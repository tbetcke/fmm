//! x86_64 AVX2 + FMA (256-bit vectors: 8 × f32, 4 × f64): the vector layer, the
//! inverse square root and the entry points.
//!
//! The entry points are `#[target_feature(enable = "avx2,fma")]` functions, which the
//! dispatch calls only after [`Isa::Avx2`] was detected. The layer methods are
//! `#[inline(always)]` and cannot carry the features themselves, so each intrinsic call
//! is an `unsafe` block; they inline into the entry points, which have them.
//!
//! **Inverse square root** (signed off in Phase 3S T2, spikes/p2p-simd/SPIKE_REPORT.md,
//! "Recommendation", item 2, and "Inverse square root, AVX2"): the estimate `vrsqrtps`,
//! whose relative error |e₀| is at most 1.5 · 2⁻¹² (Intel Intrinsics Guide), and one
//! polynomial correction. With y the estimate and r = 1 − x y², the exact correction
//! is y (1 − r)^(−½) = y (1 + c₁ r + c₂ r² + …), cₖ = C(2k, k) / 4ᵏ; it is truncated
//! after r^M and evaluated as
//!
//! y² (mul), r = 1 − x · y² (fnma), p = c₁ + r (c₂ + r (… + r c_M)) (Horner, M − 1 fma),
//! y + (y r) · p (mul, fma).
//!
//! - f32, M = 2 (`est+P2`): 6 operations. The truncation error is about c₃ |r|³ ≤ 1.2e-10
//!   for |r| ≤ 2|e₀| + e₀²; the roundings add ½ u (y², propagated) and 1 u (the last
//!   fma), so the derived bound is 1.50 u₃₂.
//! - f64, M = 5 (`est+P5`): x is converted to f32 for the estimate and the estimate
//!   back to f64 (`vcvtpd2ps`, `vrsqrtps`, `vcvtps2pd`), 11 operations. The kernel
//!   domain 2⁻¹⁰⁸ ≤ x ≤ 2⁷ (CONVENTIONS §3.13) lies in the normal range of f32, so the
//!   conversion neither overflows nor flushes; it adds 2⁻²⁵ to |e₀|, and the conversion
//!   back is exact. The truncation error is about c₆ |r|⁶ ≤ 3.5e-20; y² is exact (y has
//!   24 significant bits), so the derived bound is 1.50 u₆₄.
//!
//! These bounds are derived, not measured: nothing could be run on x86_64 hardware
//! when they were chosen. The tests of `tests/rsqrt` measure them on the x86_64 leg of
//! the `run-tests-simd` CI job. At x = 0 the estimate is +∞ and r is NaN, which the
//! mask clears. `vrsqrtps` is not architecturally defined, so results may differ
//! between Intel and AMD CPUs in the last bits (design §5.5).

use std::arch::x86_64::{
    __m256, __m256d, _CMP_EQ_OQ, _mm_rsqrt_ps, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_pd,
    _mm256_add_ps, _mm256_andnot_pd, _mm256_andnot_ps, _mm256_broadcast_sd, _mm256_broadcast_ss,
    _mm256_castpd_si256, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps,
    _mm256_cmp_pd, _mm256_cmp_ps, _mm256_cvtpd_ps, _mm256_cvtps_pd, _mm256_fmadd_pd,
    _mm256_fmadd_ps, _mm256_fnmadd_pd, _mm256_fnmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps,
    _mm256_max_pd, _mm256_max_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_rsqrt_ps, _mm256_set1_pd,
    _mm256_set1_ps, _mm256_setr_pd, _mm256_setr_ps, _mm256_slli_epi32, _mm256_slli_epi64,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps,
};

use super::p2p::{Yukawa, p2p_body, p2p_dipoles_body, p2p_many_body, p2p_skeleton};
use super::waves::{gather_body, merge_body, sums_body, waves_body};
use super::{Simd, exp::exp_body, matrix::accumulate_body, rsqrt_body};
use crate::{Isa, WaveTables};

/// The coefficients cₖ = C(2k, k) / 4ᵏ of (1 − r)^(−½) = Σₖ cₖ rᵏ, from c₁; all exact
/// in f32.
const C: [f64; 5] = [0.5, 0.375, 0.3125, 0.273_437_5, 0.246_093_75];

/// The token of AVX2 + FMA: a value exists only on a CPU with both, so the [`Simd`]
/// methods that take it are safe.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx2(());

impl Avx2 {
    /// The token, if this machine has AVX2 and FMA ([`Isa::Avx2`] is available).
    #[cfg(test)]
    pub(crate) fn try_new() -> Option<Self> {
        Isa::Avx2.is_available().then_some(Self(()))
    }

    /// The token, unchecked.
    ///
    /// # Safety
    ///
    /// The CPU must have AVX2 and FMA.
    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Simd<f32> for Avx2 {
    type V = __m256;
    type Mask = __m256;
    const W: usize = 8;
    const ISA: Isa = Isa::Avx2;

    #[inline(always)]
    fn splat(self, x: f32) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_set1_ps(x) }
    }

    #[inline(always)]
    fn load(self, x: &[f32]) -> __m256 {
        assert!(x.len() >= 8, "`x` holds fewer than 8 values");
        // SAFETY: the token proves AVX2 and FMA, and `x` holds at least the 8 values
        // read (an unaligned load).
        unsafe { _mm256_loadu_ps(x.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, v: __m256, out: &mut [f32]) {
        assert!(out.len() >= 8, "`out` holds fewer than 8 values");
        // SAFETY: the token proves AVX2 and FMA, and `out` holds at least the 8 values
        // written (an unaligned store).
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn add(self, a: __m256, b: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_add_ps(a, b) }
    }

    #[inline(always)]
    fn sub(self, a: __m256, b: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_sub_ps(a, b) }
    }

    #[inline(always)]
    fn mul(self, a: __m256, b: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_mul_ps(a, b) }
    }

    #[inline(always)]
    fn fma(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn fnma(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn eq(self, a: __m256, b: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn and_not(self, mask: __m256, v: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_andnot_ps(mask, v) }
    }

    #[inline(always)]
    fn rsqrt_estimate(self, x: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_rsqrt_ps(x) }
    }

    #[inline(always)]
    fn rsqrt(self, x: __m256) -> __m256 {
        // est+P2 (module documentation).
        // SAFETY: the token proves AVX2 and FMA.
        unsafe {
            let y = _mm256_rsqrt_ps(x);
            let r = _mm256_fnmadd_ps(x, _mm256_mul_ps(y, y), _mm256_set1_ps(1.0));
            let p = _mm256_fmadd_ps(_mm256_set1_ps(C[1] as f32), r, _mm256_set1_ps(C[0] as f32));
            _mm256_fmadd_ps(_mm256_mul_ps(y, r), p, y)
        }
    }

    #[inline(always)]
    fn load3(self, points: &[[f32; 3]]) -> [__m256; 3] {
        let p = &points[..8];
        // SAFETY: the token proves AVX2 and FMA.
        unsafe {
            [
                _mm256_setr_ps(
                    p[0][0], p[1][0], p[2][0], p[3][0], p[4][0], p[5][0], p[6][0], p[7][0],
                ),
                _mm256_setr_ps(
                    p[0][1], p[1][1], p[2][1], p[3][1], p[4][1], p[5][1], p[6][1], p[7][1],
                ),
                _mm256_setr_ps(
                    p[0][2], p[1][2], p[2][2], p[3][2], p[4][2], p[5][2], p[6][2], p[7][2],
                ),
            ]
        }
    }

    #[inline(always)]
    fn store3(self, v: [__m256; 3], points: &mut [[f32; 3]]) {
        let mut lanes = [[0.0; 8]; 3];
        for (lane, vk) in lanes.iter_mut().zip(v) {
            <Self as Simd<f32>>::store(self, vk, lane);
        }
        for (i, point) in points[..8].iter_mut().enumerate() {
            *point = [lanes[0][i], lanes[1][i], lanes[2][i]];
        }
    }

    #[inline(always)]
    fn broadcast(self, point: &[f32; 3], charge: f32) -> ([__m256; 3], __m256) {
        // SAFETY: the token proves AVX2 and FMA.
        let p = unsafe {
            [
                _mm256_broadcast_ss(&point[0]),
                _mm256_broadcast_ss(&point[1]),
                _mm256_broadcast_ss(&point[2]),
            ]
        };
        (p, Simd::splat(self, charge))
    }

    #[inline(always)]
    fn max(self, a: __m256, b: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA
        unsafe { _mm256_max_ps(a, b) }
    }

    #[inline(always)]
    fn scale_pow2(self, x: __m256, k: __m256) -> __m256 {
        // SAFETY: the token proves AVX2 and FMA; the integer lanes are the bits of x and k, reinterpreted.
        unsafe {
            _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_castps_si256(x),
                _mm256_slli_epi32::<23>(_mm256_castps_si256(k)),
            ))
        }
    }
}

impl Simd<f64> for Avx2 {
    type V = __m256d;
    type Mask = __m256d;
    const W: usize = 4;
    const ISA: Isa = Isa::Avx2;

    #[inline(always)]
    fn splat(self, x: f64) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_set1_pd(x) }
    }

    #[inline(always)]
    fn load(self, x: &[f64]) -> __m256d {
        assert!(x.len() >= 4, "`x` holds fewer than 4 values");
        // SAFETY: the token proves AVX2 and FMA, and `x` holds at least the 4 values
        // read (an unaligned load).
        unsafe { _mm256_loadu_pd(x.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, v: __m256d, out: &mut [f64]) {
        assert!(out.len() >= 4, "`out` holds fewer than 4 values");
        // SAFETY: the token proves AVX2 and FMA, and `out` holds at least the 4 values
        // written (an unaligned store).
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn add(self, a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub(self, a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul(self, a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_mul_pd(a, b) }
    }

    #[inline(always)]
    fn fma(self, a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn fnma(self, a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn eq(self, a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_cmp_pd::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn and_not(self, mask: __m256d, v: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA.
        unsafe { _mm256_andnot_pd(mask, v) }
    }

    #[inline(always)]
    fn rsqrt_estimate(self, x: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA. On the kernel domain the conversion
        // to f32 is normal (module documentation); elsewhere the result is
        // unspecified but defined.
        unsafe { _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x))) }
    }

    #[inline(always)]
    fn rsqrt(self, x: __m256d) -> __m256d {
        // est+P5 (module documentation).
        // SAFETY: the token proves AVX2 and FMA. On the kernel domain the conversion
        // to f32 is normal (module documentation).
        unsafe {
            let y = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x)));
            let r = _mm256_fnmadd_pd(x, _mm256_mul_pd(y, y), _mm256_set1_pd(1.0));
            let mut p = _mm256_set1_pd(C[4]);
            p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(C[3]));
            p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(C[2]));
            p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(C[1]));
            p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(C[0]));
            _mm256_fmadd_pd(_mm256_mul_pd(y, r), p, y)
        }
    }

    #[inline(always)]
    fn load3(self, points: &[[f64; 3]]) -> [__m256d; 3] {
        let p = &points[..4];
        // SAFETY: the token proves AVX2 and FMA.
        unsafe {
            [
                _mm256_setr_pd(p[0][0], p[1][0], p[2][0], p[3][0]),
                _mm256_setr_pd(p[0][1], p[1][1], p[2][1], p[3][1]),
                _mm256_setr_pd(p[0][2], p[1][2], p[2][2], p[3][2]),
            ]
        }
    }

    #[inline(always)]
    fn store3(self, v: [__m256d; 3], points: &mut [[f64; 3]]) {
        let mut lanes = [[0.0; 4]; 3];
        for (lane, vk) in lanes.iter_mut().zip(v) {
            <Self as Simd<f64>>::store(self, vk, lane);
        }
        for (i, point) in points[..4].iter_mut().enumerate() {
            *point = [lanes[0][i], lanes[1][i], lanes[2][i]];
        }
    }

    #[inline(always)]
    fn broadcast(self, point: &[f64; 3], charge: f64) -> ([__m256d; 3], __m256d) {
        // SAFETY: the token proves AVX2 and FMA.
        let p = unsafe {
            [
                _mm256_broadcast_sd(&point[0]),
                _mm256_broadcast_sd(&point[1]),
                _mm256_broadcast_sd(&point[2]),
            ]
        };
        (p, Simd::splat(self, charge))
    }

    #[inline(always)]
    fn max(self, a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA
        unsafe { _mm256_max_pd(a, b) }
    }

    #[inline(always)]
    fn scale_pow2(self, x: __m256d, k: __m256d) -> __m256d {
        // SAFETY: the token proves AVX2 and FMA; the integer lanes are the bits of x and k, reinterpreted.
        unsafe {
            _mm256_castsi256_pd(_mm256_add_epi64(
                _mm256_castpd_si256(x),
                _mm256_slli_epi64::<52>(_mm256_castpd_si256(k)),
            ))
        }
    }
}

/// Generates the inverse-square-root entry point of one precision.
macro_rules! rsqrt_entry {
    ($name:ident, $t:ty) => {
        #[doc = concat!(
                    "The kernel's inverse square root, masked at 0, on AVX2 + FMA in ",
                    stringify!($t), ": `out[i] = rsqrt(x[i])`, with lengths checked by the callee."
                )]
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $name(x: &[$t], out: &mut [$t]) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            rsqrt_body(s, x, out);
        }
    };
}

rsqrt_entry!(rsqrt_slice_f32, f32);
rsqrt_entry!(rsqrt_slice_f64, f64);

/// Generates the entry point of the exponential of one precision (Helmholtz CH2.4a).
macro_rules! exp_entry {
    ($name:ident, $t:ty) => {
        #[doc = concat!(
                    "The exponential (Helmholtz CH2.4a) on AVX2 + FMA in ",
                    stringify!($t), ": `out[i] = exp(x[i])`, with lengths checked by the callee."
                )]
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $name(x: &[$t], out: &mut [$t]) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            exp_body(s, x, out);
        }
    };
}

exp_entry!(exp_slice_f32, f32);
exp_entry!(exp_slice_f64, f64);

/// Target vectors per P2P block in f32 (signed off in Phase 3S T2,
/// spikes/p2p-simd/SPIKE_REPORT.md, "Recommendation", item 3).
pub(crate) const K_F32: usize = 1;
/// Target vectors per P2P block in f64 (as [`K_F32`]).
pub(crate) const K_F64: usize = 1;

/// Generates the P2P entry points of one precision, potential only and with gradients.
macro_rules! p2p_entries {
    ($potential:ident, $gradient:ident, $t:ty, $k:expr) => {
        /// P2P on AVX2 + FMA, potential only (`arch::p2p::p2p_body`), in the precision of
        /// its name, with lengths checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_body::<$t, Avx2, { $k }, false>(s, sources, charges, targets, potential, &mut []);
        }

        /// P2P on AVX2 + FMA, potential and gradient (`arch::p2p::p2p_body`), in the
        /// precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_body::<$t, Avx2, { $k }, true>(s, sources, charges, targets, potential, gradient);
        }
    };
}

p2p_entries!(p2p_f32, p2p_f32_gradient, f32, K_F32);
p2p_entries!(p2p_f64, p2p_f64_gradient, f64, K_F64);

/// Target vectors per block of the dipole P2P (`arch::p2p::p2p_dipoles_body`; Phase 6 T8),
/// both precisions, potential only and with gradients: a block of K target vectors holds
/// 7 K vectors (targets, potentials, gradients) in registers besides the source's six
/// broadcasts and the temporaries, so 1 (the charge kernel's K at most).
pub(crate) const DIPOLE_K: usize = 1;

/// Generates the dipole P2P entry points of one precision, potential only and with
/// gradients (Phase 6 T8).
macro_rules! p2p_dipole_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// Dipole P2P on AVX2 + FMA, potential only (`arch::p2p::p2p_dipoles_body`), in the
        /// precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_dipoles_body::<$t, Avx2, DIPOLE_K, false>(
                s,
                sources,
                dipoles,
                targets,
                potential,
                &mut [],
            );
        }

        /// Dipole P2P on AVX2 + FMA, potential and gradient (`arch::p2p::p2p_dipoles_body`), in
        /// the precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_dipoles_body::<$t, Avx2, DIPOLE_K, true>(
                s, sources, dipoles, targets, potential, gradient,
            );
        }
    };
}

p2p_dipole_entries!(p2p_dipoles_f32, p2p_dipoles_f32_gradient, f32);
p2p_dipole_entries!(p2p_dipoles_f64, p2p_dipoles_f64_gradient, f64);

/// Target vectors per block of the Yukawa P2P (`arch::p2p::p2p_skeleton` with
/// `arch::p2p::Yukawa`; Helmholtz CH2.4b), both precisions, potential only and with
/// gradients: the exponential's constants (19 in f64) alone exceed AVX2's 16 registers, so
/// the block holds one target vector and the constants are memory operands.
pub(crate) const YUKAWA_K: usize = 1;

/// Generates the Yukawa P2P entry points of one precision, potential only and with
/// gradients (Helmholtz CH2.4b).
macro_rules! p2p_yukawa_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// Yukawa P2P on AVX2 + FMA, potential only (`arch::p2p::p2p_skeleton` with
        /// `arch::p2p::Yukawa`), in the precision of its name, with lengths and κ ≥ 0
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $potential(
            kappa: $t,
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with them: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_skeleton::<$t, Avx2, _, YUKAWA_K, false>(
                s,
                Yukawa::new(s, kappa),
                sources,
                charges,
                targets,
                potential,
                &mut [],
            );
        }

        /// Yukawa P2P on AVX2 + FMA, potential and gradient (`arch::p2p::p2p_skeleton`
        /// with `arch::p2p::Yukawa`), in the precision of its name, with lengths and κ ≥ 0
        /// checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $gradient(
            kappa: $t,
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with them: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_skeleton::<$t, Avx2, _, YUKAWA_K, true>(
                s,
                Yukawa::new(s, kappa),
                sources,
                charges,
                targets,
                potential,
                gradient,
            );
        }
    };
}

p2p_yukawa_entries!(p2p_yukawa_f32, p2p_yukawa_f32_gradient, f32);
p2p_yukawa_entries!(p2p_yukawa_f64, p2p_yukawa_f64_gradient, f64);

/// Target vectors per block of the P2P of several charge vectors (`arch::p2p::p2p_many_body`;
/// Phase 6 T5), potential only: measured on one core of Kathleen's Xeon Gold 6248 (64
/// sources and targets, m = 8 and 32, Kathleen job 604422), 1 target vector with groups of
/// 8 vectors ran 3.9× (f32) and 4.4× (f64) faster per vector than m calls of the
/// single-vector kernel, against 3.4× and 4.5× with 2 and 8 (but slower in absolute time),
/// 2.8× and 2.9× with 1 and 4, and 1.8× and 1.6× with 1 and 2.
pub(crate) const MANY_K: usize = 1;
/// Vectors per group of the P2P of several charge vectors, potential only (as [`MANY_K`]).
pub(crate) const MANY_VG: usize = 8;
/// Target vectors per block with gradients: 1 target vector with groups of 4 vectors
/// measured 1.5× (f32) and 1.9× (f64) faster per vector, against 1.4× with groups of 2
/// and 1.0× with 1 (16 accumulators per group in 16 vector registers spill, but less than
/// the shared work saves).
pub(crate) const MANY_K_GRADIENT: usize = 1;
/// Vectors per group with gradients (as [`MANY_K_GRADIENT`]).
pub(crate) const MANY_VG_GRADIENT: usize = 4;

/// Generates the entry points of the P2P of several charge vectors of one precision,
/// potential only and with gradients (Phase 6 T5).
macro_rules! p2p_many_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// P2P of several charge vectors on AVX2 + FMA, potential only
        /// (`arch::p2p::p2p_many_body`), in the precision of its name, with lengths
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_many_body::<$t, Avx2, MANY_K, MANY_VG, false>(
                s,
                sources,
                charges,
                m,
                targets,
                potential,
                &mut [],
            );
        }

        /// P2P of several charge vectors on AVX2 + FMA, potential and gradient
        /// (`arch::p2p::p2p_many_body`), in the precision of its name, with lengths
        /// checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            p2p_many_body::<$t, Avx2, MANY_K_GRADIENT, MANY_VG_GRADIENT, true>(
                s, sources, charges, m, targets, potential, gradient,
            );
        }
    };
}

p2p_many_entries!(p2p_many_f32, p2p_many_f32_gradient, f32);
p2p_many_entries!(p2p_many_f64, p2p_many_f64_gradient, f64);

/// Row vectors per block of the column-accumulation kernel in f32 (`arch::matrix`):
/// 16 vector registers: 8 accumulators of 2 × 4 (f32: 16 rows; f64: 8 rows), 2 of A and
/// the broadcast.
pub(crate) const MATRIX_ROWS_F32: usize = 2;
/// Row vectors per block of the column-accumulation kernel in f64 (as
/// [`MATRIX_ROWS_F32`]).
pub(crate) const MATRIX_ROWS_F64: usize = 2;

/// Generates the column-accumulation entry points of one precision, separate and fused.
macro_rules! accumulate_entries {
    ($separate:ident, $fused:ident, $t:ty, $r:expr) => {
        /// C[:, c_j] += A · B[:, b_j] on AVX2 + FMA, multiply and add rounded apart
        /// (`arch::matrix::accumulate_body`), in the precision of its name, with shapes
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $separate(
            n: usize,
            a: &[$t],
            b: &[$t],
            b_offsets: &[usize],
            c: &mut [$t],
            c_offsets: &[usize],
            pack: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            accumulate_body::<$t, Avx2, { $r }, false>(s, n, a, b, b_offsets, c, c_offsets, pack);
        }

        /// The fused form of the separate entry point: one fma per step.
        ///
        /// Not inlined, as the separate entry point.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $fused(
            n: usize,
            a: &[$t],
            b: &[$t],
            b_offsets: &[usize],
            c: &mut [$t],
            c_offsets: &[usize],
            pack: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with both: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            accumulate_body::<$t, Avx2, { $r }, true>(s, n, a, b, b_offsets, c, c_offsets, pack);
        }
    };
}

accumulate_entries!(accumulate_f32, accumulate_f32_fused, f32, MATRIX_ROWS_F32);
accumulate_entries!(accumulate_f64, accumulate_f64_fused, f64, MATRIX_ROWS_F64);

/// Generates the plane-wave entry points of one precision (Phase 6 T13; the bodies of
/// `arch::waves`).
macro_rules! waves_entries {
    ($t:ty, $merge:ident, $gather:ident, $waves:ident, $sums:ident) => {
        /// [`WaveKernel::merge`](crate::WaveKernel::merge) on AVX2 + FMA, in the precision of
        /// its name, with shapes checked by the caller (`arch::waves::merge_body`).
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check finds
        /// the function by name.
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $merge(
            acc: &mut [$t],
            waves: &[$t],
            wave_offsets: &[usize],
            shifts: &[$t],
            shift_offsets: &[usize],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with them: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            merge_body::<$t, Avx2>(s, acc, waves, wave_offsets, shifts, shift_offsets);
        }

        /// [`WaveKernel::gather`](crate::WaveKernel::gather) on AVX2 + FMA, as the merge entry
        /// point (`arch::waves::gather_body`).
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $gather(v: &mut [$t], accs: &[$t], acc_offsets: &[usize], t: &[$t]) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with them: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            gather_body::<$t, Avx2>(s, v, accs, acc_offsets, t);
        }

        /// [`WaveKernel::waves`](crate::WaveKernel::waves) on AVX2 + FMA, as the merge entry
        /// point (`arch::waves::waves_body`).
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $waves(
            tables: &WaveTables<$t>,
            modes: &[$t],
            up: &mut [$t],
            down: &mut [$t],
        ) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with them: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            waves_body::<$t, Avx2, 2>(s, tables, modes, up, down);
        }

        /// [`WaveKernel::sums`](crate::WaveKernel::sums) on AVX2 + FMA, as the merge entry
        /// point (`arch::waves::sums_body`).
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        pub(crate) fn $sums(tables: &WaveTables<$t>, up: &[$t], down: &[$t], sums: &mut [$t]) {
            // SAFETY: this function enables `avx2` and `fma`, so it runs only on a CPU
            // with them: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx2::new_unchecked() };
            sums_body::<$t, Avx2>(s, tables, up, down, sums);
        }
    };
}

waves_entries!(f32, merge_f32, gather_f32, waves_f32, sums_f32);
waves_entries!(f64, merge_f64, gather_f64, waves_f64, sums_f64);
