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
    __m256, __m256d, _CMP_EQ_OQ, _mm_rsqrt_ps, _mm256_add_pd, _mm256_add_ps, _mm256_andnot_pd,
    _mm256_andnot_ps, _mm256_broadcast_sd, _mm256_broadcast_ss, _mm256_cmp_pd, _mm256_cmp_ps,
    _mm256_cvtpd_ps, _mm256_cvtps_pd, _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_fnmadd_pd,
    _mm256_fnmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd, _mm256_mul_ps,
    _mm256_rsqrt_ps, _mm256_set1_pd, _mm256_set1_ps, _mm256_setr_pd, _mm256_setr_ps,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps,
};

use super::{Simd, p2p::p2p_body, rsqrt_body};
use crate::Isa;

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
