//! x86_64 AVX-512F (512-bit vectors: 16 × f32, 8 × f64; Phase 6 T6): the vector layer,
//! the inverse square root and the entry points (docs/design/optimisation.md §5.2,
//! docs/design/simd-p2p.md §4.7).
//!
//! The entry points are `#[target_feature(enable = "avx512f")]` functions, which the
//! dispatch calls only after [`Isa::Avx512`] was detected (AVX-512F, AVX2 and FMA; the
//! compiler treats `avx512f` as implying the other two). The layer methods are
//! `#[inline(always)]` and cannot carry the feature themselves, so each intrinsic call
//! is an `unsafe` block; they inline into the entry points, which have it. Every
//! instruction used is AVX-512F: the 512-bit arithmetic and fma, `vrsqrt14ps`/`pd`,
//! compares into a mask register and zero-masked moves.
//!
//! **Masks.** [`Simd::Mask`] is a mask register, `__mmask16` (f32) or `__mmask8` (f64):
//! `eq` compares into it and `and_not(m, v)` is the zero-masked move of v under the
//! complement of m, which clears the lanes of the r² = 0 rule whatever they hold (NaN
//! and ±∞ included). Tails run on the padded stack copies of `arch::p2p` and
//! `arch::rsqrt_body`, as on every ISA (optimisation.md §5.2): the same per-lane
//! instruction sequence, so target-position invariance holds as it does elsewhere.
//!
//! **Inverse square root** (simd-p2p.md §4.3, §4.7): the estimate `vrsqrt14ps`/`pd`,
//! whose relative error |e₀| is below 2⁻¹⁴ (Intel Intrinsics Guide; Intel's SDM gives a
//! bit-exact reference, so it does not vary between implementations), then Newton steps
//! with h = x/2 (exact):
//!
//! y ← y + y (½ − h y²)   (mul y², fnma, fma),
//!
//! one in f32 and two in f64, with h formed once: 5 operations in f32 and 8 in f64,
//! against 6 and 11 on AVX2, and no detour through f32 for f64. A step squares the
//! relative error and multiplies it by about 3/2.
//!
//! - f32, one step: |e₁| ≤ 1.5 · 2⁻²⁸ = 0.09 u₃₂ before rounding; the rounding of y²
//!   (½ u, through h y² ≈ ½) and the last fma (1 u) give a derived bound of 1.6 u₃₂.
//! - f64, two steps: |e₁| ≤ 1.5 · 2⁻²⁸, |e₂| ≤ 1.5 e₁² = 0.42 u₆₄ before rounding; with
//!   the roundings of the second step, a derived bound of 1.9 u₆₄.
//!
//! The kernel domain 2⁻¹⁰⁸ ≤ x ≤ 2⁷ (CONVENTIONS §3.13) is normal in both precisions, so
//! the estimate neither flushes nor overflows. At x = 0 the estimate is +∞ and the step
//! gives NaN (0 · ∞), which the mask clears. The tests of `tests/rsqrt` measure the bounds
//! on hardware with AVX-512F (Kathleen's Cascade Lake; opportunistically CI).

use std::arch::x86_64::{
    __m512, __m512d, __mmask8, __mmask16, _CMP_EQ_OQ, _mm512_add_epi32, _mm512_add_epi64,
    _mm512_add_pd, _mm512_add_ps, _mm512_castpd_si512, _mm512_castps_si512, _mm512_castsi512_pd,
    _mm512_castsi512_ps, _mm512_cmp_pd_mask, _mm512_cmp_ps_mask, _mm512_fmadd_pd, _mm512_fmadd_ps,
    _mm512_fnmadd_pd, _mm512_fnmadd_ps, _mm512_loadu_pd, _mm512_loadu_ps, _mm512_maskz_mov_pd,
    _mm512_maskz_mov_ps, _mm512_max_pd, _mm512_max_ps, _mm512_mul_pd, _mm512_mul_ps,
    _mm512_rsqrt14_pd, _mm512_rsqrt14_ps, _mm512_set1_pd, _mm512_set1_ps, _mm512_slli_epi32,
    _mm512_slli_epi64, _mm512_storeu_pd, _mm512_storeu_ps, _mm512_sub_pd, _mm512_sub_ps,
};

use super::p2p::{Yukawa, p2p_body, p2p_dipoles_body, p2p_many_body, p2p_skeleton};
use super::waves::{gather_body, merge_body, sums_body, waves_body};
use super::{Simd, exp::exp_body, matrix::accumulate_body, rsqrt_body, sincos::sincos_body};
use crate::{Isa, WaveTables};

/// The token of AVX-512F (with AVX2 and FMA): a value exists only on a CPU with them, so
/// the [`Simd`] methods that take it are safe.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx512(());

impl Avx512 {
    /// The token, if this machine has AVX-512F, AVX2 and FMA ([`Isa::Avx512`] is
    /// available).
    #[cfg(test)]
    pub(crate) fn try_new() -> Option<Self> {
        Isa::Avx512.is_available().then_some(Self(()))
    }

    /// The token, unchecked.
    ///
    /// # Safety
    ///
    /// The CPU must have AVX-512F, AVX2 and FMA.
    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Simd<f32> for Avx512 {
    type V = __m512;
    type Mask = __mmask16;
    const W: usize = 16;
    const ISA: Isa = Isa::Avx512;

    #[inline(always)]
    fn splat(self, x: f32) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_set1_ps(x) }
    }

    #[inline(always)]
    fn load(self, x: &[f32]) -> __m512 {
        assert!(x.len() >= 16, "`x` holds fewer than 16 values");
        // SAFETY: the token proves AVX-512F, and `x` holds at least the 16 values read
        // (an unaligned load).
        unsafe { _mm512_loadu_ps(x.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, v: __m512, out: &mut [f32]) {
        assert!(out.len() >= 16, "`out` holds fewer than 16 values");
        // SAFETY: the token proves AVX-512F, and `out` holds at least the 16 values
        // written (an unaligned store).
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_add_ps(a, b) }
    }

    #[inline(always)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_sub_ps(a, b) }
    }

    #[inline(always)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_mul_ps(a, b) }
    }

    #[inline(always)]
    fn fma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn fnma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn eq(self, a: __m512, b: __m512) -> __mmask16 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn and_not(self, mask: __mmask16, v: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_maskz_mov_ps(!mask, v) }
    }

    #[inline(always)]
    fn rsqrt_estimate(self, x: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_rsqrt14_ps(x) }
    }

    #[inline(always)]
    fn rsqrt(self, x: __m512) -> __m512 {
        // The estimate and one Newton step (module documentation).
        // SAFETY: the token proves AVX-512F.
        unsafe {
            let half = _mm512_set1_ps(0.5);
            let h = _mm512_mul_ps(x, half);
            let y = _mm512_rsqrt14_ps(x);
            let r = _mm512_fnmadd_ps(h, _mm512_mul_ps(y, y), half);
            _mm512_fmadd_ps(y, r, y)
        }
    }

    #[inline(always)]
    fn load3(self, points: &[[f32; 3]]) -> [__m512; 3] {
        let mut lanes = [[0.0; 16]; 3];
        for (i, point) in points[..16].iter().enumerate() {
            for (lane, &value) in lanes.iter_mut().zip(point) {
                lane[i] = value;
            }
        }
        lanes.map(|lane| <Self as Simd<f32>>::load(self, &lane))
    }

    #[inline(always)]
    fn store3(self, v: [__m512; 3], points: &mut [[f32; 3]]) {
        let mut lanes = [[0.0; 16]; 3];
        for (lane, vk) in lanes.iter_mut().zip(v) {
            <Self as Simd<f32>>::store(self, vk, lane);
        }
        for (i, point) in points[..16].iter_mut().enumerate() {
            *point = [lanes[0][i], lanes[1][i], lanes[2][i]];
        }
    }

    #[inline(always)]
    fn broadcast(self, point: &[f32; 3], charge: f32) -> ([__m512; 3], __m512) {
        (
            point.map(|c| Simd::splat(self, c)),
            Simd::splat(self, charge),
        )
    }

    #[inline(always)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F
        unsafe { _mm512_max_ps(a, b) }
    }

    #[inline(always)]
    fn scale_pow2(self, x: __m512, k: __m512) -> __m512 {
        // SAFETY: the token proves AVX-512F; the integer lanes are the bits of x and k, reinterpreted.
        unsafe {
            _mm512_castsi512_ps(_mm512_add_epi32(
                _mm512_castps_si512(x),
                _mm512_slli_epi32::<23>(_mm512_castps_si512(k)),
            ))
        }
    }
}

impl Simd<f64> for Avx512 {
    type V = __m512d;
    type Mask = __mmask8;
    const W: usize = 8;
    const ISA: Isa = Isa::Avx512;

    #[inline(always)]
    fn splat(self, x: f64) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_set1_pd(x) }
    }

    #[inline(always)]
    fn load(self, x: &[f64]) -> __m512d {
        assert!(x.len() >= 8, "`x` holds fewer than 8 values");
        // SAFETY: the token proves AVX-512F, and `x` holds at least the 8 values read (an
        // unaligned load).
        unsafe { _mm512_loadu_pd(x.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, v: __m512d, out: &mut [f64]) {
        assert!(out.len() >= 8, "`out` holds fewer than 8 values");
        // SAFETY: the token proves AVX-512F, and `out` holds at least the 8 values
        // written (an unaligned store).
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn add(self, a: __m512d, b: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub(self, a: __m512d, b: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_mul_pd(a, b) }
    }

    #[inline(always)]
    fn fma(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn fnma(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn eq(self, a: __m512d, b: __m512d) -> __mmask8 {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn and_not(self, mask: __mmask8, v: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_maskz_mov_pd(!mask, v) }
    }

    #[inline(always)]
    fn rsqrt_estimate(self, x: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F.
        unsafe { _mm512_rsqrt14_pd(x) }
    }

    #[inline(always)]
    fn rsqrt(self, x: __m512d) -> __m512d {
        // The estimate and two Newton steps (module documentation).
        // SAFETY: the token proves AVX-512F.
        unsafe {
            let half = _mm512_set1_pd(0.5);
            let h = _mm512_mul_pd(x, half);
            let mut y = _mm512_rsqrt14_pd(x);
            let r = _mm512_fnmadd_pd(h, _mm512_mul_pd(y, y), half);
            y = _mm512_fmadd_pd(y, r, y);
            let r = _mm512_fnmadd_pd(h, _mm512_mul_pd(y, y), half);
            _mm512_fmadd_pd(y, r, y)
        }
    }

    #[inline(always)]
    fn load3(self, points: &[[f64; 3]]) -> [__m512d; 3] {
        let mut lanes = [[0.0; 8]; 3];
        for (i, point) in points[..8].iter().enumerate() {
            for (lane, &value) in lanes.iter_mut().zip(point) {
                lane[i] = value;
            }
        }
        lanes.map(|lane| <Self as Simd<f64>>::load(self, &lane))
    }

    #[inline(always)]
    fn store3(self, v: [__m512d; 3], points: &mut [[f64; 3]]) {
        let mut lanes = [[0.0; 8]; 3];
        for (lane, vk) in lanes.iter_mut().zip(v) {
            <Self as Simd<f64>>::store(self, vk, lane);
        }
        for (i, point) in points[..8].iter_mut().enumerate() {
            *point = [lanes[0][i], lanes[1][i], lanes[2][i]];
        }
    }

    #[inline(always)]
    fn broadcast(self, point: &[f64; 3], charge: f64) -> ([__m512d; 3], __m512d) {
        (
            point.map(|c| Simd::splat(self, c)),
            Simd::splat(self, charge),
        )
    }

    #[inline(always)]
    fn max(self, a: __m512d, b: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F
        unsafe { _mm512_max_pd(a, b) }
    }

    #[inline(always)]
    fn scale_pow2(self, x: __m512d, k: __m512d) -> __m512d {
        // SAFETY: the token proves AVX-512F; the integer lanes are the bits of x and k, reinterpreted.
        unsafe {
            _mm512_castsi512_pd(_mm512_add_epi64(
                _mm512_castpd_si512(x),
                _mm512_slli_epi64::<52>(_mm512_castpd_si512(k)),
            ))
        }
    }
}

/// Generates the inverse-square-root entry point of one precision.
macro_rules! rsqrt_entry {
    ($name:ident, $t:ty) => {
        #[doc = concat!(
                    "The kernel's inverse square root, masked at 0, on AVX-512F in ",
                    stringify!($t), ": `out[i] = rsqrt(x[i])`, with lengths checked by the callee."
                )]
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $name(x: &[$t], out: &mut [$t]) {
            // SAFETY: this function enables `avx512f` (and so, to the compiler, `avx2`
            // and `fma`), and the dispatch calls it only where `Isa::Avx512` is
            // available: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
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
                    "The exponential (Helmholtz CH2.4a) on AVX-512F in ",
                    stringify!($t), ": `out[i] = exp(x[i])`, with lengths checked by the callee."
                )]
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $name(x: &[$t], out: &mut [$t]) {
            // SAFETY: this function enables `avx512f` (and so, to the compiler, `avx2`
            // and `fma`), and the dispatch calls it only where `Isa::Avx512` is
            // available: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            exp_body(s, x, out);
        }
    };
}

exp_entry!(exp_slice_f32, f32);
exp_entry!(exp_slice_f64, f64);

/// Generates the entry point of the sine and cosine of one precision (Helmholtz CH3.4a).
macro_rules! sincos_entry {
    ($name:ident, $t:ty) => {
        #[doc = concat!(
                            "The sine and cosine (Helmholtz CH3.4a) on AVX-512F in ",
                            stringify!($t),
                            ": `sin[i] = sin(theta[i])`, `cos[i] = cos(theta[i])`, with lengths \
                     checked by the callee."
                        )]
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $name(theta: &[$t], sin: &mut [$t], cos: &mut [$t]) {
            // SAFETY: this function enables `avx512f` (and so, to the compiler, `avx2`
            // and `fma`), and the dispatch calls it only where `Isa::Avx512` is
            // available: a call from outside such a context is `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            sincos_body(s, theta, sin, cos);
        }
    };
}

sincos_entry!(sincos_slice_f32, f32);
sincos_entry!(sincos_slice_f64, f64);

/// Target vectors per P2P block in f32 (optimisation.md §5.2): 32 vector registers hold
/// two vectors' 2 × 7 accumulators and coordinates with room for the inverse square
/// root's chains. Measured on one core of Kathleen's Xeon Gold 6248 (`p2p_kernels --quick`,
/// the geometric means of the gathered W1 cells, Kathleen job 604543): K = 2 against
/// K = 1 and 4 within 5 %, best or tied in both precisions and outputs (f64 φ, ∇φ 1.62
/// Gpairs/s against 1.56 at K = 1; fmm-validate/results/phase6-avx512.md).
pub(crate) const K_F32: usize = 2;
/// Target vectors per P2P block in f64 (as [`K_F32`]).
pub(crate) const K_F64: usize = 2;

/// Generates the P2P entry points of one precision, potential only and with gradients.
macro_rules! p2p_entries {
    ($potential:ident, $gradient:ident, $t:ty, $k:expr) => {
        /// P2P on AVX-512F, potential only (`arch::p2p::p2p_body`), in the precision of
        /// its name, with lengths checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_body::<$t, Avx512, { $k }, false>(s, sources, charges, targets, potential, &mut []);
        }

        /// P2P on AVX-512F, potential and gradient (`arch::p2p::p2p_body`), in the
        /// precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_body::<$t, Avx512, { $k }, true>(s, sources, charges, targets, potential, gradient);
        }
    };
}

p2p_entries!(p2p_f32, p2p_f32_gradient, f32, K_F32);
p2p_entries!(p2p_f64, p2p_f64_gradient, f64, K_F64);

/// Target vectors per block of the dipole P2P (`arch::p2p::p2p_dipoles_body`; Phase 6 T8),
/// both precisions, potential only and with gradients: a block of K target vectors holds
/// 7 K vectors (targets, potentials, gradients) in registers besides the source's six
/// broadcasts and the temporaries, so 2 (the charge kernel's K at most).
pub(crate) const DIPOLE_K: usize = 2;

/// Generates the dipole P2P entry points of one precision, potential only and with
/// gradients (Phase 6 T8).
macro_rules! p2p_dipole_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// Dipole P2P on AVX-512F, potential only (`arch::p2p::p2p_dipoles_body`), in the
        /// precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_dipoles_body::<$t, Avx512, DIPOLE_K, false>(
                s,
                sources,
                dipoles,
                targets,
                potential,
                &mut [],
            );
        }

        /// Dipole P2P on AVX-512F, potential and gradient (`arch::p2p::p2p_dipoles_body`), in
        /// the precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_dipoles_body::<$t, Avx512, DIPOLE_K, true>(
                s, sources, dipoles, targets, potential, gradient,
            );
        }
    };
}

p2p_dipole_entries!(p2p_dipoles_f32, p2p_dipoles_f32_gradient, f32);
p2p_dipole_entries!(p2p_dipoles_f64, p2p_dipoles_f64_gradient, f64);

/// Target vectors per block of the Yukawa P2P (`arch::p2p::p2p_skeleton` with
/// `arch::p2p::Yukawa`; Helmholtz CH2.4b), both precisions, potential only and with
/// gradients: as NEON's, with the same 32 registers (K = 2 timed against it on Kathleen,
/// report `fmm-validate/results/helmholtz-ch2-4b.md`).
pub(crate) const YUKAWA_K: usize = 1;

/// Generates the Yukawa P2P entry points of one precision, potential only and with
/// gradients (Helmholtz CH2.4b).
macro_rules! p2p_yukawa_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// Yukawa P2P on AVX-512F, potential only (`arch::p2p::p2p_skeleton` with
        /// `arch::p2p::Yukawa`), in the precision of its name, with lengths and κ ≥ 0
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $potential(
            kappa: $t,
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_skeleton::<$t, Avx512, _, 1, 1, YUKAWA_K, false>(
                s,
                Yukawa::new(s, kappa),
                sources,
                charges.as_chunks::<1>().0,
                targets,
                potential,
                &mut [],
            );
        }

        /// Yukawa P2P on AVX-512F, potential and gradient (`arch::p2p::p2p_skeleton`
        /// with `arch::p2p::Yukawa`), in the precision of its name, with lengths and κ ≥ 0
        /// checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $gradient(
            kappa: $t,
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_skeleton::<$t, Avx512, _, 1, 1, YUKAWA_K, true>(
                s,
                Yukawa::new(s, kappa),
                sources,
                charges.as_chunks::<1>().0,
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
/// Phase 6 T5), potential only: AVX2's choice, 1 target vector with groups of 8 vectors.
pub(crate) const MANY_K: usize = 1;
/// Vectors per group of the P2P of several charge vectors, potential only (as [`MANY_K`]).
pub(crate) const MANY_VG: usize = 8;
/// Target vectors per block with gradients: 1 target vector with groups of 4 vectors, as
/// on AVX2; 16 accumulators fit AVX-512's 32 registers without spilling.
pub(crate) const MANY_K_GRADIENT: usize = 1;
/// Vectors per group with gradients (as [`MANY_K_GRADIENT`]).
pub(crate) const MANY_VG_GRADIENT: usize = 4;

/// Generates the entry points of the P2P of several charge vectors of one precision,
/// potential only and with gradients (Phase 6 T5).
macro_rules! p2p_many_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// P2P of several charge vectors on AVX-512F, potential only
        /// (`arch::p2p::p2p_many_body`), in the precision of its name, with lengths
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_many_body::<$t, Avx512, MANY_K, MANY_VG, false>(
                s,
                sources,
                charges,
                m,
                targets,
                potential,
                &mut [],
            );
        }

        /// P2P of several charge vectors on AVX-512F, potential and gradient
        /// (`arch::p2p::p2p_many_body`), in the precision of its name, with lengths
        /// checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            p2p_many_body::<$t, Avx512, MANY_K_GRADIENT, MANY_VG_GRADIENT, true>(
                s, sources, charges, m, targets, potential, gradient,
            );
        }
    };
}

p2p_many_entries!(p2p_many_f32, p2p_many_f32_gradient, f32);
p2p_many_entries!(p2p_many_f64, p2p_many_f64_gradient, f64);

/// Row vectors per block of the column-accumulation kernel in f32 (`arch::matrix`): 32
/// vector registers hold 4 × 4 accumulators (f32: 64 rows; f64: 32 rows), 4 of A and the
/// broadcast. Measured inside the FMM on one core of Kathleen (the N = 10⁶ cube,
/// `Batched`, Kathleen job 604543): R = 4 against 2 took the downward pass from 2.90 to
/// 2.50 s at f64 p = 8 and from 1.35 to 1.21 s at f32 p = 8, and left p = 6 within 6 %.
pub(crate) const MATRIX_ROWS_F32: usize = 4;
/// Row vectors per block of the column-accumulation kernel in f64 (as
/// [`MATRIX_ROWS_F32`]).
pub(crate) const MATRIX_ROWS_F64: usize = 4;

/// Generates the column-accumulation entry points of one precision, separate and fused.
macro_rules! accumulate_entries {
    ($separate:ident, $fused:ident, $t:ty, $r:expr) => {
        /// C[:, c_j] += A · B[:, b_j] on AVX-512F, multiply and add rounded apart
        /// (`arch::matrix::accumulate_body`), in the precision of its name, with shapes
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "avx512f")]
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
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            accumulate_body::<$t, Avx512, { $r }, false>(s, n, a, b, b_offsets, c, c_offsets, pack);
        }

        /// The fused form of the separate entry point: one fma per step.
        ///
        /// Not inlined, as the separate entry point.
        #[target_feature(enable = "avx512f")]
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
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            accumulate_body::<$t, Avx512, { $r }, true>(s, n, a, b, b_offsets, c, c_offsets, pack);
        }
    };
}

accumulate_entries!(accumulate_f32, accumulate_f32_fused, f32, MATRIX_ROWS_F32);
accumulate_entries!(accumulate_f64, accumulate_f64_fused, f64, MATRIX_ROWS_F64);

/// Generates the plane-wave entry points of one precision (Phase 6 T13; the bodies of
/// `arch::waves`).
macro_rules! waves_entries {
    ($t:ty, $merge:ident, $gather:ident, $waves:ident, $sums:ident) => {
        /// [`WaveKernel::merge`](crate::WaveKernel::merge) on AVX-512F, in the precision of
        /// its name, with shapes checked by the caller (`arch::waves::merge_body`).
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check finds
        /// the function by name.
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $merge(
            acc: &mut [$t],
            waves: &[$t],
            wave_offsets: &[usize],
            shifts: &[$t],
            shift_offsets: &[usize],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            merge_body::<$t, Avx512>(s, acc, waves, wave_offsets, shifts, shift_offsets);
        }

        /// [`WaveKernel::gather`](crate::WaveKernel::gather) on AVX-512F, as the merge entry
        /// point (`arch::waves::gather_body`).
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $gather(v: &mut [$t], accs: &[$t], acc_offsets: &[usize], t: &[$t]) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            gather_body::<$t, Avx512>(s, v, accs, acc_offsets, t);
        }

        /// [`WaveKernel::waves`](crate::WaveKernel::waves) on AVX-512F, as the merge entry
        /// point (`arch::waves::waves_body`).
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $waves(
            tables: &WaveTables<$t>,
            modes: &[$t],
            up: &mut [$t],
            down: &mut [$t],
        ) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            waves_body::<$t, Avx512, { super::waves::BLOCK }>(s, tables, modes, up, down);
        }

        /// [`WaveKernel::sums`](crate::WaveKernel::sums) on AVX-512F, as the merge entry
        /// point (`arch::waves::sums_body`).
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        pub(crate) fn $sums(tables: &WaveTables<$t>, up: &[$t], down: &[$t], sums: &mut [$t]) {
            // SAFETY: this function enables `avx512f` (and so `avx2` and `fma`), so it
            // runs only on a CPU with them: a call from outside such a context is
            // `unsafe` and asserts it.
            let s = unsafe { Avx512::new_unchecked() };
            sums_body::<$t, Avx512>(s, tables, up, down, sums);
        }
    };
}

waves_entries!(f32, merge_f32, gather_f32, waves_f32, sums_f32);
waves_entries!(f64, merge_f64, gather_f64, waves_f64, sums_f64);
