//! The spike's thin vector layer: one trait over the vector types of each ISA, and the
//! inverse-square-root candidates of design simd-p2p.md §4.3, written once over it.
//!
//! Every method is `#[inline(always)]` and `unsafe`: on x86_64 the intrinsics need the
//! `avx2` and `fma` features, which only the `#[target_feature]` entry points in
//! `kernels` and `rsqrt_study` enable. Callers must run on a CPU with the ISA of the
//! vector type, and pointers must be valid for the lanes read or written.

use nd_fmm_math::RealScalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;

/// f32 or f64, with the constants the study needs.
pub trait Elem: RealScalar + std::fmt::Debug + Default {
    /// Unit roundoff u_T: 2⁻²⁴ or 2⁻⁵³.
    const U: f64;
    /// "f32" or "f64".
    const NAME: &'static str;
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;
}

impl Elem for f32 {
    const U: f64 = 1.0 / 16_777_216.0;
    const NAME: &'static str = "f32";
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
}

impl Elem for f64 {
    const U: f64 = 1.0 / 9_007_199_254_740_992.0;
    const NAME: &'static str = "f64";
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
}

/// A vector of `W` lanes of `E`.
#[allow(clippy::missing_safety_doc)]
pub trait Vf: Copy {
    /// The lane type.
    type E: Elem;
    /// The lane count.
    const W: usize;
    /// The ISA, for reports.
    const ISA: &'static str;
    /// FP operations of the estimate (x86 f64: conversion, estimate, conversion).
    const EST_OPS: usize;
    /// FP operations of `step_prep`: 0 on NEON, 1 elsewhere.
    const STEP_PREP_OPS: usize;

    unsafe fn splat(x: Self::E) -> Self;
    unsafe fn load(p: *const Self::E) -> Self;
    unsafe fn store(self, p: *mut Self::E);
    /// Broadcasts one source triple to three vectors (NEON `ld3r`).
    unsafe fn splat3(p: *const Self::E) -> [Self; 3];
    /// Loads `W` interleaved triples into x, y and z vectors (NEON `ld3`).
    unsafe fn load3(p: *const Self::E) -> [Self; 3];
    unsafe fn add(self, b: Self) -> Self;
    unsafe fn sub(self, b: Self) -> Self;
    unsafe fn mul(self, b: Self) -> Self;
    /// a·b + c, fused.
    unsafe fn fma(a: Self, b: Self, c: Self) -> Self;
    /// c − a·b, fused.
    unsafe fn fnma(a: Self, b: Self, c: Self) -> Self;
    /// `rho` where `r2 != 0`, else +0 (compare and and-not).
    unsafe fn mask_zero(r2: Self, rho: Self) -> Self;
    /// The hardware estimate of 1/√x.
    unsafe fn est(x: Self) -> Self;
    /// What the step form needs of x: x on NEON (FRSQRTS takes x), ½x elsewhere.
    unsafe fn step_prep(x: Self) -> Self;
    /// (3 − x·y²)/2 from `step_prep(x)` and y²: one FRSQRTS on NEON, one fma elsewhere.
    unsafe fn step(p: Self, y2: Self) -> Self;
    unsafe fn sqrt(self) -> Self;
    unsafe fn div(self, b: Self) -> Self;
    /// The sum of the lanes.
    unsafe fn hsum(self) -> Self::E;
}

/// One inverse-square-root formulation, without the r² = 0 mask.
pub trait Rsqrt<V: Vf> {
    /// Name in the reports.
    const NAME: &'static str;
    /// Vector FP operations per evaluation, including the estimate.
    const OPS: usize;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn rsqrt(x: V) -> V;
}

/// Estimate and `N` Newton steps in the fma form: h = ½x once, then per step
/// y² (mul), t = ½ − h y² (fnma), y + y t (fma).
pub struct Newton<const N: usize>;

impl<V: Vf, const N: usize> Rsqrt<V> for Newton<N> {
    const NAME: &'static str = ["est", "est+N1", "est+N2", "est+N3", "est+N4"][N];
    const OPS: usize = V::EST_OPS + 1 + 3 * N;
    #[inline(always)]
    unsafe fn rsqrt(x: V) -> V {
        // SAFETY: the caller runs on the vector type's ISA (module docs).
        unsafe {
            let h = x.mul(V::splat(V::E::from_f64(0.5)));
            let half = V::splat(V::E::from_f64(0.5));
            let mut y = V::est(x);
            for _ in 0..N {
                let t = V::fnma(h, y.mul(y), half);
                y = V::fma(y, t, y);
            }
            y
        }
    }
}

/// Estimate and `N` steps in the step form, y · (3 − x y²)/2: FRSQRTS on NEON
/// (green-kernels on NEON), the rlst form 1.5 − ½x y² on x86 (green-kernels on AVX2).
pub struct Steps<const N: usize>;

impl<V: Vf, const N: usize> Rsqrt<V> for Steps<N> {
    const NAME: &'static str = ["est", "est+S1", "est+S2", "est+S3", "est+S4"][N];
    // NEON: mul, FRSQRTS, mul per step; x86: + one mul for ½x.
    const OPS: usize = V::EST_OPS + V::STEP_PREP_OPS + 3 * N;
    #[inline(always)]
    unsafe fn rsqrt(x: V) -> V {
        // SAFETY: the caller runs on the vector type's ISA (module docs).
        unsafe {
            let p = V::step_prep(x);
            let mut y = V::est(x);
            for _ in 0..N {
                y = y.mul(V::step(p, y.mul(y)));
            }
            y
        }
    }
}

/// Coefficients of (1 − r)^(−½) = Σ cₖ rᵏ: cₖ = C(2k, k) / 4ᵏ.
pub const POLY: [f64; 9] = [
    1.0,
    0.5,
    0.375,
    0.3125,
    0.2734375,
    0.24609375,
    0.2255859375,
    0.20947265625,
    0.196380615234375,
];

/// Estimate and the polynomial correction of design §4.3 truncated after r^M:
/// r = 1 − x y² (mul, fnma), p = c₁ + c₂ r + … + c_M r^(M−1) (Horner, M − 1 fma),
/// y + (y r) p (mul, fma). Ops: estimate + M + 3 (M = 1 is a Newton step).
pub struct Poly<const M: usize>;

impl<V: Vf, const M: usize> Rsqrt<V> for Poly<M> {
    const NAME: &'static str = [
        "", "est+P1", "est+P2", "est+P3", "est+P4", "est+P5", "est+P6", "est+P7", "est+P8",
    ][M];
    const OPS: usize = V::EST_OPS + M + 3;
    #[inline(always)]
    unsafe fn rsqrt(x: V) -> V {
        // SAFETY: the caller runs on the vector type's ISA (module docs).
        unsafe { poly_correct::<V, M>(x, V::est(x)) }
    }
}

/// One correction of `y` ≈ 1/√x by the polynomial of order M.
#[inline(always)]
unsafe fn poly_correct<V: Vf, const M: usize>(x: V, y: V) -> V {
    // SAFETY: the caller runs on the vector type's ISA (module docs).
    unsafe {
        let one = V::splat(V::E::from_f64(1.0));
        let r = V::fnma(x, y.mul(y), one);
        let mut p = V::splat(V::E::from_f64(POLY[M]));
        for k in (1..M).rev() {
            p = V::fma(p, r, V::splat(V::E::from_f64(POLY[k])));
        }
        V::fma(y.mul(r), p, y)
    }
}

/// `N` Newton steps (fma form), then the polynomial of order M.
pub struct NewtonPoly<const N: usize, const M: usize>;

impl<V: Vf, const N: usize, const M: usize> Rsqrt<V> for NewtonPoly<N, M> {
    const NAME: &'static str = match (N, M) {
        (1, 2) => "est+N1+P2",
        (1, 3) => "est+N1+P3",
        (2, 2) => "est+N2+P2",
        _ => "est+Nn+Pm",
    };
    const OPS: usize = V::EST_OPS + 1 + 3 * N + M + 3;
    #[inline(always)]
    unsafe fn rsqrt(x: V) -> V {
        // SAFETY: the caller runs on the vector type's ISA (module docs).
        unsafe {
            let y = <Newton<N> as Rsqrt<V>>::rsqrt(x);
            poly_correct::<V, M>(x, y)
        }
    }
}

/// 1/√x by `sqrt` then a division: the yardstick and the scalar path.
pub struct SqrtDiv;

impl<V: Vf> Rsqrt<V> for SqrtDiv {
    const NAME: &'static str = "sqrt+div";
    const OPS: usize = 2;
    #[inline(always)]
    unsafe fn rsqrt(x: V) -> V {
        // SAFETY: the caller runs on the vector type's ISA (module docs).
        unsafe { V::splat(V::E::from_f64(1.0)).div(x.sqrt()) }
    }
}
