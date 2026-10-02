//! AVX2 + FMA vector types: f32 × 8 and f64 × 4, and the integer bit-trick estimate
//! for f64. Every method calls intrinsics that need `avx`, `avx2` or `fma`; the
//! callers are inlined into `#[target_feature(enable = "avx2,fma")]` entry points and
//! run only after `is_x86_feature_detected!` confirmed both.

use core::arch::x86_64::*;

use super::{Rsqrt, Vf};

/// Eight f32 lanes.
#[derive(Clone, Copy)]
pub struct F32x8(__m256);

/// Four f64 lanes.
#[derive(Clone, Copy)]
pub struct F64x4(__m256d);

impl Vf for F32x8 {
    type E = f32;
    const W: usize = 8;
    const ISA: &'static str = "avx2";
    const EST_OPS: usize = 1;
    const STEP_PREP_OPS: usize = 1;

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_set1_ps(x) })
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        // SAFETY: AVX is available; the caller guarantees 8 readable lanes at `p`.
        Self(unsafe { _mm256_loadu_ps(p) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        // SAFETY: AVX is available; the caller guarantees 8 writable lanes at `p`.
        unsafe { _mm256_storeu_ps(p, self.0) }
    }
    #[inline(always)]
    unsafe fn splat3(p: *const f32) -> [Self; 3] {
        // SAFETY: AVX is available; the caller guarantees 3 readable values at `p`.
        unsafe {
            [
                Self(_mm256_broadcast_ss(&*p)),
                Self(_mm256_broadcast_ss(&*p.add(1))),
                Self(_mm256_broadcast_ss(&*p.add(2))),
            ]
        }
    }
    #[inline(always)]
    unsafe fn load3(p: *const f32) -> [Self; 3] {
        // SAFETY: AVX is available; the caller guarantees 24 readable values at `p`.
        unsafe {
            let g = |c: usize| {
                _mm256_setr_ps(
                    *p.add(c),
                    *p.add(3 + c),
                    *p.add(6 + c),
                    *p.add(9 + c),
                    *p.add(12 + c),
                    *p.add(15 + c),
                    *p.add(18 + c),
                    *p.add(21 + c),
                )
            };
            [Self(g(0)), Self(g(1)), Self(g(2))]
        }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_add_ps(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn sub(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_sub_ps(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_mul_ps(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn fma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: FMA is available (module docs).
        Self(unsafe { _mm256_fmadd_ps(a.0, b.0, c.0) })
    }
    #[inline(always)]
    unsafe fn fnma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: FMA is available (module docs).
        Self(unsafe { _mm256_fnmadd_ps(a.0, b.0, c.0) })
    }
    #[inline(always)]
    unsafe fn mask_zero(r2: Self, rho: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        unsafe {
            let zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(r2.0, _mm256_setzero_ps());
            Self(_mm256_andnot_ps(zero, rho.0))
        }
    }
    #[inline(always)]
    unsafe fn est(x: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_rsqrt_ps(x.0) })
    }
    #[inline(always)]
    unsafe fn step_prep(x: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        unsafe { x.mul(Self::splat(0.5)) }
    }
    #[inline(always)]
    unsafe fn step(p: Self, y2: Self) -> Self {
        // SAFETY: AVX and FMA are available (module docs).
        unsafe { Self::fnma(p, y2, Self::splat(1.5)) }
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }
    #[inline(always)]
    unsafe fn div(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_div_ps(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn hsum(self) -> f32 {
        let mut a = [0.0_f32; 8];
        // SAFETY: AVX is available; `a` has 8 lanes.
        unsafe { self.store(a.as_mut_ptr()) };
        ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
    }
}

impl Vf for F64x4 {
    type E = f64;
    const W: usize = 4;
    const ISA: &'static str = "avx2";
    // Convert to f32, rsqrtps on 4 lanes, convert back.
    const EST_OPS: usize = 3;
    const STEP_PREP_OPS: usize = 1;

    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_set1_pd(x) })
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        // SAFETY: AVX is available; the caller guarantees 4 readable lanes at `p`.
        Self(unsafe { _mm256_loadu_pd(p) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        // SAFETY: AVX is available; the caller guarantees 4 writable lanes at `p`.
        unsafe { _mm256_storeu_pd(p, self.0) }
    }
    #[inline(always)]
    unsafe fn splat3(p: *const f64) -> [Self; 3] {
        // SAFETY: AVX is available; the caller guarantees 3 readable values at `p`.
        unsafe {
            [
                Self(_mm256_broadcast_sd(&*p)),
                Self(_mm256_broadcast_sd(&*p.add(1))),
                Self(_mm256_broadcast_sd(&*p.add(2))),
            ]
        }
    }
    #[inline(always)]
    unsafe fn load3(p: *const f64) -> [Self; 3] {
        // SAFETY: AVX is available; the caller guarantees 12 readable values at `p`.
        unsafe {
            let g =
                |c: usize| _mm256_setr_pd(*p.add(c), *p.add(3 + c), *p.add(6 + c), *p.add(9 + c));
            [Self(g(0)), Self(g(1)), Self(g(2))]
        }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_add_pd(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn sub(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_sub_pd(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_mul_pd(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn fma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: FMA is available (module docs).
        Self(unsafe { _mm256_fmadd_pd(a.0, b.0, c.0) })
    }
    #[inline(always)]
    unsafe fn fnma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: FMA is available (module docs).
        Self(unsafe { _mm256_fnmadd_pd(a.0, b.0, c.0) })
    }
    #[inline(always)]
    unsafe fn mask_zero(r2: Self, rho: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        unsafe {
            let zero = _mm256_cmp_pd::<_CMP_EQ_OQ>(r2.0, _mm256_setzero_pd());
            Self(_mm256_andnot_pd(zero, rho.0))
        }
    }
    #[inline(always)]
    unsafe fn est(x: Self) -> Self {
        // SAFETY: AVX is available (module docs). The kernel domain (CONVENTIONS
        // §3.13) lies in the normal range of f32, so the conversion neither overflows
        // nor produces a subnormal.
        unsafe { Self(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x.0)))) }
    }
    #[inline(always)]
    unsafe fn step_prep(x: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        unsafe { x.mul(Self::splat(0.5)) }
    }
    #[inline(always)]
    unsafe fn step(p: Self, y2: Self) -> Self {
        // SAFETY: AVX and FMA are available (module docs).
        unsafe { Self::fnma(p, y2, Self::splat(1.5)) }
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_sqrt_pd(self.0) })
    }
    #[inline(always)]
    unsafe fn div(self, b: Self) -> Self {
        // SAFETY: AVX is available (module docs).
        Self(unsafe { _mm256_div_pd(self.0, b.0) })
    }
    #[inline(always)]
    unsafe fn hsum(self) -> f64 {
        let mut a = [0.0_f64; 4];
        // SAFETY: AVX is available; `a` has 4 lanes.
        unsafe { self.store(a.as_mut_ptr()) };
        (a[0] + a[1]) + (a[2] + a[3])
    }
}

/// f64 without the f32 detour: the integer bit-trick guess
/// y₀ = bits⁻¹(0x5FE6EB50C7B537A9 − (bits(x) ≫ 1)) (relative error ≤ 3.42e-2,
/// Lomont), then `N` Newton steps in the fma form.
pub struct BitTrick<const N: usize>;

impl<const N: usize> Rsqrt<F64x4> for BitTrick<N> {
    const NAME: &'static str = [
        "bits", "bits+N1", "bits+N2", "bits+N3", "bits+N4", "bits+N5",
    ][N];
    // Shift and subtract (integer), ½x, 3 per step.
    const OPS: usize = 2 + 1 + 3 * N;
    #[inline(always)]
    unsafe fn rsqrt(x: F64x4) -> F64x4 {
        // SAFETY: AVX2 and FMA are available (module docs).
        unsafe {
            let bits = _mm256_castpd_si256(x.0);
            let magic = _mm256_set1_epi64x(0x5FE6_EB50_C7B5_37A9);
            let guess = _mm256_sub_epi64(magic, _mm256_srli_epi64::<1>(bits));
            let mut y = F64x4(_mm256_castsi256_pd(guess));
            let h = x.mul(F64x4::splat(0.5));
            let half = F64x4::splat(0.5);
            for _ in 0..N {
                let t = F64x4::fnma(h, y.mul(y), half);
                y = F64x4::fma(y, t, y);
            }
            y
        }
    }
}
