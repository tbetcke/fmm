//! NEON vector types: f32 × 4 and f64 × 2. NEON is part of the aarch64 base
//! architecture. Every intrinsic call still needs `unsafe` here, because these methods
//! carry no `#[target_feature(enable = "neon")]` (rustc 1.98: the feature being enabled
//! for the target does not make the call safe).

use core::arch::aarch64::*;

use super::Vf;

/// Four f32 lanes.
#[derive(Clone, Copy)]
pub struct F32x4(float32x4_t);

/// Two f64 lanes.
#[derive(Clone, Copy)]
pub struct F64x2(float64x2_t);

impl Vf for F32x4 {
    type E = f32;
    const W: usize = 4;
    const ISA: &'static str = "neon";
    const EST_OPS: usize = 1;
    const STEP_PREP_OPS: usize = 0;

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vdupq_n_f32(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        // SAFETY: the caller guarantees 4 readable lanes at `p`.
        Self(unsafe { vld1q_f32(p) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        // SAFETY: the caller guarantees 4 writable lanes at `p`.
        unsafe { vst1q_f32(p, self.0) }
    }
    #[inline(always)]
    unsafe fn splat3(p: *const f32) -> [Self; 3] {
        // SAFETY: the caller guarantees 3 readable values at `p`.
        let v = unsafe { vld3q_dup_f32(p) };
        [Self(v.0), Self(v.1), Self(v.2)]
    }
    #[inline(always)]
    unsafe fn load3(p: *const f32) -> [Self; 3] {
        // SAFETY: the caller guarantees 12 readable values at `p`.
        let v = unsafe { vld3q_f32(p) };
        [Self(v.0), Self(v.1), Self(v.2)]
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vaddq_f32(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn sub(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vsubq_f32(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vmulq_f32(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn fma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vfmaq_f32(c.0, a.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn fnma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vfmsq_f32(c.0, a.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn mask_zero(r2: Self, rho: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe {
            Self(vreinterpretq_f32_u32(vbicq_u32(
                vreinterpretq_u32_f32(rho.0),
                vceqzq_f32(r2.0),
            )))
        }
    }
    #[inline(always)]
    unsafe fn est(x: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vrsqrteq_f32(x.0)) }
    }
    #[inline(always)]
    unsafe fn step_prep(x: Self) -> Self {
        x
    }
    #[inline(always)]
    unsafe fn step(p: Self, y2: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vrsqrtsq_f32(p.0, y2.0)) }
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vsqrtq_f32(self.0)) }
    }
    #[inline(always)]
    unsafe fn div(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vdivq_f32(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn hsum(self) -> f32 {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { vaddvq_f32(self.0) }
    }
}

impl Vf for F64x2 {
    type E = f64;
    const W: usize = 2;
    const ISA: &'static str = "neon";
    const EST_OPS: usize = 1;
    const STEP_PREP_OPS: usize = 0;

    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vdupq_n_f64(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        // SAFETY: the caller guarantees 2 readable lanes at `p`.
        Self(unsafe { vld1q_f64(p) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        // SAFETY: the caller guarantees 2 writable lanes at `p`.
        unsafe { vst1q_f64(p, self.0) }
    }
    #[inline(always)]
    unsafe fn splat3(p: *const f64) -> [Self; 3] {
        // SAFETY: the caller guarantees 3 readable values at `p`.
        let v = unsafe { vld3q_dup_f64(p) };
        [Self(v.0), Self(v.1), Self(v.2)]
    }
    #[inline(always)]
    unsafe fn load3(p: *const f64) -> [Self; 3] {
        // SAFETY: the caller guarantees 6 readable values at `p`.
        let v = unsafe { vld3q_f64(p) };
        [Self(v.0), Self(v.1), Self(v.2)]
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vaddq_f64(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn sub(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vsubq_f64(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vmulq_f64(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn fma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vfmaq_f64(c.0, a.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn fnma(a: Self, b: Self, c: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vfmsq_f64(c.0, a.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn mask_zero(r2: Self, rho: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe {
            Self(vreinterpretq_f64_u64(vbicq_u64(
                vreinterpretq_u64_f64(rho.0),
                vceqzq_f64(r2.0),
            )))
        }
    }
    #[inline(always)]
    unsafe fn est(x: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vrsqrteq_f64(x.0)) }
    }
    #[inline(always)]
    unsafe fn step_prep(x: Self) -> Self {
        x
    }
    #[inline(always)]
    unsafe fn step(p: Self, y2: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vrsqrtsq_f64(p.0, y2.0)) }
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vsqrtq_f64(self.0)) }
    }
    #[inline(always)]
    unsafe fn div(self, b: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { Self(vdivq_f64(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn hsum(self) -> f64 {
        // SAFETY: NEON is part of the aarch64 base architecture.
        unsafe { vaddvq_f64(self.0) }
    }
}
