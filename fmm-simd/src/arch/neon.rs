//! aarch64 NEON (128-bit vectors: 4 × f32, 2 × f64): the vector layer, the inverse
//! square root and the entry points.
//!
//! NEON is part of the aarch64 base architecture, but on the pinned toolchain an
//! intrinsic is safe to call only inside a function that lists `neon` in its own
//! `#[target_feature]` (crate documentation, "Toolchain"). The layer methods are
//! `#[inline(always)]` and cannot list it, so each intrinsic call is an `unsafe`
//! block; the entry points list it, and the dispatch calls them in `unsafe`.
//!
//! **Inverse square root** (signed off in Phase 3S T2, spikes/p2p-simd/SPIKE_REPORT.md,
//! "Recommendation", item 2): `sqrt` then a division, FSQRT and FDIV, in both
//! precisions, not the estimate FRSQRTE with Newton steps. FSQRT and FDIV run on the
//! divider beside the four FP pipes, which the rest of the P2P kernel keeps busy, so
//! inside the kernel this route was faster than every estimate route that meets the
//! contract, by 19–53% on the Apple M3 Max. Both operations are correctly rounded, so
//! the error is at most (1 + u)^(3/2) (1 + u) − 1 ≈ 1.5 u_T relative, everywhere in the
//! domain; the spike measured 1.50 u₃₂ (every f32 in [1, 4)) and 1.50 u₆₄ (10⁷ samples),
//! and the tests of `tests/rsqrt` measure it again. At x = 0 it gives +∞, which the
//! mask clears. The result equals the scalar path's 1/√x bit for bit.

use std::arch::aarch64::{
    float32x4_t, float32x4x3_t, float64x2_t, float64x2x3_t, uint32x4_t, uint64x2_t, vaddq_f32,
    vaddq_f64, vbicq_u32, vbicq_u64, vceqq_f32, vceqq_f64, vdivq_f32, vdivq_f64, vdupq_n_f32,
    vdupq_n_f64, vfmaq_f32, vfmaq_f64, vfmsq_f32, vfmsq_f64, vld1q_f32, vld1q_f64, vld3q_dup_f32,
    vld3q_dup_f64, vld3q_f32, vld3q_f64, vmulq_f32, vmulq_f64, vreinterpretq_f32_u32,
    vreinterpretq_f64_u64, vreinterpretq_u32_f32, vreinterpretq_u64_f64, vrsqrteq_f32,
    vrsqrteq_f64, vsqrtq_f32, vsqrtq_f64, vst1q_f32, vst1q_f64, vst3q_f32, vst3q_f64, vsubq_f32,
    vsubq_f64,
};

use super::p2p::{p2p_body, p2p_dipoles_body, p2p_many_body};
use super::{Simd, matrix::accumulate_body, rsqrt_body};
use crate::Isa;

/// The token of NEON: a value exists only on a CPU with NEON, so the [`Simd`] methods
/// that take it are safe.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neon(());

impl Neon {
    /// The token, if this machine has NEON ([`Isa::Neon`] is available).
    #[cfg(test)]
    pub(crate) fn try_new() -> Option<Self> {
        Isa::Neon.is_available().then_some(Self(()))
    }

    /// The token, unchecked.
    ///
    /// # Safety
    ///
    /// The CPU must have NEON.
    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Simd<f32> for Neon {
    type V = float32x4_t;
    type Mask = uint32x4_t;
    const W: usize = 4;
    const ISA: Isa = Isa::Neon;

    #[inline(always)]
    fn splat(self, x: f32) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vdupq_n_f32(x) }
    }

    #[inline(always)]
    fn load(self, x: &[f32]) -> float32x4_t {
        assert!(x.len() >= 4, "`x` holds fewer than 4 values");
        // SAFETY: the token proves NEON, and `x` holds at least the 4 values read.
        unsafe { vld1q_f32(x.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, v: float32x4_t, out: &mut [f32]) {
        assert!(out.len() >= 4, "`out` holds fewer than 4 values");
        // SAFETY: the token proves NEON, and `out` holds at least the 4 values written.
        unsafe { vst1q_f32(out.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn add(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vaddq_f32(a, b) }
    }

    #[inline(always)]
    fn sub(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vsubq_f32(a, b) }
    }

    #[inline(always)]
    fn mul(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vmulq_f32(a, b) }
    }

    #[inline(always)]
    fn fma(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vfmaq_f32(c, a, b) }
    }

    #[inline(always)]
    fn fnma(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vfmsq_f32(c, a, b) }
    }

    #[inline(always)]
    fn eq(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vceqq_f32(a, b) }
    }

    #[inline(always)]
    fn and_not(self, mask: uint32x4_t, v: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(v), mask)) }
    }

    #[inline(always)]
    fn rsqrt_estimate(self, x: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vrsqrteq_f32(x) }
    }

    #[inline(always)]
    fn rsqrt(self, x: float32x4_t) -> float32x4_t {
        // SAFETY: the token proves NEON.
        unsafe { vdivq_f32(vdupq_n_f32(1.0), vsqrtq_f32(x)) }
    }

    #[inline(always)]
    fn load3(self, points: &[[f32; 3]]) -> [float32x4_t; 3] {
        assert!(points.len() >= 4, "`points` holds fewer than 4 triples");
        // SAFETY: the token proves NEON, and `points` holds at least the 4 triples,
        // 12 contiguous values, read.
        let v = unsafe { vld3q_f32(points.as_ptr().cast::<f32>()) };
        [v.0, v.1, v.2]
    }

    #[inline(always)]
    fn store3(self, v: [float32x4_t; 3], points: &mut [[f32; 3]]) {
        assert!(points.len() >= 4, "`points` holds fewer than 4 triples");
        // SAFETY: the token proves NEON, and `points` holds at least the 4 triples,
        // 12 contiguous values, written.
        unsafe {
            vst3q_f32(
                points.as_mut_ptr().cast::<f32>(),
                float32x4x3_t(v[0], v[1], v[2]),
            )
        }
    }

    #[inline(always)]
    fn broadcast(self, point: &[f32; 3], charge: f32) -> ([float32x4_t; 3], float32x4_t) {
        // SAFETY: the token proves NEON, and `point` holds the 3 values read.
        let p = unsafe { vld3q_dup_f32(point.as_ptr()) };
        ([p.0, p.1, p.2], self.splat(charge))
    }
}

impl Simd<f64> for Neon {
    type V = float64x2_t;
    type Mask = uint64x2_t;
    const W: usize = 2;
    const ISA: Isa = Isa::Neon;

    #[inline(always)]
    fn splat(self, x: f64) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vdupq_n_f64(x) }
    }

    #[inline(always)]
    fn load(self, x: &[f64]) -> float64x2_t {
        assert!(x.len() >= 2, "`x` holds fewer than 2 values");
        // SAFETY: the token proves NEON, and `x` holds at least the 2 values read.
        unsafe { vld1q_f64(x.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, v: float64x2_t, out: &mut [f64]) {
        assert!(out.len() >= 2, "`out` holds fewer than 2 values");
        // SAFETY: the token proves NEON, and `out` holds at least the 2 values written.
        unsafe { vst1q_f64(out.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn add(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vaddq_f64(a, b) }
    }

    #[inline(always)]
    fn sub(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vsubq_f64(a, b) }
    }

    #[inline(always)]
    fn mul(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vmulq_f64(a, b) }
    }

    #[inline(always)]
    fn fma(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vfmaq_f64(c, a, b) }
    }

    #[inline(always)]
    fn fnma(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vfmsq_f64(c, a, b) }
    }

    #[inline(always)]
    fn eq(self, a: float64x2_t, b: float64x2_t) -> uint64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vceqq_f64(a, b) }
    }

    #[inline(always)]
    fn and_not(self, mask: uint64x2_t, v: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v), mask)) }
    }

    #[inline(always)]
    fn rsqrt_estimate(self, x: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vrsqrteq_f64(x) }
    }

    #[inline(always)]
    fn rsqrt(self, x: float64x2_t) -> float64x2_t {
        // SAFETY: the token proves NEON.
        unsafe { vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(x)) }
    }

    #[inline(always)]
    fn load3(self, points: &[[f64; 3]]) -> [float64x2_t; 3] {
        assert!(points.len() >= 2, "`points` holds fewer than 2 triples");
        // SAFETY: the token proves NEON, and `points` holds at least the 2 triples,
        // 6 contiguous values, read.
        let v = unsafe { vld3q_f64(points.as_ptr().cast::<f64>()) };
        [v.0, v.1, v.2]
    }

    #[inline(always)]
    fn store3(self, v: [float64x2_t; 3], points: &mut [[f64; 3]]) {
        assert!(points.len() >= 2, "`points` holds fewer than 2 triples");
        // SAFETY: the token proves NEON, and `points` holds at least the 2 triples,
        // 6 contiguous values, written.
        unsafe {
            vst3q_f64(
                points.as_mut_ptr().cast::<f64>(),
                float64x2x3_t(v[0], v[1], v[2]),
            )
        }
    }

    #[inline(always)]
    fn broadcast(self, point: &[f64; 3], charge: f64) -> ([float64x2_t; 3], float64x2_t) {
        // SAFETY: the token proves NEON, and `point` holds the 3 values read.
        let p = unsafe { vld3q_dup_f64(point.as_ptr()) };
        ([p.0, p.1, p.2], self.splat(charge))
    }
}

/// Generates the inverse-square-root entry point of one precision.
macro_rules! rsqrt_entry {
    ($name:ident, $t:ty) => {
        #[doc = concat!(
                    "The kernel's inverse square root, masked at 0, on NEON in ", stringify!($t),
                    ": `out[i] = rsqrt(x[i])`, with lengths checked by the callee."
                )]
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $name(x: &[$t], out: &mut [$t]) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            rsqrt_body(s, x, out);
        }
    };
}

rsqrt_entry!(rsqrt_slice_f32, f32);
rsqrt_entry!(rsqrt_slice_f64, f64);

/// Target vectors per P2P block in f32 (signed off in Phase 3S T2,
/// spikes/p2p-simd/SPIKE_REPORT.md, "Recommendation", item 3).
pub(crate) const K_F32: usize = 2;
/// Target vectors per P2P block in f64 (as [`K_F32`]).
pub(crate) const K_F64: usize = 4;

/// Generates the P2P entry points of one precision, potential only and with gradients.
macro_rules! p2p_entries {
    ($potential:ident, $gradient:ident, $t:ty, $k:expr) => {
        /// P2P on NEON, potential only (`arch::p2p::p2p_body`), in the precision of
        /// its name, with lengths checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            p2p_body::<$t, Neon, { $k }, false>(s, sources, charges, targets, potential, &mut []);
        }

        /// P2P on NEON, potential and gradient (`arch::p2p::p2p_body`), in the
        /// precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            p2p_body::<$t, Neon, { $k }, true>(s, sources, charges, targets, potential, gradient);
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
        /// Dipole P2P on NEON, potential only (`arch::p2p::p2p_dipoles_body`), in the
        /// precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            p2p_dipoles_body::<$t, Neon, DIPOLE_K, false>(
                s,
                sources,
                dipoles,
                targets,
                potential,
                &mut [],
            );
        }

        /// Dipole P2P on NEON, potential and gradient (`arch::p2p::p2p_dipoles_body`), in
        /// the precision of its name, with lengths checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            p2p_dipoles_body::<$t, Neon, DIPOLE_K, true>(
                s, sources, dipoles, targets, potential, gradient,
            );
        }
    };
}

p2p_dipole_entries!(p2p_dipoles_f32, p2p_dipoles_f32_gradient, f32);
p2p_dipole_entries!(p2p_dipoles_f64, p2p_dipoles_f64_gradient, f64);

/// Target vectors per block of the P2P of several charge vectors (`arch::p2p::p2p_many_body`;
/// Phase 6 T5), potential only: measured on the M3 Max (64 sources and targets, m = 8 and
/// 32), 2 target vectors with groups of 8 vectors ran 4.6× (f32) and 4.1× (f64) faster per
/// vector than m calls of the single-vector kernel, against 4.4× and 4.0× with 1 and 8,
/// and 3.1× and 2.8× with 2 and 4.
pub(crate) const MANY_K: usize = 2;
/// Vectors per group of the P2P of several charge vectors, potential only (as [`MANY_K`]).
pub(crate) const MANY_VG: usize = 8;
/// Target vectors per block with gradients: 1 target vector with groups of 4 vectors
/// measured 1.8× (f32) and 1.9× (f64) faster per vector, against 1.2× with 2 and 2 and
/// 1.4–1.5× with 1 and 2 (16 accumulators per group; 8 vectors would not fit the 32
/// registers).
pub(crate) const MANY_K_GRADIENT: usize = 1;
/// Vectors per group with gradients (as [`MANY_K_GRADIENT`]).
pub(crate) const MANY_VG_GRADIENT: usize = 4;

/// Generates the entry points of the P2P of several charge vectors of one precision,
/// potential only and with gradients (Phase 6 T5).
macro_rules! p2p_many_entries {
    ($potential:ident, $gradient:ident, $t:ty) => {
        /// P2P of several charge vectors on NEON, potential only
        /// (`arch::p2p::p2p_many_body`), in the precision of its name, with lengths
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $potential(
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
        ) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            p2p_many_body::<$t, Neon, MANY_K, MANY_VG, false>(
                s,
                sources,
                charges,
                m,
                targets,
                potential,
                &mut [],
            );
        }

        /// P2P of several charge vectors on NEON, potential and gradient
        /// (`arch::p2p::p2p_many_body`), in the precision of its name, with lengths
        /// checked by the caller.
        ///
        /// Not inlined, as the potential-only entry point.
        #[target_feature(enable = "neon")]
        #[inline(never)]
        pub(crate) fn $gradient(
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: &mut [[$t; 3]],
        ) {
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            p2p_many_body::<$t, Neon, MANY_K_GRADIENT, MANY_VG_GRADIENT, true>(
                s, sources, charges, m, targets, potential, gradient,
            );
        }
    };
}

p2p_many_entries!(p2p_many_f32, p2p_many_f32_gradient, f32);
p2p_many_entries!(p2p_many_f64, p2p_many_f64_gradient, f64);

/// Row vectors per block of the column-accumulation kernel in f32 (`arch::matrix`):
/// 32 vector registers: 16 accumulators of 4 × 4 (f32: 16 rows; f64: 8 rows), 4 of A
/// and the broadcast.
pub(crate) const MATRIX_ROWS_F32: usize = 4;
/// Row vectors per block of the column-accumulation kernel in f64 (as
/// [`MATRIX_ROWS_F32`]).
pub(crate) const MATRIX_ROWS_F64: usize = 4;

/// Generates the column-accumulation entry points of one precision, separate and fused.
macro_rules! accumulate_entries {
    ($separate:ident, $fused:ident, $t:ty, $r:expr) => {
        /// C[:, c_j] += A · B[:, b_j] on NEON, multiply and add rounded apart
        /// (`arch::matrix::accumulate_body`), in the precision of its name, with shapes
        /// checked by the caller.
        ///
        /// Not inlined, so that the dispatch stays one call and the inlining check
        /// finds the function by name.
        #[target_feature(enable = "neon")]
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
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            accumulate_body::<$t, Neon, { $r }, false>(s, n, a, b, b_offsets, c, c_offsets, pack);
        }

        /// The fused form of the separate entry point: one fma per step.
        ///
        /// Not inlined, as the separate entry point.
        #[target_feature(enable = "neon")]
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
            // SAFETY: this function enables `neon`, so it runs only on a CPU with NEON:
            // a call from outside a NEON context is `unsafe` and asserts it.
            let s = unsafe { Neon::new_unchecked() };
            accumulate_body::<$t, Neon, { $r }, true>(s, n, a, b, b_offsets, c, c_offsets, pack);
        }
    };
}

accumulate_entries!(accumulate_f32, accumulate_f32_fused, f32, MATRIX_ROWS_F32);
accumulate_entries!(accumulate_f64, accumulate_f64_fused, f64, MATRIX_ROWS_F64);
