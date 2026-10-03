//! The per-ISA code: the vector layer, the inverse square root and the entry points
//! (docs/design/simd-p2p.md §5.3).
//!
//! Each module holds the code of one [`Isa`], compiled only for its architecture, and
//! implements [`Simd`], the crate's thin layer over its vector types:
//!
//! - [`scalar`]: one lane, plain Rust, available everywhere;
//! - `neon` (aarch64): `float32x4_t` and `float64x2_t`;
//! - `avx2` (x86_64): `__m256` and `__m256d`.
//!
//! Code that runs on vectors is written once, generic over [`Simd`]
//! ([`rsqrt_body`] and the P2P kernel body [`p2p::p2p_body`]), and instantiated inside
//! one `#[target_feature]` entry point per ISA, precision and, for P2P, output, so
//! that the layer methods, all `#[inline(always)]`, and their intrinsics inline into a
//! function that has the features. [`rsqrt_slice_f32`] and [`rsqrt_slice_f64`]
//! dispatch to the inverse-square-root entry points, [`p2p_f32`] and [`p2p_f64`] to
//! the P2P ones.
//!
//! The scalar ISA runs P2P by the plain loop [`scalar::p2p`], which equals
//! `nd_fmm_ref::p2p::p2p` bit for bit wherever r² ≠ 0, not by the generic body at
//! W = 1: that body forms r² with fma and the gradient term as (q ρ) ρ² d, so it
//! rounds differently. The tests run the generic body on the scalar layer as well.
//!
//! # Unsafe
//!
//! `unsafe` is allowed only in these modules and in the dispatch (crate documentation,
//! "Unsafe"). The layer is safe to call: an ISA's methods take its token
//! (`neon::Neon`, `avx2::Avx2`) by value, and a token can only be made on a CPU
//! that has the ISA. Inside the methods, every intrinsic call is an `unsafe` block,
//! because an `#[inline(always)]` method cannot carry `#[target_feature]` (crate
//! documentation, "Toolchain"); its `// SAFETY:` comment cites the token, and for
//! loads and stores the slice length the method asserts.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod p2p;
pub(crate) mod scalar;
#[cfg(test)]
mod tests;

use crate::{Isa, IsaUnavailable, SimdScalar};

/// The largest lane count of any ISA (AVX2, f32), the size of the padded stack copies
/// that tails use.
pub(crate) const MAX_LANES: usize = 8;

/// The vector operations of one ISA on `T` (docs/design/simd-p2p.md §5.3).
///
/// Implemented by a zero-sized token per ISA, [`scalar::Scalar`], `neon::Neon` and
/// `avx2::Avx2`, for f32 and f64. A token exists only on a CPU that has its ISA, so
/// every method is safe. [`Self::V`] holds [`Self::W`] lanes. Arithmetic is IEEE 754
/// lane by lane, each operation rounded once (an fma once in all); the vector paths
/// therefore agree with [`scalar::Scalar`] bit for bit in every method except the two
/// inverse square roots.
///
/// Every method is `#[inline(always)]` in every implementation: a method that is not
/// inlined turns its intrinsic into an out-of-line call (crate `CLAUDE.md`, "Inlining
/// check").
pub(crate) trait Simd<T: SimdScalar>: Copy {
    /// A vector of [`Self::W`] lanes of `T`.
    type V: Copy;
    /// The result of a lane-wise comparison: all bits set in a lane where it holds.
    type Mask: Copy;
    /// The number of lanes, [`Isa::lanes`] of [`Self::ISA`].
    const W: usize;
    /// The ISA.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "names the ISA in the tests' messages")
    )]
    const ISA: Isa;

    /// Every lane equal to `x`.
    fn splat(self, x: T) -> Self::V;
    /// The first [`Self::W`] values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` holds fewer than [`Self::W`] values.
    fn load(self, x: &[T]) -> Self::V;
    /// Writes the lanes of `v` to the first [`Self::W`] values of `out`.
    ///
    /// # Panics
    ///
    /// If `out` holds fewer than [`Self::W`] values.
    fn store(self, v: Self::V, out: &mut [T]);
    /// a + b.
    fn add(self, a: Self::V, b: Self::V) -> Self::V;
    /// a − b.
    fn sub(self, a: Self::V, b: Self::V) -> Self::V;
    /// a · b.
    fn mul(self, a: Self::V, b: Self::V) -> Self::V;
    /// a · b + c, fused (one rounding).
    fn fma(self, a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// c − a · b, fused (one rounding).
    fn fnma(self, a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// The lanes where a == b (IEEE: +0 equals −0, NaN equals nothing).
    fn eq(self, a: Self::V, b: Self::V) -> Self::Mask;
    /// `v` where `mask` is clear and +0 where it is set: the bitwise and-not, which
    /// also clears a NaN or an infinity.
    fn and_not(self, mask: Self::Mask, v: Self::V) -> Self::V;
    /// The hardware estimate of 1/√x: FRSQRTE (relative error below 2⁻⁸) on NEON,
    /// `vrsqrtps` (at most 1.5 · 2⁻¹², documented) on AVX2, through f32 for f64; on
    /// the scalar path 1/√x itself. Not the kernel's inverse square root; see
    /// [`Self::rsqrt`].
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "a reference for the tests; no kernel uses it")
    )]
    fn rsqrt_estimate(self, x: Self::V) -> Self::V;
    /// The kernel's inverse square root, 1/√x within 4 u_T relative on the kernel
    /// domain 2⁻¹⁰⁸ ≤ x ≤ 2⁷ (CONVENTIONS §3.13; design §4.3), in the formulation
    /// signed off in Phase 3S T2 for the ISA and precision. Its value at x = 0 is +∞
    /// or NaN; [`Self::rsqrt_masked`] clears it. Outside the domain the result is
    /// unspecified.
    fn rsqrt(self, x: Self::V) -> Self::V;
    /// The first [`Self::W`] triples of `points`, as the vectors of their x, y and z
    /// components (NEON `ld3`).
    ///
    /// # Panics
    ///
    /// If `points` holds fewer than [`Self::W`] triples.
    fn load3(self, points: &[[T; 3]]) -> [Self::V; 3];
    /// Writes the x, y and z vectors `v` to the first [`Self::W`] triples of `points`,
    /// the inverse of [`Self::load3`] (NEON `st3`).
    ///
    /// # Panics
    ///
    /// If `points` holds fewer than [`Self::W`] triples.
    fn store3(self, v: [Self::V; 3], points: &mut [[T; 3]]);
    /// One source in every lane: its three coordinates and its charge (NEON `ld3r`).
    fn broadcast(self, point: &[T; 3], charge: T) -> ([Self::V; 3], Self::V);

    /// The inverse square root with the coincident-pair mask of design §4.4:
    /// [`Self::rsqrt`] where x ≠ 0 and +0 where x = 0.
    #[inline(always)]
    fn rsqrt_masked(self, x: Self::V) -> Self::V {
        let zero = self.eq(x, self.splat(T::zero()));
        self.and_not(zero, self.rsqrt(x))
    }
}

/// `out[i] = s.rsqrt_masked(x[i])` for every i, in whole vectors, and the tail in a
/// stack copy padded with its last value, through the same vector code (design §4.5).
///
/// # Panics
///
/// If `x` and `out` have different lengths.
#[inline(always)]
pub(crate) fn rsqrt_body<T: SimdScalar, S: Simd<T>>(s: S, x: &[T], out: &mut [T]) {
    const { assert!(S::W <= MAX_LANES) };
    assert_eq!(
        x.len(),
        out.len(),
        "`x` and `out` must have the same length"
    );
    let whole = x.len() - x.len() % S::W;
    let (x_whole, rest) = x.split_at(whole);
    let (out_whole, out_rest) = out.split_at_mut(whole);
    for (xv, ov) in x_whole.chunks(S::W).zip(out_whole.chunks_mut(S::W)) {
        s.store(s.rsqrt_masked(s.load(xv)), ov);
    }
    if let Some(&last) = rest.last() {
        let mut padded = [last; MAX_LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let mut result = [T::zero(); MAX_LANES];
        s.store(s.rsqrt_masked(s.load(&padded)), &mut result);
        out_rest.copy_from_slice(&result[..rest.len()]);
    }
}

/// Generates the dispatch of the inverse-square-root entry points for one precision.
macro_rules! rsqrt_dispatch {
    ($name:ident, $t:ty, $entry:ident) => {
        #[doc = concat!(
                    "The kernel's inverse square root, masked at 0, of every value of `x` into ",
                    "`out`, on `isa`, in ", stringify!($t), " (`rsqrt::rsqrt_slice`)."
                )]
        ///
        /// # Panics
        ///
        /// If `isa` is not available on this machine, or `x` and `out` have different
        /// lengths.
        pub(crate) fn $name(isa: Isa, x: &[$t], out: &mut [$t]) {
            if !isa.is_available() {
                panic!("{}", IsaUnavailable { isa });
            }
            match isa {
                Isa::Scalar => rsqrt_body(scalar::Scalar, x, out),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (checked above), so the CPU has NEON,
                // the feature the entry point enables.
                Isa::Neon => unsafe { neon::$entry(x, out) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (checked above), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                Isa::Avx2 => unsafe { avx2::$entry(x, out) },
                // The ISAs of other architectures, which are never available.
                _ => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

rsqrt_dispatch!(rsqrt_slice_f32, f32, rsqrt_slice_f32);
rsqrt_dispatch!(rsqrt_slice_f64, f64, rsqrt_slice_f64);

/// An [`Isa`] that this machine can run: the proof that makes the dispatch to the
/// `#[target_feature]` P2P entry points sound.
///
/// [`AvailableIsa::new`] checks [`Isa::is_available`] once, so that
/// [`P2pKernel`](crate::P2pKernel), which holds one, dispatches every call without
/// checking again (docs/design/simd-p2p.md §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvailableIsa(Isa);

impl AvailableIsa {
    /// `isa`, if this machine can run it.
    pub(crate) fn new(isa: Isa) -> Result<Self, IsaUnavailable> {
        if isa.is_available() {
            Ok(Self(isa))
        } else {
            Err(IsaUnavailable { isa })
        }
    }

    /// The ISA.
    pub(crate) fn isa(self) -> Isa {
        self.0
    }
}

/// Generates the dispatch of the P2P entry points for one precision.
macro_rules! p2p_dispatch {
    ($name:ident, $t:ty, $potential:ident, $gradient:ident) => {
        /// P2P on `isa` ([`P2pKernel::evaluate`](crate::P2pKernel::evaluate)) in the
        /// precision of its name, with lengths already checked by the caller.
        pub(crate) fn $name(
            isa: AvailableIsa,
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: Option<&mut [[$t; 3]]>,
        ) {
            match (isa.0, gradient) {
                (Isa::Scalar, gradient) => {
                    scalar::p2p(sources, charges, targets, potential, gradient)
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has NEON, the
                // feature the entry point enables.
                (Isa::Neon, None) => unsafe {
                    neon::$potential(sources, charges, targets, potential)
                },
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as above.
                (Isa::Neon, Some(gradient)) => unsafe {
                    neon::$gradient(sources, charges, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                (Isa::Avx2, None) => unsafe {
                    avx2::$potential(sources, charges, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx2, Some(gradient)) => unsafe {
                    avx2::$gradient(sources, charges, targets, potential, gradient)
                },
                // The ISAs of other architectures, which are never available.
                (isa, _) => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

p2p_dispatch!(p2p_f32, f32, p2p_f32, p2p_f32_gradient);
p2p_dispatch!(p2p_f64, f64, p2p_f64, p2p_f64_gradient);
