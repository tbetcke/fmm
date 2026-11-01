//! The per-ISA code: the vector layer, the inverse square root and the entry points
//! (docs/design/simd-p2p.md §5.3; the matrix kernel, docs/design/optimisation.md §2.11).
//!
//! Each module holds the code of one [`Isa`], compiled only for its architecture, and
//! implements [`Simd`], the crate's thin layer over its vector types:
//!
//! - [`scalar`]: one lane, plain Rust, available everywhere;
//! - `neon` (aarch64): `float32x4_t` and `float64x2_t`;
//! - `avx2` (x86_64): `__m256` and `__m256d`;
//! - `avx512` (x86_64, Phase 6 T6): `__m512` and `__m512d`, with mask registers.
//!
//! Code that runs on vectors is written once, generic over [`Simd`]
//! ([`rsqrt_body`], the P2P kernel body [`p2p::p2p_body`] and the column-accumulation
//! body [`matrix::accumulate_body`]), and instantiated inside one `#[target_feature]`
//! entry point per ISA, precision and, for P2P, output, for the matrix kernel rounding,
//! so that the layer methods, all `#[inline(always)]`, and their intrinsics inline into
//! a function that has the features. [`rsqrt_slice_f32`] and [`rsqrt_slice_f64`]
//! dispatch to the inverse-square-root entry points, [`p2p_f32`] and [`p2p_f64`] to
//! the P2P ones, [`p2p_many_f32`] and [`p2p_many_f64`] to those of several charge vectors
//! (the body [`p2p::p2p_many_body`]; Phase 6 T5), [`p2p_dipoles_f32`] and
//! [`p2p_dipoles_f64`] to the dipole ones (the body [`p2p::p2p_dipoles_body`]; Phase 6 T8),
//! [`p2p_yukawa_f32`] and [`p2p_yukawa_f64`] to the Yukawa ones (the skeleton
//! [`p2p::p2p_skeleton`] with [`p2p::Yukawa`], which also runs Laplace's [`p2p::p2p_body`];
//! Helmholtz CH2.4b), [`accumulate_f32`] and
//! [`accumulate_f64`] to the matrix ones, and [`exp_slice_f32`] and [`exp_slice_f64`] to
//! those of the exponential (the body [`exp::exp_body`]; Helmholtz CH2.4a), which the
//! scalar ISA runs on the scalar layer. The
//! scalar ISA runs the matrix body on the scalar layer, whose chain is the vector
//! paths' exactly.
//!
//! The scalar ISA runs P2P by the plain loop [`scalar::p2p`] (several vectors by
//! [`scalar::p2p_many`], the same terms per vector), which equals
//! `nd_fmm_ref::p2p::p2p` bit for bit wherever r² ≠ 0 (dipoles by [`scalar::p2p_dipoles`],
//! `nd_fmm_ref::p2p::p2p_dipoles`'s loop; Yukawa by [`scalar::p2p_yukawa`],
//! `nd_fmm_ref::helmholtz::p2p::p2p_yukawa`'s loop), not by the generic body at
//! W = 1: that body forms r² with fma and the gradient term as (q ρ) ρ² d, so it
//! rounds differently. The tests run the generic body on the scalar layer as well.
//!
//! # Unsafe
//!
//! `unsafe` is allowed only in these modules and in the dispatch (crate documentation,
//! "Unsafe"). The layer is safe to call: an ISA's methods take its token
//! (`neon::Neon`, `avx2::Avx2`, `avx512::Avx512`) by value, and a token can only be made on a CPU
//! that has the ISA. Inside the methods, every intrinsic call is an `unsafe` block,
//! because an `#[inline(always)]` method cannot carry `#[target_feature]` (crate
//! documentation, "Toolchain"); its `// SAFETY:` comment cites the token, and for
//! loads and stores the slice length the method asserts.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
pub(crate) mod exp;
pub(crate) mod matrix;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod p2p;
pub(crate) mod scalar;
pub(crate) mod sincos;
#[cfg(test)]
mod tests;
pub(crate) mod waves;

use crate::{Isa, IsaUnavailable, MulAdd, SimdScalar, WaveTables};

/// The largest lane count of any ISA (AVX-512, f32), the size of the padded stack copies
/// that tails use.
pub(crate) const MAX_LANES: usize = 16;

/// The vector operations of one ISA on `T` (docs/design/simd-p2p.md §5.3).
///
/// Implemented by a zero-sized token per ISA, [`scalar::Scalar`], `neon::Neon`,
/// `avx2::Avx2` and `avx512::Avx512`, for f32 and f64. A token exists only on a CPU that has its ISA, so
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
    /// The result of a lane-wise comparison: all bits set in a lane where it holds, or
    /// on AVX-512 a mask register with one bit per lane.
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
    /// `v` where `mask` is clear and +0 where it is set: the bitwise and-not (on
    /// AVX-512 the zero-masked move under the complement of the mask), which also clears
    /// a NaN or an infinity.
    fn and_not(self, mask: Self::Mask, v: Self::V) -> Self::V;
    /// The hardware estimate of 1/√x: FRSQRTE (relative error below 2⁻⁸) on NEON,
    /// `vrsqrtps` (at most 1.5 · 2⁻¹², documented) on AVX2, through f32 for f64,
    /// `vrsqrt14ps`/`pd` (below 2⁻¹⁴) on AVX-512; on the scalar path 1/√x itself. Not the kernel's inverse square root; see
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
    /// The larger of a and b, lane by lane (Helmholtz CH2.4a, [`exp::exp`]'s clamp). Where
    /// a lane of a or b is NaN, or the lanes are zeros of opposite signs, the result is
    /// unspecified.
    fn max(self, a: Self::V, b: Self::V) -> Self::V;
    /// x · 2ⁿ, lane by lane, where k = c + n carries the integer n in the low bits of its
    /// significand, c = 1.5 · 2^m with m the explicit significand bits (52, 23; Helmholtz
    /// CH2.4a, [`exp::exp`]): the integer sum, wrapping, of the bits of x and the bits of k
    /// shifted left by m. Exact where x and x · 2ⁿ are normal and |n| < 2^(m − 1);
    /// unspecified otherwise.
    fn scale_pow2(self, x: Self::V, k: Self::V) -> Self::V;

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
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (checked above), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                Isa::Avx512 => unsafe { avx512::$entry(x, out) },
                // The ISAs of other architectures, which are never available.
                _ => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

rsqrt_dispatch!(rsqrt_slice_f32, f32, rsqrt_slice_f32);
rsqrt_dispatch!(rsqrt_slice_f64, f64, rsqrt_slice_f64);

/// Generates the dispatch of the exponential's entry points for one precision (Helmholtz
/// CH2.4a).
macro_rules! exp_dispatch {
    ($name:ident, $t:ty, $entry:ident) => {
        #[doc = concat!(
                            "The exponential of every value of `x` into `out`, on `isa`, in ",
                            stringify!($t), " (`exp::exp_slice`)."
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
                Isa::Scalar => exp::exp_body(scalar::Scalar, x, out),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (checked above), so the CPU has NEON,
                // the feature the entry point enables.
                Isa::Neon => unsafe { neon::$entry(x, out) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (checked above), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                Isa::Avx2 => unsafe { avx2::$entry(x, out) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (checked above), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                Isa::Avx512 => unsafe { avx512::$entry(x, out) },
                // The ISAs of other architectures, which are never available.
                _ => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

exp_dispatch!(exp_slice_f32, f32, exp_slice_f32);
exp_dispatch!(exp_slice_f64, f64, exp_slice_f64);

/// Generates the dispatch of the sine and cosine's entry points for one precision
/// (Helmholtz CH3.4a).
macro_rules! sincos_dispatch {
    ($name:ident, $t:ty, $entry:ident) => {
        #[doc = concat!(
                            "The sine and cosine of every value of `theta` into `sin` and `cos`, on `isa`, in ",
                            stringify!($t), " (`sincos::sincos_slice`)."
                        )]
        ///
        /// # Panics
        ///
        /// If `isa` is not available on this machine, or `theta`, `sin` and `cos` do not
        /// all have the same length.
        pub(crate) fn $name(isa: Isa, theta: &[$t], sin: &mut [$t], cos: &mut [$t]) {
            if !isa.is_available() {
                panic!("{}", IsaUnavailable { isa });
            }
            match isa {
                Isa::Scalar => sincos::sincos_body(scalar::Scalar, theta, sin, cos),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (checked above), so the CPU has NEON,
                // the feature the entry point enables.
                Isa::Neon => unsafe { neon::$entry(theta, sin, cos) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (checked above), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                Isa::Avx2 => unsafe { avx2::$entry(theta, sin, cos) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (checked above), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                Isa::Avx512 => unsafe { avx512::$entry(theta, sin, cos) },
                // The ISAs of other architectures, which are never available.
                _ => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

sincos_dispatch!(sincos_slice_f32, f32, sincos_slice_f32);
sincos_dispatch!(sincos_slice_f64, f64, sincos_slice_f64);

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
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                (Isa::Avx512, None) => unsafe {
                    avx512::$potential(sources, charges, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx512, Some(gradient)) => unsafe {
                    avx512::$gradient(sources, charges, targets, potential, gradient)
                },
                // The ISAs of other architectures, which are never available.
                (isa, _) => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

p2p_dispatch!(p2p_f32, f32, p2p_f32, p2p_f32_gradient);
p2p_dispatch!(p2p_f64, f64, p2p_f64, p2p_f64_gradient);

/// Generates the dispatch of the dipole P2P entry points for one precision (Phase 6 T8).
macro_rules! p2p_dipoles_dispatch {
    ($name:ident, $t:ty, $potential:ident, $gradient:ident) => {
        /// Dipole P2P on `isa`
        /// ([`P2pKernel::evaluate_dipoles`](crate::P2pKernel::evaluate_dipoles)) in the
        /// precision of its name, with lengths already checked by the caller.
        pub(crate) fn $name(
            isa: AvailableIsa,
            sources: &[[$t; 3]],
            dipoles: &[[$t; 3]],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: Option<&mut [[$t; 3]]>,
        ) {
            match (isa.0, gradient) {
                (Isa::Scalar, gradient) => {
                    scalar::p2p_dipoles(sources, dipoles, targets, potential, gradient)
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has NEON, the
                // feature the entry point enables.
                (Isa::Neon, None) => unsafe {
                    neon::$potential(sources, dipoles, targets, potential)
                },
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as above.
                (Isa::Neon, Some(gradient)) => unsafe {
                    neon::$gradient(sources, dipoles, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                (Isa::Avx2, None) => unsafe {
                    avx2::$potential(sources, dipoles, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx2, Some(gradient)) => unsafe {
                    avx2::$gradient(sources, dipoles, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                (Isa::Avx512, None) => unsafe {
                    avx512::$potential(sources, dipoles, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx512, Some(gradient)) => unsafe {
                    avx512::$gradient(sources, dipoles, targets, potential, gradient)
                },
                // The ISAs of other architectures, which are never available.
                (isa, _) => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

p2p_dipoles_dispatch!(
    p2p_dipoles_f32,
    f32,
    p2p_dipoles_f32,
    p2p_dipoles_f32_gradient
);
p2p_dipoles_dispatch!(
    p2p_dipoles_f64,
    f64,
    p2p_dipoles_f64,
    p2p_dipoles_f64_gradient
);

/// Generates the dispatch of the Yukawa P2P entry points for one precision (Helmholtz
/// CH2.4b).
macro_rules! p2p_yukawa_dispatch {
    ($name:ident, $t:ty, $potential:ident, $gradient:ident) => {
        /// Yukawa P2P on `isa`
        /// ([`P2pKernel::evaluate_yukawa`](crate::P2pKernel::evaluate_yukawa)) in the
        /// precision of its name, with lengths and κ ≥ 0 already checked by the caller.
        pub(crate) fn $name(
            isa: AvailableIsa,
            kappa: $t,
            sources: &[[$t; 3]],
            charges: &[$t],
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: Option<&mut [[$t; 3]]>,
        ) {
            match (isa.0, gradient) {
                (Isa::Scalar, gradient) => {
                    scalar::p2p_yukawa(kappa, sources, charges, targets, potential, gradient)
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has NEON, the
                // feature the entry point enables.
                (Isa::Neon, None) => unsafe {
                    neon::$potential(kappa, sources, charges, targets, potential)
                },
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as above.
                (Isa::Neon, Some(gradient)) => unsafe {
                    neon::$gradient(kappa, sources, charges, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                (Isa::Avx2, None) => unsafe {
                    avx2::$potential(kappa, sources, charges, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx2, Some(gradient)) => unsafe {
                    avx2::$gradient(kappa, sources, charges, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                (Isa::Avx512, None) => unsafe {
                    avx512::$potential(kappa, sources, charges, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx512, Some(gradient)) => unsafe {
                    avx512::$gradient(kappa, sources, charges, targets, potential, gradient)
                },
                // The ISAs of other architectures, which are never available.
                (isa, _) => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

p2p_yukawa_dispatch!(p2p_yukawa_f32, f32, p2p_yukawa_f32, p2p_yukawa_f32_gradient);
p2p_yukawa_dispatch!(p2p_yukawa_f64, f64, p2p_yukawa_f64, p2p_yukawa_f64_gradient);

/// Generates the dispatch of the entry points of the P2P of several charge vectors for one
/// precision (Phase 6 T5).
macro_rules! p2p_many_dispatch {
    ($name:ident, $t:ty, $potential:ident, $gradient:ident) => {
        /// P2P of several charge vectors on `isa`
        /// ([`P2pKernel::evaluate_many`](crate::P2pKernel::evaluate_many)) in the precision
        /// of its name, with lengths already checked by the caller.
        pub(crate) fn $name(
            isa: AvailableIsa,
            sources: &[[$t; 3]],
            charges: &[$t],
            m: usize,
            targets: &[[$t; 3]],
            potential: &mut [$t],
            gradient: Option<&mut [[$t; 3]]>,
        ) {
            match (isa.0, gradient) {
                (Isa::Scalar, gradient) => {
                    scalar::p2p_many(sources, charges, m, targets, potential, gradient)
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has NEON, the
                // feature the entry point enables.
                (Isa::Neon, None) => unsafe {
                    neon::$potential(sources, charges, m, targets, potential)
                },
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as above.
                (Isa::Neon, Some(gradient)) => unsafe {
                    neon::$gradient(sources, charges, m, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                (Isa::Avx2, None) => unsafe {
                    avx2::$potential(sources, charges, m, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx2, Some(gradient)) => unsafe {
                    avx2::$gradient(sources, charges, m, targets, potential, gradient)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                (Isa::Avx512, None) => unsafe {
                    avx512::$potential(sources, charges, m, targets, potential)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx512, Some(gradient)) => unsafe {
                    avx512::$gradient(sources, charges, m, targets, potential, gradient)
                },
                // The ISAs of other architectures, which are never available.
                (isa, _) => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

p2p_many_dispatch!(p2p_many_f32, f32, p2p_many_f32, p2p_many_f32_gradient);
p2p_many_dispatch!(p2p_many_f64, f64, p2p_many_f64, p2p_many_f64_gradient);

/// Generates the dispatch of the column-accumulation entry points for one precision.
macro_rules! accumulate_dispatch {
    ($name:ident, $t:ty, $separate:ident, $fused:ident) => {
        /// C[:, c_j] += A · B[:, b_j] on `isa`
        /// ([`MatrixKernel::accumulate_columns`](crate::MatrixKernel::accumulate_columns))
        /// in the precision of its name, with shapes already checked by the caller.
        #[expect(
            clippy::too_many_arguments,
            reason = "the shape of `MatrixKernel::accumulate_columns`, plus the ISA and mode"
        )]
        pub(crate) fn $name(
            isa: AvailableIsa,
            mul_add: MulAdd,
            n: usize,
            a: &[$t],
            b: &[$t],
            b_offsets: &[usize],
            c: &mut [$t],
            c_offsets: &[usize],
            pack: &mut [$t],
        ) {
            use matrix::{SCALAR_ROWS, accumulate_body};
            match (isa.0, mul_add) {
                (Isa::Scalar, MulAdd::Separate) => accumulate_body::<$t, _, SCALAR_ROWS, false>(
                    scalar::Scalar,
                    n,
                    a,
                    b,
                    b_offsets,
                    c,
                    c_offsets,
                    pack,
                ),
                (Isa::Scalar, MulAdd::Fused) => accumulate_body::<$t, _, SCALAR_ROWS, true>(
                    scalar::Scalar,
                    n,
                    a,
                    b,
                    b_offsets,
                    c,
                    c_offsets,
                    pack,
                ),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has NEON, the
                // feature the entry point enables.
                (Isa::Neon, MulAdd::Separate) => unsafe {
                    neon::$separate(n, a, b, b_offsets, c, c_offsets, pack)
                },
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as above.
                (Isa::Neon, MulAdd::Fused) => unsafe {
                    neon::$fused(n, a, b, b_offsets, c, c_offsets, pack)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX2 and
                // FMA, the features the entry point enables.
                (Isa::Avx2, MulAdd::Separate) => unsafe {
                    avx2::$separate(n, a, b, b_offsets, c, c_offsets, pack)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx2, MulAdd::Fused) => unsafe {
                    avx2::$fused(n, a, b, b_offsets, c, c_offsets, pack)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX-512F,
                // AVX2 and FMA, the features the entry point enables or implies.
                (Isa::Avx512, MulAdd::Separate) => unsafe {
                    avx512::$separate(n, a, b, b_offsets, c, c_offsets, pack)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                (Isa::Avx512, MulAdd::Fused) => unsafe {
                    avx512::$fused(n, a, b, b_offsets, c, c_offsets, pack)
                },
                // The ISAs of other architectures, which are never available.
                (isa, _) => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

accumulate_dispatch!(accumulate_f32, f32, accumulate_f32, accumulate_f32_fused);
accumulate_dispatch!(accumulate_f64, f64, accumulate_f64, accumulate_f64_fused);

/// Generates the dispatch of the plane-wave entry points for one precision (Phase 6 T13):
/// the scalar layer runs the generic bodies of [`waves`], the other ISAs their entry points.
macro_rules! waves_dispatch {
    ($t:ty, $merge:ident, $gather:ident, $waves:ident, $sums:ident) => {
        /// [`WaveKernel::merge`](crate::WaveKernel::merge) on `isa` in the precision of its
        /// name, with shapes already checked by the caller.
        pub(crate) fn $merge(
            isa: AvailableIsa,
            acc: &mut [$t],
            waves: &[$t],
            wave_offsets: &[usize],
            shifts: &[$t],
            shift_offsets: &[usize],
        ) {
            match isa.0 {
                Isa::Scalar => waves::merge_body(
                    scalar::Scalar,
                    acc,
                    waves,
                    wave_offsets,
                    shifts,
                    shift_offsets,
                ),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has NEON, the
                // feature the entry point enables.
                Isa::Neon => unsafe {
                    neon::$merge(acc, waves, wave_offsets, shifts, shift_offsets)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX2 and FMA,
                // the features the entry point enables.
                Isa::Avx2 => unsafe {
                    avx2::$merge(acc, waves, wave_offsets, shifts, shift_offsets)
                },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa` is available (`AvailableIsa`), so the CPU has AVX-512F, AVX2
                // and FMA, the features the entry point enables or implies.
                Isa::Avx512 => unsafe {
                    avx512::$merge(acc, waves, wave_offsets, shifts, shift_offsets)
                },
                // The ISAs of other architectures, which are never available.
                isa => unreachable!("`{isa}` is not available on this machine"),
            }
        }

        /// [`WaveKernel::gather`](crate::WaveKernel::gather) on `isa` in the precision of its
        /// name, with shapes already checked by the caller.
        pub(crate) fn $gather(
            isa: AvailableIsa,
            v: &mut [$t],
            accs: &[$t],
            acc_offsets: &[usize],
            t: &[$t],
        ) {
            match isa.0 {
                Isa::Scalar => waves::gather_body(scalar::Scalar, v, accs, acc_offsets, t),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as in the merge dispatch.
                Isa::Neon => unsafe { neon::$gather(v, accs, acc_offsets, t) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as in the merge dispatch.
                Isa::Avx2 => unsafe { avx2::$gather(v, accs, acc_offsets, t) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as in the merge dispatch.
                Isa::Avx512 => unsafe { avx512::$gather(v, accs, acc_offsets, t) },
                isa => unreachable!("`{isa}` is not available on this machine"),
            }
        }

        /// [`WaveKernel::waves`](crate::WaveKernel::waves) on `isa` in the precision of its
        /// name, with shapes already checked by the caller.
        pub(crate) fn $waves(
            isa: AvailableIsa,
            tables: &WaveTables<$t>,
            modes: &[$t],
            up: &mut [$t],
            down: &mut [$t],
        ) {
            match isa.0 {
                Isa::Scalar => waves::waves_body::<_, _, { waves::BLOCK }>(
                    scalar::Scalar,
                    tables,
                    modes,
                    up,
                    down,
                ),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as in the merge dispatch.
                Isa::Neon => unsafe { neon::$waves(tables, modes, up, down) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as in the merge dispatch.
                Isa::Avx2 => unsafe { avx2::$waves(tables, modes, up, down) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as in the merge dispatch.
                Isa::Avx512 => unsafe { avx512::$waves(tables, modes, up, down) },
                isa => unreachable!("`{isa}` is not available on this machine"),
            }
        }

        /// [`WaveKernel::sums`](crate::WaveKernel::sums) on `isa` in the precision of its
        /// name, with shapes already checked by the caller.
        pub(crate) fn $sums(
            isa: AvailableIsa,
            tables: &WaveTables<$t>,
            up: &[$t],
            down: &[$t],
            sums: &mut [$t],
        ) {
            match isa.0 {
                Isa::Scalar => waves::sums_body(scalar::Scalar, tables, up, down, sums),
                #[cfg(target_arch = "aarch64")]
                // SAFETY: as in the merge dispatch.
                Isa::Neon => unsafe { neon::$sums(tables, up, down, sums) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as in the merge dispatch.
                Isa::Avx2 => unsafe { avx2::$sums(tables, up, down, sums) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as in the merge dispatch.
                Isa::Avx512 => unsafe { avx512::$sums(tables, up, down, sums) },
                isa => unreachable!("`{isa}` is not available on this machine"),
            }
        }
    };
}

waves_dispatch!(f32, merge_f32, gather_f32, waves_f32, sums_f32);
waves_dispatch!(f64, merge_f64, gather_f64, waves_f64, sums_f64);
