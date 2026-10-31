//! # `nd-fmm-simd`
//!
//! Hand-written SIMD kernels for the host path of the nd-project FMM, with a scalar
//! fallback and run-time instruction-set dispatch. P2P, the near field of the Laplace
//! FMM, is the first kernel ([`P2pKernel`]); the column-accumulation kernel of the host's
//! batched M2L ([`MatrixKernel`], Phase 6 T2) the second, and the plane-wave M2L's shifts
//! and Fourier passes ([`WaveKernel`], Phase 6 T13) the third. The crate needs no MPI and no
//! external SIMD library: the vector code uses `core::arch` intrinsics only. Its design is
//! [docs/design/simd-p2p.md][design] (Phase 3S), and for the matrix kernel
//! docs/design/optimisation.md §2.3 and §2.11 (Phase 6).
//!
//! **Status.** Phase 3S T5: [`P2pKernel::evaluate`] runs the vector P2P kernel on
//! NEON and AVX2 + FMA, and the reference's loop on the scalar path. Each ISA has its
//! vector layer and the inverse square root of the kernel, which
//! [`rsqrt::rsqrt_slice`] exposes for tests and reports. Phase 6 T2:
//! [`MatrixKernel::accumulate_columns`] on every ISA, in the [`MulAdd`] modes. Phase 6
//! T5: [`P2pKernel::evaluate_many`], P2P of several charge vectors at once. Phase 6 T6:
//! [`Isa::Avx512`], every kernel on AVX-512F, selectable but not yet the default. Phase 6 T8:
//! [`P2pKernel::evaluate_dipoles`], P2P of point dipoles. Phase 6 T13: [`WaveKernel`], the
//! plane-wave M2L's loops on every ISA, bit for bit the scalar path's.
//!
//! ## Instruction sets
//!
//! | [`Isa`] | Architecture | Vector | Lanes f32 / f64 | Available |
//! | --- | --- | --- | --- | --- |
//! | [`Isa::Scalar`] | any | none | 1 / 1 | always |
//! | [`Isa::Neon`] | aarch64 | 128-bit NEON | 4 / 2 | on every aarch64 CPU |
//! | [`Isa::Avx2`] | x86_64 | 256-bit AVX2 + FMA | 8 / 4 | if the CPU has AVX2 and FMA |
//! | [`Isa::Avx512`] | x86_64 | 512-bit AVX-512F | 16 / 8 | if the CPU has AVX-512F, AVX2 and FMA |
//!
//! Every variant exists on every target, but the code of an ISA is compiled only for its
//! architecture and [`Isa::is_available`] is false elsewhere.
//!
//! ## Dispatch
//!
//! - On aarch64 NEON is part of the base architecture, so [`Isa::Neon`] needs no
//!   detection. On x86_64, `is_x86_feature_detected!` for `avx2` and `fma` decides
//!   [`Isa::Avx2`].
//! - [`Isa::detect`] picks the default ISA: NEON on aarch64; on x86_64 AVX-512 where
//!   AVX-512F, AVX2 and FMA are detected (since Phase 6 T11, from measurements inside the
//!   FMM on a fully loaded node; docs/design/optimisation.md §5.3), else AVX2; else
//!   scalar.
//! - [`P2pKernel::new`] fails with [`IsaUnavailable`] for an ISA the machine cannot
//!   run, so a kernel always holds an available ISA. [`P2pKernel::detect`] uses
//!   [`Isa::detect`]. Every ISA is selectable explicitly, for tests and benchmarks.
//! - [`P2pKernel::evaluate`] checks the slice lengths and calls one entry point per ISA;
//!   the branch runs once per call.
//! - The crate never reads an environment variable to choose an ISA.
//!
//! ## The kernel
//!
//! Targets in lanes (design §4.1): each lane holds one target, a block holds K vectors
//! of targets, and each source is broadcast to every lane and visited in input order.
//! The loop order, the inverse square root and K are those signed off in Phase 3S T2
//! (spikes/p2p-simd/SPIKE_REPORT.md, "Recommendation"), and for AVX-512 those of Phase 6
//! T6 (docs/design/optimisation.md §5.2):
//!
//! | [`Isa`] | 1/r, f32 | 1/r, f64 | K, f32 / f64 | targets per block |
//! | --- | --- | --- | --- | --- |
//! | [`Isa::Scalar`] | `sqrt`, division | `sqrt`, division | — | 1 (the reference's loop) |
//! | [`Isa::Neon`] | FSQRT, FDIV | FSQRT, FDIV | 2 / 4 | 8 / 8 |
//! | [`Isa::Avx2`] | `vrsqrtps` + degree-2 correction | `vrsqrtps` via f32 + degree-5 correction | 1 / 1 | 8 / 4 |
//! | [`Isa::Avx512`] | `vrsqrt14ps` + one Newton step | `vrsqrt14pd` + two Newton steps | 2 / 2 | 32 / 16 |
//!
//! Per pair and lane, with d = x − y: r² = fma(d₂, d₂, fma(d₁, d₁, d₀²)); ρ = 1/√r²,
//! cleared to +0 where r² = 0; φ += q ρ, and with gradients ∇φ −= (q ρ) ρ² d. A call
//! runs whole blocks in place, then smaller blocks while whole vectors remain, and
//! the last partial vector on a stack copy padded with the last target, by the same
//! vector code: no target ever takes a scalar tail.
//!
//! ## Semantics
//!
//! [`P2pKernel::evaluate`] has the signature and semantics of `nd_fmm_ref::p2p::p2p`
//! (design §3, requirements 1–6):
//!
//! - φᵢ += Σⱼ qⱼ / |xᵢ − yⱼ| and ∇φᵢ += −Σⱼ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³, without the
//!   factor 1/(4π) (CONVENTIONS §3.1), added into the outputs.
//! - **Coincident pairs.** A pair contributes nothing exactly when r² = |xᵢ − yⱼ|² = 0
//!   (CONVENTIONS §3.13, "Fast kernels"). The reference skips exact coincidence,
//!   xᵢ == yⱼ in all three components; the two rules agree on leaf-scaled data and
//!   differ only when distinct points have a separation whose square underflows to
//!   zero, where the reference returns non-finite values. The accuracy contract holds
//!   on the kernel domain of §3.13, r² = 0 or 2⁻¹⁰⁸ ≤ r² ≤ 2⁷.
//! - **Order.** Each target adds its sources in input order, starting from the value
//!   already in the output. Evaluating sources `[..k]` and then `[k..]` equals
//!   evaluating them at once, and a target's result does not depend on its position
//!   in `targets` or on how many targets there are, bit for bit.
//! - **Determinism.** The same inputs on the same machine, ISA and build give the same
//!   bits. Results differ between ISAs, and from the reference, within the accuracy
//!   below. The scalar path uses the reference's formulas in the reference's order and
//!   equals it bit for bit wherever r² ≠ 0. On AVX2 the estimate `vrsqrtps` is not
//!   architecturally defined, so results may also differ between Intel and AMD CPUs in
//!   the last bits (design §5.5); AVX-512's `vrsqrt14ps`/`pd` has a bit-exact reference
//!   in Intel's documentation.
//! - **Accuracy** (design §3, requirement 2), for pairs in the kernel domain: each
//!   potential term within 8 u_T of the reference's term, relative, and each gradient
//!   component within 16 u_T relative to |q| / r², with u_T the unit roundoff (2⁻²⁴,
//!   2⁻⁵³). In f32 the gradient contract holds for 2⁻⁸⁴ ≤ r² ≤ 2⁷ and |q| ≤ 1, where
//!   the intermediate (q ρ) ρ² stays normal (§3.13, "Range of the terms in f32"). The
//!   terms are added in input order without compensation, as by the reference, so a
//!   sum carries the reference's summation error.
//! - No allocation, no threads, no MPI; the kernel writes only into the caller's
//!   slices. Mismatched lengths panic, as in the reference.
//!
//! ## Several charge vectors
//!
//! [`P2pKernel::evaluate_many`] (Phase 6 T5, C6.3; docs/design/optimisation.md §4.3)
//! evaluates m charge vectors at the same sources and targets: the m charges of a source
//! adjacent, the m vectors' outputs one vector after the other (CONVENTIONS §3.13,
//! "Several charge vectors"). The targets stay in lanes and the vectors run in groups
//! held in registers (on NEON, AVX2 and AVX-512 8 vectors without gradients and 4 with, then halves
//! for the rest); per source and target vector d, r² and ρ are formed once and every vector
//! of the group then takes the steps of [`P2pKernel::evaluate`] with its own charge, in
//! the same order. So vector v of the result is `evaluate` of vector v bit for bit, on
//! every ISA, and every guarantee above holds per vector. The scalar path forms d, r² and
//! r once per pair and adds each vector's term as its single-vector loop does.

//! ## Dipoles
//!
//! [`P2pKernel::evaluate_dipoles`] (Phase 6 T8, C6.4) adds the potential d · w / r³ of
//! point dipoles and its gradient (CONVENTIONS §3.7, "Dipole sources"), with the
//! semantics of `nd_fmm_ref::p2p::p2p_dipoles`. The vector kernels run the charge
//! kernel's loop order, blocks, tails, coincident-pair mask and inverse square root, with
//! the dipole terms formed as ρ³ s and ρ³ (dₖ − 3 s ρ² wₖ), s = d · w, so that no
//! intermediate exceeds the term's bound by more than ρ³ (docs/design/simd-p2p.md §4.8,
//! which derives the contract); the scalar path is the reference's loop.
//!
//! ## The matrix kernel
//!
//! [`MatrixKernel::accumulate_columns`] adds A · B[:, b_j] into C[:, c_j] for lists of
//! columns, A an n × n column-major matrix: the product of each M2L offset's table with
//! the multipoles of its pairs, added into the targets' own locals. Four B columns are
//! packed k-major into the caller's buffer, and a block of R vectors of rows of their four
//! C columns stays in registers while k runs from 0 to n − 1 (R = 4 on NEON and
//! AVX-512, 2 on AVX2, 4 rows on the scalar path); the rows left over run as single vectors, then one at a
//! time. Every element receives c ← c + B[k, b_j] · A[r, k] for k ascending, with the
//! product and the sum rounded apart ([`MulAdd::Separate`], the chain of
//! `nd_fmm_tables::MatrixSet::apply`) or fused ([`MulAdd::Fused`]), from its own row and
//! column only: the same bits on every ISA, for every grouping of the columns.
//!
//! ## Unsafe
//!
//! `unsafe` is allowed only in the architecture modules and in the dispatch that calls
//! their `#[target_feature]` entry points (design §5.4): calling an entry point after
//! [`P2pKernel::new`] or [`rsqrt::rsqrt_slice`] has checked the ISA, intrinsics inside
//! the vector layer, and vector loads and stores from slices whose bounds the code has
//! checked. The layer's methods are safe to call: they take a zero-sized token of their
//! ISA, which can only be made on a CPU that has it. Every block carries a
//! `// SAFETY:` comment, which the crate-level lints `unsafe_op_in_unsafe_fn` and
//! `clippy::undocumented_unsafe_blocks` (both denied) enforce. Every public function is
//! safe.
//!
//! ## Testing on each architecture
//!
//! Every test that runs the kernel prints the ISAs it ran; `--show-output` shows it
//! for passing tests.
//!
//! - aarch64 (Apple silicon, the CI arm64 runner): `cargo test -p nd-fmm-simd --
//!   --show-output` runs scalar and NEON.
//! - x86_64 (the CI x86_64 runner): the same command runs scalar and, if the CPU has
//!   them, AVX2 + FMA and AVX-512F. Most runners lack AVX-512F (docs/design/optimisation.md
//!   §5.4): its tests then skip, and every test line names it under "not available
//!   here"; Kathleen's Cascade Lake nodes run it (Phase 6 T6).
//! - The release-mode accuracy tests (exhaustive and sampled, from Phase 3S T4 on) are
//!   `#[ignore]`d: `cargo test -p nd-fmm-simd --release -- --ignored --show-output`.
//! - The other architecture's code is checked by
//!   `cargo clippy -p nd-fmm-simd --all-targets --target <other> -- -D warnings`, with
//!   `<other>` `x86_64-apple-darwin` on Apple silicon or `aarch64-unknown-linux-gnu`
//!   on x86_64 Linux (installed with `rustup target add`).
//! - On an Apple silicon Mac with Rosetta 2, `cargo test -p nd-fmm-simd --target
//!   x86_64-apple-darwin` should run the x86_64 build, for a quick check before CI.
//!   This is unconfirmed: the development machine had no Rosetta 2 when the crate was
//!   created, nor in Phase 3S T4, and the binary failed with "Bad CPU type in
//!   executable", so the AVX2 path runs only on the x86_64 CI runner. Rosetta also
//!   emulates the estimate instructions, so accuracy contracts count only from real
//!   x86_64 hardware.
//!
//! ## Toolchain
//!
//! The design relies on language features of Rust 1.86, 1.87 and 1.89. Confirmed on
//! stable rustc 1.98.0 and 1.99.0 for both `aarch64-apple-darwin` and
//! `x86_64-apple-darwin`, each with a compiled example in the test module `toolchain`:
//!
//! - **Safe `#[target_feature]` functions** (target_feature 1.1, Rust 1.86): stable. A
//!   safe function may carry `#[target_feature(enable = "…")]`. Calling it needs no
//!   `unsafe` inside a function whose own `#[target_feature]` enables the same
//!   features, and needs `unsafe` everywhere else; it coerces only to an `unsafe fn`
//!   pointer.
//! - **Safe intrinsics in matching contexts** (Rust 1.87): stable. Value-only
//!   intrinsics, such as arithmetic, fma, compares, the inverse-square-root estimates
//!   and splats, are safe inside a function whose `#[target_feature]` lists their
//!   features. Intrinsics that take pointers (loads and stores) stay `unsafe`. On
//!   aarch64 this holds for NEON too, but only where the function lists `neon` itself:
//!   that `neon` is enabled for every aarch64 target does not count (error E0133), so
//!   NEON entry points are `#[target_feature(enable = "neon")]` functions and the
//!   dispatch calls them in `unsafe` like the AVX2 ones.
//! - **AVX-512 target features** (Rust 1.89): stable. `#[target_feature(enable =
//!   "avx512f")]`, the `__m512d` intrinsics and `is_x86_feature_detected!("avx512f")`
//!   compile; [`Isa::Avx512`] (Phase 6 T6) uses them.
//!
//! Two limits on this toolchain shape the vector layer of design §5.3:
//!
//! - `#[inline(always)]` cannot be combined with `#[target_feature]` (rejected:
//!   "cannot use `#[inline(always)]` with `#[target_feature]`"). A layer method
//!   that is `#[inline(always)]` therefore carries no target features of its own, and
//!   its intrinsic calls are `unsafe` there; it inlines into the entry point, which has
//!   them.
//! - `#[target_feature]` is rejected on a safe trait method, but allowed on an
//!   `unsafe` one.
//!
//! [design]: https://github.com/tbetcke/fmm/blob/main/docs/design/simd-p2p.md

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod arch;
pub mod exp;
mod isa;
mod kernel;
mod matrix;
pub mod rsqrt;
#[cfg(test)]
mod toolchain;
mod waves;

pub use isa::{Isa, IsaUnavailable};
pub use kernel::P2pKernel;
pub use matrix::{MatrixKernel, MulAdd, UnknownMulAdd};
pub use waves::{WaveBlock, WaveKernel, WaveTables};

use nd_fmm_math::RealScalar;

/// The precisions with SIMD kernels: [`f32`] and [`f64`].
///
/// The trait is sealed: it cannot be implemented outside this crate, so every kernel
/// has a vector path for every implementing type. f32 and f64 are also the only
/// [`RealScalar`] types, so code generic over `RealScalar` can require it without
/// excluding a caller.
pub trait SimdScalar: RealScalar + sealed::Sealed {}

impl SimdScalar for f32 {}
impl SimdScalar for f64 {}

/// Seals [`SimdScalar`], and dispatches generic code to the entry points of its
/// precision.
mod sealed {
    use crate::arch::{self, AvailableIsa, exp::ExpConstants};
    use crate::{Isa, MulAdd, WaveTables};

    /// Implemented only for f32 and f64.
    pub trait Sealed: Sized {
        /// [`rsqrt_slice`](crate::rsqrt::rsqrt_slice) in this precision.
        fn rsqrt_slice(isa: Isa, x: &[Self], out: &mut [Self]);

        /// The constants of the exponential in this precision (`arch::exp`; Helmholtz
        /// CH2.4a).
        const EXP: ExpConstants;

        /// `Simd::scale_pow2` of the scalar layer: x · 2ⁿ from the bits of x and of
        /// k = 1.5 · 2^m + n (Helmholtz CH2.4a).
        fn scale_pow2(x: Self, k: Self) -> Self;

        /// [`exp_slice`](crate::exp::exp_slice) in this precision.
        fn exp_slice(isa: Isa, x: &[Self], out: &mut [Self]);

        /// [`P2pKernel::evaluate`](crate::P2pKernel::evaluate) in this precision, with
        /// lengths already checked.
        fn p2p(
            isa: AvailableIsa,
            sources: &[[Self; 3]],
            charges: &[Self],
            targets: &[[Self; 3]],
            potential: &mut [Self],
            gradient: Option<&mut [[Self; 3]]>,
        );

        /// [`P2pKernel::evaluate_many`](crate::P2pKernel::evaluate_many) in this
        /// precision, with lengths already checked.
        fn p2p_many(
            isa: AvailableIsa,
            sources: &[[Self; 3]],
            charges: &[Self],
            m: usize,
            targets: &[[Self; 3]],
            potential: &mut [Self],
            gradient: Option<&mut [[Self; 3]]>,
        );

        /// [`P2pKernel::evaluate_dipoles`](crate::P2pKernel::evaluate_dipoles) in this
        /// precision, with lengths already checked.
        fn p2p_dipoles(
            isa: AvailableIsa,
            sources: &[[Self; 3]],
            dipoles: &[[Self; 3]],
            targets: &[[Self; 3]],
            potential: &mut [Self],
            gradient: Option<&mut [[Self; 3]]>,
        );

        /// [`MatrixKernel::accumulate_columns`](crate::MatrixKernel::accumulate_columns)
        /// in this precision, with shapes already checked.
        #[expect(
            clippy::too_many_arguments,
            reason = "the shape of `MatrixKernel::accumulate_columns`, plus the ISA and mode"
        )]
        fn accumulate(
            isa: AvailableIsa,
            mul_add: MulAdd,
            n: usize,
            a: &[Self],
            b: &[Self],
            b_offsets: &[usize],
            c: &mut [Self],
            c_offsets: &[usize],
            pack: &mut [Self],
        );

        /// [`WaveKernel::merge`](crate::WaveKernel::merge) in this precision, with shapes
        /// already checked.
        fn merge(
            isa: AvailableIsa,
            acc: &mut [Self],
            waves: &[Self],
            wave_offsets: &[usize],
            shifts: &[Self],
            shift_offsets: &[usize],
        );

        /// [`WaveKernel::gather`](crate::WaveKernel::gather) in this precision, with shapes
        /// already checked.
        fn gather(
            isa: AvailableIsa,
            v: &mut [Self],
            accs: &[Self],
            acc_offsets: &[usize],
            t: &[Self],
        );

        /// [`WaveKernel::waves`](crate::WaveKernel::waves) in this precision, with shapes
        /// already checked.
        fn waves(
            isa: AvailableIsa,
            tables: &WaveTables<Self>,
            modes: &[Self],
            up: &mut [Self],
            down: &mut [Self],
        );

        /// [`WaveKernel::sums`](crate::WaveKernel::sums) in this precision, with shapes
        /// already checked.
        fn sums(
            isa: AvailableIsa,
            tables: &WaveTables<Self>,
            up: &[Self],
            down: &[Self],
            sums: &mut [Self],
        );
    }

    impl Sealed for f32 {
        fn rsqrt_slice(isa: Isa, x: &[f32], out: &mut [f32]) {
            arch::rsqrt_slice_f32(isa, x, out);
        }

        const EXP: ExpConstants = arch::exp::F32;

        #[inline(always)]
        fn scale_pow2(x: f32, k: f32) -> f32 {
            f32::from_bits(x.to_bits().wrapping_add(k.to_bits() << 23))
        }

        fn exp_slice(isa: Isa, x: &[f32], out: &mut [f32]) {
            arch::exp_slice_f32(isa, x, out);
        }

        #[inline]
        fn p2p(
            isa: AvailableIsa,
            sources: &[[f32; 3]],
            charges: &[f32],
            targets: &[[f32; 3]],
            potential: &mut [f32],
            gradient: Option<&mut [[f32; 3]]>,
        ) {
            arch::p2p_f32(isa, sources, charges, targets, potential, gradient);
        }

        #[inline]
        fn p2p_many(
            isa: AvailableIsa,
            sources: &[[f32; 3]],
            charges: &[f32],
            m: usize,
            targets: &[[f32; 3]],
            potential: &mut [f32],
            gradient: Option<&mut [[f32; 3]]>,
        ) {
            arch::p2p_many_f32(isa, sources, charges, m, targets, potential, gradient);
        }

        #[inline]
        fn p2p_dipoles(
            isa: AvailableIsa,
            sources: &[[f32; 3]],
            dipoles: &[[f32; 3]],
            targets: &[[f32; 3]],
            potential: &mut [f32],
            gradient: Option<&mut [[f32; 3]]>,
        ) {
            arch::p2p_dipoles_f32(isa, sources, dipoles, targets, potential, gradient);
        }

        #[inline]
        fn accumulate(
            isa: AvailableIsa,
            mul_add: MulAdd,
            n: usize,
            a: &[f32],
            b: &[f32],
            b_offsets: &[usize],
            c: &mut [f32],
            c_offsets: &[usize],
            pack: &mut [f32],
        ) {
            arch::accumulate_f32(isa, mul_add, n, a, b, b_offsets, c, c_offsets, pack);
        }

        #[inline]
        fn merge(
            isa: AvailableIsa,
            acc: &mut [f32],
            waves: &[f32],
            wave_offsets: &[usize],
            shifts: &[f32],
            shift_offsets: &[usize],
        ) {
            arch::merge_f32(isa, acc, waves, wave_offsets, shifts, shift_offsets);
        }

        #[inline]
        fn gather(
            isa: AvailableIsa,
            v: &mut [f32],
            accs: &[f32],
            acc_offsets: &[usize],
            t: &[f32],
        ) {
            arch::gather_f32(isa, v, accs, acc_offsets, t);
        }

        #[inline]
        fn waves(
            isa: AvailableIsa,
            tables: &WaveTables<f32>,
            modes: &[f32],
            up: &mut [f32],
            down: &mut [f32],
        ) {
            arch::waves_f32(isa, tables, modes, up, down);
        }

        #[inline]
        fn sums(
            isa: AvailableIsa,
            tables: &WaveTables<f32>,
            up: &[f32],
            down: &[f32],
            sums: &mut [f32],
        ) {
            arch::sums_f32(isa, tables, up, down, sums);
        }
    }

    impl Sealed for f64 {
        fn rsqrt_slice(isa: Isa, x: &[f64], out: &mut [f64]) {
            arch::rsqrt_slice_f64(isa, x, out);
        }

        const EXP: ExpConstants = arch::exp::F64;

        #[inline(always)]
        fn scale_pow2(x: f64, k: f64) -> f64 {
            f64::from_bits(x.to_bits().wrapping_add(k.to_bits() << 52))
        }

        fn exp_slice(isa: Isa, x: &[f64], out: &mut [f64]) {
            arch::exp_slice_f64(isa, x, out);
        }

        #[inline]
        fn p2p(
            isa: AvailableIsa,
            sources: &[[f64; 3]],
            charges: &[f64],
            targets: &[[f64; 3]],
            potential: &mut [f64],
            gradient: Option<&mut [[f64; 3]]>,
        ) {
            arch::p2p_f64(isa, sources, charges, targets, potential, gradient);
        }

        #[inline]
        fn p2p_many(
            isa: AvailableIsa,
            sources: &[[f64; 3]],
            charges: &[f64],
            m: usize,
            targets: &[[f64; 3]],
            potential: &mut [f64],
            gradient: Option<&mut [[f64; 3]]>,
        ) {
            arch::p2p_many_f64(isa, sources, charges, m, targets, potential, gradient);
        }

        #[inline]
        fn p2p_dipoles(
            isa: AvailableIsa,
            sources: &[[f64; 3]],
            dipoles: &[[f64; 3]],
            targets: &[[f64; 3]],
            potential: &mut [f64],
            gradient: Option<&mut [[f64; 3]]>,
        ) {
            arch::p2p_dipoles_f64(isa, sources, dipoles, targets, potential, gradient);
        }

        #[inline]
        fn accumulate(
            isa: AvailableIsa,
            mul_add: MulAdd,
            n: usize,
            a: &[f64],
            b: &[f64],
            b_offsets: &[usize],
            c: &mut [f64],
            c_offsets: &[usize],
            pack: &mut [f64],
        ) {
            arch::accumulate_f64(isa, mul_add, n, a, b, b_offsets, c, c_offsets, pack);
        }

        #[inline]
        fn merge(
            isa: AvailableIsa,
            acc: &mut [f64],
            waves: &[f64],
            wave_offsets: &[usize],
            shifts: &[f64],
            shift_offsets: &[usize],
        ) {
            arch::merge_f64(isa, acc, waves, wave_offsets, shifts, shift_offsets);
        }

        #[inline]
        fn gather(
            isa: AvailableIsa,
            v: &mut [f64],
            accs: &[f64],
            acc_offsets: &[usize],
            t: &[f64],
        ) {
            arch::gather_f64(isa, v, accs, acc_offsets, t);
        }

        #[inline]
        fn waves(
            isa: AvailableIsa,
            tables: &WaveTables<f64>,
            modes: &[f64],
            up: &mut [f64],
            down: &mut [f64],
        ) {
            arch::waves_f64(isa, tables, modes, up, down);
        }

        #[inline]
        fn sums(
            isa: AvailableIsa,
            tables: &WaveTables<f64>,
            up: &[f64],
            down: &[f64],
            sums: &mut [f64],
        ) {
            arch::sums_f64(isa, tables, up, down, sums);
        }
    }
}
