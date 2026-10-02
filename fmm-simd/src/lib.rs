//! # `nd-fmm-simd`
//!
//! Hand-written SIMD kernels for the host path of the nd-project FMM, with a scalar
//! fallback and run-time instruction-set dispatch. P2P, the near field of the Laplace
//! FMM, is the first kernel ([`P2pKernel`]). The crate needs no MPI and no external
//! SIMD library: the vector code uses `core::arch` intrinsics only. Its design is
//! [docs/design/simd-p2p.md][design] (Phase 3S).
//!
//! **Status.** This version is the scaffold of Phase 3S T3: the public surface,
//! detection and dispatch work, and every ISA runs the scalar path. The per-ISA vector
//! layer and inverse square root (T4) and the vector P2P kernel (T5) fill in the
//! architecture modules behind the same interface.
//!
//! ## Instruction sets
//!
//! | [`Isa`] | Architecture | Vector | Lanes f32 / f64 | Available |
//! | --- | --- | --- | --- | --- |
//! | [`Isa::Scalar`] | any | none | 1 / 1 | always |
//! | [`Isa::Neon`] | aarch64 | 128-bit NEON | 4 / 2 | on every aarch64 CPU |
//! | [`Isa::Avx2`] | x86_64 | 256-bit AVX2 + FMA | 8 / 4 | if the CPU has AVX2 and FMA |
//!
//! AVX-512 is deferred (design §4.7); an AVX-512 machine runs [`Isa::Avx2`]. Every
//! variant exists on every target, but the code of an ISA is compiled only for its
//! architecture and [`Isa::is_available`] is false elsewhere.
//!
//! ## Dispatch
//!
//! - On aarch64 NEON is part of the base architecture, so [`Isa::Neon`] needs no
//!   detection. On x86_64, `is_x86_feature_detected!` for `avx2` and `fma` decides
//!   [`Isa::Avx2`].
//! - [`Isa::detect`] picks the widest available ISA: NEON on aarch64, AVX2 if detected
//!   on x86_64, else scalar.
//! - [`P2pKernel::new`] fails with [`IsaUnavailable`] for an ISA the machine cannot
//!   run, so a kernel always holds an available ISA. [`P2pKernel::detect`] uses
//!   [`Isa::detect`]. Every ISA is selectable explicitly, for tests and benchmarks.
//! - [`P2pKernel::evaluate`] checks the slice lengths and calls one entry point per ISA;
//!   the branch runs once per call.
//! - The crate never reads an environment variable to choose an ISA.
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
//!   bits. Results differ between ISAs, and from the reference, within the accuracy of
//!   design §3, requirement 2. The scalar path uses the reference's formulas in the
//!   reference's order and equals it bit for bit wherever r² ≠ 0.
//! - No allocation, no threads, no MPI; the kernel writes only into the caller's
//!   slices. Mismatched lengths panic, as in the reference.
//!
//! ## Unsafe
//!
//! `unsafe` is allowed only in the architecture modules and in the dispatch that calls
//! their `#[target_feature]` entry points (design §5.4): calling an entry point after
//! [`P2pKernel::new`] has checked the ISA, and vector loads and stores from slices
//! whose bounds the code has checked. Every block carries a `// SAFETY:` comment, which
//! the crate-level lints `unsafe_op_in_unsafe_fn` and
//! `clippy::undocumented_unsafe_blocks` (both denied) enforce. Every public function is
//! safe. The scaffold has no `unsafe` yet.
//!
//! ## Testing on each architecture
//!
//! Every test that runs the kernel prints the ISAs it ran; `--show-output` shows it
//! for passing tests.
//!
//! - aarch64 (Apple silicon, the CI arm64 runner): `cargo test -p nd-fmm-simd --
//!   --show-output` runs scalar and NEON.
//! - x86_64 (the CI x86_64 runner): the same command runs scalar and, if the CPU has it,
//!   AVX2 + FMA.
//! - The release-mode accuracy tests (exhaustive and sampled, from Phase 3S T4 on) are
//!   `#[ignore]`d: `cargo test -p nd-fmm-simd --release -- --ignored --show-output`.
//! - The other architecture's code is checked by
//!   `cargo clippy -p nd-fmm-simd --all-targets --target <other> -- -D warnings`, with
//!   `<other>` `x86_64-apple-darwin` on Apple silicon or `aarch64-unknown-linux-gnu`
//!   on x86_64 Linux (installed with `rustup target add`).
//! - On an Apple silicon Mac with Rosetta 2, `cargo test -p nd-fmm-simd --target
//!   x86_64-apple-darwin` should run the x86_64 build, for a quick check before CI.
//!   This is unconfirmed: the development machine had no Rosetta 2 when the crate was
//!   created, and the binary failed with "Bad CPU type in executable". Rosetta also
//!   emulates the estimate instructions, so accuracy contracts count only from real
//!   x86_64 hardware.
//!
//! ## Toolchain
//!
//! The design relies on language features of Rust 1.86, 1.87 and 1.89. Confirmed on
//! stable rustc 1.98.0 for both `aarch64-apple-darwin` and `x86_64-apple-darwin`, each
//! with a compiled example in the test module `toolchain`:
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
//!   compile, for the deferred path.
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
mod isa;
mod kernel;
#[cfg(test)]
mod toolchain;

pub use isa::{Isa, IsaUnavailable};
pub use kernel::P2pKernel;

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

/// Seals [`SimdScalar`].
mod sealed {
    /// Implemented only for f32 and f64.
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
