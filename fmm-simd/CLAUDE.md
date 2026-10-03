# nd-fmm-simd

Purpose: hand-written SIMD kernels for the host path, P2P first: `core::arch` intrinsics
for aarch64 NEON and x86_64 AVX2 + FMA (AVX-512 deferred), a scalar fallback, and
run-time ISA dispatch (docs/design/simd-p2p.md).
Phase and components: Phase 3S, C3S.3 and C3S.4 (tasks T3–T5 in docs/phase3s/).

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  P2P implements §3.1 and excludes a pair by r² = 0 (§3.13, "Fast kernels"); its
  accuracy contract holds on the kernel domain of §3.13.
- `P2pKernel::evaluate` keeps the signature and semantics of `nd_fmm_ref::p2p::p2p`:
  it adds (+=) into its outputs, applies no 1/(4π), and each target adds its sources in
  input order. Chunk and target-position invariance hold bit for bit on every ISA.
- Generic over `T: SimdScalar` (f32 and f64, sealed); no allocation, threads or MPI in
  a kernel. Write only into the caller's slices.
- Hand-written `core::arch` intrinsics only: no portable SIMD crate (`pulp`, `wide`,
  `std::simd`), no C or assembly, no SVE. Stable Rust.
- Dispatch: `Isa::detect` once at construction (NEON on aarch64; AVX2 + FMA by
  `is_x86_feature_detected!` on x86_64, else scalar). Every `Isa` variant exists on
  every target; the code of an ISA compiles only for its architecture (`cfg`). Never
  read an environment variable to choose an ISA.
- Unsafe (design §5.4): only in `src/arch/` and in the dispatch of `src/kernel.rs` that
  calls `#[target_feature]` entry points. The uses are calling an entry point after
  `P2pKernel::new` (or the dispatch in `src/arch/mod.rs`) checked the ISA, intrinsics
  inside the vector layer, and vector loads and stores from slices whose bounds the
  code checked. The layer trait `arch::Simd` is safe to call: its methods take the
  ISA's zero-sized token (`Neon`, `Avx2`), which exists only on a CPU with the ISA;
  the entry points make it with `new_unchecked`, the tests with `try_new`. Every block
  has a `// SAFETY:` comment; the crate denies `unsafe_op_in_unsafe_fn` and
  `clippy::undocumented_unsafe_blocks`. Every public function is safe. Prefer safe
  `#[target_feature]` functions and safe intrinsics (crate docs, "Toolchain") wherever
  they remove an `unsafe` block.
- ISA coverage: every test that runs a kernel prints the ISAs it ran (`common::kernels`
  in tests/p2p, `common::isas` in tests/rsqrt and `on_every_isa!` in
  src/arch/tests.rs do it). A task report lists which ISAs ran on which machine, from
  the test output, and never reports a path that did not run as passing. Accuracy
  contracts count only from real hardware, not from Rosetta 2.
- Inlining check: a missing `#[inline(always)]` on a vector-layer method turns every
  intrinsic into a call; the kernel still passes every test but runs many times
  slower. From T4 on, inspect the release disassembly of each entry point's inner loop
  and report its instruction count and that it contains no call. The entry points
  are `#[inline(never)]`, so they keep their names (`nd_fmm_simd::arch::neon::
  rsqrt_slice_f32`, `…::avx2::rsqrt_slice_f64`, `…::neon::p2p_f32`,
  `…::avx2::p2p_f64_gradient`, …). A P2P entry point has one source loop per block
  size it runs (K, K/2, …, 1, and the padded tail); the source loops are the
  innermost loops that contain the inverse square root. From the repository root:

  ```sh
  # Build the release test binaries; cargo prints their paths ("Executable ...").
  cargo test -p nd-fmm-simd --release --no-run
  cargo test -p nd-fmm-simd --release --no-run --target x86_64-apple-darwin  # on Apple silicon
  # Every loop of one entry point, with its instruction count and calls.
  objdump -d -C --no-show-raw-insn target/release/deps/rsqrt-<hash> \
    | awk -v f='nd_fmm_simd::arch::neon::rsqrt_slice_f32' -f fmm-simd/tools/inner_loops.awk
  objdump -d -C --no-show-raw-insn target/x86_64-apple-darwin/release/deps/rsqrt-<hash> \
    | awk -v f='nd_fmm_simd::arch::avx2::rsqrt_slice_f32' -f fmm-simd/tools/inner_loops.awk
  ```

  On x86_64 Linux, build the native binary and cross-build `--target
  aarch64-unknown-linux-gnu` (needs a linker for it, or inspect the `.rlib` objects
  instead). Identical functions may be merged and share one symbol.
- Timings are reported, never asserted, and never taken in CI.
- Debug-mode `cargo test -p nd-fmm-simd` stays under 30 seconds. Exhaustive and large
  sampled checks are `#[ignore]` tests run in release; the debug run checks a strided
  subset.
- Before finishing:
  - `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`;
  - the cross-target clippy `cargo clippy -p nd-fmm-simd --all-targets --target <other>
    -- -D warnings`, with `<other>` `x86_64-apple-darwin` on Apple silicon or
    `aarch64-unknown-linux-gnu` on x86_64 Linux;
  - `cargo test -p nd-fmm-simd -- --show-output` (no MPI needed);
  - `cargo test -p nd-fmm-simd --release -- --ignored --show-output`, once a task adds
    ignored tests.

## Allowed dependencies
nd-fmm-math, thiserror; dev-dependencies: nd-fmm-ref, proptest.
No MPI, no octree, no rlst, no CubeCL, no external SIMD crate, and no dependency on
nd-fmm-validate (not even as a dev-dependency). Anything else needs asking first.

## Test oracle
`nd_fmm_ref::p2p::p2p` for terms and semantics (the scalar path equals it bit for bit
wherever r² ≠ 0) and `nd_fmm_ref::p2p::direct_sum` for sums; `1/√x` in f64 (f32 inputs)
or double-double (f64 inputs) for the inverse square root; the kernel's own results for
chunk and target-position invariance and determinism.
