# nd-fmm-simd

Purpose: hand-written SIMD kernels for the host path, P2P first: `core::arch` intrinsics
for aarch64 NEON, x86_64 AVX2 + FMA and, since Phase 6 T6, x86_64 AVX-512F (`Isa::Avx512`,
`arch::avx512`; docs/design/optimisation.md §5), a scalar fallback, and run-time ISA
dispatch (docs/design/simd-p2p.md). Since Phase 6 T2 also the
column-accumulation kernel of the host's batched M2L, `MatrixKernel` (C[:, c_j] +=
A · B[:, b_j], the chain of `nd_fmm_tables::MatrixSet::apply`, and an opt-in fused mode;
docs/design/optimisation.md §2.3, §2.11).
Phase and components: Phase 3S, C3S.3 and C3S.4 (tasks T3–T5 in docs/phase3s/); Phase 6,
C6.6 (task T2 in docs/phase6/), C6.3 (task T5: `P2pKernel::evaluate_many`, P2P of
several charge vectors), C6.7 (task T6: AVX-512, every kernel; report
fmm-validate/results/phase6-avx512.md) and C6.4 (task T8: `P2pKernel::evaluate_dipoles`, the dipole
P2P, `arch::p2p::p2p_dipoles_body`, the scalar path `arch::scalar::p2p_dipoles`; contract
in docs/design/simd-p2p.md §4.8, tests in `tests/p2p/dipoles.rs`) and C6.5 (task T13: the
plane-wave kernels, `WaveKernel` with `WaveTables`, bodies in `arch::waves`, tests in
`tests/waves.rs`). Helmholtz CH2.4a: the exponential e^x for x ≤ 0 (`arch::exp`, the public
`exp::exp_slice`, tests in `tests/exp`; docs/design/simd-p2p.md §4.9), for the Yukawa P2P of
CH2.4b. Helmholtz CH2.4b: `P2pKernel::evaluate_yukawa`, the Yukawa P2P e^{−κr}/r, on the P2P
loop skeleton `arch::p2p::p2p_skeleton` that `evaluate` shares (the pair kernels
`arch::p2p::{Laplace, Yukawa}`), the scalar path `arch::scalar::p2p_yukawa`; contract in
docs/design/simd-p2p.md §4.10, tests in `tests/p2p/yukawa.rs`.

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  P2P implements §3.1 and excludes a pair by r² = 0 (§3.13, "Fast kernels"); its
  accuracy contract holds on the kernel domain of §3.13.
- `P2pKernel::evaluate` keeps the signature and semantics of `nd_fmm_ref::p2p::p2p`:
  it adds (+=) into its outputs, applies no 1/(4π), and each target adds its sources in
  input order. Chunk and target-position invariance hold bit for bit on every ISA.
- `P2pKernel::evaluate_many` (Phase 6 T5; `arch::p2p::p2p_many_body`, the scalar path
  `arch::scalar::p2p_many`) gives every (target, vector) exactly the instruction sequence
  `evaluate` gives that vector's charges: the distance and ρ are formed once per source
  and target vector and shared by a group of vectors held in registers (the `MANY_*`
  constants per ISA), and nothing per vector may contract or reorder. So vector v equals
  `evaluate` of vector v bit for bit on every ISA, for every group and block size (tested:
  `tests/p2p/many.rs` against `evaluate`, `src/arch/tests.rs` every K and group against
  `p2p_body`); m = 1 runs `evaluate`. The charges are point-major (m per source), the
  outputs vector after vector.
- `P2pKernel::evaluate_dipoles` (Phase 6 T8) keeps the semantics of
  `nd_fmm_ref::p2p::p2p_dipoles` as `evaluate` keeps `p2p`'s (+=, no 1/(4π), sources in
  input order, chunk and target-position invariance, r² = 0 skipped; the scalar path bit for
  bit the reference). Its gradient is formed as ρ³ (dₖ − (3 s ρ²) wₖ), never with ρ⁵, so f32
  stays in range (CONVENTIONS §3.13, "Range of the terms in f32"). Contract: 32 u_T (φ,
  relative to |d| / r²) and 176 u_T (∇φ per component, relative to |d| / r³), derived in
  docs/design/simd-p2p.md §4.8; measured at most 10.5 u and 33.8 u on NEON over 10⁶ pairs.
- The P2P skeleton (Helmholtz CH2.4b; proposal §6.2): `arch::p2p::p2p_skeleton` holds the
  blocks, tails, loads, broadcast, d, r² and stores; a `PairKernel` (a `Copy` type whose
  `#[inline(always)]` method does everything after r²) is a type parameter. Laplace's is the
  zero-sized `Laplace`, whose method is `evaluate`'s source loop operation for operation, and
  `p2p_body` is the skeleton with it. Constraint 1 of the Helmholtz proposal binds every
  change to the skeleton or to `Laplace`: Laplace's entry points must stay the same
  instruction for instruction (CH2.4b: every function of the crate identical to `main`'s,
  aarch64 and x86_64, default and one codegen unit), its bits unchanged and its benchmarks
  within noise (D9). Nothing per kernel may become a run-time value in the loop.
- `P2pKernel::evaluate_yukawa` (Helmholtz CH2.4b) keeps the semantics of
  `nd_fmm_ref::helmholtz::p2p::p2p_yukawa` as `evaluate` keeps `p2p`'s (+=, no 1/(4π),
  sources in input order, chunk and target-position invariance, r² = 0 skipped; the scalar
  path bit for bit the reference) and panics unless κ is finite and κ ≥ 0. Its pair kernel
  forms ρ, r = r² ρ, e = exp((−κ) r) (`arch::exp::exp`), (q e) ρ and, with gradients,
  ((q e) ρ · fma(κ, r, 1)) (ρ ρ) dₖ; at κ = 0 every step is Laplace's, so the vector kernels
  then equal `evaluate` bit for bit (tested; keep it so). Contract: (20 + 12 κr) u_T (φ,
  relative to |q| e^{−κr}/r) and (52 + 12 κr) u_T (∇φ per component, relative to
  |q| (1 + κr) e^{−κr}/r²) for κr ≤ 64 (f32) or 512 (f64), derived in
  docs/design/simd-p2p.md §4.10; measured at most 10.2 u at small κr and 3 u per unit of κr
  beyond, over 10⁶ pairs on NEON (M3 Max), AVX2 and AVX-512 (Kathleen). One target vector
  per block on every ISA (`YUKAWA_K`; measured against two on NEON and AVX-512, AVX2 by its
  16 registers).
- `MatrixKernel::accumulate_columns` gives every element of C the chain of
  `MatrixSet::apply` (k ascending, multiply and add rounded apart; `MulAdd::Separate`)
  or one fma per step (`MulAdd::Fused`), from its own row of A and column of B only: the
  bits do not depend on the ISA, the row block, the column group or the other columns of
  a call, and every ISA equals the scalar layer exactly. Keep it so; never let a step
  contract or reassociate.
- `WaveKernel` (Phase 6 T13; `arch::waves`): `merge`, `gather`, `waves` and `sums` give
  every value the chain of `nd_fmm_tables::plane_wave`'s `shift_add`, `add` + `shift`,
  `waves_from_modes` and `sums_from_waves` (one fma per product and sum, the order of each
  output fixed), from its own inputs only: every ISA equals the scalar layer bit for bit,
  whatever W, the register block (`waves`: 2 on AVX2's 16 registers, 4 elsewhere) or the
  padding. `WaveKernel::tables` builds the Fourier passes' tables in the kernel's layout
  (rows padded to its lanes with zeros); tables of one kernel are refused by another lane
  count. Slice the rows and offsets once per block and iterate them: bounds checks inside
  the loops cost NEON's `waves` half its time (Phase 6 T13). `tests/waves.rs` restates the
  reference loops and checks every ISA bit for bit.
- `arch::exp::exp` (Helmholtz CH2.4a): e^x within 4 u_T relative on [x_min, 0], x_min =
  (e_min + 1) ln 2 (every result normal), exactly 1 at ±0, the value at x_min below it
  (design §4.9; measured 1.324 u₃₂ exhaustively and 1.228 u₆₄ on 10⁷ samples, on every ISA). It
  is written once over `Simd` from IEEE-rounded operations only (`fma`, `fnma`, `sub`, `max`,
  and `scale_pow2`'s integer add on the bits), so every ISA equals the scalar layer bit for
  bit (`tests/exp/invariance.rs`); keep it so: no estimate instruction, no ISA-specific
  step. Its constants are `SimdScalar`'s (`sealed::Sealed::EXP`).
- Adding items to the crate can change rustc's partition into codegen units and with it the
  inlining in untouched functions (Helmholtz CH2.4a: NEON's `sums_f32` called `sums_body`'s
  first closure out of line in the default release build; with `-C codegen-units=1` every
  existing function was identical to `main`'s). Compare a change's existing entry points with
  `main`'s at the default settings and with one codegen unit, and report both.
- Generic over `T: SimdScalar` (f32 and f64, sealed); no allocation, threads or MPI in
  a kernel (`WaveKernel::tables` allocates, once). Write only into the caller's slices.
- Hand-written `core::arch` intrinsics only: no portable SIMD crate (`pulp`, `wide`,
  `std::simd`), no C or assembly, no SVE. Stable Rust.
- Dispatch: `Isa::detect` once at construction (NEON on aarch64; on x86_64 AVX-512F
  with AVX2 + FMA, else AVX2 + FMA, by `is_x86_feature_detected!`; else scalar). Every
  `Isa` variant exists on every target; the code of an ISA compiles only for its
  architecture (`cfg`). Never read an environment variable to choose an ISA.
  `Isa::Avx512` is available where the CPU has AVX-512F, AVX2 and FMA, and since Phase 6
  T11 (decision 7 of docs/phase6/README.md, from T11's in-FMM measurements on Kathleen;
  optimisation.md §5.3) `Isa::detect` returns it there; `Isa::Avx2` stays selectable. The
  rule is a function of the feature set (`isa::Features`), unit-tested with forced sets;
  keep it so.
- Unsafe (design §5.4): only in `src/arch/` and in the dispatch of `src/kernel.rs` that
  calls `#[target_feature]` entry points. The uses are calling an entry point after
  `P2pKernel::new` (or the dispatch in `src/arch/mod.rs`) checked the ISA, intrinsics
  inside the vector layer, and vector loads and stores from slices whose bounds the
  code checked. The layer trait `arch::Simd` is safe to call: its methods take the
  ISA's zero-sized token (`Neon`, `Avx2`, `Avx512`), which exists only on a CPU with
  the ISA; the entry points make it with `new_unchecked`, the tests with `try_new`. Every block
  has a `// SAFETY:` comment; the crate denies `unsafe_op_in_unsafe_fn` and
  `clippy::undocumented_unsafe_blocks`. Every public function is safe. Prefer safe
  `#[target_feature]` functions and safe intrinsics (crate docs, "Toolchain") wherever
  they remove an `unsafe` block.
- ISA coverage: every test that runs a kernel prints the ISAs it ran and those this
  machine cannot run (`common::kernels` in tests/p2p, `common::isas` in tests/rsqrt and tests/exp,
  `kernels` in tests/matrix.rs and `on_every_isa!` in src/arch/tests.rs do it); on an
  x86_64 CPU without AVX-512F its tests skip with "not available here: … avx512", never
  fail (GitHub's x86_64 runners have it about half the time; Kathleen always). A task
  report lists which ISAs ran on which machine, from the test output, and never reports
  a path that did not run as passing. Accuracy contracts count only from real hardware,
  not from Rosetta 2.
- Inlining check: a missing `#[inline(always)]` on a vector-layer method turns every
  intrinsic into a call; the kernel still passes every test but runs many times
  slower. From T4 on, inspect the release disassembly of each entry point's inner loop
  and report its instruction count and that it contains no call. The entry points
  are `#[inline(never)]`, so they keep their names (`nd_fmm_simd::arch::neon::
  rsqrt_slice_f32`, `…::avx2::rsqrt_slice_f64`, `…::neon::p2p_f32`,
  `…::avx2::p2p_f64_gradient`, `…::neon::accumulate_f64`, `…::avx2::accumulate_f32_fused`,
  …). A P2P entry point has one source loop per block size it runs (K, K/2, …, 1, and the
  padded tail); the source loops are the innermost loops that contain the inverse square
  root. A matrix entry point's k loops are the innermost loops with R × 4 multiplies and
  adds, or fmas (Phase 6 T2: NEON 43 instructions at R = 4, AVX2 27 separate and 19
  fused at R = 2; Phase 6 T6: AVX-512 45 separate and 29 fused at R = 4). The AVX-512
  P2P source loops (Phase 6 T6, K = 2): 36, 50, 42 and 56 instructions for f32 φ, f32 φ
  and ∇φ, f64 φ, f64 φ and ∇φ (18, 25, 21 and 28 per target vector, against AVX2's 24,
  35, 29 and 42). The exponential's loops (Helmholtz CH2.4a, `…::exp_slice_f32` and `_f64`):
  NEON 27 and 39 (14 and 20 operations, the rest moves), AVX2 23 and 33, AVX-512 23 and
  29. The Yukawa P2P source loops (Helmholtz CH2.4b, K = 1; whole blocks, then the tail),
  f32 φ, f32 φ and ∇φ, f64 φ, f64 φ and ∇φ: NEON 40/40, 61/50, 52/52, 78/65; AVX2 49/49,
  78/64, 71/68, 87/84; AVX-512 37/37, 48/46, 46/46, 58/56. From the repository root:

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
wherever r² ≠ 0) and `nd_fmm_ref::p2p::direct_sum` for sums (for Yukawa
`nd_fmm_ref::helmholtz::p2p::{p2p_yukawa, direct_sum_yukawa}`, and `evaluate` itself at
κ = 0); `1/√x` in f64 (f32 inputs)
or double-double (f64 inputs) for the inverse square root; the kernel's own results for
chunk and target-position invariance and determinism. For `MatrixKernel` the matrix–vector
loop of `MatrixSet::apply`, restated in `tests/matrix.rs` (this crate cannot depend on
nd-fmm-tables), bit for bit in both modes, and the separate mode for the fused mode's
bound.
