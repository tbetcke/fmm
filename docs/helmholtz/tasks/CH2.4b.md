# CH2.4b Yukawa P2P on a shared skeleton

Started 2026-10-31 on `tbetcke/h-ch2-4b`. Roadmap: "CH2.4b Yukawa P2P on a shared skeleton"
(H2; the second half of CH2.4, seventh on the priority list). Proposal: §10 CH2.4; §2
(constraint 1), §5.5, §5.7, §6.1, §6.2.

## Goal

`P2pKernel::evaluate_yukawa`, the potential e^{−κr}/r and its gradient for real charges on
every ISA in both precisions, built on a P2P loop skeleton that Laplace's `evaluate` shares,
with Laplace's P2P unchanged bit for bit and its benchmarks within noise on the three
machines.

## Acceptance (proposal §10, verbatim)

> an accuracy contract like simd-p2p.md's; cost per pair against Laplace; Laplace P2P bit
> for bit unchanged and at least 90 % of the spike (C3S.4)

CH2.4b's *Done when* (roadmap): CH2.4's: an accuracy contract like simd-p2p.md's (§4.9
onwards) on every ISA; cost per pair against Laplace on the three machines, beside
green-kernels' modified Helmholtz; Laplace's P2P bit for bit, its benchmarks within noise
(D9), the plane-wave M2L steps (`plane_wave_parts`) on locust included (CH2.4a's question
2, signed off 2026-10-31), and at least 90 % of the spike (C3S.4).

## Read first

- `fmm-simd/CLAUDE.md`; `fmm-simd/src/arch/p2p.rs` (`p2p_body`, `block`), `arch/mod.rs`
  (`Simd`, the dispatch macros), `arch/exp.rs`, `arch/scalar.rs`, the entry points in
  `arch/{neon,avx2,avx512}.rs`, `src/kernel.rs`, `src/lib.rs`, `tests/p2p/`.
- docs/design/simd-p2p.md §3 (requirements), §4.2–§4.5, §4.8 (the dipole contract's
  derivation, the model for this one), §4.9.
- `nd_fmm_ref::helmholtz::p2p::{p2p_yukawa, direct_sum_yukawa}` (`YukawaTerm`).
- green-kernels' `modified_helmholtz_3d.rs` at the pinned commit; `spikes/p2p-simd`
  (`examples/compare.rs`, `src/green.rs`).
- The D9 scripts of CH2.0 and CH2.4a (previous sessions' scratchpads: `d9_run.sh`, `d9.py`,
  `bits.sh`, `d9_pwp.sh`), recreated in this session's scratchpad.

## Changes

`nd-fmm-simd` (library) and `spikes/p2p-simd` (one new example), plus docs. No manifest,
dependency, convention, table, layout or `nd-fmm-plan` change. `nd-fmm-exec` does not call
the Yukawa kernel yet (CH2.1 does).

- **The skeleton** (`arch/p2p.rs`). A crate-private trait for the pair arithmetic,
  `PairKernel<T, S: Simd<T>>: Copy`, with one `#[inline(always)]` method that adds one
  source's term at one target vector, from d = x − y, r² and the broadcast charge, into φ
  and, if `G`, the gradient. `p2p_body`, `sub_block` and `block` take the pair kernel as a
  value and keep everything else: blocks of K, K/2, …, 1 vectors, the padded tail, the loads,
  the broadcast, d and r², the stores. Laplace's pair kernel is a zero-sized `Laplace` whose
  method is the present source-loop body, operation for operation:
  ρ = rsqrt_masked(r²); φ = fma(q, ρ, φ), or t = q ρ, φ = φ + t, t₃ = t (ρ ρ),
  gₖ = fnma(t₃, dₖ, gₖ). The several-vector and dipole bodies are not touched (each could
  become a later user; rule of two, proposal §6.2).
- **Yukawa's pair kernel**, `Yukawa { kappa, minus_kappa, one }` (vectors splatted once per
  call):
  ```text
  ρ = rsqrt_masked(r²)              as Laplace (+0 at r² = 0)
  r = r² ρ                          (0 at r² = 0, so e = 1 and the term is 0)
  e = exp(−κ r)                     arch::exp::exp (CH2.4a), x = (−κ) r ≤ 0
  potential only:  φ = fma(q e, ρ, φ)
  with gradients:  t = (q e) ρ;  φ = φ + t;  f = fma(κ, r, 1);  t₃ = (t f)(ρ ρ);
                   gₖ = fnma(t₃, dₖ, gₖ)
  ```
  so the term is q e^{−κr}/r and the gradient −q (κr + 1) e^{−κr} d / r³. At κ = 0 every step
  equals Laplace's (q e = q, f = 1 exactly), so the Yukawa kernel at κ = 0 gives `evaluate`'s
  bits on every ISA: a test, and a check that the skeleton is shared rather than copied.
  The f32 gradient's range is Laplace's (|t f| ≤ |q| ρ (1 + κr) and e^{−κr}(1 + κr) ≤ 1).
- **Entry points** per ISA and precision, `p2p_yukawa_f32`, `…_f32_gradient`, `…_f64`,
  `…_f64_gradient`, `#[inline(never)]`, with blocks of `YUKAWA_K` target vectors per ISA
  (chosen from K ∈ {1, 2} by the NEON and Kathleen timings: the exponential's 19 (f64)
  constants spill on AVX2's 16 registers); a dispatch macro `p2p_yukawa_dispatch!` as the
  dipoles'. Laplace's entry points pass `Laplace` and are otherwise unchanged.
- **The scalar ISA**, `scalar::p2p_yukawa`: `nd_fmm_ref::helmholtz::p2p::p2p_yukawa`'s loop
  (r = √r², e = exp(−(κ r)), (q e) / r, (q ((−κr − 1) e / (r² r))) dₖ), r² = 0 skipped, so
  it equals the reference bit for bit wherever r² ≠ 0, as the Laplace scalar path does.
- **Public API**, `P2pKernel::evaluate_yukawa(&self, kappa: T, sources, charges, targets,
  potential, gradient: Option<…>)`, with `evaluate`'s semantics (+=, no 1/(4π), sources in
  input order, r² = 0 skipped, chunk and target-position invariance) and its panics, plus a
  panic unless κ is finite and κ ≥ 0 (the exponential is specified for x ≤ 0; the reference
  takes any finite κ, §3.15's range is Im k ≥ 0). `sealed::Sealed::p2p_yukawa`; the crate
  docs' table and a section "Yukawa".
- **The contract** (design §4.10, new: "The Yukawa P2P (Helmholtz CH2.4b)"), derived to
  first order as §4.8 does, against the reference's term in the same precision, in the form
  (a + b κr) u_T: the potential term relative to |q| e^{−κr}/r, each gradient component
  relative to |q| (1 + κr) e^{−κr}/r², on the kernel domain of §3.13 (f32 gradients from
  2⁻⁸⁴ as Laplace's) and 0 ≤ κr ≤ −x_min (every e^{−κr} normal); beyond, the exponential's
  clamp makes the term at most 2^(e_min + 2) |q| / r. The κr part is the conditioning of
  e^{−κr} in r, which the reference shares (its own r carries about 2.5 u κr); a first count
  gives a ≈ 16 and 48, b ≈ 10. The measured maxima go beside it.
- **Cost per pair**, `spikes/p2p-simd/examples/yukawa.rs` (the spike already holds
  green-kernels and `nd-fmm-validate`'s W1 cells): on the gathered W1 cells n_t = 16, 32, 64,
  f32 and f64, potential and potential with gradient, one thread, κ = 1 in leaf-scaled
  units (the exponential has no branch, so the time does not depend on κ): Gpairs/s of
  `evaluate_yukawa` and `evaluate` on every ISA, green-kernels' `ModifiedHelmholtz3dKernel`
  and `Laplace3dKernel`, the ratio Yukawa / Laplace per kernel, and the accuracy of each
  against `direct_sum_yukawa` (green-kernels' 1/(4π) and gradient sign checked on a pair
  first, as C3S.6 did for Laplace).
- **Docs**: design §4.10 and the Outcome; `fmm-simd/CLAUDE.md` (the skeleton rule, the Yukawa
  rule, the entry points of the inlining check, the oracle); the crate README if it lists
  the kernels; the report `fmm-validate/results/helmholtz-ch2-4b.md`.

## Tests that prove it

All on every ISA the machine offers (`common::kernels`, printing those it cannot run).

- **Terms** (`tests/p2p/yukawa.rs`): per pair against `p2p_yukawa` in the same precision, r²
  log-uniform over the kernel domain (f32 gradients from 2⁻⁸⁴), κr log-uniform in
  [10⁻⁶, −x_min] plus κ = 0, |q| ≤ 1: within the contract (a + b κr) u_T; 10⁴ pairs in
  debug, 10⁶ per precision in an ignored release test that prints the maxima by κr decade.
- **Sums**: against `direct_sum_yukawa` on the W1 sets and random sets to 4,096 sources,
  κ ∈ {0.1, 1, 10} (leaf units): within 1e-14 (f64) and 1e-6 (f32) relative to
  Σ |q| e^{−κr}/r (Σ |q| (1 + κr) e^{−κr}/r² per gradient component), or twice the
  reference's error on the same inputs where that is larger (C3S.4's rule).
- **κ = 0**: `evaluate_yukawa(0, …)` equals `evaluate` bit for bit on every ISA, with and
  without gradients.
- **Scalar ISA**: equals `p2p_yukawa` bit for bit wherever r² ≠ 0.
- **Coincident, empty and accumulating inputs** as the reference (r² = 0 adds a zero).
- **Invariance**: chunk (sources split anywhere) and target-position invariance bit for bit
  on every ISA.
- **Every block size** (`src/arch/tests.rs`): the Yukawa body at K = 1, 2, 4 equals K = 1 bit
  for bit on every ISA; the Laplace body through the skeleton as the existing tests do.
- **Panics**: the length messages of the reference; κ < 0, NaN, ±∞.
- **Laplace**: the whole existing suite, no existing test edited.

## Checks

- `/verify`: format, Clippy default, examples and workspace all-targets, the default and
  workspace tests, `cargo doc`; `nd-fmm-simd`'s cross-target Clippy
  (`--target x86_64-apple-darwin`), its tests and its ignored release tests; the MPI list at
  2 and 4 ranks (`nd-fmm-exec` links the changed kernel); no licence check (no manifest).
- **The inlining check** on the Yukawa entry points (NEON natively, AVX2 and AVX-512 from the
  `x86_64-apple-darwin` build): no call in any source loop, instruction counts reported.
- **Constraint 1, the code**: every existing entry point of the crate (P2P, many, dipoles,
  rsqrt, exp, matrix, waves) disassembled on `main` and on the branch, aarch64 and x86_64,
  at the default settings and with `-C codegen-units=1`, compared instruction for
  instruction (adrp pages and RIP displacements normalised); the P2P source loops must be
  identical; any other difference is reported (CH2.4a's codegen-unit finding).
- **Constraint 1, the bits**: the FMM outputs of `main` and the branch compared byte for
  byte, CH2.0's 18 cases (`bits.sh`), on the M3 Max, locust and Kathleen.

## Machine runs

D9 as signed off (five runs per side, A B B A A B B A A B, the spread max − min of `main`'s
five run medians, one-sided, one repeat of a failing row allowed with both reported), `main`
in a detached worktree with its own target directory, BLAS threads at 1. Per machine:

1. `p2p_kernels` (`nd-fmm-validate`, unchanged), `--isa <ISA>`, one thread: the gathered W1
   rows n_t = 16, 32, 64, f32 and f64, φ and φ with ∇φ: 12 rows per ISA. NEON on the M3 Max
   and locust; AVX-512 and AVX2 on Kathleen.
2. `plane_wave_parts --ps 8,12 --repeats 15`, f32 and f64, NEON / AVX-512: the steps
   `sums`, `waves`, `merge`, `gather` and the sum per box (20 rows). locust is required
   (signed off); the M3 Max and Kathleen as well, since each target partitions the crate
   into codegen units differently.
3. `tools/bench/run.sh` host, `--strategy auto`, f32 and f64, p = 3 (P2P-dominated) and 8
   (PlaneWave), N = 10⁶, 10 evaluations after 2 warm-ups: 4 rows.

Then:
- **C3S.4** on the M3 Max: the branch's NEON gathered W1 rows from (1) against the spike's
  prototype run once beside it (`spikes/p2p-simd`, its targets-in-lanes NEON kernel), at
  least 90 %.
- **Cost per pair**: `yukawa` on the M3 Max (outside the sandbox, quiet machine), locust
  (idle checked) and Kathleen (one `singlenode` job; AVX2 and AVX-512).
- **Contract on x86**: Kathleen `check.sbatch simd` (the ignored release tests on scalar,
  AVX2 and AVX-512). CI's `run-tests-simd` covers x86_64 and arm64 again.

Kathleen: one `singlenode` job builds both trees and alternates them (as CH2.0's 614707).
locust: `tools/gh200/sync.sh`, `remote.sh`, the main tree as `detached-<main sha>`.

## Constraints and risks

- **Constraint 1** applies in full: the change edits Laplace's P2P body.
  1. No new runtime dispatch: the pair kernel is a type parameter resolved at compile time;
     Laplace's is zero-sized, and K and G stay const generics.
  2. No layout or table change.
  3. Laplace's bits: the same operations in the same order, checked by the suite, the
     disassembly and the 18-case byte comparison on three machines.
  4. Benchmarks: the D9 runs above and C3S.4.
  If any fails, the proposal's fallback applies: Yukawa gets its own copy of the skeleton
  and Laplace's body is restored as on `main`, reported in the PR.
- Risk: the trait method does not inline into the `#[target_feature]` entry points, or
  changes register allocation in Laplace's loops. The disassembly comparison shows it before
  any timing.
- Risk: AVX2's register pressure with the exponential's constants (memory operands); the
  Yukawa K per ISA is chosen by measurement, and AVX2's cost per pair is reported as
  measured on Kathleen.
- Risk: the contract's κr term. It is the conditioning of e^{−κr}, shared by the reference;
  if the measured b sits far below the derived one, the report says so, as §4.8 does.
- Scope: about 1,000–1,300 changed lines, half of them tests and docs, in one library crate
  and one spike example, with runs on three machines. Not split further: constraint 1 needs
  the shared skeleton and its D9 runs in one pull request, and the cost example is small.

## Decisions

- The user, 2026-10-31: "Proceed as planned" (the shared skeleton for Laplace's charges and
  Yukawa; the several-vector and dipole bodies untouched), and the D9 set "P2P +
  plane-wave + 4 FMM" (Machine runs 1–3).

## Pending runs

(filled in step 6)

## Open questions

(filled when they arise)
