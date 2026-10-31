# Helmholtz progress log

Newest entries at the bottom. `/next-task` appends one entry per task, in that task's pull
request, using this template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Laplace: unchanged bit for bit (how shown), or not touched
- Follow-ups:
```

## 2026-10-26: Planning imported
- Done: the proposal, draft 0.8, in `docs/design/helmholtz-fmm-proposal.md` (draft 0.7 reviewed
  against RLST 0.12 from the repositories only, nothing compiled); its Section 10 as
  `docs/helmholtz/roadmap.md`; the skills `/next-task`, `/verify` and `/milestone-gate` in
  `.claude/skills/`; root `CLAUDE.md` pointed at them.
- Tests: none.
- Checks: none run; the compile and test checks of the review are CHR.2 and CHR.4.
- Decisions: none yet; D0 (the proposal as the plan) and D9 (noise) come first.
- Laplace: not touched.
- Follow-ups: `/next-task` (D0, D9, then CHR.2 and CHR.4).

## 2026-10-26: CHR.2 Move to RLST 0.12
- Done: `rlst` 0.12.0 for the workspace and `octree/` (no source change; green-kernels'
  `[patch]` stays on 0.10.2); `deny.toml` and the CI job `run-licence-check` (fftw-sys
  stated as GPL); `cmake` dropped from CI (no crate runs it: trap job 608744 and on
  locust), kept unused on locust and Kathleen; docs.
  Report `fmm-validate/results/helmholtz-chr-2.md`.
- Tests: no new test; 48 Laplace outputs of `main` and the branch compared byte for byte
  (identical); the licence check's negative case with FFTW enabled.
- Checks: ran `/verify` on the M3 Max (all pass; MPI at 2 and 4), Kathleen 608726 (all pass
  but the known `spikes/p2p-simd` failures), 608727 (correctness, 2 nodes), 608733 (weekly
  examples), 608744 (cmake trap), locust (every check of root `CLAUDE.md` but its MPI
  list, dropped by the user; the ignored CUDA tests; the cmake trap). Nothing pending.
- Decisions: D0 and D9 signed off 2026-10-26; D9 clarified the same day: one-sided
  (faster passes with identical bits), five runs per side in A B B A order from CH2.0 on,
  never pooled; the licence check a separate CI job.
- Laplace: unchanged bit for bit (48 full outputs, fixtures, bit-identity and invariance
  tests); benchmarks within noise (D9 as clarified) on the M3 Max, Kathleen and locust.
- Follow-ups: CHR.4 (next runnable); CHR.3's `fftw` job needs an allowance for the fftw-sys
  clarification.

## 2026-10-27: CHR.4 FFTW on the machines
- Done: `tools/fftw/probe.sh` (with a README) prints a machine's FFTW as RLST sees it and
  runs RLST 0.12's own tests with `fftw_system` in a copy outside the workspace; the
  `fftw-probe` workflow runs it on CI. Records in `tools/m3max/machine.md` (new),
  `tools/gh200/machine.md`, `tools/kathleen/machine.md`, `tools/ci/machine.md` (new).
  Report `fmm-validate/results/helmholtz-chr-4.md`.
- Tests: no workspace code changed. RLST's `cargo test --locked --features fftw_system`
  (789 passed, `fftw_check_allocations` 10 checks, `fftw_determinism` 10 scenarios) on
  each machine. FFTW: M3 Max Homebrew 3.3.11 without SIMD; locust spack 3.3.11 NEON
  (loaded from the view, not the RPM's 3.3.8); Kathleen module 3.3.10
  SSE2/AVX/AVX2/AVX-512; CI Ubuntu 3.3.10 SSE2/AVX. Both precisions everywhere.
- Checks: ran `/verify`'s default and workspace checks on the M3 Max, `shellcheck` and
  `dash -n` on the probe, the probe on the four machines (Kathleen job 608854, CI run
  40481567890), and PR #127's CI. Nothing pending.
- Decisions: none.
- Laplace: not touched (no crate, manifest or lock file changed).
- Follow-ups: S9 (CH0.9) is unblocked; its FFTW timings on the M3 Max are scalar code (R22),
  an input to D7.

## 2026-10-27: CH0.1 S1: rotation blocks at large degree
- Done: the spike `spikes/rotation-blocks/` (a member outside the default members): the
  Gumerov–Duraiswami y-rotation blocks in the orthonormal real basis (N Rₙ in the
  storage of CONVENTIONS §3.6), in a compact cos/sin layout, with z-y-z blocks built
  from them; an mpmath reference (`tools/reference.py`, Wigner's sum at 0.61 n + 40
  digits, the real-basis map checked against the definition). Report
  `spikes/rotation-blocks/SPIKE_REPORT.md`.
- Tests: six in `tests/accuracy.rs`, all passing in release: 12,914 sampled reference
  entries at n ≤ 1000 (worst 5.0e-15 over six angles, 3.4e-15 at the octree's); ‖O Oᵀ − I‖ ≤ 4.4e-14 to
  n = 1000; N-conjugated `rotation::blocks` at p ≤ 30 (1.8e-14); the definition at
  n ≤ 20; the β/π − β sign rule to n = 1000 (6.7e-15, not bit for bit); the group
  property at n = 500 (1.2e-14). Conjugated blocks: 1e-14 lost from n = 28–43, 1e-8
  from n = 94–140, non-finite at 515. Time: 1.8 ns per stored value (3.8 ms at p = 144,
  1.2 s at p = 1000), about 4.5× faster than `rotation::blocks` + N at 20 ≤ p ≤ 82
  (2.2× per stored value; the compact y-rotation set has half the values).
- Checks: ran `/verify` on the M3 Max (default and workspace checks, `cargo doc`, the
  licence check, no fftw-sys; the spike's tests with `--include-ignored` in release);
  `reference.py --check` byte for byte. No machine runs. Nothing pending.
- Decisions: signed off 2026-10-27, as recommended (report §8): GD at every degree for the
  Helmholtz tables, the conjugated blocks only as the test reference at p ≤ 30; the
  generate-or-store threshold of §5.2/§7 a memory rule from the node budget, with the
  per-call generation cost (large near the root) measured in CH0.8 or CH2.2.
- Laplace: not touched (only the spike, its member entry and its lock-file entry).
- Follow-ups: CH1.2 implements GD with the wedge-only recursion and the telescoped,
  exponent-carrying Legendre functions (report §2.3); §3.15 item 5 takes the sign rule.

## 2026-10-27: CH0.2 S2: coaxial coefficients
- Done: the spike `spikes/coaxial/` (a member outside the default members): the
  Gumerov–Duraiswami recursions in m and n for (S|R) and (R|R) = (S|S) in the real
  orthonormal basis, in the triangle n′ ≥ n with the index symmetry, and the halved
  table; the coefficients by projection on a sphere (FMM3D's stabilised point and shoot,
  least squares from value and radial derivative); the FMM-weighted error (weights from
  the jₙ envelope over the box); an mpmath reference (`tools/reference.py`, the Gaunt sum,
  checked against the addition theorems to 5.0e-49) and a full-table arbitrary-precision
  check (`tools/full_table.py`). Report `spikes/coaxial/SPIKE_REPORT.md`.
- Tests: seven in `tests/accuracy.rs`, all passing in release: 21,809 reference entries in
  120 configurations (p ≤ 371, k a ≤ 300, real, complex, imaginary k), projection
  ≤ 1.2e-12 FMM-weighted everywhere, recursion ≤ 8.0e-15 where stable; whole tables at
  k a = 40 ≤ 8.6e-14; start values 3.3e-14; (R|R) imaginary parts structurally zero by
  the recursion and below 4.4e-14 (M2M weights) by the outer-sphere projection;
  i^{n−n′} (E|F) real for imaginary k; the triangle shared bit for bit by both variants;
  the rectangular table a block of the square bit for bit.
- Findings: R2 confirmed. For real k the recursion meets 1e-12 weighted to k a = 50 and
  fails beyond (1.6e-8 at 100, O(1) from 200), independent of ε and p; strongly damped
  and imaginary k show no instability to 300. Per-entry relative error is unreachable at
  large p by any f64 method (tables span 10²⁰⁰). Time per table (one thread): recursion
  85 ms, projection 5.1 s at p = 371; a halved M2L table is 132 MiB there.
- Checks: ran `/verify` on the M3 Max (format, Clippy default, examples, workspace and the
  spike's all-targets, `cargo doc`, the default (823) and workspace (884) tests, the
  licence check, no fftw-sys; the spike's tests with `--include-ignored` in release);
  `reference.py --check` byte for byte. A self-review by a subagent; its findings are
  fixed (the weights, test 6, the reality check, the report's numbers). No machine runs.
  Nothing pending.
- Decisions: signed off 2026-10-27, as recommended (report §8): (1) the coaxial tables
  by the recursion (triangle, index symmetry) where Re(k) a ≤ 40, by projection above;
  (2) CH0.2's acceptance met in the FMM-weighted error, bound 1e-11, not per-entry
  relative error; (3) R2's fallback the projection instead of signatures, with the
  coaxial table memory (132 MiB per M2L table at p = 371) and the table-free projection
  carried into §7 and CH0.8; (4) §3.15 item 6 states the symmetries, (S|S) = (R|R) real
  for real k, the Yukawa powers of i and the recursion signs; the scale's radius is left
  to CH0.3 and CH1.1 (proposal §3.4's half-width until then; the spike used the
  circumradius).
- Laplace: not touched (only the spike, its member entry and its lock-file entry).
- Follow-ups: CH0.7 (S7) uses the coaxial prototypes; the table code (CH1.2/CH2.2)
  implements the rule of question 1; §3.15 item 6 takes the symmetries and the Yukawa
  phases; CH0.3 and CH1.1 fix the scale's radius; CH0.8 takes the coaxial table memory
  and the table-free projection; the proposal's R2 and §7 take decision 3 at its next
  draft.

## 2026-10-28: CH0.3 S3: radial functions
- Done: the spike `spikes/radial/` (a member outside the default members): jₙ, hₙ of
  complex argument (0 ≤ arg z ≤ π/2), iₙ, kₙ of real argument and their derivatives,
  scaled by sⁿ and the exponential factor; j and i by Miller's method in ratio form
  (FMM3D's start test), h and k upward, derivatives from neighbouring orders; an mpmath
  reference (`tools/reference.py`: J and I from mpmath, h and k from their exact finite
  sums, DLMF 10.49.6 and 10.49.12); the range, truncation and timing examples. Report
  `spikes/radial/SPIKE_REPORT.md`.
- Tests: five in `tests/accuracy.rs`, all passing in release: 13,174 reference entries in
  571 sets over every operator's range (leaves to c_leaf = 2, levels to k a = 650, orders
  to 1482, |z| from 10⁻¹² to 4550, real k, k = κ(1 + iη) with η = 0.1, 1, 10, imaginary k,
  Yukawa), worst 6.5e-14 (≤ 3.6e-15 on the leaf and low-frequency ranges); the Wronskians
  on a dense grid ≤ 5.4e-14; Yukawa's real path equal to the complex path at z = ix bit
  for bit; finite everywhere (Yukawa to κρ = 4550) and exact at z = 0; scale consistency.
- Findings: the exponential factor is needed for range (without it kₙ at the nearest
  distance underflows f64 from κa ≈ 405, and Yukawa leaves have no wavelength bound); in f64 every power-scale radius works
  (10^±54 at worst), the double factorials cannot be absorbed (10^±2000 at k a = 650);
  f32 would need a per-degree scale on low-frequency levels (D3). FMM3D 2.1.0's scale is
  min(|k| w, 1) with the full width. Yukawa's degree for a relative tolerance grows like
  √(κ a) (43 at κa = 10, 105 at 100, 324 at 1000 for 1e-10). Time: j 19–31 ns per value,
  h 5–7 ns. The reference needed exact finite sums for h and k (mpmath's `besselk` does
  not converge at large argument and moderate order).
- Checks: ran `/verify` on the M3 Max (format, Clippy default, examples, workspace
  all-targets, `cargo doc`, the default (823) and workspace (885) tests, the licence
  check, no fftw-sys; the spike's tests with `--include-ignored` in release);
  `reference.py --check` byte for byte. A self-review by a subagent; its findings are
  fixed (the projection's own h to |z| = 4550, η = 10, Yukawa's orders and both M2L
  distances, the report's numbers). No machine runs.
  Nothing pending.
- Decisions: signed off 2026-10-27, as recommended (report §8, questions 1–2): the
  scaling ĵₙ = e^{−|Im z|} jₙ/sⁿ, ĥₙ = e^{Im z} hₙ sⁿ (likewise i, k and the
  derivatives) with s_l = min(|k| a_l, 1), a_l the circumradius, the exponential factor
  on every level and kind of k, no double factorials ("Circumradius, exp always"); kₙ
  with k₀ = e^{−x}/x ("k₀ = e^{−x}/x").
- Laplace: not touched (only the spike, its member entry and its lock-file entry).
- Follow-ups: CH1.1 writes §3.15 item 3 from the sign-offs; CH1.2 implements §2's
  algorithms (a real-arithmetic j for real k); D3 takes the f32 range finding; the
  proposal's next draft corrects §3.4 (FMM3D's width; the factor on every kind of k) and
  §5.5/CH2.3 (Yukawa's degree grows like √(κ a)).

## 2026-10-28: CH0.8a S8: workload census, first pass
- Done: the spike `spikes/census/` (a member outside the default members): W1–W4 in the
  root cube; a prototype maximum leaf width (nd-octree's points-per-leaf rule plus "refine a
  box with a point wider than c_leaf λ", the leaves handed to `Octree::new` as one key each
  with `max_fine_keys = 1`, so nd-octree balances them); per-level counts from `Plan::new`
  (boxes, occupied boxes, leaves, U, V per distance and reflection class, W, X, P2P point
  pairs); the first-pass model (formula p, the bound of §3.6 with C_T = 2 and θ = 1, the
  regime by operation count, N_φ, §7's memory and operation counts, the replicated top).
  The sweep: 378 trees (W1 to kD = 1000, W2 to 500, W3 resolved to 100 and 10⁶ at 200 and
  500, W4 at N = 10⁵ and 10⁶; c_leaf 0.25–2 and none; n = 128 and 64), ε = 10⁻⁴, 10⁻⁶, 10⁻⁸,
  22 minutes and 40 GB on the M3 Max. Report `spikes/census/SPIKE_REPORT.md`, tables in
  `spikes/census/results/`.
- Tests: six in `tests/census.rs` (the plan's eight), all passing in release and debug:
  the prototype equals nd-octree's tree leaf for leaf without a width (seven workloads);
  the width rule holds and refines; every count against a brute-force oracle from the list
  definitions on four adaptive trees; 15 distance and 34 reflection classes; the formula p
  against §3.5's tables; |h_n| against S3's reference (3.5e-15); the bound against §3.6's
  classes on RCR and widths; the cost and memory formulas against §7 and §6.7.
- Findings: the census reproduces §7 and §6.6 from real trees (W1 at kD = 500: the same
  boxes, regimes and V on RCR, M2L within 4 %; kD = 1000 hybrid 200 against 196, peak
  signatures 5.35 against 5.0 GiB; replicated top 93, 268 and 237 MiB exactly). §12's D11
  rule is degenerate: leaf cost never grows as c_leaf shrinks, so it picks 0.25 at 3–15×
  the total work. A revised rule (the largest c_leaf with no high-frequency leaf at 10⁻⁶ and
  10⁻⁸ on any workload) gives c_leaf = 1: within 2 % of the best total on resolved
  workloads to kD = 200, +22–25 % on W1 at 500 and 1000, 10.4× on under-resolved W3 at 500.
  At 10⁻⁴ the first-pass rule makes 0.5–1 λ levels H. The under-resolved W3 at kD = 500 needs
  42 GiB of signature scratch on one level (R17). Low-frequency M2L is 71–88 % of the M2L
  on W1. Batch sizes for S5 and S9 per H level in `results/batches.md`.
- Checks: ran `/verify` on the M3 Max: format, Clippy default, examples and workspace
  all-targets, `cargo doc`, the default (823) and workspace (891) tests, `cargo check
  --workspace --all-targets`, the licence check, no fftw-sys; the spike's Clippy and tests in
  release and debug. A self-review by a subagent: nothing blocking; its 16 points fixed (the
  W4 seed's wording, constraint 9's wording, the per-class test noted, the oracle extended to
  every count, report numbers). No machine runs. Nothing pending.
- Decisions: signed off 2026-10-28, "1–4 as recommended" (report §9): (1) D11 c_leaf = 1
  by the revised rule (the largest c_leaf with no high-frequency leaf on any workload at
  10⁻⁶ and 10⁻⁸), replacing §12's degenerate rule, revisable by CH0.8b; (2) D12 the
  workloads of §9.1 unchanged, W3 at 10⁶ and kD = 500 named the node-budget and
  leaf-rule-cost case; (3) high-frequency leaf levels at 10⁻⁴ accepted for now, the
  threshold left to CH0.8b with S6's constants and S7's p; (4) the proposal's next draft
  takes D11's revised rule, §7's numbers from `results/summary.md` once CH0.8b confirms
  them, and the full-cube scratch (9.2 GiB on level 5, 42 GiB on level 6 with the leaf
  rule).
- Laplace: not touched (only the spike, its member entry and its lock-file entry).
- Follow-ups: CH0.8b repeats the census with S7's p and S6's constants (`examples/census.rs
  -- sweep`); CH3.0 implements the maximum leaf width in nd-octree with the signed-off
  c_leaf; S5 and S9 take their batch sizes from `results/batches.md`; CH4.1 and CH8.1 the
  scratch and replicated-top numbers.

## 2026-10-28: CH1.4 FMM3D convention check and fixtures
- Done:
  - The draft §3.15 (`docs/helmholtz/conventions-3.15-draft.md`), items 1–9 of proposal
    §4 from the proposal and the S1–S3 sign-offs. Item 1 is fixed by the check; the rest
    are marked proposed or open.
  - `tools/fmm3d/driver.f90` gains the modes `helmholtz-check` (the two-point check,
    which stops unless FMM3D computes the draft's kernel) and `helmholtz` (`h3ddirectcg`
    or `hfmm3d` on a problem file). Its Laplace usage is unchanged.
  - `tools/fmm3d/build.sh driver`; `tools/fmm3d/helmholtz.py` (formats, the
    identification among 16 candidates, the R12 sweep).
  - `tools/fixtures/gen_helmholtz_fmm3d.py` writes `fmm-ref/fixtures/helmholtz_fmm3d.json`
    (687 KB): `h3ddirectcg` with 200 sources and 40 targets; k = 1, 25, 1 + 0.5i,
    25 + 2.5i, i, 25i; real and complex charges.
  - The Kathleen steps in `fmm3d.sbatch`.
  - Report `fmm-validate/results/helmholtz-ch1-4.md`.
- Tests:
  - The two-point check on the M3 Max and on Kathleen. Both FMM3D routines compute
    q e^{ikr}/(4πr) with the gradient in the target and the charge unconjugated, for
    real, complex and imaginary k, to 2.6e-16; exactly one of 16 candidate conventions
    matches each case.
  - Every fixture value agrees with the draft's kernel in mpmath at 30 digits to
    ≤ 1.0e-15 relative ℓ2. `--check` is byte for byte on the M3 Max at 1 and 8 threads,
    and within 1.7e-16 on Kathleen (gcc 12.3).
  - The Laplace mode of `main`'s driver and the branch's write byte-identical outputs.
- Findings (R12, M3 Max at N = 10⁵, Kathleen at N = 10⁶):
  - `hfmm3d` with imaginary zk meets its eps for κD ≤ 30, and at eps ≥ 1e-6 everywhere
    (to 1.7×).
  - For κD 50–300 at eps 1e-9 and 1e-12 it misses in 9 of 12 cube cells and 10 of 12
    sphere cells at N = 10⁵, and in 12 and 10 of 12 at N = 10⁶ (Kathleen), by up to
    12,000×.
  - On the sphere at N = 10⁶ and eps 1e-12, the gradient is off by 2× for every k, real
    k included.
  - Complex k with Im kD = 100 misses alike; k = 100 + 30i passes.
  - The misses begin where FMM3D's decay cut-off leaves an M2L level with Im(k) w ≳ 12
    active (a hypothesis that fits every cell).
- Checks:
  - Ran `/verify` on the M3 Max: format, Clippy default, examples and workspace
    all-targets, `cargo doc`, the default (823) and workspace (891) tests. No Rust
    changed.
  - Also shellcheck, `bash -n`, the Python 3.9 parse and the generator's `--check`.
  - Kathleen job 614095: the check and the fixtures. A self-review by a subagent: its two
    blocking points are fixed (the record cases' label, the far-field factor 4π ik in
    the draft), and so are the report's counts.
  - Kathleen job 614099 (node-c11b-009, 40 threads): the check, the fixtures and R12 at
    N = 10⁶, all passed. Nothing pending.
- Decisions: signed off 2026-10-28, "1–3 as recommended" (report §5):
  1. the draft §3.15 is CH1.1's starting point, with no correction from the check. Where
     the factor 4π ik is applied is left to CH1.1, which takes the draft's proposal (the
     crossing operators M2L, M2P and P2L) or the alternative;
  2. CH2.5 runs and is not skipped. It compares against FMM3D only where FMM3D met its eps
     in R12: κD ≤ 30 at every eps, and eps ≥ 1e-6 with up to 2× allowed beyond. The
     sampled direct sum is the measure, with FMM3D's error stated beside it. CH5.3
     restricts its grid the same way for Im kD ≳ 50;
  3. the proposal's next draft takes R12's result into R12's row and §9.2, and FMM3D's
     thresh rule beside §3.13's.
- Laplace: not touched (tools, docs and one fixture file only; the driver's Laplace
  outputs are byte for byte the same).
- Follow-ups:
  - CH1.1 writes §3.15 from the draft and decides where 4π ik goes.
  - CH1.3 tests `nd-fmm-ref`'s direct sum against the fixtures, to round-off, not bits.
  - CH2.5 and CH5.3 take question 2's answer.

## 2026-10-29: CH0.7 S7: truncation under realistic geometry
- Done:
  - The spike `spikes/truncation/` (a member outside the default members): a double-double
    scalar (`src/real.rs`, about 32 digits) and generic ports of S1's rotation blocks,
    S2's coaxial tables (recursion and projection), S2's and S3's radial functions and
    S2's basis.
  - The chains C1 (P2M → M2L → L2P), C2 (P2M → M2M → M2L → L2L → L2P), C3m (M2M twice,
    M2P) and C3l (P2L, L2L twice) on all 316 offsets, with uniform, corner and W1-patch
    placements, against the double-double direct sum.
  - The sweep (`examples/sweep.rs`): k a on a half-octave grid from 0.016 to 256 (κ a to
    32 for η = 1 and Yukawa); real k, η = 0.1 and 1, imaginary k; ε from 10⁻³ to 10⁻¹⁰;
    the margin ε/2; f64 at every calibrated p on all 316 offsets.
  - Report `spikes/truncation/SPIKE_REPORT.md`, tables in `results/summary.md`.
- Tests:
  - Double-double identities to 10⁻³⁰.
  - S1's, S2's and S3's mpmath fixtures, to their f64 rounding.
  - Orthogonality at degree 300; Y(Qx) = O(Q) Y(x) on all 316 directions.
  - Recursion against projection to 2·10⁻²⁵ at k a = 90.5.
  - Test 6: every chain at high p leaves at most 10⁻²⁰ (10⁻¹⁶ where Im(k) a ≥ 16), so the
    sweep measures truncation alone.
  - The search, and the formula against §3.5's tables.
  - The record of the L2L projection failure.
  - Output of test 6 in `results/chains.log`.
- Findings:
  - p for real k is nearly flat below k a ≈ 8 (the Laplace limit) and above the formula of
    §3.5 there (30 against 22 at 10⁻⁶, 55 against 35 at 10⁻¹⁰), set by corner clusters
    and the gradient. From k a = 32 it is 5–15 % below the formula.
  - The closed form max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉) never
    under-predicts the per-level degree p* at the real-k grid points (over by ≤ 9).
  - M2M and L2L over three levels need nothing extra for real k; the two-level chain needs
    up to 3 degrees at low frequency.
  - Strong damping and Yukawa need 2–2.5 times the formula at κ a = 16–32 in the per-pair
    relative norm. η = 0.1's multi-level δ grows with Im(k) a (7 at k a = 181, > 8
    unresolved at 256, a level always pruned).
  - f64 meets every real-k p on all 316 offsets, but not strong damping from κ a = 16.
  - S2's outer-sphere projection of the shift tables fails in L2L with complex k.
- Checks: ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
  all-targets; `cargo doc` and the spike's doc with `-D warnings`; the default (823) and
  workspace (901) tests; `cargo check --workspace --all-targets`; the licence check; no
  fftw-sys. The spike's tests in release with `--include-ignored`. A self-review by a
  subagent; its five blocking points were fixed, among them the worst offset, the f64 check
  moved to all 316 offsets, the search test and the plan. The sweep ran about 20 hours on
  the M3 Max; the η = 0.1 multi-level chains at k a = 181 and 256 ran last, on the user's
  request. Nothing pending.
- Decisions: signed off 2026-10-29, "1–4 as recommended" (report §8):
  1. p* of `results/summary.md` is S7's output for S6, CH0.8b and §7; for real k the closed
     form max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉); CH3.3 confirms it
     in the full FMM;
  2. decaying kernels (η = 1, Yukawa): the per-pair tables are upper bounds; CH2.3
     calibrates Yukawa's p in the full FMM with the global norm, with pruning and θ_D;
     until then S6 and CH0.8b use the tables only below the pruning cut;
  3. the L2L tables for CH1.2, amending S2's signed-off question 1: S2's outer-sphere
     projection serves M2L and M2M only; CH1.2 builds L2L by the recursion where stable,
     checks every table in both directions' weights, and tests an inner-sphere projection
     for the rest;
  4. the proposal's next draft replaces §3.5's formula and tables by p* and the closed
     form, and records the finding for Yukawa and strong damping.
- Laplace: not touched (only the spike, its member entry and its lock-file entry).
- Follow-ups: S6 (CH0.6) and CH0.8b take p* or the closed form; CH1.2 the L2L tables;
  CH2.3, CH3.3 and CH4.3 confirm p in the full FMM.

## 2026-10-29: CH1.1 CONVENTIONS §3.15
- Done:
  - D5 and D2 signed off at the start (charges now, dipoles later; A, modules in the layer
    crates).
  - `docs/CONVENTIONS.md` §3.15, the Helmholtz family, from CH1.4's draft. Its parts are
    the kernel and factors, the angular basis, radial functions and scaling, storage,
    rotations, coaxial translations (with S2's start column and recursions), signatures
    (reserved), precision and range (f32 reserved), inputs and outputs, and verification.
  - `CONVENTION_VERSION = 3`, with §3.15 among the versioned sections.
  - The draft file reduced to a pointer that lists how §3.15 departs from it.
  - The check `tools/fixtures/check_3_15.py`.
  - The five fixtures regenerated, their headers only; the FMM3D fixture's header now
    names §3.15.
- Tests:
  - `check_3_15.py`: 101 checks in mpmath at 40 digits, all passing, for real, complex and
    imaginary k and real and complex charges. They cover the kernel's gradient and its
    k → 0 limit; the radial relations; the harmonics against Racah's through N; the series
    with 4π ik and Yukawa's with 4πκ; the leaf operators in stored form, every applied
    exponent ≤ 0; the rotation blocks; the coaxial symmetries, the start column and both
    recursions against projected tables; and P2M → M2L → L2P and the M2M and L2L chains
    in stored form against the direct sum.
  - The generators' `--check`: dipoles and FMM3D byte for byte; the harmonics fixtures
    regenerated deterministically, with a header-only diff.
  - `crosscheck_scipy.py` passed.
- Checks:
  - Ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
    all-targets; `cargo doc`; the default tests (823) and workspace tests (902), all
    passing.
  - The MPI test list at 2 and 4 ranks, 22 of 22 passing.
  - Not run: the licence check and the fftw-sys check (no manifest changed); benchmarks
    (no code path changed).
  - A self-review by a subagent. Its one blocking point is fixed: S2's sign-off 4 asks
    §3.15 for the recursions, which are now stated and checked. Its other points are
    taken: Yukawa's real tables, f32 reserved, FMM3D's thresh, the stored table
    normalisation and the M2M/L2L cancellation in the check, and the (R|R) reflection.
  - No machine runs. Nothing pending.
- Decisions:
  - Signed off 2026-10-29 at planning, each as recommended:
    1. 4π ik is applied by M2L (in its (S|R) step), M2P and P2L;
    2. stored coefficients carry e^{∓β_l}, β_l = Im(k) a_l, besides s_lⁿ;
    3. `CONVENTION_VERSION` 3.
  - Signed off 2026-10-29, "1 and 2 as recommended" (PR #135):
    1. §3.15 as written, CH1.1's acceptance;
    2. the proposal's next draft takes three corrections: §3.1 (the locals carry 4π ik),
       §3.4 (the circumradius and the per-box damping factor) and §4 item 6 (Yukawa's
       tables are real as i^{n−n′} (E|F)).
- Laplace:
  - No code path changed; only the version constant. Cached tables and tune files from
    version 2 are rejected once and rebuilt.
  - The tables come out identical; a re-tune may choose differently, so device runs with
    a tuning cache may change in their last bits, within the FMM's bounds.
  - The default, workspace and MPI tests pass with the regenerated fixtures.
- Follow-ups:
  - CH1.2 implements §3.15 in `nd-fmm-math` (D2: A), with the identities of
    "Verification".
  - CH1.3 checks `nd-fmm-ref` against the FMM3D fixtures.
  - Old `-c2-` cache files can be removed on each machine (Kathleen's quota).

## 2026-10-29: CH1.2a Orthonormal harmonics and rotation blocks
- Done:
  - CH1.2 split in the roadmap into CH1.2a (S1: harmonics and rotations), CH1.2b (S3:
    radial functions) and CH1.2c (S2: coaxial coefficients). Each fits one reviewable PR.
  - `nd_fmm_math::orthonormal`, generic over `T`. It holds:
    - `legendre`: S1's telescoped recursion, with a binary exponent per column (span
      2^±256 in f64, 2^±32 in f32), and the column divided by sin θ;
    - `harmonics` and `harmonics_derivatives`: Y, ∂θ Y and (1/sin θ) ∂φ Y in the layout
      of §3.6. They need no trigonometric calls; φ = 0 on the axis; ∂θ uses the m-ladder;
      both derivatives are regular at the poles;
    - `rotation`: the Gumerov–Duraiswami y-blocks in S1's compact layout, `supplement`
      (π − β) and `blocks(α, β, γ)` in the layout of `rotation::blocks`.
  - The fixture `fmm-math/fixtures/helmholtz_angular.json` (514 KB), from
    `tools/fixtures/gen_helmholtz_angular.py`:
    - harmonics and derivatives at seven directions to degree 1000, by the plain recursion
      at 60 and 100 digits;
    - S1's 54 rotation sets, value for value.
- Tests: `fmm-math/tests/orthonormal/`, 20 tests (2 of them ignored, run in release).
  Results in f64:
  - against the Racah harmonics through N at p = 30: ≤ 1e-13;
  - the addition formula to n = 300: ≤ 1e-12;
  - orthonormality by quadrature at n ≤ 30: ≤ 1e-13;
  - derivatives against differences: ≤ 1e-7;
  - against the fixture to n = 1000: worst 5.7e-14 for the value, 5.9e-14 for ∂θ, 5.1e-14
    for ∂φ;
  - block entries to n = 1000: worst 5.0e-15, as S1;
  - orthogonality: 4.3e-14 at n = 1000;
  - against N-conjugated `rotation::blocks` at p ≤ 30, Yₙ(Q x̂) = Oⁿ Yₙ(x̂) at p = 50, the
    group property, and the π − β rule: all within tolerance;
  - degree independence, f32 at p ≤ 8, and the panics.
- Checks:
  - Ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
    all-targets; `cargo doc` (`nd-fmm-math` with `-D warnings`).
  - The default tests (845) and workspace tests (924), all passing.
  - After the self-review fixes (the exponent loop in `legendre` and tests only), re-ran
    `nd-fmm-math`'s tests, the ignored ones in release, and workspace Clippy.
  - The generator's `--check` is byte for byte.
  - Not run: the MPI tests, because no code an MPI crate uses changed (a new module only).
  - A self-review by a subagent: nothing blocking. Its points are taken: test 2's scale,
    the rescale loop, f32 and scratch tests for the blocks, and the plan's tests as run.
  - No machine runs. Nothing pending.
- Decisions: D2 applied (A). Signed off 2026-10-30 (PR #136, CI failure): test 2's bound
  becomes max(1e-13, 2.5e-16 n), "degree-scaled". On Linux x86_64 ∂θ Y at n = 1000 came
  to 1.10e-13: glibc's `hypot` differs from macOS's in the last bit, and Yₙ's condition
  number in θ is about n.
- Laplace: not touched. `lib.rs` declares the new module; no Laplace function changed.
- Follow-ups:
  - CH1.2b, the radial functions, and then CH1.2c, the coaxial coefficients, which uses
    `orthonormal` for the projection.
  - Performance for the operators (CH3.x): the Legendre step costs a sqrt and three
    divisions per (n, m), the normalisation a sqrt per entry, the ∂θ ladder 2–3 sqrts,
    and `row_one` 4 sqrts. Away from the poles the second recursion over sin θ is
    `current/s`. A table of √ factors per p in the scratch would remove most of this.

## 2026-10-30: CH1.2b Radial functions and scaling
- Done:
  - Branched from `main` beside CH1.2a's open PR #136, with the roadmap's split of CH1.2
    taken unchanged from that branch.
  - `nd_fmm_math::radial`, generic over `T`: `bessel_j` and `hankel_h` (complex z,
    Im z ≥ 0), `modified_i` and `modified_k` (real x), with optional derivatives and the
    scaling of §3.15 (e^{∓Im z} and s^{∓n}; k₀ = e^{−x}/x). They hold S3's algorithms:
    - Miller's ratios for j and i, with the start test in f64 for every `T`;
    - the upward recursion for h and k;
    - derivatives from the neighbouring orders.
  - Nothing allocates. Arguments that are not finite panic. The module documents the
    tested ranges and the tangential gradient's limit at z = 0.
  - `nd_fmm_math::Complex`, re-exported from `num-complex` 0.4 (a new workspace
    dependency, signed off 2026-10-30). `radial` divides by Smith's algorithm in a
    private helper: num-complex's textbook division took the Wronskian at |z| = 800 from
    5.6e-14 to 1.0e-13, beyond its bound.
  - The fixture `fmm-math/fixtures/helmholtz_radial.json` (1.3 MB), from
    `tools/fixtures/gen_helmholtz_radial.py`: S3's 571 sets and 13,174 rows, value for
    value, under the fmm-math header.
- Tests: `fmm-math/tests/radial/`, 18 tests. Results in f64:
  - against the fixture on every operator's range: worst 7.1e-14 (ĥ at n = 1482,
    |z| = 1501), at most 3.6e-15 on the leaf and low-frequency ranges;
  - the Wronskians: 5.6e-14; the cross product: 4.6e-14; the modified Wronskian: 5.8e-15;
  - Yukawa two ways (îₙ = i⁻ⁿ ĵₙ(ix), k̂ₙ = −iⁿ ĥₙ(ix)): bit for bit;
  - k̂₀ = 1/x exactly, and orders 0–2 against their closed forms;
  - finite on every range; z = 0 exact; the scale; prefixes; f32 at 4.4e-7; the panics.
- Checks:
  - Ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
    all-targets (14 crates re-checked after a touch); `cargo doc`.
  - The default tests (846) and workspace tests (925), all passing.
  - After the self-review fixes, re-ran `nd-fmm-math`'s tests, Clippy and doctests.
  - The generator's `--check` is byte for byte.
  - After the switch to `num-complex`, all of the above again (default 844 and workspace
    923 tests, all passing; +2 panic tests, −3 unit tests and 1 doctest of the own type),
    plus
    `cargo check --workspace --all-targets`, `cargo deny --locked check licenses` (ok)
    and `cargo tree -e normal` (no fftw-sys).
  - Not run: the MPI tests, because no code an MPI crate runs changed (a new module and
    a dependency edge only); `--ignored` in release, because the branch has no ignored
    test.
  - A self-review by a subagent: nothing blocking. Its points are taken: finiteness, z²
    underflow, and the documented start order and ranges.
  - No machine runs. Nothing pending.
- Decisions: D2 applied (A). Signed off 2026-10-30 (PR #137): "num-complex" as the
  complex type, in place of an own `Complex<T>`.
- Laplace: not touched. `lib.rs` declares a new module and a re-export; `nd-fmm-math` gains
  the dependency `num-complex`, which no Laplace code calls.
- Follow-ups:
  - CH1.2c, the coaxial coefficients, uses `radial` for its start column and its
    projection.
  - A real-arithmetic jₙ for real k (S3 §7: j costs 3–5 times h), measured when the leaf
    operators need it.

## 2026-10-30: CH1.2c Coaxial coefficients
- Done:
  - `nd_fmm_math::coaxial`, generic over `T`. `CoaxialTable` stores per order the triangle
    n′ ≥ n of degrees max(p_src, p_tgt) by min(p_src, p_tgt), in §3.15's stored
    normalisation. It reads the rest through the index symmetry, so one table serves both
    orientations (`reversed`: M2M's table is L2L's).
  - The generators:
    - S2's recursion (`regular`, `singular_regular`), with the start column from `radial`;
    - S2's three projections (`projection_singular_regular`, `projection_regular_outer`,
      and `projection_regular_inner`, extended to complex k), each filling the triangle
      from the half it computes.
  - §3.15's rule in `m2l`, `m2m` and `l2l` (`uses_recursion`: Re(k) a ≤ 40, a the M2L
    level's or the shift parent's circumradius), on S2's spheres; `l2l` reads `m2m`'s
    table reversed (signed off 2026-10-30, below).
  - Every regular radial column is evaluated to the square table's length (Miller's
    start), so a rectangular table is a block of the square one bit for bit.
  - The fixture `fmm-math/fixtures/helmholtz_coaxial.json` (1.2 MB) from
    `tools/fixtures/gen_helmholtz_coaxial.py`: S2's 120 sets and 21,809 entries value for
    value, with the fmm-math header and each set's geometry.
  - The example `coaxial_time`, and the report `fmm-validate/results/helmholtz-ch1-2c.md`.
- Tests: `fmm-math/tests/coaxial/`, 23 tests (5 of them ignored, run in release). In the
  FMM-weighted error of S2:
  - the rule's tables against the fixture: 6.8e-13 at worst (bound 1e-11); the recursion
    8.0e-15 where stable, and per entry 8.1e-14 at k a ≤ 3 (1.4e-13 on CI's Linux; bound
    1e-12);
  - the addition theorems pointwise, with the reflection and the symmetry: 5.9e-15;
  - real k: the recursion's (R|R) exactly real; Re (S|R) = (R|R) 1.4e-14; the outer
    projection's imaginary parts 4.1e-14, the inner one's measured 0;
  - Yukawa: i^{n−n′} (E|F) exactly real to κ a = 300;
  - rectangular blocks and `reversed` bit for bit for all five generators;
  - test 7, the threshold at k a = 40, every kind: 6.5e-14;
  - test 8, L2L at k a 40 to 300: the outer projection (M2M's orientation, read reversed)
    and the inner projection agree within 6.8e-13 for every kind of k; the recursion
    fails above the threshold for real and weakly damped k, as S2 found for M2M;
  - f32 at p = 8: 8.8e-7; the panics.
- Checks:
  - Ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
    all-targets; `cargo doc` (`nd-fmm-math` with `-D warnings`).
  - The default tests (887) and workspace tests (966), all passing.
  - The release tests with `--include-ignored` (85 s); the generator's `--check` is byte
    for byte.
  - After the self-review fixes (test code and docs only), re-ran `nd-fmm-math`'s Clippy
    and its tests, debug and release.
  - Not run: the MPI tests (no code an MPI crate runs changed); the licence and fftw-sys
    checks (no manifest changed).
  - A self-review by a subagent: nothing blocking. Its points are taken: the stricter
    reality check, three panic tests, and the plan's API and two report figures corrected.
  - No machine runs. Nothing pending. The timings ran at a load average of about 40.
- Decisions: signed off 2026-10-30, "1 and 2 as recommended" (report §5):
  1. one shift table, the outer projection in M2M's orientation, for M2M and L2L above the
     threshold; the inner projection is its check;
  2. §3.15's generator text corrected: the (S|R) projection is on the target box's
     (inner) sphere.
  Both are in CONVENTIONS §3.15's "The generator"; no stored quantity changes, so
  `CONVENTION_VERSION` stays 3.
  Signed off 2026-10-30 (PR #149, CI failure): test 1's per-entry bound at k a ≤ 3
  becomes 1e-12, "1e-12 flat". On CI's glibc an entry came to 1.4e-13 (8.1e-14 on the
  M3 Max), from the last bit of e^{iz} in the start column.
- Laplace: not touched. `lib.rs` declares the new module; `radial::div` became
  `pub(crate)`; CONVENTIONS changed in §3.15 only.
- Follow-ups:
  - CH1.3 applies the tables (RCR translations) in `nd-fmm-ref`. CH1.2 is complete with
    this task.
  - The recursion computes a, β and γ with a square root per entry; a table of them per
    order would speed up the build if CH2.2 finds the build cost matters.

## 2026-10-30: CH1.3a Direct-sum oracle
- Done:
  - CH1.3 split in the roadmap into CH1.3a (the direct-sum oracle), CH1.3b (leaf
    operators), CH1.3c (translations, direct and RCR) and CH1.3d (the diagonal form in
    extended precision). CH1.3d waits for CH0.6's signatures and is off the priority list;
    CH2.0 and CH3.0 now need CH1.3a–c.
  - `nd_fmm_ref::helmholtz::p2p`: `p2p` (generic over `T`) and `direct_sum` (f64,
    Neumaier per component) for e^{ikr}/r with complex k and complex charges, and
    `p2p_yukawa` and `direct_sum_yukawa` for e^{−κr}/r with real charges, potentials and
    gradients, exact coincidences skipped, no 1/(4π) (§3.15).
  - The term is written so that at k = iκ the complex path repeats the Yukawa
    operations (plus exact zeros), and at k = 0 the potential is Laplace's q/r.
  - Laplace's `p2p.rs`: `Neumaier` and `check_pair` became `pub(crate)`, nothing else.
- Tests: `fmm-ref/tests/helmholtz/`, 31 tests. Results in f64:
  - against CH1.4's FMM3D fixture (12 cases): relative ℓ2 at most 9.9e-16 (bound 1e-14),
    per point by term sum 4.6e-15 (bound 5e-14; the fixture's own error is up to 2.7e-15);
  - k → 0 against Laplace's direct sum at |k| = 1e-2, 1e-4, 1e-6, real, complex and
    imaginary k: within |k| r_max, and within (|k| r_max)²/2 once ik Σq is removed; the
    gradient within (|k| r_max)². For real k, §9's tolerances: Re φ̃ within (k r)²/2,
    Im φ̃/k − Σq within (k r)²/6, Im ∇φ̃ within (k r)³/3. Every error falls as the
    bound's power of k. At k = 0 the potential is bit for bit Laplace's;
  - Yukawa two ways at κ = 0.5, 25, 300: bit for bit, f64 and f32, real charges and
    complex charges as two real vectors, `p2p` and `direct_sum`;
  - `p2p` against `direct_sum` 1.1e-15; f32 2.7e-7; the gradient against central
    differences 1.0e-8; accumulation, coincidences, reciprocity, exact scaling, empty
    inputs, and the panics.
- Checks:
  - Ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
    all-targets; `cargo doc` (`nd-fmm-ref` with `-D warnings`).
  - The default tests (920) and workspace tests (999), all passing.
  - After the self-review fixes (tests and docs only), re-ran `nd-fmm-ref`'s Clippy, its
    tests (192), workspace Clippy, its strict docs and the format check.
  - Not run: the MPI tests (no code an MPI crate runs changed); the licence check (no
    manifest changed).
  - A self-review by a subagent: nothing blocking. Its points are taken: the second-order
    k → 0 check, the plan's bound aligned with the code, the real-charge Yukawa
    comparison, the "bit for bit" wording, two panic tests, and comments.
  - No machine runs. Nothing pending.
- Decisions: the split, the user, 2026-10-30, "Defer to CH1.3d" (the diagonal form after
  CH0.6). No question for sign-off.
- Laplace: not touched (`lib.rs` declares the new module; two private items of
  `p2p.rs` became `pub(crate)`; its tests pass).
- Follow-ups:
  - CH1.3b, the leaf operators in §3.15's stored form, uses this oracle for its series.
  - §9 item 1 asks for the oracle on the workloads of §9.1; CH1.3a tested the fixture and
    seeded cubes only, so H1.G should not count item 1 as shown yet.
  - `fmm-ref/Cargo.toml` and README still describe the crate as Laplace's only; worth a
    one-line change with the next manifest edit.

## 2026-10-30: CH1.3b Leaf operators
- Done:
  - `nd_fmm_ref::helmholtz::leaf`, generic over `T`: `p2m`, `p2l`, `l2p` and `m2p` for any
    k and complex charges (one complex expansion as two real planes, §3.15 "Storage"), and
    `p2m_yukawa`, `p2l_yukawa`, `l2p_yukawa` and `m2p_yukawa` for k = iκ and real charges
    (M̂ᵞ = i⁻ⁿ M̂, L̂ᵞ = iⁿ L̂). All accumulate, with gradients, in §3.15's stored form: the
    scaled radial functions of `radial`, the factor 4π ik (4πκ) on P2L and M2P, and only
    non-positive exponents.
  - Gradients are the radial derivative plus `harmonics_derivatives`' tangential terms, in
    the same frame (φ = 0 on the axis). L2P at the box centre takes the limit k/(3s) for
    n = 1.
  - `helmholtz::Level`: k with the level's a = √3 r, s = min(|k| a, 1) and β = Im(k) a;
    it rejects k outside §3.15's range.
  - `helmholtz::Workspace`: the operators' scratch.
  - For real k and real charges, `p2m` leaves the imaginary plane exactly zero, which is
    §3.15's real multipole.
- Tests: `fmm-ref/tests/helmholtz/leaf_*.rs` and `panics.rs`. All f64 unless stated.
  - Series, P2M → M2P and P2L → L2P against `direct_sum`, relative ℓ2:
    - every kind of k at |k| a = 0.5, 5 and 30, and Im(k) a = 100 (and about 70) at
      |k| a = 100;
    - sources and targets include the box corners, the centre and the ±z-axis;
    - at round-off at p₀ (worst 4.2e-14, bound 1e-12);
    - truncated (above 1e-7) at a lower pₜ, and falling in between.
  - Closed forms of degrees 0 and 1 (j₀, j₁, h₀, h₁) at s < 1 and s = 1, Im(k) a to 50:
    1.5e-14 (bound 1e-12). They pin the scale, the damping and 4π ik.
  - k → 0 against Laplace's `p2m`, `p2l`, `m2p` and `l2p` under the diagonal map of proposal
    §3.4, written out in `leaf_limit.rs`:
    - at |k| a = 10⁻², 10⁻⁴ and 10⁻⁶, every kind and Yukawa's real path;
    - bounds from the Taylor remainders: (|k| a)² for real parts, |k| a for the damping and
      for degree 0's imaginary part;
    - worst error over bound 0.54.
  - Yukawa two ways, κ a = 0.5 to 300: the coefficients and potentials agree bit for bit,
    the gradients within 3.3e-17 (bound 1e-13). For real charges the complex path's
    imaginary parts are exactly 0.
  - Strong damping, Im(k) a = 1000, where e^β overflows: every output finite.
  - f32 at p = 8: 5.4e-7.
  - 16 panic tests.
- Checks:
  - Ran `/verify` on the M3 Max:
    - format; Clippy default, examples and workspace all-targets; `cargo doc`
      (`nd-fmm-ref` with `-D warnings`);
    - the default tests (951 passed) and workspace tests (1030 passed).
  - After the self-review fixes (tests and docs only), re-ran the format check, workspace
    Clippy, `nd-fmm-ref`'s tests (222) and its strict docs.
  - Not run: the MPI tests (no code an MPI crate runs changed); the licence check (no
    manifest changed).
  - A self-review by a subagent: nothing blocking. Taken:
    - targets on the ±z-axis (the gradient frame's poles);
    - the real-charge check of the Yukawa path;
    - two doc corrections, and the plan aligned with the tests.
  - No machine runs. Nothing pending.
- Decisions: none. No question for sign-off.
- Laplace: not touched (new modules under `helmholtz`; `lib.rs` documentation only).
- Follow-ups:
  - CH1.3c, the translations (direct and RCR), takes `Level` and `Workspace` from here and
    tests against these operators' series.
  - Not in CH1.3b, for the drivers' tasks: the several-vector (`_many`) forms, and a
    real-only P2M for real k with real charges (here it is `p2m`'s real plane).
  - The strong-damping test checks finiteness only. Correct damping is shown by the closed
    forms (Im(k) a ≤ 50) and the series (≤ 100).

## 2026-10-30: CH1.3c Translations
- Done:
  - `nd_fmm_ref::helmholtz::rcr`: `m2m`, `l2l`, `m2l` and their `_yukawa` forms, generic
    over `T`, by proposal §5.2:
    - rotate the shift onto +z (a z-rotation per order, then the transposed y-block of
      `orthonormal::rotation::y_blocks`);
    - apply the coaxial table per order to the slots ±m, rectangular p_in → p_out;
    - rotate back.
    - In §3.15's stored form: M2M by s_cⁿ / s_P^{n′}, L2L by s_c^{n′} / s_Pⁿ, M2L by
      4π ik e^{2β − Im(k)|t|}; Yukawa through the real tables i^{n−n′} Ê, i^{n′−n} Ê and
      −i^{n+n′} Ê with 4πκ.
    - The tables come from `rcr::{m2m_table, l2l_table, m2l_table}` (§3.15's rule through
      `CoaxialTable::{m2m, l2l, m2l}`).
    - M2M and L2L assert a parent–child pair; M2L one level.
  - `nd_fmm_ref::helmholtz::direct`, the same six operators by projection:
    - the leaf evaluators' field at a Gauss–Legendre × uniform product quadrature on the
      output's sphere, projected on the real harmonics;
    - divided by 4π ik ĥₙ for a multipole; for a local, least squares from the value and
      ρ times the radial derivative (jₙ has zeros); Yukawa by k̂ₙ and îₙ;
    - general frames, with the spheres' nesting or separation asserted.
  - `Workspace` gains the y-blocks, their scratch and the translation buffers. `leaf`'s
    private helpers became `pub(super)`.
- Tests: `fmm-ref/tests/helmholtz/translation_*.rs` (24 tests and 13 panic tests; 2 ignored,
  run in release):
  - Addition theorems, RCR against `direct_sum`, every kind of k at |k| a 0.5, 5 and 30:
    - P2M → M2M → M2P, P2L → L2L → L2P and P2M → M2L → L2P;
    - ≤ 1e-12 at p₀ (worst 3.0e-14), above 1e-7 at pₜ, falling in between.
  - RCR against direct, in the FMM-weighted measure (per degree, weighted by the radial
    function where the output is evaluated):
    - children, the M2L offsets including the nearest and ±z, rectangular degrees,
      |k| a 0.5 and 5: worst 2.4e-14;
    - all 316 offsets: 8.4e-15;
    - |k| a = 30, Re(k) a = 50 above the projection threshold, and Im(k) a = 100:
      9.0e-14 relative to the larger of the output and the input;
    - relative to the output alone wherever Im(k) a ≤ 5.
  - Rectangular RCR is the square one's block, bit for bit. Real k keeps a real M2M and L2L
    exactly real. Finite at Im(k) a = 1000. f32 at 1.4e-6.
  - k → 0 against Laplace's `rotation` under §3.4's map, RCR and direct, |k| a 1e-2 to
    1e-6, every kind and Yukawa:
    - within |k| d (worst 0.29 of it);
    - real k: real parts within (|k| d)², imaginary parts within |k| d at degree 0 and
      (|k| d)² above (worst 0.22).
  - Yukawa two ways: 9.5e-15. RCR is bit for bit in all but two of 28 cases.
- Checks:
  - Ran `/verify` on the M3 Max: format; Clippy default, examples and workspace
    all-targets; `cargo doc`.
  - The default tests (976) and workspace tests (1055), all passing.
  - `cargo test -p nd-fmm-ref --release -- --ignored`: passed.
  - After the self-review fixes (tests and docs, one doc comment in `rcr`), re-ran:
    - the format check, workspace Clippy and the strict docs;
    - `nd-fmm-ref`'s tests (246);
    - every translation test in release, ignored included.
  - Not run: the MPI tests (no code an MPI crate runs changed); the licence check (no
    manifest changed).
  - A self-review by a subagent: nothing blocking. Taken:
    - the strict measure to Im(k) a ≤ 5;
    - §9's degree-0 rule for real k's imaginary parts;
    - the low-frequency overflow range documented;
    - doc corrections.
  - No machine runs. Nothing pending.
- Decisions: none. No question for sign-off.
- Laplace: not touched (new modules under `helmholtz`; `lib.rs` documentation only).
- Findings:
  - Under strong damping RCR's output loses relative accuracy while staying at round-off
    relative to its input:
    - M2L to 2.1e-10 at κ a = 30 and |t| = 3w, where e^{−Im(k)(|t| − 2a)} ≈ 1e-19;
    - L2L to a far child about 1e-11 at Im(k) a = 100.
    - The coaxial sum cancels; the recursion and projection tables agree with each other
      there. §5.5's pruning removes such contributions; relevant to CH2.3's θ_D.
  - RCR M2M and L2L overflow at low frequency once p log₁₀(1/s) ≳ 300 (documented).
- Follow-ups:
  - CH1.3d (the diagonal form) after CH0.6.
  - CH2.0 and CH3.0 can take the translations as their reference.
  - The drivers' tasks need the several-vector forms.

## 2026-10-31: CH2.0 Laplace's real host drivers made generic over the kernel
- Done:
  - The level calls' walking of the views, the threads and scratch sets and the batched
    M2L's blocks moved from `operator.rs` into `nd-fmm-exec/src/driver.rs`
    (crate-private), generic over `driver::HostKernel<T>`. Its methods take the level
    call's level.
  - Laplace's `Kernels` implements the trait by delegating to its unchanged methods,
    ignoring the level. `LaplaceOperator` calls the drivers. The plane-wave M2L stays in
    `operator`.
  - No public API, table, layout, manifest or convention change.
  - p reaches Laplace's inner loops at run time, before and after (report §1).
- Tests: no new test. Laplace is the trait's only implementor, and every existing test
  runs through it.
- Checks:
  - `/verify` on the M3 Max: format; Clippy default, examples, workspace all-targets and
    `nd-fmm-exec` with `cpu,metal`; the CUDA check; `cargo doc`, also `nd-fmm-exec`'s
    private items with `-D warnings`.
  - The default tests (976) and workspace tests (1055).
  - The MPI list at 2 and 4 ranks; `mpi_exec` at 8 ranks; the ignored release tests of
    `nd-fmm-exec`; `multi_rank` at 2, 4 and 8 ranks in release.
  - The full outputs of `main` and the branch identical byte for byte: 18 cases on each
    of the M3 Max, locust and Kathleen.
  - D9 on the three machines (report §3):
    - M3 Max: 12 of 12 within noise.
    - locust: 11 of 12; f32 p = 3 Batched failed by 0.2 ms and passed on a repeat.
    - Kathleen (jobs 614707, 614723): 10 of 12. f64 p = 6 Dense and Classes are +5.4 %
      from code placement: equal to `main` when both are built with
      `-C llvm-args=-align-loops=64`.
  - A self-review by a subagent: nothing blocking. Doc corrections taken (the level's
    meaning, "compile time", the CLAUDE.md rule, the plan as built).
- Decisions: the user, 2026-10-30, "Proceed as planned" (Laplace the only kernel) and the
  12 host benchmarks. Signed off 2026-10-31, the user: "Accept both as recommended":
  the two Kathleen rows accepted as code placement, and D9 amended so that a failing row
  may be repeated once, both results reported (report §4).
- Laplace: unchanged bit for bit (the byte comparison on three machines and the test
  suite). Benchmarks: within noise except the two Kathleen rows above (accepted).
- Follow-ups:
  - CH2.1 (Yukawa) or CH3.5/CH3.1 (complex) is the trait's second user, with per-level
    p_l and tables through the level.
  - Code placement moves Laplace's times by up to ±5–13 % between builds (Kathleen f64
    Dense/Classes, M3 Max f32 Classes/Rotation). Later D9 runs may meet it again. An
    aligned build (`-align-loops=64`) is the tool to tell placement from work.

## 2026-10-31: CH2.4a The vectorised exponential
- Done:
  - CH2.4 split into CH2.4a (this) and CH2.4b (Yukawa P2P on a shared skeleton), the user,
    "Split a/b".
  - `nd_fmm_simd::arch::exp::exp`, e^x for x ≤ 0 on the vector layer:
    - clamp to x_min = (e_min + 1) ln 2;
    - n by the constant 1.5 · 2^m and a Cody–Waite reduction;
    - Taylor of degree 13 (f64) or 7 (f32) by Horner;
    - n added to the exponent bits (`Simd::scale_pow2`, no integer conversion).
  - The layer gains `max` and `scale_pow2` on every ISA. `exp::exp_slice` is public, for
    tests and reports, like `rsqrt_slice`.
  - Every step is an IEEE-rounded operation, so every ISA equals the scalar ISA bit for bit.
  - Design `docs/design/simd-p2p.md` §4.9; report `fmm-validate/results/helmholtz-ch2-4a.md`.
- Tests: `fmm-simd/tests/exp/` (11, plus 2 ignored release tests) and the layer test
  `layer_exponent_operations_match_scalar`.
  - Contract 4 u_T on [x_min, 0]: f32 exhaustive, 1.324 u; f64 on 10⁷ samples against
    double-double, 1.228 u.
    - Identical on the M3 Max (scalar, NEON) and Kathleen (scalar, AVX2, AVX-512).
    - Derived bounds 3.3 u₃₂ and 3.2 u₆₄.
  - The reduction's boundaries; e^{±0} = 1 exactly; below x_min the value at x_min.
  - Bit identity across ISAs, lane and length independence, panics.
  - The oracle against 90-digit values.
- Checks:
  - Ran `/verify` on the M3 Max, all passed:
    - format; Clippy default, examples and workspace all-targets;
    - `cargo test` (989) and `cargo test --workspace` (1,068); `cargo doc`;
    - `nd-fmm-simd`'s Clippy native and x86_64; its tests and ignored release tests;
    - the MPI list at 2 and 4 ranks.
  - Kathleen jobs 615552 and 615691 (`check.sbatch simd`): all passed.
  - Inlining check: no call in any exponential loop; NEON 27/39, AVX2 23/33, AVX-512 23/29
    instructions (f32/f64).
  - Not run:
    - the licence check (no manifest changed);
    - Laplace's `tools/bench/run.sh` (the plan leaves it to CH2.4b);
    - locust.
  - A self-review by a subagent: nothing blocking. Taken:
    - the f64 oracle compares before its 2ⁿ scaling (the first figure, 1.519 u₆₄, was half
      the oracle's own error near x_min);
    - two bounds and one wording corrected; the plan updated.
  - Nothing pending.
- Decisions: the split (above). Questions for sign-off, report §5:
  1. NEON f32 `sums`, 4–6 % slower from a codegen-unit shift: accept, fix in source, or
     one codegen unit. Signed off 2026-10-31: accepted.
  2. The plane-wave steps on locust in CH2.4b's D9 runs. Signed off 2026-10-31: yes.
- Laplace: bits unchanged (only additions; the test suite and the MPI list).
  - Code: x86_64 all 124 existing functions identical to `main`'s; aarch64 49 of 54.
    - The five that differ are the plane-wave `sums` step, from rustc's codegen-unit
      partition; with one codegen unit, 69 of 69 are identical.
  - D9 on the M3 Max (`plane_wave_parts`, A B B A order): f32 p = 8 `sums` failed once
    (+6.1 %) and passed on the repeat (+3.6 %); every other row passes, f64 included.
- Follow-ups:
  - CH2.4b: the Yukawa P2P on `exp`, with Laplace's D9 on the three machines (the
    plane-wave steps on locust included, question 2).
  - Adding code to `nd-fmm-simd` can move inlining in untouched kernels (fmm-simd/CLAUDE.md
    now says so). Compare entry points with `main`'s at the default settings and with one
    codegen unit.

## 2026-10-31: CH2.4b Yukawa P2P on a shared skeleton
- Done:
  - `arch::p2p::p2p_skeleton`: the P2P loop (blocks, tails, loads, broadcast, d, r², stores)
    generic over a `PairKernel`, a compile-time type parameter. Laplace's `Laplace` is
    zero-sized and is the old source loop operation for operation; `p2p_body` is the
    skeleton with it. The several-vector and dipole bodies are untouched.
  - `P2pKernel::evaluate_yukawa`: e^{−κr}/r and its gradient for real charges, every ISA,
    f32 and f64. The pair kernel `Yukawa`: ρ, r = r² ρ, e = exp((−κ) r) (CH2.4a), (q e) ρ
    and ((q e) ρ · fma(κ, r, 1)) (ρ ρ) dₖ. The scalar ISA is `p2p_yukawa`'s loop. Panics
    unless κ is finite and κ ≥ 0.
  - `YUKAWA_K = 1` on every ISA (measured on NEON and AVX-512; K = 2 spills).
  - Design `docs/design/simd-p2p.md` §4.10; report `fmm-validate/results/helmholtz-ch2-4b.md`;
    the cost example `spikes/p2p-simd/examples/yukawa.rs`.
- Tests: `fmm-simd/tests/p2p/yukawa.rs` (17, plus 4 ignored release tests) and
  `p2p_yukawa_body_is_independent_of_k` in `src/arch/tests.rs`.
  - Terms within the derived contract (20 + 12 κr) u_T and (52 + 12 κr) u_T over 10⁶ pairs per
    precision, κr to 64 (f32) and 512 (f64): at most 10.2 u at small κr, about 3 u per unit
    of κr beyond. The same on NEON (M3 Max), AVX2 and AVX-512 (Kathleen).
  - Sums against `direct_sum_yukawa` within C3S.4's rule (7.9e-7, 2.9e-15).
  - κ = 0 equals `evaluate` bit for bit on every vector ISA; the scalar ISA equals
    `p2p_yukawa` bit for bit; invariance and K independence bit for bit; panics.
- Checks:
  - `/verify` on the M3 Max, all passed: format; Clippy default, examples, workspace and
    `nd-fmm-simd` (native and x86_64); `cargo test` (1,006) and `--workspace` (1,086);
    `cargo doc`; `nd-fmm-simd`'s tests and ignored release tests; the MPI list at 2 and 4
    ranks; the spike's test.
  - Kathleen job 616123 (`check.sbatch simd`, node-c11a-039): passed.
  - Inlining check: no call in any Yukawa source loop (NEON 40–78, AVX2 49–87, AVX-512 37–58
    instructions).
  - Not run: the licence check (no default member's manifest changed).
- Constraint 1, all four conditions:
  - no new dispatch (type parameter, const generics); no layout or table change;
  - every existing function of `nd-fmm-simd` identical to `main`'s on aarch64 and x86_64,
    macOS and Linux, default and one codegen unit;
  - the 18 full FMM outputs byte for byte `main`'s on the M3 Max, locust and Kathleen (job
    616122, node-c11a-034);
  - D9 on the three machines (`p2p_kernels` gathered W1, `plane_wave_parts` at p = 8 and 12,
    `run.sh` Auto at p = 3 and 8): 120 of 120 rows within noise, no repeat;
  - C3S.4: 95–101 % of the Phase 3S prototype on the M3 Max.
- Cost per pair against Laplace: 2.3–4.5× over every ISA, precision and output (M3 Max f64
  4.5× at most); 1.2–4.1× faster than green-kernels' modified Helmholtz on the same machine
  and vector width (its convention checked: 1/(4π), ∂/∂x).
- Decisions: the user, 2026-10-31, "Proceed as planned" and the D9 set. Questions for
  sign-off, report §7: 1. the contract as derived; 2. κ ≥ 0 required. Both signed off
  2026-10-31, the user: "Accept both sign-offs as recommended".
- Laplace: unchanged in code (instruction for instruction), bits and benchmarks.
- Follow-ups:
  - CH2.1 (Yukawa operator) needs a several-vector Yukawa P2P if it evaluates several
    charge vectors (`evaluate_many` has no Yukawa form yet).
  - The dipole and several-vector bodies could move onto the skeleton later (rule of two).
  - CH3.4 (Helmholtz P2P) builds on the same skeleton with a sincos.


## 2026-10-31: CH3.0a nd-octree's maximum leaf width
- Done:
  - D1 signed off: kD ≤ 500 at 10⁻³ to 10⁻⁸, as recommended.
  - CH3.0 split into CH3.0a (this), CH3.0b (the builder's leaf rule on W1–W4) and CH3.0c
    (complex densities); the user, "Split a/b/c, proceed".
  - `OctreeOptions::with_min_leaf_level` (the name chosen by the user): every box that
    holds a key and lies above the level is refined, capped at `max_level`; default 0,
    no rule. It is the key-space form of proposal §3.7's maximum leaf width.
  - The rule acts in `create_local_tree` only, below the blocks, never in the coarse
    tree's weight rounds. So the blocks stay nodes of the one-rank tree, and the leaves
    are the one-rank leaves on every rank count (O1).
  - `octree/CLAUDE.md` updated. No other crate, manifest, convention or `nd-fmm-plan`
    change. Record: the plan `docs/helmholtz/tasks/CH3.0a.md` (no separate report; no
    measurement).
- Tests:
  - `test_create_local_tree_min_leaf_level_against_brute_force`: against a plain
    recursion, key for key, for max_keys 1, 8 and 64, max_level 3, 5 and 16, levels 0–6,
    from the root and from the octants. Level 0 equals the call without the option, and a
    level above `max_level` acts as `max_level`.
  - `test_coarse_tree_by_weight_keeps_the_one_rank_leaves`: three cases with the level
    set.
  - `test_mpi_weighted_partition`: the cube and the Plummer sphere, 2,000 points each, at
    levels 0, 4 and 5. It checks every invariant, the one-rank leaves, the brute force
    after balance and the leaf levels. Passes at 1, 2, 3, 4 and 8 ranks.
- Checks:
  - `/verify` on the M6:
    - format; Clippy default, examples and workspace all-targets; `nd-octree`'s Clippy
      with `strict`, its `cargo doc` with `-D warnings` and its examples;
    - `cargo test` (1,010) and `--workspace` (1,090); `cargo doc`;
    - the MPI list at 2 and 4 ranks.
  - A self-review by a subagent: nothing blocking; its doc and count fixes taken.
  - Not run: the licence check (no manifest changed); locust and Kathleen (no machine
    run in the plan).
- Decisions: D1 signed off; the split; constraint 1's fourth condition by the byte
  comparison without D9 timing runs (the user, "Bytes only, no D9").
- Laplace: unchanged bit for bit. The default adds no condition (the tree tests at level
  0), and the FMM outputs of `main` and the branch are identical byte for byte on the M6:
  36 of 36 (cube and Plummer, N = 10⁵, f64, p = 3, 6 and 8, all six strategies).
- Follow-ups:
  - CH3.0b passes the shallowest level whose boxes are at most c_leaf λ wide (D11:
    c_leaf = 1), from k and the domain; no rule for Yukawa.
  - The census spike's serial prototype (`spikes/census/src/tree.rs`) can move onto the
    option.

## 2026-10-31: CH3.0b The Helmholtz builder's leaf rule
- Done:
  - `nd_fmm_exec::helmholtz` (new module), proposal §6.1's home of the Helmholtz builder:
    - `Wavenumber::new(re, im)` classifies k as §3.15 states (`WavenumberKind`: Yukawa for
      Re k = 0, real for Im k = 0, complex otherwise). It rejects non-finite parts, k = 0
      (Laplace), Re k < 0 and Im k < 0 (`WavenumberError`); signed zeros count as zero.
    - `wavelength()`, 2π/Re k (`None` for Yukawa); `C_LEAF = 1` (D11).
    - `min_leaf_level(k, domain, c_leaf)`: the shallowest level whose boxes are at most
      c_leaf λ wide, capped at 16, 0 for Yukawa; the value for CH3.0a's
      `OctreeOptions::with_min_leaf_level`. It reads k, the global domain and c_leaf only
      (constraint 9).
    - `FmmBuilder` is untouched; CH3.1's Helmholtz FMM passes the level to its tree.
  - `nd_fmm_validate::helmholtz_workloads`: W1–W4 of proposal §9.1 in the root cube, ported
    point for point from the census spike (points only; charges come with CH3.0c/CH3.1).
  - `nd-octree` is now a direct dependency of `nd-fmm-validate` (the user: "Add
    nd-octree"); one new edge in `Cargo.lock`, no new package.
  - `fmm-exec/CLAUDE.md` and `fmm-validate/CLAUDE.md` updated. Record:
    `docs/helmholtz/tasks/CH3.0b.md` (the sweep's table; no separate report).
- Tests:
  - `fmm-exec/tests/helmholtz/` (17): the classification and rejections. The level
    against the census spike's search (1,675 pairs of kD and c_leaf, boundaries
    included) and against the closed form ⌈log₂(k w/(2π c))⌉ (1,459 pairs away from a
    boundary). Also minimality, known levels, complex k at Re k's level, Yukawa 0, the
    cap, the domain's scaling and the panics.
  - `nd-fmm-validate`: 5 unit tests of the workloads. `tests/helmholtz_leaves.rs` (10 s in
    debug) runs `Fmm`'s tree with the rule on W1–W4 at kD ≤ 300: no leaf that holds a
    point is wider than c_leaf λ, and Yukawa's tree is `Fmm`'s key for key.
  - `spikes/census/tests/ported_workloads.rs`: the workloads equal the spike's bit for bit.
  - The ignored `tests/helmholtz_leaves_sweep.rs` covers D1's range in release (6 s on the
    M6): W1 and W2 to kD = 500 (3.2 million points), W3 resolved to 100 (4.0 million)
    and at 10⁶ for 200 and 500, and W4 at 10⁵. No occupied leaf is wider than 0.995 λ;
    every tree's leaves at kD = 500 lie on level 7 (0.62 λ), as the census found.
- Checks:
  - `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
    `cargo test` (1,034) and `--workspace` (1,115); `cargo doc`; the licence check (ok);
    `cargo tree -e normal` (no fftw-sys); the MPI list at 2 and 4 ranks (22 of 22 passed);
    `tests/helmholtz_leaves.rs` at 2 ranks by hand (passed).
  - A self-review by a subagent found nothing blocking. Its points were taken: occupied
    leaves stated and the widest empty leaf recorded, a closed form as an independent
    path, kD = 500 left to the sweep in debug, and the wording.
  - Not run: locust and Kathleen (no machine run in the plan).
- Decisions: the dependency (above). Questions for sign-off:
  1. λ = 2π/Re k for complex k.
  2. "No leaf wider than c_leaf λ" read as no leaf that holds a point. Empty leaves reach
     9.9 λ on W1 and 19.9 λ on the clusters at kD = 500.
  Both signed off 2026-10-31, the user: "1 and 2 as recommended".
- Laplace: not touched. There is no change to `fmm.rs`, the drivers, the tables,
  `nd-octree`, `nd-fmm-plan` or `nd-fmm-simd`; the Laplace tests pass unchanged.
- Follow-ups:
  - CH3.0c: complex densities.
  - CH3.1 passes `min_leaf_level` to its tree, and keeps empty leaves (possibly many λ
    wide) out of every expansion path.
