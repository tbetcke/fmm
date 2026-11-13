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

## 2026-11-01: CH3.0c Complex densities
- Done:
  - `nd_fmm_exec::helmholtz::densities` (new; `helmholtz.rs` moved to `helmholtz/mod.rs`
    unchanged but for the module line and re-exports):
    - `Densities` (k's kind, `ChargeKind`, m): the sizes of §3.15's table
      (`multipole_size`, `local_size`, `multipole_planes`, `local_planes`), the chunk sizes
      and indices.
    - The leaf chunks, which §3.15 leaves to CH3.0 (not versioned), fixed in the module
      docs: a complex vector is two adjacent real columns, Re then Im, of §3.13's
      several-vector layout. Every chunk is then a §3.13 chunk of c m source and o_m
      output vectors, §3.15's output order is the column order, and Yukawa's complex
      vectors are its real path's 2m vectors value for value.
    - `Charges` in (`check_charges`, `write_leaf_charges`); `Output` out in split planes
      (`potential_of`, `gradient_of` give `Split`), `write_leaf_output` with Laplace's
      scales 4π r and 4π r².
    - `p2m` and `l2p` in the leaf's own frame at k̂ = k r, one reference call per point
      (`LeafScratch`, no allocation): one complex expansion per complex vector.
  - `fmm-exec/CLAUDE.md` updated. Record: `docs/helmholtz/tasks/CH3.0c.md` (no separate
    report; no measurement).
- Tests: `fmm-exec/tests/helmholtz/densities.rs` (9 tests, no MPI, 0.3 s in debug):
  - sizes and indices against §3.15's table and §3.13's closed forms, for every path and
    m = 1, 3;
  - charges in, with each error;
  - P2M and L2P bit for bit against `nd_fmm_ref::helmholtz::leaf` on every path, with and
    without gradients, vector v equal to the one-vector operator, and accumulation;
  - Yukawa's complex vectors bit for bit its 2m real vectors;
  - end to end against the direct sum at k̂ = 0.875, 5 and 10 (|k| D to 160), p = 20 to
    40: worst relative L2 5.7e-13 (P2M → M2P) and 1.1e-11 (P2L → L2P → output pass).
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo test` (1,043) and `--workspace` (1,125); `cargo doc`; the MPI list
  at 2 and 4 ranks (22 of 22 passed). A self-review by a subagent: nothing blocking; its should-fix points
  taken (no-gradient path, k̂ = 10, accumulation). Not run: the licence check (no
  manifest changed); locust and Kathleen (no machine run in the plan).
- Decisions: none taken. Questions for sign-off: 1. CH3.0c's "in" and "out" read as the
  types, chunks, leaf P2M and L2P and the output pass (the FMM's are CH3.1's); 2. the
  leaf chunks fixed in the module docs, not in CONVENTIONS. Both signed off 2026-11-01,
  the user: "yes to both".
- Laplace: not touched. There is no change to `fmm.rs`, `operator.rs`, `driver.rs`, the
  tables, `nd-octree`, `nd-fmm-plan` or `nd-fmm-simd`; the Laplace tests pass unchanged.
- Follow-ups:
  - CH3.1: the operator with P2L, M2P and P2P on the chunks; it can reuse Laplace's charge
    load and output pass with m → `columns` and `output_columns`; its translations form a
    level's s and β from the same k̂ as the leaf operators.

## 2026-11-01: CH3.5 The shared drivers extended to complex expansions
- Done:
  - Closed as a record, without a code change (the user: "Close as a record"). The
    record is `docs/helmholtz/tasks/CH3.5.md`.
  - The finding: the target-by-target drivers of CH2.0 take each box block's size from
    the plan's stores (`multipole_size(level)`, `local_size(level)`, which may differ) and
    pass whole blocks to the kernel's bodies. Two planes per complex expansion, real-k
    multipoles of (p + 1)² m beside locals of 2 (p + 1)² m, and the real shifts of M2M
    and L2L on both planes therefore all live in the kernel. CH3.1's RCR operator runs on
    the unchanged drivers.
  - Only the batched M2L block (`driver::m2l_block`) assumes Laplace's layout: square
    tables, and m expansions of n values per box, multipole and local alike. The
    two-plane case there (`PLANES`, rectangular real-to-complex tables, the per-target
    epilogue of proposal §6.2) moves to CH3.2, its first user, with constraint 1 as in
    CH2.0. A kernel with another layout returns `None` from `batched_m2l`.
  - The docs of `HostKernel` and `Columns` in `fmm-exec/src/driver.rs`, plus
    `fmm-exec/CLAUDE.md`, now say this. No code line changed.
- Tests: none new (no code). Evidence: `nd-fmm-plan`'s tests with differing,
  level-dependent multipole and local sizes (`src/evaluator_tests.rs`,
  `src/operator_tests.rs`, `tests/mpi_regressions.rs`), which prove the stores and the
  exchanges; a reading of `driver.rs`. `Kernels` is its only kernel, so `driver.rs`
  stays untested with unequal sizes until CH3.1, whose kernel is the end-to-end proof.
- Checks:
  - `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
    `cargo doc`, plus `nd-fmm-exec`'s private docs with `-D warnings` (the new links);
    `cargo test` (1,044) and `--workspace` (1,125).
  - A self-review by a subagent: nothing blocking. Its wording points were taken: Dense's
    epilogue, the leaf frames' units, the dummy `Columns`, and the limits of the
    evidence.
  - Not run: the MPI list locally (no compiled change; CI runs it at 2 and 4 ranks);
    machine runs and D9 (no compiled change to Laplace); the licence check (no manifest
    changed).
- Decisions: the scope, by the user on 2026-11-01 ("Close as a record"); CH3.2's *Done
  when* gains the batched two-plane case. Questions for sign-off: 1. the proposal's §6.2
  and §10 text left unchanged, the roadmap and the record carrying the move; 2. Dense's
  epilogue decided in CH3.2. Both signed off 2026-11-01, the user: "1 and 2 as
  recommended".
- Laplace: not touched; `driver.rs` changed in doc comments only.
- Follow-ups:
  - CH3.1: the Helmholtz kernel implements `HostKernel` with `batched_m2l` `None` (and an
    empty `Columns`). It passes `helmholtz::leaf::{p2l, m2p}` a `Level` in the frame's
    units: M2P's W boxes lie one level below the call's level and P2L's X leaves one
    above it.
  - CH3.2: Dense's epilogue per pair or per target (a finish hook in `m2l_target`).
  - CH3.2: the batched block's two-plane case.

## 2026-11-01: CH3.1a The Helmholtz operator
- Done:
  - CH3.1 split into a/b/c (the user: "a/b/c"). The user also decided ("Yes, both as
    recommended"): the degree is a rule behind a setting, and CH3.1b extracts the shared
    point pipeline of `fmm.rs`, with "minimal timing runs to check the impact" for D9.
  - `nd_fmm_tables::helmholtz` (new): `HelmholtzTables`, the per-level RCR tables. The
    coaxial tables are `rcr`'s and `CoaxialTable::m2l`'s with the stored form's factors
    folded in (f64, rounded to T). The tables and y-blocks are keyed by exact integer
    classes (15 distances, 49 polar angles), their values formed from the integers with
    correctly rounded square roots: the same classes on every machine. (A first version
    keyed them by the bits of `rcr`'s floating-point angles and distances, for
    bit-identity with `rcr`; the user found it brittle and chose exact classes in this PR.)
    The octant blocks are always stored, the M2L blocks within a byte budget (default
    512 MiB). The table-driven M2M, L2L and M2L take m vectors of one or two planes, with
    real shifts for real k.
  - `nd_fmm_exec::helmholtz::degree` (new): `Degrees` (a tolerance, or named per level) and
    `degree`, S7's closed form at Re(k) a. Levels 0 and 1 get degree 0, since nothing reads
    their expansions.
  - `nd_fmm_exec::helmholtz::operator` (new): `HelmholtzOperator` for real and complex k,
    real and complex charges and m vectors. It implements `FmmSizes`, `FmmOperator` (on the
    unchanged shared drivers, no batched M2L) and `PairOperator`. P2M and L2P are
    `Densities`'; P2L and M2P are `nd_fmm_ref::helmholtz::leaf`'s in the points' units;
    P2P is the reference kernel. Yukawa is refused (CH2.1's path).
  - The units, found by the chain tests: a box of level l holds r_l φ̃ in its own units. So
    the L2L tables carry ½ and P2L scales its charges by r̂(t|s); both are exact. The module
    docs and `fmm-exec/CLAUDE.md` and `fmm-tables/CLAUDE.md` record this.
- Tests:
  - `fmm-tables/tests/helmholtz/` (10 tests, 0.6 s in debug):
    - Against `rcr` in the FMM-weighted measure, within 2e-13 times max(1, |k̂| a)
      (1.7e-14 measured): M2L on all 316 offsets on three levels and five k; M2M and L2L
      on the 8 octants with rectangular degrees (L2L against ½ `rcr`); both above the
      projection threshold too.
    - Bit for bit: stored against per-call blocks, vectors one at a time. Also f32, the
      class counts (15, 49) and the refusals, among them non-finite power factors.
  - `fmm-exec/tests/helmholtz/{degree, operator}.rs` (9 new tests; the binary runs in
    5.8 s in debug):
    - The closed form on S7's grid (≥ p*), and `resolve` against degrees computed by hand.
    - The chains far, v4, x, w and near against the direct sum: below ε at 10⁻⁶ and
      10⁻¹⁰, for kD = 1, 20, 100 and 20 + 2i, with real and complex charges.
    - At p = 5–8, where truncation dominates (10⁻⁵ to 10⁻¹), the same chains of
      `nd-fmm-ref` in absolute units, within 2e-13 (1.2e-14 measured).
    - P2P bit for bit `p2p::p2p`; vectors bit for bit; real charges bit for bit complex
      ones with Im q = 0; sizes and errors.
  - The level calls through the drivers need MPI, so they are tested in CH3.1b.
- Checks:
  - `/verify` on the M6, all passed: format; Clippy default, examples and workspace
    all-targets; `cargo doc` (and `-D warnings` on the two crates, private items
    included); `cargo test` (1,065) and `--workspace` (1,146); the MPI list at 2 and 4
    ranks (22 of 22).
  - A self-review by a subagent found no correctness bug. Its points were taken: the plan
    updated to what was built, P2P against `p2p`, non-finite power factors refused, a
    message for too many points in a leaf, the budget docs, and a hand-computed `resolve`.
  - Not run: the licence check (no manifest changed); locust and Kathleen (no machine run
    in the plan).
- Decisions: the split and the two design choices above. Questions for sign-off (the PR):
  1. The y-blocks built in T, as `rcr` builds them, so that stored and per-call blocks
     agree. In f32 that departs from constraint 5.
  2. The level units (a box holds r_l φ̃) recorded in the module docs now and proposed for
     CONVENTIONS §3.15 at its next revision.
  3. Complex k's degree at Re(k) a until CH5.2.
  All three signed off 2026-11-01, the user: "1–3 as recommended". The keys: exact integer
  classes, the user, 2026-11-01 ("switch to A in this PR").
- Laplace: not touched. There is no change to `LaplaceOperator`, `driver.rs`, Laplace's
  tables, `nd-fmm-plan`, `nd-octree` or `nd-fmm-simd`; the Laplace tests and the MPI list
  pass unchanged.
- Follow-ups:
  - CH3.1b: the FMM, with the level calls against `PerPair`.
  - CH3.2: Dense, Classes and Batched.
  - CH3.4: the vectorised P2P in place of the reference.

## 2026-11-01: CH3.1b The Helmholtz FMM
- Done:
  - The point pipeline (`fmm-exec/src/pipeline.rs`, crate-private): steps 3, 4's keys and
    octree, 5, 6 and 8 of `FmmBuilder::build` (`domain`, `octree`, `Points::route`,
    `Points::load_coordinates`) and the contiguous block of helpers from `agree` to
    `unpack_output_many` (the charge load, the output pass, the moves; the dipole and device
    helpers of the block with them), moved verbatim from `fmm.rs`. `Fmm` holds a `Points`;
    `unpack_output_many` takes the output's two buffers. No public item of `fmm` changed.
  - `nd_fmm_exec::helmholtz::fmm`: `HelmholtzFmmBuilder` and `HelmholtzFmm` on any number
    of ranks, the pipeline with the leaf rule and CH3.1a's operator on `Evaluator`. The c m
    columns (Re q, Im q per complex vector) are packed point-major, so the charge load,
    the output pass and the unpacking run unchanged with the columns as the vectors
    (CH3.0c's chunks). `HelmholtzFmmError`; doc-hidden `pair_calls` and
    `reference_output` for the tests. No overlap, device, near rule, work cut or cache.
  - `tests/helmholtz_fmm.rs` in CI's MPI list (`.github/scripts/run-mpi-tests.sh`); root
    and `fmm-exec/CLAUDE.md` updated. Record: `docs/helmholtz/tasks/CH3.1b.md`.
- Tests: `tests/helmholtz_fmm.rs` (its own MPI test): six scenarios (cube and clusters,
  kD = 1, 20, 20 + 2i, 100; ε = 10⁻³ and 10⁻⁶ by S7's closed form, p = 8 named; real and
  complex charges; one scenario with rank 0 empty), each against the direct sum (2.0e-8 to
  2.2e-4 within its bound; 5.3e-13 and 2.0e-10 at ε = 10⁻⁶), the reference output pass, 4
  threads (2 and 4 in release), `PerPair` and the one-rank FMM, all bit for bit; two
  vectors, real against complex charges, the errors. 36 s at one rank in debug.
- Checks: `/verify` on the M6: format; Clippy default, examples, workspace all-targets and
  `nd-fmm-exec` with `cpu,metal`; `cargo check -p nd-fmm-exec --features cuda`; `cargo
  doc` (and `-D warnings` on `nd-fmm-exec` with private items); `cargo test` (1,067) and
  `--workspace` (1,148); the MPI list at 2 and 4 ranks (24 of 24); `mpi_exec`,
  `invariance` and `helmholtz_fmm` at 8 ranks; `helmholtz_fmm` in release at 1 and 4.
  A self-review by a subagent: no correctness bug, the extraction a pure move; its points
  taken (the one-rank reference built on every rank, a complex-k tolerance scenario, an
  empty rank, real-charge vectors, two more errors, the CLAUDE.md wording). Not run: the
  ignored `multi_rank.rs` (the user, "Skip it"); locust and Kathleen (the user, "M6, 3
  rows"); the licence check (no manifest changed).
- Decisions: the user, 2026-11-01, each as recommended: the extraction as planned, the
  minimal D9 runs on the M6 (three rows), `multi_rank` not run. No question for sign-off.
- Laplace: `fmm.rs` changed (constraint 1). Outputs bit for bit: every existing test, and
  36 of 36 `nd-fmm-bench --write-output` files of `main` (540f134) and the branch
  identical byte for byte on the M6. D9 on the M6: f32 p = 3 +0.7 %, f64 p = 6 +1.3 %,
  f64 p = 8 +0.4 % against spreads of 5.7 %, 6.7 % and 9.0 % of `main`'s medians, all
  pass; a thermal drift of about 9 % through the session widened the spreads, and the
  adjacent A/B pairs, which cancel it, differ by medians of −0.1 %, −0.1 % and +0.4 %.
- Follow-ups:
  - CH3.1c: the acceptance runs on W1, W3, W4 (kD ≤ 100) and W1 at kD = 500.
  - CH3.3: S7's closed form overshoots on low-frequency levels (p = 16 at ε = 10⁻³ gives
    1e-7 to 1e-8; p = 34 at 10⁻⁶ gives 5e-13).
  - Later: overlap, the near rule and the work cut for the Helmholtz FMM (their cost model
    is Laplace's).

## 2026-11-01: CH3.1c The acceptance runs
- Done:
  - `nd_fmm_validate::helmholtz_accuracy` (new): the Helmholtz FMM against
    `nd_fmm_ref::helmholtz::p2p::direct_sum` (times 1/(4π), which the reference leaves
    out) at 1,000 sampled targets on the workloads of `helmholtz_workloads`, §9.1's charges
    and relative ℓ2 norms of φ and ∇φ, each rank a contiguous block, the oracle sharded
    over ranks and threads, and a rank-invariant checksum of the whole output for
    determinism. The `helmholtz_accuracy` example (not registered for the weekly job: it
    runs for minutes) prints a Markdown row per case and thread count.
  - The report `fmm-validate/results/helmholtz-ch3-1c.md`; record
    `docs/helmholtz/tasks/CH3.1c.md`.
  - CH3.1 is done with this task: once it is merged, the tasks that need CH3.1 (CH3.2,
    CH3.3, CH4.1, CH6.1, the device tasks) no longer wait for it.
- Tests: `fmm-validate/tests/helmholtz_accuracy.rs` (one MPI test, 26 s at one rank in
  debug): the checksum independent of the partition and sensitive to one value, the blocks
  a partition, the sharded oracle the serial one bit for bit, and W1 (kD = 10, two
  vectors) and the W4 clusters (complex k, real charges) within ε with one checksum at 1
  and 2 threads and on one rank; passed at 1, 2 and 3 ranks.
- Runs (the acceptance; M6, release, 1 × 6 unless stated): W1 at kD = 10, 30, 100 and
  500, W3 at ten points per wavelength at kD = 10, 30 and 100 (4.0 million points), W4's
  four distributions at N = 10⁵ and kD = 1, 10, 100, each at ε = 10⁻³, 10⁻⁶ and 10⁻⁸
  (S7's degrees), plus real charges, eight vectors and kD = 100 + 10i on W1: 61 cases,
  every error of φ and ∇φ below ε, the largest 3.7e-3 ε. Not run: W3 at kD = 100 at 10⁻⁸
  (about 21 GB). Deterministic: equal checksums at 1 and 6 threads on 13 cases, and on 1,
  2 and 3 ranks on 3 cases, W1 at kD = 500 on 1 and 3. W1 at kD = 500 evaluates in 89, 508
  and 1,131 s at the three ε.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc` (and `-D warnings` on `nd-fmm-validate`); `cargo test` (1,068) and
  `--workspace` (1,149); the new test at 2 and 3 ranks by hand. A self-review by a
  subagent: no defect in the harness; its report corrections and the missing 10⁻⁸ rows
  were taken. Not run: the MPI list (no change to `nd-octree`, `nd-fmm-plan` or
  `nd-fmm-exec`); locust and Kathleen (none in the plan); the licence check (no
  dependency changed).
- Decisions: questions for sign-off (the PR): 1. "against the direct sum" read as
  relative ℓ2 errors of φ and ∇φ at most ε with S7's degrees at ε; 2. W3 at kD = 100,
  ε = 10⁻⁸ left out on the M6's memory (or run on Kathleen in CH3.3). Both signed off
  2026-11-01, the user: "1 and 2 as recommended".
- Laplace: not touched (no compiled change outside `nd-fmm-validate`).
- Follow-ups:
  - CH3.3: S7's closed form leaves errors 3 × 10³ to 6 × 10⁶ below ε on low-frequency
    trees (p = 16, 34, 46 on every low-frequency level); the high-frequency levels set the
    error. The harness is its measuring tool.
  - CH3.4 and CH3.2: the cost (reference P2P, RCR at p = 34–46: 5.7–8 times 10⁻³'s time at
    10⁻⁶).
  - CH3.6: the harness's oracle and norms for the FMM3D comparison.

## 2026-11-01: CH3.4a The vectorised sine and cosine
- Done:
  - CH3.4 split into CH3.4a (this), CH3.4b (the Helmholtz P2P on the skeleton) and CH3.4c
    (the operator's P2P); the user, "Split a/b/c", and the plan, "Proceed as planned".
  - `arch::sincos` (new): sin θ and cos θ from one reduction by π (n by the shifter,
    t = θ − n π₁ exact by Sterbenz, − n π₂ rounded once; two parts, not the planned
    three), Taylor polynomials in t² (f64 to t²¹ and t²⁰, f32 to t¹³ and t¹²) and the sign
    (−1)ⁿ = 1 − 2b² formed exactly. Only the existing layer operations, so every ISA gives
    the scalar layer's bits; no layer method added. 25 (f32) and 33 (f64) operations.
  - `sincos::sincos_slice` (public), entry points per ISA, `Sealed::{SINCOS,
    sincos_slice}`; design `docs/design/simd-p2p.md` §4.11 with the derivation (2.7 u
    sine, 3.4 u cosine, first order; `fmm-simd/tools/sincos_bound.py`);
    `fmm-simd/CLAUDE.md`. Record: `docs/helmholtz/tasks/CH3.4a.md`.
- Tests: `fmm-simd/tests/sincos/` (11, plus 2 ignored release tests): the 4 u_T absolute
  contract on |θ| ≤ 2¹⁰, f32 over every float of [0, 2¹⁰] and its negation (1.886 u sine,
  2.566 u cosine), f64 on 10⁷ samples against double-double (1.890 u, 2.396 u), the
  reduction's boundaries and the multiples of π; odd and even bit for bit for θ ≠ 0;
  every ISA the scalar ISA bit for bit; lane and length independence; the oracle against
  80-digit values and libm; panics.
- Checks: `/verify` on the M6: format; Clippy default, examples, workspace all-targets and
  `nd-fmm-simd` native and x86_64; `cargo test` (1,080) and `--workspace` (1,161);
  `cargo doc` (and `-D warnings` on `nd-fmm-simd`, private items); `nd-fmm-simd`'s tests
  and ignored release tests. Kathleen job 617308 (`check.sbatch simd`, node-c11a-094):
  passed, the same maxima on scalar, AVX2 and AVX-512. The inlining check: no call; NEON
  45/61, AVX2 42/58, AVX-512 37/45 instructions. A self-review by a subagent: no defect;
  its points taken. Not run: the MPI list (no existing function changed); the licence
  check (no manifest changed).
- Decisions: the split and the plan, as above. Question for sign-off (the PR): 1. sin(±0)
  = +0 (the sine's last step adds a +0 correction to t = −0), the symmetry bit for bit for
  θ ≠ 0, against the plan's "sin ±0 with its sign". Signed off 2026-11-01, the user:
  "Accept question 1 as recommended".
- Laplace: unchanged in code. Every existing function of `nd-fmm-simd` identical to
  `main`'s instruction for instruction in the release rlibs of aarch64 and x86_64, macOS
  and Linux, at the default settings and with one codegen unit (8 builds); so no D9 run.
- Follow-ups:
  - CH3.4b: the Helmholtz P2P on the skeleton, e^{iθ} by this sine and cosine and
    e^{−Im(k) r} by CH2.4a's exponential; its contract over Re(k) r ≤ 22 c_leaf.
  - CH3.4c: the operator's P2P.

## 2026-11-01: CH3.4b Helmholtz P2P on the skeleton
- Done:
  - The P2P skeleton with planes: `PairKernel<T, S, QP, OP>`, QP charge planes per source
    (point-major) and OP output planes per target (stride n), compile-time; Laplace and
    Yukawa the one-plane case, the tail and `sub_block`'s range kept as CH2.4b's for one
    plane. The user, 2026-11-01, each "(Recommended)": "Proceed as planned", "Split planes",
    "Undamped and damped".
  - `P2pKernel::evaluate_helmholtz` with `HelmholtzCharges` (real; complex Re q, Im q per
    source) and split output planes: the pair kernel `arch::p2p::Helmholtz<V, DAMPED>`
    (θ = Re(k) r by `arch::sincos`, e^{−Im(k) r} by `arch::exp` only when Im k > 0, the
    gradient factor t (f − iθ)), 16 entry points and a selector per ISA, `HELMHOLTZ_K` = 1,
    the scalar path `scalar::p2p_helmholtz` (the reference's loop). Design
    `docs/design/simd-p2p.md` §4.12; `fmm-simd/CLAUDE.md`; the spike example
    `spikes/p2p-simd/examples/helmholtz.rs`. Report `fmm-validate/results/helmholtz-ch3-4b.md`;
    record `docs/helmholtz/tasks/CH3.4b.md`.
- Tests: `tests/p2p/helmholtz.rs` (23, plus 4 ignored release tests): terms within
  (32 + 12 |k| r) u_T and (66 + 12 |k| r) u_T over Re(k) r ≤ 2¹⁰ (measured at most 10.3u and
  9.0u per unit of |k| r over 10⁶ pairs, NEON, AVX2, AVX-512); sums against `direct_sum`; the
  real plane `evaluate`'s at k = 0 and `evaluate_yukawa`'s at k = iκ bit for bit; complex
  charges with Im q = 0 as real ones; the scalar path the reference bit for bit; invariance;
  coincident and empty inputs; panics. `src/arch/tests.rs`: every K and both charge kinds and
  dampings bit for bit, and the Laplace and Yukawa identities on every layer.
- Checks: `/verify` on the M6 (format; Clippy default, examples, workspace all-targets,
  `nd-fmm-simd` for x86_64; `cargo test` 1,104 and `--workspace` 1,186; `cargo doc`; the
  ignored release tests; the spike's test; the MPI list at 2 and 4 ranks, 24 of 24);
  Kathleen jobs 617314 and 617322 (`check.sbatch simd`, before and after the review's wider
  sampling) passed; the inlining check (no call; NEON
  59–167, AVX2 70–181, AVX-512 53–122 instructions). Not run: the licence check.
- Runs: cost per pair (one thread, W1): complex charges at real k 3.8–5.9× Laplace and
  1.4–1.7× Yukawa on the default ISAs of the M6, locust and Kathleen (job 617315); damping
  1.2–1.5× more; 1.7–11.9× faster than green-kernels' Helmholtz. K = 1 (K = 2 no faster on
  NEON or AVX-512).
- Decisions: the three above. Questions for sign-off (report §7): 1. the contract with the
  f32 gradient range (1 + |k| r)/r³ ≤ 2¹²⁶; 2. constraint 1 as met; 3. the panic for
  Re k < 0. Signed off 2026-11-01, the user: "i accept 1 and 2", and for 3 "A as
  recommended" (the panic kept).
- Laplace: `nd-fmm-simd`'s skeleton changed (constraint 1). Every Laplace function on NEON
  and AVX2 `main`'s instruction for instruction at the default settings (aarch64 and x86_64,
  macOS and Linux), every other one too but NEON's `p2p_yukawa_f32_gradient` on macOS (two
  accumulators exchange registers in its source loop); AVX-512's `p2p_f32_gradient` and `p2p_f64_gradient` keep `main`'s source loops up to
  register names but control the block loop differently: D9 on Kathleen (jobs 617315 and
  617321, `p2p_kernels --isa avx512` and the FMM at p = 3 and 8, A B B A A B B A A B) 16 of
  16 rows within noise, outputs equal; the FMM's outputs of `main` and the branch byte for
  byte identical in 18 of 18 cases (f64, p = 3, 6, 8, every strategy, AVX-512). With one codegen unit NEON's `merge` differs (not in
  the default build).
- Follow-ups:
  - CH3.4c: `HelmholtzOperator`'s P2P through `evaluate_helmholtz` (the chunks' layout is the
    kernel's: Re plane, then Im plane), the reference kept as the trusted path.

## 2026-11-01: CH3.4c The Helmholtz operator's P2P
- Done:
  - `HelmholtzOperator::{with_p2p, p2p_kernel}`, `HelmholtzFmmBuilder::p2p_kernel` and
    `HelmholtzFmm::p2p_kernel` with Laplace's `P2pChoice` (default `Auto`): P2P through
    `P2pKernel::evaluate_helmholtz` on the ISA, one call per vector on all the leaf's
    targets (the chunk's own strengths for one vector, gathered into scratch for several;
    the output planes of columns 2v and 2v + 1 as they lie, the kernel's layout);
    `Reference` keeps `nd_fmm_ref::helmholtz::p2p::p2p`, the trusted path. An unavailable
    ISA is `IsaUnavailable` from the operator and `SettingsError::P2pIsaUnavailable` from
    the builder, before any collective.
  - `nd_fmm_validate::helmholtz_accuracy::{Case, Row}::p2p` and the example's `--p2p`.
    Record `docs/helmholtz/tasks/CH3.4c.md` (with the times); `fmm-exec/CLAUDE.md`,
    `fmm-validate/CLAUDE.md`.
  - CH3.4 is done with this task: once it is merged, CH3.6 waits for CH3.3 alone.
- Tests: `fmm-exec/tests/helmholtz/p2p.rs` (new): every available ISA against `Reference`
  pair by pair within CH3.4b's per-term contract summed over the sources (worst 0.019 of the
  bound on NEON), f64 and f32, k up to 128 + 16i (Re k̂ r to 53), real and complex charges,
  one and two vectors, with and without gradients, a coarser neighbour; `Isa::Scalar` the
  reference bit for bit, `Auto` that of `Isa::detect`. `tests/helmholtz/operator.rs`:
  `Reference` and `Isa::Scalar` bit for bit `nd_fmm_ref`. `tests/helmholtz_fmm.rs`: every
  scenario on `Auto` (threads, pair calls and the one-rank FMM bit for bit, as before); on
  two scenarios every ISA within 1e-13 of `Reference` (6.9e-16 measured), `Scalar` bit for
  bit; an unavailable ISA refused on every rank. Debug run at one rank 36 → 46 s.
- Runs (M6, release, one rank, NEON against `Reference`, two launches in both orders): the
  evaluation 0.15–0.48 s faster at one thread on W1 kD = 100, the W4 cube at kD = 10 and W3
  at kD = 30 (2–9 % at ε = 10⁻³, under 1 % at 10⁻⁶: the RCR translations dominate); the
  6-thread runs vary by up to 25 % between launches; errors unchanged, `Reference`
  checksums CH3.1c's, each kernel's equal at 1 and 6 threads.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc` (and `-D warnings` on `nd-fmm-exec` and `nd-fmm-validate`); `cargo test`
  (1,106) and `--workspace` (1,188); the MPI list at 2 and 4 ranks (24 of 24);
  `helmholtz_fmm` in release at 1 and 3 ranks. A self-review by a subagent: no blocking
  defect; its points taken (one vector and no gradients on every ISA, the notes). Not run:
  AVX2 and AVX-512 (CI's x86_64 runner runs the operator and FMM tests on its ISAs);
  locust and Kathleen (none in the plan); the licence check (no manifest changed).
- Decisions: none; no question for sign-off.
- Laplace: not touched (no Laplace file changed; `P2pChoice` reused as it is).
- Follow-ups:
  - CH3.2: the translations are where the Helmholtz FMM's time is.
  - CH3.3, then CH3.6 (timed with this kernel).

## 2026-11-01: CH3.3 Calibration of p_l over (k a_l, ε)
- Done:
  - The plan's three choices, the user on 2026-11-01: "Calibrate, new default", "Both
    branches", "ε/2, verify ≤ ε" (each "(Recommended)").
  - `nd_fmm_exec::helmholtz::DegreeRule` (public: a plateau ⌈γ L² + α L + β⌉ and a branch
    ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉, ceilings forgiving 10⁻⁹), `DegreeRule::CALIBRATED`
    (γ = 0.30, α = 0.25, β = 2.60, C = 0.80), `degree` on it (the default of
    `Degrees::Tolerance`), `s7_degree` (S7's closed form, the old default, kept as the
    per-pair bound).
  - `nd_fmm_validate::helmholtz_calibration` (MPI-free: `Curve`, `Point`, the fit),
    `helmholtz_accuracy::Row::circumradii`, the `helmholtz_calibrate` example (plateau and
    level scans, `fit`), `helmholtz_accuracy --rule s7`. Report
    `fmm-validate/results/helmholtz-ch3-3.md`; record `docs/helmholtz/tasks/CH3.3.md`.
- Tests: `fmm-exec/tests/helmholtz/degree.rs` (S7's closed form at its grid, now through
  `s7_degree`; the calibrated rule at least every measured degree, at most S7's and
  monotone on a grid of x and ε; `DegreeRule`'s shapes; `resolve`), the doctest;
  `tests/helmholtz/operator.rs`'s per-pair chains on `s7_degree` (their degrees and bits
  `main`'s); `helmholtz_calibration`'s unit tests (9); `tests/helmholtz_accuracy.rs` a
  plateau scan on W1 at kD = 10 and the default rule at most S7's.
- Runs (M6, release, 1 × 6, NEON P2P): 10 plateau scans (W4 ×4 at kD = 1, 10; W1, W3 at
  kD = 10; ε = 10⁻³ … 10⁻¹⁰) and level scans (W1 at kD = 30, 100, top levels at 200 and
  300; W3 at kD = 30; the W4 cube and clusters at kD = 100; ε = 10⁻³ … 10⁻⁸): 176 points,
  S7's closed form at or above every one (the acceptance). The plateau 7, 9, 12, 15, 20, 24
  at 10⁻³ … 10⁻⁸ against S7's 16 … 46. Verification of the default against ε: CH3.1c's
  62 cases (now with W3 at kD = 100, 10⁻⁸) and 26 held out, all within ε, the largest
  0.63 ε (W1, kD = 500, 10⁻⁸); checksums equal at 1 and 6 threads (15 cases) and at 1, 2,
  3 ranks (2). Times against S7's degrees: 2.7× at 10⁻³, 6–8× at 10⁻⁶ and 10⁻⁸, 4.9× on
  W1 at kD = 500, 10⁻⁶ (98 against 478 s).
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc`; `cargo test` (1,118) and `--workspace` (1,200); the MPI list at 2 and 4
  ranks (24 of 24). A self-review by a subagent: no code defect; the report's case counts
  (62 and 26, not 63 and 27), the acceptance's wording, a caveat above x = 65, stale plan
  text, doc wording and a saturating subtraction fixed. Not run: locust and Kathleen (none
  in the plan); the licence check (no manifest dependency changed; `fmm-validate/Cargo.toml`
  gained an `[[example]]` only).
- Decisions: the three above. Questions for sign-off (the PR): 1. the quadratic plateau;
  2. the high-frequency branch above x = 65 an extrapolation below S7's p* at small ε
  (checked at kD = 500 only); 3. the per-pair chain tests on `s7_degree`. Signed off
  2026-11-01, the user: "1–3 as recommended" (the quadratic plateau accepted; the branch
  accepted for CH3.6 at kD ≤ 500, its form left to CH4.3, a kD ≈ 1000, 10⁻⁸ check on
  Kathleen before relying on it there; the chain tests on `s7_degree`).
- Laplace: not touched (no Laplace file changed).
- Follow-ups:
  - CH3.6: on this rule (W1 at kD = 500, 10⁻⁶ in 98 s).
  - CH4.3: the high-frequency branch, x + (a L + b) x^{1/3} the candidate (over-predicts at
    10⁻³ by up to 8 on the top levels; under S7's p* beyond x ≈ 64 at small ε).
  - A check at kD ≈ 1000, 10⁻⁸ (Kathleen) before relying on the rule there.

## 2026-11-02: CH3.6 Gate: RCR on every level against FMM3D
- Done:
  - `tools/fmm3d/driver.f90`: the `helmholtz` mode's optional FULL argument (φ and ∇φ at
    every point, `NDFMMHF1`).
  - `nd_fmm_validate::helmholtz_fmm3d` (the driver's files, its `seconds=` line, FMM3D's
    errors at the samples, the full-target crosscheck of a distributed output: relative ℓ2
    over all points and the worst error per box of the deepest level over the reference's
    RMS), `helmholtz_accuracy::{Run, run_output}`, the `helmholtz_fmm3d` example (`problem`,
    `compare`), `tools/fmm3d/run.sh helmholtz` and `fmm3d.sbatch`'s step (the gate's cases
    by default). Report `fmm-validate/results/helmholtz-ch3-6.md`; record
    `docs/helmholtz/tasks/CH3.6.md`.
  - `run.sh`: the detected FMM3D variants list without a leading space (the log headers
    named no variant).
- Tests: `helmholtz_fmm3d`'s unit tests (7: the file round trips in the driver's byte
  layout, the `seconds=` line, `box_of` at corners and faces, the crosscheck norms);
  `tests/helmholtz_accuracy.rs` (W1 at kD = 10, every point sampled, the direct sum as a
  stand-in FMM3D: the crosscheck equals the sampled errors to 10⁻¹², the box errors equal
  on one rank and several; one rank in CI, 2 and 3 by hand on the M6).
- Runs (Kathleen, one node each, release, BLAS threads 1; ours 4 ranks × 10 threads with
  the AVX-512 P2P, FMM3D v2.1.0 default and fastker on 40 threads): job 617649 (driver),
  617650 (W1 at kD = 10, 30, 100, 500; W1 at kD = 100 on one thread), 617651 (W3 at
  kD = 10, 30, 100; W4 ×4 at kD = 1, 10, 100); ε = 10⁻³, 10⁻⁶, 10⁻⁸. Ours within ε at the
  samples on all 57 points, the largest 0.63 ε (∇φ, W1 at kD = 500, 10⁻⁸); over all targets
  at most 0.43 ε. FMM3D misses its eps at 10⁻³ at kD = 100 (W1 1.23, W3 2.03, the cube
  2.12 ε) and is at most 0.04 ε at 10⁻⁸. Times at equal ε: ours 0.29–0.65 of FMM3D default
  and 0.55–1.31 of fastker at 10⁻³, 0.93–3.4 and 1.65–5.5 at 10⁻⁸; W1 at kD = 500 32, 52,
  100 s against 49, 63, 72 (default) and 33, 44, 49 (fastker). Worst box over the
  all-target ℓ2: ours up to 146×, FMM3D 37×. M6 smoke runs (W1 at kD = 10, 30, 100, the
  W4 cube at N = 10⁴) agree.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc` (and `-D warnings` for nd-fmm-validate); `cargo test` (1,125) and
  `--workspace` (1,207); `tools/fmm3d/build.sh driver` and `helmholtz-check` on the M6, the
  driver relinked on Kathleen. Not run: the MPI list at 2 and 4 ranks (no nd-octree,
  nd-fmm-plan or nd-fmm-exec change; the new test is in `helmholtz_accuracy`, not in the
  list), the licence check (no dependency change), D9 (Laplace untouched). A self-review by
  a subagent: no blocking finding; report wording fixed (the problem file holds sample
  indices, the lowest error ratio, the time-ratio filter, the one-thread checksums).
- Decisions: none taken. Questions for sign-off (the PR): 1. the worst-box concentration
  (ours 30–146× the ℓ2 error at 10⁻⁶ and 10⁻⁸, FMM3D up to 37×); 2. the acceptance's
  reading (same requested ε, FMM3D not gated, the box measure on the deepest level without
  offset classes, one charge vector rather than §9.1's eight). Signed off 2026-11-02, the user:
  "both decisions accepted with your recommendations" (1: the gate on §9.1's ℓ2, the
  concentration a follow-up before H4; 2: the reading accepted).
- Laplace: not touched (no library file changed; `run.sh`'s Laplace steps get only the
  variants fix).
- Follow-ups:
  - The worst-box concentration (question 1): where it comes from (the boxes at the cube
    face tangency on W1 at 10⁻⁶ and 10⁻⁸, not a boundary effect), and whether a pointwise
    criterion should join §9.1's.
  - Performance at matched accuracy (§9.2: after CH4.5 and CH7.6); FMM3D over-delivers by
    up to 100× at 10⁻⁸.
  - H3.G needs CH3.2.

## 2026-11-02: CH3.2a Dense
- Done:
  - CH3.2 split into a (Dense), b (Classes) and c (Batched, the measurements per p on the
    three machines), each one reviewable PR; CH3.2a taken as named by the user (off the
    priority list, which ended with CH3.6).
  - `nd_fmm_tables::helmholtz::dense::DenseM2l`: per level and offset the complex M2L as one
    2n × n column-major matrix [A_r; A_i], column j Rotation's M2L of e_j in f64
    (`HelmholtzTables<f64>`), rounded to T; `select` (the levels within a byte budget from
    the finest up), `level_bytes`, `m2l` (one plane: k ascending over 2n rows; two planes:
    one pass, the four real products combined per k).
  - `nd_fmm_exec::helmholtz::{HelmholtzM2l, DEFAULT_DENSE_BUDGET}`,
    `HelmholtzOperator::{with_m2l, m2l_choice, m2l_on, dense_bytes}`,
    `HelmholtzFmmBuilder::{m2l, dense_budget}`: Dense on the levels that fit, RCR on the
    others, per pair in the kernel's `m2l` (no driver change, `batched_m2l` stays `None`).
  - `nd-fmm-validate`: the `helmholtz_m2l` example (per p, Dense against Rotation per pair),
    `helmholtz_accuracy`'s `--m2l` and `--dense-budget`. Report
    `fmm-validate/results/helmholtz-ch3-2a.md`; record `docs/helmholtz/tasks/CH3.2a.md`.
- Tests: `fmm-tables/tests/helmholtz/dense.rs` (every offset of levels 2–4 for five k̂
  against Rotation, at most 3.7e-14 per unit of conditioning in the FMM-weighted measure,
  bound 2e-13; a sample against `rcr`; columns bit for bit Rotation of e_j; vectors; one
  plane against two with Im = 0; f32 the f64 tables rounded; `select`; refusals);
  `fmm-exec/tests/helmholtz/dense.rs` (the operator pair by pair against Rotation within
  1e-14, 7.2e-16 measured; the budget; a zero budget bit for bit Rotation);
  `tests/helmholtz_fmm.rs` `check_dense` on two scenarios (the FMM within 1e-14 of
  Rotation, 7.9e-16 measured, and the direct-sum bound; 2 threads, pair by pair and the
  one-rank FMM bit for bit; the finest level alone). One-rank debug run 46 → 61 s on the
  M6, 7 s in release.
- Runs (M6, release, one rank, one thread, BLAS threads 1, every launch behind a quiet
  gate): `helmholtz_m2l` at p = 3–20, two launches:
  Dense faster per pair to p = 5 (2.5–2.7× at p = 3), level at 6–7, 0.42× at p = 12 and
  0.22× at p = 20 (memory-bound: 2 (p + 1)⁴ values per pair); tables 21 MB at p = 7, 331 MB
  at 15, 983 MB at 20 per level, built in 0.02–1.4 s. `helmholtz_accuracy --m2l` both orders
  on W1 at kD = 30 and 100, the W4 cube at kD = 10 and W3 at kD = 30, ε = 10⁻³ and 10⁻⁶:
  Dense 0.97–1.03× Rotation's evaluation time at p = 7 (the cube and W3 at 10⁻³), 1.5–2.6×
  above, the errors Rotation's to the printed digits, each strategy's checksum the same in
  both launches. The first, ungated runs had timed §2 on a `helmholtz_accuracy` built before
  the one-pass chain (1.4–4.6×); rerun at the user's question about a quiet machine.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc`; `cargo test` (1,137) and `--workspace` (1,219); `helmholtz_fmm` in release at
  one rank; the MPI list at 2 and 4 ranks, 26 of 26 (`helmholtz_fmm` 75 and 72 s;
  `mpi_exec`, unchanged, at 2 ranks hit the script's local 300 s default at opt-level 0 and
  passed in 298 s with `RUN_TIMEOUT=900`, CI's limit). A self-review by a subagent: no code defect; its
  points taken (the debug runtime, the plan and report wording, tighter tolerances, a
  length refusal test, no f64 build without a dense level, example overflow). Not run:
  locust and Kathleen (CH3.2c's); D9 (Laplace untouched); the licence check (no dependency
  change; `fmm-validate/Cargo.toml` gains an `[[example]]`).
- Decisions: none taken. Questions for sign-off (the PR): 1. Dense stays opt-in, no Auto
  rule until CH3.2c; 2. Dense's two-plane chain is what CH3.2c's Batched reproduces bit for
  bit or declares within rounding; 3. `DEFAULT_DENSE_BUDGET` 1 GiB per rank, no node budget. Signed
  off 2026-11-02, the user, after the merge: "i agree with the recommendations" (1–3 as
  recommended).
- Laplace: not touched (no Laplace file, no driver; `nd-fmm-tables` gains a submodule of
  `helmholtz` only).
- Follow-ups:
  - CH3.2b (Classes): 16 class tables per level, about 20× less to read per level.
  - CH3.2c (Batched): where dense tables can pay, read once per block of targets.
  - The M2L y-blocks of `HelmholtzTables` are still stored for a level that runs Dense.

## 2026-11-02: CH3.2b Classes
- Done:
  - CH3.2a's sign-offs recorded first (8693987; the user, after #171 merged: "i agree with
    the recommendations").
  - `nd_fmm_tables::helmholtz::classes`: `CubeTransforms` (Oⁿ(P) of the orthonormal
    coefficients for the 48 elements of O_h: rotations factored into quarter turns about z
    and y, the y-block of π/2 from `orthonormal::rotation::y_blocks` at cos 0, sin 1
    exactly; Oⁿ(P) = (−1)ⁿ Oⁿ(−P) for improper P; the 16 elements that fix the z-axis exact
    signed permutations, the 32 others dense per-degree blocks), and `ClassM2l` (per level
    the 16 class matrices by `DenseM2l`'s build, bit for bit its matrices of the
    representatives; per pair O(Pᵀ), Dense's chain into zeroed scratch, O(P) added;
    `select` from the finest level up with the transforms charged; `ClassScratch`).
    `helmholtz::dense`'s build and chain moved into shared helpers, the same operations.
  - `nd_fmm_exec::helmholtz`: `HelmholtzM2l::Classes` (`with_m2l` within the dense budget,
    `m2l_on`, `class_bytes`, the class scratch per thread), the builder's `dense_budget`
    covering both.
  - `nd-fmm-validate`: `helmholtz_m2l` with Classes (all offsets, the z-axis and moving
    offsets apart, the transforms alone, bytes and build), `helmholtz_accuracy --m2l
    classes` (`Row::m2l`). Report `fmm-validate/results/helmholtz-ch3-2b.md`; record
    `docs/helmholtz/tasks/CH3.2b.md`.
- Tests: `fmm-tables/tests/helmholtz/classes.rs` (**Y**_n(P x̂) = Oⁿ(P) **Y**_n(x̂) for all 48
  P at p = 12, 2.2e-15; orthogonality, the homomorphism, the z-axis elements' structure
  including the x–y exchanges, −I; every offset of levels 2–4 for five k̂ against Dense,
  3.7e-14 per unit of conditioning within 2e-13; the class matrices and the representatives
  bit for bit Dense; vectors, one plane against two, f32, `select`, refusals);
  `fmm-exec/tests/helmholtz/dense.rs` (pair by pair against Rotation and Dense, 5.6e-15
  within 5e-14; the budget; a zero budget bit for bit Rotation); `tests/helmholtz_fmm.rs`
  `check_tables` for Dense and Classes on two scenarios (Classes 2.4e-16 of Rotation's output
  within 1e-14; 2 threads, pair by pair, the one-rank FMM bit for bit; the finest level
  alone). One-rank debug run 61 → 69 s on the M6.
- Runs (M6, release, one rank, one thread, BLAS threads 1, both examples rebuilt first, the
  launches started after the user confirmed the machine quiet, every launch gated, none
  void): `helmholtz_m2l` at p = 3–20 twice: Classes 1.27–1.31× Rotation's speed at p = 3,
  level at 5, 0.71–0.75× at 7, 0.22× at 20; 0.48–0.51× Dense's at p = 3 (transforms about
  half the pair), 0.86–1.00× from p = 12 (6–11 %); tables 1.2, 18 and 53 MB at p = 7, 15,
  20 (Dense 21, 331, 983), built in at most 0.07 s. `helmholtz_accuracy --m2l` in both orders
  on CH3.2a's cases: Classes 1.23–1.25× Rotation's evaluation on the W4 cube at 10⁻³ and W3,
  1.7–3.0× on W1 and at 10⁻⁶, 1.07–1.42× Dense's; errors Rotation's to the printed digits,
  checksums the same in both launches.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc`; `cargo test` (1,156) and `--workspace` (1,238); the MPI list at 2 and 4 ranks,
  24 of 24 (`helmholtz_fmm` 62 and 49 s); after the review's fixes again format, workspace
  Clippy, `nd-fmm-exec`'s `helmholtz` tests, `nd-fmm-tables`' tests and the docs. A
  self-review by a subagent: no blocking finding; its points taken (the plan brought in line
  with the code, the x–y exchange test, recorded measurements, doc wording, a test constant
  from the degrees, the example's transposes hoisted out of the timed loop). Not run: locust
  and Kathleen (CH3.2c's); D9 (Laplace untouched); the licence check (no dependency change).
- Decisions: none taken. Questions for sign-off (the PR): 1. Classes stays opt-in, no Auto
  rule until CH3.2c; 2. Classes shares Dense's budget, the transforms counted; 3. whether
  §3.15 should state the orthonormal transforms at its next version bump (they follow from
  §3.12 and §3.15 as they stand); 4. the correction of CH3.2a's "bound by memory" (the
  arithmetic binds per pair), left in this report rather than in CH3.2a's. Signed off
  2026-11-02, the user: "i agree with the recommendations" (1: opt-in; 2: the shared budget,
  CH3.2c's rule decides the levels; 3: §3.15 at its next bump, not now; 4: the correction
  here, CH3.2a's report unchanged).
- Laplace: not touched (no Laplace file, no driver; `nd-fmm-tables` gains a submodule of
  `helmholtz`, Laplace's `symmetry` used, not changed).
- Follow-ups:
  - The dense transforms on the compact y-block with two signed permutations: about half
    their multiply–adds ((n + 1)² + n² of (2n + 1)² entries are non-zero).
  - CH3.2c: the batched M2L, where tables can pay; Classes' tables could carry it to
    degrees where Dense's do not fit if the transforms batch too.

## 2026-11-02: CH3.2c Batched, and the measurements per p
- Done:
  - Decisions before the work (the user, 2026-11-02): Helmholtz's own block, not the
    two-plane case of `driver::m2l_block` ("Own Helmholtz block"); Dense's tables in split
    planes, the two-plane chain Dense to rounding ("Split planes, 2n²").
  - `nd_fmm_tables::helmholtz::batched::BatchedM2l`: Dense's matrices re-laid as A_r and
    A_i, each n × n column-major, bit for bit Dense's entries; Dense's `select` and bytes.
  - `nd_fmm_exec::helmholtz`: `HelmholtzM2l::Batched`, `batched_bytes`,
    `with_m2l_mul_add`/`m2l_mul_add` and the builder's `m2l_mul_add`; the crate-private
    `src/helmholtz/batched.rs`: the M2L level call by blocks of `M2L_BLOCK_TARGETS` on
    `Execution::blocks` with `nd_fmm_simd::MatrixKernel` (P2P's ISA), two calls per offset
    for real multipoles, four for complex ones (L_r negated around the A_i M_i call), and the
    pair body with the same calls. `driver.rs`, `nd-fmm-simd` and Laplace unchanged.
  - `nd-fmm-validate`: `helmholtz_m2l` with Batched, separate and fused (the operator's calls
    on a block of 32 pairs per offset); `helmholtz_accuracy --m2l batched,batched-fused`.
    Report `fmm-validate/results/helmholtz-ch3-2c.md`; record `docs/helmholtz/tasks/CH3.2c.md`.
- Tests: `fmm-tables/tests/helmholtz/batched.rs` (the planes Dense's halves bit for bit on
  every offset of levels 2–4 for five k̂; bytes and `select`; f32; refusals);
  `fmm-exec/tests/helmholtz/dense.rs` (pair by pair on every offset of levels 2–4: Dense
  bit for bit with real multipoles, 7.4e-16 with complex ones, fused 6.9e-16, within
  `BATCHED_BOUND` 1e-14; the same bits on every ISA in both modes; the budget; a zero budget
  Rotation bit for bit); `tests/helmholtz_fmm.rs` `check_tables` with Batched on two scenarios
  (Rotation's output within 7.6e-16, Dense's bit for bit with real multipoles and within
  3.3e-16 with complex ones; 2 threads, pair by pair, two vectors through the block against
  the pair body, the one-rank FMM bit for bit; the fused mode within 7.6e-16 of Rotation and
  its one-rank FMM). One-rank debug run 69 → 122 s on the M6.
- Runs (release, one rank, one thread, BLAS threads 1; the M6 behind the quiet gate after
  the user's go-ahead, one void launch repeated; locust; Kathleen job 618018):
  `helmholtz_m2l` at p = 3–20 twice: fused Batched 1.84×, 1.77× and 4.52× Rotation's speed
  per pair at p = 7 with two planes on the M6, locust and Kathleen, level at p = 10 on the
  M6 and locust, between 15 and 18 on Kathleen; separate Batched 0.99–1.25× Dense's speed on
  the M6, 2.8–8.5× on Kathleen. `helmholtz_accuracy` on CH3.2a's cases in both orders:
  fused Batched 0.72–0.74×, 0.78–0.82× and 0.39–0.42× Rotation's evaluation on the W4 cube
  at 10⁻³ and W3; 1.27–2.76× on the M6 and locust where the levels with tables carry
  p = 15–35, 0.80–1.13× on Kathleen; errors Rotation's to the printed digits, checksums the
  same in both orders.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc`; `cargo test` (1,167) and `--workspace` (1,249); the MPI list at 2 and 4
  ranks, 24 of 24 (`helmholtz_fmm` 197 and 148 s at opt-level 0); after the review's fixes
  format, workspace Clippy, `nd-fmm-exec`'s `helmholtz` tests and `helmholtz_fmm` again.
  A self-review by a subagent: no blocking finding; its points taken (two vectors through the
  block, the bounds' measured values, doc wording). Not run: D9 and the Laplace
  bit-identity runs beyond CI's list (no Laplace file, no driver change); the licence check
  (no dependency change).
- Decisions: none taken beyond the two before the work. Questions for sign-off (the PR):
  1. Batched opt-in, no Auto rule here; a per-level rule by p as a follow-up before H3.G;
  2. the fused mode added as an opt-in; 3. the debug test cost; 4. the proposal's §6.2 and
  §10 left as written, the roadmap and record carrying the own block. Signed off
  2026-11-02, the user: "i agree with the recommendations" (1: opt-in, the rule by p a
  follow-up before H3.G; 2: the fused mode kept; 3: the debug cost kept; 4: as stated).
- Laplace: not touched (no Laplace file, no `driver.rs`, `nd-fmm-simd` or `nd-fmm-plan`
  change; the driver's `Columns`, `Execution::blocks` and `M2L_BLOCK_TARGETS` used as they are).
- Follow-ups:
  - The M2L rule per level (question 1): fused Batched below a machine's crossover degree,
    Rotation above, within the budget; with sparse levels (W1) counted. Added to the
    roadmap as CH3.2d at the sign-off, before H3.G.
  - CH3.2a–c are done with this task; H3.G waits for CH3.2d.

## 2026-11-02: CH3.2d The M2L rule per level
- Done:
  - Taken ahead of CH0.6 (the first runnable task in file order once the priority list was
    done) at the user's choice ("CH3.2d").
  - `nd_fmm_tables::helmholtz::batched::BatchedM2l::select_within`: Dense's choice from the
    finest level up among the levels of degree at most a crossover.
  - `nd_fmm_exec::helmholtz`: `HelmholtzM2l::Auto` (fused Batched on the levels of degree
    at most the crossover that fit the budget, Rotation above), `auto_batched_max_degree(isa,
    planes)` (NEON 10 and 12, AVX-512 15 and 15, AVX2 11 and 12, scalar 6 and 8, two planes
    and one), `auto_m2l_mul_add(isa)` (fused, separate for the scalar kernel without a
    hardware fma), `with_m2l_max_degree`, `max_degree`; the named mode kept across
    `with_m2l` and `with_p2p`; the builder's `m2l_max_degree`, the crossover agreed over the
    ranks (all-reduce min), `HelmholtzFmm::m2l_max_degree`.
  - `nd-fmm-validate`: `helmholtz_accuracy` with `auto`, `auto:<d>` and `sweep`
    (`sweep_crossovers`), `Case::m2l_max_degree`, `Case::m2l_mul_add` as an `Option`,
    `Row::m2l_max_degree`; `helmholtz_m2l --isa`; `tools/helmholtz/ch3-2d-runs.sh`. Report
    `fmm-validate/results/helmholtz-ch3-2d.md`; record `docs/helmholtz/tasks/CH3.2d.md`.
- Tests: `fmm-tables/tests/helmholtz/batched.rs` (`select_within` over degrees, crossovers
  and budgets; `usize::MAX` is `select`); `fmm-exec/tests/helmholtz/dense.rs` (every offset
  of levels 2–4 level by level: fused Batched bit for bit at or below the crossover,
  Rotation bit for bit above; the extremes; the budget; the rule and the mode of each ISA in
  either order of the calls; a named mode kept; the crossovers' order);
  `tests/helmholtz_fmm.rs` `check_auto` (one degree more above the finest level, so the
  clusters' tree splits by degree: Rotation of the same degrees within `DENSE_TOLERANCE`,
  the direct sum, 2 threads, pair by pair, the one-rank FMM bit for bit; the rule's
  crossover agreed); `fmm-validate/tests/helmholtz_accuracy.rs` (Auto below every degree
  Rotation's checksum, above every degree Batched's in Auto's mode; `sweep_crossovers`).
  One-rank debug run of `helmholtz_fmm` 122 → 150 s on the M6.
- Runs (release, one rank, one thread, BLAS threads 1; the M6 gated after the user's
  go-ahead, none void; locust; Kathleen job 618430): `helmholtz_accuracy --m2l
  rotation,auto,sweep` and the reverse on CH3.2a's seven cases and two with real charges,
  on the M6, locust, Kathleen AVX-512 and Kathleen AVX2. With the final constants the
  rule's choice is the best per-level choice on 34 of 36 case × machine rows, 1.001 on one
  (a tie), and 1.106 on one: Kathleen AVX-512, W1 at kD = 30 and 10⁻⁶, a sparse level at
  p = 15 batched (the cube at p = 15 needs Batched, so no degree-only crossover meets both).
  Errors the same for every choice. `helmholtz_m2l --isa` avx2 and scalar on Kathleen,
  scalar on the M6: the fused scalar kernel 2.4–50× slower than Rotation on Kathleen.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace all-targets;
  `cargo doc`; `cargo test` and `--workspace`; the MPI list at 2 and 4 ranks, 24 of 24;
  after the review's and the mode's changes `helmholtz_fmm` again at 1, 2 and 4 ranks and
  the full set once more. A self-review by a subagent: no blocking finding; its points
  taken (a split by degree in `check_auto`, the named mode across calls, `sweep_crossovers`
  tested, the report's figures, the f32 caveat, the bookkeeping). Not run: D9 and the
  Laplace bit-identity runs beyond CI's list (no Laplace file, no driver change); the
  licence check (no dependency change).
- Decisions: the margin's miss accepted during the work (the user, 2026-11-02, "Accept,
  follow-up"): CH4.0, the sparse-level term, added to the roadmap. Questions for sign-off
  (the PR): 1. Auto as the default of `HelmholtzFmmBuilder`; 2. the scalar kernel's mode;
  3. the debug cost. Signed off 2026-11-02, the user: "i agree with the recommendations"
  (1: Auto the builder's default, switched on this branch, the operator's own default and
  the harness's `Case` Rotation; 2: separate without a hardware fma; 3: kept).
- Laplace: not touched (no Laplace file, no `driver.rs`, `nd-fmm-simd` or `nd-fmm-plan`
  change).
- Follow-ups:
  - CH4.0: the rule's sparse-level term (the roadmap).
  - The crossovers are f64's; f32 takes them unmeasured (D3: f64 until H7).


## 2026-11-02: H3.G Gate
- Done:
  - `/milestone-gate H3`, the first Helmholtz gate. Report `docs/helmholtz/gates/H3.md`: the
    gate condition, all 16 tasks (#158–#174) and their sign-offs, CI on `main`, D1 and D11,
    proposal §9's items, constraint 1 per task, the measurements against §3.5, §5.3, §7,
    §5.7 and §11, corrections C1–C6, recommendations and a retrospective on the loop.
  - The gate met: CH3.6's FMM within ε at all 57 case × ε points (largest 0.63 ε sampled,
    0.43 ε over all targets; Kathleen jobs 617649–617651).
  - Roadmap: H3.G ticked; the status line; D6 signed off; CH0.8b on the calibrated p;
    CH4.0 gains the per-node dense budget, CH4.1 the resident memory at the calibrated
    degrees; CH4.0b (the worst-box concentration) and CH4.0c (the degree rule at
    kD ≈ 1000 and the under-resolved W3) added; a new priority list for H6 (CH0.4, CH6.1,
    CH6.2).
  - Proposal (draft 0.8 kept): C1–C6 applied, with a line in its Appendix A ("Amended at
    H3.G, 2026-11-02").
- Tests: none (documents only).
- Checks: none of cargo's (no code). `tools/roadmap/sync_issues.py --out` parses the
  edited roadmap (a dry run against the issues: H4's issue would change, nothing applied).
  CI on `main`: run-tests passed on #158–#174, except `run-tests-simd (x86_64)`, cancelled
  by its 5-minute limit on #166, #169 and #172; #174's run passed every job.
- Decisions: signed off 2026-11-02, the user: "i agree with the recommendations" (report
  §6): 1. H3 closed; 2. D6, H6 first; 3. C1–C6 applied; 4. CH4.0b and CH4.0c added after
  CH4.0; 5. `run-tests-simd`'s `timeout-minutes` from 5 to 10, in a separate PR; 6. the
  priority list for H6.
- Laplace: not touched (documents only).
- Follow-ups:
  - `run-tests-simd`'s limit, the separate PR of question 5.
  - The retrospective's changes to `/next-task` and `/verify` (a roadmap line for every
    signed-off follow-up; CI job times against their limits) and the debug test time of
    `helmholtz_fmm` (36 → 150 s). `.claude/skills/` is not writable from the session; for
    the user.

## 2026-11-02: CH0.4 S4: plane-wave quadratures per k w_l
- Done:
  - The first task of the H6 priority list. Spike `spikes/pw-quadrature/` (a workspace
    member outside `default-members`; num-complex and rayon from the workspace, a path
    dev-dependency on `spikes/plane-wave`): a parser of FMM3D's Helmholtz tables
    (`hwts_3d_iprec*.txt`, `hnumfour.f`, `hnumphys.f`) with `hwts3e`'s domain lookup,
    precisions and node maps; the kernel and point-to-point chain evaluators; an
    independent generator for any κ = k w (the real-t contour, Gauss–Legendre panels,
    pivoted Gram–Schmidt compression, Bessel-bound α and Fourier counts); the sweeps
    `domains`, `rays`, `levels` and the probe `alpha`; a fixture of five domains cut with
    NIST's notice; `tools/fetch.sh` (FMM3D's sources under `$FMM3D_ROOT`, outside the
    repository). Report `spikes/pw-quadrature/SPIKE_REPORT.md`, results in its `results/`.
  - Findings: FMM3D tabulates its quadratures (437 domains of complex κ up to 16π × 12π,
    5 precisions), it does not generate them; every λ sum meets its stated error to 1.5×
    (5.5× at iprec 4 near 16π, round-off); FMM3D's α counts miss the stated error by up to
    119× at the region's corner, and ε on 11 of 138 reference level rows by up to 1.8×;
    the tables cover every unpruned level of the reference trees with Re κ_l < 16π (only
    levels 2–3 at kD = 500 fall outside); the generator meets ε everywhere measured at
    1.5–1.9× FMM3D's wave values for real κ and 2.4–4.4× for Yukawa and strongly damped κ;
    FMM3D's plane waves stop at boxes of 8 wavelengths, not the 32 of its documentation.
- Tests: `tests/fixture.rs` (the parser, the lookup on all 437 rectangles, the λ sums
  within 1.5× the stated error at three points of each fixture domain, the grid sums
  against term-by-term evaluation, the untruncated chain against the kernel at t − s to
  10⁻¹³, the Laplace limit against `spikes/plane-wave`'s §3.14 tables);
  `tests/generator.rs` (ε on the grid, off it within 2ε, the chain within 2ε, real t,
  converged panels; the regression κ of the first count rule); `tests/fmm3d_full.rs`
  (ignored: all 437 × 5 tables). Debug run on the M6: fixture 18 s, generator 13 s.
- Checks: `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo test` (742 s) and `--workspace` (1,320 s); `cargo doc`; `cargo
  check --workspace --all-targets`; `cargo deny --locked check licenses`; no fftw-sys.
  After the self-review's changes the spike's format, Clippy and tests, default and
  ignored, again. Not run: the MPI list and D9 (no MPI crate, no Laplace path, no
  dependency of a default member changed). A self-review by a subagent: one blocking
  finding, the fixtures' NIST notice cut to 5 of its lines (fixed: the whole notice and
  a dated change note); the rest taken (the report's wording on α, pruned counts and
  ratio ranges; `results/alpha.md`; the generator's errors at the levels; its bounds
  documented; the tests' tolerances recorded).
- Decisions: questions for sign-off (the PR): 1. CH6.1's quadratures from FMM3D's tables
  (iprec 1–3) extracted with NIST's notice, or from the generator: signed off 2026-11-03: "i would like to revert my request not to use the nist tables and go vack to your earlier plan", FMM3D's tables
  (an earlier answer that day, the generator, withdrawn); 2. the
  α counts by measurement, not from `hnumphys`; 3. the proposal's "32 wavelengths"
  corrected to 8 (§5.3, §9.2, §11, §13): 2 and 3 signed off 2026-11-03, "yes to 2 and 3";
  the proposal corrected in this PR, marked "amended at CH0.4".
- Laplace: not touched (a new spike crate; the root manifest's `members` and one lock
  entry).
- Follow-ups:
  - CH6.1's line in the roadmap carries S4's recommendation, subject to the sign-offs.
  - The generator returns its best quadrature at its bounds without saying so; CH6.1
    checks the result if it uses it (report §4).

## 2026-11-03: CH6.1a The quadratures
- Done:
  - The first task of CH6.1 (priority list item 2), split on this branch into CH6.1a (the
    quadratures), CH6.1b (the pair chain, with a CONVENTIONS §3.15 subsection for
    sign-off) and CH6.1c (merge and shift in the FMM).
  - `nd_fmm_tables::helmholtz::plane_wave::quadrature`: FMM3D's quadratures for iprec 1–3
    (`hwts_iprec{1,2,3}.txt`, 437 domains each, 32,574 nodes, 1.68 MB; NIST's notice and
    a dated change note kept), written by the new `tools/fixtures/helmholtz_pw_quadrature.sh`
    from FMM3D v2.1.0 and parsed on first use; `hwts3e`'s lookup (`domain_of`,
    `rectangle`, `Precision::for_tolerance`; negative and non-finite parts refused) and
    node maps; `Quadrature::new(κ, precision)` with α and Fourier counts not FMM3D's but
    S4's Bessel bound for complex tₖ (share e/(4s) of the stated error, the corner
    ρ = 4√2 resolved, the chain's aliasing covered); `bessel_j_norms`, |J_m| of complex
    argument by Miller's recurrence normalised by e^{iz} at Im z ≤ 0.
  - `spikes/pw-quadrature/examples/counts.rs` (a path dependency on `nd-fmm-tables`):
    every value bit for bit FMM3D's files, the node maps the spike's to 10⁻¹⁵, the counts
    against FMM3D's. Record with the results: `docs/helmholtz/tasks/CH6.1a.md`.
- Tests: `fmm-tables/tests/helmholtz/plane_wave_quadrature.rs`: the data (rectangles,
  types, totals, the fixture domains' first and last nodes), the lookup on every rectangle
  and outside, the node maps and the λ sum within 1.5× the stated error, |J_m| against the
  trapezoid (|z| to 380) and its tails against the power series, the counts at the
  fixture domains' centres (debug), determinism; ignored: the kernel at nine points of every
  domain and precision and the chain at every centre (worst 1.08, 1.19, 1.50 and 0.98,
  0.72, 1.18 × the stated error at iprec 1–3, bounds 2 and 3), and S4 §5's reference
  levels. Debug: the new tests 10.8 s, the crate's run 41 s with its build on the M6.
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc` (one warning, a link, fixed); `cargo test` and `--workspace`;
  `cargo check --workspace --all-targets`; `cargo deny --locked check licenses`; after the
  review's fixes format, workspace Clippy, `cargo doc` (clean) and `cargo test -p
  nd-fmm-tables` again; `cargo test -p nd-fmm-tables --release --test helmholtz plane_wave
  -- --ignored` (380 s; before the review's fixes, which changed only tests, the docs and
  the guard of `bessel_j_norms` below |z| = 10⁻¹⁵⁰, outside the counts' arguments);
  the spike's Clippy and `examples/counts.rs`. Not run: the MPI list and D9 (no MPI crate,
  no Laplace path). A self-review by a subagent: nothing blocking; its points taken (the
  link, the plan's text, the sweep's results recorded, the range of `bessel_j_norms`, the
  last nodes and the Bessel tails tested).
- Decisions: none. The split of CH6.1 into a/b/c (this branch).
- Laplace: not touched (a new module of `nd-fmm-tables`; `plane_wave` and every Laplace
  path unchanged).
- Follow-ups:
  - CH6.1b: the pair chain and the convention, for sign-off (the roadmap).
  - CH6.2: whether fewer waves than the corner's would do (a sentence in CH6.2's line).

## 2026-11-03: CH6.1b The generator
- Done:
  - The second task of CH6.1 (priority list item 2), split on this branch: CH6.1b became
    the generator (this task), the pair chain with its CONVENTIONS §3.15 subsection moved
    to CH6.1c (sign-off), merge and shift renamed CH6.1d. The pair chain, the generator
    and the convention together came to about 1,800 lines in two crates.
  - `nd_fmm_ref::helmholtz::plane_wave` (f64, a builder of quadratures): `generate(κ, ε)`,
    S4's generator for any κ = k w with Re κ, Im κ ≥ 0 (the contour of real t, composite
    Gauss–Legendre, column-pivoted Gram–Schmidt with refitted weights, α and Fourier
    counts by the Bessel bound), in the spike's order of operations with its own
    Gauss–Legendre rule and Jₙ; `Quadrature` with `evaluate(u)` and `kernel_error()` (the
    largest |Q − G| on the region's grid); `kernel`; `Shortfall { error, bounds }` with
    `Bound::{Floor, Panels, Compression, Tightening}`, where the spike returned silently
    at its bounds.
  - `spikes/pw-quadrature/examples/generated.rs` (the spike gains `nd-fmm-ref`): the port
    against the spike's `generate`, every λₖ, tₖ, wₖ, Nₖ and Fₖ bit for bit at all 35 κ × ε
    of the spike's tests (14 s in release). Record: `docs/helmholtz/tasks/CH6.1b.md`.
- Tests: `fmm-ref/tests/helmholtz/plane_wave.rs`: S4's checks (ε on the grid, 2ε at 200
  seeded points off it, 2ε in the point-to-point chain, every tₖ real, no shortfall) at
  its five debug cases, and ignored at its full set (9 κ × 3 ε and 3 regression cases);
  determinism bit for bit; the κ panic. Unit tests: the panels converge; a forced panel
  cap and ε below the floor reported (`Floor`, `Panels`, `Tightening`); no shortfall when
  ε is met; Jₙ against the trapezoid; Gauss–Legendre on polynomials. `Bound::Compression`
  has no test of its own (no cheap case reaches the compression's floor). Debug on the
  M6: the helmholtz test binary 21.2 → 22.2 s, the lib's 0.01 → 24.7 s (the forced-bound
  test, ten grid sums of the generator's give-up path).
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc`; `cargo check --workspace --all-targets`; `cargo deny --locked
  check licenses`; no fftw-sys in the default members' tree; `cargo test` (744 s) and
  `--workspace` (1,019 s), all passed. After the self-review's fixes (tests and docs
  only) Clippy, `cargo doc` and `cargo test -p nd-fmm-ref` again; the ignored release tests
  (`cargo test -p nd-fmm-ref --release --test helmholtz plane_wave -- --ignored`, 23.5 s);
  the spike's Clippy and `examples/generated.rs`, both again on the final branch. Not run: the MPI list and D9 (no MPI crate,
  no Laplace path). A self-review by a subagent: nothing blocking in the code (the
  arithmetic token for token the spike's, the shortfall's bookkeeping read-only); its
  points taken (the by-hand runs recorded, two doc bounds corrected, determinism by bits
  with tₖ, the tests' measures named, the counts' sums tested, the plan's API names).
- Decisions: none. The split of CH6.1b (this branch).
- Laplace: not touched (a new module of `nd-fmm-ref`; a doc line of
  `nd_fmm_tables::helmholtz::plane_wave` for the renamed tasks).
- Follow-ups:
  - CH6.1c: the pair chain and its convention, for sign-off, tested with FMM3D's and this
    generator's quadratures (the roadmap).

## 2026-11-03: CH6.1c The pair chain
- Done:
  - The third task of CH6.1 (priority list item 2): `nd_fmm_tables::helmholtz::plane_wave::pair`,
    `PlaneWaveLevel<T>`, one level's exponential M2L in the orthonormal basis and stored form
    of CONVENTIONS §3.15, built in f64 (rounded to T) from CH6.1a's `Quadrature` or any
    `Nodes` (CH6.1b's generated ones): the factors Pₖ(n, m) = i⁻ⁿ cₙₘ 𝒫̄ₙᵐ((s/κ) iλₖ,
    (s/κ) tₖ; s²) (`harmonic_factors`, the solid normalised Legendre polynomials, finite as
    κ → 0), the steps (into a frame by Oⁿ(Q), modes, waves, shifts, sums, local, back by
    Oⁿ(Q)ᵀ) and the pair `m2l` along any of §3.14's six directions, in `HelmholtzTables`'
    units and slices (one or two multipole planes). The waves represent r_l φ̃ itself (e^β on
    each conversion). `classes::rotation_blocks` (pub(crate)): Oⁿ of one cube rotation,
    `CubeTransforms`' blocks bit for bit, so the frames need not build all 48.
  - CONVENTIONS §3.15 "Exponential M2L" (unversioned, for sign-off), and
    `tools/fixtures/check_3_15_plane_wave.py` (mpmath, 30 digits): the representation
    hₙ Yₙᵐ = 1/(iⁿ⁺¹κ) ∫∫ Yₙᵐ(ŝ) e^{iκŝ·u} up and down, the solid recursion, the plane-wave
    expansion at complex ŝ, and the stored forms, for real, complex and imaginary κ.
  - The degree study (`degree_study`, ignored; docs/helmholtz/tasks/CH6.1c.md, "Results"):
    the quadrature represents a degree-p multipole of in-box charges to its error only
    where p resolves it; below, the waves are a truncated series in λ and the quadrature
    errs on the tail, more for the finer precisions. At the FMM's degrees some precision
    keeps the chain within about 1–2.3× RCR against the direct sum (iprec 3 at p ≤ 16,
    iprec 2 at p = 20–24), but `hwts3e`'s precision from ε alone does not (ε = 10⁻⁸,
    iprec 3, p ≈ 24: 14–25×).
- Tests: `fmm-tables/tests/helmholtz/plane_wave_pair.rs`: against RCR (`HelmholtzTables::m2l`)
  by P2M of charges at the source box's corners, centre and seeded points and L2P at 27
  target points, box widths, at degrees that resolve the multipole (base 20, 36, 60 for
  iprec 1–3, plus ⌈0.9 |κ|⌉), every offset along `Direction::of` and every merge-and-shift
  direction, real k (one and two planes), complex and strongly damped k, levels 2–9 (κ from
  40 to 0.08 + 0.16i): worst 0.76× the stated error (bound 3×), generated quadratures 0.07 ε
  (bound 3ε); debug sample 11 s, release ignored 36 s. The factors against the real
  harmonics, homogeneity, shifts composing, frames round trip, the up and down waves
  against M2P, the panics.
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc`; `cargo test` (12 min 55 s) and `--workspace` (16 min 49 s),
  all passed (before the review's fixes, which changed only nd-fmm-tables' tests, a unit
  test and docs); on the final tree format, Clippy default and workspace, `cargo doc` and
  `cargo test -p nd-fmm-tables` (41 s wall, the helmholtz binary 13.1 s, of which the new
  tests 11 s); the ignored release tests (`cargo test --release -p nd-fmm-tables --test
  helmholtz plane_wave_pair -- --ignored`, 36 s, and `degree_study`); `uv run
  tools/fixtures/check_3_15_plane_wave.py`, all passed. Not run: the MPI list and D9 (no
  MPI crate, no Laplace path). A self-review by a subagent: nothing blocking in the code
  (signs, frames, layouts and units checked by hand); its points taken (panic tests for
  the slice lengths and a specific κ message, `rotation_blocks` tested against
  `CubeTransforms` bit for bit, the convention's bound sentence, the record's wording and
  the check's run recorded); left for CH6.1d: f32's range (the factors and e^β are folded
  into the tables, |Pₖ| up to about 10³⁴ at iprec 3), and the exact κ = 2k̂ comparison of
  `PlaneWaveLevel::new`.
- Decisions: three sign-offs, all 2026-11-03, "as recommended": (1) the convention, as
  written; (2) the acceptance as tested at resolving degrees; (3) CH6.1d pairs each level's
  quadrature with its degree, the coarsest precision within about 1.1× RCR's error against
  the direct sum, Rotation where none is (CH6.1d's roadmap line).
- Laplace: not touched (new Helmholtz modules of `nd-fmm-tables` and one pub(crate)
  function in `helmholtz::classes`; no Laplace file changed).
- Follow-ups:
  - CH6.1d: merge and shift, and the quadrature chosen per level with p_l and ε by
    measurement (a sentence in CH6.1d's line).

## 2026-11-03: CH6.1d Merge and shift in the FMM
- Done:
  - The fourth task of CH6.1 (priority list item 2), split on this branch into merge and
    shift in the FMM (CH6.1d) and the quadrature per level by measurement (CH6.1e), about
    2,000 lines in three crates together.
  - `nd_fmm_tables::helmholtz::plane_wave::merge`: `MergeLevel` (a level's `PlaneWaveLevel`
    with the merge's source shifts slot → parent, 104 tables shared by (rotated offset, up
    or down), and child shifts parent → child, in §3.14's geometry), `PlaneWaveM2l`, `shift`
    and `add`; `precision_for_degree` and `select`, each level's precision from its degree
    where κ_l lies in the tables, provisional from CH6.1c's degree study (iprec 1 to p = 8,
    2 to 12, 3 to 16, Rotation above). `PlaneWaveLevel::bytes`.
  - `nd-fmm-exec`: `HelmholtzM2l::PlaneWave` (opt-in), `PlaneWavePrecision` (`Rule` or
    `PerLevel`), `with_plane_wave_precision`, `plane_wave_precision(level)`,
    `plane_wave_bytes`, `with_plane_wave_boxes`, `HelmholtzFmmBuilder::plane_wave_precision`;
    the level call `src/helmholtz/operator_plane_wave.rs`, Laplace's order in scalar loops on
    the driver's `Execution` (no Laplace file changed); the pair body CH6.1c's translation
    along `PlaneWaveTables::pair_direction`.
- Tests: `fmm-tables/tests/helmholtz/plane_wave_merge.rs` (the shifts' factorisation on
  every direction, slot and child, 7.5e-14 relative against 1e-12; the shared tables bit
  for bit; the rule; `build`); `fmm-exec/tests/helmholtz/plane_wave.rs` (the precisions by
  the rule and named; the pair body bit for bit `PlaneWaveLevel::m2l`, one and two planes,
  two vectors); `check_plane_wave` in `tests/helmholtz_fmm.rs` on two scenarios (one and
  two multipole planes): the direct sum (1.000× and 1.013× Rotation's error, printed), the
  pair path within 1e-14 (2.2e-16 in release on every level; in debug builds of a tree
  with waves on several levels the finest level's alone), 2 threads and the one-rank FMM
  bit for bit, two vectors, named precisions (none on every level Rotation bit for bit).
  Debug cost about 27 s of the one-rank `helmholtz_fmm` run on the M6.
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc`; `cargo test` (19 min 16 s) and `--workspace` (23 min 57 s), all
  passed; the MPI list at 2 and 4 ranks (`run-mpi-tests.sh`, every executable passed;
  `helmholtz_fmm` 276 s and 249 s, `mpi_exec` 350 s and 430 s); `helmholtz_fmm` in release
  at one rank (21.7 s). After the self-review's fixes (docs, `plane_wave_bytes`'s pair
  scratch): Clippy and `cargo doc` of both crates, the MPI list (built after them) and
  `tests/helmholtz` `plane_wave` again. Not run: D9 (no Laplace path changed). A self-review
  by a subagent: nothing blocking (offsets, up and down halves, the accumulator order, the
  layouts and the level k̂ checked against §3.14 and Laplace's level call); its points taken
  (the pair scratch in `plane_wave_bytes`, docs of `m2l_on`, the panics and f64, the
  plan's file name, the release run recorded).
- Decisions: none. The split of CH6.1d into d/e (this branch).
- Laplace: not touched (new Helmholtz modules of `nd-fmm-tables` and `nd-fmm-exec`;
  Laplace's plane-wave geometry and the driver's `Execution` used, not changed).
- Follow-ups:
  - CH6.1e: the precision rule by the FMM's errors on CH3.1c's cases, CH6.1's 1.1×
    Rotation asserted (the roadmap).
  - f32 under `PlaneWave` is untested and not refused (factors to about 10³⁴): D3 and
    CH7.1 (§3.15 "Precision and range", f64 only until then).


## 2026-11-03: CH6.1e The quadrature per level, measured
- Done:
  - The fifth and last task of CH6.1 (priority list item 2): the precision rule of
    `HelmholtzM2l::PlaneWave` by the FMM's own errors. `nd_fmm_tables::helmholtz::plane_wave::merge`:
    `precision_for_tolerance` (iprec 2 for ε ≥ `PLANE_WAVE_MIN_TOLERANCE` = 10⁻⁶, none
    below), `precision_for_degree` (named degrees: iprec 2 to `MAX_PLANE_WAVE_DEGREE` = 15,
    the calibrated plateau at 10⁻⁶), `select(k̂₀, degrees, tolerance)`, `RULE_PRECISION`; the
    provisional degree rule of CH6.1d replaced. `nd-fmm-exec`:
    `HelmholtzOperator::with_plane_wave_tolerance`, which `HelmholtzFmmBuilder` sets from
    `Degrees::Tolerance`.
  - `nd-fmm-validate`: `helmholtz_accuracy::{Case::plane_wave, Row::plane_wave}`, the
    example's `--m2l planewave|planewave:<q>|planewave-scan` and its "÷ Rotation" column;
    `tools/helmholtz/ch6-1e-runs.sh`; the report `fmm-validate/results/helmholtz-ch6-1e.md`.
  - The scan (the M6): each level with a quadrature at iprec 1, 2 and 3 alone on CH3.1c's
    matrix and at 10⁻⁴, 10⁻⁵, 10⁻⁷. The precision follows ε, not p or κ_l: iprec 1 misses
    1.1× Rotation's from 10⁻³ (1.145×), iprec 2 holds from 10⁻³ to 10⁻⁶ (worst 1.044×) and
    misses from 10⁻⁷, iprec 3 misses on the clustered trees' deep levels (1.116× at 10⁻⁷,
    1.76× at 10⁻⁸, κ_l ≤ 0.08). The rule within 1.070× on 87 cases; every level at
    ε < 10⁻⁶ left to Rotation.
- Tests: the rule in `fmm-tables/tests/helmholtz/plane_wave_merge.rs` and
  `fmm-exec/tests/helmholtz/plane_wave.rs` (by tolerance, by degree, the plateau);
  `check_plane_wave` in `fmm-exec/tests/helmholtz_fmm.rs` now asserts 1.1× Rotation's
  (`PLANE_WAVE_RATIO`; 1.000× on both scenarios); `check_plane_wave` in
  `fmm-validate/tests/helmholtz_accuracy.rs` on W1 at kD = 10 and W4's clusters. Debug cost
  (a loaded M6): the harness test 36 s (26 s before), `check_plane_wave` 47 s of
  `helmholtz_fmm` (27 s).
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc`; `cargo test` (705 s) and `--workspace` (1,094 s), all passed;
  the MPI list at 2 and 4 ranks (`run-mpi-tests.sh`, every executable passed;
  `helmholtz_fmm` 176 s and 153 s); `tests/helmholtz_accuracy.rs` in release. One-rank
  debug times: `helmholtz_fmm` 188 s (about 177 s after CH6.1d), `helmholtz_accuracy` 31 s
  (26 s), `fmm-exec`'s `tests/helmholtz` 30 s. Not run: D9 (no Laplace path changed). A
  first attempt was stopped by Claude Code when the M6 ran out of memory (the large scan's
  22.9 GB point beside a workspace test run); the checks ran again alone. A self-review by
  a subagent: one blocking defect fixed (the harness test asserted waves at 10⁻⁸, where the
  rule takes none), numbers in docs and the report corrected, a timeout around the
  script's `mpirun`.
- Decisions: the PR's four questions, signed off 2026-11-03 (the user: "1–4 as recommended"): (1)
  the acceptance as read (the two large cases at 10⁻⁸ left out, Rotation there; asserted on
  the tests' small scenarios, the matrix in the report); (2) Rotation below 10⁻⁶, no κ
  guard for iprec 3; (3) CH6.1f before H6.G, after CH6.2 (priority list item 4); (4) the
  waves buffer bounded and counted in the node budget before CH6.2's times.
- Laplace: not touched (Helmholtz modules of `nd-fmm-tables` and `nd-fmm-exec`, and
  `nd-fmm-validate`; no Laplace file changed).
- Follow-ups:
  - CH6.1f: quadratures for ε < 10⁻⁶ matched to the degree (a new roadmap task before
    H6.G).
  - CH6.2: the times of the rule's iprec 2 against Rotation and the table strategies (the
    report's §7: 0.25–0.65 of Rotation's evaluation at 10⁻⁶ on a loaded M6), and the waves
    buffer bounded and counted in the node budget first (9.0–11.8 GB under the rule on the
    large cases; a sentence in CH6.2's line).
  - The scan's level 6 of W3 at kD = 100 at iprec 3 (22.9 GB of waves) swapped and the run
    after it ran the M6 out of memory: the large scan stopped there, W1 at kD = 500 not
    scanned (the report's §7, §8).

## 2026-11-03: CH6.2a The waves buffer bounded
- Done:
  - CH6.2 (priority list item 3) split on this branch into the bounded waves buffer (CH6.2a)
    and the measurements (CH6.2b), CH6.1e's sign-off 4 putting the bound before any time.
  - `nd-fmm-exec`: the exponential M2L's level call by chunks of whole parents
    (`operator_plane_wave.rs`, `PlaneBuffer`): per chunk the distinct V sources of its
    computed children, listed with a position per box, their waves on the pool, then merge
    and shift over the chunk's parents; the output the same at every bound.
    `HelmholtzOperator::{with_plane_wave_buffer, plane_wave_buffer,
    plane_wave_buffer_bytes, plane_wave_counts}`, `PlaneWaveCounts`, `MIN_BUFFER_BOXES`
    (216); `HelmholtzFmmBuilder::plane_wave_budget` per node,
    `DEFAULT_PLANE_WAVE_NODE_BUDGET` (8 GiB), divided by the ranks per node that `build`
    now agrees (one `split_shared`, one all-reduce, every rank), `HelmholtzFmm::ranks_per_node`.
  - `nd-fmm-validate`: `Case::plane_wave_budget`, `Row::{plane_wave_buffer,
    plane_wave_counts}`, the example's `--plane-wave-budget MB|whole` and rank 0's peak
    resident size; `tools/helmholtz/ch6-2a-runs.sh`; the report
    `fmm-validate/results/helmholtz-ch6-2a.md`.
  - Kathleen (job 628518, one node): W1 at kD = 500 and W3 at kD = 100 at 10⁻⁶, one checksum
    per case at every bound from the floor to the whole level, at 1 × 40 and 40 × 1. At one
    rank the whole level's waves 8.9 and 11.7 GB (peak 14.6 and 17.1 GB); 4 GiB 1.03× and
    1.04× its time (peak 10.0 and 9.7 GB), 1 GiB 1.05× and 1.11×, 256 MiB 1.34× and 1.43×,
    the floor 13× and 28×. At 40 ranks per node (job 628529, with repeats) the default
    (107 MB per rank) 1.16× and 1.28×, 8 GiB 1.04× and 1.11×, 16 GiB 1.01× and 1.06×.
- Tests: `check_plane_wave_buffer` in `fmm-exec/tests/helmholtz_fmm.rs` (the floor, twice
  the floor and in release no bound bit for bit the default; the floor at 2 threads and
  against the one-rank FMM; the counts agreed over the ranks; on one rank the floor chunked,
  8 chunks against 4 on the clusters); `the_waves_buffer_is_bounded` in
  `fmm-exec/tests/helmholtz/plane_wave.rs` (the sizes and `plane_wave_bytes` at eight
  bounds). One-rank debug `helmholtz_fmm` 195 s on the M6 (188 s before).
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc`; `cargo test --workspace` (22 min 52 s, before the review's test
  changes) and `cargo test` (20 min 28 s, after them), all passed; the MPI list at 2 and 4
  ranks (`run-mpi-tests.sh`, every executable passed; `helmholtz_fmm` 187 s and 165 s);
  `helmholtz_fmm` and `helmholtz_accuracy` in release. Kathleen jobs 628518 and 628529
  (above; the report's §2–§4). Not run:
  D9 (no Laplace path changed). A self-review by a subagent: no code defect (the chunking,
  the rollback, the offsets and the collectives checked); its points taken (a test that the
  floor chunks, a real middle bound, the counts agreed over the ranks, the byte formula for
  any bound, stale docs, the plan's run text), and the debug run trimmed after the new
  checks took it to 339 s.
- Decisions: signed off 2026-11-04, both as recommended: the default waves budget 8 GiB
  per node (the runs took 4 GiB, the default before; at 40 ranks per node 8 GiB costs
  4–11 % against 16–28 %); the bound not a hard cap (the floor and the positions outside
  it), accepted as documented.
- Laplace: not touched (Helmholtz files of `nd-fmm-exec`, `nd-fmm-validate`; the driver's
  `Execution` used, not changed).
- Follow-ups:
  - CH6.2b: the times at the default budget and the ranks × threads split (a sentence in
    CH6.2b's line).
  - The bound is not a hard cap (the floor of 216 boxes per level, the positions outside
    it), and a chunk's halo is recomputed; a cut by spatial blocks rather than Morton runs
    could lower it if CH6.2b finds the cost matters (the report's §3).


## 2026-11-04: CH6.2b The M2L timed per level
- Done:
  - CH6.2b (priority list item 3) split on this branch into our strategies timed per level
    (CH6.2b) and FMM3D with D8 (CH6.2c), as its line allowed.
  - `nd-fmm-exec`: `HelmholtzOperator::m2l_times`, every M2L level call timed and summed per
    level (two clock reads per call, no value changed; reset with the counts when the
    tables are chosen).
  - `nd-fmm-validate`: `Row::{m2l_seconds, v_pairs, first_evaluate}`, `Case::evaluations`,
    the example's `--levels` (a table per run and level: degree, the method that ran, V
    pairs, seconds, ns per pair) and `--evaluations n`; `tools/helmholtz/ch6-2b-runs.sh`
    (ABBA, the macOS gate), `tools/helmholtz/ch6-2b-tables.py`; the report
    `fmm-validate/results/helmholtz-ch6-2b.md`.
  - The M6 (1 × 6) and locust (1 × 72; the large cases also 8 × 9, 72 × 1): at 10⁻⁶ the
    exponential M2L the fastest method on almost every level below the coarsest, 0.19–0.40×
    Rotation's M2L on the M6's cases of 8 × 10⁴ points and more (evaluation 0.27–0.52×
    Auto's), 0.45× and 0.21× on W1 at kD = 500 and W3 at kD = 100 on locust; at 10⁻³ no
    single strategy best (a per-level choice up to 1.32× faster); level 2 9–24× Rotation's
    time at 72 threads (parents split the level call); Batched above Auto's crossover 2–10×
    Rotation's.
- Tests: `check_m2l_times` in `fmm-exec/tests/helmholtz_fmm.rs` (in `check_auto` and
  `check_plane_wave`: an entry per level, none on level 0, a time on every level with V
  pairs, on every rank or none); `check_levels` and two evaluations of one build in
  `fmm-validate/tests/helmholtz_accuracy.rs` (the V pairs against one rank). One-rank debug
  on the M6: `helmholtz_fmm` 196–201 s (195 s before), `helmholtz_accuracy` 41 s (36 s;
  the extra evaluation run); 279–301 s and 64 s on a loaded M6 (load average 7).
- Checks: ran `/verify` on the M6: format; Clippy default, examples and workspace
  all-targets; `cargo doc`; `cargo test` (21 min, passed); `cargo test --workspace` was
  terminated (exit 143) inside the spikes after every default member had passed, so the
  non-default members (nd-fmm-kernels and the spikes) ran on their own and passed; the
  MPI list at 2 and 4 ranks (`run-mpi-tests.sh`, every executable passed; `helmholtz_fmm`
  185 s and 162 s); `helmholtz_accuracy` in release at 1 and 2 ranks. Kathleen's runs
  (job 628611, `matrix` and `large` at 1 × 40, completed, outputs not fetched; `large` at
  4 × 10 and 40 × 1 not submitted) moved to CH6.2c: the login node refused every
  connection from 2026-11-04 10:10. Not run: D9 (no Laplace path). A self-review by a subagent: one
  blocking defect fixed (the macOS gate counted the launch: the M6's run made again), its
  points taken (the per-level seconds' digits, level 0 asserted, the doc paragraph, the
  tables' parsing, the plan as built).
- Decisions: none. The acceptance amended on 2026-11-04 (the user, "yes, go ahead with
  that change"): done on the M6 and locust, Kathleen's runs moved to CH6.2c. dps31 (a
  shared Threadripper 3970X, AVX2 only, loaded at the time) was checked as a substitute
  and not used: it is not Kathleen's AVX-512 class.
- Laplace: not touched (the Helmholtz operator, `nd-fmm-validate`'s Helmholtz harness,
  tools).
- Follow-ups:
  - CH6.2c: the coarse levels' level call at many threads (split below the parent, or a
    rule by the level's boxes) and the per-level choice at 10⁻³, both for D8 (sentences in
    CH6.2c's line).
  - Kathleen's runs (fetch job 628611, `large` at 4 × 10 and 40 × 1, the report's §5):
    a sentence in CH6.2c's line.


## 2026-11-05: CH7.5a The plane-wave kernels
- Done:
  - CH7.5a (priority list item 1) split on this branch into the kernels (CH7.5a), the
    device's `PlaneWave` behind an opt-in setting (CH7.5b) and the measurements with the
    adoption (CH7.5c); Helmholtz's device exponential M2L renamed CH7.5d (the user, 2026-11-05:
    "a/b/c, Helmholtz → d"; constraint 1: "Go ahead").
  - `nd-fmm-kernels`, module `plane_wave`: Laplace's merge and shift (CONVENTIONS §3.14) as
    four launches per level call: `outgoing` (per source and frame the multipole rotated,
    its modes and its up and down waves), `merge` (per accumulator Σ shift × wave over its
    slots in order), `gather` (per incoming block its accumulators summed in order, times
    its child shift), `local` (per target the three frames in order: the sums, the local in
    the frame, rotated back and added). `PlaneWaveTables` (the tables of
    `nd_fmm_tables::PlaneWaveTables` from plain arrays), `PlaneWaveLists`/`PlaneWavePlan`
    (a level call's lists, validated on the host: CSR offsets, every index, distinct local
    columns, the tables' shifts, the 32-bit index range), `PlaneWaveScratch`,
    `PlaneWaveLayout` (`Cube { units }`, 128 units by default on Metal and CUDA; `Cpu`).
    Every chain is the order of the host function it restates, every multiply–add an
    explicit fma; no atomics.
  - `nd-fmm-tables`: read-only accessors `outgoing_powers`, `incoming_powers`,
    `multipole_rotation`, `local_rotation`, `child_shifts` of `PlaneWaveTables`.
- Tests: `fmm-kernels/tests/kernels/translate/plane_wave.rs`: on the full and an adaptive
  region (every seventh box absent, every fifth target without a V row) at p = 3 and 8 in
  f32 and f64, every layout, every step bit for bit a host `mul_add` replica (the
  accumulators and incoming blocks also `shift_add`, `add` and `shift` of the device's
  own buffers), repeated calls bit-identical, four launches; the waves within 11.5 u_T
  (bound 128) and the locals within 18.7 u_T (bound 1024; 110.5 at f64 p = 18) of the
  tables' functions, relative to each block's or column's largest value; the locals
  against Σ `PlaneWaveTables::m2l` per pair within twice the host level call's own
  distance plus 64 u_T (the host's 15–97 u_T at p ≤ 12, 1995 u_T at f64 p = 18: merge and
  shift equals the pair path to rounding that grows with p); one parent's 216 slots into
  its children on levels 2, 9 and 16 within 0.007 of the quadrature's bound E Σ|q| of
  `direct::m2l` at p = 3 and 8 (every p ≤ 20 at most 0.113 in the ignored sweep); the
  refusals (a cube of no unit, a plan of other tables). `arrays_match_the_steps` in
  `fmm-tables/src/plane_wave/tests.rs`. CPU runtime: 3 tests in about 1.3 s, the ignored
  sweep 16 s (release, the M6).
- Checks: ran `/verify` on the M6: format; Clippy default, examples, workspace all-targets
  and `nd-fmm-kernels` with `cpu,metal`; `cargo check` with `cuda`; `cargo doc`; the
  kernels' tests without a feature and on the CPU runtime with the ignored ones;
  `cargo test` (15 min) and `cargo test --workspace` (20.5 min), all passed; the MPI list
  at 2 and 4 ranks (`run-mpi-tests.sh`, every executable passed). Metal (the M6, outside
  the sandbox): the 66 ignored kernel tests passed. locust (H100): Clippy with `cpu,cuda`
  and the 166 kernel tests with `--include-ignored` (77 CUDA) passed, the plane-wave tests
  again after the review's fixes; the CPU runtime's default plane-wave tests 2.6 s on
  Grace. Not run: D9 (no Laplace path changed). A self-review by a subagent: one blocking
  point fixed (a plan validated against tables with more source shifts than a launch's
  would read out of bounds: now refused, tested), its other points taken (32-bit index
  checks, every buffer compared on repeats, f32 at p = 12, docs).
- Decisions: none. The proposal amended at CH7.5a (signed off 2026-11-05 on PR #187,
  "Please do as recommended"): §10's row CH7.5 split into CH7.5a–c (Laplace's plane-wave
  M2L on the device) and CH7.5d (the exponential M2L, the former CH7.5),
  §6.5's device list and the amendment log updated.
- Laplace: not touched (a new module of `nd-fmm-kernels` that no path calls yet;
  additive accessors in `nd-fmm-tables`, no table or step changed).
- Follow-ups:
  - CH7.5b: the lists in the order of CH7.5a's test, several vectors as columns, chunks
    within the kernels' 32-bit index range (a sentence in CH7.5b's line).
  - CH7.5c: the kernels' layouts measured (`GPU_UNITS` = 128 a first choice) and the
    merge's irregular lists (a sentence in CH7.5c's line).

## 2026-11-05: CH7.5b The device's `PlaneWave`
- Done:
  - `nd-fmm-exec` (feature `gpu`): `fmm::DevicePlaneWave` (`Dense`, the default: "PlaneWave,
    run as dense on the device", `Dense`'s bits; `Kernels`: merge and shift on the device),
    `FmmBuilder::{device_plane_wave, device_plane_wave_budget}`,
    `DeviceOptions::{plane_wave, plane_wave_budget}`. Under `PlaneWave` with `Kernels` and
    M2L on the device: the host's `PlaneWaveTables` uploaded once; per level the lists of
    the host's order from the V view (parents in index order, the slots from the computed
    children's rows, every accumulator over its slots ascending, each computed child's
    accumulators in order); chunks of whole parents within the budget (default the
    translation scratch budget) and the kernels' 32-bit index range; the m vectors as
    columns; four launches per chunk; one scratch for the widest chunk; all in the memory
    check. `DeviceReport::{plane_wave, plane_wave_layout, plane_waves,
    plane_wave_scratch_bytes}`, `PlaneWaveReport`; module docs "Plane-wave M2L on the
    device".
  - `nd-fmm-kernels` (`plane_wave` only): `PlaneWaveScratch::covering` (one scratch for
    several plans; a launch checks the scratch holds at least its plan's buffers),
    `PlaneWaveScratch::bytes_of`, `PlaneWaveTables::bytes_of`, `PlaneWaveLists::bytes`.
- Tests: `device_common::check_plane_waves` (without the setting `Dense`'s bits, also with
  it under `Dense`; with it every kind on the device, the report, the output and every
  level's expansions within the FMM bounds of the host's `PlaneWave`, three evaluations,
  a second build and one parent per chunk bit for bit, the transfer formula with four
  launches per chunk, three vectors), run by `tests/mpi_exec.rs`'s `check_plane_wave`
  (CPU runtime) and by `tests/device_metal.rs`, `tests/device_cuda.rs` and
  `tests/device_ranks.rs`; in `check_plane_wave` on several ranks M2L alone on the device
  against the one-rank `Fmm` bit for bit; the plane-wave gate of `tests/device_fmm.rs`
  (within 1.1× the device `Dense`'s φ and ∇φ L2 errors on the cube and the Plummer sphere
  at N = 10⁵: measured 0.941–1.011 on the CPU runtime f64 p = 8, 12, 18, Metal f32 p = 8,
  CUDA f32 p = 8 and f64 p = 8, 12, 18); the kernels' `check_covering`. The tests were
  written after the code in this task, not before.
- Checks: ran `/verify` on the M6 (results in the PR); the device runs of the plan's
  "Runs" on the M6 (CPU runtime at 1, 2 and 4 ranks, Metal) and locust (CUDA), all passed.
  Not run: D9 (CH7.5c's: no default and no other strategy changes).
- Decisions: none. Question for sign-off (the PR): 1. the roadmap's "the CubeCL CPU
  runtime in CI" for `nd-fmm-exec`, which no CI job builds with `cpu` (its device checks
  ran by hand; the kernels' run in CI). Recommended: accept as is. Signed off 2026-11-05,
  the user: "sign off the question as recommended".
- Laplace: no default changed; without `DevicePlaneWave::Kernels` every device build keeps
  its tables, launches and bits (`check_plane_waves`: `PlaneWave` and `Dense` with and
  without the setting bit for bit `Dense`; every existing device check passed on the CPU
  runtime, Metal and CUDA); the host path untouched.
- Follow-ups:
  - CH7.5c: the chunk budget measured with the layouts (a sentence in CH7.5c's line).

## 2026-11-05: CH7.5c Measured, and adopted or dropped (adopted 2026-11-06)
- Done:
  - `nd-fmm-exec` (feature `gpu`): `fmm::DevicePlaneWaveLayout` (`Auto`, by backend, as
    before; `Cube { units }`) and `FmmBuilder::device_plane_wave_layout`, resolved in
    `DeviceOptions::plane_wave_layout` and checked at build by `PlaneWaveLayout::check`;
    every layout gives the same bits.
  - `nd-fmm-validate`'s `layout_sweep`: the part `planewave` (Dense against merge and shift
    at each cube of 32–1,024 units and each chunk budget of 128 MB–4 GB, M2L by
    `KindTiming` and the evaluation; per level with each level call's chunks and items) and
    `--dist cube|plummer`.
  - The report `fmm-validate/results/helmholtz-ch7-5c.md`: locust's H100 (the cube and the
    Plummer sphere at N = 10⁶ and 10⁷, f64 p = 8, 12, 18 and f32 p = 8; the units by p at
    f64 p = 10–20) and the M6's Metal (f32 p = 8, both problems and sizes). Merge and shift
    takes 0.30–0.38× the dense M2L's time at p = 8, 0.19–0.20× at f64 p = 12 and
    0.19–0.21× at f64 p = 18 with 256-unit cubes on the H100 (the evaluation 0.79–0.90×,
    0.52–0.68×, 0.36–0.45×), 0.47–0.60× on Metal (the evaluation 0.65–0.69×). T13's gate (the
    gain at p ≥ 12) is met.
  - The adoption (2026-11-06, after the sign-off): `DevicePlaneWave::Kernels` the default
    of `PlaneWave` on every device (`Dense` the named opt-out);
    `DevicePlaneWaveLayout::resolve(info, p)` with 256-unit cubes on CUDA from p = 13
    (`device::CUDA_PLANE_WAVE_WIDE_{UNITS,FROM}`), 128 otherwise, the CPU layout on the CPU
    runtime; `device::default_plane_wave_budget`, 1 GB on Metal (`METAL_PLANE_WAVE_BUDGET`,
    at most an eighth of the device's memory), the translation budget elsewhere.
- Tests: `device_common::check_plane_waves` builds named cube layouts (one unit on the CPU
  runtime; 32 and 256 units on the GPUs), each the default's bits with the transfer
  formula, and refuses a cube of no unit on every rank; the smoke run of
  `layout_sweep --part planewave` on the CPU runtime by hand (the example has no test).
  After the adoption: `check_plane_waves` turned round (the default runs merge and shift,
  bit for bit `Kernels` named; `DevicePlaneWave::Dense` gives `Dense`'s bits; either setting
  under `Dense` changes nothing), `mpi_exec`'s run-as-dense check names `Dense`, the
  `device_fmm` gate runs the default; the unit test
  `plane_wave_defaults_follow_the_backend_and_p`.
- Checks: ran `/verify` on the M6 (format; Clippy default, examples, workspace all-targets,
  `nd-fmm-exec` and `nd-fmm-validate` with `cpu,metal`; `cargo check` with `cuda`; `cargo
  doc`, also `nd-fmm-exec` with `cpu`; `cargo test` 21 min and `cargo test --workspace`
  50 min; the MPI list at 2 and 4 ranks; `mpi_exec` on the CPU runtime at 1, 2 and 4 ranks
  (21 and 24 device plane-wave checks) and the ignored `device_ranks` at 2 and 4 ranks), all
  passed; `mpi_exec` again at 1 and 2 ranks after the review's over-cap refusal. After the
  adoption: `cargo test -p nd-fmm-exec --features cpu --release` (every executable;
  `device_tune` failed in the first run and passed in the second, and fails the same way
  on `main` b69e031: the CPU runtime's tuner on the M6 sometimes picks Rotation, and the
  wide-candidate check assumes Dense's M2L GEMM, a flaky test outside this task), `mpi_exec`
  and `device_ranks` at 2 and 4 ranks (24 device plane-wave checks), the MPI list at 2 and
  4 ranks; on the M6's Metal `device_metal` and the `device_fmm` gate (φ L2 1.011× and
  0.999× `Dense`'s); on locust `device_cuda` (the 32- and 256-unit cubes on the GPU), the
  gate (every L2 ratio ≤ 1.0114) and Clippy with `cpu,cuda`; Clippy (default, examples,
  workspace, `cpu,metal`), `cargo doc` and `cargo test --workspace` (27 min); all passed. D9 on locust
  (`nd-fmm-bench --backend cuda`, f32 and f64, p = 8, N = 10⁶, `Auto`; A `main` b69e031,
  B the branch, A B B A A B B A A B): f32 −0.8 % (`main`'s spread 0.27 ms), f64 −6.0 %
  (spread 4.16 ms), the errors identical, both pass. A self-review by a
  subagent: nothing blocking; its points taken (the report's budget range and claims, the
  near rule's share of the evaluation column, the over-cap refusal, docs).
- Decisions: the PR's three questions, signed off 2026-11-06 ("sign off all three questions
  as recommended"): 1. adopt merge and shift as the default of `PlaneWave` on every device,
  with cubes of 256 units from p = 13 on CUDA and a 1 GB plane-wave budget on Metal; 2. a
  task for the device `Auto` (CH7.5e); 3. D9 by the device tests and one CUDA ABBA run.
- Laplace: a named `PlaneWave` on a device now runs merge and shift (within 1.1× `Dense`'s
  errors, the gate); `DevicePlaneWave::Dense` keeps `Dense`'s bits. The device `Auto` never
  resolves to `PlaneWave`, so every unnamed device evaluation and the host keep their code
  and bits (D9 above).
- Follow-ups: CH7.5e (the device `Auto` on `PlaneWave`, by backend, precision, p and N: at
  N = 10⁵ the gate's printed times put merge and shift behind `Dense` on the cube); the
  flaky `device_tune` wide-candidate check, a separate fix.

## 2026-11-06: CH0.6 S6: stability of the diagonal form
- Done: the spike `spikes/diagonal` (`nd-fmm-spike-diagonal`, a workspace member, not a
  default one): the signature grid of §5.4 in double-double, S → F and F → R per ring, the
  transfer functions in double-double (rounded per sample, built in f64 as the seed does,
  interpolated in ŝ · t̂), the 34 classes and their sample permutations, the bound's Hankel
  term, and the sweep against S7's double-double RCR and direct sum on S7's placements
  (612 units plus §3.6's four levels, 188.7 core-hours on locust); an additive
  `impl Real for f32` in S7's spike. The report `spikes/diagonal/SPIKE_REPORT.md`, results
  in `spikes/diagonal/results/` (`summary.md`, `times.md`, `sweep.log`); the raw rows
  (68 MB) archived in `~/projects/fmm-archive/locust/runs/ch0-6/`, not committed. Findings:
  C_T = 0.153 and θ = 1/2 with a floor of 176 u for T rounded from double-double, once the
  bound carries e^{Im(k)(|t| + 2a)} (without it the damped cases exceed it by up to 10⁴⁴;
  with e^{Im(k)|t|} by 1.6·10⁵ at the W1 patches); T built in f64 exceeds it by 47×,
  interpolated T needs C_T = 0.23; the reading of Gumerov and Duraiswami confirmed in 3D
  (0.3–0.5× the 1-D seed's errors with its model of T); §3.6's kD = 500 table redone (none,
  4, 4, 4–6 at 10⁻⁶ with the formula's p; none on levels 2–5 with the calibrated p); f32
  meets 10⁻³ on every class for real k from k a = 8 and 10⁻⁴ from k a = 45; RCR per pair
  5–53× the diagonal pair at p = 24–256 (prototypes, one Grace core).
- Tests: `cargo test -p nd-fmm-spike-diagonal --release -- --include-ignored`, 18 tests:
  the plan's 1–11 (the chain in double-double equals S7's RCR to 10⁻²⁰, real k and η = 0.1
  to k a 16, not the plan's 32; the seed's geometry; determinism at 1 and 3 threads).
- Checks: ran `/verify` on the M6 at 01ef39d: format, Clippy (default, examples, workspace
  all-targets), `cargo doc`, `cargo deny --locked check licenses`, `cargo check --workspace
  --all-targets`, no `fftw-sys` in the default graph, `cargo test` and `cargo test
  --workspace`, the spike tests of `nd-fmm-spike-diagonal` and `nd-fmm-spike-truncation` in
  release with the ignored ones; all passed. After it only the summary example changed:
  format and `cargo clippy -p nd-fmm-spike-diagonal --all-targets -- -D warnings` again,
  passed. The MPI list not run (no MPI crate changed). A self-review by a subagent: nothing
  blocking in the code or the scope; its corrections taken (the damping factor's need at the
  corners as well, 610× without the 2a; the acceptance stated with the corrected bound; the
  reading of Gumerov and Duraiswami stated where tested; the bandwidth and seed numbers;
  pruned cells and the f32 counts in `summary.rs`, the tables' Markdown), the summary
  regenerated byte for byte from the archived rows.
- Decisions: questions for sign-off in the report's §9: 1. D3, f32 on high-frequency levels
  at ε ≥ 10⁻⁴ by the per-class bound; 2. the bound of §3.6 (C_T 0.15, C₀ 200 u, θ 1/2, the
  damping factor), a correction of §3.6; 3. T from extended precision or interpolated; 4.
  §3.5's p_high constant against the 2007 report's eq. (25); 5. CONVENTIONS §3.15
  "Signatures" as proposed, written in CH1.3d; 6. the PR's size; 7. the damped bound's
  conservatism. Signed off 2026-11-06, the user: "i agree with the recommendations" (all
  seven as recommended): D3 recorded; the proposal's §3.5, §3.6, §5.4, §5.6, §1 and §6.5
  amended (Appendix A, "Amended at CH0.6").
- Laplace: not touched (a spike crate, a members entry, `Cargo.lock`, an additive impl in
  S7's spike); D9 does not apply.
- Follow-ups: CH1.3d writes CONVENTIONS §3.15 "Signatures" (question 5); CH4.3 measures the
  damped classes in the FMM-weighted norm and looks at the shorter transfer function (the
  roadmap's CH4.3 line); CH0.8b recounts §7 with S6's constants (its line already says so).

## 2026-11-06: CH6.2c CH6.2b's Kathleen runs
- Done:
  - The split (`/next-task`'s size rule, Kathleen back): the former CH6.2c (FMM3D with D8)
    into CH6.2c (CH6.2b's Kathleen runs), CH6.2d (beside FMM3D's exponential M2L, the
    corner ρ = 4√2) and CH6.2e (D8, the coarse levels' remedy); D8 now prepared by CH6.2e,
    the priority list's item 4 names all five.
  - Kathleen's runs: job 628611's outputs fetched (`matrix` and `large` at 1 × 40,
    2026-11-04) and job 629953 run (`large` at 4 × 10 and 40 × 1). CH6.2b's report
    (`fmm-validate/results/helmholtz-ch6-2b.md`) has Kathleen's summary rows (§2), a splits
    table (§4), §5 written, the observations for D8 checked (§7, retitled for CH6.2e) and
    Kathleen's per-level tables in the appendix. Findings: at 10⁻⁶ the exponential M2L
    0.24–0.61× Rotation's M2L on W1 at kD = 500, W3 and the W4 cases at kD = 1 (W1 at
    kD = 500 0.50×, W3 at kD = 100 0.24×; the evaluation 0.31–0.65× Auto's), 0.73–0.92×
    on the other W4 cases at kD = 100; level 2 4.4–25× Rotation's time at 1 × 40 (on W3 at
    kD = 100 5.4–8.1× at 1 × 40, 5.5–7.9× at 40 × 1, 1.3–1.8× at 4 × 10); Auto's
    crossover 15 on AVX-512 runs Batched on degree-15 levels
    (33× Rotation's on W1 at kD = 10's one level; on levels of many parents the exponential
    M2L is faster); at 10⁻³ a per-level choice up to 2.08× the best single strategy; 1 × 40
    the fastest split on W1 at kD = 500, within 10 % of 4 × 10 on W3 at kD = 100, 40 × 1
    the slowest for the exponential M2L.
  - Raw outputs in `~/projects/fmm-archive/kathleen/logs/ch6-2b/` (not committed).
- Tests: none (no code). Every evaluation's checksum equal in both launches of all 152 ABBA
  pairs, and each large case's the same at 1 × 40, 4 × 10 and 40 × 1 across the two
  sources (1e22995, e6b29c0); the exponential M2L's errors 0.92–1.07× Rotation's.
- Checks: no Rust, manifest or tool change, so `/verify`'s cargo checks do not apply (none
  run); the tables regenerated with `tools/helmholtz/ch6-2b-tables.py`; Kathleen jobs
  628611 and 629953, every launch exit 0. A self-review by a subagent (the tables
  regenerated byte for byte, checksums, peaks, jobs and sources rechecked): its points
  taken (2.08× and 1.62×, not 2.10× and 1.64×; the p ≥ 15 rule and its p = 13 exception;
  the ranges over all 34 levels of degree 15–16; level 2 compared on W3 alone across the
  splits, not by threads).
- Decisions: none.
- Laplace: not touched.
- Follow-ups: CH6.2e's line: Kathleen's level 2 on W3 at kD = 100 (1.3–1.8× Rotation's at
  4 × 10 against 5.4–8.1× at 1 × 40 and 5.5–7.9× at 40 × 1) for the coarse levels' remedy,
  Auto's AVX-512 crossover against the exponential M2L, the per-level gain at 10⁻³; CH4.0
  already covers Batched on sparse levels.

## 2026-11-07: CH6.2d Beside FMM3D's exponential M2L
- Done:
  - The split (`/next-task`'s size rule): CH6.2d into FMM3D's per-level times and accuracy
    (CH6.2d) and the counts' corner (CH6.2e); D8 renamed CH6.2f (its references in the
    roadmap and CH6.2b's report follow). H9 added with CH9.1 (the user: "can you schedule an
    in depth comparison with fmm3d when all optimizations are applied?"), and H9 in the range
    of `/milestone-gate` and the roadmap issue sync (the user: "you have my permission to
    update the skills").
  - `tools/fmm3d/hfmm3d-levels.patch`, applied by `build.sh` to the variants `timed` and
    `fastker-timed`: per-level timers in `hfmm3dmain`'s step 3 (setup, list 4, multipoles to
    waves, the parents' loop with its plane-wave and list-3 thread-seconds, point-and-shoot)
    and the switch `FMM3D_PW_LEVEL` (one level on plane waves, or none);
    `helmholtz_fmm3d errors` (FMM3D's outputs alone against the oracle);
    `tools/helmholtz/ch6-2d-runs.sh` (`check`, `times`, `scan`) and `ch6-2d-tables.py`.
  - Our level call's two passes timed (the user: "yes, add the per-pass timers to our level
    call"): `HelmholtzOperator::plane_wave_times` (`PlaneWaveTimes`: the waves, and merge and
    shift, per level), `helmholtz_accuracy::Row::plane_wave_seconds`, two columns of
    `helmholtz_accuracy --levels`.
  - The report `fmm-validate/results/helmholtz-ch6-2d.md` (Kathleen and locust): FMM3D's
    M2L over ours by level size 0.22/0.45/0.56 at 10⁻³ and 0.44/0.89/1.11 at 10⁻⁶ (coarse,
    middle, full; the levels the two trees share); merge and shift 69–95 % of ours per coarse
    level and 2.4–3.8× FMM3D's parents' loop there in wall time, at par (0.81–1.13×) on the
    full levels, both codes able to use at most 8 threads on level 2; on locust (clean,
    after the user's "Wait for gridcl") 0.31/0.73/1.03 and 0.60/1.95/1.93, merge and shift
    2.3× FMM3D's loop on level 2; per call ours faster on the large cases (W1 at kD = 500, 10⁻⁶: 22.4 s against
    FMM3D's 61.6 and 43.2 s); the scan: FMM3D's point-and-shoot far below ε, one level at
    κ ≈ 12.5 on FMM3D's plane waves 0.9–2.0 × 10⁻³ at 10⁻³ against 2–4 × 10⁻⁵ for ours with
    the same nodes; FMM3D misses ε at 10⁻³ on three cases.
- Tests: `check_plane_wave_times` in `fmm-exec/tests/helmholtz_fmm.rs` (both passes timed and
  within the level's M2L time on every level with waves, zero elsewhere, every rank or none);
  `check_levels` on the exponential-M2L rows of `fmm-validate/tests/helmholtz_accuracy.rs`;
  the patch's `check` (patched builds byte for byte the unpatched ones, both ε, two cases, on
  Kathleen, locust and the M6). A self-review by a subagent: nothing blocking; its points
  taken (the patched builds' labels, the 15.4 ms mean, per-level shares, wall-time ratios
  not per parent, levels the two trees do not share left out of the groups, hedged causes,
  docs, the run script's `problem` exit).
- Checks: ran `/verify` on the M6: format, Clippy (default, examples, workspace all-targets),
  `cargo doc`, `cargo test` (888 s) and `cargo test --workspace` (1,337 s), the MPI list at 2
  and 4 ranks (`helmholtz_fmm` 188 and 167 s), all passed; `tests/helmholtz_fmm.rs` 203 s and
  `tests/helmholtz_accuracy.rs` 42 s at one rank in debug. Kathleen jobs 630125 and 630164,
  every launch exit 0. locust 2026-11-07 01:24–01:50 after ten gridcl-free minutes (76
  launches, exit 0); the earlier locust runs beside a gridcl test kept as records only.
- Decisions: none.
- Laplace: not touched (the Helmholtz operator's timers, `nd-fmm-validate`, `tools/`).
- Follow-ups: CH6.2e's line (FMM3D's counts too few at iprec 1 in the FMM; iprec 2 still to
  measure); CH6.2f's line (merge and shift's split into shifts and conversions on the coarse
  levels, before the coarse-level remedy); CH9.1.

## 2026-11-07: CH6.2e The counts' corner
- Done:
  - The α counts for a smaller reach: `Quadrature::with_reach` (`new` is the corner 4√2,
    bit for bit), `assert_reach`, `MergeLevel::with_reach`, `PlaneWaveM2l::build_with_reach`,
    `HelmholtzOperator::with_plane_wave_reach`, `plane_wave_reach`, `plane_wave_waves`,
    `HelmholtzFmmBuilder::plane_wave_reach`; `helmholtz_accuracy --reach` and
    `planewave-scan:<iprec>`, the rows' reach and waves per level, a "waves" column in
    `--levels`; `tools/helmholtz/ch6-2e-runs.sh` and `ch6-2e-tables.py`.
  - The report `fmm-validate/results/helmholtz-ch6-2e.md` (locust 1 × 72, 2026-11-07
    07:43–08:07; the M6 1 × 6 08:28–08:31, the user: "the m6 is quiet now"). A pair needs the
    corner: the point-to-point chain with counts for ρ ≤ 4 errs there by up to 1.1·10⁴,
    1.1·10⁷ and 1.7·10⁹ × the stated error at iprec 1–3. The FMM did not on D12's workloads:
    on CH6.1e's 60 verification cases at 10⁻³ to 10⁻⁶ the corner, 5, 4.5 and 4.25 within
    1.070× Rotation's; 4 misses on three cases at 10⁻⁶ and kD ≥ 100 (1.60–1.69×), 3.5 on 12,
    3 on 17; the scan (166 levels at iprec 2) the corner and 4.5 within 1.044×, 4 one level
    (the W4 cube's level 2 at κ = 25, 1.59×). The threshold lies between 4 and 4.25, about
    3√2. At 4.5 the waves 0.84–0.86×, the exponential M2L 0.75–0.96× per level and
    0.80–0.92× per case on locust (0.79–0.95× and 0.80–0.91× on the M6), the large cases'
    evaluation 0.89–0.95×. CH6.2d's question answered: at iprec 2 fewer counts than ours hold
    (reach 4.5, 4.25), FMM3D's reach 4 does not on three cases.
- Tests: `the_counts_for_a_smaller_reach`, `a_reach_within_the_source_box_panics` and the
  ignored `the_counts_for_a_smaller_reach_on_every_domain` (kernel within 2×, chain within 3×
  on ρ ≤ reach at 4 and 4.5) in nd-fmm-tables; `build_with_reach_takes_the_counts_of_the_reach`;
  the reach checks of `check_plane_wave` in `fmm-exec/tests/helmholtz_fmm.rs` (the corner bit
  for bit the default, reach 4 within 1.1× Rotation's with fewer waves, 2 threads, the
  one-rank FMM) and of `fmm-validate/tests/helmholtz_accuracy.rs`. A self-review by a
  subagent: its points taken (the large cases' bytes, 4.25 not scanned, hedged causes, the
  reach checked eagerly, the chain asserted in the ignored test, the tables from several
  outputs); its convention-version point rebutted (§3.15's "Exponential M2L" is unversioned).
- Checks: ran `/verify` on the quiet M6: format, Clippy (default, examples, workspace
  all-targets), `cargo doc`, `cargo test` (1219 passed, 909 s) and `cargo test --workspace`
  (1333 passed, 1236 s), the MPI list at 2 and 4 ranks (1051 s), `cargo test --release -p
  nd-fmm-tables -- --ignored` (15 passed, 553 s), all passed; `tests/helmholtz_fmm.rs` 211 s
  and `tests/helmholtz_accuracy.rs` 46 s at one rank in debug (203 and 42 s before);
  `shellcheck` not installed, not run. Pending: none.
- Decisions: questions 1. the counts' reach 4.5 in place of the corner 4√2 (CH6.2g) and
  2. CH6.2g before CH6.2f (D8), both signed off 2026-11-07 (the user: "1 and 2 as recommended").
- Laplace: not touched (the Helmholtz quadrature, merge and operator, `nd-fmm-validate`,
  `tools/`); `mpi_exec`, `mpi_threading` and `invariance` passed at 2 and 4 ranks.
- Follow-ups: CH6.2g (the reach adopted, with an M2L pair at the largest V offset against
  Rotation on every domain), next; CH6.2f after it.

## 2026-11-08: CH6.2g The counts' reach adopted (not adopted: the corner kept)
- Done:
  - The check CH6.2e's question 1 asked for, one M2L pair at the largest V offsets at the
    level's degree against Rotation on every domain, written first (nd-fmm-exec's
    `tests/helmholtz/plane_wave_reach.rs`), failed for reach 4.5: at the 24 offsets
    (±3, ±3, ±2) the counts for 4.5 add more than 10× the stated error over converged counts
    from Re κ = 7π on the weakly damped domains (47× from 8π, 2·10⁴× near 16π), against the
    direct sum at 10⁻⁶ up to about 2,000× Rotation's error; the corner's add at most 3.2×.
    Asked with that evidence, the user answered "Keep the corner (Recommended)"; asked again
    after a correction (the first question said wrongly that CH6.2e had no level with waves
    above κ ≈ 25: W1 at kD = 500 has κ = 31.25 on level 4, where the FMM held at 0.986×), the
    answer stood. The default-change steps (`DEFAULT_REACH`, the restated tests) were
    implemented and reverted before commit.
  - The test kept as the guard of the default counts (at most 10× the stated error added over
    converged counts on every domain below Im κ = 5π, an empirical bound; reach 4.5's
    printed); a paragraph in the quadrature module's docs and the crate `CLAUDE.md` files.
  - `tail_count`'s sampling checked (a throwaway example at 64–4,096 steps): the 4.5-above-corner
    inversions are the 64 steps' (none at 1,024 and 4,096), all with Im κ ≥ 17, one or two waves
    in all; kept at 64.
  - The report `fmm-validate/results/helmholtz-ch6-2g.md` (the M6, release, accuracy only);
    raw lines in `~/projects/fmm-archive/m6/ch6-2g/`.
- Tests: `plane_wave_reach::the_counts_at_the_largest_offsets` (debug: CH6.1a's fixture domains
  and domain 92 at 10⁻⁶, 16 s on the M6; asserts reach 4.5 adds more than 100× on domain 92)
  and the ignored `…_on_every_domain` (release, 437 domains less those from Im κ = 5π, at 10⁻³
  and 10⁻⁶, about 40 s; passed: the corner adds at most 2.45× and 3.21×). A self-review by a
  subagent: its points taken (an uncommitted import fix, the 10× bound called empirical, the
  damped carve-out named a deferred effect, causes hedged, "real κ's edge" made one term).
- Checks: ran on the M6: format, Clippy (default, examples, workspace all-targets: the last
  first failed on an unused import, fixed), `cargo doc`, `cargo test` (1220 passed, 784 s),
  `cargo test --workspace` (1334 passed, 1001 s), the MPI list at 2 and 4 ranks (passed: all 12 executables at 2 and 4 ranks (1168 s with the build));
  the new ignored test in release. The helmholtz test executable's debug run 31 s (about
  15 s before; under nd-fmm-exec's one-minute budget). Not run: the planned locust
  verification (the default is unchanged, outputs bit for bit `main`'s). Pending: none.
- Decisions: CH6.2e's question 1 (reach 4.5) withdrawn 2026-11-08 (the user: "Keep the corner
  (Recommended)", twice); no new sign-off.
- Laplace: not touched (a new Helmholtz test, docs); `mpi_exec`, `mpi_threading` and
  `invariance` in the MPI list (passed: all 12 executables at 2 and 4 ranks (1168 s with the build)).
- Follow-ups: CH6.2f (D8), next, with two sentences added from the report's §3: the
  exponential M2L's largest-offset pairs at the level's degree err about 10× ε relative at
  10⁻⁶ on the low-κ plateau (CH6.1c's truncated tail), and on levels with Im κ ≥ 5π both
  counts' chains err 10⁻⁹–10⁻⁵ absolute (whether the rule keeps Rotation there).

## 2026-11-08: CH6.2f The coarse levels' merge and shift
- Done:
  - Split by `/next-task`'s size rule from the former CH6.2f: the coarse levels here, D8's
    rule in CH6.2h.
  - Merge and shift's parts timed (`PlaneWaveTimes::{busy, shifts, children, conversions}`,
    each worker's clock, no value changed): level 2 ran on **one thread** and level 3 on
    eight under the shared drivers' split by parents (`PARENT_GRAIN` = 64 boxes), not slow
    per parent (Kathleen 15–18 ms of thread time per parent against FMM3D's 23–38 ms);
    the conversions take 76–90 % of levels 2–3's thread time. CH6.2d's open question answered.
  - `PlaneWaveSplit` (`Auto` the default, `Parents`, `Children`; operator and builder
    setters): by child, one task per box, each computed child running its parent's call for
    itself alone (the same arithmetic and order, so the same bits). `Auto` takes it where the
    split by parents makes at most half as many pieces holding a computed box as the pool
    has threads. The rule changed twice on the branch: the first (≤ 64 × threads boxes) took
    level 4 at 72 threads, 1.24–1.32× slower; the second counted pieces without work on P
    ranks and kept level 3 by parents at 4 × 10.
  - Runs: locust 1 × 72, Kathleen 1 × 40 and 4 × 10 (jobs 630738 and 630756), the M6 1 × 6
    (the user's go-ahead). At one rank `Auto` takes levels 2–3 by child: 0.38–0.73×
    Rotation's time on locust, 0.43–0.88× on Kathleen (before up to 12.3× and 7.8×), level 2
    on the M6 0.44–0.69×; elsewhere 0.87–1.03× the split by parents'. Our level-2 merge and
    shift is now 0.06–0.37× FMM3D's parents' loop (before 2.3–5.1×). CH6.2c's 4 × 10
    anomaly was the same one-thread level seen from two sides; at 4 × 10 `Auto` takes levels
    2–3 by child (0.57–1.10× Rotation's).
  - The report `fmm-validate/results/helmholtz-ch6-2f.md`; `tools/helmholtz/ch6-2f-runs.sh`,
    `ch6-2f-tables.py`; the harness's `--split` and parts columns; raw outputs in
    `~/projects/fmm-archive/{locust,kathleen,m6}/logs/ch6-2f*`.
- Tests: `check_plane_wave_split` (in `check_plane_wave`: `Auto` the default, never by child
  at 1 thread, by child on some rank at 2; `Parents` at 2 threads, `Children` at 1 and 2 bit
  for bit the default; `Children` on P ranks against one rank); the floor at 2 threads by
  child and by parents; `check_plane_wave_times` (the parts within `busy`);
  `the_split_is_a_setting`; the unit test `pieces_count_the_split_by_parents_with_work`;
  nd-fmm-validate's parts columns and `Children`'s checksum. The runs' checksums agree across
  the three settings and 1 × 40 against 4 × 10. A self-review by a subagent: its points taken
  (`Parents` at the floor restored, the sign-off on constraint 9, records, doc ranges).
- Checks: ran on the M6: format, Clippy (default, examples, workspace all-targets),
  `cargo doc`, `cargo test` (1222 passed), `cargo test --workspace` (1336 passed), the MPI
  list at 2 and 4 ranks (all 12 executables passed, at a 900 s launch timeout; the default
  300 s timed out three launches on a loaded M6). `tests/helmholtz_fmm.rs`' one-rank debug
  run 216 s (211 s before). Pending: none.
- Decisions: signed off 2026-11-08 (the user: "1 as recommended; schedule a Laplace task
  for 2"): (1) `Auto` may read the threads and the rank's computed boxes (constraint 9 names a
  rank's share; no bit depends on the split), as recommended; (2) a Laplace task for the same
  64-box grain in its plane-wave M2L, CH6.2i, after CH6.2h on the priority list.
- Laplace: not touched (`driver.rs` read only; the Helmholtz operator, harness and tools);
  `mpi_exec`, `mpi_threading` and `invariance` passed at 2 and 4 ranks.
- Follow-ups: CH6.2h (D8), next, with a sentence added: levels 2–3 now gain over Rotation,
  and a two-phase split (the repeated shifts) only if D8's margins need it.
  CH6.2i (Laplace's coarse levels), added by question 2.

## 2026-11-08: CH6.2h D8: the exponential M2L's level rule
- Done:
  - The rule under `HelmholtzM2l::Auto`, behind `with_auto_plane_wave` and the builder's
    `auto_plane_wave` (on by default since D8's sign-off): the exponential M2L on a level where
    p_l ≥ d_pw (`auto_plane_wave_min_degree`: Batched's crossover + 1, 12 on AVX-512 with two
    planes; agreed over the ranks and reported, named for runs by
    `m2l_plane_wave_min_degree`), CH6.1e's rule gives a quadrature and Im κ_l < 5π
    (`AUTO_PLANE_WAVE_MAX_IM_KAPPA`), f64 only; Batched at p_l ≤ d_b within the budget on the
    rest; Rotation else. Each level runs an existing level call (CH6.2f's split included).
  - The harness's `--m2l auto-pw` and `auto-pw:<d>`; `tools/helmholtz/ch6-2h-runs.sh`,
    `ch6-2h-tables.py`.
  - Runs: the M6 1 × 6 (the user's go-ahead), Kathleen 1 × 40 and 4 × 10 (jobs 630761,
    630762, with `auto-pw:16`), locust 1 × 72 (2026-11-09, quiet: others ≤ 10 % during). Against today's `Auto` the M2L takes 0.14–0.70× on the M6 and
    0.03–0.55× (matrix) and 0.23–0.49× (large) on Kathleen at 10⁻⁶; at 10⁻³ 0.47–1.07× and
    0.26–1.19× (above 1 within the ABBA spread). Within 1.10× of the
    per-level best except ms-scale cases at 10⁻³, all on few-box levels, mostly Batched at
    p = 7 on level 2 (today's `Auto` too). d_pw = 16 on AVX-512 is 1.6–35× slower at 10⁻⁶,
    so 12. Errors 0.916–1.063× Rotation's. No boxes term needed for the exponential M2L on
    CH6.2f's level call (at most 3.7 ms lost per evaluation at 10⁻³). locust: 0.21–0.69× at
    10⁻⁶ (two ms-scale cases 1.05–1.07, within their spread); at 10⁻³ Batched at p = 7–8
    behind Rotation below about 3.5·10⁵ pairs at 72 threads (1.0–28×, today's `Auto`'s
    choice), to CH4.0.
  - The report `fmm-validate/results/helmholtz-ch6-2h.md`; raw outputs in
    `~/projects/fmm-archive/{m6,kathleen,locust}/logs/ch6-2h*`.
- Tests: `auto_takes_the_waves_by_its_rule`, `auto_waves_by_the_machine_class_and_the_damping`
  (`tests/helmholtz/plane_wave.rs`); `check_auto_plane_wave` in `tests/helmholtz_fmm.rs` (bit
  for bit PlaneWave at d_b = d_pw = 0, the split within 1.1× Rotation's errors, 2 threads,
  P ranks against one, the agreed d_pw); nd-fmm-validate's `check_auto` extended.
- Checks: ran on the M6, again after the defaults turned on: format, Clippy (default,
  examples, workspace all-targets), `cargo doc`, `cargo test --workspace` (1339 passed), the
  MPI list at 2 and 4 ranks (all 12 passed), `cargo test -p nd-fmm-validate --test helmholtz_accuracy`; `cargo test` on the
  default members not run apart from the workspace run. `tests/helmholtz_fmm.rs`' one-rank
  debug run 255 s (216 s before).
- Decisions: D8 signed off 2026-11-08 (the user: "1–5 as recommended"): (1) the rule is
  `Auto`'s default (builder, operator and `Case::new`; the tests that pin CH3.2d's rule name
  it off); (2) d_pw = 12 on AVX-512 with two planes, d_b + 1 elsewhere; (3) Im κ_l < 5π;
  (4) no boxes term, Batched's few-box levels to CH4.0; (5) proposal §5.3 and §6.4 amended.
- Laplace: not touched (the Helmholtz operator and builder, the harness and tools);
  `mpi_exec`, `mpi_threading` and `invariance` passed at 2 and 4 ranks.
- Follow-ups: Batched's few-box levels and, on locust at 72 threads, its levels below about
  3.5·10⁵ pairs: a sentence added to CH4.0; CH6.2i next.

## 2026-11-09: CH6.2i Laplace's plane-wave M2L on the coarse levels
- Done:
  - Measured (`scaling --kinds`: the M2L per level with its boxes and the split by parents'
    pieces; the sweep `ch6-2i`): Laplace's merge and shift ran level 2 on one thread and
    level 3 on eight at every pool size (busy 0.6–1.4 and 2.7–9.9); at N = 10⁶ levels 2–3
    were 7–16 % of the M2L and 1.9–4.1 % of the evaluation on locust (72), 2.2–3.0 % on
    Kathleen (40), 0.5–0.8 % on the M6 (6); at 10⁷ at most 0.4 %.
  - The split by child in Laplace's level call (`Kernels::plane_child`, `plane_parent` with a
    `serve` mask; the same arithmetic and order, so the same bits); `PlaneWaveSplit` moved to
    `operator` (re-exported from `helmholtz` and `fmm`), `LaplaceOperator::with_plane_wave_split`,
    `FmmBuilder::plane_wave_split`, `Fmm::{plane_wave_split, set_plane_wave_split}`;
    `driver::parent_pieces` (Helmholtz's `pieces` moved; no existing driver code changed).
  - `Auto` (the default on the branch) by child where 4 × pieces ≤ threads
    (`PLANE_CHILD_SHARE`): Helmholtz's half took Plummer's level 4 (32 pieces) by child at 72
    threads, 1.02–1.22× slower. Levels 2–3 under `Auto` 0.27–0.49× on locust, 0.39–0.54× on
    Kathleen, 0.81–0.96× on the M6; by child on every level the M2L 1.4–2.3×.
  - D9 (`main` b73c362 against the branch, A B B A A B B A A B, nd-fmm-bench at N = 10⁶,
    PlaneWave): no row slower on locust, Kathleen (job 630796) and the M6; locust f32 p = 8 −1.5 % and f64 p = 18
    −3.8 % faster beyond the spread; errors equal in every run.
  - The report `fmm-validate/results/helmholtz-ch6-2i.md`; raw outputs in
    `~/projects/fmm-archive/{locust,kathleen,m6}/logs/ch6-2i/`.
- Tests: `check_plane_wave_split` in `tests/mpi_exec.rs` (every split bit for bit the
  default at 1, 2, 4 and the most threads, overlapped; `Auto` never by child at one thread,
  by child at four; `Children` on P ranks against the one-rank `Fmm`); `driver::parent_pieces`
  and the harness's `parent_pieces` unit tests; `tests/scaling.rs` (the levels sum to the M2L).
  A self-review by a subagent: no blocking points; its others taken (a two-thread `Children`
  check on every rank count, docs, the report's ranges).
- Checks: ran on the M6: format, Clippy (default, examples, workspace all-targets),
  `cargo doc`, `cargo test` (1226 passed), `cargo test --workspace` (1340 passed), the MPI
  list at 2 and 4 ranks (24 of 24), `tests/mpi_exec.rs` at 1, 2, 4 and 8 ranks
  (passed: 169, 229, 220 and 218 s). Pending: none.
- Decisions: signed off 2026-11-09 (the user: "ok, 1–3 as recommended"): (1) constraint 1,
  `Auto` as Laplace's default; (2) Laplace's rule at a quarter of the threads, Helmholtz's at
  half; (3) `driver.rs`'s added function.
- Laplace: changed on the plane-wave level call, outputs bit for bit at every split (tests),
  D9 as above.
- Follow-ups: none.


## 2026-11-09: CH6.1f The quadratures below 10⁻⁶ on the pair chain
- Done:
  - Split (`/next-task`'s size rule): CH6.1f the study on the pair chain, CH6.1g the FMM after
    the sign-off.
  - `nd_fmm_ref::helmholtz::plane_wave::generate_for_degree` (`plane_wave/degree.rs`): the λ
    rule fitted to the exponential M2L's chain at degree p (Gegenbauer's truncated expansions)
    against nd-fmm-ref's RCR at degree p, on the boxes' corners, face centres and centres and
    the region's 98 offsets; `Quadrature::degree`; `rule::panel_rule_to` (`panel_rule` the
    same arithmetic, so `generate`'s bits are unchanged).
  - The study (`degree_matched_study`, 260 rows, 13 levels at 10⁻⁷ and 10⁻⁸, the multipole by
    M2M from children at p as in the FMM): FMM3D's tables fail on the plateau's degrees
    (iprec 2 1.08–4.1× RCR's error at 10⁻⁷, 0.67–6.3× at 10⁻⁸; iprec 3 up to 15× and 53×) and
    hold on the high branch's (0.93–1.00×). Candidate 4, the resolving degree
    p_w = max(p_l, 24) with iprec 2 at 10⁻⁷ and max(p_l, 32) with iprec 3 at 10⁻⁸: 0.10–0.81×
    and 0.010–0.17×. Candidate 3, `generate_for_degree` at ε/20: 1.000–1.010× at 1.6–2.9×
    iprec 3's waves, about a minute to generate at p = 24. Candidate 2 erratic.
  - The fit's face centres: without them the chain met ε on the samples but erred 2.68ε and
    7.91ε between them at p = 24 (0.06ε and 0.02ε with them).
  - The report `fmm-validate/results/helmholtz-ch6-1f.md` (the M6, release, 34.8 minutes).
- Tests: nd-fmm-ref `degree.rs`'s unit tests (the expansions, RCR's normalisation) and
  `tests/helmholtz/plane_wave.rs` (against RCR off the samples within 2ε; ignored at the FMM's
  degrees, 0.00–0.06ε; determinism, `degree()`, κ = 0); nd-fmm-tables
  `tests/helmholtz/plane_wave_pair.rs` (through `PlaneWaveLevel` within 3ε, 0.01ε; ignored at
  the FMM's degrees, 0.01–0.16ε; the study).
- Checks: ran on the M6: format, Clippy (default, examples, workspace all-targets),
  `cargo doc`, `cargo test` (1233 passed), `cargo test --workspace` (1347 passed), the three
  ignored release tests above. Debug runs: nd-fmm-ref's Helmholtz tests 46 s (22 s before),
  nd-fmm-tables' 35 s. MPI list not run (no MPI crate changed). Pending: none.
- Decisions: report §5 signed off 2026-11-09 (the user: "1–3 as recommended"): (1) candidate 4
  for CH6.1g, the resolving degree; (2) the degree raised on the level first; (3)
  `generate_for_degree` kept as a trusted path only.
- Laplace: not touched (nd-fmm-ref's Helmholtz quadrature builder and tests only; no table,
  driver or kernel changed).
- Follow-ups: CH6.1g (roadmap), the FMM below 10⁻⁶ by the signed-off rule.

## 2026-11-11: CH6.1g The quadrature below 10⁻⁶ in the FMM
- Done:
  - The rule (`nd_fmm_tables::helmholtz::plane_wave::merge`): below ε = 10⁻⁶ iprec 3 with the
    level's degree raised to `plane_wave_degree` = max(p_l, 28) for 10⁻⁷ ≤ ε < 10⁻⁶ and
    max(p_l, 32) for 10⁻⁸ ≤ ε < 10⁻⁷, Rotation below 10⁻⁸ (`resolving_degree`,
    `RESOLVING_*`); named degrees unchanged.
  - The operator (`HelmholtzOperator`) keeps the degrees given (`base_degree`) and, under
    PlaneWave's rule and under Auto, builds its RCR tables at p_w on the levels with waves (the
    whole level, CH6.1f's sign-off 2(a)); `degree` and `HelmholtzFmm::degrees` report p_w.
  - CH6.1f's iprec 2 at 10⁻⁷ did not hold in the FMM: its own error floor lies above
    Rotation's on the tests' trees at any degree (1.25–6.6× at 24 and 32), which the pair
    chain's worst-case comparison hid; iprec 3 at 24 gave 0.04–0.10×. Asked mid-task.
  - Runs: Kathleen 1 × 40 (jobs 632472, 632473), CH6.1e's 26 cases at 10⁻⁷ and 10⁻⁸, Rotation
    and PlaneWave in both orders: errors 0.004–0.80× (φ) and 0.004–0.54× (∇φ) Rotation's at
    10⁻⁷, 0.003–0.48× and 0.002–0.32× at 10⁻⁸; the M2L 0.42–1.46× and 0.26–0.84×; the
    evaluation 0.54–1.05× at 10⁻⁷ and 0.37–0.81× at 10⁻⁸ on the cases above 0.1 s (two ms
    cases slower); the rest of the evaluation at p_w 1.29–1.88×.
  - The report `fmm-validate/results/helmholtz-ch6-1g.md`; `tools/helmholtz/ch6-1g-runs.sh`,
    `ch6-1g-tables.py`; raw outputs in `~/projects/fmm-archive/kathleen/logs/ch6-1g/`.
- Tests: nd-fmm-tables `the_rule_pairs_tolerances_with_precisions`,
  `the_rule_raises_the_degree_below_the_level_degrees_tolerance`, `select` at 10⁻⁷–10⁻⁹;
  nd-fmm-exec `the_rule_raises_the_degrees_below_the_level_degrees_tolerance` and
  `auto_takes_the_waves_by_its_rule` (`tests/helmholtz/plane_wave.rs`), `check_resolving_degree`
  in `tests/helmholtz_fmm.rs` (the clusters at kD = 1 and the cube at kD = 20 + 2i at 10⁻⁷ and
  10⁻⁸: 0.003–0.011× Rotation's, 2 threads, the one-rank FMM, Auto bit for bit PlaneWave's at
  10⁻⁸ and Rotation's at 10⁻⁷); nd-fmm-validate's W1 at 10⁻⁸ under the rule.
- Checks: see the PR (the M6; `/verify` after the Auto change).
- Decisions: signed off 2026-11-11: (1) "iprec 3 at max(p,28)" for 10⁻⁷ ≤ ε < 10⁻⁶ (in place
  of CH6.1f's iprec 2 at 24); (2) D8 amended, "Lift for 10⁻⁸–10⁻⁷ only (Recommended)": Auto
  takes the waves at 10⁻⁸ ≤ ε < 10⁻⁷, not at 10⁻⁷ ≤ ε < 10⁻⁶; proposal §5.3 amended; (3) "Add
  CH6.1h, next on the list (Recommended)".
- Laplace: not touched (the Helmholtz merge rule, operator and builder; tests and tools).
- Follow-ups: CH6.1h (roadmap, priority list item 6), the degree raised on the M2L alone.

## 2026-11-11: CH6.1h The resolving degree on the M2L alone
- Done:
  - `HelmholtzTables::build_with_m2l_degrees` (nd-fmm-tables): a level's expansions and M2L at
    q_l, its translations to and from the parent on the first p_l degrees (`m2l_degree`;
    `translate` on the first degrees of larger expansions; `build` unchanged).
  - `PlaneWaveRaise` (nd-fmm-exec; `with_plane_wave_raise`, `translation_degree`,
    `HelmholtzFmmBuilder::plane_wave_raise`): `Level` (CH6.1g's, the default, bit for bit),
    `Leaves` (the M2M to the parent and the L2L from it at p_l), `M2l` (also the leaves' P2M,
    L2P, P2L and M2P at p_l through a scratch expansion). `M2l` was added after a probe on the
    M6 held 1.1× with the leaves at p_l, against CH6.1f §4's reading (the plan's open
    question 1).
  - The harness's `--raise level|leaves|m2l`; `tools/helmholtz/ch6-1h-runs.sh`,
    `ch6-1h-tables.py`.
  - Runs: Kathleen 1 × 40 (jobs 632717, 632725), CH6.1g's 26 cases in both orders and 8 more W4
    cases at 10⁻⁷: `Leaves` at `Level`'s errors (at most 0.81× Rotation's), 0.93–1.03× its
    evaluation; `M2l` the rest 1.03–1.35× Rotation's (`Level` 1.36–1.88×; the cases run in both orders), the evaluation
    0.66–0.96× `Level`'s, errors 0.11–1.00× at 10⁻⁷ (17 cases), 1.53× (φ) on the clusters at
    kD = 10 and 10⁻⁸ (fails), 0.06–1.00× on the other 16.
  - The report `fmm-validate/results/helmholtz-ch6-1h.md`; raw outputs in
    `~/projects/fmm-archive/kathleen/logs/ch6-1h/`.
- Tests: nd-fmm-tables `m2m_and_l2l_on_the_first_degrees_of_raised_levels`,
  `m2l_degrees_below_the_degrees_are_refused`; nd-fmm-exec
  `the_raised_levels_translations_keep_the_degrees_given`, `the_raised_levels_leaf_operators_by_the_form`
  (`tests/helmholtz/plane_wave.rs`), `check_m2l_raise` in `tests/helmholtz_fmm.rs` (both forms on
  the four resolving scenarios within `PLANE_WAVE_RATIO`, 2 threads, the one-rank FMM bit for
  bit); nd-fmm-validate's W1 at 10⁻⁸ under both forms.
- Checks: see the PR.
- Decisions: signed off 2026-11-11 (report §5): (1) "M2l at 1e-7, Leaves at 1e-8
  (Recommended)": `PlaneWaveRaise::Rule`, the default; (2) "Lift it (Recommended)": `Auto` takes
  the exponential M2L at 10⁻⁷ ≤ ε < 10⁻⁶ too (D8 and proposal §5.3 amended).
- Laplace: not touched (nd-fmm-tables' `helmholtz` module, the Helmholtz operator and builder;
  tests, harness, tools).
- Follow-ups: none; CH6.1 is done, next H6.G (`/milestone-gate H6`).

## 2026-11-12: H6.G Gate
- Done:
  - `/milestone-gate H6`. Report `docs/helmholtz/gates/H6.md`:
    - the gate condition, D8 adopted under `Auto`;
    - all 17 tasks (#179–#204), each with its report and sign-offs, no pending run, 31
      questions signed off;
    - CI on `main` and its job times against their limits;
    - proposal §9's items and constraint 1 (CH6.2i, the one Laplace change);
    - the measurements against §5.3, §3.5, §7, §9.2 and §2;
    - corrections C1–C8, recommendations, a retrospective on the loop.
  - The gap the gate found: the default at 10⁻⁷ and 10⁻⁸ had never run on D1's large cases
    (CH6.1e's sign-off 1 left them to Rotation; CH6.1g and CH6.1h did not bring them back).
    Closed on the branch (question 2, "Run first, then close") by Kathleen job 633000
    (node-c11a-048, one rank × 40, `tools/helmholtz/h6-gate-runs.sh`, Rotation against
    `auto-pw` with the rule's raise in both orders, `main`'s 4b5b37f):
    - W1 at kD = 500 and W3 at kD = 100 at 10⁻⁷ and 10⁻⁸: errors 0.45–0.84× Rotation's;
    - the evaluation 0.27–0.65×, the M2L 0.22–0.61×, the build 1.0–1.4×;
    - each launch's peak 19.1–22.5 GB, the waves buffer at its 8 GiB bound;
    - checksums equal in both orders. Raw outputs in
      `~/projects/fmm-archive/kathleen/logs/h6-gate/`; the report's §8.
  - Roadmap:
    - H6.G ticked;
    - the status line rewritten to the current state;
    - the priority list for H4 (CH4.0, CH4.0b, CH4.0c, CH1.3d, CH4.1–CH4.5, CH0.9, CHR.3,
      CH4.6);
    - CH0.8b counts the waves;
    - H3's and H6's ticked lines condensed (1,292 → 698 lines), the full lines in git
      history (4b5b37f).
  - Proposal (draft 0.8 kept): C1–C8 applied, a line in its Appendix A ("Amended at H6.G"),
    and the missing lines of CH0.4, CH6.2h, CH6.1g and CH6.1h.
- Tests: none (documents and a run script).
- Checks:
  - None of cargo's (no Rust or manifest change). `bash -n` on the run script; `shellcheck`
    not installed.
  - `tools/roadmap/sync_issues.py`'s parser reads the edited roadmap (every milestone ends
    with its gate).
  - CI on `main`: every H6 merge passed every job. #188's run-tests (CH7.5b) was cancelled
    at its 30-minute limit on a cold-cache build after a Rust release; its next run passed.
- Decisions: signed off 2026-11-12 (the user):
  1–2. "Run first, then close (Recommended)";
  3. C1–C8, "Accept all (Recommended)";
  4. the H4 priority list, "As proposed (Recommended)";
  5–6. "run-tests limit 30→45 min, Condense the roadmap".
- Laplace: not touched (documents only).
- Follow-ups:
  - run-tests' `timeout-minutes` 30 → 45, the separate PR of question 5.
  - The retrospective's changes to `/next-task`, `/verify` and `/milestone-gate`, for the
    user (`.claude/skills/` is not writable from the session):
    - re-run cases an earlier sign-off excluded when a scope widens;
    - pair and FMM evidence both before a numerical rule changes;
    - an Appendix A line with every proposal amendment;
    - the MPI crates' debug tests at opt-level 1;
    - done roadmap lines kept short.

## 2026-11-12: CH4.0 The M2L rule's sparse-level term
- Done:
  - The term (`LevelShape`, `with_level_shapes`, `auto_batched_min_columns`): `Auto` runs
    Batched on a level only where m̂, the estimate of its columns per table read from the
    level's V pairs and boxes over the whole tree (summed over the ranks at build), reaches
    m*(p) of the machine class, besides p_l ≤ d_b; the builder agrees a table of m* over the
    ranks (all-reduce max) as it agrees the crossover; `batched_term(false)` gives CH3.2d's
    rule.
  - m* from both measures: the pair fit per degree (`helmholtz_m2l` at 1–64 columns, a/c + b)
    at each class's margin, set from the FMM's per-level times at one thread (1.45 on AVX-512
    and AVX2 with two planes, 1.2 with one, 1.15 on NEON; NEON's the larger of the M6's and
    locust's, but m*(11) with one plane the M6's, where the levels contradict locust's fit).
  - The split (`batched_block_targets`, `with_m2l_block_targets`, the builder's
    `m2l_block_targets`): Batched's blocks ⌈rows / threads⌉ within 8 to 64 on a level of
    fewer blocks than threads; no bit changes.
  - The node budget (`dense_node_budget`, `DEFAULT_DENSE_NODE_BUDGET` 8 GiB): each rank
    min(1 GiB, node budget / ranks per node).
  - The harness's `--term`, `--block`, the levels' boxes, m̂, visits and exact columns per
    read; `tools/helmholtz/ch4-0-runs.sh`, `ch4-0-tables.py` (`pairs`, `fmm`, `rule`).
  - Runs: the pair sweeps (the M6, locust, Kathleen job 633057); CH3.2d's nine cases at one
    thread (the M6, locust, Kathleen 633066), CH6.2h's matrix and large cases at the machines'
    threads (the M6 1 × 6, locust 1 × 72, Kathleen 633065 at 1 × 40); the confirmation at the
    final constants (Kathleen 633071, locust).
  - Results: CH3.2d's sweeps within 1.049 of the per-level best on every case of the three
    machines (its miss 1.196 → 1.000; the rule's own runs within 1.010 on AVX-512, 1.035 on
    AVX2, 1.003 on locust); at the machines' threads the default's evaluation 0.36–1.07× the
    previous default's, within 1.09 (the M6), 1.21 (Kathleen) and 1.28 (locust) of the
    per-level best, the excess on ms-scale cases. Under D8's default the term changed no level
    of these cases (the same checksums in all 172 launches); the gain at many threads is the
    split's.
  - The report `fmm-validate/results/helmholtz-ch4-0.md`; raw outputs in
    `~/projects/fmm-archive/{m6,locust,kathleen}/logs/ch4-0/`.
- Tests: nd-fmm-exec `auto_reads_the_level_shapes` (a sparse level Rotation's bits, dense ones
  Batched's, a named m* table), `the_level_shapes_are_one_per_level`,
  `the_columns_per_read_on_hand_shapes`, `the_term_and_the_split_widths`
  (`tests/helmholtz/dense.rs`); `check_batched_term` in `tests/helmholtz_fmm.rs` (the shapes
  and m* the same on every rank and the one-rank tree's, a level sent to Rotation by the term,
  within `DENSE_TOLERANCE` of Rotation, 2 threads and the one-rank FMM bit for bit, the
  library's default against the direct sum and one rank, block widths bit for bit, the node
  budget's share); nd-fmm-validate's columns and a named width (`tests/helmholtz_accuracy.rs`).
- Checks: `/verify` on the M6: fmt, Clippy (default, examples, workspace), docs, the
  workspace tests at opt-level 1 (2,600 passed), the MPI list at 2 and 4 ranks (24 of 24),
  `helmholtz_fmm` again at 1, 2 and 4 ranks after the last test change; the one-rank debug run
  of `helmholtz_fmm` 526 s at opt-level 0 (load 2–6), `check_batched_term` 91 s of it (21 %).
  Nothing pending.
- Decisions: questions for sign-off in the PR: (1) the default node budget, 8 GiB; (2) CH4.0
  done with its residual at 40–72 threads left to CH4.0f, and CH4.0f's place in the list; (3)
  proposal §6.4 amended for the term and the split. Signed off (the user, 2026-11-12, on #211): (1) "I agree to your
  recommendation", 8 GiB, with a few ranks per node and threads per rank the documented advice;
  (2) "Q2 as recommended", CH4.0 done and CH4.0f off the priority list; (3) "yes to Q3",
  proposal §6.4 and its Appendix A line amended on the PR.
- Laplace: not touched (the Helmholtz operator, builder and batched level call; tests,
  harness, tools; `driver.rs` used, not changed).
- Follow-ups: CH4.0f (roadmap, after CH4.0e), Batched's split across the matrix's rows; CH8.0
  (roadmap, H8; the user, on #211), one copy of the M2L tables per node in an MPI shared window.
  Laplace has no shared tables either: Phase 5N spiked one (node-m2l.md §2.6) and its decision 6
  left it out, to revisit in Phase 5S, which did not take it up.

## 2026-11-12: CH4.0b The worst-box concentration
- Done:
  - `helmholtz_accuracy::Density` (random, a plane wave's trace, constant) and
    `Case::{density, domain, max_points_per_leaf}`; `helmholtz_boxes` (the direct sum at every
    point or a per-box sample, §9.1's box measure, a target's place per level, the levels' parts
    of the error) and its example; `tools/helmholtz/ch4-0b-runs.sh`.
  - The cause (the M6, Rotation, against the direct sum at every point): placement. A level's
    error at targets beyond 1.5 half-widths from their box centre is 19–45× its RMS within one
    half-width (W1 at kD = 100, the dominant level, 10⁻³ to 10⁻⁸); by an argument from the
    operators' form (not measured) the same for sources. The coarsest
    level on the degree rule's plateau carries most of it (70–96 % of Σ|Δ∇φ|² on W1 at
    kD = 100). W1's six tangent points are box vertices on every level, so the levels' worst
    cases add there; with a margin the worst boxes are boxes the sphere clips at a corner (to
    30.6 ε at 10⁻⁸). Not the lists (one leaf level: the same), the boundary or a defect; which
    operator of a level was not separated.
  - The smooth densities (W1 at kD = 10, 30, 100 on the M6; 500 on Kathleen job 633154): the
    plane wave's trace and the constant density meet ε in ℓ2 at 10⁻³, 10⁻⁶ and 10⁻⁸ under
    Rotation and the default; the constant density is the hardest (∇φ 0.66 ε at kD = 100,
    10⁻⁸); p_w at 10⁻⁶ needs no change. Below D1 the constant density misses ε in ℓ2 (1.5× at
    10⁻⁹, 3.4× at 10⁻¹⁰ on W1 at kD = 100), all from level 2's high-frequency degree.
  - A pointwise criterion would take the rule at ε/10 (10⁻³, 10⁻⁶; 1.1–2.3× the evaluation)
    and ε/30–ε/100 (10⁻⁸; 2.9–5.2×).
  - The report `fmm-validate/results/helmholtz-ch4-0b.md`; raw outputs in
    `~/projects/fmm-archive/{m6,kathleen}/logs/ch4-0b/`.
- Tests: `helmholtz_boxes`' unit tests (offsets, the per-box sample, the level shares, the
  worst boxes); `tests/helmholtz_accuracy.rs` `check_densities` (the plane wave's trace through
  `check_fmm`, the constant density within ε, a named domain and leaf size with the leaves'
  count) and `check_box_study` (the reference against the oracle bit for bit, the box measure
  against `crosscheck` bit for bit), at 1, 2 and 4 ranks.
- Checks: `/verify` on the M6: fmt, Clippy (default, examples, workspace), docs, the default
  and workspace tests at opt-level 1, `tests/helmholtz_accuracy.rs` at 2 and 4 ranks, its
  one-rank debug run 95–99 s against `main`'s 83–87 s (loaded M6). Nothing pending.
- Decisions: signed off (the user, 2026-11-12, on #212, "Sign off 1 and 2 as recommended"): (1) ε stays §9.1's
  relative ℓ2 tolerance and the box measure joins §9.1 as a gate bound of 100 ε (proposal §9.1
  and its Appendix A amended; the degree rule's documentation notes the pointwise errors);
  (2) the gates keep §9.1's random charges (D12 unchanged), and CH4.0c, CH4.3 and CH4.4 verify
  on the constant density and the plane wave's trace too (their roadmap lines).
- Laplace: not touched (nd-fmm-validate's Helmholtz modules, tests, tools, docs).
- Follow-ups: CH4.0c (roadmap: its runs include the constant density, which shows the
  high-frequency branch short below 10⁻⁸); CH4.3 and CH4.4 (roadmap: the smooth densities in
  their verification; CH4.4 the 100 ε bound).

## 2026-11-13: CH4.0c The degree rule beyond its measured range
- Done:
  - W1 at kD = 1000 (N = 7,957,748; levels 2 and 3 at x = 216.5 and 108.3, beyond the rule's
    measured x ≤ 65) against the direct sum on Kathleen (one rank × 40 threads): within ε in
    §9.1's ℓ2 at 10⁻⁸, 10⁻⁷, 10⁻⁶ and 10⁻³, under Rotation and the default, for the random
    charges, the plane wave's trace and the constant density; at most 0.83 ε (∇φ over the
    samples and one target per level-8 box, 10⁻⁸, Rotation). No correction, no S7 degrees.
  - The levels at 10⁻⁸ (each raised to the rule at 10⁻¹¹): levels 2–4 0.6 %, 1.0 % and 0.6 % of Σ|e|² (φ),
    level 6 (x = 13.5, p = 27, where the high-frequency branch first exceeds the plateau) 85 %.
  - §9 item 8: the under-resolved W3 (N = 10⁶) within ε at kD = 200 (at most 0.61 ε) and 500
    (0.72 ε), the three densities, 10⁻⁸, 10⁻⁶, 10⁻³, both M2Ls.
  - A defect, fixed: `HelmholtzFmmBuilder` sized the exponential M2L's waves buffer for the
    plan's boxes before setting its bound, so the buffer held a whole level until then (218 GB
    on W3 at kD = 500, refused by the node; job 633324). The bound is set first;
    `HelmholtzOperator::plane_wave_buffer_peak_bytes` reports the most the buffer held.
  - The rule's measured range (10⁻⁸ ≤ ε ≤ 10⁻³, x ≤ 65; checked at x = 108 and 216) stated in
    `helmholtz::degree`'s docs and fmm-exec/CLAUDE.md; CH3.3's 10⁻¹⁸ to 10⁻¹ named as a
    comparison of formulas (review M6, M7).
  - `helmholtz_boxes` prints each run's degrees, errors and times as it finishes;
    `tools/helmholtz/ch4-0c-runs.sh`. Report `fmm-validate/results/helmholtz-ch4-0c.md`; raw
    outputs in `~/projects/fmm-archive/kathleen/logs/ch4-0c/` (jobs 633320, 633321, 633323,
    633324, 633429–633431).
- Tests: `check_plane_wave_buffer` (`tests/helmholtz_fmm.rs`) asserts the floor build's peak
  buffer is the bound's buffer (red before the fix, green after). The acceptance is the
  machine runs against the direct sum.
- Checks: `/verify` on the M6: fmt, Clippy (default, examples, workspace), docs, the default
  and workspace tests at opt-level 1, the MPI list at 2 and 4 ranks (`helmholtz_fmm` 43 s and
  75 s). The opt-level 0 debug time was not measured (one assertion added to an existing check).
  Nothing pending.
- Decisions: question 1 of the PR, proposal §3.5's check at x = 216: signed off 2026-11-13
  after the merge, "Sign off as recommended"; the proposal amended in its own PR.
- Laplace: not touched (the Helmholtz builder and operator, nd-fmm-validate's example, docs).
- Follow-ups: CH4.0d (roadmap: the under-resolved W3's leaves of about one point and the M2L
  of its empty boxes, 61 % of 2.06 million leaves, every V pair run; level 7's M2L 86 % of the
  evaluation at kD = 500; the design of 2026-11-13: empty boxes left out by an opt-in plan rule,
  then V pairs of few points as near pairs); CH4.0g (roadmap: Laplace's empty boxes counted,
  the rule adopted only if it pays); CH4.3 (the high-frequency recalibration; the tightest level at
  kD = 1000 is where the branch leaves the plateau).
