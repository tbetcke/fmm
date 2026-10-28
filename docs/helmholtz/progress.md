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
- Decisions: none taken. Questions for sign-off in report §5:
  1. the draft as CH1.1's starting point;
  2. R12's consequence for CH2.5 and CH5.3 (compare against FMM3D only where it meets its
     eps, rather than skipping CH2.5);
  3. the proposal's next draft.
- Laplace: not touched (tools, docs and one fixture file only; the driver's Laplace
  outputs are byte for byte the same).
- Follow-ups:
  - CH1.1 writes §3.15 from the draft and decides where 4π ik goes.
  - CH1.3 tests `nd-fmm-ref`'s direct sum against the fixtures, to round-off, not bits.
  - CH2.5 and CH5.3 take question 2's answer.
