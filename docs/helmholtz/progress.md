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
