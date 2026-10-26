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
  stated as GPL); `cmake` dropped from CI (no crate runs it: trap job 608744), kept unused on locust and Kathleen; docs.
  Report `fmm-validate/results/helmholtz-chr-2.md`.
- Tests: no new test; 48 Laplace outputs of `main` and the branch compared byte for byte
  (identical); the licence check's negative case with FFTW enabled.
- Checks: ran `/verify` on the M3 Max (all pass; MPI at 2 and 4), Kathleen 608726 (all pass
  but the known `spikes/p2p-simd` failures), 608727 (correctness, 2 nodes), 608733 (weekly
  examples), locust's checks so far; pending on locust: the ignored CUDA tests, the cmake
  trap, the D9 benchmark. locust's MPI list dropped by the user.
- Decisions: D0 and D9 signed off 2026-10-26; D9 clarified the same day: one-sided
  (faster passes with identical bits), five runs per side in A B B A order from CH2.0 on,
  never pooled; the licence check a separate CI job.
- Laplace: unchanged bit for bit (48 full outputs, fixtures, bit-identity and invariance
  tests); benchmarks within noise (D9 as clarified) on the M3 Max and Kathleen, locust
  pending.
- Follow-ups: CHR.4 (next runnable); CHR.3's `fftw` job needs an allowance for the fftw-sys
  clarification.
