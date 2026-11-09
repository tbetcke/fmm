# CH6.2i Laplace's plane-wave M2L on the coarse levels

Started 2026-11-09 on `tbetcke/h-ch6-2i`. Roadmap: "CH6.2i Laplace's plane-wave M2L on the
coarse levels" (H6; priority list item 5; added 2026-11-08 by CH6.2f's question 2, the user:
"schedule a Laplace task for 2"). Proposal: §10 CH6.2; §2 (constraints 1, 2, 9); CH6.2f's
report §2, §9 and its plan (`docs/helmholtz/tasks/CH6.2f.md`); Phase 6 T13's report
(`fmm-validate/results/phase6-planewave.md` §5, its appendix).

## Goal

Measure whether Laplace's plane-wave M2L holds its coarse levels to few threads, as
Helmholtz's did, and if they weigh in, let merge and shift split below the parent there,
bit for bit, adopted only where it gains beyond D9's noise.

## Acceptance (roadmap, CH6.2i; proposal §10 CH6.2 for the H6 milestone)

> *Done when:* Laplace's plane-wave M2L per level with the threads its merge and shift keeps
> busy, on locust (1 × 72), Kathleen (1 × 40) and the M6 (with the user's go-ahead), on
> Phase 6 T13's cases; if the coarse levels are held to few threads and weigh in the
> evaluation, a split below the parent (as CH6.2f's by child), bit for bit the split by
> parents at 1, 2 and many threads and on P ranks, adopted where it gains beyond D9's noise,
> Laplace's fixtures and bit-identity tests unchanged; else recorded as no change.
> *Sign-off:* constraint 1 (a change to a Laplace path).

## What the code shows (before any run)

`LaplaceOperator::m2l` under `PlaneWave` (`fmm-exec/src/operator.rs`) runs, per vector and
frame, the outgoing waves on `Execution::chunks` (box by box, every thread busy), then merge
and shift on `Execution::parents`, whose `split_parents` halves only while a piece holds more
than `PARENT_GRAIN` = 64 boxes. So level 2 (64 boxes) runs merge and shift on **one** thread
and level 3 (512) on at most 8, at every pool size. Unlike Helmholtz, Laplace's p is the same
on every level, so a coarse box costs what a fine one does: on the cube at N = 10⁶ (finest
level about 5) level 2 is 64 of about 37,000 boxes, but at 72 threads its one-thread piece
is about 14 % of level 5's wall time, and level 3's 64-box pieces as much again. An
estimate: on locust the coarse levels may be 15–30 % of the plane-wave M2L's wall time, on
Kathleen (40) less, on the M6 (6) little. The runs decide.

## Read first

- `fmm-exec/src/operator.rs` (`m2l` under `PlaneWave`, module docs "The plane-wave M2L"),
  `fmm-exec/src/operator_plane_wave.rs` (`plane_parents`, `plane_parent`),
  `fmm-exec/src/driver.rs` (`Execution::{chunks, parents}`, `split_parents`, `PARENT_GRAIN`).
- Helmholtz's split: `fmm-exec/src/helmholtz/operator_plane_wave.rs` (`PlaneWaveSplit`,
  `pieces`, the `Auto` rule, `child`/`parent_call`'s serve mask).
- `fmm-exec/tests/mpi_exec.rs` (`check_plane_wave`), `tests/invariance.rs`.
- `fmm-validate/src/scaling.rs` (`--kinds`, `KindTimings`), `tools/scaling/run.sh` (`t13-host`).

## Changes

Stage A, the measurement (no Laplace path change):

1. **The harness** (`nd-fmm-validate`): `scaling --kinds` reports M2L per level (the slowest
   rank's `KindTime::levels`, already measured by `KindTiming::Synchronous`) with the level's
   boxes with a V row, in the report and the summary. Threads kept busy per level: the
   level's M2L time at 1 thread over its time at n threads (both passes; the waves pass is
   split box by box, so a level whose ratio stays near 1–8 is merge and shift's grain).
2. **The sweep** `ch6-2i` in `tools/scaling/run.sh`: T13's cases where `Auto` takes
   `PlaneWave` (the cube and the Plummer sphere at N = 10⁶, f64 p = 8, 12, 18, f32 p = 8, at
   1 thread and every core; N = 10⁷ at every core), `--kinds`.

Stage B, only if stage A shows the coarse levels held and weighing in (else the task records
no change):

3. **The split by child in Laplace's level call** (`operator.rs`, `operator_plane_wave.rs`):
   CH6.2f's design. Under the split by child a (vector, frame) pass runs on
   `Execution::chunks`, one task per box: a computed child finds its parent's run of
   siblings, their sources by slot, sums each accumulator whose mask holds it over its slots
   in ascending order from zero (`WaveKernel::merge`), gathers them in ascending order
   (`WaveKernel::gather`) and converts, `plane_parent` restricted to that child, so the same
   bits. `Auto` takes it where the split by parents makes at most half as many pieces holding
   a computed box as the pool has threads (Helmholtz's rule; it reads the threads, which no
   bit depends on). The choice is once per level call: no dispatch below it; the split by
   parents' code path is unchanged.
4. **One setting for both operators**: `PlaneWaveSplit` moves from `helmholtz` to
   `operator` (re-exported from `helmholtz` and `fmm` under the same name, so Helmholtz's API
   is unchanged), with `LaplaceOperator::with_plane_wave_split` and
   `FmmBuilder::plane_wave_split` (default `Auto` if adopted, else `Parents`); `pieces` moves
   beside `split_parents` in `driver.rs` (a new crate-private function restating it; no
   existing driver code changes) and both operators call it. The harness's `--split`.
5. Docs: the operator module docs, `fmm-exec/CLAUDE.md`, `fmm-validate/CLAUDE.md`.

No convention, table, layout, dependency or `nd-fmm-plan` change.

## Tests that prove it

- Stage A: `fmm-validate/tests/scaling.rs`: the per-level M2L entries present on the
  `planewave` run, their sum the M2L kind's total.
- Stage B, `fmm-exec/tests/mpi_exec.rs`'s `check_plane_wave` (every scenario and precision
  it runs): `Children` and `Parents` named against the default bit for bit at 1 thread, 2
  threads and the scenario's most threads, overlapped too; on P ranks each against the
  one-rank `Fmm` (the existing check, each setting); `Auto` by child at 2 threads on a level
  of at most 128 boxes and never at 1 thread (a doc-hidden count). `tests/invariance.rs`
  under `PlaneWave` unchanged (the default). Trusted path: the split by parents, itself
  checked against the direct sum, `Dense` and the pair path (T13).
- Laplace's fixtures and bit-identity tests unchanged (none edited).
- No timing asserted.

## Checks

`/verify` (format, Clippy default/examples/workspace, `cargo doc`, `cargo test`,
`cargo test --workspace`, the MPI list at 2 and 4 ranks); stage B also `tests/mpi_exec.rs`
at 8 ranks by hand, and the one-rank debug run of `tests/mpi_exec.rs` against the crate's
budget.

## Machine runs

- Stage A: `tools/scaling/run.sh ch6-2i` on locust (1 × 72, a quiet node, no gridcl test),
  Kathleen (a `singlenode` job, 1 × 40) and the M6 (1 × 6, only with the user's go-ahead).
  Decides whether stage B runs.
- Stage B: the same cases with `--split parents|children|auto`, then D9 for the adoption:
  five runs per side, A B B A A B B A A B, `main` (A) against the branch's default (B), the
  evaluation time on the cases `Auto` changes and on one it leaves alone, on the three
  machines; the branch faster beyond `main`'s spread with the output bit for bit adopts it,
  slower beyond it fails.

## Constraints and risks

- Constraint 1 (sign-off): stage B changes Laplace's plane-wave level call. (1) No dispatch
  below the level call: one choice per level call; (2) layouts and tables unchanged; (3)
  outputs bit for bit (every split gives every local the same arithmetic and order; tested);
  (4) benchmarks within noise, or faster: D9 on the three machines.
- Constraint 2: the split by child gives every local the split by parents' bits.
- Constraint 9: `Auto` reads the threads and the rank's computed boxes (signed off for
  Helmholtz at CH6.2f, question 1).
- Risk: the split by child repeats shared accumulators (1.2–3× the parents' thread time in
  Helmholtz); Laplace's merge and shift may be a larger share than Helmholtz's conversions
  were, so it may gain less. The measurement decides; a split in two phases (each parent's
  accumulators once, then its children) is the alternative if the repetition eats the gain.

## Pending runs

- Kathleen job 630796 (`run.sbatch` with the D9 script: `main` b73c362 against bf7c262, five
  runs per side): decides Kathleen's D9 rows (report §5).

Ran: locust 2026-11-09 (the sweep at every split; D9), Kathleen job 630794 (the sweep at every
split, node-c11b-017), the M6 (the sweep and D9, with the user's go-ahead).

## Results

[fmm-validate/results/helmholtz-ch6-2i.md](../../../fmm-validate/results/helmholtz-ch6-2i.md):
by parents level 2 ran on one thread and level 3 on eight (levels 2–3 1.9–4.1 % of the
evaluation at N = 10⁶ on locust); `Auto` takes them by child, 0.27–0.54× their time; D9 no row
slower, two locust rows faster beyond the spread.

Departures from the plan, each from a measurement:
- **`Auto` at a quarter of the threads**, not Helmholtz's half: at 72 threads on locust the
  half rule took the Plummer sphere's level 4 (32 pieces) by child, 1.02–1.22× slower
  (`PLANE_CHILD_SHARE` = 4; report §4). Helmholtz keeps its rule.
- **The tests**: `Auto` by child is asserted at four threads (the quarter rule keeps two
  threads by parents); `Children` runs at two threads on every rank count (bit for bit the
  default; on P ranks also the one-rank `Fmm`), `Children` and `Parents` at one thread and at
  the most threads overlapped.
- Stage A and B ran together on Kathleen (`--split parents` is the code before the split).

## Open questions

Asked with the plan, answered 2026-11-09 (the user):
1. Constraint 1, may stage B change Laplace's plane-wave level call: "Measure, then split
   (Recommended)": stage A first, stage B on this branch only if the coarse levels are held
   and weigh in, adopted by D9 on the three machines, else recorded as no change.
2. The setting: "One shared public type (Recommended)": `PlaneWaveSplit` in `operator`,
   re-exported from `helmholtz` and `fmm`, `FmmBuilder::plane_wave_split` and
   `LaplaceOperator::with_plane_wave_split`.
3. The M6's timing runs: "Yes, when I start them" (the `ch6-2i` sweep and D9's ABBA runs,
   gated on a quiet machine).

Signed off 2026-11-09 (the user: "ok, 1–3 as recommended"), the report's §7:
1. Constraint 1: `PlaneWaveSplit::Auto` adopted as Laplace's default.
2. Laplace's rule at a quarter of the threads, Helmholtz's at half, each by its own
   measurement.
3. `driver.rs`: the one added crate-private function (`parent_pieces`) accepted.
