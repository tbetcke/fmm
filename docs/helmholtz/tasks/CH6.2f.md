# CH6.2f The coarse levels' merge and shift

Started 2026-11-08 on `tbetcke/h-ch6-2f`. Roadmap: "CH6.2f The coarse levels' merge and
shift" (H6; priority list item 4; split on this branch, `/next-task`'s size rule, from the
former CH6.2f, D8, whose rule is now CH6.2h). Proposal: §10 CH6.2; §2 (constraints 1, 2, 7,
9), §5.3, §6.4; CH6.2b's report §7, CH6.2c's (CH6.2b's report §4–§5), CH6.2d's report §3.

## Goal

Find where the exponential M2L's coarse levels spend their time and let merge and shift use
the pool's threads there, so that CH6.2h's level rule need not read the threads.

## Acceptance (proposal §10, verbatim)

CH6.2: "per-level accuracy and time beside FMM3D's on the same problems; adopted as a strategy
where it wins, else kept opt-in or dropped (D8)". D8 is CH6.2h's. This sub-task's, from the
roadmap after the split:

> *Done when:* the parts per level on Kathleen (1 × 40) and locust (1 × 72) on W1 and W3 at
> kD = 100 and W1 at kD = 500, ε = 10⁻³ and 10⁻⁶, beside FMM3D's parents' loop (CH6.2d's
> report §3); the split below the parent bit for bit the split by parents at 1, 2 and many
> threads, at the buffer's floor and on P ranks the one-rank output; levels 2 and 3's
> exponential M2L time against Rotation's and against the split by parents on both machines
> (the M6 with the user's go-ahead), with the 4 × 10 anomaly of CH6.2c checked against it.

## What the code shows (before any run)

`HelmholtzOperator::plane_wave_m2l` runs merge and shift on the shared drivers'
`Execution::parents`, which halves a chunk at parent boundaries only while a piece holds more
than `driver::PARENT_GRAIN` = 64 boxes (Laplace's grain, Phase 6 T13). Level 2 has 64 boxes,
so its merge and shift runs whole on **one thread**; level 3 (512 boxes) in at most 8 pieces.
CH6.2d's "level 2 holds both codes to 8 threads" was an assumption ("its busy threads are not
measured"): ours held level 2 to one. Its 149 ms on Kathleen is then about 19 ms per parent,
against FMM3D's about 38 ms per parent on 7.7 threads; the per-parent cost may not be the
problem at all. The thread-seconds below test this before anything else.

## Read first

- `fmm-exec/src/helmholtz/operator_plane_wave.rs` (the level call, `parents`, `parent_call`),
  `fmm-exec/src/driver.rs` (`Execution::{chunks, parents}`, `split_parents`, `PARENT_GRAIN`;
  read only), the operator module docs ("M2L").
- `fmm-exec/tests/helmholtz_fmm.rs` (`check_plane_wave`, `check_plane_wave_times`).
- `fmm-validate/src/helmholtz_accuracy.rs` (`Row::plane_wave_seconds`, the levels table),
  `examples/helmholtz_accuracy.rs`, `tools/helmholtz/ch6-2b-runs.sh`, `ch6-2d-runs.sh`,
  `ch6-2d-tables.py`; CH6.2d's report §3 and appendix.

## Changes

`nd-fmm-exec` (the Helmholtz operator only), `nd-fmm-validate` (the harness), `tools/helmholtz/`.
No convention, table, dependency, `nd-fmm-plan` or Laplace change: `driver.rs` is used, not
changed (`Execution::chunks` exists).

1. **The parts timed** (no value changed). `PlaneWaveTimes` gains the thread-seconds of merge
   and shift's parts, summed over the workers' own clocks: `shifts` (each accumulator over its
   slots: `shift_add`), `children` (each child's sum of its accumulators and its child shift)
   and `conversions` (each child's sums from waves, local from sums and rotation back), and
   `busy` (the parents pass's thread-seconds in all, so busy ÷ `parents` is the threads it
   kept busy, FMM3D's column). Each scratch set's `PlaneScratch` carries the clocks; the level
   call drains them into the level's entry after each pass. Clock reads per parent and frame:
   a few, against milliseconds of work.
2. **The split by child.** A setting `PlaneWaveSplit` (`Auto`, the default; `Parents`, the
   split by parents as before; `Children`), `HelmholtzOperator::with_plane_wave_split`,
   `plane_wave_split()`, `HelmholtzFmmBuilder::plane_wave_split`. Under `Children` a chunk's
   merge and shift for (vector, frame) runs on `Execution::chunks` over its boxes, one box per
   task: a computed child finds its parent's run of siblings by key, their sources by slot
   (from the computed siblings' V rows, as `parent_call`), sums each accumulator its mask holds
   over its slots present in ascending order from zero (`shift_add`), adds them in ascending
   accumulator order, shifts, converts and adds its local rotated back: exactly the arithmetic
   and order `parent_call` gives that child, so the same bits; the siblings' shared
   accumulators are computed once per child that needs them (more shifts, the conversions
   split eight ways). `Auto` takes `Children` for a chunk where the split by parents leaves
   threads idle, a chunk of at most `PARENT_GRAIN` × threads boxes (the threads of the pool;
   no bit depends on the choice, so constraint 2 holds and constraint 9 is not engaged),
   `Parents` elsewhere and serially. Whether `Auto` is the default is this task's measurement:
   if `Children` loses where `Auto` takes it, the rule changes before the PR. **Changed after
   locust's first cases (2026-11-08):** the first rule, a chunk of at most `PARENT_GRAIN` ×
   threads boxes, took `Children` on level 4 of W1 and W3 at kD = 100 at 72 threads,
   1.24–1.36× slower than `Parents` there (which already kept about 50 threads busy; the split
   by child's thread time 1.5–1.9× the parents' on levels 2–4, the repeated shifts 3–9×).
   Now `Children` where the split by parents makes at most half as many pieces as the pool
   has threads (`pieces` restates the driver's halving), the measured repetition rounded up
   to 2. **Changed after Kathleen's 4 × 10 (2026-11-08):** a rank's level also holds boxes
   it does not compute, whose pieces have no work but counted, so level 3 stayed split by
   parents at 4 × 10 on 3 busy threads; now the pieces holding a computed box are counted.
3. **The harness.** `helmholtz_accuracy --plane-wave-split <auto|parents|children>`, the row's
   setting and its parts per level (`Row::plane_wave_parts`), a column per part and the busy
   threads in `--levels`; `tools/helmholtz/ch6-2f-runs.sh` and `ch6-2f-tables.py`.
4. Docs: the operator module docs ("M2L"), `fmm-exec/CLAUDE.md` and `fmm-validate/CLAUDE.md`.

## Tests that prove it

- `check_plane_wave` (`fmm-exec/tests/helmholtz_fmm.rs`, two scenarios): `Children` and
  `Parents` named against the default bit for bit at 1 thread, at 2 threads and at the
  buffer's floor (the chunked level call); on P ranks each against the one-rank FMM (the
  existing check, every setting); `Auto` at 2 threads taking `Children` on a level of at most
  128 boxes (asserted through a doc-hidden count of the chunks it split by child, or the
  parts' clocks). Trusted path: the split by parents, itself checked against Rotation, the
  pair path and the direct sum.
- `check_plane_wave_times`: the parts timed on every level with waves and V pairs, their sum
  within `busy`, zero elsewhere; every rank or none.
- `tests/helmholtz/plane_wave.rs`: the setting's default and setters.
- `fmm-validate/tests/helmholtz_accuracy.rs`: the parts' columns on the exponential-M2L rows.
- No timing asserted.

## Checks

`/verify` (format, Clippy default/examples/workspace, `cargo doc`, `cargo test`,
`cargo test --workspace`, the MPI list at 2 and 4 ranks). The debug run of
`tests/helmholtz_fmm.rs` at one rank against nd-fmm-exec's minute budget (31 s before).

## Machine runs

`tools/helmholtz/ch6-2f-runs.sh` (from `ch6-2b-runs.sh`'s `large` and `matrix` stages): W1
and W3 at kD = 100 and W1 at kD = 500, ε = 10⁻³ and 10⁻⁶, `--levels --evaluations`, under
`PlaneWave` with the split by parents and by child (and `Auto`), and Rotation; each launch twice
in reversed order.
- **Kathleen** (a `singlenode` job, 1 × 40; and 4 × 10 on W3 at kD = 100 for CH6.2c's
  anomaly): decides the parts, the busy threads and the split's gain on 40 cores.
- **locust** (1 × 72, quiet node, no gridcl test running): the same at 72 threads.
- **the M6** (1 × 6) only with the user's go-ahead (they run other jobs on it).
FMM3D's parents' loop is CH6.2d's (same problems, same machines), not run again.

## Constraints and risks

- Constraint 1: no Laplace path changes (`driver.rs` read only; Laplace's own plane-wave M2L
  has the same grain, an observation for the report, not this task's change).
- Constraint 2: the split by child gives each local the split by parents' bits; tests at 1,
  2 threads, the floor and on P ranks.
- Constraint 7: no new buffer (the child body uses the thread scratch it has).
- Risk: the split by child repeats the shared accumulators' shifts up to the masks' sizes; on
  a level whose shifts dominate it may lose at few threads, which the `Auto` rule's grain
  threshold and the measurement decide.

## Pending runs

None. Ran: locust 1 × 72 (2026-11-08, the second rule; the first rule's pass a record);
Kathleen job 630738 (1 × 40 and 4 × 10, the second rule) and job 630756 (the third rule:
W1 and W3 at kD = 100 at 1 × 40 and 4 × 10, W1 at kD = 500 at 4 × 10; `Auto` takes level 3
by child at 4 × 10, one rank unchanged); the M6 1 × 6 on W1 and W3 at kD = 100 (the third
rule, with the user's go-ahead, 2026-11-08).

## Results

[fmm-validate/results/helmholtz-ch6-2f.md](../../../fmm-validate/results/helmholtz-ch6-2f.md):
level 2 ran on one thread (the 64-box grain), not slow per parent; by child levels 2–3 take
0.38–0.88× Rotation's time at one rank, `Auto` the better split on every level.

## Open questions

(none yet)
