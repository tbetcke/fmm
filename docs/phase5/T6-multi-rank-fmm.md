# Phase 5 / T6 — the Laplace FMM on any number of ranks (C5.1, host)

With `Redistribution` (T5), `Fmm` can take points in any distribution over the ranks and
move them to the ranks that own their leaves. This task does that inside `build` and
`evaluate`, with unchanged signatures, and removes `PointsNotOwned`. Then it checks the
phase's first gate: **the distributed result equals the one-rank result** on 2, 4 and 8
ranks, on every debug scenario and at the Phase 4 workload points. The device path stays
on one rank until T8.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-plan/CLAUDE.md, fmm-validate/CLAUDE.md;
- docs/phase5/README.md ("Requirements" 1–4, 8–10, "Design decisions": "The oracles",
  "Accuracy measures", "Workloads", "Ranks on the M3 Max", "Tests and MPI"; "Exit gate"
  C5.1 host);
- docs/design/distributed-fmm.md, signed off: §3 (the weights the FMM passes, if any), §4,
  §5, §9 and §10, and §12 for T6;
- fmm-exec/src/fmm.rs: the module docs ("Redistribution (C5.1)"), `build` steps 1–8,
  `PointsNotOwned`, `DeviceNeedsOneRank`, `Fmm::evaluate`, `scaled_output`,
  `StageTimings`, `BuildTimings`;
- fmm-exec/tests/mpi_exec.rs (`cases`, `share`, `built`, `exact`, `relative_l2`,
  `evaluate_threaded`, `evaluate_every_kernel`, the `ownership`, `single leaf` and
  `device backends` scenarios), and tests/{accuracy, adaptive}.rs;
- `nd_fmm_validate::fmm_accuracy` (`run`, `measure`, `Oracle`) and the examples
  `fmm_accuracy` and `calibrate`, which refuse more than one rank today;
- the T5 code (`nd_fmm_plan::redistribute`).

Do:
- **`build`**, with the steps renumbered in the module docs:
  - Keys from every rank's own sources and targets, as today. If T4 added weights and
    the design fixes the FMM's weight, pass it to the octree.
  - One `Redistribution` for sources and one for targets, as the design fixes. The f64
    coordinates are forwarded and converted to leaf-scaled values on the owner
    (CONVENTIONS §3.13), as step 8 does today for local points. The counts per leaf come
    from the redistributions.
  - The cheap path for points already on their owners, if the design kept one.
  - `PointsNotOwned` is removed. The error enum and docs state what replaces it, if
    anything (design §4).
  - A device backend on several ranks still returns `DeviceNeedsOneRank` after the
    redistribution until T8, on every rank, with no collective.
  - `BuildTimings` gains the redistribution as the design names it.
- **`evaluate`**: forward this evaluation's charges to the owners (into the source
  chunks), run the stages as today, and return the output to the caller's ranks and
  order. Both moves are collective, and a rank with no points enters them.
  `StageTimings` gains the two moves. The signature and the output order are unchanged.
- **Determinism** (requirement 3): for fixed input on every rank and a fixed rank count,
  two evaluations, two builds and every thread count give the same bits.
- **Docs**: the crate and module docs drop "until C5.1" and describe the distributed
  `Fmm`: where points go, the collectives per build and per evaluation, determinism,
  what differs from one rank and by how much, and the threads rule per rank.
  Update fmm-exec/CLAUDE.md: the MPI section's "Until C5.1" line, the scenario rules,
  and the one-rank reference.

Tests that define done:
- **`tests/mpi_exec.rs`**, every scenario on every rank count (1 in CI's root job; 2 and
  4 in the multi-rank job if kept; 8 by hand):
  - `built()` no longer turns `PointsNotOwned` into a stop. Every scenario runs
    distributed;
  - each scenario that builds an `Fmm` also builds the **one-rank reference**, the same
    settings over the union of every rank's points on a one-rank communicator (README,
    "The oracles"), and checks:
    - the multi-rank output within the signed-off tolerance of requirement 2, for φ and
      ∇φ;
    - the errors against the direct sum as before;
  - `evaluate_threaded` keeps checking 2, 4 and 8 threads bit for bit, now per rank
    count;
  - the input distributions of the README ("Workloads") in at least the uniform and a
    clustered scenario: all on rank 0, a seeded random share, one empty rank, and the
    octree's own partition (the old `device backends` shape). Each is within the
    tolerance of the one-rank result, and two of them are within it of each other;
  - the `ownership` scenario becomes a scenario that passes the same points in two
    distributions and gets equal results;
  - `single leaf` runs on every rank count, or says why it cannot;
  - a tiny problem on more ranks than the coarse tree has blocks: T4's agreed error or
    refinement, on every rank, with no hang;
  - the debug run at one rank stays under a minute. Run the one-rank reference only
    where the budget allows, and say where it was left out.
- **The C5.1 host gate** (ignored, release, a new executable `tests/multi_rank.rs` with
  its own single MPI test, or in `accuracy.rs` and `adaptive.rs` if they can drop their
  one-rank assertion): the cube and the Plummer sphere at N = 10⁵, at the workload points
  (f64 p = 3, 8, 18; f32 p = 3, 8), from a seeded random share per rank, on 2, 4 and 8
  ranks by hand:
  - the output within the tolerance of requirement 2 of the one-rank run;
  - the errors against `direct_sum` within 0.1% (f64) and 1% (f32) of the one-rank run's;
  - the C3.2 and C3.3 gates still pass.

  It prints ranks, threads and every rank's leaf and point counts.
- **`nd-fmm-validate`**: `fmm_accuracy::run` and `measure` take each rank's share and
  gather or reduce what they report. The oracle is computed once (on rank 0 or sharded
  by targets, as you see fit). `fmm_accuracy` and `calibrate` run on any rank count,
  and the smoke test in `tests/fmm_accuracy.rs` passes at 1 and 2 ranks. Register
  `fmm_accuracy` with templated-examples, if it runs in the weekly job's time at 3
  ranks.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug, one rank, within budget) and
  `--release -- --ignored` at one rank;
- `tests/mpi_exec.rs` at 2, 4 and 8 ranks by hand, under an external timeout with the
  macOS loopback flags; the ignored gate at 2, 4 and 8 ranks (release);
- the device features still build and pass at one rank (`--features cpu --release`;
  `--features metal --release -- --ignored` by hand), and a device backend at 2 ranks
  returns `DeviceNeedsOneRank` on both ranks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate` at 1 and 2 ranks;
- the multi-rank CI job, if kept: report its time against T3's budget.

Report:
- a table per workload point and rank count: φ and ∇φ difference from one rank (relative
  L2 and max), the errors against the direct sum beside the one-rank run's;
- per-rank leaves, points and near pairs (the load balance as built);
- the build and evaluation time at 1, 2, 4 and 8 ranks with one thread each, and the
  redistribution's share (release; reported, not asserted; T10 measures properly).

If a difference exceeds the tolerance, break it down by operator kind and list (README,
"Accuracy measures") and stop.

Do not:
- change the plan, the exchanges or the evaluator (T7, T9), or the octree beyond
  passing T4's options;
- change any operator, the host defaults or the one-rank results. A one-rank run must
  equal today's bit for bit; check it on every scenario at one rank. The one exception:
  if the design fixed a new order of points within a leaf (§4), one-rank sums change in
  the last bits. Then say so, and show that the one-rank errors against the direct sum
  equal the old ones to the printed digits;
- enable the device on several ranks (T8);
- tune a tolerance to pass.
