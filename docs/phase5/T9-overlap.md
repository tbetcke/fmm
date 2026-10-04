# Phase 5 / T9 — overlap of exchanges with local work (C5.2)

Every exchange of the evaluator is blocking: rlst 0.9.0 wraps the blocking
`MPI_Neighbor_alltoallv`, and the coarse gather is a blocking all-gather-v. While a rank
waits for its ghosts, it computes nothing. This task makes the exchanges non-blocking
and overlaps them with work that does not need their data, as
`docs/design/distributed-fmm.md` §8 specifies:
- the mechanism of decision 10 (scoped rsmpi point-to-point, or an approved rlst
  addition);
- the order of decision 9;
- the "communication hidden" criterion of decision 11.

The default recommendation keeps the accumulation order of fmm-plan-redesign §7.5. It
hides the source exchange behind the local upward pass, and the multipole exchange of
the finer levels behind the downward pass of the coarser ones. The output then stays
bit-identical to the blocking path.

Read first:
- root CLAUDE.md, fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase5/README.md ("Requirements" 3–6, 10; "Design decisions": "Non-blocking
  communication", "Accumulation order", "Threads and BLAS"; "Exit gate" C5.2);
- docs/design/distributed-fmm.md, signed off: §6 (how the hook's events fire under
  overlap), §7 (the device part, if any), §8, §9, and §12 for T9;
- docs/design/fmm-plan-redesign.md §7.2–§7.5, §8 and §10;
- fmm-plan/src/{evaluator, exchange}.rs; rlst 0.9.0's `GhostCommunicator` (registry
  sources) for the neighbour lists, counts and offsets; rsmpi 0.8.2's `request` module
  (`scope`, `immediate_send`, `immediate_receive_into`, `wait`, `test`) and
  `immediate_all_gather_varcount_into`;
- fmm-exec/src/fmm.rs (`Fmm::evaluate`, `StageTimings`);
- the T6 and T7 code and reports.

Do:
- **Non-blocking exchanges** in `nd_fmm_plan::exchange`, by the signed-off mechanism,
  next to the blocking ones, which stay:
  - with scoped point-to-point: per neighbour one receive and one send, posted inside one
    `request::scope`, with the value counts and displacements rebuilt from the ghost
    communicator's public counts and offsets; tags that cannot collide between the
    source exchange and the per-level multipole exchanges; no message to the rank
    itself; empty neighbours skipped consistently on both sides;
  - with an rlst addition: only if decision 10 chose it and the new rlst release is in
    the workspace; the pin is changed in its own commit.

  The received data must equal the blocking exchange's bit for bit. Test it in
  `tests/mpi_regressions.rs` with the existing exchange checks, run through both paths.
- **Overlapped stages** in `Evaluator`, as the design fixes them. A scoped request
  cannot outlive its scope, so an overlapped stage is one method that posts, works and
  waits: for example `exchange_sources_and_upward_local`, and a downward pass that waits
  for each level's ghost multipoles just before that level. Progress: call `test` (or
  the design's equivalent) between level calls, as often as the design fixes.
  - The blocking stages stay public and unchanged. The overlapped evaluation is an
    option of `Evaluator` (for example `Evaluator::evaluate_overlapped`, or a setting),
    with the stage order checked in debug builds as today.
  - The hook's events (T7) fire at the same logical points with the same index lists:
    "send" before the data is read for posting, "received" after the wait.
- **Accumulation order**: by default the order of fmm-plan-redesign §7.5, unchanged, so
  the overlapped output is bit-identical to the blocking output. If decision 9 accepted
  an opt-in reordering (for example P2P before L2P, or local before ghost row parts),
  implement it as a separate option. Document its order in the operator and evaluator
  docs ("Accumulation rule"), and test that it is deterministic and within requirement
  2's tolerance of the default. Otherwise do not implement it.
- **The coarse gather**: non-blocking behind the last levels of `upward_local`, if the
  design found it possible. Otherwise unchanged.
- **`nd-fmm-exec`**: `FmmBuilder::overlap(bool)` (or the design's name), off by default
  until the sign-off of this task's numbers decides the default. `StageTimings` reports,
  per exchange, the total time from post to completion and the exposed wait. The
  evaluation's stage times stay comparable between the two paths.
- **The device part**, only if the design §7 and its sign-off (T1, question 10) include one
  (for example downloading the packed multipoles of level l while the device computes
  level l − 1). Otherwise the device runs the blocking stages, and `Fmm` with a device
  backend ignores `overlap(true)` and reports it.
- Docs: `nd-fmm-plan` (exchange and evaluator docs: non-blocking exchanges, the
  overlapped stages, progress, determinism) and `nd-fmm-exec` (`overlap`, `StageTimings`
  fields); update both CLAUDE.md files.

Tests that define done:
- `tests/mpi_regressions.rs`, every scenario, 1, 2 and 4 ranks:
  - the non-blocking exchanges deliver bit for bit what the blocking ones do;
  - the recording operator sees the same calls, groupings and hook events, in the
    overlapped order the design fixes.
- `tests/mpi_exec.rs`, every scenario, every rank count it runs at: with
  `overlap(true)`, the output bit-identical to `overlap(false)`, and over 2, 4 and 8
  threads (`evaluate_threaded`). With an opt-in reordering, deterministic and within the
  tolerance instead.
- The shadow check of T7 passes with overlap on.
- A stress case for the protocol, by hand at 8 ranks: the graded scenario and one empty
  rank, repeated 100 evaluations, with no hang and the same bits every time.
- **The C5.2 criterion** (decision 11), measured on the benchmark case the design names
  (for example the cube at N = 10⁶, p = 8, f64, 8 ranks, one thread each; release; by
  hand):
  - exposed wait per exchange against the blocking path's exchange time, as medians over
    repeated evaluations;
  - the evaluation time with and without overlap.

  Reported, never asserted. If the criterion is not met, report the measurements, the
  cause (no progress, too little work to hide behind) and a recommendation, and stop.
  Do not reorder anything that decision 9 did not accept.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-fmm-plan`;
- `tests/mpi_regressions.rs` and `tests/mpi_exec.rs` at 1, 2, 4 and 8 ranks by hand,
  under an external timeout with the macOS loopback flags; the ignored C5.1 gate with
  overlap on at 4 ranks;
- the device features at one rank and at 2 ranks on the CPU runtime (`--features cpu
  --release`);
- the multi-rank CI job, if kept.

Report:
- the C5.2 measurement table;
- what the exposed wait is made of: the source exchange, each multipole level, the
  coarse gather;
- the bytes and messages per exchange at 8 ranks, so that T10 and a later network run can
  estimate the gain at network bandwidths;
- a recommendation for the default of `overlap`.

Do not:
- change the accumulation order unless decision 9 accepted it, and then only behind its
  own option;
- use `unsafe` or raw MPI FFI anywhere in the workspace (requirement 6);
- call MPI from a rayon worker. Every post, test and wait runs on the calling thread;
- remove the blocking path.
