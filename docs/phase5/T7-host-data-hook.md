# Phase 5 / T7 — nd-fmm-plan: the data-movement hook (part of C5.1, device)

On one rank the `Evaluator` writes nothing outside operator calls except `reset`'s
zeroing. That is why the device operator could keep every store on the device in Phase 4
without a plan change (device-path.md §4.3). On several ranks the evaluator also moves
host data around three exchanges: it reads values to send and writes values it received.
A device operator does not see those moves. device-path.md §4.5 sketched a
kernel-agnostic hook that tells the operator about them, and T1 refined it in
`docs/design/distributed-fmm.md` §6. This task adds the hook to `nd-fmm-plan` and proves
that it covers every movement. T8 then uses it on the device.

Read first:
- root CLAUDE.md, fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase5/README.md ("Requirements" 3–6, "Exit gate" C5.1 device);
- docs/design/distributed-fmm.md, signed off: §6 (the events, the shadow check), §8 (how
  overlap will fire the events) and §12 for T7;
- docs/design/device-path.md §4.2–§4.5;
- fmm-plan/src/{evaluator, exchange, operator, store}.rs: `reset`, the stages,
  `SourceExchange::forward`, `MultipoleExchange::forward`/`forward_all`,
  `CoarseExchange::gather`, the batch types and how they are built;
- the T6 code (`Fmm` on several ranks) and fmm-exec/tests/mpi_exec.rs.

Do:
- In `nd_fmm_plan::operator`: the event type (`HostData` or the design's name) and
  `FmmOperator::host_data(&mut self, event)` with a default that does nothing. The events
  and their arguments are exactly the design's. Starting from device-path.md §4.5, there
  is one event before each exchange reads the host store (with the indices it will send
  and mutable access to the store) and one after it wrote the store (with the indices it
  wrote), for:
  - `reset`;
  - the source exchange;
  - the coarse gather;
  - the multipole exchange (per level, or once for every level, as the design fixes).
- In `nd_fmm_plan::exchange`: the accessors the events need that are missing
  (`CoarseExchange::own_blocks()`, or as the design names them).
- The evaluator fires each event at its place in every stage, on every rank count,
  including when the exchange moves nothing. On one rank the events still come, with
  empty index lists, so that an operator handles one and several ranks alike.
- `PerPair` forwards `host_data` to nothing (the default), unless the design says
  otherwise.
- Docs: the operator docs gain "Host data": which events come, when, with what, and the
  guarantee that no host store is read or written outside operator calls and these
  events. The evaluator docs list the events per stage. Update fmm-plan/CLAUDE.md (code
  map, the rule for operators that keep their own copies) and the crate docs.

Tests that define done:
- **The host path is unchanged**: every `tests/mpi_regressions.rs` check and every
  `fmm-exec/tests/mpi_exec.rs` scenario passes unchanged, and the `Fmm` output is bit
  for bit what it was before this task. Check it on the scenarios at 1 and 2 ranks
  against outputs recorded before the change.
- **The recording operator** in `tests/mpi_regressions.rs` also records the events. In
  every scenario on every rank count it checks:
  - their order relative to the level calls;
  - that the index lists equal the exchanges' own (`send_leaves`, `ghost_leaves`,
    `send_boxes`, `receive_boxes`, the coarse blocks);
  - that every event comes on every rank.
- **The shadow check** (design §6), in `nd-fmm-exec`'s tests:
  - a shadow operator wraps `LaplaceOperator`. It keeps its own copies of the five
    stores and computes every batch on its copies, never on the evaluator's slices;
  - it learns of host data only through the events. At "send" events it writes the
    values to send into the host store; at "received" events it copies the received
    values into its own copies;
  - its output, read from its own target-output copy, equals the plain `Fmm` output bit
    for bit on every `tests/mpi_exec.rs` scenario that it can run within the debug
    budget, at 1, 2 and 4 ranks;
  - with the hook disabled (the shadow ignores the events), it differs on 2 ranks in
    a scenario with ghosts. The test asserts that difference, so the check cannot pass
    vacuously.

  If the batch types cannot be built from the shadow's own stores with the public API,
  add the smallest public constructors the design names, documented, in `nd-fmm-plan`.
  The shadow wrapper is test code in `fmm-exec/tests/`, not a public type.
- At 8 ranks, the shadow check by hand on the uniform and the graded scenarios.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-fmm-plan`;
- `tests/mpi_regressions.rs` and `tests/mpi_exec.rs` at 1, 2 and 4 ranks by hand under an
  external timeout (macOS loopback flags), and the shadow check at 8;
- the device features of `nd-fmm-exec` at one rank (`--features cpu --release`;
  `cargo check -p nd-fmm-exec --features cuda`), unchanged;
- the multi-rank CI job, if kept.

Report:
- the events per stage on one rank and on several, with their index lists and sizes for
  the cube at N = 10⁵ on 4 ranks (p = 8);
- the list of every host read and write of the evaluator, with file and function, and
  the event that covers it;
- the cost of the hook on the host path (the default no-op), as one-rank and 4-rank
  evaluation times before and after (release; reported, not asserted).

Do not:
- change what the evaluator moves, the exchanges' order or the accumulation order;
- put device or Laplace code in `nd-fmm-plan`;
- change the device operator (T8) or add overlap (T9);
- use the index FMM (removed in T2).
