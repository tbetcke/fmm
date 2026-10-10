# Phase 5N / T4 — the strategy rule for the node, tables at scale, ranks × threads (C5N.3)

T3's design, `docs/design/node-m2l.md`, signed off (README decisions 5–8), fixes:
- how `M2lStrategy::Auto` resolves with the node context;
- how `build` learns and agrees that context;
- how tables are loaded when many ranks build at once;
- what is done, or deferred, about one copy of the tables per rank;
- what the documentation recommends for ranks × threads.

This task builds exactly that in `nd-fmm-exec` (and in `nd-fmm-tables`' cache, if the
design asks for it), runs the before/after measurements on every machine available after
decision 0 (the M3 Max and locust, and one to four Kathleen nodes if Kathleen is usable),
and ends the phase with the design-document update. Without Kathleen, every Kathleen item
below is "not applicable", and the PR lists what a later Kathleen run must confirm. Where this brief and the signed-off design
differ, the design wins; say where in the PR.

Read first:
- `docs/design/node-m2l.md` as signed off, and the sign-off record in docs/phase5n/
  README.md (decisions 5–8);
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-tables/CLAUDE.md, fmm-validate/CLAUDE.md;
- `fmm-exec/src/tables.rs` (`M2lStrategy`, `resolve`, `AUTO_DENSE_MAX_P`,
  `Tables::load_or_build`), `fmm-exec/src/fmm.rs` (`FmmBuilder`, `build` steps 1 and 7, the
  device path's `split_shared` and `RankPlacement`, `Fmm::strategy`, `Fmm::threading`),
  `fmm-exec/src/threading.rs` (`ThreadingReport`, the module documentation), `fmm-exec/src/
  tune.rs` (the device path resolves `Auto` by its own rule; keep it);
- `fmm-tables/src/cache.rs` (`TableCache`, `store`, `load`, `load_or_build`);
- `fmm-exec/tests/mpi_exec.rs` (the scenarios and how they set the strategy) and
  `fmm-exec/tests/multi_rank.rs`;
- `fmm-validate/src/scaling.rs`, `examples/scaling.rs`, `tools/scaling/run.sh` and its
  Slurm mode (T2);
- the T10 reports and `fmm-validate/results/phase5n-kathleen.md`.

Do:
- **The rule.** Implement the signed-off rule:
  - in `M2lStrategy::resolve`, or in a new resolution step that takes the node context (a
    small struct of the agreed inputs, documented), never returning `Auto`;
  - its documentation states the rule, its inputs and its source (`node-m2l.md`, the
    measurements), replacing the Phase 2 T8 paragraph where the rule changed, and keeping
    the device path's note (with a device backend and M2L on the device, `Auto` follows the
    `tune` module, not this);
  - `AUTO_DENSE_MAX_P` is kept, changed or deprecated as the design says; anything that
    cites it (docs, the harness, the T10 reports' text is history and stays) is updated.
- **The node context in `build`.**
  - Every rank resolves the same strategy: the inputs are agreed (ranks per node from one
    `split_shared` on the host path if the design uses it, threads per rank as set, p,
    precision), and any input that one rank could see differently is agreed in an existing
    collective where possible (step 1's all-reduce), or a counted new one.
  - The collective count changes only as the design states; update the `fmm` module
    documentation ("Collectives") and distributed-fmm.md §9.1's table note.
  - Ranks with no points take part, as in every collective (root CLAUDE.md, "MPI").
- **What is reported.** `Fmm::strategy()` returns the resolved strategy (as today); the
  node context and the reason for the choice are reported where the design puts them (for
  example in `ThreadingReport` and its `Display`), so every benchmark line shows them.
- **Tables at scale** (decision 7), as signed off: for example one rank per node builds and
  stores into the cache while the others wait at a barrier and load, or a documented
  warm-up step; with no `table_cache`, behaviour as the design states. Any change to
  `TableCache` keeps its guarantees (no partial file under a final name; two writers of
  the same key both succeed; loads check the header, versions, key and checksum) and gets
  tests in `nd-fmm-tables`.
- **One copy per node** (decision 6): build only what was signed off; if the decision was
  "left to C6.6" or an upstream rsmpi ask, write nothing here beyond the documentation of
  the decision (and the upstream issue's link, if one was opened with the user's
  approval).
- **Ranks × threads.** The `threading` module documentation gains the measured guidance
  for many cores per node (one rank per socket or NUMA domain with threads, as the design
  recommends, distributed-fmm.md §14.4 S6), with the machines' numbers cited; the
  `scaling` example and `tools/scaling/` gain the knobs the design names (for example a
  sweep over ranks per node at fixed cores, and the strategy column in every summary).
- **Tests:**
  - unit tests of the rule: every input combination the design names resolves as the
    table in `node-m2l.md` says, and never to `Auto`;
  - in `tests/mpi_exec.rs`, within its budget: at 2 and 4 ranks every rank resolves the
    same strategy; an explicit strategy is never overridden; with the resolved strategy
    fixed, the output is bit for bit the one-rank `Fmm` of that strategy over the union in
    rank order (Phase 5's guarantee, now stated per resolved strategy); the debug run at
    one rank stays within its budget (report the time);
  - if the one-rank default changed (decision 5): every test or fixture that assumed Dense
    at p ≤ 8 is updated to name its strategy explicitly where it checks Dense, and the
    change is listed in the PR;
  - the table loading at scale: a test at 2 and 4 ranks that a cold cache directory is
    filled once and every rank loads equal tables.
- **Before/after runs** (release, BLAS variables at 1), with the harness:
  - on each available machine, N = 10⁶ cube and Plummer sphere, f64 and f32 at p = 8 and f64 at
    p = 3 and 12: `Auto` before (Phase 5) and after, against the best explicit strategy, at
    every ranks-per-node count of T3's grid and the threads splits;
  - the exit gate's measure: at p = 8 the default within the signed-off margin of the best
    strategy at every count measured;
  - errors against the direct sum for the new default (f64 to the printed digits of the
    explicit strategy's run, f32 within 1%);
  - with Kathleen: in jobs of at most 2 nodes, within the node-hours of decision 3
    (proposed 30); locust with the load checked before and after and stated.
  - Report them in a "Phase 5N, T4" section of `fmm-validate/results/phase5n-kathleen.md`,
    or, without Kathleen, of a new `fmm-validate/results/phase5n-node-m2l.md` (and short
    notes in the two T10 reports pointing to it, if their conclusions changed), stating
    the machine, ranks per node × threads, build and job ids.
- **Design documents**, with the Phase 5N outcome:
  - laplace-fmm-plan.md: the revision note at the top; §7, a Phase 5N section (status per
    component C5N.1–C5N.3 with the numbers, and a "Recommendation for Phase 5S"); §8.3
    (Kathleen and the Slurm harness, if used); §9.1 (the Phase 5N risks with what was found); §9.2
    (the strategy question answered, or what stays open);
  - distributed-fmm.md: a note in §15 on the node rule and, with Kathleen, on its first
    inter-node numbers (T2), pointing to the reports; without Kathleen, that the
    inter-node figures stay unmeasured and why (decision 0);
  - `docs/design/node-m2l.md`: an outcome section (decisions as taken, measured
    numbers);
  - workspace-structure.md §6: a Phase 5N row (done);
  - docs/phase5n/README.md: tick the exit checklist;
  - the Phase 5S and Phase 6 briefs: correct any fact this phase changed (the rule, the
    configurations to assume, table loading), saying which in the PR.

Tests that define done:
- The rule's unit tests; the `mpi_exec` checks above at 1, 2 and 4 ranks (CI) and by hand
  at 8 ranks on the M3 Max, locust and (if usable) Kathleen; the table-loading test.
- Every Phase 5 test passes: `tests/mpi_exec.rs`, `tests/multi_rank.rs` (the C5.1 host
  gate) at 2, 4 and 8 ranks, `tests/device_ranks.rs` on the CPU runtime at 2 ranks, and
  nd-fmm-validate's `tests/scaling.rs` at 1 and 2 ranks.
- The before/after table shows the exit gate's margin met on every machine and count
  measured, or the PR says plainly where it is not, with the numbers.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` and `-p nd-fmm-validate` (and
  `-p nd-fmm-tables` if the cache changed);
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks on the M3 Max, and at 8 ranks on
  the M3 Max, locust and (if usable) Kathleen;
- with Kathleen, in a job, the root checks once on the final branch.

Do not:
- build anything the sign-off did not accept (no shared-memory windows without the rsmpi
  addition, no `unsafe`, no new dependency);
- change the device path's strategy rule (`tune`), the leaf size, `P2pChoice::Auto`, or
  any default other than the strategy rule;
- change a test's tolerance to make it pass; a bit-level change outside what decision 5
  allows is a defect;
- assert timings, or commit CSV or raw output beyond the report;
- submit any Kathleen job larger than 2 nodes (80 cores).
