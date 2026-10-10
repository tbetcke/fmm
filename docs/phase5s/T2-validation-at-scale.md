# Phase 5S / T2 — validation at scale (C5S.1; S7)

Every Phase 5 check compares a multi-rank run with the one-rank `Fmm` over the union of the
points: in tests every rank builds it on `SimpleCommunicator::self_comm()`, and the
`scaling` harness (`nd_fmm_validate::scaling::run`, `--reference`) gathers every point to
rank 0 and builds it there. Neither fits at N = 10⁸–10⁹ on thousands of ranks. T3–T8 change
the structure under the evaluator, and each must show that no bit moved, at the scales
where the change matters.

This task builds S7 (distributed-fmm.md §14.4) before any structural change, as T1's
design specifies:
- **rank-count invariance**: a run on P ranks equals a run on P/k ranks bit for bit, when
  rank r of the small run holds the concatenation of the inputs of ranks k r … k r + k − 1
  of the large one (requirement 3);
- the distributed direct-sum sample, which the harness has since Phase 5 T10, checked at
  N ≥ 10⁸;
- generation by index (`Workload`), so no rank holds the global set;
- a Kathleen correctness job that runs the MPI suites and the invariance check at 2 nodes.

Read first:
- docs/phase5s/README.md ("Requirements" 2, 3, 9; "The oracles"; "Tests and MPI") and
  `docs/design/scale-out.md` §5 and §9 (T1, signed off);
- distributed-fmm.md §14.4 (S7) and §14.5;
- fmm-validate/CLAUDE.md; `nd_fmm_validate::scaling` (`Workload::{point, charge, share,
  sample}`, `Input`, the distributed direct sum, the reference built on rank 0, `Report`,
  `RankRow`) and `fmm-validate/tests/scaling.rs`; `fmm-validate/examples/scaling.rs`;
- `fmm-exec/tests/mpi_exec.rs` (`cases`, `share`, the one-rank comparison) and
  `fmm-exec/tests/multi_rank.rs` (the C5.1 host gate);
- `.github/scripts/run-mpi-tests.sh` and the `run-tests-mpi` job;
- tools/kathleen/README.md (Phase 5N T1: launcher, modules, job templates).

Do:
- **The invariance check as a test.** In `nd-fmm-exec` (a scenario of `tests/mpi_exec.rs`,
  or a new MPI-owning test executable if T1 places it there), runnable at 1, 2 and 4 ranks:
  - every rank generates its inputs by index, so any rank can produce any rank's share;
  - for every k that divides the world size (at 4 ranks: k = 2 and 4), the small run
    lives on a sub-communicator of P/k ranks (`split` of the world; the other ranks take
    part in the split and skip the run, or run a duplicate, without any collective only
    some ranks enter), and its rank r holds the concatenation of the inputs of world
    ranks k r … k r + k − 1, in that order;
  - each world rank compares its own targets' output, bit for bit, with the small run's
    output for the same targets (moved back by index, not by gathering everything);
  - workloads: the cube, the Plummer sphere and one graded scenario with one empty rank,
    N ≈ 10⁴, f64 p = 8 and f32 p = 3, gradients, a random share; overlap off and on;
  - the verdict is all-reduced, so the test fails on every rank or on none.
  It must fit `run-tests-mpi`'s budget (15 minutes for the cached job; report the added
  time at 2 and 4 ranks) and the one-minute debug budget of `mpi_exec` at one rank, or
  live in its own executable registered in `run-mpi-tests.sh`.
- **The invariance check at scale.** An `nd-fmm-validate` mode (`scaling --invariance k`,
  or a separate example; not registered with `run-examples` if it times) that runs the
  same workload on P and on P/k ranks in one job, compares bit for bit by index, and
  prints the number of differing values per rank (reduced). It must not gather the
  outputs: each rank of the large run sends its targets' values to the rank of the small
  run that holds them, or both runs compare against an order both can compute.
- **The harness at N ≥ 10⁸.**
  - `--reference` refuses (agreed error, clear message) above a size rank 0 can hold, or
    T1's alternative applies;
  - the distributed direct-sum sample checked at N = 10⁸: its cost per rank O(N/P), its
    compensated f64 sum, and its result against a run at a smaller rank count within
    0.1% (f64) and 1% (f32);
  - per-rank memory of the harness itself measured (it must not hold O(N) on any rank).
- **A Kathleen correctness job** in `tools/kathleen/` (a script or job template,
  documented in its README): at 2 nodes in the `test` QoS, the MPI test list of
  `run-mpi-tests.sh` at 2, 4 and 8 ranks and at 80 ranks across the nodes, the
  invariance check at 80 against 40, 20 and 10 ranks, and the `scaling` smoke run. Every
  launch under `timeout`. Its output is a short table (result and wall time per launch).
- **Runs**: the test at 1, 2, 4 and 8 ranks on the M3 Max and on locust; the at-scale
  check on locust at 512 against 64 ranks oversubscribed (not timed), and (Kathleen) at 160
  against 20 ranks in one 4-node job and at 1,280 against 160 oversubscribed on the same 4
  nodes; the harness at N = 10⁸ (Kathleen, 4 nodes; without Kathleen at N = 10⁷ on
  locust). No job larger than 4 nodes.
- fmm-validate/CLAUDE.md, fmm-exec/CLAUDE.md and the root CLAUDE.md "Checks" ("On
  Kathleen (by hand)") updated.

Tests that define done:
- The invariance test passes at 1, 2 and 4 ranks (CI) and at 8 by hand, overlap off and
  on, f32 and f64. A deliberately broken input order (concatenation in the wrong order)
  makes it fail on every rank (shown once, by hand; not committed as a failing test).
- The at-scale check reports 0 differing values at 512 against 64 ranks on locust and
  (Kathleen) at 160 against 20 and 1,280 against 160.
- The harness runs at N = 10⁸ (N = 10⁷ without Kathleen) without `--reference`, with
  errors against the direct sum matching a run at a quarter of the ranks within 0.1%
  (f64).
- (Kathleen) The correctness job passes and is documented.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks (with the new executable, if any)
  on the M3 Max and on locust;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate` and its smoke test at 2 ranks.

Do not:
- change `nd-octree`, `nd-fmm-plan` or the library code of `nd-fmm-exec` (this task adds
  checks only; a defect found is reported and fixed in its own commit with its test);
- gather all outputs or all points to one rank in the at-scale check;
- assert timings, or commit raw outputs;
- run on Kathleen's login nodes beyond short, light commands, or submit a job larger than
  4 nodes.
