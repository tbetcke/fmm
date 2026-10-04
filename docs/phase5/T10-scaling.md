# Phase 5 / T10 — scaling on the M3 Max, design-document update (C5.3; gate: scaling report)

This task measures what the phase delivered, on the only machine available: the M3 Max,
one node, ranks over shared memory (docs/phase5/README.md, decision 2). It covers:
- strong and weak scaling of the host path, stage by stage;
- load balance and communication per rank;
- the cost of redistribution;
- the effect of overlap;
- ranks against threads at a fixed core count;
- what was not measured, stated plainly: inter-node scaling, and the device on several
  ranks as a timing.

It ends the phase with the design-document update, as Phase 4 T13 did.

Read first:
- root CLAUDE.md, fmm-validate/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-plan/CLAUDE.md;
- docs/phase5/README.md ("Requirements" 8–10, "Ranks on the M3 Max", "Threads and BLAS",
  "Exit gate", "Decisions to sign off", "Exit checklist");
- docs/design/distributed-fmm.md, signed off: §9 (memory per rank, replicated data), §11
  (scaling method) and every decision as signed off;
- docs/design/laplace-fmm-plan.md §7 (Phase 3 "C3.5, threads", Phase 3S "FMM", Phase 4
  benchmarks) and §8.3;
- `fmm-validate/results/phase4-m3max.md` (the format of a results report);
- the T4–T9 reports;
- `nd_fmm_validate::{bench, fmm_accuracy}` and the `p2p_fmm` and `device_fmm` examples
  (how stage times, medians and the machine line are printed).

Do:
- **A scaling example** `scaling` in `nd-fmm-validate`, on any rank count, not
  registered with templated-examples (it times). It runs one configuration per launch
  and prints one block per run: machine, ranks, threads per rank, the BLAS variables,
  precision, p, strategy, N, distribution, overlap, the input distribution.
  - Per rank: leaves, points, near pairs, V pairs; build time by part (domain, octree,
    plan, redistribution, tables, evaluator with its exchange builds); evaluation time
    by stage (charge forward, each exchange as total and exposed wait, upward, downward,
    leaves, output return).
  - Reduced over ranks: max, min and mean per stage, and the load imbalance of the
    compute stages (max over mean).
  - Per exchange: bytes and messages per rank.
  - Peak memory per rank, or the design's formula evaluated, and say which.
  - Medians over repeated evaluations, with the repetition count printed.
  - A driver script (or documented loop) launches the sweep under `mpirun` with the
    loopback flags and an external timeout per launch.
- **Runs** (release, outside the sandbox, so the machine line is known; every BLAS
  thread variable set to 1; at most 12 ranks × threads):
  - **Strong scaling**: the cube and the Plummer sphere at N = 10⁶ (and N = 10⁵, to show
    where the problem runs out of work), f64 p = 3 and 8, f32 p = 8, the default strategy,
    1, 2, 4, 8 and 12 ranks with one thread each. Overlap off and on (T9). Input from a
    seeded random share per rank.
  - **Weak scaling**: N = 10⁵ per rank for the cube (N = 10⁵ × P, uniform), and the
    Plummer sphere with N = 10⁵ × P, 1–12 ranks, the same points. Report the efficiency
    against one rank.
  - **Ranks against threads at 12 cores**: 1 × 12, 2 × 6, 4 × 3, 6 × 2 and 12 × 1, on
    the N = 10⁶ cube and Plummer at p = 8, f64.
  - **Redistribution**: build and per-evaluation cost from all points on rank 0, from a
    random share, and from points already on their owners, at 8 ranks.
  - **Load balance**: the per-rank work as built (T4's weights, if any) on the cube, the
    Plummer sphere and the Gaussian clusters at 8 ranks, and how the slowest rank's
    stage times relate to it.
  - 16 ranks (the efficiency cores included) once for the N = 10⁶ cube, labelled as
    such, or not at all. Say which.
  - Every run's errors against the direct sum at the sampled targets match the one-rank
    run's to the printed digits (f64) and within 1% (f32). Use T6's machinery; say which
    runs were checked.
- **Device on several ranks**: a correctness table only (T8's gate results, the
  transfers and syncs per evaluation at 1, 2 and 4 ranks). No device time is presented
  as scaling (decision 3).
- **The inter-node run, documented**: one command (or a short job script) that runs the
  sweep on a cluster with Open MPI or MPICH. Name the numbers it would settle: exchange
  cost at network bandwidth, the value of overlap there, and scaling beyond one node
  (the replicated coarse data at large P, design §9). Check that it builds on Linux,
  for example through the CI job's environment. Claim no result from it.
- **The report**: paste the output into the PR and keep it as
  `fmm-validate/results/phase5-m3max.md`, in the format of the Phase 4 report. It states
  that every figure is one node of the M3 Max over shared memory.
- **Design documents**, with the Phase 5 outcome:
  - laplace-fmm-plan.md:
    - the revision note at the top;
    - §5 (the distributed `Fmm` as built: redistribution, the hook, overlap);
    - §7, Phase 5: status per component C5.1–C5.3 with the measured differences
      against one rank and the scaling tables, and a "Recommendation for Phase 6";
    - §8.1 (the "Distributed" and "Topology" rows as built; the multi-rank CI job;
      `IndexFmm` retired) and §8.3 (scaling as measured);
    - §9.1 (the Phase 5 risks, with what was found) and §9.2 (the questions answered
      and still open: inter-node scaling, the device on several GPUs);
  - fmm-plan-redesign.md: short outcome notes in §9 (redistribution as built) and §10
    (overlap as built), and a note that the index FMM of §1.6 and §11 was removed in
    Phase 5 T2. Leave the signed-off text otherwise as it is;
  - device-path.md: §14 and a short note in §17 (the device on several ranks as built:
    the hook, transfers and syncs per evaluation, errors);
  - workspace-structure.md: §3 and §3.1 (the new surfaces of `nd-fmm-plan`,
    `nd-octree`, `nd-fmm-exec` and `nd-fmm-validate`), §4 (redistribution and overlap
    done) and §6 (Phase 5 done);
  - docs/design/distributed-fmm.md: the decisions as taken and the measured numbers
    replacing its models where they differ, in an outcome section, as device-path.md §17
    does;
  - docs/phase5/README.md: tick the exit checklist.

Tests that define done:
- A smoke test of the `scaling` core in `nd-fmm-validate` (a tiny problem, two
  repetitions, at 1 and 2 ranks) that checks the report's fields and the errors against
  the one-rank run. It sits in its own test executable if it initialises MPI.
- The T4–T9 test suites still pass.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate` at 1 rank, and the smoke test at 2
  ranks by hand;
- the sweep on the M3 Max, as above;
- the multi-rank CI job, if kept.

Do not:
- assert timings, or commit CSV or other output beyond the results report;
- change the library to improve a number. A defect found here is reported, and fixed in
  its own commit with its test;
- present a model as a measurement, or a one-node figure as inter-node scaling. Every
  number is labelled "measured (M3 Max, ranks × threads, build)" or "model";
- add criterion or any other dependency without asking.
