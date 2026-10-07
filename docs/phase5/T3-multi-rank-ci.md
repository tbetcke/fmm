# Phase 5 / T3 — a multi-rank CI job

CI runs every test on one MPI rank. The ghost layer, the exchanges and the global levels
exist only on several ranks, and so far they have been checked only by hand
(fmm-plan/CLAUDE.md, "Multi-rank runs"). Phase 5 makes every `Fmm` scenario run on any
rank count. A defect there should fail a pull request, not wait for a hand run. It was
decided on 2026-10-04 to add a multi-rank job (docs/phase5/README.md, decision 4). This
task builds it and measures it, so that the sign-off can keep, change or drop it, as
Phase 4 T4 did for `run-tests-kernels`. The job runs on GitHub's runners only; locust
(decision 2, revised on 2026-10-05) adds by-hand runs beside it, never CI runs.

Read first:
- root CLAUDE.md ("Checks", "MPI"), fmm-plan/CLAUDE.md ("Multi-rank runs"),
  fmm-exec/CLAUDE.md (the MPI-owning test executables), octree/CLAUDE.md ("MPI
  examples");
- docs/phase5/README.md ("Requirements" 4 and 9, "Design decisions": "Tests and MPI",
  "Ranks on locust", "Machines");
- tools/gh200/README.md ("MPI at n ranks");
- `.github/workflows/run-tests.yml` (the three jobs and how `run-tests-kernels` caches)
  and `.github/workflows/run-examples.yml`;
- fmm-plan/tests/mpi_regressions.rs and fmm-exec/tests/mpi_exec.rs: how they behave on
  several ranks today. Most `mpi_exec` scenarios stop with `PointsNotOwned` on several
  ranks until T6; `device backends` runs distributed.

Do:
- Add a job `run-tests-mpi` to `.github/workflows/run-tests.yml`, on `ubuntu-latest`,
  for pull requests to `main`, alongside the existing jobs. It:
  - installs the same native prerequisites as `run-tests` (Open MPI included);
  - caches the cargo registry and the target directory, as `run-tests-kernels` does;
  - builds the MPI-owning test executables in debug, without running them
    (`cargo test -p … --test … --no-run`). It finds each executable's path from cargo's
    JSON output (`--message-format=json`), not from a glob over `target/debug/deps`;
  - runs each at 2 and at 4 ranks, under `timeout` and with a step `timeout-minutes`,
    with `RUST_MIN_STACK=8388608` and `--test-threads=1`. The executables:
    - `nd-fmm-plan`: `mpi_regressions`;
    - `nd-fmm-exec`: `mpi_exec`; `mpi_threading`, if it is meaningful on several ranks
      (say which);
  - runs the `nd-octree` MPI examples at 2 and 4 ranks, if their time allows (they run
    weekly at 3 ranks today). Say whether you included them;
  - prints the CPU, the core count and `mpirun --version` first.
- Open MPI on the runner:
  - `ubuntu-latest` has 4 vCPUs. Open MPI counts slots by cores and may refuse 4 ranks;
    use `--oversubscribe` (or `--map-by :OVERSUBSCRIBE`) where needed, and say what the
    runner reports;
  - the runner is not root, so no `--allow-run-as-root`;
  - do not use the macOS loopback flags there. If the TCP interface selection hangs, pin
    the interface or use the shared-memory transport, and say what was needed.
- Make every multi-rank test fail on every rank or on none, so a failure never leaves a
  rank blocked. If a test fails or hangs at 2 or 4 ranks on the runner but not on the
  M3 Max or locust, report it and stop. Do not mark it ignored to get the job green.
- **Measure**, over at least three runs of the job (re-run it on the PR):
  - the cold and the cached wall time of the job, and of build and run separately;
  - the time of each executable at 2 and 4 ranks;
  - any flaky result.

  Estimate how the times grow once T6 runs every `mpi_exec` scenario on several ranks:
  those scenarios now stop early, and after T6 they also build a one-rank reference.
  Propose a budget (for example 15 minutes cached) and what to drop if T6 exceeds it.
- Root CLAUDE.md, "Checks": describe the new job like the other two (what it runs, at
  which ranks, that it covers correctness only). Change "CI never runs anything on more
  than one rank" under "MPI" accordingly. Keep the rule that every multi-rank run by
  hand has an external timeout.
- fmm-plan/CLAUDE.md and fmm-exec/CLAUDE.md: say that CI now runs their MPI executables
  at 2 and 4 ranks, and that 8 ranks and the ignored tests stay by hand, on the M3 Max
  and on locust.

Tests that define done:
- The job passes on the PR at 2 and 4 ranks, three times.
- A deliberately failing assertion on one rank only, pushed to a throwaway branch, makes
  the job fail within its timeout instead of hanging. Show the log in the PR, and do not
  merge the branch.
- The existing jobs are unchanged.

Must pass:
- the four CI jobs on the PR;
- the multi-rank runs by hand on the M3 Max at 2 and 4 ranks, with the loopback flags
  and an external timeout, to compare with the runner;
- the same runs by hand on locust at 2, 4 and 8 ranks, under an external timeout and
  without the loopback flags (tools/gh200/README.md, "MPI at n ranks"), a Linux run to
  compare with the runner;
- the root checks.

Report:
- the measurements above, as a table;
- the proposed budget;
- a recommendation: keep, change (for example, only `mpi_regressions` on every PR and
  `mpi_exec` nightly) or drop. Recorded as decision 4 at sign-off.

Do not:
- change any test, scenario or library code to make the job pass. A defect found here
  is reported, and fixed in its own commit with its test, only if it is small and you
  say so;
- run ignored or release tests in the job;
- add timings, benchmarks or more than 4 ranks to CI.
