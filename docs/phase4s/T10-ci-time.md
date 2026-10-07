# Phase 4S / T10 — CI time: package installs, the kernel job's second test run, and warm caches (C4S.9)

The CI workflow (`.github/workflows/run-tests.yml`) has grown with Phases 4 and 4S. On
pull request #70 (2026-10-07) the job `run-tests-kernels` took 13, 17 and 29 minutes on
three runs of the same documentation-only change:
- **The package install.** "Install MPI and libclang (for nd-fmm-exec)" took 16 s, 134 s
  and 842 s. In the slowest run `apt-get` fetched 67 MB from `azure.archive.ubuntu.com`
  at 86 kB/s. A slow mirror is transient, but every run downloads the same packages
  again, and the root job `run-tests` installs the same libraries and more in five
  steps, also from the mirror.
- **A second test run.** "Kernel compilation measurement (CubeCL profiling log)", added
  in Phase 4 T4 (`a27923e`), runs the whole `nd-fmm-kernels` suite a second time, with
  `CUBECL_DEBUG_LOG` set, to print the compile times of the kernel variants. That is
  3–4.5 minutes of every run, after "Run tests" has just run the same tests (about 4
  minutes).
- **Cold caches.** CI runs only on `pull_request`. A cache saved by a pull request's run
  is scoped to that pull request, so every pull request's first run builds from
  scratch. T2 measured this job at 541 s cold against 143 s warm (PR #63). The root job
  `run-tests` has no cargo cache at all.

This task makes CI faster and less exposed to slow mirrors **without dropping or
weakening any check**: the same commands run with the same features and give the same
results, and the compile-time table is still printed. It changes the workflow and the
documents that describe it, and no Rust code.

Read first:
- root CLAUDE.md ("Checks": the three jobs and their commands), fmm-kernels/CLAUDE.md
  (the compile-time measurement, "CI"), docs/phase4s/README.md (decision 6, decision
  13);
- `.github/workflows/run-tests.yml` (all three jobs) and the other workflows
  (`run-examples.yml`, `run-dependency-checks.yml`), so nothing they rely on changes;
- docs/design/laplace-fmm-plan.md §8.1 (the CI paragraph: the kernel job's measured
  times) and device-path.md §17.1 (the CI row);
- docs/design/workspace-structure.md §2 ("the default members and the root CI job never
  build CubeCL");
- GitHub's documentation of `actions/cache`: scope and restore rules (a pull request can
  restore caches of its base branch), the 10 GB limit per repository and eviction.

Do:
1. **Baseline.** From the GitHub API (`gh run list`, `gh run view --json jobs`), take
   the job and step times of the last 10 or more `run-tests` runs. Separate each pull
   request's first run (cold) from its later runs, and report a table per job and step:
   median and range. Note every run where a package install took more than a minute.
2. **One kernel test run, not two.** Set `CUBECL_DEBUG_LOG` on the "Run tests" step and
   run the awk summary (`fmm-kernels/tools/compile_times.awk`) on its log in a following
   step. Then drop the second run. Check that the test output, the test count and the
   "backends run" lines are the same with the log on, and that the log adds at most a few
   seconds; say so with numbers. If the log changes what the tests do or print, keep the
   measurement as a separate run, but only on `workflow_dispatch` or a weekly schedule,
   and say why.
3. **Package installs that need no mirror.**
   - Cache the downloaded `.deb` files with the first-party `actions/cache`, keyed on the
     runner image and the package list, in both jobs that install packages. On a hit,
     `apt-get install` only unpacks. On a miss it downloads, as today, and saves the
     cache.
   - Measure whether `--no-install-recommends` shrinks the 67 MB without losing what
     `mpi-sys`, `rlst` and bindgen need (the build passes, and the same tests run).
   - Add `apt-get` retries and a bounded timeout (`-o Acquire::Retries=…`,
     `-o Acquire::http::Timeout=…`), so that a stalled mirror is retried rather than
     waited on.
   - Measure whether `apt-get update` can be skipped on a cache hit. If it cannot be
     skipped safely, keep it.
   - A third-party action (for example one that caches apt packages) only after asking:
     decision 13.
   - **Not this:** moving `cargo check -p nd-fmm-exec --features cuda` into the root job
     to save the kernel job its install. The root job never builds CubeCL
     (workspace-structure.md §2). Mention it as considered and rejected.
4. **Warm caches for every pull request.**
   - Run the workflow on pushes to `main` as well (decision 13), so that `main` holds
     caches that every pull request can restore from its first run on.
   - Give the root job a cargo cache like the kernel job's (registry, git index,
     `target/`).
   - Add `restore-keys` prefixes, so that a changed `Cargo.lock` starts from the newest
     older cache instead of from nothing.
   - State the cache sizes per job and their total against the 10 GB repository limit,
     and what is evicted first.
5. **Fail fast where waiting is pointless.** A `timeout-minutes` per job, about twice its
   warm time and above its worst cold time, and one on each package-install step. A
   timed-out install fails the run, and the failure's message says that a re-run is the
   remedy.
6. **Measure after.** At least three cold and three warm runs: a new pull request, then
   re-runs and pushes, and runs on `main`. The same table as item 1, before against
   after, per job and step. Report the compile-time table of item 2 from a run, to show
   it is still printed. Times are reported, never asserted.
7. **Docs.**
   - Root CLAUDE.md "Checks": the jobs as they are now (triggers; the kernel job's
     single test run with the log). The commands that developers run stay the same.
   - fmm-kernels/CLAUDE.md: where the compile times now come from.
   - laplace-fmm-plan.md §8.1, device-path.md §17.1 (the CI row) and the Phase 4S
     README: the new times.

Tests that define done:
- **Every check still runs.** Compare the workflow before and after, command by command:
  - `cargo fmt -- --check`, both clippy commands, `RUST_MIN_STACK=8388608 cargo test`
    and `cargo doc --no-deps` in `run-tests`;
  - both `run-tests-simd` runners with their three commands;
  - in `run-tests-kernels`: clippy with `cpu`, both CUDA type-checks, and the release
    test run with `--show-output`.

  The pull request lists each check of the old workflow and where it runs now.
- **Same results.** A run of the new workflow passes on the pull request, and on `main`
  after the merge. The test counts per job equal those of the baseline runs (the same
  test executables and tests).
- **The compile times** of the kernel variants are printed in a run of the new workflow,
  with the same columns as before.
- **The caches restore.** A pull request's first run after a push to `main` restores the
  caches (the step log says "Cache restored from key …").

Must pass:
- the root checks and the stricter workspace checks on the M3 Max (nothing in the Rust
  code changes, so these confirm the tree, not the workflow);
- the workflow itself, green on the pull request, and on `main` after the merge.

Do not:
- drop, weaken or move a check, or change a command's features or flags beyond what
  items 2–5 say;
- build CubeCL in the root job;
- add a third-party action, a self-hosted runner or a GPU job;
- change Rust code or `Cargo.toml`;
- assert a time.
