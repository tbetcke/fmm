# Phase 4S / T6 — `nd-fmm-bench`: one command, a Markdown table (C4S.6)

The Phase 4 harnesses (`device_fmm`, `fmm_accuracy`, `autotune`, the kernel examples in
`nd-fmm-validate`) are research tools. They run fixed problem sets, compare strategies,
and print long reports for a task's pull request. What is missing is a benchmark that
anyone can run on any machine with one command:
- choose N, the precision, the expansion degree and the backend;
- get the overall evaluation time as minimum, maximum and mean, and the time of each
  operator kind (P2P, M2L, …);
- receive a Markdown file with one clean table.

This task adds it as a new crate, `fmm-bench/` (package `nd-fmm-bench`). It runs on the
M3 Max (host, CPU runtime, Metal) and on locust (host, CUDA).

Read first:
- root CLAUDE.md, fmm-validate/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4s/README.md ("Design decisions": "The benchmark crate", "Timing"; decision 5);
- the T5 report and `nd_fmm_exec::fmm::{KindTiming, KindTimings, StageTimings,
  BuildTimings, DeviceReport}`;
- `nd_fmm_validate::{bench, points, rng, metrics, fmm_accuracy, device_fmm}`, and the
  `device_fmm` example (argument parsing, machine lines, the direct-sum reference);
- fmm-validate/results/phase4-m3max.md ("Setup"): what a report states about the machine.

Do:
- **The crate.** `fmm-bench/`, package `nd-fmm-bench`, version `0.1.0-dev`,
  `publish = false`, inheriting `[workspace.package]`, a default member. It is a binary
  (`src/main.rs`, binary name `nd-fmm-bench`) over a small library (`src/lib.rs`) that
  holds the configuration, the measurement and the Markdown writer, so that they can be
  tested.
  - Dependencies, all workspace entries: `nd-fmm-exec`, `nd-fmm-validate` (points, the
    seeded generator, the direct sum, the machine lines), `mpi`.
  - Features `gpu`, `cpu`, `metal` and `cuda`, passed through as in `nd-fmm-validate`.
  - No new external dependency (no `clap`, `serde` or `toml`; the README, decision 5).
    Add the root `[workspace.dependencies]` entry and the crate `CLAUDE.md`.
- **The problem.** N points drawn uniformly in the unit cube [0, 1]³, from
  `nd_fmm_validate::points::cube` with a fixed seed. Sources equal targets. Charges are
  uniform in [−1, 1] from the same generator. Gradients are on by default
  (`--no-gradients` turns them off). The tree is adaptive with the library's default
  `max_level` and leaf size, unless `--leaf-size` or `--max-level` is given. The
  defaults:
  - N = 1,000,000;
  - f32;
  - p = 6;
  - the host backend.

  Every value is printed in the report.
- **The command line.** Each option takes one value or a comma-separated list, and the
  run is their Cartesian product:
  - `--n 1000000,4000000` (also accepts `1e6`);
  - `--precision f32,f64`;
  - `--degree 3,6,8` (the expansion degree p, 0 ≤ p ≤ 20; the request's "degree of
    freedom", confirmed as p with decision 5);
  - `--backend host,cpu,metal,cuda`;
  - `--threads n` (host and host-fallback threads; default every core, which the
    report states);
  - `--strategy auto|dense|classes|rotation` (default `auto`);
  - `--repeats r` (default 10);
  - `--warmup w` (default 2);
  - `--kinds sync|device|off` (default `sync`; T5's modes);
  - `--table-cache DIR` and `--tuning-cache DIR` (both optional; default none);
  - `--accuracy targets|off` (default 1,000 sampled targets);
  - `--output FILE.md` (default `bench-results/<host>-<date>.md`; the directory is
    created).

  A combination the backend refuses, such as f64 on Metal, becomes a row "refused:
  <error>", and the run goes on. `--help` prints the options. `--quick` sets N = 10⁴ and
  `--repeats 2`, for smoke runs.
- **The measurement**, per combination:
  1. **Build.** Build the `Fmm` once and record `BuildTimings`: total, tables, and the
     device part (open, upload, tuning).
  2. **Warm-up.** Run `--warmup` evaluations. They compile the device kernels and are
     not counted.
  3. **Overall time.** Run `--repeats` evaluations with kind timings `Off`. Record the
     wall time of `Fmm::evaluate` each time, then report min, max, mean, median and
     standard deviation.
  4. **Per kind.** Unless `--kinds off`: build a second `Fmm` with the chosen T5 mode,
     run `--warmup` evaluations, then `--repeats` evaluations. Report per kind the min,
     max and mean of its per-evaluation total, its share of the summed kinds, and the
     calls per evaluation. Also report the remainder "other" (load, transfers, the
     download, scaling) and the sum of the kinds against step 3's mean, so the overhead
     of the mode is visible.
  5. **Accuracy.** The relative L2 error of φ (and ∇φ) at the sampled targets against
     the f64 direct sum (`nd_fmm_validate`'s reference), from one evaluation. It is a
     sanity column, not a gate.
  6. **Device facts.** On a device: the device line (`DeviceInfo`), the resolved
     strategy and GEMM per level (`DeviceReport`), and the transfers, launches and
     syncs of one evaluation.

  The charges are the same in every evaluation. The output is checked bit-identical
  across the repeats on a device (requirement 6 of Phase 4). If it is not, the row is
  flagged rather than aborted.
- **The report.** One Markdown file:
  - **Header.** Date. Host name. CPU model and core count. The GPU (`DeviceInfo`, and on
    CUDA the driver and runtime version where CubeCL reports them). rustc, the CubeCL
    version, and the source revision (`git describe --always --dirty` in a checkout,
    else the `.source-revision` file that T1's sync script writes). The full command
    line. The BLAS and rayon thread variables.
  - **Summary table**, one row per combination: backend, precision, N, p, strategy,
    build (s), evaluation min / mean / max (ms), standard deviation, error φ, error ∇φ.
  - **Kind table**, one row per combination: P2M, M2M, M2L, L2L, P2L, M2P, L2P and P2P
    as mean ms (with min–max in a second table or in brackets, whichever reads better),
    "other", and the sum against the overall mean.
  - **Per combination**, a short section: the tree (leaves, levels, points per leaf),
    the device facts, and every kind's min / mean / max and calls.
  - **Footer.** The timing method in three sentences, and "measured on <host>; never
    asserted".

  Write the file as the run proceeds (rewrite it after each combination), so an
  interrupted run keeps its rows. Also print the summary table on stdout and progress
  on stderr.
- **One command.** `tools/bench/run.sh`, a short POSIX shell script:
  - it picks the cargo features from `--backend`: the union over the listed backends
    (`host,cuda` → `--features cuda`; host alone needs none);
  - sets every BLAS thread variable to 1;
  - runs `cargo run --release -p nd-fmm-bench --features … -- "$@"`.

  On locust the script also sources T1's environment script if it is not yet active.
  Document both commands in the crate's README section of `fmm-bench/CLAUDE.md` and in
  tools/gh200/README.md. Example:
  `tools/bench/run.sh --backend cuda --precision f32,f64 --degree 3,6,8`.
- **Ignore the output.** Add `bench-results/` to `.gitignore`. Phase reports are copied
  by hand into `fmm-bench/results/`, as T8 does.
- **Root CLAUDE.md.** Add `nd-fmm-bench` to the crate list and the benchmark command to
  "Commands".

Tests that define done:
- **Unit tests** in the library: parsing of every option, lists, `1e6`, bad input
  (exits with the usage line, from the binary only), the Cartesian product, and the
  Markdown writer on a fixed set of measurements (a golden string: column order,
  alignment, units).
- **A smoke run** as an integration test, `tests/smoke.rs`, which initialises MPI once.
  It runs the host at N = 2,000, p = 3 in f32 and f64, `--repeats 2`, into a temporary
  file, and checks that the file has a row per combination and a kind table whose kinds
  are non-negative and sum to at most the mean evaluation. With `--features cpu`, the
  same on the CPU runtime.
- **By hand on the M3 Max:** host and Metal at N = 10⁶, f32, p = 6. Paste the table into
  the pull request.
- **On locust**, if T4 has merged: host and CUDA at N = 10⁶, f32 and f64, p = 6.
  Otherwise host only, and say so.

Must pass:
- `cargo fmt --all`, then the root checks and the stricter workspace checks (the crate
  is a default member);
- `cargo clippy -p nd-fmm-bench --all-targets -- -D warnings`, and with
  `--features cpu,metal`;
- `cargo check -p nd-fmm-bench --features cuda`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-bench`, and with `--features cpu
  --release`.

Do not:
- change `nd-fmm-exec`'s or `nd-fmm-validate`'s behaviour. A helper that the bench needs
  from `nd-fmm-validate` may be made public, with docs, in its own commit;
- assert timings, or commit benchmark output beyond a phase report;
- register the binary with templated-examples or CI (a timing run);
- add dependencies beyond the workspace crates and `mpi`.
