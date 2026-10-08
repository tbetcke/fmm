# nd-fmm-bench

Purpose: the benchmark of the nd-project FMM that anyone can run with one command: N
points uniform in the unit cube, f32 or f64, an expansion degree p and a backend; the
evaluation time (min, median, mean, max, standard deviation), the time per operator kind
and the error against the direct sum, as one Markdown file. A binary (`src/main.rs`,
`nd-fmm-bench`) over a small library (`src/lib.rs`: `options`, `measure`, `report`,
`machine`) that holds the configuration, the measurement and the Markdown writer.
Phase and components: Phase 4S, C4S.6 (task T6 in docs/phase4s/; decision 5); the `load`
and `output` columns, `--reuse-output` and `--output-pass` (task T9, C4S.8).

## Running it

From anywhere in a checkout (the report goes to `bench-results/` under the current
directory, ignored by git):

```sh
tools/bench/run.sh --help
tools/bench/run.sh                                   # host, N = 10^6, f32, p = 6
tools/bench/run.sh --quick --backend host,cpu        # a smoke run, N = 10^4
tools/bench/run.sh --backend host,metal --precision f32 --degree 6   # M3 Max, unsandboxed
tools/bench/run.sh --backend cuda --precision f32,f64 --degree 3,6,8
```

On locust, from the M3 Max outside the sandbox (tools/gh200/README.md, "The benchmark";
check the GPU and the CPU for other users' jobs first, docs/phase4s/README.md "Timing"):

```sh
tools/gh200/sync.sh
tools/gh200/remote.sh 'timeout 3600 tools/bench/run.sh --backend host,cuda --precision f32,f64'
```

`tools/bench/run.sh` picks the cargo features from `--backend` (the union over the
listed backends; the host alone needs none), sets every BLAS thread variable to 1,
sources tools/gh200/env.sh on locust if it is not active, and runs `cargo run --release
-p nd-fmm-bench --features … -- "$@"`. The same by hand:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
    GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    cargo run --release -p nd-fmm-bench --features metal -- --backend host,metal
```

Options (`--help`): `--n`, `--precision`, `--degree` and `--backend` take a list, and
the run is their Cartesian product; `--threads` (default every core), `--strategy`,
`--repeats` (10), `--warmup` (2), `--kinds sync|device|off` (`sync`), `--accuracy
TARGETS|off` (1,000), `--no-gradients`, `--leaf-size`, `--max-level`, `--table-cache`,
`--tuning-cache`, `--output` (default `bench-results/<host>-<date>-<time>.md`, UTC, so
two runs on one day do not overwrite each other) and `--quick` (N = 10⁴, 2 repeats).
From Phase 4S T9: `--reuse-output` (every evaluation, warm-ups and kind-timed ones
included, through `Fmm::evaluate_into` into one `Output`; the default stays
`Fmm::evaluate`, so older reports stay comparable) and `--output-pass auto|host|device`
(`FmmBuilder::output_pass`, default `auto`); the header's "output" row records both.

The kind table splits "other" (Phase 4S T9): `load` (`StageTimings::load`: the charges
into leaf order, on a device with their upload and scatter, and in `sync` mode the sync
after them), `output` (`StageTimings::output`: the output pass, on a device with the
download, the evaluation's one wait for the device), and "other", the rest outside the
level calls (the stages' remainder less load and output). Each combination's section
gives load and output with min, mean and max, and on a device the download's part of
output (`StageTimings::download`). They come from the kind-timed evaluations.

## Rules
- `publish = false`. A timing tool: never registered with templated-examples or CI, and
  nothing asserts a timing. Tests check the parsing, the Markdown writer (a golden
  string) and a smoke run's structure only.
- It measures through the public API of `nd-fmm-exec` and changes nothing in it or in
  `nd-fmm-validate`. A helper it needs from `nd-fmm-validate` may be made public there,
  with docs, in its own commit.
- The command line is parsed by hand: no `clap`, `serde` or `toml`.
- Every value of a run is in the report's header (README requirement 7): the machine,
  the devices, the software, the source revision (`git describe`, else the
  `.source-revision` that tools/gh200/sync.sh writes), the command line and the thread
  variables. A combination that did not run is a "refused" or "failed" row, never a
  number (requirement 8).
- Seeded and deterministic apart from timings: the points, charges and sampled targets
  come from `SplitMix64` with `measure::SEED`. Changing the problem or the seed makes
  old reports incomparable; say so in the PR.
- Reports: `bench-results/` is ignored. A phase report is copied by hand into
  `fmm-bench/results/` (Phase 4S T8: `results/phase4s-gh200.md`; T9:
  `results/phase4s-t9-host-part.md`); no other output is committed.
- MPI: the binary initialises it once at `Threading::Funneled` and runs on one rank.
  `tests/smoke.rs` owns the test executable's one MPI initialisation; add checks to its
  one test, not new `#[test]`s.
- Before finishing:
  - `cargo clippy -p nd-fmm-bench --all-targets -- -D warnings`, and with
    `--features cpu,metal`;
  - `cargo check -p nd-fmm-bench --features cuda`;
  - `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-bench`, and with `--features cpu
    --release` (the smoke run on the CPU runtime too).

## Allowed dependencies
All through [workspace.dependencies]: nd-fmm-exec, nd-fmm-validate (points, the seeded
generator, the direct-sum oracle, the error metrics, the machine lines), mpi, and
nd-fmm-tables for the `Stored` bound of `Fmm<T>` only (nd-fmm-exec does not re-export
it). Features `gpu`, `cpu`, `metal` and `cuda` pass through to nd-fmm-exec and
nd-fmm-validate. No external dependency; anything else needs asking first.

## Test oracle
`nd_fmm_validate::fmm_accuracy::Oracle` (`nd_fmm_ref::p2p::direct_sum` in f64, over the
f32-rounded charges in f32) for the error columns; hand-made reports for the Markdown
writer.
