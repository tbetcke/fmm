# Phase 4 / T12 — autotune of the M2L strategy, with a persistent cache (C4.7)

Which M2L strategy is fastest depends on the device, the precision, p and the batch
sizes (design §4, §6.7). On Metal f32 the spike expects dense at every p. For f64 the
spike's roofline model places the crossover near p ≈ 10, with nothing measured. This
task registers the device strategies and their kernel variants as candidates. It times
them at build and picks the fastest per key, and it persists the result, so that
production runs do not re-tune. The choice is resolved when the `Fmm` is built and then
fixed, so the output stays deterministic (requirement 6). Where no tuning result
exists, the static rule applies.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("Strategy selection", requirement 6, "Exit gate" C4.7);
- docs/design/device-path.md, signed off: section 10 (the tuner chosen there, its key,
  candidates and persistence), and the f64 rule for p = 9–11 as signed off;
- docs/design/laplace-fmm-plan.md §4, §6.5 and §6.7;
- `nd_fmm_tables::cache` (`TableCache`: versioned keys, checksum, stale and corrupt
  files rejected, safe for concurrent writers). The tuning cache follows the same
  rules;
- `FmmBuilder::table_cache` (no default directory, no environment variable);
- the T9, T10 and T11 code and their timing reports;
- CubeCL 0.11's autotune and its cache, if the design uses them.

Do:
- **Candidates**, per the design: dense M2L with the library GEMM (f32, where CMMA
  applies), dense with the hand-written GEMM (per cube layout), and rotation. Also the
  P2P tile size and the GEMM layouts, if the design includes them. Only candidates that
  pass T9's input precision guard are registered: no library strategy that rounds f32
  inputs to TF32, F16 or BF16 is ever timed or chosen, however fast it is.
- **Key**, per the design: backend, device name, precision, p and a bucketed batch size
  (boxes or pairs per level). Also the CubeCL version and `CONVENTION_VERSION`, so that
  a cache from another build is stale.
- **Tuning** at build, when the backend is a device and tuning is enabled:
  - time each candidate on representative batches of this `Fmm`'s plan, with kernel
    compilation excluded and launches queued as T11 queues them;
  - take the median of repeated batches;
  - pick the fastest, and fix it for the `Fmm`'s lifetime.
  - Tuning time is reported in `BuildTimings` and never inside `evaluate`. It is
    bounded: state the budget and how it is enforced.
- **Persistence**: `FmmBuilder` accepts a tuning-cache directory, as `table_cache`
  does: no default directory, no environment variable.
  - Every rank may read and write it safely.
  - A stale entry (another key, CubeCL version, convention version, or corrupted) is
    rejected and re-tuned, never trusted.
  - Without a directory, results live only as long as the `Fmm`, or tuning is off, as
    the design says.
- **Static fallback rule** for an untuned key (README, "Strategy selection"):
  - f32: dense, with the library GEMM where CMMA applies and the hand-written GEMM
    otherwise;
  - f64: dense for p ≤ 8 and rotation for p ≥ 12, with the signed-off rule for 9–11.

  This rule is what f64 on CUDA gets, since nothing here can tune it. Document it as
  provisional (Phase 0 T6 model).
- **Determinism**: every candidate is chosen at build, and `Fmm` reports the resolved
  strategy, GEMM and layout per level. Two `Fmm`s built from the same cache make the
  same choices and give bit-identical output. Without a cache, a re-tune can choose
  differently, and the docs say so.
- `nd-fmm-validate`: an example or flag that prints the tuning table (candidate times
  per key) for the cube and the Plummer sphere on Metal f32 at p = 3, 6 and 8, and on
  the CPU runtime f64 at p = 4, 8, 12 and 16 (for correctness of the machinery only;
  CPU-runtime timings say nothing about a GPU).

Tests that define done (CPU runtime, and Metal by hand; each prints the backends it ran):
- With a candidate set where one candidate is made artificially slow by a test hook,
  the tuner picks another; the choice is reported and then fixed across evaluations.
- A library strategy with lower input precision than T (TF32, F16 or BF16 for f32) is
  never registered as a candidate, even when it is offered and the test hook makes it
  the fastest.
- Cache:
  - the round trip gives the same choices;
  - a cache from another key, CubeCL version or convention version is rejected and
    re-tuned;
  - a corrupted or truncated file is rejected without a panic;
  - two concurrent writers leave a valid file (as `TableCache`'s test does);
  - no file is written without a directory.
- The static rule is used exactly when no tuning result exists, and gives the strategies
  above (unit tests over p and precision).
- Outputs: every tuned choice gives an output within the README's FMM bounds of the host
  output, on the debug scenarios. Two builds from the same cache are bit-identical.
- Tuning time stays within the stated budget at the C3.2 size.

Must pass:
- `cargo test -p nd-fmm-kernels --features cpu --release`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged), and
  `--features cpu --release`, and on Metal by hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- clippy on the three crates, without features and with `--features cpu,metal`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report: the tuning tables on Metal f32 (and the CPU runtime f64, marked as not
representative); the chosen strategy per key; the tuning time; and the static rule as
implemented, with what remains unmeasured (f64 on a GPU).

Do not:
- tune inside `evaluate`, or let a choice change during an `Fmm`'s lifetime;
- read environment variables, or write anywhere without a caller-supplied directory;
- change the host strategy `Auto` or its rule;
- assert timings. Tests use the hook, not real speed differences.
