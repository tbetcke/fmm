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
- the T9, T10 and T11 code and their timing reports.

Do:
- **Candidates**, per the design: dense M2L with the library GEMM (f32, where CMMA
  applies), dense with the hand-written GEMM (per cube layout), and rotation, with
  `Dense` only where its tables fit. Also, per level, the GEMM and the offset-chunk
  budget, and for P2P the units per cube (32, 64, 128) and the plane-per-leaf layout,
  and the CPU layout's K if decision 10 added it (device-path.md §10.3). Only
  candidates that pass T9's input precision guard are registered: no library strategy
  that rounds f32 inputs to TF32, F16 or BF16 is ever timed or chosen, however fast it
  is.
- **The coefficient-major GEMM layout** (deferred from T9 on 2026-10-04, README decision
  12): a gather that writes X with each coefficient's values across the batch
  contiguous, so that the library and the hand-written GEMM see the spike's orientation
  (M = n, N = columns) instead of device-path.md §6.4's (M = columns, N = n), with the
  reduction reading Y in the same layout. T9 measured the orientation alone at 71–84%
  of the spike's for the library (spikes/cubecl-gemm, same process); its
  `m2l_kernels` example is the baseline. Add it as a candidate layout for M2M, L2L and
  M2L if it is faster, keeping the summation order inside each product fixed and
  documented; it changes §6.4's layout, so record the outcome for the design update.
- **Key**, per the design (device-path.md §10.2): backend, device name and compiler,
  precision, p and gradients; for the GEMM and layout choices the level's V (or
  octant) pairs bucketed to a power of two, and for the P2P layout the mean points per
  leaf, likewise. Also the CubeCL version, the `nd-fmm-kernels` version,
  `CONVENTION_VERSION`, the candidate set and the file format, so that a cache from
  another build is stale.
- **Tuning** at build, after the uploads, only when the backend is a device, a
  tuning-cache directory is given and the cache lacks the key (device-path.md §10.4):
  - time each candidate on this `Fmm`'s own largest level of that kind, with kernel
    compilation excluded (one warm-up launch) and launches queued as T11 queues them;
  - take the median of five batches of at least 10 ms each;
  - pick the fastest, and fix it for the `Fmm`'s lifetime.
  - Tuning time is reported in `BuildTimings::device` and never inside `evaluate`. It
    is bounded: 10 s per build by default, enforced by a deadline checked between
    candidates, with the static rule's candidate timed first so that a cut leaves at
    least it. Without `table_cache`, the dense candidate at p ≥ 12 is skipped.
- **Persistence**: `FmmBuilder::tuning_cache(dir)` accepts a tuning-cache directory,
  as `table_cache` does: no default directory, no environment variable. One file per
  (backend, device, precision, p), with the `TableCache` rules (device-path.md §10.5).
  - Every rank may read and write it safely.
  - A stale entry (another key, CubeCL version, convention version, or corrupted) is
    rejected and re-tuned, never trusted.
  - Without a directory, no tuning runs and the static rule applies.
- **Static fallback rule** for an untuned key (README, "Strategy selection"):
  - f32: dense; M2M and L2L with the library GEMM where CMMA applies and the
    hand-written GEMM otherwise, M2L with the hand-written GEMM (README, "M2L
    strategies", changed on 2026-10-04 after T9);
  - f64: dense for p ≤ 11 and rotation for p ≥ 12 (device-path.md §10.5).

  This rule is what f64 on CUDA gets, since nothing here can tune it. Document it as
  provisional until a GPU measures rotation (Phase 0 T6 model). Enter T10's measured
  Metal f32 rotation efficiency into the model and report what it implies; moving the
  boundary is a sign-off decision.
- **Determinism**: every candidate is chosen at build, and `Fmm` reports the resolved
  strategy, GEMM and layout per level. Two `Fmm`s built from the same cache make the
  same choices and give bit-identical output. Without a directory the static rule
  applies on every build. A re-tune that replaces a stale entry can choose
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
  - no file is written, and no tuning runs, without a directory.
- The static rule is used exactly when no tuning-cache directory is given, and gives
  the strategies above (unit tests over p and precision).
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
- call CubeCL's autotune (`LocalTuner`) or enable its `persistence` feature
  (device-path.md §10.1, §3.4);
- change the host strategy `Auto` or its rule;
- assert timings. Tests use the hook, not real speed differences.
