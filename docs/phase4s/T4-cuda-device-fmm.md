# Phase 4S / T4 — the device FMM on CUDA: the Phase 4 gates in f32 and f64 (C4S.4)

T2 makes the kernels run on CUDA. This task runs the whole device path of `nd-fmm-exec`
on the H100: the device operator, the evaluation boundary, autotune and the C4.8 gate.
It makes the Phase 4 acceptance tests that ran on Metal (f32) and the CPU runtime (f64)
also run on CUDA, in **both** precisions. It is the first time f64 runs on a GPU here.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-kernels/CLAUDE.md (with T2's "CUDA" section),
  tools/gh200/README.md;
- docs/phase4s/README.md ("Requirements", "Accuracy measures", "Workloads");
- docs/phase4/README.md ("Requirements on the device path", "Accuracy measures", "Exit
  gate" C4.1 and C4.8);
- the T2 and T3 reports, and the T3 sign-off;
- `fmm-exec/tests/device_metal.rs`, `device_fmm.rs`, `device_tune.rs`, `tune_common/`,
  `device_common/`, `operator/device_*.rs`, `accuracy.rs`, `adaptive.rs`: what each runs
  on which backend and why;
- docs/design/device-path.md §4.1, §7.2, §8, §10, §12, §13.

Do:
- **A CUDA test executable**, `tests/device_cuda.rs`, the counterpart of
  `tests/device_metal.rs`: one MPI-initialising test, `#[ignore]`d, run by hand on
  locust. It runs every Metal scenario through `device_common::check_backend`:
  - with every kind on the host fallback, the output bit for bit the host path's, and
    the transfers of the formula;
  - with the default placement, the FMM bounds against the host, determinism, and the
    transfers;
  - in **f32 and f64**. Metal refuses f64, so the f64 refusal scenario becomes "f64 is
    accepted on CUDA";
  - the tuner scenarios of `tune_common` at f32 p = 8 and f64 p = 6, where Metal ran f32
    only.

  Where the Metal file's scenarios are Metal-specific, such as the library GEMM in use,
  the CUDA file states what CUDA does instead (the library rejected by the guard, the
  hand-written GEMM in its place) and checks that.
- **The operator tests** of `tests/operator/device_*.rs`, gated today on `metal`, gain
  CUDA arms.
- **The C4.8 gate on CUDA**, in `tests/device_fmm.rs`: a CUDA block beside the CPU-runtime
  and Metal blocks, with the cube and the Plummer sphere at N = 10⁵, `max_level` 16, 64
  per leaf, eight charge vectors, gradients:
  - f32 at p = 3 and 8;
  - f64 at p = 8, 12 and 18;
  - the Gaussian clusters at p = 8 in both precisions.

  Against the host of the same settings:
  - the output within 1e-12 (f64) and 1e-5 (f32) relative L2;
  - the errors against the direct sum within 0.1% (f64) and 5% (f32) of the host run's;
  - two evaluations bit for bit;
  - the transfers the design's minimum;
  - the tuning budget at the C3.2 size, as on Metal.

  Also add N = 10⁶ on the cube at f32 p = 8 and f64 p = 8: the GPU has 96 GB and the
  host direct sum at 1,000 targets is cheap.
- **The ignored accuracy gates** `tests/accuracy.rs` and `tests/adaptive.rs` repeat their
  problems on CUDA with every kind on the device, f32 and f64, as they do on the CPU
  runtime (f64) and Metal (f32).
- **`DeviceTimestamps` on CUDA.** With `device_timestamps(true)`, check that the stage
  windows add no sync and do not change the output. Check whether neighbouring windows
  overlap on CUDA, as they did on Metal (T11). If T5 has merged, run the same check for
  `KindTiming::Device` per call. Report it in fmm-exec/CLAUDE.md.
- **Grace as the host.** `nd-fmm-validate`'s machine lines (`bench::cpu_model`,
  `cores`) print "unknown" on Grace: its `/proc/cpuinfo` has no `model name`. Make them
  name the CPU on Linux aarch64, from `/proc/cpuinfo`'s `CPU implementer`/`CPU part`
  (0x41/0xd4f is Neoverse-V2) or `/sys/devices/system/cpu`, without a new dependency.
  Make `--threads` default to the cores on Linux (it already falls back to
  `available_parallelism`; check it). Use one small commit.
- **`fmm_accuracy --backend cuda`** and **`device_fmm --device cuda`** run on locust, f32
  and f64, at the default sizes, as a smoke check. The full runs are T8.
- **Fix what fails.** Each fix gets its own commit with a test, on the CPU runtime if the
  defect shows there. If a gate fails, report the measured errors with the breakdown by
  kind (host fallback per kind, Phase 4 README) and stop. If a failure follows from a T3
  finding that the sign-off has not settled, say so and stop.
- **Root CLAUDE.md and fmm-exec/CLAUDE.md:** the "On locust (by hand)" checks, in the
  style of the Metal ones:
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release --
  --ignored`, run through `tools/gh200/remote.sh`.

Tests that define done:
- On locust, by hand: every new CUDA test passes in f32 and f64, with the backends run
  printed ("backends run: cpu, cuda"). That means `tests/device_cuda.rs`, the CUDA blocks
  of `tests/device_fmm.rs` and `tests/device_tune.rs`, the operator tests, and
  `tests/accuracy.rs` and `tests/adaptive.rs` on CUDA.
- On the M3 Max: every existing test still passes (host, CPU runtime, Metal by hand),
  and the CUDA blocks report "not compiled".

Must pass:
- `cargo fmt --all`, then the root checks and the stricter workspace checks on the M3
  Max;
- the device-path checks of fmm-exec/CLAUDE.md on the M3 Max;
- on locust:
  - `cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda -- -D warnings`;
  - `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release`, and
    with `-- --ignored`;
  - the T2 kernel suite again.

Do not:
- change defaults, layouts, the tuner's candidates or the static rule (T7);
- change a tolerance to pass;
- add a CUDA job to CI (decision 6 covers only a type-check).
