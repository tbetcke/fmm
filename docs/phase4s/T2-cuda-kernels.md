# Phase 4S / T2 — `nd-fmm-kernels` on CUDA: the tests run, the facts recorded (C4S.2)

Every kernel in `nd-fmm-kernels` is runtime-generic and has been type-checked with
`--features cuda` since Phase 4 T4. None has ever run on CUDA. The test harness cannot
run them there either:
- `tests/kernels/common.rs` is gated on `cpu` or `metal`;
- `tests_on!` has a `cpu:` and a `metal:` arm only;
- `backends_line` hard-codes CUDA as "type-checked, not run".

This task makes the kernel suite run on locust's H100 in f32 and f64, fixes what fails,
and records the CUDA facts that later tasks rely on. It also runs the GEMM spike on CUDA,
as Phase 4 documented.

The default layouts stay as they are (CUDA shares Metal's GPU defaults, below). Tuning
for Hopper is T7.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, tools/gh200/README.md (T1);
- docs/phase4s/README.md ("Requirements", "Design decisions", decision 4);
- docs/design/device-path.md §1.2 (F2, F9, F12, F13, F14, F17, F21), §3.1, §5, §6, §13;
- spikes/cubecl-gemm/SPIKE_REPORT.md ("CUDA run");
- the CubeCL 0.11.0-pre.4 sources for CUDA: `cubecl-cuda` (`src/runtime.rs`
  `DeviceService::init`, `src/compiler.rs`, `src/compute/context.rs`), `cubecl-llvm`
  (`src/nvptx/`), `cudarc-0.19.10/build.rs`.

What the code does on CUDA today (read on 2026-10-05; confirm on the device):
- **Opening the device.** `Device::open(Cuda)` calls `cubecl::Device::cuda(0)`. Device 0
  is hard-coded; locust has one GPU. A missing driver gives `NoDevice`. There is no
  CUDA counterpart of the Metal MSL check.
- **Every GPU default is Metal's.** The `Metal | Cuda` arms set:
  - P2P `Cube { units: 64 }` (`p2p.rs:124-133`);
  - leaf 64 units with tiles of up to 32 points (`leaf.rs:147-161`);
  - GEMM 32 rows, 64 units, 4 columns per unit (`translate.rs:236-254`);
  - rotation (p + 1)² rounded up to the plane size (`rotation.rs:289-298`);
  - elementwise `GPU_UNITS = 256` and `GPU_MAX_CUBES = 65,535`, "wgpu's limit per
    dimension" (`device.rs:282-284`), which also caps CUDA, so large launches
    grid-stride.

  Shared memory is read from `DeviceInfo` (on H100 CubeCL reports the opt-in maximum,
  about 227 KB, against 32 KB on Metal).
- **Compiler path.** CUDA compiles through LLVM to NVPTX by default (F17). That needs the
  `tracel-llvm` bundle (T1) and, for transcendental calls only, `libdevice.10.bc` under
  `CUDA_PATH`. The kernels call no sin, cos, exp or pow. On this path `inverse_sqrt` is
  the polyfill `1 / sqrt(x)` (spikes/device-arith/REPORT.md, "How each backend
  lowers"). NVRTC (`cubecl/cuda-cpp`) is not enabled anywhere (decision 4).
- **Precisions.** f64 is registered with full arithmetic. TF32 is registered for
  conversion, so `cubek-matmul` would switch f32 CMMA stages to TF32. Also, the LLVM
  backend keeps only f16 × f16 CMMA, so the f32 library probe is expected to fail first.
  Either way the input-precision guard keeps the library GEMM off on CUDA (device-path.md
  §6.5).
- **Timing.** `times_on_device()` is true (CUDA events). `client.name()` is `"cuda"`.
  It does not name the compiler path, and the tuning key's compiler field inherits that.
- **Alignment.** `LIBRARY_TABLE_ALIGNMENT = 256` exists for wgpu. CUDA's allocator
  aligns to 512. This matters only to the library path, which is off.

Do:
- **The harness.**
  - Gate `tests/kernels/common.rs` on `any(cpu, metal, cuda)`, and add a `cuda:` arm to
    `tests_on!`. CUDA tests are `#[ignore = "CUDA: run by hand on locust"]`, like Metal.
  - `backends_line` reports CUDA as run when it ran, and as "not compiled" or "not run"
    otherwise. It never says "passed" for a backend that did not run.
  - Register every kernel test on the `cuda:` arm: the same list as Metal, plus f64
    (Metal has no f64), so every f64 test that runs on the CPU runtime also runs on
    CUDA. The library-GEMM tests run on CUDA too: they must observe the guard rejecting
    the library and the hand-written fallback taking over, and must not panic.
- **Run** on locust (from `tools/gh200/remote.sh`, after `sync.sh`):
  `cargo test -p nd-fmm-kernels --release --features cuda -- --ignored --show-output`,
  and the CPU runtime there as well. Report per test file:
  - passed and failed tests;
  - the `DeviceInfo` line;
  - the first-launch (compile) times from `CUBECL_DEBUG_LOG` and `tools/compile_times.awk`;
  - the suite's wall time.
- **Fix what fails, smallest change first.** Typical suspects:
  - shared-memory sizes computed from a limit CUDA reports differently;
  - grid limits;
  - `sync_plane` and plane operations;
  - u32 index widths in grid-stride loops;
  - alignment of sub-buffer offsets (CUDA takes byte offsets on raw pointers);
  - f64 code paths never run on a GPU before.

  Each fix gets its own commit with a test that fails before it, on the CPU runtime
  where the defect can be shown there. A defect in CubeCL itself is reported with a
  minimal reproducer and worked around in `nd-fmm-kernels` only if the workaround is
  small; otherwise stop and report. Do not change a kernel's formulation, order or
  layout semantics without asking (fmm-kernels/CLAUDE.md).
- **Accuracy as measured, not as assumed.** The operator tolerances of Phase 4 (README
  "Accuracy measures": 1e-14 or twice the host error for dense f64 operators, 1e-13 for
  rotation and the leaf operators, 1e-5 in f32; the P2P contract) apply to CUDA as they
  stand. If a CUDA f64 test fails a tolerance, report the measured error and its
  breakdown and stop; T3 decides whether the arithmetic rules need a CUDA note. Bit
  identity:
  - the bit-for-bit tests (copies, scatters, frames, the hand-written GEMM and rotation
    as host `mul_add` replicas) must hold on CUDA;
  - if one does not, report which operation differs (LLVM NVPTX's `contract` flag, F17,
    is the first suspect).
- **The GEMM spike on CUDA:** `cargo run -p nd-fmm-spike-cubecl-gemm --release
  --no-default-features --features cuda -- --backends cuda`, f32 and f64 (with
  `--force-f64` if still needed). Keep the output as
  `spikes/cubecl-gemm/results-gh200-0.11.md` and add a short "CUDA on GH200" section to
  SPIKE_REPORT.md:
  - whether `Strategy::Auto` and the explicit strategies run at all in f32 on the LLVM
    backend;
  - the hand-written kernels' f64 GFLOP/s against the H100's datasheet peak (labelled);
  - the correctness check result.
- **Facts.** A new section in fmm-kernels/CLAUDE.md, "CUDA (Phase 4S)":
  - how to run the suite on locust;
  - the `DeviceInfo` values (plane size, shared memory, units per cube, cube-count
    limits, memory);
  - f64 and TF32 as registered;
  - the compiler path;
  - the test budget measured on the H100.

  Also a draft of device-path.md §18, "CUDA on Grace Hopper (Phase 4S)": §18.1 facts,
  with F-numbers continuing the F-table, read and confirmed on the device.
- **`GPU_MAX_CUBES`:** keep 65,535 for every backend unless a test shows a defect. Raising
  it on CUDA is a T7 tuning question.
- **CI** (decision 6, decided 2026-10-05). Add two steps to the `run-tests-kernels` job
  of `.github/workflows/run-tests.yml`, after its clippy step:
  - `cargo check -p nd-fmm-kernels --features cuda`;
  - `cargo check -p nd-fmm-exec --features cuda`.

  The job already caches the `tracel-llvm` bundle that `cubecl-cuda` needs. The runner
  has no `nvcc`, so cudarc assumes its default CUDA version, which is enough for a
  type-check. Measure the extra time cold and warm on the PR's CI runs, and report it.
  Update the root CLAUDE.md's description of the job. CUDA is type-checked in CI, never
  run.

Tests that define done:
- On locust, every kernel test passes on CUDA in f32 and f64, or each failure is reported
  with its measured error and stopped on. The closing line says "backends run: cpu,
  cuda".
- On the M3 Max, the suite still passes on the CPU runtime and on Metal (by hand), and
  the closing line reports CUDA as not compiled or not run.
- The spike's CUDA run passes its correctness check (1e-12 f64, 1e-5 f32).

Must pass:
- `cargo fmt --all`, then the root checks;
- the kernel checks of the root CLAUDE.md (clippy with `cpu,metal`, `cargo check
  --features cuda`, the CPU-runtime tests, the Metal run by hand);
- `cargo clippy -p nd-fmm-kernels --all-targets --features cuda -- -D warnings` (on
  locust or the Mac);
- on locust: `cargo test -p nd-fmm-kernels --release --features cpu,cuda -- --ignored
  --show-output` (`cpu` and `cuda` in one build, if they combine; otherwise one after the
  other, and say so).

Do not:
- change defaults, layouts or candidates for performance (T7), or `nd-fmm-exec` beyond
  what a fix needs (T4);
- enable `cuda-cpp` (NVRTC) or any other CubeCL feature without decision 4;
- loosen a tolerance to pass; report instead;
- time anything beyond the spike run and the suite's wall time.
