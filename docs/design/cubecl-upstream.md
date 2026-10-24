# Observations for the CubeCL developers

A collection, not a report: observations about CubeCL that may be worth raising upstream
(tracel-ai/cubecl), gathered so that a summary for the CubeCL developers can be written
from one place. Every entry names the version it was seen with, the evidence and where the
full numbers are. Nothing here has been filed upstream unless the entry says so. Moving
off the pinned `=0.11.0-pre.4` stays a separate decision (root CLAUDE.md). That decision
was Phase 5S T1s (2026-10-11): the pin is `=0.11.0` (`cubek-matmul`/`cubek-std` `=0.3.0`).
T1s re-checked every entry against the 0.11.0 sources; each carries a **0.11.0** line
(still present, fixed or changed, with the 0.11.0 file and line; paths are relative to
the crate's directory in the cargo registry).

Started in Phase 4S T11 (2026-10-08). Add an entry whenever a task finds something
upstream-worthy; keep the source document as the place for the full numbers.

## Entries

### 1. CUDA: pinned staging of host-to-device copies below 100 MiB costs more than it saves on GH200

- **Version:** CubeCL 0.11.0-pre.4, CUDA through LLVM NVPTX, CUDA 12.6, driver 565.57.01,
  NVIDIA GH200 480GB (Grace with an H100, NVLink-C2C).
- **Where in CubeCL:** `cubecl-server` `src/command/staging.rs`, `Staging::of`: data
  under `STAGE_MAX` (100 MiB) that is not already pinned is first copied into a slice of
  the stream's pinned pool (`Command::write_to_gpu` → `reserve_pinned`,
  `Bytes::copy_into`), then sent with `cuMemcpyHtoDAsync`; data above it is sent from
  where it is.
- **Observation** (measured, locust, `Client::write` of a `Vec<f32>` handed over with
  `Bytes::from_elems`, then `Client::sync`; median of 10):

  | size | path | time from `write` to the end of `sync` |
  | ---: | --- | ---: |
  | 40 MB | pinned staging (below `STAGE_MAX`) | 2.73 ms |
  | 80 MB | pinned staging | 5.14 ms |
  | 160 MB | direct from pageable memory (above `STAGE_MAX`) | 1.65 ms |

  The device-to-host copy of the same sizes into pinned memory runs at about 290 GB/s
  (0.55 ms for 160 MB), so the 40 MB transfer itself should take about 0.14 ms; the rest
  is consistent with the serial staging `memcpy` on the server thread at about 16 GB/s
  (one Grace core; *inferred* from the times, not profiled per step). On this machine the
  larger, unstaged write finishes sooner than a write a quarter of its size.
- **Why it matters here:** the charge upload of every FMM evaluation (40 MB at N = 10⁷ in
  f32, 80 MB in f64) takes the staged path; the staging copy is about 2.5 / 4.8 ms of a
  60 / 103 ms evaluation.
- **No workaround through the public API:** `Bytes` that are already pinned
  (`AllocationProperty::Pinned`) skip the staging, but only CubeCL makes them (the results
  of `read`, or `Client::staging`, which itself copies); the threshold is a constant.
- **Possible questions:** a lower or configurable `STAGE_MAX` (or none) on devices with
  a coherent, fast host link such as GH200 or GB200; a way for a caller to obtain pinned
  `Bytes` to fill directly (for example a public `Client::reserve_pinned(size)`).
- **Full numbers:** spikes/download-path/REPORT.md, "The upload, step by step" and
  "Upstream"; raw output in spikes/download-path/results-gh200.md.
- **0.11.0: still present** (source; not re-measured). `cubecl-server`
  `src/command/staging.rs` is unchanged: `STAGE_MAX` = 100 MiB (line 15), the staging
  test at line 43; `Command::write_to_gpu` at `src/command/base.rs:446` (pre.4: 359)
  still stages through the private `reserve_pinned` (base.rs:299). No public
  `Client::reserve_pinned` and no configuration of the threshold; `Client::staging`
  (`cubecl-runtime` `src/client.rs:844`) still copies.

### 2. CUDA and wgpu: host pools grow on first use; `memory_persistent_allocation` does not reach them

- **Version:** 0.11.0-pre.4, CUDA (locust) and wgpu-msl (M3 Max).
- **Observation:** the pinned pool (CUDA) and the staging pool (wgpu) allocate only when
  they grow and are reused afterwards: one `cuMemAllocHost` of 58–61 ms (or about 15 ms,
  depending on the pool serving the size) for the first download of a size, none for
  later ones (`nsys`, 1 against 10 downloads). `Client::memory_persistent_allocation`
  switches only the device pool's mode (`device_memory().mode` on CUDA, the main pool on
  wgpu), so it does not cover the host pools; measured identical. Not a defect; possibly
  worth documenting, or a way to reserve a host-pool size up front.
- **Full numbers:** spikes/download-path/REPORT.md, "CubeCL's host pools".
- **0.11.0: still present** (source). CUDA: `cubecl-cuda` `src/compute/server.rs:409-416`
  sets only `device_memory().mode` (now also for streams created later); wgpu:
  `cubecl-wgpu` `src/compute/server.rs:728` sets the stream's main pool, while staging
  moved to a separate auxiliary memory (`src/compute/stream.rs:84-86`, `reserve_staging`
  at 402) that the mode does not reach. New: `Client::memory_dedicated_allocation`
  (`cubecl-runtime` `src/client.rs:638`).

### 3. With the profiling log on, a failed launch panics at the launch

- **Version:** 0.11.0-pre.4; seen on the CPU runtime (CI, then the M3 Max) and Metal.
- **Observation:** with `CUBECL_DEBUG_LOG` set, the profiling logger is at a timing level
  and `Client::launch_inner` profiles every launch synchronously; a launch that fails (a
  kernel that does not compile, a cube larger than the device allows) panics there with
  "An execution error happened during profiling", by design ("the logger's timing levels
  opted into profiling and keep their loud failure"). Without the log the same failure is
  attached to the output buffer and returned by the next `read_one` (F9). The failure is
  attached in both cases, so a caller that catches the panic still sees the error at the
  download. Turning on a debug log thus changes the error behaviour of a program that
  handles launch errors; worth documenting, or a non-panicking mode for logging.
- **Where it showed:** `nd-fmm-kernels`'s launch-error test (Phase 4S T11), which failed
  in the CPU-runtime CI job (it sets `CUBECL_DEBUG_LOG` for the compile-time report) and
  passed locally without the log; PR #74.
- **0.11.0: still present.** `cubecl-runtime` `src/client.rs:1207` (pre.4: 1147) keeps the
  `panic!` under the same comment; seen again in T1s, where the CPU-runtime run with
  `CUBECL_DEBUG_LOG` set printed "the launch panicked in CubeCL's profiler (its log is
  on)" in `nd-fmm-kernels`' launch-error test, which passes as before.

### 4. CUDA: an f32 CMMA runs at TF32 precision while reporting F32 types

- **Version:** CubeCL 0.11.0, `cubek-matmul` 0.3.0, CUDA through LLVM NVPTX, CUDA 12.6,
  NVIDIA GH200 480GB (Phase 5S T1s, 2026-10-12). Not seen with 0.11.0-pre.4, where the
  same probe failed (F28: "No tile size is available for the problem").
- **Observation:** `launch_ref` with `multi_level::Strategy::SimpleCyclicCmma` on f32
  operands (`MatmulElems::from_single_dtype(f32)`, the call of `nd-fmm-kernels`'
  `library_gemm`) succeeds and leaves every stage and register type of the
  `MatmulElems` at F32, yet the result has TF32 accuracy: the largest error relative to
  Σ|x b| is 1.5–2.1e-4 over batched shapes with n = 81 and 100 (f32 rounding would give
  about 1e-7; TF32's unit roundoff is 4.9e-4). The device registers no CMMA form with
  f32 operands, only F16, BF16 and TF32 16 × 16 × 8 (`features.matmul.cmma`);
  `type_usage(tf32)` is `Conversion` only, `supports_type(tf32)` is true. 0.11.0 keeps the
  TF32 CMMA form on NVPTX (`cubecl-llvm` `src/shared/lowered_features.rs:55-96`), where
  pre.4 kept only F16 (`cubecl-cuda` 0.11.0-pre.4 `src/runtime.rs:436-443`);
  `cubek-matmul`'s `adjust_dtypes` (`src/multi_level/definition/blueprint.rs:89-113`)
  switches to TF32 only on its own copy of the element types, which the caller's
  `MatmulElems` does not see.
- **Why it matters here:** `nd-fmm-kernels`' input-precision guard read the resolved types
  only, so on 0.11.0 the f32 GEMM of M2M and L2L (n ≥ 81) under `GemmPolicy::Auto` went to
  the library: two kernel tests failed (the library's error 1.48e-3 against the bound
  6.57e-4), the CUDA f32 p = 8 FMM output changed in every value (up to 1.3e-4 of the
  largest |φ|), and the C3.2 gate on CUDA failed (device error 17.9 times the host's).
  T1s added a second part to the guard: the device must register a CMMA form whose
  operands are T (`registers_cmma` in `fmm-kernels/src/translate.rs`); CUDA f32 then
  runs the hand-written kernel again, as on pre.4, and Metal keeps the library (its f32
  output bit for bit pre.4's).
- **Possible questions:** report the precision actually used through `MatmulElems` (or
  refuse an f32 request on a device without an f32 CMMA form unless the caller opts into
  TF32); document that f32 inputs may be rounded to TF32 on CUDA.
- **Full numbers:** fmm-validate/results/phase5s-t1s.md.

### 5. Metal: a matmul with few rows is refused as "Plane dimension unsupported: 32"

- **Version:** CubeCL 0.11.0, `cubek-matmul` 0.3.0, wgpu-msl on the M3 Max (seen
  2026-10-24 in an `nd-fmm-bench` run at f32 p = 8, revision aff3514); the same message
  in the Phase 4 and Phase 4S reports under 0.11.0-pre.4.
- **Observation:** `launch_ref` with `multi_level::Strategy::SimpleCyclicCmma` on f32
  operands fails with `MatmulAvailabilityError::PlaneDimUnsupported { plane_dim: 32 }`,
  displayed as "Plane dimension unsupported: 32. Only 32 & 64 are supported." The plane
  size is not the cause. `infer_blueprint_plane`
  (`src/multi_level/routines/selector/plane.rs:103-127`) sizes the stage as
  `row_count = max_planes / (tile_factor · precision_factor)`, with the units per cube
  capped at 256 on macOS (line 106), so 256 / 32 = 8 planes; `precision_factor` is 2 for
  4-byte stages (line 112); and `tile_factor` is 8 whenever `m ≤ 4 tile_m` or
  `n ≤ 4 tile_n` (line 118). Metal registers only 8 × 8 × 8 CMMA forms (`cubecl-cpp`
  `src/metal/dialect.rs:72-104`), so every f32 problem with m ≤ 32 or n ≤ 32 gets
  `row_count` = 8 / 16 = 0, and the zero is reported as `PlaneDimUnsupported` with the
  device's plane size (lines 123-127). The error text is a fixed string
  (`src/definition/error.rs:150-153`), hence the contradiction. The tiny-problem path
  (`is_tiny`, `src/multi_level/routines/selector/mod.rs:16-25`, enabled by the simple
  routine) does not catch these shapes: it needs two of m, n, k at most 2 tile edges
  (16), and here only m is.
- **Where it showed:** `nd-fmm-kernels`' library probe (`probe_library` in
  `fmm-kernels/src/translate.rs`), box-major, so the matmul's m is the columns per group
  and n = k the order (p + 1)². Rejected on Metal at f32 p = 8 (n = 81): M2M and L2L level
  calls with 1 or 8 columns per group (`[8, 1, 81]`, `[8, 8, 81]`), and Phase 4's dense
  M2L at level 2 with 32 (`[316, 32, 81]`, also at n = 169 and 289); accepted from 64
  columns per group. Each rejected view runs the hand-written GEMM, so the outputs are
  correct and the cost is small (those level calls have 8 to 3096 pairs).
- **0.11.0: present** (source and run, above). Under pre.4 the same text appears in
  fmm-validate/results/phase4-m3max.md ("The device dense M2L per level") and
  fmm-bench/results/phase4s-t11-download-path.md; not re-checked against the pre.4
  sources.
- **Possible questions:** report a stage that does not fit as its own error (or as
  "no stage fits the problem") rather than `PlaneDimUnsupported`, whose message names a
  supported plane size; or select a smaller stage for small m or n instead of refusing.
- **Full numbers:** the level calls and their rejections in
  fmm-validate/results/phase4-m3max.md and fmm-bench/results/phase4s-t11-download-path.md;
  the 2026-10-24 run's report was not kept in the repository.

### Earlier observations, recorded elsewhere (pointers; re-check before quoting)

- `PLANE_POS` not lowered for NVPTX; the compile panic is dropped and the launch fails
  silently (a later `read_one` returns stale data as `Ok`). Fixed upstream by
  tracel-ai/cubecl#1714 (merged 2026-09-25, not yet in a release). device-path.md F29;
  fmm-kernels/CLAUDE.md, "CUDA". **0.11.0: fixed** (source): `cubecl-llvm`
  `src/shared/builtins.rs:136-141` sets `PlanePos` for every GPU target (pre.4
  `src/nvptx/builtins.rs:93-107` did not), with the runtime tests
  `test_kernel_topology_plane_pos` (`cubecl-core` `src/runtime_tests/topology.rs:103`)
  and `test_expansion_panic_fails_the_read` (`src/runtime_tests/launch.rs:309`). The
  kernels still derive the plane position from `UNIT_POS`; not exercised on CUDA.
- wgpu-msl falls back to WGSL silently when the GPU family check fails, and `client.name()`
  says `wgpu<msl>` either way. device-path.md F4. **0.11.0: fixed** (source): the name
  follows the compiler chosen (`cubecl-wgpu` `src/runtime.rs:612-616`, `wgpu<msl>` or
  `wgpu<wgsl>`), a failed family check logs a warning (`src/backend/metal.rs:170-173`),
  and a device pinned to Metal refuses instead of falling back (`src/runtime.rs:28-32`).
- Client creation can panic (wgpu "No possible adapter available"), with no `try_`
  variant; allocation failure panics. device-path.md F3, F7. **0.11.0: changed**
  (source): `Runtime::client` still panics (`cubecl-wgpu` `src/runtime.rs:51`, "Unable to
  acquire wgpu device"), but wgpu gains `try_init_setup` and `try_init_device`
  (`src/runtime.rs:493-573`, `WgpuInitError::AdapterUnavailable` at 296);
  `Server::initialize_memory` returns a `Result` (`cubecl-runtime` `src/server/base.rs:615`)
  and an allocation error is attached to the buffer and reported at the next sync
  (`src/client.rs:803-815`). Whether a real out-of-memory still panics is unverified.
- `cubecl-metal` (`metal-native`) 0.11.0-pre.4 does not compile (`DebugInformation` not
  imported in `src/compute/context.rs`). spikes/device-arith/Cargo.toml, REPORT.md.
  **0.11.0: fixed**: imported at `cubecl-metal` `src/compute/context.rs:6`; `cargo check`
  of `cubecl` 0.11.0 with `metal-native` passes on the M3 Max (T1s).
- cubecl-opt's `InstCombinePass` fuses every lone product feeding an add into an fma on
  every backend, regardless of fast-math flags. device-path.md F16, F34;
  spikes/device-arith/REPORT.md. **0.11.0: still present** (source):
  `cubecl-opt` `src/passes/inst_combine.rs` is unchanged (`lone_product` at 28,
  `FmaOp::new` at 80), added unconditionally by `cubecl-cpp` `src/shared/base.rs:289`,
  `cubecl-llvm` `src/shared/base.rs:441` and `cubecl-wgpu`
  `src/compiler/wgsl/compiler.rs:160`. The CPU runtime still compiles with
  `default<O3>` (`cubecl-llvm` `src/cpu/jit/engine.rs:142`) and no fma contraction at
  the LLVM level on the CPU target (`src/shared/to_llvm/math.rs:361-367`).
- wgpu-msl compiles MSL with Metal's default fast math (no `MTLMathMode` set): `sqrt` and
  division not correctly rounded, subnormals flushed. device-path.md F16;
  spikes/device-arith/REPORT.md. **0.11.0: still present** (source): `cubecl-wgpu`
  `src/backend/base.rs:174-186` passes the MSL through with no math options (wgpu 30
  has none); only `cubecl-metal` sets `MTLMathMode::Safe` (`src/compute/context.rs:73`).
- `cubek-matmul`'s CMMA strategies are `Unavailable` on the CUDA LLVM path at every shape
  tried; `Strategy::Auto` panics on errors other than `Unavailable`. device-path.md F20,
  F28. **0.11.0: changed**: the CMMA strategies now launch for f32 on CUDA, at TF32
  precision (entry 4); `Strategy::Auto` still panics on other errors (`cubek-matmul`
  0.3.0 `src/strategy.rs:85-98`).
- An empty timing window measures time on CUDA (events) but nothing on wgpu.
  device-path.md F30. **0.11.0: still present** (source): wgpu returns
  `ProfileError::NotMeasured` (`cubecl-wgpu` `src/compute/timings.rs:549-563`); CUDA still
  times with device events (`cubecl-cuda` `src/runtime.rs:211`). Not re-run.
