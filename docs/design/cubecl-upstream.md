# Observations for the CubeCL developers

A collection, not a report: observations about CubeCL that may be worth raising upstream
(tracel-ai/cubecl), gathered so that a summary for the CubeCL developers can be written
from one place. Every entry names the version it was seen with, the evidence and where the
full numbers are. Nothing here has been filed upstream unless the entry says so. Moving
off the pinned `=0.11.0-pre.4` stays a separate decision (root CLAUDE.md).

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

### Earlier observations, recorded elsewhere (pointers; re-check before quoting)

- `PLANE_POS` not lowered for NVPTX; the compile panic is dropped and the launch fails
  silently (a later `read_one` returns stale data as `Ok`). Fixed upstream by
  tracel-ai/cubecl#1714 (merged 2026-09-25, not yet in a release). device-path.md F29;
  fmm-kernels/CLAUDE.md, "CUDA".
- wgpu-msl falls back to WGSL silently when the GPU family check fails, and `client.name()`
  says `wgpu<msl>` either way. device-path.md F4.
- Client creation can panic (wgpu "No possible adapter available"), with no `try_`
  variant; allocation failure panics. device-path.md F3, F7.
- `cubecl-metal` (`metal-native`) 0.11.0-pre.4 does not compile (`DebugInformation` not
  imported in `src/compute/context.rs`). spikes/device-arith/Cargo.toml, REPORT.md.
- cubecl-opt's `InstCombinePass` fuses every lone product feeding an add into an fma on
  every backend, regardless of fast-math flags. device-path.md F16, F34;
  spikes/device-arith/REPORT.md.
- wgpu-msl compiles MSL with Metal's default fast math (no `MTLMathMode` set): `sqrt` and
  division not correctly rounded, subnormals flushed. device-path.md F16;
  spikes/device-arith/REPORT.md.
- `cubek-matmul`'s CMMA strategies are `Unavailable` on the CUDA LLVM path at every shape
  tried; `Strategy::Auto` panics on errors other than `Unavailable`. device-path.md F20,
  F28.
- An empty timing window measures time on CUDA (events) but nothing on wgpu.
  device-path.md F30.
