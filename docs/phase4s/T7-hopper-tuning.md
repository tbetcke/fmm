# Phase 4S / T7 — the device path tuned for Hopper (C4S.7)

After T2 and T4, CUDA runs correctly with the Metal GPU defaults, which were measured on
a 40-core laptop GPU with 32 KB of shared memory:
- P2P cubes of 64 units;
- leaf tiles of 32 points;
- GEMM tiles of 32 rows × 64 units × 4 columns;
- the 128 MB scratch budget;
- the 65,535-cube grid cap.

An H100 has 132 SMs, about 227 KB of shared memory per block, 1,024 threads per block,
fast f64, and a much cheaper sync. This task measures where the Metal defaults leave the
H100 idle, and adds CUDA layouts, tuner candidates and static defaults where measurement
shows a gain. It also settles, by measurement, the provisional f64 dense/rotation rule
of Phase 4.

It changes **layouts and choices only**. The formulation, the accumulation order,
determinism and the results' bits under a given layout stay as signed off.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-bench/CLAUDE.md;
- docs/phase4s/README.md ("Timing", "Peaks", decision 9);
- the T2, T3, T4, T5 and T6 reports;
- docs/design/device-path.md §6 (every kernel's mapping), §10 (the tuner), §13.4 (the P2P
  peak model), §17;
- `nd_fmm_exec::tune` (candidates, keys, the static rule, `STATIC_F64_DENSE_MAX_P`);
- the kernel harness examples of `nd-fmm-validate`: `p2p_kernels`, `leaf_kernels`,
  `translation_kernels`, `m2l_kernels`, `rotation_kernels`, `autotune`.

Do:
- **Peaks and a model for the H100.**
  - The examples hard-code the M3 Max: `PEAK_GFLOPS = 14_300` in `m2l_kernels`,
    `rotation_kernels` and `translation_kernels`, plus `BREAK_EVEN_F32_M3` and the
    spike's M3 numbers. Make the peak a per-device value: a small table keyed by
    `DeviceInfo`'s name, with "unknown peak" for anything else. The M3 Max output must
    not change.
  - For the GH200's H100, use the datasheet: f32 (non-tensor), f64 (non-tensor; CubeCL
    has no FP64 MMA), and HBM3 bandwidth. Read the clocks with `nvidia-smi -q -d CLOCK`
    and say which clock the peak assumes. Label the peaks "datasheet".
  - Derive the P2P peak model for the H100 as device-path.md §13.4 did for the M3 Max
    (lanes × clock / operations per pair, f32 and f64).
- **Measure first, per kernel, on CUDA** (the T2 harnesses, release, on locust, GPU idle):
  - **P2P:** Gpairs/s on W1 and W2, against the model; `Cube` units ∈ {32, 64, 128,
    256}, `Plane` with 2–8 leaves per cube, and the tile sizes the shared memory now
    allows.
  - **Leaf operators:** units and tile against the shared-memory limit.
  - **GEMM (M2M, L2L, dense M2L):** GFLOP/s per level and p against the f32 and f64
    peaks; rows, units and columns per unit; larger tiles that fit 227 KB. Is the
    hand-written kernel compute-bound or memory-bound at each p?
  - **Rotation:** units per cube, and several targets per cube at small p (device-path.md
    §6.6 left it as a candidate).
  - **Elementwise kernels:** whether `GPU_MAX_CUBES` = 65,535 costs anything on CUDA.
  - **Scratch budget:** 128 MB against 512 MB to 2 GB (96 GB HBM).
  - **Launches and syncs:** the cost per launch and per sync on CUDA, measured as on
    Metal. Is a sync now cheap enough that the `Synchronous` kind timings of T5 barely
    perturb the total?

  Profile the slowest kind of the FMM with Nsight Compute (`ncu`) or Nsight Systems
  (`nsys`) from the CUDA toolkit. If performance counters are restricted to
  administrators on locust, say so and use CUDA event timings alone.
- **Then change, where the measurement shows at least a 10% gain on the FMM level that
  uses it:**
  - add `Cuda`-specific defaults in the `match` arms that today read `Metal | Cuda`
    (`p2p.rs`, `leaf.rs`, `translate.rs`, `rotation.rs`, `device.rs`), with a comment
    citing the measurement;
  - add CUDA candidates to the tuner (`tune.rs`), keeping the tuning budget;
  - **new layouts are comptime variants of existing kernels**: the same owners, the same
    order, the same arithmetic. Every new layout runs the existing bit-for-bit and
    tolerance tests on the CPU runtime, Metal and CUDA. A faster kernel that needs a
    different order or formulation is reported, not built (fmm-kernels/CLAUDE.md);
  - a change to a Metal default needs the same evidence on the M3 Max, and is out of
    scope unless asked.
- **The static rule on CUDA** (decision 9). Measure the dense against the rotation M2L,
  f64 at p ∈ {4, 6, 8, 10, 11, 12, 14, 16, 18, 20} and f32 at p ∈ {2, …, 10}, on the C3.2
  cube at N = 10⁵ and 10⁶. Use `rotation_kernels` per level, and `nd-fmm-bench` end to
  end with `--strategy dense,rotation`.
  - Propose the static rule for CUDA from these numbers: per precision, the p at which
    rotation wins.
  - The rule must be keyed by backend. Today it is one rule for every GPU (`tune.rs`),
    and Metal's must not change.
  - The change is made in this task only after the sign-off of decision 9. Until then,
    report the proposal.
- **End to end.** `nd-fmm-bench --backend cuda` at N = 10⁶, f32 p ∈ {3, 6, 8}, f64
  p ∈ {3, 6, 8, 12}, before and after the changes, with the kind table. This is the
  task's headline: the speed-up per kind and overall.
- **Docs:**
  - device-path.md §18.2, "Layouts and rules on CUDA as measured": a table of each
    kernel's Metal default, CUDA default and the measurement behind it, with the f64
    rule;
  - fmm-kernels/CLAUDE.md: the CUDA defaults;
  - fmm-exec/CLAUDE.md: the CUDA tuner candidates.

Tests that define done:
- Every new layout is covered by the existing kernel tests (each layout on each backend,
  CPU runtime included), and all pass on the M3 Max (CPU runtime, Metal by hand) and on
  locust (CUDA).
- The tuner tests (`tests/device_tune.rs`, `tune_common`) pass on CUDA with the new
  candidates, within the budget.
- T4's C4.8 CUDA gate still passes with the new defaults: f32 and f64, output within the
  bounds, deterministic.
- With CUDA's defaults changed and Metal's not, the Metal and CPU-runtime outputs are
  bit-identical to before this task (check on the `mpi_exec` scenarios).

Must pass:
- `cargo fmt --all`, then the root checks and the stricter workspace checks;
- the kernel and device-path checks of the root CLAUDE.md, fmm-kernels/CLAUDE.md and
  fmm-exec/CLAUDE.md, on the M3 Max;
- on locust: the CUDA commands of T2 and T4.

Do not:
- change a kernel's formulation, summation order or arithmetic rules, or the tolerances;
- change the Metal or CPU-runtime defaults, or the host path;
- use TF32, F16 or BF16 inputs, or any library strategy that rounds f32 inputs (Phase 4
  rule; the guard stays);
- overlap kinds on several CUDA streams (it changes the accumulation order; out of
  scope, README);
- assert timings.
