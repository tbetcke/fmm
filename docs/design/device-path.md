# The device path: design for Phase 4

As of 2026-10-03. Written for [docs/phase4/README.md](../phase4/README.md) (T1), and
**signed off on 2026-10-03** with every recommendation of Section 16 accepted.
**Updated at the end of Phase 4 (2026-10-04, T13):** Section 17 records the decisions as
taken in T2–T13 and the measured numbers that replace this document's models where
they differ; short notes in Sections 6.5, 8.1, 10.5 and 13.4 point to it. The design
sections are otherwise left as signed off. It is
the Phase 4 counterpart of [simd-p2p.md](simd-p2p.md) and of the Phase 3 design
[fmm-plan-redesign.md](fmm-plan-redesign.md), and it ties together what
[laplace-fmm-plan.md](laplace-fmm-plan.md) §6 sketches: where data lives during an
evaluation, how a device operator stays consistent with the `Evaluator`, how kernels
batch and order their work under the accumulation rule, and how precision, fallback,
autotune and testing fit.

Conventions of this document:
- "Design §x" is laplace-fmm-plan.md, "redesign §x" is fmm-plan-redesign.md, "README"
  is docs/phase4/README.md.
- CubeCL facts cite the 0.11.0-pre.4 crates (and `cubek-*` 0.3.0-pre.4) as
  `crate-version/path`, read from the published `.crate` files. *Inferred* marks a fact
  taken from code structure, names or third-party documentation rather than read line
  by line. Nothing here was compiled or run: T2 and T3 confirm what they touch.
- Every number marked *model* or *estimate* is not a measurement. Measured numbers name
  their source (the Phase 0 spike, Phase 3 T12, Phase 3S T7).
- CubeCL's vector type is `Vector<T, N>` in 0.10 and 0.11
  (`cubecl-core-0.11.0-pre.4/src/frontend/container/vector/base.rs`). This document
  says "vector" or `Vector`.

> Where this document and docs/CONVENTIONS.md differ, **the conventions file takes
> precedence.** Phase 4 changes no convention except the §3.13 addition that T3 drafts.

**Recommendation in one paragraph.** Add a crate `nd-fmm-kernels` that hides CubeCL
behind its own `Device`, `DeviceBuffer` and per-family launch wrappers, and a device
operator in `nd-fmm-exec` that wraps `LaplaceOperator` for its host fallback. On one rank
the `Evaluator` writes nothing outside operator calls except `reset`'s zeroing, so the
operator keeps every store on the device and `Fmm` tells it when an evaluation starts,
uploads the charges and downloads the output once: two transfers and one sync per
evaluation, and **no `nd-fmm-plan` change (no T4b)**. The device backend is refused on
more than one rank until C5.1, for which Section 4.5 sketches the plan hook. M2M, L2L and
dense M2L share one *grouped translation*: a gather in batch order, one grouped GEMM per
level (chunked by offset), and a reduction per target in row order, which is bit-identical
to one launch per offset in index order at about 1% of its launches. P2P is one cube per
target leaf with sources staged through shared memory in near-row order; P2M and P2L give
each unit a coefficient and add the points in point order; rotation M2L is one cube per
target walking its V row. The run-time backend is a value, not a type parameter
(CubeCL 0.11's `Client` is no longer generic over the runtime). M2L strategies are tuned
by a strategy-level tuner in `nd-fmm-exec` at build, with a cache only in a directory the
caller passes; CubeCL's own autotune is not used.

| Question (Section 16) | Recommendation | Section |
| --- | --- | --- |
| 1. Requirements and tolerances | keep 1–11; device on one rank only until C5.1; dense f64 operator bound "1e-14, or twice the host's error"; FMM tolerances confirmed | 2, 9 |
| 2. Residency, T4b | operator-held device stores, `Fmm` drives the evaluation boundary; no T4b in Phase 4; hook designed for C5.1 | 4 |
| 3. Generic or enum | run-time value (`Backend` in exec, `Device` in kernels); no `R: Runtime` anywhere | 3.2 |
| 4. Dense M2L launches | grouped GEMM per level and offset chunk, then a row-ordered reduction | 6.4 |
| 5. `Classes` on the device | run as `Dense` from `M2lClasses::expand` | 6.8 |
| 6. f64 rule for p = 9–11 | dense through p = 11, rotation from p = 12 (provisional) | 10.5 |
| 7. CPU runtime and `threads(n)` | no rayon pool with the CPU backend; `threads(n)` caps the units per cube of the CPU layouts | 11 |
| 8. Autotune | strategy-level tuner in `nd-fmm-exec`, cache only in a caller-supplied directory | 10 |
| 9. C4.2 target | ≥ 25% of the P2P peak model on W2 (N = 10⁵), ≥ 10% on W1 at n_t = 64 with ≥ 4,096 target leaves per launch | 13.4 |
| 10. Metal compiler (new) | keep wgpu-msl; switch to `metal-native` only if T3 finds wgpu-msl's math mode breaks §3.13 | 5.3 |

## 1. Starting point

### 1.1 What Phase 3 and 3S built

The device path plugs into the following, unchanged.

- **The interface** (`nd_fmm_plan::operator`). `FmmOperator: FmmSizes` with one method
  per level and kind: `p2m`, `m2m` (once per `UpwardPass`), `m2l`, `p2l`, `l2l`, `l2p`,
  `m2p`, `p2p`. Each takes a batch struct (`P2m` … `P2p`) holding `level`,
  `index: &BoxIndex`, the views, shared inputs and an exclusive output. Every call adds;
  each target's contributions come in the order of its row ("Accumulation rule").
- **Both views of each list** (`nd_fmm_plan::lists`):
  - `Csr` (`row_offsets()`, `entries()`, both `&[u32]`) for X, near, W, P2M and L2P;
  - `GroupedCsr<G>` as `VList` (G = `u16` offset index), `Children` and `Parents`
    (G = `u8` octant), with the rows (`row_offsets()`, `sources()`, `groups()`) and the
    batches (`batch_offsets()`, `batch_targets()`, `batch_sources()`). A batch lists
    each target at most once, ascending; walking batches 0, 1, … meets each target in
    row order (`lists.rs`, "Guarantees").
- **The stores** (`nd_fmm_plan::store`):
  - `LevelBuffers<T>`: one allocation per kind; level l is a column-major
    (p + 1)² × K_l matrix, box i at `offsets()[l] + i · size(l)`, a GEMM operand;
    `as_slice()` for one upload;
  - `LeafStore<T>`: CSR with `point_offsets()` (`usize`) and `point_size()`; the
    sources (local leaves, then ghost leaves), target input and target output.
- **The evaluator** (`nd_fmm_plan::evaluator`): `Evaluator<'p, C, Op, P>` owns `Data`
  (the five stores), the three exchanges and the operator; its public stages are
  `reset`, `exchange_sources`, `upward_local`, `upward_global`, `exchange_multipoles`,
  `downward`, `evaluate_leaves`, with a debug check of their order.
- **`LaplaceOperator<T>`** (`nd_fmm_exec::operator`): `Kernels<T>` (p, `Tables<T>`,
  gradients, `max_leaf_points`, the P2P kernel) with one body per target,
  `Kernels::{p2m,m2m,m2l,p2l,l2l,l2p,m2p,p2p}_target`, run serially or in the `Fmm`'s
  rayon pool by `Execution::{boxes, leaves}`. It also implements `PairOperator`.
- **`Tables<T>`** (`nd_fmm_exec::tables`): one resolved `M2lStrategy` (`Dense`,
  `Classes`, `Rotation`; `Auto` resolves to `Dense` for p ≤ 8). Its families are private
  (`Families`); T5 adds crate-private accessors, no public change.
- **`nd-fmm-tables`**: `MatrixSet` (column-major, entry (r, c) of matrix i at
  i n² + r + c n, CONVENTIONS §3.12 "Matrix layout"); `M2mTables`, `L2lTables`,
  `M2lTables` (`matrices()`); `M2lClasses` (`expand()` to `M2lTables`); `RotationTables`
  (`tables(Operator)` → `ShiftTables` with `forward_blocks(polar)`,
  `backward_blocks(polar)`, `azimuth_factors(a)`, `coaxial_factors(d)`, `shift(t)`).
- **Geometry** (`nd_fmm_exec::geometry`): `relative_frame` from integer key indices,
  exact in f32 and f64 (CONVENTIONS §3.13, "Relative frames").
- **`Fmm`** (`nd_fmm_exec::fmm`): `FmmBuilder::build` (steps 1–8; step 1's all-reduce
  agrees the settings, the pool, the supplied domain and the finiteness of the points;
  step 3 (points inside the domain) and step 5 (`PointsNotOwned`) have their own),
  `Fmm::evaluate` (charges into the source chunks, `reset`, the six stages timed into
  `StageTimings`, output scaled by 1/(4π r_t)).
- **Phase 3S**: `nd_fmm_simd::P2pKernel` (NEON 4.03 Gpairs/s f32 φ at one thread,
  measured, Phase 3S T7), the C3S.4 contract (8 / 16 u_T per term).
- **The Phase 0 spike** (`spikes/cubecl-gemm`): the hand-written `tiled-smem`,
  `tiled-reg-cols` and `tiled-reg-rows` kernels; library CMMA at 3.5–4.4 TFLOP/s for
  p ≥ 8, B ≥ 10⁴ on Metal f32 (measured on 0.10.0; T2 re-measures on 0.11).

### 1.2 CubeCL 0.11.0-pre.4 facts the design relies on

`cubek-matmul` and `cubek-std` 0.3.0-pre.4 depend on `cubecl =0.11.0-pre.4`
(`cubek-matmul-0.3.0-pre.4/Cargo.toml`), so they are the matching pre-releases.

| # | Fact | Source |
| --- | --- | --- |
| F1 | The client is one type, `Client`, no longer generic over the runtime; kernels launch as `k::launch::<F>(&client, count, dim, …)` without `R` | `cubecl-runtime-0.11.0-pre.4/src/client.rs` (struct `Client`); `cubecl-macros-0.11.0-pre.4/src/generate/launch.rs` |
| F2 | `cubecl::Device` is a run-time enum (`Cuda`, `Hip`, `Metal`, `Wgpu`, `Cpu`) with fallible constructors `Device::cpu()`, `cuda(i)`, `metal_msl(kind)` → `Result<_, DeviceUnavailable>` (`NotLinked`, `NoSuchDevice`); `Device::client()` panics on an unlinked runtime | `cubecl-0.11.0-pre.4/src/device.rs` |
| F3 | Creating a client can still panic (wgpu: "No possible adapter available"); there is no `try_` variant | `cubecl-wgpu-0.11.0-pre.4/src/runtime.rs` |
| F4 | `metal` = `wgpu-msl` (wgpu with the MSL compiler); `metal-native` = the new `cubecl-metal` crate. wgpu-msl falls back to WGSL if the GPU family check fails (silently) or the MSL 3.2 canary fails (with a `log::warn!`). `client.name()` is `"wgpu<msl>"` for every Metal device once `msl` is compiled in, so it cannot tell; only the MSL path registers `Plane::Sync` and the CMMA combinations | `cubecl-0.11.0-pre.4/Cargo.toml`; `cubecl-wgpu-0.11.0-pre.4/src/backend/metal.rs` (`register_metal_features`, `register_features`), `src/compiler/base.rs` (`AutoCompiler::init`), `src/runtime.rs` (`runtime_name`) |
| F5 | Kernel buffers are slices `&[F]` / `&mut [F]`; the launch argument `BufferArg::from_raw_parts(handle, len)` is `unsafe`; `Array<E>` is a local array | `cubecl-core-0.11.0-pre.4/src/frontend/container/slice/launch.rs` |
| F6 | Handles take byte offsets (`offset_start`, `offset_end`); on wgpu an offset must respect the storage-buffer alignment (*inferred*) | `cubecl-runtime-0.11.0-pre.4/src/server/handle.rs`; `cubecl-wgpu-0.11.0-pre.4/src/runtime.rs` |
| F7 | Allocation failure **panics** ("failed to reserve … bytes"), on wgpu and CPU | `cubecl-wgpu-0.11.0-pre.4/src/compute/server.rs`; `cubecl-cpu-0.11.0-pre.4/src/compute/server.rs` |
| F8 | `properties().memory.max_memory()` reports the limit (Metal: `recommendedMaxWorkingSetSize`; CPU: RAM or cgroup); `memory_usage()` the use | `cubecl-ir-0.11.0-pre.4/src/properties.rs`; `cubecl-wgpu-0.11.0-pre.4/src/backend/metal.rs` |
| F9 | A launch returns `()`; compile and resource errors are attached to the output buffers and surface as `ServerError` on `read_one`, `check` or `sync_buffers`; `read` panics, `read_one` returns `Result` | `cubecl-runtime-0.11.0-pre.4/src/client.rs`, `src/server/base.rs` |
| F10 | Streams: `StreamPolicy::PerThread` by default, the `StreamId` derived from the OS thread; work on one stream is in order. wgpu encodes launches and submits every 32 tasks; `flush` submits, `sync` waits; all of a device's wgpu streams submit to one ordered queue. The CPU runtime drains its stream (spin, then yield) before a host write and before every shared-memory launch, and its `flush` waits | `cubecl-environment-0.11.0-pre.4/src/stream/policy.rs`; `cubecl-wgpu-0.11.0-pre.4/src/compute/stream.rs`, `src/compute/timings.rs`; `cubecl-cpu-0.11.0-pre.4/src/compute/stream.rs` (`enqueue_task`, `submit`) |
| F11 | `#[comptime]` parameters, `sync_cube`, plane operations (`plane_sum`, `plane_shuffle`, …), `fma` are as in 0.10; `SharedMemory::new(n)` is now `Shared::<[T]>::new_slice(n)` | `cubecl-core-0.11.0-pre.4/src/frontend/` (`synchronization.rs`, `plane.rs`, `operation/fma.rs`, `container/shared_memory.rs`) |
| F12 | Capability: `properties().supports_type(..)` and `type_usage(..)`; wgpu-msl registers F16 and F32 only (no f64); CPU registers F64; `cubecl-cpp` 0.11 registers F64 for CUDA again | `cubecl-ir-0.11.0-pre.4/src/properties.rs`; `cubecl-wgpu-0.11.0-pre.4/src/backend/metal.rs`; `cubecl-cpu-0.11.0-pre.4/src/runtime.rs`; `cubecl-cpp-0.11.0-pre.4/src/shared/base.rs` |
| F13 | Hardware properties: plane size, `max_shared_memory_size`, `max_units_per_cube`, `max_cube_count`; Apple Silicon plane size 32; Metal shared memory from the adapter's `max_compute_workgroup_storage_size` at run time (32 KB on the M3 Max, README); CPU plane size 1 | `cubecl-ir-0.11.0-pre.4/src/properties.rs`; `cubecl-wgpu-0.11.0-pre.4/src/runtime.rs`; `cubecl-cpu-0.11.0-pre.4/src/runtime.rs` |
| F14 | Profiling: `client.profile`, `profile_start`/`profile_end` → `ProfileDuration`, resolved later; `TimingMethod::Device` on wgpu when the adapter has timestamp queries (it flushes, not syncs, at the start), otherwise `System` (syncs at both ends); the CPU runtime flushes and uses host clocks | `cubecl-runtime-0.11.0-pre.4/src/client.rs`; `cubecl-wgpu-0.11.0-pre.4/src/compute/stream.rs`; `cubecl-cpu-0.11.0-pre.4/src/compute/stream.rs` |
| F15 | Autotune (`LocalTuner`, `TunableSet`, `AutotuneKey`, `anchor`) runs only where called; its persistent cache is a database under `<workspace>/target/environment` by default (or a user cache dir, or a `cubecl.toml` setting) when the `persistence` feature is on, which it is in `cubecl-runtime`'s defaults; the `cubecl` facade depends on every sub-crate with `default-features = false` | `cubecl-runtime-0.11.0-pre.4/src/tune/`, `Cargo.toml`; `cubecl-environment-0.11.0-pre.4/src/persistence/root.rs`; `cubecl-0.11.0-pre.4/Cargo.toml` |
| F16 | Fast math is per kernel (`#[cube(fast_math = …)]`), empty by default; the C++/MSL and LLVM emitters ignore the flags. The CPU runtime compiles with LLVM `default<O3>` and sets no fast-math flags, so LLVM itself neither contracts nor reassociates. But cubecl-opt's `InstCombinePass`, which the LLVM, C++/MSL and WGSL compilers all run, unconditionally fuses every product whose only use is an add or subtract into an fma (read; T2 measured its effect: CPU-runtime f64 GEMM no longer bit-identical to an unfused reference) wgpu-msl hands its MSL to `create_shader_module_passthrough`, and wgpu-hal compiles it with a bare `MTLCompileOptions::new()`, setting no math mode (read); Apple documents fast math as that object's default (*inferred* from Apple's documentation, not measured). `cubecl-metal` sets `MTLMathMode::Safe` | `cubecl-ir-0.11.0-pre.4/src/properties.rs`; `cubecl-opt-0.11.0-pre.4/src/passes/inst_combine.rs`; `cubecl-llvm-0.11.0-pre.4/src/cpu/jit/engine.rs`, `src/shared/to_llvm/math.rs`, `src/shared/base.rs` (pass list); `cubecl-wgpu-0.11.0-pre.4/src/backend/base.rs`; `wgpu-hal-30.0.1/src/metal/device.rs` (`ShaderInput::Msl`); `cubecl-metal-0.11.0-pre.4/src/compute/context.rs` |
| F17 | CUDA: the default compiler is LLVM to PTX (`CudaBackend::Llvm`); NVRTC is used only with the `cpp` feature (`cuda-cpp` on the facade). On the LLVM path fadd, fsub and fmul carry LLVM's `contract` flag for NVPTX, so multiply–adds may be fused; fdiv carries none. The NVRTC path passes no `--use_fast_math` or `--fmad` flag | `cubecl-cuda-0.11.0-pre.4/src/compiler.rs`, `src/compute/context.rs`, `Cargo.toml` (`cubecl-llvm` with `nvptx`); `cubecl-llvm-0.11.0-pre.4/src/shared/to_llvm/math.rs` (`fma_contraction`) |
| F18 | CPU runtime: a launch becomes one task per unit position of the cube, and the JIT entry loops over every cube, so its parallelism is the cube's unit count, not the cube count. Tasks go to the least-loaded worker of a pool with one worker per active logical CPU (physical cores first, SMT siblings after; on the M3 Max 16 workers, efficiency cores included); the workers are pinned where the OS allows, which on macOS is only a hint (T2); kernels with `sync_cube` or shared memory get one dedicated worker per unit, and the pool grows. Idle workers poll for 200 µs, then park. No thread-count setting; only the stack size (`CUBECL_CPU_STACK_SIZE` or `CUBECL_CPU_STACK_MB`) | `cubecl-cpu-0.11.0-pre.4/src/compute/threadpool/` (`scheduler/dispatcher.rs`, `scheduler/mod.rs`), `src/compute/affinity/mod.rs`; `cubecl-llvm-0.11.0-pre.4/src/cpu/entrypoint.rs`, `src/shared/base.rs` |
| F19 | CPU runtime: `sync_cube` is a spin barrier; shared memory is emulated from a pool, and the stream drains before a shared-memory launch; plane size 1, no plane features | `cubecl-llvm-0.11.0-pre.4/src/cpu/synchronization.rs`; `cubecl-cpu-0.11.0-pre.4/src/compute/threadpool/mod.rs` |
| F20 | `cubek_matmul::launch_ref(&Strategy, &Client, lhs, rhs, out, &mut MatmulElems) -> Result<(), MatmulSetupError>`; `Strategy::{Tiled(..), MultiLevel(..), Auto}`; `Auto` does not autotune (it tries `SimpleCyclicCmma`, falls back to `SimpleUnit` on `Unavailable`, and panics on any other error) | `cubek-matmul-0.3.0-pre.4/src/launch.rs`, `src/strategy.rs` |
| F21 | Metal CMMA offers (f32, f32, f32) at 8 × 8 × 8; no f64 MMA anywhere. On a backend that registers TF32 (CUDA), the accelerated routines switch f32 stage and register types to TF32 | `cubecl-cpp-0.11.0-pre.4/src/metal/dialect.rs`; `cubek-matmul-0.3.0-pre.4/src/multi_level/definition/blueprint.rs` |
| F22 | Matmul shapes are `[batch…, m, k]` with numpy-style batch broadcasting; row or column major from the last two strides | `cubek-matmul-0.3.0-pre.4/src/definition/base.rs`; `cubek-std-0.3.0-pre.4/src/matrix_layout.rs` |
| F23 | No persistent kernel cache for wgpu-msl or the CPU runtime (in-memory per process only); `[compilation] cache` is off by default | `cubecl-wgpu-0.11.0-pre.4/src/compute/server.rs`; `cubecl-cpu-0.11.0-pre.4/src/runtime.rs`; `cubecl-runtime-0.11.0-pre.4/src/config/compilation.rs` |
| F24 | CubeCL reads `cubecl.toml` (searched upward from the working directory) and several `CUBECL_*` variables itself | `cubecl-environment-0.11.0-pre.4/src/config/mod.rs`; `cubecl-runtime-0.11.0-pre.4/src/config/base.rs` |

Facts F1, F7, F9, F10, F16 and F18 shape the design most. F1 makes the run-time backend
natural (Section 3.2). F7 means device memory must be checked before allocating
(Section 4.6). F9 means device errors surface at the one download (Section 12). F10
means every launch must come from one thread (Section 11). F16 is why T3 must measure
Metal's arithmetic before any kernel is written (Section 9.3).

## 2. Requirements

The eleven requirements of the README, and how this design meets each.

| # | Requirement | How the design meets it | Change? |
| --- | --- | --- | --- |
| 1 | Same interface | `DeviceOperator<T>` implements `FmmSizes` and `FmmOperator`; the `Evaluator` drives it; `FmmBuilder::backend` is one more setting, `Backend::Host` the default (Section 3.3) | clarified: device backend on one rank only until C5.1 (Section 4.4) |
| 2 | Same data and conventions | device stores mirror the evaluator's layouts value for value (§3.6, §3.7, §3.13); tables uploaded from `nd-fmm-tables` as built and rounded; frames from integer box coordinates on the device (Section 6.2); 1/(4π) and r_t applied by `Fmm::output` unchanged | none |
| 3 | Device-resident data | plan views, geometry and tables uploaded once at build; points once; charges and output once per evaluation (Section 4.2); counted by the transfer accounting | none |
| 4 | Accumulation rule | every kernel adds each target's contributions in row order, starting from the value already in the output; the grouped reduction reproduces the per-offset order exactly (Sections 6.1, 6.4) | none |
| 5 | No atomics | P2P, leaf operators and reductions are target- or coefficient-owned; GEMM outputs go to distinct columns (Section 6) | none |
| 6 | Deterministic | fixed launch sequence on one stream; explicit matmul strategies; layouts, chunks and tuned choices fixed at build (Sections 8, 9, 10) | none |
| 7 | Precision is a capability | `Device::supports(Precision::F64)` from CubeCL's type registry; refusal in step 1 of `build` (Section 5) | none |
| 8 | Host fallback | per-kind placement; fallback calls run `LaplaceOperator`'s own `FmmOperator` method on host mirrors with explicit transfers; bit-identical with every kind on the host (Section 7) | none |
| 9 | Safe at the boundary | `unsafe` only in `nd-fmm-kernels` (`launch_unchecked`, `BufferArg::from_raw_parts`, F5); `nd-fmm-exec` sees only `nd-fmm-kernels` types | none |
| 10 | Tested on what can run | CPU runtime f32 and f64, Metal f32 ignored and by hand; every test prints `DeviceInfo` (Section 13) | none |
| 11 | Measured, never asserted | timings only in `nd-fmm-validate` examples, Metal, release, compilation excluded (Section 13.3) | none |

**Proposed clarification of requirement 1 and of the README's MPI rule** ("On more than
one rank `Fmm` still reports `PointsNotOwned` until C5.1, for either backend"). That
holds, but on several ranks a build can also succeed, when every point happens to lie
in a leaf of its own rank (`fmm-exec/tests/mpi_exec.rs`, module docs). The host path
then runs distributed. The device path cannot yet (Section 4.4). Proposal: with a
device backend on more than one rank, `build` returns
`SettingsError::DeviceNeedsOneRank { ranks }` **after** the `PointsNotOwned` check of
step 5, so that `PointsNotOwned` still wins where it applies. The rank count is the same
on every rank, so the error needs no agreement.

**Tolerances of "Accuracy measures".** Confirmed, with one proposed addition:

- Operators, f64: 1e-14 for dense M2M, L2L and M2L. The device GEMM sums each product
  in its own order (blocked, possibly fused) and adds the result to the output once,
  where the host adds each product into the output (`MatrixSet::apply`). The worst-case
  rounding bound of an n-term dot product is n u Σ|terms|, (p + 1)² u ≈ 4.9e-14 at
  p = 20 (model), above 1e-14; the typical √n u is about 2.3e-15. Proposal, as for P2P
  in C3S.4: **within 1e-14, or within twice the host operator's measured error on the
  same cell, whichever is larger.** It changes nothing where the GEMM behaves typically
  and prevents a p = 20 cell from failing on a valid summation order.
- Operators, f32: 1e-5 against the f64 reference, and rotation and the leaf operators
  at 1e-13 in f64: confirmed.
- P2P: the provisional C3S.4 contract until T3 proposes the device one.
- FMM against the host FMM: relative L2 1e-12 (f64) and 1e-5 (f32): confirmed
  (Section 9.2).
- FMM errors against `direct_sum`: within 0.1% (f64) and 5% (f32) of the host run's:
  confirmed, with the condition it rests on stated: truncation dominates the error.
  That holds on the cube and the Plummer sphere at p ≤ 8 in f32 (f32 within 5% of f64
  there, Phase 3 T12) but not for ∇φ on the sphere surface, whose f32 floor of 1.1e-5
  is rounding; the README already keeps f32 gradient tests off the sphere.

## 3. Crates and public surfaces

### 3.1 `nd-fmm-kernels`

Package `nd-fmm-kernels`, library `nd_fmm_kernels`, in `fmm-kernels/`, as the README
fixes. No MPI, no `nd-fmm-plan`, no rayon: it takes plain `&[u32]` index arrays and
`&[T]` data, so it builds and tests alone.

| Module | Contents | Task |
| --- | --- | --- |
| `device` | `BackendKind`, `Precision`, `Device`, `DeviceInfo`, transfer and launch counters | T4 |
| `buffer` | `DeviceBuffer<E>`, `DeviceSlice`, `DeviceSliceMut` (buffer + element range) | T4 |
| `error` | `KernelError` | T4 |
| `movement` | `zero`, `gather_columns`, `scatter_add_columns`, `scatter_values` | T4 |
| `view` | `IndexView` (a CSR), `GroupedView` (rows, batches, row-to-batch map), `BoxCoordinates`, `LeafCoordinates` | T5 (types), T6–T9 (use) |
| `p2p` | `P2pLayout`, `p2p` | T6 |
| `leaf` | `HarmonicsCoefficients`, `p2m`, `l2p`, `p2l`, `m2p` | T7 |
| `gemm` | `TiledLayout`, `LibraryStrategy`, `GemmChoice`, the hand-written kernel, the library call with its input-precision guard | T8, T9 |
| `translate` | `TranslationTables`, `GroupedLaunch`, `grouped` (gather → GEMM → reduce or scatter-add) | T8, T9 |
| `rotation` | `RotationTablesDevice`, `m2l_rotation` | T10 |

The surfaces that decide something, as sketches (doc comments abridged; T4–T10 refine
names, not semantics):

```rust
/// A backend this crate can name. Whether it is compiled in is a run-time question,
/// so a host-only build can still parse "metal" and refuse it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind { Cpu, Metal, Cuda }

impl BackendKind {
    /// True if this build has the backend's cargo feature.
    pub fn is_compiled(self) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision { F32, F64 }

/// An opened device: one `cubecl::Device` and its `Client` (F1, F2), what it reports,
/// and the counters of transfers, launches and syncs. Every launch is issued from the
/// thread that owns the `Device` (F10); it is `Send`, not shared between threads.
pub struct Device { /* cubecl::Device, cubecl Client, DeviceInfo, Counters */ }

impl Device {
    /// Opens the default device of `kind`: `cubecl::Device::{cpu, metal_msl, cuda(0)}`.
    /// Metal must come up with the MSL compiler, recognised by the features only that
    /// path registers (`Plane::Sync` and the CMMA combinations, F4; `client.name()`
    /// says "wgpu<msl>" either way). A WGSL fallback is refused as `NoDevice`.
    ///
    /// # Errors
    /// `NotCompiled`, `NoDevice`. A panic inside CubeCL's client creation (F3) is not
    /// caught; the constructors of F2 enumerate devices first, so it is not expected.
    pub fn open(kind: BackendKind) -> Result<Self, KernelError>;
    pub fn info(&self) -> &DeviceInfo;
    /// Whether the device does arithmetic in `precision` (F12).
    pub fn supports(&self, precision: Precision) -> bool;
    /// Allocates `len` zeroed elements; checks `len` against the memory limit first,
    /// since CubeCL panics on allocation failure (F7).
    pub fn alloc<E: DeviceElement>(&mut self, len: usize) -> Result<DeviceBuffer<E>, KernelError>;
    pub fn upload<E: DeviceElement>(&mut self, data: &[E]) -> Result<DeviceBuffer<E>, KernelError>;
    /// Overwrites `slice` with `data`; no sync.
    pub fn write<E: DeviceElement>(&mut self, slice: DeviceSliceMut<'_, E>, data: &[E]) -> Result<(), KernelError>;
    /// Copies `slice` into `out`; waits for every queued launch (one sync), and returns
    /// any launch error attached to the buffer (F9).
    pub fn download<E: DeviceElement>(&mut self, slice: DeviceSlice<'_, E>, out: &mut [E]) -> Result<(), KernelError>;
    pub fn sync(&mut self) -> Result<(), KernelError>;
    pub fn counters(&self) -> Counters;
}

/// For reports: backend, device name, CubeCL version, compiler (`client.name()`),
/// precisions, plane size, shared memory, units per cube, memory limit.
pub struct DeviceInfo { /* public fields */ }
impl fmt::Display for DeviceInfo { /* "metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32" */ }

/// Bytes and calls of uploads and downloads, launches and syncs, since the last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters { pub upload_bytes: u64, pub uploads: u64, pub download_bytes: u64,
                      pub downloads: u64, pub launches: u64, pub syncs: u64 }

/// `len` elements of f32, f64 or u32 on one device; freed on drop. Index arrays are
/// always u32 on the device: the u16 offset indices and u8 octants of `GroupedCsr` are
/// widened at upload.
pub struct DeviceBuffer<E: DeviceElement> { /* handle, len, device id */ }

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum KernelError {
    NotCompiled { backend: BackendKind },
    NoDevice { backend: BackendKind, reason: String },
    UnsupportedPrecision { backend: BackendKind, precision: Precision },
    /// The request does not fit under `max_memory()` (F8).
    OutOfMemory { requested: u64, limit: u64 },
    /// A store or view needs more than u32 indices.
    TooLarge { what: &'static str, len: usize },
    /// A buffer of another `Device`.
    WrongDevice,
    /// A launch or transfer failed on the device (F9), with CubeCL's message.
    Device { reason: String },
}
```

Launch wrappers, one per family; each is safe, validates its index arrays on the host at
upload (so kernels can be launched unchecked) and allocates nothing:

```rust
pub mod movement {
    /// buffer[range] = 0 (+0.0 bit pattern).
    pub fn zero<E>(device: &mut Device, slice: DeviceSliceMut<'_, E>) -> Result<(), KernelError>;
    /// y[:, j] = x[:, idx[j]], columns of n values (n comptime).
    pub fn gather_columns<T>(device: &mut Device, n: u32, x: DeviceSlice<'_, T>,
                             idx: DeviceSlice<'_, u32>, y: DeviceSliceMut<'_, T>) -> Result<(), KernelError>;
    /// x[:, idx[j]] += y[:, j]. The indices of one launch are distinct (checked at
    /// upload in debug builds).
    pub fn scatter_add_columns<T>(/* as above */) -> Result<(), KernelError>;
    /// x[idx[j]] = y[j]: the charges into their slots of the source store.
    pub fn scatter_values<T>(/* as above */) -> Result<(), KernelError>;
}

pub mod translate {
    /// out[:, t] += Σ over (s, g) of row t, in row order, of A_g · input[:, s]:
    /// M2M, L2L and dense M2L of one level (Section 6.4).
    pub fn grouped<T: Float>(device: &mut Device, launch: &GroupedLaunch, tables: &TranslationTables<T>,
                             input: DeviceSlice<'_, T>, output: DeviceSliceMut<'_, T>,
                             gemm: GemmChoice, scratch: &mut Scratch<T>) -> Result<(), KernelError>;
}

pub mod p2p {
    /// Adds the near field of one level's target leaves into `output` (Section 6.2).
    pub fn p2p<T: Float>(device: &mut Device, launch: &P2pLaunch<'_, T>, layout: P2pLayout)
                         -> Result<(), KernelError>;
}
```

`leaf::{p2m, l2p, p2l, m2p}` and `rotation::m2l_rotation` follow the same pattern
(Sections 6.3 and 6.6).

### 3.2 Generic over the runtime, or chosen at run time

| | `R: Runtime` threaded through `nd-fmm-exec` | A run-time value |
| --- | --- | --- |
| Fits CubeCL 0.11 | no: 0.11 removed `R` from the client and the launch (F1) | yes: `cubecl::Device` is already an enum and `Client` is one type (F1, F2) |
| `nd-fmm-exec` without `cubecl` (requirement 9) | needs `Runtime` re-exported, and leaks it into `Fmm<'o, T, C, R>` | sees only `nd_fmm_kernels::Device` |
| `Fmm`'s API (requirement 1) | a new type parameter on `Fmm` | unchanged; `backend` is a builder setting, as the README asks |
| CLI and tests (`--backend metal`) | a `match` that instantiates one type per backend | parse a value |
| Compile time | `nd-fmm-exec`'s device code compiled once per runtime | once |
| Cost per call | none | none: kernels are still generic over `F: Float` and comptime p; the backend is a value in the client |

**Recommendation: a run-time value.** CubeCL 0.11 decided most of this: kernels are
generic over the float type and comptime parameters only, and the runtime is a property
of the `Client`. `nd-fmm-kernels` keeps `cubecl::Device` and `Client` inside its own
`Device`. Per-backend choices (cube layouts, tile sizes, the GEMM) are values looked up
from `DeviceInfo::backend` at build (Section 6.1).

### 3.3 The `nd-fmm-exec` side

Settings and reports on `FmmBuilder` and `Fmm`. The `Backend` enum and the kinds exist
without the `gpu` feature, so a host-only build refuses `Metal` with a typed error
rather than failing to compile:

```rust
/// Where the operators of an FMM run. Text form: host, cpu, metal, cuda.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend { #[default] Host, Cpu, Metal, Cuda }

/// One operator kind, for placement and reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorKind { P2m, M2m, M2l, P2l, L2l, L2p, M2p, P2p }

impl<T> FmmBuilder<T> {
    /// The backend (default `Host`; README "Design decisions").
    pub fn backend(self, backend: Backend) -> Self;
    /// Runs these kinds on the host fallback even with a device backend: a test aid
    /// (requirement 8). Kinds without a device kernel yet fall back regardless.
    pub fn host_fallback(self, kinds: impl IntoIterator<Item = OperatorKind>) -> Self;
    /// Synchronises after every stage so that `StageTimings` time each stage (Section 8.3).
    pub fn synchronous_stages(self, on: bool) -> Self;
    /// The autotune cache directory (Section 10); without it the static rule applies.
    pub fn tuning_cache(self, dir: impl Into<PathBuf>) -> Self;
}

impl<'o, T, C> Fmm<'o, T, C> {
    pub fn backend(&self) -> Backend;
    /// With a device backend: the device, the kind placement, the resolved strategy,
    /// GEMM and layouts per level, the tuning outcome.
    pub fn device_report(&self) -> Option<&DeviceReport>;
    /// Transfers, launches and syncs of the last evaluation (and of the build).
    pub fn device_counters(&self) -> Option<DeviceCounters>;
}
```

New `SettingsError` variants, all `Copy` as the enum is today:
`BackendNotCompiled { backend }`, `NoDevice { backend }` (the reason cannot ride in the
`Copy` enum, and a failed build has no `Fmm` to report it;
`Backend::probe(self) -> Result<DeviceInfo, String>`, a thin wrapper over
`Device::open`, returns it to a caller who wants it),
`PrecisionUnsupported { backend }`, `DeviceMemory { needed: u64, limit: u64 }`,
`DeviceNeedsOneRank { ranks: usize }`. `FmmError` gains `Device(String)` for failures
after the settings were accepted (Section 12).

**The device operator.** In a new module `nd_fmm_exec::device` (feature `gpu`):

```rust
/// The Laplace operator on a device; see Sections 4 and 7.
pub struct DeviceOperator<T: SimdScalar> {
    host: LaplaceOperator<T>,      // the fallback bodies, the host tables, the P2P choice
    device: nd_fmm_kernels::Device,
    placement: Placement,          // per OperatorKind: Device or Host
    stores: DeviceStores<T>,       // multipoles, locals (one buffer each), sources,
                                   // target input, target output, charge slots
    views: DeviceViews,            // per level: the uploaded views and launch plans
    tables: DeviceTables<T>,       // dense octants always; dense M2L or rotation
    mirrors: Option<HostMirrors<T>>, // only if some kind is on the host (Section 7)
    choices: Resolved,             // strategy, GEMM and layouts per level, fixed at build
}
```

It **wraps** `LaplaceOperator` rather than sharing its tables by reference: the fallback
needs a complete host operator (same tables, gradients, `max_leaf_points`, P2P choice,
pool), and the device tables are uploaded from it once and then never read on the host.
Under `Rotation` the device additionally needs the dense octant tables (README), which
the build loads or builds through `table_cache` beside the host's rotation tables.

`Fmm` holds `Evaluator<'o, C, ExecOperator<T>, Plan>`, where the crate-private

```rust
enum ExecOperator<T> { Host(LaplaceOperator<T>), #[cfg(feature = "gpu")] Device(Box<DeviceOperator<T>>) }
```

implements `FmmSizes` and `FmmOperator` by delegation (one `match` per level call).
`Fmm::operator()` keeps returning `&LaplaceOperator<T>`: the host operator, or the
device operator's fallback operator. Without `gpu` the enum has one variant and the host
path compiles to what it is today.

### 3.4 Cargo features

| Crate | Feature | Builds |
| --- | --- | --- |
| `nd-fmm-kernels` | none | kernel definitions and host-side types; no runtime; `BackendKind::is_compiled` false everywhere |
| | `cpu` | `cubecl/cpu` (the CubeCL CPU runtime and its `tracel-llvm` bundle) |
| | `metal` | `cubecl/metal` (= `wgpu-msl`, F4) |
| | `cuda` | `cubecl/cuda`; type-checked only |
| `nd-fmm-exec` | `gpu` | `dep:nd-fmm-kernels`, `Backend` support, `DeviceOperator` |
| | `cpu`, `metal`, `cuda` | `gpu` plus the matching `nd-fmm-kernels` feature |
| `nd-fmm-validate` | `gpu`, `cpu`, `metal`, `cuda` | passed through to `nd-fmm-exec`; the device rows of the examples |

None is on by default. CubeCL's `persistence` feature stays off with `cpu` and `cuda`
**only because all three workspace entries set `default-features = false`** (root
`Cargo.toml`): the facade's `default` enables `cubecl-core/default`,
which enables `cubecl-runtime/default`, which includes `persistence`
(`cubecl-0.11.0-pre.4/Cargo.toml`, `cubecl-core-0.11.0-pre.4/Cargo.toml`,
`cubecl-runtime-0.11.0-pre.4/Cargo.toml`), and the defaults of `cubek-matmul` and
`cubek-std` turn on `cubecl/default` (`cubek-matmul-0.3.0-pre.4/Cargo.toml`,
`cubek-std-0.3.0-pre.4/Cargo.toml`). Cargo unifies features, so any crate in the graph
that takes one of the three with its defaults turns persistence on; T4 checks with
`cargo tree -e features`, and `fmm-kernels/CLAUDE.md` keeps the rule.

**Corrected by T4 (measured, accepted at its sign-off on 2026-10-03):** with `metal`,
persistence is on whatever the workspace sets. `cubecl-wgpu-0.11.0-pre.4/Cargo.toml`
takes `cubecl-cpp` without `default-features = false`, and `cubecl-cpp`'s `default`
enables `cubecl-runtime/default`. Opening a Metal device creates an empty store
`target/environment/default.db` (schema only). Nothing is recorded in it as long as no
crate calls CubeCL's autotune or throughput measurement or enables `[compilation]
cache`, which the workspace never does. A `metal-native`
feature (`cubecl-metal`) is not added unless the sign-off of question 10 asks for it.

## 4. Device data and residency

### 4.1 What lives where

Notation for the formulas: s bytes per value (4 or 8); n_c = (p + 1)²; K_l boxes held
on level l and K = Σ K_l; J leaves of the numbering (J_loc local); N_s sources and N_t
targets on the rank; o = 1, or 4 with gradients; E_V, E_M (M2M, both passes), E_L (L2L),
E_U (near entries), E_W, E_X pairs per view, summed over levels.

| Data | Bytes on the device | Uploaded | Downloaded |
| --- | --- | --- | --- |
| multipoles, locals | 2 K n_c s, one buffer each, the `LevelBuffers` layout | never (zeroed by a kernel) | never (fallback only, Section 7) |
| source store | 4 N_s s, the `LeafStore` layout of §3.13 | coordinates once at build; charges every evaluation (N_s s, scattered by `charge_slots`) | never |
| target input | 3 N_t s | once at build | never |
| target output | o N_t s | never (zeroed) | once per evaluation |
| index arena (u32) | about 12 E_V + 12 (E_M + E_L) + 4 (E_U + E_W + E_X) + 8 J + 8 K + 4 N_s: views, row-to-batch maps (Section 6.4), point offsets, `charge_slots` | once at build, one call | never |
| geometry (u32) | 16 J (level and index per leaf) + 12 K (index per box) | once at build | never |
| tables | dense: (316 + 16) n_c² s; rotation: the M2L `ShiftTables` (94 B + 64 p + 15 C) s plus 16 n_c² s for the dense octants (B, C as in `nd_fmm_tables::rotation`) | once at build | never |
| GEMM scratch | at most S_scr (default 128 MB, *model*; Section 6.4) | never | never |

Everything is allocated at build: an evaluation allocates nothing on the device, so the
allocation panic of F7 cannot happen inside `evaluate`. Stores are one buffer per kind
and kernels take an element offset as a scalar argument, rather than sub-buffer handles,
because wgpu binding offsets must be aligned (F6) and level offsets are not.

**Per evaluation** (one rank, every kind on the device):

| Direction | Bytes | Calls |
| --- | --- | --- |
| up | N_s s (charges) | 1 |
| down | o N_t s (target output) | 1 |
| syncs | | 1 (the download) |

For the C3.2 cube (N_s = N_t = 10⁵, gradients) that is 2.0 MB in f32 and 4.0 MB in f64.
The README's minimum is met: nothing else crosses.

**Per build**: the source store and target input ((4 N_s + 3 N_t) s, 2 calls), the index
arena and the geometry (2 calls), the tables (one call per family: 1–3). About 7 calls;
the bytes are those of the table above.

### 4.2 Every host-side write of the `Evaluator` outside operator calls

Read from `fmm-plan/src/evaluator.rs` and `fmm-plan/src/exchange.rs`. "Reads" are listed
too, because a stale host value that an exchange *sends* is as wrong as one it writes.

| Stage | Function | What it does | On one rank | On several ranks |
| --- | --- | --- | --- | --- |
| construction | `Evaluator::new` → `Data::new` | allocates the five stores zeroed (`LevelBuffers::from_index`, `LeafStore::new`) | same | same |
| `reset` | `Evaluator::reset` → `Data::reset` → `LevelBuffers::clear` (multipoles, locals), `LeafStore::clear` (target output) | fills them with `T::default()` (+0.0); sources and target input are kept | **writes**: every multipole, local and target output | same |
| step 1 | `Evaluator::exchange_sources` → `SourceExchange::forward` | reads the local chunks `sources.chunk(leaf)` of `send_leaves` into `send_buffer`; `forward_send_values` writes the received chunks into the ghost tail `sources.range_mut(ghost_leaves())` | no ghost leaves and empty `send_leaves`: the neighbour collective runs with empty buffers; **nothing read or written** | **reads** local source chunks (coordinates and this evaluation's charges); **writes** the ghost tail |
| step 3 | `Evaluator::upward_global` → `CoarseExchange::gather` | copies this rank's coarse blocks (`multipoles.chunk(level, i)` for b in `rank_blocks(rank)`) into `send_buffer`; all-gather-v into `gathered`; writes every *other* rank's block into its slot | the only coarse block is the root (`compute_coarse_tree` returns `[root]` on one rank, `octree/src/octree/implementation.rs`), which is this rank's: **reads the root multipole** into its send buffer, before the global M2M forms it, so the value read is `reset`'s +0.0 on host and device alike; **writes nothing** | **reads** own coarse-block multipoles (formed in `upward_local`); **writes** the other ranks' blocks, which are ghost boxes on this rank (`OctreeOptions::with_ghost_children`, octree docs) |
| step 4 | `Evaluator::exchange_multipoles` → `MultipoleExchange::forward_all` → `forward(level)` for l = 0..L | per level: copies the `send_boxes(l)` multipoles into the send buffer; neighbour all-to-all; scatters the receive buffer into the `receive_boxes(l)` slots | no ghost boxes, so both lists are empty on every level: **nothing read or written** | **reads** sent non-ghost multipoles of every level; **writes** the ghost boxes named by V rows of l and W rows of l − 1 |

The caller's writes, through the `Evaluator`'s API, are listed for completeness:
`Fmm::build` step 8 writes the leaf-scaled coordinates (`sources_mut`,
`target_input_mut`) once, and `Fmm::evaluate` writes the charges every evaluation
(`local_sources_mut`). Operator calls write only their batch output. Nothing else in
`nd-fmm-plan` writes a store.

On one rank the octree has no ghost keys: `generate_all_keys` builds ghosts only when
`size > 1` (`octree/src/octree/implementation.rs`). It does have one `Global` key, the
root, which it inserts after the leaf loop unless the root is itself a leaf
(`all_keys.entry(morton::root()).or_insert(KeyType::Global)`). So on one rank
`m2m_global(0)` holds the root's row (`fmm-plan/src/plan.rs`, first pass) and the
global pass forms the root multipole: an operator call, not a host write. The exchanges
have nothing to move. That leaves exactly one evaluator-side write per evaluation,
`reset`'s zeroing, and one read whose result is never observed: the root multipole, +0.0
at that point, goes into `CoarseExchange`'s `gathered` buffer. `CoarseExchange` has
public `gathered()` and `chunk()`, but the `Evaluator` keeps it in a private field and
exposes no accessor, and nothing reads it back.

### 4.3 Keeping the device copy consistent: the options

**(a) The operator keeps device stores; `Fmm` tells it where an evaluation starts.**
No plan change. `Fmm` already calls the stages itself (`Fmm::evaluate`), so it can call
the operator between them through `Evaluator::operator_mut`:

```rust
// Fmm::evaluate, device backend, one rank:
write charges into the host chunks              // as today; read by host-fallback kinds
self.evaluator.reset();                         // host zeroing, as today
op.begin_evaluation(&charges_by_leaf)?;         // zero kernels; upload charges; scatter
stages 1..6 as today                            // operator calls launch kernels, no sync
let output = op.read_output()?;                 // one download, one sync
self.output_from(output)                        // scaling and order, as today
```

`reset` ↔ `begin_evaluation`'s zero kernels; caller writes ↔ the point upload at build
(`DeviceOperator::load_points(&sources, &targets)`, the leaf-scaled source store with
zero charges and the target input, copied by `build` after step 8 from
`Evaluator::source_store()` and `target_input_store()`, since those accessors and
`operator_mut()` cannot borrow the evaluator at once; one allocation, at build) and the charge upload; the
exchanges ↔ nothing to do on one rank (Section 4.2). Consistent by construction on one
rank, with a guard: `DeviceOperator::new` checks that the plan has no ghost leaf and no
ghost box (the condition under which no exchange writes), and `build` refuses a device
backend on several ranks (Section 2).

On several ranks (a) does not work without a plan change. The values to download before
an exchange are the exchange's `send_leaves`, `send_boxes(l)` and own coarse blocks, and
the slots to upload after it are its `ghost_leaves`, `receive_boxes(l)` and the other
ranks' blocks. The exchanges are private fields of the `Evaluator`, and the host stores
have no `&mut` accessor outside operator calls (`multipoles()` and `locals()` are
shared). The operator could only download every non-ghost multipole of a level inside
the last call that writes it (`p2m(l)`), one sync per level, and upload ghost slots
inside the first call that reads them: correct but neither minimal nor simple.

**(b) An `nd-fmm-plan` hook around the four data movements** (redesign §10). The
`Evaluator` tells the operator before and after each movement, with the index lists the
movement uses, so a device operator downloads exactly what is sent and uploads exactly
what arrives. Exact on every rank count; sketched in Section 4.5. It changes
`nd-fmm-plan` and needs multi-rank `IndexFmm` checks (a T4b), and in Phase 4 it would
have no user beyond one rank, where it does what (a) does.

**(c) Alternatives considered and rejected:**
- *The host stores as the source of truth, with write-back.* Every device call
  downloads its output level into the batch's `&mut` output slice. Simple, but a sync
  and a full download per call: the residency requirement fails.
- *Detecting host writes* (generation counters or checksums of host buffers). Costs a
  pass over every store per evaluation and still needs the exchange layouts.
- *Device-only stores with zero-sized host stores.* The `Evaluator` validates positive
  sizes (`EvaluatorError::ZeroSize`) and the batch types carry host slices, so it is a
  plan redesign, not a hook.

**Recommendation: (a) for Phase 4, without an `nd-fmm-plan` change; T4b is not
needed.** The device backend runs on one rank, where the only evaluator-side write is
the zeroing that `begin_evaluation` mirrors. (b) is the C5.1 path; Section 4.5 gives its
shape so that C5.1 starts from a designed hook.

The price of (a), stated: the `Evaluator` still allocates and zeroes its host stores.
The host memory is the host path's (unchanged), and `reset` costs a memset of
(2 K n_c + o N_t) s bytes per evaluation, about 5 MB for the cube at p = 8 in f32 and
about 41 MB for the Plummer sphere at p = 18 in f64 (*model*, Section 4.6), roughly
0.1–1 ms (*estimate*). T5 and T11 measure it; if it matters, a later plan option
(stores the operator owns) belongs to C5.2's "device-resident ghost buffers".

### 4.4 The rule on several ranks in Phase 4

`build` with a device backend on more than one rank: steps 1–5 run as today (so an input
error or `PointsNotOwned` is reported as on the host path), then every rank returns
`SettingsError::DeviceNeedsOneRank { ranks }`. Every rank sees the same `comm.size()`,
so no collective is needed and none is skipped. `tests/mpi_exec.rs` on two ranks (T5)
therefore neither hangs nor diverges: each scenario stops with an agreed error.

### 4.5 The C5.1 hook, sketched (not part of Phase 4)

A kernel-agnostic extension of `FmmOperator`, with a default that does nothing, so
`LaplaceOperator`, `PerPair` and the index FMMs are unchanged and the host path stays
bit-identical (the evaluator's own movements do not change):

```rust
/// A data movement of the evaluator outside the operator calls (redesign §10), with the
/// index lists it uses. "Send" events come before the exchange reads the host store,
/// "Received" events after it wrote it. Never called with communication pending.
pub enum HostData<'a, T> {
    Reset,
    SendSources { leaves: &'a [u32], sources: &'a mut LeafStore<T> },          // send_leaves
    ReceivedSources { leaves: Range<usize>, sources: &'a LeafStore<T> },        // ghost_leaves
    SendCoarse { blocks: &'a [(usize, u32)], multipoles: &'a mut LevelBuffers<T> },
    ReceivedCoarse { exchange: &'a CoarseExchange<T>, multipoles: &'a LevelBuffers<T> },
    SendMultipoles { exchange: &'a MultipoleExchange<T>, multipoles: &'a mut LevelBuffers<T> }, // all levels
    ReceivedMultipoles { level: usize, boxes: &'a [u32], multipoles: &'a LevelBuffers<T> },
}

pub trait FmmOperator: FmmSizes {
    /* the eight level calls */
    fn host_data(&mut self, event: HostData<'_, Self::Value>) {}
}
```

- On one rank `SendCoarse` lists no blocks (the gather goes to the rank itself), so a
  device operator syncs once per evaluation there too.
- `SendMultipoles` comes once, before level 0, with every level's `send_boxes`, so the
  device downloads them in one sync, not one per level.
- `CoarseExchange` would gain an `own_blocks()` accessor; everything else it needs is
  public already (`send_leaves`, `ghost_leaves`, `send_boxes`, `receive_boxes`,
  `block`, `rank_blocks`).
- **The `IndexFmm` check**: a shadow operator wraps `BatchedIndexFmm`, computes into its
  own copies of the stores and learns of host data only through `host_data`. It must
  equal the plain index FMM on every `tests/mpi_regressions.rs` scenario and path on 1,
  2 and 4 ranks; with the hook disabled it must fail on 2 ranks (the test sees the
  ghosts). That proves the hook covers every host write, kernel-agnostically.

### 4.6 Device memory, and what `Fmm` does when a configuration does not fit

*Model* from the table of Section 4.1, one rank, gradients, S_scr = 128 MB, tables of
the strategy named; K and E_V from Phase 3 T12 for the cube (K = 4,681, E_V = 640,584)
and estimated for the Plummer sphere (K ≈ 6,500 from its 5,678 leaves) and for the cube
at N = 10⁶ (K ≈ 37,000, E_V ≈ 5.6 × 10⁶).

| Problem | p, precision, M2L | Coefficients | Points | Views | Tables | Scratch | Total (MB, *model*) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cube 10⁵ | 3, f32, dense | 0.6 | 4.8 | 8.4 | 0.3 | 77 | 91 |
| cube 10⁵ | 8, f32, dense | 3.0 | 4.8 | 8.4 | 8.7 | 128 | 153 |
| cube 10⁵ | 8, f64, dense | 6.1 | 9.2 | 8.4 | 17 | 128 | 169 |
| cube 10⁵ | 12, f64, rotation | 13 | 9.2 | 8.4 | 6.0 | 0 | 36 |
| cube 10⁵ | 18, f64, rotation | 27 | 9.2 | 8.4 | 24 | 0 | 69 |
| cube 10⁵ | 20, f64, dense | 33 | 9.2 | 8.4 | 517 | 128 | 695 |
| Plummer 10⁵ | 8, f32, dense | 4.2 | 4.8 | 11 | 8.7 | 128 | 157 |
| Plummer 10⁵ | 18, f64, rotation | 38 | 9.2 | 11 | 24 | 0 | 81 |
| cube 10⁶ | 8, f32, dense | 24 | 48 | 73 | 8.7 | 128 | 282 |
| cube 10⁶ | 18, f64, rotation | 216 | 92 | 73 | 24 | 0 | 405 |
| cube 10⁶ | 20, f64, dense | 264 | 92 | 73 | 517 | 128 | 1,074 |

Every workload of the phase fits the M3 Max (64 GB unified, Metal's working-set limit
reported by F8) by two orders of magnitude. For larger N the coefficients, points and
views grow with N and the tables and scratch do not: per 10⁶ points on the cube about
429 MB (dense p = 20, f64) and 381 MB (rotation p = 18, f64) beside a fixed 645 MB and
24 MB. A 40 GB card then holds the dense p = 20 f64 case to about N ≈ 9 × 10⁷ and
rotation at p = 18 to about 10⁸; an 80 GB card about 1.8 × 10⁸ and 2.1 × 10⁸ (*model*;
uniform trees, K and E_V per point as at N = 10⁶). The dense M2L tables (492 MB in f64 at
p = 20) dominate only at small N.

**At build**, `DeviceOperator::new` sums the formula before allocating anything and
compares it with `max_memory()` less `memory_usage()` (F8). If it does not fit, `build`
returns `SettingsError::DeviceMemory { needed, limit }`, on the one rank the device path
runs on. Allocation is never attempted past the limit, because CubeCL panics on failure
(F7). Where the backend reports no limit, the check is skipped and stated in
`DeviceReport`.

## 5. Precision and capability

### 5.1 The f64 check per backend

`Device::supports(Precision::F64)` is `client.properties().supports_type(f64)` with
arithmetic among its uses (F12):

| Backend | f64 | Source |
| --- | --- | --- |
| CPU runtime | yes | `cubecl-cpu-0.11.0-pre.4/src/runtime.rs` registers F64 |
| Metal (wgpu-msl) | no | `cubecl-wgpu-0.11.0-pre.4/src/backend/metal.rs` registers F16 and F32 only |
| CUDA | yes, in principle | `cubecl-cpp-0.11.0-pre.4/src/shared/base.rs` registers F64 (0.10.0 had it commented out); never run here |

### 5.2 The refusal path (requirement 7)

In step 1 of `FmmBuilder::build`, with the settings: `Backend` compiled in
(`BackendNotCompiled`), `Device::open` (`NoDevice`), then `supports(precision)`
(`PrecisionUnsupported`). All three are local results agreed by step 1's existing
all-reduce (`agree` in `fmm.rs`): the rank that found the error returns it, the others
`FmmError::OtherRank`. No new collective, no panic, no cast: an f64 `FmmBuilder` with
`Backend::Metal` never builds.

### 5.3 The CPU runtime as the f64 correctness backend, and Metal's math mode

The CPU runtime is the only backend here that runs f64. It compiles with LLVM O3 and no
fast-math flags (F16): no reassociation, correctly rounded division and square root
(*inferred* from the absence of fast-math flags; T3 measures). It does contract: before
LLVM, cubecl-opt's `InstCombinePass` fuses every product whose only use is an add or
subtract into an fma, on every backend and with no switch to turn it off (F16, read;
found by T2). The host's Rust code does not contract. So a device kernel there that
repeats the host's operations in the host's order equals the host bit for bit only
where no inexact product feeds an add or subtract, or where T3 finds a formulation that
keeps the fusion out; elsewhere it agrees to rounding. Sections 6 and 13 use bit
identity only where it holds.

Metal is the opposite case. wgpu-msl passes its MSL through
`create_shader_module_passthrough`, and wgpu-hal 30.0.1 compiles it with a bare
`MTLCompileOptions::new()` that sets no math mode (read, F16; the options with language
version 3.2 in `cubecl-wgpu`'s `backend/metal.rs` belong only to the canary). Apple
documents fast math as the default of those options (*inferred*, not measured). That may
mean approximate division and `rsqrt`, reassociation and flushing to zero: exactly what
CONVENTIONS §3.13's r² = 0 argument excludes. T3 measures it. If a formulation cannot
restore the argument, the fallback is the `metal-native` runtime (`cubecl-metal`), which
compiles with `MTLMathMode::Safe` and precise math functions (F16). That changes the
`metal` feature's meaning and is sign-off question 10.

## 6. Kernel mapping

Refines design §6.4. Each kernel is generic over `F: Float` with p (or n_c), the
gradient flag and its layout parameters as comptime values (F11).

### 6.1 Rules every kernel follows

- **Ownership.** Every output value has exactly one owning unit per launch: a target
  point (P2P, L2P, M2P), a coefficient of a box (P2M, P2L, reductions, rotation), or a
  column of a temporary (GEMM). No atomics (requirement 5).
- **Accumulation.** The owner loads the output value into a register, adds its
  contributions one by one in row order, and stores it: `acc = out; acc = acc + c₁;
  acc = acc + c₂; …; out = acc`. That is the same sequence of roundings as the host's
  in-place `+=` (requirement 4), and it makes a split of a row into several launches
  give the same bits (chunk invariance).
- **One thread, one stream.** Every launch is issued from the thread that called the
  operator, never from a rayon worker, so all of them land on one CubeCL stream and run
  in order (F10). Kernels of consecutive level calls need no sync between them.
- **Frames from integer coordinates.** The index arena holds, per leaf, its level and
  index (i_x, i_y, i_z), and per box of each level its index. A kernel forms,
  with L = max(l_s, l_t), N = (2 i_s + 1) 2^(L − l_s) − (2 i_t + 1) 2^(L − l_t) in `i32`
  (|N| ≤ 131,070), converts
  it to F (exact) and multiplies by 2^(l_t − L) from a table of exact powers of two, as
  `geometry::relative_frame` does: no step rounds (CONVENTIONS §3.13, "Relative frames").
  T6 tests the device frames against `relative_frame` bit for bit.
- **Per-backend layouts.** Each family has a layout value (units per cube, tile sizes,
  vector width) chosen at build from `DeviceInfo::backend`, or by the tuner (Section
  10). Layouts are comptime parameters, so each (layout, p, precision) is one compiled
  kernel.
- **Launch hygiene.** Index arrays are validated on the host at upload (every entry in
  range), so kernels launch with `launch_unchecked`; debug builds launch the checked
  form. A level call with an empty view launches nothing.

### 6.2 P2P

**GPU layout (default on Metal and CUDA).**

| Item | Choice |
| --- | --- |
| parallel unit | one cube per target leaf of the level (row r of `near`); unit u owns target points u, u + U, u + 2U, … |
| comptime | U (units per cube, also the source tile): 64 by default on Metal; candidates 32, 64, 128; gradients |
| shared memory | one tile of U mapped sources and charges, 4 U s bytes (1 KB at U = 64 in f32) |
| loop, per target block | load φ̂ (and ĝ) from the output; for each source leaf j of row r, in row order: the frame (ĉ(j\|t), r̂(j\|t)), identity for j = t; for each tile of j's points: each unit maps one source, ŷ = ĉ + r̂ u_j (u_j itself for j = t), and its charge into shared memory; `sync_cube`; each unit adds the tile's sources in point order; `sync_cube`; finally store |
| r² = 0 | as the signed-off T3 addition to §3.13 requires |
| reads | `near` (row offsets, entries), source and target point offsets, the source store, target input, leaf coordinates; writes the target output of the level's leaves |

- **Order.** Each target adds its sources in near-row order and, inside a leaf, in point
  order: the order of `Kernels::p2p_target` with `P2pKernel` (simd-p2p.md requirement 4).
  Inside one pair term the arithmetic is the T3 formulation.
- **Mapping.** r̂ u_j is exact (a power of two times u), so fl(ĉ + r̂ u_j) is the host's
  value whether or not the backend fuses it into an fma. Reassociation could still
  change it; T3 checks it.
- **Leaves larger than a tile** loop over target blocks and source tiles. **Empty
  leaves**: n_t = 0 or n_j = 0 is the same for every unit of the cube, so the exits are
  uniform and no `sync_cube` diverges. The partial last tile and target block run the
  same code with masked units.
- **Small leaves.** The Plummer sphere averages 17.6 points per leaf (Phase 3S T7), so a
  64-unit cube would idle most units. A second GPU layout gives each target leaf one
  plane (32 units on Metal, F13) and several leaves to a cube, each plane staging its
  own tile and syncing with `sync_plane`. Both layouts are candidates for T6 and T12.
- **Grid.** One cube per target leaf; above the per-dimension cube-count limit
  (`max_cube_count`, F13) a 2-D grid.

**CPU layout (only if decision 10 sets a CPU-runtime target).** Targets in
`Vector<F, N>` lanes with the host's width and K vectors per unit, one unit per core
(units per cube capped by `threads(n)`, Section 11), each unit a contiguous range of
target leaves, sources broadcast in near-row order, no shared memory, no `sync_cube`,
explicit `fma`, `sqrt` and division (T3's prototype). The layout value is chosen at build
from `DeviceInfo::backend == Cpu`; the two layouts share the launch wrapper, the frames
and the order, so every P2P test runs on both. Not designed further here.

### 6.3 P2M, L2P, P2L and M2P

**Harmonics.** Regular and irregular solid harmonics (and their gradient ladders) by the
Cartesian recursion of CONVENTIONS §3.5, into a local array of n_c values (n_c of degree
p + 1 for M2P's gradient), with p comptime and the loops over n and m unrolled. The
recursion coefficients are **formed in the kernel**, operation for operation as
`nd_fmm_math::harmonics` forms them (`fmm-math/src/harmonics.rs`): for `regular`, `1 /
int(2m)`, `int(2n − 1) · z / int((n + m)(n − m))` and `−r² / int((n + m)(n − m))`; for
`irregular`, one `inv_r2 = 1 / r²` per point, then `int(2m − 1) · inv_r2`, `int(2m + 1)
· z · inv_r2`, `int(2n − 1) · z · inv_r2` and `−int((n − 1 − m)(n − 1 + m)) · inv_r2`,
products in that order. Their integers are comptime constants; the divisions and
products run in F. Not precomputed on the host, because:
- on the CPU runtime the device harmonics then follow the host's operations; they
  equal the host's bit for bit only if T3 finds a formulation that keeps cubecl-opt's
  fma fusion out of the recursion (F16, Section 5.3), and agree to rounding otherwise;
- on Metal they agree to the rounding of its division, which T3 measures; precomputed
  reciprocals would differ from the host everywhere.

Local arrays of 441 values at p = 20 are large for a GPU unit, but p = 20 runs in f64
only, on the CPU runtime; f32 stops at p = 8 (81 values).

| Operator | Parallel unit | Order inside the target | Reads |
| --- | --- | --- | --- |
| P2M | one cube per local leaf of the level; unit c owns coefficient slots c, c + U, …. Points in tiles: each unit computes the harmonics of one point of the tile into shared memory (tile × n_c values), `sync_cube`, then each owner adds q_j conj(R(u_j)) for the tile's points in point order | points in point order: the order of `leaf::p2m` (each source in turn, `add_conjugate`) | `p2m` view, source offsets, source store |
| P2L | one cube per target box with a non-empty X row; the same owner scheme, irregular harmonics at the frame (ĉ(t\|j), r̂(t\|j)) | X row in leaf-index order, then point order (`leaf::p2l`) | `x` view, box and leaf coordinates, source store |
| L2P | one cube per local leaf, one unit per target point; the leaf box's local staged once in shared memory (n_c s) | one contribution, summed as `leaf::l2p`'s `Expansion::evaluate` sums it | `l2p` view, target offsets, target input |
| M2P | one cube per local leaf, one unit per target point; walks the W row (boxes on l + 1, ascending) staging each multipole in shared memory; frame (ĉ(s\|t), r̂(s\|t)) | W row in box-index order; inside a box as `leaf::m2p` | `w` view, box and leaf coordinates, multipoles of l + 1 |

The coefficient-owner scheme replaces the "reduction in shared memory or plane
operations" of design §6.4 for P2M and P2L: it keeps the host's point order, so it needs
no reduction tree and repeats the host's sum exactly. Shared memory per tile is
P n_c s: 10 KB for 32 points at p = 8 in f32 (within Metal's 32 KB, F13), 28 KB for 8
points at p = 20 in f64 on the CPU runtime. L2P runs before M2P because the evaluator
calls them in that order (redesign §7.5): two launches on one stream.

### 6.4 M2M, L2L and dense M2L: one grouped translation

The three translations with dense tables have the same shape (redesign §6.4): for each
group g (octant or offset) of a level, gather the batch's inputs, multiply by table g,
and add into the batch's targets. `translate::grouped` does all three.

**Layout.** A gathered matrix X has one column per batch entry, n_c values contiguous,
like a level buffer. Read row-major it is k × n_c, and a table stored column-major
(§3.12) is Aᵀ read row-major. So Y = A X is, in row-major terms, (k × n_c) · (n_c × n_c)
= X_rm · Aᵀ_rm: a plain row-major GEMM with M = k, N = K = n_c and no transposition. The
spike's kernels compute C = A X with X row-major n_c × B; T8 swaps the roles of rows and
columns accordingly.

**Three launch structures for one level of dense M2L**, with the uniform cube at p = 8 in
f32 as the example (*model*: K_4 = 4,096 boxes and about 600,000 of the 640,584 V pairs
on level 4; about 10 µs per launch on wgpu/Metal, *estimate* from the spike's 9–22 µs
for a 13 MFLOP GEMM; 8.4 GFLOP of M2L at the spike's 3.5 TFLOP/s is about 2.4 ms):

| | (A) per offset, index order | (B) grouped GEMM, row-ordered reduction | (C) stacked |
| --- | --- | --- | --- |
| launches per level | 3 per non-empty offset: ≤ 948; 2,844 for levels 2–4, about 28 ms of launch overhead, launch-bound by about 10× | 3 per offset chunk (gather, grouped GEMM, reduction): about 12 on level 4, 18 for levels 2–4 | 2 per box chunk (one GEMM M_l · [A₀ᵀ … A₃₁₅ᵀ], reduction) |
| flops | the needed 2 n_c² E_l | the needed | 316 K_l / E_l ≈ 2.2× the needed on level 4 |
| temporary memory | X and Y of the largest batch, 2 n_c k_max s (2.3 MB) | X and Y of an offset chunk, ≤ S_scr; all of level 4 would be 389 MB | Z = 316 n_c K_l s, 419 MB on level 4; chunked by boxes |
| conflict-free | yes: targets distinct in a batch | yes: GEMM columns distinct; the reduction owns targets | yes: as (B) |
| each target's order | L ← L + Y_d, offsets ascending | the reduction does `acc = L[:, t]; for e in row t: acc = acc + Y[:, pos(e)]; L[:, t] = acc`: the same additions in the same order as (A), so **bit-identical to (A)** with the same GEMM | the same reduction; equal to (A) only if Z's columns come from the same kernel and k-order |
| library GEMM | one launch per offset | one batched launch per chunk, groups padded to the chunk's largest batch (F22), or one launch per group | one large launch |

**Choice: (B).** It is (A)'s arithmetic in (A)'s order with about 1% of its launches,
needs no wasted flops, and bounds its memory by S_scr. A grouped launch is safe here
because no cube writes a target: the GEMM writes columns of Y, and only the reduction,
which owns each target, writes L. (A) stays in T9 as the test reference, bit for bit.

Details:
- **pos(e)**, the column of Y holding row entry e's product, is the "row-to-batch" map:
  built on the host at build by walking the batches in group order with one cursor per
  target (O(E), from `GroupedCsr`'s public accessors), uploaded once (4 E bytes).
- **Chunks** are contiguous ranges of groups whose X and Y fit in S_scr (default 128 MB,
  *model*; T9 measures). Rows are sorted by group, so a chunk's entries in row t are a
  contiguous sub-range; its start per row and chunk is precomputed at build. Chunks run
  in ascending group order, so each target still meets its offsets in index order.
- **The grouped GEMM** (hand-written) takes a tile schedule, the list of (group, first
  column) of every tile of the chunk, built at build: one launch, no idle cubes, each
  tile using its group's table.
- **L2L**: a child has one parent, so its "reduction" is a scatter-add whose targets
  are distinct over the whole level (`movement::scatter_add_columns`, one launch).
  **M2M**: a parent has up to eight children by octant, so the reduction runs, with
  groups = octants. Both passes (`m2m_local`, `m2m_global`) use the same code.
- **Inside one contribution**: the hand-written GEMM sums over k in ascending order into
  one accumulator per output, starting from zero (fused or not as T3's rules say); the
  library sums 8 × 8 × 8 MMA tiles in k-tile order, inside a tile in the hardware's
  order, fixed for a device and driver (*inferred*). The result is added to L once.
  The host adds each product into L directly (`MatrixSet::apply`), so device and host
  differ in rounding (Section 9.2).
- **Rejected**: accumulating in the GEMM from a gathered copy of L (β = 1), which would
  repeat the host's sequence for an unfused hand-written kernel. It costs a gather and
  a scatter of L per batch (the stage is memory-bound at small p), and the library path
  cannot follow it.

### 6.5 The library matmul and the hand-written GEMM

| Precision | p | GEMM | Why |
| --- | --- | --- | --- |
| f32 | ≤ 7 | hand-written | the signed-off rule (README): library CMMA from p = 8. The spike found CMMA rejected at p = 4 (n_c = 25; it "needs Nc ≳ 64") and the unit path 10× slower there; p = 5–7 were not measured |
| f32 | ≥ 8 | library, `Strategy::MultiLevel(SimpleCyclicCmma)` named explicitly, where the input-precision guard passes; else hand-written | 3.5–4.4 TFLOP/s on Metal (spike); the strategy `Auto` tries first (F20). *Changed after T9 for M2L* (Section 17): M2L runs the hand-written kernel at every p by default; M2M and L2L keep this rule |
| f64 | all | hand-written | no f64 MMA (F21); the library's unit path is 2.2–2.6× below the hand-written kernel (spike model) |

- Never `Strategy::Auto`: it can change with the setup error it meets, and it panics on
  any error other than `Unavailable` (F20).
- **Input-precision guard** (README): the library is used for f32 only if the backend
  registers no TF32 type, since the accelerated routines otherwise switch f32 stages
  to TF32 (F21, `adjust_dtypes`). Metal registers none; CUDA does, so f32 on CUDA uses
  the hand-written kernel. T9 checks the resolved `MatmulElems`, not the backend name.
- A shape the library rejects at build (`MatmulSetupError::Unavailable`) goes to the
  hand-written kernel, decided by shape alone, so the choice is the same on every
  evaluation. The choice per level is fixed at build and reported.
- The hand-written kernel's layout per backend (design §6.5): row blocks (coalesced X)
  on Metal, column strips (X in cache) on the CPU runtime, from the spike's
  `tiled-reg-cols` and `tiled-reg-rows`.

### 6.6 Rotation M2L

| Item | Choice |
| --- | --- |
| parallel unit | one cube per target box with a non-empty V row, U ≥ n_c units (n_c rounded up to the plane size: 96 at p = 8, 448 at p = 20) |
| shared memory | two working vectors of n_c values (7 KB at p = 20 in f64) |
| loop | for each (source s, offset d) of row t in offset order: load the multipole of s; the five steps of `ShiftTables::apply`: z-rotation (one unit per (+m, −m) pair), forward y-blocks (one unit per output slot (n, m), summing the degree's 2n + 1 inputs), coaxial step (one unit per (j, i), summing over n in the order-i matrix), backward y-blocks, z-rotation back, each followed by `sync_cube`; the last step adds into the unit's accumulator, which started from the local as L2L left it. `Up` and `Down` offsets run the coaxial step alone (with the parity for `Down`) |
| order | V row in offset order (requirement 4); inside a pair, the order of `ShiftTables::apply` |
| device tables | the M2L `ShiftTables` only (M2M and L2L use the dense octant GEMM, README): forward blocks [polar][B], backward blocks [polar][B], azimuth factors [azimuth][2p], coaxial factors [distance][C], and per offset a u32 quadruple (alignment, polar, azimuth, distance), concatenated at upload from the public accessors; no change to `nd-fmm-tables` |

On the CPU runtime a kernel that repeats `ShiftTables::apply` step for step can equal
`RotationTables::m2l` bit for bit; T10 tests that where it holds. Many units idle in the
steps with fewer outputs; packing several targets into one cube at small p is a T10
candidate.

### 6.7 Top levels

Levels 0 and 1 have no V lists, and near the root M2M and L2L have at most eight parents
or children. With grouped launches each level costs at most about three launches per
kind, so the top levels add about 30–50 launches per evaluation, 0.3–0.5 ms
(*estimate*). Running them on the host instead needs a download, a host call and an
upload, so at least one sync (1.5 ms on wgpu/Metal, spike): more than it saves. **The top
levels stay on the device.** T11 may merge further: the M2L tables do not depend on the
level, so one grouped GEMM over every level's V batches can run at the first `m2l` call,
and each level's reduction after its L2L. The additions per target are unchanged, so the
output is bit-identical; T11 adopts it only if it measures the per-level launches as
significant.

### 6.8 `Classes` on the device

**Run as `Dense` from `M2lClasses::expand()`.** The device holds the expanded 316 tables
(dense memory), uploaded once and then dropped on the host; the host keeps its class
tables for the fallback. `DeviceReport` names the strategy "Classes, run as dense on the
device".
- *For*: every host strategy stays selectable with a device backend, so
  `tests/mpi_exec.rs`, which runs `Dense`, `Classes` and `Rotation`, runs unchanged; the
  host keeps the memory saving that `Classes` exists for.
- *Against*: the device saves no memory; the fit check covers that. `expand` forms each
  matrix as T_L(P) M2L(d₀) T_M(Pᵀ) in T and "agrees with `M2lTables::build` up to
  rounding" (`fmm-tables/src/m2l.rs`), so device `Classes` and device `Dense` agree to
  rounding, not bit for bit, as on the host. Expanding costs O(p⁵) per offset on the
  host at build, less than building the dense tables (O(p⁶)).

## 7. Host fallback and migration

### 7.1 Placement per kind

`Placement` maps each `OperatorKind` to `Device` or `Host`. A kind runs on the host if
`FmmBuilder::host_fallback` names it, or if no device kernel exists for it yet under the
resolved strategy (for example `Rotation`'s M2L before T10). The placement is fixed at
build and reported (`DeviceReport`).

### 7.2 How a kind runs on the host

The device operator owns host mirrors, created only if some kind is on the host:
`LevelBuffers::new(lens, sizes)` for multipoles and locals and `LeafStore::new(counts,
o)` for the target output, the public constructors of `nd-fmm-plan`. A host-fallback
call:

1. downloads the call's device inputs and its output region into the mirrors (exact
   copies; one sync);
2. builds the same batch struct with the call's own `level`, `index` and views, the
   mirrors' slices for multipoles, locals and target output, and the evaluator's batch
   slices for sources and target input (current on the host: `Fmm` writes them);
3. calls the wrapped `LaplaceOperator`'s own `FmmOperator` method, so the bodies are
   `Kernels::*_target`, serially or on the `Fmm`'s pool;
4. uploads the output region.

At the deepest level `m2p` receives an empty `LevelSlice` for level L + 1 (its rows are
all empty); the fallback passes the batch's own slice through, since `LevelSlice` has no
public constructor.

| Kind (level l) | Downloaded before | Uploaded after |
| --- | --- | --- |
| `p2m` | multipoles of l | multipoles of l |
| `m2m` (either pass) | multipoles of l + 1 and of l | multipoles of l |
| `m2l` | multipoles of l, locals of l | locals of l |
| `p2l` | locals of l | locals of l |
| `l2l` | locals of l − 1 and of l | locals of l |
| `l2p` | locals of l, the level's target output | the level's target output |
| `m2p` | multipoles of l + 1, the level's target output | the level's target output |
| `p2p` | the level's target output | the level's target output |

Each fallback call costs one sync and n_c s bytes per box moved (o s per target point).
T5 states the exact per-evaluation formula for its scenarios and checks it against the
counters.

### 7.3 Why every kind on the host equals the host path bit for bit

By induction over the evaluator's calls, the device stores equal the host path's stores
after each call:
- **Start**: `reset` zeroes the host path's stores; `begin_evaluation` writes +0.0 into
  the device stores (`T::default()`, the same bits). Coordinates and charges are exact
  copies, and on one rank the exchanges move nothing (Section 4.2).
- **A call**: the mirrors receive exact copies of device stores, equal to the host
  path's by hypothesis; the same `LaplaceOperator` method (same tables, gradients,
  `max_leaf_points`, P2P kernel) runs on the same views and data, and is bit-identical
  for every thread count (C3.5); the upload copies its output exactly.
- **End**: the download is exact, and `Fmm` scales the output with the same code.

The wrapped `LaplaceOperator` must therefore be built exactly as `build` builds the host
operator (`LaplaceOperator::new(tables, gradients, max_leaf_points).with_p2p(..)`,
`.with_pool(..)`), which T5's tests check by comparing every `tests/mpi_exec.rs`
scenario with the host path bit for bit.

### 7.4 Migration, T5 to T11

| Task | Kinds on the device by default | Everything else |
| --- | --- | --- |
| T5 | none: the device path moves data as designed and computes every kind on the host fallback | bit-identical to the host path |
| T6 | P2P | host fallback |
| T7 | P2M, L2P, P2L, M2P | host fallback |
| T8 | M2M (both passes), L2L, for every strategy | host fallback |
| T9 | M2L under `Dense` and `Classes` | `Rotation`'s M2L on the host |
| T10 | M2L under `Rotation` | — |
| T11 | every kind, by default; launch scheduling; transfers at the minimum | the fallback stays as a test aid |

T6, T7 and T8 change disjoint kinds and can merge in any order (README, "Tasks"). Each
task's FMM test runs with its kinds on the device and the rest on the fallback, which
isolates a kind when an accuracy check fails.

## 8. Launches, synchronisation and timing

### 8.1 Launches and syncs per evaluation

Every kind on the device, one rank; c_l the offset chunks of level l (Section 6.4); a
level call with an empty view launches nothing:

| Stage | Launches |
| --- | --- |
| `begin_evaluation` | 3 zero kernels (multipoles, locals, target output) + 1 charge scatter |
| `upward_local` | per level: `p2m` 1; `m2m` 3 (gather, GEMM, reduction); on one rank level 0 has no local-pass row, since the root is `Global` |
| `upward_global` | on one rank `m2m(0)` of the global pass for the root, which is `Global` (Section 4.2): 3 (gather, GEMM, reduction), none if the root is a leaf; every other level's global view is empty |
| `downward` | per level: `l2l` 3 (gather, GEMM, scatter-add); `m2l` 3 c_l (dense) or 1 (rotation); `p2l` 1 |
| `evaluate_leaves` | per level: `l2p` 1, `m2p` 1, `p2p` 1 |
| `read_output` | 1 download, **1 sync** |

That is about 50 launches for the uniform cube at p = 8 (levels 0–4, one leaf level, no
W or X: 4 + `p2m` 1 + local M2M 9 + global M2M 3 + L2L 12 + M2L 18 with four offset
chunks on level 4 + L2P and P2P 2 = 49) and about 150 for the Plummer sphere (leaves on
six levels) (*model*), against roughly 3,000 with one launch per offset. *Measured
(T13, Metal f32, N = 10⁵):* 40, 43 and 46 launches on the cube at p = 3, 6 and 8 (34 under
`Rotation`), 101–107 on the Plummer sphere (87 under `Rotation`); 64–133 on the cube at
N = 10⁶ (Section 17). At about 10 µs
a launch (*estimate*), launch overhead is 0.5–1.5 ms per evaluation, about one sync's
cost. wgpu submits the queued work every 32 tasks (F10), so the device starts before the
evaluation's one sync.

### 8.2 The one sync

The only sync is `read_output`'s download (F9: launch errors surface there too). On Metal
and CUDA uploads (`write`) and launches do not wait. On the CPU runtime the stream drains
before a host write and before every shared-memory launch (F10), so there the charge
upload waits for any work still queued (none at the start of an evaluation) and each
GPU-shaped P2P, P2M, P2L or rotation launch waits for the work before it: launches still
run in order, but the host does not run ahead of the device. The CPU runtime is a
correctness backend, so this costs no target. Host-fallback calls add one sync each
(Section 7.2); with every kind on the device there are none.

### 8.3 `StageTimings` without a sync per stage

`Fmm::evaluate` keeps timing each stage call on the host. With a device backend that
measures **enqueue time**, and the device's work shows in `output`, which contains the
waiting download. The docs of `StageTimings` say so. Two ways to time stages properly:

- **Synchronous stages** (`FmmBuilder::synchronous_stages(true)`, T5): `Fmm` calls the
  operator's `sync` after every stage. Exact per-stage wall times, at the cost of five
  more syncs (about 7.5 ms on wgpu/Metal, *estimate*). For reports only, never the
  default.
- **Device timestamps** (T11, where `TimingMethod::Device`, F14): one profile window per
  stage (`profile_start`/`profile_end`), resolved after the download. On wgpu a window
  flushes at its start but does not wait (F14), so normal evaluations keep one sync.
  CubeCL budgets 28 live timestamp query sets on Metal (Metal's cap is 32 per device;
  `CUBECL_MAX_METAL_TIMING_QUERY_SETS` overrides it) and reuses its sets past the budget
  (`cubecl-wgpu-0.11.0-pre.4/src/compute/timings.rs`); six stages stay under it. On the CPU
  runtime a window waits for the stream at both ends (F14, F10), so there it is the
  synchronous mode. T11 adopts it only after
  checking that it adds no sync and does not change the output; otherwise the
  synchronous mode stays the only stage timing.

`BuildTimings` gains a `device` field: opening the device, uploads, the device's extra
tables and tuning.

## 9. Determinism and accuracy

### 9.1 Bit-identical from evaluation to evaluation and run to run (requirement 6)

For a fixed tree, backend, device, build and resolved choices:
- one thread enqueues every launch, so all of them run in order on one stream (F10);
- the launch sequence, the chunks, the tile schedules, the layouts and the GEMM per level
  are fixed at build from the plan and the device (or read from the tuning cache);
- no atomics, and no reduction whose order depends on scheduling: every owner adds in a
  fixed order (Section 6.1);
- the library matmul runs one explicitly named strategy on fixed shapes (its tile
  selection is a function of the shape, *inferred*), never `Auto` (F20);
- the tuned choices are fixed at build; two `Fmm`s built from the same cache, or under
  the static rule, choose the same.

Without a tuning cache a re-tune at a later build may choose differently, so outputs of
two such builds agree within the FMM bounds, not bit for bit; the docs say so (T12).

### 9.2 What differs from the host path, and the tolerances

| Source | Where | Effect |
| --- | --- | --- |
| summation inside a product | GEMM translations (Section 6.4): k-order, blocking, CMMA tiles; one addition into the output instead of one per product | rounding-level differences, ≤ (p + 1)² u_T relative to the terms (worst case), typically √n_c u_T |
| fma contraction | CUDA: the default LLVM path marks fadd, fsub and fmul `contract` (F17, read); Metal under fast math (*inferred*, F16); on every backend, the CPU runtime included, cubecl-opt's `InstCombinePass` fuses each product whose only use is an add or subtract (F16, read; T2 measured it) | one rounding fewer per multiply–add |
| inverse square root and division | P2P, harmonics, rotation | per T3: correctly rounded on the CPU runtime (*inferred*), approximate under Metal's fast math (*inferred*) |
| reassociation, flush to zero | Metal under fast math (*inferred*) | T3 decides the formulation or the runtime (Section 5.3) |

Where none applies, a device kernel that repeats the host's operations in the host's
order can equal the host bit for bit: on the CPU runtime the charge scatter and the
zeroing always, and the frames where their products are exact (powers of two and small
integers, so fusing them changes nothing). The harmonics, P2M, L2P, P2L, M2P and
rotation are multiply–add chains, which cubecl-opt fuses, so they are bit-identical
only if T3 finds a formulation that keeps the fusion out (T7, T10). The tests assert bit
identity where it holds, and tolerances elsewhere.

**The README's FMM tolerances are confirmed** (*estimate*): per-operator differences of
about 1e-15 relative in f64 stay far below 1e-12 in the relative L2 of the output after
the dozen or so translations a value passes through; in f32 the largest differences are
in near-field sums, within the 1e-6 sum contract of C3S.4, so the relative L2 difference
is of order 1e-6 at most, inside 1e-5. The 0.1% and 5% rules hold where truncation
dominates (Section 2). The operator bounds stand with the addition of Section 2 for
dense f64.

### 9.3 What T3 must establish before T6 and T7

Per backend (Metal f32; CPU runtime f32 and f64; CUDA from the code generator, not
measured):
1. whether `sqrt`, division, `inverse_sqrt` and `inverse_sqrt` with one Newton step are
   correctly rounded, and their errors in u_T over the §3.13 domain;
2. whether `a * b + c` is contracted, whether an explicit `fma` is honoured, whether
   `x − (y + z)` is reassociated, and whether subnormals flush to zero;
3. whether ŷ = fl(ĉ + r̂ u_s), d_k = fl(u_t,k − ŷ_k) and r² survive as written, so that
   r² = 0 exactly for coincident stored points and every other r² ≥ 2⁻¹⁰⁶; if not, the
   formulation or option that restores it, or the switch to `metal-native` (Section 5.3);
4. the P2P formulation per backend and precision, and its pair-term and sum errors
   against `nd_fmm_ref::p2p` and `direct_sum` (the device P2P contract);
5. a degree-20 harmonic against `nd-fmm-math` and one GEMM against `MatrixSet::apply`
   per backend, so that T7 and T8 know which of their tests can be bit for bit;
6. decision 10: the CPU-shaped P2P against `nd-fmm-simd`.

## 10. Autotune (C4.7)

### 10.1 CubeCL's autotune or a strategy-level tuner

| | CubeCL's autotune (`LocalTuner`, F15) | A strategy-level tuner in `nd-fmm-exec` |
| --- | --- | --- |
| what it chooses | one kernel among candidates per call key | the M2L strategy for the whole `Fmm`, then per level the GEMM and layouts, and the P2P layout |
| when | at the first call with a new key, so inside the first `evaluate` | at build, outside `evaluate` (README) |
| dense against rotation | cannot: the strategy decides which tables are built and uploaded, and how much memory is used, before any call | yes |
| persistence | a database under `<workspace>/target/environment`, a user cache directory or what `cubecl.toml` says (F15, F24); process-wide configuration | a file in the directory the caller passes, nowhere else |
| fixed for the `Fmm`'s lifetime (requirement 6) | not by itself: the key and the cache decide per call | yes |

**Recommendation: a strategy-level tuner in `nd-fmm-exec`** (module `tune`, T12), on
timing primitives of `nd-fmm-kernels` (`Device::sync` and its counters). CubeCL's
autotune is never called, and its `persistence` feature stays off (with `metal` it is
on but unused, and leaves an empty store; Section 3.4), so the
rule of the host table cache holds: a directory only when the caller passes one, no
default directory, no environment variable read by this workspace. CubeCL's own
configuration file and variables (F24) are documented in `fmm-kernels/CLAUDE.md`, not
set or read by the crates.

### 10.2 The key

- backend, device name and compiler (`DeviceInfo`), precision, p, gradients;
- for the GEMM and layout choices, the level's V pairs (or octant pairs) bucketed to a
  power of two; for the P2P layout, the mean points per leaf bucketed to a power of two;
- versions: CubeCL, `nd-fmm-kernels`, `CONVENTION_VERSION`, the candidate set and the
  file format. A different version makes an entry stale.

### 10.3 The candidates

- M2L strategy: `Dense` with the library GEMM (f32, p ≥ 8, the input-precision
  guard passed), `Dense` with the hand-written GEMM in each backend layout, `Rotation`
  (if its kernel exists). `Dense` only where its tables fit (Section 4.6).
- Per level: the GEMM (library or hand-written, layouts) and the offset-chunk budget.
- P2P: the units per cube (32, 64, 128) and the plane-per-leaf layout (Section 6.2);
  the CPU layout's K, if decision 10 adds it.
- Never a library strategy that rounds f32 inputs to TF32, F16 or BF16: such a strategy
  is not registered, so it is neither timed nor chosen (Section 6.5).

### 10.4 When tuning runs

At build, after the uploads, only when a tuning-cache directory is given and the cache
lacks the key. Each candidate is compiled and launched once (warm-up, excluded), then
timed on this `Fmm`'s own largest level of that kind, queued as in an evaluation, median
of five batches of at least 10 ms each. The static rule's candidate is timed first, so a
budget cut leaves at least it. Budget: 10 s per build by default (*estimate*: about six
candidates for the strategy plus a few per level bucket, each about 0.3 s with
compilation), enforced by a deadline checked between candidates. The time is reported
in `BuildTimings::device`. Building the dense tables for a candidate goes through
`table_cache` when given; without one, the dense candidate at p ≥ 12 is skipped (8.5 s
to build at p = 16, Phase 3 T8).

### 10.5 Persistence, staleness and the static rule

- `FmmBuilder::tuning_cache(dir)`: one file per (backend, device, precision, p), with
  the `TableCache` rules: a header with magic, format version and key, a checksum,
  atomic replacement by rename, safe for concurrent writers on every rank. A stale,
  truncated or corrupted file is rejected without a panic and re-tuned.
- Without a directory: no tuning, and the static rule applies:
  - **f32**: `Dense`; the library GEMM where p ≥ 8 and the guard passes, the
    hand-written GEMM otherwise;
  - **f64**: `Dense` for p ≤ 11, `Rotation` for p ≥ 12 (**provisional**).

**The f64 rule for p = 9–11.** Nothing here can time f64 on a GPU. The spike's roofline
model (central, A100) gives the rotation efficiency that would tie with the hand-written
dense GEMM: 13.2% of f64 peak at p = 8 and 5.4% at p = 12. Interpolating the dense
efficiency between them gives about 10.7%, 8.6% and 6.9% at p = 9, 10 and 11 (*model*).

No rotation M2L has been measured on a GPU. On one host P-core the table-driven rotation
reaches about 9% of f64 peak (*estimate*, against 4 FMA pipes × 2 lanes × 2 flops ×
4.05 GHz = 64.8 GFLOP/s):
- one M2L at p = 12 is (p + 1)(10p² + 32p + 9)/3 = 7,943 multiply–adds plus 26 additions
  (`ShiftTables::apply`, `fmm-tables/src/rotation.rs`), about 15,900 flops. Phase 3 T12's
  2.65 µs per V pair at p = 12 is the whole downward stage (L2L and P2L included) divided
  by the V pairs, so it bounds the M2L from above: at least 6.0 GFLOP/s, at least 9.3%;
- Phase 2's single table rotation at p = 18, about 8 µs (quoted in design §7, Phase 3,
  "Per-pair cost") for 24,225 multiply–adds (48,500 flops), gives 6.1 GFLOP/s, 9.3%.

**What the numbers say.** If a GPU rotation reached the same ~9% of its f64 peak, the
model puts p = 9 on the dense side (break-even 10.7%), p = 10 at a tie (8.6%, within
about 10%), and p = 11 on the rotation side by about 1.35× (6.9%). Dense keeps p = 11
only if the GPU rotation stays below about 7% of peak, which its short per-degree loops
make plausible but nothing measures. The rule as signed off, `Dense` through p = 11 and
`Rotation` from p = 12, is therefore **provisional**: it stays the static fallback for an
untuned key (f64 on CUDA included), and autotune (C4.7) replaces it wherever it can
measure. T10's Metal f32 rotation efficiency enters the model in T12; moving the
boundary is a sign-off decision. *As built (T12, decision 13):* the rule is kept, and the
device's `Auto` follows it rather than the host rule; Section 17 adds what T13 measured
for f32 at N = 10⁶. The dense tables are small at these degrees (52 MB in
f64 at p = 11), so memory does not decide it.

## 11. Threads and BLAS

The CPU runtime's parallelism is the number of units in a cube, not a setting: a launch
becomes one task per unit, each looping over every cube; tasks go to workers, one per
active logical CPU (efficiency cores included; pinning is only a hint on macOS), and kernels with `sync_cube` or shared memory get a dedicated
worker per unit, growing the pool (F18). Idle workers poll for 200 µs, then park.

| | Refuse `threads > 1` with the CPU backend | Keep rayon idle while device work runs |
| --- | --- | --- |
| oversubscription | none from rayon; the runtime's own threads remain | none if the two never run at once |
| holds by construction? | yes | yes: every launch is enqueued from the calling thread (F10 requires it), and a fallback call waits for the device (its download) before its host body runs on the pool |
| use of n | lost: the CPU runtime's thread count is not set by anything | can give n a meaning |

**Recommendation.** Both rules, each where it fits:
- **With the CPU backend no rayon pool is built, and `threads(n)` caps the units per
  cube of every CPU layout** (the column-strip GEMM, the movement kernels, and the CPU
  P2P layout if decision 10 adds it). The rank then has at most n busy runtime threads
  in those kernels, and the existing rule, ranks × n within the physical cores, covers
  the CPU runtime. GPU-shaped kernels (shared memory, `sync_cube`) keep their own cube
  size on the CPU runtime; they are correctness-only there and may start more threads
  than n, which their docs say. Host-fallback kinds run serially with this backend.
- **With Metal or CUDA** the pool of `threads(n)` is built as today and used only by
  host-fallback kinds; rayon is idle whenever device work runs, by the construction
  above. Worker threads never launch and never call MPI.

Phase 4 makes no BLAS call: every GEMM is a CubeCL kernel, compiled by the backend's own
compiler (on the CPU runtime, LLVM JIT code). `ThreadingReport` adds the CPU runtime's
units per cube; its BLAS variables and warnings are unchanged.

## 12. Errors

| Failure | When | Surfaces as | Agreed by |
| --- | --- | --- | --- |
| backend not compiled in | `build`, step 1 | `SettingsError::BackendNotCompiled` | step 1's existing all-reduce |
| no device (no adapter, MSL unavailable) | step 1, `Device::open` | `SettingsError::NoDevice` (the reason from `Backend::probe`, Section 3.3) | step 1 |
| unsupported precision (f64 on Metal) | step 1 | `SettingsError::PrecisionUnsupported` | step 1 |
| device on several ranks | after step 5 | `SettingsError::DeviceNeedsOneRank` | nothing needed: the same `comm.size()` everywhere |
| does not fit in device memory | after the plan, before any allocation | `SettingsError::DeviceMemory` | nothing needed: one rank |
| an upload, a table or a tuning step fails | the rest of the build | `FmmError::Device(reason)` | nothing needed: one rank |
| a launch or transfer fails during `evaluate` | attached to the output buffers, seen at the one download (F9) | `Fmm::evaluate` returns `FmmError::Device(reason)`; the `Fmm` keeps the error and every later `evaluate` returns it | nothing needed: one rank |
| an invariant breaks (index out of range, layout mismatch) | anywhere | panic, as on the host path | — |
| CubeCL's own panics: client creation (F3), allocation (F7), LLVM optimisation failure | — | panic | prevented where possible: the device constructors enumerate first (F2), memory is checked before allocating (Section 4.6) |

During `evaluate` the operator does not stop at the first failure: launches cost nothing
to enqueue, the stages run to the end (so on several ranks, later, no collective would be
skipped), and the error is reported once, at the download. C5.1 must agree such an error
on every rank; the `HostData` events of Section 4.5 that download are where it can be
seen early.

## 13. Testing and benchmarking

### 13.1 Test layers per task

| Layer | Oracle | Where | Tasks |
| --- | --- | --- | --- |
| kernel primitives: buffers, zero, gather, scatter-add, scatter | host loops, bit for bit | `nd-fmm-kernels`, no MPI | T4 |
| a kernel per operator | `nd-fmm-ref` (`direct`, `leaf`, `p2p`), `nd-fmm-math`, `nd-fmm-tables` (`MatrixSet::apply`, `RotationTables`) at the canonical frames, levels 2, 9 and 16 of a dyadic domain | `nd-fmm-kernels`, no MPI | T6–T10 |
| the device operator against the host operator, per kind | `LaplaceOperator` on the same batch, with a `Plan` built locally by the public `Plan::from_key_types` (as `fmm-plan/src/plan_tests.rs` does; `fmm-exec/tests/operator/` builds no plan and tests the per-pair methods `p2m_leaf`, `m2m_pair`, `m2l_pair`, … instead) | `nd-fmm-exec` tests, no MPI | T5–T10 |
| the device FMM against the host FMM and the direct sum | the host `Fmm` with the same settings; `direct_sum` | `tests/mpi_exec.rs` (a helper `evaluate_on_backends` beside `evaluate_threaded`), the ignored `tests/accuracy.rs` and `tests/adaptive.rs` | T5–T11 |
| transfers, launches, syncs | the formulas of Sections 4.1, 7.2 and 8.1 | `tests/mpi_exec.rs` | T5, T11 |

### 13.2 Which backend runs what

- **CPU runtime**, f32 and f64, small shapes (p ≤ 8 and B ≤ 10³ in the default run;
  the p = 20 sweeps and large shapes `#[ignore]`d): every kernel (the hand-written GEMM,
  not the library), every operator check, the debug FMM scenarios. CI if T4 keeps the
  job; otherwise by hand.
- **Metal**, f32: the same tests plus the library GEMM, `#[ignore]`d, by hand, outside
  the sandbox.
- **CUDA**: `cargo check --features cuda` only.

### 13.3 Compile time, and printing the backends

- Each test process opens one `Device` per backend (`OnceLock`) and reuses it, because
  CubeCL caches compiled kernels per process only (F23). Each test keeps its comptime
  variants (p, precision, layout) few. T4 measures the CPU-runtime suite and fixes its
  budget (README target: under five minutes warm, release).
- Every device test prints the `DeviceInfo` of each backend it ran, and a closing line
  "backends run: …; not run: …". A backend that did not run is never reported as
  passing, and CUDA is "type-checked, not run".

### 13.4 Benchmark harness, metrics and performance targets

In `nd-fmm-validate` behind `gpu` (T6–T13): stage times (synchronous mode), GFLOP/s and
GB/s against the spike's peaks (M3 Max f32 14.3 TFLOP/s, derived; 400 GB/s), P2P pairs
per second against host NEON at 1 and 12 threads, and launches, syncs and transfers per
evaluation. Metal, release, compilation excluded, many launches per sync, median over
repeated batches, the number of runs stated.

**C4.5**: the dense M2L GEMM at least 80% of the spike's throughput at the same (p,
columns) on Metal f32, GEMM only; gather and reduction reported separately (README).

**C4.2, derived here.** A peak model for P2P on the M3 Max GPU (*model*):
- 40 cores × 128 f32 lanes × 1.4 GHz = 7.2 × 10¹² lane-operations per second (the
  spike's 14.3 TFLOP/s counting an fma as one operation);
- per pair, potential only: 3 subtractions (d), 1 multiply and 2 fmas (r²), 1 inverse
  square root, 1 select (r² = 0), 1 multiply (q ρ), 1 add (φ̂) = 10 operations;
  with gradients 5 more (ρ², q ρ³, 3 fmas into ĝ) = 15;
- so **720 Gpairs/s (φ) and 480 Gpairs/s (φ, ∇φ)**. If the inverse square root takes 4
  issue slots, 550 and 400 (*assumption*, which T6 checks).

Proposed targets, measured and reported in T6, never asserted (*measured:* met with a
wide margin, 45–60% of the model; Section 17):
- W2 all-pairs, N = 10⁵, f32: at least **25%** of the model (180 and 120 Gpairs/s);
- W1 FMM-shaped, n_t = 64, f32, with at least 4,096 target leaves per launch: at least
  **10%** (72 and 48 Gpairs/s). That is at least 1.5× the 12 performance cores of host
  NEON (4.03 and 2.61 Gpairs/s per core, measured in Phase 3S T7, times 12 = 48 and 31,
  *model*).

The W1 pool of Phase 3S (64 target sets cycled through cache) fills only 64 cubes, a
small fraction of 40 GPU cores; the GPU row needs a level-sized launch, at least the
4,096 target leaves of the C3.2 cube (a T6 brief change, Section 15).

## 14. Multi-rank and overlap compatibility

**C5.1, the device on several ranks** (exchanges staged through host buffers) needs:
- the hook of Section 4.5 (a T4b-style `nd-fmm-plan` task with its `IndexFmm` check on
  1, 2 and 4 ranks), so the device operator downloads exactly the sent values and
  uploads exactly the received ones;
- device errors agreed on every rank before the next collective (Section 12);
- a device per rank (`Device::open` with an index from the local rank).

Nothing in Phase 4 precludes it: the device stores have the plan's layouts, ghost slots
included; the exchanges stay index-based; `DeviceNeedsOneRank` is one check to remove;
and the stage boundaries where the hook fires are the ones `Fmm` already sees.

**C5.2, overlap and device-resident ghost buffers** needs exchanges split into start and
finish (redesign §10) and device buffers the operator owns instead of the evaluator's
host stores; Section 4.3 names the latter. Neither conflicts with this design.

**Overlapping P2P with the far field on a second stream** (README, out of scope). On
Metal it would gain little as built: all of a device's wgpu streams submit to one ordered
queue (F10; stated in `cubecl-wgpu-0.11.0-pre.4/src/compute/timings.rs`), so there is no
real concurrency. More important, P2P adds into the target output after L2P and M2P
(redesign §7.5); run concurrently it needs its own buffer, added at the end, which changes
every target's rounding, deterministically. **Worth proposing later, not now**: as an
opt-in for C5.2 or Phase 6 with a documented order, measured first on a backend with
concurrent streams (CUDA, or `metal-native`'s multistream,
`cubecl-metal-0.11.0-pre.4/src/tests_multistream.rs`, *inferred*).

## 15. Task check

| Task | Modules | Tests | Brief change proposed |
| --- | --- | --- | --- |
| T4 | `fmm-kernels/src/{lib, device, buffer, error, movement}.rs`; `fmm-kernels/CLAUDE.md` | capability per backend; round trips (sizes 0, 1, 7, 10³, 10⁶; −0, subnormal, extremes, ±∞, NaN) for f32, f64, u32; zero, gather, scatter-add and scatter against host loops; proptest | device index arrays are u32 only (u16/u8 widened at upload), so no u16/u8 buffers; add `scatter_values` (assign by index) for the charges; no CSR gather/scatter is needed; `Device::open` uses CubeCL 0.11's fallible `cubecl::Device` constructors and refuses Metal without the MSL compiler; T4 checks with `cargo tree -e features` that CubeCL's `persistence` is off |
| T4b | not needed | — | — |
| T5 | `fmm-exec/src/device.rs` (`DeviceOperator`, mirrors, `begin_evaluation`, `read_output`), `fmm.rs` (`Backend`, `OperatorKind`, settings, `ExecOperator`), `tables.rs` (crate-private family accessors) | full-fallback bit identity on every `mpi_exec.rs` scenario; transfers against Sections 4.1 and 7.2; settings errors; the threads rule; `view` types and their upload | the device backend runs on one rank (`DeviceNeedsOneRank` after step 5); `Fmm` drives `begin_evaluation` and `read_output`; `synchronous_stages`; the threads rule of Section 11 |
| T6 | `fmm-kernels/src/p2p.rs`; the P2P kind in `device.rs`; a row in `nd_fmm_validate::p2p_kernels` | as the brief, plus device frames against `relative_frame` bit for bit | the GPU W1 row uses at least 4,096 target leaves per launch, not the 64-set pool; the C4.2 target of Section 13.4; the plane-per-leaf layout as a candidate |
| T7 | `fmm-kernels/src/leaf/{harmonics, p2m, l2p, p2l, m2p}.rs` | as the brief; bit identity with `nd-fmm-math` and `nd_fmm_ref::leaf` on the CPU runtime only where T3 finds a formulation free of cubecl-opt's fma fusion (Section 5.3) | P2M and P2L use the coefficient-owner mapping in point order (Section 6.3), not a shared-memory or plane reduction; coefficients formed in the kernel |
| T8 | `fmm-kernels/src/{gemm/tiled, translate, view}.rs` | as the brief; the grouped call against per-octant launches bit for bit | the GEMM writes a temporary (β = 0); the grouped translation of Section 6.4 (gather in batch order, grouped GEMM with a tile schedule, row-ordered reduction for M2M, scatter-add for L2L) |
| T9 | `fmm-kernels/src/gemm/library.rs`; dense M2L through `translate::grouped` with offset chunks | as the brief; grouped against per-offset (A) bit for bit with the same GEMM; the TF32 guard from `MatmulElems` | the launch structure is (B) of Section 6.4; `Classes` runs as dense from `expand` |
| T10 | `fmm-kernels/src/rotation.rs` | as the brief; bit identity with `RotationTables::m2l` on the CPU runtime where it holds | one cube per target box (Section 6.6) |
| T11 | scheduling in `device.rs`; device timestamps; `--backend` in `fmm_accuracy` | as the brief; one sync per evaluation | top levels stay on the device; the cross-level M2L merge is optional (Section 6.7) |
| T12 | `fmm-exec/src/tune.rs`; `FmmBuilder::tuning_cache` | as the brief | tuning runs only with a tuning-cache directory; the static f64 rule is dense through p = 11 |
| T13 | `nd-fmm-validate` `device_fmm` | as the brief | none |

Facts in the briefs corrected at sign-off (README, T1, T2, T3 and T6):
- The README ("The CubeCL CPU runtime", decision 10), T1, T2, T3 and T6 called
  CubeCL's vector type `Line<T>`; in 0.10 and 0.11 it is `Vector<T, N>`
  (`cubecl-core-0.11.0-pre.4/src/frontend/container/vector/base.rs`; the spike already
  uses `Vector`).
- The README, T1 §11, T2 and T3 said the CPU runtime "starts one OS thread per unit of
  a cube". In 0.11 a launch becomes one *task* per unit; tasks without `sync_cube` or
  shared memory share a pool of one worker per logical CPU, and only barrier or
  shared-memory kernels get a dedicated worker per unit (F18). T2 confirmed it (spike
  report, "CubeCL 0.11.0-pre.4").

## 16. Questions for sign-off

Signed off on 2026-10-03: every recommendation below is accepted as stated.

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | The requirements and tolerances | Keep requirements 1–11. Clarify that the device backend runs on one rank until C5.1 (`DeviceNeedsOneRank`, after `PointsNotOwned`). Dense f64 operators: within 1e-14, or twice the host operator's error on the same cell, whichever is larger. Every other tolerance as the README states (Sections 2, 9.2) |
| 2 | The residency option, and T4b | (a): device stores in the operator, `Fmm` drives `begin_evaluation` and `read_output`; on one rank the only evaluator-side write is `reset`'s zeroing. **No T4b in Phase 4**; the hook of Section 4.5 is the C5.1 design (Section 4) |
| 3 | Runtime-generic code or a run-time enum | A run-time value: `Backend` in `nd-fmm-exec`, `Device` over `cubecl::Device` in `nd-fmm-kernels`; CubeCL 0.11 removed `R` from the client (Section 3.2) |
| 4 | The dense M2L launch structure | (B): grouped GEMM per level and offset chunk, then a row-ordered reduction per target; bit-identical to per-offset launches at about 1% of their launches (Section 6.4) |
| 5 | `Classes` on the device | Run as `Dense` from `M2lClasses::expand`, reported as such (Section 6.8) |
| 6 | The f64 static rule for p = 9–11 | `Dense` through p = 11, `Rotation` from p = 12, provisional until a GPU measures rotation (Section 10.5) |
| 7 | The CPU runtime and `threads(n)` | No rayon pool with the CPU backend; `threads(n)` caps the units per cube of the CPU layouts. With Metal and CUDA the pool serves host-fallback kinds only, and is idle during device work by construction (Section 11) |
| 8 | Autotune and its persistence | A strategy-level tuner in `nd-fmm-exec` at build; a cache file only in a caller-supplied directory, with the `TableCache` rules; CubeCL's autotune unused and its persistence feature off (Section 10) |
| 9 | The C4.2 performance target | At least 25% of the P2P peak model (720 / 480 Gpairs/s) on W2 at N = 10⁵, and at least 10% on W1 at n_t = 64 with at least 4,096 target leaves per launch, Metal f32 (Section 13.4) |
| 10 | (new) The Metal runtime | Keep `metal` = wgpu-msl (README). If T3 finds that its default math mode breaks the §3.13 argument and no formulation restores it, switch the `metal` feature to `metal-native` (`cubecl-metal`, safe math mode) by a separate sign-off (Section 5.3) |

## 17. Outcome: decisions as taken and measured numbers (Phase 4, T2–T13)

Every number here is measured on the Apple M3 Max (Metal f32 for timings; the CubeCL
CPU runtime for f64 correctness) unless marked *model*; no f64 GPU run and no CUDA run
was made, and CUDA is type-checked only. The full tables are in
`fmm-validate/results/phase4-m3max.md` (T13) and the task reports (the T2–T12 pull requests);
laplace-fmm-plan.md §7, Phase 4, carries the status per component.

### 17.1 Decisions as taken

| Topic (section) | As designed | As taken |
| --- | --- | --- |
| CubeCL (1.2) | 0.11.0-pre.4 | pinned (T2), with `cubek-matmul`/`cubek-std` 0.3.0-pre.4; the spike ported and re-measured (Metal unchanged within noise; CPU runtime at O3) |
| fma (F16, 5.3) | measure whether a formulation keeps cubecl-opt's fusion out | none found (T3): device kernels write `fma` explicitly where a result is pinned, and match the host within tolerances, not bit for bit, except copies, scatters, frames and GEMMs/rotations written as host `mul_add` replicas |
| §3.13 addition (9.3) | drafted by T3 | signed off (decision 3): the C3S.4 P2P contract unchanged, on flushing backends for q = 0 or \|q\| ≥ 2⁻¹⁰⁰; ŷ by an explicit fma, `inverse_sqrt` without a Newton step, masking by compare and select |
| Metal runtime (5.3) | keep wgpu-msl | kept (decision 11); no `metal-native` |
| persistence (3.4) | off | off with `cpu` and `cuda`; on with `metal` through CubeCL's manifests, an empty store, unused (T4 sign-off) |
| CI (13.2) | T4 measures the CPU-runtime job | kept as built (`run-tests-kernels`: 4 min 53 s cold, 51 s warm); `nd-fmm-kernels` a member, not a default member (decision 7) |
| CPU runtime target (decision 10) | if T3's ratio ≤ 1.5 | set (T3: 1.004); T6's CPU layout of P2P at 1.16× `nd_fmm_simd::P2pKernel` per pair at one thread (target 1.5), 2.2× on all cores (reported) |
| frames (6.1) | a table of exact powers of two | the powers formed exactly in the kernel from integer keys (T6), bit for bit `relative_frame` |
| P2P layouts (6.2) | cube; plane per leaf a candidate; CPU layout if decision 10 | all three built and tested on every backend; cube 64 by default on GPUs; the tuner picks plane (2 per cube) or cube 32 on small leaves |
| P2M, P2L (6.3) | coefficient owners in point order | as designed; tiles of up to 32 points' harmonics in shared memory (18 at p = 20 in f64) |
| M2M, L2L, dense M2L (6.4) | structure (B) | as designed (T8, T9); (B) equals (A) bit for bit with the same GEMM, and on Metal the library equals the hand-written kernel bit for bit |
| library for M2L (6.5) | f32, p ≥ 8 | **changed after T9 (2026-10-04):** M2L runs the hand-written kernel at every p by default; the library pads each offset run to its widest batch (44–66% useful columns) and was slower on every FMM level measured; `DeviceGemm::Library` keeps it selectable. M2M and L2L keep the library rule (equal batches, 1.3–4.2× faster where it applies) |
| C4.5 gate (13.4) | 80% of the spike's GEMM | met in 2 of 12 cells; the cause measured (operand orientation, 16–29% for the library); accepted as analysed (decision 12). The coefficient-major layout that would give the spike's orientation was measured in T12 and is not a candidate (decision 13) |
| rotation (6.6) | one cube per target box | as designed (T10), plus slots u, u + U, … so that fewer units than (p + 1)² also run (needed on the CPU runtime); bit for bit a host `mul_add` replica |
| top levels, merge (6.7) | on the device; merge M2L across levels if significant | on the device; not merged: an evaluation enqueues in 0.26–0.68 ms for 40–107 launches (T11) |
| stage timing (8.3) | device timestamps if they add no sync | built (T11) and opt-in (`device_timestamps`): no sync, same bits, but the windows of neighbouring stages overlap on Metal; `synchronous_stages` is the per-stage measure, used by T13 |
| autotune (10) | strategy-level tuner | as designed (T12, decision 13): the strategy tuned before the tables are built; the device's `Auto` by the static rule; the budget bounds candidate starts; the coefficient-major layout not a candidate |
| leaf size (decision 8) | the Phase 3S rule on the device | applied in T13: it picks 64 (128 is 1.9% faster, below 5%), so no device default is added |

### 17.2 Measured numbers against the models

| Item (section) | Model or estimate | Measured |
| --- | --- | --- |
| launches per evaluation, every kind on the device (8.1) | about 49 (cube, p = 8), about 150 (Plummer) | cube N = 10⁵: 40 / 43 / 46 at p = 3 / 6 / 8 (34 under `Rotation`); Plummer: 101 / 101 / 107 (87); cube N = 10⁶: 64 / 97 / 133 (41) |
| launch overhead (8.1) | about 10 µs a launch, 0.5–1.5 ms per evaluation | enqueueing a whole evaluation 0.26–0.68 ms (T11) |
| transfers per evaluation (4.1) | the charges up, the output down, one sync | exactly that on every run: 0.4 MB up and 1.6 MB down at N = 10⁵ in f32 with gradients (4 / 16 MB at N = 10⁶) |
| P2P, Metal f32 (13.4) | peak model 720 / 480 Gpairs/s; targets 25% (W2) and 10% (W1) | W1 n_t = 64: 352 / 286 Gpairs/s (49% / 60% of the model); W2 N = 10⁵: 326 / 269 (45% / 56%) (T6) |
| M2L stage (6.4) | 8.4 GFLOP at 3.5 TFLOP/s ≈ 2.4 ms (cube, p = 8) | 8.5 ms in 24 launches (T9; GEMM 72%, gather 13%, reduction 12%); the downward stage of the device FMM 6.1 ms at N = 10⁵ (T13) |
| rotation against dense (6.6, 10.5) | rotation needs 10–14% of f32 peak to compete (spike) | 1.8–3.0% of f32 peak; dense faster per pair at every p from 2 to 16 on the C3.2 cube (1.4–3.3×, T10); at N = 10⁵ the tuner chose dense at every f32 p. **At N = 10⁶ (cube) it chose rotation at p = 6 and 8**: the evaluation 74 ms against 82 ms dense at p = 8, a tie at p = 6 (T13), because the dense downward stage per V pair grows from 9.5 ns at N = 10⁵ to 11.6 ns at N = 10⁶ while rotation's falls from 11.8 to 10.3 ns (synchronous stages; the cause is not analysed) |
| device memory (4.6) | dense M2L 13 MB (f32, p = 8) plus scratch 128 MB | 95–172 MB per `Fmm` at N = 10⁵ (dense), 20–30 MB (rotation); 322–358 MB at N = 10⁶ |
| evaluation, Metal f32 against the host (README target: none) | — | tuned device 2.1–10.9 ms at N = 10⁵ (15–73 ms at N = 10⁶): 35–125× the host at one thread, 4.0–12.9× at 12 threads (T13) |
| tuning time (10.4) | about 6 candidates × 0.3 s plus a few per bucket, within 10 s | 2.9–6.7 s per build on Metal f32 (T12, T13); a rebuild from the cache 0.08–0.12 s |
| CPU-runtime suite (13.3) | under 5 minutes | 16 s warm for `nd-fmm-kernels` (T12) |

