# nd-fmm-kernels

Purpose: every CubeCL (`#[cube]`) kernel of the FMM behind safe wrappers: backend
selection and the f64 capability check, device buffers, the data movement primitives,
the plan's views on the device (`view`, from T5), and (from T6) the operator kernels:
P2P (`p2p`, T6), the leaf operators P2M, L2P, P2L and M2P (`leaf`, T7), the grouped
translations M2M and L2L (`translate`, T8), dense M2L (`translate`, T9) and rotation M2L
(`rotation`, T10) (docs/design/device-path.md §3.1), and (Phase 4S T9) the output pass of
`nd-fmm-exec` on the device (`movement::gather_output`).
Phase and components: Phase 4, C4.1 (T4, T5), C4.2–C4.6 (T6–T10) and the timing windows
of C4.8 (T11) in docs/phase4/; Phase 4S, C4S.8 (T9) and C4S.10 (T11: transfers without a
host copy) in docs/phase4s/.

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  Device kernels follow §3.13, "Device kernels".
- Views (`view`, T5): uploaded from plain `u32` arrays in the layout of the plan's
  `Csr` and `GroupedCsr` (no `nd-fmm-plan` dependency), validated on the host, one
  `IndexBuffer` per array (its bound covers the whole buffer); a malformed view panics.
- Transfers without a host copy (Phase 4S T11, C4S.10, decision 14;
  spikes/download-path/REPORT.md): `Device::download_view(slice)` returns a
  `HostValues<E>` that owns CubeCL's host buffer of the download (pinned memory from
  CUDA's pool, a mapped staging buffer on wgpu) and derefs to `&[E]` through
  `CubeElement::from_bytes` (no `unsafe`); it counts one download and one sync and has
  `download`'s checks and errors (a launch error surfaces at it, device-path.md §12).
  `download` is the copy of a view. `Device::write_owned(slice, Vec<E>)` hands the `Vec`
  to `Bytes::from_elems` without the copy `write` makes (`write` is `write_owned` of a
  `to_vec`), with `write`'s counters and checks. Drop a view before the next download:
  held across one, it makes CubeCL's host pool keep a second buffer. Tests: the view
  equals `download` and `write_owned` equals `write` bit for bit and in their counters
  (`tests/kernels/round_trip.rs`: f32, f64 and u32, 0, 1, 7 and 10⁵ values at an odd
  offset and whole), the refusals of a foreign buffer (`capability.rs`), and a launch
  error at both (the unit tests of `device`: a plane sum on the CPU runtime, which does
  not lower it, and a cube larger than the device allows on Metal and CUDA).
- The output gather (Phase 4S T9, `movement::gather_output`, device-path.md §18.4): φ and
  ∇φ in the caller's order from the leaf-ordered target output, each value converted to
  f64, divided by its leaf's f64 scale and rounded once to T (`F::cast_from(f64::cast_from(x)
  / s)`), so bit for bit `nd-fmm-exec`'s host pass `T::from_f64(x.to_f64() / s)` where the
  device divides f64 correctly rounded (the CPU runtime, CUDA, F33). Its input is a
  `movement::OutputOrder` (the leaves' offsets, every target's point and leaf, two f64
  scales per leaf), validated at upload (every point within its leaf) so that the launch
  reads without bounds checks; its upload and the gather need f64 arithmetic and refuse
  without it (`KernelError::UnsupportedPrecision`, Metal). One elementwise launch, the
  units capped on the CPU runtime. Never fuse the scaling, multiply by a reciprocal or
  divide in f32: that changes output bits. Tested against the host loop bit for bit
  (`tests/kernels/movement.rs`: f32 and f64, 0, 1, 7 and 10⁵ targets, an empty leaf, with
  and without gradients; the bound checks; the refusal on Metal).
- CPU units: `Device::limit_units(n)` caps the units per cube of the CPU runtime's
  elementwise launches (default `CPU_MAX_UNITS`); `nd-fmm-exec` passes `threads(n)`
  (device-path.md §11). Every later CPU layout honours the cap (the CPU layout of P2P
  does: at most the cap, one unit per contiguous range of target leaves).
- P2P (T6, `p2p`): one launch per level call, three layouts of one formulation
  (`P2pLayout`): `Cube` (one cube per target leaf, a shared-memory tile of U sources,
  default on Metal with U = 64 and on CUDA with U = 32, Phase 4S T7), `Plane` (one plane per target leaf, several per
  cube, `sync_plane`, plane and lane from `UNIT_POS` (CUDA, below); a candidate for
  T12) and `Cpu` (targets in `Vector<T, N>` lanes of the host's width, K N = 8 per
  block, default on the CPU runtime). Every layout adds
  each target's sources in near-row order and point order from the value in the output,
  so every test runs on every layout, and the GPU layouts also run on the CPU runtime
  (correctness only: planes of one unit, at most one unit per core). Frames come from
  `LeafCoordinates` in integer arithmetic (`near_frames` writes them for tests); the
  leaf stores' offsets are `PointOffsets`, validated at upload. Do not change the
  formulation, the tile structure or the order without a sign-off; report a faster
  variant instead.
- Leaf operators (T7, `leaf`): P2M, P2L, L2P and M2P, one launch per level call, p
  comptime up to `MAX_DEGREE` (20). The harmonics follow `nd_fmm_math::harmonics`
  operation for operation (coefficients formed in the kernel, integers comptime, one
  `inv_r2` per point, M2P's gradient from the recursion run to p + 1), the loops over n
  and m unrolled (`#[unroll]`) into a local `Array`; the contractions and gradient
  ladders are `nd_fmm_ref::leaf`'s and `harmonics`'s, in their order. Frames come from
  `BoxCoordinates` and `LeafCoordinates` through `frame.rs` (shared with P2P), applied as
  (u − ĉ) · 2^k, exact. Two layouts (`LeafLayout`): `Cube { units, tile }` (one cube per
  box or target leaf; P2M and P2L by coefficient owners with `tile` points' harmonics in
  shared memory, L2P and M2P one unit per target with each entry's coefficients staged in
  shared memory; default on Metal with 64 units and on CUDA with 32 (Phase 4S T7), tiles
  of up to 32 points)
  and `Cpu` (one unit per core, contiguous rows, no shared memory; default on the CPU
  runtime, units capped by `Device::limit_units`). Both add every output's
  contributions in the plan's order; tests run every layout on every backend. Bit
  identity with the host does not apply (contraction, T3 rule 6): tests use the operator
  bounds.
- Grouped translations (T8, T9, `translate`): structure (B) of device-path.md §6.4, for
  M2M and L2L (8 octant groups) and dense M2L (the 316 offsets in index order, `u16`
  groups, multipoles and locals in separate buffers). Tables are uploaded once as
  `Tables` (matrix g at g n²; with `library` also the library copy at 256-byte aligned
  strides, `library_stride`). Per level call and chunk (contiguous batch entries within
  the scratch budget, `default_scratch_bytes`), three launches: `movement::gather_columns` in
  batch order, one grouped GEMM over a tile schedule built at build (`TileSchedule`:
  group, first column, columns per tile), and `Accumulate::Rows` (a reduction per target
  in row order, reading the view's row-to-batch map; M2M, M2L) or `Accumulate::Scatter`
  (`scatter_add_columns`; L2L); the budget by device, `default_scratch_bytes` (128 MB, on
  CUDA 2 GB, at most an eighth of the device memory; Phase 4S T7: the reduction walks
  every entry of each row once per chunk, so its cost grows with the chunks). Structure (A) (`PerGroupPlan`, `per_group`: one gather,
  GEMM and scatter-add per group with a column) is the test reference, bit for bit (B)
  with the same GEMM; `grouped_stage` runs one stage of every chunk, for profiling.
  The hand-written GEMM (`GemmLayout`): `Cube { rows, columns, per_unit }` (one cube per
  tile; default on Metal: up to 32 units along the rows, 64 in all, 4 columns per unit;
  on CUDA (Phase 4S T7): the power of two at or above n / 4 units along the rows, 16 to
  128, 64 units in all (more where the rows need them), 8 columns per unit) and `Cpu { block, per_unit }` (one cube, units capped by
  `Device::units_cap`, contiguous tiles; default on the CPU runtime); each output one
  accumulator from zero, `fma` with k ascending, stored once (β = 0): bit for bit a host
  `mul_add` loop in that order, on Metal, CUDA and the CPU runtime (T3 rule 6;
  tested). The library GEMM (`GemmPolicy::Auto`, f32, p ≥ 8, GPU only): `cubek-matmul`'s
  `Strategy::MultiLevel(SimpleCyclicCmma)` named explicitly, one batched launch per
  chunk: runs of contiguous groups padded to the run's widest batch ([G_c, k, n]; a
  group wider than the budget in pieces), reading the library copy of the tables from
  the chunk's first group (wgpu binds only aligned offsets), only if a probe launch of
  every chunk shape succeeds at build and the resolved `MatmulElems` keep T for every
  stage and register type (the input-precision guard; tested with lowered types); else
  the hand-written kernel for the whole view, decided at build by the shape alone
  (`GroupedPlan::gemm`, `library_rejection`, `gemm_columns` for the padding). Chunks and
  layouts never change the bits of the hand-written path (tested); on Metal the library
  measured bit for bit the hand-written kernel, across chunkings and against (A) (T9;
  not assumed elsewhere). Do not change the summation order without a sign-off.
  From T12 `PlanSettings::orientation` (`Orientation`): box-major (the default, §6.4) or
  coefficient-major (blocks of w columns, coefficient k of column c of block b at
  (b n + k) w + c; the gather `movement::gather_coefficients`, the same hand-written
  units, accumulators and k order, so its products are box-major's bit for bit, tested on
  every layout and chunking; the library reads the tables as column-major A; L2L then
  takes the reduction). Measured slower than the best box-major choice on every FMM level
  of Metal f32 (`nd-fmm-exec`'s `tune` docs), so `nd-fmm-exec` does not register it; keep
  it tested. `Tables::release_library_copy` frees a library copy no plan needs (T12), and
  `VERSION` keys the tuning cache.
- Rotation M2L (T10, `rotation`, device-path.md §6.6): the M2L family of the Phase 2
  rotation tables uploaded once as `RotationTables` from plain arrays in the storage of
  `nd_fmm_tables::rotation::ShiftTables` (`RotationArrays`: forward and backward y-blocks
  per polar angle, azimuth factors, coaxial factors, and per offset four `u32`: alignment,
  polar, azimuth, distance; no `nd-fmm-tables` dependency, no conversion of the storage).
  `RotationPlan` uploads the rows of a V view with a pair; `m2l` is one launch per level.
  Two layouts (`RotationLayout`): `Cube { units }` (one cube per row with a pair, unit u
  the owner of slots u, u + U, …, two working vectors of (p + 1)² values in shared memory,
  `sync_cube` after each step; default on Metal and CUDA: (p + 1)² units rounded up to the
  plane size, one slot each, the fastest of the layouts measured on CUDA, Phase 4S T7) and `Cpu` (one unit per core, at most `Device::units_cap`,
  contiguous rows, local arrays; default on the CPU runtime). Each pair repeats
  `ShiftTables::apply` step for step (z-rotation, forward y-blocks, coaxial step, backward
  y-blocks, z-rotation back added into the owner's accumulator; `Up`/`Down` the coaxial
  step alone, added term by term into the accumulator, with the parity for `Down`), each
  output one accumulator from zero with an explicit `fma` per term in the host's order:
  bit for bit a host `mul_add` replica on every layout and backend (tested on the CPU
  runtime, Metal and CUDA), not `RotationTables::m2l`, which rounds each product
  (tolerance, T3 rule 6; 35–100% of the values agree bit for bit). Do not change the step order or
  the device table layout without a sign-off.
- Timing windows (T11, device-path.md §8.3): `Device::open_window` and `close_window`
  wrap CubeCL's `profile_start`/`profile_end`; `WindowTime::resolve` reads the time after
  the work has run. `Device::times_on_device` is true only on a GPU backend whose runtime
  times on the device (Metal with timestamp queries, CUDA): there a window submits the
  queued work without waiting. The CPU runtime reports device timing but drains its
  stream at both ends of a window, so `close_window` counts two syncs for every window
  not timed on the device (`Counters::windows`, `Counters::syncs`). A window without a
  launch measures nothing (`resolve` gives `None`) on every backend: `Device` counts the
  launches inside it and drops the time CUDA's events and the CPU runtime's clock give
  an empty window (Phase 4S T2). Never call CubeCL's `profile` closure form, autotune or
  throughput measurement for this.
- Generic over the float type (`DeviceFloat`: f32, f64) and comptime parameters (p, n,
  layouts); the backend is a run-time value (`BackendKind`, `Device`), never an
  `R: Runtime` type parameter (device-path.md §3.2). Every kernel runs on every runtime;
  layouts are chosen per backend from `DeviceInfo`. No allocation per launch: launch
  wrappers allocate nothing, and buffers are allocated at build.
- Unsafe (root CLAUDE.md, device-path.md requirement 9): only CubeCL launches
  (`launch_unchecked`) and buffer views (`BufferArg::from_raw_parts`). Every block has a
  `// SAFETY:` comment naming the bounds the wrapper checked; the crate denies
  `unsafe_op_in_unsafe_fn` and `clippy::undocumented_unsafe_blocks`. Every public
  function is safe: a wrapper checks lengths, owners and index bounds on the host
  (`IndexBuffer::bound`) before launching unchecked, and panics on a violated
  precondition.
- Backend coverage: every runtime test prints the device it ran on and a closing line
  "backends run: …; not run: …" (`tests/kernels/common.rs`, `run` and the `tests_on!`
  macro). A task report lists which backends ran, from the test output, and never
  reports a backend that did not run as passing. A backend compiled in but not run by a
  test is "in its own test" (CPU) or "in its own ignored test" (Metal, CUDA).
- Accumulation and determinism (README requirements 4–6, device-path.md §6.1):
  - each output value has one owning unit per launch; the owner loads it, adds its
    contributions one by one in the target's row order, and stores it, so splitting a
    row into several launches gives the same bits;
  - no atomics anywhere; reductions use shared memory or plane operations in a fixed
    order; scatters require distinct indices per launch (debug builds check);
  - no kernel or matmul strategy chosen per call (never `Strategy::Auto`), and every
    launch from the thread that owns the `Device`, so one stream runs them in order.
- Arithmetic (signed off with T3, CONVENTIONS §3.13 "Device kernels";
  spikes/device-arith/REPORT.md, "Recommendation"):
  - assume only what §3.13 lists: `+ − ×` correctly rounded; any `a · b ± c` may be
    fused (a lone one by cubecl-opt's `InstCombinePass` on every backend, any one on
    CUDA, where LLVM's NVPTX back end also fuses products with other uses; Phase 4S T3),
    so write `fma` where the result must be pinned; `sqrt`, division, `inverse_sqrt` within 2.5 u_T on normal
    arguments; subnormals may flush (Metal does in arithmetic, not in copies);
  - no compensated summation, no `x − x` or `x + 0.0` tricks, nothing that depends on
    the sign of a zero;
  - P2P: ŷ = `fma(r̂, u_s, ĉ)`, `inverse_sqrt` without a Newton step, masking by
    compare and select;
  - bit identity with host loops only where §3.13 rule 6 allows it (copies, scatters,
    zeroing, frames, a GEMM in the host order with `mul_add`); tolerances elsewhere.
- Kernel tests: `nd-fmm-kernels` builds and tests without MPI. Each test process
  shares one `Device` per backend (`tests/kernels/common.rs`), because CubeCL compiles
  kernels once per process. Keep comptime variants few per test; reuse the column
  sizes (p + 1)² for p ∈ {0, 3, 8, 20}.
- Test budget (T4, measured on the M3 Max): `cargo test -p nd-fmm-kernels --features
  cpu --release` runs under **2 minutes** warm (test run only, build excluded), no
  default test over **30 s**; larger shapes and sweeps (p = 20 at full size, B > 10³)
  are `#[ignore]`. T4's suite: 30 tests in 0.4 s (release), 32 in 1.1 s (debug). T6's
  P2P tests (`tests/kernels/p2p.rs`, every layout in f32 and f64) add 8 runtime tests
  of a few seconds together. T7's leaf tests (`tests/kernels/leaf.rs`, p ∈ {0, 3, 8},
  both layouts, f32 and f64) add 8 runtime tests of about 4 s together (104 kernel
  variants, 3.9 s of first launches, the slowest 0.24 s); the p = 20 sweep in f64
  (`degree_20_sweep_on_the_cpu_runtime`, 17 variants, up to 2.5 s each to compile, 20 s)
  is `#[ignore]`. T8's translation tests (`tests/kernels/translate.rs`: the GEMM at
  p ∈ {0, 1, 3, 8, 12, 20} and k ∈ {0, 1, 7, 64}, level calls, chunks and per-octant
  structure at p ∈ {0, 3, 8}, both layouts, f32 and f64) add 4 runtime tests of about
  4 s together; the GEMM of 1000 columns on the CPU runtime is `#[ignore]`. T10's
  rotation tests (`tests/kernels/translate/rotation.rs`: random V rows at p ∈ {0, 3, 8}
  and the special offsets at p ∈ {1, 3, 8}, both layouts, f32 and f64) add 3 runtime
  tests of about 1.2 s together. T12 runs the level-call tests in both orientations and
  the gather coefficient-major too (no new test functions).
  `cargo test -p nd-fmm-kernels` without features builds and passes in seconds.
  Kernel compilation, from CubeCL's profiling log (the first launch of each variant
  includes its compilation): `CUBECL_DEBUG_LOG=<file> cargo test -p nd-fmm-kernels
  --features cpu --release -q`, then `awk -f fmm-kernels/tools/compile_times.awk
  <file>` (T4: 34 variants, 0.27 s on the M3 Max). In CI the log comes from the job's
  one test run, which sets `CUBECL_DEBUG_LOG` (Phase 4S T10: the test output, the test
  count and the "backends run" lines are the same with the log on), and the awk summary
  is the next step.
- CubeCL features: the workspace entries set `default-features = false`, so `cpu` and
  `cuda` build without CubeCL's `persistence`. With `metal` it is on regardless:
  `cubecl-wgpu` 0.11.0-pre.4 takes `cubecl-cpp` with its defaults, which enable
  `cubecl-runtime/default`. A Metal run then creates an empty store
  `target/environment/default.db` (T4; accepted at its sign-off, device-path.md §3.4).
  Never call CubeCL's autotune or throughput
  measurement, and never enable `[compilation] cache`, so that nothing is recorded in
  it. Check with `cargo tree -p nd-fmm-kernels -e features --features <backend>` after
  any dependency change.
- Never read an environment variable to choose a backend or device. CubeCL's own
  variables (`CUBECL_CPU_STACK_MB`, `CUBECL_DEBUG_LOG`) and `cubecl.toml` are documented,
  not set.
- Sandbox (macOS, Claude Code):
  - the `cpu` feature's first build downloads the `tracel-llvm` bundle into
    `~/.cache/tracel/` and installs it into `~/Library/Application Support/tracel/`
    (Linux: `~/.local/share/tracel/`); both are allowed in the sandbox;
  - Metal needs GPU access and fails inside the sandbox ("No possible adapter
    available", or `NoDevice`). Build sandboxed with `cargo build --tests -p
    nd-fmm-kernels --release --features metal`, then run `cargo test -p nd-fmm-kernels
    --release --features metal -- --ignored --show-output` outside it, and say so.
- Before finishing:
  - `cargo clippy -p nd-fmm-kernels --all-targets -- -D warnings` and the same with
    `--features cpu,metal`;
  - `cargo check -p nd-fmm-kernels --features cuda`;
  - `cargo test -p nd-fmm-kernels` and `cargo test -p nd-fmm-kernels --features cpu
    --release -- --show-output`;
  - by hand on the M3 Max: `cargo test -p nd-fmm-kernels --release --features metal --
    --ignored --show-output`;
  - `cargo doc -p nd-fmm-kernels --no-deps`.

## CUDA (Phase 4S)
Measured on locust's H100 (GH200) on 2026-10-06 (Phase 4S T2; docs/design/device-path.md
§18.1 has the facts with their sources).
- Running the suite: on the M3 Max, outside the sandbox, `tools/gh200/sync.sh`, then
  `tools/gh200/remote.sh 'cargo test -p nd-fmm-kernels --release --features cpu,cuda --
  --ignored --show-output'` (the CUDA tests and the ignored CPU ones); without
  `--ignored`, the CPU runtime's tests on Grace. `cpu` and `cuda` combine in one build.
  Clippy: `--features cpu,cuda`. CUDA tests are `#[ignore = "CUDA: run by hand on
  locust"]` and are registered with `tests_on!(cuda: …)`: Metal's list plus every f64
  test and the property tests, 56 tests (58 from Phase 4S T9, with the output gather in
  f32 and f64; Metal's list gains its f64 refusal instead).
- `DeviceInfo`: `cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64`; plane
  size 32 (fixed), shared memory 232,448 B per cube (the opt-in maximum; Metal 32 KB),
  1024 units per cube, cube counts (2³¹ − 1, 65,535, 65,535), memory 102,005,473,280 B.
  `GPU_MAX_CUBES` stays 65,535 for every backend: measured in Phase 4S T7, lifting it on
  CUDA made `zero` and `gather_columns` 1.2–1.7× slower over 2²⁶–2²⁸ values (striding
  units win; device-path.md §18.2).
- Defaults on CUDA (Phase 4S T7, measured on the H100, device-path.md §18.2): P2P `Cube`
  of 32 units (`CUDA_CUBE_UNITS`), the leaf operators `Cube` of 32 units with tiles of 32
  (`CUDA_LEAF_UNITS`), the GEMM with the power of two at or above n / 4 rows (16 to 128),
  64 units and 8 columns per unit (`CUDA_GEMM_*`), the chunk budget 2 GB
  (`CUDA_SCRATCH_BYTES`, `default_scratch_bytes`), the rotation layout as on Metal. Each
  is an existing layout parameterised, tested on every backend by the kernel tests (the
  CPU runtime with fewer units: the GEMM with 8 columns per unit, the P2P and leaf cube
  layouts), and on CUDA with its own values. Metal and the CPU runtime keep theirs.
- Precisions: f32 and f64 registered with arithmetic. TF32 is registered for
  conversion; it does not come into play, because the library probe already fails:
  `cubek-matmul`'s `SimpleCyclicCmma` refuses every f32 shape the tests build ("No tile
  size is available for the problem"), so `GemmPolicy::Auto` takes the hand-written
  kernel (tested on CUDA: `library_level_call`, `m2l_library`).
- Compiler path: LLVM to NVPTX (`client.name()` is `"cuda"`; it does not name the
  path). `PLANE_POS` is not lowered for NVPTX: compiling a kernel that reads it panics on
  CubeCL's server thread, and the launch is dropped silently (a later download returns
  the buffer's old contents, no error). Fixed upstream after 0.11.0-pre.4
  (tracel-ai/cubecl#1714, merged 2026-09-25, not yet released). Until the pin includes
  it, do not use `PLANE_POS`; derive it from `UNIT_POS` in a 1-D cube. `UNIT_POS_PLANE` (the reproducer), `sync_plane` and
  `sync_cube` (the tests) work.
- Every bit-for-bit test holds on CUDA in f32 and f64 (copies, scatters, frames, the
  hand-written GEMM, (B) against (A), rotation against its `mul_add` replica, P2P and
  leaf layouts against each other), and every operator stays within the Phase 4 bounds.
  The P2P pair-terms test passes on CUDA without Metal's subnormal-flushing allowance
  (CUDA's subnormal arithmetic itself is T3's to measure).
- Test budget (release, H100, one process per test file): the 56 CUDA tests in 34 s
  wall, 537 kernel variants and 18 s of first launches (compilation included), the
  slowest 0.54 s (`HarmonicsKernel`, f64). On the first run the leaf tests' first
  launches took 11.1 s against 7.2 s on the second (the driver's PTX cache in
  `CUDA_CACHE_PATH`, presumably). The whole ignored run (CUDA plus the CPU runtime's
  p = 20 sweep and GEMM of 1000 columns) in 55 s; the CPU runtime's 91 default tests on
  Grace in 24 s.

## Allowed dependencies
cubecl, cubek-matmul, cubek-std (the pinned workspace entries), nd-fmm-math, thiserror;
dev-dependencies: nd-fmm-ref, nd-fmm-tables, proptest. No MPI, no nd-fmm-plan, no
rayon. Anything else needs asking first.

## Test oracle
Plain host loops (buffers, zeroing, gather, scatter), bit for bit; `nd-fmm-ref`
(`direct`, `leaf`, `p2p`), `nd-fmm-math` and `nd-fmm-tables` (`MatrixSet::apply`,
`RotationTables`; for rotation M2L also a host `mul_add` replica of the kernel, bit for
bit) for the operator kernels, at the canonical frames, levels 2, 9 and 16 of a dyadic
domain.
