# nd-fmm-kernels

Purpose: every CubeCL (`#[cube]`) kernel of the FMM behind safe wrappers: backend
selection and the f64 capability check, device buffers, the data movement primitives,
the plan's views on the device (`view`, from T5), and (from T6) the operator kernels:
P2P (`p2p`, T6), the leaf operators P2M, L2P, P2L and M2P (`leaf`, T7), the grouped
translations M2M and L2L (`translate`, T8) and dense M2L (`translate`, T9), then rotation
M2L (T10) (docs/design/device-path.md §3.1).
Phase and components: Phase 4, C4.1 (T4, T5) and C4.2–C4.6 (T6–T10) in docs/phase4/.

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  Device kernels follow §3.13, "Device kernels".
- Views (`view`, T5): uploaded from plain `u32` arrays in the layout of the plan's
  `Csr` and `GroupedCsr` (no `nd-fmm-plan` dependency), validated on the host, one
  `IndexBuffer` per array (its bound covers the whole buffer); a malformed view panics.
- CPU units: `Device::limit_units(n)` caps the units per cube of the CPU runtime's
  elementwise launches (default `CPU_MAX_UNITS`); `nd-fmm-exec` passes `threads(n)`
  (device-path.md §11). Every later CPU layout honours the cap (the CPU layout of P2P
  does: at most the cap, one unit per contiguous range of target leaves).
- P2P (T6, `p2p`): one launch per level call, three layouts of one formulation
  (`P2pLayout`): `Cube` (one cube per target leaf, a shared-memory tile of U sources,
  default on Metal and CUDA with U = 64), `Plane` (one plane per target leaf, several per
  cube, `sync_plane`; a candidate for T12) and `Cpu` (targets in `Vector<T, N>` lanes of
  the host's width, K N = 8 per block, default on the CPU runtime). Every layout adds
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
  shared memory; default on Metal and CUDA with 64 units and tiles of up to 32 points)
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
  the scratch budget, default 128 MB), three launches: `movement::gather_columns` in
  batch order, one grouped GEMM over a tile schedule built at build (`TileSchedule`:
  group, first column, columns per tile), and `Accumulate::Rows` (a reduction per target
  in row order, reading the view's row-to-batch map; M2M, M2L) or `Accumulate::Scatter`
  (`scatter_add_columns`; L2L). Structure (A) (`PerGroupPlan`, `per_group`: one gather,
  GEMM and scatter-add per group with a column) is the test reference, bit for bit (B)
  with the same GEMM; `grouped_stage` runs one stage of every chunk, for profiling.
  The hand-written GEMM (`GemmLayout`): `Cube { rows, columns, per_unit }` (one cube per
  tile; default on Metal and CUDA: up to 32 units along the rows, 64 in all, 4 columns
  per unit) and `Cpu { block, per_unit }` (one cube, units capped by
  `Device::units_cap`, contiguous tiles; default on the CPU runtime); each output one
  accumulator from zero, `fma` with k ascending, stored once (β = 0): bit for bit a host
  `mul_add` loop in that order, on Metal and the CPU runtime (T3 rule 6; tested). The
  library GEMM (`GemmPolicy::Auto`, f32, p ≥ 8, GPU only): `cubek-matmul`'s
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
  reports a backend that did not run as passing. CUDA is "type-checked, not run".
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
  - assume only what §3.13 lists: `+ − ×` correctly rounded; any lone `a · b ± c` may
    be fused (cubecl-opt's `InstCombinePass`, every backend), so write `fma` where the
    result must be pinned; `sqrt`, division, `inverse_sqrt` within 2.5 u_T on normal
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
  4 s together; the GEMM of 1000 columns on the CPU runtime is `#[ignore]`.
  `cargo test -p nd-fmm-kernels` without features builds and passes in seconds.
  Kernel compilation, from CubeCL's profiling log (the first launch of each variant
  includes its compilation): `CUBECL_DEBUG_LOG=<file> cargo test -p nd-fmm-kernels
  --features cpu --release -q`, then `awk -f fmm-kernels/tools/compile_times.awk
  <file>` (T4: 34 variants, 0.27 s on the M3 Max).
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

## Allowed dependencies
cubecl, cubek-matmul, cubek-std (the pinned workspace entries), nd-fmm-math, thiserror;
dev-dependencies: nd-fmm-ref, nd-fmm-tables, proptest. No MPI, no nd-fmm-plan, no
rayon. Anything else needs asking first.

## Test oracle
Plain host loops (buffers, zeroing, gather, scatter), bit for bit; `nd-fmm-ref`
(`direct`, `leaf`, `p2p`), `nd-fmm-math` and `nd-fmm-tables` (`MatrixSet::apply`,
`RotationTables`) for the operator kernels, at the canonical frames, levels 2, 9 and 16
of a dyadic domain.
