//! The device path (Phase 4, C4.1; feature `gpu`): [`DeviceOperator`], the Laplace
//! operator with its data on a device of `nd-fmm-kernels`, and what an
//! [`Fmm`](crate::fmm::Fmm) reports about it ([`DeviceReport`], [`DeviceCounters`]).
//! The design is docs/design/device-path.md, signed off on 2026-10-03; its sections are
//! cited as §x.
//!
//! # Backends
//!
//! [`Backend`] chooses where the operators run: [`Backend::Host`] (the default, the host
//! path of [`operator`](crate::operator)), or a device of `nd-fmm-kernels`, a run-time
//! value, never a type parameter (§3.2):
//!
//! | Backend | Feature | Precisions | Runs here |
//! | --- | --- | --- | --- |
//! | [`Cpu`](Backend::Cpu) | `cpu` | f32, f64 | the CubeCL CPU runtime: the correctness backend |
//! | [`Metal`](Backend::Metal) | `metal` | f32 | wgpu with the MSL compiler, outside the macOS sandbox |
//! | [`Cuda`](Backend::Cuda) | `cuda` | f32, f64 | LLVM to NVPTX; type-checked in CI, run by hand on an H100 (Phase 4S) |
//!
//! Each feature enables `gpu`. A backend that is not compiled in, a device that cannot be
//! opened and a precision the device does not do arithmetic in (f64 on Metal) are
//! refused by [`FmmBuilder::build`](crate::fmm::FmmBuilder::build) with a
//! [`SettingsError`], agreed on every rank by step 1's
//! existing all-reduce of input errors. [`Backend::probe`] opens a device and returns
//! why it is unavailable, which the `Copy` error cannot carry. Until C5.1 a device runs
//! on one rank only: on several, `build` returns
//! [`DeviceNeedsOneRank`](crate::fmm::SettingsError::DeviceNeedsOneRank) on every rank
//! after its step 5, so that `PointsNotOwned` still wins where it applies (§4.4).
//!
//! # Residency (§4.1, §4.3 option (a))
//!
//! Everything is allocated on the device at build, so no evaluation allocates there:
//!
//! | Data | On the device | Uploaded | Downloaded |
//! | --- | --- | --- | --- |
//! | multipoles, locals | one buffer each, the `LevelBuffers` layout | never (zeroed by a kernel) | never, but by host-fallback calls |
//! | source store | the `LeafStore` layout of CONVENTIONS §3.13 | coordinates once at build; the charges every evaluation, scattered into their slots | never |
//! | target input | the `LeafStore` layout | once at build | never |
//! | target output | the `LeafStore` layout | never (zeroed) | once per evaluation, and by host-fallback calls |
//! | plan views | one index buffer per array ([`DeviceViews`]) | once at build | never |
//! | geometry | box and leaf indices ([`BoxCoordinates`], [`LeafCoordinates`]) | once at build | never |
//! | tables | the dense octant tables of M2M and L2L; the dense M2L tables under `Dense`, expanded from the classes under `Classes` (§6.8); the M2L family of the rotation tables under `Rotation` (T10); for a family whose level calls may take the library GEMM, also its library copy (matrices at 256-byte aligned strides) | once at build | never |
//! | translation plans (T8, T9, T10) | per M2M, L2L and dense M2L level call on the device: the tile schedule of the hand-written GEMM, or the padded gather indices of the library GEMM; per rotation M2L level call, its rows with a pair | once at build | never |
//! | translation scratch (T8, T9) | the gathered inputs and products of the widest chunk, shared by every level call | never (allocated at build) | never |
//!
//! Before allocating, [`DeviceOperator::new`] sums the bytes of every buffer and refuses a
//! configuration that does not fit in the memory the device reports as available
//! ([`SettingsError::DeviceMemory`]), because CubeCL panics when an allocation fails
//! (§4.6).
//!
//! The `Evaluator` keeps its host stores. On one rank the only write it makes to them
//! outside operator calls is `reset`'s zeroing (§4.2), which
//! [`begin_evaluation`](DeviceOperator::begin_evaluation) mirrors: it zeroes the device
//! stores, uploads the charges and scatters them into the source store. The exchanges
//! move nothing on one rank, and [`DeviceOperator::new`] checks that the plan has no
//! ghost leaf and no ghost box, the condition under which none writes. The points are
//! uploaded once per build by [`load_points`](DeviceOperator::load_points), from copies
//! of the evaluator's source store and target input that `build` takes after its step 8.
//! [`read_output`](DeviceOperator::read_output) downloads the target output once.
//!
//! **Per evaluation**, with every kind on the device (T11): one upload of N_s s bytes
//! (the charges), one download of o N_t s bytes (the target output) and one sync (the
//! download), for N_s sources, N_t targets, s bytes per value and o = 1, or 4 with
//! gradients. Host-fallback calls add their own transfers (next section).
//!
//! # Host fallback (§7)
//!
//! Every operator kind can run on the host ([`OperatorKind`], [`Placement`]); a kind
//! runs there if [`FmmBuilder::host_fallback`](crate::fmm::FmmBuilder::host_fallback)
//! names it. **From T10, every kind has a device kernel under every strategy** (next
//! sections), so a kind runs on the host only by request. A host-fallback level call
//!
//! 1. downloads its device inputs and its output region into host mirrors (exact
//!    copies);
//! 2. builds the same batch with the call's own level, index and views, the mirrors for
//!    multipoles, locals and target output, and the evaluator's own slices for sources
//!    and target input (which `Fmm` keeps current on the host);
//! 3. runs the wrapped [`LaplaceOperator`]'s own method, so the bodies are those of the
//!    host path, serially or on the `Fmm`'s pool;
//! 4. uploads its output region.
//!
//! | Kind (level l) | Downloaded before | Uploaded after |
//! | --- | --- | --- |
//! | P2M | multipoles of l | multipoles of l |
//! | M2M (either pass) | multipoles of l and l + 1 (one range) | multipoles of l |
//! | M2L | multipoles of l, locals of l | locals of l |
//! | P2L | locals of l | locals of l |
//! | L2L | locals of l − 1 and l (one range) | locals of l |
//! | L2P | locals of l, the level's target output | the level's target output |
//! | M2P | multipoles of l + 1, the level's target output | the level's target output |
//! | P2P | the level's target output | the level's target output |
//!
//! A call whose view has no entry, or a leaf call whose leaves have no target, adds
//! nothing and moves nothing. Every download is one sync. With every kind on the host
//! the device path's output equals the host path's bit for bit (§7.3): the device stores
//! start from the same +0.0, every transfer is a copy, and the same operator, built as
//! `build` builds the host operator, runs on the same data in the same order.
//!
//! # P2P on the device (T6)
//!
//! P2P runs on the device by default: one launch of `nd_fmm_kernels::p2p::p2p` per level
//! whose near view has an entry and whose leaves have a target, from the level's near view,
//! the leaf indices, the point offsets and the device stores, adding into the device target
//! output. It moves no data, so the P2P rows of the table above vanish from an evaluation's
//! transfers; L2P and M2P before it on the same level (on the device, or on the host
//! fallback, which uploads its output region first) add to the target output before P2P
//! does (the order of `evaluate_leaves`, all on one stream). The layout is fixed at
//! build ([`DeviceReport::p2p_layout`]): by default the CPU layout on the CPU runtime, whose
//! units per cube `threads(n)` caps, and the cube layout of 64 units on Metal and CUDA
//! (device-path.md §6.2). The kernel follows CONVENTIONS §3.13, "Device kernels", and adds
//! each target's sources in the order of the host path, so its output agrees with the host
//! P2P within the device P2P contract (docs/phase4/README.md, "Accuracy measures"), not bit
//! for bit; `host_fallback` with [`OperatorKind::P2p`] restores the host P2P.
//!
//! # The leaf operators on the device (T7)
//!
//! P2M, P2L, L2P and M2P run on the device by default: one launch of
//! `nd_fmm_kernels::leaf::{p2m, p2l, l2p, m2p}` per level whose view has an entry (and,
//! for L2P and M2P, whose leaves have a target), from the level's view, the box and leaf
//! indices, the point offsets and the device stores, adding into the device multipoles
//! (P2M), locals (P2L) or target output (L2P, M2P). They move no data, so their rows of
//! the table above vanish from an evaluation's transfers. The order of every target is
//! the host path's (the plan's rows, points in point order), the frames are formed on the
//! device from the integer indices, and the harmonics by the recursion of `nd-fmm-math`
//! (CONVENTIONS §3.5, §3.13), so the output agrees with the host path within the FMM
//! bounds, not bit for bit (contraction, device-path.md §9.2). The layout is fixed at
//! build ([`DeviceReport::leaf_layout`]): by default the CPU layout on the CPU runtime,
//! whose units per cube `threads(n)` caps, and the cube layout of 64 units with tiles of
//! 32 points on Metal and CUDA (device-path.md §6.3;
//! [`FmmBuilder::device_leaf_layout`](crate::fmm::FmmBuilder::device_leaf_layout)).
//! `host_fallback` with any of the four kinds restores its host operator.
//!
//! # M2M and L2L on the device (T8)
//!
//! M2M (both passes) and L2L run on the device by default, for every strategy: the
//! grouped translation of `nd_fmm_kernels::translate` with the dense octant tables (built
//! here under `Rotation`, through `table_cache` when given), per level call a gather of
//! the view's input columns in batch order, one grouped GEMM over the level's octants and
//! a reduction per parent in row order (M2M) or a scatter-add (L2L, one parent per child):
//! three launches per chunk, with no transfer (device-path.md §6.4). The plans (chunks,
//! tile schedules) are built and uploaded at build, one per level and view, and the
//! scratch is allocated once, sized by the widest chunk under the scratch budget
//! ([`FmmBuilder::device_scratch_budget`](crate::fmm::FmmBuilder::device_scratch_budget),
//! default 128 MB); the chunks do not change the bits. The GEMM is fixed per level call at
//! build ([`DeviceReport::translations`]): the hand-written kernel in the backend's layout
//! ([`DeviceReport::gemm_layout`]), or, under
//! [`DeviceGemm::Auto`] and [`DeviceGemm::Library`] in f32 at p ≥ 8 on a GPU, the library
//! matmul (CMMA, named explicitly) where a probe at build accepts the level's shape and
//! its element types keep f32 (the input-precision guard). The hand-written GEMM sums each
//! product in a fixed order with explicit fmas and adds it into the output once, so the
//! device and the host (`MatrixSet::apply`, which adds each product into the output)
//! differ by rounding: the output agrees with the host path within the FMM bounds, not
//! bit for bit; under `Rotation` the host's M2M and L2L are rotations, the device's dense.
//! `host_fallback` with [`OperatorKind::M2m`] or [`OperatorKind::L2l`] restores the host
//! operator.
//!
//! # Dense M2L on the device (T9)
//!
//! M2L runs on the device by default under `Dense` and `Classes`: per level the same
//! grouped translation with the level's V view, grouped by the 316 offsets in index
//! order, and the dense M2L tables (`M2lTables`, through `table_cache` when given; under
//! `Classes` the 316 matrices of `M2lClasses::expand`, uploaded once and dropped on the
//! host, which keeps its class tables for the fallback; [`DeviceReport::strategy_name`]
//! says "Classes, run as dense on the device"; §6.8). Per chunk of offsets within the
//! scratch budget: a gather of the source multipoles in batch order, one grouped GEMM and
//! a reduction that adds each box's products into its local in row order, so each box
//! meets its offsets in index order; three launches per chunk, no transfer. The GEMM is
//! fixed per level at build ([`DeviceReport::translations`]): by default
//! ([`DeviceGemm::Auto`]) the hand-written kernel at every p and precision, because the
//! library's padding made it slower on every FMM level measured (see [`DeviceGemm`]);
//! under [`DeviceGemm::Library`], in f32 at p ≥ 8 on a GPU, the library matmul (CMMA,
//! named explicitly) where every chunk shape passes a probe and the input-precision
//! guard, padded per chunk to its widest offset batch
//! ([`TranslationReport::gemm_columns`]), and the hand-written kernel otherwise. The output
//! agrees with the host path within the FMM bounds, not bit for bit (the GEMM adds each
//! product into the local once, the host each term). Under `Rotation` M2L runs the rotation
//! kernel (next section); `host_fallback` with [`OperatorKind::M2l`] restores the host
//! operator under every strategy.
//!
//! The 316 dense tables take 316 (p + 1)⁴ values on the device, in MB (10⁶ bytes); the
//! library copy, uploaded only where the library may run (f32, p ≥ 8, a GPU, under
//! [`DeviceGemm::Library`]), as much
//! again (each matrix padded to 256 bytes):
//!
//! | p | 3 | 8 | 12 | 16 | 18 | 20 |
//! | --- | ---: | ---: | ---: | ---: | ---: | ---: |
//! | f32 | 0.32 | 8.29 | 36.1 | 106 | 165 | 246 |
//! | f64 | 0.65 | 16.6 | 72.2 | 211 | 329 | 492 |
//!
//! # Rotation M2L on the device (T10)
//!
//! Under `Rotation` M2L runs on the device by default: one launch of
//! `nd_fmm_kernels::rotation::m2l` per level whose V view has a pair, from the level's V
//! view and the M2L family of the host's `RotationTables`, uploaded once in its own storage
//! (the accessors of `ShiftTables` concatenated, the geometry of each offset as four `u32`;
//! `nd-fmm-tables` unchanged), adding into the device locals. Each box's row is one cube
//! of the cube layout (or a unit's share of the CPU layout), which walks the row in offset
//! order and repeats `ShiftTables::apply` step for step for each pair (z-rotation,
//! y-blocks, coaxial step, y-blocks back, z-rotation back; the coaxial step alone on the z
//! axis), with an explicit fma per multiply–add and the last step added into the box's
//! accumulator, which started from the local as L2L left it (device-path.md §6.6). It
//! moves no data. Its rows are uploaded at build; the layout is fixed at build
//! ([`DeviceReport::rotation_layout`]): the CPU layout on the CPU runtime, whose units per
//! cube `threads(n)` caps, and a cube of (p + 1)² units rounded up to the plane size on
//! Metal and CUDA. M2M and L2L stay the dense octant GEMMs (previous sections), whose
//! tables are built or loaded here under `Rotation`. The host rounds every product before
//! adding it and the kernel fuses them, so the output agrees with the host path within the
//! FMM bounds, not bit for bit. The tables take 94 B + 64 p + 15 C values with
//! B = (p + 1)(2p + 1)(2p + 3)/3 and C = (p + 1)(p + 2)(2p + 3)/6, and 5,056 bytes of
//! geometry, in MB:
//!
//! | p | 3 | 8 | 12 | 16 | 18 | 20 |
//! | --- | ---: | ---: | ---: | ---: | ---: | ---: |
//! | f32 | 0.04 | 0.39 | 1.16 | 2.58 | 3.59 | 4.85 |
//! | f64 | 0.07 | 0.77 | 2.31 | 5.15 | 7.18 | 9.69 |
//!
//! # Transfer accounting
//!
//! The operator counts every upload and download (calls and bytes), every launch, every
//! sync and every timing window, of the build and of the last evaluation
//! ([`DeviceCounters`]), and splits the transfers by the data they move ([`DataKind`],
//! [`Traffic`]).
//!
//! # The device FMM (T11): scheduling, syncs and stage timing (§8)
//!
//! **By default every operator kind runs on the device**, under every strategy, and
//! [`DeviceReport::placement`] says so; [`host_fallback`](crate::fmm::FmmBuilder::host_fallback)
//! stays as a test aid. The data stay on the device for the whole evaluation, and an
//! evaluation moves the design's minimum: the charges up (one upload of N_s s bytes) and
//! the output down (one download of o N_t s bytes), nothing else.
//!
//! Every level call queues its launches on the one stream and returns: no launch waits
//! for another, and **an evaluation syncs once**, at the download of the output, where
//! launch errors also surface (§8.2). The top levels stay on the device, since a host
//! call there would cost a sync (§6.7). The launches per evaluation (§8.1), with c the
//! offset chunks of a dense M2L level call (one with the default scratch budget, unless
//! a level's V batches exceed it):
//!
//! | Stage | Launches |
//! | --- | --- |
//! | `begin_evaluation` | 3 zero kernels (multipoles, locals, target output; none for an empty store) and the charge scatter |
//! | upward, local pass | per level: P2M 1; M2M 3 per chunk (gather, GEMM, reduction) |
//! | upward, global pass | on one rank the root's M2M: 3, none if the root is a leaf |
//! | downward | per level: L2L 3 per chunk (gather, GEMM, scatter-add); M2L 3 c (dense) or 1 (rotation); P2L 1 |
//! | leaves | per level: L2P 1, M2P 1, P2P 1 |
//!
//! A level call whose view has no entry launches nothing. The level calls of dense M2L
//! are not merged into one GEMM over every level (§6.7 allows it): it would save at most
//! two launches per level with a V list, which T11 measured as a negligible share of an
//! evaluation. On Metal and CUDA neither uploads nor launches wait; on the CPU runtime
//! the stream drains before a host write and before every launch with shared memory
//! (F10), which costs time but no extra sync of the operator and changes no value.
//!
//! **Stage timing** ([`StageTiming`], [`DeviceReport::stage_timing`]): `Fmm` times every
//! stage on the host, which on a device measures the enqueueing of its launches (the
//! default, [`StageTiming::Enqueue`]); [`synchronous_stages`](crate::fmm::FmmBuilder::synchronous_stages)
//! times them whole on every backend, at a sync per stage. With
//! [`device_timestamps`](crate::fmm::FmmBuilder::device_timestamps), where the device
//! times on itself (Metal with timestamp queries, CUDA by events;
//! `nd_fmm_kernels::Device::times_on_device`), each stage with device work also runs in a
//! timing window ([`open_stage`](DeviceOperator::open_stage),
//! [`close_stage`](DeviceOperator::close_stage); five per evaluation, one per
//! [`DeviceStage`]), which submits the queued work without waiting for it; the windows
//! are read after the download, and `Fmm` returns their times in
//! [`StageTimings::device`](crate::fmm::StageTimings::device). They add no sync, launch
//! nothing and change no bit (tested on Metal), but they are opt-in: on Metal the device
//! runs independent passes of neighbouring stages concurrently, so the windows overlap
//! and their sum exceeds the evaluation's wall time (T11 measured 11.9 ms of stage spans
//! in a 7.3 ms evaluation), adjacent windows sometimes report the same span, and the
//! flushes double the enqueue time (about 0.3–0.6 ms). A stage's time is its span on the
//! device, not its share of the evaluation. A window the device did not time itself (it
//! waited: a wgpu stream that found the device's timestamp queries spent) turns the
//! windows off for the later evaluations of the operator. The CPU runtime drains its
//! stream at both ends of a window, so there no window is opened.
//!
//! # Autotune (T12, C4.7; [`crate::tune`])
//!
//! Every grouped level call keeps the GEMM choice it was built with (the GEMM, the layout
//! of its operands and the chunk budget; [`TranslationReport`]), and the P2P layout is one
//! choice: under `DeviceGemm::Auto` and `DeviceP2pLayout::Auto` each is the choice of its
//! decision, one per (kind, pair bucket) and one per points-per-leaf bucket. At build the
//! operator takes them from the decisions taken earlier in the build (the M2L GEMM timed
//! with the strategy), from the tuning cache, or by the static rule, and builds every plan
//! with its choice. With a tuning cache that lacks a decision, the operator provides for
//! every candidate (the library copies of the tables where the library applies, the
//! scratch of its padded layout), times the candidates at the end of
//! [`new`](DeviceOperator::new) (the GEMMs, on its own buffers, each on the largest level
//! call of its bucket) and in [`load_points`](DeviceOperator::load_points) (P2P, after the
//! points are uploaded), rebuilds the plans whose choice changed, and releases what no
//! chosen plan needs; the report's tables, scratch and memory follow. The M2L strategy
//! itself is decided before the operator exists (`tune_strategy`), since it decides
//! the tables. Every decision, with where its choice came from and its candidates' times,
//! is in [`DeviceReport::tuning`]. Tuning launches belong to the build counters; an
//! evaluation zeroes every buffer the tuning wrote, so the output does not depend on it.
//!
//! # Determinism (requirement 6)
//!
//! Every launch and transfer is issued from the thread that calls the operator, so they
//! run in order on one CubeCL stream; host-fallback calls are bit-identical for every
//! thread count. The launch sequence, the chunks, the tile schedules, the layouts and the
//! GEMM of every level call are fixed at build from the plan, the device and the choices
//! of the build (static rule, cache or tuning, T12), so for a fixed tree, backend, device,
//! build and choices the output is bit-identical from evaluation to evaluation, and from
//! build to build of the same input with the same choices (tested on every
//! `tests/mpi_exec.rs` scenario, and for builds from one tuning cache in
//! `tests/device_tune.rs`). Timing windows change nothing.
//!
//! # Differences from the host path (§9.2)
//!
//! With every kind on the device the output agrees with the host path's within the FMM
//! bounds of docs/phase4/README.md (relative L2 1e-12 in f64, 1e-5 in f32, φ and ∇φ),
//! not bit for bit: the device GEMMs add each product into the output once, in their own
//! order, the rotation and leaf kernels fuse multiply–adds, and P2P follows the device
//! formulation of CONVENTIONS §3.13, "Device kernels". The errors against the direct sum
//! are those of the host path (`tests/device_fmm.rs`, the C4.8 gate). Under `Rotation`
//! the host translates M2M and L2L by rotation and the device by the dense octant
//! tables; `Classes` runs as dense on the device.
//!
//! # Errors (§12)
//!
//! A failed upload or allocation at build is
//! [`FmmError::Device`]. During an evaluation the operator
//! keeps the first failure, skips its remaining device work (the evaluator's stages
//! still run to the end), and [`read_output`](DeviceOperator::read_output) returns it;
//! `Fmm::evaluate` then returns `FmmError::Device`, and so does every later evaluation.
//! A violated invariant (a malformed view, a ghost in the plan) panics, as on the host
//! path.
//!
//! # Threads (§11)
//!
//! With [`Backend::Cpu`] no rayon pool is built: `threads(n)` caps the units per cube of
//! the CPU runtime's launches at n ([`nd_fmm_kernels::Device::limit_units`]), so the
//! rank keeps at most n of CubeCL's workers busy in them (the movement kernels and the
//! CPU layouts of P2P, the leaf operators and the GEMM), and host-fallback kinds run
//! serially. Kernels with shared memory
//! or barriers (the GPU layouts of P2P, if chosen there) keep their own cube size on the
//! CPU runtime. With Metal or CUDA the pool of `threads(n)` is built as on the
//! host path and serves only host-fallback kinds; a fallback call waits for the device
//! (its download) before its body runs on the pool, and every launch is issued from the
//! calling thread, so the pool is idle whenever device work runs. Worker threads never
//! launch and never call MPI.
//!
//! # Example
//!
//! The FMM of the crate example on the CPU runtime, which with every kind on the host
//! fallback gives the host path's output bit for bit:
//!
//! ```no_run
//! use nd_fmm_exec::fmm::{Backend, FmmBuilder};
//!
//! let universe = mpi::initialize().expect("MPI initialises once");
//! let comm = universe.world();
//! let points: Vec<[f64; 3]> = (0..10_000)
//!     .map(|i| {
//!         let t = i as f64;
//!         [(0.37 * t).sin(), (0.71 * t).cos(), (0.13 * t).sin()]
//!     })
//!     .collect();
//! let charges = vec![1.0; points.len()];
//!
//! let builder = FmmBuilder::<f64>::new(6).gradients(true);
//! let mut host = builder.build(&points, &points, &comm).unwrap();
//! let mut device = builder
//!     .clone()
//!     .backend(Backend::Cpu)
//!     .build(&points, &points, &comm)
//!     .expect("the CPU runtime is compiled in");
//! assert_eq!(host.evaluate(&charges).unwrap(), device.evaluate(&charges).unwrap());
//! let report = device.device_report().expect("a device backend");
//! let counters = device.device_counters().expect("a device backend");
//! println!("{report}\nlast evaluation: {:?}", counters.evaluation);
//! ```

use std::any::Any;
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mpi::traits::Equivalence;
use nd_fmm_kernels::leaf::{LeafLayout, SourceInputs, TargetInputs};
use nd_fmm_kernels::movement::{scatter_values, zero};
use nd_fmm_kernels::p2p::{P2pInputs, P2pLayout};
pub use nd_fmm_kernels::rotation::RotationLayout;
use nd_fmm_kernels::rotation::{
    Alignment as DeviceAlignment, RotationArrays, RotationPlan, RotationTables as DeviceRotation,
    Shift as DeviceShift,
};
use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, GroupedPlan, Operands, Orientation, PlanSettings, Tables,
    TranslationScratch, grouped,
};
pub use nd_fmm_kernels::translate::{Gemm, GemmLayout};
use nd_fmm_kernels::view::{
    BoxCoordinates, GroupedArrays, GroupedView, IndexView, LeafCoordinates, PointOffsets,
};
pub use nd_fmm_kernels::view::{CsrImage, GroupedImage};
use nd_fmm_kernels::{
    BackendKind, Device, DeviceBuffer, DeviceElement, DeviceFloat, DeviceSlice, DeviceSliceMut,
    IndexBuffer, KernelError, TimingWindow, WindowTime,
};
pub use nd_fmm_kernels::{Counters, DeviceInfo};
use nd_fmm_plan::lists::{Csr, GroupedCsr, VList};
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, UpwardPass,
};
use nd_fmm_plan::plan::Plan;
use nd_fmm_plan::store::{LeafStore, LevelBuffers};
use nd_fmm_tables::cache::{Stored, TableKind};
use nd_fmm_tables::rotation::{Alignment, Operator as RotationOperator, ShiftTables};
use nd_fmm_tables::{
    CacheOutcome, L2lTables, M2lTables, M2mTables, MatrixSet, RotationTables as HostRotationTables,
    TableCache,
};
use nd_octree::morton;

use crate::fmm::{
    Backend, DeviceGemm, DeviceLeafLayout, DeviceP2pLayout, DeviceStage, DeviceStageTimings,
    FmmError, OperatorKind, Placement, SettingsError,
};
use crate::operator::{LaplaceOperator, SimdScalar};
use crate::tables::M2lStrategy;
use crate::tune::{
    Candidate, Decision, GemmChoice, GemmKind, MIN_TUNED_PAIRS, Rejection, Tuner, TuningReport,
    bucket, gemm_candidates, library_applies, p2p_candidates, static_gemm, static_p2p,
    static_strategy, time_launches,
};

/// The precisions of the device operator, f32 and f64: the bounds of the host operator
/// and of the table cache, and a float type of `nd-fmm-kernels`.
pub trait DeviceScalar: DeviceFloat + SimdScalar + Stored + Equivalence + Default {}

impl<T: DeviceFloat + SimdScalar + Stored + Equivalence + Default> DeviceScalar for T {}

impl Backend {
    /// Opens the device of this backend and returns what it reports, or why it cannot be
    /// opened: the reason [`SettingsError::NoDevice`] cannot carry (§3.3, §12). The
    /// device is closed again; a build opens its own.
    ///
    /// # Errors
    ///
    /// The message of `nd_fmm_kernels::KernelError`: the backend is not compiled in, or
    /// no device came up. For [`Backend::Host`], that it opens no device.
    pub fn probe(self) -> Result<DeviceInfo, String> {
        match self.kind() {
            None => Err("the host backend opens no device".to_owned()),
            Some(kind) => Device::open(kind)
                .map(|device| device.info().clone())
                .map_err(|error| error.to_string()),
        }
    }

    /// The backend of `nd-fmm-kernels`, `None` for [`Backend::Host`].
    pub(crate) fn kind(self) -> Option<BackendKind> {
        match self {
            Self::Host => None,
            Self::Cpu => Some(BackendKind::Cpu),
            Self::Metal => Some(BackendKind::Metal),
            Self::Cuda => Some(BackendKind::Cuda),
        }
    }
}

/// Opens the device of `backend` for values of `T` at step 1 of `build`, with the CPU
/// runtime's units capped at `threads`; `None` for the host.
pub(crate) fn open_device<T: Stored>(
    backend: Backend,
    threads: usize,
) -> Result<Option<Device>, SettingsError> {
    let Some(kind) = backend.kind() else {
        return Ok(None);
    };
    let mut device = Device::open(kind).map_err(|error| match error {
        KernelError::NotCompiled { .. } => SettingsError::BackendNotCompiled { backend },
        _ => SettingsError::NoDevice { backend },
    })?;
    let precision = match T::PRECISION {
        nd_fmm_tables::cache::Precision::F32 => nd_fmm_kernels::Precision::F32,
        nd_fmm_tables::cache::Precision::F64 => nd_fmm_kernels::Precision::F64,
    };
    if !device.supports(precision) {
        return Err(SettingsError::PrecisionUnsupported { backend });
    }
    device.limit_units(u32::try_from(threads).unwrap_or(u32::MAX));
    Ok(Some(device))
}

/// The kinds of data the device operator moves, for the transfer accounting
/// ([`DeviceCounters`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataKind {
    /// The leaf-scaled source coordinates (with zero charges) and target positions:
    /// once per build.
    Points,
    /// The plan views, the point offsets of the leaf stores, the charge slots and the
    /// plans of the device translations (T8, T9): once per build.
    Indices,
    /// The box and leaf indices: once per build.
    Geometry,
    /// The translation tables: once per build.
    Tables,
    /// The charges: once per evaluation.
    Charges,
    /// The target output: once per evaluation.
    Output,
    /// Multipoles moved by host-fallback calls.
    FallbackMultipoles,
    /// Locals moved by host-fallback calls.
    FallbackLocals,
    /// Target output moved by host-fallback calls.
    FallbackTargetOutput,
}

impl DataKind {
    /// Every kind, in report order.
    pub const ALL: [Self; 9] = [
        Self::Points,
        Self::Indices,
        Self::Geometry,
        Self::Tables,
        Self::Charges,
        Self::Output,
        Self::FallbackMultipoles,
        Self::FallbackLocals,
        Self::FallbackTargetOutput,
    ];

    /// A short name for reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::Indices => "indices",
            Self::Geometry => "geometry",
            Self::Tables => "tables",
            Self::Charges => "charges",
            Self::Output => "output",
            Self::FallbackMultipoles => "fallback multipoles",
            Self::FallbackLocals => "fallback locals",
            Self::FallbackTargetOutput => "fallback target output",
        }
    }
}

impl fmt::Display for DataKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Uploads and downloads of one kind of data: calls and bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    /// Host-to-device copies.
    pub uploads: u64,
    /// Bytes copied to the device.
    pub upload_bytes: u64,
    /// Device-to-host copies, each one sync.
    pub downloads: u64,
    /// Bytes copied to the host.
    pub download_bytes: u64,
}

impl Traffic {
    /// The transfers between two readings of a device's counters.
    fn between(before: Counters, after: Counters) -> Self {
        Self {
            uploads: after.uploads - before.uploads,
            upload_bytes: after.upload_bytes - before.upload_bytes,
            downloads: after.downloads - before.downloads,
            download_bytes: after.download_bytes - before.download_bytes,
        }
    }

    /// The sum of two.
    fn plus(self, other: Self) -> Self {
        Self {
            uploads: self.uploads + other.uploads,
            upload_bytes: self.upload_bytes + other.upload_bytes,
            downloads: self.downloads + other.downloads,
            download_bytes: self.download_bytes + other.download_bytes,
        }
    }
}

/// The transfers of a build or an evaluation by [`DataKind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrafficByData([Traffic; DataKind::ALL.len()]);

impl TrafficByData {
    /// The transfers of one kind of data.
    pub fn get(&self, data: DataKind) -> Traffic {
        self.0[data as usize]
    }

    /// The transfers of every kind together.
    pub fn total(&self) -> Traffic {
        self.0.iter().fold(Traffic::default(), |a, &b| a.plus(b))
    }

    fn add(&mut self, data: DataKind, traffic: Traffic) {
        self.0[data as usize] = self.0[data as usize].plus(traffic);
    }
}

/// The transfers, launches and syncs of a device operator: of its build and of the last
/// evaluation ([module documentation](self#transfer-accounting)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceCounters {
    /// Everything from opening the device to the end of the build (allocations zero
    /// their buffers by a kernel, so the build launches too).
    pub build: Counters,
    /// Everything from [`begin_evaluation`](DeviceOperator::begin_evaluation) to
    /// [`read_output`](DeviceOperator::read_output) of the last evaluation, its stage
    /// syncs included.
    pub evaluation: Counters,
    /// The transfers of the build by kind of data.
    pub build_traffic: TrafficByData,
    /// The transfers of the last evaluation by kind of data.
    pub evaluation_traffic: TrafficByData,
}

/// A table family uploaded to the device, for [`DeviceReport`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceTable {
    /// The family: `M2M`, `L2L`, `M2L`, `M2L (expanded from the classes)` or
    /// `M2L (rotation)`, and the library copies.
    pub name: &'static str,
    /// The number of matrices; for the rotation tables, the offsets they translate
    /// across.
    pub matrices: usize,
    /// The bytes on the device.
    pub bytes: u64,
}

/// One device level call of M2M, L2L (T8) or M2L (T9), fixed at build, for
/// [`DeviceReport`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslationReport {
    /// [`OperatorKind::M2m`], [`OperatorKind::L2l`] or [`OperatorKind::M2l`].
    pub kind: OperatorKind,
    /// The pass of an M2M call; `None` for L2L and M2L.
    pub pass: Option<UpwardPass>,
    /// The level of the call (of the parents for M2M, of the children for L2L, of the
    /// targets and sources for M2L).
    pub level: usize,
    /// The pairs of the view.
    pub pairs: usize,
    /// The GEMM that runs.
    pub gemm: Gemm,
    /// The layout of the gathered inputs and products: box-major (T12: coefficient-major
    /// is an option of `nd-fmm-kernels` that the tuner does not register, `tune`).
    pub orientation: Orientation,
    /// The chunk budget in bytes (T12: the default unless set or tuned).
    pub budget: u64,
    /// The chunks: three launches each.
    pub chunks: usize,
    /// The columns the GEMMs compute over the chunks: the pairs with the hand-written
    /// kernel, and with the library also its padding columns
    /// ([`GroupedPlan::gemm_columns`]).
    pub gemm_columns: usize,
    /// Why the library does not run where the rule would choose it
    /// ([`GroupedPlan::library_rejection`]).
    pub library_rejection: Option<String>,
}

/// One device level call of rotation M2L (T10), fixed at build, for [`DeviceReport`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotationReport {
    /// The level of the targets and sources.
    pub level: usize,
    /// The pairs of the view.
    pub pairs: usize,
    /// The rows with a pair: the cubes of the cube layout.
    pub rows: usize,
}

/// How [`StageTimings`](crate::fmm::StageTimings) time the stages of a device
/// evaluation (docs/design/device-path.md §8.3), fixed at build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StageTiming {
    /// One timing window per stage with device work, timed by the device itself and read
    /// after the evaluation's one download: no sync of its own
    /// ([`FmmBuilder::device_timestamps`](crate::fmm::FmmBuilder::device_timestamps), on
    /// Metal and CUDA; opt-in).
    DeviceTimestamps,
    /// A sync after the charge upload and after every stage
    /// ([`FmmBuilder::synchronous_stages`](crate::fmm::FmmBuilder::synchronous_stages)):
    /// each stage timed whole on the host.
    Synchronous,
    /// The host clock only, which times the enqueueing of each stage: the default, and on
    /// the CPU runtime also with `device_timestamps(true)`, since its timing windows wait
    /// for it.
    #[default]
    Enqueue,
}

impl fmt::Display for StageTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DeviceTimestamps => "device timestamps, one window per stage, no sync",
            Self::Synchronous => "synchronous stages, a sync after each",
            Self::Enqueue => "host clock, enqueue time only",
        })
    }
}

/// What a device operator runs on and how, fixed at build, for reports.
#[derive(Debug)]
pub struct DeviceReport {
    /// The device.
    pub info: DeviceInfo,
    /// Where each operator kind runs, in [`OperatorKind::ALL`] order.
    pub placement: [Placement; 8],
    /// The kinds the builder asked to run on the host ([`FmmBuilder::host_fallback`]).
    ///
    /// [`FmmBuilder::host_fallback`]: crate::fmm::FmmBuilder::host_fallback
    pub requested_fallback: Vec<OperatorKind>,
    /// The resolved M2L strategy of the host operator; [`strategy_name`](Self::strategy_name)
    /// says how the device runs it.
    pub strategy: M2lStrategy,
    /// The table families on the device.
    pub tables: Vec<DeviceTable>,
    /// What the table cache did for the tables the device needs beyond the host's (the
    /// dense octant tables under `Rotation`); empty without a cache or without such
    /// tables.
    pub cache_outcomes: Vec<(TableKind, CacheOutcome)>,
    /// The cap on the units per cube of the CPU runtime's launches, `threads(n)`;
    /// `None` on the GPU backends.
    pub cpu_units: Option<u32>,
    /// The layout of the device P2P kernel (T6), fixed at build: by default the CPU
    /// layout on the CPU runtime and the cube layout on the GPUs
    /// ([`FmmBuilder::device_p2p_layout`](crate::fmm::FmmBuilder::device_p2p_layout)).
    pub p2p_layout: P2pLayout,
    /// The layout of the device leaf operators P2M, L2P, P2L and M2P (T7), fixed at
    /// build for the FMM's degree and precision: by default the CPU layout on the CPU
    /// runtime and the cube layout on the GPUs
    /// ([`FmmBuilder::device_leaf_layout`](crate::fmm::FmmBuilder::device_leaf_layout)).
    pub leaf_layout: LeafLayout,
    /// The layout of the hand-written GEMM of the device translations (T8, T9), fixed at
    /// build for the FMM's degree: by default the CPU layout on the CPU runtime and the
    /// cube layout on the GPUs ([`GemmLayout::default_for`]).
    pub gemm_layout: GemmLayout,
    /// The device level calls of M2M, L2L and (under `Dense` and `Classes`) M2L with a
    /// pair, in the order local M2M, global M2M, L2L, M2L and by level: the GEMM and
    /// chunks of each (T8, T9).
    pub translations: Vec<TranslationReport>,
    /// The layout of the device rotation M2L (T10), fixed at build for the FMM's degree:
    /// by default the CPU layout on the CPU runtime and the cube layout of (p + 1)² units,
    /// rounded up to the plane size, on the GPUs ([`RotationLayout::default_for`]).
    pub rotation_layout: RotationLayout,
    /// The device level calls of rotation M2L with a pair, under `Rotation`, by level:
    /// one launch each (T10).
    pub rotations: Vec<RotationReport>,
    /// The bytes of the translation scratch: the gathered inputs and the products of
    /// the widest chunk (T8, T9).
    pub scratch_bytes: u64,
    /// The bytes of every buffer the operator allocates on the device, as summed for the
    /// check before allocating, less what tuning released after it (T12).
    pub memory_needed: u64,
    /// The bytes the device reported as available before the allocation, `None` if the
    /// backend reports no limit; the check was then skipped.
    pub memory_available: Option<u64>,
    /// How the stages of an evaluation are timed: by the host clock, unless
    /// `synchronous_stages` asks for a sync per stage or `device_timestamps` for timing
    /// windows (where the device times on itself).
    pub stage_timing: StageTiming,
    /// What the tuner did (T12, [`crate::tune`]): every decision of the build, with its
    /// source (the static rule, the cache, or tuned now) and the candidates' times; `None`
    /// only before the points are loaded.
    pub tuning: Option<TuningReport>,
}

impl DeviceReport {
    /// Where `kind` runs.
    pub fn placement(&self, kind: OperatorKind) -> Placement {
        self.placement[kind as usize]
    }

    /// How the device runs the strategy (device-path.md §6.6, §6.8): `Dense`, "Classes,
    /// run as dense on the device" (the 316 tables of `M2lClasses::expand` on the device,
    /// the class tables on the host for the fallback), or `Rotation` (the rotation kernel,
    /// T10); with M2L on the host fallback by request, the strategy and "M2L on the host
    /// fallback".
    pub fn strategy_name(&self) -> String {
        match (self.strategy, self.placement(OperatorKind::M2l)) {
            (M2lStrategy::Dense, Placement::Device) => "Dense".to_owned(),
            (M2lStrategy::Classes, Placement::Device) => {
                "Classes, run as dense on the device".to_owned()
            }
            (M2lStrategy::Rotation, Placement::Device) => "Rotation".to_owned(),
            (strategy, _) => format!("{strategy:?}, M2L on the host fallback"),
        }
    }

    /// The device level calls of `kind`.
    pub fn translations_of(&self, kind: OperatorKind) -> impl Iterator<Item = &TranslationReport> {
        self.translations.iter().filter(move |t| t.kind == kind)
    }
}

impl fmt::Display for DeviceReport {
    /// A few lines: the device, the placement, the tables and the memory.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "device: {}", self.info)?;
        let placement: Vec<String> = OperatorKind::ALL
            .iter()
            .map(|&kind| format!("{kind} {}", self.placement(kind)))
            .collect();
        writeln!(f, "placement: {}", placement.join(", "))?;
        let tables: Vec<String> = self
            .tables
            .iter()
            .map(|t| format!("{} ({} matrices, {} B)", t.name, t.matrices, t.bytes))
            .collect();
        writeln!(
            f,
            "strategy {}; tables on the device: {}",
            self.strategy_name(),
            if tables.is_empty() {
                "none".to_owned()
            } else {
                tables.join(", ")
            }
        )?;
        writeln!(f, "P2P layout: {}", self.p2p_layout)?;
        writeln!(f, "leaf-operator layout: {}", self.leaf_layout)?;
        for (name, kinds) in [
            ("M2M/L2L", &[OperatorKind::M2m, OperatorKind::L2l][..]),
            ("M2L", &[OperatorKind::M2l][..]),
        ] {
            let calls: Vec<&TranslationReport> = self
                .translations
                .iter()
                .filter(|t| kinds.contains(&t.kind))
                .collect();
            if calls.is_empty() {
                continue;
            }
            let library = calls.iter().filter(|t| t.gemm == Gemm::Library).count();
            let chunks: usize = calls.iter().map(|t| t.chunks).sum();
            writeln!(
                f,
                "{name}: {} level calls in {chunks} chunks, the library GEMM in {library}, \
                 the hand-written {} in the others",
                calls.len(),
                self.gemm_layout,
            )?;
            for t in calls.iter().filter(|t| {
                t.orientation != Orientation::BoxMajor
                    || t.budget != DEFAULT_SCRATCH_BYTES
                    || matches!(t.gemm, Gemm::HandWritten(l) if l != self.gemm_layout)
            }) {
                writeln!(
                    f,
                    "  {} level {}: {}, {}, {} MB",
                    t.kind,
                    t.level,
                    t.gemm,
                    t.orientation,
                    t.budget >> 20
                )?;
            }
        }
        if !self.rotations.is_empty() {
            let pairs: usize = self.rotations.iter().map(|r| r.pairs).sum();
            writeln!(
                f,
                "M2L: rotation in {} level calls of {pairs} pairs, one launch each, layout {}",
                self.rotations.len(),
                self.rotation_layout
            )?;
        }
        if !self.translations.is_empty() {
            writeln!(f, "translation scratch: {} B", self.scratch_bytes)?;
        }
        if let Some(units) = self.cpu_units {
            writeln!(f, "CPU runtime: at most {units} units per cube")?;
        }
        writeln!(f, "stage timing: {}", self.stage_timing)?;
        if let Some(tuning) = &self.tuning {
            writeln!(f, "{tuning}")?;
        }
        match self.memory_available {
            Some(available) => write!(
                f,
                "device memory: {} B needed, {available} B available",
                self.memory_needed
            ),
            None => write!(
                f,
                "device memory: {} B needed; the backend reports no limit, not checked",
                self.memory_needed
            ),
        }
    }
}

/// The settings of a [`DeviceOperator`] beyond its host operator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceOptions {
    /// Kinds to run on the host even where a device kernel exists.
    pub host_fallback: Vec<OperatorKind>,
    /// The table cache of the tables the device needs beyond the host's.
    pub table_cache: Option<PathBuf>,
    /// The layout of the device P2P (T6); [`DeviceP2pLayout::Auto`] by default.
    pub p2p_layout: DeviceP2pLayout,
    /// The layout of the device leaf operators (T7); [`DeviceLeafLayout::Auto`] by
    /// default.
    pub leaf_layout: DeviceLeafLayout,
    /// The GEMM of the device translations M2M, L2L and M2L (T8, T9); [`DeviceGemm::Auto`]
    /// by default.
    pub gemm: DeviceGemm,
    /// The scratch budget of the device translations in bytes; `None` for
    /// [`DEFAULT_SCRATCH_BYTES`].
    pub scratch_budget: Option<u64>,
    /// How to time the stages; [`StageTiming::Enqueue`] by default.
    /// [`StageTiming::DeviceTimestamps`] falls back to it on a device that does not time on
    /// itself.
    pub stage_timing: StageTiming,
}

impl DeviceP2pLayout {
    /// The kernel layout of this choice on the device `info`: for
    /// [`Auto`](DeviceP2pLayout::Auto) the default of `nd-fmm-kernels`
    /// (`P2pLayout::default_for`: the CPU layout on the CPU runtime, the cube layout of
    /// 64 units on the GPUs).
    pub fn resolve(self, info: &DeviceInfo) -> P2pLayout {
        match self {
            Self::Auto => P2pLayout::default_for(info),
            Self::Cube(units) => P2pLayout::Cube { units },
            Self::Plane(planes) => P2pLayout::Plane { planes },
            Self::Cpu => P2pLayout::Cpu {
                vector_bits: nd_fmm_kernels::p2p::CPU_VECTOR_BITS,
            },
        }
    }
}

impl DeviceLeafLayout {
    /// The kernel layout of this choice on the device `info` at degree p in `T`: for
    /// [`Auto`](DeviceLeafLayout::Auto) the default of `nd-fmm-kernels`
    /// (`LeafLayout::default_for`: the CPU layout on the CPU runtime, the cube layout of
    /// 64 units with tiles of up to 32 points on the GPUs).
    pub fn resolve<T: DeviceFloat>(self, info: &DeviceInfo, p: usize) -> LeafLayout {
        match self {
            Self::Auto => LeafLayout::default_for(info, p, T::FLOAT),
            Self::Cube { units, tile } => LeafLayout::Cube { units, tile },
            Self::Cpu => LeafLayout::Cpu,
        }
    }
}

/// The kinds with a device kernel (T6, T7, T8, T9, T10): on the device unless
/// `host_fallback` names them; M2L under every strategy (dense under `Dense` and
/// `Classes`, T9; the rotation kernel under `Rotation`, T10).
const DEVICE_KINDS: [OperatorKind; 8] = [
    OperatorKind::P2m,
    OperatorKind::M2m,
    OperatorKind::M2l,
    OperatorKind::P2l,
    OperatorKind::L2l,
    OperatorKind::L2p,
    OperatorKind::M2p,
    OperatorKind::P2p,
];

/// The leaf operators among them (T7).
const LEAF_KINDS: [OperatorKind; 4] = [
    OperatorKind::P2m,
    OperatorKind::P2l,
    OperatorKind::L2p,
    OperatorKind::M2p,
];

/// The views of one level on the device, uploaded from the plan's
/// (`nd_fmm_plan::lists::LevelLists`). Box views have one row per box of the level,
/// leaf views one row per local leaf of the level.
#[derive(Debug)]
pub struct LevelViews {
    /// P2M: leaf indices per box.
    pub p2m: IndexView,
    /// M2M of the local pass: children on the next finer level by octant.
    pub m2m_local: GroupedView,
    /// M2M of the global pass.
    pub m2m_global: GroupedView,
    /// L2L: the parent on the next coarser level by octant.
    pub l2l: GroupedView,
    /// M2L: V-list sources by offset index.
    pub v: GroupedView,
    /// P2L: X-list leaf indices per box.
    pub x: IndexView,
    /// P2P: near-list leaf indices per leaf, the leaf itself included.
    pub near: IndexView,
    /// M2P: W-list boxes on the next finer level per leaf.
    pub w: IndexView,
    /// L2P: the leaf's box per leaf.
    pub l2p: IndexView,
}

/// Every plan view of an FMM on the device, the box and leaf indices, and the index
/// arrays of the leaf stores, uploaded once at build ([module documentation](self)).
#[derive(Debug)]
pub struct DeviceViews {
    levels: Vec<LevelViews>,
    boxes: BoxCoordinates,
    leaves: LeafCoordinates,
    source_offsets: PointOffsets,
    target_offsets: PointOffsets,
    charge_slots: IndexBuffer,
}

impl DeviceViews {
    /// The views of `level`.
    pub fn level(&self, level: usize) -> &LevelViews {
        &self.levels[level]
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// The index of every box of every level.
    pub fn boxes(&self) -> &BoxCoordinates {
        &self.boxes
    }

    /// The level and index of every leaf.
    pub fn leaves(&self) -> &LeafCoordinates {
        &self.leaves
    }

    /// The `nleaves + 1` point offsets of the source store.
    pub fn source_offsets(&self) -> &PointOffsets {
        &self.source_offsets
    }

    /// The `nlocal + 1` point offsets of the target input and output.
    pub fn target_offsets(&self) -> &PointOffsets {
        &self.target_offsets
    }

    /// The position in the source store of every charge, in leaf order (CONVENTIONS
    /// §3.13, "Source chunks"): the targets of the charge scatter.
    pub fn charge_slots(&self) -> &IndexBuffer {
        &self.charge_slots
    }

    /// Downloads every array, for tests and reports.
    ///
    /// # Errors
    ///
    /// As `Device::download`.
    pub fn download(&self, device: &mut Device) -> Result<ViewsImage, KernelError> {
        let levels = self
            .levels
            .iter()
            .map(|views| {
                Ok(LevelViewsImage {
                    p2m: views.p2m.download(device)?,
                    m2m_local: views.m2m_local.download(device)?,
                    m2m_global: views.m2m_global.download(device)?,
                    l2l: views.l2l.download(device)?,
                    v: views.v.download(device)?,
                    x: views.x.download(device)?,
                    near: views.near.download(device)?,
                    w: views.w.download(device)?,
                    l2p: views.l2p.download(device)?,
                })
            })
            .collect::<Result<_, KernelError>>()?;
        let indices = |device: &mut Device, buffer: &IndexBuffer| {
            let mut out = vec![0u32; buffer.len()];
            device
                .download(buffer.buffer().as_slice(), &mut out)
                .map(|()| out)
        };
        Ok(ViewsImage {
            levels,
            boxes: self.boxes.download(device)?,
            leaves: self.leaves.download(device)?,
            source_offsets: self.source_offsets.download(device)?,
            target_offsets: self.target_offsets.download(device)?,
            charge_slots: indices(device, &self.charge_slots)?,
        })
    }
}

/// The views of one level as downloaded ([`DeviceViews::download`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelViewsImage {
    /// P2M.
    pub p2m: CsrImage,
    /// M2M, local pass.
    pub m2m_local: GroupedImage,
    /// M2M, global pass.
    pub m2m_global: GroupedImage,
    /// L2L.
    pub l2l: GroupedImage,
    /// M2L.
    pub v: GroupedImage,
    /// P2L.
    pub x: CsrImage,
    /// P2P.
    pub near: CsrImage,
    /// M2P.
    pub w: CsrImage,
    /// L2P.
    pub l2p: CsrImage,
}

/// Every array of [`DeviceViews`] as downloaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewsImage {
    /// The views of each level.
    pub levels: Vec<LevelViewsImage>,
    /// The index of every box, level after level.
    pub boxes: Vec<Vec<[u32; 3]>>,
    /// The level and index of every leaf.
    pub leaves: Vec<(u32, [u32; 3])>,
    /// The point offsets of the source store.
    pub source_offsets: Vec<u32>,
    /// The point offsets of the target input and output.
    pub target_offsets: Vec<u32>,
    /// The position of every charge in the source store.
    pub charge_slots: Vec<u32>,
}

/// The data stores of an evaluation on the device ([module documentation](self)).
#[derive(Debug)]
struct DeviceStores<T: DeviceFloat> {
    multipoles: DeviceBuffer<T>,
    locals: DeviceBuffer<T>,
    sources: DeviceBuffer<T>,
    target_input: DeviceBuffer<T>,
    target_output: DeviceBuffer<T>,
    /// The charges of an evaluation in leaf order, before the scatter.
    charges: DeviceBuffer<T>,
}

/// The translation tables on the device, with their library copies where a level call may
/// take the library GEMM (`nd_fmm_kernels::translate::Tables`).
#[derive(Debug)]
struct DeviceTables<T: DeviceFloat> {
    m2m: Tables<T>,
    l2l: Tables<T>,
    /// The dense M2L tables, under `Dense` and `Classes` (expanded from the classes).
    m2l: Option<Tables<T>>,
    /// The M2L family of the rotation tables, under `Rotation` (T10).
    rotation: Option<DeviceRotation<T>>,
}

/// One grouped level call on the device: its plan and the GEMM choice it was built with
/// (T12: the static rule, the cache, or the tuner).
#[derive(Debug)]
struct LevelCall {
    plan: GroupedPlan,
    choice: GemmChoice,
}

/// The plans of the device M2M, L2L (T8) and dense M2L (T9) level calls, one per level and
/// view (empty where the kind runs on the host), their shared scratch, and the plans of the
/// rotation M2L level calls (T10; empty unless M2L runs on the device under `Rotation`).
#[derive(Debug)]
struct Translations<T: DeviceFloat> {
    m2m_local: Vec<LevelCall>,
    m2m_global: Vec<LevelCall>,
    l2l: Vec<LevelCall>,
    m2l: Vec<LevelCall>,
    scratch: TranslationScratch<T>,
    rotation: Vec<RotationPlan>,
    rotation_layout: RotationLayout,
}

impl<T: DeviceFloat> Translations<T> {
    /// The level calls of `kind` (and `pass` for M2M).
    fn calls(&self, kind: OperatorKind, pass: Option<UpwardPass>) -> &[LevelCall] {
        match (kind, pass) {
            (OperatorKind::M2m, Some(UpwardPass::Local)) => &self.m2m_local,
            (OperatorKind::M2m, Some(UpwardPass::Global)) => &self.m2m_global,
            (OperatorKind::L2l, None) => &self.l2l,
            (OperatorKind::M2l, None) => &self.m2l,
            _ => unreachable!("M2M with a pass, L2L or M2L"),
        }
    }

    /// The level calls of `kind` (and `pass` for M2M), mutably.
    fn calls_mut(&mut self, kind: OperatorKind, pass: Option<UpwardPass>) -> &mut Vec<LevelCall> {
        match (kind, pass) {
            (OperatorKind::M2m, Some(UpwardPass::Local)) => &mut self.m2m_local,
            (OperatorKind::M2m, Some(UpwardPass::Global)) => &mut self.m2m_global,
            (OperatorKind::L2l, None) => &mut self.l2l,
            (OperatorKind::M2l, None) => &mut self.m2l,
            _ => unreachable!("M2M with a pass, L2L or M2L"),
        }
    }
}

/// A grouped level call: (kind, pass of M2M, level).
type CallId = (OperatorKind, Option<UpwardPass>, usize);

/// The grouped level calls, in report order: (kind, pass).
const GROUPED: [(OperatorKind, Option<UpwardPass>); 4] = [
    (OperatorKind::M2m, Some(UpwardPass::Local)),
    (OperatorKind::M2m, Some(UpwardPass::Global)),
    (OperatorKind::L2l, None),
    (OperatorKind::M2l, None),
];

/// The host copies of the multipoles and locals that host-fallback calls work on.
#[derive(Debug)]
struct Mirrors<T> {
    multipoles: LevelBuffers<T>,
    locals: LevelBuffers<T>,
}

/// The device with the accounting of its transfers and the first failure of an
/// evaluation.
#[derive(Debug)]
struct Link {
    device: Device,
    traffic: TrafficByData,
    error: Option<KernelError>,
}

impl Link {
    /// Runs `op` on the device unless an earlier operation failed, counts its transfers
    /// as `data`, and keeps its failure.
    fn run<R>(
        &mut self,
        data: DataKind,
        op: impl FnOnce(&mut Device) -> Result<R, KernelError>,
    ) -> Option<R> {
        if self.error.is_some() {
            return None;
        }
        let before = self.device.counters();
        let result = op(&mut self.device);
        self.traffic
            .add(data, Traffic::between(before, self.device.counters()));
        result.map_err(|error| self.error = Some(error)).ok()
    }

    /// [`run`](Self::run) at build, where a failure ends the build.
    fn build<R>(
        &mut self,
        data: DataKind,
        op: impl FnOnce(&mut Device) -> Result<R, KernelError>,
    ) -> Result<R, FmmError> {
        match self.run(data, op) {
            Some(value) => Ok(value),
            None => Err(FmmError::Device(
                self.error
                    .take()
                    .map_or_else(String::new, |error| error.to_string()),
            )),
        }
    }
}

/// Which level store a host-fallback transfer moves.
#[derive(Clone, Copy, Debug)]
enum Level {
    Multipoles,
    Locals,
}

/// The Laplace operator with its data on a device, and a host fallback per operator
/// kind; see the [module documentation](self).
///
/// It wraps the [`LaplaceOperator`] that runs its host-fallback kinds, built as
/// `FmmBuilder::build` builds the host operator, and implements [`FmmSizes`] and
/// [`FmmOperator`] for the `Evaluator`. [`Fmm`](crate::fmm::Fmm) drives the evaluation
/// boundary: [`begin_evaluation`](Self::begin_evaluation) after the evaluator's `reset`,
/// [`read_output`](Self::read_output) after its stages.
pub struct DeviceOperator<T: DeviceScalar> {
    host: LaplaceOperator<T>,
    link: Link,
    placement: [Placement; 8],
    /// The layout of the device P2P, fixed at build.
    p2p_layout: P2pLayout,
    /// The layout of the device leaf operators, fixed at build.
    leaf_layout: LeafLayout,
    stores: DeviceStores<T>,
    /// `nlevels + 1` offsets of the levels in the multipole and local buffers, in values.
    level_offsets: Vec<usize>,
    views: DeviceViews,
    tables: DeviceTables<T>,
    translations: Translations<T>,
    mirrors: Option<Mirrors<T>>,
    /// The target output on the host: the mirror of host-fallback calls, and where
    /// [`read_output`](Self::read_output) downloads to.
    output: LeafStore<T>,
    report: DeviceReport,
    build_counters: Counters,
    build_traffic: TrafficByData,
    evaluation_counters: Counters,
    /// Whether stages are timed by timing windows ([`StageTiming::DeviceTimestamps`]);
    /// turned off for good if a window was not timed on the device.
    stage_windows: bool,
    /// The open timing window of the current stage.
    window: Option<TimingWindow>,
    /// The closed windows of this evaluation, resolved by `read_output`.
    pending: Vec<(DeviceStage, WindowTime)>,
    /// The device times of the stages of the last evaluation.
    stage_timings: Option<DeviceStageTimings>,
    /// The settings the operator was built with.
    options: DeviceOptions,
    /// The tuner until the points are loaded (T12): the translations are tuned at the end
    /// of [`new`](Self::new), P2P in [`load_points`](Self::load_points), which ends it.
    tuner: Option<Tuner>,
    /// Per level, the local leaves of its P2P level call and their near-field pairs
    /// (targets times sources over the near list), for the P2P decision.
    p2p_levels: Vec<(Range<usize>, usize)>,
    /// The mean points per local leaf, (N_s + N_t) / (2 leaves) rounded up: the bucket of
    /// the P2P decision.
    mean_points: usize,
}

impl<T: DeviceScalar> fmt::Debug for DeviceOperator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceOperator")
            .field("report", &self.report)
            .field("error", &self.link.error)
            .finish_non_exhaustive()
    }
}

/// The bytes of a CSR view of `nrows` rows and `len` entries on the device.
fn csr_bytes(nrows: usize, len: usize) -> u64 {
    Device::buffer_bytes::<u32>(nrows + 1) + Device::buffer_bytes::<u32>(len)
}

/// The bytes of a grouped view on the device: seven index buffers.
fn grouped_bytes<G: Copy + Into<usize>>(view: &GroupedCsr<G>) -> u64 {
    let len = view.sources().len();
    Device::buffer_bytes::<u32>(view.row_offsets().len())
        + Device::buffer_bytes::<u32>(view.batch_offsets().len())
        + 5 * Device::buffer_bytes::<u32>(len)
}

/// The bytes of the views of every level.
fn view_bytes(plan: &Plan) -> u64 {
    plan.levels()
        .iter()
        .map(|lists| {
            let csr = |view: &Csr| csr_bytes(view.nrows(), view.len());
            csr(lists.p2m())
                + grouped_bytes(lists.m2m_local())
                + grouped_bytes(lists.m2m_global())
                + grouped_bytes(lists.l2l())
                + grouped_bytes(lists.v())
                + csr(lists.x())
                + csr(lists.near())
                + csr(lists.w())
                + csr(lists.l2p())
        })
        .sum()
}

/// CSR point offsets of `counts`, as `u32`.
fn point_offsets(counts: &[usize]) -> Result<Vec<u32>, KernelError> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut total = 0usize;
    offsets.push(0);
    for &n in counts {
        total += n;
        offsets.push(u32::try_from(total).map_err(|_| KernelError::TooLarge {
            what: "leaf store",
            len: total,
        })?);
    }
    Ok(offsets)
}

/// The position of every charge in a source store with `counts` points per leaf and
/// four values per point, in leaf order: leaf j's charge k at 4 P_j + 3 n_j + k
/// (CONVENTIONS §3.13, "Source chunks").
fn charge_slots(counts: &[usize]) -> Result<Vec<u32>, KernelError> {
    let total: usize = counts.iter().sum();
    let mut slots = Vec::with_capacity(total);
    let mut start = 0usize;
    for &n in counts {
        let first = start + 3 * n;
        for k in 0..n {
            slots.push(u32::try_from(first + k).map_err(|_| KernelError::TooLarge {
                what: "source store",
                len: 4 * total,
            })?);
        }
        start += 4 * n;
    }
    Ok(slots)
}

/// The arrays of a grouped view of the plan.
fn grouped_arrays<G: Copy + Into<usize>>(view: &GroupedCsr<G>) -> GroupedArrays<'_, G> {
    GroupedArrays {
        row_offsets: view.row_offsets(),
        sources: view.sources(),
        groups: view.groups(),
        batch_offsets: view.batch_offsets(),
        batch_targets: view.batch_targets(),
        batch_sources: view.batch_sources(),
    }
}

/// The plan of one grouped level call on the device (T8, T9), built through `link` as
/// index data: `GroupedPlan::new` for the plan's `view` (octant or offset groups) with
/// `sources_bound` input columns.
fn build_plan<G: Copy + Into<usize> + Into<u32>, T: DeviceFloat>(
    link: &mut Link,
    view: &GroupedCsr<G>,
    sources_bound: usize,
    settings: &PlanSettings,
    tables: &Tables<T>,
    scratch: &mut TranslationScratch<T>,
) -> Result<GroupedPlan, FmmError> {
    link.build(DataKind::Indices, |d| {
        GroupedPlan::new(
            d,
            &grouped_arrays(view),
            sources_bound,
            settings,
            tables,
            scratch,
        )
    })
}

/// The plan of the grouped level call (`kind`, `pass`, `level`) of `plan` with
/// `settings`, over the device tables of its family (T8, T9; T12 builds candidates and
/// tuned choices the same way).
fn plan_level_call<T: DeviceFloat>(
    link: &mut Link,
    plan: &Plan,
    (kind, pass, level): (OperatorKind, Option<UpwardPass>, usize),
    settings: &PlanSettings,
    tables: &DeviceTables<T>,
    scratch: &mut TranslationScratch<T>,
) -> Result<GroupedPlan, FmmError> {
    let lists = plan.level(level);
    let (index, nlevels) = (plan.index(), plan.nlevels());
    let bound = |l: usize| if l < nlevels { index.len(l) } else { 0 };
    match (kind, pass) {
        (OperatorKind::M2m, Some(UpwardPass::Local)) => build_plan(
            link,
            lists.m2m_local(),
            bound(level + 1),
            settings,
            &tables.m2m,
            scratch,
        ),
        (OperatorKind::M2m, _) => build_plan(
            link,
            lists.m2m_global(),
            bound(level + 1),
            settings,
            &tables.m2m,
            scratch,
        ),
        (OperatorKind::L2l, _) => build_plan(
            link,
            lists.l2l(),
            bound(level.wrapping_sub(1)),
            settings,
            &tables.l2l,
            scratch,
        ),
        _ => build_plan(
            link,
            lists.v(),
            bound(level),
            settings,
            tables
                .m2l
                .as_ref()
                .expect("M2L on the device has its tables"),
            scratch,
        ),
    }
}

/// The report of every grouped level call with a pair, in the order local M2M, global
/// M2M, L2L, M2L and by level.
fn translation_reports<T: DeviceFloat>(translations: &Translations<T>) -> Vec<TranslationReport> {
    GROUPED
        .iter()
        .flat_map(|&(kind, pass)| {
            translations
                .calls(kind, pass)
                .iter()
                .enumerate()
                .filter(|(_, call)| !call.plan.is_empty())
                .map(move |(level, call)| TranslationReport {
                    kind,
                    pass,
                    level,
                    pairs: call.plan.len(),
                    gemm: call.plan.gemm(),
                    orientation: call.plan.orientation(),
                    budget: call.choice.budget,
                    chunks: call.plan.nchunks(),
                    gemm_columns: call.plan.gemm_columns(),
                    library_rejection: call.plan.library_rejection().map(str::to_owned),
                })
        })
        .collect()
}

/// A key's index on its level as `u32`.
fn key_index(key: morton::MortonKey) -> (u32, [u32; 3]) {
    let (level, index) = morton::decode(key);
    (level as u32, index.map(|c| c as u32))
}

/// The M2L family of the rotation tables as the device takes them (T10): the slices of
/// `ShiftTables`' accessors concatenated in index order, its storage unchanged, and the
/// geometry of each offset with `u32` indices. [`arrays`](Self::arrays) borrows them as
/// the `nd_fmm_kernels::rotation::RotationArrays` that `RotationTables::upload` takes;
/// public for tests and reports that launch the kernel themselves.
#[derive(Clone, Debug)]
pub struct RotationHostArrays<T> {
    p: usize,
    forward: Vec<T>,
    backward: Vec<T>,
    azimuth: Vec<T>,
    coaxial: Vec<T>,
    shifts: Vec<DeviceShift>,
}

impl<T: DeviceScalar> RotationHostArrays<T> {
    /// Copies the arrays of `tables` (the M2L family, `RotationTables::tables(Operator::M2l)`;
    /// any family converts the same way).
    ///
    /// # Panics
    ///
    /// If a table index does not fit in `u32`.
    pub fn new(tables: &ShiftTables<T>) -> Self {
        fn concat<T: Copy>(count: usize, piece: impl Fn(usize) -> Vec<T>) -> Vec<T> {
            (0..count).flat_map(piece).collect()
        }
        let index = |i: usize| u32::try_from(i).expect("a table index fits in u32");
        Self {
            p: tables.p(),
            forward: concat(tables.polar_count(), |i| tables.forward_blocks(i).to_vec()),
            backward: concat(tables.polar_count(), |i| tables.backward_blocks(i).to_vec()),
            azimuth: concat(tables.azimuth_count(), |i| {
                tables.azimuth_factors(i).to_vec()
            }),
            coaxial: concat(tables.distance_count(), |i| {
                tables.coaxial_factors(i).to_vec()
            }),
            shifts: (0..tables.count())
                .map(|d| {
                    let shift = tables.shift(d);
                    DeviceShift {
                        alignment: match shift.alignment {
                            Alignment::Up => DeviceAlignment::Up,
                            Alignment::Down => DeviceAlignment::Down,
                            Alignment::Rotated { polar, azimuth } => DeviceAlignment::Rotated {
                                polar: index(polar),
                                azimuth: index(azimuth),
                            },
                        },
                        distance: index(shift.distance),
                    }
                })
                .collect(),
        }
    }

    /// The arrays, borrowed.
    pub fn arrays(&self) -> RotationArrays<'_, T> {
        RotationArrays {
            p: self.p,
            forward: &self.forward,
            backward: &self.backward,
            azimuth: &self.azimuth,
            coaxial: &self.coaxial,
            shifts: &self.shifts,
        }
    }
}

impl<T: DeviceScalar> DeviceOperator<T> {
    /// Creates the operator on `device` for `plan` with `source_counts` and
    /// `target_counts` points per local leaf: checks that everything fits in the memory
    /// the device reports, then allocates the stores (zeroed), uploads the plan views,
    /// the geometry and the tables, and keeps `host` for the host-fallback kinds. The
    /// points follow with [`load_points`](Self::load_points).
    ///
    /// `host` must be built as the host path builds its operator (its tables, gradients,
    /// `max_leaf_points`, P2P kernel and pool), so that host-fallback kinds give the
    /// host path's bits. `tuner` holds the build's tuning key, cache and budget (T12): the
    /// level calls' GEMMs and the P2P layout come from its decisions, cache or static rule,
    /// and the GEMM decisions it lacks are tuned at the end of this call ([module
    /// documentation](self#autotune-t12-c47-cratetune)).
    ///
    /// # Errors
    ///
    /// [`SettingsError::DeviceMemory`] (as [`FmmError::InvalidSettings`]) if the
    /// buffers do not fit, before anything is allocated; [`FmmError::Device`] if an
    /// allocation or upload fails.
    ///
    /// # Panics
    ///
    /// If the plan has a ghost leaf or a ghost box (the device runs on one rank, §4.3),
    /// the counts do not have one entry per local leaf, or a view is malformed.
    pub fn new(
        host: LaplaceOperator<T>,
        device: Device,
        plan: &Plan,
        source_counts: &[usize],
        target_counts: &[usize],
        options: &DeviceOptions,
        tuner: Tuner,
    ) -> Result<Self, FmmError> {
        let index = plan.index();
        let nlevels = plan.nlevels();
        let leaves = index.leaves();
        assert!(
            leaves.ghosts().is_empty()
                && (0..nlevels).all(|l| index.kinds(l).iter().all(|kind| !kind.is_ghost())),
            "the device operator needs a plan without ghost leaves and ghost boxes (one rank)"
        );
        assert_eq!(
            source_counts.len(),
            leaves.nlocal(),
            "one source count per leaf"
        );
        assert_eq!(
            target_counts.len(),
            leaves.nlocal(),
            "one target count per leaf"
        );
        let p = host.p();
        let n = (p + 1) * (p + 1);
        let o = host.target_output_point_size();
        let strategy = host.tables().strategy();

        // The dense octant tables: the host's, or under `Rotation` built or loaded here.
        let mut cache_outcomes = Vec::new();
        let octants: Option<(M2mTables<T>, L2lTables<T>)> = match host.tables().octant_families() {
            Some(_) => None,
            None => Some(match &options.table_cache {
                Some(dir) => {
                    let cache = TableCache::new(dir);
                    let (m2m, m2m_outcome) = cache.load_or_build::<M2mTables<T>>(p);
                    let (l2l, l2l_outcome) = cache.load_or_build::<L2lTables<T>>(p);
                    cache_outcomes.push((TableKind::M2m, m2m_outcome));
                    cache_outcomes.push((TableKind::L2l, l2l_outcome));
                    (m2m, l2l)
                }
                None => (M2mTables::build(p), L2lTables::build(p)),
            }),
        };
        let (m2m, l2l) = match (&octants, host.tables().octant_families()) {
            (Some((m2m, l2l)), _) => (m2m.matrices(), l2l.matrices()),
            (None, Some((m2m, l2l))) => (m2m.matrices(), l2l.matrices()),
            (None, None) => unreachable!("one of the two holds the octant tables"),
        };
        let expanded = host.tables().classes_m2l().map(|classes| classes.expand());
        let m2l: Option<&MatrixSet<T>> = host
            .tables()
            .dense_m2l()
            .or(expanded.as_ref())
            .map(|tables| tables.matrices());
        let rotation = host.tables().rotation_m2l().map(RotationHostArrays::new);

        // The kinds with a device kernel run on the device unless `host_fallback` names
        // them: P2P from T6, P2M, P2L, L2P and M2P from T7, M2M and L2L from T8, M2L under
        // `Dense` and `Classes` from T9 and under `Rotation` from T10.
        let mut placement = [Placement::Host; 8];
        for kind in DEVICE_KINDS {
            if !options.host_fallback.contains(&kind) {
                placement[kind as usize] = Placement::Device;
            }
        }
        let on_device = |kind: OperatorKind| placement[kind as usize] == Placement::Device;
        // M2L on the device runs the dense tables or the rotation kernel, by strategy.
        let dense_m2l = on_device(OperatorKind::M2l) && m2l.is_some();
        let rotation = rotation.filter(|_| on_device(OperatorKind::M2l));
        debug_assert!(!on_device(OperatorKind::M2l) || dense_m2l != rotation.is_some());

        // The GEMM of each grouped level call (T8, T9, T12): the builder's `device_gemm`
        // and scratch budget where it names them; under `DeviceGemm::Auto` the choice of the
        // call's decision, taken earlier in this build (the M2L GEMM tuned with the
        // strategy) or found in the tuning cache, else the static rule. A family whose level
        // calls may take the library GEMM gets the library copy of its tables; with
        // decisions to tune, every candidate is provided for (the library copies where the
        // library applies, the scratch of its padded layout), and what no chosen plan needs
        // is released after tuning.
        let info = device.info().clone();
        let backend = info.backend;
        let gemm_layout = GemmLayout::default_for(&info, n);
        let tunable_gemm = options.gemm == DeviceGemm::Auto;
        let choice_of = |kind: OperatorKind, pairs: usize| -> GemmChoice {
            let fixed = static_gemm(
                &info,
                T::FLOAT,
                n,
                kind,
                options.gemm,
                options.scratch_budget,
            );
            if !tunable_gemm {
                return fixed;
            }
            let decision = Decision::Gemm {
                kind,
                bucket: bucket(pairs),
            };
            let mut choice = tuner
                .decided(decision)
                .or_else(|| tuner.cached(decision))
                .and_then(|c| c.gemm())
                .unwrap_or(fixed);
            if let Some(budget) = options.scratch_budget {
                choice.budget = budget;
            }
            choice
        };
        let grouped_on_device =
            |kind: OperatorKind| on_device(kind) && (kind != OperatorKind::M2l || dense_m2l);
        // Every grouped level call with a pair: (kind, pass, level, pairs, batch offsets).
        let calls: Vec<(CallId, usize, &[u32])> = GROUPED
            .iter()
            .filter(|(kind, _)| grouped_on_device(*kind))
            .flat_map(|&(kind, pass)| {
                plan.levels().iter().enumerate().map(move |(level, lists)| {
                    let (len, offsets) = match (kind, pass) {
                        (OperatorKind::M2m, Some(UpwardPass::Local)) => {
                            (lists.m2m_local().len(), lists.m2m_local().batch_offsets())
                        }
                        (OperatorKind::M2m, _) => {
                            (lists.m2m_global().len(), lists.m2m_global().batch_offsets())
                        }
                        (OperatorKind::L2l, _) => (lists.l2l().len(), lists.l2l().batch_offsets()),
                        _ => (lists.v().len(), lists.v().batch_offsets()),
                    };
                    ((kind, pass, level), len, offsets)
                })
            })
            .collect();
        // Whether a GEMM decision is left to tune: a level call large enough whose decision
        // the build has not taken and the cache does not hold.
        let tuning = tuner.enabled()
            && tunable_gemm
            && calls.iter().any(|&((kind, _, _), pairs, _)| {
                let decision = Decision::Gemm {
                    kind,
                    bucket: bucket(pairs),
                };
                pairs >= MIN_TUNED_PAIRS
                    && tuner.decided(decision).is_none()
                    && tuner.cached(decision).is_none()
            });
        let tuning_library = tuning && library_applies(backend, T::FLOAT, n);
        let library = |kind: OperatorKind| {
            grouped_on_device(kind)
                && (tuning_library
                    || calls
                        .iter()
                        .any(|c| c.0.0 == kind && c.1 > 0 && choice_of(kind, c.1).is_library()))
        };
        let mut tables_report = Vec::new();
        let mut table = |name: &'static str, set: &MatrixSet<T>, library: bool| {
            tables_report.push(DeviceTable {
                name,
                matrices: set.count(),
                bytes: Device::buffer_bytes::<T>(set.as_slice().len()),
            });
            if library {
                tables_report.push(DeviceTable {
                    name: match name {
                        "M2M" => "M2M (library copy)",
                        "L2L" => "L2L (library copy)",
                        _ => "M2L (library copy)",
                    },
                    matrices: set.count(),
                    bytes: Tables::<T>::bytes(n, set.count(), true)
                        - Tables::<T>::bytes(n, set.count(), false),
                });
            }
        };
        table("M2M", m2m, library(OperatorKind::M2m));
        table("L2L", l2l, library(OperatorKind::L2l));
        if let Some(m2l) = m2l.filter(|_| dense_m2l) {
            let name = if expanded.is_some() {
                "M2L (expanded from the classes)"
            } else {
                "M2L"
            };
            table(name, m2l, library(OperatorKind::M2l));
        }
        if let Some(rotation) = &rotation {
            tables_report.push(DeviceTable {
                name: "M2L (rotation)",
                matrices: rotation.shifts.len(),
                bytes: DeviceRotation::<T>::bytes(&rotation.arrays()),
            });
        }

        // The plans of the device M2M, L2L and M2L level calls: their index buffers and the
        // widest chunk, at most, before anything is allocated (T8, T9); with decisions to
        // tune, also the widest chunk of any candidate (T12).
        let (mut plan_bytes, mut scratch_columns) = (0u64, 0usize);
        let tuning_settings = GemmChoice {
            budget: options.scratch_budget.unwrap_or(DEFAULT_SCRATCH_BYTES),
            ..GemmChoice::library(T::FLOAT)
        }
        .settings(&info, n);
        for &((kind, _, _), pairs, batch_offsets) in &calls {
            let size =
                choice_of(kind, pairs)
                    .settings(&info, n)
                    .size(backend, T::FLOAT, batch_offsets);
            plan_bytes += size.bytes;
            scratch_columns = scratch_columns.max(size.columns);
            if tuning {
                let size = tuning_settings.size(backend, T::FLOAT, batch_offsets);
                scratch_columns = scratch_columns.max(size.columns);
            }
        }
        let scratch_bytes = TranslationScratch::<T>::bytes(scratch_columns * n);
        // The rows of each rotation M2L level call (T10).
        if rotation.is_some() {
            plan_bytes += plan
                .levels()
                .iter()
                .map(|lists| RotationPlan::bytes(lists.v().row_offsets()))
                .sum::<u64>();
        }

        // Every buffer, summed before anything is allocated (§4.6).
        let nboxes: usize = (0..nlevels).map(|l| index.len(l)).sum();
        let (nsources, ntargets): (usize, usize) =
            (source_counts.iter().sum(), target_counts.iter().sum());
        let nleaves = leaves.len();
        let needed = 2 * Device::buffer_bytes::<T>(nboxes * n)
            + Device::buffer_bytes::<T>(4 * nsources)
            + Device::buffer_bytes::<T>(3 * ntargets)
            + Device::buffer_bytes::<T>(o * ntargets)
            + Device::buffer_bytes::<T>(nsources)
            + view_bytes(plan)
            + Device::buffer_bytes::<u32>(nleaves + 1)
            + Device::buffer_bytes::<u32>(leaves.nlocal() + 1)
            + Device::buffer_bytes::<u32>(nsources)
            + Device::buffer_bytes::<u32>(3 * nboxes)
            + Device::buffer_bytes::<u32>(4 * nleaves)
            + tables_report.iter().map(|t| t.bytes).sum::<u64>()
            + plan_bytes
            + scratch_bytes;
        let available = device.available_memory();
        if let Some(limit) = available
            && needed > limit
        {
            return Err(SettingsError::DeviceMemory { needed, limit }.into());
        }

        let mut link = Link {
            device,
            traffic: TrafficByData::default(),
            error: None,
        };
        let level_offsets: Vec<usize> = std::iter::once(0)
            .chain((0..nlevels).scan(0, |total, l| {
                *total += index.len(l) * n;
                Some(*total)
            }))
            .collect();

        // An allocation zeroes its buffer by a kernel and transfers nothing, so the data
        // kind it is counted under stays without traffic.
        let mut alloc = |len: usize| link.build(DataKind::Points, |d| d.alloc::<T>(len));
        let stores = DeviceStores {
            multipoles: alloc(nboxes * n)?,
            locals: alloc(nboxes * n)?,
            sources: alloc(4 * nsources)?,
            target_input: alloc(3 * ntargets)?,
            target_output: alloc(o * ntargets)?,
            charges: alloc(nsources)?,
        };

        let mut levels = Vec::with_capacity(nlevels);
        for (level, lists) in plan.levels().iter().enumerate() {
            let boxes = |l: Option<usize>| l.filter(|&l| l < nlevels).map_or(0, |l| index.len(l));
            let csr = |link: &mut Link, view: &Csr, columns: usize| {
                link.build(DataKind::Indices, |d| {
                    IndexView::upload(d, view.row_offsets(), view.entries(), columns)
                })
            };
            let grouped = |link: &mut Link, view: &GroupedCsr<u8>, bound: usize| {
                link.build(DataKind::Indices, |d| {
                    GroupedView::upload(d, &grouped_arrays(view), bound)
                })
            };
            levels.push(LevelViews {
                p2m: csr(&mut link, lists.p2m(), nleaves)?,
                m2m_local: grouped(&mut link, lists.m2m_local(), boxes(Some(level + 1)))?,
                m2m_global: grouped(&mut link, lists.m2m_global(), boxes(Some(level + 1)))?,
                l2l: grouped(&mut link, lists.l2l(), boxes(level.checked_sub(1)))?,
                v: link.build(DataKind::Indices, |d| {
                    GroupedView::upload(d, &grouped_arrays(lists.v()), index.len(level))
                })?,
                x: csr(&mut link, lists.x(), nleaves)?,
                near: csr(&mut link, lists.near(), nleaves)?,
                w: csr(&mut link, lists.w(), boxes(Some(level + 1)))?,
                l2p: csr(&mut link, lists.l2p(), index.len(level))?,
            });
        }
        let source_offsets = point_offsets(source_counts).map_err(device_error)?;
        let target_offsets = point_offsets(target_counts).map_err(device_error)?;
        let slots = charge_slots(source_counts).map_err(device_error)?;
        let box_indices: Vec<Vec<[u32; 3]>> = (0..nlevels)
            .map(|l| index.keys(l).iter().map(|&key| key_index(key).1).collect())
            .collect();
        let leaf_indices: Vec<(u32, [u32; 3])> =
            leaves.keys().iter().map(|&key| key_index(key)).collect();
        let views = DeviceViews {
            levels,
            source_offsets: link.build(DataKind::Indices, |d| {
                PointOffsets::upload(d, &source_offsets)
            })?,
            target_offsets: link.build(DataKind::Indices, |d| {
                PointOffsets::upload(d, &target_offsets)
            })?,
            charge_slots: link.build(DataKind::Indices, |d| d.upload_indices(&slots))?,
            boxes: link.build(DataKind::Geometry, |d| {
                BoxCoordinates::upload(d, &box_indices)
            })?,
            leaves: link.build(DataKind::Geometry, |d| {
                LeafCoordinates::upload(d, &leaf_indices)
            })?,
        };

        let upload = |link: &mut Link, set: &MatrixSet<T>, library: bool| {
            link.build(DataKind::Tables, |d| {
                Tables::upload(d, set.as_slice(), n, library)
            })
        };
        let tables = DeviceTables {
            m2m: upload(&mut link, m2m, library(OperatorKind::M2m))?,
            l2l: upload(&mut link, l2l, library(OperatorKind::L2l))?,
            m2l: match m2l.filter(|_| dense_m2l) {
                Some(m2l) => Some(upload(&mut link, m2l, library(OperatorKind::M2l))?),
                None => None,
            },
            rotation: match &rotation {
                Some(rotation) => Some(link.build(DataKind::Tables, |d| {
                    DeviceRotation::upload(d, &rotation.arrays())
                })?),
                None => None,
            },
        };

        // The scratch and the plans of the device translations (T8, T9, T12); a library
        // choice probes its shapes here, so the GEMM is fixed before the first evaluation.
        let mut scratch = link.build(DataKind::Points, |d| {
            TranslationScratch::<T>::new(d, scratch_columns * n)
        })?;
        gemm_layout
            .check(link.device.info())
            .map_err(device_error)?;
        let mut level_calls: [Vec<LevelCall>; 4] = Default::default();
        for &((kind, pass, level), pairs, _) in &calls {
            let choice = choice_of(kind, pairs);
            let grouped_plan = plan_level_call(
                &mut link,
                plan,
                (kind, pass, level),
                &choice.settings(&info, n),
                &tables,
                &mut scratch,
            )?;
            let slot = GROUPED
                .iter()
                .position(|&g| g == (kind, pass))
                .expect("a grouped level call");
            level_calls[slot].push(LevelCall {
                plan: grouped_plan,
                choice,
            });
        }
        let [m2m_local, m2m_global, l2l_calls, m2l_calls] = level_calls;
        let mut translations = Translations {
            m2m_local,
            m2m_global,
            l2l: l2l_calls,
            m2l: m2l_calls,
            scratch,
            rotation: Vec::new(),
            rotation_layout: RotationLayout::default_for(link.device.info(), p),
        };

        // The rows of the rotation M2L level calls and the layout (T10).
        let rotation_layout = translations.rotation_layout;
        let mut rotation_plans = Vec::new();
        let mut rotation_report = Vec::new();
        if rotation.is_some() {
            rotation_layout
                .check(link.device.info(), p, T::FLOAT)
                .map_err(device_error)?;
            for (level, lists) in plan.levels().iter().enumerate() {
                let view = lists.v();
                let rotation_plan = link.build(DataKind::Indices, |d| {
                    RotationPlan::new(d, &grouped_arrays(view), index.len(level))
                })?;
                if !rotation_plan.is_empty() {
                    rotation_report.push(RotationReport {
                        level,
                        pairs: rotation_plan.len(),
                        rows: rotation_plan.active_rows(),
                    });
                }
                rotation_plans.push(rotation_plan);
            }
        }
        translations.rotation = rotation_plans;

        // P2P: the near-field pairs of each level's leaves and the mean points per leaf,
        // for the P2P decision (T12); the layout of the builder, or under
        // `DeviceP2pLayout::Auto` the cached choice or the static rule.
        let p2p_levels: Vec<(Range<usize>, usize)> = plan
            .levels()
            .iter()
            .enumerate()
            .map(|(level, lists)| {
                let range = leaves.local(level);
                let near = lists.near();
                let pairs = (0..near.nrows())
                    .map(|r| {
                        let sources: usize =
                            near.row(r).iter().map(|&s| source_counts[s as usize]).sum();
                        target_counts[range.start + r] * sources
                    })
                    .sum();
                (range, pairs)
            })
            .collect();
        let mean_points = (nsources + ntargets).div_ceil(2 * leaves.nlocal().max(1));
        let p2p_layout = match options.p2p_layout {
            DeviceP2pLayout::Auto => tuner
                .cached(Decision::P2p {
                    bucket: bucket(mean_points),
                })
                .and_then(|c| c.p2p())
                .unwrap_or_else(|| static_p2p(link.device.info())),
            layout => layout.resolve(link.device.info()),
        };
        p2p_layout
            .check(link.device.info(), T::FLOAT)
            .map_err(device_error)?;
        let leaf_layout = options.leaf_layout.resolve::<T>(link.device.info(), p);
        if LEAF_KINDS
            .iter()
            .any(|&kind| placement[kind as usize] == Placement::Device)
        {
            leaf_layout
                .check(link.device.info(), p, T::FLOAT)
                .map_err(device_error)?;
        }
        let lens: Vec<usize> = (0..nlevels).map(|l| index.len(l)).collect();
        let sizes = vec![n; nlevels];
        let mirrors = placement.contains(&Placement::Host).then(|| Mirrors {
            multipoles: LevelBuffers::new(&lens, &sizes),
            locals: LevelBuffers::new(&lens, &sizes),
        });
        debug_assert!(
            mirrors
                .as_ref()
                .is_none_or(|m| m.multipoles.offsets() == &level_offsets[..nlevels])
        );
        let stage_timing = match options.stage_timing {
            StageTiming::DeviceTimestamps if !link.device.times_on_device() => StageTiming::Enqueue,
            timing => timing,
        };
        let report = DeviceReport {
            info: link.device.info().clone(),
            placement,
            requested_fallback: options.host_fallback.clone(),
            strategy,
            tables: tables_report,
            cache_outcomes,
            cpu_units: (backend == BackendKind::Cpu).then(|| link.device.units_cap()),
            p2p_layout,
            leaf_layout,
            gemm_layout,
            translations: translation_reports(&translations),
            rotation_layout,
            rotations: rotation_report,
            scratch_bytes,
            memory_needed: needed,
            memory_available: available,
            stage_timing,
            tuning: None,
        };
        let mut operator = Self {
            host,
            link,
            placement,
            p2p_layout,
            leaf_layout,
            stores,
            level_offsets,
            views,
            tables,
            translations,
            mirrors,
            output: LeafStore::new(target_counts, o),
            report,
            build_counters: Counters::default(),
            build_traffic: TrafficByData::default(),
            evaluation_counters: Counters::default(),
            stage_windows: stage_timing == StageTiming::DeviceTimestamps,
            window: None,
            pending: Vec::with_capacity(DeviceStage::ALL.len()),
            stage_timings: None,
            options: options.clone(),
            tuner: Some(tuner),
            p2p_levels,
            mean_points,
        };
        operator.tune_translations(plan)?;
        Ok(operator)
    }

    /// Uploads the leaf-scaled source store (coordinates, with any charges) and target
    /// input, as `build` writes them in its step 8 (CONVENTIONS §3.13): two uploads, once
    /// per build. The charges of each evaluation overwrite their slots. Then takes the P2P
    /// decision (timed on the loaded points with a tuning cache that lacks it, T12), ends
    /// the tuning (storing what was tuned) and records the build's counters.
    ///
    /// # Errors
    ///
    /// [`FmmError::Device`] if an upload fails.
    ///
    /// # Panics
    ///
    /// If the stores do not have the layout of the operator's counts.
    pub fn load_points(
        &mut self,
        sources: &LeafStore<T>,
        target_input: &LeafStore<T>,
    ) -> Result<(), FmmError> {
        let stores = &mut self.stores;
        self.link.build(DataKind::Points, |d| {
            d.write(stores.sources.as_slice_mut(), sources.as_slice())
        })?;
        self.link.build(DataKind::Points, |d| {
            d.write(stores.target_input.as_slice_mut(), target_input.as_slice())
        })?;
        self.tune_p2p();
        if let Some(tuner) = self.tuner.take() {
            self.report.tuning = Some(tuner.finish());
        }
        self.finish_build();
        Ok(())
    }

    /// Records the counters of the build: from here on the counters belong to
    /// evaluations.
    fn finish_build(&mut self) {
        self.build_counters = self.link.device.counters();
        self.build_traffic = self.link.traffic;
    }

    /// The GEMM decisions of the grouped level calls (T12, [`crate::tune`]): under
    /// `DeviceGemm::Auto`, one decision per (kind, pair bucket) with a level call, M2L
    /// first, then M2M and L2L, larger buckets first, each timed on its bucket's largest
    /// level call (with a tuning cache that lacks it) or taken from the cache or the
    /// static rule; then the plans of the level calls whose choice changed are rebuilt, and
    /// what no chosen plan needs (library copies, scratch) is released.
    fn tune_translations(&mut self, plan: &Plan) -> Result<(), FmmError> {
        let Some(mut tuner) = self.tuner.take() else {
            return Ok(());
        };
        if self.options.gemm != DeviceGemm::Auto {
            self.tuner = Some(tuner);
            return Ok(());
        }
        tuner.start_phase();
        let info = self.link.device.info().clone();
        let n = self.n();
        let budget = self.options.scratch_budget;
        // The largest level call of each decision.
        let mut decisions: Vec<(Decision, CallId, usize)> = Vec::new();
        for &(kind, pass) in &GROUPED {
            for (level, call) in self.translations.calls(kind, pass).iter().enumerate() {
                let pairs = call.plan.len();
                if pairs == 0 {
                    continue;
                }
                let decision = Decision::Gemm {
                    kind,
                    bucket: bucket(pairs),
                };
                match decisions.iter_mut().find(|d| d.0 == decision) {
                    Some(entry) if entry.2 >= pairs => {}
                    Some(entry) => *entry = (decision, (kind, pass, level), pairs),
                    None => decisions.push((decision, (kind, pass, level), pairs)),
                }
            }
        }
        let rank = |kind: OperatorKind| match kind {
            OperatorKind::M2l => 0,
            OperatorKind::M2m => 1,
            _ => 2,
        };
        decisions.sort_by_key(|&(decision, (kind, _, _), _)| {
            let Decision::Gemm { bucket, .. } = decision else {
                unreachable!("GEMM decisions")
            };
            (rank(kind), std::cmp::Reverse(bucket))
        });
        for &(decision, call, pairs) in &decisions {
            let candidates: Vec<Candidate> =
                gemm_candidates(&info, T::FLOAT, n, call.0, budget.is_none())
                    .into_iter()
                    .map(|mut choice| {
                        if let Some(budget) = budget {
                            choice.budget = budget;
                        }
                        Candidate::Gemm(choice)
                    })
                    .collect();
            tuner.decide(
                decision,
                (Some(call.2), pairs),
                candidates,
                |candidate, deadline| {
                    let choice = candidate.gemm().expect("a GEMM candidate");
                    self.time_grouped(plan, call, &choice, deadline)
                },
            );
        }
        tuner.end_phase();
        // Every level call takes its decision's choice.
        for &(kind, pass) in &GROUPED {
            for level in 0..self.translations.calls(kind, pass).len() {
                let call = &self.translations.calls(kind, pass)[level];
                let pairs = call.plan.len();
                if pairs == 0 {
                    continue;
                }
                let decision = Decision::Gemm {
                    kind,
                    bucket: bucket(pairs),
                };
                let Some(choice) = tuner.decided(decision).and_then(|c| c.gemm()) else {
                    continue;
                };
                if choice == call.choice {
                    continue;
                }
                let Self {
                    link,
                    tables,
                    translations,
                    ..
                } = self;
                let rebuilt = plan_level_call(
                    link,
                    plan,
                    (kind, pass, level),
                    &choice.settings(&info, n),
                    tables,
                    &mut translations.scratch,
                )?;
                translations.calls_mut(kind, pass)[level] = LevelCall {
                    plan: rebuilt,
                    choice,
                };
            }
        }
        self.tuner = Some(tuner);
        self.release_unused()?;
        self.report.translations = translation_reports(&self.translations);
        Ok(())
    }

    /// The order n = (p + 1)² of the tables.
    fn n(&self) -> usize {
        let p = self.host.p();
        (p + 1) * (p + 1)
    }

    /// Times the grouped level call (`kind`, `pass`, `level`) with a plan built for
    /// `choice` (T12), on the operator's own buffers (their values do not matter: every
    /// evaluation zeroes them first).
    fn time_grouped(
        &mut self,
        plan: &Plan,
        (kind, pass, level): (OperatorKind, Option<UpwardPass>, usize),
        choice: &GemmChoice,
        deadline: Instant,
    ) -> Result<Duration, Rejection> {
        let info = self.link.device.info().clone();
        let n = self.n();
        if let GemmKind::HandWritten(layout) = choice.gemm {
            layout
                .check(&info)
                .map_err(|e| Rejection::Unregistered(e.to_string()))?;
        }
        let input = match kind {
            OperatorKind::M2m => level + 1,
            OperatorKind::L2l => level - 1,
            _ => level,
        };
        let (input, output) = (
            self.level_values(input..input + 1),
            self.level_values(level..level + 1),
        );
        let Self {
            link,
            views,
            tables,
            stores,
            translations,
            ..
        } = self;
        let family = match kind {
            OperatorKind::M2m => &tables.m2m,
            OperatorKind::L2l => &tables.l2l,
            _ => tables
                .m2l
                .as_ref()
                .expect("M2L on the device has its tables"),
        };
        if choice.is_library() && !family.has_library_copy() {
            return Err(Rejection::Unregistered(
                "the tables have no library copy".into(),
            ));
        }
        let candidate = plan_level_call(
            link,
            plan,
            (kind, pass, level),
            &choice.settings(&info, n),
            tables,
            &mut translations.scratch,
        )
        .map_err(|e| Rejection::Failed(e.to_string()))?;
        if choice.is_library() && candidate.gemm() != Gemm::Library {
            return Err(Rejection::Unregistered(format!(
                "the library does not take the level's shapes: {}",
                candidate.library_rejection().unwrap_or("not a candidate")
            )));
        }
        if choice.budget < DEFAULT_SCRATCH_BYTES && candidate.nchunks() <= 1 {
            return Err(Rejection::Unregistered(
                "one chunk under either budget: the default budget's plan".into(),
            ));
        }
        let level_views = &views.levels[level];
        let (view, accumulate) = match (kind, pass) {
            (OperatorKind::M2m, Some(UpwardPass::Local)) => {
                (&level_views.m2m_local, Accumulate::Rows)
            }
            (OperatorKind::M2m, _) => (&level_views.m2m_global, Accumulate::Rows),
            (OperatorKind::L2l, _) => (&level_views.l2l, Accumulate::Scatter),
            _ => (&level_views.v, Accumulate::Rows),
        };
        let scratch = &mut translations.scratch;
        time_launches(&mut link.device, deadline, |d| {
            let operands = match kind {
                OperatorKind::M2m => Operands::Shared {
                    buffer: &mut stores.multipoles,
                    input: input.clone(),
                    output: output.clone(),
                },
                OperatorKind::L2l => Operands::Shared {
                    buffer: &mut stores.locals,
                    input: input.clone(),
                    output: output.clone(),
                },
                _ => Operands::Separate {
                    input: stores.multipoles.slice(input.clone()),
                    output: stores.locals.slice_mut(output.clone()),
                },
            };
            grouped(d, &candidate, view, accumulate, family, operands, scratch)
        })
        .map_err(Rejection::from)
    }

    /// Releases what no chosen plan needs after tuning (T12): the library copy of a family
    /// whose level calls all run the hand-written kernel, and the scratch beyond the widest
    /// chunk of the chosen plans; the report's tables, scratch and memory follow.
    fn release_unused(&mut self) -> Result<(), FmmError> {
        let n = self.n();
        let uses_library = |translations: &Translations<T>, kinds: &[OperatorKind]| {
            GROUPED.iter().any(|&(kind, pass)| {
                kinds.contains(&kind)
                    && translations
                        .calls(kind, pass)
                        .iter()
                        .any(|c| c.plan.gemm() == Gemm::Library)
            })
        };
        let mut freed = 0u64;
        for (kinds, name) in [
            (&[OperatorKind::M2m][..], "M2M (library copy)"),
            (&[OperatorKind::L2l][..], "L2L (library copy)"),
            (&[OperatorKind::M2l][..], "M2L (library copy)"),
        ] {
            if uses_library(&self.translations, kinds) {
                continue;
            }
            let family = match kinds[0] {
                OperatorKind::M2m => Some(&mut self.tables.m2m),
                OperatorKind::L2l => Some(&mut self.tables.l2l),
                _ => self.tables.m2l.as_mut(),
            };
            if let Some(family) = family
                && family.has_library_copy()
            {
                freed += family.release_library_copy();
                self.report.tables.retain(|t| t.name != name);
            }
        }
        let columns = GROUPED
            .iter()
            .flat_map(|&(kind, pass)| self.translations.calls(kind, pass))
            .map(|c| c.plan.columns())
            .max()
            .unwrap_or(0);
        if columns * n < self.translations.scratch.len() {
            let old = TranslationScratch::<T>::bytes(self.translations.scratch.len());
            self.translations.scratch = self.link.build(DataKind::Points, |d| {
                TranslationScratch::<T>::new(d, columns * n)
            })?;
            let new = TranslationScratch::<T>::bytes(columns * n);
            freed += old - new;
            self.report.scratch_bytes = new;
        }
        self.report.memory_needed = self.report.memory_needed.saturating_sub(freed);
        Ok(())
    }

    /// The P2P decision (T12, [`crate::tune`]): under `DeviceP2pLayout::Auto` with P2P on
    /// the device, timed on the level with the most near-field pairs (with a tuning cache
    /// that lacks it), after the points are loaded; else from the cache or the static rule.
    fn tune_p2p(&mut self) {
        let Some(mut tuner) = self.tuner.take() else {
            return;
        };
        let best = self
            .p2p_levels
            .iter()
            .enumerate()
            .filter(|(_, (_, pairs))| *pairs > 0)
            .max_by_key(|(_, (_, pairs))| *pairs)
            .map(|(level, (leaves, pairs))| (level, leaves.clone(), *pairs));
        if self.placement(OperatorKind::P2p) == Placement::Device
            && self.options.p2p_layout == DeviceP2pLayout::Auto
            && let Some((level, leaves, pairs)) = best
        {
            tuner.start_phase();
            let info = self.link.device.info().clone();
            let candidates = p2p_candidates(&info)
                .into_iter()
                .map(Candidate::P2p)
                .collect();
            let decision = Decision::P2p {
                bucket: bucket(self.mean_points),
            };
            let choice = tuner.decide(
                decision,
                (Some(level), pairs),
                candidates,
                |candidate, deadline| {
                    let layout = candidate.p2p().expect("a P2P candidate");
                    self.time_p2p(level, &leaves, layout, deadline)
                },
            );
            tuner.end_phase();
            if let Some(layout) = choice.p2p() {
                self.p2p_layout = layout;
                self.report.p2p_layout = layout;
            }
        }
        self.tuner = Some(tuner);
    }

    /// Times the P2P level call of `level` (its local leaves `leaves`) in `layout` (T12),
    /// on the operator's loaded points; the target output it adds into is zeroed by every
    /// evaluation.
    fn time_p2p(
        &mut self,
        level: usize,
        leaves: &Range<usize>,
        layout: P2pLayout,
        deadline: Instant,
    ) -> Result<Duration, Rejection> {
        layout
            .check(self.link.device.info(), T::FLOAT)
            .map_err(|e| Rejection::Unregistered(e.to_string()))?;
        let gradients = self.host.gradients();
        let Self {
            link,
            views,
            stores,
            ..
        } = self;
        let inputs = P2pInputs {
            near: &views.levels[level].near,
            first_leaf: leaves.start,
            leaves: &views.leaves,
            source_offsets: &views.source_offsets,
            sources: stores.sources.as_slice(),
            target_offsets: &views.target_offsets,
            target_input: stores.target_input.as_slice(),
        };
        let output = &mut stores.target_output;
        time_launches(&mut link.device, deadline, |d| {
            nd_fmm_kernels::p2p::p2p(d, layout, gradients, &inputs, output.as_slice_mut())
        })
        .map_err(Rejection::from)
    }

    /// Starts an evaluation, after the evaluator's `reset`: zeroes the multipoles, the
    /// locals and the target output on the device (+0.0, the bits of `reset`), uploads
    /// `charges` (one per source, in leaf order: the k-th source of the source store in
    /// leaf order has the k-th charge) and scatters them into their slots of the source
    /// store. Four launches and one upload, no sync. Resets the evaluation counters. With
    /// [`StageTiming::DeviceTimestamps`] it runs in the timing window of
    /// [`DeviceStage::Load`].
    ///
    /// A failure is kept and returned by [`read_output`](Self::read_output).
    ///
    /// # Panics
    ///
    /// If `charges` does not hold one value per source.
    pub fn begin_evaluation(&mut self, charges: &[T]) {
        assert_eq!(
            charges.len(),
            self.stores.charges.len(),
            "one charge per source"
        );
        self.link.device.reset_counters();
        self.link.traffic = TrafficByData::default();
        self.pending.clear();
        self.stage_timings = None;
        self.open_stage();
        let (stores, views) = (&mut self.stores, &self.views);
        // The zero kernels transfer nothing; they are counted with the output they clear.
        for buffer in [
            &mut stores.multipoles,
            &mut stores.locals,
            &mut stores.target_output,
        ] {
            self.link
                .run(DataKind::Output, |d| zero(d, buffer.as_slice_mut()));
        }
        self.link.run(DataKind::Charges, |d| {
            d.write(stores.charges.as_slice_mut(), charges)
        });
        self.link.run(DataKind::Charges, |d| {
            scatter_values(
                d,
                stores.charges.as_slice(),
                views.charge_slots.as_slice(),
                stores.sources.as_slice_mut(),
            )
        });
        self.close_stage(DeviceStage::Load);
    }

    /// Opens the timing window of a stage, with [`StageTiming::DeviceTimestamps`]: no
    /// sync on a device that times on itself (device-path.md §8.3). `Fmm` calls it before
    /// each stage with device work.
    pub fn open_stage(&mut self) {
        if self.stage_windows {
            self.window = self.link.run(DataKind::Output, Device::open_window);
        }
    }

    /// Closes the timing window of `stage`, opened by [`open_stage`](Self::open_stage);
    /// its time is read by [`read_output`](Self::read_output). A window the device did not
    /// time itself (it waited for the stream: a wgpu stream that found the device's
    /// timestamp queries spent) turns the windows off for every later evaluation.
    pub fn close_stage(&mut self, stage: DeviceStage) {
        let Some(window) = self.window.take() else {
            return;
        };
        if let Some(time) = self.link.run(DataKind::Output, |d| d.close_window(window)) {
            self.stage_windows &= time.on_device();
            self.pending.push((stage, time));
        }
    }

    /// The device times of the stages of the last evaluation, with
    /// [`StageTiming::DeviceTimestamps`] and if every window was timed on the device.
    pub fn stage_timings(&self) -> Option<DeviceStageTimings> {
        self.stage_timings
    }

    /// Waits for every queued launch and transfer: one sync. `Fmm` calls it after every
    /// stage with `synchronous_stages`. A failure is kept for
    /// [`read_output`](Self::read_output).
    pub fn sync(&mut self) {
        self.link.run(DataKind::Output, Device::sync);
    }

    /// Ends an evaluation: downloads the target output (one download, one sync) and
    /// returns it, in the evaluator's layout, unscaled.
    ///
    /// # Errors
    ///
    /// The first failure of the evaluation, if any device operation failed (a launch
    /// error surfaces at this download, §12).
    pub fn read_output(&mut self) -> Result<&LeafStore<T>, KernelError> {
        let (stores, output) = (&self.stores, &mut self.output);
        self.link.run(DataKind::Output, |d| {
            d.download(stores.target_output.as_slice(), output.as_mut_slice())
        });
        // The stage windows, read after the download: the stream is idle. A window
        // without device work measures nothing and times zero.
        if self.link.error.is_none() && !self.pending.is_empty() {
            let mut timings = DeviceStageTimings::default();
            let mut on_device = true;
            for (stage, time) in self.pending.drain(..) {
                on_device &= time.on_device();
                timings.set(stage, time.resolve().unwrap_or_default());
            }
            self.stage_timings = on_device.then_some(timings);
        }
        self.evaluation_counters = self.link.device.counters();
        match &self.link.error {
            Some(error) => Err(error.clone()),
            None => Ok(&self.output),
        }
    }

    /// Downloads the multipoles and the locals of every level, in the `LevelBuffers`
    /// layout (box t of level l at the level's offset plus t (p + 1)²), for tests and
    /// reports: two downloads, counted toward the evaluation counters until the next
    /// [`begin_evaluation`](Self::begin_evaluation).
    ///
    /// # Errors
    ///
    /// As `Device::download`.
    pub fn download_expansions(&mut self) -> Result<(Vec<T>, Vec<T>), KernelError> {
        let mut multipoles = vec![T::default(); self.stores.multipoles.len()];
        let mut locals = vec![T::default(); self.stores.locals.len()];
        let device = &mut self.link.device;
        device.download(self.stores.multipoles.as_slice(), &mut multipoles)?;
        device.download(self.stores.locals.as_slice(), &mut locals)?;
        Ok((multipoles, locals))
    }

    /// The host operator of the fallback kinds.
    pub fn host(&self) -> &LaplaceOperator<T> {
        &self.host
    }

    /// The host operator, mutably (for `set_serial`).
    pub fn host_mut(&mut self) -> &mut LaplaceOperator<T> {
        &mut self.host
    }

    /// The device.
    pub fn device(&self) -> &Device {
        &self.link.device
    }

    /// The views on the device.
    pub fn views(&self) -> &DeviceViews {
        &self.views
    }

    /// Downloads every view, for tests and reports. Its transfers count toward the
    /// evaluation counters until the next [`begin_evaluation`](Self::begin_evaluation).
    ///
    /// # Errors
    ///
    /// As `Device::download`.
    pub fn download_views(&mut self) -> Result<ViewsImage, KernelError> {
        self.views.download(&mut self.link.device)
    }

    /// What the operator runs on and how.
    pub fn report(&self) -> &DeviceReport {
        &self.report
    }

    /// The transfers, launches and syncs of the build and of the last evaluation.
    pub fn counters(&self) -> DeviceCounters {
        DeviceCounters {
            build: self.build_counters,
            evaluation: self.evaluation_counters,
            build_traffic: self.build_traffic,
            evaluation_traffic: self.link.traffic,
        }
    }

    /// Where `kind` runs.
    pub fn placement(&self, kind: OperatorKind) -> Placement {
        self.placement[kind as usize]
    }

    /// The value range of `levels` in the multipole and local buffers.
    fn level_values(&self, levels: Range<usize>) -> Range<usize> {
        self.level_offsets[levels.start]..self.level_offsets[levels.end]
    }

    /// The value range of the local leaves `leaves` in the target output.
    fn output_values(&self, leaves: &Range<usize>) -> Range<usize> {
        let offsets = self.output.point_offsets();
        let o = self.output.point_size();
        offsets[leaves.start] * o..offsets[leaves.end] * o
    }

    /// Downloads `levels` of `store` into its mirror (nothing for an empty range).
    fn fetch_levels(&mut self, store: Level, levels: Range<usize>) {
        let values = self.level_values(levels);
        if values.is_empty() {
            return;
        }
        let (stores, mirrors) = (&self.stores, self.mirrors.as_mut().expect(MIRRORS));
        let (buffer, mirror, data) = match store {
            Level::Multipoles => (
                &stores.multipoles,
                &mut mirrors.multipoles,
                DataKind::FallbackMultipoles,
            ),
            Level::Locals => (
                &stores.locals,
                &mut mirrors.locals,
                DataKind::FallbackLocals,
            ),
        };
        self_fetch(
            &mut self.link,
            data,
            buffer.slice(values.clone()),
            &mut mirror.as_mut_slice()[values],
        );
    }

    /// Uploads `level` of `store` from its mirror.
    fn send_level(&mut self, store: Level, level: usize) {
        let values = self.level_values(level..level + 1);
        if values.is_empty() {
            return;
        }
        let (stores, mirrors) = (&mut self.stores, self.mirrors.as_ref().expect(MIRRORS));
        let (buffer, mirror, data) = match store {
            Level::Multipoles => (
                &mut stores.multipoles,
                &mirrors.multipoles,
                DataKind::FallbackMultipoles,
            ),
            Level::Locals => (
                &mut stores.locals,
                &mirrors.locals,
                DataKind::FallbackLocals,
            ),
        };
        self_send(
            &mut self.link,
            data,
            buffer.slice_mut(values.clone()),
            &mirror.as_slice()[values],
        );
    }

    /// Downloads the target output of `leaves` into the host output.
    fn fetch_output(&mut self, leaves: &Range<usize>) {
        let values = self.output_values(leaves);
        if !values.is_empty() {
            self_fetch(
                &mut self.link,
                DataKind::FallbackTargetOutput,
                self.stores.target_output.slice(values.clone()),
                &mut self.output.as_mut_slice()[values],
            );
        }
    }

    /// Uploads the target output of `leaves` from the host output.
    fn send_output(&mut self, leaves: &Range<usize>) {
        let values = self.output_values(leaves);
        if !values.is_empty() {
            self_send(
                &mut self.link,
                DataKind::FallbackTargetOutput,
                self.stores.target_output.slice_mut(values.clone()),
                &self.output.as_slice()[values],
            );
        }
    }

    /// True if a host-fallback call can go ahead: no earlier failure.
    fn healthy(&self) -> bool {
        self.link.error.is_none()
    }
}

/// The panic message of a host-fallback call without mirrors.
const MIRRORS: &str = "host-fallback kinds have host mirrors";

/// [`Link::run`] for a download.
fn self_fetch<E: DeviceElement>(
    link: &mut Link,
    data: DataKind,
    slice: DeviceSlice<'_, E>,
    out: &mut [E],
) {
    link.run(data, |d| d.download(slice, out));
}

/// [`Link::run`] for an upload.
fn self_send<E: DeviceElement>(
    link: &mut Link,
    data: DataKind,
    slice: DeviceSliceMut<'_, E>,
    values: &[E],
) {
    link.run(data, |d| d.write(slice, values));
}

/// A kernel error at build as an [`FmmError`].
fn device_error(error: KernelError) -> FmmError {
    FmmError::Device(error.to_string())
}

impl<T: DeviceScalar> FmmSizes for DeviceOperator<T> {
    type Value = T;

    fn multipole_size(&self, level: usize) -> usize {
        self.host.multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.host.local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.host.source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.host.target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.host.target_output_point_size()
    }
}

impl<T: DeviceScalar> DeviceOperator<T> {
    /// The plan and view of the M2M (of `pass`) or L2L level call of `level`.
    fn translation(
        &self,
        kind: OperatorKind,
        level: usize,
        pass: Option<UpwardPass>,
    ) -> (&GroupedPlan, &GroupedView) {
        let (translations, views) = (&self.translations, &self.views.levels[level]);
        match (kind, pass) {
            (OperatorKind::M2m, Some(UpwardPass::Local)) => {
                (&translations.m2m_local[level].plan, &views.m2m_local)
            }
            (OperatorKind::M2m, Some(UpwardPass::Global)) => {
                (&translations.m2m_global[level].plan, &views.m2m_global)
            }
            (OperatorKind::L2l, None) => (&translations.l2l[level].plan, &views.l2l),
            _ => unreachable!("M2M with a pass or L2L"),
        }
    }

    /// M2M (of `pass`, children on `level + 1` into parents on `level`) or L2L (parents on
    /// `level − 1` into children on `level`) on the device: the grouped translation of
    /// the level's view with the dense octant tables, three launches per chunk (T8).
    fn translate_on_device(&mut self, kind: OperatorKind, level: usize, pass: Option<UpwardPass>) {
        let (input, accumulate) = match kind {
            OperatorKind::M2m => (level + 1, Accumulate::Rows),
            _ => (level - 1, Accumulate::Scatter),
        };
        let input = self.level_values(input..input + 1);
        let output = self.level_values(level..level + 1);
        let Self {
            translations,
            views,
            tables,
            stores,
            link,
            ..
        } = self;
        let views = &views.levels[level];
        let (plan, view, table, buffer) = match (kind, pass) {
            (OperatorKind::M2m, Some(UpwardPass::Local)) => (
                &translations.m2m_local[level].plan,
                &views.m2m_local,
                &tables.m2m,
                &mut stores.multipoles,
            ),
            (OperatorKind::M2m, Some(UpwardPass::Global)) => (
                &translations.m2m_global[level].plan,
                &views.m2m_global,
                &tables.m2m,
                &mut stores.multipoles,
            ),
            (OperatorKind::L2l, None) => (
                &translations.l2l[level].plan,
                &views.l2l,
                &tables.l2l,
                &mut stores.locals,
            ),
            _ => unreachable!("M2M with a pass or L2L"),
        };
        let scratch = &mut translations.scratch;
        link.run(DataKind::Output, |d| {
            grouped(
                d,
                plan,
                view,
                accumulate,
                table,
                Operands::Shared {
                    buffer,
                    input,
                    output,
                },
                scratch,
            )
        });
    }

    /// M2L of `level` on the device, from the multipoles into the locals of the level: under
    /// `Rotation` the rotation kernel over the level's V view, one launch (T10,
    /// device-path.md §6.6); otherwise the grouped translation with the 316 dense offset
    /// tables, three launches per chunk (T9, device-path.md §6.4).
    fn m2l_on_device(&mut self, level: usize) {
        let values = self.level_values(level..level + 1);
        let Self {
            translations,
            views,
            tables,
            stores,
            link,
            ..
        } = self;
        if let Some(rotation) = &tables.rotation {
            let (plan, view) = (&translations.rotation[level], &views.levels[level].v);
            let layout = translations.rotation_layout;
            link.run(DataKind::Output, |d| {
                nd_fmm_kernels::rotation::m2l(
                    d,
                    layout,
                    plan,
                    view,
                    rotation,
                    stores.multipoles.slice(values.clone()),
                    stores.locals.slice_mut(values),
                )
            });
            return;
        }
        let table = tables
            .m2l
            .as_ref()
            .expect("M2L on the device has its tables");
        let (plan, view) = (&translations.m2l[level].plan, &views.levels[level].v);
        let scratch = &mut translations.scratch;
        link.run(DataKind::Output, |d| {
            grouped(
                d,
                plan,
                view,
                Accumulate::Rows,
                table,
                Operands::Separate {
                    input: stores.multipoles.slice(values.clone()),
                    output: stores.locals.slice_mut(values),
                },
                scratch,
            )
        });
    }

    /// P2M (`irregular` false) or P2L of `level` on the device: one launch from the
    /// level's view, adding into the multipoles or locals of the level (T7).
    fn expand_on_device(&mut self, level: usize, irregular: bool) {
        let values = self.level_values(level..level + 1);
        let views = &self.views;
        let inputs = SourceInputs {
            view: if irregular {
                &views.levels[level].x
            } else {
                &views.levels[level].p2m
            },
            level,
            boxes: &views.boxes,
            leaves: &views.leaves,
            source_offsets: &views.source_offsets,
            sources: self.stores.sources.as_slice(),
        };
        let (layout, p) = (self.leaf_layout, self.host.p());
        let out = if irregular {
            &mut self.stores.locals
        } else {
            &mut self.stores.multipoles
        };
        self.link.run(DataKind::Output, |d| {
            let out = out.slice_mut(values);
            if irregular {
                nd_fmm_kernels::leaf::p2l(d, layout, p, &inputs, out)
            } else {
                nd_fmm_kernels::leaf::p2m(d, layout, p, &inputs, out)
            }
        });
    }

    /// L2P (`irregular` false) or M2P of the leaves `leaves` of `level` on the device: one
    /// launch from the level's view, the locals of the level or the multipoles of the
    /// level below, adding into the target output (T7).
    fn evaluate_on_device(&mut self, level: usize, leaves: &Range<usize>, irregular: bool) {
        let entry_level = level + usize::from(irregular);
        let values = self.level_values(entry_level..entry_level + 1);
        let views = &self.views;
        let view = if irregular {
            &views.levels[level].w
        } else {
            &views.levels[level].l2p
        };
        debug_assert_eq!(view.nrows(), leaves.len());
        let inputs = TargetInputs {
            view,
            level,
            first_leaf: leaves.start,
            boxes: &views.boxes,
            leaves: &views.leaves,
            target_offsets: &views.target_offsets,
            target_input: self.stores.target_input.as_slice(),
        };
        let (layout, p, gradients) = (self.leaf_layout, self.host.p(), self.host.gradients());
        let coefficients = if irregular {
            self.stores.multipoles.slice(values)
        } else {
            self.stores.locals.slice(values)
        };
        let output = &mut self.stores.target_output;
        self.link.run(DataKind::Output, |d| {
            let output = output.as_slice_mut();
            if irregular {
                nd_fmm_kernels::leaf::m2p(d, layout, p, gradients, &inputs, coefficients, output)
            } else {
                nd_fmm_kernels::leaf::l2p(d, layout, p, gradients, &inputs, coefficients, output)
            }
        });
    }
}

/// Every kind on the host fallback ([module documentation](self#host-fallback-7)):
/// download, the host operator's own method on the mirrors, upload; every kind with a
/// device kernel on the device unless it falls back ([module
/// documentation](self#p2p-on-the-device-t6), [the leaf
/// operators](self#the-leaf-operators-on-the-device-t7), [M2M and
/// L2L](self#m2m-and-l2l-on-the-device-t8), [dense M2L](self#dense-m2l-on-the-device-t9),
/// [rotation M2L](self#rotation-m2l-on-the-device-t10)).
impl<T: DeviceScalar> FmmOperator for DeviceOperator<T> {
    fn p2m(&mut self, batch: P2m<'_, T>) {
        let level = batch.level;
        if batch.leaves.is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::P2m) == Placement::Device {
            self.expand_on_device(level, false);
            return;
        }
        self.fetch_levels(Level::Multipoles, level..level + 1);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_mut().expect(MIRRORS);
        self.host.p2m(P2m {
            multipoles: mirrors.multipoles.level_mut(level),
            ..batch
        });
        self.send_level(Level::Multipoles, level);
    }

    fn m2m(&mut self, batch: M2m<'_, T>) {
        let level = batch.level;
        if batch.children.is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::M2m) == Placement::Device {
            debug_assert_eq!(
                batch.children.len(),
                self.translation(OperatorKind::M2m, level, Some(batch.pass))
                    .0
                    .len()
            );
            self.translate_on_device(OperatorKind::M2m, level, Some(batch.pass));
            return;
        }
        self.fetch_levels(Level::Multipoles, level..level + 2);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_mut().expect(MIRRORS);
        let (multipoles, child_multipoles) = mirrors.multipoles.parent_child_mut(level);
        self.host.m2m(M2m {
            multipoles,
            child_multipoles,
            ..batch
        });
        self.send_level(Level::Multipoles, level);
    }

    fn m2l(&mut self, batch: M2l<'_, T>) {
        let level = batch.level;
        if batch.pairs.is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::M2l) == Placement::Device {
            debug_assert_eq!(
                batch.pairs.len(),
                match self.tables.rotation {
                    Some(_) => self.translations.rotation[level].len(),
                    None => self.translations.m2l[level].plan.len(),
                }
            );
            self.m2l_on_device(level);
            return;
        }
        self.fetch_levels(Level::Multipoles, level..level + 1);
        self.fetch_levels(Level::Locals, level..level + 1);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_mut().expect(MIRRORS);
        self.host.m2l(M2l {
            multipoles: mirrors.multipoles.level(level),
            locals: mirrors.locals.level_mut(level),
            ..batch
        });
        self.send_level(Level::Locals, level);
    }

    fn p2l(&mut self, batch: P2l<'_, T>) {
        let level = batch.level;
        if batch.x.is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::P2l) == Placement::Device {
            self.expand_on_device(level, true);
            return;
        }
        self.fetch_levels(Level::Locals, level..level + 1);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_mut().expect(MIRRORS);
        self.host.p2l(P2l {
            locals: mirrors.locals.level_mut(level),
            ..batch
        });
        self.send_level(Level::Locals, level);
    }

    fn l2l(&mut self, batch: L2l<'_, T>) {
        let level = batch.level;
        if batch.parents.is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::L2l) == Placement::Device {
            debug_assert_eq!(
                batch.parents.len(),
                self.translation(OperatorKind::L2l, level, None).0.len()
            );
            self.translate_on_device(OperatorKind::L2l, level, None);
            return;
        }
        self.fetch_levels(Level::Locals, level - 1..level + 1);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_mut().expect(MIRRORS);
        let (locals, parent_locals) = mirrors.locals.child_parent_mut(level - 1);
        self.host.l2l(L2l {
            locals,
            parent_locals,
            ..batch
        });
        self.send_level(Level::Locals, level);
    }

    fn l2p(&mut self, batch: L2p<'_, T>) {
        let (level, leaves) = (batch.level, batch.leaves.clone());
        if batch.boxes.is_empty() || self.output_values(&leaves).is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::L2p) == Placement::Device {
            self.evaluate_on_device(level, &leaves, false);
            return;
        }
        self.fetch_levels(Level::Locals, level..level + 1);
        self.fetch_output(&leaves);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_ref().expect(MIRRORS);
        self.host.l2p(L2p {
            locals: mirrors.locals.level(level),
            target_output: self.output.range_mut(leaves.clone()),
            ..batch
        });
        self.send_output(&leaves);
    }

    fn m2p(&mut self, batch: M2p<'_, T>) {
        let (level, leaves) = (batch.level, batch.leaves.clone());
        if batch.w.is_empty() || self.output_values(&leaves).is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::M2p) == Placement::Device {
            // A non-empty W list names boxes of level + 1, which then exists.
            self.evaluate_on_device(level, &leaves, true);
            return;
        }
        // A non-empty W list names boxes of level + 1, which then exists.
        self.fetch_levels(Level::Multipoles, level + 1..level + 2);
        self.fetch_output(&leaves);
        if !self.healthy() {
            return;
        }
        let mirrors = self.mirrors.as_ref().expect(MIRRORS);
        self.host.m2p(M2p {
            multipoles: mirrors.multipoles.level(level + 1),
            target_output: self.output.range_mut(leaves.clone()),
            ..batch
        });
        self.send_output(&leaves);
    }

    fn p2p(&mut self, batch: P2p<'_, T>) {
        let leaves = batch.leaves.clone();
        if batch.near.is_empty() || self.output_values(&leaves).is_empty() || !self.healthy() {
            return;
        }
        if self.placement(OperatorKind::P2p) == Placement::Device {
            // One launch: the level's near view on the device, rows `leaves` (T6).
            let views = &self.views;
            let near = &views.levels[batch.level].near;
            debug_assert_eq!(near.nrows(), leaves.len());
            let inputs = P2pInputs {
                near,
                first_leaf: leaves.start,
                leaves: &views.leaves,
                source_offsets: &views.source_offsets,
                sources: self.stores.sources.as_slice(),
                target_offsets: &views.target_offsets,
                target_input: self.stores.target_input.as_slice(),
            };
            let (layout, gradients) = (self.p2p_layout, self.host.gradients());
            let output = &mut self.stores.target_output;
            self.link.run(DataKind::Output, |d| {
                nd_fmm_kernels::p2p::p2p(d, layout, gradients, &inputs, output.as_slice_mut())
            });
            return;
        }
        self.fetch_output(&leaves);
        if !self.healthy() {
            return;
        }
        self.host.p2p(P2p {
            target_output: self.output.range_mut(leaves.clone()),
            ..batch
        });
        self.send_output(&leaves);
    }
}

/// What [`Fmm`](crate::fmm::Fmm) needs of a device operator, for either precision: the
/// operator is created for the concrete f32 or f64 ([`driver`]) and held as
/// `Box<dyn DeviceDriver<T>>`, so `Fmm` keeps its bounds.
pub(crate) trait DeviceDriver<T: SimdScalar>: FmmOperator<Value = T> + Send {
    fn host(&self) -> &LaplaceOperator<T>;
    fn host_mut(&mut self) -> &mut LaplaceOperator<T>;
    fn load_points(
        &mut self,
        sources: &LeafStore<T>,
        target_input: &LeafStore<T>,
    ) -> Result<(), FmmError>;
    fn begin_evaluation(&mut self, charges: &[T]);
    fn open_stage(&mut self);
    fn close_stage(&mut self, stage: DeviceStage);
    fn stage_timings(&self) -> Option<DeviceStageTimings>;
    fn sync(&mut self);
    fn read_output(&mut self) -> Result<&LeafStore<T>, KernelError>;
    fn report(&self) -> &DeviceReport;
    fn counters(&self) -> DeviceCounters;
    fn download_views(&mut self) -> Result<ViewsImage, KernelError>;
    fn download_expansions(&mut self) -> Result<(Vec<T>, Vec<T>), KernelError>;
}

impl<T: DeviceScalar> DeviceDriver<T> for DeviceOperator<T> {
    fn host(&self) -> &LaplaceOperator<T> {
        DeviceOperator::host(self)
    }
    fn host_mut(&mut self) -> &mut LaplaceOperator<T> {
        DeviceOperator::host_mut(self)
    }
    fn load_points(
        &mut self,
        sources: &LeafStore<T>,
        target_input: &LeafStore<T>,
    ) -> Result<(), FmmError> {
        DeviceOperator::load_points(self, sources, target_input)
    }
    fn begin_evaluation(&mut self, charges: &[T]) {
        DeviceOperator::begin_evaluation(self, charges);
    }
    fn open_stage(&mut self) {
        DeviceOperator::open_stage(self);
    }
    fn close_stage(&mut self, stage: DeviceStage) {
        DeviceOperator::close_stage(self, stage);
    }
    fn stage_timings(&self) -> Option<DeviceStageTimings> {
        DeviceOperator::stage_timings(self)
    }
    fn sync(&mut self) {
        DeviceOperator::sync(self);
    }
    fn read_output(&mut self) -> Result<&LeafStore<T>, KernelError> {
        DeviceOperator::read_output(self)
    }
    fn report(&self) -> &DeviceReport {
        DeviceOperator::report(self)
    }
    fn counters(&self) -> DeviceCounters {
        DeviceOperator::counters(self)
    }
    fn download_views(&mut self) -> Result<ViewsImage, KernelError> {
        DeviceOperator::download_views(self)
    }
    fn download_expansions(&mut self) -> Result<(Vec<T>, Vec<T>), KernelError> {
        DeviceOperator::download_expansions(self)
    }
}

/// `value` as a `U`, where `T` and `U` are one type: `SimdScalar` is implemented for f32
/// and f64 only, so generic code that has checked the type may convert between `T` and
/// the concrete type it names.
fn same_type<T: 'static, U: 'static>(value: T) -> U {
    *(Box::new(value) as Box<dyn Any>)
        .downcast::<U>()
        .unwrap_or_else(|_| panic!("the two types are one"))
}

/// Creates the device operator of `host` for the precision of `T` ([`DeviceOperator::new`]).
pub(crate) fn driver<T: SimdScalar + Stored + Equivalence + Default>(
    host: LaplaceOperator<T>,
    device: Device,
    plan: &Plan,
    (source_counts, target_counts): (&[usize], &[usize]),
    options: &DeviceOptions,
    tuner: Tuner,
) -> Result<Box<dyn DeviceDriver<T>>, FmmError> {
    fn concrete<E: DeviceScalar, T: SimdScalar>(
        host: LaplaceOperator<T>,
        device: Device,
        plan: &Plan,
        (source_counts, target_counts): (&[usize], &[usize]),
        options: &DeviceOptions,
        tuner: Tuner,
    ) -> Result<Box<dyn DeviceDriver<T>>, FmmError> {
        let host: LaplaceOperator<E> = same_type(host);
        let operator = DeviceOperator::new(
            host,
            device,
            plan,
            source_counts,
            target_counts,
            options,
            tuner,
        )?;
        let boxed: Box<dyn DeviceDriver<E>> = Box::new(operator);
        Ok(same_type(boxed))
    }
    let counts = (source_counts, target_counts);
    match T::PRECISION {
        nd_fmm_tables::cache::Precision::F32 => {
            concrete::<f32, T>(host, device, plan, counts, options, tuner)
        }
        nd_fmm_tables::cache::Precision::F64 => {
            concrete::<f64, T>(host, device, plan, counts, options, tuner)
        }
    }
}

/// The precision of `T` in `nd-fmm-kernels`' terms.
pub(crate) fn precision_of<T: Stored>() -> nd_fmm_kernels::Precision {
    match T::PRECISION {
        nd_fmm_tables::cache::Precision::F32 => nd_fmm_kernels::Precision::F32,
        nd_fmm_tables::cache::Precision::F64 => nd_fmm_kernels::Precision::F64,
    }
}

/// The M2L strategy of a device FMM of degree p under `M2lStrategy::Auto` with M2L on the
/// device (T12, [`crate::tune`], "When tuning runs"), decided before the tables are built,
/// since it decides which tables the host and the device build: from the cache, by the
/// static rule, or by timing rotation against dense with each M2L GEMM candidate on the
/// plan's largest V level, standalone on `device` (the level's multipoles and locals, its
/// V view, and the candidate's tables through `table_cache` when given, uploaded for the
/// timing and freed after). At p ≥ 12 without `table_cache` the dense candidate would
/// take seconds to build, so the strategy keeps the static rule there and is not stored.
/// Where dense wins, the GEMM it won with becomes the decision of the M2L level calls of
/// the largest level's bucket.
pub(crate) fn tune_strategy<T: Stored>(
    tuner: &mut Tuner,
    device: &mut Device,
    plan: &Plan,
    p: usize,
    table_cache: Option<&Path>,
) -> M2lStrategy {
    match T::PRECISION {
        nd_fmm_tables::cache::Precision::F32 => {
            strategy::<f32>(tuner, device, plan, p, table_cache)
        }
        nd_fmm_tables::cache::Precision::F64 => {
            strategy::<f64>(tuner, device, plan, p, table_cache)
        }
    }
}

/// [`tune_strategy`] in `T`.
fn strategy<T: DeviceScalar>(
    tuner: &mut Tuner,
    device: &mut Device,
    plan: &Plan,
    p: usize,
    table_cache: Option<&Path>,
) -> M2lStrategy {
    let precision = T::FLOAT;
    let info = device.info().clone();
    let n = (p + 1) * (p + 1);
    let fixed = static_strategy(precision, p);
    let dense: Vec<Candidate> = gemm_candidates(&info, precision, n, OperatorKind::M2l, false)
        .into_iter()
        .map(Candidate::Dense)
        .collect();
    let first = match fixed {
        M2lStrategy::Rotation => Candidate::Rotation,
        _ => dense[0],
    };
    let candidates: Vec<Candidate> = if fixed == M2lStrategy::Rotation {
        std::iter::once(Candidate::Rotation).chain(dense).collect()
    } else {
        dense.into_iter().chain([Candidate::Rotation]).collect()
    };
    if tuner.enabled()
        && tuner.cached(Decision::Strategy).is_none()
        && p >= 12
        && table_cache.is_none()
    {
        tuner.record_static(
            Decision::Strategy,
            first,
            "the dense candidate at p ≥ 12 needs a table cache: not tuned, not stored".to_owned(),
        );
        return fixed;
    }
    let (level, pairs) = (0..plan.nlevels())
        .map(|l| (l, plan.level(l).v().len()))
        .max_by_key(|&(l, k)| (k, std::cmp::Reverse(l)))
        .unwrap_or((0, 0));
    tuner.start_phase();
    let mut bench: Option<StrategyBench<'_, T>> = None;
    let choice = tuner.decide(
        Decision::Strategy,
        (Some(level), pairs),
        candidates,
        |candidate, deadline| {
            if bench.is_none() {
                bench = Some(StrategyBench::new(device, plan, level)?);
            }
            let bench = bench.as_mut().expect("created above");
            match candidate {
                Candidate::Dense(choice) => {
                    bench.time_dense(device, (p, table_cache), choice, deadline)
                }
                _ => bench.time_rotation(device, (p, table_cache), deadline),
            }
        },
    );
    drop(bench);
    tuner.end_phase();
    if choice.strategy() == Some(M2lStrategy::Dense) {
        tuner.record_dense_gemm(bucket(pairs));
    }
    choice.strategy().unwrap_or(fixed)
}

/// The device data of the strategy decision (T12): the largest V level's multipoles and
/// locals (zero), its V view (host and device), and each candidate's tables, created when
/// first timed.
struct StrategyBench<'a, T: DeviceFloat> {
    boxes: usize,
    host_view: &'a VList,
    view: GroupedView,
    multipoles: DeviceBuffer<T>,
    locals: DeviceBuffer<T>,
    dense: Option<(Tables<T>, TranslationScratch<T>)>,
    rotation: Option<(DeviceRotation<T>, RotationPlan)>,
}

impl<'a, T: DeviceScalar> StrategyBench<'a, T> {
    fn new(device: &mut Device, plan: &'a Plan, level: usize) -> Result<Self, Rejection> {
        let boxes = plan.index().len(level);
        let host_view = plan.level(level).v();
        let view = GroupedView::upload(device, &grouped_arrays(host_view), boxes)?;
        Ok(Self {
            boxes,
            host_view,
            view,
            multipoles: device.alloc(0)?,
            locals: device.alloc(0)?,
            dense: None,
            rotation: None,
        })
    }

    /// Allocates the level's multipoles and locals for order n, once.
    fn buffers(&mut self, device: &mut Device, n: usize) -> Result<(), Rejection> {
        if self.multipoles.len() != self.boxes * n {
            self.multipoles = device.alloc(self.boxes * n)?;
            self.locals = device.alloc(self.boxes * n)?;
        }
        Ok(())
    }

    /// Times the level's dense M2L with `choice`.
    fn time_dense(
        &mut self,
        device: &mut Device,
        (p, table_cache): (usize, Option<&Path>),
        choice: &GemmChoice,
        deadline: Instant,
    ) -> Result<Duration, Rejection> {
        let n = (p + 1) * (p + 1);
        let info = device.info().clone();
        if let GemmKind::HandWritten(layout) = choice.gemm {
            layout
                .check(&info)
                .map_err(|e| Rejection::Unregistered(e.to_string()))?;
        }
        self.buffers(device, n)?;
        if self.dense.is_none() {
            let library = library_applies(info.backend, T::FLOAT, n);
            let batch_offsets = self.host_view.batch_offsets();
            let columns = [
                GemmChoice::hand_written(GemmLayout::default_for(&info, n)),
                GemmChoice::library(T::FLOAT),
            ]
            .iter()
            .map(|c| {
                c.settings(&info, n)
                    .size(info.backend, T::FLOAT, batch_offsets)
                    .columns
            })
            .max()
            .unwrap_or(0);
            let needed =
                Tables::<T>::bytes(n, 316, library) + TranslationScratch::<T>::bytes(columns * n);
            if let Some(limit) = device.available_memory()
                && needed > limit
            {
                return Err(Rejection::Skipped(format!(
                    "the dense tables and scratch need {needed} bytes, {limit} are available"
                )));
            }
            let tables: M2lTables<T> = match table_cache {
                Some(dir) => TableCache::new(dir).load_or_build(p).0,
                None => M2lTables::build(p),
            };
            let tables = Tables::upload(device, tables.matrices().as_slice(), n, library)?;
            let scratch = TranslationScratch::<T>::new(device, columns * n)?;
            self.dense = Some((tables, scratch));
        }
        let (tables, scratch) = self.dense.as_mut().expect("created above");
        if choice.is_library() && !tables.has_library_copy() {
            return Err(Rejection::Unregistered(
                "the tables have no library copy".into(),
            ));
        }
        let plan = GroupedPlan::new(
            device,
            &grouped_arrays(self.host_view),
            self.boxes,
            &choice.settings(&info, n),
            tables,
            scratch,
        )?;
        if choice.is_library() && plan.gemm() != Gemm::Library {
            return Err(Rejection::Unregistered(format!(
                "the library does not take the level's shapes: {}",
                plan.library_rejection().unwrap_or("not a candidate")
            )));
        }
        let (view, multipoles, locals) = (&self.view, &self.multipoles, &mut self.locals);
        time_launches(device, deadline, |d| {
            grouped(
                d,
                &plan,
                view,
                Accumulate::Rows,
                tables,
                Operands::Separate {
                    input: multipoles.as_slice(),
                    output: locals.as_slice_mut(),
                },
                scratch,
            )
        })
        .map_err(Rejection::from)
    }

    /// Times the level's rotation M2L in the backend's default layout.
    fn time_rotation(
        &mut self,
        device: &mut Device,
        (p, table_cache): (usize, Option<&Path>),
        deadline: Instant,
    ) -> Result<Duration, Rejection> {
        let n = (p + 1) * (p + 1);
        let layout = RotationLayout::default_for(device.info(), p);
        layout
            .check(device.info(), p, T::FLOAT)
            .map_err(|e| Rejection::Unregistered(e.to_string()))?;
        self.buffers(device, n)?;
        if self.rotation.is_none() {
            let tables: HostRotationTables<T> = match table_cache {
                Some(dir) => TableCache::new(dir).load_or_build(p).0,
                None => HostRotationTables::build(p),
            };
            let arrays = RotationHostArrays::new(tables.tables(RotationOperator::M2l));
            let rotation = DeviceRotation::upload(device, &arrays.arrays())?;
            let plan = RotationPlan::new(device, &grouped_arrays(self.host_view), self.boxes)?;
            self.rotation = Some((rotation, plan));
        }
        let (rotation, plan) = self.rotation.as_ref().expect("created above");
        let (view, multipoles, locals) = (&self.view, &self.multipoles, &mut self.locals);
        time_launches(device, deadline, |d| {
            nd_fmm_kernels::rotation::m2l(
                d,
                layout,
                plan,
                view,
                rotation,
                multipoles.as_slice(),
                locals.as_slice_mut(),
            )
        })
        .map_err(Rejection::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_slots_follow_the_source_chunks() {
        // Leaves of 2, 0 and 3 points: chunks of 8, 0 and 12 values.
        assert_eq!(charge_slots(&[2, 0, 3]).unwrap(), vec![6, 7, 17, 18, 19]);
        assert_eq!(point_offsets(&[2, 0, 3]).unwrap(), vec![0, 2, 2, 5]);
        assert!(charge_slots(&[]).unwrap().is_empty());
    }

    #[test]
    fn traffic_by_data_adds_up() {
        let mut by = TrafficByData::default();
        let t = Traffic {
            uploads: 1,
            upload_bytes: 8,
            downloads: 2,
            download_bytes: 16,
        };
        by.add(DataKind::Charges, t);
        by.add(DataKind::Output, t);
        by.add(DataKind::Output, t);
        assert_eq!(by.get(DataKind::Output).downloads, 4);
        assert_eq!(by.total().upload_bytes, 24);
        assert_eq!(DataKind::ALL.len(), 9);
        assert!(
            DataKind::ALL
                .iter()
                .enumerate()
                .all(|(i, &k)| k as usize == i)
        );
    }

    #[test]
    fn same_type_converts_only_between_equal_types() {
        let x: f32 = same_type::<f32, f32>(1.5);
        assert_eq!(x, 1.5);
        assert!(std::panic::catch_unwind(|| same_type::<f32, f64>(1.5)).is_err());
    }
}
