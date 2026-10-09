//! The user-facing FMM: [`FmmBuilder`], [`Fmm`] and its [`Output`] (C3.2).
//!
//! [`FmmBuilder::build`] takes source and target points in the caller's order, on any
//! rank, builds the octree, the plan of `nd-fmm-plan`, the tables, the
//! [`LaplaceOperator`] and the evaluator, moves every point to the rank that owns its
//! leaf, and loads the points there in the leaf-scaled form of CONVENTIONS §3.13.
//! [`Fmm::evaluate`] takes charges in the caller's source order, moves them to the
//! owners, runs the evaluator stage by stage, and returns potentials (and gradients) in
//! the caller's target order on the caller's rank, with 1/(4π) and the target leaf radius
//! applied once (§3.1, §3.13, "Output"):
//!
//! φ(x) = φ̂ / (4π r_t),  ∇φ(x) = ĝ / (4π r_t²),
//!
//! so φ(x) = Σⱼ qⱼ / (4π |x − yⱼ|), the free-space Green's function of −Δ. A source and a
//! target at the same point do not interact (§3.13, "Coincident pairs").
//!
//! # Build
//!
//! [`FmmBuilder::build`] is collective. In order:
//!
//! 1. Checks the settings (among them that this rank's CPU runs the P2P kernel), the MPI
//!    threading level when `threads` > 1, the supplied domain ([`Domain::new`]) and that
//!    every point is finite, builds the thread pool if `threads` > 1, opens the device
//!    of a device [`Backend`] (compiled in, a device that comes up, the precision
//!    supported), and agrees the outcome on every rank (one all-reduce). With a device
//!    backend it splits the communicator by node once (`split_shared`, Phase 5 T8): the
//!    rank's place there is in the device report, and on the CPU runtime the units per
//!    cube are capped at the node's cores over its ranks.
//! 2. Counts the points of all ranks (one all-reduce); none at all is an error.
//! 3. Takes the supplied domain, or `compute_global_bounding_box` over the sources and
//!    targets of every rank (after one all-reduce pair that rejects points spanning no
//!    volume, which that function cannot handle). Checks that every point lies strictly
//!    inside it and agrees the outcome (one all-reduce).
//! 4. Builds one octree from the finest-level keys (`points_to_morton` at level 16) of
//!    this rank's sources and targets together, with `max_level`, `max_points_per_leaf`
//!    (`OctreeOptions::with_max_fine_keys`) and the ghost-children layer, and the plan
//!    of its box index and lists. The octree's partition weighs every key it is passed,
//!    so the points (`nd-octree`'s default since Phase 5 T4); the FMM passes no other
//!    weight (docs/design/distributed-fmm.md §3.5).
//! 5. Moves every point to the rank that owns its leaf (Phase 5 T6, design §4.4): one
//!    `nd_fmm_plan::redistribute::Redistribution` for the sources and one for the
//!    targets, routed by the keys of step 4, then the f64 coordinates of each forwarded
//!    to the owners, into leaf order. Within a leaf the points lie in the order (origin
//!    rank, position on that rank).
//! 6. Takes the points of every local leaf from the redistributions, the radius of every
//!    local leaf, and the output order (local).
//! 7. All-reduces the largest number of sources in a leaf, which sizes the P2P scratch;
//!    with a device backend resolves the M2L strategy of [`M2lStrategy::Auto`] for the
//!    device (Phase 4 T12: from the tuning cache, by timing rotation against dense, or by
//!    the static rule; on several ranks with a tuning cache rank 0's, broadcast, Phase 5
//!    T8); builds or loads the tables, the host operator, and the evaluator with the
//!    per-leaf counts (its own collectives).
//! 8. Writes the leaf-scaled source coordinates and target positions, from the forwarded
//!    f64 coordinates, into the evaluator's stores, once ([`leaf_coordinates`]). With a
//!    device backend (Phase 5 T8) it then builds the device operator around the host
//!    operator, from the evaluator's source counts (the ghost leaves' included) and the
//!    index lists of its exchanges: it checks that its buffers fit in device memory,
//!    allocates them, uploads the views, the exchanges' columns and the tables, and takes
//!    its GEMM decisions; swaps it in for the host operator; uploads the points, after
//!    which the device takes its P2P decision and ends its tuning (device work only); and
//!    agrees the outcome on every rank (one all-reduce).
//!
//! It also reads the BLAS thread variables once, for [`Fmm::threading`].
//!
//! An error that depends on one rank's input is agreed by every rank before the next
//! collective: the rank that found it returns it, the others [`FmmError::OtherRank`];
//! errors that every rank sees alike ([`NoPoints`](FmmError::NoPoints),
//! [`DegenerateExtent`](FmmError::DegenerateExtent), and those of the plan, the
//! redistributions and the evaluator) are returned on every rank. No communication sits
//! in a rank-dependent branch.
//!
//! # Evaluation
//!
//! [`Fmm::evaluate`] is collective: it agrees the length of the charge vector (one
//! all-reduce), forwards the charges to the ranks that own their sources, into leaf order
//! (one all-to-all-v), writes them after the coordinates of each source chunk (§3.13,
//! "Source chunks"), resets the evaluator and runs its six stages, scales the target
//! output of the owned targets in leaf order (the output pass, next section), and moves
//! it back to the caller's ranks and order (one all-to-all-v), timing each step
//! ([`StageTimings`]). For a fixed input on every rank, a fixed rank count and fixed
//! settings, two evaluations are bit-identical (the accumulation order of
//! `nd_fmm_plan::evaluator`), for every number of threads. [`Fmm::evaluate_into`]
//! (Phase 4S T9, decision 11) does the same into the buffers of an [`Output`] the caller
//! reuses, with the same bits; `evaluate` evaluates into a fresh one.
//!
//! # The output pass (Phase 4S T9)
//!
//! The evaluator leaves the target output in leaf order and leaf-scaled (§3.13); the
//! output pass turns it into φ and ∇φ, dividing each value by 4π r_t or 4π r_t² in f64
//! and rounding once to `T`: 1/(4π) and the leaf radius applied once, here (§3.1, §3.13,
//! "Output"). It runs on the rank that owns the targets, in their received order, which
//! is leaf order, and writes φ, or φ and ∇φ, of each target next to each other: the o = 1
//! or 4 values per target that the backward move of the targets' redistribution returns
//! to the caller's ranks and order, where they are copied into the [`Output`] (Phase 5
//! T6). At build `Fmm` stores, for every owned target, its point in leaf order and its
//! local leaf (`u32` each: 8 N_t bytes), and for every local leaf the two scales (16
//! bytes), formed as the pass before T9 formed them. Where it runs ([`OutputPass`],
//! [`FmmBuilder::output_pass`], [`Fmm::output_pass`]):
//! - **on the host** (the host path, Metal, and every backend with [`OutputPass::Host`]):
//!   one loop over the owned targets, on the `Fmm`'s pool when it has one (threads > 1,
//!   not [`set_serial`](Fmm::set_serial)) and serially otherwise, each thread writing a
//!   contiguous range of the values and gathering them from the store. On Metal and CUDA
//!   the pool is idle by then (device-path.md §11);
//! - **on the device** (by default on a device with f64 arithmetic: the CPU runtime and
//!   CUDA): `nd_fmm_kernels::movement::gather_output` makes the same values on the device
//!   from a copy of the order uploaded once, with the same f64 division and one rounding to
//!   `T`; the evaluation still downloads o N_t values once and syncs once, and the host
//!   places them next to each other (one more launch per evaluation, device-path.md §4.1).
//!
//! Both give the bits of the pass before T9, which the tests keep as their oracle
//! (`Fmm::reference_output`). The charge load is parallel the same way: the forwarded
//! charges, already in leaf order, are copied into the host's source chunks (by leaves),
//! on the pool; on a device the forwarded buffer itself is the upload, which the device
//! takes over without a copy (Phase 4S T11); with P2M, P2L and P2P on the device the
//! host's source store, which feeds only host-fallback calls of those kinds, is not
//! written at all. The output is read in place from CubeCL's host copy of the download (no
//! host buffer of the operator in between).
//!
//! On request ([`FmmBuilder::kind_timings`], [`Fmm::set_kind_timings`]; Phase 4S T5) it
//! also times every level call by operator kind and level ([`KindTiming`],
//! [`StageTimings::kinds`]), on the host path and every device backend, with the same
//! output bit for bit.
//!
//! # P2P kernel
//!
//! [`FmmBuilder::p2p_kernel`] chooses the kernel of P2P ([`P2pChoice`]): by default
//! the SIMD kernel of `nd-fmm-simd` on the widest ISA of the machine, or the reference
//! loop of `nd-fmm-ref`, or an ISA named explicitly. Every other operator is the same.
//! Kernels differ in the last bits of each near-field term, so outputs of different
//! choices agree to rounding, not bit for bit; on one machine, ISA and build they are
//! reproducible ([`operator`](crate::operator#p2p-kernel), "P2P kernel").
//! [`Fmm::p2p_kernel`] reports the kernel that runs, for reports next to
//! [`Fmm::threading`].
//!
//! # Threads (C3.5)
//!
//! With [`FmmBuilder::threads`] n > 1, the `Fmm` owns a rayon pool of n threads and the
//! operator runs every level call in it, each target on one thread
//! ([`LaplaceOperator::with_pool`]); with n = 1 (the default) there is no pool. Every
//! collective, and so every MPI call, stays on the calling thread, which needs MPI at
//! [`Threading::Funneled`] or above. The rules for threads, MPI and BLAS, and how to
//! launch, are in [`threading`](crate::threading).
//!
//! # Backends (Phase 4)
//!
//! [`FmmBuilder::backend`] chooses where the operators run: [`Backend::Host`], the
//! default and the host path above, or a device backend of the `gpu` feature
//! ([`Backend::Cpu`], [`Backend::Metal`], [`Backend::Cuda`]). A device backend keeps
//! the interface, the plan and the evaluator: the operator holds its data on the device,
//! [`Fmm::evaluate`] tells it where an evaluation starts and reads its output once, and
//! the output scaling is the same code. Every operator kind can run on the host
//! fallback ([`OperatorKind`], [`Fmm::placement`]); with every kind there the output
//! equals the host path's bit for bit. By default (Phase 4 T11) every kind runs on the
//! device under every strategy: P2P ([`DeviceP2pLayout`]), P2M, L2P, P2L and M2P
//! ([`DeviceLeafLayout`]), M2M, L2L and dense M2L as grouped GEMMs ([`DeviceGemm`]) and
//! M2L under `Rotation` by the rotation kernel. The data stay on the device for the whole
//! evaluation, which moves the charges up and the output down and syncs once, at that
//! download; the output agrees with the host path's within the FMM bounds of
//! docs/phase4/README.md, and its errors against the direct sum are the host path's.
//! [`StageTimings`] time each stage on the host and, on request and where the device
//! times on itself, on the device ([`FmmBuilder::device_timestamps`]). The `device` module (feature `gpu`)
//! documents the residency, the transfers, the scheduling, the fallback, the errors and
//! the threads rule; docs/design/device-path.md is the design.
//!
//! **Autotune** (Phase 4 T12, C4.7; the `tune` module, feature `gpu`): with
//! [`FmmBuilder::tuning_cache`] the device path times its candidates at build (the M2L
//! strategy under `Auto`, the GEMM of each level call under `DeviceGemm::Auto`, the P2P
//! layout under `DeviceP2pLayout::Auto`) within [`FmmBuilder::tuning_budget`], keeps the
//! fastest for the `Fmm`'s lifetime and stores it in the caller's directory; a later
//! build with the same key reads it. Without a directory the static rule applies: f32
//! `Dense`; f64 `Dense` at every p on CUDA (Phase 4S decision 9), and on Metal and the CPU
//! runtime `Dense` to p = 11 and `Rotation` above (provisional); the GEMMs of
//! `DeviceGemm::Auto` and the backend's P2P layout. Two builds from the same cache give
//! the same bits; `evaluate` never tunes.
//!
//! # Several ranks (C5.1)
//!
//! Each rank passes any subset of the points, none included, and gets the output of its
//! own targets in the order it passed them; where the points are is the library's
//! business (Phase 5 T6, docs/design/distributed-fmm.md §4).
//! - **Where points go.** The octree (step 4) partitions the coarse blocks over the ranks
//!   by points, independently of where the points were passed; its leaves are those of
//!   the one-rank tree of the same points. Each point goes to the rank that owns its leaf
//!   (step 5), and each charge with its source on every evaluation; the output comes back
//!   to the target's rank. Within a leaf the points lie in the order (origin rank,
//!   position on that rank).
//! - **Collectives.** Per build, besides those of step 1–3, `Octree::new`, `Plan::new` and
//!   `Evaluator::new`: two `Redistribution::new` (each an all-to-all of counts, an
//!   all-reduce of the errors, a communicator duplicate and an all-to-all-v of the keys),
//!   two all-to-all-v's of the coordinates, and the all-reduce of step 7. Per
//!   evaluation: the all-reduce of the charge length, the forward of the charges (an
//!   all-to-all-v), the evaluator's exchanges (the source exchange, the coarse gather and
//!   the multipole exchange of every level), and the backward of the output (an
//!   all-to-all-v). Every rank enters each one, a rank with no points included.
//! - **Equal to one rank.** On P ranks the output is bit for bit the output of the
//!   one-rank `Fmm` of the same settings over the union of every rank's points in rank
//!   order (rank 0's points, then rank 1's, …): its leaves are the same (Phase 5 T4), the
//!   points of a leaf lie in the same order, the near and X rows are ordered by the source
//!   leaf's (level, key) (Phase 5 T5), and ghost and global values are exact copies or the
//!   same sums (design §5.1, §5.3). Two distributions of the same points over the ranks
//!   put the points of a leaf in different orders, so their outputs differ in the last
//!   bits: within 100 u_T relative L2 over every target, φ and ∇φ (1.1e-14 in f64, 6.0e-6
//!   in f32; design §5.4). The errors against the direct sum are the one-rank run's.
//! - **Determinism.** For fixed input on every rank, a fixed rank count and fixed
//!   settings, the output is bit-identical from evaluation to evaluation, from build to
//!   build, and for every number of threads.
//! - **Threads.** The `threads(n)` of [`FmmBuilder::threads`] are per rank: ranks × n stay
//!   within the physical cores ([`threading`](crate::threading)), and every MPI call,
//!   the moves included, is made on the calling thread.
//! - **Errors.** `PointsNotOwned` (until Phase 5 T6) is gone; a redistribution that does
//!   not fit MPI's `i32` counts is [`FmmError::Redistribution`], on every rank.
//! - **The device** (Phase 5 T8) runs on any number of ranks: every rank opens the default
//!   device of its backend, and the device operator mirrors the evaluator's exchanges
//!   (the `device` module, "Several ranks"). Per evaluation it adds one all-reduce, after
//!   the output's download, that agrees a device error of any rank; the rank that failed
//!   returns [`FmmError::Device`], the others [`FmmError::OtherRank`], and from then on
//!   the charge-length agreement carries the kept error. [`Fmm::exchange_sizes`] says
//!   what the exchanges move on the rank.
//!
//! # Precision
//!
//! `T` is f64 or f32 (`Stored + SimdScalar + Equivalence`). Coordinates are always f64
//! and are rounded to `T` once, as leaf-scaled values; tables are built in f64 and
//! rounded. f32 is accurate to its floor at p ≤ 8 (design §4); a larger p is allowed but
//! gains nothing.

use std::f64::consts::PI;
use std::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "gpu")]
use mpi::traits::{Communicator, Root};
use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::{Evaluator, EvaluatorError};
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, HostData, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p,
};
use nd_fmm_plan::plan::{Plan, PlanError};
use nd_fmm_plan::redistribute::{Redistribution, RedistributionError};
use nd_fmm_plan::store::{LeafSliceMut, LeafStore};
use nd_fmm_tables::cache::{Stored, TableKind};
use nd_fmm_tables::{CacheOutcome, TableCache};
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::octree::compute_global_bounding_box;
use nd_octree::{Octree, OctreeOptions, PhysicalBox, points_to_morton};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rlst::SliceArray;
use thiserror::Error;

#[cfg(feature = "gpu")]
use nd_fmm_kernels::{TimingWindow, WindowTime};

#[cfg(feature = "gpu")]
use crate::device::{
    self, DeviceCounters, DeviceDriver, DeviceOptions, DeviceOutput, DeviceReport, ViewsImage,
};
use crate::geometry::{Domain, GeometryError, leaf_coordinates, radius};
use crate::operator::{Isa, LaplaceOperator, P2pChoice, SimdScalar};
use crate::tables::{M2lStrategy, Tables};
use crate::threading::{REQUIRED_MPI_THREADING, ThreadingReport};

/// The largest degree p that [`FmmBuilder::build`] accepts: CONVENTIONS §3.9 covers M2L
/// up to p = 20 in f64.
pub const MAX_DEGREE: usize = 20;

/// The default `max_level` of [`FmmBuilder`]: 16, the deepest level of a Morton key.
pub const DEFAULT_MAX_LEVEL: usize = DEEPEST_LEVEL as usize;

/// The default `max_points_per_leaf` of [`FmmBuilder`].
pub const DEFAULT_MAX_POINTS_PER_LEAF: usize = 64;

/// The default [`FmmBuilder::tuning_budget`]: 10 s (docs/design/device-path.md §10.4).
pub const DEFAULT_TUNING_BUDGET: Duration = Duration::from_secs(10);

/// The deepest level of a Morton key.
const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// Where the operators of an FMM run (Phase 4; docs/design/device-path.md §3.3). Text
/// form, for command lines: `host`, `cpu`, `metal`, `cuda`.
///
/// The enum exists without the `gpu` feature, so that a host-only build names a device
/// backend and refuses it with [`SettingsError::BackendNotCompiled`] rather than failing
/// to compile. The device path is the `device` module (feature `gpu`).
///
/// ```
/// use nd_fmm_exec::fmm::Backend;
///
/// assert_eq!(Backend::default(), Backend::Host);
/// assert_eq!("metal".parse(), Ok(Backend::Metal));
/// assert_eq!(Backend::Cpu.to_string(), "cpu");
/// assert!(Backend::Host.is_compiled());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend {
    /// The host path: [`LaplaceOperator`] on the calling thread or the `Fmm`'s pool.
    #[default]
    Host,
    /// The CubeCL CPU runtime (feature `cpu`): f32 and f64, the correctness backend.
    Cpu,
    /// Metal, wgpu with the MSL compiler (feature `metal`): f32 only.
    Metal,
    /// CUDA (feature `cuda`): f32 and f64; type-checked in CI, run by hand on an H100
    /// (Phase 4S).
    Cuda,
}

impl Backend {
    /// Every backend, in report order.
    pub const ALL: [Self; 4] = [Self::Host, Self::Cpu, Self::Metal, Self::Cuda];

    /// The text form: `host`, `cpu`, `metal` or `cuda`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// True for every backend but [`Host`](Self::Host).
    pub fn is_device(self) -> bool {
        self != Self::Host
    }

    /// True if this build can run the backend: always for [`Host`](Self::Host), and for
    /// a device backend if its cargo feature is enabled.
    pub fn is_compiled(self) -> bool {
        match self {
            Self::Host => true,
            Self::Cpu => cfg!(feature = "cpu"),
            Self::Metal => cfg!(feature = "metal"),
            Self::Cuda => cfg!(feature = "cuda"),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The layout of the device P2P kernel ([`FmmBuilder::device_p2p_layout`]; Phase 4 T6,
/// docs/design/device-path.md §6.2): the parallel mapping of the near field onto a
/// device, with the same formulation, order and results within the P2P contract in each.
/// Every layout gives the same bits from evaluation to evaluation. Ignored by
/// [`Backend::Host`].
///
/// The enum exists without the `gpu` feature, as [`Backend`] does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceP2pLayout {
    /// By backend: the CPU layout on the CPU runtime, the cube layout of 64 units on
    /// Metal and CUDA.
    #[default]
    Auto,
    /// One cube of this many units per target leaf, sources staged through shared
    /// memory in tiles of as many sources (on the CPU runtime correctness only, at most
    /// one unit per core).
    Cube(u32),
    /// One plane per target leaf and this many planes per cube, each plane staging its
    /// own tile: for small leaves. Needs a device with one plane size.
    Plane(u32),
    /// Targets in vector lanes of the host's width, one unit per core, each a contiguous
    /// range of target leaves: the layout of the CPU runtime, whose units per cube
    /// `threads(n)` caps.
    Cpu,
}

/// The layout of the device leaf operators P2M, L2P, P2L and M2P
/// ([`FmmBuilder::device_leaf_layout`]; Phase 4 T7, docs/design/device-path.md §6.3): the
/// parallel mapping of the boxes and target leaves of a level onto a device, with the same
/// arithmetic and order, and so results within the leaf-operator bounds, in each. Every
/// layout gives the same bits from evaluation to evaluation. Ignored by [`Backend::Host`].
///
/// The enum exists without the `gpu` feature, as [`Backend`] does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceLeafLayout {
    /// By backend: the CPU layout on the CPU runtime; on Metal and CUDA the cube layout of
    /// 64 units with tiles of 32 points, fewer where the shared memory holds fewer at p.
    #[default]
    Auto,
    /// One cube of `units` units per box (P2M, P2L: coefficient owners, the harmonics of
    /// `tile` points at a time staged in shared memory) or per target leaf (L2P, M2P: one
    /// unit per target point). On the CPU runtime correctness only, at most one unit per
    /// core.
    Cube {
        /// Units per cube, at least 1.
        units: u32,
        /// Points per tile of P2M and P2L, from 1 to `units`.
        tile: u32,
    },
    /// One unit per core, each a contiguous range of boxes or target leaves, no shared
    /// memory: the layout of the CPU runtime, whose units per cube `threads(n)` caps.
    Cpu,
}

/// The GEMM of the device translations M2M, L2L and dense M2L ([`FmmBuilder::device_gemm`];
/// Phase 4 T8 and T9, docs/design/device-path.md §6.4, §6.5). Every choice adds each
/// target's products in row order and gives the same bits from evaluation to evaluation.
/// Ignored by [`Backend::Host`].
///
/// The enum exists without the `gpu` feature, as [`Backend`] does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceGemm {
    /// The default: for M2M and L2L the rule of the design, the library matmul (CMMA,
    /// named explicitly) in f32 at p ≥ 8 on a GPU where it accepts the level's shape and
    /// keeps f32 inputs, the hand-written kernel otherwise; for M2L the hand-written
    /// kernel. Decided per level call at build and reported.
    ///
    /// M2L departs from the design's rule (decided after Phase 4 T9): its library GEMM
    /// needs one shape per launch, so it pads each run of offsets to the run's widest
    /// batch, and on the M3 Max only 44–66% of the padded columns were useful on the C3.2
    /// cube and the Plummer sphere; there the hand-written kernel was faster on every
    /// level the library took (for example 3.5 ms against 5.5 ms on the cube's level 4 at
    /// p = 8). [`Library`](Self::Library) keeps the library M2L selectable.
    #[default]
    Auto,
    /// The design's rule for every translation, M2L included: the library matmul in f32
    /// at p ≥ 8 on a GPU where it accepts the level's shape and keeps f32 inputs, the
    /// hand-written kernel otherwise.
    Library,
    /// The hand-written kernel everywhere, in the backend's layout.
    HandWritten,
}

/// Where the output pass of an evaluation runs ([`FmmBuilder::output_pass`]; Phase 4S T9,
/// decision 12): the pass that turns the leaf-ordered, leaf-scaled target output into φ
/// and ∇φ in the caller's order, dividing by 4π r_t and 4π r_t² in f64 and rounding once
/// to `T` (CONVENTIONS §3.1, §3.13, "Output"). Both passes give the same bits; the
/// [module documentation](self#the-output-pass-phase-4s-t9) describes them.
///
/// The enum exists without the `gpu` feature, as [`Backend`] does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OutputPass {
    /// The default: on the device where the backend is a device with f64 arithmetic (the
    /// CPU runtime, CUDA), on the host otherwise (the host path, Metal).
    #[default]
    Auto,
    /// On the host, on every backend: the host pass, which a device backend runs on the
    /// downloaded leaf-ordered output.
    Host,
    /// On the device: [`FmmBuilder::build`] refuses it on the host backend and on a device
    /// without f64 arithmetic (Metal) with [`SettingsError::OutputPassUnsupported`].
    Device,
}

/// The error of parsing a [`Backend`] from text it does not name.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("`{text}` is not a backend; expected `host`, `cpu`, `metal` or `cuda`")]
pub struct UnknownBackend {
    /// The text that was parsed.
    pub text: String,
}

/// Parses the [`Display`](fmt::Display) form, whether or not the backend is compiled in.
impl FromStr for Backend {
    type Err = UnknownBackend;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|backend| backend.name() == s)
            .ok_or_else(|| UnknownBackend { text: s.to_owned() })
    }
}

/// One operator kind, for the placement of the device path and for reports
/// (docs/design/device-path.md §7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    /// P2M.
    P2m,
    /// M2M, both passes.
    M2m,
    /// M2L.
    M2l,
    /// P2L.
    P2l,
    /// L2L.
    L2l,
    /// L2P.
    L2p,
    /// M2P.
    M2p,
    /// P2P.
    P2p,
}

impl OperatorKind {
    /// Every kind, in the order of the operator interface.
    pub const ALL: [Self; 8] = [
        Self::P2m,
        Self::M2m,
        Self::M2l,
        Self::P2l,
        Self::L2l,
        Self::L2p,
        Self::M2p,
        Self::P2p,
    ];

    /// The name: `P2M`, `M2M`, …
    pub fn name(self) -> &'static str {
        match self {
            Self::P2m => "P2M",
            Self::M2m => "M2M",
            Self::M2l => "M2L",
            Self::P2l => "P2L",
            Self::L2l => "L2L",
            Self::L2p => "L2P",
            Self::M2p => "M2P",
            Self::P2p => "P2P",
        }
    }
}

impl fmt::Display for OperatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Where an operator kind runs ([`Fmm::placement`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// On the host: the host path, or the host fallback of a device backend.
    Host,
    /// On the device, by a kernel of `nd-fmm-kernels`.
    Device,
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Host => "host",
            Self::Device => "device",
        })
    }
}

/// A setting of [`FmmBuilder`] that [`FmmBuilder::build`] rejects.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SettingsError {
    /// The degree is larger than [`MAX_DEGREE`].
    #[error("p = {p} exceeds {max}, the largest degree CONVENTIONS §3.9 covers for M2L", max = MAX_DEGREE)]
    DegreeTooLarge {
        /// The requested degree.
        p: usize,
    },
    /// The maximum level is deeper than the deepest level of a Morton key.
    #[error("max_level = {max_level} exceeds the deepest level {deepest}", deepest = DEEPEST)]
    MaxLevelTooDeep {
        /// The requested maximum level.
        max_level: usize,
    },
    /// `max_points_per_leaf` is zero.
    #[error("max_points_per_leaf must be at least 1")]
    ZeroPointsPerLeaf,
    /// `threads` is zero.
    #[error("threads must be at least 1")]
    ZeroThreads,
    /// The P2P kernel is [`P2pChoice::Isa`] with an instruction set this machine cannot
    /// run ([`Isa::is_available`]).
    #[error("p2p_kernel: instruction set `{isa}` is not available on this machine")]
    P2pIsaUnavailable {
        /// The requested instruction set.
        isa: Isa,
    },
    /// The backend's cargo feature is not enabled in this build ([`Backend::is_compiled`]).
    #[error("backend {backend} is not compiled in (enable the `{backend}` feature)")]
    BackendNotCompiled {
        /// The requested backend.
        backend: Backend,
    },
    /// The backend is compiled in, but no device came up (no adapter, Metal inside the
    /// macOS sandbox, or Metal without the MSL compiler). The reason is not carried, so
    /// the error stays `Copy`; `Backend::probe` (feature `gpu`) returns it.
    #[error("no {backend} device could be opened")]
    NoDevice {
        /// The requested backend.
        backend: Backend,
    },
    /// The device does no arithmetic in the FMM's precision (f64 on Metal).
    #[error("the {backend} device does not support this precision")]
    PrecisionUnsupported {
        /// The requested backend.
        backend: Backend,
    },
    /// The device buffers of the FMM do not fit in the memory the device reports as
    /// available; nothing was allocated.
    #[error("the FMM needs {needed} bytes on the device, {limit} bytes are available")]
    DeviceMemory {
        /// The bytes of every device buffer.
        needed: u64,
        /// The bytes available.
        limit: u64,
    },
    /// [`KindTiming::Device`] on a backend that does not time on the device: the host,
    /// and the CPU runtime, whose timing windows wait for it (Phase 4S T5).
    #[error("kind_timings(Device) needs a device that times on itself; {backend} does not")]
    KindTimingUnsupported {
        /// The requested backend.
        backend: Backend,
    },
    /// [`OutputPass::Device`] on a backend without a device that does f64 arithmetic: the
    /// host, and Metal (Phase 4S T9).
    #[error("output_pass(Device) needs a device with f64 arithmetic; {backend} has none")]
    OutputPassUnsupported {
        /// The requested backend.
        backend: Backend,
    },
}

/// Which of the two point sets a point belongs to, for error messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointSet {
    /// The sources.
    Sources,
    /// The targets.
    Targets,
}

impl fmt::Display for PointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sources => "source",
            Self::Targets => "target",
        })
    }
}

/// Why an [`Fmm`] cannot be built or evaluated.
///
/// See the [module documentation](self#build) for which errors every rank returns and
/// which only the rank that found them.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FmmError {
    /// A builder setting is out of range.
    #[error("invalid settings: {0}")]
    InvalidSettings(#[from] SettingsError),
    /// The supplied domain is not a valid cubic domain.
    #[error("invalid domain: {0}")]
    InvalidDomain(#[from] GeometryError),
    /// A point has a NaN or infinite coordinate.
    #[error("{set} point {index} at {point:?} is not finite")]
    NonFinitePoint {
        /// The point set.
        set: PointSet,
        /// The point's position in the caller's slice.
        index: usize,
        /// The point.
        point: [f64; 3],
    },
    /// A point does not lie strictly inside the domain.
    #[error("{set} point {index} at {point:?} is not strictly inside the domain {domain:?}")]
    PointOutsideDomain {
        /// The point set.
        set: PointSet,
        /// The point's position in the caller's slice.
        index: usize,
        /// The point.
        point: [f64; 3],
        /// The corners `[xmin, ymin, zmin, xmax, ymax, zmax]` of the domain.
        domain: [f64; 6],
    },
    /// No rank has a source or a target.
    #[error("no rank has a source or a target point")]
    NoPoints,
    /// No domain is supplied and the points span no volume (they all coincide), or
    /// their extent overflows, so `compute_global_bounding_box` cannot build one.
    #[error("the points have extent {extent}, which is not positive and finite; supply a domain")]
    DegenerateExtent {
        /// The largest extent max_k (max xₖ − min xₖ) over all points of all ranks.
        extent: f64,
    },
    /// The points could not be moved to the ranks that own their leaves (step 5): the
    /// `Redistribution` of the sources or of the targets failed, on every rank alike
    /// (the rank that found the defect returns it, the others
    /// [`RedistributionError::OtherRank`]). From `Fmm` only
    /// [`RedistributionError::Overflow`] can arise: a rank that sends or receives so many
    /// points that their coordinates (three values each) or their output (up to four)
    /// overflow MPI's `i32` counts, about 5 × 10⁸ points; it is agreed on every rank. Its
    /// keys come from points strictly inside the domain, so they are valid, and its plan
    /// is its octree's (Phase 5 T6, docs/design/distributed-fmm.md §4.6).
    #[error("redistributing the points failed: {0}")]
    Redistribution(#[source] RedistributionError),
    /// The charge vector does not have one entry per source of this rank.
    #[error("{actual} charges given for {expected} sources")]
    ChargesLength {
        /// The number of sources on this rank.
        expected: usize,
        /// The number of charges given.
        actual: usize,
    },
    /// The plan could not be built.
    #[error("building the plan failed: {0}")]
    Plan(#[source] PlanError),
    /// The evaluator could not be built.
    #[error("building the evaluator failed: {0}")]
    Evaluator(#[source] EvaluatorError),
    /// More than one thread is requested, but MPI provides a threading level below
    /// `required` ([`Threading::Funneled`]); see [`threading`](crate::threading).
    #[error(
        "threads > 1 needs MPI at {required:?} or above, but it provides {provided:?}; \
         initialise MPI with mpi::initialize_with_threading(Threading::Funneled)"
    )]
    MpiThreading {
        /// The level more than one thread requires.
        required: Threading,
        /// The level MPI provides.
        provided: Threading,
    },
    /// The thread pool could not be built.
    #[error("building the thread pool failed: {0}")]
    ThreadPool(String),
    /// A device operation failed after the settings were accepted: an allocation or
    /// upload at build, or a launch or transfer of an evaluation. An `Fmm` whose
    /// evaluation failed returns it from every later evaluation.
    #[error("device error: {0}")]
    Device(String),
    /// The input is invalid on another rank.
    #[error("the input is invalid on another rank")]
    OtherRank,
}

/// The settings of an FMM, and [`build`](Self::build).
///
/// | Setting | Default |
/// | --- | --- |
/// | degree p ([`new`](Self::new)) | none; 0 ≤ p ≤ [`MAX_DEGREE`] |
/// | [`strategy`](Self::strategy) | [`M2lStrategy::Auto`]: `Dense` for p ≤ 8, `Rotation` above |
/// | [`gradients`](Self::gradients) | off |
/// | [`max_level`](Self::max_level) | [`DEFAULT_MAX_LEVEL`], 16 |
/// | [`max_points_per_leaf`](Self::max_points_per_leaf) | [`DEFAULT_MAX_POINTS_PER_LEAF`], 64 |
/// | [`domain`](Self::domain) | `compute_global_bounding_box` of all points |
/// | [`table_cache`](Self::table_cache) | none: tables are built |
/// | [`threads`](Self::threads) | 1: no pool, the calling thread |
/// | [`p2p_kernel`](Self::p2p_kernel) | [`P2pChoice::Auto`]: the kernel of `nd-fmm-simd` on the widest ISA of this machine |
/// | [`backend`](Self::backend) | [`Backend::Host`]: the host path |
/// | [`host_fallback`](Self::host_fallback) | none |
/// | [`synchronous_stages`](Self::synchronous_stages) | off |
/// | [`device_timestamps`](Self::device_timestamps) | off |
/// | [`kind_timings`](Self::kind_timings) | [`KindTiming::Off`] |
/// | [`output_pass`](Self::output_pass) | [`OutputPass::Auto`]: on the device where it does f64 arithmetic |
/// | [`device_p2p_layout`](Self::device_p2p_layout) | [`DeviceP2pLayout::Auto`]: by backend |
/// | [`device_leaf_layout`](Self::device_leaf_layout) | [`DeviceLeafLayout::Auto`]: by backend |
/// | [`device_gemm`](Self::device_gemm) | [`DeviceGemm::Auto`]: by precision, p and backend |
/// | [`device_scratch_budget`](Self::device_scratch_budget) | 128 MB; 2 GB on CUDA |
/// | [`tuning_cache`](Self::tuning_cache) | none: no tuning, the static rule |
/// | [`tuning_budget`](Self::tuning_budget) | 10 s |
///
/// For `T = f32`, p > 8 is accepted but lies beyond the useful range (design §4): the
/// error is then at the f32 floor already.
#[derive(Clone, Debug)]
pub struct FmmBuilder<T> {
    p: usize,
    strategy: M2lStrategy,
    gradients: bool,
    max_level: usize,
    max_points_per_leaf: usize,
    domain: Option<[f64; 6]>,
    table_cache: Option<PathBuf>,
    threads: usize,
    p2p: P2pChoice,
    backend: Backend,
    host_fallback: Vec<OperatorKind>,
    synchronous_stages: bool,
    device_timestamps: bool,
    kind_timing: KindTiming,
    output_pass: OutputPass,
    device_p2p_layout: DeviceP2pLayout,
    device_leaf_layout: DeviceLeafLayout,
    device_gemm: DeviceGemm,
    device_scratch_budget: Option<u64>,
    tuning_cache: Option<PathBuf>,
    tuning_budget: Duration,
    #[cfg(feature = "gpu")]
    tuning_hook: crate::tune::TuningHook,
    value: PhantomData<fn() -> T>,
}

impl<T> FmmBuilder<T> {
    /// The settings for degree `p`, with every other setting at its default.
    pub fn new(p: usize) -> Self {
        Self {
            p,
            strategy: M2lStrategy::Auto,
            gradients: false,
            max_level: DEFAULT_MAX_LEVEL,
            max_points_per_leaf: DEFAULT_MAX_POINTS_PER_LEAF,
            domain: None,
            table_cache: None,
            threads: 1,
            p2p: P2pChoice::Auto,
            backend: Backend::Host,
            host_fallback: Vec::new(),
            synchronous_stages: false,
            device_timestamps: false,
            kind_timing: KindTiming::Off,
            output_pass: OutputPass::Auto,
            device_p2p_layout: DeviceP2pLayout::Auto,
            device_leaf_layout: DeviceLeafLayout::Auto,
            device_gemm: DeviceGemm::Auto,
            device_scratch_budget: None,
            tuning_cache: None,
            tuning_budget: DEFAULT_TUNING_BUDGET,
            #[cfg(feature = "gpu")]
            tuning_hook: crate::tune::TuningHook::default(),
            value: PhantomData,
        }
    }

    /// Sets the M2L strategy, which also chooses the tables of M2M and L2L
    /// ([`M2lStrategy`]).
    pub fn strategy(mut self, strategy: M2lStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Requests the gradients ∇φ as well as the potentials.
    pub fn gradients(mut self, gradients: bool) -> Self {
        self.gradients = gradients;
        self
    }

    /// Sets the deepest level a leaf may lie on, 0–16.
    pub fn max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// Sets the refinement target: a box with more points is refined unless it lies on
    /// `max_level`, so leaves there may hold more. Passed to
    /// `OctreeOptions::with_max_fine_keys`, which counts distinct finest-level keys:
    /// a point that is a source and a target counts once, and so do points closer than
    /// a level-16 box.
    pub fn max_points_per_leaf(mut self, max_points_per_leaf: usize) -> Self {
        self.max_points_per_leaf = max_points_per_leaf;
        self
    }

    /// Supplies the domain: a cube ([`Domain::new`]) that contains every point strictly.
    /// Without it, the domain is `compute_global_bounding_box` of all points.
    pub fn domain(mut self, domain: PhysicalBox) -> Self {
        self.domain = Some(domain.coordinates());
        self
    }

    /// Loads the tables from the cache directory `dir`, building and storing those
    /// that are missing ([`Tables::load_or_build`]). Without it, no directory is used.
    pub fn table_cache(mut self, dir: impl Into<PathBuf>) -> Self {
        self.table_cache = Some(dir.into());
        self
    }

    /// Sets the number of threads of the level calls, at least 1 (the default).
    ///
    /// With n > 1, [`build`](Self::build) creates a rayon pool of n threads that the
    /// `Fmm` owns, and MPI must provide at least [`Threading::Funneled`]. The output is
    /// bit-identical for every n. Choose n so that ranks × n stays within the physical
    /// cores of a node ([`threading`](crate::threading)).
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Sets the P2P kernel ([`P2pChoice`]; default [`P2pChoice::Auto`]).
    ///
    /// [`build`](Self::build) rejects [`P2pChoice::Isa`] with an ISA this machine cannot
    /// run with [`SettingsError::P2pIsaUnavailable`]. The check is local, so on a
    /// cluster with mixed CPUs one rank may fail it; like every input error it is
    /// agreed by all ranks in step 1, and the others return [`FmmError::OtherRank`].
    /// The choice changes the output only in the last bits (the accuracy of
    /// [`P2pChoice`]); [`Fmm::p2p_kernel`] reports the kernel that runs.
    pub fn p2p_kernel(mut self, choice: P2pChoice) -> Self {
        self.p2p = choice;
        self
    }

    /// Sets where the operators run ([`Backend`]; default [`Backend::Host`], the host
    /// path, which no other setting of a device backend changes).
    ///
    /// [`build`](Self::build) refuses a device backend that is not compiled in
    /// ([`SettingsError::BackendNotCompiled`]), whose device cannot be opened
    /// ([`SettingsError::NoDevice`]) or does not do arithmetic in `T`
    /// ([`SettingsError::PrecisionUnsupported`], f64 on Metal), or whose buffers do not
    /// fit in device memory ([`SettingsError::DeviceMemory`]). It runs on any number of
    /// ranks, every rank opening the default device of the backend, which the ranks of a
    /// node share (Phase 5 T8; the `device` module, "Several ranks"). With
    /// [`Backend::Cpu`], `threads(n)` builds no rayon pool and caps the units per cube of
    /// the CPU runtime instead, at most the node's cores over its ranks (the `device`
    /// module, "Threads").
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Runs these operator kinds on the host fallback even with a device backend: a
    /// test aid (requirement 8 of docs/phase4/README.md). From Phase 4 T10 every kind has
    /// a device kernel under every strategy. Ignored by [`Backend::Host`].
    pub fn host_fallback(mut self, kinds: impl IntoIterator<Item = OperatorKind>) -> Self {
        self.host_fallback = kinds.into_iter().collect();
        self
    }

    /// Sets the layout of the device P2P kernel ([`DeviceP2pLayout`]; default
    /// [`DeviceP2pLayout::Auto`], by backend). [`build`](Self::build) returns
    /// [`FmmError::Device`] if the device cannot run it (more units or shared memory than
    /// it has, or a plane layout on a device whose plane size varies). Fixed at build and
    /// reported by `Fmm::device_report` (feature `gpu`). Ignored by [`Backend::Host`] and
    /// when P2P runs on the host fallback.
    pub fn device_p2p_layout(mut self, layout: DeviceP2pLayout) -> Self {
        self.device_p2p_layout = layout;
        self
    }

    /// Sets the layout of the device leaf operators P2M, L2P, P2L and M2P
    /// ([`DeviceLeafLayout`]; default [`DeviceLeafLayout::Auto`], by backend).
    /// [`build`](Self::build) returns [`FmmError::Device`] if the device cannot run it at
    /// the FMM's degree (more units or shared memory than it has). Fixed at build and
    /// reported by `Fmm::device_report` (feature `gpu`). Ignored by [`Backend::Host`] and
    /// when all four run on the host fallback.
    pub fn device_leaf_layout(mut self, layout: DeviceLeafLayout) -> Self {
        self.device_leaf_layout = layout;
        self
    }

    /// Sets the GEMM of the device translations M2M, L2L and dense M2L ([`DeviceGemm`];
    /// default [`DeviceGemm::Auto`]). Fixed at build and reported by `Fmm::device_report`
    /// (feature `gpu`). Ignored by [`Backend::Host`] and when all three run on the host
    /// fallback.
    pub fn device_gemm(mut self, gemm: DeviceGemm) -> Self {
        self.device_gemm = gemm;
        self
    }

    /// Sets the scratch budget of the device translations in bytes: the gathered inputs
    /// and the products of one chunk of a level call together (default 128 MB,
    /// docs/design/device-path.md §6.4; on CUDA 2 GB, at most an eighth of the device's
    /// memory, measured in Phase 4S T7, `nd_fmm_kernels::translate::default_scratch_bytes`).
    /// A level call whose batches need more runs in several chunks, with the same bits.
    /// Allocated once at build, no larger than the widest chunk, and counted in the device
    /// memory check. Ignored by [`Backend::Host`].
    pub fn device_scratch_budget(mut self, bytes: u64) -> Self {
        self.device_scratch_budget = Some(bytes);
        self
    }

    /// With a device backend, synchronises with the device after the charge upload and
    /// after every stage, so that [`StageTimings`] time each stage rather than its
    /// enqueueing (docs/design/device-path.md §8.3). Seven more syncs per evaluation, and
    /// the same output: for reports only, off by default. Ignored by [`Backend::Host`].
    pub fn synchronous_stages(mut self, on: bool) -> Self {
        self.synchronous_stages = on;
        self
    }

    /// With a device backend that times on the device itself (Metal, CUDA; not the CPU
    /// runtime, whose timing windows wait for it), times every stage with device work by
    /// the device's timestamps: one timing window per stage, resolved after the
    /// evaluation's one download, with no sync of its own ([`DeviceStageTimings`],
    /// docs/design/device-path.md §8.3). Off by default: on Metal the windows of
    /// neighbouring stages overlap, so the stage times are spans, not a breakdown of the
    /// evaluation, and the windows add some enqueue time (the `device` module, "Stage
    /// timing"). The output is the same either way. Ignored by [`Backend::Host`], on the
    /// CPU runtime and with [`synchronous_stages`](Self::synchronous_stages).
    pub fn device_timestamps(mut self, on: bool) -> Self {
        self.device_timestamps = on;
        self
    }

    /// Times every level call by operator kind ([`KindTiming`]; default
    /// [`KindTiming::Off`]; Phase 4S T5), into [`StageTimings::kinds`]. For reports only:
    /// the output is the same bit for bit in every mode, and only the syncs and windows
    /// [`KindTiming`] documents are added.
    ///
    /// It composes with [`synchronous_stages`](Self::synchronous_stages) and
    /// [`device_timestamps`](Self::device_timestamps) and implies neither: each adds its
    /// own syncs or windows. [`build`](Self::build) refuses [`KindTiming::Device`] on a
    /// backend that does not time on the device (the host and the CPU runtime) with
    /// [`SettingsError::KindTimingUnsupported`], agreed by step 1's all-reduce.
    pub fn kind_timings(mut self, mode: KindTiming) -> Self {
        self.kind_timing = mode;
        self
    }

    /// Sets where the output pass runs ([`OutputPass`]; default [`OutputPass::Auto`], on
    /// the device where it does f64 arithmetic; Phase 4S T9, decision 12). The output is
    /// the same bit for bit either way; [`Fmm::output_pass`] reports where it runs.
    /// [`build`](Self::build) refuses [`OutputPass::Device`] on the host backend and on a
    /// device without f64 arithmetic (Metal) with [`SettingsError::OutputPassUnsupported`],
    /// agreed by step 1's all-reduce.
    pub fn output_pass(mut self, pass: OutputPass) -> Self {
        self.output_pass = pass;
        self
    }

    /// With a device backend, tunes the device path's choices against the timings of this
    /// device and keeps them in the directory `dir` (Phase 4 T12, C4.7;
    /// docs/design/device-path.md §10): the M2L strategy under [`M2lStrategy::Auto`], the
    /// GEMM of the M2M, L2L and dense M2L level calls under [`DeviceGemm::Auto`], and the
    /// P2P layout under [`DeviceP2pLayout::Auto`]. Decisions the cache file of this
    /// (backend, device, precision, p) holds are taken from it; the others are timed at
    /// build, within [`tuning_budget`](Self::tuning_budget), and stored. Without it, no
    /// directory is used and no tuning runs: the static rule applies (the `tune` module,
    /// feature `gpu`). No default directory, no environment variable. Ignored by
    /// [`Backend::Host`].
    pub fn tuning_cache(mut self, dir: impl Into<PathBuf>) -> Self {
        self.tuning_cache = Some(dir.into());
        self
    }

    /// Sets the time the tuner may spend in one build (default 10 s; Phase 4 T12): checked
    /// before every candidate and between its timed batches, so that no candidate starts
    /// past it. Decisions it leaves untuned take the static rule and are not stored.
    /// Ignored without [`tuning_cache`](Self::tuning_cache).
    pub fn tuning_budget(mut self, budget: Duration) -> Self {
        self.tuning_budget = budget;
        self
    }

    /// A test aid for the tuner (Phase 4 T12; feature `gpu`): `hook` adjusts measured
    /// times and offers extra candidates, which go through registration like the others
    /// (`tune::TuningHook`). Not for production runs.
    #[cfg(feature = "gpu")]
    pub fn tuning_hook(mut self, hook: crate::tune::TuningHook) -> Self {
        self.tuning_hook = hook;
        self
    }

    /// Checks the settings that do not depend on the input; whether this machine can
    /// run the P2P kernel depends on the rank's CPU.
    fn check_settings(&self) -> Result<(), SettingsError> {
        if self.p > MAX_DEGREE {
            return Err(SettingsError::DegreeTooLarge { p: self.p });
        }
        if self.max_level > DEEPEST {
            return Err(SettingsError::MaxLevelTooDeep {
                max_level: self.max_level,
            });
        }
        if self.max_points_per_leaf == 0 {
            return Err(SettingsError::ZeroPointsPerLeaf);
        }
        if self.threads == 0 {
            return Err(SettingsError::ZeroThreads);
        }
        if let Err(error) = self.p2p.resolve() {
            return Err(SettingsError::P2pIsaUnavailable { isa: error.isa });
        }
        Ok(())
    }

    /// The rayon threads of the level calls: `threads`, but 1 with [`Backend::Cpu`],
    /// which builds no pool (docs/design/device-path.md §11).
    fn rayon_threads(&self) -> usize {
        if self.backend == Backend::Cpu {
            1
        } else {
            self.threads
        }
    }

    /// With more than one rayon thread: checks that MPI provides `provided` ≥
    /// [`REQUIRED_MPI_THREADING`] and builds the pool.
    fn pool(&self, provided: Threading) -> Result<Option<Arc<ThreadPool>>, FmmError> {
        if self.rayon_threads() == 1 {
            return Ok(None);
        }
        if provided < REQUIRED_MPI_THREADING {
            return Err(FmmError::MpiThreading {
                required: REQUIRED_MPI_THREADING,
                provided,
            });
        }
        ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .thread_name(|i| format!("nd-fmm-exec-{i}"))
            .build()
            .map(|pool| Some(Arc::new(pool)))
            .map_err(|error| FmmError::ThreadPool(error.to_string()))
    }
}

impl<T: Stored + SimdScalar + Equivalence + Default> FmmBuilder<T> {
    /// Builds the FMM of `sources` and `targets`, in the caller's order; see the
    /// [module documentation](self#build).
    ///
    /// The two sets may overlap or coincide. A rank may pass no points.
    ///
    /// # Collective operation
    ///
    /// Every rank of `comm` must call it, with the same settings. Its collectives are
    /// those of the module documentation, then those of `Octree::new`, `Plan::new` and
    /// `Evaluator::new`.
    ///
    /// # Errors
    ///
    /// Every [`FmmError`] but [`ChargesLength`](FmmError::ChargesLength); see the
    /// module documentation for which ranks return which. With `threads` > 1,
    /// [`FmmError::MpiThreading`] if MPI provides less than [`Threading::Funneled`].
    pub fn build<'o, C: CommunicatorCollectives>(
        &self,
        sources: &[[f64; 3]],
        targets: &[[f64; 3]],
        comm: &'o C,
    ) -> Result<Fmm<'o, T, C>, FmmError> {
        let start = Instant::now();
        // Step 1: settings, MPI threading and the pool, the supplied domain, finite
        // points.
        //
        // The MPI level is the same on every rank in practice (one library, and every
        // rank asks for the same level), and so are the settings, so the threading check
        // needs no collective of its own; it rides on this step's agreement, which also
        // covers a pool that fails to build on one rank. Every collective of the build
        // and of `evaluate` runs on this thread; the pool's threads never call MPI.
        //
        // A device backend is opened here, so that a backend not compiled in, a device
        // that does not come up and a precision it does not support are input errors of
        // this rank, agreed by the same all-reduce (docs/design/device-path.md §5.2).
        let provided = mpi::environment::threading_support();
        let mut open_time = Duration::ZERO;
        let local = self
            .check_settings()
            .map_err(FmmError::from)
            .and_then(|()| {
                let pool = self.pool(provided)?;
                check_finite(sources, PointSet::Sources)?;
                check_finite(targets, PointSet::Targets)?;
                let supplied = self.domain.map(|c| Domain::new(&PhysicalBox::new(c)));
                let supplied = supplied.transpose()?;
                let start = Instant::now();
                let device = open_device::<T>(self.backend, self.threads)?;
                open_time = start.elapsed();
                let on_device = device.as_ref().is_some_and(times_on_device);
                check_kind_timing(self.kind_timing, self.backend, on_device)?;
                let f64_device = device.as_ref().is_some_and(does_f64);
                check_output_pass(self.output_pass, self.backend, f64_device)?;
                Ok((pool, supplied, device))
            });
        let (pool, supplied, mut device) = agree(comm, local)?;
        let threading = ThreadingReport::read(self.rayon_threads(), provided);
        // A device per rank (Phase 5 T8, docs/design/distributed-fmm.md §7.4): every rank of
        // a device build opens the default device of the backend (above), and the ranks of
        // a node share it. One split of the communicator by node places the rank there;
        // every rank has the same backend (agreed above), so every rank splits or none.
        let ranks = self.place_rank(comm, device.as_mut());

        // Step 2: are there points at all?
        let mut total = 0u64;
        comm.all_reduce_into(
            &((sources.len() + targets.len()) as u64),
            &mut total,
            SystemOperation::sum(),
        );
        if total == 0 {
            return Err(FmmError::NoPoints);
        }

        // Step 3: the domain, and every point strictly inside it.
        let coordinates: Vec<f64> = sources.iter().chain(targets).flatten().copied().collect();
        let points = SliceArray::from_shape(&coordinates, [3, sources.len() + targets.len()]);
        let domain = match supplied {
            Some(domain) => domain,
            None => {
                let extent = global_extent(sources, targets, comm);
                if !(extent > 0.0 && extent.is_finite()) {
                    return Err(FmmError::DegenerateExtent { extent });
                }
                // The same box on every rank, so the same outcome.
                Domain::new(&compute_global_bounding_box(&points, comm))?
            }
        };
        agree(comm, check_inside(sources, targets, &domain))?;
        let domain_time = start.elapsed();

        // Step 4: the octree of all points, and its plan.
        let start = Instant::now();
        let keys = points_to_morton(&points, DEEPEST, &domain.physical_box());
        let options = OctreeOptions::new()
            .with_max_level(self.max_level)
            .with_max_fine_keys(self.max_points_per_leaf)
            .with_ghost_children(true);
        let octree = Octree::new(&keys, options, comm);
        let octree_time = start.elapsed();
        let start = Instant::now();
        let plan = Plan::new(&octree).map_err(FmmError::Plan)?;
        let plan_time = start.elapsed();

        // Step 5: every point to the rank that owns its leaf (Phase 5 T6,
        // docs/design/distributed-fmm.md §4.4): one redistribution for the sources and one
        // for the targets, routed by the keys of step 4, and the f64 coordinates forwarded
        // into the received order, which is leaf order. Each `new` agrees its own errors,
        // so every rank returns here alike.
        let start = Instant::now();
        let nlocal = plan.index().leaves().nlocal();
        let (source_keys, target_keys) = keys.split_at(sources.len());
        let source_route =
            Redistribution::new(&octree, &plan, source_keys).map_err(FmmError::Redistribution)?;
        let target_route =
            Redistribution::new(&octree, &plan, target_keys).map_err(FmmError::Redistribution)?;
        // Three coordinates per point move now, and up to four output values per target on
        // every evaluation. `max_per_item` is the same on every rank, so is this outcome.
        let per_item = if self.gradients { 4 } else { 3 };
        if source_route.max_per_item().min(target_route.max_per_item()) < per_item {
            return Err(FmmError::Redistribution(RedistributionError::Overflow));
        }
        let (source_coordinates, target_coordinates) = coordinates.split_at(3 * sources.len());
        let owned_sources = source_route.forward(source_coordinates, 3);
        let owned_targets = target_route.forward(target_coordinates, 3);

        // Step 6: the points of every local leaf, its radius, and the output order.
        let source_leaves = LeafRanges::new(source_route.counts());
        let target_leaves = LeafRanges::new(target_route.counts());
        let radii: Vec<f64> = (0..nlocal)
            .map(|j| radius(plan.index().leaves().level(j), &domain))
            .collect();
        let outputs = OutputOrder::new(&target_leaves, &radii);
        let redistribute_time = start.elapsed();

        // Step 7: tables, operator, evaluator.
        let start = Instant::now();
        let local_max = source_leaves.counts.iter().copied().max().unwrap_or(0);
        let mut max_leaf_points = 0usize;
        comm.all_reduce_into(&local_max, &mut max_leaf_points, SystemOperation::max());
        // With a device backend, the tuner (Phase 4 T12) and the M2L strategy: under
        // `Auto` with M2L on the device, the cached, tuned or static choice of the device
        // path, for which the tables are built; on several ranks with a tuning cache rank
        // 0's, broadcast (Phase 5 T8).
        let tuning_start = Instant::now();
        let (strategy, tuner) = self.device_strategy(device.as_mut(), &plan, comm);
        let tuning_time = tuning_start.elapsed();
        let start = start + tuning_time;
        let (tables, cache_outcomes) = match &self.table_cache {
            Some(dir) => Tables::load_or_build(self.p, strategy, &TableCache::new(dir)),
            None => (Tables::build(self.p, strategy), Vec::new()),
        };
        let tables_time = start.elapsed();
        let start = Instant::now();
        let mut operator = LaplaceOperator::new(tables, self.gradients, max_leaf_points)
            .with_p2p(self.p2p)
            .expect("step 1 checked that this machine runs the P2P kernel");
        if let Some(pool) = pool {
            operator = operator.with_pool(pool);
        }
        // The evaluator is built with the host engine; a device engine replaces it below,
        // once the evaluator's exchanges know the ghost leaves and the slots they move.
        let operator = ExecOperator::new(Engine::Host(operator), self.kind_timing, false);
        let mut evaluator = Evaluator::new(
            plan,
            comm,
            operator,
            &source_leaves.counts,
            &target_leaves.counts,
        )
        .map_err(FmmError::Evaluator)?;
        let evaluator_time = start.elapsed();

        // Step 8: the leaf-scaled coordinates, once, from the forwarded coordinates in
        // leaf order (CONVENTIONS §3.13).
        let start = Instant::now();
        let leaf_keys = evaluator.plan().index().leaves().keys()[..nlocal].to_vec();
        let as_point = |values: &[f64]| [values[0], values[1], values[2]];
        for (j, &key) in leaf_keys.iter().enumerate() {
            let chunk = evaluator.sources_mut(j);
            let owned = &owned_sources[3 * source_leaves.offsets[j]..];
            for (u, x) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(owned.chunks(3)) {
                *u = leaf_coordinates(as_point(x), key, &domain);
            }
            let chunk = evaluator.target_input_mut(j);
            let owned = &owned_targets[3 * target_leaves.offsets[j]..];
            for (u, x) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(owned.chunks(3)) {
                *u = leaf_coordinates(as_point(x), key, &domain);
            }
        }
        let load_time = start.elapsed();

        // With a device backend (Phase 5 T8, docs/design/distributed-fmm.md §7.1): the
        // device operator, built from the evaluator's source counts (the ghost leaves'
        // included) and exchanges, swapped in for the host engine (the sizes are the same),
        // and the points uploaded. A failure on any rank is agreed by one all-reduce, so
        // that every rank returns here alike.
        let start = Instant::now();
        let mut device_time = open_time + tuning_time;
        if let Some(device) = device {
            let attached = self.attach_device(
                &mut evaluator,
                device,
                &target_leaves.counts,
                &outputs,
                tuner,
                ranks,
            );
            agree(comm, attached)?;
        }
        device_time += start.elapsed();

        // The host's source store feeds the host path and host-fallback calls of the kinds
        // that read sources; with those on the device, the charges go to the device alone
        // (Phase 4S T9), but for the local leaves the source exchange sends, whose charges
        // the exchange reads from the host store (Phase 5 T8).
        let engine = &evaluator.operator().engine;
        let host_sources = !engine.is_device()
            || [OperatorKind::P2m, OperatorKind::P2l, OperatorKind::P2p]
                .into_iter()
                .any(|kind| engine.placement(kind) == Placement::Host);
        let sent_leaves = if host_sources {
            Vec::new()
        } else {
            let mut leaves = evaluator.source_exchange().send_leaves().to_vec();
            leaves.sort_unstable();
            leaves.dedup();
            leaves
        };

        // The values per target the output pass forms and the backward move returns: φ,
        // or φ and ∇φ.
        let output_values = if self.gradients { 4 } else { 1 };
        Ok(Fmm {
            octree,
            evaluator,
            domain,
            strategy: strategy.resolve(self.p),
            backend: self.backend,
            synchronous_stages: self.synchronous_stages && self.backend.is_device(),
            host_sources,
            sent_leaves,
            device_error: None,
            charges: vec![T::zero(); source_route.nreceived()],
            received_output: vec![T::zero(); output_values * target_route.nreceived()],
            returned_output: vec![T::zero(); output_values * target_route.nsent()],
            sources: source_route,
            targets: target_route,
            source_leaves,
            target_leaves,
            outputs,
            radii,
            max_leaf_points,
            cache_outcomes,
            threading,
            build_timings: BuildTimings {
                domain: domain_time.saturating_sub(open_time),
                octree: octree_time,
                plan: plan_time,
                redistribute: redistribute_time,
                tables: tables_time,
                evaluator: evaluator_time,
                load: load_time,
                device: device_time,
            },
        })
    }

    /// With a device backend (feature `gpu`): where this rank sits among the ranks and on
    /// its node (Phase 5 T8, docs/design/distributed-fmm.md §7.4), from one
    /// `split_shared` of `comm`; on the CPU runtime it caps the units per cube of `device`
    /// at `threads(n)` and at the node's cores over its ranks. Nothing on the host.
    #[cfg(feature = "gpu")]
    fn place_rank<C: CommunicatorCollectives>(
        &self,
        comm: &C,
        device: Option<&mut OpenedDevice>,
    ) -> device::RankPlacement {
        let Some(device) = device else {
            return device::RankPlacement::default();
        };
        let node = comm.split_shared(comm.rank());
        let placement = device::RankPlacement {
            rank: comm.rank() as usize,
            ranks: comm.size() as usize,
            local_rank: node.rank() as usize,
            local_ranks: node.size() as usize,
            cores: std::thread::available_parallelism().map_or(1, usize::from),
            threads: self.threads,
            follows_rank0: self.tuning_cache.is_some() && comm.size() > 1 && comm.rank() != 0,
        };
        device.limit_units(u32::try_from(placement.cpu_units()).unwrap_or(u32::MAX));
        placement
    }

    /// Without the `gpu` feature no device is opened.
    #[cfg(not(feature = "gpu"))]
    fn place_rank<C: CommunicatorCollectives>(
        &self,
        comm: &C,
        device: Option<&mut OpenedDevice>,
    ) -> RankPlacement {
        let _ = comm;
        if let Some(never) = device {
            match *never {}
        }
        RankPlacement
    }

    /// Replaces the host engine of `evaluator` by the device operator around its host
    /// operator (Phase 5 T8, docs/design/distributed-fmm.md §7.1): built on `device` from
    /// the evaluator's source counts (every leaf of the numbering, the ghost leaves'
    /// included), the index lists of its exchanges and `target_counts`, and loaded with the
    /// points of the evaluator's stores. Device work only, no collective.
    ///
    /// # Errors
    ///
    /// As [`DeviceOperator::new`](crate::device::DeviceOperator::new) and `load_points`;
    /// the evaluator is left with a placeholder engine, to be dropped.
    #[cfg(feature = "gpu")]
    fn attach_device<C: CommunicatorCollectives>(
        &self,
        evaluator: &mut FmmEvaluator<'_, C, T>,
        device: OpenedDevice,
        target_counts: &[usize],
        outputs: &OutputOrder,
        tuner: Option<Tuner>,
        ranks: device::RankPlacement,
    ) -> Result<(), FmmError> {
        let options = DeviceOptions {
            host_fallback: self.host_fallback.clone(),
            table_cache: self.table_cache.clone(),
            p2p_layout: self.device_p2p_layout,
            leaf_layout: self.device_leaf_layout,
            gemm: self.device_gemm,
            scratch_budget: self.device_scratch_budget,
            stage_timing: if self.synchronous_stages {
                device::StageTiming::Synchronous
            } else if self.device_timestamps {
                device::StageTiming::DeviceTimestamps
            } else {
                device::StageTiming::Enqueue
            },
            output_pass: self.output_pass,
            ranks,
        };
        let tuner = tuner.expect("a device build has a tuner");
        let on_device = device.times_on_device();
        let caller = device::CallerOrder {
            points: &outputs.points,
            leaves: &outputs.leaves,
            scales: outputs.scales.as_flattened(),
        };
        // The host operator moves into the device operator; until the swap the evaluator
        // holds a placeholder of degree 0, never called.
        let engine = std::mem::replace(
            &mut evaluator.operator_mut().engine,
            Engine::Host(detached_operator()),
        );
        let Engine::Host(host) = engine else {
            unreachable!("the evaluator is built with the host engine")
        };
        let lists =
            device::ExchangeLists::new(evaluator.multipole_exchange(), evaluator.coarse_exchange());
        let counts = (evaluator.source_exchange().leaf_counts(), target_counts);
        let driver = device::driver(
            host,
            device,
            evaluator.plan(),
            counts,
            &lists,
            caller,
            &options,
            tuner,
        )?;
        *evaluator.operator_mut() =
            ExecOperator::new(Engine::Device(driver), self.kind_timing, on_device);
        // The device's copy of the points, from copies of the stores: the store accessors
        // and `operator_mut` cannot borrow the evaluator at once (device-path.md §4.3). The
        // source store holds every leaf of the numbering; its ghost tail is zero until the
        // first source exchange.
        let stores = (
            evaluator.source_store().clone(),
            evaluator.target_input_store().clone(),
        );
        evaluator
            .operator_mut()
            .engine
            .load_points(&stores.0, &stores.1)
    }

    /// Without the `gpu` feature no device is opened.
    #[cfg(not(feature = "gpu"))]
    fn attach_device<C: CommunicatorCollectives>(
        &self,
        evaluator: &mut FmmEvaluator<'_, C, T>,
        device: OpenedDevice,
        target_counts: &[usize],
        outputs: &OutputOrder,
        tuner: Option<Tuner>,
        ranks: RankPlacement,
    ) -> Result<(), FmmError> {
        let _ = (evaluator, target_counts, outputs, tuner, ranks);
        match device {}
    }

    /// With a device: the tuner of the build (Phase 4 T12) and the M2L strategy to build
    /// the tables for: the builder's, or under `Auto` with M2L on the device the device
    /// path's (cached, tuned, or the static rule; `tune::static_strategy`). On the host,
    /// or with M2L on the host fallback, the builder's strategy, which the tables resolve
    /// by the host rule.
    ///
    /// On several ranks with a tuning cache (Phase 5 T8, docs/design/distributed-fmm.md
    /// §7.6) rank 0 alone tunes, reads and writes the cache, and broadcasts its strategy
    /// (one broadcast, on every rank); the other ranks take it, and the static rule for
    /// the GEMMs and the P2P layout, so that two runs with the same cache give the same
    /// bits on every rank and no rank times while another tunes on the shared device.
    #[cfg(feature = "gpu")]
    fn device_strategy<C: CommunicatorCollectives>(
        &self,
        device: Option<&mut OpenedDevice>,
        plan: &Plan,
        comm: &C,
    ) -> (M2lStrategy, Option<Tuner>) {
        let Some(device) = device else {
            return (self.strategy, None);
        };
        let key = crate::tune::TuningKey::new(
            device.info(),
            device::precision_of::<T>(),
            self.p,
            self.gradients,
        );
        let shared = self.tuning_cache.is_some() && comm.size() > 1;
        let follower = shared && comm.rank() != 0;
        let mut tuner = Tuner::new(
            key,
            self.tuning_cache.as_deref().filter(|_| !follower),
            self.tuning_budget,
            self.tuning_hook.clone(),
        );
        let tuned =
            self.strategy == M2lStrategy::Auto && !self.host_fallback.contains(&OperatorKind::M2l);
        let mut strategy = if tuned && !follower {
            device::tune_strategy::<T>(
                &mut tuner,
                device,
                plan,
                self.p,
                self.table_cache.as_deref(),
                self.device_scratch_budget,
            )
        } else {
            self.strategy
        };
        if shared {
            let mut code = strategy_code(strategy);
            comm.process_at_rank(0).broadcast_into(&mut code);
            strategy = strategy_from_code(code);
        }
        (strategy, Some(tuner))
    }

    /// Without the `gpu` feature: the builder's strategy, no tuner.
    #[cfg(not(feature = "gpu"))]
    fn device_strategy<C: CommunicatorCollectives>(
        &self,
        device: Option<&mut OpenedDevice>,
        plan: &Plan,
        comm: &C,
    ) -> (M2lStrategy, Option<Tuner>) {
        let _ = (device, plan, comm);
        (self.strategy, None)
    }
}

/// The M2L strategy as one byte, for the broadcast of rank 0's tuned strategy.
#[cfg(feature = "gpu")]
fn strategy_code(strategy: M2lStrategy) -> u8 {
    match strategy {
        M2lStrategy::Dense => 0,
        M2lStrategy::Classes => 1,
        M2lStrategy::Rotation => 2,
        M2lStrategy::Auto => 3,
    }
}

/// The M2L strategy of [`strategy_code`].
#[cfg(feature = "gpu")]
fn strategy_from_code(code: u8) -> M2lStrategy {
    match code {
        0 => M2lStrategy::Dense,
        1 => M2lStrategy::Classes,
        2 => M2lStrategy::Rotation,
        _ => M2lStrategy::Auto,
    }
}

/// The operator the evaluator holds while its host operator moves into the device
/// operator ([`FmmBuilder::attach_device`]): degree 0, built in microseconds, never
/// called.
#[cfg(feature = "gpu")]
fn detached_operator<T: SimdScalar + Stored>() -> LaplaceOperator<T> {
    LaplaceOperator::new(Tables::build(0, M2lStrategy::Dense), false, 1)
}

/// Where a rank sits, without the `gpu` feature: no device, so nothing to say.
#[cfg(not(feature = "gpu"))]
#[derive(Clone, Copy, Debug)]
struct RankPlacement;

/// The tuner of a device build (feature `gpu`); without it, no value.
#[cfg(feature = "gpu")]
type Tuner = crate::tune::Tuner;
#[cfg(not(feature = "gpu"))]
type Tuner = std::convert::Infallible;

/// An opened device: `nd_fmm_kernels::Device` with the `gpu` feature; without it no
/// device can be opened, and the type has no value.
#[cfg(feature = "gpu")]
type OpenedDevice = nd_fmm_kernels::Device;
#[cfg(not(feature = "gpu"))]
type OpenedDevice = std::convert::Infallible;

/// Opens the device of `backend` at step 1 of the build; `None` for the host.
#[cfg(feature = "gpu")]
fn open_device<T: Stored>(
    backend: Backend,
    threads: usize,
) -> Result<Option<OpenedDevice>, SettingsError> {
    device::open_device::<T>(backend, threads)
}

/// Refuses [`KindTiming::Device`] on `backend` unless its device times on itself
/// (`on_device`).
fn check_kind_timing(
    mode: KindTiming,
    backend: Backend,
    on_device: bool,
) -> Result<(), SettingsError> {
    if mode == KindTiming::Device && !on_device {
        return Err(SettingsError::KindTimingUnsupported { backend });
    }
    Ok(())
}

/// Refuses [`OutputPass::Device`] on `backend` unless its device does f64 arithmetic
/// (`f64_device`).
fn check_output_pass(
    pass: OutputPass,
    backend: Backend,
    f64_device: bool,
) -> Result<(), SettingsError> {
    if pass == OutputPass::Device && !f64_device {
        return Err(SettingsError::OutputPassUnsupported { backend });
    }
    Ok(())
}

/// Whether `device` does f64 arithmetic (CUDA, the CPU runtime), for the output pass on the
/// device ([`OutputPass`]).
#[cfg(feature = "gpu")]
fn does_f64(device: &OpenedDevice) -> bool {
    device.supports(nd_fmm_kernels::Precision::F64)
}

/// Without the `gpu` feature no device is opened.
#[cfg(not(feature = "gpu"))]
fn does_f64(device: &OpenedDevice) -> bool {
    match *device {}
}

/// Whether `device` times on itself (`nd_fmm_kernels::Device::times_on_device`: Metal,
/// CUDA), for [`KindTiming::Device`].
#[cfg(feature = "gpu")]
fn times_on_device(device: &OpenedDevice) -> bool {
    device.times_on_device()
}

/// Without the `gpu` feature no device is opened.
#[cfg(not(feature = "gpu"))]
fn times_on_device(device: &OpenedDevice) -> bool {
    match *device {}
}

/// Without the `gpu` feature only the host runs: every device backend is not compiled
/// in.
#[cfg(not(feature = "gpu"))]
#[expect(
    clippy::extra_unused_type_parameters,
    reason = "the signature of the `gpu` build, which checks T's precision"
)]
fn open_device<T: Stored>(
    backend: Backend,
    _threads: usize,
) -> Result<Option<OpenedDevice>, SettingsError> {
    match backend {
        Backend::Host => Ok(None),
        _ => Err(SettingsError::BackendNotCompiled { backend }),
    }
}

/// The evaluator of an [`Fmm`], which owns its plan.
type FmmEvaluator<'o, C, T> = Evaluator<'o, C, ExecOperator<T>, Plan>;

/// Where the level calls of the evaluator run: [`LaplaceOperator`] on the host path, or
/// the device operator (feature `gpu`), which wraps one for its host fallback
/// (docs/design/device-path.md §3.3). Without `gpu` the enum has the host variant only.
#[cfg_attr(
    feature = "gpu",
    expect(
        clippy::large_enum_variant,
        reason = "one per Fmm; boxing the host operator would add an indirection to every \
                  level call of the host path"
    )
)]
enum Engine<T: SimdScalar + Stored + Equivalence + Default> {
    Host(LaplaceOperator<T>),
    #[cfg(feature = "gpu")]
    Device(Box<dyn DeviceDriver<T>>),
}

impl<T: SimdScalar + Stored + Equivalence + Default> Engine<T> {
    /// The host operator, or the device operator's fallback operator.
    fn host(&self) -> &LaplaceOperator<T> {
        match self {
            Self::Host(operator) => operator,
            #[cfg(feature = "gpu")]
            Self::Device(driver) => driver.host(),
        }
    }

    /// The host operator, mutably.
    fn host_mut(&mut self) -> &mut LaplaceOperator<T> {
        match self {
            Self::Host(operator) => operator,
            #[cfg(feature = "gpu")]
            Self::Device(driver) => driver.host_mut(),
        }
    }

    /// True for the device operator.
    fn is_device(&self) -> bool {
        !matches!(self, Self::Host(_))
    }

    /// Uploads the points to the device; nothing on the host.
    #[cfg(feature = "gpu")]
    fn load_points(
        &mut self,
        sources: &LeafStore<T>,
        target_input: &LeafStore<T>,
    ) -> Result<(), FmmError> {
        match self {
            Self::Host(_) => Ok(()),
            Self::Device(driver) => driver.load_points(sources, target_input),
        }
    }

    /// Waits for the device; nothing on the host.
    fn sync(&mut self) {
        match self {
            Self::Host(_) => {}
            #[cfg(feature = "gpu")]
            Self::Device(driver) => driver.sync(),
        }
    }

    /// Opens the timing window of a stage on the device, if it times stages; nothing on
    /// the host.
    fn open_stage(&mut self) {
        match self {
            Self::Host(_) => {}
            #[cfg(feature = "gpu")]
            Self::Device(driver) => driver.open_stage(),
        }
    }

    /// Closes the timing window of `stage`; nothing on the host.
    fn close_stage(&mut self, stage: DeviceStage) {
        match self {
            Self::Host(_) => {
                let _ = stage;
            }
            #[cfg(feature = "gpu")]
            Self::Device(driver) => driver.close_stage(stage),
        }
    }

    /// Where `kind` runs: on the host for the host path, as the device report says for a
    /// device.
    fn placement(&self, kind: OperatorKind) -> Placement {
        match self {
            Self::Host(_) => {
                let _ = kind;
                Placement::Host
            }
            #[cfg(feature = "gpu")]
            Self::Device(driver) => driver.report().placement(kind),
        }
    }
}

/// The operator of the evaluator: the [`Engine`] that runs the level calls, and the
/// per-kind timer of [`FmmBuilder::kind_timings`] around them (Phase 4S T5). Every level
/// call passes through the [`FmmOperator`] implementation below, the one place where it
/// is timed; with [`KindTiming::Off`] it only checks the mode and delegates.
struct ExecOperator<T: SimdScalar + Stored + Equivalence + Default> {
    engine: Engine<T>,
    kinds: KindRecorder,
}

impl<T: SimdScalar + Stored + Equivalence + Default> ExecOperator<T> {
    /// The operator of `engine`, timing its level calls by `mode`; `on_device` if its
    /// device times on itself.
    fn new(engine: Engine<T>, mode: KindTiming, on_device: bool) -> Self {
        let placement = OperatorKind::ALL.map(|kind| engine.placement(kind));
        Self {
            engine,
            kinds: KindRecorder::new(mode, placement, on_device),
        }
    }

    /// Starts the timer of a level call of `kind` with a pair: a timing window for a
    /// device kind with [`KindTiming::Device`], the host clock otherwise.
    fn start_call(&mut self, kind: OperatorKind) -> CallClock {
        match &mut self.engine {
            #[cfg(feature = "gpu")]
            Engine::Device(driver)
                if self.kinds.timings.mode == KindTiming::Device
                    && self.kinds.timings.get(kind).placement == Placement::Device =>
            {
                CallClock::Window(driver.open_call_window())
            }
            _ => {
                let _ = kind;
                CallClock::Host(Instant::now())
            }
        }
    }

    /// Stops the timer of the level call of `kind` on `level` and adds its time: with
    /// [`KindTiming::Synchronous`] after a sync of the device (nothing on the host); a
    /// window is closed and resolved after the evaluation's download.
    fn end_call(&mut self, kind: OperatorKind, level: usize, clock: CallClock) {
        match clock {
            CallClock::Host(start) => {
                if self.kinds.timings.mode == KindTiming::Synchronous {
                    self.engine.sync();
                }
                self.kinds.add(kind, level, start.elapsed());
            }
            #[cfg(feature = "gpu")]
            CallClock::Window(window) => {
                let Engine::Device(driver) = &mut self.engine else {
                    unreachable!("windows are opened on a device")
                };
                let time = window.and_then(|window| driver.close_call_window(window));
                self.kinds.add(kind, level, Duration::ZERO);
                if let Some(time) = time {
                    self.kinds.windows.push((kind, level, time));
                }
            }
        }
    }
}

/// The timer of one level call ([`ExecOperator::start_call`]).
enum CallClock {
    /// The host clock at the start of the call.
    Host(Instant),
    /// The timing window of a device call; `None` after a device failure, which
    /// `read_output` returns.
    #[cfg(feature = "gpu")]
    Window(Option<TimingWindow>),
}

/// The per-kind times of the current evaluation ([`FmmBuilder::kind_timings`]).
struct KindRecorder {
    timings: KindTimings,
    /// Whether the device times on itself, for [`KindTiming::Device`].
    on_device: bool,
    /// With [`KindTiming::Device`]: the windows of this evaluation's device calls, with
    /// their kind and level, resolved after its download.
    #[cfg(feature = "gpu")]
    windows: Vec<(OperatorKind, usize, WindowTime)>,
}

impl KindRecorder {
    /// The recorder of `mode`, with the placement of every kind in
    /// [`OperatorKind::ALL`] order.
    fn new(mode: KindTiming, placement: [Placement; 8], on_device: bool) -> Self {
        Self {
            on_device,
            timings: KindTimings {
                mode,
                kinds: placement.map(|placement| KindTime {
                    placement,
                    calls: 0,
                    levels: [Duration::ZERO; MAX_LEVELS],
                }),
            },
            #[cfg(feature = "gpu")]
            windows: Vec::new(),
        }
    }

    /// Clears the times of the last evaluation.
    fn begin(&mut self) {
        for kind in &mut self.timings.kinds {
            kind.calls = 0;
            kind.levels = [Duration::ZERO; MAX_LEVELS];
        }
        #[cfg(feature = "gpu")]
        self.windows.clear();
    }

    /// The times of the evaluation; `None` with [`KindTiming::Off`].
    fn timings(&self) -> Option<KindTimings> {
        (self.timings.mode != KindTiming::Off).then_some(self.timings)
    }

    /// Counts a call of `kind` on `level` and adds `time` to it.
    fn add(&mut self, kind: OperatorKind, level: usize, time: Duration) {
        let entry = &mut self.timings.kinds[kind as usize];
        entry.calls += 1;
        entry.levels[level] += time;
    }

    /// With a device, after the evaluation's download: adds the times of the windows to
    /// their calls. False if a window was not timed on the device (it waited for the
    /// stream), so that its times are not the device's.
    #[cfg(feature = "gpu")]
    fn resolve_windows(&mut self) -> bool {
        let mut on_device = true;
        for (kind, level, time) in self.windows.drain(..) {
            on_device &= time.on_device();
            self.timings.kinds[kind as usize].levels[level] += time.resolve().unwrap_or_default();
        }
        on_device
    }
}

/// Delegates to the engine's host operator.
impl<T: SimdScalar + Stored + Equivalence + Default> FmmSizes for ExecOperator<T> {
    type Value = T;

    fn multipole_size(&self, level: usize) -> usize {
        self.engine.host().multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.engine.host().local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.engine.host().source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.engine.host().target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.engine.host().target_output_point_size()
    }
}

/// A batch of the operator interface, for the per-kind timer: its kind, and whether its
/// view has a pair (a call with an empty view does nothing and is not timed).
trait LevelCall {
    /// The operator kind of the call.
    const KIND: OperatorKind;

    /// True if the call's view has an entry.
    fn has_pairs(&self) -> bool;
}

/// Implements [`LevelCall`] for each batch, with its kind and its view.
macro_rules! level_call {
    ($($batch:ident: $kind:ident, $view:ident);* $(;)?) => {
        $(
            impl<T> LevelCall for $batch<'_, T> {
                const KIND: OperatorKind = OperatorKind::$kind;

                fn has_pairs(&self) -> bool {
                    !self.$view.is_empty()
                }
            }
        )*
    };
}

level_call!(
    P2m: P2m, leaves;
    M2m: M2m, children;
    M2l: M2l, pairs;
    P2l: P2l, x;
    L2l: L2l, parents;
    L2p: L2p, boxes;
    M2p: M2p, w;
    P2p: P2p, near;
);

/// Calls the engine's method of every level call; with a [`KindTiming`] other than `Off`,
/// a call with a pair runs between [`ExecOperator::start_call`] and
/// [`ExecOperator::end_call`].
macro_rules! delegate {
    ($($kind:ident: $batch:ident),* $(,)?) => {
        impl<T: SimdScalar + Stored + Equivalence + Default> FmmOperator for ExecOperator<T> {
            $(
                fn $kind(&mut self, batch: $batch<'_, T>) {
                    let timed = (self.kinds.timings.mode != KindTiming::Off
                        && batch.has_pairs())
                    .then(|| {
                        let kind = <$batch<'_, T> as LevelCall>::KIND;
                        (kind, batch.level, self.start_call(kind))
                    });
                    match &mut self.engine {
                        Engine::Host(operator) => operator.$kind(batch),
                        #[cfg(feature = "gpu")]
                        Engine::Device(driver) => driver.$kind(batch),
                    }
                    if let Some((kind, level, clock)) = timed {
                        self.end_call(kind, level, clock);
                    }
                }
            )*

            /// The evaluator's data movements, to the engine: nothing on the host path; the
            /// device operator mirrors them (Phase 5 T8).
            fn host_data(&mut self, event: HostData<'_, T>) {
                match &mut self.engine {
                    Engine::Host(operator) => operator.host_data(event),
                    #[cfg(feature = "gpu")]
                    Engine::Device(driver) => driver.host_data(event),
                }
            }
        }
    };
}

delegate!(p2m: P2m, m2m: M2m, m2l: M2l, p2l: P2l, l2l: L2l, l2p: L2p, m2p: M2p, p2p: P2p);

/// Returns `local` on this rank if every rank's `local` is `Ok`, and otherwise an error
/// on every rank: this rank's own, or [`FmmError::OtherRank`]. One all-reduce.
fn agree<C: CommunicatorCollectives, V>(
    comm: &C,
    local: Result<V, FmmError>,
) -> Result<V, FmmError> {
    let mut valid = false;
    comm.all_reduce_into(&local.is_ok(), &mut valid, SystemOperation::logical_and());
    match local {
        Ok(_) if !valid => Err(FmmError::OtherRank),
        result => result,
    }
}

/// The first point of `points` with a coordinate that is not finite, as an error.
fn check_finite(points: &[[f64; 3]], set: PointSet) -> Result<(), FmmError> {
    match points.iter().position(|x| !x.iter().all(|c| c.is_finite())) {
        Some(index) => Err(FmmError::NonFinitePoint {
            set,
            index,
            point: points[index],
        }),
        None => Ok(()),
    }
}

/// The first point not strictly inside `domain`, as an error.
fn check_inside(
    sources: &[[f64; 3]],
    targets: &[[f64; 3]],
    domain: &Domain,
) -> Result<(), FmmError> {
    let c = domain.physical_box().coordinates();
    let inside = |x: &[f64; 3]| (0..3).all(|k| c[k] < x[k] && x[k] < c[k + 3]);
    for (set, points) in [(PointSet::Sources, sources), (PointSet::Targets, targets)] {
        if let Some(index) = points.iter().position(|x| !inside(x)) {
            return Err(FmmError::PointOutsideDomain {
                set,
                index,
                point: points[index],
                domain: c,
            });
        }
    }
    Ok(())
}

/// The largest extent max_k (max xₖ − min xₖ) of the points of every rank. Two
/// all-reduces; +∞ − (−∞) is never formed, since some rank has a point.
fn global_extent<C: CommunicatorCollectives>(
    sources: &[[f64; 3]],
    targets: &[[f64; 3]],
    comm: &C,
) -> f64 {
    let (mut lower, mut upper) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for x in sources.iter().chain(targets) {
        for k in 0..3 {
            lower[k] = lower[k].min(x[k]);
            upper[k] = upper[k].max(x[k]);
        }
    }
    let (mut global_lower, mut global_upper) = ([0.0; 3], [0.0; 3]);
    comm.all_reduce_into(&lower[..], &mut global_lower[..], SystemOperation::min());
    comm.all_reduce_into(&upper[..], &mut global_upper[..], SystemOperation::max());
    (0..3)
        .map(|k| global_upper[k] - global_lower[k])
        .fold(0.0, f64::max)
}

/// The points of a point set that this rank owns, in leaf order (the received order of
/// its `Redistribution`): the points of local leaf j are `offsets[j]..offsets[j + 1]`.
#[derive(Clone, Debug)]
struct LeafRanges {
    counts: Vec<usize>,
    offsets: Vec<usize>,
}

impl LeafRanges {
    /// The ranges of `counts[j]` points in local leaf j.
    fn new(counts: &[usize]) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);
        for &n in counts {
            offsets.push(offsets.last().unwrap() + n);
        }
        Self {
            counts: counts.to_vec(),
            offsets,
        }
    }

    /// The positions of the points of local leaf `j`.
    fn range(&self, j: usize) -> std::ops::Range<usize> {
        self.offsets[j]..self.offsets[j + 1]
    }

    /// The number of points.
    fn len(&self) -> usize {
        *self.offsets.last().unwrap()
    }
}

/// The points per task below which the charge load ([`load_source_charges`]) stops
/// splitting its work.
const LOAD_GRAIN: usize = 4096;

/// Where the output pass of an evaluation finds every target (Phase 4S T9): for every
/// target this rank owns, in the received order of its `Redistribution` (leaf order,
/// Phase 5 T6), its point in leaf order and its local leaf, and for every local leaf the
/// two scales of CONVENTIONS §3.13, "Output". Built once by [`FmmBuilder::build`]: 8 N_t
/// bytes for N_t owned targets, and 16 bytes per local leaf. The pass writes in this
/// order, and the backward move of the targets' redistribution returns its values to the
/// caller's ranks and order. With the output pass on the device (`OutputPass`) the device
/// holds a copy, which the device report's memory counts; for the device it is the order
/// of the output it writes (`device::CallerOrder`).
#[derive(Clone, Debug)]
struct OutputOrder {
    /// For every owned target in received order: its point in leaf order (the received
    /// order is leaf order, so target r is point r).
    points: Vec<u32>,
    /// For every owned target in received order: its local leaf.
    leaves: Vec<u32>,
    /// For every local leaf: 4π r_t and 4π r_t², formed as the pass before T9 formed them.
    scales: Vec<[f64; 2]>,
}

impl OutputOrder {
    /// The order of the owned targets `targets`, whose local leaf j has r_t = `radii[j]`.
    ///
    /// # Panics
    ///
    /// If this rank owns more than `u32::MAX` targets.
    fn new(targets: &LeafRanges, radii: &[f64]) -> Self {
        let n = targets.len();
        let index = |i: usize| u32::try_from(i).expect("at most u32::MAX targets on a rank");
        let points = (0..n).map(index).collect();
        let mut leaves = vec![0; n];
        for j in 0..radii.len() {
            leaves[targets.range(j)].fill(index(j));
        }
        let scales = radii
            .iter()
            .map(|&r| [4.0 * PI * r, 4.0 * PI * r * r])
            .collect();
        Self {
            points,
            leaves,
            scales,
        }
    }
}

/// Writes `charges`, the charges of the owned sources in leaf order (forwarded by the
/// sources' `Redistribution`), into the charge slots of the source chunks `chunks` of the
/// local leaves (CONVENTIONS §3.13, "Source chunks"): leaf j gets
/// `charges[sources.range(j)]`. On `pool` when given, the leaves split in halves under
/// `rayon::join` into disjoint parts of at most [`LOAD_GRAIN`] points (or one leaf);
/// serially otherwise. Copies only, so the same values either way (Phase 4S T9).
fn load_source_charges<T: Copy + Send + Sync>(
    chunks: LeafSliceMut<'_, T>,
    sources: &LeafRanges,
    charges: &[T],
    pool: Option<&ThreadPool>,
) {
    match pool {
        Some(pool) => pool.install(|| split_source_charges(chunks, 0, sources, charges)),
        None => write_source_charges(chunks, 0, sources, charges),
    }
}

/// [`load_source_charges`] on the calling thread, for the leaves of `chunks`, the first
/// of which is local leaf `first`.
fn write_source_charges<T: Copy>(
    mut chunks: LeafSliceMut<'_, T>,
    first: usize,
    sources: &LeafRanges,
    charges: &[T],
) {
    for (r, chunk) in chunks.chunks_mut().enumerate() {
        let points = sources.range(first + r);
        let (_, values) = chunk.split_at_mut(3 * points.len());
        values.copy_from_slice(&charges[points]);
    }
}

/// [`load_source_charges`] on the current pool, halving the leaves of `chunks` (the
/// first local leaf `first`) until a half holds at most [`LOAD_GRAIN`] points or one leaf.
fn split_source_charges<T: Copy + Send + Sync>(
    mut chunks: LeafSliceMut<'_, T>,
    first: usize,
    sources: &LeafRanges,
    charges: &[T],
) {
    let n = chunks.nleaves();
    let points = sources.offsets[first + n] - sources.offsets[first];
    if n < 2 || points <= LOAD_GRAIN {
        write_source_charges(chunks, first, sources, charges);
        return;
    }
    let mid = n / 2;
    let (left, right) = chunks.split_at_mut(mid);
    rayon::join(
        || split_source_charges(left, first, sources, charges),
        || split_source_charges(right, first + mid, sources, charges),
    );
}

/// The output pass on the host (Phase 4S T9): φ and ∇φ of every owned target of `order`
/// from the leaf-ordered target output `data` (the `LeafStore` layout with the point
/// offsets `offsets`: the host's store, or since Phase 4S T11 the device's download read
/// in place), into `values` in the received order, o values per target: φ, or with
/// `gradients` φ and ∇φ (o = 4), the layout the backward move of the targets'
/// redistribution sends (Phase 5 T6). Each value is `T::from_f64(x.to_f64() / scale)`, as
/// the pass before T9 ([`scaled_output`]) computes it, so the output is that pass's bit
/// for bit.
///
/// One loop over the targets: on `pool` when given, rayon's split of `values` into ranges
/// that each thread writes in sequence, reading the store gathered; on the calling thread
/// otherwise. Every element of `values` (o values per target) is written once.
fn gather_output<T: SimdScalar + Default>(
    data: &[T],
    offsets: &[usize],
    order: &OutputOrder,
    gradients: bool,
    pool: Option<&ThreadPool>,
    values: &mut [T],
) {
    let o = if gradients { 4 } else { 1 };
    debug_assert_eq!(values.len(), o * order.points.len());
    // Target i's leaf j, the leaf's first point and the target's place k in the leaf.
    let place = |i: usize| {
        let j = order.leaves[i] as usize;
        let first = offsets[j];
        (j, first, order.points[i] as usize - first)
    };
    let scaled = |x: T, scale: f64| T::from_f64(RealScalar::to_f64(x) / scale);
    let write = |(i, target): (usize, &mut [T])| {
        let (j, first, k) = place(i);
        let [phi_scale, g_scale] = order.scales[j];
        if gradients {
            // Leaf j holds φ̂ of its n_j points, then their ĝ, from 4 P_j.
            let base = 4 * first;
            let g = base + (offsets[j + 1] - first) + 3 * k;
            target[0] = scaled(data[base + k], phi_scale);
            for c in 0..3 {
                target[1 + c] = scaled(data[g + c], g_scale);
            }
        } else {
            target[0] = scaled(data[first + k], phi_scale);
        }
    };
    match pool {
        Some(pool) => pool.install(|| values.par_chunks_mut(o).enumerate().for_each(write)),
        None => values.chunks_mut(o).enumerate().for_each(write),
    }
}

/// After the output pass on the device (Phase 4S T9): `device`, φ of every owned target in
/// received order and then, with gradients, ∇φ in the `[T; 3]` order of [`Output`], copied
/// into `values` in the layout of [`gather_output`] (o values per target), for the
/// backward move. On `pool` when given; each element once.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn interleave_output<T: SimdScalar>(
    device: &[T],
    gradients: bool,
    pool: Option<&ThreadPool>,
    values: &mut [T],
) {
    if !gradients {
        values.copy_from_slice(device);
        return;
    }
    let n = device.len() / 4;
    let (phi, g) = device.split_at(n);
    let g = g.as_chunks::<3>().0;
    let write = |(i, target): (usize, &mut [T])| {
        target[0] = phi[i];
        target[1..].copy_from_slice(&g[i]);
    };
    match pool {
        Some(pool) => pool.install(|| values.par_chunks_mut(4).enumerate().for_each(write)),
        None => values.chunks_mut(4).enumerate().for_each(write),
    }
}

/// After the backward move (Phase 5 T6): `values`, o values per target in the caller's
/// order (φ, or φ and ∇φ), copied into `output`'s buffers: cleared first and their
/// allocations reused (`collect_into_vec`, `unzip_into_vecs`), every element written once,
/// with no zero fill. On `pool` when given, serially otherwise. `gradient` becomes `Some`
/// with gradients and `None` without.
fn unpack_output<T: SimdScalar>(
    values: &[T],
    gradients: bool,
    pool: Option<&ThreadPool>,
    output: &mut Output<T>,
) {
    if gradients {
        let n = values.len() / 4;
        let both = |i: usize| {
            let v = &values[4 * i..4 * i + 4];
            (v[0], [v[1], v[2], v[3]])
        };
        let (potential, gradient) = (
            &mut output.potential,
            output.gradient.get_or_insert_with(Vec::new),
        );
        match pool {
            Some(pool) => pool.install(|| {
                (0..n)
                    .into_par_iter()
                    .map(both)
                    .unzip_into_vecs(potential, gradient);
            }),
            None => {
                potential.clear();
                gradient.clear();
                potential.reserve(n);
                gradient.reserve(n);
                for (phi, g) in (0..n).map(both) {
                    potential.push(phi);
                    gradient.push(g);
                }
            }
        }
    } else {
        output.gradient = None;
        let potential = &mut output.potential;
        match pool {
            Some(pool) => {
                pool.install(|| values.par_iter().copied().collect_into_vec(potential));
            }
            None => {
                potential.clear();
                potential.extend_from_slice(values);
            }
        }
    }
}

/// The wall time of each step of [`FmmBuilder::build`] on this rank, for reports only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BuildTimings {
    /// Steps 1–3: validation and the domain.
    pub domain: Duration,
    /// `Octree::new`, with the keys of the points.
    pub octree: Duration,
    /// `Plan::new`.
    pub plan: Duration,
    /// Steps 5 and 6 (Phase 5 T6): the two redistributions of the points to the ranks
    /// that own their leaves (sources and targets: `Redistribution::new` and the forward
    /// of the coordinates, collective, with the wait for other ranks), the radii of the
    /// local leaves and the output order.
    pub redistribute: Duration,
    /// Building or loading the tables.
    pub tables: Duration,
    /// The operator and `Evaluator::new` (stores and exchanges).
    pub evaluator: Duration,
    /// Step 8: the leaf-scaled coordinates.
    pub load: Duration,
    /// With a device backend: opening the device, the tuning (Phase 4 T12; with
    /// [`FmmBuilder::tuning_cache`]), the device operator (its tables and uploads) and the
    /// upload of the points; zero on the host.
    pub device: Duration,
}

impl BuildTimings {
    /// The sum of all steps.
    pub fn total(&self) -> Duration {
        self.domain
            + self.octree
            + self.plan
            + self.redistribute
            + self.tables
            + self.evaluator
            + self.load
            + self.device
    }
}

/// The wall time of each stage of one [`Fmm::evaluate`] on this rank, for reports only.
///
/// The six evaluator stages are those of `nd_fmm_plan::evaluator`; the collective ones
/// include the time spent waiting for other ranks.
///
/// With a device backend each stage is timed on the host as it is called, which
/// measures the time to *enqueue* its launches, not their run time: the device's work
/// shows in [`output`](Self::output), which contains the evaluation's one waiting
/// download (docs/design/device-path.md §8.3). Two ways to time the device's work:
/// - [`device`](Self::device): with [`FmmBuilder::device_timestamps`], where the device
///   times on itself (Metal, CUDA), the span of each stage on the device, from one timing
///   window per stage, without a sync;
/// - [`FmmBuilder::synchronous_stages`] waits for the device after every stage, so that
///   each stage is timed whole on the host, at the cost of a sync per stage (the only
///   way on the CPU runtime).
///
/// Host-fallback calls wait for the device themselves (their downloads).
///
/// With [`FmmBuilder::kind_timings`] (Phase 4S T5), [`kinds`](Self::kinds) holds the time
/// of every level call by operator kind and level ([`KindTimings`]), and
/// [`remainder`](Self::remainder) what the stages spent outside the level calls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimings {
    /// Moving the charges to the ranks that own their sources, into leaf order: the
    /// forward of the sources' `Redistribution` (collective: one all-to-all-v, with the
    /// wait for other ranks; Phase 5 T6, docs/design/distributed-fmm.md §4.4).
    pub forward_charges: Duration,
    /// Writing the charges, in leaf order, into the host's source chunks (on a device only
    /// with P2M, P2L or P2P on the host fallback), on the `Fmm`'s pool when it has one
    /// (Phase 4S T9); with a device backend also zeroing the device stores and uploading
    /// and scattering the charges.
    pub load: Duration,
    /// Stage 1: the source exchange (collective).
    pub exchange_sources: Duration,
    /// Stage 2: P2M and the local M2M.
    pub upward_local: Duration,
    /// Stage 3: the coarse gather (collective) and the global M2M.
    pub upward_global: Duration,
    /// Stage 4: the multipole exchange (collective).
    pub exchange_multipoles: Duration,
    /// Stage 5: L2L, M2L and P2L.
    pub downward: Duration,
    /// Stage 6: L2P, M2P and P2P.
    pub evaluate_leaves: Duration,
    /// The output pass: scaling the target output of the targets this rank owns, in leaf
    /// order, on the host or on the device ([`OutputPass`]; Phase 4S T9), and with a device
    /// backend downloading it ([`download`](Self::download)).
    pub output: Duration,
    /// Moving the output back to the ranks and the order the caller passed the targets in:
    /// the backward of the targets' `Redistribution` (collective: one all-to-all-v, with
    /// the wait for other ranks) and the copy into the [`Output`] (Phase 5 T6,
    /// docs/design/distributed-fmm.md §4.4).
    pub backward_output: Duration,
    /// With a device backend: the part of [`output`](Self::output) spent in
    /// `read_output`, the evaluation's one download and its sync, by the host clock
    /// (Phase 4S T9). It is the evaluation's one wait for the device, so in a default
    /// evaluation it also holds the device work still queued. Zero on the host.
    /// `output` less `download` is the host's part of the output pass.
    pub download: Duration,
    /// With a device backend that times on the device: the device time of each stage
    /// with device work ([`FmmBuilder::device_timestamps`]); `None` otherwise.
    pub device: Option<DeviceStageTimings>,
    /// The time of every level call by operator kind and level
    /// ([`FmmBuilder::kind_timings`], [`KindTiming`]); `None` with [`KindTiming::Off`],
    /// and with [`KindTiming::Device`] for an evaluation in which a window was not timed
    /// on the device.
    pub kinds: Option<KindTimings>,
}

impl StageTimings {
    /// The sum of all stages.
    pub fn total(&self) -> Duration {
        self.forward_charges
            + self.load
            + self.exchange_sources
            + self.upward_local
            + self.upward_global
            + self.exchange_multipoles
            + self.downward
            + self.evaluate_leaves
            + self.output
            + self.backward_output
    }

    /// [`total`](Self::total) less [`KindTimings::total`], at least zero: the time outside
    /// the level calls (moving and loading the charges, the exchanges, the download, the
    /// scaling and the move of the output, and on a device the enqueueing between calls).
    /// `None` without [`kinds`](Self::kinds). With [`KindTiming::Device`] it subtracts device time from
    /// host time, so it is the host's time beyond the device work, not a stage.
    pub fn remainder(&self) -> Option<Duration> {
        self.kinds
            .map(|kinds| self.total().saturating_sub(kinds.total()))
    }
}

/// How [`Fmm::evaluate`] times its level calls by operator kind
/// ([`FmmBuilder::kind_timings`], [`StageTimings::kinds`]; Phase 4S T5), for reports only.
///
/// Every level call passes through one place, the operator of the evaluator, which in a
/// mode other than `Off` times each call whose view has an entry and adds its time to its
/// kind and level ([`KindTimings`]); a call with an empty view does nothing, and is neither
/// timed nor counted. An evaluation makes at most 148 such calls (eight kinds, levels 0 to
/// 16, M2M in two passes): 14 on the uniform level-4 tree of the C3.2 gate, 3–119 on the
/// trees of `tests/mpi_exec.rs`. In every mode the output is the same bit for bit: no
/// call or launch moves, and no kind changes where it runs.
///
/// The modes compose with [`FmmBuilder::synchronous_stages`] and
/// [`FmmBuilder::device_timestamps`]: each adds its own syncs or windows, and none implies
/// another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KindTiming {
    /// No timing: the default. The evaluation adds no timer, sync or window; each level
    /// call checks the mode and is delegated.
    #[default]
    Off,
    /// Every call timed by the host clock on the calling thread, one `Instant::now` pair
    /// per call, the only cost on the host path. There the call is synchronous already
    /// (with threads, the rayon pool joins before it returns), so its time is its whole
    /// run; on the C3.2 tree at p = 6 and one thread the kinds add up to 0.9999 of the
    /// stages that call them.
    ///
    /// On a device backend `Fmm` syncs the device after every call (on the device or the
    /// host fallback), and once after the charge upload so that the first call starts on
    /// an idle device: one sync per call plus one per evaluation, besides the evaluation's
    /// own (and those of `synchronous_stages`, if set). A call's time is then the wall
    /// time of its device work, its enqueueing and one sync; the sum over the kinds
    /// exceeds what an evaluation without the syncs spends by about the sync cost times
    /// the calls. Report-only, never the default; on the CPU runtime the only mode, and on
    /// Metal the one whose kinds are a breakdown.
    Synchronous,
    /// Every call of a kind on the device timed by the device itself: one timing window
    /// per call (`nd_fmm_kernels::Device::open_window`), with no sync, read after the
    /// evaluation's download. Device time only: no launch, enqueue or sync time. A kind on
    /// the host fallback is timed by the host clock, and so includes the wait of its first
    /// download for the device work queued before it.
    ///
    /// Only where the device times on itself (Metal with timestamp queries, CUDA by
    /// events): [`FmmBuilder::build`] refuses it on the host and on the CPU runtime with
    /// [`SettingsError::KindTimingUnsupported`]. Each window flushes the queued work on
    /// Metal, which adds enqueue time to the evaluation.
    ///
    /// **Not a breakdown on Metal.** There the windows of neighbouring calls overlap and
    /// misattribute the device's time between calls (Phase 4S T5, the M3 Max, f32, the
    /// uniform cube at N = 10⁵, p = 8): M2L 0.25 ms against 6.15 ms with `Synchronous`
    /// and L2P 4.67 ms against 0.37 ms (medians of ten); in every evaluation the call
    /// windows of some stage added up to more than that stage's own window, and their sum
    /// ranged from 0.71 to 1.59 of the evaluation's wall time. Use `Synchronous` there.
    ///
    /// **A breakdown on CUDA** (one stream, events in order; Phase 4S T5, locust's H100,
    /// the same cube in f32 and f64): the call windows of every stage added up to less
    /// than its own window in every evaluation (f32 downward 1.64 ms of calls in a 1.65 ms
    /// span), and per kind they lie at or below the `Synchronous` times, which carry a sync
    /// each (f32 M2L 1.48 ms against 1.56 ms, P2P 0.31 against 0.33 ms; medians of ten).
    Device,
}

impl fmt::Display for KindTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Synchronous => "synchronous, a sync after each call on a device",
            Self::Device => "device timestamps, one window per device call",
        })
    }
}

/// The most levels of a tree: levels 0 to 16, the deepest level of a Morton key.
pub const MAX_LEVELS: usize = DEEPEST + 1;

/// The level calls of one operator kind in one [`Fmm::evaluate`] ([`KindTimings`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindTime {
    /// Where the kind ran ([`Fmm::placement`]); with [`KindTiming::Device`], a kind on
    /// the device has device times, one on the host host times.
    pub placement: Placement,
    /// The level calls with an entry in their view; for M2M both passes.
    pub calls: usize,
    /// The time of each level's calls, by level (of the targets; of the parents for M2M,
    /// of the children for L2L); zero for a level without a call.
    pub levels: [Duration; MAX_LEVELS],
}

impl KindTime {
    /// The time over every level.
    pub fn total(&self) -> Duration {
        self.levels.iter().sum()
    }
}

/// The time of every level call of one [`Fmm::evaluate`] by operator kind and level
/// ([`FmmBuilder::kind_timings`], Phase 4S T5), for reports only. What a time measures is
/// the mode's ([`KindTiming`]); a kind without a call has zero calls and zero time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindTimings {
    /// The mode that measured the times; never [`KindTiming::Off`].
    pub mode: KindTiming,
    /// By kind, in [`OperatorKind::ALL`] order.
    kinds: [KindTime; 8],
}

impl KindTimings {
    /// The calls of `kind`.
    pub fn get(&self, kind: OperatorKind) -> &KindTime {
        &self.kinds[kind as usize]
    }

    /// Every kind with its calls, in [`OperatorKind::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = (OperatorKind, &KindTime)> {
        OperatorKind::ALL.into_iter().zip(&self.kinds)
    }

    /// The sum over every kind.
    pub fn total(&self) -> Duration {
        self.kinds.iter().map(KindTime::total).sum()
    }
}

/// A stage of [`Fmm::evaluate`] that runs device work, timed by one timing window
/// ([`DeviceStageTimings`]). The exchanges have no window: on one rank they move nothing,
/// on several the device's part of them is a few transfers and launches (Phase 5 T8),
/// and the coarse gather's download falls in [`UpwardGlobal`](Self::UpwardGlobal).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceStage {
    /// Zeroing the device stores (at the evaluator's `reset`), uploading and scattering
    /// the charges.
    Load,
    /// Stage 2: P2M and the local M2M.
    UpwardLocal,
    /// Stage 3: the global M2M.
    UpwardGlobal,
    /// Stage 5: L2L, M2L and P2L.
    Downward,
    /// Stage 6: L2P, M2P and P2P.
    EvaluateLeaves,
}

impl DeviceStage {
    /// Every stage, in evaluation order.
    pub const ALL: [Self; 5] = [
        Self::Load,
        Self::UpwardLocal,
        Self::UpwardGlobal,
        Self::Downward,
        Self::EvaluateLeaves,
    ];
}

/// The device time of each [`DeviceStage`] of one [`Fmm::evaluate`], from the device's
/// own timestamps, for reports only (docs/design/device-path.md §8.3): one timing window
/// per stage, opened before its first launch and closed after its last, with no sync;
/// the times are read after the evaluation's one download. A window spans the device
/// work of its stage from the start of its first pass to the end of its last; the device
/// may run passes of neighbouring stages concurrently (Metal does), so the spans can
/// overlap and their sum can exceed the evaluation's wall time. A stage without device
/// work times zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeviceStageTimings {
    /// [`DeviceStage::Load`].
    pub load: Duration,
    /// [`DeviceStage::UpwardLocal`].
    pub upward_local: Duration,
    /// [`DeviceStage::UpwardGlobal`].
    pub upward_global: Duration,
    /// [`DeviceStage::Downward`].
    pub downward: Duration,
    /// [`DeviceStage::EvaluateLeaves`].
    pub evaluate_leaves: Duration,
}

impl DeviceStageTimings {
    /// The time of `stage`.
    pub fn get(&self, stage: DeviceStage) -> Duration {
        match stage {
            DeviceStage::Load => self.load,
            DeviceStage::UpwardLocal => self.upward_local,
            DeviceStage::UpwardGlobal => self.upward_global,
            DeviceStage::Downward => self.downward,
            DeviceStage::EvaluateLeaves => self.evaluate_leaves,
        }
    }

    /// Sets the time of `stage`.
    pub fn set(&mut self, stage: DeviceStage, time: Duration) {
        *match stage {
            DeviceStage::Load => &mut self.load,
            DeviceStage::UpwardLocal => &mut self.upward_local,
            DeviceStage::UpwardGlobal => &mut self.upward_global,
            DeviceStage::Downward => &mut self.downward,
            DeviceStage::EvaluateLeaves => &mut self.evaluate_leaves,
        } = time;
    }

    /// The sum of all stages.
    pub fn total(&self) -> Duration {
        DeviceStage::ALL.iter().map(|&stage| self.get(stage)).sum()
    }
}

/// The result of one [`Fmm::evaluate`] or [`Fmm::evaluate_into`], in the caller's target
/// order on this rank.
#[derive(Clone, Debug, PartialEq)]
pub struct Output<T> {
    /// φ(xᵢ) = Σⱼ qⱼ / (4π |xᵢ − yⱼ|) at every target.
    pub potential: Vec<T>,
    /// ∇φ(xᵢ) at every target, if the FMM was built with gradients.
    pub gradient: Option<Vec<[T; 3]>>,
    /// The wall time of each stage.
    pub timings: StageTimings,
}

/// The number of entries of each interaction list over all levels, on this rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ListSizes {
    /// U-list pairs: the near list without each leaf itself.
    pub u: usize,
    /// V-list pairs (M2L).
    pub v: usize,
    /// W-list pairs (M2P).
    pub w: usize,
    /// X-list pairs (P2L).
    pub x: usize,
}

/// What the evaluator's exchanges move on this rank per evaluation, in points and boxes
/// ([`Fmm::exchange_sizes`], Phase 5 T8; docs/design/distributed-fmm.md §7.2): the sizes
/// behind the device path's transfers on several ranks. Every count is zero on one rank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExchangeSizes {
    /// The source points of the ghost leaves, which the source exchange receives (G_s).
    pub ghost_sources: usize,
    /// The source points of the local leaves it sends, a leaf sent to two ranks twice.
    pub sent_sources: usize,
    /// The local leaves it sends, each once: the leaves whose charges a device
    /// evaluation writes into the host store.
    pub sent_leaves: usize,
    /// This rank's coarse blocks, which the coarse gather sends (B_o); none on one rank.
    pub sent_blocks: usize,
    /// The other ranks' coarse blocks, which it receives (B_r).
    pub received_blocks: usize,
    /// Per level, the boxes the multipole exchange sends (S_l), a box sent to two ranks
    /// twice.
    pub sent_boxes: Vec<usize>,
    /// Per level, the ghost boxes it receives (R_l).
    pub received_boxes: Vec<usize>,
}

/// A built FMM; see the [module documentation](self).
///
/// It owns the octree (which borrows the communicator for `'o`), the plan, the
/// operator with its tables and, with more than one thread, its thread pool, and the
/// evaluator with its stores, in which the leaf-scaled points stay loaded between
/// evaluations.
pub struct Fmm<'o, T, C = SimpleCommunicator>
where
    T: Stored + SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    octree: Octree<'o, C>,
    evaluator: FmmEvaluator<'o, C, T>,
    domain: Domain,
    strategy: M2lStrategy,
    backend: Backend,
    /// With a device backend: sync after every stage.
    synchronous_stages: bool,
    /// Whether an evaluation writes the charges into the host's source store: on the host
    /// path, and on a device with P2M, P2L or P2P on the host fallback (Phase 4S T9).
    host_sources: bool,
    /// Otherwise the local leaves the source exchange sends, ascending, whose charges an
    /// evaluation writes into the host's source store for it (Phase 5 T8); none on one
    /// rank.
    sent_leaves: Vec<u32>,
    /// The failure of a device evaluation, returned by every later evaluation.
    device_error: Option<String>,
    /// The route of the sources from the caller's ranks to their owners (Phase 5 T6).
    sources: Redistribution,
    /// The route of the targets, and of their output back.
    targets: Redistribution,
    /// The owned sources of every local leaf, in leaf order.
    source_leaves: LeafRanges,
    /// The owned targets of every local leaf, in leaf order.
    target_leaves: LeafRanges,
    /// The charges of the owned sources in leaf order, as the forward move leaves them on
    /// the host path (reused by every evaluation).
    charges: Vec<T>,
    /// The output pass's values, o per owned target in leaf order, for the backward move.
    received_output: Vec<T>,
    /// The backward move's values, o per target in the caller's order.
    returned_output: Vec<T>,
    /// The output pass's position and leaf of every owned target, and the scales of every
    /// leaf.
    outputs: OutputOrder,
    /// r_t of every local leaf, for the test oracle ([`Fmm::reference_output`]).
    radii: Vec<f64>,
    max_leaf_points: usize,
    cache_outcomes: Vec<(TableKind, CacheOutcome)>,
    threading: ThreadingReport,
    build_timings: BuildTimings,
}

impl<'o, T, C> Fmm<'o, T, C>
where
    T: Stored + SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    /// Evaluates the potentials, and the gradients if built with them, of the charges
    /// `charges` at every target; see the [module documentation](self#evaluation).
    ///
    /// `charges[i]` is the charge of source i of [`FmmBuilder::build`], on this rank.
    /// Allocates the output: [`evaluate_into`](Self::evaluate_into) a fresh [`Output`],
    /// with the same bits.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: one all-reduce, then the collectives of the evaluator's
    /// stages; with a device backend on several ranks one more all-reduce after the
    /// output's download, which agrees the device errors (Phase 5 T8).
    ///
    /// # Errors
    ///
    /// [`FmmError::ChargesLength`] on a rank whose `charges` does not have one entry per
    /// source, [`FmmError::OtherRank`] on the others; nothing is evaluated. With a
    /// device backend, [`FmmError::Device`] if a device operation of this or an earlier
    /// evaluation failed on this rank, and [`FmmError::OtherRank`] on the other ranks.
    pub fn evaluate(&mut self, charges: &[T]) -> Result<Output<T>, FmmError> {
        let mut output = Output {
            potential: Vec::new(),
            gradient: None,
            timings: StageTimings::default(),
        };
        self.evaluate_into(charges, &mut output)?;
        Ok(output)
    }

    /// [`evaluate`](Self::evaluate) into `output`, whose buffers are reused (Phase 4S T9,
    /// decision 11): the same bits, without allocating the output where its buffers have
    /// the capacity.
    ///
    /// `output` may hold anything: [`potential`](Output::potential) is rewritten to one
    /// value per target, [`gradient`](Output::gradient) to one per target if the FMM was
    /// built with gradients (its buffer reused if present, allocated if absent) and to
    /// `None` otherwise, and [`timings`](Output::timings) to this evaluation's. A buffer of
    /// the wrong length is resized, never refused; every element is written once, with no
    /// zero fill first. On an error `output` is left as it was.
    ///
    /// # Collective operation
    ///
    /// As [`evaluate`](Self::evaluate).
    ///
    /// # Errors
    ///
    /// As [`evaluate`](Self::evaluate), agreed on every rank alike.
    pub fn evaluate_into(&mut self, charges: &[T], output: &mut Output<T>) -> Result<(), FmmError> {
        // A kept device error rides on the agreement of the charge length (Phase 5 T8,
        // decision 12): the rank that failed returns it, the others `OtherRank`, before any
        // other collective of the evaluation.
        let local = if let Some(reason) = &self.device_error {
            Err(FmmError::Device(reason.clone()))
        } else if charges.len() == self.sources.nsent() {
            Ok(())
        } else {
            Err(FmmError::ChargesLength {
                expected: self.sources.nsent(),
                actual: charges.len(),
            })
        };
        agree(self.evaluator.comm(), local)?;

        // The pool of the charge load and the output pass: the operator's, unless it runs
        // serially (Phase 4S T9).
        let pool = self.evaluator.operator().engine.host().pool().cloned();
        let mut timings = StageTimings::default();
        // The charges to the ranks that own their sources, in leaf order (Phase 5 T6): on
        // the host path into the reused buffer, on a device into a fresh one that the
        // upload takes over without a copy (Phase 4S T11).
        let start = Instant::now();
        let upload = if self.evaluator.operator().engine.is_device() {
            Some(self.sources.forward(charges, 1))
        } else {
            self.sources.forward_into(charges, 1, &mut self.charges);
            None
        };
        timings.forward_charges = start.elapsed();
        let start = Instant::now();
        if self.host_sources {
            load_source_charges(
                self.evaluator.local_sources_mut(),
                &self.source_leaves,
                upload.as_deref().unwrap_or(&self.charges),
                pool.as_deref(),
            );
        } else if let Some(upload) = &upload {
            // The charges of the leaves the source exchange sends, which it reads from the
            // host store (Phase 5 T8, docs/design/distributed-fmm.md §7.1).
            for &j in &self.sent_leaves {
                let points = self.source_leaves.range(j as usize);
                let chunk = self.evaluator.sources_mut(j as usize);
                chunk[3 * points.len()..].copy_from_slice(&upload[points]);
            }
        }
        timings.load = start.elapsed();

        let sync = self.synchronous_stages;
        let evaluator = &mut self.evaluator;
        evaluator.reset();
        evaluator.operator_mut().kinds.begin();
        #[cfg(feature = "gpu")]
        if let ExecOperator {
            engine: Engine::Device(driver),
            kinds,
        } = evaluator.operator_mut()
        {
            // With synchronous kind timings, the first level call starts on an idle
            // device: its own sync, beside that of `synchronous_stages` (each documents it).
            let sync_kinds = kinds.timings.mode == KindTiming::Synchronous;
            let upload = upload.expect("a device evaluation forwards into a fresh buffer");
            timings.load += timed(|| {
                driver.begin_evaluation(upload);
                if sync {
                    driver.sync();
                }
                if sync_kinds {
                    driver.sync();
                }
            });
        }
        // A stage with device work runs in a timing window (`device_timestamps`); the
        // exchanges have none.
        let mut stage = |device: Option<DeviceStage>,
                         run: &mut dyn FnMut(&mut FmmEvaluator<'o, C, T>)| {
            timed(|| {
                if device.is_some() {
                    evaluator.operator_mut().engine.open_stage();
                }
                run(evaluator);
                if let Some(device) = device {
                    evaluator.operator_mut().engine.close_stage(device);
                }
                if sync {
                    evaluator.operator_mut().engine.sync();
                }
            })
        };
        timings.exchange_sources = stage(None, &mut |e| e.exchange_sources());
        timings.upward_local = stage(Some(DeviceStage::UpwardLocal), &mut |e| e.upward_local());
        timings.upward_global = stage(Some(DeviceStage::UpwardGlobal), &mut |e| {
            e.upward_global();
        });
        timings.exchange_multipoles = stage(None, &mut |e| e.exchange_multipoles());
        timings.downward = stage(Some(DeviceStage::Downward), &mut |e| e.downward());
        timings.evaluate_leaves = stage(Some(DeviceStage::EvaluateLeaves), &mut |e| {
            e.evaluate_leaves();
        });

        // The output pass, on the rank that owns each target, in leaf order.
        let start = Instant::now();
        let gradients = self.gradients();
        let ExecOperator { engine, kinds } = self.evaluator.operator_mut();
        match engine {
            Engine::Host(_) => {
                timings.kinds = kinds.timings();
                let store = self.evaluator.target_output_store();
                gather_output(
                    store.as_slice(),
                    store.point_offsets(),
                    &self.outputs,
                    gradients,
                    pool.as_deref(),
                    &mut self.received_output,
                );
            }
            #[cfg(feature = "gpu")]
            Engine::Device(driver) => {
                let failed = match timed_value(|| driver.read_output(), &mut timings.download) {
                    Ok(DeviceOutput::LeafOrder { values, offsets }) => {
                        gather_output(
                            &values,
                            offsets,
                            &self.outputs,
                            gradients,
                            pool.as_deref(),
                            &mut self.received_output,
                        );
                        None
                    }
                    Ok(DeviceOutput::CallerOrder(values)) => {
                        interleave_output(
                            &values,
                            gradients,
                            pool.as_deref(),
                            &mut self.received_output,
                        );
                        None
                    }
                    Err(error) => Some(error.to_string()),
                };
                timings.device = driver.stage_timings();
                // The call windows, read after the download, like the stage windows.
                let on_device = kinds.resolve_windows();
                timings.kinds = kinds.timings().filter(|_| on_device);
                // A device error of any rank, at any sync of the evaluation (the
                // mid-evaluation download on several ranks included), is agreed before the
                // backward move: one all-reduce with a device on several ranks (Phase 5 T8,
                // decision 12; design §7.5). Every rank finished the stages, so none waits.
                let comm = self.evaluator.comm();
                let ok = if comm.size() > 1 {
                    let mut ok = false;
                    comm.all_reduce_into(
                        &failed.is_none(),
                        &mut ok,
                        SystemOperation::logical_and(),
                    );
                    ok
                } else {
                    failed.is_none()
                };
                if let Some(reason) = failed {
                    self.device_error = Some(reason.clone());
                    return Err(FmmError::Device(reason));
                }
                if !ok {
                    return Err(FmmError::OtherRank);
                }
            }
        }
        timings.output = start.elapsed();

        // The output back to the caller's ranks and order (Phase 5 T6).
        let start = Instant::now();
        let o = if gradients { 4 } else { 1 };
        self.targets
            .backward_into(&self.received_output, o, &mut self.returned_output);
        unpack_output(&self.returned_output, gradients, pool.as_deref(), output);
        timings.backward_output = start.elapsed();
        output.timings = timings;
        Ok(())
    }

    /// Returns the octree.
    pub fn octree(&self) -> &Octree<'o, C> {
        &self.octree
    }

    /// Returns the plan: the box index and the interaction lists.
    pub fn plan(&self) -> &Plan {
        self.evaluator.plan()
    }

    /// Returns the operator: the host operator, or with a device backend the operator
    /// of its host fallback.
    pub fn operator(&self) -> &LaplaceOperator<T> {
        self.evaluator.operator().engine.host()
    }

    /// Returns the backend the operators run on ([`FmmBuilder::backend`]).
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns where `kind` runs: on the host for [`Backend::Host`]; with a device
    /// backend as its device report says (from Phase 4 T10 every kind runs on the device
    /// unless [`FmmBuilder::host_fallback`] names it).
    pub fn placement(&self, kind: OperatorKind) -> Placement {
        self.evaluator.operator().engine.placement(kind)
    }

    /// Returns the domain.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Returns the expansion degree p.
    pub fn p(&self) -> usize {
        self.operator().p()
    }

    /// Returns the resolved strategy, never [`M2lStrategy::Auto`]: on the host `Auto` by
    /// [`M2lStrategy::resolve`]; with a device backend and M2L on the device, `Auto` by the
    /// device path's choice (Phase 4 T12: from the tuning cache, tuned at build, or the
    /// static rule of the `tune` module, feature `gpu`), for which the tables were built.
    pub fn strategy(&self) -> M2lStrategy {
        self.strategy
    }

    /// Returns whether the output holds gradients.
    pub fn gradients(&self) -> bool {
        self.operator().gradients()
    }

    /// Returns the P2P kernel as it runs: [`P2pChoice::Reference`], or
    /// [`P2pChoice::Isa`] with the ISA of the kernel, for [`P2pChoice::Auto`] the one
    /// it picked on this machine. For reports, next to [`threading`](Self::threading).
    pub fn p2p_kernel(&self) -> P2pChoice {
        self.operator().p2p_kernel()
    }

    /// Returns the number of sources this rank passed to [`FmmBuilder::build`], the length
    /// of its charge vector.
    pub fn nsources(&self) -> usize {
        self.sources.nsent()
    }

    /// Returns the number of targets this rank passed to [`FmmBuilder::build`], the length
    /// of its output.
    pub fn ntargets(&self) -> usize {
        self.targets.nsent()
    }

    /// Returns the number of sources and of targets this rank owns: those in its local
    /// leaves, wherever the caller passed them (Phase 5 T6). On one rank,
    /// ([`nsources`](Self::nsources), [`ntargets`](Self::ntargets)).
    pub fn owned_points(&self) -> (usize, usize) {
        (self.sources.nreceived(), self.targets.nreceived())
    }

    /// Returns the number of levels, the global maximum level plus one.
    pub fn nlevels(&self) -> usize {
        self.plan().nlevels()
    }

    /// Returns the number of local leaves, on every level.
    pub fn nleaves(&self) -> usize {
        self.radii.len()
    }

    /// Returns the number of sources in each local leaf, in leaf order
    /// (`nd_fmm_plan::index::LeafNumbering`): the sources this rank owns, from every rank.
    pub fn source_counts(&self) -> &[usize] {
        &self.source_leaves.counts
    }

    /// Returns the number of targets in each local leaf, in leaf order: the targets this
    /// rank owns, from every rank.
    pub fn target_counts(&self) -> &[usize] {
        &self.target_leaves.counts
    }

    /// Returns the largest number of sources in a leaf, over all ranks.
    pub fn max_leaf_points(&self) -> usize {
        self.max_leaf_points
    }

    /// Returns the number of pairs in each interaction list on this rank, over all
    /// levels.
    pub fn list_sizes(&self) -> ListSizes {
        self.plan()
            .levels()
            .iter()
            .fold(ListSizes::default(), |sizes, lists| ListSizes {
                u: sizes.u + lists.near().len() - lists.near().nrows(),
                v: sizes.v + lists.v().len(),
                w: sizes.w + lists.w().len(),
                x: sizes.x + lists.x().len(),
            })
    }

    /// Returns what the evaluator's exchanges move on this rank per evaluation
    /// ([`ExchangeSizes`], Phase 5 T8), from their index lists.
    pub fn exchange_sizes(&self) -> ExchangeSizes {
        let sources = self.evaluator.source_exchange();
        let multipoles = self.evaluator.multipole_exchange();
        let coarse = self.evaluator.coarse_exchange();
        let mut sent_leaves = sources.send_leaves().to_vec();
        sent_leaves.sort_unstable();
        sent_leaves.dedup();
        let nlevels = multipoles.nlevels();
        ExchangeSizes {
            ghost_sources: sources.ghost_counts().iter().sum(),
            sent_sources: sources
                .send_leaves()
                .iter()
                .map(|&j| sources.leaf_counts()[j as usize])
                .sum(),
            sent_leaves: sent_leaves.len(),
            sent_blocks: coarse.sent_blocks().len(),
            received_blocks: coarse.received_blocks().count(),
            sent_boxes: (0..nlevels)
                .map(|l| multipoles.send_boxes(l).len())
                .collect(),
            received_boxes: (0..nlevels)
                .map(|l| multipoles.receive_boxes(l).len())
                .collect(),
        }
    }

    /// Returns what the table cache did for each table family, in the order of
    /// [`Tables::kinds`]; empty without a cache.
    pub fn cache_outcomes(&self) -> &[(TableKind, CacheOutcome)] {
        &self.cache_outcomes
    }

    /// Returns the wall time of each step of the build.
    pub fn build_timings(&self) -> BuildTimings {
        self.build_timings
    }

    /// Returns the threading report: the rayon threads, the MPI threading level and the
    /// BLAS thread variables as `build` read them ([`threading`](crate::threading)).
    pub fn threading(&self) -> &ThreadingReport {
        &self.threading
    }

    /// Sets how the next evaluations time their level calls by operator kind
    /// ([`FmmBuilder::kind_timings`], [`KindTiming`]; Phase 4S T5): the same build, for
    /// comparisons and for timing the evaluation and its kinds in separate evaluations
    /// without a second build. The output is the same bit for bit in every mode. Local:
    /// no collective.
    ///
    /// # Errors
    ///
    /// [`SettingsError::KindTimingUnsupported`] for [`KindTiming::Device`] where the
    /// device does not time on itself (the host, the CPU runtime); the mode is unchanged.
    pub fn set_kind_timings(&mut self, mode: KindTiming) -> Result<(), SettingsError> {
        let kinds = &mut self.evaluator.operator_mut().kinds;
        check_kind_timing(mode, self.backend, kinds.on_device)?;
        kinds.timings.mode = mode;
        Ok(())
    }

    /// With `serial`, runs the next evaluations on the calling thread even if the FMM
    /// has a pool ([`LaplaceOperator::set_serial`]): the serial path of the same build,
    /// for comparisons and timings. The output is the same bit for bit.
    pub fn set_serial(&mut self, serial: bool) {
        self.evaluator
            .operator_mut()
            .engine
            .host_mut()
            .set_serial(serial);
    }

    /// The multipoles and locals of the last evaluation, in the `LevelBuffers` layout of
    /// `nd-fmm-plan` (level after level, (p + 1)² values per box), for tests and reports:
    /// copies of the evaluator's stores on the host path; with a device backend, downloaded
    /// from the device (two downloads, counted toward the evaluation's device counters
    /// until the next evaluation). The values are unscaled (CONVENTIONS §3.7).
    ///
    /// # Errors
    ///
    /// [`FmmError::Device`] if a download fails.
    pub fn expansions(&mut self) -> Result<(Vec<T>, Vec<T>), FmmError> {
        match &mut self.evaluator.operator_mut().engine {
            Engine::Host(_) => Ok((
                self.evaluator.multipoles().as_slice().to_vec(),
                self.evaluator.locals().as_slice().to_vec(),
            )),
            #[cfg(feature = "gpu")]
            Engine::Device(driver) => driver
                .download_expansions()
                .map_err(|error| FmmError::Device(error.to_string())),
        }
    }

    /// Returns where the output pass runs ([`FmmBuilder::output_pass`], Phase 4S T9): on
    /// the host for [`Backend::Host`]; with a device backend as its device report says
    /// (by default on the device where it does f64 arithmetic, the CPU runtime and CUDA,
    /// and on the host on Metal).
    pub fn output_pass(&self) -> Placement {
        match &self.evaluator.operator().engine {
            Engine::Host(_) => Placement::Host,
            #[cfg(feature = "gpu")]
            Engine::Device(driver) => driver.report().output_pass,
        }
    }

    /// The test oracle of the output pass (Phase 4S T9), for tests only: φ and ∇φ of the
    /// last evaluation by the pass before T9, one serial loop over the leaves that divides
    /// each value of the leaf-ordered target output by 4π r_t or 4π r_t² in f64 and
    /// scatters it into the received order of the owned targets, then moved to the caller's
    /// ranks and order by the targets' `Redistribution`, φ and ∇φ in separate moves (Phase
    /// 5 T6). On the host it reads the evaluator's target output; with a device backend it
    /// downloads the device's (one download, counted toward the evaluation's device
    /// counters until the next evaluation). The output of [`evaluate`](Self::evaluate)
    /// equals it bit for bit, on every backend and with either [`OutputPass`]. Before the
    /// first evaluation it holds zeros. Its timings are zero.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: one all-to-all-v, two with gradients; with a device
    /// backend first one all-reduce, which agrees the download.
    ///
    /// # Errors
    ///
    /// [`FmmError::Device`] if the download fails, on that rank, and
    /// [`FmmError::OtherRank`] on the others.
    #[doc(hidden)]
    pub fn reference_output(&mut self) -> Result<Output<T>, FmmError> {
        let gradients = self.gradients();
        let (potential, gradient) = match &mut self.evaluator.operator_mut().engine {
            Engine::Host(_) => scaled_output(
                self.evaluator.target_output_store(),
                &self.target_leaves,
                &self.radii,
                gradients,
            ),
            #[cfg(feature = "gpu")]
            Engine::Device(driver) => {
                let store = driver
                    .download_target_output()
                    .map_err(|error| FmmError::Device(error.to_string()));
                let store = agree(self.evaluator.comm(), store)?;
                scaled_output(&store, &self.target_leaves, &self.radii, gradients)
            }
        };
        let potential = self.targets.backward(&potential, 1);
        let gradient = gradient.map(|gradient| {
            let flat = self.targets.backward(gradient.as_flattened(), 3);
            flat.as_chunks::<3>().0.to_vec()
        });
        Ok(Output {
            potential,
            gradient,
            timings: StageTimings::default(),
        })
    }
}

/// The device path's reports (feature `gpu`).
#[cfg(feature = "gpu")]
impl<'o, T, C> Fmm<'o, T, C>
where
    T: Stored + SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    /// With a device backend: the device, the placement of every operator kind, the
    /// tables on the device and the memory ([`DeviceReport`]); `None` on the host.
    pub fn device_report(&self) -> Option<&DeviceReport> {
        match &self.evaluator.operator().engine {
            Engine::Device(driver) => Some(driver.report()),
            Engine::Host(_) => None,
        }
    }

    /// With a device backend: the transfers, launches and syncs of the build and of the
    /// last evaluation ([`DeviceCounters`]); `None` on the host.
    pub fn device_counters(&self) -> Option<DeviceCounters> {
        match &self.evaluator.operator().engine {
            Engine::Device(driver) => Some(driver.counters()),
            Engine::Host(_) => None,
        }
    }

    /// A test hook (Phase 5 T8, decision 12): with a device backend, the next evaluation
    /// fails on this rank with a device error at its mid-evaluation sync (the download of
    /// the sent multipoles, `HostData::SendMultipoles`), which every rank then agrees:
    /// this rank returns [`FmmError::Device`], the others [`FmmError::OtherRank`], from that
    /// evaluation on. Nothing on the host.
    #[doc(hidden)]
    pub fn inject_device_error(&mut self) {
        if let Engine::Device(driver) = &mut self.evaluator.operator_mut().engine {
            driver.inject_error();
        }
    }

    /// With a device backend: downloads every plan view, the geometry and the index
    /// arrays from the device, for tests and reports ([`ViewsImage`]); `None` on the
    /// host. Its transfers count toward the evaluation counters until the next
    /// evaluation.
    ///
    /// # Errors
    ///
    /// [`FmmError::Device`] if a download fails.
    pub fn download_device_views(&mut self) -> Option<Result<ViewsImage, FmmError>> {
        match &mut self.evaluator.operator_mut().engine {
            Engine::Device(driver) => Some(
                driver
                    .download_views()
                    .map_err(|error| FmmError::Device(error.to_string())),
            ),
            Engine::Host(_) => None,
        }
    }
}

/// The target output `store`, scaled (CONVENTIONS §3.13, "Output") and in the received
/// order of the owned targets `targets` (since Phase 5 T6; the caller's order before),
/// with r_t of every local leaf in `radii`: the output pass before Phase 4S T9, one serial
/// loop over the leaves into zeroed vectors, kept as the test oracle of [`gather_output`]
/// and [`interleave_output`] ([`Fmm::reference_output`]).
fn scaled_output<T: SimdScalar + Default>(
    store: &LeafStore<T>,
    targets: &LeafRanges,
    radii: &[f64],
    gradients: bool,
) -> (Vec<T>, Option<Vec<[T; 3]>>) {
    let ntargets = targets.len();
    let mut potential = vec![T::zero(); ntargets];
    let mut gradient = gradients.then(|| vec![[T::zero(); 3]; ntargets]);
    for (j, &r) in radii.iter().enumerate() {
        let points = targets.range(j);
        let (phi_hat, g_hat) = store.chunk(j).split_at(points.len());
        let (phi_scale, g_scale) = (4.0 * PI * r, 4.0 * PI * r * r);
        for (&phi, i) in phi_hat.iter().zip(points.clone()) {
            potential[i] = T::from_f64(RealScalar::to_f64(phi) / phi_scale);
        }
        if let Some(gradient) = gradient.as_mut() {
            for (g, i) in g_hat.as_chunks::<3>().0.iter().zip(points) {
                gradient[i] = g.map(|gk| T::from_f64(RealScalar::to_f64(gk) / g_scale));
            }
        }
    }
    (potential, gradient)
}

/// Runs `stage` and returns its wall time.
fn timed(stage: impl FnOnce()) -> Duration {
    let start = Instant::now();
    stage();
    start.elapsed()
}

/// Runs `f`, stores its wall time in `time` and returns its value.
#[cfg(feature = "gpu")]
fn timed_value<R>(f: impl FnOnce() -> R, time: &mut Duration) -> R {
    let start = Instant::now();
    let value = f();
    *time = start.elapsed();
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, for test data.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// Uniform in [−1, 1).
        fn value(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    /// The points of `n` points in `nleaves` local leaves, in leaf order: random leaves,
    /// leaf 1 empty.
    fn random_ranges(rng: &mut Rng, n: usize, nleaves: usize) -> LeafRanges {
        let mut counts = vec![0; nleaves];
        for _ in 0..n {
            let j = rng.below(nleaves - 1);
            counts[if j >= 1 { j + 1 } else { j }] += 1;
        }
        LeafRanges::new(&counts)
    }

    /// No pool for one thread, a pool of `threads` otherwise.
    fn pool(threads: usize) -> Option<ThreadPool> {
        (threads > 1).then(|| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
    }

    fn bits<T: SimdScalar>(values: &[T]) -> Vec<u64> {
        values
            .iter()
            .map(|&v| RealScalar::to_f64(v).to_bits())
            .collect()
    }

    /// The charge load (Phase 4S T9; the charges in leaf order since Phase 5 T6) at one and
    /// four threads, against a serial loop over the points, bit for bit: the charges of the
    /// host's source store, the coordinates untouched. 20,000 sources in 700 leaves, so
    /// that the load splits below [`LOAD_GRAIN`].
    #[test]
    fn the_charge_load_is_bit_identical_at_any_thread_count() {
        let mut rng = Rng(1);
        let (n, nleaves) = (20_000, 700);
        let sources = random_ranges(&mut rng, n, nleaves);
        let charges: Vec<f64> = (0..n).map(|_| rng.value()).collect();
        let mut store = LeafStore::<f64>::new(&sources.counts, 4);
        for x in store.as_mut_slice() {
            *x = rng.value();
        }
        // Point by point: point k of leaf j has the charge of owned source offsets[j] + k.
        let mut want = store.clone();
        let mut range = want.range_mut(0..nleaves);
        for (j, chunk) in range.chunks_mut().enumerate() {
            let count = sources.counts[j];
            for k in 0..count {
                chunk[3 * count + k] = charges[sources.offsets[j] + k];
            }
        }
        for threads in [1, 4] {
            let pool = pool(threads);
            let mut got = store.clone();
            load_source_charges(got.range_mut(0..nleaves), &sources, &charges, pool.as_ref());
            assert_eq!(
                bits(got.as_slice()),
                bits(want.as_slice()),
                "{threads} threads: the source store"
            );
        }
    }

    /// The output passes (Phase 4S T9) against the pass before T9 ([`scaled_output`]), bit
    /// for bit, in the received order of the owned targets and its layout for the backward
    /// move (Phase 5 T6): the host pass ([`gather_output`]) at one and four threads, and
    /// the device pass's values interleaved ([`interleave_output`]); and the unpack after
    /// the backward move ([`unpack_output`]) into a fresh output and into one of the wrong
    /// size and gradients; f32 and f64, gradients off and on.
    #[test]
    fn the_output_passes_are_the_pass_before_t9_bit_for_bit() {
        output_passes::<f64>();
        output_passes::<f32>();
    }

    fn output_passes<T: SimdScalar + Default>() {
        let mut rng = Rng(2);
        let (n, nleaves) = (20_000, 700);
        let targets = random_ranges(&mut rng, n, nleaves);
        // Radii of levels 0 to 16 of domains of any size.
        let radii: Vec<f64> = (0..nleaves)
            .map(|_| (1.5 + rng.value()) * 2f64.powi(-(rng.below(17) as i32)))
            .collect();
        let order = OutputOrder::new(&targets, &radii);
        for gradients in [false, true] {
            let o = if gradients { 4 } else { 1 };
            let mut store = LeafStore::<T>::new(&targets.counts, o);
            for x in store.as_mut_slice() {
                *x = T::from_f64(rng.value());
            }
            let (potential, gradient) = scaled_output(&store, &targets, &radii, gradients);
            // The pass before T9 in the layout of the backward move: o values per target.
            let mut reference = Vec::with_capacity(o * n);
            for i in 0..n {
                reference.push(potential[i]);
                if let Some(gradient) = &gradient {
                    reference.extend(gradient[i]);
                }
            }
            let want = Output {
                potential,
                gradient,
                timings: StageTimings::default(),
            };
            let want_bits = |output: &Output<T>| {
                assert_eq!(output.gradient.is_some(), gradients);
                let mut values = bits(&output.potential);
                if let Some(gradient) = &output.gradient {
                    values.extend(bits(gradient.as_flattened()));
                }
                values
            };
            for threads in [1, 4] {
                let pool = pool(threads);
                let what = format!("{threads} threads, gradients {gradients}");
                let mut values = vec![T::zero(); o * n];
                let (data, offsets) = (store.as_slice(), store.point_offsets());
                gather_output(data, offsets, &order, gradients, pool.as_ref(), &mut values);
                assert_eq!(bits(&values), bits(&reference), "{what}: the host pass");
                // The device pass's values: φ, then ∇φ, in the received order.
                let mut device = want.potential.clone();
                device.extend(want.gradient.iter().flatten().flatten());
                let mut interleaved = vec![T::zero(); o * n];
                interleave_output(&device, gradients, pool.as_ref(), &mut interleaved);
                assert_eq!(
                    bits(&interleaved),
                    bits(&reference),
                    "{what}: the device pass"
                );
                let mut fresh = Output {
                    potential: Vec::new(),
                    gradient: None,
                    timings: StageTimings::default(),
                };
                unpack_output(&reference, gradients, pool.as_ref(), &mut fresh);
                assert_eq!(want_bits(&fresh), want_bits(&want), "{what}: the unpack");
                let mut wrong = Output {
                    potential: vec![T::zero(); 3],
                    gradient: (!gradients).then(|| vec![[T::zero(); 3]; 5]),
                    timings: StageTimings::default(),
                };
                unpack_output(&reference, gradients, pool.as_ref(), &mut wrong);
                assert_eq!(
                    want_bits(&wrong),
                    want_bits(&want),
                    "{what}: the unpack into a resized output"
                );
            }
        }
    }
}
