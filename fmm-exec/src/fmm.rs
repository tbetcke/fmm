//! The user-facing FMM: [`FmmBuilder`], [`Fmm`] and its [`Output`] (C3.2).
//!
//! [`FmmBuilder::build`] takes source and target points in the caller's order, builds
//! the octree, the plan of `nd-fmm-plan`, the tables, the [`LaplaceOperator`] and the
//! evaluator, and loads the points in the leaf-scaled form of CONVENTIONS §3.13.
//! [`Fmm::evaluate`] takes charges in the caller's source order, runs the evaluator stage
//! by stage, and returns potentials (and gradients) in the caller's target order, with
//! 1/(4π) and the target leaf radius applied once (§3.1, §3.13, "Output"):
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
//!    supported), and agrees the outcome on every rank (one all-reduce).
//! 2. Counts the points of all ranks (one all-reduce); none at all is an error.
//! 3. Takes the supplied domain, or `compute_global_bounding_box` over the sources and
//!    targets of every rank (after one all-reduce pair that rejects points spanning no
//!    volume, which that function cannot handle). Checks that every point lies strictly
//!    inside it and agrees the outcome (one all-reduce).
//! 4. Builds one octree from the finest-level keys (`points_to_morton` at level 16) of
//!    the sources and the targets together, with `max_level`, `max_points_per_leaf`
//!    (`OctreeOptions::with_max_fine_keys`) and the ghost-children layer, and the plan
//!    of its box index and lists.
//! 5. Finds every point's leaf with `Octree::local_leaf`. If any rank has a point in a
//!    leaf another rank owns, every rank returns [`FmmError::PointsNotOwned`] with the
//!    global count (one all-reduce): points are not redistributed until C5.1. With a
//!    device backend on more than one rank, every rank then returns
//!    [`SettingsError::DeviceNeedsOneRank`], without a collective.
//! 6. Sorts sources and targets into leaf order, stably, and keeps both permutations.
//! 7. All-reduces the largest number of sources in a leaf, which sizes the P2P scratch;
//!    with a device backend resolves the M2L strategy of [`M2lStrategy::Auto`] for the
//!    device (Phase 4 T12: from the tuning cache, by timing rotation against dense, or by
//!    the static rule); builds or loads the tables, the operator (with a device backend
//!    the device operator around it, which checks that its buffers fit in device memory,
//!    allocates them, uploads the views and tables, and takes its GEMM decisions), and the
//!    evaluator with the per-leaf counts (its own collectives).
//! 8. Writes the leaf-scaled source coordinates and target positions into the
//!    evaluator's stores, once ([`leaf_coordinates`]), and with a device backend uploads
//!    them to the device, which then takes its P2P decision and ends its tuning (device
//!    work only, no collective).
//!
//! It also reads the BLAS thread variables once, for [`Fmm::threading`].
//!
//! An error that depends on one rank's input is agreed by every rank before the next
//! collective: the rank that found it returns it, the others [`FmmError::OtherRank`];
//! errors that every rank sees alike ([`NoPoints`](FmmError::NoPoints),
//! [`DegenerateExtent`](FmmError::DegenerateExtent),
//! [`PointsNotOwned`](FmmError::PointsNotOwned), and those of the plan and the
//! evaluator) are returned on every rank. No communication sits in a rank-dependent
//! branch.
//!
//! # Evaluation
//!
//! [`Fmm::evaluate`] is collective: it agrees the length of the charge vector (one
//! all-reduce), writes the charges after the coordinates of each source chunk (§3.13,
//! "Source chunks"), resets the evaluator and runs its six stages, timing each one
//! ([`StageTimings`]), and scales the target output into the caller's order. For a fixed
//! tree, ranks, input and P2P kernel, two evaluations are bit-identical (the
//! accumulation order of `nd_fmm_plan::evaluator`), for every number of threads.
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
//! `Dense`, f64 `Dense` to p = 11 and `Rotation` above (provisional), the GEMMs of
//! `DeviceGemm::Auto` and the backend's P2P layout. Two builds from the same cache give
//! the same bits; `evaluate` never tunes.
//!
//! # Redistribution (C5.1)
//!
//! Points and charges are taken, and potentials returned, in the caller's order, on the
//! caller's rank. C5.1 will move points to the ranks that own their leaves behind these
//! signatures (`docs/design/fmm-plan-redesign.md` §9): one `Redistribution` for the
//! sources and one for the targets in `build`, the charges forwarded and the output sent
//! back in `evaluate`. Until then, step 5 rejects input that would need it.
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
use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::{Evaluator, EvaluatorError};
use nd_fmm_plan::operator::{FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p};
use nd_fmm_plan::plan::{Plan, PlanError};
use nd_fmm_plan::store::LeafStore;
use nd_fmm_tables::cache::{Stored, TableKind};
use nd_fmm_tables::{CacheOutcome, TableCache};
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::octree::compute_global_bounding_box;
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, points_to_morton};
use rayon::{ThreadPool, ThreadPoolBuilder};
use rlst::SliceArray;
use thiserror::Error;

#[cfg(feature = "gpu")]
use crate::device::{self, DeviceCounters, DeviceDriver, DeviceOptions, DeviceReport, ViewsImage};
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
    /// A device backend on more than one rank: the device path runs on one rank until
    /// C5.1 (docs/design/device-path.md §4.4). Returned on every rank.
    #[error("a device backend runs on one rank, not {ranks}, until C5.1")]
    DeviceNeedsOneRank {
        /// The number of ranks.
        ranks: usize,
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
    /// Some points lie in leaves that other ranks own; returned on every rank. Points are
    /// not redistributed until C5.1.
    #[error(
        "{count} points of all ranks lie in leaves owned by other ranks; points are not \
         redistributed yet (C5.1)"
    )]
    PointsNotOwned {
        /// The number of such points, summed over all ranks.
        count: u64,
    },
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
/// | [`device_p2p_layout`](Self::device_p2p_layout) | [`DeviceP2pLayout::Auto`]: by backend |
/// | [`device_leaf_layout`](Self::device_leaf_layout) | [`DeviceLeafLayout::Auto`]: by backend |
/// | [`device_gemm`](Self::device_gemm) | [`DeviceGemm::Auto`]: by precision, p and backend |
/// | [`device_scratch_budget`](Self::device_scratch_budget) | 128 MB |
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
    /// fit in device memory ([`SettingsError::DeviceMemory`]); on more than one rank it
    /// returns [`SettingsError::DeviceNeedsOneRank`] (docs/design/device-path.md §4.4).
    /// With [`Backend::Cpu`], `threads(n)` builds no rayon pool and caps the units per
    /// cube of the CPU runtime instead (the `device` module, "Threads").
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
    /// docs/design/device-path.md §6.4). A level call whose batches need more runs in
    /// several chunks, with the same bits. Allocated once at build and counted in the
    /// device memory check. Ignored by [`Backend::Host`].
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
    ///
    /// # Panics
    ///
    /// If `Octree::new` panics: on several ranks when the coarse tree has fewer blocks
    /// than ranks (too few distinct points, or a small `max_level`).
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
                Ok((pool, supplied, device))
            });
        let (pool, supplied, device) = agree(comm, local)?;
        let threading = ThreadingReport::read(self.rayon_threads(), provided);

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

        // Step 5: the local leaf of every point.
        let start = Instant::now();
        let nlocal = plan.index().leaves().nlocal();
        let mut leaves = Vec::with_capacity(keys.len());
        let mut not_owned = 0u64;
        for &key in &keys {
            let leaf = octree
                .local_leaf(key)
                .ok()
                .flatten()
                .and_then(|leaf| plan.index().find_leaf(leaf))
                .filter(|&j| (j as usize) < nlocal);
            match leaf {
                Some(j) => leaves.push(j),
                None => not_owned += 1,
            }
        }
        let mut count = 0u64;
        comm.all_reduce_into(&not_owned, &mut count, SystemOperation::sum());
        if count > 0 {
            return Err(FmmError::PointsNotOwned { count });
        }
        // The device path runs on one rank until C5.1. Every rank sees the same size, so
        // every rank returns here and no collective is skipped (device-path.md §4.4).
        if self.backend.is_device() && comm.size() > 1 {
            return Err(SettingsError::DeviceNeedsOneRank {
                ranks: comm.size() as usize,
            }
            .into());
        }

        // Step 6: leaf order, stable.
        let (source_leaves, target_leaves) = leaves.split_at(sources.len());
        let sources_by_leaf = LeafOrder::new(source_leaves, nlocal);
        let targets_by_leaf = LeafOrder::new(target_leaves, nlocal);
        let leaf_keys: Vec<MortonKey> = plan.index().leaves().keys()[..nlocal].to_vec();
        let radii: Vec<f64> = (0..nlocal)
            .map(|j| radius(plan.index().leaves().level(j), &domain))
            .collect();
        let sort_time = start.elapsed();

        // Step 7: tables, operator, evaluator.
        let start = Instant::now();
        let local_max = sources_by_leaf.counts.iter().copied().max().unwrap_or(0);
        let mut max_leaf_points = 0usize;
        comm.all_reduce_into(&local_max, &mut max_leaf_points, SystemOperation::max());
        // With a device backend, the tuner (Phase 4 T12) and the M2L strategy: under
        // `Auto` with M2L on the device, the cached, tuned or static choice of the device
        // path, for which the tables are built.
        let mut device = device;
        let tuning_start = Instant::now();
        let (strategy, tuner) = self.device_strategy(device.as_mut(), &plan);
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
        let start_device = Instant::now();
        let operator = self.exec_operator(
            operator,
            device,
            &plan,
            (&sources_by_leaf.counts, &targets_by_leaf.counts),
            tuner,
        )?;
        let mut device_time = open_time + tuning_time + start_device.elapsed();
        let mut evaluator = Evaluator::new(
            plan,
            comm,
            operator,
            &sources_by_leaf.counts,
            &targets_by_leaf.counts,
        )
        .map_err(FmmError::Evaluator)?;
        let evaluator_time = start
            .elapsed()
            .saturating_sub(device_time - open_time - tuning_time);

        // Step 8: the leaf-scaled coordinates, once.
        let start = Instant::now();
        for (j, &key) in leaf_keys.iter().enumerate() {
            let chunk = evaluator.sources_mut(j);
            for (u, &i) in chunk
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(sources_by_leaf.points(j))
            {
                *u = leaf_coordinates(sources[i], key, &domain);
            }
            let chunk = evaluator.target_input_mut(j);
            for (u, &i) in chunk
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(targets_by_leaf.points(j))
            {
                *u = leaf_coordinates(targets[i], key, &domain);
            }
        }
        let load_time = start.elapsed();
        // The device's copy of the points, from copies of the stores: the store accessors
        // and `operator_mut` cannot borrow the evaluator at once (device-path.md §4.3).
        let start = Instant::now();
        if evaluator.operator().is_device() {
            let stores = (
                evaluator.source_store().clone(),
                evaluator.target_input_store().clone(),
            );
            evaluator.operator_mut().load_points(&stores.0, &stores.1)?;
        }
        device_time += start.elapsed();

        Ok(Fmm {
            octree,
            evaluator,
            domain,
            strategy: strategy.resolve(self.p),
            backend: self.backend,
            synchronous_stages: self.synchronous_stages && self.backend.is_device(),
            leaf_charges: if self.backend.is_device() {
                vec![T::default(); sources_by_leaf.len()]
            } else {
                Vec::new()
            },
            device_error: None,
            sources: sources_by_leaf,
            targets: targets_by_leaf,
            radii,
            max_leaf_points,
            cache_outcomes,
            threading,
            build_timings: BuildTimings {
                domain: domain_time.saturating_sub(open_time),
                octree: octree_time,
                plan: plan_time,
                sort: sort_time,
                tables: tables_time,
                evaluator: evaluator_time,
                load: load_time,
                device: device_time,
            },
        })
    }

    /// The operator of the evaluator: the host operator, or with a device backend the
    /// device operator that wraps it.
    #[cfg_attr(not(feature = "gpu"), allow(clippy::unnecessary_wraps))]
    fn exec_operator(
        &self,
        operator: LaplaceOperator<T>,
        device: Option<OpenedDevice>,
        plan: &Plan,
        counts: (&[usize], &[usize]),
        tuner: Option<Tuner>,
    ) -> Result<ExecOperator<T>, FmmError> {
        match device {
            None => Ok(ExecOperator::Host(operator)),
            #[cfg(feature = "gpu")]
            Some(device) => {
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
                };
                let tuner = tuner.expect("a device build has a tuner");
                let driver = device::driver(operator, device, plan, counts, &options, tuner)?;
                Ok(ExecOperator::Device(driver))
            }
            #[cfg(not(feature = "gpu"))]
            Some(never) => {
                let _ = (plan, counts, tuner);
                match never {}
            }
        }
    }

    /// With a device: the tuner of the build (Phase 4 T12) and the M2L strategy to build
    /// the tables for: the builder's, or under `Auto` with M2L on the device the device
    /// path's (cached, tuned, or the static rule; `tune::static_strategy`). On the host,
    /// or with M2L on the host fallback, the builder's strategy, which the tables resolve
    /// by the host rule.
    #[cfg(feature = "gpu")]
    fn device_strategy(
        &self,
        device: Option<&mut OpenedDevice>,
        plan: &Plan,
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
        let mut tuner = Tuner::new(
            key,
            self.tuning_cache.as_deref(),
            self.tuning_budget,
            self.tuning_hook.clone(),
        );
        let strategy = if self.strategy == M2lStrategy::Auto
            && !self.host_fallback.contains(&OperatorKind::M2l)
        {
            device::tune_strategy::<T>(
                &mut tuner,
                device,
                plan,
                self.p,
                self.table_cache.as_deref(),
            )
        } else {
            self.strategy
        };
        (strategy, Some(tuner))
    }

    /// Without the `gpu` feature: the builder's strategy, no tuner.
    #[cfg(not(feature = "gpu"))]
    fn device_strategy(
        &self,
        device: Option<&mut OpenedDevice>,
        plan: &Plan,
    ) -> (M2lStrategy, Option<Tuner>) {
        let _ = (device, plan);
        (self.strategy, None)
    }
}

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

/// The operator of the evaluator: [`LaplaceOperator`] on the host path, or the device
/// operator (feature `gpu`), which wraps one for its host fallback
/// (docs/design/device-path.md §3.3). Every call is delegated; without `gpu` the enum
/// has the host variant only.
#[cfg_attr(
    feature = "gpu",
    expect(
        clippy::large_enum_variant,
        reason = "one per Fmm; boxing the host operator would add an indirection to every \
                  level call of the host path"
    )
)]
enum ExecOperator<T: SimdScalar + Stored + Equivalence + Default> {
    Host(LaplaceOperator<T>),
    #[cfg(feature = "gpu")]
    Device(Box<dyn DeviceDriver<T>>),
}

impl<T: SimdScalar + Stored + Equivalence + Default> ExecOperator<T> {
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
    #[cfg_attr(not(feature = "gpu"), allow(clippy::unnecessary_wraps))]
    fn load_points(
        &mut self,
        sources: &LeafStore<T>,
        target_input: &LeafStore<T>,
    ) -> Result<(), FmmError> {
        match self {
            Self::Host(_) => {
                let _ = (sources, target_input);
                Ok(())
            }
            #[cfg(feature = "gpu")]
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
}

/// Delegates to the variant.
impl<T: SimdScalar + Stored + Equivalence + Default> FmmSizes for ExecOperator<T> {
    type Value = T;

    fn multipole_size(&self, level: usize) -> usize {
        self.host().multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.host().local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.host().source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.host().target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.host().target_output_point_size()
    }
}

/// Calls the variant's method of every level call.
macro_rules! delegate {
    ($($kind:ident: $batch:ident),* $(,)?) => {
        impl<T: SimdScalar + Stored + Equivalence + Default> FmmOperator for ExecOperator<T> {
            $(
                fn $kind(&mut self, batch: $batch<'_, T>) {
                    match self {
                        Self::Host(operator) => operator.$kind(batch),
                        #[cfg(feature = "gpu")]
                        Self::Device(driver) => driver.$kind(batch),
                    }
                }
            )*
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

/// A point set in leaf order: the points of local leaf j are
/// `order[offsets[j]..offsets[j + 1]]`, as positions in the caller's slice, ascending.
#[derive(Clone, Debug)]
struct LeafOrder {
    counts: Vec<usize>,
    offsets: Vec<usize>,
    order: Vec<usize>,
}

impl LeafOrder {
    /// Sorts the points with local leaf indices `leaves` by leaf, stably (a counting
    /// sort).
    fn new(leaves: &[u32], nlocal: usize) -> Self {
        let mut counts = vec![0; nlocal];
        for &j in leaves {
            counts[j as usize] += 1;
        }
        let mut offsets = Vec::with_capacity(nlocal + 1);
        offsets.push(0);
        for &n in &counts {
            offsets.push(offsets.last().unwrap() + n);
        }
        let mut next = offsets[..nlocal].to_vec();
        let mut order = vec![0; leaves.len()];
        for (i, &j) in leaves.iter().enumerate() {
            order[next[j as usize]] = i;
            next[j as usize] += 1;
        }
        Self {
            counts,
            offsets,
            order,
        }
    }

    /// The caller's positions of the points of local leaf `j`, in leaf order.
    fn points(&self, j: usize) -> &[usize] {
        &self.order[self.offsets[j]..self.offsets[j + 1]]
    }

    /// The number of points.
    fn len(&self) -> usize {
        self.order.len()
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
    /// Steps 5 and 6: the leaf of every point and the leaf order.
    pub sort: Duration,
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
            + self.sort
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
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimings {
    /// Writing the charges into the source chunks; with a device backend also zeroing
    /// the device stores and uploading and scattering the charges.
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
    /// Scaling the target output into the caller's order; with a device backend also
    /// downloading it.
    pub output: Duration,
    /// With a device backend that times on the device: the device time of each stage
    /// with device work ([`FmmBuilder::device_timestamps`]); `None` otherwise.
    pub device: Option<DeviceStageTimings>,
}

impl StageTimings {
    /// The sum of all stages.
    pub fn total(&self) -> Duration {
        self.load
            + self.exchange_sources
            + self.upward_local
            + self.upward_global
            + self.exchange_multipoles
            + self.downward
            + self.evaluate_leaves
            + self.output
    }
}

/// A stage of [`Fmm::evaluate`] that runs device work, timed by one timing window
/// ([`DeviceStageTimings`]). The exchanges move nothing on the one rank a device runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceStage {
    /// Zeroing the device stores, uploading and scattering the charges.
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

/// The result of one [`Fmm::evaluate`], in the caller's target order on this rank.
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
    /// With a device backend: the charges of an evaluation in leaf order, for the upload.
    leaf_charges: Vec<T>,
    /// The failure of a device evaluation, returned by every later evaluation.
    device_error: Option<String>,
    sources: LeafOrder,
    targets: LeafOrder,
    /// r_t of every local leaf.
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
    /// Allocates the output.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: one all-reduce, then the collectives of the evaluator's
    /// stages.
    ///
    /// # Errors
    ///
    /// [`FmmError::ChargesLength`] on a rank whose `charges` does not have one entry per
    /// source, [`FmmError::OtherRank`] on the others; nothing is evaluated. With a
    /// device backend, [`FmmError::Device`] if a device operation of this or an earlier
    /// evaluation failed.
    pub fn evaluate(&mut self, charges: &[T]) -> Result<Output<T>, FmmError> {
        let local = if charges.len() == self.sources.len() {
            Ok(())
        } else {
            Err(FmmError::ChargesLength {
                expected: self.sources.len(),
                actual: charges.len(),
            })
        };
        agree(self.evaluator.comm(), local)?;
        // A device runs on one rank, so this return skips no collective of another rank.
        if let Some(reason) = &self.device_error {
            return Err(FmmError::Device(reason.clone()));
        }

        let mut timings = StageTimings::default();
        let start = Instant::now();
        let mut chunks = self.evaluator.local_sources_mut();
        let mut leaf_charges = self.leaf_charges.iter_mut();
        for (j, chunk) in chunks.chunks_mut().enumerate() {
            let points = self.sources.points(j);
            let (_, values) = chunk.split_at_mut(3 * points.len());
            for (q, &i) in values.iter_mut().zip(points) {
                *q = charges[i];
                if let Some(slot) = leaf_charges.next() {
                    *slot = charges[i];
                }
            }
        }
        timings.load = start.elapsed();

        let sync = self.synchronous_stages;
        let evaluator = &mut self.evaluator;
        evaluator.reset();
        #[cfg(feature = "gpu")]
        if let ExecOperator::Device(driver) = evaluator.operator_mut() {
            let leaf_charges = &self.leaf_charges;
            timings.load += timed(|| {
                driver.begin_evaluation(leaf_charges);
                if sync {
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
                    evaluator.operator_mut().open_stage();
                }
                run(evaluator);
                if let Some(device) = device {
                    evaluator.operator_mut().close_stage(device);
                }
                if sync {
                    evaluator.operator_mut().sync();
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

        let start = Instant::now();
        let gradients = self.gradients();
        let (potential, gradient) = match self.evaluator.operator_mut() {
            ExecOperator::Host(_) => scaled_output(
                self.evaluator.target_output_store(),
                &self.targets,
                &self.radii,
                gradients,
            ),
            #[cfg(feature = "gpu")]
            ExecOperator::Device(driver) => match driver.read_output() {
                Ok(store) => {
                    let output = scaled_output(store, &self.targets, &self.radii, gradients);
                    timings.device = driver.stage_timings();
                    output
                }
                Err(error) => {
                    let reason = error.to_string();
                    self.device_error = Some(reason.clone());
                    return Err(FmmError::Device(reason));
                }
            },
        };
        timings.output = start.elapsed();
        Ok(Output {
            potential,
            gradient,
            timings,
        })
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
        self.evaluator.operator().host()
    }

    /// Returns the backend the operators run on ([`FmmBuilder::backend`]).
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns where `kind` runs: on the host for [`Backend::Host`]; with a device
    /// backend as its device report says (from Phase 4 T10 every kind runs on the device
    /// unless [`FmmBuilder::host_fallback`] names it).
    pub fn placement(&self, kind: OperatorKind) -> Placement {
        match self.evaluator.operator() {
            ExecOperator::Host(_) => {
                let _ = kind;
                Placement::Host
            }
            #[cfg(feature = "gpu")]
            ExecOperator::Device(driver) => driver.report().placement(kind),
        }
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

    /// Returns the number of sources on this rank.
    pub fn nsources(&self) -> usize {
        self.sources.len()
    }

    /// Returns the number of targets on this rank.
    pub fn ntargets(&self) -> usize {
        self.targets.len()
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
    /// (`nd_fmm_plan::index::LeafNumbering`).
    pub fn source_counts(&self) -> &[usize] {
        &self.sources.counts
    }

    /// Returns the number of targets in each local leaf, in leaf order.
    pub fn target_counts(&self) -> &[usize] {
        &self.targets.counts
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

    /// With `serial`, runs the next evaluations on the calling thread even if the FMM
    /// has a pool ([`LaplaceOperator::set_serial`]): the serial path of the same build,
    /// for comparisons and timings. The output is the same bit for bit.
    pub fn set_serial(&mut self, serial: bool) {
        self.evaluator.operator_mut().host_mut().set_serial(serial);
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
        match self.evaluator.operator_mut() {
            ExecOperator::Host(_) => Ok((
                self.evaluator.multipoles().as_slice().to_vec(),
                self.evaluator.locals().as_slice().to_vec(),
            )),
            #[cfg(feature = "gpu")]
            ExecOperator::Device(driver) => driver
                .download_expansions()
                .map_err(|error| FmmError::Device(error.to_string())),
        }
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
        match self.evaluator.operator() {
            ExecOperator::Device(driver) => Some(driver.report()),
            ExecOperator::Host(_) => None,
        }
    }

    /// With a device backend: the transfers, launches and syncs of the build and of the
    /// last evaluation ([`DeviceCounters`]); `None` on the host.
    pub fn device_counters(&self) -> Option<DeviceCounters> {
        match self.evaluator.operator() {
            ExecOperator::Device(driver) => Some(driver.counters()),
            ExecOperator::Host(_) => None,
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
        match self.evaluator.operator_mut() {
            ExecOperator::Device(driver) => Some(
                driver
                    .download_views()
                    .map_err(|error| FmmError::Device(error.to_string())),
            ),
            ExecOperator::Host(_) => None,
        }
    }
}

/// The target output `store`, scaled (CONVENTIONS §3.13, "Output") and in the caller's
/// order of `targets`, with r_t of every local leaf in `radii`.
fn scaled_output<T: SimdScalar + Default>(
    store: &LeafStore<T>,
    targets: &LeafOrder,
    radii: &[f64],
    gradients: bool,
) -> (Vec<T>, Option<Vec<[T; 3]>>) {
    let ntargets = targets.len();
    let mut potential = vec![T::zero(); ntargets];
    let mut gradient = gradients.then(|| vec![[T::zero(); 3]; ntargets]);
    for (j, &r) in radii.iter().enumerate() {
        let points = targets.points(j);
        let (phi_hat, g_hat) = store.chunk(j).split_at(points.len());
        let (phi_scale, g_scale) = (4.0 * PI * r, 4.0 * PI * r * r);
        for (&phi, &i) in phi_hat.iter().zip(points) {
            potential[i] = T::from_f64(RealScalar::to_f64(phi) / phi_scale);
        }
        if let Some(gradient) = gradient.as_mut() {
            for (g, &i) in g_hat.as_chunks::<3>().0.iter().zip(points) {
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
