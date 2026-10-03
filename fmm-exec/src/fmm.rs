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
//!    every point is finite, builds the thread pool if `threads` > 1, and agrees the
//!    outcome on every rank (one all-reduce).
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
//!    global count (one all-reduce): points are not redistributed until C5.1.
//! 6. Sorts sources and targets into leaf order, stably, and keeps both permutations.
//! 7. All-reduces the largest number of sources in a leaf, which sizes the P2P scratch;
//!    builds or loads the tables, the operator, and the evaluator with the per-leaf
//!    counts (its own collectives).
//! 8. Writes the leaf-scaled source coordinates and target positions into the
//!    evaluator's stores, once ([`leaf_coordinates`]).
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
use std::sync::Arc;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::{Evaluator, EvaluatorError};
use nd_fmm_plan::plan::{Plan, PlanError};
use nd_fmm_tables::cache::{Stored, TableKind};
use nd_fmm_tables::{CacheOutcome, TableCache};
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::octree::compute_global_bounding_box;
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, points_to_morton};
use rayon::{ThreadPool, ThreadPoolBuilder};
use rlst::SliceArray;
use thiserror::Error;

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

/// The deepest level of a Morton key.
const DEEPEST: usize = DEEPEST_LEVEL as usize;

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

    /// With more than one thread: checks that MPI provides `provided` ≥
    /// [`REQUIRED_MPI_THREADING`] and builds the pool.
    fn pool(&self, provided: Threading) -> Result<Option<Arc<ThreadPool>>, FmmError> {
        if self.threads == 1 {
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
        let provided = mpi::environment::threading_support();
        let local = self
            .check_settings()
            .map_err(FmmError::from)
            .and_then(|()| {
                let pool = self.pool(provided)?;
                check_finite(sources, PointSet::Sources)?;
                check_finite(targets, PointSet::Targets)?;
                let supplied = self.domain.map(|c| Domain::new(&PhysicalBox::new(c)));
                Ok((pool, supplied.transpose()?))
            });
        let (pool, supplied) = agree(comm, local)?;
        let threading = ThreadingReport::read(self.threads, provided);

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
        let (tables, cache_outcomes) = match &self.table_cache {
            Some(dir) => Tables::load_or_build(self.p, self.strategy, &TableCache::new(dir)),
            None => (Tables::build(self.p, self.strategy), Vec::new()),
        };
        let tables_time = start.elapsed();
        let start = Instant::now();
        let mut operator = LaplaceOperator::new(tables, self.gradients, max_leaf_points)
            .with_p2p(self.p2p)
            .expect("step 1 checked that this machine runs the P2P kernel");
        if let Some(pool) = pool {
            operator = operator.with_pool(pool);
        }
        let mut evaluator = Evaluator::new(
            plan,
            comm,
            operator,
            &sources_by_leaf.counts,
            &targets_by_leaf.counts,
        )
        .map_err(FmmError::Evaluator)?;
        let evaluator_time = start.elapsed();

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

        Ok(Fmm {
            octree,
            evaluator,
            domain,
            strategy: self.strategy.resolve(self.p),
            sources: sources_by_leaf,
            targets: targets_by_leaf,
            radii,
            max_leaf_points,
            cache_outcomes,
            threading,
            build_timings: BuildTimings {
                domain: domain_time,
                octree: octree_time,
                plan: plan_time,
                sort: sort_time,
                tables: tables_time,
                evaluator: evaluator_time,
                load: load_time,
            },
        })
    }
}

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
}

impl BuildTimings {
    /// The sum of all steps.
    pub fn total(&self) -> Duration {
        self.domain + self.octree + self.plan + self.sort + self.tables + self.evaluator + self.load
    }
}

/// The wall time of each stage of one [`Fmm::evaluate`] on this rank, for reports only.
///
/// The six evaluator stages are those of `nd_fmm_plan::evaluator`; the collective ones
/// include the time spent waiting for other ranks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimings {
    /// Writing the charges into the source chunks.
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
    /// Scaling the target output into the caller's order.
    pub output: Duration,
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
    evaluator: Evaluator<'o, C, LaplaceOperator<T>, Plan>,
    domain: Domain,
    strategy: M2lStrategy,
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
    /// source, [`FmmError::OtherRank`] on the others; nothing is evaluated.
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

        let mut timings = StageTimings::default();
        let start = Instant::now();
        let mut chunks = self.evaluator.local_sources_mut();
        for (j, chunk) in chunks.chunks_mut().enumerate() {
            let points = self.sources.points(j);
            let (_, values) = chunk.split_at_mut(3 * points.len());
            for (q, &i) in values.iter_mut().zip(points) {
                *q = charges[i];
            }
        }
        timings.load = start.elapsed();

        let evaluator = &mut self.evaluator;
        evaluator.reset();
        timings.exchange_sources = timed(|| evaluator.exchange_sources());
        timings.upward_local = timed(|| evaluator.upward_local());
        timings.upward_global = timed(|| evaluator.upward_global());
        timings.exchange_multipoles = timed(|| evaluator.exchange_multipoles());
        timings.downward = timed(|| evaluator.downward());
        timings.evaluate_leaves = timed(|| evaluator.evaluate_leaves());

        let start = Instant::now();
        let (potential, gradient) = self.output();
        timings.output = start.elapsed();
        Ok(Output {
            potential,
            gradient,
            timings,
        })
    }

    /// The target output of the last evaluation, scaled (CONVENTIONS §3.13, "Output")
    /// and in the caller's order.
    fn output(&self) -> (Vec<T>, Option<Vec<[T; 3]>>) {
        let ntargets = self.targets.len();
        let mut potential = vec![T::zero(); ntargets];
        let mut gradient = self.gradients().then(|| vec![[T::zero(); 3]; ntargets]);
        let store = self.evaluator.target_output_store();
        for (j, &r) in self.radii.iter().enumerate() {
            let points = self.targets.points(j);
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

    /// Returns the octree.
    pub fn octree(&self) -> &Octree<'o, C> {
        &self.octree
    }

    /// Returns the plan: the box index and the interaction lists.
    pub fn plan(&self) -> &Plan {
        self.evaluator.plan()
    }

    /// Returns the operator.
    pub fn operator(&self) -> &LaplaceOperator<T> {
        self.evaluator.operator()
    }

    /// Returns the domain.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Returns the expansion degree p.
    pub fn p(&self) -> usize {
        self.operator().p()
    }

    /// Returns the resolved strategy, never [`M2lStrategy::Auto`].
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
        self.evaluator.operator_mut().set_serial(serial);
    }
}

/// Runs `stage` and returns its wall time.
fn timed(stage: impl FnOnce()) -> Duration {
    let start = Instant::now();
    stage();
    start.elapsed()
}
