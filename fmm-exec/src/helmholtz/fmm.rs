//! The Helmholtz FMM (Helmholtz CH3.1b; docs/design/helmholtz-fmm-proposal.md §6.1–§6.3;
//! CONVENTIONS §3.15): [`HelmholtzFmmBuilder`] and [`HelmholtzFmm`], for real and complex
//! k with real and complex charges, on any number of ranks, RCR on every level.
//!
//! # Build
//!
//! [`HelmholtzFmmBuilder::build`] runs the point pipeline that [`Fmm`](crate::fmm::Fmm)
//! runs (the crate-private `pipeline` module; the `fmm` module documentation, "Build" and
//! "Several ranks"), with the Helmholtz operator in place of Laplace's:
//!
//! 1. the settings, the thread pool, finite points and the supplied domain, agreed on
//!    every rank; Yukawa (Re k = 0) is refused here, as its path is Helmholtz CH2.1's;
//! 2. whether any rank has a point;
//! 3. the domain, the supplied one or the bounding box of every rank's points, with every
//!    point strictly inside it;
//! 4. the octree of every point, with the leaf rule (proposal §3.7):
//!    [`min_leaf_level`] of k, the domain and c_leaf passed to
//!    `OctreeOptions::with_min_leaf_level`, so that no leaf holding a point is wider than
//!    c_leaf λ; and its plan. There is no near rule and no work cut: both weigh the work
//!    with Laplace's cost model;
//! 5. and 6. every point to the rank that owns its leaf, the leaf ranges, radii and the
//!    output order;
//! 7. the degree of every level ([`Degrees::resolve`]), the [`HelmholtzOperator`] with its
//!    tables, on the pool, and the evaluator; the largest leaf is agreed on every rank;
//! 8. the leaf-scaled coordinates (CONVENTIONS §3.13).
//!
//! The degrees, the leaf level and the tables depend on k, the domain, ε and c_leaf only
//! (proposal §2, constraint 9), so every rank finds the same.
//!
//! # Evaluation
//!
//! [`HelmholtzFmm::evaluate`] takes the m charge vectors of the build
//! ([`Charges`], all real or all complex as [`HelmholtzFmmBuilder::charges`] says) at this
//! rank's sources, in the caller's order, and returns the [`Output`] at this rank's
//! targets, in the caller's order, in §3.15's split planes, 1/(4π) included. The leaf
//! chunks of [`Densities`] are §3.13 chunks of c m source and o_m output vectors (the
//! `densities` module, "The leaf chunks"), so the evaluation is `Fmm`'s with those as the
//! vectors: the c m columns (Re q⁰, Im q⁰, …) packed point-major and forwarded to their
//! owners, the charge load, the six blocking stages of the evaluator, the output pass with
//! Laplace's scales 4π r and 4π r², the backward move, and the columns unpacked into the
//! output's planes.
//!
//! # Determinism
//!
//! The level calls run target by target on the shared drivers, each target's
//! contributions in the order of its row, so the output is the same at every thread count
//! and, on P ranks, the one-rank FMM's over the union of the points in rank order, bit for
//! bit (proposal §2, constraint 2), as for `Fmm`.
//!
//! Not yet: overlap, the device path, the near rule and the work cut, table caches and the
//! vectorised P2P (CH3.4).
//!
//! ```no_run
//! use mpi::traits::Communicator;
//! use nd_fmm_exec::helmholtz::{Charges, Degrees, HelmholtzFmmBuilder, Wavenumber};
//!
//! let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//! let points: Vec<[f64; 3]> = (0..1000)
//!     .map(|i| {
//!         let t = i as f64 / 1000.0;
//!         [t, (7.0 * t).fract(), (13.0 * t).fract()]
//!     })
//!     .collect();
//! let k = Wavenumber::new(20.0, 0.0).unwrap();
//! let mut fmm = HelmholtzFmmBuilder::new(k, Degrees::Tolerance(1e-6))
//!     .gradients(true)
//!     .build::<f64, _>(&points, &points, &world)
//!     .unwrap();
//! let (re, im) = (vec![1.0; points.len()], vec![0.5; points.len()]);
//! let output = fmm.evaluate(&[Charges::Complex { re: &re, im: &im }]).unwrap();
//! let phi = output.potential_of(0);
//! println!("Re φ = {}, Im φ = {:?}", phi.re[0], phi.im.map(|im| im[0]));
//! ```

use std::sync::Arc;

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PerPair,
};
use nd_fmm_plan::plan::Plan;
use nd_fmm_tables::helmholtz::DEFAULT_BLOCK_BUDGET;
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::{Octree, OctreeOptions, PhysicalBox};
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

use super::{
    C_LEAF, ChargeKind, Charges, DegreeError, Degrees, Densities, DensityError, HelmholtzOperator,
    OperatorError, Output, Wavenumber, WavenumberKind, min_leaf_level,
};
use crate::fmm::{DEFAULT_MAX_POINTS_PER_LEAF, FmmError, PointSet, SettingsError};
use crate::geometry::Domain;
use crate::operator::SimdScalar;
use crate::pipeline::{
    self, Points, check_finite, gather_output, load_source_charges, pack_charges,
    unpack_output_many,
};
use crate::threading::REQUIRED_MPI_THREADING;

/// The deepest level of a Morton key, 16 (CONVENTIONS §3.12).
const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// Why a Helmholtz FMM was not built or not evaluated.
#[derive(Debug, Error)]
pub enum HelmholtzFmmError {
    /// A step that the Helmholtz FMM shares with `Fmm` failed: the settings, the points,
    /// the domain, the plan, the redistribution, the evaluator or the thread pool.
    #[error(transparent)]
    Build(FmmError),
    /// c_leaf is not positive and finite.
    #[error("c_leaf must be positive and finite, got {0}")]
    LeafWidth(f64),
    /// The degrees could not be resolved.
    #[error("the degrees: {0}")]
    Degree(#[from] DegreeError),
    /// The operator refused k (Yukawa) or its inputs.
    #[error("the operator: {0}")]
    Operator(#[from] OperatorError),
    /// The charges of an evaluation do not fit the build.
    #[error("the charges: {0}")]
    Charges(#[from] DensityError),
    /// The input is invalid on another rank.
    #[error("the input is invalid on another rank")]
    OtherRank,
}

impl From<FmmError> for HelmholtzFmmError {
    /// [`FmmError::OtherRank`] as [`HelmholtzFmmError::OtherRank`], the rest as
    /// [`HelmholtzFmmError::Build`].
    fn from(error: FmmError) -> Self {
        match error {
            FmmError::OtherRank => Self::OtherRank,
            error => Self::Build(error),
        }
    }
}

/// The settings of a Helmholtz FMM ([module documentation](self)).
#[derive(Clone, Debug)]
pub struct HelmholtzFmmBuilder {
    k: Wavenumber,
    degrees: Degrees,
    charges: ChargeKind,
    vectors: usize,
    gradients: bool,
    threads: usize,
    domain: Option<[f64; 6]>,
    max_points_per_leaf: usize,
    max_level: usize,
    block_budget: usize,
    c_leaf: f64,
    pair_calls: bool,
}

impl HelmholtzFmmBuilder {
    /// The settings for wavenumber `k` with the degrees `degrees`: complex charges, one
    /// vector, potentials only, one thread, the domain from the points,
    /// [`DEFAULT_MAX_POINTS_PER_LEAF`] points per leaf, leaves down to level 16, the
    /// default block budget of the M2L y-blocks and the leaf width [`C_LEAF`].
    pub fn new(k: Wavenumber, degrees: Degrees) -> Self {
        Self {
            k,
            degrees,
            charges: ChargeKind::Complex,
            vectors: 1,
            gradients: false,
            threads: 1,
            domain: None,
            max_points_per_leaf: DEFAULT_MAX_POINTS_PER_LEAF,
            max_level: DEEPEST,
            block_budget: DEFAULT_BLOCK_BUDGET,
            c_leaf: C_LEAF,
            pair_calls: false,
        }
    }

    /// Real or complex charges (CONVENTIONS §3.15's real-charge flag; default complex).
    pub fn charges(mut self, charges: ChargeKind) -> Self {
        self.charges = charges;
        self
    }

    /// The number m ≥ 1 of charge vectors of every evaluation (default 1). Vector v of the
    /// output is the one-vector FMM's of vector v, bit for bit.
    pub fn vectors(mut self, vectors: usize) -> Self {
        self.vectors = vectors;
        self
    }

    /// Whether the output holds the gradients (default no).
    pub fn gradients(mut self, gradients: bool) -> Self {
        self.gradients = gradients;
        self
    }

    /// The threads of the level calls, the charge load and the output pass (default 1).
    /// The output is the same at every thread count. With more than one, MPI must provide
    /// at least `Threading::Funneled`.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// The domain `[x_min, y_min, z_min, x_max, y_max, z_max]`, a cube that holds every
    /// point strictly inside, the same on every rank (default: the bounding box of every
    /// rank's points).
    pub fn domain(mut self, domain: PhysicalBox) -> Self {
        self.domain = Some(domain.coordinates());
        self
    }

    /// The octree's refinement target, points per leaf (default
    /// [`DEFAULT_MAX_POINTS_PER_LEAF`]). The leaf rule refines further where k needs it.
    pub fn max_points_per_leaf(mut self, max_points_per_leaf: usize) -> Self {
        self.max_points_per_leaf = max_points_per_leaf;
        self
    }

    /// The deepest leaf level, at most 16 (default 16).
    pub fn max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// The bytes of the stored M2L y-blocks (`nd_fmm_tables::helmholtz::HelmholtzTables`;
    /// default [`DEFAULT_BLOCK_BUDGET`]); the output does not depend on it.
    pub fn block_budget(mut self, bytes: usize) -> Self {
        self.block_budget = bytes;
        self
    }

    /// The maximum leaf width in wavelengths (default [`C_LEAF`], decision D11).
    pub fn c_leaf(mut self, c_leaf: f64) -> Self {
        self.c_leaf = c_leaf;
        self
    }

    /// For tests: run the operator pair by pair (`nd_fmm_plan::operator::PerPair`) in place
    /// of its level calls. The output is the same bit for bit.
    #[doc(hidden)]
    pub fn pair_calls(mut self, on: bool) -> Self {
        self.pair_calls = on;
        self
    }

    /// The settings' own errors, checked on every rank before any collective.
    fn check_settings(&self) -> Result<(), HelmholtzFmmError> {
        if self.k.kind() == WavenumberKind::Yukawa {
            return Err(OperatorError::Yukawa.into());
        }
        let settings = if self.max_level > DEEPEST {
            Some(SettingsError::MaxLevelTooDeep {
                max_level: self.max_level,
            })
        } else if self.max_points_per_leaf == 0 {
            Some(SettingsError::ZeroPointsPerLeaf)
        } else if self.threads == 0 {
            Some(SettingsError::ZeroThreads)
        } else if self.vectors == 0 {
            Some(SettingsError::ZeroVectors)
        } else {
            None
        };
        if let Some(error) = settings {
            return Err(FmmError::InvalidSettings(error).into());
        }
        if !(self.c_leaf > 0.0 && self.c_leaf.is_finite()) {
            return Err(HelmholtzFmmError::LeafWidth(self.c_leaf));
        }
        if let Degrees::Tolerance(_) = self.degrees {
            // The tolerance's own check; the levels come with the plan.
            self.degrees.resolve(&self.k, &unit_domain(), 0)?;
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
            .thread_name(|i| format!("nd-fmm-exec-helmholtz-{i}"))
            .build()
            .map(|pool| Some(Arc::new(pool)))
            .map_err(|error| FmmError::ThreadPool(error.to_string()))
    }

    /// Builds the FMM of `sources` and `targets`, in the caller's order ([module
    /// documentation](self#build)). The two sets may overlap or coincide; a rank may pass
    /// no points.
    ///
    /// # Collective operation
    ///
    /// Every rank of `comm` must call it, with the same settings.
    ///
    /// # Errors
    ///
    /// The settings' errors ([`HelmholtzFmmError::Operator`] for Yukawa,
    /// [`HelmholtzFmmError::LeafWidth`], [`HelmholtzFmmError::Degree`],
    /// [`HelmholtzFmmError::Build`] with a `SettingsError`), a point that is not finite or
    /// outside the supplied domain, no point on any rank, and the errors of the shared
    /// steps; the rank that found an input error returns it, the others
    /// [`HelmholtzFmmError::OtherRank`].
    pub fn build<'o, T, C>(
        &self,
        sources: &[[f64; 3]],
        targets: &[[f64; 3]],
        comm: &'o C,
    ) -> Result<HelmholtzFmm<'o, T, C>, HelmholtzFmmError>
    where
        T: SimdScalar + Equivalence + Default,
        C: CommunicatorCollectives,
    {
        // Step 1: settings, MPI threading and the pool, finite points, the supplied
        // domain; one agreement.
        let provided = mpi::environment::threading_support();
        let local = self.check_settings().and_then(|()| {
            let pool = self.pool(provided)?;
            check_finite(sources, PointSet::Sources)?;
            check_finite(targets, PointSet::Targets)?;
            let supplied = self.domain.map(|c| Domain::new(&PhysicalBox::new(c)));
            let supplied = supplied.transpose().map_err(FmmError::from)?;
            Ok((pool, supplied))
        });
        let (pool, supplied) = agree(comm, local)?;

        // Step 2: are there points at all?
        let local = (sources.len() + targets.len()) as u64;
        let mut total = 0u64;
        comm.all_reduce_into(&local, &mut total, SystemOperation::max());
        if total == 0 {
            return Err(FmmError::NoPoints.into());
        }

        // Step 3: the domain, and every point strictly inside it.
        let (coordinates, domain) = pipeline::domain(sources, targets, supplied, comm)?;

        // Step 4: the octree with the leaf rule, and its plan.
        let leaf_level = min_leaf_level(&self.k, &domain, self.c_leaf);
        let options = OctreeOptions::new()
            .with_max_level(self.max_level)
            .with_max_fine_keys(self.max_points_per_leaf)
            .with_ghost_children(true)
            .with_min_leaf_level(leaf_level);
        let (keys, octree) = pipeline::octree(&coordinates, &domain, options, comm);
        let plan = Plan::new(&octree).map_err(FmmError::Plan)?;

        // Steps 5 and 6: the points to their owners. Three coordinates per point move now,
        // and on every evaluation c m strengths per source and the output values per
        // target.
        let densities = Densities::new(self.k.kind(), self.charges, self.vectors);
        let output_values = densities.target_output_point_size(self.gradients);
        let per_item = output_values.max(3).max(densities.columns());
        let (points, owned) = Points::route(
            &octree,
            &plan,
            (&keys, sources.len()),
            &coordinates,
            &domain,
            per_item,
        )?;
        drop(keys);
        drop(coordinates);

        // Step 7: the degrees, the operator and the evaluator. The largest number of
        // sources in a leaf, over every rank, sizes P2P's scratch.
        let local = points
            .source_leaves
            .counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let mut max_leaf_points = 0usize;
        comm.all_reduce_into(&local, &mut max_leaf_points, SystemOperation::max());
        let degrees = agree(
            comm,
            self.degrees
                .resolve(&self.k, &domain, plan.nlevels())
                .map_err(HelmholtzFmmError::from),
        )?;
        let mut operator = HelmholtzOperator::with_block_budget(
            self.k,
            densities,
            &domain,
            &degrees,
            self.gradients,
            max_leaf_points,
            self.block_budget,
        )?;
        if let Some(pool) = &pool {
            operator = operator.with_pool(Arc::clone(pool));
        }
        let calls = if self.pair_calls {
            Calls::Pairs(PerPair(operator))
        } else {
            Calls::Levels(operator)
        };
        let mut evaluator = Evaluator::new(
            plan,
            comm,
            calls,
            &points.source_leaves.counts,
            &points.target_leaves.counts,
        )
        .map_err(FmmError::Evaluator)?;

        // Step 8: the leaf-scaled coordinates.
        points.load_coordinates(&owned, &mut evaluator, &domain);
        drop(owned);

        let strengths = densities.columns();
        Ok(HelmholtzFmm {
            octree,
            evaluator,
            domain,
            densities,
            gradients: self.gradients,
            degrees,
            min_leaf_level: leaf_level,
            max_leaf_points,
            pool,
            packed: Vec::new(),
            charges: vec![T::zero(); strengths * points.sources.nreceived()],
            received_output: vec![T::zero(); output_values * points.targets.nreceived()],
            returned_output: vec![T::zero(); output_values * points.targets.nsent()],
            points,
        })
    }
}

/// The unit cube, for the tolerance's own check.
fn unit_domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).expect("the unit cube")
}

/// Agrees `local` on every rank: the rank that failed returns its error, the others
/// [`HelmholtzFmmError::OtherRank`] (one all-reduce).
fn agree<C: CommunicatorCollectives, V>(
    comm: &C,
    local: Result<V, HelmholtzFmmError>,
) -> Result<V, HelmholtzFmmError> {
    let mut valid = false;
    comm.all_reduce_into(&local.is_ok(), &mut valid, SystemOperation::logical_and());
    match local {
        Ok(_) if !valid => Err(HelmholtzFmmError::OtherRank),
        result => result,
    }
}

/// The operator of the evaluator: the level calls, or (for tests) the same bodies pair by
/// pair.
enum Calls<T: SimdScalar + Equivalence + Default> {
    Levels(HelmholtzOperator<T>),
    Pairs(PerPair<HelmholtzOperator<T>>),
}

impl<T: SimdScalar + Equivalence + Default> Calls<T> {
    fn operator(&self) -> &HelmholtzOperator<T> {
        match self {
            Self::Levels(operator) | Self::Pairs(PerPair(operator)) => operator,
        }
    }
}

/// Every method delegates to the operator.
macro_rules! delegate {
    ($($kind:ident: $batch:ident),* $(,)?) => {
        $(
            fn $kind(&mut self, batch: $batch<'_, T>) {
                match self {
                    Self::Levels(operator) => FmmOperator::$kind(operator, batch),
                    Self::Pairs(pairs) => pairs.$kind(batch),
                }
            }
        )*
    };
}

impl<T: SimdScalar + Equivalence + Default> FmmSizes for Calls<T> {
    type Value = T;

    fn multipole_size(&self, level: usize) -> usize {
        self.operator().multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.operator().local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.operator().source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.operator().target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.operator().target_output_point_size()
    }
}

impl<T: SimdScalar + Equivalence + Default> FmmOperator for Calls<T> {
    delegate!(p2m: P2m, m2m: M2m, m2l: M2l, p2l: P2l, l2l: L2l, l2p: L2p, m2p: M2p, p2p: P2p);
}

/// A Helmholtz FMM built on the points of every rank ([module documentation](self)).
pub struct HelmholtzFmm<'o, T, C = SimpleCommunicator>
where
    T: SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    octree: Octree<'o, C>,
    evaluator: Evaluator<'o, C, Calls<T>, Plan>,
    domain: Domain,
    densities: Densities,
    gradients: bool,
    degrees: Vec<usize>,
    min_leaf_level: usize,
    max_leaf_points: usize,
    pool: Option<Arc<ThreadPool>>,
    /// The caller's columns packed point-major (c m per source) for the forward move, when
    /// there is more than one column; allocated by the first evaluation that needs it.
    packed: Vec<T>,
    /// The strengths of the owned sources in leaf order, c m per source.
    charges: Vec<T>,
    /// The output pass's values, o per owned target in leaf order.
    received_output: Vec<T>,
    /// The backward move's values, o per target in the caller's order.
    returned_output: Vec<T>,
    /// The routes, the leaf ranges, the radii and the output order.
    points: Points,
}

impl<'o, T, C> HelmholtzFmm<'o, T, C>
where
    T: SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    /// Evaluates the potentials, and the gradients if built with them, of the m charge
    /// vectors `charges` at every target of this rank ([module
    /// documentation](self#evaluation)): `charges[v]` holds vector v at this rank's
    /// sources of [`HelmholtzFmmBuilder::build`], in their order. Allocates the output:
    /// [`evaluate_into`](Self::evaluate_into) a fresh [`Output`], with the same bits.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: one all-reduce, then the collectives of the evaluator's
    /// stages and of the two moves.
    ///
    /// # Errors
    ///
    /// [`HelmholtzFmmError::Charges`] on a rank whose charges do not fit the build
    /// ([`Densities::check_charges`]), [`HelmholtzFmmError::OtherRank`] on the others;
    /// nothing is evaluated.
    pub fn evaluate(&mut self, charges: &[Charges<'_, T>]) -> Result<Output<T>, HelmholtzFmmError> {
        let mut output = Output::new(&self.densities, 0, self.gradients);
        self.evaluate_into(charges, &mut output)?;
        Ok(output)
    }

    /// [`evaluate`](Self::evaluate) into `output`, whose buffers are reused: the same bits.
    /// An `output` of another path (vectors, or real and complex planes) is replaced by
    /// one of this path first; its buffers are resized as needed, every element written
    /// once. On an error `output` is left as it was.
    ///
    /// # Collective operation
    ///
    /// As [`evaluate`](Self::evaluate).
    ///
    /// # Errors
    ///
    /// As [`evaluate`](Self::evaluate).
    pub fn evaluate_into(
        &mut self,
        charges: &[Charges<'_, T>],
        output: &mut Output<T>,
    ) -> Result<(), HelmholtzFmmError> {
        let local = self
            .densities
            .check_charges(charges, self.points.sources.nsent())
            .map_err(HelmholtzFmmError::from);
        agree(self.evaluator.comm(), local)?;
        let pool = self.pool.as_deref();

        // The c m columns, Re q then Im q per complex vector, packed point-major and moved
        // to the ranks that own their sources, in leaf order.
        let columns: Vec<&[T]> = charges
            .iter()
            .flat_map(|vector| match *vector {
                Charges::Real(q) => [Some(q), None],
                Charges::Complex { re, im } => [Some(re), Some(im)],
            })
            .flatten()
            .collect();
        let w = columns.len();
        if w > 1 {
            pack_charges(&columns, pool, &mut self.packed);
        }
        let input = if w > 1 { &self.packed[..] } else { columns[0] };
        self.points
            .sources
            .forward_into(input, w, &mut self.charges);
        load_source_charges(
            self.evaluator.local_sources_mut(),
            &self.points.source_leaves,
            &self.charges,
            (w, None),
            pool,
        );

        self.evaluator.evaluate();

        // The output pass with the output columns as the vectors, the move back, and the
        // columns into the output's planes.
        let o = self.densities.output_columns();
        let store = self.evaluator.target_output_store();
        gather_output(
            store.as_slice(),
            store.point_offsets(),
            &self.points.outputs,
            (self.gradients, o),
            pool,
            &mut self.received_output,
        );
        let per_target = self.densities.target_output_point_size(self.gradients);
        self.points.targets.backward_into(
            &self.received_output,
            per_target,
            &mut self.returned_output,
        );
        if output.vectors() != self.densities.vectors()
            || output.is_complex() != (o == 2 * self.densities.vectors())
        {
            *output = Output::new(&self.densities, 0, self.gradients);
        }
        let buffers = (&mut output.potential, &mut output.gradient);
        unpack_output_many(&self.returned_output, (self.gradients, o), pool, buffers);
        Ok(())
    }

    /// The output of the last evaluation by the output pass of
    /// [`Densities::write_leaf_output`] (Helmholtz CH3.0c) leaf by leaf, moved back column
    /// by column, for tests only: [`evaluate`](Self::evaluate) returns it bit for bit.
    ///
    /// # Collective operation
    ///
    /// Every rank must call it: the backward moves of every column.
    #[doc(hidden)]
    pub fn reference_output(&self) -> Output<T> {
        let ntargets = self.points.targets.nreceived();
        let mut received = Output::new(&self.densities, ntargets, self.gradients);
        let store = self.evaluator.target_output_store();
        for (j, &radius) in self.points.radii.iter().enumerate() {
            let targets: Vec<usize> = self.points.target_leaves.range(j).collect();
            self.densities.write_leaf_output(
                store.chunk(j),
                radius,
                self.gradients,
                &targets,
                &mut received,
            );
        }
        let mut output = Output::new(&self.densities, 0, self.gradients);
        for column in 0..self.densities.output_columns() {
            let at = column * ntargets..(column + 1) * ntargets;
            let moved = self
                .points
                .targets
                .backward(&received.potential[at.clone()], 1);
            output.potential.extend(moved);
            if let (Some(all), Some(gradient)) = (output.gradient.as_mut(), &received.gradient) {
                let flat = self.points.targets.backward(gradient[at].as_flattened(), 3);
                all.extend_from_slice(flat.as_chunks::<3>().0);
            }
        }
        output
    }

    /// The octree.
    pub fn octree(&self) -> &Octree<'o, C> {
        &self.octree
    }

    /// The plan.
    pub fn plan(&self) -> &Plan {
        self.evaluator.plan()
    }

    /// The operator.
    pub fn operator(&self) -> &HelmholtzOperator<T> {
        self.evaluator.operator().operator()
    }

    /// The domain.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// The path: k's kind, the charges' kind and the vectors.
    pub fn densities(&self) -> &Densities {
        &self.densities
    }

    /// Whether the output holds gradients.
    pub fn gradients(&self) -> bool {
        self.gradients
    }

    /// The degree p_l of every level.
    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    /// The number of levels of the plan.
    pub fn nlevels(&self) -> usize {
        self.degrees.len()
    }

    /// The number of local leaves.
    pub fn nleaves(&self) -> usize {
        self.points.radii.len()
    }

    /// The leaf level of the leaf rule: no leaf holding a point lies above it (unless
    /// the maximum level does).
    pub fn min_leaf_level(&self) -> usize {
        self.min_leaf_level
    }

    /// The largest number of sources in a leaf, over every rank.
    pub fn max_leaf_points(&self) -> usize {
        self.max_leaf_points
    }

    /// This rank's sources and targets, as passed to the build.
    pub fn npoints(&self) -> (usize, usize) {
        (self.points.sources.nsent(), self.points.targets.nsent())
    }
}
