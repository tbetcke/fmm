//! The point pipeline that [`Fmm`](crate::fmm::Fmm) and the Helmholtz FMM
//! ([`HelmholtzFmm`](crate::helmholtz::HelmholtzFmm)) share (Helmholtz CH3.1b): the steps
//! of a build and an evaluation that do not depend on the operator. Crate-private.
//!
//! - Build: the domain and the points strictly inside it ([`domain`], step 3 of
//!   [`FmmBuilder::build`](crate::fmm::FmmBuilder::build)); the Morton keys and the octree
//!   ([`octree`], step 4's first part); the two redistributions, the coordinates forwarded
//!   into leaf order, the leaf ranges, the radii and the output order ([`Points::route`],
//!   steps 5 and 6); the leaf-scaled coordinates in the evaluator's chunks
//!   ([`Points::load_coordinates`], step 8). The `fmm` module documentation, "Build" and
//!   "Several ranks", describes them; the caller adds its own steps between them (for
//!   `Fmm` the near rule, the work cut, the tables and the device).
//! - Evaluation: the charge load into the source chunks ([`load_source_charges`]), the
//!   packing of several vectors ([`pack_charges`]), the output pass ([`gather_output`]) and
//!   the unpacking after the backward move ([`unpack_output_many`]), with the agreement of
//!   local errors ([`agree`]). The `fmm` module documentation, "Evaluation", "The output
//!   pass" and "Several charge vectors", describes them.
//!
//! Every function is the code `fmm` ran before the extraction, statement for statement, so
//! `Fmm`'s output is unchanged bit for bit. The Helmholtz FMM runs the same chunk layouts
//! with its columns as the vectors (`helmholtz::densities`, "The leaf chunks").

use std::f64::consts::PI;

use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::operator::FmmOperator;
use nd_fmm_plan::plan::Plan;
use nd_fmm_plan::redistribute::{Redistribution, RedistributionError};
use nd_fmm_plan::store::LeafSliceMut;
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::octree::compute_global_bounding_box;
use nd_octree::{MortonKey, Octree, OctreeOptions, points_to_morton};
use rayon::ThreadPool;
use rayon::prelude::*;
use rlst::SliceArray;

use crate::fmm::{FmmError, Output, PointSet, SourceType};
use crate::geometry::{Domain, leaf_coordinates, radius};
use crate::operator::SimdScalar;

/// The deepest level of a Morton key (CONVENTIONS §3.12).
const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// Step 3 of a build: the coordinates of `sources` then `targets` (x, y, z per point), and
/// the domain, `supplied` or else the bounding box of every rank's points, with every
/// point strictly inside it, agreed on every rank.
///
/// # Collective operation
///
/// Without `supplied` two all-reduces of [`global_extent`] and those of
/// `compute_global_bounding_box`; then [`agree`]'s all-reduce.
///
/// # Errors
///
/// [`FmmError::DegenerateExtent`] on every rank where the points span nothing; the domain's
/// own error; [`FmmError::PointOutsideDomain`] on the rank with such a point and
/// [`FmmError::OtherRank`] on the others.
pub(crate) fn domain<C: CommunicatorCollectives>(
    sources: &[[f64; 3]],
    targets: &[[f64; 3]],
    supplied: Option<Domain>,
    comm: &C,
) -> Result<(Vec<f64>, Domain), FmmError> {
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
    Ok((coordinates, domain))
}

/// Step 4's first part: the Morton keys of the points `coordinates` (of [`domain`]) at
/// the deepest level, and the octree of every rank's keys with `options`.
///
/// # Collective operation
///
/// Those of `Octree::new`.
pub(crate) fn octree<'o, C: CommunicatorCollectives>(
    coordinates: &[f64],
    domain: &Domain,
    options: OctreeOptions,
    comm: &'o C,
) -> (Vec<MortonKey>, Octree<'o, C>) {
    let points = SliceArray::from_shape(coordinates, [3, coordinates.len() / 3]);
    let keys = points_to_morton(&points, DEEPEST, &domain.physical_box());
    let octree = Octree::new(&keys, options, comm);
    (keys, octree)
}

/// The points of a build where they are owned (steps 5 and 6): the routes of the sources
/// and targets to the ranks that own their leaves and back, the owned points of every
/// local leaf, r of every local leaf and the output order.
pub(crate) struct Points {
    /// The route of the sources from the caller's ranks to their owners (Phase 5 T6).
    pub(crate) sources: Redistribution,
    /// The route of the targets, and of their output back.
    pub(crate) targets: Redistribution,
    /// The owned sources of every local leaf, in leaf order.
    pub(crate) source_leaves: LeafRanges,
    /// The owned targets of every local leaf, in leaf order.
    pub(crate) target_leaves: LeafRanges,
    /// r_t of every local leaf.
    pub(crate) radii: Vec<f64>,
    /// The output pass's position and leaf of every owned target, and the scales of every
    /// leaf.
    pub(crate) outputs: OutputOrder,
}

/// The coordinates of the owned sources and targets in leaf order, as forwarded by
/// [`Points::route`], until [`Points::load_coordinates`] writes them into the chunks.
pub(crate) type OwnedCoordinates = (Vec<f64>, Vec<f64>);

impl Points {
    /// Steps 5 and 6: every point to the rank that owns its leaf (Phase 5 T6,
    /// docs/design/distributed-fmm.md §4.4), one redistribution for the sources and one
    /// for the targets, routed by their `keys` (the `nsources` sources, then the targets,
    /// as [`octree`] gave them) on `plan` of `octree`, and the f64 `coordinates` of [`domain`] forwarded into
    /// the received order, which is leaf order; then the points of every local leaf, its
    /// radius and the output order. `per_item` is the largest number of values per point
    /// that a move of the build's evaluations carries.
    ///
    /// # Collective operation
    ///
    /// Those of two `Redistribution::new` and two forwards; every rank must enter.
    ///
    /// # Errors
    ///
    /// [`FmmError::Redistribution`], on every rank alike: each `new` agrees its own
    /// errors, and `max_per_item` is the same on every rank.
    pub(crate) fn route<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        plan: &Plan,
        (keys, nsources): (&[MortonKey], usize),
        coordinates: &[f64],
        domain: &Domain,
        per_item: usize,
    ) -> Result<(Self, OwnedCoordinates), FmmError> {
        let nlocal = plan.index().leaves().nlocal();
        let (source_keys, target_keys) = keys.split_at(nsources);
        let source_route =
            Redistribution::new(octree, plan, source_keys).map_err(FmmError::Redistribution)?;
        let target_route =
            Redistribution::new(octree, plan, target_keys).map_err(FmmError::Redistribution)?;
        if source_route.max_per_item().min(target_route.max_per_item()) < per_item {
            return Err(FmmError::Redistribution(RedistributionError::Overflow));
        }
        let (source_coordinates, target_coordinates) = coordinates.split_at(3 * nsources);
        let owned_sources = source_route.forward(source_coordinates, 3);
        let owned_targets = target_route.forward(target_coordinates, 3);

        let source_leaves = LeafRanges::new(source_route.counts());
        let target_leaves = LeafRanges::new(target_route.counts());
        let radii: Vec<f64> = (0..nlocal)
            .map(|j| radius(plan.index().leaves().level(j), domain))
            .collect();
        let outputs = OutputOrder::new(&target_leaves, &radii);
        let points = Self {
            sources: source_route,
            targets: target_route,
            source_leaves,
            target_leaves,
            radii,
            outputs,
        };
        Ok((points, (owned_sources, owned_targets)))
    }

    /// Step 8: the leaf-scaled coordinates (CONVENTIONS §3.13), once, from the forwarded
    /// coordinates `owned` in leaf order, into the source chunks and the target input of
    /// every local leaf of `evaluator`, in the units of `domain`.
    pub(crate) fn load_coordinates<C, Op, T>(
        &self,
        owned: &OwnedCoordinates,
        evaluator: &mut Evaluator<'_, C, Op, Plan>,
        domain: &Domain,
    ) where
        C: CommunicatorCollectives,
        Op: FmmOperator<Value = T>,
        T: SimdScalar,
    {
        let (owned_sources, owned_targets) = owned;
        let nlocal = self.radii.len();
        let leaf_keys = evaluator.plan().index().leaves().keys()[..nlocal].to_vec();
        let as_point = |values: &[f64]| [values[0], values[1], values[2]];
        for (j, &key) in leaf_keys.iter().enumerate() {
            let chunk = evaluator.sources_mut(j);
            let owned = &owned_sources[3 * self.source_leaves.offsets[j]..];
            for (u, x) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(owned.chunks(3)) {
                *u = leaf_coordinates(as_point(x), key, domain);
            }
            let chunk = evaluator.target_input_mut(j);
            let owned = &owned_targets[3 * self.target_leaves.offsets[j]..];
            for (u, x) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(owned.chunks(3)) {
                *u = leaf_coordinates(as_point(x), key, domain);
            }
        }
    }
}

pub(crate) fn agree<C: CommunicatorCollectives, V>(
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
pub(crate) fn check_finite(points: &[[f64; 3]], set: PointSet) -> Result<(), FmmError> {
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
pub(crate) fn check_inside(
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
pub(crate) fn global_extent<C: CommunicatorCollectives>(
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
pub(crate) struct LeafRanges {
    pub(crate) counts: Vec<usize>,
    pub(crate) offsets: Vec<usize>,
}

impl LeafRanges {
    /// The ranges of `counts[j]` points in local leaf j.
    pub(crate) fn new(counts: &[usize]) -> Self {
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
    pub(crate) fn range(&self, j: usize) -> std::ops::Range<usize> {
        self.offsets[j]..self.offsets[j + 1]
    }

    /// The number of points.
    pub(crate) fn len(&self) -> usize {
        *self.offsets.last().unwrap()
    }
}

/// The points per task below which the charge load ([`load_source_charges`]) stops
/// splitting its work.
pub(crate) const LOAD_GRAIN: usize = 4096;

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
pub(crate) struct OutputOrder {
    /// For every owned target in received order: its point in leaf order (the received
    /// order is leaf order, so target r is point r).
    pub(crate) points: Vec<u32>,
    /// For every owned target in received order: its local leaf.
    pub(crate) leaves: Vec<u32>,
    /// For every local leaf: 4π r_t and 4π r_t², formed as the pass before T9 formed them.
    pub(crate) scales: Vec<[f64; 2]>,
}

impl OutputOrder {
    /// The order of the owned targets `targets`, whose local leaf j has r_t = `radii[j]`.
    ///
    /// # Panics
    ///
    /// If this rank owns more than `u32::MAX` targets.
    pub(crate) fn new(targets: &LeafRanges, radii: &[f64]) -> Self {
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
/// sources' `Redistribution`), m = `vectors` per source, into the charge slots of the
/// source chunks `chunks` of the local leaves (CONVENTIONS §3.13, "Source chunks" and
/// "Several charge vectors"): leaf j gets `charges[m sources.range(j)]`, point-major as
/// forwarded. On `pool` when given, the leaves split in halves under `rayon::join` into
/// disjoint parts of at most [`LOAD_GRAIN`] points (or one leaf); serially otherwise.
/// Copies only, so the same values either way (Phase 4S T9).
pub(crate) fn load_source_charges<T: SimdScalar>(
    chunks: LeafSliceMut<'_, T>,
    sources: &LeafRanges,
    charges: &[T],
    (vectors, moments): (usize, Option<StrengthLayout<'_>>),
    pool: Option<&ThreadPool>,
) {
    let input = (sources, charges, vectors, moments);
    match pool {
        Some(pool) => pool.install(|| split_source_charges(chunks, 0, input)),
        None => write_source_charges(chunks, 0, input),
    }
}

/// The input of the charge load: the owned sources' leaves, the strengths, the strengths
/// per source and, with dipoles, how to lay them out.
pub(crate) type ChargeInput<'a, T> = (&'a LeafRanges, &'a [T], usize, Option<StrengthLayout<'a>>);

/// How the charge load lays out the strengths of a source type with dipoles (Phase 6 T8;
/// CONVENTIONS §3.13, "Source chunks"): the w = 3 or 4 values of a source arrive together
/// (its charge first, if any), and leaf j's chunk takes the n charges, then the n moments
/// d̂ = d / r_j, each component `T::from_f64(d.to_f64() / r_j)`: one rounding in f64 (r_j =
/// w / 2^(l+1) is exact, so this is fl(d · 2^(l+1) / w)), then the rounding to T.
#[derive(Clone, Copy)]
pub(crate) struct StrengthLayout<'a> {
    pub(crate) sources: SourceType,
    /// r of every local leaf.
    pub(crate) radii: &'a [f64],
}

/// [`load_source_charges`] on the calling thread, for the leaves of `chunks`, the first
/// of which is local leaf `first`.
pub(crate) fn write_source_charges<T: SimdScalar>(
    mut chunks: LeafSliceMut<'_, T>,
    first: usize,
    (sources, charges, m, moments): ChargeInput<'_, T>,
) {
    for (r, chunk) in chunks.chunks_mut().enumerate() {
        let points = sources.range(first + r);
        let (_, values) = chunk.split_at_mut(3 * points.len());
        let input = &charges[m * points.start..m * points.end];
        let layout = moments.map(|layout| (layout.sources, Some(layout.radii[first + r])));
        write_leaf_strengths(values, input, m, layout);
    }
}

/// One local leaf's strengths: `values` its chunk after the coordinate triples, `input`
/// its sources' m strengths each, as forwarded. Copied as they are without `moments`; with
/// `moments` = (the source type, the leaf's r) the charges first and then the moments,
/// each component `T::from_f64(d.to_f64() / r)`, or copied as they are with `None` for r
/// (already scaled, Phase 6 T8d: [`scale_moments`]).
pub(crate) fn write_leaf_strengths<T: SimdScalar>(
    values: &mut [T],
    input: &[T],
    m: usize,
    moments: Option<(SourceType, Option<f64>)>,
) {
    let Some((sources, radius)) = moments else {
        values.copy_from_slice(input);
        return;
    };
    let n = input.len() / m;
    let q = usize::from(sources.has_charges());
    let (leaf_charges, leaf_moments) = values.split_at_mut(q * n);
    for (j, strengths) in input.chunks_exact(m).enumerate() {
        if q == 1 {
            leaf_charges[j] = strengths[0];
        }
        for k in 0..3 {
            let d = strengths[q + k];
            leaf_moments[3 * j + k] = match radius {
                Some(radius) => T::from_f64(d.to_f64() / radius),
                None => d,
            };
        }
    }
}

/// Scales the moments of the forwarded strengths `upload` (w = 3 or 4 per source, leaf
/// order, the charge first if any) into their leaf's units in place, for the device's
/// upload (Phase 6 T8d): each component `T::from_f64(d.to_f64() / r_j)`, the bits the charge
/// load writes into the host's chunks; the charges stay as they are. On `pool` when given,
/// halving the leaves as the charge load does.
pub(crate) fn scale_moments<T: SimdScalar>(
    upload: &mut [T],
    sources: &LeafRanges,
    layout: StrengthLayout<'_>,
    pool: Option<&ThreadPool>,
) {
    let leaves = 0..layout.radii.len();
    match pool {
        Some(pool) => pool.install(|| split_scale_moments(upload, sources, layout, leaves)),
        None => write_scaled_moments(upload, sources, layout, leaves),
    }
}

/// [`scale_moments`] on the calling thread for the local leaves `leaves`, whose strengths
/// `upload` holds.
pub(crate) fn write_scaled_moments<T: SimdScalar>(
    upload: &mut [T],
    sources: &LeafRanges,
    StrengthLayout {
        sources: kind,
        radii,
    }: StrengthLayout<'_>,
    leaves: std::ops::Range<usize>,
) {
    let w = kind.strengths();
    let q = usize::from(kind.has_charges());
    let first = sources.offsets[leaves.start];
    for j in leaves {
        let radius = radii[j];
        for point in sources.range(j) {
            let at = w * (point - first) + q;
            for d in &mut upload[at..at + 3] {
                *d = T::from_f64(d.to_f64() / radius);
            }
        }
    }
}

/// [`scale_moments`] on the current pool for the local leaves `leaves`, halving them until
/// a half holds at most [`LOAD_GRAIN`] points or one leaf.
pub(crate) fn split_scale_moments<T: SimdScalar>(
    upload: &mut [T],
    sources: &LeafRanges,
    layout: StrengthLayout<'_>,
    leaves: std::ops::Range<usize>,
) {
    let points = sources.offsets[leaves.end] - sources.offsets[leaves.start];
    if leaves.len() < 2 || points <= LOAD_GRAIN {
        write_scaled_moments(upload, sources, layout, leaves);
        return;
    }
    let mid = leaves.start + leaves.len() / 2;
    let w = layout.sources.strengths();
    let (left, right) =
        upload.split_at_mut(w * (sources.offsets[mid] - sources.offsets[leaves.start]));
    rayon::join(
        || split_scale_moments(left, sources, layout, leaves.start..mid),
        || split_scale_moments(right, sources, layout, mid..leaves.end),
    );
}

/// [`load_source_charges`] on the current pool, halving the leaves of `chunks` (the
/// first local leaf `first`) until a half holds at most [`LOAD_GRAIN`] points or one leaf.
pub(crate) fn split_source_charges<T: SimdScalar>(
    mut chunks: LeafSliceMut<'_, T>,
    first: usize,
    input: ChargeInput<'_, T>,
) {
    let sources = input.0;
    let n = chunks.nleaves();
    let points = sources.offsets[first + n] - sources.offsets[first];
    if n < 2 || points <= LOAD_GRAIN {
        write_source_charges(chunks, first, input);
        return;
    }
    let mid = n / 2;
    let (left, right) = chunks.split_at_mut(mid);
    rayon::join(
        || split_source_charges(left, first, input),
        || split_source_charges(right, first + mid, input),
    );
}

/// The m = `charges.len()` vectors of the caller's charges packed point-major for the
/// forward move (Phase 6 T5): `packed[i m + v] = charges[v][i]`, every element written
/// once (the buffer reused, resized). On `pool` when given, by ranges of sources.
pub(crate) fn pack_charges<T: Copy + Send + Sync>(
    charges: &[&[T]],
    pool: Option<&ThreadPool>,
    packed: &mut Vec<T>,
) {
    let (m, n) = (charges.len(), charges.first().map_or(0, |c| c.len()));
    let value = |k: usize| charges[k % m][k / m];
    match pool {
        Some(pool) => pool.install(|| {
            (0..m * n)
                .into_par_iter()
                .map(value)
                .collect_into_vec(packed)
        }),
        None => {
            packed.clear();
            packed.extend((0..m * n).map(value));
        }
    }
}

/// The charges and moments of an evaluation with both (Phase 6 T8) packed point-major,
/// four values per source in the caller's order: q, d₀, d₁, d₂. On the pool if given.
pub(crate) fn pack_sources<T: Copy + Send + Sync>(
    charges: &[T],
    dipoles: &[[T; 3]],
    pool: Option<&ThreadPool>,
    packed: &mut Vec<T>,
) {
    let value = |k: usize| match k % 4 {
        0 => charges[k / 4],
        c => dipoles[k / 4][c - 1],
    };
    let n = 4 * charges.len();
    match pool {
        Some(pool) => pool.install(|| (0..n).into_par_iter().map(value).collect_into_vec(packed)),
        None => {
            packed.clear();
            packed.extend((0..n).map(value));
        }
    }
}

/// The output pass on the host (Phase 4S T9): φ and ∇φ of every owned target of `order`
/// from the leaf-ordered target output `data` (the `LeafStore` layout with the point
/// offsets `offsets`: the host's store, or since Phase 4S T11 the device's download read
/// in place), into `values` in the received order, o m values per target: for each of
/// the m vectors in turn (Phase 6 T5; m = 1 before) φ, or with `gradients` φ and ∇φ
/// (o = 4), the layout the backward move of the targets' redistribution sends (Phase 5
/// T6). Leaf j's chunk holds the m vectors' φ̂ one vector after the other, then their ĝ
/// (CONVENTIONS §3.13, "Several charge vectors"). Each value is
/// `T::from_f64(x.to_f64() / scale)`, as the pass before T9 ([`scaled_output`]) computes
/// it, so the output is that pass's bit for bit.
///
/// One loop over the targets: on `pool` when given, rayon's split of `values` into ranges
/// that each thread writes in sequence, reading the store gathered; on the calling thread
/// otherwise. Every element of `values` (o m values per target) is written once.
pub(crate) fn gather_output<T: SimdScalar + Default>(
    data: &[T],
    offsets: &[usize],
    order: &OutputOrder,
    (gradients, m): (bool, usize),
    pool: Option<&ThreadPool>,
    values: &mut [T],
) {
    let o = if gradients { 4 } else { 1 };
    debug_assert_eq!(values.len(), o * m * order.points.len());
    // Target i's leaf j, the leaf's first point and the target's place k in the leaf.
    let place = |i: usize| {
        let j = order.leaves[i] as usize;
        let first = offsets[j];
        (j, first, order.points[i] as usize - first)
    };
    let scaled = |x: T, scale: f64| T::from_f64(RealScalar::to_f64(x) / scale);
    let write = |(i, target): (usize, &mut [T])| {
        let (j, first, k) = place(i);
        let n = offsets[j + 1] - first;
        let [phi_scale, g_scale] = order.scales[j];
        // Leaf j's chunk: o m values per point from o m P_j, vector v's φ̂ at v n + k and
        // its ĝ after the m n potentials, at m n + 3 (v n + k).
        let base = o * m * first;
        for (v, target) in target.chunks_exact_mut(o).enumerate() {
            target[0] = scaled(data[base + v * n + k], phi_scale);
            if gradients {
                let g = base + m * n + 3 * (v * n + k);
                for c in 0..3 {
                    target[1 + c] = scaled(data[g + c], g_scale);
                }
            }
        }
    };
    match pool {
        Some(pool) => pool.install(|| values.par_chunks_mut(o * m).enumerate().for_each(write)),
        None => values.chunks_mut(o * m).enumerate().for_each(write),
    }
}

/// After the output pass on the device (Phase 4S T9): `device`, for each of the m vectors
/// in turn (Phase 6 T5) φ of every owned target in received order and then, with
/// gradients, ∇φ in the `[T; 3]` order of [`Output`], copied into `values` in the layout of
/// [`gather_output`] (o m values per target), for the backward move. On `pool` when given;
/// each element once.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn interleave_output<T: SimdScalar>(
    device: &[T],
    (gradients, m): (bool, usize),
    pool: Option<&ThreadPool>,
    values: &mut [T],
) {
    if !gradients && m == 1 {
        values.copy_from_slice(device);
        return;
    }
    let o = if gradients { 4 } else { 1 };
    let n = device.len() / (o * m);
    let write = |(i, target): (usize, &mut [T])| {
        for (v, target) in target.chunks_exact_mut(o).enumerate() {
            let vector = &device[v * o * n..(v + 1) * o * n];
            target[0] = vector[i];
            if gradients {
                target[1..].copy_from_slice(&vector[n + 3 * i..n + 3 * i + 3]);
            }
        }
    };
    match pool {
        Some(pool) => pool.install(|| values.par_chunks_mut(o * m).enumerate().for_each(write)),
        None => values.chunks_mut(o * m).enumerate().for_each(write),
    }
}

/// After the backward move (Phase 5 T6): `values`, o values per target in the caller's
/// order (φ, or φ and ∇φ), copied into `output`'s buffers: cleared first and their
/// allocations reused (`collect_into_vec`, `unzip_into_vecs`), every element written once,
/// with no zero fill. On `pool` when given, serially otherwise. `gradient` becomes `Some`
/// with gradients and `None` without.
pub(crate) fn unpack_output<T: SimdScalar>(
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

/// After the backward move with m vectors (Phase 6 T5): `values`, o m values per target in
/// the caller's order (for each vector in turn φ, or φ and ∇φ), copied into the output's
/// buffers `phi` and `grad` vector after vector (`phi[v N + i]`, `grad[v N + i]` for N
/// targets; [`OutputMany`](crate::fmm::OutputMany)'s layout, and since Helmholtz CH3.1b that of
/// `helmholtz::Output` with the output columns as the m vectors), the allocations reused,
/// every element written once. On `pool` when given, serially otherwise. `grad` becomes
/// `Some` with gradients and `None` without.
pub(crate) fn unpack_output_many<T: SimdScalar>(
    values: &[T],
    (gradients, m): (bool, usize),
    pool: Option<&ThreadPool>,
    (phi, grad): (&mut Vec<T>, &mut Option<Vec<[T; 3]>>),
) {
    let o = if gradients { 4 } else { 1 };
    let n = values.len() / (o * m);
    // Element k = v N + i of the output: target i's value of vector v.
    let at = |k: usize| o * (m * (k % n) + k / n);
    let potential = |k: usize| values[at(k)];
    let gradient = |k: usize| {
        let x = at(k) + 1;
        [values[x], values[x + 1], values[x + 2]]
    };
    match pool {
        Some(pool) => pool.install(|| {
            (0..m * n)
                .into_par_iter()
                .map(potential)
                .collect_into_vec(phi);
            if gradients {
                (0..m * n)
                    .into_par_iter()
                    .map(gradient)
                    .collect_into_vec(grad.get_or_insert_with(Vec::new));
            }
        }),
        None => {
            phi.clear();
            phi.extend((0..m * n).map(potential));
            if gradients {
                let g = grad.get_or_insert_with(Vec::new);
                g.clear();
                g.extend((0..m * n).map(gradient));
            }
        }
    }
    if !gradients {
        *grad = None;
    }
}
