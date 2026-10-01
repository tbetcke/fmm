//! The MPI scenarios of `nd-fmm-exec`. Keep MPI initialisation in this one test: MPI
//! cannot be initialised again after finalisation within the same process. Add
//! scenarios to `cases`, not new `#[test]`s (T9, T10 and T11 extend it).
//!
//! CI runs the scenarios on one rank. Each one is written for any rank count: every
//! rank generates the same global point set, passes its share of the keys to the
//! octree, and loads the points that fall into its own leaves, so a run by hand on
//! several ranks (under an external timeout, with the loopback flags of the root
//! `CLAUDE.md` on macOS) exercises ghosts and the global levels.
//!
//! Scenarios:
//! - **table order against the plan** (CONVENTIONS §3.12): on a uniform level-3 tree in
//!   the dyadic domain, every (level, offset) batch of the V-list view has the offset
//!   index of its direction, both as its position in `V_LIST_DIRECTIONS` and as
//!   `m2l_offset_index`; every octant batch of the M2M and L2L views has the child
//!   index of its children; and the centre difference of every V pair, from
//!   `morton::physical_box`, is 2 r_l d exactly.
//! - **batched against per-pair**: on a small adaptive tree with W and X lists, points
//!   loaded by hand in the leaf-scaled layout of §3.13, the evaluator with
//!   `LaplaceOperator` and with `PerPair<LaplaceOperator>` gives bit-identical target
//!   output. As a smoke check (not the C3.2 gate, which is T9's), its potentials and
//!   gradients match the direct sum to a loose tolerance.
//!
//! The scenarios of `Fmm` (T9) compare with `direct_sum` in f64 over all sources at
//! every target, divided by 4π (the oracle expands 1/|x − y|, `Fmm` returns
//! Σ q / (4π |x − y|)). Error measure: relative L2 error over all targets of all ranks
//! (docs/phase3/README.md, "Error measures"), charges uniform in [−1, 1).
//! - **uniform cube, every strategy**: N = 2,000, sources equal to targets, the default
//!   tree; for p ∈ {2, 4, 6, 8} and `Dense`, `Classes` and `Rotation`, the error
//!   decreases with p and is below 1e-4 at p = 8, and the three strategies agree to
//!   1e-12 (relative L2 between them).
//! - **distinct and nested point sets**: sources and targets disjoint, and the sources
//!   a subset of the targets, p = 6.
//! - **gradients**: ∇φ against the `direct_sum` gradient, p = 6; the potentials equal
//!   those of the FMM without gradients bit for bit.
//! - **uniform tree**: `max_level` 3, `max_points_per_leaf` 1, four points in every
//!   level-3 box of a supplied unit-cube domain: every leaf lies on level 3, the W and X
//!   lists are empty.
//! - **single leaf**: `max_level` 0, the root the only leaf: P2P alone, equal to the
//!   direct sum to 1e-14. One rank only (a root-only coarse tree has one block).
//! - **no targets on this rank**: rank 0 passes no targets and gets empty output.
//! - **input errors**: a supplied non-cubic domain, a point outside a supplied domain,
//!   p > 20, a non-finite point and a charge vector of the wrong length each give
//!   their `FmmError`, agreed on every rank.
//! - **f32 against f64**: p = 6; the f32 output against the f64 output, and both
//!   against the direct sum.
//! - **repeatability**: `evaluate` twice gives bit-identical output, and a second charge
//!   vector gives bit for bit the output of a fresh build with it.
//! - **ownership**: every rank passes the same complete point set. On one rank the FMM
//!   runs; on several, every rank returns `PointsNotOwned` with the same count.
//!
//! Every `Fmm` scenario but the last passes each rank its share of the points (every
//! `size`-th). On several ranks those points generally lie in leaves of other ranks;
//! the scenario then checks that every rank returns `PointsNotOwned` and stops, since
//! points are not redistributed until C5.1. If the build succeeds instead, the
//! checks run distributed.

use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{Fmm, FmmBuilder, FmmError, Output, PointSet, SettingsError};
use nd_fmm_exec::geometry::{Domain, GeometryError, leaf_coordinates, radius};
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS;
use nd_fmm_plan::lists::GroupedCsr;
use nd_fmm_plan::operator::{FmmOperator, PerPair};
use nd_fmm_plan::plan::Plan;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::geometry::m2l_offset_index;
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, morton};

type Scenario = fn(&SimpleCommunicator);

#[test]
fn distributed_scenarios() {
    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    let cases: [(&str, Scenario); 12] = [
        ("table order against the plan", table_order),
        ("batched against per-pair", batched_against_per_pair),
        ("uniform cube, every strategy", uniform_cube_strategies),
        ("distinct and nested point sets", distinct_and_nested),
        ("gradients", gradients),
        ("uniform tree", uniform_tree),
        ("single leaf", single_leaf),
        ("no targets on this rank", no_targets_on_this_rank),
        ("input errors", input_errors),
        ("f32 against f64", f32_against_f64),
        ("repeatability", repeatability),
        ("ownership", ownership),
    ];
    for (name, scenario) in cases {
        eprintln!("rank {}: {name}", comm.rank());
        scenario(&comm);
    }
}

/// The dyadic domain of Phase 2, a = (−1.25, 0.5, 2) and w = 3: every centre, centre
/// difference and half-width on levels 0–16 is exact in f64.
fn dyadic_domain() -> Domain {
    Domain::new(&PhysicalBox::new([-1.25, 0.5, 2.0, 1.75, 3.5, 5.0])).unwrap()
}

/// SplitMix64, as in the operator tests.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }
}

/// The octree of this rank's share of `points` (every `size`-th point from `rank`), with
/// the ghost-children layer that `Plan::new` requires.
fn octree<'c>(
    points: &[[f64; 3]],
    domain: &Domain,
    max_level: usize,
    capacity: usize,
    comm: &'c SimpleCommunicator,
) -> Octree<'c, SimpleCommunicator> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let fine_keys: Vec<MortonKey> = points
        .iter()
        .skip(rank)
        .step_by(size)
        .map(|&x| morton::from_physical_point(x, &domain.physical_box(), DEEPEST_LEVEL as usize))
        .collect();
    let options = OctreeOptions::new()
        .with_max_level(max_level)
        .with_max_fine_keys(capacity)
        .with_ghost_children(true);
    Octree::new(&fine_keys, options, comm)
}

/// The midpoint of `morton::physical_box(key)`, per axis.
fn midpoint(key: MortonKey, domain: &Domain) -> [f64; 3] {
    let c = morton::physical_box(key, &domain.physical_box()).coordinates();
    core::array::from_fn(|k| (c[k] + c[k + 3]) / 2.0)
}

/// Checks the octant of every pair of a parent–child view, rows and batches: `child`
/// and `parent` give the keys of an entry's child and parent.
fn check_octants<G: Copy + Into<usize>>(
    view: &GroupedCsr<G>,
    pair: impl Fn(usize, usize) -> (MortonKey, MortonKey),
) -> usize {
    for t in 0..view.nrows() {
        let (sources, groups) = view.row(t);
        for (&s, &o) in sources.iter().zip(groups) {
            let (child, parent) = pair(t, s as usize);
            assert_eq!(morton::child_index(child), o.into(), "row {t}");
            assert_eq!(morton::parent(child), Some(parent));
        }
    }
    let mut pairs = 0;
    for o in 0..view.ngroups() {
        let (targets, sources) = view.batch(o);
        for (&t, &s) in targets.iter().zip(sources) {
            let (child, parent) = pair(t as usize, s as usize);
            assert_eq!(morton::child_index(child), o, "octant batch {o}");
            assert_eq!(morton::parent(child), Some(parent));
            pairs += 1;
        }
    }
    pairs
}

/// The table order of `nd-fmm-tables` against the views of the plan (module
/// documentation). Error measure: exact equality (integers, and centre differences that
/// are exact in the dyadic domain).
fn table_order(comm: &SimpleCommunicator) {
    let domain = dyadic_domain();
    // One point at the centre of every level-3 box: a uniform level-3 tree.
    let points: Vec<[f64; 3]> = (0..512)
        .map(|j| {
            let index = [j >> 6, (j >> 3) & 7, j & 7];
            midpoint(morton::from_index_and_level(index, 3), &domain)
        })
        .collect();
    let octree = octree(&points, &domain, 3, 1, comm);
    let plan = Plan::new(&octree).expect("the plan builds");
    let index = plan.index();
    let mut counts = [0usize; 3];
    for level in 0..plan.nlevels() {
        let lists = plan.level(level);
        let r = radius(level, &domain);
        let v = lists.v();
        let check_pair = |t: usize, s: usize, d: usize| {
            let (target, source) = (index.key(level, t), index.key(level, s));
            let ((_, ti), (_, si)) = (morton::decode(target), morton::decode(source));
            let offset: [i64; 3] = core::array::from_fn(|k| ti[k] as i64 - si[k] as i64);
            assert_eq!(
                offset, V_LIST_DIRECTIONS[d],
                "level {level}, offset index {d}"
            );
            assert_eq!(m2l_offset_index(offset), Some(d));
            let (ct, cs) = (midpoint(target, &domain), midpoint(source, &domain));
            let shift: [f64; 3] = core::array::from_fn(|k| ct[k] - cs[k]);
            assert_eq!(shift, offset.map(|dk| 2.0 * r * dk as f64), "level {level}");
        };
        for t in 0..v.nrows() {
            let (sources, offsets) = v.row(t);
            for (&s, &d) in sources.iter().zip(offsets) {
                check_pair(t, s as usize, d as usize);
            }
        }
        for d in 0..v.ngroups() {
            let (targets, sources) = v.batch(d);
            for (&t, &s) in targets.iter().zip(sources) {
                check_pair(t as usize, s as usize, d);
                counts[0] += 1;
            }
        }
        for children in [lists.m2m_local(), lists.m2m_global()] {
            counts[1] += check_octants(children, |t, c| {
                (index.key(level + 1, c), index.key(level, t))
            });
        }
        counts[2] += check_octants(lists.l2l(), |t, p| {
            (index.key(level, t), index.key(level - 1, p))
        });
    }
    let mut total = [0usize; 3];
    comm.all_reduce_into(&counts[..], &mut total[..], SystemOperation::sum());
    eprintln!(
        "rank {}: table order: {} V pairs, {} M2M pairs, {} L2L pairs checked ({} in all)",
        comm.rank(),
        counts[0],
        counts[1],
        counts[2],
        total.iter().sum::<usize>()
    );
    assert!(
        total.iter().all(|&n| n > 0),
        "every view has pairs: {total:?}"
    );
}

/// The points of the adaptive scenario: a cloud over the whole domain and a dense blob
/// near one corner, so the tree is graded and has W and X lists.
fn adaptive_points(domain: &Domain) -> Vec<[f64; 3]> {
    let mut rng = SplitMix64(0x7891);
    let (a, w) = (domain.lower(), domain.side());
    let mut points: Vec<[f64; 3]> = (0..80)
        .map(|_| core::array::from_fn(|k| a[k] + w * rng.range(0.001, 0.999)))
        .collect();
    points.extend(
        (0..80).map(|_| -> [f64; 3] { core::array::from_fn(|k| a[k] + w * rng.range(0.03, 0.09)) }),
    );
    points
}

/// The points of `points` that lie in each local leaf of `index`, in point order.
fn points_by_leaf(points: &[[f64; 3]], index: &BoxIndex, domain: &Domain) -> Vec<Vec<usize>> {
    let mut leaves = vec![Vec::new(); index.leaves().nlocal()];
    for (j, &x) in points.iter().enumerate() {
        let fine = morton::from_physical_point(x, &domain.physical_box(), DEEPEST_LEVEL as usize);
        if let Some(leaf) = index.local_leaf_containing(fine) {
            leaves[leaf as usize].push(j);
        }
    }
    leaves
}

/// Runs the evaluator with `operator` on the points loaded by leaf; returns the target
/// output store, every local leaf after another.
fn evaluate<Op: FmmOperator<Value = f64>>(
    plan: &Plan,
    comm: &SimpleCommunicator,
    operator: Op,
    domain: &Domain,
    (sources, charges, source_leaves): (&[[f64; 3]], &[f64], &[Vec<usize>]),
    (targets, target_leaves): (&[[f64; 3]], &[Vec<usize>]),
) -> Vec<f64> {
    let index = plan.index();
    let source_counts: Vec<usize> = source_leaves.iter().map(Vec::len).collect();
    let target_counts: Vec<usize> = target_leaves.iter().map(Vec::len).collect();
    let mut evaluator = Evaluator::new(plan, comm, operator, &source_counts, &target_counts)
        .expect("the evaluator builds");
    for (leaf, (in_sources, in_targets)) in source_leaves.iter().zip(target_leaves).enumerate() {
        let key = index.leaf_key(leaf);
        let chunk = evaluator.sources_mut(leaf);
        let n = in_sources.len();
        for (r, &j) in in_sources.iter().enumerate() {
            chunk[3 * r..3 * r + 3]
                .copy_from_slice(&leaf_coordinates::<f64>(sources[j], key, domain));
            chunk[3 * n + r] = charges[j];
        }
        let chunk = evaluator.target_input_mut(leaf);
        for (r, &j) in in_targets.iter().enumerate() {
            chunk[3 * r..3 * r + 3]
                .copy_from_slice(&leaf_coordinates::<f64>(targets[j], key, domain));
        }
    }
    evaluator.evaluate();
    evaluator.target_output_store().as_slice().to_vec()
}

/// The batched operator against the per-pair adapter, and a smoke check against the
/// direct sum (module documentation).
fn batched_against_per_pair(comm: &SimpleCommunicator) {
    let domain = dyadic_domain();
    let sources = adaptive_points(&domain);
    let mut rng = SplitMix64(0x7892);
    let charges: Vec<f64> = sources.iter().map(|_| rng.range(-1.0, 1.0)).collect();
    // The targets: the sources (coincident pairs excluded) and a few other points.
    let mut targets = sources.clone();
    let (a, w) = (domain.lower(), domain.side());
    targets.extend(
        (0..20)
            .map(|_| -> [f64; 3] { core::array::from_fn(|k| a[k] + w * rng.range(0.001, 0.999)) }),
    );

    let octree = octree(&sources, &domain, 6, 4, comm);
    let plan = Plan::new(&octree).expect("the plan builds");
    let index = plan.index();
    let source_leaves = points_by_leaf(&sources, index, &domain);
    let target_leaves = points_by_leaf(&targets, index, &domain);

    // The tree must exercise every list.
    let local = [
        (0..plan.nlevels())
            .map(|l| plan.level(l).w().len())
            .sum::<usize>(),
        (0..plan.nlevels())
            .map(|l| plan.level(l).x().len())
            .sum::<usize>(),
        (0..plan.nlevels())
            .map(|l| plan.level(l).v().len())
            .sum::<usize>(),
    ];
    let mut lists = [0usize; 3];
    comm.all_reduce_into(&local[..], &mut lists[..], SystemOperation::sum());
    assert!(lists.iter().all(|&n| n > 0), "W, X and V pairs: {lists:?}");

    // The scratch must hold the largest source leaf of any rank.
    let local_max = source_leaves.iter().map(Vec::len).max().unwrap_or(0);
    let mut max_leaf_points = 0usize;
    comm.all_reduce_into(&local_max, &mut max_leaf_points, SystemOperation::max());

    let p = 6;
    let op = LaplaceOperator::new(Tables::build(p, M2lStrategy::Auto), true, max_leaf_points);
    let input = (&sources[..], &charges[..], &source_leaves[..]);
    let batched = evaluate(
        &plan,
        comm,
        op.clone(),
        &domain,
        input,
        (&targets, &target_leaves),
    );
    let per_pair = evaluate(
        &plan,
        comm,
        PerPair(op),
        &domain,
        input,
        (&targets, &target_leaves),
    );
    assert_eq!(batched.len(), per_pair.len());
    let differing = batched
        .iter()
        .zip(&per_pair)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "batched and per-pair output differ in {differing} values"
    );

    // Smoke check against the direct sum over all sources: φ = φ̂ / r_t, ∇φ = ĝ / r_t²
    // (1/|x − y| units, no 4π), relative L2 over the local targets.
    let mut offset = 0;
    let mut sums = [0.0f64; 4];
    for (leaf, in_targets) in target_leaves.iter().enumerate() {
        let r = radius(index.leaves().level(leaf), &domain);
        let n = in_targets.len();
        let (potential, gradient) = batched[offset..offset + 4 * n].split_at(n);
        offset += 4 * n;
        let x: Vec<[f64; 3]> = in_targets.iter().map(|&j| targets[j]).collect();
        let mut phi = vec![0.0; n];
        let mut grad = vec![[0.0; 3]; n];
        direct_sum(&sources, &charges, &x, &mut phi, Some(&mut grad));
        for j in 0..n {
            sums[0] += (potential[j] / r - phi[j]).powi(2);
            sums[1] += phi[j].powi(2);
            for k in 0..3 {
                sums[2] += (gradient[3 * j + k] / (r * r) - grad[j][k]).powi(2);
                sums[3] += grad[j][k].powi(2);
            }
        }
    }
    assert_eq!(offset, batched.len());
    let mut total = [0.0f64; 4];
    comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
    let (potential_error, gradient_error) =
        ((total[0] / total[1]).sqrt(), (total[2] / total[3]).sqrt());
    eprintln!(
        "rank {}: batched = per-pair bit for bit ({} values); W, X, V pairs {lists:?}; \
         p = {p}: relative L2 error vs direct sum {potential_error:.3e} (potential), \
         {gradient_error:.3e} (gradient)",
        comm.rank(),
        batched.len()
    );
    assert!(potential_error < 1e-3, "potential: {potential_error:e}");
    assert!(gradient_error < 1e-2, "gradient: {gradient_error:e}");
}

/// `n` points uniform in the unit cube [0, 1)³.
fn unit_cube_points(rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(0.0, 1.0)))
        .collect()
}

/// `n` charges uniform in [−1, 1).
fn random_charges(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
}

/// This rank's share of `items`: every `size`-th item from `rank`.
fn share<V: Copy>(items: &[V], comm: &SimpleCommunicator) -> Vec<V> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    items.iter().skip(rank).step_by(size).copied().collect()
}

/// The FMM, or `None` on several ranks if the points are not owned (module
/// documentation). Panics on any other error.
fn built<'o, T: nd_fmm_tables::cache::Stored + Equivalence + Default>(
    result: Result<Fmm<'o, T>, FmmError>,
    comm: &SimpleCommunicator,
) -> Option<Fmm<'o, T>> {
    match result {
        Ok(fmm) => Some(fmm),
        Err(FmmError::PointsNotOwned { count }) if comm.size() > 1 => {
            eprintln!(
                "rank {}: {count} points not owned; not redistributed until C5.1, scenario \
                 stops",
                comm.rank()
            );
            None
        }
        Err(error) => panic!("rank {}: the FMM does not build: {error}", comm.rank()),
    }
}

/// The exact potential (and gradient) of `charges` at `sources` at every target:
/// `direct_sum` divided by 4π.
fn exact(sources: &[[f64; 3]], charges: &[f64], targets: &[[f64; 3]]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut phi = vec![0.0; targets.len()];
    let mut grad = vec![[0.0; 3]; targets.len()];
    direct_sum(sources, charges, targets, &mut phi, Some(&mut grad));
    let scale = 4.0 * std::f64::consts::PI;
    (
        phi.iter().map(|v| v / scale).collect(),
        grad.iter().map(|g| g.map(|gk| gk / scale)).collect(),
    )
}

/// The relative L2 error ‖a − e‖₂ / ‖e‖₂ over the values of all ranks.
fn relative_l2<T: RealScalar>(approx: &[T], exact: &[f64], comm: &SimpleCommunicator) -> f64 {
    assert_eq!(approx.len(), exact.len());
    let mut sums = [0.0f64; 2];
    for (&a, &e) in approx.iter().zip(exact) {
        sums[0] += (RealScalar::to_f64(a) - e).powi(2);
        sums[1] += e * e;
    }
    let mut total = [0.0f64; 2];
    comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
    (total[0] / total[1]).sqrt()
}

/// The relative L2 error of 3-vectors over all ranks, with the Euclidean norm per
/// vector.
fn relative_l2_vectors<T: RealScalar>(
    approx: &[[T; 3]],
    exact: &[[f64; 3]],
    comm: &SimpleCommunicator,
) -> f64 {
    let approx: Vec<T> = approx.iter().flatten().copied().collect();
    let exact: Vec<f64> = exact.iter().flatten().copied().collect();
    relative_l2(&approx, &exact, comm)
}

/// The values as f64 bit patterns (the conversion to f64 is exact).
fn bits<T: RealScalar>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect()
}

/// The output's values as bit patterns: potentials, then gradients.
fn output_bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut values = bits(&output.potential);
    if let Some(gradient) = &output.gradient {
        values.extend(bits(gradient.as_flattened()));
    }
    values
}

/// The sum of `local` over all ranks.
fn global_sum(local: usize, comm: &SimpleCommunicator) -> usize {
    let mut total = 0usize;
    comm.all_reduce_into(&local, &mut total, SystemOperation::sum());
    total
}

/// Uniform cube, sources equal to targets, p ∈ {2, 4, 6, 8}, every strategy (module
/// documentation).
fn uniform_cube_strategies(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7893);
    let points = unit_cube_points(&mut rng, 2000);
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let (phi, _) = exact(&points, &charges, &sources);
    let strategies = [
        M2lStrategy::Dense,
        M2lStrategy::Classes,
        M2lStrategy::Rotation,
    ];
    let ps = [2, 4, 6, 8];
    let mut errors = [[0.0f64; 4]; 3];
    for (k, &p) in ps.iter().enumerate() {
        let mut outputs: Vec<Vec<f64>> = Vec::new();
        for (s, &strategy) in strategies.iter().enumerate() {
            let builder = FmmBuilder::<f64>::new(p).strategy(strategy);
            let Some(mut fmm) = built(builder.build(&sources, &sources, comm), comm) else {
                return;
            };
            assert_eq!(fmm.strategy(), strategy);
            let output = fmm.evaluate(&local_charges).expect("the FMM evaluates");
            errors[s][k] = relative_l2(&output.potential, &phi, comm);
            outputs.push(output.potential);
        }
        for s in 1..strategies.len() {
            let difference = relative_l2(&outputs[s], &outputs[0], comm);
            assert!(
                difference < 1e-12,
                "p = {p}: {:?} differs from Dense by {difference:e} (relative L2)",
                strategies[s]
            );
        }
    }
    for (s, strategy) in strategies.iter().enumerate() {
        eprintln!(
            "rank {}: uniform cube, N = {}, {strategy:?}: relative L2 error of φ at p = \
             {ps:?}: {}",
            comm.rank(),
            points.len(),
            errors[s].map(|e| format!("{e:.3e}")).join(", ")
        );
        assert!(
            errors[s].windows(2).all(|e| e[1] < e[0]),
            "{strategy:?}: the error does not decrease with p: {:?}",
            errors[s]
        );
        assert!(
            errors[s][3] < 1e-4,
            "{strategy:?}: p = 8: {:e}",
            errors[s][3]
        );
    }
}

/// Sources and targets disjoint, and sources a subset of the targets (module
/// documentation). Tolerance 1e-3, ten times the p = 6 error of the strategy scenario:
/// a smoke check of the permutations, which a wrong order fails by O(1).
fn distinct_and_nested(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7894);
    let sources = unit_cube_points(&mut rng, 1500);
    let charges = random_charges(&mut rng, sources.len());
    let others = unit_cube_points(&mut rng, 1000);
    // The sources in the middle of the targets, so their positions shift.
    let nested: Vec<[f64; 3]> = others[..500]
        .iter()
        .chain(&sources)
        .chain(&others[500..])
        .copied()
        .collect();
    let (local_sources, local_charges) = (share(&sources, comm), share(&charges, comm));
    for (name, targets) in [("disjoint", &others), ("nested", &nested)] {
        let local_targets = share(targets, comm);
        let builder = FmmBuilder::<f64>::new(6);
        let Some(mut fmm) = built(builder.build(&local_sources, &local_targets, comm), comm) else {
            return;
        };
        assert_eq!(fmm.strategy(), M2lStrategy::Dense, "Auto at p = 6");
        assert_eq!(
            (fmm.nsources(), fmm.ntargets()),
            (local_sources.len(), local_targets.len())
        );
        let output = fmm.evaluate(&local_charges).expect("the FMM evaluates");
        assert_eq!(output.potential.len(), local_targets.len());
        assert!(output.gradient.is_none());
        let (phi, _) = exact(&sources, &charges, &local_targets);
        let error = relative_l2(&output.potential, &phi, comm);
        eprintln!(
            "rank {}: {name}: {} sources, {} targets, p = 6: relative L2 error of φ {error:.3e}",
            comm.rank(),
            sources.len(),
            targets.len()
        );
        assert!(error < 1e-3, "{name}: {error:e}");
    }
}

/// Gradients against the `direct_sum` gradient (module documentation). Tolerance 1e-2,
/// the smoke tolerance of the batched-against-per-pair scenario.
fn gradients(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7895);
    let points = unit_cube_points(&mut rng, 2000);
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(6).gradients(true);
    let Some(mut fmm) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    assert!(fmm.gradients());
    let output = fmm.evaluate(&local_charges).expect("the FMM evaluates");
    let gradient = output.gradient.as_ref().expect("gradients requested");
    assert_eq!(gradient.len(), sources.len());
    let (phi, grad) = exact(&points, &charges, &sources);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(gradient, &grad, comm);
    eprintln!(
        "rank {}: gradients, N = {}, p = 6: relative L2 error {potential_error:.3e} (φ), \
         {gradient_error:.3e} (∇φ)",
        comm.rank(),
        points.len()
    );
    assert!(potential_error < 1e-3, "φ: {potential_error:e}");
    assert!(gradient_error < 1e-2, "∇φ: {gradient_error:e}");

    // The potentials do not depend on whether gradients are computed.
    let builder = FmmBuilder::<f64>::new(6);
    let Some(mut without) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    let potential = without.evaluate(&local_charges).unwrap().potential;
    assert_eq!(bits(&potential), bits(&output.potential));
}

/// A uniform level-3 tree: no W or X list (module documentation).
fn uniform_tree(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7896);
    // Four points in every level-3 box of the unit cube, away from its faces.
    let points: Vec<[f64; 3]> = (0..512)
        .flat_map(|b: usize| {
            let index = [b >> 6, (b >> 3) & 7, b & 7];
            (0..4)
                .map(|_| core::array::from_fn(|k| (index[k] as f64 + rng.range(0.05, 0.95)) / 8.0))
                .collect::<Vec<[f64; 3]>>()
        })
        .collect();
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(4)
        .max_level(3)
        .max_points_per_leaf(1)
        .domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
    let Some(mut fmm) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    assert_eq!(fmm.nlevels(), 4);
    let leaves = fmm.plan().index().leaves();
    assert!(
        (0..fmm.nleaves()).all(|j| leaves.level(j) == 3),
        "every leaf lies on level 3"
    );
    assert_eq!(global_sum(fmm.nleaves(), comm), 512);
    let sizes = fmm.list_sizes();
    let (w, x) = (global_sum(sizes.w, comm), global_sum(sizes.x, comm));
    assert_eq!((w, x), (0, 0), "W and X lists of a uniform tree");
    assert!(global_sum(sizes.v, comm) > 0 && global_sum(sizes.u, comm) > 0);
    let output = fmm.evaluate(&local_charges).expect("the FMM evaluates");
    let (phi, _) = exact(&points, &charges, &sources);
    let error = relative_l2(&output.potential, &phi, comm);
    eprintln!(
        "rank {}: uniform level-3 tree, N = {}, p = 4: {} V and {} U pairs; relative L2 \
         error of φ {error:.3e}",
        comm.rank(),
        points.len(),
        global_sum(sizes.v, comm),
        global_sum(sizes.u, comm)
    );
    assert!(error < 1e-2, "{error:e}");
}

/// The root as the only leaf: P2P alone (module documentation). Error measure: relative
/// L2 against the direct sum, 1e-14.
fn single_leaf(comm: &SimpleCommunicator) {
    if comm.size() > 1 {
        eprintln!("rank {}: single leaf: one rank only", comm.rank());
        return;
    }
    let mut rng = SplitMix64(0x7897);
    let points = unit_cube_points(&mut rng, 200);
    let charges = random_charges(&mut rng, points.len());
    let builder = FmmBuilder::<f64>::new(3).max_level(0).gradients(true);
    let mut fmm = builder
        .build(&points, &points, comm)
        .expect("the FMM builds");
    assert_eq!((fmm.nlevels(), fmm.nleaves()), (1, 1));
    let output = fmm.evaluate(&charges).expect("the FMM evaluates");
    let (phi, grad) = exact(&points, &charges, &points);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(output.gradient.as_ref().unwrap(), &grad, comm);
    eprintln!(
        "rank {}: single leaf, N = {}: relative L2 error {potential_error:.3e} (φ), \
         {gradient_error:.3e} (∇φ)",
        comm.rank(),
        points.len()
    );
    assert!(potential_error < 1e-14, "φ: {potential_error:e}");
    assert!(gradient_error < 1e-14, "∇φ: {gradient_error:e}");
}

/// Rank 0 has no targets (module documentation).
fn no_targets_on_this_rank(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7898);
    let points = unit_cube_points(&mut rng, 500);
    let charges = random_charges(&mut rng, points.len());
    let targets = unit_cube_points(&mut rng, 300);
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let local_targets = if comm.rank() == 0 {
        Vec::new()
    } else {
        share(&targets, comm)
    };
    let builder = FmmBuilder::<f64>::new(4).gradients(true);
    let Some(mut fmm) = built(builder.build(&sources, &local_targets, comm), comm) else {
        return;
    };
    let output = fmm.evaluate(&local_charges).expect("the FMM evaluates");
    assert_eq!(output.potential.len(), local_targets.len());
    assert_eq!(output.gradient.as_ref().unwrap().len(), local_targets.len());
    if comm.rank() == 0 {
        assert_eq!(fmm.ntargets(), 0);
        assert!(fmm.target_counts().iter().all(|&n| n == 0));
    }
    eprintln!(
        "rank {}: no targets on rank 0: {} targets here",
        comm.rank(),
        output.potential.len()
    );
}

/// Every input error gives its `FmmError`, agreed on every rank (module documentation).
fn input_errors(comm: &SimpleCommunicator) {
    let (rank, last) = (comm.rank(), comm.size() - 1);
    let mut rng = SplitMix64(0x7899);
    let points = share(&unit_cube_points(&mut rng, 400), comm);
    let unit = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    // A supplied domain that is not a cube: the same on every rank.
    let builder =
        FmmBuilder::<f64>::new(4).domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 2.0]));
    let error = builder.build(&points, &points, comm).err();
    assert!(
        matches!(
            error,
            Some(FmmError::InvalidDomain(GeometryError::NotCubic { .. }))
        ),
        "{error:?}"
    );

    // A source of rank 0 outside the supplied domain.
    let mut outside = points.clone();
    if rank == 0 {
        outside[3] = [1.5, 0.5, 0.5];
    }
    let builder = FmmBuilder::<f64>::new(4).domain(unit);
    let error = builder.build(&outside, &points, comm).err();
    match (rank, &error) {
        (
            0,
            Some(FmmError::PointOutsideDomain {
                set, index, point, ..
            }),
        ) => {
            assert_eq!(
                (*set, *index, *point),
                (PointSet::Sources, 3, [1.5, 0.5, 0.5])
            );
        }
        (r, Some(FmmError::OtherRank)) if r != 0 => {}
        _ => panic!("rank {rank}: point outside the domain: {error:?}"),
    }

    // A target of the last rank that is not finite.
    let mut non_finite = points.clone();
    if rank == last {
        non_finite[0] = [0.5, f64::NAN, 0.5];
    }
    let error = FmmBuilder::<f64>::new(4)
        .build(&points, &non_finite, comm)
        .err();
    match (rank, &error) {
        (r, Some(FmmError::NonFinitePoint { set, index, .. })) if r == last => {
            assert_eq!((*set, *index), (PointSet::Targets, 0));
        }
        (r, Some(FmmError::OtherRank)) if r != last => {}
        _ => panic!("rank {rank}: non-finite point: {error:?}"),
    }

    // Settings out of range.
    for (builder, expected) in [
        (
            FmmBuilder::<f64>::new(21),
            SettingsError::DegreeTooLarge { p: 21 },
        ),
        (
            FmmBuilder::<f64>::new(4).max_level(17),
            SettingsError::MaxLevelTooDeep { max_level: 17 },
        ),
        (
            FmmBuilder::<f64>::new(4).max_points_per_leaf(0),
            SettingsError::ZeroPointsPerLeaf,
        ),
    ] {
        let error = builder.build(&points, &points, comm).err();
        assert_eq!(error, Some(FmmError::InvalidSettings(expected)));
    }

    // No points on any rank, and points that span no volume.
    let error = FmmBuilder::<f64>::new(4).build(&[], &[], comm).err();
    assert_eq!(error, Some(FmmError::NoPoints));
    let error = FmmBuilder::<f64>::new(4)
        .build(&[[0.5; 3]], &[[0.5; 3]], comm)
        .err();
    assert_eq!(error, Some(FmmError::DegenerateExtent { extent: 0.0 }));

    // A charge vector of the wrong length on rank 0.
    let Some(mut fmm) = built(
        FmmBuilder::<f64>::new(2).build(&points, &points, comm),
        comm,
    ) else {
        return;
    };
    let charges = vec![1.0; points.len() - usize::from(rank == 0)];
    let error = fmm.evaluate(&charges).err();
    match (rank, &error) {
        (0, Some(FmmError::ChargesLength { expected, actual })) => {
            assert_eq!((*expected, *actual), (points.len(), points.len() - 1));
        }
        (r, Some(FmmError::OtherRank)) if r != 0 => {}
        _ => panic!("rank {rank}: wrong charge length: {error:?}"),
    }
    eprintln!("rank {rank}: input errors as expected");
}

/// f32 at p = 6 against f64 (module documentation). The f64 run and the oracle use the
/// f32 charges, so only the arithmetic differs. Tolerances: the f32 output within 1e-5
/// of the f64 output (relative L2, about 100 ε₃₂), and its error against the direct sum
/// within 1e-5 of the f64 error.
fn f32_against_f64(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789a);
    let points = unit_cube_points(&mut rng, 2000);
    let charges32: Vec<f32> = random_charges(&mut rng, points.len())
        .iter()
        .map(|&q| q as f32)
        .collect();
    let charges: Vec<f64> = charges32.iter().map(|&q| f64::from(q)).collect();
    let sources = share(&points, comm);
    let (local32, local64) = (share(&charges32, comm), share(&charges, comm));
    let Some(mut fmm32) = built(
        FmmBuilder::<f32>::new(6)
            .gradients(true)
            .build(&sources, &sources, comm),
        comm,
    ) else {
        return;
    };
    let Some(mut fmm64) = built(
        FmmBuilder::<f64>::new(6)
            .gradients(true)
            .build(&sources, &sources, comm),
        comm,
    ) else {
        return;
    };
    let output32 = fmm32.evaluate(&local32).expect("the f32 FMM evaluates");
    let output64 = fmm64.evaluate(&local64).expect("the f64 FMM evaluates");
    let (phi, _) = exact(&points, &charges, &sources);
    let difference = relative_l2(&output32.potential, &output64.potential, comm);
    let gradient_difference = relative_l2_vectors(
        output32.gradient.as_ref().unwrap(),
        output64.gradient.as_ref().unwrap(),
        comm,
    );
    let (error32, error64) = (
        relative_l2(&output32.potential, &phi, comm),
        relative_l2(&output64.potential, &phi, comm),
    );
    eprintln!(
        "rank {}: f32 against f64, N = {}, p = 6: relative L2 difference {difference:.3e} (φ), \
         {gradient_difference:.3e} (∇φ); error of φ {error32:.3e} (f32), {error64:.3e} (f64)",
        comm.rank(),
        points.len()
    );
    assert!(difference < 1e-5, "φ: {difference:e}");
    assert!(gradient_difference < 1e-5, "∇φ: {gradient_difference:e}");
    assert!(
        (error32 - error64).abs() < 1e-5,
        "{error32:e} against {error64:e}"
    );
}

/// Two evaluations agree bit for bit, and a second charge vector gives the output of a
/// fresh build (module documentation).
fn repeatability(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789b);
    let points = unit_cube_points(&mut rng, 1000);
    let first = random_charges(&mut rng, points.len());
    let second = random_charges(&mut rng, points.len());
    let sources = share(&points, comm);
    let (first, second) = (share(&first, comm), share(&second, comm));
    let builder = FmmBuilder::<f64>::new(4).gradients(true);
    let Some(mut fmm) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    let a = fmm.evaluate(&first).unwrap();
    let b = fmm.evaluate(&first).unwrap();
    assert_eq!(output_bits(&a), output_bits(&b), "two evaluations differ");
    let again = fmm.evaluate(&second).unwrap();
    let Some(mut fresh) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    let reference = fresh.evaluate(&second).unwrap();
    assert_eq!(
        output_bits(&again),
        output_bits(&reference),
        "a second charge vector differs from a fresh build"
    );
    assert_ne!(output_bits(&a), output_bits(&again));
    eprintln!(
        "rank {}: repeatability: {} values bit for bit",
        comm.rank(),
        output_bits(&a).len()
    );
}

/// Every rank passes the complete point set: the FMM runs on one rank, and on several
/// every rank returns `PointsNotOwned` with the same count (module documentation).
fn ownership(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789c);
    let points = unit_cube_points(&mut rng, 600);
    let charges = random_charges(&mut rng, points.len());
    let result = FmmBuilder::<f64>::new(4).build(&points, &points, comm);
    if comm.size() == 1 {
        let mut fmm = result.expect("on one rank every point is owned");
        let output = fmm.evaluate(&charges).expect("the FMM evaluates");
        let (phi, _) = exact(&points, &charges, &points);
        let error = relative_l2(&output.potential, &phi, comm);
        eprintln!("rank 0: ownership: one rank, relative L2 error of φ {error:.3e} (p = 4)");
        assert!(error < 1e-2, "{error:e}");
        return;
    }
    let count = match result {
        Err(FmmError::PointsNotOwned { count }) => count,
        Err(error) => panic!("rank {}: {error}", comm.rank()),
        Ok(_) => panic!(
            "rank {}: every rank holds every point, yet the build succeeds",
            comm.rank()
        ),
    };
    let (mut lowest, mut highest) = (0u64, 0u64);
    comm.all_reduce_into(&count, &mut lowest, SystemOperation::min());
    comm.all_reduce_into(&count, &mut highest, SystemOperation::max());
    assert_eq!(lowest, highest, "the ranks disagree on the count");
    assert!(count > 0);
    eprintln!(
        "rank {}: ownership: PointsNotOwned with {count} points, on every rank",
        comm.rank()
    );
}
