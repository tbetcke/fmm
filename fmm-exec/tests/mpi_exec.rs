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

use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::geometry::{Domain, leaf_coordinates, radius};
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables};
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
    let cases: [(&str, Scenario); 2] = [
        ("table order against the plan", table_order),
        ("batched against per-pair", batched_against_per_pair),
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
