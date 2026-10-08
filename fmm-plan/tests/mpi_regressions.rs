//! Public-API regression tests. Keep MPI initialization in one test: MPI cannot
//! be initialized again after finalization within the same process.
use std::{borrow::Borrow, collections::HashMap};

use mpi::{collective::SystemOperation, traits::*};
use nd_fmm_plan::{
    evaluator::Evaluator,
    exchange::{CoarseExchange, MultipoleExchange, SourceExchange},
    interaction_manager::V_LIST_DIRECTIONS,
    lists::{GroupedCsr, offset_index},
    operator::{
        FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator, PerPair,
        UpwardPass,
    },
    plan::Plan,
    store::{LeafSliceMut, LeafStore, LevelBuffers},
};
use nd_octree::{
    Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, morton, morton::is_ancestor,
    octree::KeyType, points_to_morton,
};
use rlst::{distributed_tools::array_tools::gather_to_all, rlst_dynamic_array};

fn keys(points: &[[f64; 3]]) -> Vec<u64> {
    let mut array = rlst_dynamic_array!(f64, [3, points.len()]);
    for (j, point) in points.iter().enumerate() {
        for i in 0..3 {
            array[[i, j]] = point[i];
        }
    }
    points_to_morton(
        &array,
        DEEPEST_LEVEL as usize,
        &PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
}

/// Inclusive index bounds of a box, expressed in cells of the deepest level.
fn deepest_bounds(key: u64) -> [[u64; 2]; 3] {
    let (level, index) = morton::decode(key);
    let size = 1u64 << (DEEPEST_LEVEL as usize - level);
    [0usize, 1, 2].map(|dim| {
        let low = index[dim] as u64 * size;
        [low, low + size - 1]
    })
}

/// Two boxes touch when their closed cubes intersect and neither contains the
/// other. Written independently of the implementation under test.
fn touching(a: u64, b: u64) -> bool {
    if is_ancestor(a, b) || is_ancestor(b, a) {
        return false;
    }
    let a_bounds = deepest_bounds(a);
    let b_bounds = deepest_bounds(b);
    (0..3).all(|dim| {
        a_bounds[dim][0] <= b_bounds[dim][1] + 1 && b_bounds[dim][0] <= a_bounds[dim][1] + 1
    })
}

/// Brute-force U-, V-, W- and X-lists of `key` from the global key set, using
/// only the geometric definitions. `keys` and `leaves` must be sorted.
fn oracle_lists(keys: &[u64], leaves: &[u64], key: u64) -> [Vec<u64>; 4] {
    let key_is_leaf = leaves.binary_search(&key).is_ok();
    let key_level = morton::level(key);
    let key_parent = morton::parent(key);
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut w = Vec::new();
    let mut x = Vec::new();
    for &other in keys {
        let other_is_leaf = leaves.binary_search(&other).is_ok();
        let other_level = morton::level(other);
        if key_is_leaf && other_is_leaf && other != key && touching(other, key) {
            u.push(other);
        }
        if let Some(parent) = key_parent {
            if other_level == key_level
                && let Some(other_parent) = morton::parent(other)
                && other_parent != parent
                && touching(other_parent, parent)
                && !touching(other, key)
            {
                v.push(other);
            }
            if other_is_leaf
                && other_level < key_level
                && touching(parent, other)
                && !touching(key, other)
            {
                x.push(other);
            }
        }
        if key_is_leaf
            && other_level > key_level
            && let Some(other_parent) = morton::parent(other)
            && touching(other_parent, key)
            && !touching(other, key)
        {
            w.push(other);
        }
    }
    [u, v, w, x]
}

/// The brute-force U-, V-, W- and X-lists of every non-ghost key of `all_keys`, by key,
/// over the gathered global tree. The reference for the plan, the multipole ghosts and
/// the pairs that an evaluation issues.
type OracleLists = HashMap<u64, [Vec<u64>; 4]>;

/// Compute the [`OracleLists`] from the local key map and the sorted global leaves.
fn oracle_of(all_keys: &HashMap<u64, KeyType>, global_leaves: &[u64]) -> OracleLists {
    let mut keys: Vec<u64> = morton::get_interior_keys(global_leaves)
        .into_iter()
        .collect();
    keys.extend_from_slice(global_leaves);
    keys.sort_unstable();
    keys.dedup();
    all_keys
        .iter()
        .filter(|(_, kind)| !kind.is_ghost())
        .map(|(&key, _)| (key, oracle_lists(&keys, global_leaves, key)))
        .collect()
}

/// The rows of a grouped view as sorted (target, source, group) triples, checking that
/// every row is strictly ascending in its group.
fn grouped_rows<G: Copy + Into<usize>>(view: &GroupedCsr<G>) -> Vec<(u32, u32, usize)> {
    let mut triples = Vec::new();
    for t in 0..view.nrows() {
        let (sources, groups) = view.row(t);
        assert!(
            groups
                .windows(2)
                .all(|pair| pair[0].into() < pair[1].into())
        );
        triples.extend(
            sources
                .iter()
                .zip(groups)
                .map(|(&s, &g)| (t as u32, s, g.into())),
        );
    }
    triples.sort_unstable();
    triples
}

/// The batches of a grouped view as sorted (target, source, group) triples, checking that
/// every batch lists its targets strictly ascending, so each at most once.
fn grouped_batches<G: Copy + Into<usize>>(view: &GroupedCsr<G>) -> Vec<(u32, u32, usize)> {
    let mut triples = Vec::new();
    for g in 0..view.ngroups() {
        let (targets, sources) = view.batch(g);
        assert!(targets.windows(2).all(|pair| pair[0] < pair[1]));
        triples.extend(targets.iter().zip(sources).map(|(&t, &s)| (t, s, g)));
    }
    triples.sort_unstable();
    triples
}

/// Compare the index-form lists of every local non-ghost box against the brute-force
/// oracle, through the key and index maps, and check the invariants of the views.
fn check_plan(
    name: &str,
    all_keys: &HashMap<u64, KeyType>,
    oracle: &OracleLists,
    plan: &Plan,
    coarse_blocks: &[u64],
) {
    let index = plan.index();
    let leaves = index.leaves();
    let mut own_blocks = coarse_blocks.to_vec();
    own_blocks.sort_unstable();
    assert_eq!(plan.coarse_blocks(), own_blocks, "{name}: coarse blocks");

    // The numbering: every held key, in Morton order per level.
    let mut held = 0;
    for level in 0..index.nlevels() {
        let level_keys = index.keys(level);
        assert!(level_keys.windows(2).all(|pair| pair[0] < pair[1]));
        for (i, &key) in level_keys.iter().enumerate() {
            assert_eq!(morton::level(key), level);
            assert_eq!(
                index.kind(level, i),
                all_keys[&key],
                "{name}: kind of {key}"
            );
            assert_eq!(index.find(key), Some((level, i as u32)));
        }
        held += level_keys.len();
    }
    assert_eq!(held, all_keys.len(), "{name}: held keys");

    // The leaves: local ones by (level, key), then the ghosts named by U or X, by key.
    let mut local: Vec<(usize, u64)> = all_keys
        .iter()
        .filter(|&(_, &kind)| kind == KeyType::LocalLeaf)
        .map(|(&key, _)| (morton::level(key), key))
        .collect();
    local.sort_unstable();
    let numbered: Vec<(usize, u64)> = (0..leaves.nlocal())
        .map(|j| (leaves.level(j), leaves.key(j)))
        .collect();
    assert_eq!(numbered, local, "{name}: local leaves");
    let ghost_leaves: Vec<u64> = leaves.ghosts().map(|j| leaves.key(j)).collect();
    assert!(ghost_leaves.windows(2).all(|pair| pair[0] < pair[1]));
    let mut named = Vec::new();

    let leaf_key = |j: &u32| leaves.key(*j as usize);
    for level in 0..index.nlevels() {
        let lists = plan.level(level);
        let local = leaves.local(level);
        let key_on = |level: usize| move |i: &u32| index.key(level, *i as usize);

        for (i, &key) in index.keys(level).iter().enumerate() {
            let kind = index.kind(level, i);
            if kind.is_ghost() {
                assert!(lists.v().row(i).0.is_empty() && lists.x().row(i).is_empty());
                assert!(lists.l2l().row(i).0.is_empty() && lists.p2m().row(i).is_empty());
                assert!(lists.m2m_local().row(i).0.is_empty());
                assert!(lists.m2m_global().row(i).0.is_empty());
                continue;
            }
            let expected = &oracle[&key];
            let mut v: Vec<u64> = lists.v().row(i).0.iter().map(key_on(level)).collect();
            let mut x: Vec<u64> = lists.x().row(i).iter().map(leaf_key).collect();
            v.sort_unstable();
            x.sort_unstable();
            let (u, w) = match index.box_leaf(level, i) {
                Some(leaf) => {
                    let r = leaf as usize - local.start;
                    let near: Vec<u64> = lists.near().row(r).iter().map(leaf_key).collect();
                    assert!(
                        near.contains(&key),
                        "{name}: near list of {key} lacks itself"
                    );
                    let mut u: Vec<u64> = near.into_iter().filter(|&e| e != key).collect();
                    u.sort_unstable();
                    let w: Vec<u64> = lists.w().row(r).iter().map(key_on(level + 1)).collect();
                    (u, w)
                }
                None => (Vec::new(), Vec::new()),
            };
            for (list_name, actual, expected) in [
                ("U", &u, &expected[0]),
                ("V", &v, &expected[1]),
                ("W", &w, &expected[2]),
                ("X", &x, &expected[3]),
            ] {
                assert_eq!(
                    actual, expected,
                    "{name}: index-form {list_name}-list of {key}"
                );
            }
            named.extend(
                u.iter()
                    .chain(&x)
                    .copied()
                    .filter(|entry| all_keys[entry].is_ghost()),
            );

            // Octant views: the children of non-ghost interiors, the parent of every box.
            let (children, octants) = match kind {
                KeyType::LocalInterior => lists.m2m_local().row(i),
                KeyType::Global => lists.m2m_global().row(i),
                _ => (&[][..], &[][..]),
            };
            let children: Vec<u64> = children.iter().map(key_on(level + 1)).collect();
            if kind == KeyType::LocalInterior || kind == KeyType::Global {
                assert_eq!(children, morton::children(key).unwrap(), "{name}: children");
                assert_eq!(octants, [0, 1, 2, 3, 4, 5, 6, 7]);
            } else {
                assert!(children.is_empty());
            }
            let (parents, octants) = lists.l2l().row(i);
            let parents: Vec<u64> = parents.iter().map(key_on(level.wrapping_sub(1))).collect();
            assert_eq!(parents, morton::parent(key).into_iter().collect::<Vec<_>>());
            if level > 0 {
                assert_eq!(octants, [morton::child_index(key) as u8]);
            }
        }

        // The same pairs in both views, with offset indices that match the keys.
        for view in [lists.m2m_local(), lists.m2m_global(), lists.l2l()] {
            assert_eq!(grouped_rows(view), grouped_batches(view), "{name}: octants");
        }
        let v_pairs = grouped_rows(lists.v());
        assert_eq!(v_pairs, grouped_batches(lists.v()), "{name}: V batches");
        for &(t, s, d) in &v_pairs {
            let direction = {
                let (_, target) = morton::decode(index.key(level, t as usize));
                let (_, source) = morton::decode(index.key(level, s as usize));
                [0, 1, 2].map(|dim| target[dim] as i64 - source[dim] as i64)
            };
            assert_eq!(V_LIST_DIRECTIONS[d], direction);
            assert_eq!(offset_index(direction), Some(d));
        }

        // Rows cover the boxes of the level, and the local leaves of the level, once each.
        let nboxes = index.len(level);
        for nrows in [
            lists.v().nrows(),
            lists.x().nrows(),
            lists.m2m_local().nrows(),
            lists.m2m_global().nrows(),
            lists.l2l().nrows(),
            lists.p2m().nrows(),
        ] {
            assert_eq!(nrows, nboxes, "{name}: box rows of level {level}");
        }
        for nrows in [lists.near().nrows(), lists.w().nrows(), lists.l2p().nrows()] {
            assert_eq!(nrows, local.len(), "{name}: leaf rows of level {level}");
        }
        for (r, leaf) in local.enumerate() {
            assert_eq!(lists.l2p().row(r), [leaves.box_index(leaf)]);
        }
    }

    named.sort_unstable();
    named.dedup();
    assert_eq!(ghost_leaves, named, "{name}: ghost leaves");
}

/// The value that the owner of `key` sends as the `j`-th entry of its chunk.
fn ghost_value(key: u64, j: usize) -> u64 {
    key.wrapping_mul(64).wrapping_add(j as u64)
}

/// Values per source point, as for Laplace (CONVENTIONS §3.13: coordinates, then charge).
const SOURCE_POINT_SIZE: usize = 4;

/// A source count per leaf that every rank can recompute from the key; zero for about one
/// key in five.
fn hashed_count(key: u64) -> usize {
    (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize % 5
}

/// The number of fine keys of `points` (sorted) that lie in `leaf`.
fn points_in(points: &[u64], leaf: u64) -> usize {
    points.iter().filter(|&&p| is_ancestor(leaf, p)).count()
}

/// Ghost keys and values that this rank receives in one exchange.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Traffic {
    keys: usize,
    values: usize,
}

impl Traffic {
    /// The sum over all ranks.
    fn total<C: CommunicatorCollectives>(self, comm: &C) -> Self {
        let mut all = [0usize; 2];
        comm.all_reduce_into(
            &[self.keys, self.values][..],
            &mut all[..],
            SystemOperation::sum(),
        );
        Self {
            keys: all[0],
            values: all[1],
        }
    }
}

/// Forward seeded source chunks with `count(key)` points of every local leaf, and check
/// that every ghost leaf receives its owner's count and values.
fn check_source_exchange<C: CommunicatorCollectives>(
    name: &str,
    comm: &C,
    plan: &Plan,
    count: impl Fn(u64) -> usize,
) -> Traffic {
    let leaves = plan.index().leaves();
    let nlocal = leaves.nlocal();
    let local_counts: Vec<usize> = (0..nlocal).map(|j| count(leaves.key(j))).collect();
    let mut exchange = SourceExchange::<u64>::new(plan, comm, &local_counts, SOURCE_POINT_SIZE)
        .unwrap_or_else(|error| panic!("{name}: source exchange: {error}"));
    assert_eq!(&exchange.leaf_counts()[..nlocal], local_counts, "{name}");
    assert_eq!(exchange.ghost_leaves(), leaves.ghosts(), "{name}");
    for (g, j) in leaves.ghosts().enumerate() {
        let key = leaves.key(j);
        assert_eq!(
            exchange.ghost_counts()[g],
            count(key),
            "{name}: owner's count of ghost leaf {key}"
        );
    }
    assert!(
        exchange
            .send_leaves()
            .iter()
            .all(|&j| (j as usize) < nlocal),
        "{name}: only local leaves are sent"
    );

    let mut sources = exchange.new_store();
    for j in 0..nlocal {
        let key = leaves.key(j);
        for (k, value) in sources.chunk_mut(j).iter_mut().enumerate() {
            *value = ghost_value(key, k);
        }
    }
    exchange.forward(&mut sources);
    for j in 0..leaves.len() {
        let key = leaves.key(j);
        let expected: Vec<u64> = (0..count(key) * SOURCE_POINT_SIZE)
            .map(|k| ghost_value(key, k))
            .collect();
        assert_eq!(
            sources.chunk(j),
            expected,
            "{name}: source chunk of leaf {key}"
        );
    }
    Traffic {
        keys: exchange.communicator().total_receive_count(),
        values: exchange.communicator().receive_buffer_len(),
    }
}

/// Forward the multipoles of every non-ghost box with `sizes[l]` values on level l, and
/// check that exactly the ghosts of the V- and W-lists of the oracle receive their
/// owner's values. Returns the traffic.
fn check_multipole_exchange<C: CommunicatorCollectives>(
    name: &str,
    comm: &C,
    all_keys: &HashMap<u64, KeyType>,
    oracle: &OracleLists,
    plan: &Plan,
    sizes: &[usize],
) -> Traffic {
    let index = plan.index();
    let mut exchange = MultipoleExchange::<u64>::new(plan, comm, sizes)
        .unwrap_or_else(|error| panic!("{name}: multipole exchange: {error}"));
    let mut multipoles = LevelBuffers::<u64>::from_index(index, sizes);
    for level in 0..index.nlevels() {
        for i in 0..index.len(level) {
            if !index.kind(level, i).is_ghost() {
                let key = index.key(level, i);
                for (k, value) in multipoles.chunk_mut(level, i).iter_mut().enumerate() {
                    *value = ghost_value(key, k);
                }
            }
        }
    }
    exchange.forward_all(&mut multipoles);

    // The ghosts of the V- and W-lists, by level.
    let mut expected = vec![Vec::new(); index.nlevels()];
    for &entry in oracle.values().flat_map(|[_, v, w, _]| v.iter().chain(w)) {
        if all_keys[&entry].is_ghost() {
            expected[morton::level(entry)].push(entry);
        }
    }

    let mut traffic = Traffic::default();
    for (level, expected) in expected.iter_mut().enumerate() {
        expected.sort_unstable();
        expected.dedup();
        let received = exchange.receive_boxes(level);
        assert!(received.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
        let received_keys: Vec<u64> = received
            .iter()
            .map(|&i| index.key(level, i as usize))
            .collect();
        assert_eq!(
            &received_keys, expected,
            "{name}: received ghosts of level {level}"
        );
        for i in 0..index.len(level) {
            let key = index.key(level, i);
            let chunk = multipoles.chunk(level, i);
            if index.kind(level, i).is_ghost() && received.binary_search(&(i as u32)).is_err() {
                assert!(
                    chunk.iter().all(|&v| v == 0),
                    "{name}: unrequested ghost {key}"
                );
            } else {
                for (k, &value) in chunk.iter().enumerate() {
                    assert_eq!(value, ghost_value(key, k), "{name}: multipole of {key}");
                }
            }
        }
        let communicator = exchange.communicator(level);
        traffic.keys += communicator.total_receive_count();
        traffic.values += communicator.receive_buffer_len();
    }
    traffic
}

/// Gather the multipoles of every rank's coarse blocks and check that every rank then
/// holds every block's values, in global key order, in the gathered buffer and in the
/// block's slot.
fn check_coarse_gather<C: CommunicatorCollectives>(
    name: &str,
    comm: &C,
    plan: &Plan,
    sizes: &[usize],
) -> Traffic {
    let index = plan.index();
    let rank = comm.rank() as usize;
    let mut gather = CoarseExchange::<u64>::new(plan, comm, sizes)
        .unwrap_or_else(|error| panic!("{name}: coarse gather: {error}"));
    let all_blocks = gather_to_all(plan.coarse_blocks(), comm);
    assert_eq!(
        gather.keys(),
        all_blocks,
        "{name}: coarse blocks in rank order"
    );
    assert!(
        all_blocks.windows(2).all(|pair| pair[0] < pair[1]),
        "{name}: rank order is key order"
    );
    assert_eq!(
        &gather.keys()[gather.rank_blocks(rank)],
        plan.coarse_blocks()
    );

    let mut multipoles = LevelBuffers::<u64>::from_index(index, sizes);
    for &key in plan.coarse_blocks() {
        let (level, i) = index.find(key).unwrap();
        for (k, value) in multipoles
            .chunk_mut(level, i as usize)
            .iter_mut()
            .enumerate()
        {
            *value = ghost_value(key, k);
        }
    }
    gather.gather(&mut multipoles);
    let first = gather.gathered().to_vec();
    for (b, &key) in gather.keys().iter().enumerate() {
        let (level, i) = gather.block(b);
        assert_eq!(
            index.key(level, i as usize),
            key,
            "{name}: slot of block {key}"
        );
        let expected: Vec<u64> = (0..sizes[level]).map(|k| ghost_value(key, k)).collect();
        assert_eq!(gather.chunk(b), expected, "{name}: gathered block {key}");
        assert_eq!(
            multipoles.chunk(level, i as usize),
            expected,
            "{name}: block {key} in its slot"
        );
    }
    gather.gather(&mut multipoles);
    assert_eq!(gather.gathered(), first, "{name}: two gathers agree");
    Traffic {
        keys: gather.keys().len(),
        values: gather.gathered().len(),
    }
}

/// A target count per leaf that every rank can recompute from the key, independent of
/// [`hashed_count`]; zero for about one key in five.
fn hashed_target_count(key: u64) -> usize {
    hashed_count(key ^ 0x5bd1_e995)
}

/// A per-pair test operator whose values do not matter. Every call adds to each output
/// value a hash of all its inputs and of the keys of the pair, so the values depend on
/// every input, ghost sources and ghost multipoles included, and a stale or doubled
/// value shows. It checks that an evaluation repeats bit for bit, not that it is right.
#[derive(Clone, Copy, Debug)]
struct Mixer;

/// The start of [`mix`]'s hash (FNV-1a), so that the root key and empty inputs still
/// give a nonzero hash.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// Add a hash of `inputs` and `keys` to every value of `output`.
fn mix(inputs: [&[u32]; 2], keys: [u64; 2], output: &mut [u32]) {
    let hash = inputs.iter().flat_map(|input| input.iter()).fold(
        OFFSET_BASIS ^ keys[0] ^ keys[1].rotate_left(32),
        |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x0100_0000_01b3),
    );
    for (k, value) in output.iter_mut().enumerate() {
        *value = value.wrapping_add((hash >> (k % 32)) as u32);
    }
}

impl FmmSizes for Mixer {
    type Value = u32;

    fn multipole_size(&self, level: usize) -> usize {
        1 + level % 3
    }

    fn local_size(&self, _level: usize) -> usize {
        2
    }

    fn source_point_size(&self) -> usize {
        2
    }

    fn target_input_point_size(&self) -> usize {
        1
    }

    fn target_output_point_size(&self) -> usize {
        2
    }
}

impl PairOperator for Mixer {
    fn p2m(&mut self, leaf: u64, sources: &[u32], multipole: &mut [u32]) {
        mix([sources, &[]], [leaf, leaf], multipole);
    }

    fn m2m(&mut self, child: u64, parent: u64, _: usize, input: &[u32], output: &mut [u32]) {
        mix([input, &[]], [child, parent], output);
    }

    fn m2l(&mut self, source: u64, target: u64, _: usize, input: &[u32], output: &mut [u32]) {
        mix([input, &[]], [source, target], output);
    }

    fn p2l(&mut self, source: u64, target: u64, sources: &[u32], local: &mut [u32]) {
        mix([sources, &[]], [source, target], local);
    }

    fn l2l(&mut self, parent: u64, child: u64, _: usize, input: &[u32], output: &mut [u32]) {
        mix([input, &[]], [parent, child], output);
    }

    fn l2p(&mut self, leaf: u64, local: &[u32], target_input: &[u32], output: &mut [u32]) {
        mix([local, target_input], [leaf, leaf], output);
    }

    fn m2p(
        &mut self,
        source: u64,
        target: u64,
        multipole: &[u32],
        target_input: &[u32],
        output: &mut [u32],
    ) {
        mix([multipole, target_input], [source, target], output);
    }

    fn p2p(
        &mut self,
        source: u64,
        target: u64,
        sources: &[u32],
        target_input: &[u32],
        output: &mut [u32],
    ) {
        mix([sources, target_input], [source, target], output);
    }
}

/// Fill every local leaf's chunk of `values` from its key and `salt`.
fn fill_by_key(plan: &Plan, mut values: LeafSliceMut<'_, u32>, salt: u64) {
    let leaves = plan.index().leaves();
    for (j, chunk) in values.chunks_mut().enumerate() {
        for (k, value) in chunk.iter_mut().enumerate() {
            *value = ghost_value(leaves.key(j) ^ salt, k) as u32;
        }
    }
}

/// The stores an evaluation writes: target output, multipoles, locals and sources
/// (ghosts included).
type Stores = (
    LeafStore<u32>,
    LevelBuffers<u32>,
    LevelBuffers<u32>,
    LeafStore<u32>,
);

/// Return the [`Stores`] of `evaluator`.
fn stores<C: CommunicatorCollectives, P: Borrow<Plan>>(
    evaluator: &Evaluator<'_, C, PerPair<Mixer>, P>,
) -> Stores {
    (
        evaluator.target_output_store().clone(),
        evaluator.multipoles().clone(),
        evaluator.locals().clone(),
        evaluator.source_store().clone(),
    )
}

/// Evaluate with [`Mixer`] and seeded variable counts (zeros included), and check that
/// evaluations are bit-identical: twice on one evaluator, and on an evaluator that owns
/// a copy of the plan. Collective.
fn check_repeated_evaluation<C: CommunicatorCollectives>(name: &str, comm: &C, plan: &Plan) {
    let leaves = plan.index().leaves();
    let sources: Vec<usize> = (0..leaves.nlocal())
        .map(|j| hashed_count(leaves.key(j)))
        .collect();
    let targets: Vec<usize> = (0..leaves.nlocal())
        .map(|j| hashed_target_count(leaves.key(j)))
        .collect();
    let mut borrowing = Evaluator::new(plan, comm, PerPair(Mixer), &sources, &targets)
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error}"));
    // This one owns a copy of the plan; it must agree with the borrowing one.
    let mut owning = Evaluator::new(plan.clone(), comm, PerPair(Mixer), &sources, &targets)
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error}"));
    fill_by_key(plan, borrowing.local_sources_mut(), 0);
    fill_by_key(plan, borrowing.local_target_inputs_mut(), 1);
    fill_by_key(plan, owning.local_sources_mut(), 0);
    fill_by_key(plan, owning.local_target_inputs_mut(), 1);
    borrowing.evaluate();
    owning.evaluate();
    let first = stores(&borrowing);

    // The check is not vacuous: some rank holds a nonzero multipole.
    let local_nonzero = first.1.as_slice().iter().any(|&value| value != 0);
    let mut nonzero = false;
    comm.all_reduce_into(&local_nonzero, &mut nonzero, SystemOperation::logical_or());
    assert!(nonzero, "{name}: every multipole is zero");

    borrowing.evaluate();
    assert!(
        stores(&borrowing) == first,
        "{name}: two evaluations differ"
    );
    assert!(
        stores(&owning) == first,
        "{name}: the evaluator that owns its plan differs"
    );
}

/// One level call: (method, level, pass of an M2M).
type Call = (&'static str, usize, Option<UpwardPass>);

/// A test operator that computes nothing. It checks every batch it receives (the
/// groupings of design §4.4, both views, buffer shapes) and records every call and every
/// pair, by key.
struct Recorder<'p> {
    plan: &'p Plan,
    calls: Vec<Call>,
    /// (method, target key, source key) of every pair issued.
    pairs: Vec<(&'static str, u64, u64)>,
}

impl Recorder<'_> {
    fn key(&self, level: usize, i: u32) -> u64 {
        self.plan.index().key(level, i as usize)
    }

    fn leaf(&self, j: u32) -> u64 {
        self.plan.index().leaf_key(j as usize)
    }

    /// Check both views of a grouped batch and return its (target, source, group)
    /// triples.
    fn grouped<G: Copy + Into<usize>>(&self, view: &GroupedCsr<G>) -> Vec<(u32, u32, usize)> {
        let rows = grouped_rows(view);
        assert_eq!(
            rows,
            grouped_batches(view),
            "both views hold the same pairs"
        );
        rows
    }
}

impl FmmSizes for Recorder<'_> {
    type Value = u32;

    fn multipole_size(&self, _level: usize) -> usize {
        1
    }

    fn local_size(&self, _level: usize) -> usize {
        1
    }

    fn source_point_size(&self) -> usize {
        1
    }

    fn target_input_point_size(&self) -> usize {
        0
    }

    fn target_output_point_size(&self) -> usize {
        1
    }
}

impl FmmOperator for Recorder<'_> {
    fn p2m(&mut self, b: P2m<'_, u32>) {
        assert_eq!(b.multipoles.len(), b.index.len(b.level));
        assert_eq!(b.leaves.nrows(), b.index.len(b.level));
        for t in 0..b.leaves.nrows() {
            for &j in b.leaves.row(t) {
                let pair = ("p2m", self.key(b.level, t as u32), self.leaf(j));
                self.pairs.push(pair);
            }
        }
        self.calls.push(("p2m", b.level, None));
    }

    fn m2m(&mut self, b: M2m<'_, u32>) {
        assert_eq!(b.multipoles.len(), b.index.len(b.level));
        assert_eq!(b.child_multipoles.len(), b.index.len(b.level + 1));
        let triples = self.grouped(b.children);
        // Complete octant batches: every parent appears in all eight, with its child of
        // that octant.
        let parents: Vec<u32> = (0..b.children.nrows() as u32)
            .filter(|&t| !b.children.row(t as usize).0.is_empty())
            .collect();
        for o in 0..8 {
            assert_eq!(b.children.batch(o).0, parents, "octant batch {o}");
        }
        let method = match b.pass {
            UpwardPass::Local => "m2m local",
            UpwardPass::Global => "m2m global",
        };
        for (t, c, o) in triples {
            let (parent, child) = (self.key(b.level, t), self.key(b.level + 1, c));
            assert_eq!(morton::children(parent).unwrap()[o], child);
            self.pairs.push((method, parent, child));
        }
        self.calls.push(("m2m", b.level, Some(b.pass)));
    }

    fn m2l(&mut self, b: M2l<'_, u32>) {
        assert_eq!(b.locals.len(), b.index.len(b.level));
        assert_eq!(b.multipoles.len(), b.index.len(b.level));
        for (t, s, d) in self.grouped(b.pairs) {
            let (target, source) = (self.key(b.level, t), self.key(b.level, s));
            let (_, ti) = morton::decode(target);
            let (_, si) = morton::decode(source);
            let offset = [0, 1, 2].map(|k| ti[k] as i64 - si[k] as i64);
            assert_eq!(offset_index(offset), Some(d), "offset index of a V pair");
            self.pairs.push(("m2l", target, source));
        }
        self.calls.push(("m2l", b.level, None));
    }

    fn p2l(&mut self, b: P2l<'_, u32>) {
        assert_eq!(b.locals.len(), b.index.len(b.level));
        for t in 0..b.x.nrows() {
            for &j in b.x.row(t) {
                let pair = ("p2l", self.key(b.level, t as u32), self.leaf(j));
                self.pairs.push(pair);
            }
        }
        self.calls.push(("p2l", b.level, None));
    }

    fn l2l(&mut self, b: L2l<'_, u32>) {
        assert_eq!(b.locals.len(), b.index.len(b.level));
        assert_eq!(b.parent_locals.len(), b.index.len(b.level - 1));
        let triples = self.grouped(b.parents);
        for t in 0..b.parents.nrows() {
            assert!(b.parents.row(t).0.len() <= 1, "one parent per child");
        }
        for (t, p, o) in triples {
            let (child, parent) = (self.key(b.level, t), self.key(b.level - 1, p));
            assert_eq!(morton::parent(child), Some(parent));
            assert_eq!(morton::child_index(child), o);
            self.pairs.push(("l2l", child, parent));
        }
        self.calls.push(("l2l", b.level, None));
    }

    fn l2p(&mut self, b: L2p<'_, u32>) {
        assert_eq!(b.leaves, b.index.leaves().local(b.level));
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        assert_eq!(b.boxes.nrows(), b.leaves.len());
        for r in 0..b.boxes.nrows() {
            for &i in b.boxes.row(r) {
                let pair = (
                    "l2p",
                    self.leaf((b.leaves.start + r) as u32),
                    self.key(b.level, i),
                );
                self.pairs.push(pair);
            }
        }
        self.calls.push(("l2p", b.level, None));
    }

    fn m2p(&mut self, b: M2p<'_, u32>) {
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        assert_eq!(b.w.nrows(), b.leaves.len());
        for r in 0..b.w.nrows() {
            for &s in b.w.row(r) {
                let target = self.leaf((b.leaves.start + r) as u32);
                self.pairs.push(("m2p", target, self.key(b.level + 1, s)));
            }
        }
        self.calls.push(("m2p", b.level, None));
    }

    fn p2p(&mut self, b: P2p<'_, u32>) {
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        assert_eq!(b.near.nrows(), b.leaves.len());
        assert_eq!(b.sources.nleaves(), b.index.leaves().len());
        for r in 0..b.near.nrows() {
            for &j in b.near.row(r) {
                let target = self.leaf((b.leaves.start + r) as u32);
                self.pairs.push(("p2p", target, self.leaf(j)));
            }
        }
        self.calls.push(("p2p", b.level, None));
    }
}

/// The level calls of one evaluation, in the order of design §7.2.
fn expected_calls(nlevels: usize) -> Vec<Call> {
    let deepest = nlevels - 1;
    let mut calls = Vec::new();
    for level in (0..=deepest).rev() {
        calls.push(("p2m", level, None));
        if level > 0 {
            calls.push(("m2m", level - 1, Some(UpwardPass::Local)));
        }
    }
    for level in (0..deepest).rev() {
        calls.push(("m2m", level, Some(UpwardPass::Global)));
    }
    for level in 1..=deepest {
        calls.extend([
            ("l2l", level, None),
            ("m2l", level, None),
            ("p2l", level, None),
        ]);
    }
    for level in 0..=deepest {
        calls.extend([
            ("l2p", level, None),
            ("m2p", level, None),
            ("p2p", level, None),
        ]);
    }
    calls
}

/// The pairs (method, target, source) of `target` with every one of `sources`.
fn pairs_of(method: &'static str, target: u64, sources: &[u64]) -> Vec<(&'static str, u64, u64)> {
    sources
        .iter()
        .map(|&source| (method, target, source))
        .collect()
}

/// Evaluate with a recording operator: every level gets every call once, in pass order;
/// every batch honours its grouping; and the pairs issued are exactly the pairs of the
/// oracle lists, each once.
fn check_batches<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    oracle: &OracleLists,
    plan: &Plan,
) {
    let nlocal = plan.index().leaves().nlocal();
    let counts: Vec<usize> = (0..nlocal)
        .map(|j| hashed_count(plan.index().leaf_key(j)))
        .collect();
    let recorder = Recorder {
        plan,
        calls: Vec::new(),
        pairs: Vec::new(),
    };
    let mut evaluator = Evaluator::new(plan, octree.comm(), recorder, &counts, &counts)
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error}"));
    evaluator.evaluate();
    let recorder = evaluator.operator();
    assert_eq!(recorder.calls, expected_calls(plan.nlevels()), "{name}");

    let mut expected = Vec::new();
    for (&key, &kind) in octree.all_keys() {
        if kind.is_ghost() {
            continue;
        }
        let entries = |method, sources: &[u64]| pairs_of(method, key, sources);
        let [u, v, w, x] = &oracle[&key];
        expected.extend(entries("m2l", v));
        expected.extend(entries("p2l", x));
        if let Some(parent) = morton::parent(key) {
            expected.push(("l2l", key, parent));
        }
        let children = morton::children(key)
            .map(|c| c.to_vec())
            .unwrap_or_default();
        match kind {
            KeyType::LocalLeaf => {
                expected.extend([("p2m", key, key), ("l2p", key, key), ("p2p", key, key)]);
                expected.extend(entries("m2p", w));
                expected.extend(entries("p2p", u));
            }
            KeyType::LocalInterior => expected.extend(entries("m2m local", &children)),
            KeyType::Global => expected.extend(entries("m2m global", &children)),
            _ => unreachable!(),
        }
    }
    expected.sort_unstable();
    let mut issued = recorder.pairs.clone();
    issued.sort_unstable();
    assert_eq!(
        issued.len(),
        expected.len(),
        "{name}: number of pairs issued"
    );
    assert!(
        issued == expected,
        "{name}: every list pair is issued exactly once"
    );
}

#[test]
fn distributed_tree_regressions() {
    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    let rank = comm.rank() as usize;
    let cloud: Vec<_> = (0..48)
        .map(|i| {
            let j = i + 17 * rank;
            [
                ((13 * j + 7) % 97) as f64 / 97.0,
                ((29 * j + 3) % 89) as f64 / 89.0,
                ((11 * j + 5) % 83) as f64 / 83.0,
            ]
        })
        .collect();
    let mut reversed = cloud.clone();
    reversed.reverse();
    // A dense blob filling the 64 level-6 cells of one level-4 cell. Next to the
    // sparse cloud it forces a strongly graded tree, so local leaves meet much
    // coarser remote neighbours and the V- and W-lists reach deep into other
    // ranks. The blob sits in a different, well separated level-4 cell on every
    // rank — a rank-independent blob would collapse into a single owner's
    // subdomain and lose the per-rank deep/shallow contrast.
    let blob: Vec<[f64; 3]> = {
        // Eight level-6 cells per step, so the blobs never touch, and an offset of
        // 0.005 keeps the points off the cell boundaries.
        let corner = |n: usize| 0.005 + (n % 8) as f64 / 8.0;
        let origin = [corner(rank), corner(3 * rank + 1), corner(5 * rank + 2)];
        (0..64usize)
            .map(|i| {
                let coordinate = |k: usize, d: usize| origin[d] + (k % 4) as f64 / 64.0;
                [
                    coordinate(i, 0),
                    coordinate(i / 4, 1),
                    coordinate(i / 16, 2),
                ]
            })
            .collect()
    };
    let graded: Vec<[f64; 3]> = cloud.iter().chain(blob.iter()).copied().collect();
    let cases = [
        (
            "unsorted unequal populations",
            cloud.clone(),
            reversed[..31].to_vec(),
            6,
            8,
        ),
        ("identical populations", cloud.clone(), cloud.clone(), 6, 8),
        (
            "coarse refinement cap",
            cloud.clone(),
            reversed.clone(),
            2,
            2,
        ),
        (
            "large leaf capacity",
            cloud.clone(),
            reversed.clone(),
            6,
            10000,
        ),
        (
            "duplicate keys",
            vec![[0.125; 3]; 17],
            vec![[0.875; 3]; 9],
            3,
            4,
        ),
        (
            "single point per population",
            vec![[0.125; 3]],
            vec![[0.875; 3]],
            3,
            4,
        ),
        ("no sources", vec![], cloud.clone(), 4, 8),
        ("no targets", cloud.clone(), vec![], 4, 8),
        (
            "uneven rank populations",
            cloud[..(rank % cloud.len()) + 1].to_vec(),
            reversed[..(rank % 5) + 2].to_vec(),
            4,
            8,
        ),
        (
            "graded corner blob",
            graded,
            reversed.iter().chain(blob.iter()).copied().collect(),
            6,
            1,
        ),
        (
            "dense max-level leaf",
            if rank == 0 {
                cloud
                    .iter()
                    .copied()
                    .chain([[0.3, 0.6, 0.4]; 1000])
                    .collect()
            } else {
                cloud.clone()
            },
            reversed.clone(),
            5,
            8,
        ),
        (
            "empty input ranks",
            if rank == 0 { cloud.clone() } else { vec![] },
            if rank + 1 == comm.size() as usize {
                reversed.clone()
            } else {
                vec![]
            },
            4,
            8,
        ),
    ];
    for (name, sources, targets, max_level, capacity) in cases {
        eprintln!("rank {rank}: {name}");
        // One tree serves both populations, as in an FMM with distinct
        // sources and targets. The ghost-children layer makes every list entry
        // a key of the tree.
        let mut fine_keys = keys(&sources);
        fine_keys.extend(keys(&targets));
        let options = OctreeOptions::new()
            .with_max_level(max_level)
            .with_max_fine_keys(capacity)
            .with_ghost_children(true);
        let octree = Octree::new(&fine_keys, options, &comm);
        let leaves = octree.leaf_keys();
        assert!(leaves.windows(2).all(|pair| pair[0] < pair[1]));

        // The oracle needs the global tree, so the gather is entered by every rank.
        let mut global_leaves = gather_to_all(leaves, &comm);
        global_leaves.sort_unstable();
        assert!(morton::is_complete_linear_and_balanced(&global_leaves));
        let oracle = oracle_of(octree.all_keys(), &global_leaves);

        // The index-form plan, against the oracle. Building it is collective.
        let plan =
            Plan::new(&octree).unwrap_or_else(|error| panic!("rank {rank}: {name}: {error}"));
        check_plan(
            name,
            octree.all_keys(),
            &oracle,
            &plan,
            octree.coarse_tree_leafs(),
        );
        // The graded tree has `Global` boxes on every rank count and spreads its lists
        // over several ranks, so the global pass and the ghost leaves are exercised.
        let global_pairs: usize = plan.levels().iter().map(|l| l.m2m_global().len()).sum();
        let mut ghost_leaves = 0usize;
        comm.all_reduce_into(
            &plan.index().leaves().ghosts().len(),
            &mut ghost_leaves,
            SystemOperation::sum(),
        );
        if name == "graded corner blob" {
            assert!(global_pairs > 0, "{name}: no global M2M pair");
            assert!(
                comm.size() == 1 || ghost_leaves > 0,
                "{name}: no ghost leaf"
            );
        }

        // The exchanges. Sources: counts of one, counts from a hash of the key (zeros
        // included), and the real source points per leaf.
        let mut source_points = gather_to_all(&keys(&sources), &comm);
        source_points.sort_unstable();
        let ones = check_source_exchange(name, &comm, &plan, |_| 1).total(&comm);
        let hashed = check_source_exchange(name, &comm, &plan, hashed_count).total(&comm);
        let points =
            check_source_exchange(name, &comm, &plan, |key| points_in(&source_points, key))
                .total(&comm);
        // One ghost key per ghost leaf, whatever the counts.
        assert_eq!(ones.keys, ghost_leaves, "{name}: source ghost keys");
        assert_eq!(ones.values, ghost_leaves * SOURCE_POINT_SIZE, "{name}");
        assert_eq!(hashed.keys, ghost_leaves, "{name}: source ghost keys");
        assert_eq!(points.keys, ghost_leaves, "{name}: source ghost keys");
        // Multipole sizes per level: uniform in one scenario, growing with the level in
        // the others.
        let sizes: Vec<usize> = (0..plan.nlevels())
            .map(|level| {
                if name == "identical populations" {
                    3
                } else {
                    level + 1
                }
            })
            .collect();
        let multipoles =
            check_multipole_exchange(name, &comm, octree.all_keys(), &oracle, &plan, &sizes)
                .total(&comm);
        let coarse = check_coarse_gather(name, &comm, &plan, &sizes);
        if comm.size() > 1 && name == "graded corner blob" {
            assert!(
                ones.keys > 0 && multipoles.keys > 0,
                "{name}: no ghost exchanged"
            );
        }
        if name == "dense max-level leaf" {
            let leaves = plan.index().leaves();
            let local_max = (0..leaves.nlocal())
                .map(|j| points_in(&source_points, leaves.key(j)))
                .max()
                .unwrap_or(0);
            let mut dense = 0usize;
            comm.all_reduce_into(&local_max, &mut dense, SystemOperation::max());
            assert!(dense >= 1000, "{name}: the dense leaf holds {dense} points");
        }
        if rank == 0 && (name == "graded corner blob" || name == "dense max-level leaf") {
            println!(
                "traffic on {} ranks, {name}: ghost keys / values received, summed over ranks",
                comm.size()
            );
            println!("  sources, counts of one:  {ones:?}");
            println!("  sources, hashed counts:  {hashed:?}");
            println!("  sources, source points:  {points:?}");
            println!("  multipoles:              {multipoles:?}");
            println!("  coarse gather, per rank: {coarse:?}");
        }

        // The evaluator: determinism with variable counts, and the batches it issues.
        check_repeated_evaluation(name, &comm, &plan);
        check_batches(name, &octree, &oracle, &plan);
    }
}
