//! Public-API regression tests. Keep MPI initialization in one test: MPI cannot
//! be initialized again after finalization within the same process.
use std::{borrow::Borrow, collections::HashMap, ops::Range};

use mpi::{collective::SystemOperation, topology::SimpleCommunicator, traits::*};
use nd_fmm_plan::{
    evaluator::Evaluator,
    exchange::{
        self, MultipoleExchange, MultipoleRouting, ReductionExchange, SourceExchange,
        TOP_MAX_BOXES, TOP_READER_FRACTION, TopExchange,
    },
    interaction_manager::V_LIST_DIRECTIONS,
    lists::{GroupedCsr, offset_index},
    operator::{
        FmmOperator, FmmSizes, HostData, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator,
        PerPair, UpwardPass,
    },
    plan::Plan,
    redistribute::{Redistribution, RedistributionError},
    store::{LeafSliceMut, LeafStore, LevelBuffers},
};
use nd_octree::{
    Octree, OctreeOptions, PhysicalBox,
    constants::DEEPEST_LEVEL,
    morton,
    morton::is_ancestor,
    octree::{KeyType, PartitionMap},
    points_to_morton,
};
use rlst::{
    distributed_tools::{GhostCommunicator, all_to_allv, array_tools::gather_to_all},
    rlst_dynamic_array,
};

/// The forwarding tree at its narrowest (Phase 5S T8s): a fan-out of 2, so that boxes with
/// three readers or more are forwarded at 4–8 ranks, and made-up nodes of two ranks each,
/// so that the node split runs. With this rank's node.
fn narrow_tree<C: Communicator>(comm: &C) -> (MultipoleRouting, usize) {
    (
        MultipoleRouting::Tree { fan_out: 2 },
        comm.rank() as usize / 2,
    )
}

/// The default routing with this rank as its own node, as `Evaluator::new` builds it.
fn default_routing<C: Communicator>(comm: &C) -> (MultipoleRouting, usize) {
    (MultipoleRouting::default(), comm.rank() as usize)
}

/// Check what the multipole exchange of `level` reports about its traffic: the values
/// received are the receive buffer, at least one message per rank sent to or received
/// from, at most `fan_out` copies of a box from this rank under a tree, and over all
/// ranks as many messages and values sent as received. Collective.
fn check_multipole_traffic<C: CommunicatorCollectives, T: Copy + Default + Equivalence>(
    name: &str,
    comm: &C,
    exchange: &MultipoleExchange<T>,
    level: usize,
) {
    let traffic = exchange.traffic(level);
    assert_eq!(
        traffic.values_received,
        exchange.receive_buffer(level).len(),
        "{name}"
    );
    assert!(
        traffic.messages_sent >= exchange.destinations(level).len(),
        "{name}"
    );
    assert!(
        traffic.messages_received >= exchange.sources(level).len(),
        "{name}"
    );
    if let MultipoleRouting::Tree { fan_out } = exchange.routing() {
        assert!(
            exchange.fan_out(level) <= fan_out,
            "{name}: a box sent to {} ranks on level {level}",
            exchange.fan_out(level)
        );
    }
    let local = [
        traffic.messages_sent,
        traffic.messages_received,
        traffic.values_sent,
        traffic.values_received,
    ];
    let mut total = [0usize; 4];
    comm.all_reduce_into(&local[..], &mut total[..], SystemOperation::sum());
    assert_eq!(
        [total[0], total[2]],
        [total[1], total[3]],
        "{name}: messages and values sent and received over all ranks"
    );
}

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

/// Whether this rank needs the local of the non-ghost `key`: every key but a `Global` box
/// that is not an ancestor of one of the rank's own coarse blocks (P2).
fn needs_local(all_keys: &HashMap<u64, KeyType>, own_blocks: &[u64], key: u64) -> bool {
    all_keys[&key] != KeyType::Global
        || own_blocks
            .iter()
            .any(|&block| block != key && is_ancestor(key, block))
}

/// The keys the plan holds, with their kinds: since Phase 5S T5 the octree's locally
/// essential keys H ([`check_held_keys`]).
fn held_keys(plan: &Plan) -> HashMap<u64, KeyType> {
    let index = plan.index();
    (0..index.nlevels())
        .flat_map(|level| {
            index
                .keys(level)
                .iter()
                .copied()
                .zip(index.kinds(level).iter().copied())
        })
        .collect()
}

/// Compute the [`OracleLists`] from the local key map and the sorted global leaves. A
/// `Global` box whose local this rank does not need ([`needs_local`]) has empty V- and
/// X-lists (P2).
fn oracle_of(
    all_keys: &HashMap<u64, KeyType>,
    global_leaves: &[u64],
    own_blocks: &[u64],
) -> OracleLists {
    let mut keys: Vec<u64> = morton::get_interior_keys(global_leaves)
        .into_iter()
        .collect();
    keys.extend_from_slice(global_leaves);
    keys.sort_unstable();
    keys.dedup();
    all_keys
        .iter()
        .filter(|(_, kind)| !kind.is_ghost())
        .map(|(&key, _)| {
            let [u, v, w, x] = oracle_lists(&keys, global_leaves, key);
            if needs_local(all_keys, own_blocks, key) {
                (key, [u, v, w, x])
            } else {
                (key, [u, Vec::new(), w, Vec::new()])
            }
        })
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

/// The held keys of `plan` against brute force over the gathered tree (Phase 5S T4, T5;
/// docs/design/scale-out.md §3.1, §4): exactly H = N(S) ∪ C(N(S)), so no key is held that
/// no list, halo or parent-child step of S names; and every kind the octree's: an own key
/// local, a shared ancestor `Global` (on one rank the root unless it is the only leaf), a
/// top-tree key above other ranks' blocks only a ghost of the rank of its first block,
/// every other key a ghost of its block's rank. Returns this rank's |H| and the number of
/// top-tree keys (blocks and ancestors of one) among them.
///
/// # Collective operation
/// Two all-gathers (the blocks and their counts); every rank must call it.
fn check_held_keys<C: CommunicatorCollectives>(
    name: &str,
    comm: &C,
    held: &HashMap<u64, KeyType>,
    octree: &Octree<'_, C>,
    global_leaves: &[u64],
) -> [usize; 2] {
    let nranks = comm.size() as usize;
    let rank = comm.rank() as usize;
    let own_blocks = octree.coarse_tree_leafs();
    let mut counts = vec![0usize; nranks];
    comm.all_gather_into(&own_blocks.len(), &mut counts[..]);
    let blocks = gather_to_all(own_blocks, comm);
    let mut owners = Vec::with_capacity(blocks.len());
    for (r, &count) in counts.iter().enumerate() {
        owners.extend(std::iter::repeat_n(r, count));
    }
    assert!(blocks.windows(2).all(|pair| pair[0] < pair[1]), "{name}");

    let mut tree: Vec<u64> = morton::get_interior_keys(global_leaves)
        .into_iter()
        .collect();
    tree.extend_from_slice(global_leaves);
    tree.sort_unstable();
    tree.dedup();
    let in_tree = |key: u64| tree.binary_search(&key).is_ok();
    let is_leaf = |key: u64| global_leaves.binary_search(&key).is_ok();
    // The first block at or below `key`, and the block a key below the blocks lies in.
    let first_block = |key: u64| blocks.iter().position(|&block| is_ancestor(key, block));
    let block_of = |key: u64| blocks.iter().position(|&block| is_ancestor(block, key));

    // T: every block and every ancestor of one.
    let mut top: Vec<u64> = Vec::new();
    for &block in &blocks {
        let mut key = Some(block);
        while let Some(k) = key {
            top.push(k);
            key = morton::parent(k);
        }
    }
    top.sort_unstable();
    top.dedup();

    // S: the own keys and the shared ancestors; N(S) and C(N(S)) by geometry.
    let own = |key: u64| own_blocks.iter().any(|&block| is_ancestor(block, key));
    let shared = |key: u64| {
        own_blocks
            .iter()
            .any(|&block| block != key && is_ancestor(key, block))
    };
    let s: Vec<u64> = tree
        .iter()
        .copied()
        .filter(|&key| own(key) || shared(key))
        .collect();
    let mut essential = s.clone();
    for &key in &s {
        let (level, index) = morton::decode(key);
        let cells = 1i64 << level;
        for offset in 0..27 {
            let d = [offset % 3, offset / 3 % 3, offset / 9].map(|d| d as i64 - 1);
            let cell = [0, 1, 2].map(|k| index[k] as i64 + d[k]);
            if cell.iter().all(|&c| (0..cells).contains(&c)) {
                let neighbour = morton::from_index_and_level(cell.map(|c| c as usize), level);
                if in_tree(neighbour) {
                    essential.push(neighbour);
                }
            }
        }
    }
    essential.sort_unstable();
    essential.dedup();
    let children: Vec<u64> = essential
        .iter()
        .filter(|&&key| !is_leaf(key))
        .flat_map(|&key| morton::children(key).unwrap())
        .collect();
    essential.extend(children);
    essential.sort_unstable();
    essential.dedup();

    let mut actual: Vec<u64> = held.keys().copied().collect();
    actual.sort_unstable();
    let extra: Vec<u64> = actual
        .iter()
        .copied()
        .filter(|key| essential.binary_search(key).is_err())
        .collect();
    let missing: Vec<u64> = essential
        .iter()
        .copied()
        .filter(|key| actual.binary_search(key).is_err())
        .collect();
    assert!(
        extra.is_empty() && missing.is_empty(),
        "rank {rank}: {name}: held keys not H: {} extra {extra:?}, {} missing {missing:?}",
        extra.len(),
        missing.len()
    );

    for (&key, &kind) in held {
        let above_blocks = top.binary_search(&key).is_ok() && blocks.binary_search(&key).is_err();
        let one_rank_root = nranks == 1 && key == morton::root() && !is_leaf(key);
        let expected = if shared(key) || one_rank_root {
            KeyType::Global
        } else if above_blocks {
            KeyType::GhostInterior(owners[first_block(key).expect("an ancestor of a block")])
        } else {
            let owner = owners[block_of(key).expect("a key below the blocks lies in one")];
            match (owner == rank, is_leaf(key)) {
                (true, true) => KeyType::LocalLeaf,
                (true, false) => KeyType::LocalInterior,
                (false, true) => KeyType::GhostLeaf(owner),
                (false, false) => KeyType::GhostInterior(owner),
            }
        };
        assert_eq!(kind, expected, "rank {rank}: {name}: kind of {key}");
    }
    let top_held = held
        .keys()
        .filter(|key| top.binary_search(key).is_ok())
        .count();
    [held.len(), top_held]
}

/// Whether this rank computes the multipole of the held `key`: a local key, or a `Global`
/// box it owns, the owner of a top-tree key being the rank of its first descendant block
/// (the octree's partition map; docs/design/scale-out.md §4.1).
fn computes<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    held: &HashMap<u64, KeyType>,
    key: u64,
) -> bool {
    match held[&key] {
        KeyType::LocalLeaf | KeyType::LocalInterior => true,
        KeyType::Global => {
            let map = octree.partition_map();
            map.owner(key) == map.rank()
        }
        KeyType::GhostLeaf(_) | KeyType::GhostInterior(_) => false,
    }
}

/// Compare the index-form lists of every local non-ghost box against the brute-force
/// oracle, through the key and index maps, and check the invariants of the views: a
/// `Global` box has its children in its global M2M row where `computed` says this rank
/// owns it, and an empty row elsewhere (Phase 5S T5).
fn check_plan(
    name: &str,
    all_keys: &HashMap<u64, KeyType>,
    oracle: &OracleLists,
    plan: &Plan,
    coarse_blocks: &[u64],
    computed: impl Fn(u64) -> bool,
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
            assert_eq!(
                plan.computes(level, i),
                computed(key),
                "{name}: owner of {key}"
            );
            if kind == KeyType::LocalInterior || (kind == KeyType::Global && computed(key)) {
                assert_eq!(children, morton::children(key).unwrap(), "{name}: children");
                assert_eq!(octants, [0, 1, 2, 3, 4, 5, 6, 7]);
            } else {
                assert!(children.is_empty());
            }
            // P2: no parent for a `Global` box whose local this rank does not need.
            let (parents, octants) = lists.l2l().row(i);
            let parents: Vec<u64> = parents.iter().map(key_on(level.wrapping_sub(1))).collect();
            let parent = morton::parent(key).filter(|_| needs_local(all_keys, &own_blocks, key));
            assert_eq!(parents, parent.into_iter().collect::<Vec<_>>(), "{name}");
            if parent.is_some() {
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

        // P1: X and near rows in the entry leaf's (level, key), on every rank count.
        for view in [lists.x(), lists.near()] {
            for t in 0..view.nrows() {
                let row: Vec<(usize, u64)> = view
                    .row(t)
                    .iter()
                    .map(|&j| (leaves.level(j as usize), leaves.key(j as usize)))
                    .collect();
                assert!(
                    row.windows(2).all(|pair| pair[0] < pair[1]),
                    "{name}: X or near row {t} of level {level} not in (level, key) order"
                );
            }
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

/// Check the [`exchange::Traffic`] of one exchange against its communicator, and that the
/// messages and values sent over all ranks are those received: a neighbour without values
/// is skipped on both sides. Collective.
fn check_traffic<C: CommunicatorCollectives>(
    name: &str,
    comm: &C,
    traffic: exchange::Traffic,
    communicator: &GhostCommunicator<u64>,
) {
    assert_eq!(
        traffic.values_sent,
        communicator.send_buffer_len(),
        "{name}"
    );
    assert_eq!(
        traffic.values_received,
        communicator.receive_buffer_len(),
        "{name}"
    );
    assert!(
        traffic.messages_sent <= communicator.out_ranks().len(),
        "{name}"
    );
    assert!(
        traffic.messages_received <= communicator.in_ranks().len(),
        "{name}"
    );
    let local = [
        traffic.messages_sent,
        traffic.messages_received,
        traffic.values_sent,
        traffic.values_received,
    ];
    let mut total = [0usize; 4];
    comm.all_reduce_into(&local[..], &mut total[..], SystemOperation::sum());
    assert_eq!(
        [total[0], total[2]],
        [total[1], total[3]],
        "{name}: messages and values sent and received over all ranks"
    );
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
    let mut overlapped = sources.clone();
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

    // The non-blocking exchange (Phase 5 T9) delivers the same bits, into a ghost tail
    // that holds other values until the wait, with `test` calls during the work.
    let ghosts = exchange.ghost_leaves();
    overlapped
        .range_mut(ghosts.clone())
        .as_mut_slice()
        .fill(u64::MAX);
    let (untouched, _) = exchange.forward_overlapped(&mut overlapped, |store, flight| {
        let untouched = store
            .range(ghosts.clone())
            .as_slice()
            .iter()
            .all(|&value| value == u64::MAX);
        for _ in 0..3 {
            flight.test();
        }
        untouched
    });
    assert!(
        untouched,
        "{name}: the ghost tail is written after the wait"
    );
    assert!(
        overlapped == sources,
        "{name}: the non-blocking source exchange differs from the blocking one"
    );
    check_traffic(name, comm, exchange.traffic(), exchange.communicator());
    Traffic {
        keys: exchange.communicator().total_receive_count(),
        values: exchange.communicator().receive_buffer_len(),
    }
}

/// A multipole store of `plan` with `sizes[l]` values on level l, the boxes this rank
/// computes ([`computes`]) seeded from their keys with [`ghost_value`], every other zero.
fn seeded_multipoles<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    held: &HashMap<u64, KeyType>,
    plan: &Plan,
    sizes: &[usize],
) -> LevelBuffers<u64> {
    let index = plan.index();
    let mut multipoles = LevelBuffers::<u64>::from_index(index, sizes);
    for level in 0..index.nlevels() {
        for i in 0..index.len(level) {
            let key = index.key(level, i);
            if computes(octree, held, key) {
                for (k, value) in multipoles.chunk_mut(level, i).iter_mut().enumerate() {
                    *value = ghost_value(key, k);
                }
            }
        }
    }
    multipoles
}

/// The keys of `boxes` on `level`.
fn keys_of(plan: &Plan, level: usize, boxes: &[u32]) -> Vec<u64> {
    boxes
        .iter()
        .map(|&i| plan.index().key(level, i as usize))
        .collect()
}

/// Check that `multipoles` holds its owner's values exactly in the slots this rank
/// computes and in `received[l]` (keys, per level), and zero elsewhere.
fn check_slots<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    held: &HashMap<u64, KeyType>,
    plan: &Plan,
    multipoles: &LevelBuffers<u64>,
    received: &[Vec<u64>],
) {
    let index = plan.index();
    for (level, received) in received.iter().enumerate().take(index.nlevels()) {
        for i in 0..index.len(level) {
            let key = index.key(level, i);
            let chunk = multipoles.chunk(level, i);
            if computes(octree, held, key) || received.binary_search(&key).is_ok() {
                for (k, &value) in chunk.iter().enumerate() {
                    assert_eq!(value, ghost_value(key, k), "{name}: multipole of {key}");
                }
            } else {
                assert!(
                    chunk.iter().all(|&v| v == 0),
                    "{name}: {key} written but not received"
                );
            }
        }
    }
}

/// Check the upward reduction (Phase 5S T5; docs/design/scale-out.md §4.2): its levels are
/// 1 to the deepest block level; on level l it receives exactly the children of the
/// `Global` boxes this rank owns that other ranks compute, and sends only boxes this rank
/// computes; forwarded deepest first it writes their owners' values into those slots and
/// nothing else; the non-blocking reduction (sends posted level by level, `test` calls,
/// waits deepest first, on a second exchange) delivers the same bits; and messages and
/// values sent equal those received over all ranks. Returns the reduction and the
/// traffic (keys and values received). Collective.
fn check_reduction<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    held: &HashMap<u64, KeyType>,
    plan: &Plan,
    sizes: &[usize],
) -> (ReductionExchange<u64>, Traffic) {
    let comm = octree.comm();
    let index = plan.index();
    let mut exchange = ReductionExchange::<u64>::new(plan, comm, sizes)
        .unwrap_or_else(|error| panic!("{name}: reduction: {error}"));
    let deepest = octree
        .partition_map()
        .blocks()
        .iter()
        .map(|&block| morton::level(block))
        .max()
        .unwrap();
    assert_eq!(
        exchange.levels(),
        1..deepest + 1,
        "{name}: reduction levels"
    );

    // The children of the owned `Global` boxes that other ranks compute, by level.
    let mut expected = vec![Vec::new(); index.nlevels()];
    for (&key, &kind) in held {
        if kind == KeyType::Global && computes(octree, held, key) {
            for child in morton::children(key).unwrap() {
                if !computes(octree, held, child) {
                    expected[morton::level(child)].push(child);
                }
            }
        }
    }
    let mut traffic = Traffic::default();
    for (level, expected) in expected.iter_mut().enumerate() {
        expected.sort_unstable();
        if !exchange.levels().contains(&level) {
            assert!(expected.is_empty(), "{name}: a child on level {level}");
            continue;
        }
        let received = exchange.receive_boxes(level);
        assert!(received.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
        assert_eq!(
            &keys_of(plan, level, received),
            expected,
            "{name}: reduction receives of level {level}"
        );
        for key in keys_of(plan, level, exchange.send_boxes(level)) {
            assert!(
                computes(octree, held, key),
                "{name}: the reduction sends {key}, which another rank computes"
            );
        }
        let communicator = exchange.communicator(level);
        traffic.keys += communicator.total_receive_count();
        traffic.values += communicator.receive_buffer_len();
    }

    let seeded = seeded_multipoles(octree, held, plan, sizes);
    let mut multipoles = seeded.clone();
    for level in exchange.levels().rev() {
        exchange.forward(level, &mut multipoles);
    }
    check_slots(name, octree, held, plan, &multipoles, &expected);

    let mut second = ReductionExchange::<u64>::new(plan, comm, sizes)
        .unwrap_or_else(|error| panic!("{name}: reduction: {error}"));
    let mut overlapped = seeded;
    let levels = second.levels();
    let (current, _) = second.forward_all_overlapped(&mut overlapped, |flight, multipoles| {
        let mut current = true;
        for level in levels.rev() {
            flight.send(level, multipoles);
            flight.test();
            flight.wait(level, multipoles);
            current &= flight.exchange().receive_buffer(level) == exchange.receive_buffer(level);
        }
        current
    });
    assert!(current, "{name}: a reduction receive buffer after its wait");
    assert!(
        overlapped == multipoles,
        "{name}: the non-blocking reduction differs from the blocking one"
    );
    for level in exchange.levels() {
        check_traffic(
            name,
            comm,
            second.traffic(level),
            second.communicator(level),
        );
    }
    (exchange, traffic)
}

/// The top-tree keys each rank reads but neither computes nor receives through the
/// reduction (the readers of the replicated top), by key, from the oracle's V- and
/// W-lists.
fn top_needs<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    held: &HashMap<u64, KeyType>,
    oracle: &OracleLists,
    plan: &Plan,
    reduction: &ReductionExchange<u64>,
) -> Vec<u64> {
    let map = octree.partition_map();
    let index = plan.index();
    let mut needs: Vec<u64> = oracle
        .values()
        .flat_map(|[_, v, w, _]| v.iter().chain(w))
        .copied()
        .filter(|&key| map.top_kind(key).is_some() && !computes(octree, held, key))
        .filter(|&key| {
            let (level, i) = index.find(key).unwrap();
            !reduction.receives(level, i)
        })
        .collect();
    needs.sort_unstable();
    needs.dedup();
    needs
}

/// Check the replicated top (Phase 5S T5; docs/design/scale-out.md §4.3): R is the same on
/// every rank; with every rank's readers counted by brute force (each rank's
/// [`top_needs`], gathered), it holds exactly the top-tree keys with at least
/// max(2, ⌈P/16⌉) readers (the tests stay far below the cap), in rank order, each rank's
/// keys its own by (level, key); the gather sends this rank's keys, writes exactly the
/// keys of R this rank holds but neither computes nor receives through the reduction, and
/// delivers every owner's values to every rank. Returns the exchange and its size (keys, values). Collective.
fn check_top<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    held: &HashMap<u64, KeyType>,
    oracle: &OracleLists,
    plan: &Plan,
    sizes: &[usize],
    reduction: &ReductionExchange<u64>,
) -> (TopExchange<u64>, Traffic) {
    let comm = octree.comm();
    let map = octree.partition_map();
    let nranks = comm.size() as usize;
    let rank = comm.rank() as usize;
    let index = plan.index();
    let mut top = TopExchange::<u64>::new(plan, comm, sizes, reduction)
        .unwrap_or_else(|error| panic!("{name}: replicated top: {error}"));

    let (shortest, longest) = extremes(comm, top.keys().len());
    let hash = top.keys().iter().fold(OFFSET_BASIS, |hash, &key| {
        (hash ^ key).wrapping_mul(0x0100_0000_01b3)
    });
    let mut hashes = [0u64; 2];
    comm.all_reduce_into(&hash, &mut hashes[0], SystemOperation::min());
    comm.all_reduce_into(&hash, &mut hashes[1], SystemOperation::max());
    assert!(
        shortest == longest && hashes[0] == hashes[1],
        "{name}: R differs between ranks"
    );

    let mut readers = gather_to_all(&top_needs(octree, held, oracle, plan, reduction), comm);
    readers.sort_unstable();
    let threshold = nranks.div_ceil(TOP_READER_FRACTION).max(2);
    let mut expected = Vec::new();
    for group in readers.chunk_by(|a, b| a == b) {
        if group.len() >= threshold {
            expected.push(group[0]);
        }
    }
    assert!(expected.len() <= TOP_MAX_BOXES, "{name}: the cap binds");
    assert_eq!(top.threshold(), threshold, "{name}: threshold");
    let mut keys = top.keys().to_vec();
    keys.sort_unstable();
    assert_eq!(keys, expected, "{name}: the keys of R");
    for r in 0..nranks {
        let own = &top.keys()[top.rank_keys(r)];
        assert!(own.iter().all(|&key| map.owner(key) == r), "{name}");
        assert!(
            own.windows(2)
                .all(|pair| (morton::level(pair[0]), pair[0]) < (morton::level(pair[1]), pair[1])),
            "{name}: R by (level, key)"
        );
    }
    assert_eq!(top.sent_keys(), top.rank_keys(rank), "{name}: sent keys");
    let received: Vec<usize> = (0..top.keys().len())
        .filter(|&r| {
            let key = top.keys()[r];
            index.find(key).is_some_and(|(level, i)| {
                !computes(octree, held, key) && !reduction.receives(level, i)
            })
        })
        .collect();
    assert_eq!(top.received_keys(), received, "{name}: received keys");
    for (r, &key) in top.keys().iter().enumerate() {
        assert_eq!(top.slot(r), index.find(key), "{name}: slot of {key} in R");
    }

    let mut multipoles = seeded_multipoles(octree, held, plan, sizes);
    top.gather(&mut multipoles);
    for (r, &key) in top.keys().iter().enumerate() {
        let expected: Vec<u64> = (0..sizes[morton::level(key)])
            .map(|k| ghost_value(key, k))
            .collect();
        assert_eq!(top.chunk(r), expected, "{name}: gathered {key}");
    }
    let mut written = vec![Vec::new(); index.nlevels()];
    for &r in top.received_keys() {
        let key = top.keys()[r];
        written[morton::level(key)].push(key);
    }
    for level in &mut written {
        level.sort_unstable();
    }
    check_slots(name, octree, held, plan, &multipoles, &written);
    let size = Traffic {
        keys: top.keys().len(),
        values: top.gathered().len(),
    };
    (top, size)
}

/// Forward the multipoles of every box this rank computes with `sizes[l]` values on
/// level l, and check that the boxes of the oracle's V- and W-lists that other ranks
/// compute, without those the reduction and the replicated top deliver, receive their
/// owner's values, and nothing else; that every box sent is one this rank computes; and
/// that every multipole a list of this rank reads, and every child its owned `Global`
/// boxes need, reaches it exactly once over the three exchanges. The forwarding tree
/// (Phase 5S T8s), at its default and at its narrowest ([`narrow_tree`]), blocking and
/// non-blocking, receives the same boxes as the direct exchange with the same bits in every
/// slot and receive buffer, and sends no box to more ranks than its fan-out. Returns the
/// traffic of the direct exchange and the chunks the narrow tree forwarded, summed over the
/// ranks.
fn check_multipole_exchange<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    all_keys: &HashMap<u64, KeyType>,
    oracle: &OracleLists,
    plan: &Plan,
    sizes: &[usize],
    (reduction, top): (&ReductionExchange<u64>, &TopExchange<u64>),
) -> (Traffic, usize) {
    let comm = octree.comm();
    let index = plan.index();
    let direct = (MultipoleRouting::Direct, 0);
    let build = |(routing, node): (MultipoleRouting, usize)| {
        MultipoleExchange::<u64>::new(plan, comm, sizes, reduction, top, routing, node)
            .unwrap_or_else(|error| panic!("{name}: multipole exchange {routing:?}: {error}"))
    };
    let mut exchange = build(direct);
    let mut multipoles = seeded_multipoles(octree, all_keys, plan, sizes);
    let mut overlapped = multipoles.clone();
    exchange.forward_all(&mut multipoles);

    // The non-blocking exchange (Phase 5 T9), on an exchange of its own whose receive
    // buffers start zeroed, with the levels waited for deepest first and `test` calls in
    // between, delivers the same bits into the slots and the receive buffers; a level's
    // receive buffer is current once it is waited for.
    let nlevels = index.nlevels();
    let mut second = build(direct);
    let (current, _) = second.forward_all_overlapped(&mut overlapped, |flight, multipoles| {
        let mut current = true;
        for level in (0..nlevels).rev() {
            flight.test();
            flight.wait(level, multipoles);
            current &= flight.exchange().receive_buffer(level) == exchange.receive_buffer(level);
        }
        current
    });
    assert!(current, "{name}: a receive buffer after its wait");
    assert!(
        overlapped == multipoles,
        "{name}: the non-blocking multipole exchange differs from the blocking one"
    );
    for level in 0..nlevels {
        assert_eq!(
            second.receive_buffer(level),
            exchange.receive_buffer(level),
            "{name}: receive buffer of level {level}"
        );
        check_multipole_traffic(name, comm, &second, level);
    }

    // The forwarding tree, slot by slot against the direct exchange: the same boxes
    // received, the same bits in every slot and receive buffer, blocking and overlapped
    // (the levels waited for in order, as the evaluator does).
    let mut forwarded = 0usize;
    for (label, routing) in [
        ("tree", default_routing(comm)),
        ("narrow tree", narrow_tree(comm)),
    ] {
        let name = &format!("{name}, {label}");
        let mut tree = build(routing);
        let mut blocking = seeded_multipoles(octree, all_keys, plan, sizes);
        tree.forward_all(&mut blocking);
        assert!(
            blocking == multipoles,
            "{name}: the stores differ from the direct exchange's"
        );
        for level in 0..nlevels {
            assert_eq!(
                tree.receive_boxes(level),
                exchange.receive_boxes(level),
                "{name}: received boxes of level {level}"
            );
            assert_eq!(
                tree.receive_buffer(level),
                exchange.receive_buffer(level),
                "{name}: receive buffer of level {level}"
            );
            for key in keys_of(plan, level, tree.send_boxes(level)) {
                assert!(computes(octree, all_keys, key), "{name}: sends {key}");
            }
            check_multipole_traffic(name, comm, &tree, level);
            forwarded += tree.forwarded(level);
        }
        let mut tree = build(routing);
        let mut overlapped = seeded_multipoles(octree, all_keys, plan, sizes);
        tree.forward_all_overlapped(&mut overlapped, |flight, multipoles| {
            for level in 0..nlevels {
                flight.test();
                flight.wait(level, multipoles);
            }
        });
        assert!(
            overlapped == multipoles,
            "{name}: the non-blocking tree differs from the direct exchange"
        );
        for level in 0..nlevels {
            assert_eq!(
                tree.receive_buffer(level),
                exchange.receive_buffer(level),
                "{name}: non-blocking receive buffer of level {level}"
            );
        }
    }
    let mut total_forwarded = 0usize;
    comm.all_reduce_into(&forwarded, &mut total_forwarded, SystemOperation::sum());

    // The boxes of the V- and W-lists that other ranks compute, by level; those the
    // multipole exchange receives are the rest after the reduction and the top.
    let mut needed = vec![Vec::new(); nlevels];
    for &entry in oracle.values().flat_map(|[_, v, w, _]| v.iter().chain(w)) {
        if !computes(octree, all_keys, entry) {
            needed[morton::level(entry)].push(entry);
        }
    }
    let mut expected = vec![Vec::new(); nlevels];
    for (level, needed) in needed.iter_mut().enumerate() {
        needed.sort_unstable();
        needed.dedup();
        expected[level] = needed
            .iter()
            .copied()
            .filter(|&key| {
                let (_, i) = index.find(key).unwrap();
                !reduction.receives(level, i) && !top.replicates(key)
            })
            .collect();
    }

    let mut traffic = Traffic::default();
    for (level, expected) in expected.iter().enumerate() {
        let received = exchange.receive_boxes(level);
        assert!(received.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
        assert_eq!(
            &keys_of(plan, level, received),
            expected,
            "{name}: received multipoles of level {level}"
        );
        for key in keys_of(plan, level, exchange.send_boxes(level)) {
            assert!(
                computes(octree, all_keys, key),
                "{name}: the multipole exchange sends {key}, which another rank computes"
            );
        }
        traffic.keys += exchange.receive_boxes(level).len();
        traffic.values += exchange.receive_buffer(level).len();
    }
    check_slots(name, octree, all_keys, plan, &multipoles, &expected);

    // Every multipole a list reads, and every child of an owned `Global` box, once.
    let mut delivered = vec![Vec::new(); nlevels];
    for (level, delivered) in delivered.iter_mut().enumerate() {
        if reduction.levels().contains(&level) {
            delivered.extend(keys_of(plan, level, reduction.receive_boxes(level)));
        }
        delivered.extend(keys_of(plan, level, exchange.receive_boxes(level)));
    }
    for &r in top.received_keys() {
        let key = top.keys()[r];
        delivered[morton::level(key)].push(key);
    }
    for (level, delivered) in delivered.iter_mut().enumerate() {
        delivered.sort_unstable();
        let unique = delivered.len();
        delivered.dedup();
        assert_eq!(
            unique,
            delivered.len(),
            "{name}: a multipole received twice"
        );
        for key in &needed[level] {
            assert!(
                delivered.binary_search(key).is_ok(),
                "{name}: {key} is read but never received"
            );
        }
    }
    (traffic, total_forwarded)
}

/// A target count per leaf that every rank can recompute from the key, independent of
/// [`hashed_count`]; zero for about one key in five.
fn hashed_target_count(key: u64) -> usize {
    hashed_count(key ^ 0x5bd1_e995)
}

/// A per-pair test operator whose values do not matter. Every call mixes into each output
/// value a hash of all its inputs and of the keys of the pair, so the values depend on
/// every input, received sources and multipoles included, and a stale or doubled value
/// shows; the mixing does not commute, so a contribution in another order shows too
/// (Phase 5S T5). It checks that an evaluation repeats bit for bit, and that several
/// ranks give a value the bits of one rank, not that it is right.
#[derive(Clone, Copy, Debug)]
struct Mixer;

/// The start of [`mix`]'s hash (FNV-1a), so that the root key and empty inputs still
/// give a nonzero hash.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// Mix a hash of `inputs` and `keys` into every value of `output`: rotate, then add, so
/// that two contributions in the other order give other bits.
fn mix(inputs: [&[u32]; 2], keys: [u64; 2], output: &mut [u32]) {
    let hash = inputs.iter().flat_map(|input| input.iter()).fold(
        OFFSET_BASIS ^ keys[0] ^ keys[1].rotate_left(32),
        |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x0100_0000_01b3),
    );
    for (k, value) in output.iter_mut().enumerate() {
        *value = value.rotate_left(7).wrapping_add((hash >> (k % 32)) as u32);
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

    // The overlapped evaluation (Phase 5 T9) gives the same stores bit for bit, after a
    // blocking one, after itself, and followed by a blocking one.
    borrowing.evaluate_overlapped();
    assert!(
        stores(&borrowing) == first,
        "{name}: the overlapped evaluation differs from the blocking one"
    );
    owning.evaluate_overlapped();
    owning.evaluate_overlapped();
    assert!(
        stores(&owning) == first,
        "{name}: two overlapped evaluations differ"
    );
    owning.evaluate();
    assert!(
        stores(&owning) == first,
        "{name}: a blocking evaluation after an overlapped one differs"
    );

    // Every routing of the multipole exchange gives the stores of the default bit for bit
    // (Phase 5S T8s), blocking and overlapped.
    for (label, (routing, node)) in [
        ("direct", (MultipoleRouting::Direct, 0)),
        ("narrow tree", narrow_tree(comm)),
    ] {
        let mut other = Evaluator::with_routing(
            plan,
            comm,
            PerPair(Mixer),
            &sources,
            &targets,
            routing,
            node,
        )
        .unwrap_or_else(|error| panic!("{name}, {label}: evaluator: {error}"));
        fill_by_key(plan, other.local_sources_mut(), 0);
        fill_by_key(plan, other.local_target_inputs_mut(), 1);
        other.evaluate();
        assert!(
            stores(&other) == first,
            "{name}, {label}: the stores differ"
        );
        other.evaluate_overlapped();
        assert!(
            stores(&other) == first,
            "{name}, {label}: the overlapped stores differ"
        );
    }
}

/// The stores of a [`Mixer`] evaluation of `plan` on `comm` (hashed counts, data by key,
/// as [`check_repeated_evaluation`]), and per level the boxes whose multipoles the
/// evaluation received: through the reduction, the replicated top and the multipole
/// exchange. Collective.
fn mixed<C: CommunicatorCollectives>(plan: &Plan, comm: &C) -> (Stores, Vec<Vec<u32>>) {
    let leaves = plan.index().leaves();
    let sources: Vec<usize> = (0..leaves.nlocal())
        .map(|j| hashed_count(leaves.key(j)))
        .collect();
    let targets: Vec<usize> = (0..leaves.nlocal())
        .map(|j| hashed_target_count(leaves.key(j)))
        .collect();
    let mut evaluator = Evaluator::new(plan, comm, PerPair(Mixer), &sources, &targets)
        .unwrap_or_else(|error| panic!("evaluator: {error}"));
    fill_by_key(plan, evaluator.local_sources_mut(), 0);
    fill_by_key(plan, evaluator.local_target_inputs_mut(), 1);
    evaluator.evaluate();
    let mut received = vec![Vec::new(); plan.nlevels()];
    let reduction = evaluator.reduction_exchange();
    for level in reduction.levels() {
        received[level].extend_from_slice(reduction.receive_boxes(level));
    }
    let top = evaluator.top_exchange();
    for &r in top.received_keys() {
        let (level, i) = top.slot(r).unwrap();
        received[level].push(i);
    }
    let multipoles = evaluator.multipole_exchange();
    for (level, received) in received.iter_mut().enumerate() {
        received.extend_from_slice(multipoles.receive_boxes(level));
        received.sort_unstable();
    }
    (stores(&evaluator), received)
}

/// Evaluate the order-sensitive [`Mixer`] on `plan` and on the plan of the one-rank tree
/// over every rank's `fine_keys` (built on `self_comm`), with the same counts and data
/// by key, and check by key (Phase 5S T5; docs/design/scale-out.md §9.1): every multipole
/// this rank computes or receives, the shared ancestors' among them, the local of every
/// box it holds but a ghost, and the output of every local leaf are bit for bit the
/// one-rank values; every other multipole is zero. The leaves of the two trees agree.
/// The verdict is agreed on every rank (one all-reduce), so the check fails on every rank
/// or on none. Returns the number of `Global` multipoles this rank formed. Collective.
fn check_against_one_rank<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    plan: &Plan,
    fine_keys: &[u64],
    options: OctreeOptions,
) -> usize {
    let comm = octree.comm();
    let rank = comm.rank();
    let all_keys = gather_to_all(fine_keys, comm);
    let mut leaves = gather_to_all(octree.leaf_keys(), comm);
    leaves.sort_unstable();
    let single_comm = SimpleCommunicator::self_comm();
    let single = Octree::new(&all_keys, options, &single_comm);
    assert_eq!(single.leaf_keys(), &leaves, "{name}: the one-rank tree");
    let one = Plan::new(&single).unwrap_or_else(|error| panic!("{name}: one rank: {error}"));

    let ((outputs, multipoles, locals, _), received) = mixed(plan, comm);
    let ((one_outputs, one_multipoles, one_locals, _), _) = mixed(&one, &single_comm);
    let index = plan.index();
    let one_index = one.index();
    // The defects of this rank (the first few named), agreed on every rank below, so that
    // the check fails on every rank or on none.
    let mut defects = 0usize;
    let mut named = Vec::new();
    let mut defect = |what: String| {
        defects += 1;
        if named.len() < 4 {
            named.push(what);
        }
    };
    let mut formed = 0;
    for (level, received) in received.iter().enumerate() {
        for i in 0..index.len(level) {
            let key = index.key(level, i);
            let kind = index.kind(level, i);
            let (one_level, j) = one_index.find(key).expect("the same tree");
            let had = plan.computes(level, i) || received.binary_search(&(i as u32)).is_ok();
            if had {
                if multipoles.chunk(level, i) != one_multipoles.chunk(one_level, j as usize) {
                    defect(format!("multipole of {key} ({kind:?})"));
                }
                formed += usize::from(kind == KeyType::Global && plan.computes(level, i));
            } else if multipoles.chunk(level, i).iter().any(|&v| v != 0) {
                defect(format!("multipole of {key} ({kind:?}) written"));
            }
            if !kind.is_ghost() && locals.chunk(level, i) != one_locals.chunk(one_level, j as usize)
            {
                defect(format!("local of {key} ({kind:?})"));
            }
        }
    }
    let leaves = index.leaves();
    let one_leaves = one_index.leaves();
    for j in 0..leaves.nlocal() {
        let key = leaves.key(j);
        let k = one_index.find_leaf(key).expect("the same leaves") as usize;
        if k >= one_leaves.nlocal() || outputs.chunk(j) != one_outputs.chunk(k) {
            defect(format!("output of leaf {key}"));
        }
    }
    let mut total = 0usize;
    comm.all_reduce_into(&defects, &mut total, SystemOperation::sum());
    assert!(
        total == 0,
        "rank {rank}: {name}: {total} values differ from one rank's over all ranks, here \
         {defects}: {named:?}"
    );
    formed
}

/// A point in [0, 1)³ from `i` and `k`, the same on every rank.
fn hashed_unit(i: usize, k: u64) -> f64 {
    let hash = (i as u64 ^ k.wrapping_mul(0x5851_f42d_4c95_7f2d))
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(29)
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

/// `n` points of the unit cube, uniform, the i-th from its index.
fn cube(range: Range<usize>) -> Vec<[f64; 3]> {
    range
        .map(|i| [hashed_unit(i, 1), hashed_unit(i, 2), hashed_unit(i, 3)])
        .collect()
}

/// Points of a Plummer sphere of scale 0.05 at the centre of the unit cube (radii cut at
/// 0.45), the i-th from its index.
fn plummer(range: Range<usize>) -> Vec<[f64; 3]> {
    range
        .map(|i| {
            let mass = hashed_unit(i, 4).max(1e-9);
            let radius = (0.05 / (mass.powf(-2.0 / 3.0) - 1.0).max(1e-12).sqrt()).min(0.45);
            let cos = 2.0 * hashed_unit(i, 5) - 1.0;
            let sin = (1.0 - cos * cos).sqrt();
            let phi = std::f64::consts::TAU * hashed_unit(i, 6);
            [
                0.5 + radius * sin * phi.cos(),
                0.5 + radius * sin * phi.sin(),
                0.5 + radius * cos,
            ]
        })
        .collect()
}

/// One level call: (method, level, pass of an M2M), or one host-data event: (event,
/// level of a `SendReduction`, `ReceivedReduction` or `ReceivedMultipoles`, else 0,
/// `None`).
type Call = (&'static str, usize, Option<UpwardPass>);

/// The index lists of one host-data event as the recorder saw them: (event, level, the
/// lists). `SendSources`: the leaves; `ReceivedSources`: the ghost leaves;
/// `SendReduction`, `ReceivedReduction`: the sent and the received boxes of the level;
/// `SendMultipoles`: the sent keys of the replicated top, then the sent boxes of every
/// level; `ReceivedTop`: the received keys of the top; `ReceivedMultipoles`: the received
/// boxes.
type EventLists = (&'static str, usize, Vec<Vec<usize>>);

/// A test operator that computes nothing. It checks every batch it receives (the
/// groupings of design §4.4, both views, buffer shapes) and records every call and every
/// pair, by key, and every host-data event with its index lists
/// (`docs/design/distributed-fmm.md` §6).
struct Recorder<'p> {
    plan: &'p Plan,
    calls: Vec<Call>,
    /// (method, target key, source key) of every pair issued.
    pairs: Vec<(&'static str, u64, u64)>,
    /// The index lists of every host-data event, in order.
    events: Vec<EventLists>,
    /// The events whose stores or buffers had the wrong shape, or whose packed values or
    /// sent blocks were not as documented.
    shape_errors: Vec<&'static str>,
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

    fn host_data(&mut self, event: HostData<'_, u32>) {
        let nleaves = self.plan.index().leaves().len();
        let index = self.plan.index();
        let (name, level, lists, shaped) = match event {
            HostData::Reset => ("reset", 0, Vec::new(), true),
            HostData::SendSources { leaves, sources } => (
                "send sources",
                0,
                vec![leaves.iter().map(|&j| j as usize).collect()],
                sources.nleaves() == nleaves,
            ),
            HostData::ReceivedSources { leaves, sources } => (
                "received sources",
                0,
                vec![leaves.collect()],
                sources.nleaves() == nleaves,
            ),
            HostData::SendReduction {
                level,
                exchange,
                multipoles,
            } => {
                // The sent boxes are final and this rank's: local, or `Global` and owned.
                let boxes = exchange.send_boxes(level);
                let shaped = multipoles.nlevels() == index.nlevels()
                    && boxes.iter().all(|&i| self.plan.computes(level, i as usize));
                (
                    "send reduction",
                    level,
                    vec![boxes.iter().map(|&i| i as usize).collect()],
                    shaped,
                )
            }
            HostData::ReceivedReduction {
                level,
                exchange,
                multipoles,
            } => {
                let boxes = exchange.receive_boxes(level);
                let size = exchange.size(level);
                let buffer = exchange.receive_buffer(level);
                let shaped = buffer.len() == boxes.len() * size
                    && boxes
                        .iter()
                        .zip(buffer.chunks_exact(size))
                        .all(|(&i, chunk)| multipoles.chunk(level, i as usize) == chunk);
                (
                    "received reduction",
                    level,
                    vec![boxes.iter().map(|&i| i as usize).collect()],
                    shaped,
                )
            }
            HostData::SendMultipoles {
                top,
                exchange,
                multipoles,
            } => {
                let mut lists = vec![top.sent_keys().collect::<Vec<_>>()];
                lists.extend(
                    (0..exchange.nlevels())
                        .map(|l| exchange.send_boxes(l).iter().map(|&i| i as usize).collect()),
                );
                // The sent keys of the top are this rank's boxes, final after the
                // reduction: local, or `Global` and owned.
                let shaped = multipoles.nlevels() == index.nlevels()
                    && (0..index.nlevels()).all(|l| multipoles.len(l) == index.len(l))
                    && top.sent_keys().all(|r| {
                        top.slot(r)
                            .is_some_and(|(l, i)| self.plan.computes(l, i as usize))
                    });
                ("send multipoles", 0, lists, shaped)
            }
            HostData::ReceivedTop { top, multipoles } => {
                // The packed values are the slots' values.
                let shaped = top.received_keys().iter().all(|&r| {
                    top.slot(r)
                        .is_some_and(|(l, i)| top.chunk(r) == multipoles.chunk(l, i as usize))
                });
                (
                    "received top",
                    0,
                    vec![top.received_keys().to_vec()],
                    shaped,
                )
            }
            HostData::ReceivedMultipoles {
                level,
                exchange,
                multipoles,
            } => {
                let boxes = exchange.receive_boxes(level);
                let size = exchange.size(level);
                let buffer = exchange.receive_buffer(level);
                let shaped = buffer.len() == boxes.len() * size
                    && boxes
                        .iter()
                        .zip(buffer.chunks_exact(size))
                        .all(|(&i, chunk)| multipoles.chunk(level, i as usize) == chunk);
                (
                    "received multipoles",
                    level,
                    vec![boxes.iter().map(|&i| i as usize).collect()],
                    shaped,
                )
            }
        };
        if !shaped {
            self.shape_errors.push(name);
        }
        self.calls.push((name, level, None));
        self.events.push((name, level, lists));
    }
}

/// [`expected_calls`] or [`expected_overlapped_calls`].
type ExpectedCalls = fn(usize, Range<usize>) -> Vec<Call>;

/// The global M2M calls of one evaluation from level L − 1 down, each after the events
/// of the reduction of the level below it if that is a reduction level (Phase 5S T5).
fn global_pass(deepest: usize, reduction: &Range<usize>) -> Vec<Call> {
    let mut calls = Vec::new();
    for level in (0..deepest).rev() {
        if reduction.contains(&(level + 1)) {
            calls.extend([
                ("send reduction", level + 1, None),
                ("received reduction", level + 1, None),
            ]);
        }
        calls.push(("m2m", level, Some(UpwardPass::Global)));
    }
    calls
}

/// The level calls and host-data events of one evaluation, in the order of design §7.2
/// and of `docs/design/distributed-fmm.md` §6.2, with the reduction levels `reduction`.
fn expected_calls(nlevels: usize, reduction: Range<usize>) -> Vec<Call> {
    let deepest = nlevels - 1;
    let mut calls = vec![
        ("reset", 0, None),
        ("send sources", 0, None),
        ("received sources", 0, None),
    ];
    for level in (0..=deepest).rev() {
        calls.push(("p2m", level, None));
        if level > 0 {
            calls.push(("m2m", level - 1, Some(UpwardPass::Local)));
        }
    }
    calls.extend(global_pass(deepest, &reduction));
    calls.extend([("send multipoles", 0, None), ("received top", 0, None)]);
    for level in 0..=deepest {
        calls.push(("received multipoles", level, None));
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

/// The level calls and host-data events of one overlapped evaluation
/// (`Evaluator::evaluate_overlapped`, `docs/design/distributed-fmm.md` §6.5, §8.1): the
/// calls of [`expected_calls`] in the same order, `ReceivedSources` after the local upward
/// pass, and each `ReceivedMultipoles` of level l just before the downward calls of l.
fn expected_overlapped_calls(nlevels: usize, reduction: Range<usize>) -> Vec<Call> {
    let deepest = nlevels - 1;
    let mut calls = vec![("reset", 0, None), ("send sources", 0, None)];
    for level in (0..=deepest).rev() {
        calls.push(("p2m", level, None));
        if level > 0 {
            calls.push(("m2m", level - 1, Some(UpwardPass::Local)));
        }
    }
    calls.push(("received sources", 0, None));
    calls.extend(global_pass(deepest, &reduction));
    calls.extend([("send multipoles", 0, None), ("received top", 0, None)]);
    calls.push(("received multipoles", 0, None));
    for level in 1..=deepest {
        calls.extend([
            ("received multipoles", level, None),
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

/// The index lists of the host-data events of one evaluation with the exchanges
/// `sources`, `reduction`, `top` and `multipoles`, in order (see [`EventLists`]).
fn event_lists<T: Equivalence + Copy + Default>(
    nlevels: usize,
    sources: &SourceExchange<T>,
    reduction: &ReductionExchange<T>,
    top: &TopExchange<T>,
    multipoles: &MultipoleExchange<T>,
) -> Vec<EventLists> {
    let as_usize = |values: &[u32]| values.iter().map(|&v| v as usize).collect::<Vec<_>>();
    let mut events = vec![
        ("reset", 0, Vec::new()),
        ("send sources", 0, vec![as_usize(sources.send_leaves())]),
        (
            "received sources",
            0,
            vec![sources.ghost_leaves().collect()],
        ),
    ];
    for level in (1..nlevels).rev() {
        if reduction.levels().contains(&level) {
            events.push((
                "send reduction",
                level,
                vec![as_usize(reduction.send_boxes(level))],
            ));
            events.push((
                "received reduction",
                level,
                vec![as_usize(reduction.receive_boxes(level))],
            ));
        }
    }
    let mut sends = vec![top.sent_keys().collect::<Vec<usize>>()];
    sends.extend((0..nlevels).map(|l| as_usize(multipoles.send_boxes(l))));
    events.push(("send multipoles", 0, sends));
    events.push(("received top", 0, vec![top.received_keys().to_vec()]));
    events.extend((0..nlevels).map(|l| {
        (
            "received multipoles",
            l,
            vec![as_usize(multipoles.receive_boxes(l))],
        )
    }));
    events
}

/// The index lists the host-data events of one evaluation must carry, in order (see
/// [`EventLists`]), from exchanges built independently of the evaluator's, with the
/// recorder's sizes and `counts`. Collective.
fn expected_events<C: CommunicatorCollectives>(
    name: &str,
    comm: &C,
    plan: &Plan,
    counts: &[usize],
    (routing, node): (MultipoleRouting, usize),
) -> Vec<EventLists> {
    let nlevels = plan.nlevels();
    let sizes = vec![1; nlevels];
    let sources = SourceExchange::<u32>::new(plan, comm, counts, 1)
        .unwrap_or_else(|error| panic!("{name}: source exchange: {error}"));
    let reduction = ReductionExchange::<u32>::new(plan, comm, &sizes)
        .unwrap_or_else(|error| panic!("{name}: reduction: {error}"));
    let top = TopExchange::<u32>::new(plan, comm, &sizes, &reduction)
        .unwrap_or_else(|error| panic!("{name}: replicated top: {error}"));
    let multipoles =
        MultipoleExchange::<u32>::new(plan, comm, &sizes, &reduction, &top, routing, node)
            .unwrap_or_else(|error| panic!("{name}: multipole exchange: {error}"));
    event_lists(nlevels, &sources, &reduction, &top, &multipoles)
}

/// Check the recorded host-data events of `recorder` against `expected`, and the kinds of
/// what they name: every box sent is one this rank computes (a local leaf or interior
/// box, or a `Global` box it owns), every slot written one another rank computes.
/// Returns the defects found, so that the verdict can be agreed on every rank.
fn event_defects(recorder: &Recorder<'_>, expected: &[EventLists]) -> Vec<String> {
    let index = recorder.plan.index();
    let mut defects: Vec<String> = recorder
        .shape_errors
        .iter()
        .map(|event| format!("{event}: a store, a buffer or a sent block"))
        .collect();
    if recorder.events.len() != expected.len() {
        defects.push(format!(
            "{} events, expected {}",
            recorder.events.len(),
            expected.len()
        ));
    }
    for (got, want) in recorder.events.iter().zip(expected) {
        if got != want {
            defects.push(format!("{} (level {}): index lists differ", want.0, want.1));
        }
    }
    let plan = recorder.plan;
    for (event, level, lists) in &recorder.events {
        match *event {
            "send sources" if lists[0].iter().any(|&j| j >= index.leaves().nlocal()) => {
                defects.push("send sources: a leaf that is not local".into());
            }
            "send multipoles" => {
                for (l, boxes) in lists[1..].iter().enumerate() {
                    if boxes.iter().any(|&i| !plan.computes(l, i)) {
                        defects.push(format!("send multipoles: a box on level {l} not ours"));
                    }
                }
            }
            "send reduction" if lists[0].iter().any(|&i| !plan.computes(*level, i)) => {
                defects.push(format!("send reduction: a box on level {level} not ours"));
            }
            "received reduction" | "received multipoles"
                if lists[0].iter().any(|&i| plan.computes(*level, i)) =>
            {
                defects.push(format!("{event}: a box on level {level} of ours"));
            }
            _ => {}
        }
    }
    defects
}

/// Evaluate with a recording operator: every level gets every call once, in pass order;
/// every batch honours its grouping; and the pairs issued are exactly the pairs of the
/// oracle lists, each once, the global M2M pairs on the owner of the parent only. The host-data events (`docs/design/distributed-fmm.md` §6)
/// come in their place among the level calls, on every rank, with the index lists of the
/// exchanges; their verdict is agreed on every rank. With `overlapped`, through
/// `Evaluator::evaluate_overlapped` (Phase 5 T9): the same calls, groupings, pairs, events
/// and lists, the events in the overlapped places. The multipole exchange routes by
/// `routing` (Phase 5S T8s), with this rank's node.
fn check_batches<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    oracle: &OracleLists,
    plan: &Plan,
    overlapped: bool,
    routing: (MultipoleRouting, usize),
) {
    let (name, expected_calls): (&str, ExpectedCalls) = if overlapped {
        (&format!("{name}, overlapped"), expected_overlapped_calls)
    } else {
        (name, expected_calls)
    };
    let nlocal = plan.index().leaves().nlocal();
    let counts: Vec<usize> = (0..nlocal)
        .map(|j| hashed_count(plan.index().leaf_key(j)))
        .collect();
    let comm = octree.comm();
    let expected_events = expected_events(name, comm, plan, &counts, routing);
    let recorder = Recorder {
        plan,
        calls: Vec::new(),
        pairs: Vec::new(),
        events: Vec::new(),
        shape_errors: Vec::new(),
    };
    let mut evaluator =
        Evaluator::with_routing(plan, comm, recorder, &counts, &counts, routing.0, routing.1)
            .unwrap_or_else(|error| panic!("{name}: evaluator: {error}"));
    if overlapped {
        evaluator.evaluate_overlapped();
    } else {
        evaluator.evaluate();
    }
    let recorder = evaluator.operator();
    let calls = expected_calls(plan.nlevels(), plan.reduction_levels());

    // The events, agreed: every rank receives each of them, in order, with its lists.
    let mut defects = event_defects(recorder, &expected_events);
    if recorder.calls != calls {
        defects.push("the order of the calls and events".into());
    }
    // The evaluator's exchanges, read through its accessors (Phase 5 T8), name the lists
    // its events carried.
    let accessed = event_lists(
        plan.nlevels(),
        evaluator.source_exchange(),
        evaluator.reduction_exchange(),
        evaluator.top_exchange(),
        evaluator.multipole_exchange(),
    );
    let sources = evaluator.source_exchange();
    if accessed != recorder.events || sources.leaf_counts()[..nlocal] != counts[..] {
        defects.push("the exchanges' accessors and the events' lists differ".into());
    }
    let mut per_rank = [0usize; 2];
    comm.all_reduce_into(
        &recorder.events.len(),
        &mut per_rank[0],
        SystemOperation::min(),
    );
    comm.all_reduce_into(
        &recorder.events.len(),
        &mut per_rank[1],
        SystemOperation::max(),
    );
    let events = 5 + plan.nlevels() + 2 * plan.reduction_levels().len();
    if per_rank != [events; 2] {
        defects.push(format!(
            "events per rank between {} and {}, expected {events}",
            per_rank[0], per_rank[1]
        ));
    }
    let mut failed = 0usize;
    comm.all_reduce_into(&defects.len(), &mut failed, SystemOperation::sum());
    assert!(
        failed == 0,
        "rank {}: {name}: {failed} host-data defects on all ranks, here: {defects:?}",
        comm.rank()
    );
    assert_eq!(recorder.calls, calls, "{name}");

    let held = held_keys(plan);
    let mut expected = Vec::new();
    for (&key, &kind) in &held {
        if kind.is_ghost() {
            continue;
        }
        let entries = |method, sources: &[u64]| pairs_of(method, key, sources);
        let [u, v, w, x] = &oracle[&key];
        expected.extend(entries("m2l", v));
        expected.extend(entries("p2l", x));
        if let Some(parent) = morton::parent(key)
            && needs_local(&held, octree.coarse_tree_leafs(), key)
        {
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
            // Only the owner forms a `Global` box (Phase 5S T5).
            KeyType::Global if computes(octree, &held, key) => {
                expected.extend(entries("m2m global", &children))
            }
            KeyType::Global => {}
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

/// The input distributions of docs/phase5/README.md ("Workloads"), each derived on every
/// rank from the scenario's global items (docs/design/distributed-fmm.md §10.2).
#[derive(Clone, Copy, Debug)]
enum Distribution {
    /// Rank 0 holds every item.
    RankZero,
    /// Item i goes to rank `hash(seed, i) % P`.
    Share,
    /// Every item is on the rank that owns its leaf.
    Owners,
    /// The share, with the last rank's items given to rank 0.
    OneEmptyRank,
}

const DISTRIBUTIONS: [Distribution; 4] = [
    Distribution::RankZero,
    Distribution::Share,
    Distribution::Owners,
    Distribution::OneEmptyRank,
];

/// The rank of item `i` in the seeded share.
fn share_rank(i: usize, nranks: usize) -> usize {
    ((i as u64 ^ 0x2545_f491_4f6c_dd1d).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize % nranks
}

/// For every rank, the indices into `global` of the items it holds, ascending.
fn inputs<C: CommunicatorCollectives>(
    distribution: Distribution,
    global: &[u64],
    octree: &Octree<'_, C>,
) -> Vec<Vec<usize>> {
    let nranks = octree.comm().size() as usize;
    let mut inputs = vec![Vec::new(); nranks];
    for (i, &key) in global.iter().enumerate() {
        let rank = match distribution {
            Distribution::RankZero => 0,
            Distribution::Share => share_rank(i, nranks),
            Distribution::Owners => octree.owner_rank(key).unwrap(),
            Distribution::OneEmptyRank => match share_rank(i, nranks) {
                last if last + 1 == nranks => 0,
                rank => rank,
            },
        };
        inputs[rank].push(i);
    }
    inputs
}

/// The minimum and the maximum of `value` over the ranks.
fn extremes<C: CommunicatorCollectives>(comm: &C, value: usize) -> (usize, usize) {
    let mut min = 0usize;
    let mut max = 0usize;
    comm.all_reduce_into(&value, &mut min, SystemOperation::min());
    comm.all_reduce_into(&value, &mut max, SystemOperation::max());
    (min, max)
}

/// The bits of `values`, for bit-for-bit comparisons.
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Redistribute the items `global` (every rank's keys of one population, in rank order)
/// from every input distribution, and check that every item arrives exactly once, on the
/// rank that owns its leaf, in that leaf, in (origin rank, position) order; that the
/// counts are those the plan's leaves imply; that payloads which name their origin
/// arrive with it and round-trip through `backward`, for `f64` with three values per
/// item and `u32` with one; that two forwards agree bit for bit; and that `forward`, point
/// to point on its fixed routes since Phase 5S T8 (S5), gives the bits of the dense
/// all-to-all-v of the same items (rlst's `all_to_allv`, ordered by leaf and origin).
/// Collective.
fn check_redistribution<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    plan: &Plan,
    population: &str,
    global: &[u64],
) {
    let comm = octree.comm();
    let rank = comm.rank() as usize;
    let leaves = plan.index().leaves();
    let mut sorted = global.to_vec();
    sorted.sort_unstable();
    let expected_counts: Vec<usize> = (0..leaves.nlocal())
        .map(|j| points_in(&sorted, leaves.key(j)))
        .collect();

    for distribution in DISTRIBUTIONS {
        let label = format!("{name}, {population}, {distribution:?}");
        let inputs = inputs(distribution, global, octree);
        let mine: Vec<u64> = inputs[rank].iter().map(|&i| global[i]).collect();
        let redistribution = Redistribution::new(octree, plan, &mine)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(redistribution.nsent(), mine.len(), "{label}");
        let (min, max) = extremes(comm, redistribution.max_per_item());
        assert_eq!(min, max, "{label}: max_per_item differs between ranks");
        assert!(min >= 3, "{label}: max_per_item {min}");

        // The counts: those of the plan's leaves, summing to this rank's items and, over
        // the ranks, to every item.
        assert_eq!(redistribution.counts(), expected_counts, "{label}: counts");
        let nreceived = redistribution.nreceived();
        assert_eq!(redistribution.counts().iter().sum::<usize>(), nreceived);
        let mut total = 0usize;
        comm.all_reduce_into(&nreceived, &mut total, SystemOperation::sum());
        assert_eq!(total, global.len(), "{label}: items received");

        // Payloads that name their origin (rank, position).
        let named = |r: usize, p: usize| [r as f64, p as f64, (r * 1_000_000 + p) as f64];
        let tag = |r: usize, p: usize| ((r as u32) << 20) | p as u32;
        let f64_payload: Vec<f64> = (0..mine.len()).flat_map(|p| named(rank, p)).collect();
        let u32_payload: Vec<u32> = (0..mine.len()).map(|p| tag(rank, p)).collect();
        let forwarded = redistribution.forward(&f64_payload, 3);
        let forwarded_u32 = redistribution.forward(&u32_payload, 1);
        assert_eq!(forwarded.len(), 3 * nreceived, "{label}");
        let again = redistribution.forward(&f64_payload, 3);
        assert_eq!(bits(&again), bits(&forwarded), "{label}: two forwards");
        let mut into = vec![0.0; forwarded.len()];
        redistribution.forward_into(&f64_payload, 3, &mut into);
        assert_eq!(bits(&into), bits(&forwarded), "{label}: forward_into");

        // Every received item: on the owner of its leaf, in that leaf, after the items of
        // lower (origin rank, position) in the leaf, with its origin's payload.
        let mut start = 0;
        for (j, &count) in redistribution.counts().iter().enumerate() {
            let leaf = leaves.key(j);
            let origins = &redistribution.origins()[start..start + count];
            assert!(
                origins.windows(2).all(|pair| pair[0] < pair[1]),
                "{label}: order within leaf {leaf}"
            );
            for (r, &(origin, position)) in (start..start + count).zip(origins) {
                let (o, p) = (origin as usize, position as usize);
                assert_eq!(
                    bits(&forwarded[3 * r..3 * r + 3]),
                    bits(&named(o, p)),
                    "{label}: payload of received item {r}"
                );
                assert_eq!(forwarded_u32[r], tag(o, p), "{label}: u32 payload");
                let key = global[inputs[o][p]];
                assert_eq!(octree.owner_rank(key), Ok(rank), "{label}: owner of {key}");
                assert_eq!(
                    octree.local_leaf(key),
                    Ok(Some(leaf)),
                    "{label}: leaf of {key}"
                );
                assert_eq!(
                    plan.index().local_leaf_containing(key),
                    Some(j as u32),
                    "{label}: leaf index of {key}"
                );
            }
            start += count;
        }

        // The dense oracle (S5): the same payloads moved to the owners by an all-to-all-v
        // of P counts and ordered by (leaf, origin rank, position) are forward's bits.
        let nranks = comm.size() as usize;
        let mut by_owner = vec![Vec::new(); nranks];
        for (p, &key) in mine.iter().enumerate() {
            let owner = octree.owner_rank(key).expect("a valid key");
            by_owner[owner].extend_from_slice(&f64_payload[3 * p..3 * p + 3]);
        }
        let counts: Vec<usize> = by_owner.iter().map(Vec::len).collect();
        let flat: Vec<f64> = by_owner.concat();
        let (_, dense) = all_to_allv(comm, &counts, &flat);
        let mut dense: Vec<(u32, usize, usize, [f64; 3])> = dense
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&item| {
                let (o, p) = (item[0] as usize, item[1] as usize);
                let leaf = plan
                    .index()
                    .local_leaf_containing(global[inputs[o][p]])
                    .expect("a received item lies in a local leaf");
                (leaf, o, p, item)
            })
            .collect();
        dense.sort_by_key(|&(leaf, o, p, _)| (leaf, o, p));
        let dense: Vec<f64> = dense.iter().flat_map(|&(.., item)| item).collect();
        assert_eq!(bits(&dense), bits(&forwarded), "{label}: the dense oracle");

        // Forward then backward is the identity.
        let returned = redistribution.backward(&forwarded, 3);
        assert_eq!(
            bits(&returned),
            bits(&f64_payload),
            "{label}: f64 round trip"
        );
        let returned_u32 = redistribution.backward(&forwarded_u32, 1);
        assert_eq!(returned_u32, u32_payload, "{label}: u32 round trip");
        let mut into = vec![0.0; f64_payload.len()];
        redistribution.backward_into(&forwarded, 3, &mut into);
        assert_eq!(bits(&into), bits(&f64_payload), "{label}: backward_into");
    }
}

/// Whether `f` panics, without printing the panic.
fn panics(f: impl FnOnce()) -> bool {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    result.is_err()
}

/// The errors of a redistribution, each agreed on every rank without a hang: invalid keys
/// on one rank, a plan of another tree, a `per_item` above the maximum, and a payload of
/// the wrong length on one rank, which still lets the others complete. Collective.
fn check_redistribution_errors<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    plan: &Plan,
    global: &[u64],
) {
    let comm = octree.comm();
    let rank = comm.rank() as usize;
    let nranks = comm.size() as usize;
    let mine: Vec<u64> = inputs(Distribution::Share, global, octree)[rank]
        .iter()
        .map(|&i| global[i])
        .collect();

    // A key off the finest level on the last rank, an invalid key on rank 0.
    for (bad_rank, bad_key) in [(nranks - 1, morton::root()), (0, morton::invalid_key())] {
        let mut keys = mine.clone();
        let position = keys.len() / 2;
        if rank == bad_rank {
            keys.insert(position, bad_key);
        }
        let error = Redistribution::new(octree, plan, &keys).err();
        let expected = if rank == bad_rank {
            RedistributionError::InvalidKey { index: position }
        } else {
            RedistributionError::OtherRank
        };
        assert_eq!(error, Some(expected), "{name}: invalid key {bad_key}");
    }

    // The plan of a uniform level-1 tree, not of this octree (unless it is one).
    let root = morton::root();
    let mut map: HashMap<u64, KeyType> = morton::children(root)
        .unwrap()
        .into_iter()
        .map(|child| (child, KeyType::LocalLeaf))
        .collect();
    map.insert(root, KeyType::LocalInterior);
    let one_rank = PartitionMap::new(vec![root], vec![0, 1], &[false], 0);
    let other = Plan::from_key_types(&map, 2, &one_rank).unwrap();
    let local_mismatch = octree.leaf_keys()[..] != morton::children(root).unwrap()[..];
    let mut any_mismatch = false;
    comm.all_reduce_into(
        &local_mismatch,
        &mut any_mismatch,
        SystemOperation::logical_or(),
    );
    let expected = match (local_mismatch, any_mismatch) {
        (true, _) => Some(RedistributionError::PlanMismatch),
        (false, true) => Some(RedistributionError::OtherRank),
        (false, false) => None,
    };
    let error = Redistribution::new(octree, &other, &mine).err();
    assert_eq!(error, expected, "{name}: plan of another tree");

    let redistribution =
        Redistribution::new(octree, plan, &mine).unwrap_or_else(|error| panic!("{name}: {error}"));
    // Too many values per item: every rank panics alike, before communicating.
    let too_many = redistribution.max_per_item() + 1;
    assert!(
        panics(|| {
            redistribution.forward::<u8>(&[], too_many);
        }),
        "{name}: forward with too many values per item"
    );
    assert!(
        panics(|| {
            redistribution.backward::<u8>(&[], too_many);
        }),
        "{name}: backward with too many values per item"
    );

    // A payload of the wrong length on rank 0: it takes part and then panics, and the
    // other ranks complete the exchange and the next one.
    let payload: Vec<u32> = (0..mine.len() as u32).collect();
    let forwarded = if rank == 0 {
        let wrong = vec![0u32; mine.len() + 1];
        assert!(
            panics(|| {
                redistribution.forward(&wrong, 1);
            }),
            "{name}: a payload of the wrong length"
        );
        let results = vec![0u32; redistribution.nreceived() + 1];
        assert!(
            panics(|| {
                redistribution.backward(&results, 1);
            }),
            "{name}: results of the wrong length"
        );
        redistribution.forward(&payload, 1)
    } else {
        redistribution.forward(&payload, 1);
        let results = vec![0u32; redistribution.nreceived()];
        redistribution.backward(&results, 1);
        redistribution.forward(&payload, 1)
    };
    let returned = redistribution.backward(&forwarded, 1);
    assert_eq!(returned, payload, "{name}: round trip after a panic");
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
    // The chunks the narrow forwarding tree forwarded over every scenario, summed over the
    // ranks (Phase 5S T8s).
    let mut tree_forwarded = 0usize;
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
        // The index-form plan, against the oracle. Building it is collective.
        let plan =
            Plan::new(&octree).unwrap_or_else(|error| panic!("rank {rank}: {name}: {error}"));
        let held = held_keys(&plan);
        let oracle = oracle_of(&held, &global_leaves, octree.coarse_tree_leafs());
        check_plan(
            name,
            &held,
            &oracle,
            &plan,
            octree.coarse_tree_leafs(),
            |key| computes(&octree, &held, key),
        );
        // The held set is H, the locally essential keys (Phase 5S T5).
        let [essential, top] = check_held_keys(name, &comm, &held, &octree, &global_leaves);
        let mut most = [0usize; 2];
        comm.all_reduce_into(&[essential, top][..], &mut most[..], SystemOperation::max());
        if rank == 0 && name == "graded corner blob" {
            println!(
                "held keys on {} ranks, {name}, max over the ranks: H {}, top-tree keys of H {}",
                comm.size(),
                most[0],
                most[1]
            );
        }
        // The graded tree has `Global` boxes on every rank count and spreads its lists
        // over several ranks, so the global pass and the ghost leaves are exercised.
        let local_pairs: usize = plan.levels().iter().map(|l| l.m2m_global().len()).sum();
        let mut global_pairs = 0usize;
        comm.all_reduce_into(&local_pairs, &mut global_pairs, SystemOperation::sum());
        let mut ghost_leaves = 0usize;
        comm.all_reduce_into(
            &plan.index().leaves().ghosts().len(),
            &mut ghost_leaves,
            SystemOperation::sum(),
        );
        // Every `Global` box is a shared ancestor (Phase 5S T5), so P2 drops no row.
        let local_pruned = held
            .keys()
            .filter(|&&key| !needs_local(&held, octree.coarse_tree_leafs(), key))
            .count();
        let mut pruned = 0usize;
        comm.all_reduce_into(&local_pruned, &mut pruned, SystemOperation::sum());
        if name == "graded corner blob" {
            assert!(global_pairs > 0, "{name}: no global M2M pair");
            assert!(
                comm.size() == 1 || ghost_leaves > 0,
                "{name}: no ghost leaf"
            );
        }
        // (On one rank the root is the block, `Global` and without rows.)
        if comm.size() > 1 {
            assert_eq!(pruned, 0, "{name}: a `Global` box without downward rows");
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
        let (reduction, reduced) = check_reduction(name, &octree, &held, &plan, &sizes);
        let reduced = reduced.total(&comm);
        let (top, top_size) = check_top(name, &octree, &held, &oracle, &plan, &sizes, &reduction);
        let (multipoles, forwarded) = check_multipole_exchange(
            name,
            &octree,
            &held,
            &oracle,
            &plan,
            &sizes,
            (&reduction, &top),
        );
        let multipoles = multipoles.total(&comm);
        tree_forwarded += forwarded;
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
            println!("  reduction:               {reduced:?}");
            println!("  multipoles:              {multipoles:?}");
            println!(
                "  replicated top, per rank: {top_size:?} (threshold {})",
                top.threshold()
            );
        }

        // The evaluator: determinism with variable counts, the bits of one rank, and the
        // batches it issues.
        check_repeated_evaluation(name, &comm, &plan);
        check_against_one_rank(name, &octree, &plan, &fine_keys, options);
        check_batches(name, &octree, &oracle, &plan, false, default_routing(&comm));
        check_batches(name, &octree, &oracle, &plan, true, default_routing(&comm));
        let narrow = &format!("{name}, narrow tree");
        check_batches(narrow, &octree, &oracle, &plan, false, narrow_tree(&comm));
        check_batches(narrow, &octree, &oracle, &plan, true, narrow_tree(&comm));

        // The redistribution of the sources and of the targets, from every input
        // distribution of the scenario's global points (every rank's, in rank order).
        let global_sources = gather_to_all(&keys(&sources), &comm);
        let global_targets = gather_to_all(&keys(&targets), &comm);
        check_redistribution(name, &octree, &plan, "sources", &global_sources);
        check_redistribution(name, &octree, &plan, "targets", &global_targets);
        check_redistribution_errors(name, &octree, &plan, &global_sources);
    }
    if rank == 0 {
        println!(
            "narrow forwarding tree on {} ranks: {tree_forwarded} chunks forwarded",
            comm.size()
        );
    }
    // From 4 ranks some box of the scenarios has three readers or more; at many ranks the
    // scenarios' few points per rank may leave none below R.
    if (4..=8).contains(&comm.size()) {
        assert!(tree_forwarded > 0, "the narrow tree forwarded nothing");
    }

    // The owned top tree against one rank on the cube and the Plummer sphere (Phase 5S T5):
    // 200 points per rank, generated by index, each rank its own range.
    let nranks = comm.size() as usize;
    for (name, points) in [
        ("cube", cube(200 * rank..200 * (rank + 1))),
        ("Plummer sphere", plummer(200 * rank..200 * (rank + 1))),
    ] {
        eprintln!("rank {rank}: {name}");
        let fine_keys = keys(&points);
        let options = OctreeOptions::new()
            .with_max_level(8)
            .with_max_fine_keys(16)
            .with_ghost_children(true);
        let octree = Octree::new(&fine_keys, options, &comm);
        let plan =
            Plan::new(&octree).unwrap_or_else(|error| panic!("rank {rank}: {name}: {error}"));
        let formed = check_against_one_rank(name, &octree, &plan, &fine_keys, options);
        let mut total = 0usize;
        comm.all_reduce_into(&formed, &mut total, SystemOperation::sum());
        let mut deepest = 0usize;
        comm.all_reduce_into(
            &plan.reduction_levels().len(),
            &mut deepest,
            SystemOperation::max(),
        );
        if rank == 0 {
            println!(
                "{name} on {nranks} ranks: {total} `Global` multipoles formed on their owners, \
                 {deepest} reduction levels, all bit for bit one rank's"
            );
        }
    }
}
