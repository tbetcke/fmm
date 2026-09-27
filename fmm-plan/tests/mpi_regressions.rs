//! Public-API regression tests. Keep MPI initialization in one test: MPI cannot
//! be initialized again after finalization within the same process.
use std::collections::HashMap;

use fmm_plan::{
    fmm::index_fmm::run_index_fmm,
    ghost_communicator::{FmmGhostCommunicator, LevelChunkSizes},
    interaction_manager::InteractionManager,
};
use mpi::{collective::SystemOperation, traits::*};
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

/// Compare the interaction lists of every local non-ghost key against the
/// brute-force oracle over the gathered global tree.
fn check_interaction_lists(
    name: &str,
    all_keys: &HashMap<u64, KeyType>,
    lists: &InteractionManager,
    global_leaves: &[u64],
) {
    let mut keys: Vec<u64> = morton::get_interior_keys(global_leaves)
        .into_iter()
        .collect();
    keys.extend_from_slice(global_leaves);
    keys.sort_unstable();
    keys.dedup();

    let mut expected_entries: Vec<u64> = all_keys
        .iter()
        .filter(|(_, key_type)| !key_type.is_ghost())
        .map(|(&key, _)| key)
        .collect();
    expected_entries.sort_unstable();
    for map in [
        lists.u_list(),
        lists.v_list(),
        lists.w_list(),
        lists.x_list(),
    ] {
        let mut entries: Vec<u64> = map.keys().copied().collect();
        entries.sort_unstable();
        assert_eq!(entries, expected_entries, "{name}: entry set");
    }

    for &key in &expected_entries {
        let expected = oracle_lists(&keys, global_leaves, key);
        let actual = [
            &lists.u_list()[&key],
            &lists.v_list()[&key],
            &lists.w_list()[&key],
            &lists.x_list()[&key],
        ];
        for (index, list_name) in ["U", "V", "W", "X"].into_iter().enumerate() {
            assert_eq!(
                *actual[index], expected[index],
                "{name}: {list_name}-list of {key}"
            );
            assert!(actual[index].windows(2).all(|pair| pair[0] < pair[1]));
            assert!(!actual[index].contains(&key));
            // The tree is built with the ghost-children layer, so every entry is
            // a key of the local map and carries its own classification.
            for entry in actual[index] {
                assert!(
                    all_keys.contains_key(entry),
                    "{name}: {list_name}-list entry {entry} of {key} is not a key of the tree"
                );
            }
        }
    }
}

/// The value that the owner of `key` sends as the `j`-th entry of its chunk.
fn ghost_value(key: u64, j: usize) -> u64 {
    key.wrapping_mul(64).wrapping_add(j as u64)
}

/// Exchange the data of every ghost in the interaction lists and check that
/// each ghost holder receives its owner's values, forward and backward.
fn check_ghost_exchange<C: CommunicatorCollectives>(
    name: &str,
    octree: &Octree<'_, C>,
    lists: &InteractionManager,
    chunk_sizes: LevelChunkSizes,
) {
    let all_keys = octree.all_keys();
    let entries: Vec<u64> = [
        lists.u_list(),
        lists.v_list(),
        lists.w_list(),
        lists.x_list(),
    ]
    .into_iter()
    .flat_map(|map| map.values().flatten().copied())
    .collect();
    let mut expected: Vec<u64> = entries
        .iter()
        .copied()
        .filter(|entry| all_keys[entry].is_ghost())
        .collect();
    expected.sort_unstable();
    expected.dedup();

    let mut ghosts = FmmGhostCommunicator::<u64>::new(octree, entries, chunk_sizes.clone());
    assert_eq!(ghosts.nlevels(), octree.global_max_level() + 1);

    let mut received: Vec<u64> = Vec::new();
    for level in 0..ghosts.nlevels() {
        assert_eq!(
            ghosts.chunk_size(level),
            chunk_sizes.chunk_size(level).unwrap()
        );
        for (position, &key) in ghosts.receive_keys(level).iter().enumerate() {
            assert_eq!(morton::level(key), level, "{name}: receive key level");
            assert_eq!(ghosts.receive_position(key), Some(position));
            received.push(key);
        }
        // Only the owner of a key sends it.
        for key in ghosts.send_keys(level) {
            assert!(
                matches!(
                    all_keys.get(key),
                    Some(KeyType::LocalLeaf | KeyType::LocalInterior)
                ),
                "{name}: sent key {key} is not local to its owner"
            );
        }
        for (key, chunk) in ghosts.send_chunks_mut(level) {
            for (j, value) in chunk.iter_mut().enumerate() {
                *value = ghost_value(key, j);
            }
        }
    }
    received.sort_unstable();
    assert_eq!(received, expected, "{name}: received ghost keys");
    // The graded tree always spreads its interaction lists over several ranks,
    // so an exchange without any ghost would mean the check is vacuous.
    let mut global_received = 0usize;
    octree.comm().all_reduce_into(
        &received.len(),
        &mut global_received,
        SystemOperation::sum(),
    );
    if name == "graded corner blob" && octree.comm().size() > 1 {
        assert!(global_received > 0, "{name}: no ghosts were exchanged");
    }

    ghosts.forward_all();
    for level in 0..ghosts.nlevels() {
        let chunk_size = ghosts.chunk_size(level);
        for (key, chunk) in ghosts.receive_chunks(level) {
            let values: Vec<u64> = (0..chunk_size).map(|j| ghost_value(key, j)).collect();
            assert_eq!(chunk, values, "{name}: forwarded values of {key}");
            assert_eq!(ghosts.receive_chunk(key), Some(chunk));
        }
    }

    // Send the received values back to the owners.
    for level in 0..ghosts.nlevels() {
        ghosts.send_buffer_mut(level).fill(0);
        ghosts.backward(level);
        for (key, chunk) in ghosts.send_chunks_mut(level) {
            for (j, &value) in chunk.iter().enumerate() {
                assert_eq!(
                    value,
                    ghost_value(key, j),
                    "{name}: returned values of {key}"
                );
            }
        }
    }
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

        // The interaction lists are built without communication, but the oracle
        // needs the global tree, so the gather is entered by every rank.
        let lists = InteractionManager::new(&octree);
        let mut global_leaves = gather_to_all(leaves, &comm);
        global_leaves.sort_unstable();
        assert!(morton::is_complete_linear_and_balanced(&global_leaves));
        check_interaction_lists(name, octree.all_keys(), &lists, &global_leaves);

        let chunk_sizes = if name == "identical populations" {
            LevelChunkSizes::Uniform(3)
        } else {
            LevelChunkSizes::PerLevel((1..=DEEPEST_LEVEL as usize + 1).collect())
        };
        check_ghost_exchange(name, &octree, &lists, chunk_sizes);

        // Every leaf must receive every leaf index exactly once.
        if let Err(message) = run_index_fmm(&octree, &lists) {
            panic!("rank {rank}: {name}: index FMM: {message}");
        }
    }
}
