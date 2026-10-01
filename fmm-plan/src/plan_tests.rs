//! Pure local tests of the box index and the index-form lists, on the synthetic key maps
//! of `interaction_manager_tests.rs`. No MPI.
use std::collections::{BTreeSet, HashMap};

use nd_octree::{MortonKey, morton, octree::KeyType};

use super::{Plan, PlanError};
use crate::interaction_manager::{
    V_LIST_DIRECTIONS,
    tests::{ListMaps, all_trees, key_types, uniform_leaves},
};
use crate::lists::{Csr, GroupedCsr, NOCTANTS, NOFFSETS, offset, offset_index};

/// Reclassify the root as `Global`, as on one rank.
pub(crate) fn with_global_root(map: &HashMap<MortonKey, KeyType>) -> HashMap<MortonKey, KeyType> {
    let mut map = map.clone();
    if map[&morton::root()] == KeyType::LocalInterior {
        map.insert(morton::root(), KeyType::Global);
    }
    map
}

/// Pretend that the level-one half with x-index 1 and everything below it belong to rank
/// 1. The lists depend only on the leaf/interior classification, so they do not change.
pub(crate) fn ghost_half(map: &HashMap<MortonKey, KeyType>) -> HashMap<MortonKey, KeyType> {
    let mut map = map.clone();
    for (&key, kind) in map.iter_mut() {
        if morton::level(key) == 0 {
            continue;
        }
        let (_, index) = morton::decode(morton::ancestor_at_level(key, 1).unwrap());
        if index[0] == 1 {
            *kind = match *kind {
                KeyType::LocalLeaf => KeyType::GhostLeaf(1),
                _ => KeyType::GhostInterior(1),
            };
        }
    }
    map
}

/// Every key map of these tests, with a name: each tree all local, with a `Global` root,
/// and with a `Global` root and a ghost half.
pub(crate) fn cases() -> Vec<(String, HashMap<MortonKey, KeyType>)> {
    let mut cases = Vec::new();
    for (t, leaves) in all_trees().into_iter().enumerate() {
        let local = key_types(&leaves);
        let global = with_global_root(&local);
        let ghosts = ghost_half(&global);
        cases.push((format!("tree {t}, local"), local));
        cases.push((format!("tree {t}, global root"), global));
        cases.push((format!("tree {t}, ghost half"), ghosts));
    }
    cases
}

pub(crate) fn nlevels(map: &HashMap<MortonKey, KeyType>) -> usize {
    map.keys().map(|&key| morton::level(key)).max().unwrap() + 1
}

pub(crate) fn plan(map: &HashMap<MortonKey, KeyType>) -> Plan {
    Plan::from_key_types(map, nlevels(map), &[morton::root()]).unwrap()
}

/// The same map, inserted in the order selected by `order`.
fn reinserted(map: &HashMap<MortonKey, KeyType>, order: usize) -> HashMap<MortonKey, KeyType> {
    let mut entries: Vec<(MortonKey, KeyType)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|&(key, _)| key);
    match order {
        0 => {}
        1 => entries.reverse(),
        _ => entries.sort_unstable_by_key(|&(key, _)| key.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
    }
    let mut reinserted = HashMap::new();
    for (key, kind) in entries {
        reinserted.insert(key, kind);
    }
    reinserted
}

/// The rows of a grouped view as (target, source, group), in row order.
fn row_triples<G: Copy + Into<usize>>(view: &GroupedCsr<G>) -> Vec<(u32, u32, usize)> {
    (0..view.nrows())
        .flat_map(|t| {
            let (sources, groups) = view.row(t);
            sources
                .iter()
                .zip(groups)
                .map(move |(&s, &g)| (t as u32, s, g.into()))
        })
        .collect()
}

/// The batches of a grouped view as (target, source, group), batch after batch.
fn batch_triples<G: Copy + Into<usize>>(view: &GroupedCsr<G>) -> Vec<(u32, u32, usize)> {
    (0..view.ngroups())
        .flat_map(|g| {
            let (targets, sources) = view.batch(g);
            targets.iter().zip(sources).map(move |(&t, &s)| (t, s, g))
        })
        .collect()
}

/// The entries of a CSR view as keys, using `key_of` to translate each entry.
fn row_keys(view: &Csr, t: usize, key_of: impl Fn(u32) -> MortonKey) -> Vec<MortonKey> {
    view.row(t).iter().map(|&e| key_of(e)).collect()
}

fn sorted(mut keys: Vec<MortonKey>) -> Vec<MortonKey> {
    keys.sort_unstable();
    keys
}

#[test]
fn numbering_is_dense_morton_and_independent_of_insertion_order() {
    for (name, map) in cases() {
        let plan = plan(&map);
        for order in 0..3 {
            assert_eq!(
                Plan::from_key_types(&reinserted(&map, order), nlevels(&map), &[morton::root()]),
                Ok(plan.clone()),
                "{name}: insertion order {order}"
            );
        }

        let index = plan.index();
        assert_eq!(index.nlevels(), nlevels(&map));
        assert_eq!(plan.nlevels(), nlevels(&map));
        let mut all = Vec::new();
        for level in 0..index.nlevels() {
            let keys = index.keys(level);
            assert!(keys.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
            assert_eq!(index.len(level), keys.len());
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(morton::level(key), level);
                assert_eq!(index.kind(level, i), map[&key]);
                assert_eq!(index.find(key), Some((level, i as u32)));
            }
            all.extend_from_slice(keys);
        }
        assert_eq!(sorted(all), sorted(map.keys().copied().collect()), "{name}");
        assert_eq!(index.find(morton::invalid_key()), None);
        let deepest = morton::from_index_and_level([0, 0, 0], index.nlevels());
        assert_eq!(index.find(deepest), None);

        // Consecutive children.
        for level in 0..index.nlevels() {
            for (i, &key) in index.keys(level).iter().enumerate() {
                let held = morton::children(key)
                    .filter(|_| level + 1 < index.nlevels())
                    .filter(|children| children.iter().all(|child| map.contains_key(child)));
                match (held, index.first_child(level, i)) {
                    (Some(children), Some(first)) => {
                        for (o, child) in children.into_iter().enumerate() {
                            assert_eq!(index.key(level + 1, first as usize + o), child);
                            assert_eq!(morton::child_index(child), o);
                        }
                    }
                    (None, None) => {}
                    (held, first) => panic!("{name}: {key}: children {held:?}, first {first:?}"),
                }
            }
        }
    }
}

#[test]
fn leaves_are_numbered_local_by_level_then_named_ghosts_by_key() {
    let mut ghost_cases = 0;
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        let leaves = index.leaves();

        // Local leaves by (level, key).
        let mut local: Vec<(usize, MortonKey)> = map
            .iter()
            .filter(|&(_, &kind)| kind == KeyType::LocalLeaf)
            .map(|(&key, _)| (morton::level(key), key))
            .collect();
        local.sort_unstable();
        assert_eq!(leaves.nlocal(), local.len());
        for (j, &(level, key)) in local.iter().enumerate() {
            assert_eq!(leaves.key(j), key, "{name}");
            assert_eq!(index.leaf_key(j), key);
            assert_eq!(leaves.level(j), level);
            assert!(leaves.local(level).contains(&j));
        }
        let mut next = 0;
        for level in 0..index.nlevels() {
            assert_eq!(leaves.local(level).start, next);
            next = leaves.local(level).end;
        }
        assert_eq!(next, leaves.nlocal());

        // Ghost leaves: exactly those named by a U- or X-list of a non-ghost key, by key.
        let reference = ListMaps::new(&map);
        let named: BTreeSet<MortonKey> = [reference.u_list(), reference.x_list()]
            .into_iter()
            .flat_map(|lists| lists.values().flatten().copied())
            .filter(|entry| map[entry].is_ghost())
            .collect();
        let ghosts: Vec<MortonKey> = leaves.ghosts().map(|j| leaves.key(j)).collect();
        assert_eq!(ghosts, named.into_iter().collect::<Vec<_>>(), "{name}");
        ghost_cases += usize::from(!ghosts.is_empty());

        // The box and leaf numberings are inverse to each other.
        for j in 0..leaves.len() {
            let (level, i) = (leaves.level(j), leaves.box_index(j) as usize);
            assert_eq!(index.key(level, i), leaves.key(j));
            assert_eq!(index.box_leaf(level, i), Some(j as u32));
            assert_eq!(index.find_leaf(leaves.key(j)), Some(j as u32));
        }
        let numbered: usize = (0..index.nlevels())
            .map(|level| {
                (0..index.len(level))
                    .filter(|&i| index.box_leaf(level, i).is_some())
                    .count()
            })
            .sum();
        assert_eq!(numbered, leaves.len());

        // A finest-level key finds its local leaf, and only a local one.
        for level in 0..index.nlevels() {
            for (i, &key) in index.keys(level).iter().enumerate() {
                let fine = morton::finest_outer_descendent(key);
                let expected = match index.kind(level, i) {
                    KeyType::LocalLeaf => index.box_leaf(level, i),
                    KeyType::GhostLeaf(_) => None,
                    _ => continue,
                };
                assert_eq!(index.local_leaf_containing(fine), expected, "{name}: {key}");
            }
        }
    }
    assert!(ghost_cases > 0);
}

#[test]
fn index_lists_equal_the_per_key_rule() {
    // Count non-empty lists so that the comparison cannot pass vacuously.
    let mut populated = [0usize; 4];
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        let reference = ListMaps::new(&map);
        let leaf_key = |j: u32| index.leaf_key(j as usize);

        for level in 0..index.nlevels() {
            let lists = plan.level(level);
            let local = index.leaves().local(level);
            for (i, &key) in index.keys(level).iter().enumerate() {
                let kind = index.kind(level, i);
                let v = sorted(
                    lists
                        .v()
                        .row(i)
                        .0
                        .iter()
                        .map(|&s| index.key(level, s as usize))
                        .collect(),
                );
                let x = sorted(row_keys(lists.x(), i, leaf_key));
                if kind.is_ghost() {
                    assert!(v.is_empty() && x.is_empty(), "{name}: ghost {key}");
                    assert!(lists.m2m_local().row(i).0.is_empty());
                    assert!(lists.m2m_global().row(i).0.is_empty());
                    assert!(lists.l2l().row(i).0.is_empty());
                    assert!(lists.p2m().row(i).is_empty());
                    continue;
                }
                assert_eq!(v, reference.v_list()[&key], "{name}: V of {key}");
                assert_eq!(x, reference.x_list()[&key], "{name}: X of {key}");

                let (u, w) = match index.box_leaf(level, i) {
                    Some(leaf) => {
                        let r = leaf as usize - local.start;
                        let near = row_keys(lists.near(), r, leaf_key);
                        assert!(
                            near.contains(&key),
                            "{name}: near list of {key} lacks itself"
                        );
                        let u = sorted(near.into_iter().filter(|&e| e != key).collect());
                        let w = row_keys(lists.w(), r, |s| index.key(level + 1, s as usize));
                        (u, w)
                    }
                    None => (Vec::new(), Vec::new()),
                };
                assert_eq!(u, reference.u_list()[&key], "{name}: U of {key}");
                assert_eq!(w, reference.w_list()[&key], "{name}: W of {key}");
                for (count, list) in populated.iter_mut().zip([&u, &v, &w, &x]) {
                    *count += usize::from(!list.is_empty());
                }
            }
        }
    }
    assert!(populated.iter().all(|&count| count > 0), "{populated:?}");
}

#[test]
fn v_batches_hold_each_target_at_most_once_and_partition_the_pairs() {
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        for level in 0..index.nlevels() {
            let v = plan.level(level).v();
            assert_eq!(v.ngroups(), NOFFSETS);
            let mut count = 0;
            for (d, &direction) in V_LIST_DIRECTIONS.iter().enumerate() {
                let (targets, sources) = v.batch(d);
                assert_eq!(targets.len(), sources.len());
                assert!(
                    targets.windows(2).all(|pair| pair[0] < pair[1]),
                    "{name}: batch {d}"
                );
                let distinct: BTreeSet<u32> = sources.iter().copied().collect();
                assert_eq!(distinct.len(), sources.len(), "{name}: batch {d} sources");
                for (&t, &s) in targets.iter().zip(sources) {
                    let (target, source) =
                        (index.key(level, t as usize), index.key(level, s as usize));
                    assert_eq!(offset(target, source), direction);
                }
                count += targets.len();
            }
            assert_eq!(count, v.len());
            let mut rows = row_triples(v);
            let mut batches = batch_triples(v);
            rows.sort_unstable();
            batches.sort_unstable();
            assert_eq!(rows, batches, "{name}: level {level}");
        }
    }
}

#[test]
fn octant_batches_pair_children_with_parents_and_partition_the_pairs() {
    let mut global_pairs = 0;
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        for level in 0..index.nlevels() {
            let lists = plan.level(level);

            // Expected parent-child pairs, from the keys alone.
            let mut expected_local = Vec::new();
            let mut expected_global = Vec::new();
            let mut expected_l2l = Vec::new();
            for (i, &key) in index.keys(level).iter().enumerate() {
                let kind = index.kind(level, i);
                let pairs = match kind {
                    KeyType::LocalInterior => Some(&mut expected_local),
                    KeyType::Global => Some(&mut expected_global),
                    _ => None,
                };
                if let Some(pairs) = pairs {
                    for child in morton::children(key).unwrap() {
                        let (_, c) = index.find(child).unwrap();
                        pairs.push((i as u32, c, morton::child_index(child)));
                    }
                }
                if !kind.is_ghost()
                    && let Some(parent) = morton::parent(key)
                {
                    let (_, p) = index.find(parent).unwrap();
                    expected_l2l.push((i as u32, p, morton::child_index(key)));
                }
            }

            for (view, expected, child_level) in [
                (lists.m2m_local(), expected_local, level + 1),
                (lists.m2m_global(), expected_global, level + 1),
                (lists.l2l(), expected_l2l, level.wrapping_sub(1)),
            ] {
                assert_eq!(view.ngroups(), NOCTANTS);
                for o in 0..NOCTANTS {
                    let (targets, sources) = view.batch(o);
                    assert!(
                        targets.windows(2).all(|pair| pair[0] < pair[1]),
                        "{name}: octant {o}"
                    );
                    for (&t, &s) in targets.iter().zip(sources) {
                        let target = index.key(level, t as usize);
                        let source = index.key(child_level, s as usize);
                        let (parent, child) = if child_level > level {
                            (target, source)
                        } else {
                            (source, target)
                        };
                        assert_eq!(morton::parent(child), Some(parent), "{name}");
                        assert_eq!(morton::child_index(child), o, "{name}");
                    }
                }
                let mut batches = batch_triples(view);
                batches.sort_unstable();
                let mut expected = expected;
                expected.sort_unstable();
                assert_eq!(batches, expected, "{name}: level {level}");
            }
            global_pairs += lists.m2m_global().len();
            // Each child has one parent.
            assert!((0..lists.l2l().nrows()).all(|t| lists.l2l().row(t).0.len() <= 1));
        }
    }
    assert!(global_pairs > 0);
}

#[test]
fn target_centric_rows_match_the_groupings_in_group_order() {
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        for level in 0..index.nlevels() {
            let lists = plan.level(level);
            let check = |view: &GroupedCsr<u8>| {
                for t in 0..view.nrows() {
                    let (_, octants) = view.row(t);
                    assert!(octants.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
                }
                let mut rows = row_triples(view);
                rows.sort_unstable();
                let mut batches = batch_triples(view);
                batches.sort_unstable();
                assert_eq!(rows, batches, "{name}: level {level}");
            };
            check(lists.m2m_local());
            check(lists.m2m_global());
            check(lists.l2l());

            let v = lists.v();
            for t in 0..v.nrows() {
                let (sources, offsets) = v.row(t);
                assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
                for (&s, &d) in sources.iter().zip(offsets) {
                    let pair = (index.key(level, t), index.key(level, s as usize));
                    assert_eq!(offset_index(offset(pair.0, pair.1)), Some(d as usize));
                }
            }
            assert_eq!(v.offset_indices(), v.groups());
            assert_eq!(lists.l2l().octants(), lists.l2l().groups());

            // CSR rows are strictly ascending.
            for view in [lists.x(), lists.p2m(), lists.near(), lists.w(), lists.l2p()] {
                for t in 0..view.nrows() {
                    assert!(
                        view.row(t).windows(2).all(|pair| pair[0] < pair[1]),
                        "{name}"
                    );
                }
            }
        }
    }
}

#[test]
fn rows_cover_the_targets_of_a_level_once_each() {
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        for level in 0..index.nlevels() {
            let lists = plan.level(level);
            let nboxes = index.len(level);
            let local = index.leaves().local(level);

            let box_rows = [
                lists.v().row_offsets(),
                lists.x().row_offsets(),
                lists.m2m_local().row_offsets(),
                lists.m2m_global().row_offsets(),
                lists.l2l().row_offsets(),
                lists.p2m().row_offsets(),
            ];
            let leaf_rows = [
                lists.near().row_offsets(),
                lists.w().row_offsets(),
                lists.l2p().row_offsets(),
            ];
            for (offsets, nrows) in box_rows
                .into_iter()
                .map(|offsets| (offsets, nboxes))
                .chain(leaf_rows.into_iter().map(|offsets| (offsets, local.len())))
            {
                assert_eq!(offsets.len(), nrows + 1, "{name}: level {level}");
                assert_eq!(offsets[0], 0);
                assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
            }

            // A level buffer split into chunks of one box lines up with the box rows.
            let size = 3;
            let mut buffer = vec![0usize; size * nboxes];
            let mut visited = 0;
            for (t, chunk) in buffer.chunks_mut(size).enumerate() {
                chunk.fill(lists.v().row(t).0.len() + lists.l2l().row(t).0.len());
                visited += 1;
            }
            assert_eq!(visited, lists.v().nrows());

            // Leaf row r belongs to local leaf `local.start + r` of this level.
            for r in 0..local.len() {
                let leaf = local.start + r;
                let i = index.leaves().box_index(leaf);
                assert_eq!(index.leaves().level(leaf), level);
                assert_eq!(index.kind(level, i as usize), KeyType::LocalLeaf);
                assert_eq!(lists.l2p().row(r), [i]);
                assert!(lists.near().row(r).contains(&(leaf as u32)));
                assert_eq!(lists.p2m().row(i as usize), [leaf as u32]);
            }
            assert_eq!(lists.p2m().len(), local.len(), "{name}");
        }
    }
}

#[test]
fn entries_lie_on_the_levels_of_two_to_one_balance() {
    for (name, map) in cases() {
        let plan = plan(&map);
        let index = plan.index();
        let leaves = index.leaves();
        for level in 0..index.nlevels() {
            let lists = plan.level(level);
            assert!(
                lists
                    .v()
                    .sources()
                    .iter()
                    .all(|&s| (s as usize) < index.len(level))
            );
            if level + 1 < index.nlevels() {
                assert!(
                    lists
                        .w()
                        .entries()
                        .iter()
                        .all(|&s| (s as usize) < index.len(level + 1))
                );
            } else {
                assert!(lists.w().is_empty());
            }
            for &j in lists.x().entries() {
                assert_eq!(leaves.level(j as usize) + 1, level, "{name}: X entry");
            }
            for &j in lists.near().entries() {
                let entry_level = leaves.level(j as usize);
                assert!(
                    entry_level + 1 >= level && entry_level <= level + 1,
                    "{name}: U entry"
                );
            }
            if level <= 1 {
                assert!(lists.v().is_empty() && lists.x().is_empty(), "{name}");
            }
        }
    }
}

#[test]
fn every_v_list_offset_occurs() {
    let plan = plan(&key_types(&uniform_leaves(3)));
    let v = plan.level(3).v();
    for (d, direction) in V_LIST_DIRECTIONS.iter().enumerate() {
        assert!(!v.batch(d).0.is_empty(), "offset {direction:?}");
    }
}

#[test]
fn offset_index_inverts_v_list_directions() {
    // The closed form of CONVENTIONS §3.12.
    let kappa = |t: i64| (t + 1).clamp(0, 3);
    let indicator = |t: i64| i64::from(t.abs() <= 1);
    for (d, &direction) in V_LIST_DIRECTIONS.iter().enumerate() {
        assert_eq!(offset_index(direction), Some(d));
        let [x, y, z] = direction;
        let closed = 49 * (x + 3) + 7 * (y + 3) + (z + 3)
            - 9 * kappa(x)
            - indicator(x) * (3 * kappa(y) + indicator(y) * kappa(z));
        assert_eq!(closed, d as i64);
    }
    for x in -4i64..=4 {
        for y in -4i64..=4 {
            for z in -4i64..=4 {
                let near = x.abs().max(y.abs()).max(z.abs()) <= 1;
                let far = x.abs().max(y.abs()).max(z.abs()) > 3;
                if near || far {
                    assert_eq!(offset_index([x, y, z]), None);
                }
            }
        }
    }
}

#[test]
fn the_root_multipole_is_formed_by_the_matching_pass() {
    let local = key_types(&uniform_leaves(2));
    let root_row = |plan: &Plan, global: bool| {
        let lists = plan.level(0);
        let view = if global {
            lists.m2m_global()
        } else {
            lists.m2m_local()
        };
        view.row(0).1.to_vec()
    };
    let all_octants: Vec<u8> = (0..8).collect();

    let plan_local = plan(&local);
    assert_eq!(root_row(&plan_local, false), all_octants);
    assert!(root_row(&plan_local, true).is_empty());

    let plan_global = plan(&with_global_root(&local));
    assert_eq!(root_row(&plan_global, true), all_octants);
    assert!(root_row(&plan_global, false).is_empty());
}

#[test]
fn malformed_maps_are_errors_not_panics() {
    let at = |index: [usize; 3], level: usize| morton::from_index_and_level(index, level);

    // A hole in a uniform tree: the missing cell is not covered by a leaf parent.
    let mut holed = key_types(&uniform_leaves(2));
    holed.remove(&at([3, 3, 3], 2));
    let error = Plan::from_key_types(&holed, 3, &[]).unwrap_err();
    assert!(matches!(error, PlanError::MalformedTree { .. }), "{error}");

    // An interior neighbour without its children, as without the ghost-children layer.
    let refined = at([0, 0, 0], 1);
    let mut childless = with_global_root(&key_types(&all_trees()[2]));
    for child in morton::children(refined).unwrap() {
        childless.remove(&child);
    }
    childless.insert(refined, KeyType::GhostInterior(1));
    for order in 0..3 {
        let error = Plan::from_key_types(&reinserted(&childless, order), 3, &[]).unwrap_err();
        // The first non-ghost box in Morton order whose list names a child is the level-1
        // leaf [0, 0, 1]. Its W-list, resolved before its U-list, starts with child 0 of
        // the refined octant.
        assert_eq!(
            error,
            PlanError::UnheldListEntry {
                key: at([0, 0, 1], 1),
                entry: morton::children(refined).unwrap()[0],
            },
            "{error}"
        );
    }

    // A non-ghost interior box without its children.
    let mut orphaned = childless.clone();
    orphaned.insert(refined, KeyType::LocalInterior);
    assert_eq!(
        Plan::from_key_types(&orphaned, 3, &[]),
        Err(PlanError::MissingChild { key: refined })
    );

    // Too few levels.
    let uniform = key_types(&uniform_leaves(2));
    assert!(matches!(
        Plan::from_key_types(&uniform, 2, &[]),
        Err(PlanError::LevelOutOfRange { nlevels: 2, .. })
    ));

    // A coarse block that is not held, or held as a ghost.
    let block = at([1, 1, 1], 1);
    assert_eq!(
        Plan::from_key_types(&uniform, 3, &[at([5, 5, 5], 3)]),
        Err(PlanError::CoarseBlock {
            key: at([5, 5, 5], 3)
        })
    );
    let ghosts = ghost_half(&uniform);
    assert_eq!(
        Plan::from_key_types(&ghosts, 3, &[block]),
        Err(PlanError::CoarseBlock { key: block })
    );
}
