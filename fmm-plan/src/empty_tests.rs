//! Pure local tests of the empty-box rule on the synthetic key maps of
//! `interaction_manager_tests.rs` (through the cases of `plan_tests.rs`: each tree all
//! local, with a `Global` root, and with a ghost half). No MPI.
use std::collections::HashMap;

use nd_octree::{MortonKey, morton, octree::KeyType};

use super::EmptyBoxes;
use crate::interaction_manager::is_leaf;
use crate::near::{LeafCounts, LeafPair, NearRule};
use crate::plan::{
    Plan, PlanError, PlanRules,
    tests::{cases, nlevels, one_rank_map, rows_by_key},
};

/// A count of 0–5 that mixes the bits of `key` and `salt`, zero for about a third of the
/// keys.
fn count(key: MortonKey, salt: u64) -> usize {
    ((key ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58) as usize % 6 / 2 * 3
}

/// Leaf counts of every leaf of `map` (ghosts included) from `leaf`, and the subtree counts
/// of every key of `map`: the sum over the leaves of `map` at or below it.
fn counts_of(
    map: &HashMap<MortonKey, KeyType>,
    leaf: impl Fn(MortonKey, u64) -> usize,
) -> LeafCounts {
    let mut subtree: HashMap<MortonKey, (usize, usize)> =
        map.keys().map(|&key| (key, (0, 0))).collect();
    let mut counts = LeafCounts::new();
    for (&key, &kind) in map {
        if !is_leaf(kind) {
            continue;
        }
        let (s, t) = (leaf(key, 1), leaf(key, 2));
        counts.set_sources(key, s);
        counts.set_targets(key, t);
        let mut ancestor = Some(key);
        while let Some(a) = ancestor {
            if let Some(entry) = subtree.get_mut(&a) {
                entry.0 += s;
                entry.1 += t;
            }
            ancestor = morton::parent(a);
        }
    }
    for (key, (s, t)) in subtree {
        counts.set_box_sources(key, s);
        counts.set_box_targets(key, t);
    }
    counts
}

fn plan_with(map: &HashMap<MortonKey, KeyType>, rules: &PlanRules<'_>) -> Result<Plan, PlanError> {
    Plan::from_key_types_with_rules(map, nlevels(map), &one_rank_map(map), rules)
}

fn empty_rule(counts: &LeafCounts) -> PlanRules<'_> {
    PlanRules {
        empty_boxes: Some(counts),
        ..PlanRules::default()
    }
}

/// The rows of `without` (by key, [`rows_by_key`]'s order: V, X, L2L, local M2M, global
/// M2M, near, W) with the empty-box rule of `counts` applied, and what it left out.
fn expected_rows(
    without: &Plan,
    counts: &LeafCounts,
) -> (HashMap<MortonKey, Vec<Vec<(MortonKey, usize)>>>, EmptyBoxes) {
    let has_sources = |key: MortonKey| counts.box_sources(key).unwrap() > 0;
    let mut empty = EmptyBoxes::default();
    let mut rows = rows_by_key(without);
    for (&key, rows) in rows.iter_mut() {
        let no_targets = counts.box_targets(key).unwrap() == 0;
        let filter = |row: &mut Vec<(MortonKey, usize)>, left_out: &mut usize| {
            let before = row.len();
            row.retain(|&(entry, _)| has_sources(entry));
            *left_out += before - row.len();
        };
        if no_targets {
            empty.target_boxes += 1;
            empty.v_without_targets += rows[0].len();
            empty.x_without_targets += rows[1].len();
            empty.l2l_without_targets += rows[2].len();
            for r in [0, 1, 2] {
                rows[r].clear();
            }
            if rows.len() > 5 {
                empty.w_without_targets += rows[6].len();
                rows[6].clear();
            }
        } else {
            filter(&mut rows[0], &mut empty.v_without_sources);
            if rows.len() > 5 {
                filter(&mut rows[6], &mut empty.w_without_sources);
            }
        }
        // Both M2M rows (one of them is empty).
        filter(&mut rows[3], &mut empty.m2m_without_sources);
        filter(&mut rows[4], &mut empty.m2m_without_sources);
    }
    (rows, empty)
}

#[test]
fn the_rule_leaves_out_exactly_its_rows_and_entries() {
    let mut left_out = EmptyBoxes::default();
    for (name, map) in cases() {
        let counts = counts_of(&map, count);
        let without = plan_with(&map, &PlanRules::default()).unwrap();
        let plan = plan_with(&map, &empty_rule(&counts)).unwrap();
        let (expected, empty) = expected_rows(&without, &counts);
        let rows = rows_by_key(&plan);
        assert_eq!(rows.len(), expected.len(), "{name}: boxes with rows");
        for (key, rows) in &rows {
            // Every row in its order: the rule only shortens rows.
            assert_eq!(rows, &expected[key], "{name}: rows of {key}");
        }
        assert_eq!(plan.empty_boxes(), empty, "{name}: what the rule left out");
        // The box index and every other view are the plan's without the rule.
        assert_eq!(plan.index().nlevels(), without.index().nlevels(), "{name}");
        for level in 0..plan.nlevels() {
            assert_eq!(
                plan.index().keys(level),
                without.index().keys(level),
                "{name}"
            );
            let (a, b) = (plan.level(level), without.level(level));
            assert_eq!(a.p2m(), b.p2m(), "{name}: P2M");
            assert_eq!(a.l2p(), b.l2p(), "{name}: L2P");
        }
        assert_eq!(plan.near_pairs(), without.near_pairs(), "{name}");
        left_out.target_boxes += empty.target_boxes;
        left_out.v_without_targets += empty.v_without_targets;
        left_out.x_without_targets += empty.x_without_targets;
        left_out.w_without_targets += empty.w_without_targets;
        left_out.l2l_without_targets += empty.l2l_without_targets;
        left_out.v_without_sources += empty.v_without_sources;
        left_out.w_without_sources += empty.w_without_sources;
        left_out.m2m_without_sources += empty.m2m_without_sources;
    }
    // Every kind of entry was left out somewhere, and kept somewhere (the trees keep
    // entries: the counts are not all zero).
    let EmptyBoxes {
        target_boxes,
        v_without_targets,
        x_without_targets,
        w_without_targets,
        l2l_without_targets,
        v_without_sources,
        w_without_sources,
        m2m_without_sources,
    } = left_out;
    for (what, n) in [
        ("boxes without targets", target_boxes),
        ("V entries of boxes without targets", v_without_targets),
        ("X entries of boxes without targets", x_without_targets),
        ("W entries of boxes without targets", w_without_targets),
        ("L2L rows of boxes without targets", l2l_without_targets),
        ("V entries without sources", v_without_sources),
        ("W entries without sources", w_without_sources),
        ("M2M entries without sources", m2m_without_sources),
    ] {
        assert!(n > 0, "no {what} over the cases");
    }
}

#[test]
fn the_x_entries_of_boxes_without_targets_name_no_ghost_leaf() {
    for (name, map) in cases() {
        let counts = counts_of(&map, count);
        let plan = plan_with(&map, &empty_rule(&counts)).unwrap();
        let without = plan_with(&map, &PlanRules::default()).unwrap();
        let leaves = plan.index().leaves();
        let ghosts: Vec<MortonKey> = leaves.ghosts().map(|j| leaves.key(j)).collect();
        // The ghost leaves are those the rows still name: U and the X entries kept.
        let mut named = Vec::new();
        for level in 0..plan.nlevels() {
            let lists = plan.level(level);
            for &j in lists.near().entries().iter().chain(lists.x().entries()) {
                if j as usize >= leaves.nlocal() {
                    named.push(leaves.key(j as usize));
                }
            }
        }
        named.sort_unstable();
        named.dedup();
        assert_eq!(ghosts, named, "{name}: ghost leaves");
        let old = without.index().leaves();
        assert!(ghosts.len() <= old.ghosts().len(), "{name}");
    }
}

#[test]
fn counts_without_a_zero_give_the_plan_without_the_rule() {
    for (name, map) in cases() {
        let counts = counts_of(&map, |key, salt| count(key, salt) + 1);
        let plan = plan_with(&map, &empty_rule(&counts)).unwrap();
        let without = plan_with(&map, &PlanRules::default()).unwrap();
        assert_eq!(plan, without, "{name}");
        assert_eq!(plan.empty_boxes(), EmptyBoxes::default(), "{name}");
    }
}

#[test]
fn a_missing_count_is_an_error() {
    for (name, map) in cases() {
        // The leaves' counts alone: the interior boxes' subtree counts are missing.
        let full = counts_of(&map, |key, salt| count(key, salt) + 1);
        let mut counts = LeafCounts::new();
        for (&key, &kind) in &map {
            if is_leaf(kind) {
                counts.set_box_sources(key, full.box_sources(key).unwrap());
                counts.set_box_targets(key, full.box_targets(key).unwrap());
            }
        }
        // Every interior box but a `Global` root reads its count: the root of the one-rank
        // map is its block, so a `Global` root is pruned (P2) and reads none.
        let read = map.iter().any(|(&key, &kind)| {
            !is_leaf(kind) && (morton::level(key) > 0 || kind != KeyType::Global)
        });
        match plan_with(&map, &empty_rule(&counts)) {
            Err(PlanError::MissingCount { key }) => {
                assert!(read, "{name}");
                assert!(!is_leaf(map[&key]), "{name}: {key} is a leaf");
            }
            Ok(_) => assert!(!read, "{name}: no error"),
            other => panic!("{name}: {other:?}"),
        }
    }
}

/// The near rule of these tests: W pairs with fewer than 3 sources, X pairs with fewer than
/// 3 targets.
fn predicate(pair: LeafPair) -> bool {
    match pair {
        LeafPair::W { sources, .. } => sources < 3,
        LeafPair::X { targets, .. } => targets < 3,
    }
}

#[test]
fn the_near_rule_sees_the_entries_the_rule_keeps() {
    for (name, map) in cases() {
        let counts = counts_of(&map, count);
        let rule = NearRule::new(&counts, &predicate);
        let both = plan_with(
            &map,
            &PlanRules {
                near: Some(&rule),
                empty_boxes: Some(&counts),
            },
        )
        .unwrap();
        let empty = plan_with(&map, &empty_rule(&counts)).unwrap();
        let is_leaf_key = |key: MortonKey| map.get(&key).copied().is_some_and(is_leaf);
        // The empty rule's rows with the near rule applied to the W entries that are leaves
        // and the X entries of local leaves.
        let mut converted = (0, 0);
        let rows = rows_by_key(&both);
        for (key, mut expected) in rows_by_key(&empty) {
            if expected.len() > 5 {
                let w = std::mem::take(&mut expected[6]);
                for (s, g) in w {
                    let near = is_leaf_key(s)
                        && predicate(LeafPair::W {
                            target: key,
                            source: s,
                            sources: counts.sources(s).unwrap(),
                        });
                    if near {
                        expected[5].push((s, g));
                        converted.0 += 1;
                    } else {
                        expected[6].push((s, g));
                    }
                }
                let x = std::mem::take(&mut expected[1]);
                for (t, g) in x {
                    let pair = LeafPair::X {
                        target: key,
                        source: t,
                        targets: counts.targets(key).unwrap(),
                    };
                    if predicate(pair) {
                        expected[5].push((t, g));
                        converted.1 += 1;
                    } else {
                        expected[1].push((t, g));
                    }
                }
                expected[5].sort_unstable_by_key(|&(k, _)| (morton::level(k), k));
            }
            assert_eq!(rows[&key], expected, "{name}: rows of {key}");
        }
        assert_eq!(both.empty_boxes(), empty.empty_boxes(), "{name}");
        let pairs = both.near_pairs();
        assert_eq!((pairs.w_converted, pairs.x_converted), converted, "{name}");
    }
}
