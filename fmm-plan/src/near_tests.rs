//! Pure local tests of the near rule on the synthetic key maps of
//! `interaction_manager_tests.rs` (through the cases of `plan_tests.rs`: each tree all
//! local, with a `Global` root, and with a ghost half). No MPI.
use std::{cell::RefCell, collections::HashMap};

use nd_octree::{MortonKey, octree::KeyType};

use super::{LeafCounts, LeafPair, NearPairs, NearRule};
use crate::interaction_manager::{is_leaf, tests::ListMaps};
use crate::plan::{
    Plan, PlanError,
    tests::{cases, nlevels, one_rank_map},
};

/// A count of 0–39 that mixes the bits of `key` and `salt`.
fn count(key: MortonKey, salt: u64) -> usize {
    ((key ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58) as usize % 40
}

/// Source and target counts for every leaf of `map`, ghosts included.
fn counts_of(map: &HashMap<MortonKey, KeyType>) -> LeafCounts {
    let mut counts = LeafCounts::new();
    for (&key, &kind) in map {
        if is_leaf(kind) {
            counts.set_sources(key, count(key, 1));
            counts.set_targets(key, count(key, 2));
        }
    }
    counts
}

/// The rule of these tests: W pairs with fewer than 20 sources, X pairs with fewer than
/// 15 targets.
fn predicate(pair: LeafPair) -> bool {
    match pair {
        LeafPair::W { sources, .. } => sources < 20,
        LeafPair::X { targets, .. } => targets < 15,
    }
}

fn plan_with(map: &HashMap<MortonKey, KeyType>, rule: &NearRule<'_>) -> Result<Plan, PlanError> {
    Plan::from_key_types_with_near_rule(map, nlevels(map), &one_rank_map(map), rule)
}

fn plan_without(map: &HashMap<MortonKey, KeyType>) -> Plan {
    Plan::from_key_types(map, nlevels(map), &one_rank_map(map)).unwrap()
}

fn sorted(mut keys: Vec<MortonKey>) -> Vec<MortonKey> {
    keys.sort_unstable();
    keys
}

/// The near, W and X rows of every non-ghost box of `plan` as keys: (near without the box
/// itself, W, X), the first two empty for a box that is not a local leaf.
fn rows(plan: &Plan) -> HashMap<MortonKey, [Vec<MortonKey>; 3]> {
    let index = plan.index();
    let leaves = index.leaves();
    let mut rows = HashMap::new();
    for level in 0..index.nlevels() {
        let lists = plan.level(level);
        for (i, &key) in index.keys(level).iter().enumerate() {
            if index.kind(level, i).is_ghost() {
                continue;
            }
            let x = lists.x().row(i).iter().map(|&j| leaves.key(j as usize));
            let (near, w) = match index.box_leaf(level, i) {
                Some(leaf) => {
                    let r = leaf as usize - leaves.local(level).start;
                    let near = lists.near().row(r).iter().map(|&j| leaves.key(j as usize));
                    let w = lists
                        .w()
                        .row(r)
                        .iter()
                        .map(|&s| index.key(level + 1, s as usize));
                    (near.filter(|&e| e != key).collect(), w.collect())
                }
                None => (Vec::new(), Vec::new()),
            };
            rows.insert(key, [near, w, x.collect()]);
        }
    }
    rows
}

#[test]
fn converted_entries_are_exactly_those_the_predicate_names() {
    let mut converted = NearPairs::default();
    for (name, map) in cases() {
        let counts = counts_of(&map);
        let calls = RefCell::new(Vec::new());
        let recording = |pair: LeafPair| {
            calls.borrow_mut().push(pair);
            predicate(pair)
        };
        let plan = plan_with(&map, &NearRule::new(&counts, &recording)).unwrap();
        let reference = ListMaps::new(&map);
        let without = plan_without(&map);
        let is_leaf_key = |key: MortonKey| map.get(&key).copied().is_some_and(is_leaf);

        // The pairs the predicate must see, once each, with the right counts.
        let mut expected_calls = Vec::new();
        let mut expected = NearPairs::default();
        for (key, [near, w, x]) in rows(&plan) {
            let leaf = matches!(map[&key], KeyType::LocalLeaf);
            let (mut near_rule, mut w_rule, mut x_rule) = (Vec::new(), Vec::new(), Vec::new());
            if leaf {
                near_rule.extend(&reference.u_list()[&key]);
                for &s in &reference.w_list()[&key] {
                    if !is_leaf_key(s) {
                        w_rule.push(s);
                        continue;
                    }
                    let pair = LeafPair::W {
                        target: key,
                        source: s,
                        sources: counts.sources(s).unwrap(),
                    };
                    expected_calls.push(pair);
                    expected.w_leaf_pairs += 1;
                    if predicate(pair) {
                        expected.w_converted += 1;
                        near_rule.push(s);
                    } else {
                        w_rule.push(s);
                    }
                }
            }
            for &t in &reference.x_list()[&key] {
                if !leaf {
                    x_rule.push(t);
                    continue;
                }
                let pair = LeafPair::X {
                    target: key,
                    source: t,
                    targets: counts.targets(key).unwrap(),
                };
                expected_calls.push(pair);
                expected.x_leaf_pairs += 1;
                if predicate(pair) {
                    expected.x_converted += 1;
                    near_rule.push(t);
                } else {
                    x_rule.push(t);
                }
            }
            assert_eq!(sorted(near), sorted(near_rule), "{name}: near of {key}");
            assert_eq!(w, w_rule, "{name}: W of {key}");
            assert_eq!(sorted(x), sorted(x_rule), "{name}: X of {key}");
        }
        let mut calls = calls.into_inner();
        let key_of = |pair: &LeafPair| match *pair {
            LeafPair::W { target, source, .. } => (0, target, source),
            LeafPair::X { target, source, .. } => (1, target, source),
        };
        calls.sort_unstable_by_key(key_of);
        expected_calls.sort_unstable_by_key(key_of);
        assert_eq!(calls, expected_calls, "{name}: the predicate's calls");
        assert_eq!(plan.near_pairs(), expected, "{name}");
        assert_eq!(without.near_pairs(), NearPairs::default(), "{name}");

        // Every other view is the view without the rule.
        for (level, (with, without)) in plan.levels().iter().zip(without.levels()).enumerate() {
            assert_eq!(with.v(), without.v(), "{name}: level {level}");
            assert_eq!(
                with.m2m_local(),
                without.m2m_local(),
                "{name}: level {level}"
            );
            assert_eq!(
                with.m2m_global(),
                without.m2m_global(),
                "{name}: level {level}"
            );
            assert_eq!(with.l2l(), without.l2l(), "{name}: level {level}");
            assert_eq!(with.l2p(), without.l2p(), "{name}: level {level}");
        }

        // The ghost leaves: those of U and X, and the converted ghost W entries.
        let mut ghosts: Vec<MortonKey> = without
            .index()
            .leaves()
            .ghosts()
            .map(|j| without.index().leaf_key(j))
            .collect();
        for (key, [near, ..]) in rows(&plan) {
            ghosts.extend(
                near.into_iter()
                    .filter(|e| map[e].is_ghost() && !reference.u_list()[&key].contains(e)),
            );
        }
        let ghosts = sorted(ghosts);
        let mut ghosts = ghosts;
        ghosts.dedup();
        let numbered: Vec<MortonKey> = plan
            .index()
            .leaves()
            .ghosts()
            .map(|j| plan.index().leaf_key(j))
            .collect();
        assert_eq!(numbered, ghosts, "{name}: ghost leaves");
        let local = plan.index().leaves().nlocal();
        assert_eq!(local, without.index().leaves().nlocal(), "{name}");
        assert_eq!(
            plan.index().leaves().keys()[..local],
            without.index().leaves().keys()[..local],
            "{name}: local leaves"
        );

        converted.w_leaf_pairs += expected.w_leaf_pairs;
        converted.w_converted += expected.w_converted;
        converted.x_leaf_pairs += expected.x_leaf_pairs;
        converted.x_converted += expected.x_converted;
    }
    // Both directions convert some pairs and keep others.
    assert!(
        0 < converted.w_converted
            && converted.w_converted < converted.w_leaf_pairs
            && 0 < converted.x_converted
            && converted.x_converted < converted.x_leaf_pairs,
        "{converted:?}"
    );
}

#[test]
fn duality_holds_for_the_pairs_not_converted() {
    for (name, map) in cases() {
        if map.values().any(|kind| kind.is_ghost()) {
            continue;
        }
        let counts = counts_of(&map);
        let plan = plan_with(&map, &NearRule::new(&counts, &predicate)).unwrap();
        let rows = rows(&plan);
        for (&t, [_, w, _]) in &rows {
            for &s in w {
                let x = &rows[&s][2];
                let converted = matches!(map[&s], KeyType::LocalLeaf)
                    && predicate(LeafPair::X {
                        target: s,
                        source: t,
                        targets: counts.targets(s).unwrap(),
                    });
                assert_eq!(x.contains(&t), !converted, "{name}: W pair ({t}, {s})");
            }
        }
        for (&s, [_, _, x]) in &rows {
            for &t in x {
                let w = &rows[&t][1];
                let converted = matches!(map[&s], KeyType::LocalLeaf)
                    && predicate(LeafPair::W {
                        target: t,
                        source: s,
                        sources: counts.sources(s).unwrap(),
                    });
                assert_eq!(w.contains(&s), !converted, "{name}: X pair ({s}, {t})");
            }
        }
    }
}

#[test]
fn near_rows_stay_ordered_by_level_and_key() {
    let mut grown = 0;
    for (name, map) in cases() {
        let counts = counts_of(&map);
        let always = |_: LeafPair| true;
        let plan = plan_with(&map, &NearRule::new(&counts, &always)).unwrap();
        let leaves = plan.index().leaves();
        for level in 0..plan.nlevels() {
            for view in [plan.level(level).near(), plan.level(level).x()] {
                for t in 0..view.nrows() {
                    let row = view.row(t);
                    let mut by_key = row.to_vec();
                    by_key.sort_unstable_by_key(|&j| {
                        (leaves.level(j as usize), leaves.key(j as usize))
                    });
                    by_key.dedup();
                    assert_eq!(by_key, row, "{name}: level {level}, row {t}");
                }
            }
            // Every X entry of a leaf converts, and every leaf W entry.
            let lists = plan.level(level);
            for r in 0..lists.w().nrows() {
                for &s in lists.w().row(r) {
                    assert!(!is_leaf(plan.index().kind(level + 1, s as usize)));
                }
            }
            grown += lists.near().entries().len();
        }
        let pairs = plan.near_pairs();
        assert_eq!(pairs.w_converted, pairs.w_leaf_pairs, "{name}");
        assert_eq!(pairs.x_converted, pairs.x_leaf_pairs, "{name}");
    }
    assert!(grown > 0);
}

#[test]
fn a_rule_that_converts_nothing_leaves_the_views_unchanged() {
    for (name, map) in cases() {
        let counts = counts_of(&map);
        let never = |_: LeafPair| false;
        let plan = plan_with(&map, &NearRule::new(&counts, &never)).unwrap();
        let without = plan_without(&map);
        assert_eq!(plan.index(), without.index(), "{name}");
        assert_eq!(plan.levels(), without.levels(), "{name}");
        assert_eq!(
            plan.near_pairs().w_converted + plan.near_pairs().x_converted,
            0
        );
    }
}

#[test]
fn a_missing_count_is_an_error() {
    let mut checked = 0;
    for (name, map) in cases() {
        let full = counts_of(&map);
        let always = |_: LeafPair| true;
        let candidates = plan_with(&map, &NearRule::new(&full, &always))
            .unwrap()
            .near_pairs();
        if candidates.w_leaf_pairs + candidates.x_leaf_pairs == 0 {
            assert!(plan_with(&map, &NearRule::new(&LeafCounts::new(), &always)).is_ok());
            continue;
        }
        let error = plan_with(&map, &NearRule::new(&LeafCounts::new(), &always)).unwrap_err();
        let PlanError::MissingCount { key } = error else {
            panic!("{name}: {error:?}");
        };
        assert!(is_leaf(map[&key]), "{name}: {key}");
        checked += 1;
    }
    assert!(checked > 0);
}
