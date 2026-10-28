//! The acceptance tests of CH0.8a (docs/helmholtz/tasks/CH0.8a.md, "Tests that prove it").
//! `trees` owns the one MPI initialisation of this executable; the others need no MPI.
//! Run them with `cargo test -p nd-fmm-spike-census --release`.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use mpi::traits::Communicator;
use nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS;
use nd_fmm_spike_census::census::{
    Census, DISTANCES, NCLASSES, NDIST, distance_class, reflection_classes,
};
use nd_fmm_spike_census::model::{
    bound, coaxial_entries, conversion_cost, degree, hankel_abs, level_ka, n_phi_host, passing,
    rotation_block_reals, signature_len, width,
};
use nd_fmm_spike_census::tree::{
    DEEPEST, fine_keys, octree_of_points, octree_with_width, width_level,
};
use nd_fmm_spike_census::workloads::Workload;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_octree::octree::KeyType;
use nd_octree::{MortonKey, Octree, morton};

const MIB: f64 = 1024.0 * 1024.0;

/// The leaf keys of a tree, sorted.
fn leaves<C: mpi::traits::CommunicatorCollectives>(tree: &Octree<'_, C>) -> Vec<MortonKey> {
    let mut l = tree.leaf_keys().clone();
    l.sort_unstable();
    l
}

/// The closed interval of a key on each axis, in level-16 units.
fn extent(key: MortonKey) -> [(usize, usize); 3] {
    let (level, index) = morton::decode(key);
    let size = 1 << (DEEPEST - level);
    index.map(|i| (i * size, (i + 1) * size))
}

/// The adjacency of the list rule: the closed cubes meet and neither is an ancestor of the
/// other (nor the same box).
fn adjacent(a: MortonKey, b: MortonKey) -> bool {
    if a == b || morton::is_ancestor(a, b) || morton::is_ancestor(b, a) {
        return false;
    }
    let (ea, eb) = (extent(a), extent(b));
    (0..3).all(|i| ea[i].0 <= eb[i].1 && eb[i].0 <= ea[i].1)
}

/// The counts of `census` recomputed by brute force from the list definitions of
/// `nd_fmm_plan::interaction_manager`, over every key of the one-rank tree.
fn check_against_brute_force<C: mpi::traits::CommunicatorCollectives>(
    name: &str,
    tree: &Octree<'_, C>,
    keys: &[MortonKey],
    census: &Census,
) {
    let all = tree.all_keys();
    let leaf_set: Vec<MortonKey> = {
        let mut v: Vec<MortonKey> = all
            .iter()
            .filter(|(_, t)| **t == KeyType::LocalLeaf)
            .map(|(k, _)| *k)
            .collect();
        v.sort_unstable();
        v
    };
    let is_leaf = |k: MortonKey| leaf_set.binary_search(&k).is_ok();
    let mut by_level: Vec<Vec<MortonKey>> = vec![Vec::new(); census.levels.len()];
    for &k in all.keys() {
        by_level[morton::level(k)].push(k);
    }
    // Points per box.
    let mut count: HashMap<MortonKey, u64> = HashMap::new();
    for &k in keys {
        for l in 0..census.levels.len() {
            *count
                .entry(morton::ancestor_at_level(k, l).unwrap())
                .or_default() += 1;
        }
    }
    let n = |k: MortonKey| count.get(&k).copied().unwrap_or(0);
    let all_boxes: Vec<MortonKey> = all.keys().copied().collect();

    for (l, c) in census.levels.iter().enumerate() {
        let boxes = &by_level[l];
        assert_eq!(c.boxes, boxes.len() as u64, "{name}: boxes on level {l}");
        let occupied = boxes.iter().filter(|&&k| n(k) > 0).count() as u64;
        assert_eq!(c.occupied, occupied, "{name}: occupied on level {l}");

        // V: every entry of the plan's rows, and between occupied boxes by distance
        // class, by reflection class (34 tuples (min(|x|, |y|), max(|x|, |y|), |z|), built
        // here), and the masks of distance classes per source and per target box.
        let mut v = [0u64; NDIST];
        let mut v_entries = 0;
        let mut by_class: BTreeMap<[i64; 3], u64> = BTreeMap::new();
        for d in V_LIST_DIRECTIONS {
            let (a, b) = (d[0].abs(), d[1].abs());
            by_class.insert([a.min(b), a.max(b), d[2].abs()], 0);
        }
        let mut source_mask: HashMap<MortonKey, u16> = HashMap::new();
        let mut target_masks: BTreeMap<u16, u64> = BTreeMap::new();
        if l >= 2 {
            for &b in boxes {
                let p = morton::parent(b).unwrap();
                let mut target_mask = 0u16;
                for &s in boxes {
                    let ps = morton::parent(s).unwrap();
                    if s == b || !(adjacent(ps, p) || ps == p) || adjacent(s, b) {
                        continue;
                    }
                    v_entries += 1;
                    if n(b) == 0 || n(s) == 0 {
                        continue;
                    }
                    let (_, ib) = morton::decode(b);
                    let (_, is) = morton::decode(s);
                    let d = [0, 1, 2].map(|i| ib[i] as i64 - is[i] as i64);
                    let dc = distance_class(d);
                    v[dc] += 1;
                    let (x, y) = (d[0].abs(), d[1].abs());
                    *by_class.get_mut(&[x.min(y), x.max(y), d[2].abs()]).unwrap() += 1;
                    target_mask |= 1 << dc;
                    *source_mask.entry(s).or_default() |= 1 << dc;
                }
                if target_mask != 0 {
                    *target_masks.entry(target_mask).or_default() += 1;
                }
            }
        }
        let mut source_masks: BTreeMap<u16, u64> = BTreeMap::new();
        for &m in source_mask.values() {
            *source_masks.entry(m).or_default() += 1;
        }
        assert_eq!(by_class.len(), NCLASSES);
        assert_eq!(c.v_entries, v_entries, "{name}: V entries on level {l}");
        assert_eq!(
            c.v_by_distance, v,
            "{name}: V pairs per distance on level {l}"
        );
        assert_eq!(
            c.v_by_class,
            by_class.values().copied().collect::<Vec<u64>>(),
            "{name}: V pairs per reflection class on level {l}"
        );
        assert_eq!(
            c.source_masks, source_masks,
            "{name}: source masks on level {l}"
        );
        assert_eq!(
            c.target_masks, target_masks,
            "{name}: target masks on level {l}"
        );

        // Leaves, U, P2P, W and X, every entry and between occupied boxes.
        let (mut u, mut pairs, mut w, mut w_points, mut x, mut x_points) = (0, 0, 0, 0, 0, 0);
        let (mut u_all, mut w_all, mut x_all) = (0, 0, 0);
        let (mut leaves, mut occupied_leaves, mut leaf_points, mut max_leaf_points) = (0, 0, 0, 0);
        for &b in boxes.iter().filter(|&&b| is_leaf(b)) {
            let nb = n(b);
            leaves += 1;
            occupied_leaves += u64::from(nb > 0);
            leaf_points += nb;
            max_leaf_points = max_leaf_points.max(nb);
            pairs += nb * nb;
            for &s in &leaf_set {
                if adjacent(s, b) {
                    u_all += 1;
                    pairs += nb * n(s);
                    if nb > 0 && n(s) > 0 {
                        u += 1;
                    }
                }
            }
            for &s in &all_boxes {
                if morton::level(s) > l
                    && adjacent(morton::parent(s).unwrap(), b)
                    && !adjacent(s, b)
                {
                    w_all += 1;
                    if nb > 0 && n(s) > 0 {
                        w += 1;
                        w_points += nb;
                    }
                }
            }
        }
        if l >= 2 {
            for &b in boxes {
                let p = morton::parent(b).unwrap();
                for &s in &leaf_set {
                    if morton::level(s) == l - 1 && adjacent(s, p) && !adjacent(s, b) {
                        x_all += 1;
                        if n(b) > 0 && n(s) > 0 {
                            x += 1;
                            x_points += n(s);
                        }
                    }
                }
            }
        }
        assert_eq!(
            (
                c.leaves,
                c.occupied_leaves,
                c.leaf_points,
                c.max_leaf_points
            ),
            (leaves, occupied_leaves, leaf_points, max_leaf_points),
            "{name}: leaves on level {l}"
        );
        assert_eq!(
            (c.u_entries, c.w_entries, c.x_entries),
            (u_all, w_all, x_all),
            "{name}: U, W and X entries on level {l}"
        );
        assert_eq!(c.u_entries_occupied, u, "{name}: U on level {l}");
        assert_eq!(c.p2p_pairs, pairs, "{name}: P2P pairs on level {l}");
        assert_eq!(
            (c.w_entries_occupied, c.w_points),
            (w, w_points),
            "{name}: W on level {l}"
        );
        assert_eq!(
            (c.x_entries_occupied, c.x_points),
            (x, x_points),
            "{name}: X on level {l}"
        );
    }
}

/// Tests 1–3: the prototype tree, the width rule and the counts. The one MPI-owning test.
#[test]
fn trees() {
    let universe = mpi::initialize().expect("MPI initialises once");
    let comm = universe.world();
    assert_eq!(comm.size(), 1, "the census runs on one rank");
    let n = 128;

    // Test 1: no width limit reproduces nd-octree's own tree, bit for bit.
    let mut cases: Vec<(Workload, f64)> = vec![
        (Workload::W1, 50.0),
        (Workload::W2, 50.0),
        (Workload::w3_resolved(20.0), 20.0),
    ];
    for distribution in Distribution::ALL {
        cases.push((
            Workload::W4 {
                distribution,
                n: 10_000,
            },
            10.0,
        ));
    }
    for &(workload, kd) in &cases {
        let keys = fine_keys(&workload.points(kd));
        let own = octree_of_points(&keys, n, &comm);
        let prototype = octree_with_width(&keys, n, None, &comm);
        assert_eq!(
            leaves(&own),
            leaves(&prototype),
            "{}: the prototype without a width",
            workload.name()
        );
    }

    // Test 2: the width rule.
    let mut refined = 0;
    for &(workload, kd) in &cases[..3] {
        let keys = fine_keys(&workload.points(kd));
        let plain = leaves(&octree_of_points(&keys, n, &comm));
        for c_leaf in [0.5, 2.0] {
            let min_level = width_level(Some(c_leaf), kd);
            let tree = octree_with_width(&keys, n, min_level, &comm);
            let census = Census::new(&tree, &keys);
            let lambda = 2.0 * PI / kd;
            for (l, c) in census.levels.iter().enumerate() {
                if c.occupied_leaves > 0 {
                    assert!(
                        width(l) <= c_leaf * lambda,
                        "{}: c_leaf {c_leaf}, level {l}",
                        workload.name()
                    );
                }
            }
            // Leaves with more than n distinct keys only on the deepest level.
            let mut distinct = keys.clone();
            distinct.sort_unstable();
            distinct.dedup();
            for &leaf in tree.leaf_keys() {
                if morton::level(leaf) < DEEPEST {
                    let inside = distinct
                        .iter()
                        .filter(|&&k| {
                            morton::ancestor_at_level(k, morton::level(leaf)) == Some(leaf)
                        })
                        .count();
                    assert!(inside <= n, "{}: leaf with {inside} keys", workload.name());
                }
            }
            // A refinement of the tree without the rule.
            let all = tree.all_keys();
            assert!(
                plain.iter().all(|k| all.contains_key(k)),
                "{}: not a refinement",
                workload.name()
            );
            if plain != leaves(&tree) {
                refined += 1;
            }
        }
    }
    // W1 and W2 at kD = 50 are refined at c_leaf = 0.5; resolved W3 needs no refinement.
    assert!(refined >= 2, "the width rule refined {refined} trees");

    // Test 3: the counts against brute force, at a small leaf size for deep trees.
    let small: Vec<(Workload, f64, usize, Option<f64>)> = vec![
        (Workload::W1, 20.0, 16, None),
        (Workload::W2, 20.0, 16, Some(1.0)),
        (
            Workload::W4 {
                distribution: Distribution::Clusters,
                n: 3_000,
            },
            10.0,
            8,
            None,
        ),
        (
            Workload::W4 {
                distribution: Distribution::Plummer,
                n: 3_000,
            },
            10.0,
            8,
            Some(0.5),
        ),
    ];
    for (workload, kd, n, c_leaf) in small {
        let keys = fine_keys(&workload.points(kd));
        let tree = octree_with_width(&keys, n, width_level(c_leaf, kd), &comm);
        let census = Census::new(&tree, &keys);
        let name = format!("{} n = {n} c_leaf = {c_leaf:?}", workload.name());
        check_against_brute_force(&name, &tree, &keys, &census);
        // The adaptive trees have W and X pairs, so those counts are exercised.
        let wx: u64 = census
            .levels
            .iter()
            .map(|c| c.w_entries_occupied + c.x_entries_occupied)
            .sum();
        println!("{name}: {} levels, W + X pairs {wx}", census.levels.len());
        if matches!(workload, Workload::W4 { .. } | Workload::W2) {
            assert!(wx > 0, "{name}: no W or X pair");
        }
    }
}

/// Test 4: 15 distance classes and 34 reflection classes over the 316 offsets.
#[test]
fn offsets() {
    let mut d2: Vec<i64> = V_LIST_DIRECTIONS
        .iter()
        .map(|d| d.iter().map(|x| x * x).sum())
        .collect();
    d2.sort_unstable();
    d2.dedup();
    assert_eq!(d2, DISTANCES);
    assert_eq!(reflection_classes().len(), NCLASSES);
}

/// Test 5: the formula p reproduces proposal §3.5's tables.
#[test]
fn formula_degree() {
    let p: Vec<usize> = (2..=8).map(|l| degree(level_ka(500.0, l), 1e-6)).collect();
    assert_eq!(p, [144, 82, 50, 33, 25, 23, 22]);
    let table = [
        (50.0, [24, 30, 35]),
        (100.0, [37, 43, 48]),
        (200.0, [63, 69, 75]),
        (500.0, [135, 144, 151]),
        (1000.0, [251, 261, 271]),
        (3000.0, [699, 714, 728]),
    ];
    for (kd, row) in table {
        let got = [1e-4, 1e-6, 1e-8].map(|eps| degree(level_ka(kd, 2), eps));
        assert_eq!(got, row, "kD = {kd}");
    }
}

/// Test 6: |h_n| against the arbitrary-precision reference of spike S3 (real argument).
#[test]
fn hankel_against_reference() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../radial/fixtures/reference.json"
    );
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let exact = |v: &serde_json::Value| -> f64 { v.as_str().unwrap().parse().unwrap() };
    let (mut sets, mut values, mut worst) = (0, 0, 0f64);
    for set in fixture["sets"].as_array().unwrap() {
        if set["family"] != "h" || exact(&set["z"][1]) != 0.0 {
            continue;
        }
        let x = exact(&set["z"][0]);
        let s = exact(&set["s"]);
        let rows = set["rows"].as_array().unwrap();
        let top = rows
            .iter()
            .map(|r| r[0].as_u64().unwrap() as usize)
            .max()
            .unwrap();
        let h = hankel_abs(x, top);
        sets += 1;
        for r in rows {
            let order = r[0].as_u64().unwrap() as usize;
            let scaled = r[1].as_f64().unwrap().hypot(r[2].as_f64().unwrap());
            // ĥ = h sⁿ for real argument.
            let reference = scaled / s.powi(order as i32);
            if scaled == 0.0 || !reference.is_finite() || !h[order].is_finite() {
                continue;
            }
            let e = (h[order] - reference).abs() / reference;
            worst = worst.max(e);
            values += 1;
        }
    }
    println!("{sets} sets, {values} values, worst relative error {worst:.2e}");
    assert!(sets > 10 && values > 1000);
    assert!(worst < 1e-12, "worst {worst:.2e}");
}

/// Test 7: the bound reproduces proposal §3.6.
#[test]
fn bound_reproduces_the_proposal() {
    let k = 500.0;
    let failing = |l: usize, eps: f64| -> Vec<i64> {
        let pass = passing(k, width(l), degree(level_ka(k, l), eps), eps);
        (0..NDIST)
            .filter(|i| pass & (1 << i) == 0)
            .map(|i| DISTANCES[i])
            .collect()
    };
    // "classes on RCR" at kD = 500, ε = 10⁻⁶.
    assert_eq!(failing(2, 1e-6), Vec::<i64>::new());
    assert_eq!(failing(3, 1e-6), [4]);
    assert_eq!(failing(4, 1e-6), [4, 5]);
    assert_eq!(failing(5, 1e-6), [4, 5, 6]);
    // At 10⁻⁸: level 2 class 4, level 3 classes 4 and 5, level 4 classes 4 to 6, level 5
    // eight classes.
    assert_eq!(failing(2, 1e-8), [4]);
    assert_eq!(failing(3, 1e-8), [4, 5]);
    assert_eq!(failing(4, 1e-8), [4, 5, 6]);
    assert_eq!(failing(5, 1e-8).len(), 8);

    // "nearest class has E ≤ ε from": 1.8 λ, 17 λ, 42 λ.
    for (eps, stated, half_digit) in [(1e-4, 1.8, 0.05), (1e-6, 17.0, 0.5), (1e-8, 42.0, 0.5)] {
        let k = 2.0 * PI; // λ = 1, so w is in wavelengths
        // E is not monotone in w (p jumps by integers), so the proposal's two-digit width
        // must lie between the first width that passes and the width from which every
        // wider box passes, to half its last digit.
        let widths: Vec<f64> = (50..=10000).map(|i| i as f64 * 0.01).collect();
        let fails = |w: f64| bound(k, w, degree(k * 3f64.sqrt() / 2.0 * w, eps), 4) > eps;
        let first = widths[widths.iter().position(|&w| !fails(w)).unwrap()];
        let from = widths[widths.iter().rposition(|&w| fails(w)).unwrap() + 1];
        println!(
            "ε = {eps:e}: the nearest class first passes at {first:.2} λ, always from \
             {from:.2} λ (stated {stated} λ)"
        );
        assert!(
            first - half_digit <= stated && stated <= from + half_digit,
            "ε = {eps:e}: {stated} λ outside [{first}, {from}] λ"
        );
    }
}

/// Test 8: the memory and cost formulas reproduce §7's and §6.7's worked numbers.
#[test]
fn cost_and_memory_formulas() {
    let mib = |reals: f64| reals * 8.0 / MIB;
    assert!((mib(rotation_block_reals(20)) - 0.09).abs() < 0.01);
    assert!((mib(rotation_block_reals(144)) - 31.0).abs() < 1.0);
    assert!((mib(rotation_block_reals(261)) - 183.0).abs() < 2.0);
    // About 3 MiB per (S|R) distance at p = 82; 15 distances 235 MiB at p = 144.
    assert!((coaxial_entries(82) * 16.0 / MIB - 3.0).abs() < 0.2);
    assert!((15.0 * coaxial_entries(144) * 16.0 / MIB - 235.0).abs() < 3.0);
    // W1 at kD = 500, level 2 (§7): 56 boxes, mean V list 44.7, every class diagonal,
    // N_φ = 4P − 3 unpadded: 6.6·10⁹ and 0.28 GiB of signature scratch.
    let (p, boxes, pairs) = (144, 56.0, 56.0 * 44.7);
    let n_phi = 4 * (p + 1) - 3;
    let m2l = 2.0 * boxes * conversion_cost(p, n_phi) + pairs * signature_len(p, n_phi);
    // §7 rounds the φ step to 16P³; (2P − 1)² N_φ is 1.3 % below it here.
    assert!((m2l / 6.6e9 - 1.0).abs() < 0.03, "{m2l:e}");
    let scratch = 2.0 * boxes * signature_len(p, n_phi) * 16.0 / (1024.0 * MIB);
    assert!((scratch - 0.28).abs() < 0.01, "{scratch}");
    // §6.7's grid sizes.
    assert_eq!(n_phi_host(144), 588);
    assert_eq!(n_phi_host(261), 1080);
    assert_eq!(n_phi_host(50), 216);
}
