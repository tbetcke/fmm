//! The census sweep of CH0.8a (docs/helmholtz/tasks/CH0.8a.md): the trees of W1–W4 over
//! the kD sweep and c_leaf, counted and modelled, written as Markdown to
//! `spikes/census/results/`.
//!
//! ```sh
//! cargo run -p nd-fmm-spike-census --release --example census -- sweep
//! cargo run -p nd-fmm-spike-census --release --example census -- probe W1 1000 0.5 128
//! ```
//!
//! One rank, one thread. `probe <W1|W2|W3|W3u> <kD> <c_leaf|none> <n>` builds one tree
//! and prints its census (W3u: 10⁶ uniform points).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use nd_fmm_spike_census::census::{Census, DISTANCES, NDIST};
use nd_fmm_spike_census::model::{Regime, TreeModel, level_ka};
use nd_fmm_spike_census::tree::{fine_keys, octree_with_width, width_level};
use nd_fmm_spike_census::workloads::Workload;
use nd_fmm_validate::fmm_accuracy::Distribution;

/// The leaf sizes: 128 (Laplace's host default from p = 12) and 64 (sensitivity).
const N_LEAF: usize = 128;
const N_LEAF_SMALL: usize = 64;

/// The candidate maximum leaf widths, in wavelengths; `None` is no limit.
const C_LEAF: [Option<f64>; 7] = [
    Some(0.25),
    Some(0.5),
    Some(0.75),
    Some(1.0),
    Some(1.5),
    Some(2.0),
    None,
];

/// The tolerances.
const EPS: [f64; 3] = [1e-4, 1e-6, 1e-8];

/// The tolerance of the per-level tables and of D11.
const EPS_MAIN: f64 = 1e-6;

/// D11's pair weights (complex multiply-adds per P2P pair) and the default.
const PAIR_WEIGHTS: [f64; 3] = [2.0, 4.0, 8.0];
const PAIR_WEIGHT: f64 = 4.0;

/// Ranks per node for the replicated top: Kathleen and locust.
const RANKS_PER_NODE: [(&str, f64); 2] = [("Kathleen", 40.0), ("locust", 72.0)];

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: f64 = 1024.0 * 1024.0;

/// One counted tree.
struct Run {
    workload: Workload,
    kd: f64,
    c_leaf: Option<f64>,
    n_leaf: usize,
    census: Census,
    seconds: f64,
}

fn c_name(c: Option<f64>) -> String {
    c.map_or("none".into(), |c| format!("{c}"))
}

/// Builds and counts one tree.
fn run(
    comm: &SimpleCommunicator,
    workload: Workload,
    kd: f64,
    keys: &[u64],
    c_leaf: Option<f64>,
    n_leaf: usize,
) -> Run {
    let start = Instant::now();
    let tree = octree_with_width(keys, n_leaf, width_level(c_leaf, kd), comm);
    let census = Census::new(&tree, keys);
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{} kD = {kd} c_leaf = {} n = {n_leaf}: N = {}, leaves on {}–{}, {:.1} s",
        workload.name(),
        c_name(c_leaf),
        keys.len(),
        census.shallowest_leaf_level(),
        census.deepest_leaf_level(),
        seconds
    );
    Run {
        workload,
        kd,
        c_leaf,
        n_leaf,
        census,
        seconds,
    }
}

/// One configuration of the sweep: (workload, kD, the c_leaf values, the leaf sizes).
type Configuration = (Workload, f64, Vec<Option<f64>>, Vec<usize>);

/// The configurations of the sweep.
fn configurations() -> Vec<Configuration> {
    let all = C_LEAF.to_vec();
    let both = vec![N_LEAF, N_LEAF_SMALL];
    let mut v = Vec::new();
    for kd in [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0] {
        // The n = 64 sensitivity up to kD = 500 (kD = 1000 at c_leaf = 0.25 alone takes
        // 217 s and 24 GB at n = 128).
        let n = if kd <= 500.0 {
            both.clone()
        } else {
            vec![N_LEAF]
        };
        v.push((Workload::W1, kd, all.clone(), n));
    }
    for kd in [10.0, 20.0, 50.0, 100.0, 200.0, 500.0] {
        v.push((Workload::W2, kd, all.clone(), vec![N_LEAF]));
    }
    for kd in [10.0, 20.0, 50.0, 100.0] {
        v.push((Workload::w3_resolved(kd), kd, all.clone(), both.clone()));
    }
    for kd in [200.0, 500.0] {
        // At kD = 500 the leaves of small c_leaf hold one point at either n.
        let n = if kd <= 200.0 {
            both.clone()
        } else {
            vec![N_LEAF]
        };
        v.push((Workload::W3 { n: 1_000_000 }, kd, all.clone(), n));
    }
    for distribution in Distribution::ALL {
        for n in [100_000, 1_000_000] {
            for kd in [1.0, 10.0, 100.0] {
                v.push((
                    Workload::W4 { distribution, n },
                    kd,
                    all.clone(),
                    vec![N_LEAF],
                ));
            }
        }
    }
    v
}

fn g(x: f64) -> String {
    if x == 0.0 {
        "0".into()
    } else if x.abs() >= 1e4 || x.abs() < 1e-2 {
        format!("{x:.2e}")
    } else if x.abs() >= 100.0 {
        format!("{x:.0}")
    } else {
        format!("{x:.2}")
    }
}

/// Cost of the leaf work (D11) at ε = 10⁻⁶.
fn leaf_cost(r: &Run, w: f64) -> f64 {
    TreeModel::new(&r.census, r.kd, EPS_MAIN).ops().leaf_cost(w)
}

/// The largest finite c_leaf whose leaf cost is within 10 % of the best over the runs
/// of one (workload, kD, n).
fn largest_within(runs: &[&Run], w: f64) -> f64 {
    let costs: Vec<(Option<f64>, f64)> = runs.iter().map(|r| (r.c_leaf, leaf_cost(r, w))).collect();
    let best = costs.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    costs
        .iter()
        .filter(|(c, cost)| c.is_some() && *cost <= 1.1 * best)
        .map(|(c, _)| c.unwrap())
        .fold(0.0, f64::max)
}

fn group_key(r: &Run) -> (String, u64, usize) {
    (r.workload.name(), r.kd as u64, r.n_leaf)
}

/// The c_leaf study (D11).
fn c_leaf_report(runs: &[Run]) -> (String, f64) {
    let mut out = String::new();
    let mut groups: BTreeMap<(String, u64, usize), Vec<&Run>> = BTreeMap::new();
    for r in runs
        .iter()
        .filter(|r| matches!(r.workload, Workload::W1 | Workload::W3 { .. }))
    {
        groups.entry(group_key(r)).or_default().push(r);
    }
    writeln!(out, "# c_leaf study (D11)\n").unwrap();
    writeln!(
        out,
        "ε = 10⁻⁶, formula p. Leaf cost = w · P2P pairs + P2M + L2P + M2P + P2L (complex \
         multiply-adds), w = {PAIR_WEIGHT} unless stated; far = M2M + L2L + M2L. \"rel.\" is \
         the leaf cost relative to the best c_leaf of the row group (\"none\" included), \
         \"rel. total\" the same for leaf cost + far.\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Workload | kD | n | c_leaf | leaf levels | leaves (occupied) | P2P pairs | leaf cost | rel. | far | total (w = 4) | rel. total | build s |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|---|").unwrap();
    let mut verdicts = Vec::new();
    for ((name, kd, n), group) in &groups {
        let best = group
            .iter()
            .map(|r| leaf_cost(r, PAIR_WEIGHT))
            .fold(f64::INFINITY, f64::min);
        let total = |r: &Run| {
            let ops = TreeModel::new(&r.census, r.kd, EPS_MAIN).ops();
            ops.leaf_cost(PAIR_WEIGHT) + ops.far()
        };
        let best_total = group.iter().map(|r| total(r)).fold(f64::INFINITY, f64::min);
        for r in group {
            let m = TreeModel::new(&r.census, r.kd, EPS_MAIN);
            let ops = m.ops();
            let occupied: u64 = r.census.levels.iter().map(|c| c.occupied_leaves).sum();
            let leaf = ops.leaf_cost(PAIR_WEIGHT);
            writeln!(
                out,
                "| {name} | {kd} | {n} | {} | {}–{} | {} | {} | {} | {:.2} | {} | {} | {:.2} | {:.1} |",
                c_name(r.c_leaf),
                r.census.shallowest_leaf_level(),
                r.census.deepest_leaf_level(),
                occupied,
                g(ops.p2p_pairs),
                g(leaf),
                leaf / best,
                g(ops.far()),
                g(leaf + ops.far()),
                (leaf + ops.far()) / best_total,
                r.seconds
            )
            .unwrap();
        }
        let per_w: Vec<f64> = PAIR_WEIGHTS
            .iter()
            .map(|&w| largest_within(group, w))
            .collect();
        verdicts.push((name.clone(), *kd, *n, per_w));
    }
    writeln!(
        out,
        "\n## The largest c_leaf within 10 % of the best leaf cost\n"
    )
    .unwrap();
    writeln!(out, "| Workload | kD | n | w = 2 | w = 4 | w = 8 |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|").unwrap();
    for (name, kd, n, per_w) in &verdicts {
        writeln!(
            out,
            "| {name} | {kd} | {n} | {} | {} | {} |",
            per_w[0], per_w[1], per_w[2]
        )
        .unwrap();
    }
    // D11: the smallest of the largest values over W1 and W3 at n = 128, w = 4.
    let chosen = verdicts
        .iter()
        .filter(|v| v.2 == N_LEAF)
        .map(|v| v.3[1])
        .fold(f64::INFINITY, f64::min);
    writeln!(
        out,
        "\nD11 by the rule (every W1 and W3 row, n = {N_LEAF}, w = {PAIR_WEIGHT}): **c_leaf = {chosen}**."
    )
    .unwrap();
    (out, chosen)
}

/// The levels with an occupied leaf that are H at `eps`.
fn high_frequency_leaf_levels(r: &Run, eps: f64) -> Vec<usize> {
    let m = TreeModel::new(&r.census, r.kd, eps);
    m.levels
        .iter()
        .zip(&r.census.levels)
        .filter(|(lm, c)| lm.regime == Regime::H && c.occupied_leaves > 0)
        .map(|(lm, _)| lm.level)
        .collect()
}

/// The revised rule proposed for D11: the largest c_leaf at which no tree of the sweep
/// (n = 128) has a high-frequency leaf at ε = 10⁻⁶ or 10⁻⁸; with the trees per c_leaf that
/// have one, per ε.
fn no_high_frequency_leaf(runs: &[Run]) -> (String, f64) {
    let mut out = String::new();
    writeln!(
        out,
        "\n## Trees with a high-frequency leaf, per c_leaf and ε (n = {N_LEAF}, every workload)\n"
    )
    .unwrap();
    writeln!(out, "| c_leaf | trees | 10⁻⁴ | 10⁻⁶ | 10⁻⁸ |").unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();
    let mut chosen = 0.0;
    for c in C_LEAF {
        let trees: Vec<&Run> = runs
            .iter()
            .filter(|r| r.n_leaf == N_LEAF && r.c_leaf == c)
            .collect();
        let counts: Vec<usize> = EPS
            .iter()
            .map(|&eps| {
                trees
                    .iter()
                    .filter(|r| !high_frequency_leaf_levels(r, eps).is_empty())
                    .count()
            })
            .collect();
        writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            c_name(c),
            trees.len(),
            counts[0],
            counts[1],
            counts[2]
        )
        .unwrap();
        if let Some(c) = c
            && counts[1] == 0
            && counts[2] == 0
        {
            chosen = c;
        }
    }
    writeln!(
        out,
        "\nThe revised rule (the largest c_leaf with no high-frequency leaf at 10⁻⁶ and 10⁻⁸ on \
         any workload): **c_leaf = {chosen}**."
    )
    .unwrap();
    (out, chosen)
}

/// Whether any level with an occupied leaf is high frequency, per run and ε.
fn leaf_report(runs: &[Run]) -> String {
    let mut out = String::new();
    writeln!(out, "# High-frequency leaves\n").unwrap();
    writeln!(
        out,
        "A leaf is high frequency if its level is H by the rule of proposal §3.6 (formula p, \
         C_T = 2, θ = 1). Per tree: the widest occupied leaf in λ, its k a, and the levels \
         with an occupied leaf that are H at each ε (\"–\" for none). n = {N_LEAF}.\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Workload | kD | c_leaf | widest leaf (λ) | k a | H leaf levels, 10⁻⁴ | 10⁻⁶ | 10⁻⁸ |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|").unwrap();
    for r in runs.iter().filter(|r| r.n_leaf == N_LEAF) {
        let l = r.census.shallowest_leaf_level();
        let hf: Vec<String> = EPS
            .iter()
            .map(|&eps| {
                let levels: Vec<String> = high_frequency_leaf_levels(r, eps)
                    .iter()
                    .map(|l| l.to_string())
                    .collect();
                if levels.is_empty() {
                    "–".into()
                } else {
                    levels.join(", ")
                }
            })
            .collect();
        writeln!(
            out,
            "| {} | {} | {} | {:.3} | {:.2} | {} | {} | {} |",
            r.workload.name(),
            r.kd,
            c_name(r.c_leaf),
            0.5f64.powi(l as i32) * r.kd / (2.0 * std::f64::consts::PI),
            level_ka(r.kd, l),
            hf[0],
            hf[1],
            hf[2]
        )
        .unwrap();
    }
    out
}

/// The per-level tables of one run at ε.
fn level_tables(out: &mut String, r: &Run, eps: f64) {
    let m = TreeModel::new(&r.census, r.kd, eps);
    writeln!(
        out,
        "\n## {} at kD = {}, c_leaf = {}, n = {}, ε = {eps:e}\n\nN = {}; built and counted in {:.1} s.\n",
        r.workload.name(),
        r.kd,
        c_name(r.c_leaf),
        r.n_leaf,
        r.census.npoints,
        r.seconds
    )
    .unwrap();
    writeln!(
        out,
        "| Level | Box (λ) | k a | p | Boxes held | Occupied | Leaves (occ.) | Mean V | U | W | X | Regime | V on RCR | Multipoles MiB | Signatures GiB | Tables MiB | M2L 10⁹ | M2M+L2L 10⁹ | P2P pairs 10⁹ |"
    )
    .unwrap();
    writeln!(
        out,
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for (lm, c) in m.levels.iter().zip(&r.census.levels) {
        if lm.level < 2 && c.occupied_leaves == 0 {
            continue;
        }
        let regime = match lm.regime {
            Regime::Top => "–",
            Regime::L => "L",
            Regime::H => "H",
        };
        let rcr = if lm.v_pairs == 0 {
            "–".to_string()
        } else {
            format!("{:.0} %", 100.0 * lm.v_rcr as f64 / lm.v_pairs as f64)
        };
        writeln!(
            out,
            "| {} | {:.2} | {:.1} | {} | {} | {} | {} | {:.1} | {} | {} | {} | {regime} | {rcr} | {:.1} | {:.2} | {:.1} | {:.2} | {:.2} | {:.3} |",
            lm.level,
            lm.width_lambda,
            lm.ka,
            lm.p,
            c.boxes,
            c.occupied,
            c.occupied_leaves,
            if c.occupied > 0 { lm.v_pairs as f64 / c.occupied as f64 } else { 0.0 },
            c.u_entries_occupied,
            c.w_entries_occupied,
            c.x_entries_occupied,
            lm.multipole_bytes / MIB,
            lm.signature_bytes / GIB,
            (lm.transfer_bytes + lm.coaxial_bytes + lm.rotation_bytes) / MIB,
            lm.ops.m2l / 1e9,
            (lm.ops.m2m + lm.ops.l2l) / 1e9,
            lm.ops.p2p_pairs / 1e9,
        )
        .unwrap();
    }
    // V pairs by distance class.
    writeln!(out, "\nV pairs between occupied boxes by |t|²/w² (% of the level's pairs; a dagger marks the classes on RCR of an H level):\n").unwrap();
    let head: Vec<String> = DISTANCES.iter().map(|d| d.to_string()).collect();
    writeln!(out, "| Level | {} |", head.join(" | ")).unwrap();
    writeln!(out, "|---|{}", "---|".repeat(NDIST)).unwrap();
    for (lm, c) in m.levels.iter().zip(&r.census.levels) {
        if lm.v_pairs == 0 {
            continue;
        }
        let cells: Vec<String> = (0..NDIST)
            .map(|i| {
                let pct = 100.0 * c.v_by_distance[i] as f64 / lm.v_pairs as f64;
                let dagger = if lm.regime == Regime::H && lm.passing & (1 << i) == 0 {
                    "†"
                } else {
                    ""
                };
                format!("{pct:.1}{dagger}")
            })
            .collect();
        writeln!(out, "| {} | {} |", lm.level, cells.join(" | ")).unwrap();
    }
}

/// The summary of every (workload, kD) at c_leaf, n = 128, per ε.
fn summary(runs: &[Run], c_leaf: f64) -> String {
    let mut out = String::new();
    writeln!(out, "# Summary at c_leaf = {c_leaf}, n = {N_LEAF}\n").unwrap();
    writeln!(
        out,
        "Operation counts in 10⁹ complex multiply-adds (P2P in 10⁹ point pairs). M2L split into \
         the L levels (Rotation) and the H levels (hybrid). Memory: the multipoles of all levels \
         (complex f64; real multipoles half), the largest signature scratch of one level, the \
         largest table set of one level, the replicated top per rank (Phase 5S's 4,096-box cap, \
         and proposal §6.6's levels 2–4) and per node.\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Workload | kD | ε | N | leaf levels | depth | H levels | P2P | P2M+L2P | M2M+L2L | M2L (L) | M2L (H) | M2P+P2L | multipoles GiB | max signatures GiB | max tables GiB | top per rank MiB (cap; 2–4) | top per node GiB (40; 72) |"
    )
    .unwrap();
    writeln!(
        out,
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for r in runs
        .iter()
        .filter(|r| r.n_leaf == N_LEAF && r.c_leaf == Some(c_leaf))
    {
        for eps in EPS {
            let m = TreeModel::new(&r.census, r.kd, eps);
            let ops = m.ops();
            let (m2l_l, m2l_h) = m
                .levels
                .iter()
                .fold((0.0, 0.0), |(a, b), l| match l.regime {
                    Regime::H => (a, b + l.ops.m2l),
                    _ => (a + l.ops.m2l, b),
                });
            let multipoles: f64 = m.levels.iter().map(|l| l.multipole_bytes).sum();
            let sig = m
                .levels
                .iter()
                .map(|l| l.signature_bytes)
                .fold(0.0, f64::max);
            let tables = m
                .levels
                .iter()
                .map(|l| l.transfer_bytes + l.coaxial_bytes + l.rotation_bytes)
                .fold(0.0, f64::max);
            let (top, deepest) = m.replicated_top(&r.census);
            let h = m.high_frequency_levels();
            let h = if h.is_empty() {
                "–".to_string()
            } else {
                format!("{}–{}", h[0], h[h.len() - 1])
            };
            writeln!(
                out,
                "| {} | {} | {eps:e} | {} | {}–{} | {} | {h} | {} | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.0} (to {deepest}); {:.0} | {:.1}; {:.1} |",
                r.workload.name(),
                r.kd,
                r.census.npoints,
                r.census.shallowest_leaf_level(),
                r.census.deepest_leaf_level(),
                r.census.levels.len() - 1,
                g(ops.p2p_pairs / 1e9),
                g((ops.p2m + ops.l2p) / 1e9),
                g((ops.m2m + ops.l2l) / 1e9),
                g(m2l_l / 1e9),
                g(m2l_h / 1e9),
                g((ops.m2p + ops.p2l) / 1e9),
                multipoles / GIB,
                sig / GIB,
                tables / GIB,
                top / MIB,
                m.top_levels_2_to_4() / MIB,
                top * RANKS_PER_NODE[0].1 / GIB,
                top * RANKS_PER_NODE[1].1 / GIB,
            )
            .unwrap();
        }
    }
    out
}

/// The batch sizes of the conversions per H level (S5, S9).
fn batches(runs: &[Run], c_leaf: f64) -> String {
    let mut out = String::new();
    writeln!(out, "# Batch sizes of the φ step (S5, S9)\n").unwrap();
    writeln!(
        out,
        "Per H level at c_leaf = {c_leaf}, n = {N_LEAF}: the boxes converted S → F (sources of a \
         diagonal pair) and F → R (targets), each box 2P − 1 rings of N_φ samples; N_φ on the \
         host (a multiple of 4 with prime factors ≤ 7) and as a power of two.\n"
    )
    .unwrap();
    writeln!(out, "| Workload | kD | ε | Level | p | rings 2P − 1 | N_φ host | N_φ pow2 | S → F boxes | F → R boxes | transforms per level call (S → F) | signatures GiB |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|").unwrap();
    for r in runs
        .iter()
        .filter(|r| r.n_leaf == N_LEAF && r.c_leaf == Some(c_leaf))
    {
        for eps in EPS {
            let m = TreeModel::new(&r.census, r.kd, eps);
            for l in m.levels.iter().filter(|l| l.regime == Regime::H) {
                let rings = 2 * l.p + 1;
                writeln!(
                    out,
                    "| {} | {} | {eps:e} | {} | {} | {rings} | {} | {} | {} | {} | {} | {:.2} |",
                    r.workload.name(),
                    r.kd,
                    l.level,
                    l.p,
                    l.n_phi,
                    l.n_phi_pow2,
                    l.sig_sources,
                    l.sig_targets,
                    l.sig_sources as usize * rings,
                    l.signature_bytes / GIB
                )
                .unwrap();
            }
        }
    }
    out
}

fn sweep(comm: &SimpleCommunicator) {
    let mut runs = Vec::new();
    for (workload, kd, c_leaves, n_leaves) in configurations() {
        let keys = fine_keys(&workload.points(kd));
        for &n in &n_leaves {
            for &c in &c_leaves {
                runs.push(run(comm, workload, kd, &keys, c, n));
            }
        }
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let (mut study, by_rule) = c_leaf_report(&runs);
    let (revised, chosen) = no_high_frequency_leaf(&runs);
    study.push_str(&revised);
    std::fs::write(format!("{dir}/c-leaf.md"), study).unwrap();
    std::fs::write(format!("{dir}/leaves.md"), leaf_report(&runs)).unwrap();
    std::fs::write(format!("{dir}/summary.md"), summary(&runs, chosen)).unwrap();
    std::fs::write(format!("{dir}/batches.md"), batches(&runs, chosen)).unwrap();
    let mut levels = format!(
        "# Per-level tables\n\nc_leaf = {chosen} (D11 by the revised rule), n = {N_LEAF}, formula p, \
         C_T = 2, θ = 1. Tables: transfer functions of the passing reflection classes, the \
         (S|R) tables of the RCR distances present, and the rotation block sets (one for M2M \
         and L2L, 47 more where RCR M2L runs). U, W and X count entries between occupied \
         boxes.\n"
    );
    for r in runs
        .iter()
        .filter(|r| r.n_leaf == N_LEAF && r.c_leaf == Some(chosen))
    {
        if !matches!(r.workload, Workload::W4 { .. }) || r.kd == 100.0 {
            level_tables(&mut levels, r, EPS_MAIN);
        }
    }
    std::fs::write(format!("{dir}/levels.md"), levels).unwrap();
    // W1 at kD = 500 without the width rule and at the other tolerances, for §7.
    let mut extra = String::from("# W1 per level at kD = 500 and 1000, every ε\n");
    for r in runs.iter().filter(|r| {
        r.workload == Workload::W1
            && r.kd >= 500.0
            && r.n_leaf == N_LEAF
            && r.c_leaf == Some(chosen)
    }) {
        for eps in EPS {
            level_tables(&mut extra, r, eps);
        }
    }
    std::fs::write(format!("{dir}/w1-levels.md"), extra).unwrap();
    let total: f64 = runs.iter().map(|r| r.seconds).sum();
    eprintln!(
        "{} trees in {total:.0} s; D11 by the rule: c_leaf = {by_rule}, by the revised rule \
         {chosen}",
        runs.len()
    );
}

fn probe(comm: &SimpleCommunicator, args: &[String]) {
    let kd: f64 = args[1].parse().unwrap();
    let workload = match args[0].as_str() {
        "W1" => Workload::W1,
        "W2" => Workload::W2,
        "W3" => Workload::w3_resolved(kd),
        "W3u" => Workload::W3 { n: 1_000_000 },
        other => panic!("unknown workload {other}"),
    };
    let c_leaf = (args[2] != "none").then(|| args[2].parse().unwrap());
    let n: usize = args[3].parse().unwrap();
    let keys = fine_keys(&workload.points(kd));
    let r = run(comm, workload, kd, &keys, c_leaf, n);
    let mut out = String::new();
    level_tables(&mut out, &r, EPS_MAIN);
    println!("{out}");
}

fn main() {
    let universe = mpi::initialize().expect("MPI initialises once");
    let world = universe.world();
    assert_eq!(world.size(), 1, "the census runs on one rank");
    let comm = world.duplicate();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("sweep") => sweep(&comm),
        Some("probe") => probe(&comm, &args[1..]),
        _ => eprintln!("usage: census sweep | census probe <W1|W2|W3|W3u> <kD> <c_leaf|none> <n>"),
    }
}
