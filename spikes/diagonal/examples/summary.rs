//! The tables and fits of S6 from the sweep's rows (docs/helmholtz/tasks/CH0.6.md,
//! "Outputs").
//!
//! ```text
//! cargo run -p nd-fmm-spike-diagonal --release --example summary -- \
//!     results/runs-0.csv results/runs-1.csv ... > results/summary.md
//! ```
//!
//! The measured error of a (kind, k a, p, class, variant) is the worst over the three
//! placements of max(potential, gradient); `round` is against S7's double-double RCR,
//! `total` against the direct sum. The bound of proposal §3.6 is E = C_T u 10^{term},
//! term = log₁₀ |k||t| maxⱼ≤L (2j + 1)|hⱼ(k|t|)|; the alternative E_T = C u 10^{max T},
//! max T = log₁₀ max |T_L| over the samples, which the table build knows.

use std::collections::{BTreeMap, BTreeSet};

use nd_fmm_spike_diagonal::classes::classes;
use nd_fmm_spike_diagonal::rules::{EPSILONS, S7Table, calibrated, formula};

const U64: f64 = 1.1102230246251565e-16;
const U32: f64 = 5.960464477539063e-8;

/// One measured (kind, k a, p, class, variant), worst over placements.
#[derive(Clone, Debug, Default)]
struct Point {
    term: f64,
    max_t: f64,
    trunc: f64,
    round: f64,
    total: f64,
    seconds: f64,
}

type Key = (String, String, usize, usize, String);

fn worse(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
    }
}

fn load(paths: &[String]) -> BTreeMap<Key, Point> {
    let mut out: BTreeMap<Key, Point> = BTreeMap::new();
    for path in paths {
        let text = std::fs::read_to_string(path).expect("rows");
        for line in text.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            if c.len() < 17 || c[0] == "kind" {
                continue;
            }
            let f = |i: usize| c[i].parse::<f64>().unwrap();
            let key = (
                c[0].to_string(),
                c[1].to_string(),
                c[2].parse().unwrap(),
                c[3].parse().unwrap(),
                c[7].to_string(),
            );
            let e = out.entry(key).or_insert(Point {
                term: f(8),
                max_t: f(9),
                seconds: f(16),
                ..Default::default()
            });
            e.trunc = worse(e.trunc, worse(f(10), f(11)));
            e.round = worse(e.round, worse(f(12), f(13)));
            e.total = worse(e.total, worse(f(14), f(15)));
        }
    }
    out
}

/// Quantiles 0, 0.5, 0.9, 1 of a sample.
fn quantiles(mut v: Vec<f64>) -> [f64; 4] {
    if v.is_empty() {
        return [f64::NAN; 4];
    }
    v.sort_by(f64::total_cmp);
    let q = |x: f64| v[((v.len() - 1) as f64 * x).round() as usize];
    [q(0.0), q(0.5), q(0.9), q(1.0)]
}

/// The share of an interior box's 189 V-list offsets in each class.
fn interior_shares() -> Vec<f64> {
    let cs = classes();
    let mut count = vec![0usize; cs.len()];
    // The box at the child position (0, 0, 0) of its parent; the parent's 27 neighbours'
    // children not adjacent to the box.
    for nx in -1i32..=1 {
        for ny in -1i32..=1 {
            for nz in -1i32..=1 {
                for c in 0..8 {
                    let o = [
                        2 * nx + (c & 1),
                        2 * ny + (c >> 1 & 1),
                        2 * nz + (c >> 2 & 1),
                    ];
                    if o.iter().all(|x| x.abs() <= 1) {
                        continue;
                    }
                    let (rep, _) = nd_fmm_spike_diagonal::classes::class_of(o);
                    count[cs.iter().position(|k| k.rep == rep).unwrap()] += 1;
                }
            }
        }
    }
    assert_eq!(count.iter().sum::<usize>(), 189);
    count.into_iter().map(|n| n as f64 / 189.0).collect()
}

fn kind_eta(kind: &str) -> f64 {
    match kind {
        "complex0.1" => 0.1,
        "complex1" => 1.0,
        _ => 0.0,
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let pts = load(&paths);
    let cs = classes();
    let shares = interior_shares();
    let s7 = S7Table::load();
    let kinds: BTreeSet<String> = pts.keys().map(|k| k.0.clone()).collect();
    let units: BTreeSet<(String, String, usize)> = pts
        .keys()
        .map(|k| (k.0.clone(), k.1.clone(), k.2))
        .collect();
    println!("# CH0.6 (S6) results\n");
    println!(
        "Written by `examples/summary.rs` from {} rows files: {} units (kind, k a, p), {} \
         (unit, class, variant) points. Errors are the worst over the three placements of \
         max(potential, gradient), relative ℓ2 over the targets. u = 2⁻⁵³ (f64), 2⁻²⁴ (f32).\n",
        paths.len(),
        units.len(),
        pts.len()
    );
    let get = |kind: &str, ka: &str, p: usize, c: usize, v: &str| {
        pts.get(&(kind.to_string(), ka.to_string(), p, c, v.to_string()))
    };

    // 1. The bound of §3.6 against the rounding, per variant and kind.
    println!("## 1. Rounding against the bounds\n");
    println!(
        "measured / (u 10^term) (the proposal's E with C_T = 1) and measured / (u max|T|), \
         quantiles 0, ½, 0.9, 1, over the points whose bound exceeds the floor (u 10^term ≥ \
         10⁻¹³ in f64, 10⁻⁵ in f32), and the largest ratio among the floor points:\n"
    );
    println!(
        "| kind | variant | points | E: min / median / 90 % / max | max|T|: min / median / 90 % / max | floor points: max measured |"
    );
    println!("| --- | --- | --- | --- | --- | --- |");
    let variants = [
        "f64:2P-2",
        "f64:3(P-1)/2",
        "f64:P-1",
        "f64i:2P-2",
        "f64b:2P-2",
        "f32:2P-2",
        "f32:3(P-1)/2",
        "f32:P-1",
    ];
    for kind in &kinds {
        for v in variants {
            let u = if v.starts_with("f32") { U32 } else { U64 };
            let floor = if v.starts_with("f32") { 1e-5 } else { 1e-13 };
            let (mut re, mut rt, mut fl) = (Vec::new(), Vec::new(), 0.0f64);
            for (k, pt) in &pts {
                if &k.0 != kind || k.4 != v || !pt.round.is_finite() {
                    continue;
                }
                let e = u * 10f64.powf(pt.term);
                if e >= floor {
                    re.push(pt.round / e);
                    rt.push(pt.round / (u * 10f64.powf(pt.max_t)));
                } else {
                    fl = fl.max(pt.round);
                }
            }
            let (a, b) = (quantiles(re.clone()), quantiles(rt));
            println!(
                "| {kind} | {v} | {} | {:.1e} / {:.1e} / {:.1e} / {:.1e} | {:.1e} / {:.1e} / {:.1e} / {:.1e} | {fl:.1e} |",
                re.len(),
                a[0],
                a[1],
                a[2],
                a[3],
                b[0],
                b[1],
                b[2],
                b[3]
            );
        }
    }

    // 2. C_T and θ for the proposal's variant (f64, 2P − 2), every kind together, the
    // floor included through a term C_0 u.
    let mut ratios = Vec::new();
    let mut floor_max = 0.0f64;
    let mut worst_point = String::new();
    for (k, pt) in &pts {
        if k.4 != "f64:2P-2" || !pt.round.is_finite() {
            continue;
        }
        let e = U64 * 10f64.powf(pt.term);
        if e >= 1e-13 {
            let r = pt.round / e;
            if r > ratios.iter().copied().fold(0.0, f64::max) {
                worst_point = format!("{} k a {} p {} class {:?}", k.0, k.1, k.2, cs[k.3].rep);
            }
            ratios.push(r);
        } else {
            floor_max = floor_max.max(pt.round / U64);
        }
    }
    let q = quantiles(ratios);
    let c_t = q[3] / 2.0;
    println!("\n## 2. C_T and θ (f64, L = 2P − 2, T rounded from double-double)\n");
    println!(
        "Above the floor the ratio measured / (u 10^term) lies in [{:.1e}, {:.1e}] (median \
         {:.1e}); the largest at {worst_point}. With C_T = {c_t:.2e} (half the largest ratio) \
         the bound E = C_T u 10^term never under-predicts by more than 2×, and θ = 1/2 makes \
         E ≤ θ ε imply a measured rounding error ≤ ε. Below the floor (u 10^term < 10⁻¹³) the \
         rounding error is at most {floor_max:.0} u, so the bound with the floor is \
         E = u (C_T 10^term + {floor_max:.0}).\n",
        q[0], q[3], q[1]
    );

    // 3. Classes on RCR per (k a, ε), per rule, by the bound and by the measurement.
    let shares_of = |set: &BTreeSet<usize>| set.iter().fold(0.0, |a, &c| a + shares[c]);
    for kind in &kinds {
        let kas: BTreeSet<String> = units
            .iter()
            .filter(|u| &u.0 == kind)
            .map(|u| u.1.clone())
            .collect();
        let mut kas: Vec<String> = kas.into_iter().collect();
        kas.sort_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse().unwrap()));
        for rule in ["formula", "S7", "calibrated"] {
            println!(
                "\n## 3. {kind}, {rule}: classes on RCR per (k a, ε)\n\nEach cell: the classes \
                 (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by \
                 the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the \
                 truncation is the rule's, the same as RCR's), and the share of an \
                 interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the \
                 level is pruned or the unit is missing.\n"
            );
            let header: Vec<String> = EPSILONS.iter().map(|e| format!("{e:.0e}")).collect();
            println!("| k a | {} |", header.join(" | "));
            println!("| --- |{}", " --- |".repeat(EPSILONS.len()));
            for ka in &kas {
                let x: f64 = ka.parse().unwrap();
                let mut row = Vec::new();
                for (e, &eps) in EPSILONS.iter().enumerate() {
                    let p = match rule {
                        "formula" => formula(x, eps),
                        "S7" => s7.degree(kind, x, x, e).0,
                        _ => calibrated(x, eps),
                    };
                    let (mut by_bound, mut by_meas) = (BTreeSet::new(), BTreeSet::new());
                    let mut found = false;
                    for c in 0..cs.len() {
                        let Some(pt) = get(kind, ka, p, c, "f64:2P-2") else {
                            continue;
                        };
                        found = true;
                        let e_bound = U64 * (c_t * 10f64.powf(pt.term) + floor_max);
                        // A non-finite value fails.
                        if e_bound.is_nan() || e_bound > 0.5 * eps {
                            by_bound.insert(c);
                        }
                        if pt.round.is_nan() || pt.round > eps {
                            by_meas.insert(c);
                        }
                    }
                    if !found {
                        row.push("–".to_string());
                        continue;
                    }
                    let names = |s: &BTreeSet<usize>| {
                        let d: BTreeSet<i32> = s.iter().map(|&c| cs[c].d2).collect();
                        if d.is_empty() {
                            "none".to_string()
                        } else if d.len() > 6 {
                            format!("{} classes", s.len())
                        } else {
                            d.iter()
                                .map(|x| x.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        }
                    };
                    row.push(format!(
                        "{} / {} ({:.0} %) [{p}]",
                        names(&by_bound),
                        names(&by_meas),
                        100.0 * shares_of(&by_bound)
                    ));
                }
                println!("| {ka} | {} |", row.join(" | "));
            }
        }
    }

    // 4. f32: every (kind, k a, p, class) where an f32 variant meets 10⁻³ or 10⁻⁴.
    println!("\n## 4. f32\n");
    println!(
        "Per kind and k a: the smallest f32 total error over the classes and bandwidths at \
         any p run, and the classes (|t|²) where f32 at L = 2P − 2 meets 10⁻³ at the p of \
         the calibrated rule at 10⁻³.\n"
    );
    println!("| kind | k a | best f32 total (class, band, p) | f32 meets 10⁻³ at calibrated p |");
    println!("| --- | --- | --- | --- |");
    for kind in &kinds {
        let mut kas: Vec<String> = units
            .iter()
            .filter(|u| &u.0 == kind)
            .map(|u| u.1.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        kas.sort_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse().unwrap()));
        for ka in &kas {
            let mut best = (f64::INFINITY, String::new());
            for (k, pt) in &pts {
                if &k.0 == kind && &k.1 == ka && k.4.starts_with("f32") && pt.total < best.0 {
                    best = (pt.total, format!("{}, {}, {}", cs[k.3].d2, k.4, k.2));
                }
            }
            let x: f64 = ka.parse().unwrap();
            let p = calibrated(x, 1e-3);
            let ok: BTreeSet<i32> = (0..cs.len())
                .filter(|&c| get(kind, ka, p, c, "f32:2P-2").is_some_and(|pt| pt.total <= 1e-3))
                .map(|c| cs[c].d2)
                .collect();
            println!(
                "| {kind} | {ka} | {:.1e} ({}) | {} |",
                best.0,
                best.1,
                if ok.is_empty() {
                    "none".to_string()
                } else {
                    format!("{ok:?}")
                }
            );
        }
    }

    // 5. Interpolated and f64-built T against T rounded from double-double.
    println!("\n## 5. T interpolated in ŝ · t̂, and T built in f64\n");
    println!(
        "Ratio of the rounding error to that of T rounded from double-double (f64, 2P − 2), \
         over the points where the latter exceeds 10⁻¹³, and the largest ratio to the bound E \
         of §2:\n"
    );
    println!("| kind | variant | points | ratio min / median / max | max measured / E |");
    println!("| --- | --- | --- | --- | --- |");
    for kind in &kinds {
        for v in ["f64i:2P-2", "f64b:2P-2"] {
            let (mut r, mut over) = (Vec::new(), 0.0f64);
            for (k, pt) in &pts {
                if &k.0 != kind || k.4 != v {
                    continue;
                }
                let base = get(&k.0, &k.1, k.2, k.3, "f64:2P-2").unwrap();
                if base.round > 1e-13 && pt.round.is_finite() {
                    r.push(pt.round / base.round);
                }
                let e = U64 * (c_t * 10f64.powf(pt.term) + floor_max);
                over = over.max(pt.round / e);
            }
            let q = quantiles(r.clone());
            println!(
                "| {kind} | {v} | {} | {:.2} / {:.2} / {:.2} | {over:.2} |",
                r.len(),
                q[0],
                q[1],
                q[3]
            );
        }
    }

    // 6. The bandwidths: the worst class's total error at the formula's p.
    println!("\n## 6. The bandwidth of T (f64, formula p)\n");
    println!(
        "The worst total error over the classes, and the nearest class (|t|² = 4), for L = \
         P − 1, ⌈3(P − 1)/2⌉ and 2P − 2, at the formula's p, against ε; real k.\n"
    );
    println!(
        "| k a | ε | p | worst: P−1 / 3(P−1)/2 / 2P−2 | nearest: P−1 / 3(P−1)/2 / 2P−2 | truncation (RCR) |"
    );
    println!("| --- | --- | --- | --- | --- | --- |");
    let mut kas: Vec<String> = units
        .iter()
        .filter(|u| u.0 == "real")
        .map(|u| u.1.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    kas.sort_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse().unwrap()));
    for ka in &kas {
        let x: f64 = ka.parse().unwrap();
        for eps in [1e-4, 1e-6, 1e-8] {
            let p = formula(x, eps);
            let mut worst = [0.0f64; 3];
            let mut near = [0.0f64; 3];
            let mut tr = 0.0f64;
            for (c, class) in cs.iter().enumerate() {
                for (b, v) in ["f64:P-1", "f64:3(P-1)/2", "f64:2P-2"].iter().enumerate() {
                    if let Some(pt) = get("real", ka, p, c, v) {
                        worst[b] = worst[b].max(pt.total);
                        if class.d2 == 4 {
                            near[b] = near[b].max(pt.total);
                        }
                        tr = tr.max(pt.trunc);
                    }
                }
            }
            println!(
                "| {ka} | {eps:.0e} | {p} | {:.1e} / {:.1e} / {:.1e} | {:.1e} / {:.1e} / {:.1e} | {tr:.1e} |",
                worst[0], worst[1], worst[2], near[0], near[1], near[2]
            );
        }
    }

    // 7. Damped k: the ratio to E against Im(k) a.
    println!("\n## 7. Damped k\n");
    println!(
        "measured / E (f64, 2P − 2, above the floor) per kind and Im(k) a band: does the bound \
         need a damping factor?\n"
    );
    println!("| kind | Im(k) a | points | min / median / max |");
    println!("| --- | --- | --- | --- |");
    for kind in &kinds {
        let eta = kind_eta(kind);
        let mut bands: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for (k, pt) in &pts {
            if &k.0 != kind || k.4 != "f64:2P-2" || !pt.round.is_finite() {
                continue;
            }
            let e = U64 * 10f64.powf(pt.term);
            if e < 1e-13 {
                continue;
            }
            let im = eta * k.1.parse::<f64>().unwrap();
            bands
                .entry(if im == 0.0 {
                    -1
                } else {
                    im.log2().floor() as i64
                })
                .or_default()
                .push(pt.round / e);
        }
        for (b, v) in bands {
            let q = quantiles(v.clone());
            let label = if b < 0 && eta == 0.0 {
                "0".to_string()
            } else {
                format!("[{}, {})", 2f64.powi(b as i32), 2f64.powi(b as i32 + 1))
            };
            println!(
                "| {kind} | {label} | {} | {:.1e} / {:.1e} / {:.1e} |",
                v.len(),
                q[0],
                q[1],
                q[3]
            );
        }
    }

    // 8. Wall time per class.
    let secs: Vec<f64> = pts
        .iter()
        .filter(|(k, _)| k.4 == "f64:2P-2")
        .map(|(_, p)| p.seconds)
        .collect();
    let total: f64 = secs.iter().sum();
    println!(
        "\nSweep cost: {:.1} core-hours over {} classes run (the per-class wall times summed).",
        total / 3600.0,
        secs.len()
    );
}
