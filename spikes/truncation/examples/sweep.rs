//! The calibration sweep of CH0.7 (docs/helmholtz/tasks/CH0.7.md, "The sweep").
//!
//! ```text
//! cargo run -p nd-fmm-spike-truncation --release --example sweep -- c1 [kind] [ka...]
//! cargo run -p nd-fmm-spike-truncation --release --example sweep -- multi [kind] [ka...]
//! cargo run -p nd-fmm-spike-truncation --release --example sweep -- summary
//! ```
//!
//! `c1` calibrates p₁(k a, ε, kind) on the one-level chain: every p in a range on the 54
//! nearest offsets and the three placements, the smallest p whose worst error (potential
//! and gradient) is at most ε/2 from there on, then that p checked on all 316 offsets and
//! raised until it passes; the same p run in f64 for comparison. `multi` runs C2, C3m and
//! C3l with p_l = p₁(k a_l) + δ, the smallest δ ≥ 0 on the 54 offsets, checked on all 316.
//! Rows are appended to `results/c1.csv` and `results/multi.csv`; a (kind, k a) already
//! there is skipped, so a run can be resumed. `summary` writes `results/summary.md`.

use nd_fmm_spike_truncation::calibrate::{
    Chain, Config, Record, formula_p, run, smallest_passing, worst,
};
use nd_fmm_spike_truncation::chain::{Kind, Setup};
use nd_fmm_spike_truncation::geometry::{Placement, near_offsets, offsets};
use nd_fmm_spike_truncation::real::Dd;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::time::Instant;

const EPS: [f64; 8] = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];
const MARGIN: f64 = 2.0;

fn results_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results")
}

/// The half-octave grid 2^{j/2}, j = −12..16, and the kinds' ranges: every k a for real
/// and η = 0.1, to κ a = 32 for η = 1 and imaginary k (beyond §5.5's pruning cut).
fn grid(kind: Kind) -> Vec<f64> {
    let top = match kind {
        Kind::Real | Kind::Damped(0.1) => 16,
        _ => 10,
    };
    (-12..=top).map(|j| 2f64.powf(j as f64 / 2.0)).collect()
}

fn ka_key(ka: f64) -> String {
    format!("{ka:.6}")
}

fn kinds(arg: Option<&String>) -> Vec<Kind> {
    match arg.map(|s| s.as_str()) {
        None | Some("all") => Kind::ALL.to_vec(),
        Some("real") => vec![Kind::Real],
        Some("complex0.1") => vec![Kind::Damped(0.1)],
        Some("complex1") => vec![Kind::Damped(1.0)],
        Some("imaginary") => vec![Kind::Imaginary],
        Some(other) => panic!("unknown kind {other}"),
    }
}

/// Reads a CSV (header, rows) as maps.
fn read_csv(path: &std::path::Path) -> Vec<BTreeMap<String, String>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let cols: Vec<&str> = header.split(',').collect();
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            cols.iter()
                .zip(l.split(','))
                .map(|(c, v)| (c.to_string(), v.to_string()))
                .collect()
        })
        .collect()
}

fn append(path: &std::path::Path, header: &str, rows: &[String]) {
    let new = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    if new {
        writeln!(f, "{header}").unwrap();
    }
    for r in rows {
        writeln!(f, "{r}").unwrap();
    }
}

/// Worst (potential, gradient) per degree index, over all records or one placement.
fn worst_by(records: &[Record], n: usize, placement: Option<Placement>) -> Vec<(f64, f64)> {
    let r: Vec<Record> = records
        .iter()
        .copied()
        .filter(|r| placement.is_none_or(|p| r.placement == p))
        .collect();
    worst(&r, n)
}

const C1_HEADER: &str = "kind,ka,eps,p_formula,p_near,p,pot,grad,margin,p_pot,p_uniform,p_corners,p_patch,f64_pot,f64_grad,offsets_worst";

fn c1(kind: Kind, kas: &[f64], cfg: &Config) {
    let path = results_dir().join("c1.csv");
    let done: Vec<(String, String)> = read_csv(&path)
        .iter()
        .map(|r| (r["kind"].clone(), r["ka"].clone()))
        .collect();
    let near = near_offsets();
    let all = offsets();
    for &ka in kas {
        if done.contains(&(kind.name(), ka_key(ka))) {
            continue;
        }
        let t0 = Instant::now();
        let setup = Setup::<Dd>::new(kind, ka);
        let p_lo = (ka.floor() as usize).saturating_sub(4).max(1);
        let mut p_hi = formula_p(ka, 1e-11).max(75);
        // The scan, extended until the smallest threshold passes.
        // (p, worst error): of potential and gradient, per placement, of the potential.
        let mut curve: Vec<(usize, f64)> = Vec::new();
        let mut per_pl: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 3];
        let mut curve_pot: Vec<(usize, f64)> = Vec::new();
        let mut from = p_lo;
        loop {
            let degrees: Vec<[usize; 3]> = (from..=p_hi).map(|p| [p, 0, 0]).collect();
            let rec = run(&setup, Chain::C1, &near, &Placement::ALL, &degrees, cfg);
            let w = worst(&rec, degrees.len());
            for (d, (pot, grad)) in degrees.iter().zip(&w) {
                curve.push((d[0], pot.max(*grad)));
                curve_pot.push((d[0], *pot));
            }
            for (i, pl) in Placement::ALL.iter().enumerate() {
                for (d, (pot, grad)) in degrees.iter().zip(worst_by(&rec, degrees.len(), Some(*pl)))
                {
                    per_pl[i].push((d[0], pot.max(grad)));
                }
            }
            if smallest_passing(&curve, EPS[EPS.len() - 1] / MARGIN).is_some() || p_hi > 400 {
                break;
            }
            from = p_hi + 1;
            p_hi += 16;
        }
        // Check on every offset, raising p until it passes.
        let mut rows = Vec::new();
        for eps in EPS {
            let thr = eps / MARGIN;
            let Some(p_near) = smallest_passing(&curve, thr) else {
                rows.push(format!(
                    "{},{},{eps:e},{},,,,,,,,,,,,",
                    kind.name(),
                    ka_key(ka),
                    formula_p(ka, eps)
                ));
                continue;
            };
            let mut p = p_near;
            let (pot, grad, worst_offset) = loop {
                let rec = run(&setup, Chain::C1, &all, &Placement::ALL, &[[p, 0, 0]], cfg);
                let (pot, grad) = worst(&rec, 1)[0];
                if pot.max(grad) <= thr || p > p_near + 8 {
                    let wo = rec
                        .iter()
                        .max_by(|a, b| a.pot.max(a.grad).total_cmp(&b.pot.max(b.grad)))
                        .map(|r| all[r.offset])
                        .unwrap();
                    break (pot, grad, wo);
                }
                p += 1;
            };
            // f64 at the same p, on the 54 nearest offsets.
            let setup64 = Setup::<f64>::new(kind, ka);
            let rec64 = run(
                &setup64,
                Chain::C1,
                &near,
                &Placement::ALL,
                &[[p, 0, 0]],
                cfg,
            );
            let (f_pot, f_grad) = worst(&rec64, 1)[0];
            let opt = |o: Option<usize>| o.map_or(String::new(), |v| v.to_string());
            rows.push(format!(
                "{},{},{eps:e},{},{p_near},{p},{pot:.3e},{grad:.3e},{:.2},{},{},{},{},{f_pot:.3e},{f_grad:.3e},{}:{}:{}",
                kind.name(),
                ka_key(ka),
                formula_p(ka, eps),
                eps / pot.max(grad),
                opt(smallest_passing(&curve_pot, thr)),
                opt(smallest_passing(&per_pl[0], thr)),
                opt(smallest_passing(&per_pl[1], thr)),
                opt(smallest_passing(&per_pl[2], thr)),
                worst_offset[0],
                worst_offset[1],
                worst_offset[2],
            ));
        }
        append(&path, C1_HEADER, &rows);
        // The curve itself, for the report's plots and the fit.
        let curve_rows: Vec<String> = curve
            .iter()
            .zip(&curve_pot)
            .enumerate()
            .map(|(i, ((p, e), (_, ep)))| {
                format!(
                    "{},{},{p},{e:.3e},{ep:.3e},{:.3e},{:.3e},{:.3e}",
                    kind.name(),
                    ka_key(ka),
                    per_pl[0][i].1,
                    per_pl[1][i].1,
                    per_pl[2][i].1
                )
            })
            .collect();
        append(
            &results_dir().join("c1_curves.csv"),
            "kind,ka,p,worst,worst_pot,uniform,corners,patch",
            &curve_rows,
        );
        println!(
            "C1 {} ka {ka:.4}: {:.1} s",
            kind.name(),
            t0.elapsed().as_secs_f64()
        );
    }
}

const MULTI_HEADER: &str = "kind,ka,eps,chain,p0,p1,p2,delta,pot,grad,margin";

fn multi(kind: Kind, kas: &[f64], cfg: &Config) {
    let c1_rows = read_csv(&results_dir().join("c1.csv"));
    let p1 = |ka: f64, eps: f64| -> Option<usize> {
        c1_rows
            .iter()
            .find(|r| {
                r["kind"] == kind.name()
                    && r["ka"] == ka_key(ka)
                    && r["eps"].parse::<f64>().unwrap() == eps
            })
            .and_then(|r| r["p"].parse().ok())
    };
    let path = results_dir().join("multi.csv");
    let done: Vec<(String, String, String)> = read_csv(&path)
        .iter()
        .map(|r| (r["kind"].clone(), r["ka"].clone(), r["chain"].clone()))
        .collect();
    let near = near_offsets();
    let all = offsets();
    for &ka in kas {
        let setup = Setup::<Dd>::new(kind, ka);
        for chain in [Chain::C2, Chain::C3m, Chain::C3l] {
            if done.contains(&(kind.name(), ka_key(ka), chain.name().to_string())) {
                continue;
            }
            let levels = if chain == Chain::C2 { 2 } else { 3 };
            let t0 = Instant::now();
            // Base degrees per ε, from C1 at k a, k a/2, k a/4.
            let base: Vec<Option<[usize; 3]>> = EPS
                .iter()
                .map(|&eps| {
                    let mut d = [0usize; 3];
                    for (l, slot) in d.iter_mut().enumerate().take(levels) {
                        *slot = p1(ka / 2f64.powi(l as i32), eps)?;
                    }
                    Some(d)
                })
                .collect();
            if base.iter().all(|b| b.is_none()) {
                continue;
            }
            let mut delta = vec![0usize; EPS.len()];
            let mut settled = vec![false; EPS.len()];
            for (i, b) in base.iter().enumerate() {
                settled[i] = b.is_none();
            }
            let add = |d: [usize; 3], delta: usize| -> [usize; 3] {
                std::array::from_fn(|l| if l < levels { d[l] + delta } else { 0 })
            };
            // δ on the 54 nearest offsets.
            while settled.iter().any(|s| !s) {
                let open: Vec<usize> = (0..EPS.len()).filter(|&i| !settled[i]).collect();
                let degrees: Vec<[usize; 3]> = open
                    .iter()
                    .map(|&i| add(base[i].unwrap(), delta[i]))
                    .collect();
                let w = worst(
                    &run(&setup, chain, &near, &Placement::ALL, &degrees, cfg),
                    degrees.len(),
                );
                for (j, &i) in open.iter().enumerate() {
                    if w[j].0.max(w[j].1) <= EPS[i] / MARGIN || delta[i] >= 6 {
                        settled[i] = true;
                    } else {
                        delta[i] += 1;
                    }
                }
            }
            // On every offset, raising δ until it passes.
            let mut rows = Vec::new();
            for (i, &eps) in EPS.iter().enumerate() {
                let Some(b) = base[i] else { continue };
                let (d, pot, grad) = loop {
                    let d = add(b, delta[i]);
                    let w = worst(&run(&setup, chain, &all, &Placement::ALL, &[d], cfg), 1)[0];
                    if w.0.max(w.1) <= eps / MARGIN || delta[i] >= 8 {
                        break (d, w.0, w.1);
                    }
                    delta[i] += 1;
                };
                rows.push(format!(
                    "{},{},{eps:e},{},{},{},{},{},{pot:.3e},{grad:.3e},{:.2}",
                    kind.name(),
                    ka_key(ka),
                    chain.name(),
                    d[0],
                    d[1],
                    d[2],
                    delta[i],
                    eps / pot.max(grad)
                ));
            }
            append(&path, MULTI_HEADER, &rows);
            println!(
                "{} {} ka {ka:.4}: {:.1} s",
                chain.name(),
                kind.name(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
}

/// results/summary.md: the C1 tables against the formula, δ of the multi-level chains,
/// and the fit p = max(p_low(ε), ⌈k a + C(ε) (k a)^{1/3}⌉) that never under-predicts.
fn summary() {
    let c1_rows = read_csv(&results_dir().join("c1.csv"));
    let multi_rows = read_csv(&results_dir().join("multi.csv"));
    let mut out = String::new();
    writeln!(out, "# CH0.7 (S7) results\n").unwrap();
    writeln!(
        out,
        "Written by `examples/sweep.rs summary` from `results/c1.csv` and `results/multi.csv`. \
         p is the smallest degree whose worst relative ℓ2 error, potential and gradient, over \
         the 316 offsets and the three placements is at most ε/{MARGIN} (C1, one level); δ the \
         degree added at every level for the multi-level chains. `formula` is proposal §3.5.\n"
    )
    .unwrap();
    let mut kinds_seen: Vec<String> = Vec::new();
    for r in &c1_rows {
        if !kinds_seen.contains(&r["kind"]) {
            kinds_seen.push(r["kind"].clone());
        }
    }
    let get = |kind: &str, ka: &str, eps: f64, col: &str| -> String {
        c1_rows
            .iter()
            .find(|r| r["kind"] == kind && r["ka"] == ka && r["eps"].parse::<f64>().unwrap() == eps)
            .map_or(String::new(), |r| r[col].clone())
    };
    for kind in &kinds_seen {
        let mut kas: Vec<String> = c1_rows
            .iter()
            .filter(|r| &r["kind"] == kind)
            .map(|r| r["ka"].clone())
            .collect();
        kas.dedup();
        writeln!(out, "## {kind}: p₁ (calibrated) / formula\n").unwrap();
        let mut header = "| k a |".to_string();
        let mut rule = "| --- |".to_string();
        for eps in EPS {
            write!(header, " {eps:.0e} |").unwrap();
            rule.push_str(" --- |");
        }
        writeln!(out, "{header}\n{rule}").unwrap();
        for ka in &kas {
            let mut line = format!("| {:.3} |", ka.parse::<f64>().unwrap());
            for eps in EPS {
                write!(
                    line,
                    " {} / {} |",
                    get(kind, ka, eps, "p"),
                    get(kind, ka, eps, "p_formula")
                )
                .unwrap();
            }
            writeln!(out, "{line}").unwrap();
        }
        writeln!(out).unwrap();
        // The fit.
        writeln!(
            out,
            "Fit p = max(p_low, ⌈k a + C (k a)^(1/3)⌉), never below the calibrated p:\n"
        )
        .unwrap();
        writeln!(
            out,
            "| ε | p_low | C | largest overshoot |\n| --- | --- | --- | --- |"
        )
        .unwrap();
        for eps in EPS {
            let pts: Vec<(f64, usize)> = kas
                .iter()
                .filter_map(|ka| {
                    Some((
                        ka.parse::<f64>().unwrap(),
                        get(kind, ka, eps, "p").parse::<usize>().ok()?,
                    ))
                })
                .collect();
            let p_low = pts
                .iter()
                .filter(|(ka, _)| *ka <= 1.0)
                .map(|(_, p)| *p)
                .max()
                .unwrap_or(0);
            let c = pts
                .iter()
                .filter(|(ka, p)| *p > p_low && *ka > 1.0)
                .map(|(ka, p)| (*p as f64 - ka) / ka.cbrt())
                .fold(f64::NEG_INFINITY, f64::max);
            let c = if c.is_finite() {
                (c * 100.0).ceil() / 100.0
            } else {
                0.0
            };
            let over = pts
                .iter()
                .map(|(ka, p)| (p_low.max((ka + c * ka.cbrt()).ceil() as usize)) as i64 - *p as i64)
                .max()
                .unwrap_or(0);
            writeln!(out, "| {eps:.0e} | {p_low} | {c:.2} | {over} |").unwrap();
        }
        writeln!(out).unwrap();
        // Placements and the potential alone at ε = 10⁻⁶ and 10⁻¹⁰.
        writeln!(out, "Per placement and potential alone (p_uniform / p_corners / p_patch / p_pot), and f64 at p:\n").unwrap();
        writeln!(out, "| k a | ε | uniform | corners | patch | pot only | margin | f64 pot | f64 grad |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |").unwrap();
        for ka in &kas {
            for eps in [1e-6, 1e-10] {
                writeln!(
                    out,
                    "| {:.3} | {eps:.0e} | {} | {} | {} | {} | {} | {} | {} |",
                    ka.parse::<f64>().unwrap(),
                    get(kind, ka, eps, "p_uniform"),
                    get(kind, ka, eps, "p_corners"),
                    get(kind, ka, eps, "p_patch"),
                    get(kind, ka, eps, "p_pot"),
                    get(kind, ka, eps, "margin"),
                    get(kind, ka, eps, "f64_pot"),
                    get(kind, ka, eps, "f64_grad"),
                )
                .unwrap();
            }
        }
        writeln!(out).unwrap();
        // δ of the multi-level chains.
        writeln!(
            out,
            "Multi-level chains, δ (max over ε) and the largest δ's ε:\n"
        )
        .unwrap();
        writeln!(out, "| k a | C2 | C3m | C3l |\n| --- | --- | --- | --- |").unwrap();
        for ka in &kas {
            let mut line = format!("| {:.3} |", ka.parse::<f64>().unwrap());
            for chain in ["C2", "C3m", "C3l"] {
                let ds: Vec<(usize, String)> = multi_rows
                    .iter()
                    .filter(|r| &r["kind"] == kind && &r["ka"] == ka && r["chain"] == chain)
                    .map(|r| (r["delta"].parse().unwrap(), r["eps"].clone()))
                    .collect();
                match ds.iter().max_by_key(|d| d.0) {
                    Some((d, e)) if *d > 0 => write!(line, " {d} ({e}) |").unwrap(),
                    Some(_) => line.push_str(" 0 |"),
                    None => line.push_str(" – |"),
                }
            }
            writeln!(out, "{line}").unwrap();
        }
        writeln!(out).unwrap();
    }
    std::fs::write(results_dir().join("summary.md"), out).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = Config::default();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("summary");
    std::fs::create_dir_all(results_dir()).unwrap();
    let kas_arg: Vec<f64> = args.iter().skip(3).map(|s| s.parse().unwrap()).collect();
    match cmd {
        "c1" | "multi" => {
            for kind in kinds(args.get(2)) {
                let kas = if kas_arg.is_empty() {
                    grid(kind)
                } else {
                    kas_arg.clone()
                };
                if cmd == "c1" {
                    c1(kind, &kas, &cfg);
                } else {
                    multi(kind, &kas, &cfg);
                }
            }
        }
        "summary" => summary(),
        other => panic!("unknown command {other}"),
    }
}
