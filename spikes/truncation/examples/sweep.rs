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

/// `results/`, or `$SWEEP_RESULTS` if set (for trial runs).
fn results_dir() -> std::path::PathBuf {
    std::env::var_os("SWEEP_RESULTS").map_or_else(
        || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results"),
        std::path::PathBuf::from,
    )
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

/// The scanned error curves of C1, each (p, worst error) sorted by p: of potential and
/// gradient, of the potential alone, and per placement.
#[derive(Default)]
struct Curves {
    all: Vec<(usize, f64)>,
    pot: Vec<(usize, f64)>,
    placement: [Vec<(usize, f64)>; 3],
}

impl Curves {
    /// Runs C1 at the degrees `ps` and merges the results.
    fn add(&mut self, setup: &Setup<Dd>, offsets: &[[i32; 3]], ps: &[usize], cfg: &Config) {
        let degrees: Vec<[usize; 3]> = ps.iter().map(|&p| [p, 0, 0]).collect();
        let rec = run(setup, Chain::C1, offsets, &Placement::ALL, &degrees, cfg);
        let w = worst(&rec, degrees.len());
        for (p, (pot, grad)) in ps.iter().zip(&w) {
            self.all.push((*p, pot.max(*grad)));
            self.pot.push((*p, *pot));
        }
        for (i, pl) in Placement::ALL.iter().enumerate() {
            for (p, (pot, grad)) in ps.iter().zip(worst_by(&rec, degrees.len(), Some(*pl))) {
                self.placement[i].push((*p, pot.max(grad)));
            }
        }
        for c in self.each_mut() {
            c.sort_by_key(|x| x.0);
        }
    }

    fn has(&self, p: usize) -> bool {
        self.all.iter().any(|x| x.0 == p)
    }

    fn each(&self) -> [&Vec<(usize, f64)>; 5] {
        [
            &self.all,
            &self.pot,
            &self.placement[0],
            &self.placement[1],
            &self.placement[2],
        ]
    }

    fn each_mut(&mut self) -> [&mut Vec<(usize, f64)>; 5] {
        let [a, b, c] = &mut self.placement;
        [&mut self.all, &mut self.pot, a, b, c]
    }
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
        // The scan on the 54 nearest offsets: (p, worst error) of potential and gradient,
        // per placement, and of the potential. Coarse first (every fourth p on a wide
        // range), extended until the smallest threshold passes, then refined below each
        // threshold's first passing p.
        let mut curves = Curves::default();
        let step = if p_hi - p_lo > 24 { 4 } else { 1 };
        let mut from = p_lo;
        loop {
            let ps: Vec<usize> = (from..=p_hi).step_by(step).collect();
            curves.add(&setup, &near, &ps, cfg);
            if smallest_passing(&curves.all, EPS[EPS.len() - 1] / MARGIN).is_some() || p_hi > 400 {
                break;
            }
            from = ps.last().unwrap() + step;
            p_hi += 16;
        }
        if step > 1 {
            let mut fill = Vec::new();
            for eps in EPS {
                for curve in curves.each() {
                    if let Some(pc) = smallest_passing(curve, eps / MARGIN) {
                        for p in pc.saturating_sub(step - 1)..pc {
                            if !curves.has(p) && !fill.contains(&p) {
                                fill.push(p);
                            }
                        }
                    }
                }
            }
            fill.sort_unstable();
            if !fill.is_empty() {
                curves.add(&setup, &near, &fill, cfg);
            }
        }
        // Check on every offset in one run at p_near, p_near + 1 and p_near + 2 of every ε
        // (the tables are built once), raising p further only if none passes.
        let p_near: Vec<Option<usize>> = EPS
            .iter()
            .map(|&eps| smallest_passing(&curves.all, eps / MARGIN))
            .collect();
        let mut batch: Vec<usize> = p_near.iter().flatten().flat_map(|&p| p..=p + 2).collect();
        batch.sort_unstable();
        batch.dedup();
        let degrees: Vec<[usize; 3]> = batch.iter().map(|&p| [p, 0, 0]).collect();
        let rec_all = run(&setup, Chain::C1, &all, &Placement::ALL, &degrees, cfg);
        let w_all = worst(&rec_all, degrees.len());
        let worst_offset = |rec: &[Record], di: usize| {
            rec.iter()
                .filter(|r| r.degrees == di)
                .max_by(|a, b| a.pot.max(a.grad).total_cmp(&b.pot.max(b.grad)))
                .map(|r| all[r.offset])
                .unwrap()
        };
        let mut chosen = Vec::new();
        for (i, &eps) in EPS.iter().enumerate() {
            let thr = eps / MARGIN;
            let Some(pn) = p_near[i] else {
                chosen.push(None);
                continue;
            };
            let hit = (pn..=pn + 2).find_map(|p| {
                let di = batch.iter().position(|&b| b == p).unwrap();
                (w_all[di].0.max(w_all[di].1) <= thr)
                    .then(|| (p, w_all[di].0, w_all[di].1, worst_offset(&rec_all, di)))
            });
            let found = hit.unwrap_or_else(|| {
                let mut p = pn + 3;
                loop {
                    let rec = run(&setup, Chain::C1, &all, &Placement::ALL, &[[p, 0, 0]], cfg);
                    let (pot, grad) = worst(&rec, 1)[0];
                    if pot.max(grad) <= thr || p > pn + 8 {
                        break (p, pot, grad, worst_offset(&rec, 0));
                    }
                    p += 1;
                }
            });
            chosen.push(Some(found));
        }
        // f64 at the chosen p, on the 54 nearest offsets, in one run.
        let mut ps64: Vec<usize> = chosen.iter().flatten().map(|c| c.0).collect();
        ps64.sort_unstable();
        ps64.dedup();
        let setup64 = Setup::<f64>::new(kind, ka);
        let d64: Vec<[usize; 3]> = ps64.iter().map(|&p| [p, 0, 0]).collect();
        let w64 = worst(
            &run(&setup64, Chain::C1, &near, &Placement::ALL, &d64, cfg),
            d64.len(),
        );
        let mut rows = Vec::new();
        for (i, &eps) in EPS.iter().enumerate() {
            let thr = eps / MARGIN;
            let Some((p, pot, grad, wo)) = chosen[i] else {
                rows.push(format!(
                    "{},{},{eps:e},{},,,,,,,,,,,,",
                    kind.name(),
                    ka_key(ka),
                    formula_p(ka, eps)
                ));
                continue;
            };
            let (f_pot, f_grad) = w64[ps64.iter().position(|&q| q == p).unwrap()];
            let opt = |o: Option<usize>| o.map_or(String::new(), |v| v.to_string());
            rows.push(format!(
                "{},{},{eps:e},{},{},{p},{pot:.3e},{grad:.3e},{:.2},{},{},{},{},{f_pot:.3e},{f_grad:.3e},{}:{}:{}",
                kind.name(),
                ka_key(ka),
                formula_p(ka, eps),
                p_near[i].unwrap(),
                eps / pot.max(grad),
                opt(smallest_passing(&curves.pot, thr)),
                opt(smallest_passing(&curves.placement[0], thr)),
                opt(smallest_passing(&curves.placement[1], thr)),
                opt(smallest_passing(&curves.placement[2], thr)),
                wo[0],
                wo[1],
                wo[2],
            ));
        }
        append(&path, C1_HEADER, &rows);
        // The curve itself, for the report's plots and the fit.
        let curve_rows: Vec<String> = (0..curves.all.len())
            .map(|i| {
                format!(
                    "{},{},{},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}",
                    kind.name(),
                    ka_key(ka),
                    curves.all[i].0,
                    curves.all[i].1,
                    curves.pot[i].1,
                    curves.placement[0][i].1,
                    curves.placement[1][i].1,
                    curves.placement[2][i].1
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
            // On every offset, δ and δ + 1 of every ε in one run, raising δ further only
            // if neither passes.
            let open: Vec<usize> = (0..EPS.len()).filter(|&i| base[i].is_some()).collect();
            let degrees: Vec<[usize; 3]> = open
                .iter()
                .flat_map(|&i| {
                    [
                        add(base[i].unwrap(), delta[i]),
                        add(base[i].unwrap(), delta[i] + 1),
                    ]
                })
                .collect();
            let w = worst(
                &run(&setup, chain, &all, &Placement::ALL, &degrees, cfg),
                degrees.len(),
            );
            let mut rows = Vec::new();
            for (j, &i) in open.iter().enumerate() {
                let eps = EPS[i];
                let b = base[i].unwrap();
                let (d, pot, grad) = if w[2 * j].0.max(w[2 * j].1) <= eps / MARGIN {
                    (add(b, delta[i]), w[2 * j].0, w[2 * j].1)
                } else if w[2 * j + 1].0.max(w[2 * j + 1].1) <= eps / MARGIN {
                    delta[i] += 1;
                    (add(b, delta[i]), w[2 * j + 1].0, w[2 * j + 1].1)
                } else {
                    delta[i] += 2;
                    loop {
                        let d = add(b, delta[i]);
                        let w = worst(&run(&setup, chain, &all, &Placement::ALL, &[d], cfg), 1)[0];
                        if w.0.max(w.1) <= eps / MARGIN || delta[i] >= 8 {
                            break (d, w.0, w.1);
                        }
                        delta[i] += 1;
                    }
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
        writeln!(out, "Per placement and potential alone (p_uniform / p_corners / p_patch / p_pot, on the 54 nearest offsets, where the search runs), the margin of p on all 316, and f64 at p (54 nearest offsets; all 316 in `f64_all.csv`):\n").unwrap();
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
        // The per-level degree: a level is the parent of one two-level chain and the child
        // of the next, so p* = p₁ + max(δ_C2(k a), δ_C2(2 k a)); beside it the closed form.
        let delta = |chain: &str, ka: &str, eps: f64| -> Option<usize> {
            multi_rows
                .iter()
                .find(|r| {
                    &r["kind"] == kind
                        && r["ka"] == ka
                        && r["chain"] == chain
                        && r["eps"].parse::<f64>().unwrap() == eps
                })
                .map(|r| r["delta"].parse().unwrap())
        };
        writeln!(
            out,
            "Per-level degree p* = p₁ + the largest δ of every chain that uses the level (C2 at \
             k a and 2 k a, C3m and C3l at k a, 2 k a and 4 k a; a chain whose k a is off the \
             grid does not count) / closed form max(⌈6 log₁₀(1/ε)⌉ − 2, \
             ⌈k a + (ln 1/ε)^(2/3) (k a)^(1/3)⌉); `*` where a chain that should count is \
             missing:\n"
        )
        .unwrap();
        writeln!(out, "{header}\n{rule}").unwrap();
        let (mut under, mut over) = (i64::MAX, i64::MIN);
        let on_grid = |x: f64| kas.contains(&ka_key(x));
        for ka in &kas {
            let x: f64 = ka.parse().unwrap();
            let mut line = format!("| {x:.3} |");
            for eps in EPS {
                let p: Option<usize> = get(kind, ka, eps, "p").parse().ok();
                let mut d = 0;
                let mut missing = false;
                for (chain, roles) in [("C2", 2), ("C3m", 3), ("C3l", 3)] {
                    for l in 0..roles {
                        let parent = x * 2f64.powi(l);
                        // The chain needs k a/2 (and k a/4) on the grid to exist at all.
                        if !on_grid(parent) || !on_grid(parent / 2f64.powi(roles - 1)) {
                            continue;
                        }
                        match delta(chain, &ka_key(parent), eps) {
                            Some(v) => d = d.max(v),
                            None => missing = true,
                        }
                    }
                }
                let cf = closed_form(x, eps);
                match p {
                    Some(p) => {
                        let ps = p + d;
                        under = under.min(cf as i64 - ps as i64);
                        over = over.max(cf as i64 - ps as i64);
                        let mark = if missing { "*" } else { "" };
                        write!(line, " {ps}{mark} / {cf} |").unwrap();
                    }
                    None => line.push_str(" – |"),
                }
            }
            writeln!(out, "{line}").unwrap();
        }
        writeln!(
            out,
            "\nClosed form minus p*: from {under} to {over} (negative: the closed form \
             under-predicts; on the grid points only, the data it was fitted to).\n"
        )
        .unwrap();
    }
    // Every row meets ε/2 (the sweep's caps would record a failing row otherwise).
    let failing: Vec<String> = c1_rows
        .iter()
        .chain(&multi_rows)
        .filter(|r| r["margin"].parse::<f64>().is_ok_and(|m| m < MARGIN))
        .map(|r| {
            let chain = r.get("chain").map_or("C1", |c| c.as_str());
            format!("{} {chain} k a {} ε {}", r["kind"], r["ka"], r["eps"])
        })
        .collect();
    writeln!(
        out,
        "Rows below the margin {MARGIN} (a search cap reached): {}.\n",
        if failing.is_empty() {
            "none".to_string()
        } else {
            failing.join(", ")
        }
    )
    .unwrap();
    std::fs::write(results_dir().join("summary.md"), out).unwrap();
}

/// C1 in f64 at every calibrated p of `results/c1.csv`, on all 316 offsets and the three
/// placements, with S2's signed-off tables; written to `results/f64_all.csv`. A (kind,
/// k a) already there is skipped.
fn f64_all(cfg: &Config) {
    let c1_rows = read_csv(&results_dir().join("c1.csv"));
    let path = results_dir().join("f64_all.csv");
    let done: Vec<(String, String)> = read_csv(&path)
        .iter()
        .map(|r| (r["kind"].clone(), r["ka"].clone()))
        .collect();
    let all = offsets();
    let mut sets: Vec<(String, String)> = Vec::new();
    for r in &c1_rows {
        let key = (r["kind"].clone(), r["ka"].clone());
        if !sets.contains(&key) && !done.contains(&key) {
            sets.push(key);
        }
    }
    for (kind_name, ka_s) in sets {
        let kind = kinds(Some(&kind_name))[0];
        let ka: f64 = ka_s.parse().unwrap();
        let rows: Vec<&BTreeMap<String, String>> = c1_rows
            .iter()
            .filter(|r| r["kind"] == kind_name && r["ka"] == ka_s && !r["p"].is_empty())
            .collect();
        let mut ps: Vec<usize> = rows.iter().map(|r| r["p"].parse().unwrap()).collect();
        ps.sort_unstable();
        ps.dedup();
        let degrees: Vec<[usize; 3]> = ps.iter().map(|&p| [p, 0, 0]).collect();
        let setup = Setup::<f64>::new(kind, ka);
        let w = worst(
            &run(&setup, Chain::C1, &all, &Placement::ALL, &degrees, cfg),
            degrees.len(),
        );
        let out: Vec<String> = rows
            .iter()
            .map(|r| {
                let p: usize = r["p"].parse().unwrap();
                let (pot, grad) = w[ps.iter().position(|&q| q == p).unwrap()];
                let eps: f64 = r["eps"].parse().unwrap();
                format!(
                    "{kind_name},{ka_s},{},{p},{pot:.3e},{grad:.3e},{}",
                    r["eps"],
                    pot.max(grad) <= eps / MARGIN
                )
            })
            .collect();
        append(&path, "kind,ka,eps,p,pot,grad,meets", &out);
    }
}

/// The closed form fitted to the real-k table: max(⌈6 log₁₀(1/ε)⌉ − 2,
/// ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉).
fn closed_form(ka: f64, eps: f64) -> usize {
    let low = (6.0 * (1.0 / eps).log10()).ceil() as usize - 2;
    let high = (ka + (1.0 / eps).ln().powf(2.0 / 3.0) * ka.cbrt()).ceil() as usize;
    low.max(high)
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
        "f64all" => f64_all(&cfg),
        other => panic!("unknown command {other}"),
    }
}
