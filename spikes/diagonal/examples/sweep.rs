//! The sweep of S6 (docs/helmholtz/tasks/CH0.6.md, "The sweep"): every distinct
//! (kind, k a, p) that one of the three degree rules gives at some ε, every class,
//! placement and variant, as rows of a CSV file.
//!
//! ```text
//! cargo run -p nd-fmm-spike-diagonal --release --example sweep -- \
//!     [--kinds real,complex0.1,complex1] [--ka 2.83,4,...] [--threads N] \
//!     [--shard i/n] [--out spikes/diagonal/results/runs.csv] [--list] [--p p] \
//!     [--classes i,j,...]
//! ```
//!
//! Units already in the output file are skipped, so a run can be resumed; `--shard i/n`
//! takes every n-th unit of the list ordered by cost (largest p first), for several
//! processes with their own output files. `--list` prints the units and stops; `--p` and
//! `--classes` restrict the units to one degree and the classes to a subset (diagnostics).

use std::collections::{BTreeSet, HashSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};

use nd_fmm_spike_diagonal::classes::classes;
use nd_fmm_spike_diagonal::rules::{EPSILONS, S7Table, calibrated, formula};
use nd_fmm_spike_diagonal::run::{Band, Config, VARIANTS, run_unit};
use nd_fmm_spike_truncation::chain::Kind;
use nd_fmm_spike_truncation::geometry::Placement;

/// The k a of the sweep: S7's half-octave grid 2^{j/2}, j = 3..17, and 400.
fn ka_grid() -> Vec<f64> {
    let mut v: Vec<f64> = (3..=17).map(|j| 2f64.powf(j as f64 / 2.0)).collect();
    v.push(400.0);
    v
}

/// Whether a level at (kind, k a) carries a V list at ε: Im(k) w ≤ ln(1/ε), the pruning
/// test of proposal §5.5 with θ_D = 1 and a leaf far below (its conservative end).
pub fn below_cut(kind: Kind, ka: f64, eps: f64) -> bool {
    let im_ka = match kind {
        Kind::Real => 0.0,
        Kind::Damped(eta) => eta * ka,
        Kind::Imaginary => ka,
    };
    im_ka * 2.0 / 3f64.sqrt() <= (1.0 / eps).ln()
}

fn kind_of(name: &str) -> Kind {
    match name {
        "real" => Kind::Real,
        "complex0.1" => Kind::Damped(0.1),
        "complex1" => Kind::Damped(1.0),
        _ => panic!("unknown kind {name}"),
    }
}

/// The distinct p of the three rules at (kind, k a) over the ε below the cut.
pub fn degrees(s7: &S7Table, kind: Kind, ka: f64) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for (e, &eps) in EPSILONS.iter().enumerate() {
        if !below_cut(kind, ka, eps) {
            continue;
        }
        out.insert(formula(ka, eps));
        out.insert(s7.degree(&kind.name(), ka, ka, e).0);
        out.insert(calibrated(ka, eps));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .map(|i| args[i + 1].clone())
    };
    let kinds: Vec<Kind> = get("--kinds")
        .unwrap_or_else(|| "real,complex0.1,complex1".into())
        .split(',')
        .map(kind_of)
        .collect();
    let kas: Vec<f64> = get("--ka").map_or_else(ka_grid, |s| {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    });
    let mut cfg = Config::default();
    if let Some(t) = get("--threads") {
        cfg.threads = t.parse().unwrap();
    }
    let (shard, shards) = get("--shard").map_or((0, 1), |s| {
        let (a, b) = s.split_once('/').unwrap();
        (a.parse().unwrap(), b.parse().unwrap())
    });
    let out_path = get("--out")
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/results/runs.csv").to_string());
    let s7 = S7Table::load();
    assert!(!s7.is_empty(), "S7's p* tables not found");
    let mut units: Vec<(Kind, f64, usize)> = Vec::new();
    for &kind in &kinds {
        for &ka in &kas {
            for p in degrees(&s7, kind, ka) {
                units.push((kind, ka, p));
            }
            if let Some(p) = get("--p") {
                units.retain(|u| u.2 == p.parse::<usize>().unwrap() || u.1 != ka);
            }
        }
    }
    units.sort_by(|a, b| b.2.cmp(&a.2).then(a.1.total_cmp(&b.1)));
    let mine: Vec<_> = units
        .iter()
        .enumerate()
        .filter(|(i, _)| i % shards == shard)
        .map(|(_, u)| *u)
        .collect();
    if args.iter().any(|a| a == "--list") {
        let cost: f64 = mine.iter().map(|u| (u.2 as f64).powi(3)).sum();
        for u in &mine {
            println!("{} {:.3} {}", u.0.name(), u.1, u.2);
        }
        println!("{} units, Σ p³ = {cost:.3e}", mine.len());
        return;
    }
    // Units already written.
    let mut done: HashSet<(String, String, usize)> = HashSet::new();
    if let Ok(f) = std::fs::File::open(&out_path) {
        for line in BufReader::new(f).lines().skip(1) {
            let line = line.unwrap();
            let c: Vec<&str> = line.split(',').collect();
            done.insert((c[0].to_string(), c[1].to_string(), c[2].parse().unwrap()));
        }
    }
    let fresh = done.is_empty();
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("output file");
    if fresh {
        writeln!(
            out,
            "kind,ka,p,class,rep,d2,placement,variant,log10_term,log10_max_t,\
             trunc_pot,trunc_grad,round_pot,round_grad,total_pot,total_grad,seconds"
        )
        .unwrap();
    }
    let all = classes();
    let ids: Vec<usize> = get("--classes").map_or_else(
        || (0..all.len()).collect(),
        |s| s.split(',').map(|x| x.parse().unwrap()).collect(),
    );
    for (n, &(kind, ka, p)) in mine.iter().enumerate() {
        let key = (kind.name(), format!("{ka:.3}"), p);
        if done.contains(&key) {
            continue;
        }
        let res = run_unit(kind, ka, p, &ids, &cfg);
        let mut text = String::new();
        for c in &res.classes {
            let rep = format!("{}:{}:{}", c.info.rep[0], c.info.rep[1], c.info.rep[2]);
            for (pl, &placement) in Placement::ALL.iter().enumerate() {
                for (v, var) in VARIANTS.iter().enumerate() {
                    let b = Band::ALL.iter().position(|x| *x == var.band).unwrap();
                    let (r, t) = (c.round[pl][v], c.total[pl][v]);
                    text += &format!(
                        "{},{:.3},{},{},{},{},{},{},{:.4},{:.4},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.2}\n",
                        kind.name(),
                        ka,
                        p,
                        c.class,
                        rep,
                        c.info.d2,
                        placement.name(),
                        var.name(),
                        c.log10_term[b],
                        c.log10_max_t[b],
                        c.trunc[pl].0,
                        c.trunc[pl].1,
                        r.0,
                        r.1,
                        t.0,
                        t.1,
                        c.seconds
                    );
                }
            }
        }
        out.write_all(text.as_bytes()).unwrap();
        out.flush().unwrap();
        eprintln!(
            "[{}/{}] {} k a {ka:.3} p {p}: {:.1} s",
            n + 1,
            mine.len(),
            kind.name(),
            res.seconds
        );
    }
}
