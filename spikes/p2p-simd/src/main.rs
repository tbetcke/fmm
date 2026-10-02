//! Phase 3S T2 spike: loop orders, inverse square roots, register blocking and the
//! green-kernels baseline for the P2P kernel (docs/phase3s/T2-simd-spike.md).
//!
//! `cargo run --release -p nd-fmm-spike-p2p-simd [-- --quick]` prints the whole
//! report as Markdown on stdout (progress on stderr). Timings are reported, never
//! asserted. Run with every BLAS thread variable set to 1.

mod bench;
mod green;
mod kernels;
mod machine;
mod rng;
mod rsqrt_study;
mod simd;
#[cfg(test)]
mod tests;
mod workloads;

use std::collections::BTreeMap;

use bench::{Row, run_cells};
use kernels::Order;
use machine::ASSUMED_GHZ;
use rsqrt_study::{Candidates, Prim};
use simd::{Elem, Vf};
use workloads::{Cell, Form};

/// green-kernels' commit (root `Cargo.toml`).
const GK_COMMIT: &str = "7d757c579f8633d58163cd5b2d011a9468541f17";

/// Pipes that issue vector FP operations, for the model of design §4.6 (NEON: 4).
const NEON_PIPES: f64 = 4.0;

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    header(quick);
    let rates = primitives();
    let study = rsqrt_tables(quick);
    lane_utilisation();
    let cells = cells(quick);
    let backend = machine::pulp_backend();
    // Candidate and relaxed rows run on the gathered W1 cells only.
    fn filter<E>(p: &kernels::Proto<E>, c: Cell) -> bool {
        match p.flavour {
            "cand" | "relaxed" => matches!(c, Cell::W1(_, Form::Gathered)),
            _ => true,
        }
    }
    eprintln!("kernels f32");
    let mut rows = run_cells::<f32>(&cells, &filter, &backend);
    eprintln!("kernels f64");
    rows.extend(run_cells::<f64>(&cells, &filter, &backend));
    kernel_tables(&rows);
    candidates_in_kernel(&rows, &study);
    summary(&rows, &rates);
}

fn cells(quick: bool) -> Vec<Cell> {
    if quick {
        return vec![Cell::W1(24, Form::Gathered), Cell::W1(24, Form::PerPair)];
    }
    let mut v = Vec::new();
    for n in [8, 16, 24, 32, 64, 128] {
        v.push(Cell::W1(n, Form::PerPair));
        v.push(Cell::W1(n, Form::Gathered));
    }
    for n in [1000, 10000] {
        v.push(Cell::W2(n, false));
        v.push(Cell::W2(n, true));
    }
    v
}

fn header(quick: bool) {
    println!("# P2P SIMD spike: raw results\n");
    println!("| item | value |\n| --- | --- |");
    println!("| CPU | {} |", machine::cpu_model());
    println!("| cores | {} |", machine::cores());
    println!("| target | {} |", machine::target());
    println!("| toolchain | {} |", machine::toolchain());
    println!("| ISAs with prototypes | {} |", machine::isas().join(", "));
    println!(
        "| green-kernels | commit `{GK_COMMIT}`, pulp backend `{}` |",
        machine::pulp_backend()
    );
    let blas: Vec<String> = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ]
    .iter()
    .map(|k| {
        format!(
            "{k}={}",
            std::env::var(k).unwrap_or_else(|_| "unset".into())
        )
    })
    .collect();
    println!("| thread variables | {} |", blas.join(", "));
    println!("| clock for cycles | {ASSUMED_GHZ} GHz assumed (checked by the FMA latency below) |");
    println!(
        "| mode | {} |",
        if quick {
            "--quick (reduced smoke run)"
        } else {
            "full"
        }
    );
    println!("| timing | median of 15 batches of ≥ 20 ms, one thread |\n");
}

/// Instruction rates per vector type: (latency in cycles, throughput per cycle).
type Rates = BTreeMap<(&'static str, &'static str, &'static str), (f64, f64)>;

fn primitives() -> Rates {
    eprintln!("primitives");
    let mut rates = Rates::new();
    println!("## Single instructions\n");
    println!(
        "Latency: one dependent chain. Throughput: 16 independent chains. Cycles at the assumed clock.\n"
    );
    println!("| ISA | precision | instruction | latency (cycles) | throughput (per cycle) |");
    println!("| --- | --- | --- | --- | --- |");
    #[cfg(target_arch = "aarch64")]
    {
        prim_rows::<simd::neon::F32x4>(&mut rates);
        prim_rows::<simd::neon::F64x2>(&mut rates);
    }
    #[cfg(target_arch = "x86_64")]
    if machine::avx2_fma() {
        prim_rows::<simd::avx2::F32x8>(&mut rates);
        prim_rows::<simd::avx2::F64x4>(&mut rates);
    }
    println!();
    rates
}

fn prim_rows<V: Vf>(rates: &mut Rates) {
    for p in Prim::ALL {
        // SAFETY: called only for vector types of an ISA this CPU has (`primitives`).
        let (lat, thr) = unsafe { rsqrt_study::prim_rates::<V>(p) };
        println!(
            "| {} | {} | {} | {lat:.2} | {thr:.2} |",
            V::ISA,
            V::E::NAME,
            p.name()
        );
        rates.insert((V::ISA, V::E::NAME, p.name()), (lat, thr));
    }
}

/// Measured error of each candidate, in u_T, by (ISA, precision, name).
type Study = BTreeMap<(&'static str, &'static str, &'static str), f64>;

fn rsqrt_tables(quick: bool) -> Study {
    let mut study = Study::new();
    println!("## Inverse square root: measured\n");
    println!(
        "Error: max |relative error| in u_T; f32 {} over [1, 4), f64 on {} log-uniform samples over [2⁻¹⁰⁸, 2⁷]; plus the powers of two of the domain, their neighbours and the ends. Edge inputs show the raw result before the r² = 0 mask. Throughput on a 4,096-element buffer in L1; latency in a dependent chain.\n",
        if quick {
            "every 64th input"
        } else {
            "exhaustively"
        },
        if quick { "10⁵" } else { "10⁷" }
    );
    rsqrt_table::<f32>(quick, &mut study);
    rsqrt_table::<f64>(quick, &mut study);
    study
}

fn rsqrt_table<E: Elem + Candidates>(quick: bool, study: &mut Study) {
    println!(
        "| ISA | precision | candidate | FP ops | max error (u_T) | at x | ≤ 4 u_T | ns/elem | cycles/vector | latency (cycles) | edge inputs |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for c in E::candidates() {
        eprintln!("rsqrt {} {} {}", c.isa, E::NAME, c.name);
        let st = E::errors(&c, quick);
        study.insert((c.isa, E::NAME, c.name), st.max_u);
        let (ns, cyc) = rsqrt_study::throughput(&c);
        let (_, lat) = rsqrt_study::latency(&c);
        println!(
            "| {} | {} | {} | {} | {:.3} | {:.6e} | {} | {:.3} | {:.2} | {:.1} | {} |",
            c.isa,
            E::NAME,
            c.name,
            c.ops,
            st.max_u,
            st.at,
            if st.max_u <= 4.0 { "yes" } else { "no" },
            ns,
            cyc,
            lat,
            rsqrt_study::edge_behaviour(&c)
        );
    }
    println!();
}

fn lane_utilisation() {
    println!("## Lane utilisation of targets in lanes at the W1 leaf sizes\n");
    println!(
        "n_t / (lanes used per call), per ISA, precision and K (K·W targets per block; the last block is padded).\n"
    );
    let isas: [(&str, &str, usize); 4] = [
        ("neon", "f32", 4),
        ("neon", "f64", 2),
        ("avx2", "f32", 8),
        ("avx2", "f64", 4),
    ];
    print!("| ISA | precision | K |");
    let sizes: [usize; 8] = [8, 16, 20, 24, 32, 44, 64, 128];
    for n in sizes {
        print!(" n_t={n} |");
    }
    println!();
    println!("| --- | --- | --- |{}", " --- |".repeat(sizes.len()));
    for (isa, p, w) in isas {
        for k in [1, 2, 4] {
            print!("| {isa} | {p} | {k} |");
            for n in sizes {
                let b = k * w;
                let used = n.div_ceil(b) * b;
                print!(" {:.0}% |", 100.0 * n as f64 / used as f64);
            }
            println!();
        }
    }
    println!();
}

/// Every contract candidate inside the kernel (targets in lanes, K = 2, gathered W1).
fn candidates_in_kernel(rows: &[Row], study: &Study) {
    println!("## Inverse square root inside the kernel\n");
    println!(
        "Targets in lanes, K = 2, gathered W1 cells, Gpairs/s; the geometric mean over n_t decides the formulation among those within 4 u_T in the study.\n"
    );
    let isa = proto_isa();
    for prec in ["f32", "f64"] {
        for grad in [false, true] {
            let cand: Vec<&Row> = rows
                .iter()
                .filter(|r| r.prec == prec && r.grad == grad && r.flavour == "cand")
                .collect();
            let mut names: Vec<&'static str> = Vec::new();
            let mut cells: Vec<Cell> = Vec::new();
            for r in &cand {
                if !names.contains(&r.rsqrt) {
                    names.push(r.rsqrt);
                }
                if !cells.contains(&r.cell) {
                    cells.push(r.cell);
                }
            }
            if names.is_empty() {
                continue;
            }
            println!(
                "### {prec}, {}\n",
                if grad {
                    "potential and gradient"
                } else {
                    "potential"
                }
            );
            print!("| rsqrt | FP ops | study error (u_T) |");
            for c in &cells {
                print!(" {} |", c.label());
            }
            println!(" geomean |");
            println!("| --- | --- | --- |{} --- |", " --- |".repeat(cells.len()));
            for name in names {
                let mine: Vec<&&Row> = cand.iter().filter(|r| r.rsqrt == name).collect();
                let err = study
                    .get(&(isa, prec, name))
                    .map_or("–".into(), |e| format!("{e:.2}"));
                let ops = mine.first().map_or(0, |r| r.model_ops);
                print!("| {name} | {ops} | {err} |");
                let mut v = Vec::new();
                for c in &cells {
                    let r = mine.iter().find(|r| r.cell == *c).map(|r| **r);
                    print!(" {} |", fmt_gp(r));
                    if let Some(p) = r.and_then(|r| r.pairs_per_s) {
                        v.push(p);
                    }
                }
                println!(
                    " {:.3} |",
                    if v.is_empty() {
                        f64::NAN
                    } else {
                        geomean(&v) / 1e9
                    }
                );
            }
            println!();
        }
    }
}

fn fmt_gp(r: Option<&Row>) -> String {
    match r {
        Some(r) => match r.pairs_per_s {
            Some(p) => format!("{:.3}", p / 1e9),
            None => "check failed".into(),
        },
        None => "–".into(),
    }
}

fn kernel_tables(rows: &[Row]) {
    println!("## Kernels: every row\n");
    println!(
        "Gpairs/s counts n_s n_t pairs, coincident ones included. Accuracy against `direct_sum` relative to the term magnitudes (max) and relative L2, worst of the checked sets; \"= gathered\": per-pair calls give the same bits as one gathered call.\n"
    );
    for prec in ["f32", "f64"] {
        for grad in [false, true] {
            println!(
                "### {prec}, {}\n",
                if grad {
                    "potential and gradient"
                } else {
                    "potential"
                }
            );
            println!(
                "| cell | variant | ISA | order | K/T | rsqrt | Gpairs/s | pairs/cycle | max err φ | max err ∇ | L2 φ | L2 ∇ | check | = gathered |"
            );
            println!(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            );
            for r in rows.iter().filter(|r| r.prec == prec && r.grad == grad) {
                println!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {:.1e} | {} | {:.1e} | {} | {} | {} |",
                    r.cell.label(),
                    r.label,
                    r.isa,
                    r.order
                        .map_or("", |o| if o == Order::Til { "TIL" } else { "SIL" }),
                    if r.block > 0 {
                        r.block.to_string()
                    } else {
                        String::new()
                    },
                    r.rsqrt,
                    fmt_gp(Some(r)),
                    r.pairs_per_s
                        .map_or("–".into(), |p| format!("{:.3}", p / (ASSUMED_GHZ * 1e9))),
                    r.acc.max_pot,
                    if grad {
                        format!("{:.1e}", r.acc.max_grad)
                    } else {
                        String::new()
                    },
                    r.acc.l2_pot,
                    if grad {
                        format!("{:.1e}", r.acc.l2_grad)
                    } else {
                        String::new()
                    },
                    if r.passed { "pass" } else { "FAIL" },
                    r.same_as_gathered
                        .map_or("", |b| if b { "yes" } else { "no" }),
                );
            }
            println!();
        }
    }
}

/// The rows of one (precision, output, cell).
fn find<'a>(
    rows: &'a [Row],
    prec: &str,
    grad: bool,
    cell: Cell,
    pred: impl Fn(&Row) -> bool,
) -> Vec<&'a Row> {
    rows.iter()
        .filter(|r| r.prec == prec && r.grad == grad && r.cell == cell && pred(r))
        .collect()
}

fn best<'a>(v: &[&'a Row]) -> Option<&'a Row> {
    v.iter()
        .filter(|r| r.pairs_per_s.is_some())
        .max_by(|a, b| a.pairs_per_s.partial_cmp(&b.pairs_per_s).unwrap())
        .copied()
}

fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

fn proto_isa() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "avx2"
    }
}

fn summary(rows: &[Row], rates: &Rates) {
    let mut cells: Vec<Cell> = Vec::new();
    for r in rows {
        if !cells.contains(&r.cell) {
            cells.push(r.cell);
        }
    }
    let isa = proto_isa();
    let is = |o: Order, k: usize, fl: &'static str| {
        move |r: &Row| r.order == Some(o) && r.block == k && r.flavour == fl && r.isa == isa
    };
    println!("## Kernels: summary\n");
    println!(
        "Gpairs/s. TIL = targets in lanes (K vectors per block), SIL = sources in lanes (T targets per block), both with the \"best\" inverse square root; \"TIL gk\" is the best K with green-kernels' formulation. Ratios: best TIL over green-kernels, over the reference, and best SIL over best TIL.\n"
    );
    let mut chosen_k: BTreeMap<(&str, bool), usize> = BTreeMap::new();
    for prec in ["f32", "f64"] {
        for grad in [false, true] {
            println!(
                "### {prec}, {}\n",
                if grad {
                    "potential and gradient"
                } else {
                    "potential"
                }
            );
            println!(
                "| cell | reference | green-kernels | TIL K=1 | K=2 | K=4 | SIL T=1 | T=2 | T=4 | TIL gk | TIL/gk | TIL/ref | SIL/TIL |"
            );
            println!(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            );
            for &cell in &cells {
                let one = |pred: &dyn Fn(&Row) -> bool| {
                    find(rows, prec, grad, cell, pred).first().copied()
                };
                let reference = one(&|r| r.label == "reference");
                let gk = one(&|r| r.label == "green-kernels");
                let til: Vec<&Row> = [1, 2, 4]
                    .iter()
                    .filter_map(|&k| one(&is(Order::Til, k, "best")))
                    .collect();
                let sil: Vec<&Row> = [1, 2, 4]
                    .iter()
                    .filter_map(|&k| one(&is(Order::Sil, k, "best")))
                    .collect();
                let til_gk: Vec<&Row> = [1, 2, 4]
                    .iter()
                    .filter_map(|&k| one(&is(Order::Til, k, "gk")))
                    .collect();
                let (bt, bs, bg) = (best(&til), best(&sil), best(&til_gk));
                let ratio = |a: Option<&Row>, b: Option<&Row>| match (
                    a.and_then(|r| r.pairs_per_s),
                    b.and_then(|r| r.pairs_per_s),
                ) {
                    (Some(x), Some(y)) => format!("{:.2}", x / y),
                    _ => "–".into(),
                };
                println!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    cell.label(),
                    fmt_gp(reference),
                    fmt_gp(gk),
                    fmt_gp(til.first().copied()),
                    fmt_gp(til.get(1).copied()),
                    fmt_gp(til.get(2).copied()),
                    fmt_gp(sil.first().copied()),
                    fmt_gp(sil.get(1).copied()),
                    fmt_gp(sil.get(2).copied()),
                    bg.map_or("–".into(), |r| format!(
                        "{} (K={})",
                        fmt_gp(Some(r)),
                        r.block
                    )),
                    ratio(bt, gk),
                    ratio(bt, reference),
                    ratio(bs, bt),
                );
            }
            println!();
            // K by the geometric mean over the W1 cells.
            let mut gm = Vec::new();
            for k in [1, 2, 4] {
                let v: Vec<f64> = cells
                    .iter()
                    .filter(|c| c.is_w1())
                    .filter_map(|&c| {
                        find(rows, prec, grad, c, is(Order::Til, k, "best"))
                            .first()
                            .and_then(|r| r.pairs_per_s)
                    })
                    .collect();
                if !v.is_empty() {
                    gm.push((k, geomean(&v)));
                }
            }
            if let Some(&(k, _)) = gm.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                chosen_k.insert((prec, grad), k);
            }
            println!(
                "Targets in lanes, geometric mean over the W1 cells (Gpairs/s): {}.\n",
                gm.iter()
                    .map(|(k, g)| format!("K={k}: {:.3}", g / 1e9))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    println!("## Loop order: the decision rule of design §4.1\n");
    println!(
        "Ratio best SIL / best TIL (each over its K or T, best inverse square root), geometric mean over the W1 cells (both forms, all n_t). Sources in lanes is recommended only if ≥ 1.15 in both precisions.\n"
    );
    println!("| precision | potential | potential and gradient | both outputs |");
    println!("| --- | --- | --- | --- |");
    for prec in ["f32", "f64"] {
        let mut all = Vec::new();
        let mut per = Vec::new();
        for grad in [false, true] {
            let v: Vec<f64> = cells
                .iter()
                .filter(|c| c.is_w1())
                .filter_map(|&c| {
                    let t = best(&find(rows, prec, grad, c, |r| {
                        r.order == Some(Order::Til) && r.flavour == "best" && r.isa == isa
                    }))?;
                    let s = best(&find(rows, prec, grad, c, |r| {
                        r.order == Some(Order::Sil) && r.flavour == "best" && r.isa == isa
                    }))?;
                    Some(s.pairs_per_s? / t.pairs_per_s?)
                })
                .collect();
            all.extend(&v);
            per.push(if v.is_empty() { f64::NAN } else { geomean(&v) });
        }
        println!(
            "| {prec} | {:.3} | {:.3} | {:.3} |",
            per[0],
            per[1],
            if all.is_empty() {
                f64::NAN
            } else {
                geomean(&all)
            }
        );
    }
    println!();

    println!("## Fraction of the design §4.6 model (targets in lanes, chosen K)\n");
    println!(
        "Model: {NEON_PIPES} FP pipes × W lanes / (FP ops per pair and lane, design §4.2 with this inverse square root). Corrected: also limited by the measured throughput of the estimate and of FRSQRTS, and for sqrt+div by the divider (1/rate(FSQRT) + 1/rate(FDIV) cycles per vector; the two share one unit), from the single-instruction table. Rows for the chosen formulation and for green-kernels' (gk), at the chosen K. Pairs per cycle at {ASSUMED_GHZ} GHz.\n"
    );
    println!(
        "| precision | output | K | rsqrt | ops/pair/lane | model pairs/cycle | corrected | cell | measured pairs/cycle | of model | of corrected |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for prec in ["f32", "f64"] {
        for grad in [false, true] {
            let Some(&k) = chosen_k.get(&(prec, grad)) else {
                continue;
            };
            for &cell in cells
                .iter()
                .filter(|c| matches!(c, Cell::W1(_, Form::Gathered) | Cell::W2(_, false)))
            {
                for flavour in ["best", "gk"] {
                    let Some(r) = find(rows, prec, grad, cell, is(Order::Til, k, flavour))
                        .first()
                        .copied()
                    else {
                        continue;
                    };
                    let Some(p) = r.pairs_per_s else { continue };
                    let ops = r.model_ops as f64;
                    let model = NEON_PIPES * r.lanes as f64 / ops;
                    let est_rate = rates
                        .get(&(isa, prec, Prim::Est.name()))
                        .map_or(f64::INFINITY, |x| x.1);
                    let step_rate = rates
                        .get(&(isa, prec, Prim::Step.name()))
                        .map_or(f64::INFINITY, |x| x.1);
                    let div_cycles = if r.rsqrt == "sqrt+div" {
                        let sq = rates
                            .get(&(isa, prec, Prim::Sqrt.name()))
                            .map_or(f64::INFINITY, |x| x.1);
                        let dv = rates
                            .get(&(isa, prec, Prim::Div.name()))
                            .map_or(f64::INFINITY, |x| x.1);
                        1.0 / sq + 1.0 / dv
                    } else {
                        0.0
                    };
                    let steps = r
                        .rsqrt
                        .strip_prefix("est+S")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let est_cycles = if r.rsqrt.starts_with("est") {
                        1.0 / est_rate
                    } else {
                        0.0
                    };
                    let cyc = (ops / NEON_PIPES)
                        .max(est_cycles)
                        .max(div_cycles)
                        .max(steps / step_rate);
                    let corrected = r.lanes as f64 / cyc;
                    let meas = p / (ASSUMED_GHZ * 1e9);
                    println!(
                        "| {prec} | {} | {k} | {} | {} | {model:.3} | {corrected:.3} | {} | {meas:.3} | {:.0}% | {:.0}% |",
                        if grad { "φ, ∇φ" } else { "φ" },
                        r.rsqrt,
                        r.model_ops,
                        cell.label(),
                        100.0 * meas / model,
                        100.0 * meas / corrected
                    );
                }
            }
        }
    }
    println!();

    println!("## Relaxed f64 levels (targets in lanes, chosen K, W1 gathered)\n");
    println!(
        "| output | n_t | full: rsqrt, Gpairs/s, max err | relaxed S2: Gpairs/s (gain), max err | relaxed P4: Gpairs/s (gain), max err |"
    );
    println!("| --- | --- | --- | --- | --- |");
    for grad in [false, true] {
        let Some(&k) = chosen_k.get(&("f64", grad)) else {
            continue;
        };
        for &cell in cells
            .iter()
            .filter(|c| matches!(c, Cell::W1(_, Form::Gathered)))
        {
            let full = find(rows, "f64", grad, cell, is(Order::Til, k, "best"))
                .first()
                .copied();
            let rel = |tag: &str| {
                find(rows, "f64", grad, cell, |r| {
                    r.order == Some(Order::Til)
                        && r.block == k
                        && r.flavour == "relaxed"
                        && r.label.ends_with(tag)
                })
                .first()
                .copied()
            };
            let (Some(full), Some(s2), Some(p4)) = (full, rel("rel_s2"), rel("rel_p4")) else {
                continue;
            };
            let cmp = |r: &Row| match (r.pairs_per_s, full.pairs_per_s) {
                (Some(a), Some(b)) => format!(
                    "{:.3} ({:+.0}%), {:.1e}",
                    a / 1e9,
                    100.0 * (a / b - 1.0),
                    r.acc.max()
                ),
                _ => "–".into(),
            };
            let Cell::W1(n, _) = cell else { continue };
            println!(
                "| {} | {n} | {}, {}, {:.1e} | {} | {} |",
                if grad { "φ, ∇φ" } else { "φ" },
                full.rsqrt,
                fmt_gp(Some(full)),
                full.acc.max(),
                cmp(s2),
                cmp(p4)
            );
        }
    }
    println!();
}
