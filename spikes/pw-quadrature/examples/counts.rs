//! `nd-fmm-tables`' quadratures of the Helmholtz exponential M2L (Helmholtz CH6.1a)
//! against FMM3D's own files, at the centre and the corners of every domain of iprec 1–3:
//! the nodes and weights bit for bit the spike's parse of FMM3D's files and the node maps
//! the same to rounding; the wave and mode counts Σ Nₖ, Σ (2Fₖ + 1) against FMM3D's `hnumphys` and
//! `hnumfour`; the kernel's error on the region with each set of counts, against the
//! stated error. Prints a Markdown summary (docs/helmholtz/tasks/CH6.1a.md).
//!
//! ```text
//! cargo run --release -p nd-fmm-spike-pw-quadrature --example counts -- <FMM3D src/Helmholtz>
//! ```

use nd_fmm_spike_pw_quadrature::C;
use nd_fmm_spike_pw_quadrature::fmm3d::{self, Tables};
use nd_fmm_spike_pw_quadrature::quadrature::Quadrature as SpikeQuadrature;
use nd_fmm_spike_pw_quadrature::region::Grid;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{DOMAINS, domains};
use nd_fmm_tables::helmholtz::plane_wave::{Precision, Quadrature};
use rayon::prelude::*;

/// One (domain, κ) row.
struct Row {
    domain: usize,
    kappa: C,
    waves: (usize, usize),
    modes: (usize, usize),
    kernel: (f64, f64),
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: counts <FMM3D src/Helmholtz>");
    let fmm3d_tables = Tables::load(std::path::Path::new(&dir));
    let grid = Grid::region();
    println!(
        "| iprec | points | Σ Nₖ ours / FMM3D's, mean (range) | Σ (2Fₖ + 1) ours / FMM3D's, mean (range) | kernel / stated, ours, worst | kernel / stated, FMM3D's counts, worst |"
    );
    println!("| --- | --- | --- | --- | --- | --- |");
    let mut worst_rows = Vec::new();
    let mut by_rows = Vec::new();
    for precision in Precision::ALL {
        let p = precision.iprec();
        let ours = domains(precision);
        let rows: Vec<Row> = (1..=DOMAINS)
            .into_par_iter()
            .flat_map_iter(|d| {
                let theirs = &fmm3d_tables.precisions[p][&d];
                let mine = &ours[d - 1];
                assert_eq!(mine.nodes, theirs.xs, "nodes of iprec {p}, domain {d}");
                assert_eq!(mine.weights, theirs.ws, "weights of iprec {p}, domain {d}");
                assert_eq!(mine.stated_error, theirs.stated_error);
                let (lo, hi) = fmm3d::rectangle(d);
                let hi = C::new(hi.re * (1.0 - 1e-9), hi.im * (1.0 - 1e-9));
                [
                    lo,
                    0.5 * (lo + hi),
                    C::new(hi.re, lo.im),
                    C::new(lo.re, hi.im),
                    hi,
                ]
                .into_iter()
                .map(|kappa| {
                    let q = Quadrature::new(kappa, precision).expect("in the tables");
                    let f = SpikeQuadrature::from_fmm3d(theirs, kappa);
                    // The node maps to rounding (the two order their operations
                    // differently).
                    let close = |a: &[C], b: &[C]| {
                        a.len() == b.len()
                            && a.iter()
                                .zip(b)
                                .all(|(x, y)| (x - y).norm() <= 1e-15 * y.norm().max(1.0))
                    };
                    assert!(close(q.lambda(), &f.lambda), "λ of {p}/{d} at {kappa}");
                    assert!(close(q.weights(), &f.weight), "w of {p}/{d} at {kappa}");
                    let o = SpikeQuadrature::new(
                        kappa,
                        q.lambda().to_vec(),
                        q.weights().to_vec(),
                        q.alpha_counts().to_vec(),
                        q.fourier_counts().to_vec(),
                    );
                    let e = q.stated_error();
                    Row {
                        domain: d,
                        kappa,
                        waves: (q.wave_count(), f.physical_terms()),
                        modes: (q.mode_count(), f.fourier_terms()),
                        kernel: (
                            o.kernel_errors(&grid).absolute / e,
                            f.kernel_errors(&grid).absolute / e,
                        ),
                    }
                })
                .collect::<Vec<_>>()
            })
            .collect();
        let ratio = |f: &dyn Fn(&Row) -> (usize, usize)| {
            let r: Vec<f64> = rows
                .iter()
                .map(|row| {
                    let (a, b) = f(row);
                    a as f64 / b as f64
                })
                .collect();
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            let (lo, hi) = r
                .iter()
                .fold((f64::MAX, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
            format!("{mean:.2} ({lo:.2}–{hi:.2})")
        };
        let worst = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).fold(0.0f64, f64::max);
        println!(
            "| {p} | {} | {} | {} | {:.2} | {:.1} |",
            rows.len(),
            ratio(&|r| r.waves),
            ratio(&|r| r.modes),
            worst(&|r| r.kernel.0),
            worst(&|r| r.kernel.1),
        );
        // By row of Im κ: the mean wave ratio and the mean Σ Nₖ of each.
        let mut by_row = vec![(0.0f64, 0usize, 0usize, 0usize); fmm3d::NY];
        for r in &rows {
            let iy = (r.domain - 1) / fmm3d::NX;
            let e = &mut by_row[iy];
            e.0 += r.waves.0 as f64 / r.waves.1 as f64;
            e.1 += 1;
            e.2 += r.waves.0;
            e.3 += r.waves.1;
        }
        let line: Vec<String> = by_row
            .iter()
            .enumerate()
            .map(|(iy, e)| {
                format!(
                    "Im κ ≥ {:.2}: {:.2} ({} / {})",
                    fmm3d::IM_BREAKS[iy],
                    e.0 / e.1 as f64,
                    e.2 / e.1,
                    e.3 / e.1
                )
            })
            .collect();
        by_rows.push((p, line));
        let top = rows
            .iter()
            .max_by(|a, b| {
                (a.waves.0 as f64 / a.waves.1 as f64)
                    .total_cmp(&(b.waves.0 as f64 / b.waves.1 as f64))
            })
            .unwrap();
        worst_rows.push((p, top.domain, top.kappa, top.waves, top.modes, top.kernel));
    }
    println!();
    println!("The wave ratio by row of Im κ, mean (mean Σ Nₖ ours / FMM3D's):");
    println!();
    for (p, line) in by_rows {
        println!("- iprec {p}: {}", line.join("; "));
    }
    println!();
    println!("The largest wave ratio per precision:");
    println!();
    for (p, d, kappa, waves, modes, kernel) in worst_rows {
        println!(
            "- iprec {p}, domain {d}, κ = {kappa:.3}: Σ Nₖ {} against {}, Σ (2Fₖ + 1) {} against {}, kernel / stated {:.2} against {:.1}",
            waves.0, waves.1, modes.0, modes.1, kernel.0, kernel.1
        );
    }
}
