//! FMM3D's α counts at the domains where the kernel's error is largest
//! (SPIKE_REPORT.md §3): the error with Nₖ scaled by 1, 1.25, 1.5 and 2, where on the
//! region it peaks, and the error with ρ ≤ 4 only.
//!
//! ```text
//! cargo run --release -p nd-fmm-spike-pw-quadrature --example alpha -- <FMM3D src/Helmholtz>
//! ```

use nd_fmm_spike_pw_quadrature::fmm3d::{self, Tables};
use nd_fmm_spike_pw_quadrature::quadrature::Quadrature;
use nd_fmm_spike_pw_quadrature::region::Grid;
use nd_fmm_spike_pw_quadrature::{C, kernel};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: alpha <FMM3D src/Helmholtz>");
    let t = Tables::load(std::path::Path::new(&dir));
    let grid = Grid::region();
    // The worst domains of results/domains.md at iprec 2 and 3, and two of the real axis.
    for (p, d) in [(2usize, 88usize), (3, 85), (2, 34), (2, 11)] {
        let (lo, hi) = fmm3d::rectangle(d);
        let dom = &t.precisions[p][&d];
        let corners = [
            lo,
            0.5 * (lo + hi),
            C::new(hi.re - 1e-9, lo.im),
            C::new(lo.re, hi.im - 1e-9),
            C::new(hi.re - 1e-9, hi.im - 1e-9),
        ];
        let kw = corners
            .into_iter()
            .map(|kw| {
                (
                    Quadrature::from_fmm3d(dom, kw)
                        .kernel_errors(&grid)
                        .absolute,
                    kw,
                )
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap()
            .1;
        let q = Quadrature::from_fmm3d(dom, kw);
        print!(
            "iprec {p}, domain {d}, κ = {kw:.3}, stated {:.1e}:",
            dom.stated_error
        );
        for f in [1.0, 1.25, 1.5, 2.0] {
            let n: Vec<usize> = q
                .physical
                .iter()
                .map(|&n| (n as f64 * f).ceil() as usize)
                .collect();
            print!(
                " N × {f}: {:.2e};",
                q.kernel_errors_with(&grid, &n).absolute
            );
        }
        let (mut at, mut worst, mut within4) = ([0.0; 3], 0.0f64, 0.0f64);
        for u in grid.points() {
            let e = (q.evaluate(u, &q.physical) - kernel(kw, u)).norm();
            if e > worst {
                (worst, at) = (e, u);
            }
            if u[0].hypot(u[1]) <= 4.0 {
                within4 = within4.max(e);
            }
        }
        println!(" largest at u = {at:?}; with ρ ≤ 4 only {within4:.2e}");
    }
}
