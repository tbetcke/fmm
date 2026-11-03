//! FMM3D's tables, from the fixture (a few domains, `tools/extract_fixture.sh`): the
//! parser, the domain lookup, the node maps and the evaluator, against the kernel.

use nd_fmm_spike_pw_quadrature::fmm3d::{self, Tables};
use nd_fmm_spike_pw_quadrature::quadrature::Quadrature;
use nd_fmm_spike_pw_quadrature::region::{Grid, Pairs};
use nd_fmm_spike_pw_quadrature::{C, kernel};
use std::path::Path;

fn fixture() -> Tables {
    Tables::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

const DOMAINS: [usize; 5] = [1, 11, 139, 172, 236];

#[test]
fn parses_the_fixture() {
    let t = fixture();
    for p in [2, 3] {
        assert_eq!(t.precisions[p].keys().copied().collect::<Vec<_>>(), DOMAINS);
        for (&d, dom) in &t.precisions[p] {
            let (lo, hi) = fmm3d::rectangle(d);
            assert!(
                (dom.rk0 - lo).norm() <= 1e-14 * hi.norm(),
                "{p}/{d}: {} vs {lo}",
                dom.rk0
            );
            assert!(
                (dom.rk1 - hi).norm() <= 1e-14 * hi.norm(),
                "{p}/{d}: {} vs {hi}",
                dom.rk1
            );
            assert_eq!(dom.kind, if d < 150 { 2 } else { 1 }, "{p}/{d}");
            assert!(!dom.is_empty());
            assert_eq!(dom.ws.len(), dom.len());
            assert_eq!(dom.fourier.len(), dom.len());
            assert_eq!(dom.physical.len(), dom.len());
            assert!(dom.stated_error > 0.0 && dom.stated_error <= fmm3d::errmax(p));
        }
    }
    // Domain 1 at iprec 2 as the file has it.
    let d = &t.precisions[2][&1];
    assert_eq!((d.len(), d.kind), (21, 2));
    assert_eq!(d.xs[0], 0.4622470085766964e-03);
    assert_eq!(d.ws[0], 0.1179127434939096e-02);
    assert_eq!(d.stated_error, 0.6722916952900796e-07);
}

#[test]
fn domain_lookup_and_precision() {
    let pi = core::f64::consts::PI;
    assert_eq!(fmm3d::domain_of(C::new(0.0, 0.0)), Some(1));
    assert_eq!(fmm3d::domain_of(C::new(0.019, 0.019)), Some(1));
    assert_eq!(fmm3d::domain_of(C::new(0.02, 0.0)), Some(2));
    assert_eq!(fmm3d::domain_of(C::new(10.0, 0.0)), Some(11));
    assert_eq!(fmm3d::domain_of(C::new(10.0, 3.0)), Some(172));
    assert_eq!(fmm3d::domain_of(C::new(-1.0, -1.0)), Some(1));
    assert_eq!(
        fmm3d::domain_of(C::new(16.0 * pi - 1e-9, 12.0 * pi - 1e-9)),
        Some(437)
    );
    assert_eq!(fmm3d::domain_of(C::new(16.0 * pi, 0.0)), None);
    assert_eq!(fmm3d::domain_of(C::new(0.0, 12.0 * pi)), None);
    for d in 1..=fmm3d::DOMAINS {
        let (lo, hi) = fmm3d::rectangle(d);
        assert_eq!(fmm3d::domain_of(0.5 * (lo + hi)), Some(d));
        assert_eq!(fmm3d::domain_of(lo), Some(d));
    }
    let p: Vec<usize> = [1e-2, 5e-3, 1e-3, 5e-4, 1e-6, 5e-7, 1e-8, 5e-10, 1e-12]
        .iter()
        .map(|&e| fmm3d::precision_of(e))
        .collect();
    assert_eq!(p, [0, 0, 1, 1, 2, 2, 3, 3, 4]);
}

/// The λ sums meet the stated errors at the domain's corners and centre (α resolved),
/// so the node maps and the normalisation are FMM3D's; with FMM3D's α counts the
/// kernel's error is a few times larger (as for Laplace, CONVENTIONS §3.14's spike).
#[test]
fn reproduces_the_kernel() {
    let t = fixture();
    let grid = Grid::region();
    for p in [2, 3] {
        for (&d, dom) in &t.precisions[p] {
            let (lo, hi) = fmm3d::rectangle(d);
            for kw in [lo, 0.5 * (lo + hi), C::new(hi.re - 1e-9, lo.im)] {
                let q = Quadrature::from_fmm3d(dom, kw);
                let l = q.lambda_errors(&grid).absolute;
                let e = q.kernel_errors(&grid).absolute;
                assert!(
                    l <= 1.5 * dom.stated_error,
                    "{p}/{d} at {kw}: λ {l:.2e} vs {:.2e}",
                    dom.stated_error
                );
                assert!(
                    e <= 30.0 * dom.stated_error,
                    "{p}/{d} at {kw}: kernel {e:.2e}"
                );
            }
        }
    }
}

/// The grid sums against the term-by-term evaluation, and the chain without
/// truncation against the evaluation at t − s.
#[test]
fn fast_sums_against_the_slow_path() {
    let t = fixture();
    let kw = C::new(10.0, 3.0);
    let q = Quadrature::from_fmm3d(t.lookup(2, kw).unwrap(), kw);
    let grid = Grid::region();
    let worst_fast = q.kernel_errors(&grid).absolute;
    let worst_slow = grid
        .points()
        .map(|u| (q.evaluate(u, &q.physical) - kernel(kw, u)).norm())
        .fold(0.0, f64::max);
    assert!(
        (worst_fast - worst_slow).abs() <= 1e-14,
        "{worst_fast:e} vs {worst_slow:e}"
    );
    let pairs = Pairs::up(2);
    for &s in &pairs.sources {
        let sums = q.chain_sums(s, &pairs, false);
        let mut i = 0;
        for &z in &pairs.target_z {
            for &y in &pairs.target_xy {
                for &x in &pairs.target_xy {
                    let direct = q.evaluate([x - s[0], y - s[1], z - s[2]], &q.physical);
                    assert!((sums[i] - direct).norm() <= 1e-13, "{s:?} → {x} {y} {z}");
                    i += 1;
                }
            }
        }
    }
    // Truncated to FMM3D's Fourier counts, the chain stays within a small multiple of
    // the kernel's error.
    let c = q.chain_errors(&pairs, true).absolute;
    assert!(
        c <= 30.0 * worst_fast,
        "chain {c:.2e} vs kernel {worst_fast:.2e}"
    );
}

/// At κ → 0 FMM3D's Helmholtz table (domain 1) and its Laplace table of CONVENTIONS
/// §3.14 (`spikes/plane-wave`) represent 1/|u| on the same region: both within their
/// stated errors, the Helmholtz one with complex nodes on a contour from −iκ.
#[test]
fn laplace_limit() {
    use nd_fmm_spike_plane_wave::quadrature::Quadrature as Laplace;
    let t = fixture();
    let dom = &t.precisions[3][&1];
    let q = Quadrature::from_fmm3d(dom, C::new(0.0, 0.0));
    let laplace = Laplace::with_nodes(29);
    let grid = Grid::region();
    for u in grid.points().step_by(7) {
        let exact = kernel(C::new(0.0, 0.0), u);
        let h = q.evaluate(u, &q.converged_alpha(6.0));
        assert!((h - exact).norm() <= 1.05 * dom.stated_error, "{u:?}");
        let mut l = 0.0;
        for k in 0..laplace.nodes() {
            let (lam, n) = (laplace.lambdas[k], 400);
            let inner: f64 = (0..n)
                .map(|j| {
                    let a = core::f64::consts::TAU * j as f64 / n as f64;
                    (lam * (u[0] * a.cos() + u[1] * a.sin())).cos()
                })
                .sum();
            l += laplace.weights[k] / n as f64 * (-lam * u[2]).exp() * inner;
        }
        assert!((l - exact.re).abs() <= 1.05 * laplace.max_error, "{u:?}");
    }
}
