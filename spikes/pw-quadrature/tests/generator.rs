//! The generator against the kernel: on the region grid, at seeded points off the grid
//! (the compression was fitted on the grid), and in the chain.

use nd_fmm_spike_pw_quadrature::generator::{self, generate};
use nd_fmm_spike_pw_quadrature::region::{Grid, Pairs, SplitMix};
use nd_fmm_spike_pw_quadrature::{C, kernel};

fn check(kw: C, eps: f64) {
    let q = generate(kw, eps);
    let grid = Grid::region();
    let on_grid = q.kernel_errors(&grid).absolute;
    assert!(on_grid <= eps, "{kw} at {eps:e}: grid {on_grid:.2e}");
    let mut rng = SplitMix(7);
    let off = (0..200)
        .map(|_| {
            let u = [
                8.0 * rng.unit() - 4.0,
                8.0 * rng.unit() - 4.0,
                1.0 + 3.0 * rng.unit(),
            ];
            (q.evaluate(u, &q.physical) - kernel(kw, u)).norm()
        })
        .fold(0.0, f64::max);
    assert!(off <= 2.0 * eps, "{kw} at {eps:e}: off the grid {off:.2e}");
    let chain = q.chain_errors(&Pairs::up(2), true).absolute;
    assert!(chain <= 2.0 * eps, "{kw} at {eps:e}: chain {chain:.2e}");
    // On the contour of real t the waves never grow.
    for t in &q.t {
        assert!(t.im.abs() <= 1e-10 * t.norm().max(1.0), "{kw}: t = {t}");
    }
}

#[test]
fn meets_eps() {
    for (kw, eps) in [
        (C::new(0.0, 0.0), 1e-6),
        (C::new(5.0, 0.0), 1e-3),
        (C::new(0.0, 5.0), 1e-6),
        (C::new(10.0, 3.0), 1e-3),
        // The rays' worst chain before the Fourier counts took the maximum over the
        // larger orders (a zero of J_{F+1} stopped the count early): 1.0e-2 at 1.4e-4.
        (C::new(17.529, 0.876), 1.4e-4),
    ] {
        check(kw, eps);
    }
}

/// The plan's full set (slow unoptimised): `cargo test --release -p
/// nd-fmm-spike-pw-quadrature -- --ignored`.
#[test]
#[ignore = "slow in the dev profile; run with --release"]
fn meets_eps_everywhere() {
    let pi = core::f64::consts::PI;
    for kw in [
        C::new(0.0, 0.0),
        C::new(5.0, 0.0),
        C::new(20.0, 0.0),
        C::new(16.0 * pi, 0.0),
        C::new(0.0, 5.0),
        C::new(0.0, 30.0),
        C::new(10.0, 3.0),
        C::new(40.0, 20.0),
        C::new(10.0, 1e-3),
    ] {
        for eps in [1e-3, 1e-6, 1e-9] {
            check(kw, eps);
        }
    }
    // The rays' other large chain errors before the fix.
    for (kw, eps) in [
        (C::new(28.524, 14.262), 1.5e-10),
        (C::new(37.949, 1.897), 1.1e-3),
        (C::new(6.533, 0.0), 1.9e-7),
    ] {
        check(kw, eps);
    }
}

/// The λ rule converges: twice the panels change the sums by less than ε/4.
#[test]
fn panels_converge() {
    let (kw, eps) = (C::new(10.0, 0.0), 1e-6);
    let (_, n, _) = generator::lambda_rule(kw, eps, eps / 4.0);
    let a = generator::panel_rule(kw, eps, n);
    let b = generator::panel_rule(kw, eps, 2 * n);
    let sum = |r: &generator::Rule, z: f64, rho: f64| -> C {
        r.t.iter()
            .zip(&r.lambda)
            .zip(&r.weight)
            .map(|((&t, &l), &w)| {
                w * (-l * z).exp() * nd_fmm_spike_pw_quadrature::bessel::j_all(t * rho, 0)[0]
            })
            .sum()
    };
    for z in [1.0, 2.5, 4.0] {
        for rho in [0.0, 2.0, 5.6] {
            let d = (sum(&a, z, rho) - sum(&b, z, rho)).norm();
            assert!(d <= eps / 4.0, "z {z} ρ {rho}: {d:.2e}");
        }
    }
}
