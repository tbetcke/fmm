//! The parts of the sweep against their trusted paths: f32, the classes, the bound's
//! Hankel maximum, the interpolated and the f64-built T, the shared L2P, determinism.

use nd_fmm_spike_diagonal::classes::{class_of, classes};
use nd_fmm_spike_diagonal::grid::{Grid, round};
use nd_fmm_spike_diagonal::rules::{EPSILONS, S7Table, calibrated, formula, s7_closed};
use nd_fmm_spike_diagonal::run::{Config, cloud, l2p_many, run_unit};
use nd_fmm_spike_diagonal::transfer::{Interp, Transfer, TransferF64, angle, log10_term};
use nd_fmm_spike_truncation::chain::{Kind, Setup, Work, e2p, p2e};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::geometry::{Placement, dist2, offsets};
use nd_fmm_spike_truncation::radial;
use nd_fmm_spike_truncation::real::{Dd, Real};

#[test]
fn f32_real_matches_std() {
    for x in [0.1f32, 1.0, 2.5, 17.0, -3.25] {
        assert_eq!(<f32 as Real>::exp(x), x.exp());
        assert_eq!(<f32 as Real>::sin_cos(x), x.sin_cos());
        assert_eq!(<f32 as Real>::sqrt(x.abs()), x.abs().sqrt());
    }
    assert_eq!(<f32 as Real>::from_f64(0.1), 0.1f32);
    assert_eq!(<f32 as Real>::EPSILON, 2f64.powi(-24));
}

/// 34 classes, sizes summing to 316, the representatives' symmetries map every offset.
#[test]
fn thirty_four_classes() {
    let cs = classes();
    assert_eq!(cs.len(), 34);
    assert_eq!(cs.iter().map(|c| c.size).sum::<usize>(), 316);
    for o in offsets() {
        let (rep, s) = class_of(o);
        assert_eq!(s.apply(o), rep);
        assert!(cs.iter().any(|c| c.rep == rep));
        assert_eq!(dist2(rep), dist2(o));
    }
    // |t| = 3w on an axis, the control, is a class of six offsets.
    let axis = cs.iter().find(|c| c.rep == [0, 0, 3]).unwrap();
    assert_eq!(axis.size, 2);
    let axis_xy = cs.iter().find(|c| c.rep == [0, 3, 0]).unwrap();
    assert_eq!(axis_xy.size, 4);
}

/// Each offset's T (double-double, rounded to f64) is its representative's read through
/// the symmetry's permutation of the samples, within an ulp of max |T|.
#[test]
fn classes_permute_the_grid() {
    let p = 7;
    let grid = Grid::new(p);
    let setup = Setup::<Dd>::new(Kind::Real, 6.0);
    let mut tv = [Complex::ZERO];
    let mut eval = |tr: &Transfer, i: usize, j: usize| {
        tr.eval(tr.mu(grid.direction(i, j)), &mut tv);
        Complex::new(round::<f64>(tv[0].re), round::<f64>(tv[0].im))
    };
    let mut worst = 0.0f64;
    for o in offsets().into_iter().step_by(7) {
        let (rep, s) = class_of(o);
        let to = Transfer::new(setup.k, o, &[2 * p]);
        let tr = Transfer::new(setup.k, rep, &[2 * p]);
        let mut max = 0.0f64;
        let mut diff = 0.0f64;
        for i in 0..grid.rings() {
            for j in 0..grid.nphi {
                let (i2, j2) = s.sample(grid.rings(), grid.nphi, i, j);
                let a = eval(&to, i, j);
                let b = eval(&tr, i2, j2);
                max = max.max(a.abs());
                diff = diff.max((a - b).abs());
            }
        }
        worst = worst.max(diff / max);
    }
    assert!(worst <= 2.3e-16, "worst {worst:e}");
}

/// log₁₀ |z| max (2j + 1)|hⱼ(z)| against S7's double-double recursion where it stays in
/// range, for real and complex z.
#[test]
fn bound_term_matches_double_double() {
    for (z, l) in [
        (Complex::new(5.0, 0.0), 30),
        (Complex::new(20.0, 0.0), 80),
        (Complex::new(20.0, 2.0), 80),
        (Complex::new(100.0, 0.0), 230),
        (Complex::new(3.0, 0.3), 60),
    ] {
        let mut h = vec![Complex::<Dd>::ZERO; l + 1];
        radial::hankel(Complex::from_f64(z.re, z.im), Dd::ONE, &mut h);
        let want = h
            .iter()
            .enumerate()
            .map(|(j, hj)| ((2 * j + 1) as f64 * hj.to_f64().abs()).log10())
            .fold(f64::MIN, f64::max)
            + z.abs().log10();
        let got = log10_term(z, l);
        assert!((got - want).abs() < 1e-12, "{z:?} {l}: {got} {want}");
    }
    // Beyond f64's range it stays finite.
    assert!(log10_term(Complex::new(2.0, 0.0), 400) > 400.0);
}

/// The interpolated T against double-double T at the angle ψ ([`angle`]), within 30 u
/// max |T| over the table (a few roundings per term, the weights' Σ|w| ≈ 2.6) (which holds the peaks of |T| at ψ = 0 and π; the samples
/// lie within the window of them).
#[test]
fn interpolated_transfer_function() {
    let cfg = Config::default();
    for (ka, p, rep) in [
        (8.0, 14, [0, 2, 0]),
        (30.0, 50, [1, 2, 1]),
        (60.0, 90, [3, 3, 3]),
    ] {
        let setup = Setup::<Dd>::new(Kind::Real, ka);
        let grid = Grid::new(p);
        let tr = Transfer::new(setup.k, rep, &[2 * p]);
        let it = Interp::new(&tr, 0, cfg.sigma, cfg.half);
        let mut tv = [Complex::ZERO];
        let mut diff = 0.0f64;
        for i in (0..grid.rings()).step_by(3) {
            for j in (0..grid.nphi).step_by(5) {
                let psi = angle(grid.direction(i, j), tr.that);
                tr.eval(psi.sin_cos().1, &mut tv);
                let want = tv[0].to_f64();
                let got = it.eval(psi);
                diff = diff.max((got - want).abs());
            }
        }
        let max = it.max_abs();
        assert!(
            diff <= 30.0 * 1.1e-16 * max,
            "k a {ka} p {p}: {:e}",
            diff / max
        );
    }
}

/// T built in f64 against double-double T at the same f64 μ: to round-off where the terms
/// stay small (the farthest class), and between u and L² u times the largest term
/// max (2j + 1)|hⱼ| where they grow (the term's rounding, amplified by the f64
/// recurrence of Pₗ).
#[test]
fn f64_built_transfer_function() {
    let (ka, p) = (16.0, 30);
    let setup = Setup::<Dd>::new(Kind::Real, ka);
    let grid = Grid::new(p);
    let tr = Transfer::new(setup.k, [3, 3, 3], &[2 * p]);
    let tb = TransferF64::new(&tr, 0);
    let mut tv = [Complex::ZERO];
    let (mut max, mut diff) = (0.0f64, 0.0f64);
    for i in 0..grid.rings() {
        for j in 0..grid.nphi {
            let mu = tb.mu(grid.direction(i, j).map(round::<f64>));
            tr.eval(Dd::from_f64(mu), &mut tv);
            max = max.max(tv[0].to_f64().abs());
            diff = diff.max((tb.eval_mu(mu) - tv[0].to_f64()).abs());
        }
    }
    assert!(diff <= 1e-14 * max, "farthest class: {:e}", diff / max);
    let rep = [0, 2, 1];
    let setup = Setup::<Dd>::new(Kind::Real, ka);
    let grid = Grid::new(p);
    let tr = Transfer::new(setup.k, rep, &[2 * p]);
    let tb = TransferF64::new(&tr, 0);
    let l = (2 * p) as f64;
    let term = 10f64.powf(log10_term(setup.k.to_f64() * 5f64.sqrt(), 2 * p))
        / (setup.k.to_f64().abs() * 5f64.sqrt());
    let mut diff = 0.0f64;
    for i in 0..grid.rings() {
        for j in 0..grid.nphi {
            let mu = tb.mu(grid.direction(i, j).map(round::<f64>));
            tr.eval(Dd::from_f64(mu), &mut tv);
            diff = diff.max((tb.eval_mu(mu) - tv[0].to_f64()).abs());
        }
    }
    assert!(diff <= l * l * 1.1e-16 * term, "{diff:e} {term:e}");
    assert!(diff >= 1.1e-16 * term, "not built in f64? {diff:e}");
}

/// The shared L2P is S7's `e2p` bit for bit.
#[test]
fn l2p_many_is_s7_e2p() {
    let cfg = Config::default();
    let p = 12;
    let mut setup = Setup::<Dd>::new(Kind::Damped(0.1), 9.0);
    setup.s = [Dd::ONE; 3];
    let (pts, q) = cloud(&cfg, Placement::Patch, 1, [0; 3], 16);
    let a = p2e(true, setup.k, Dd::ONE, [Dd::ZERO; 3], &pts, &q, p);
    let b: Vec<_> = a.iter().map(|z| *z * Complex::I).collect();
    let c_t = [2.0, 1.0, 0.0].map(Dd::from_f64);
    let (tgt, _) = cloud(&cfg, Placement::Patch, 4, [2, 1, 0], 8);
    let many = l2p_many(setup.k, c_t, &[&a, &b], p, &tgt);
    let mut work = Work::new(p);
    for (e, got) in [&a, &b].iter().zip(&many) {
        let mut want = Vec::new();
        e2p(
            true,
            setup.k,
            Dd::ONE,
            c_t,
            e,
            p,
            &tgt,
            &mut work,
            &mut want,
        );
        assert_eq!(got, &want);
    }
}

/// The rules: S7's table read (4 kinds × 29 k a), the closed form off the grid,
/// proposal §3.5's numbers and CH3.3's plateau.
#[test]
fn degree_rules() {
    let s7 = S7Table::load();
    // real and η = 0.1 to k a = 256, η = 1 and Yukawa to 32.
    assert_eq!(s7.len(), 2 * 29 + 2 * 23);
    assert_eq!(s7.degree("real", 256.0, 256.0, 3), (288, false));
    assert_eq!(s7.degree("real", 2.828, 2.828, 0), (11, false));
    assert_eq!(s7.degree("complex0.1", 22.627, 22.627, 7), (62, false));
    assert_eq!(
        s7.degree("real", 400.0, 400.0, 3),
        (s7_closed(400.0, 1e-6), true)
    );
    // §3.5's example table at kD = 500, 10⁻⁶: level 2 k a 108, p 144; level 5 33.
    assert_eq!(formula(108.25, 1e-6), 144);
    assert_eq!(formula(13.53, 1e-6), 33);
    // CH3.3's plateau 7, 9, 12, 15, 20, 24 at 10⁻³ … 10⁻⁸; level 2 at kD = 500, 10⁻⁶: 131.
    let plateau: Vec<usize> = EPSILONS[..6].iter().map(|&e| calibrated(0.1, e)).collect();
    assert_eq!(plateau, [7, 9, 12, 15, 20, 24]);
    assert_eq!(calibrated(108.25, 1e-6), 131);
}

/// A unit's results do not depend on the thread count.
#[test]
fn unit_is_deterministic() {
    let mut cfg = Config::default();
    let ids = [0, 5, 33];
    cfg.threads = 1;
    let a = run_unit(Kind::Real, 8.0, 10, &ids, &cfg);
    cfg.threads = 3;
    let b = run_unit(Kind::Real, 8.0, 10, &ids, &cfg);
    for (x, y) in a.classes.iter().zip(&b.classes) {
        assert_eq!(x.round, y.round);
        assert_eq!(x.total, y.total);
        assert_eq!(x.trunc, y.trunc);
        assert_eq!(
            x.log10_max_t.map(f64::to_bits),
            y.log10_max_t.map(f64::to_bits)
        );
    }
}
