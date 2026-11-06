//! The diagonal chain against its trusted paths: the quadrature, the identity, the
//! far field, S7's RCR and the direct sum, in double-double.

use nd_fmm_spike_diagonal::grid::{Grid, phi_count, tri};
use nd_fmm_spike_diagonal::run::{Config, cloud, diagonal, diagonal_f64_built};
use nd_fmm_spike_diagonal::signature::{f2r_ring, finish, s2f_ring, times_i_pow};
use nd_fmm_spike_diagonal::transfer::{Transfer, TransferF64, log10_term};
use nd_fmm_spike_truncation::chain::{self, Kind, Setup, Work, e2p, p2e, rcr};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::geometry::{Placement, Rng, dist2};
use nd_fmm_spike_truncation::harmonics::{Harmonics, Polar, idx, len};
use nd_fmm_spike_truncation::real::{Dd, Real};
use nd_fmm_spike_truncation::rotation::{Direction, Rotation};

/// The relative ℓ2 difference of two coefficient vectors.
fn rel<T: Real>(a: &[Complex<T>], b: &[Complex<T>]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        num += (*x - *y).to_f64().norm_sqr();
        den += y.to_f64().norm_sqr();
    }
    (num / den).sqrt()
}

/// Random complex coefficients of degree p.
fn random_coeffs<T: Real>(p: usize, seed: u64) -> Vec<Complex<T>> {
    let mut rng = Rng::new(seed);
    (0..len(p))
        .map(|_| Complex::from_f64(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)))
        .collect()
}

#[test]
fn phi_count_is_padded_to_four() {
    for p in 1..200 {
        let n = phi_count(p);
        assert_eq!(n % 4, 0);
        assert!(n >= 4 * (p + 1) - 3 && n <= 4 * (p + 1));
    }
}

/// The grid integrates Yₙᵐ Yₙ′ᵐ′ to δ for every n, n′ ≤ 2p (total degree 4p = 4P − 4).
#[test]
fn grid_is_exact_to_degree_4p() {
    let p = 6;
    let grid = Grid::new(p);
    let top = 2 * p;
    let mut harm = Harmonics::<Dd>::new(top);
    let n = len(top);
    let mut gram = vec![Dd::ZERO; n * n];
    for i in 0..grid.rings() {
        for j in 0..grid.nphi {
            let s = grid.direction(i, j);
            harm.eval(&Polar::new(s), false);
            for a in 0..n {
                let wa = grid.w[i] * harm.y[a];
                for b in 0..n {
                    gram[a * n + b] += wa * harm.y[b];
                }
            }
        }
    }
    let mut worst = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            let target = if a == b { Dd::ONE } else { Dd::ZERO };
            worst = worst.max((gram[a * n + b] - target).abs().to_f64());
        }
    }
    assert!(worst < 1e-28, "worst {worst:e}");
}

/// The ring's Θ times cos/sin mφ is S7's Yₙᵐ at every sample.
#[test]
fn theta_is_s7_basis() {
    let p = 9;
    let grid = Grid::new(p);
    let mut harm = Harmonics::<Dd>::new(p);
    let mut ring = Harmonics::<Dd>::new(p);
    let mut theta: Vec<Dd> = Vec::new();
    let mut worst = 0.0f64;
    for i in [0, 3, grid.rings() - 1] {
        grid.theta(i, &mut ring, &mut theta);
        for j in [0, 1, 7, grid.nphi - 1] {
            harm.eval(&Polar::new(grid.direction(i, j)), false);
            for n in 0..=p {
                for m in 0..=n {
                    let th = theta[tri(n, m)];
                    let (c, s) = (grid.cos[j * m % grid.nphi], grid.sin[j * m % grid.nphi]);
                    let e1 = harm.y[idx(n, m as isize)] - th * c;
                    worst = worst.max(e1.abs().to_f64());
                    if m > 0 {
                        let e2 = harm.y[idx(n, -(m as isize))] - th * s;
                        worst = worst.max(e2.abs().to_f64());
                    }
                }
            }
        }
    }
    assert!(worst < 1e-29, "worst {worst:e}");
}

/// S → F → R with T ≡ 1 is the identity: double-double to 10⁻²⁸, f64 to 10⁻¹⁴.
#[test]
fn s2f_f2r_is_the_identity() {
    fn check<T: Real>(p: usize, tol: f64) {
        let grid = Grid::new(p);
        let tab = grid.tables::<T>();
        let m = random_coeffs::<T>(p, 7);
        let mut harm = Harmonics::<Dd>::new(p);
        let mut theta = Vec::new();
        let mut ab = vec![Complex::ZERO; 2 * (p + 1)];
        let mut f = vec![Complex::ZERO; grid.nphi];
        let mut acc = vec![Complex::ZERO; len(p)];
        for i in 0..grid.rings() {
            grid.theta::<T>(i, &mut harm, &mut theta);
            s2f_ring(p, &theta, &m, &tab, &mut ab, &mut f);
            f2r_ring(p, &theta, tab.w[i], &f, &tab, &mut ab, &mut acc);
        }
        finish(p, &mut acc);
        let e = rel(&acc, &m);
        assert!(e < tol, "{} p = {p}: {e:e}", T::NAME);
    }
    check::<Dd>(12, 1e-28);
    check::<f64>(40, 1e-14);
    check::<f32>(20, 1e-5);
}

/// The signature of a P2M multipole is Σ q e^{−ikŝ·(y − c)} at the samples, within the
/// truncation (double-double, p well above k a).
#[test]
fn signature_is_the_far_field() {
    let ka = 4.0;
    let p = 40;
    let mut setup = Setup::<Dd>::new(Kind::Real, ka);
    setup.s = [Dd::ONE; 3];
    let cfg = Config::default();
    let (pts, q) = cloud(&cfg, Placement::Uniform, 1, [0; 3], 32);
    let m = p2e(true, setup.k, Dd::ONE, [Dd::ZERO; 3], &pts, &q, p);
    let grid = Grid::new(p);
    let tab = grid.tables::<Dd>();
    let mut harm = Harmonics::<Dd>::new(p);
    let mut theta = Vec::new();
    let mut ab = vec![Complex::ZERO; 2 * (p + 1)];
    let mut f = vec![Complex::ZERO; grid.nphi];
    let four_pi = Dd::pi() * Dd::from_f64(4.0);
    let (mut num, mut den) = (0.0, 0.0);
    for i in [0, grid.rings() / 2, grid.rings() - 1] {
        grid.theta::<Dd>(i, &mut harm, &mut theta);
        s2f_ring(p, &theta, &m, &tab, &mut ab, &mut f);
        for (j, fj) in f.iter().enumerate() {
            let s = grid.direction(i, j);
            let mut want = Complex::ZERO;
            for (y, &qy) in pts.iter().zip(&q) {
                let dot = s[0] * y[0] + s[1] * y[1] + s[2] * y[2];
                want += (Complex::<Dd>::I * setup.k * dot * Dd::from_f64(-1.0)).exp() * qy;
            }
            // Σ q e^{−ikŝ·y} = 4π F(ŝ).
            num += (*fj * four_pi - want).to_f64().norm_sqr();
            den += want.to_f64().norm_sqr();
        }
    }
    let e = (num / den).sqrt();
    assert!(e < 1e-25, "{e:e}");
}

/// The diagonal form with L = 2P − 2 in double-double is S7's RCR at the same degree
/// (proposal §3.6: it adds no truncation of its own), on several classes and kinds.
#[test]
fn diagonal_is_rcr_in_double_double() {
    let cfg = Config::default();
    for (kind, ka, p) in [
        (Kind::Real, 4.0, 12),
        (Kind::Real, 8.0, 18),
        (Kind::Damped(0.1), 8.0, 18),
        (Kind::Real, 16.0, 26),
    ] {
        let mut setup = Setup::<Dd>::new(kind, ka);
        setup.s = [Dd::ONE; 3];
        let (pts, q) = cloud(&cfg, Placement::Corners, 1, [0; 3], 32);
        let m = p2e(true, setup.k, Dd::ONE, [Dd::ZERO; 3], &pts, &q, p);
        let grid = Grid::new(p);
        for rep in [[0, 2, 0], [1, 2, 1], [0, 0, 3], [3, 3, 3]] {
            let tr = Transfer::new(setup.k, rep, &[2 * p]);
            let got = diagonal::<Dd>(&grid, &tr, 0, &m);
            let c_t = rep.map(|x| Dd::from_f64(x as f64));
            let rot = Rotation::new(Direction::new(c_t), p);
            let table = setup.m2l_table(dist2(rep), p);
            let mut want = vec![Complex::ZERO; len(p)];
            let mut work = Work::new(p);
            rcr(&rot, &table, p, p, None, &m, &mut want, &mut work);
            let e = rel(&got, &want);
            assert!(e < 1e-20, "{kind:?} ka {ka} p {p} {rep:?}: {e:e}");
        }
    }
}

/// The double-double diagonal field is the kernel within S7's truncation at p, and the
/// f64 one agrees with it to its rounding at the farthest class, where the bound is small
/// (k|t| = 48 against L = 60).
#[test]
fn diagonal_field_is_the_direct_sum() {
    let cfg = Config::default();
    let (ka, p) = (8.0, 30);
    let mut setup = Setup::<Dd>::new(Kind::Real, ka);
    setup.s = [Dd::ONE; 3];
    let (pts, q) = cloud(&cfg, Placement::Uniform, 1, [0; 3], 64);
    let m = p2e(true, setup.k, Dd::ONE, [Dd::ZERO; 3], &pts, &q, p);
    let grid = Grid::new(p);
    let rep = [3, 3, 3];
    let (tgt, _) = cloud(&cfg, Placement::Uniform, 4, rep, 32);
    let exact = chain::direct(setup.k, &pts, &q, &tgt);
    let tr = Transfer::new(setup.k, rep, &[2 * p]);
    let c_t = rep.map(|x| Dd::from_f64(x as f64));
    let mut work = Work::new(p);
    let mut fields = Vec::new();
    let local = diagonal::<Dd>(&grid, &tr, 0, &m);
    e2p(
        true,
        setup.k,
        Dd::ONE,
        c_t,
        &local,
        p,
        &tgt,
        &mut work,
        &mut fields,
    );
    let (pot, grad) = chain::errors(&fields, &exact);
    assert!(pot < 1e-10 && grad < 1e-10, "dd: {pot:e} {grad:e}");
    let local64 = diagonal::<f64>(&grid, &tr, 0, &m);
    let mut f64_fields = Vec::new();
    e2p(
        true,
        setup.k,
        Dd::ONE,
        c_t,
        &local64,
        p,
        &tgt,
        &mut work,
        &mut f64_fields,
    );
    let (pot, grad) = chain::errors(&f64_fields, &fields);
    assert!(
        pot < 1e-13 && grad < 1e-13,
        "f64 against dd: {pot:e} {grad:e}"
    );
}

#[test]
fn times_i_pow_cycles() {
    let z = Complex::<f64>::new(1.5, -0.25);
    let mut w = z;
    for k in 0..9 {
        assert_eq!(times_i_pow(z, k), w);
        w = w * Complex::I;
    }
}

/// The 1-D seed of proposal Appendix B.1 (`seed.py`) reproduced on its geometry: one
/// source and one target on the axis at radius a on the far sides of boxes |t| = 2w and
/// 3w apart, P from §3.5's formula at 10⁻⁶, kD = 500, levels 2–5, the f64 chain against
/// the kernel, with T rounded from double-double and with T built in f64. The seed's
/// numbers (run 2026-11-06): 9.7e-8, 1.0e-4, 2.2e-2 and 1.0e1 at 2w, at most 7.0e-9 at 3w,
/// and its bound (C_T = 2, u = 1.1e-16) 2.3e-7, 7.1e-5, 2.5e-2 and 8.3. The seed integrates
/// over θ alone (the axisymmetric part); over the whole sphere the rounding errors of T at
/// the N_φ samples of a ring partly average out, so the 3D errors lie at or below the
/// seed's (measured 0.11–1.2×), never far above.
#[test]
#[ignore = "about ten seconds in release; run with --include-ignored"]
fn seed_reproduced_on_the_axis() {
    let seed_2w = [9.7e-8, 1.0e-4, 2.2e-2, 1.0e1];
    let seed_bound = [2.3e-7, 7.1e-5, 2.5e-2, 8.3];
    for ((level, &want), &bound) in (2..6).zip(&seed_2w).zip(&seed_bound) {
        let kw = 500.0 / f64::powi(2.0, level);
        let ka = 3f64.sqrt() / 2.0 * kw;
        let p = nd_fmm_spike_diagonal::rules::formula(ka, 1e-6);
        let mut setup = Setup::<Dd>::new(Kind::Real, ka);
        setup.s = [Dd::ONE; 3];
        let a = Dd::from_f64(3.0).sqrt() * Dd::from_f64(0.5);
        let src = vec![[Dd::ZERO, Dd::ZERO, -a]];
        let q = vec![Complex::<Dd>::ONE];
        let m = p2e(true, setup.k, Dd::ONE, [Dd::ZERO; 3], &src, &q, p);
        let grid = Grid::new(p);
        let mut got = [[0.0; 2]; 2];
        for (g, d) in got.iter_mut().zip([2, 3]) {
            let c_t = [Dd::ZERO, Dd::ZERO, Dd::from_f64(d as f64)];
            let tgt = vec![[Dd::ZERO, Dd::ZERO, c_t[2] + a]];
            let exact = chain::direct(setup.k, &src, &q, &tgt);
            let tr = Transfer::new(setup.k, [0, 0, d], &[2 * p]);
            let mut work = Work::new(p);
            let mut fields = Vec::new();
            let local = diagonal::<f64>(&grid, &tr, 0, &m);
            e2p(
                true,
                setup.k,
                Dd::ONE,
                c_t,
                &local,
                p,
                &tgt,
                &mut work,
                &mut fields,
            );
            g[0] = chain::errors(&fields, &exact).0;
            let built = diagonal_f64_built(&grid, &TransferF64::new(&tr, 0), &m);
            e2p(
                true,
                setup.k,
                Dd::ONE,
                c_t,
                &built,
                p,
                &tgt,
                &mut work,
                &mut fields,
            );
            g[1] = chain::errors(&fields, &exact).0;
        }
        println!(
            "level {level}: k a {ka:.1}, p {p}: 2w {:.1e} rounded, {:.1e} built (seed {want:.1e}); \
             3w {:.1e}, {:.1e}",
            got[0][0], got[0][1], got[1][0], got[1][1]
        );
        for e in got[0] {
            assert!(
                e < 2.0 * want && e > want / 20.0,
                "level {level} at 2w: {e:e}"
            );
        }
        for e in got[1] {
            assert!(e < 1e-8, "level {level} at 3w: {e:e}");
        }
        let z = setup.k.to_f64() * 2.0;
        let e = 2.0 * 1.1e-16 * 10f64.powf(log10_term(z, 2 * p));
        assert!((e / bound - 1.0).abs() < 0.05, "level {level}: bound {e:e}");
    }
}
