//! The quadrature generator of the exponential M2L (Helmholtz CH6.1b; S4's
//! `tests/generator.rs`) against the kernel e^{iκ|u|}/|u|, the integrand's closed form:
//! on the region's grid (where it was fitted), at seeded points off it, and in the
//! point-to-point chain. Errors are absolute, in box widths (FMM3D's norm, S4 §1).
//!
//! The degree-matched quadrature (Helmholtz CH6.1f, `generate_for_degree`) against RCR at
//! its degree (`leaf::p2m`, `rcr::m2l`, `leaf::l2p` of a unit charge), its chain summed here
//! term by term from the truncated plane-wave expansions (Gegenbauer's series, unscaled), at
//! seeded sources and targets that the generator did not fit.

use core::f64::consts::TAU;

use nd_fmm_math::Complex;
use nd_fmm_math::Layout;
use nd_fmm_math::radial::bessel_j;
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::plane_wave::{Quadrature, generate, generate_for_degree, kernel};
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf, rcr};

type C = Complex<f64>;

/// A seeded generator (SplitMix64), so every sample is reproducible.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The chain's sources in the source box [−½, ½]³: its centre, its corners and
/// `interior` seeded points (S4's `Pairs::up`).
fn sources(interior: usize) -> Vec<[f64; 3]> {
    let mut sources = vec![[0.0; 3]];
    for b in 0..8 {
        let h = |bit: usize| if b & bit != 0 { 0.5 } else { -0.5 };
        sources.push([h(4), h(2), h(1)]);
    }
    let mut rng = SplitMix(0x5eed_c404);
    for _ in 0..interior {
        sources.push([rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5]);
    }
    sources
}

/// The chain's targets: x, y ∈ {−3.5, −3, …, 3.5}, z ∈ {1.5, 2, …, 3.5}, the corners,
/// edge and face midpoints and centres of the up direction's target boxes.
fn targets() -> Vec<[f64; 3]> {
    let xy: Vec<f64> = (0..15).map(|i| -3.5 + 0.5 * i as f64).collect();
    let mut out = Vec::new();
    for z in (0..5).map(|i| 1.5 + 0.5 * i as f64) {
        for &y in &xy {
            for &x in &xy {
                out.push([x, y, z]);
            }
        }
    }
    out
}

/// The largest |chain − G| over the sources and targets: each source's waves at λₖ,
/// e^{λₖ s_z − i tₖ (s_x cos α + s_y sin α)}, truncated to the orders |m| ≤ Fₖ (their
/// coefficients by a DFT fine enough to be exact to round-off), sampled at the Nₖ α
/// nodes and evaluated at each target (S4's `chain_sums`, term by term).
fn chain_error(q: &Quadrature, sources: &[[f64; 3]], targets: &[[f64; 3]]) -> f64 {
    let mut worst = 0.0f64;
    for s in sources {
        let mut sums = vec![C::new(0.0, 0.0); targets.len()];
        for k in 0..q.nodes() {
            let n = q.alpha_counts()[k];
            let f = q.fourier_counts()[k] as i64;
            let (lambda, t) = (q.lambda()[k], q.t()[k]);
            let c = q.weights()[k] / n as f64;
            let wave = |alpha: f64| {
                let (sa, ca) = alpha.sin_cos();
                (lambda * s[2] - C::i() * t * (s[0] * ca + s[1] * sa)).exp()
            };
            let m = 2 * (f as usize + (t.norm() * 0.75).ceil() as usize + 32);
            let samples: Vec<C> = (0..m).map(|l| wave(TAU * l as f64 / m as f64)).collect();
            let coeffs: Vec<C> = (-f..=f)
                .map(|order| {
                    samples
                        .iter()
                        .enumerate()
                        .map(|(l, v)| {
                            v * C::from_polar(1.0, -TAU * (order * l as i64) as f64 / m as f64)
                        })
                        .sum::<C>()
                        / m as f64
                })
                .collect();
            for j in 0..n {
                let a = TAU * j as f64 / n as f64;
                let v: C = (-f..=f)
                    .zip(&coeffs)
                    .map(|(order, c)| c * C::from_polar(1.0, order as f64 * a))
                    .sum();
                let (sa, ca) = a.sin_cos();
                for (sum, u) in sums.iter_mut().zip(targets) {
                    *sum += c * v * (-lambda * u[2] + C::i() * t * (u[0] * ca + u[1] * sa)).exp();
                }
            }
        }
        for (sum, u) in sums.iter().zip(targets) {
            let d = [u[0] - s[0], u[1] - s[1], u[2] - s[2]];
            worst = worst.max((sum - kernel(q.kappa(), d)).norm());
        }
    }
    worst
}

/// S4's checks: ε on the grid, 2ε at 200 seeded points off it, 2ε in the chain (the
/// centre, the corners and two seeded sources), every tₖ real, no shortfall.
fn check(kw: C, eps: f64) {
    let q = generate(kw, eps);
    assert!(
        q.shortfall().is_none(),
        "{kw} at {eps:e}: {:?}",
        q.shortfall()
    );
    let on_grid = q.kernel_error();
    assert!(on_grid <= eps, "{kw} at {eps:e}: grid {on_grid:.2e}");
    let mut rng = SplitMix(7);
    let off = (0..200)
        .map(|_| {
            let u = [
                8.0 * rng.unit() - 4.0,
                8.0 * rng.unit() - 4.0,
                1.0 + 3.0 * rng.unit(),
            ];
            (q.evaluate(u) - kernel(kw, u)).norm()
        })
        .fold(0.0, f64::max);
    assert!(off <= 2.0 * eps, "{kw} at {eps:e}: off the grid {off:.2e}");
    let chain = chain_error(&q, &sources(2), &targets());
    assert!(chain <= 2.0 * eps, "{kw} at {eps:e}: chain {chain:.2e}");
    // On the contour of real t the waves never grow.
    for t in q.t() {
        assert!(t.im.abs() <= 1e-10 * t.norm().max(1.0), "{kw}: t = {t}");
    }
}

/// S4's five debug cases by [`check`]: absolute |Q − G| in box widths.
#[test]
fn the_generator_meets_eps() {
    for (kw, eps) in [
        (C::new(0.0, 0.0), 1e-6),
        (C::new(5.0, 0.0), 1e-3),
        (C::new(0.0, 5.0), 1e-6),
        (C::new(10.0, 3.0), 1e-3),
        // S4's worst chain before the Fourier counts took the maximum over the larger
        // orders (a zero of J_{F+1} stopped the count early): 1.0e-2 at 1.4e-4.
        (C::new(17.529, 0.876), 1.4e-4),
    ] {
        check(kw, eps);
    }
}

/// S4's full set by [`check`] (absolute |Q − G| in box widths; slow unoptimised): `cargo test --release -p nd-fmm-ref --test helmholtz
/// plane_wave -- --ignored`.
#[test]
#[ignore = "slow in the dev profile; run with --release"]
fn the_generator_meets_eps_everywhere() {
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
    // S4's other large chain errors before that fix.
    for (kw, eps) in [
        (C::new(28.524, 14.262), 1.5e-10),
        (C::new(37.949, 1.897), 1.1e-3),
        (C::new(6.533, 0.0), 1.9e-7),
    ] {
        check(kw, eps);
    }
}

/// Two calls give the same bits: λₖ, tₖ, wₖ and the counts.
#[test]
fn the_generator_is_deterministic() {
    let kw = C::new(5.0, 0.0);
    let (a, b) = (generate(kw, 1e-3), generate(kw, 1e-3));
    let bits = |v: &[C]| -> Vec<(u64, u64)> {
        v.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
    };
    assert_eq!(bits(a.lambda()), bits(b.lambda()));
    assert_eq!(bits(a.t()), bits(b.t()));
    assert_eq!(bits(a.weights()), bits(b.weights()));
    assert_eq!(a.alpha_counts(), b.alpha_counts());
    assert_eq!(a.fourier_counts(), b.fourier_counts());
}

#[test]
#[should_panic(expected = "outside the closed first quadrant")]
fn a_negative_kappa_panics() {
    let _ = generate(C::new(-1.0, 0.0), 1e-3);
}

/// e^{σ q·x} truncated at degree p, σ = ±i: Σ_{n≤p} (2n + 1) σⁿ jₙ(κ|x|) Pₙ(q·x/(κ|x|)),
/// unscaled (jₙ from `bessel_j` at scale 1, Pₙ by Legendre's recurrence), for the tests'
/// κ |x| ≥ 10⁻³.
fn truncated(kappa: C, p: usize, q: [C; 3], x: [f64; 3], sign: f64) -> C {
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    if r == 0.0 {
        return C::new(1.0, 0.0);
    }
    let z = kappa * r;
    let mut j = vec![C::new(0.0, 0.0); p + 1];
    bessel_j(z, 1.0, &mut j, None);
    let arg = (q[0] * x[0] + q[1] * x[1] + q[2] * x[2]) / z;
    let (mut prev, mut leg) = (C::new(0.0, 0.0), C::new(1.0, 0.0));
    let (sigma, mut power, mut sum) = (C::new(0.0, sign), C::new(1.0, 0.0), C::new(0.0, 0.0));
    for (n, jn) in j.iter().enumerate() {
        sum += power * jn * z.im.exp() * leg * (2 * n + 1) as f64;
        let next = (arg * leg * (2 * n + 1) as f64 - prev * n as f64) / (n + 1) as f64;
        prev = leg;
        leg = next;
        power *= sigma;
    }
    sum
}

/// The chain of a degree-matched quadrature for a unit charge at s (from its box's
/// centre), a target at y (from its box's centre) and the centres' offset d, term by term:
/// Σₖ (wₖ/Nₖ) Σⱼ S_p e^{i q·d} T_p, q = (tₖ cos αⱼ, tₖ sin αⱼ, iλₖ).
fn degree_chain(q: &Quadrature, s: [f64; 3], y: [f64; 3], d: [f64; 3]) -> C {
    let p = q.degree().expect("a degree-matched quadrature");
    let kappa = q.kappa();
    (0..q.nodes())
        .map(|k| {
            let n = q.alpha_counts()[k];
            let (lambda, t) = (q.lambda()[k], q.t()[k]);
            (0..n)
                .map(|j| {
                    let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
                    let v = [t * ca, t * sa, C::i() * lambda];
                    let shift = (C::i() * (v[0] * d[0] + v[1] * d[1] + v[2] * d[2])).exp();
                    truncated(kappa, p, v, s, -1.0) * shift * truncated(kappa, p, v, y, 1.0)
                })
                .sum::<C>()
                * q.weights()[k]
                / n as f64
        })
        .sum()
}

/// RCR at degree p for a unit charge at s, the target y + d (box widths, box width 1).
fn rcr_value(kappa: C, p: usize, s: [f64; 3], y: [f64; 3], d: [f64; 3]) -> C {
    let level = Level::new(kappa, 0.5);
    let mut ws = Workspace::new(p);
    let len = Layout::new(p).len();
    let (from, to) = (Frame::new([0.0; 3], 0.5), Frame::new(d, 0.5));
    let mut m = vec![0.0; 2 * len];
    leaf::p2m(p, &level, &from, &[s], &[C::new(1.0, 0.0)], &mut ws, &mut m);
    let table = rcr::m2l_table(p, p, &level, &from, &to);
    let mut local = vec![0.0; 2 * len];
    rcr::m2l(p, p, &level, &from, &to, &table, &mut ws, &m, &mut local);
    let x = [core::array::from_fn(|a| d[a] + y[a])];
    let mut out = [C::new(0.0, 0.0)];
    leaf::l2p(p, &level, &to, &local, &x, &mut ws, &mut out, None);
    out[0]
}

/// The largest |chain − RCR| of a degree-matched quadrature at `pairs` seeded sources and
/// targets in their boxes and the centre offsets (0, 0, 2), (3, −3, 2), (−2, 1, 3) and
/// (3, 3, 3), box widths per unit charge.
fn degree_error(q: &Quadrature, pairs: usize) -> f64 {
    let p = q.degree().expect("a degree-matched quadrature");
    let mut rng = SplitMix(0x0c46_1f00 + p as u64);
    let mut point = || [(); 3].map(|_| rng.unit() - 0.5);
    let points: Vec<([f64; 3], [f64; 3])> = (0..pairs).map(|_| (point(), point())).collect();
    let mut worst = 0.0f64;
    for d in [
        [0.0, 0.0, 2.0],
        [3.0, -3.0, 2.0],
        [-2.0, 1.0, 3.0],
        [3.0, 3.0, 3.0],
    ] {
        for &(s, y) in &points {
            let e = degree_chain(q, s, y, d) - rcr_value(q.kappa(), p, s, y, d);
            worst = worst.max(e.norm());
        }
    }
    worst
}

/// The degree-matched quadrature (CH6.1f) meets ε against RCR at its degree off its
/// samples, within 2ε at seeded points (absolute, box widths per unit charge), without a
/// shortfall; it keeps every order to its degree and states the degree.
fn check_degree(kappa: C, eps: f64, p: usize, pairs: usize) {
    let q = generate_for_degree(kappa, eps, p);
    assert!(q.shortfall().is_none(), "{:?}", q.shortfall());
    assert_eq!(q.degree(), Some(p));
    assert_eq!(q.eps(), eps);
    assert!(q.fourier_counts().iter().all(|&f| f == p));
    assert_eq!(q.wave_count(), q.alpha_counts().iter().sum::<usize>());
    let e = degree_error(&q, pairs);
    println!(
        "κ = {kappa}, ε = {eps:.0e}, p = {p}: {} nodes, {} waves, |chain − RCR| {e:.2e} \
         ({:.2} ε)",
        q.nodes(),
        q.wave_count(),
        e / eps
    );
    assert!(e <= 2.0 * eps, "{:.2} ε", e / eps);
}

#[test]
fn the_degree_matched_quadrature_meets_rcr() {
    check_degree(C::new(1.25, 0.0), 1e-5, 6, 2);
}

#[test]
#[ignore = "release: about 5 minutes on the M6"]
fn the_degree_matched_quadrature_meets_rcr_at_the_fmms_degrees() {
    for (re, im) in [(0.0078, 0.0), (0.31, 0.0), (2.5, 0.0), (1.25, 0.1875)] {
        for (eps, p) in [(1e-7, 20), (1e-8, 24)] {
            check_degree(C::new(re, im), eps / 20.0, p, 4);
        }
    }
}

/// Two calls give the same bits.
#[test]
fn the_degree_matched_quadrature_is_deterministic() {
    let kappa = C::new(0.5, 0.25);
    let (a, b) = (
        generate_for_degree(kappa, 1e-4, 4),
        generate_for_degree(kappa, 1e-4, 4),
    );
    let bits = |v: &[C]| -> Vec<(u64, u64)> {
        v.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
    };
    assert_eq!(bits(a.lambda()), bits(b.lambda()));
    assert_eq!(bits(a.weights()), bits(b.weights()));
    assert_eq!(a.alpha_counts(), b.alpha_counts());
    assert_eq!(a.fourier_counts(), b.fourier_counts());
}

/// `generate`'s quadratures state no degree.
#[test]
fn a_kernel_quadrature_has_no_degree() {
    assert_eq!(generate(C::new(5.0, 0.0), 1e-3).degree(), None);
}

#[test]
#[should_panic(expected = "κ = 0 is Laplace")]
fn a_zero_kappa_panics_for_a_degree() {
    let _ = generate_for_degree(C::new(0.0, 0.0), 1e-3, 4);
}
