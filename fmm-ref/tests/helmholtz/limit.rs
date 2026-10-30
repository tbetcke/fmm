//! The limit k → 0 (proposal §9 item 3, CONVENTIONS §3.15): the Helmholtz direct sum
//! against Laplace's `p2p::direct_sum` on a fixed set of points.
//!
//! Error measure: per target, relative to Laplace's term magnitudes Σⱼ |qⱼ|/rⱼ (potential)
//! and Σⱼ |qⱼ|/rⱼ² (gradient), and for Im φ̃/k relative to Σⱼ |qⱼ|. The bounds are the
//! Taylor remainders, with w = ikr and |w| ≤ 1:
//!
//! - every kind of k: |e^w − 1| ≤ |w| (1 + |w|), so |φ̃ − φ_L| ≤ |k| r_max (1 + |k| r_max);
//!   |(w − 1) e^w + 1| ≤ |w|², so |∇φ̃ − ∇φ_L| ≤ (|k| r_max)²;
//! - real k (the tolerances of §9, (k a)² for real parts and k a for the imaginary part
//!   of degree 0): 1 − cos x ≤ x²/2, so |Re φ̃ − φ_L| ≤ (k r_max)²/2; |sin x/x − 1| ≤ x²/6,
//!   so |Im φ̃/k − Σq| ≤ (k r_max)²/6; |x cos x − sin x| ≤ x³/3, so
//!   |Im ∇φ̃| ≤ (k r_max)³/3;
//!
//! each plus 1e-14 for round-off.

use nd_fmm_ref::helmholtz::p2p as helmholtz;
use nd_fmm_ref::p2p as laplace;

use crate::common::{C, SplitMix64, Worst, complexify, distance};

const ROUND_OFF: f64 = 1e-14;

struct Points {
    sources: Vec<[f64; 3]>,
    charges: Vec<f64>,
    /// The sources (self terms skipped), then further targets.
    targets: Vec<[f64; 3]>,
    r_max: f64,
}

fn points() -> Points {
    let mut rng = SplitMix64::new(0x3a_2026_1030);
    let sources = rng.points(100, [0.5; 3], 0.5);
    let charges = rng.real_charges(100);
    let mut targets = sources.clone();
    targets.extend(rng.points(30, [0.5; 3], 0.5));
    let mut r_max: f64 = 0.0;
    for &x in &targets {
        for &y in &sources {
            r_max = r_max.max(distance(x, y));
        }
    }
    Points {
        sources,
        charges,
        targets,
        r_max,
    }
}

/// Laplace's potential and gradient, and the term magnitudes, per target.
struct Reference {
    potential: Vec<f64>,
    gradient: Vec<[f64; 3]>,
    magnitude: Vec<f64>,
    magnitude_gradient: Vec<f64>,
    charge_sum: Vec<f64>,
    charge_magnitude: Vec<f64>,
}

fn reference(p: &Points) -> Reference {
    let n = p.targets.len();
    let mut potential = vec![0.0; n];
    let mut gradient = vec![[0.0; 3]; n];
    laplace::direct_sum(
        &p.sources,
        &p.charges,
        &p.targets,
        &mut potential,
        Some(&mut gradient),
    );
    let mut r = Reference {
        potential,
        gradient,
        magnitude: vec![0.0; n],
        magnitude_gradient: vec![0.0; n],
        charge_sum: vec![0.0; n],
        charge_magnitude: vec![0.0; n],
    };
    for (i, &x) in p.targets.iter().enumerate() {
        for (&y, &q) in p.sources.iter().zip(&p.charges) {
            if x == y {
                continue;
            }
            let d = distance(x, y);
            r.magnitude[i] += q.abs() / d;
            r.magnitude_gradient[i] += q.abs() / (d * d);
            r.charge_sum[i] += q;
            r.charge_magnitude[i] += q.abs();
        }
    }
    r
}

fn helmholtz_sum(k: C, p: &Points) -> (Vec<C>, Vec<[C; 3]>) {
    let n = p.targets.len();
    let mut potential = vec![C::new(0.0, 0.0); n];
    let mut gradient = vec![[C::new(0.0, 0.0); 3]; n];
    helmholtz::direct_sum(
        k,
        &p.sources,
        &complexify(&p.charges),
        &p.targets,
        &mut potential,
        Some(&mut gradient),
    );
    (potential, gradient)
}

const MODULI: [f64; 3] = [1e-2, 1e-4, 1e-6];

/// k of modulus `m` for each kind: real, complex (arg π/4) and imaginary.
fn kinds(m: f64) -> [(&'static str, C); 3] {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    [
        ("real", C::new(m, 0.0)),
        ("complex", C::new(m * h, m * h)),
        ("imaginary", C::new(0.0, m)),
    ]
}

#[test]
fn every_kind_tends_to_laplace() {
    let p = points();
    let r = reference(&p);
    for m in MODULI {
        let x = m * p.r_max;
        for (kind, k) in kinds(m) {
            let (potential, gradient) = helmholtz_sum(k, &p);
            let mut worst = Worst::default();
            let mut worst_gradient = Worst::default();
            for i in 0..p.targets.len() {
                let e = (potential[i] - r.potential[i]).norm() / r.magnitude[i];
                worst.record(e, || format!("target {i}"));
                let e = (0..3)
                    .map(|c| (gradient[i][c] - r.gradient[i][c]).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / r.magnitude_gradient[i];
                worst_gradient.record(e, || format!("target {i}"));
            }
            worst.check(
                x * (1.0 + x) + ROUND_OFF,
                &format!("{kind} |k| = {m:e}: potential"),
            );
            worst_gradient.check(x * x + ROUND_OFF, &format!("{kind} |k| = {m:e}: gradient"));
        }
    }
}

#[test]
fn real_k_has_the_tolerances_of_section_9() {
    let p = points();
    let r = reference(&p);
    for m in MODULI {
        let x = m * p.r_max;
        let (potential, gradient) = helmholtz_sum(C::new(m, 0.0), &p);
        let mut real = Worst::default();
        let mut imaginary = Worst::default();
        let mut imaginary_gradient = Worst::default();
        for i in 0..p.targets.len() {
            let e = (potential[i].re - r.potential[i]).abs() / r.magnitude[i];
            real.record(e, || format!("target {i}"));
            let e = (potential[i].im / m - r.charge_sum[i]).abs() / r.charge_magnitude[i];
            imaginary.record(e, || format!("target {i}"));
            let e = (0..3)
                .map(|c| gradient[i][c].im.powi(2))
                .sum::<f64>()
                .sqrt()
                / r.magnitude_gradient[i];
            imaginary_gradient.record(e, || format!("target {i}"));
        }
        real.check(x * x / 2.0 + ROUND_OFF, &format!("k = {m:e}: Re φ̃"));
        imaginary.check(x * x / 6.0 + ROUND_OFF, &format!("k = {m:e}: Im φ̃ / k"));
        imaginary_gradient.check(x.powi(3) / 3.0 + ROUND_OFF, &format!("k = {m:e}: Im ∇φ̃"));
    }
}

#[test]
fn k_zero_is_laplace() {
    // The formula at k = 0 is Laplace's kernel: the potential bit for bit (q·1/r), the
    // gradient (−q/r³ d against −(q d)/r³) to round-off.
    let p = points();
    let r = reference(&p);
    let (potential, gradient) = helmholtz_sum(C::new(0.0, 0.0), &p);
    let mut worst = Worst::default();
    for i in 0..p.targets.len() {
        assert_eq!(
            potential[i].re.to_bits(),
            r.potential[i].to_bits(),
            "target {i}"
        );
        assert_eq!(potential[i].im, 0.0, "target {i}");
        for (c, (g, l)) in gradient[i].iter().zip(r.gradient[i]).enumerate() {
            assert_eq!(g.im, 0.0, "target {i}");
            let e = (g.re - l).abs() / r.magnitude_gradient[i];
            worst.record(e, || format!("target {i}, component {c}"));
        }
    }
    worst.check(1e-15, "k = 0: gradient");
}
