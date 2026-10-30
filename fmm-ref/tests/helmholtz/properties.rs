//! Properties of `p2p` and `direct_sum`: agreement of the two, f32 against f64,
//! accumulation, coincident pairs, reciprocity, the gradient against differences of the
//! potential, exact scaling in the charges and empty inputs.
//!
//! Error measure: per target, relative to the term magnitudes (`common::magnitudes`),
//! unless the test checks bits.

use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p;

use crate::common::{C, SplitMix64, Worst, gradient_error, magnitudes};

/// One k of each kind, moderate and large.
const KS: [(f64, f64); 6] = [
    (1.0, 0.0),
    (25.0, 0.0),
    (1.0, 0.5),
    (25.0, 2.5),
    (0.0, 1.0),
    (0.0, 25.0),
];

struct Set {
    sources: Vec<[f64; 3]>,
    charges: Vec<C>,
    targets: Vec<[f64; 3]>,
}

fn set(seed: u64, n: usize) -> Set {
    let mut rng = SplitMix64::new(seed);
    let sources = rng.points(n, [0.5; 3], 0.5);
    let charges = rng.complex_charges(n);
    let mut targets = sources.clone();
    targets.extend(rng.points(n / 4, [0.5; 3], 1.0));
    Set {
        sources,
        charges,
        targets,
    }
}

fn zeros(n: usize) -> (Vec<C>, Vec<[C; 3]>) {
    (vec![C::new(0.0, 0.0); n], vec![[C::new(0.0, 0.0); 3]; n])
}

#[test]
fn p2p_agrees_with_direct_sum() {
    let s = set(0x3a_1001, 200);
    let mut worst = Worst::default();
    for (re, im) in KS {
        let k = C::new(re, im);
        let n = s.targets.len();
        let (mut a, mut ag) = zeros(n);
        let (mut b, mut bg) = zeros(n);
        p2p::p2p(k, &s.sources, &s.charges, &s.targets, &mut a, Some(&mut ag));
        p2p::direct_sum(k, &s.sources, &s.charges, &s.targets, &mut b, Some(&mut bg));
        for i in 0..n {
            let (mp, mg) = magnitudes(k, &s.sources, &s.charges, s.targets[i]);
            worst.record((a[i] - b[i]).norm() / mp, || {
                format!("k = {k}, potential {i}")
            });
            worst.record(gradient_error(ag[i], bg[i]) / mg, || {
                format!("k = {k}, gradient {i}")
            });
        }
    }
    worst.check(1e-14, "p2p against direct_sum");
}

#[test]
fn f32_agrees_with_f64() {
    // |k| r ≤ 10 on these points; the f32 inputs are rounded first and the f64 sum takes
    // the rounded values.
    let s = set(0x3a_1002, 100);
    let r32 = |x: &[[f64; 3]]| -> Vec<[f32; 3]> { x.iter().map(|p| p.map(|c| c as f32)).collect() };
    let sources = r32(&s.sources);
    let targets = r32(&s.targets);
    let charges: Vec<Complex<f32>> = s
        .charges
        .iter()
        .map(|q| Complex::new(q.re as f32, q.im as f32))
        .collect();
    let back = |x: &[[f32; 3]]| -> Vec<[f64; 3]> { x.iter().map(|p| p.map(f64::from)).collect() };
    let (sources64, targets64) = (back(&sources), back(&targets));
    let charges64: Vec<C> = charges
        .iter()
        .map(|q| C::new(f64::from(q.re), f64::from(q.im)))
        .collect();
    let mut worst = Worst::default();
    for (re, im) in [(1.0, 0.0), (4.0, 0.0), (3.0, 1.5), (0.0, 4.0)] {
        let n = targets.len();
        let zero = Complex::new(0.0_f32, 0.0);
        let mut a = vec![zero; n];
        let mut ag = vec![[zero; 3]; n];
        p2p::p2p(
            Complex::new(re as f32, im as f32),
            &sources,
            &charges,
            &targets,
            &mut a,
            Some(&mut ag),
        );
        let k = C::new(re, im);
        let (mut b, mut bg) = zeros(n);
        p2p::direct_sum(k, &sources64, &charges64, &targets64, &mut b, Some(&mut bg));
        for i in 0..n {
            let (mp, mg) = magnitudes(k, &sources64, &charges64, targets64[i]);
            let a64 = C::new(f64::from(a[i].re), f64::from(a[i].im));
            worst.record((a64 - b[i]).norm() / mp, || {
                format!("k = {k}, potential {i}")
            });
            let ag64 = ag[i].map(|g| C::new(f64::from(g.re), f64::from(g.im)));
            worst.record(gradient_error(ag64, bg[i]) / mg, || {
                format!("k = {k}, gradient {i}")
            });
        }
    }
    worst.check(1e-5, "f32 against f64");
}

#[test]
fn outputs_accumulate() {
    let s = set(0x3a_1003, 50);
    let k = C::new(3.0, 0.5);
    let n = s.targets.len();
    let mut rng = SplitMix64::new(7);
    let start: Vec<C> = rng.complex_charges(n);
    let start_g: Vec<[C; 3]> = (0..n)
        .map(|_| {
            let g = rng.complex_charges(3);
            [g[0], g[1], g[2]]
        })
        .collect();
    // direct_sum adds its compensated sum in one final addition.
    let (mut fresh, mut fresh_g) = zeros(n);
    p2p::direct_sum(
        k,
        &s.sources,
        &s.charges,
        &s.targets,
        &mut fresh,
        Some(&mut fresh_g),
    );
    let (mut acc, mut acc_g) = (start.clone(), start_g.clone());
    p2p::direct_sum(
        k,
        &s.sources,
        &s.charges,
        &s.targets,
        &mut acc,
        Some(&mut acc_g),
    );
    for i in 0..n {
        assert_eq!(acc[i], start[i] + fresh[i]);
        for c in 0..3 {
            assert_eq!(acc_g[i][c], start_g[i][c] + fresh_g[i][c]);
        }
    }
    // p2p adds each term into the output: the same to round-off.
    let (mut acc, mut acc_g) = (start.clone(), start_g.clone());
    p2p::p2p(
        k,
        &s.sources,
        &s.charges,
        &s.targets,
        &mut acc,
        Some(&mut acc_g),
    );
    let mut worst = Worst::default();
    for i in 0..n {
        let (mp, mg) = magnitudes(k, &s.sources, &s.charges, s.targets[i]);
        let scale = |m: f64, z: C| m + z.norm();
        worst.record(
            (acc[i] - start[i] - fresh[i]).norm() / scale(mp, start[i]),
            || format!("potential {i}"),
        );
        for c in 0..3 {
            let e = (acc_g[i][c] - start_g[i][c] - fresh_g[i][c]).norm();
            worst.record(e / scale(mg, start_g[i][c]), || format!("gradient {i}"));
        }
    }
    worst.check(1e-14, "p2p accumulates");
}

#[test]
fn coincident_pairs_are_skipped() {
    // Two copies of a source at the target and one other source: only the other counts.
    let a = [0.25, 0.5, 0.75];
    let b = [1.0, -0.5, 0.125];
    let q = [C::new(1.0, 2.0), C::new(-3.0, 0.5), C::new(0.75, -0.25)];
    for (re, im) in KS {
        let k = C::new(re, im);
        let (mut all, mut all_g) = zeros(1);
        p2p::p2p(k, &[a, a, b], &q, &[a], &mut all, Some(&mut all_g));
        let (mut one, mut one_g) = zeros(1);
        p2p::p2p(k, &[b], &q[2..], &[a], &mut one, Some(&mut one_g));
        assert_eq!(all, one, "k = {k}");
        assert_eq!(all_g, one_g, "k = {k}");
        let (mut d, mut dg) = zeros(1);
        p2p::direct_sum(k, &[a, a, b], &q, &[a], &mut d, Some(&mut dg));
        assert_eq!(d, one, "k = {k}");
        assert_eq!(dg, one_g, "k = {k}");
    }
    // Targets equal to the sources: finite everywhere.
    let s = set(0x3a_1004, 40);
    let (mut phi, mut g) = zeros(40);
    p2p::p2p(
        C::new(2.0, 1.0),
        &s.sources,
        &s.charges,
        &s.sources,
        &mut phi,
        Some(&mut g),
    );
    assert!(phi.iter().all(|z| z.is_finite()));
    assert!(g.iter().flatten().all(|z| z.is_finite()));
}

#[test]
fn reciprocity() {
    // The potential of a unit charge at y, at x, is that of a unit charge at x, at y.
    let mut rng = SplitMix64::new(0x3a_1005);
    let one = [C::new(1.0, 0.0)];
    for (re, im) in KS {
        let k = C::new(re, im);
        for _ in 0..20 {
            let x = rng.points(1, [0.0; 3], 2.0)[0];
            let y = rng.points(1, [0.0; 3], 2.0)[0];
            let (mut a, _) = zeros(1);
            let (mut b, _) = zeros(1);
            p2p::p2p(k, &[y], &one, &[x], &mut a, None);
            p2p::p2p(k, &[x], &one, &[y], &mut b, None);
            assert_eq!(a, b, "k = {k}");
        }
    }
}

#[test]
fn gradient_matches_differences() {
    // Central differences of the potential in each coordinate, step h = 1e-5: error
    // O(h²) and O(ε/h), at most 1.0e-8 relative here (at k = 25i).
    let s = set(0x3a_1006, 30);
    let targets = &s.targets[30..];
    let h = 1e-5;
    let mut worst = Worst::default();
    for (re, im) in KS {
        let k = C::new(re, im);
        let n = targets.len();
        let (_, mut g) = zeros(n);
        let mut phi = vec![C::new(0.0, 0.0); n];
        p2p::direct_sum(k, &s.sources, &s.charges, targets, &mut phi, Some(&mut g));
        for (i, &x) in targets.iter().enumerate() {
            let (_, mg) = magnitudes(k, &s.sources, &s.charges, x);
            let mut difference = [C::new(0.0, 0.0); 3];
            for (c, dc) in difference.iter_mut().enumerate() {
                let (mut plus, mut minus) = (x, x);
                plus[c] += h;
                minus[c] -= h;
                let (mut fp, _) = zeros(1);
                let (mut fm, _) = zeros(1);
                p2p::direct_sum(k, &s.sources, &s.charges, &[plus], &mut fp, None);
                p2p::direct_sum(k, &s.sources, &s.charges, &[minus], &mut fm, None);
                *dc = (fp[0] - fm[0]) / (plus[c] - minus[c]);
            }
            worst.record(gradient_error(g[i], difference) / mg, || {
                format!("k = {k}, target {i}")
            });
        }
    }
    worst.check(1e-7, "gradient against central differences");
}

#[test]
fn charges_scale_exactly() {
    // Doubling the charges doubles every term exactly, so both sums double bit for bit.
    let s = set(0x3a_1007, 60);
    let doubled: Vec<C> = s.charges.iter().map(|&q| q * 2.0).collect();
    for (re, im) in KS {
        let k = C::new(re, im);
        let n = s.targets.len();
        for oracle in [false, true] {
            let run = |q: &[C]| {
                let (mut phi, mut g) = zeros(n);
                if oracle {
                    p2p::direct_sum(k, &s.sources, q, &s.targets, &mut phi, Some(&mut g));
                } else {
                    p2p::p2p(k, &s.sources, q, &s.targets, &mut phi, Some(&mut g));
                }
                (phi, g)
            };
            let (a, ag) = run(&s.charges);
            let (b, bg) = run(&doubled);
            for i in 0..n {
                assert_eq!(b[i], a[i] * 2.0, "k = {k}, oracle {oracle}, target {i}");
                assert_eq!(bg[i], ag[i].map(|g| g * 2.0), "k = {k}, target {i}");
            }
        }
    }
}

#[test]
fn empty_inputs() {
    let k = C::new(1.0, 0.25);
    let start = vec![C::new(0.5, -0.5); 3];
    let mut phi = start.clone();
    p2p::p2p(k, &[], &[], &[[0.0; 3]; 3], &mut phi, None);
    assert_eq!(phi, start);
    p2p::direct_sum(k, &[], &[], &[[0.0; 3]; 3], &mut phi, None);
    assert_eq!(phi, start);
    p2p::p2p(k, &[[1.0; 3]], &[C::new(1.0, 0.0)], &[], &mut [], None);
    let mut real = vec![0.25; 2];
    p2p::p2p_yukawa(2.0, &[], &[], &[[0.0; 3]; 2], &mut real, None);
    p2p::direct_sum_yukawa(2.0, &[], &[], &[[0.0; 3]; 2], &mut real, None);
    assert_eq!(real, [0.25; 2]);
}
