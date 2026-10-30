//! Yukawa two ways (proposal §9 item 4, CONVENTIONS §3.15): the real kernel e^{−κr}/r of
//! `p2p_yukawa` and `direct_sum_yukawa` against the complex kernel at k = iκ.
//!
//! Error measure: bit identity. At k = iκ the complex term performs the real term's
//! operations, with exact zeros in the imaginary parts, so the two agree bit for bit;
//! complex charges are the two real charge vectors of §3.15 ("Storage").

use nd_fmm_math::{Complex, RealScalar};
use nd_fmm_ref::helmholtz::p2p;

use crate::common::{C, SplitMix64, complexify};

const KAPPAS: [f64; 3] = [0.5, 25.0, 300.0];

struct Set<T> {
    sources: Vec<[T; 3]>,
    charges: Vec<Complex<T>>,
    targets: Vec<[T; 3]>,
}

fn set<T: RealScalar>(seed: u64) -> Set<T> {
    let mut rng = SplitMix64::new(seed);
    let sources = rng.points(60, [0.5; 3], 0.5);
    let charges = rng.complex_charges(60);
    let mut targets = sources.clone();
    targets.extend(rng.points(20, [0.5; 3], 0.75));
    let point = |x: [f64; 3]| x.map(T::from_f64);
    Set {
        sources: sources.into_iter().map(point).collect(),
        charges: charges
            .into_iter()
            .map(|q| Complex::new(T::from_f64(q.re), T::from_f64(q.im)))
            .collect(),
        targets: targets.into_iter().map(point).collect(),
    }
}

type Field<T> = (Vec<T>, Vec<[T; 3]>);

/// The real path on one real charge vector.
fn real<T: RealScalar>(
    kappa: T,
    s: &Set<T>,
    part: impl Fn(Complex<T>) -> T,
    oracle: bool,
) -> Field<T> {
    let q: Vec<T> = s.charges.iter().map(|&q| part(q)).collect();
    let mut potential = vec![T::zero(); s.targets.len()];
    let mut gradient = vec![[T::zero(); 3]; s.targets.len()];
    if oracle {
        let to = |v: &[[T; 3]]| v.iter().map(|x| x.map(|c| c.to_f64())).collect::<Vec<_>>();
        let mut phi = vec![0.0; s.targets.len()];
        let mut g = vec![[0.0; 3]; s.targets.len()];
        p2p::direct_sum_yukawa(
            kappa.to_f64(),
            &to(&s.sources),
            &q.iter().map(|&q| q.to_f64()).collect::<Vec<_>>(),
            &to(&s.targets),
            &mut phi,
            Some(&mut g),
        );
        potential = phi.into_iter().map(T::from_f64).collect();
        gradient = g.into_iter().map(|g| g.map(T::from_f64)).collect();
    } else {
        p2p::p2p_yukawa(
            kappa,
            &s.sources,
            &q,
            &s.targets,
            &mut potential,
            Some(&mut gradient),
        );
    }
    (potential, gradient)
}

fn bits<T: RealScalar>(v: T) -> u64 {
    v.to_f64().to_bits()
}

/// The complex path at k = iκ against the real path on each part of the charges.
fn two_ways<T: RealScalar>(seed: u64) {
    let s = set::<T>(seed);
    for kappa in KAPPAS.map(T::from_f64) {
        let n = s.targets.len();
        let zero = Complex::new(T::zero(), T::zero());
        let mut potential = vec![zero; n];
        let mut gradient = vec![[zero; 3]; n];
        p2p::p2p(
            Complex::new(T::zero(), kappa),
            &s.sources,
            &s.charges,
            &s.targets,
            &mut potential,
            Some(&mut gradient),
        );
        let (re, re_g) = real(kappa, &s, |q| q.re, false);
        let (im, im_g) = real(kappa, &s, |q| q.im, false);
        for i in 0..n {
            let at = || format!("κ = {}, target {i}", kappa.to_f64());
            assert_eq!(bits(potential[i].re), bits(re[i]), "{}", at());
            assert_eq!(bits(potential[i].im), bits(im[i]), "{}", at());
            for c in 0..3 {
                assert_eq!(bits(gradient[i][c].re), bits(re_g[i][c]), "{}", at());
                assert_eq!(bits(gradient[i][c].im), bits(im_g[i][c]), "{}", at());
            }
        }
    }
}

#[test]
fn p2p_two_ways_f64() {
    two_ways::<f64>(0x3a_0001);
}

#[test]
fn p2p_two_ways_f32() {
    two_ways::<f32>(0x3a_0002);
}

#[test]
fn direct_sum_two_ways() {
    let s = set::<f64>(0x3a_0003);
    for kappa in KAPPAS {
        let n = s.targets.len();
        let mut potential = vec![C::new(0.0, 0.0); n];
        let mut gradient = vec![[C::new(0.0, 0.0); 3]; n];
        p2p::direct_sum(
            C::new(0.0, kappa),
            &s.sources,
            &s.charges,
            &s.targets,
            &mut potential,
            Some(&mut gradient),
        );
        let (re, re_g) = real(kappa, &s, |q| q.re, true);
        let (im, im_g) = real(kappa, &s, |q| q.im, true);
        for i in 0..n {
            assert_eq!(
                potential[i].re.to_bits(),
                re[i].to_bits(),
                "κ = {kappa}, {i}"
            );
            assert_eq!(
                potential[i].im.to_bits(),
                im[i].to_bits(),
                "κ = {kappa}, {i}"
            );
            for c in 0..3 {
                assert_eq!(gradient[i][c].re.to_bits(), re_g[i][c].to_bits());
                assert_eq!(gradient[i][c].im.to_bits(), im_g[i][c].to_bits());
            }
        }
    }
}

#[test]
fn real_charges_give_real_values() {
    // Real charges at k = iκ: the imaginary parts are exactly zero.
    let s = set::<f64>(0x3a_0004);
    let q: Vec<f64> = s.charges.iter().map(|q| q.re).collect();
    let n = s.targets.len();
    let mut potential = vec![C::new(0.0, 0.0); n];
    let mut gradient = vec![[C::new(0.0, 0.0); 3]; n];
    p2p::p2p(
        C::new(0.0, 2.0),
        &s.sources,
        &complexify(&q),
        &s.targets,
        &mut potential,
        Some(&mut gradient),
    );
    for i in 0..n {
        assert_eq!(potential[i].im, 0.0);
        assert!(gradient[i].iter().all(|g| g.im == 0.0));
    }
}
