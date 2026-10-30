//! The stored form of the leaf operators (Helmholtz CH1.3b; CONVENTIONS §3.15, "Radial
//! functions and scaling" and "Storage"): closed forms of degrees 0 and 1, the real
//! multipole of real k and real charges, finiteness under strong damping, and f32.

use nd_fmm_math::RealScalar;
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};

use crate::common::{C, SplitMix64, Worst, complexify, rel_l2, rel_l2_gradient};
use crate::leaf_common::{
    CENTRE, HALF, KINDS, circumradius, frame, inside, level, local_series, multipole_series, shell,
    wavenumber,
};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// The stored coefficients of slots (0, 0) and (1, −1), (1, 0), (1, 1), as complex
/// numbers.
fn low_slots(stored: &[f64]) -> [C; 4] {
    let len = stored.len() / 2;
    core::array::from_fn(|i| C::new(stored[i], stored[len + i]))
}

/// A single complex charge on the z-axis through the centre (θ = 0), where Y₀⁰ = 1/√(4π),
/// Y₁⁰ = √(3/(4π)) and Y₁^{±1} = 0: P2M and P2L against the closed forms
/// j₀ = sin z/z, j₁ = sin z/z² − cos z/z, h₀ = −i e^{iz}/z and h₁ = −e^{iz}(z + i)/z², with
/// the stored factors e^{∓β} s^{∓n} and 4π ik.
///
/// Error measure: per slot, |stored − closed form| / |closed form|, bound 1e-12 (measured
/// at most 1.5e-14 on the M3 Max; the margin is for glibc's sin and cos); the slots m = ±1
/// exactly zero.
#[test]
fn closed_forms_of_degrees_0_and_1() {
    let a = circumradius();
    let y00 = 1.0 / FOUR_PI.sqrt();
    let y10 = (3.0 / FOUR_PI).sqrt();
    let i = C::new(0.0, 1.0);
    let q = C::new(0.7, -0.4);
    let mut worst = Worst::default();
    for kind in KINDS {
        // |k| a = 0.5 has s < 1; Im(k) a reaches 50 for imaginary k.
        for ka in [0.5, 3.0, 50.0] {
            let k = wavenumber(kind, ka / a);
            let level = level(k);
            let (s, beta) = (level.scale(), level.damping());
            let mut ws = Workspace::new(1);
            for (multipole, rho) in [(true, 0.7 * a), (false, 2.5 * a)] {
                let y = [CENTRE[0], CENTRE[1], CENTRE[2] + rho];
                let mut stored = vec![0.0; 8];
                let z = k * rho;
                let expected = if multipole {
                    leaf::p2m(1, &level, &frame(), &[y], &[q], &mut ws, &mut stored);
                    let j0 = z.sin() / z;
                    let j1 = z.sin() / (z * z) - z.cos() / z;
                    let damp = (-beta).exp();
                    [q * damp * j0 * y00, q * damp * j1 / s * y10]
                } else {
                    leaf::p2l(1, &level, &frame(), &[y], &[q], &mut ws, &mut stored);
                    // e^β e^{iz}, with the exponent combined first.
                    let e = (C::new(beta, 0.0) + i * z).exp();
                    let h0 = -i * e / z;
                    let h1 = -(e * (z + i)) / (z * z);
                    let f = i * k * FOUR_PI * q;
                    [f * h0 * y00, f * h1 * s * y10]
                };
                let got = low_slots(&stored);
                let name = if multipole { "P2M" } else { "P2L" };
                for (slot, (g, e)) in [(0, 0), (2, 1)]
                    .map(|(s, e)| (got[s], expected[e]))
                    .iter()
                    .enumerate()
                {
                    let error = (g - e).norm() / e.norm();
                    worst.record(error, || format!("{kind:?} |k|a = {ka} {name} slot {slot}"));
                }
                assert_eq!(got[1], C::new(0.0, 0.0), "{kind:?} {ka} {name} m = −1");
                assert_eq!(got[3], C::new(0.0, 0.0), "{kind:?} {ka} {name} m = 1");
            }
        }
    }
    worst.check(1e-12, "closed forms");
}

/// Real k and real charges: the multipole is real (CONVENTIONS §3.15, "Storage"), and
/// `p2m` leaves its imaginary plane at exactly zero. Error measure: exact equality.
#[test]
fn real_k_and_real_charges_give_a_real_multipole() {
    let p = 20;
    let mut rng = SplitMix64::new(0x3b_5e01);
    let sources = inside(&mut rng, 40, true);
    let charges = complexify(&rng.real_charges(sources.len()));
    for ka in [0.5, 5.0, 30.0] {
        let level = level(C::new(ka / circumradius(), 0.0));
        let mut stored = vec![0.0; 2 * (p + 1) * (p + 1)];
        leaf::p2m(
            p,
            &level,
            &frame(),
            &sources,
            &charges,
            &mut Workspace::new(p),
            &mut stored,
        );
        let (re, im) = stored.split_at(stored.len() / 2);
        assert!(re.iter().any(|&x| x != 0.0));
        assert!(
            im.iter().all(|&x| x == 0.0),
            "|k|a = {ka}: imaginary plane not zero"
        );
    }
}

/// Im(k) a = 1000 (imaginary k, and arg k = π/4): e^β overflows f64, so unscaled
/// coefficients would; the stored ones are finite, every operator applying only
/// non-positive exponents (ρ ≤ a for P2M and L2P, ρ > a for P2L and M2P). Error measure:
/// finiteness; for P2M also |M̂ₙᵐ| ≤ Σ|q| (the harmonics are below √((2n + 1)/(2π)) and the
/// damped |ĵₙ| e^{Im z − β} far below 1 at |z| ≥ 10³).
#[test]
fn strong_damping_stays_finite() {
    let p = 30;
    let a = circumradius();
    assert!((1000.0_f64).exp().is_infinite());
    let mut rng = SplitMix64::new(0x3b_da4b);
    let inner = inside(&mut rng, 30, true);
    let outer = shell(&mut rng, 30, 1.5 * a);
    let charges = rng.complex_charges(inner.len().max(outer.len()));
    for k in [C::new(0.0, 1000.0 / a), C::new(1000.0 / a, 1000.0 / a)] {
        let level = level(k);
        assert_eq!(level.damping(), 1000.0 / a * a);
        let mut ws = Workspace::new(p);
        let len = 2 * (p + 1) * (p + 1);
        let mut multipole = vec![0.0; len];
        leaf::p2m(
            p,
            &level,
            &frame(),
            &inner,
            &charges[..inner.len()],
            &mut ws,
            &mut multipole,
        );
        let total: f64 = charges[..inner.len()].iter().map(|q| q.norm()).sum();
        let largest = multipole.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        println!("k = {k}: largest |M̂| = {largest:.2e}, Σ|q| = {total:.1}");
        assert!(
            largest.is_finite() && largest <= total,
            "k = {k}: |M̂| = {largest}"
        );
        let mut local = vec![0.0; len];
        leaf::p2l(
            p,
            &level,
            &frame(),
            &outer,
            &charges[..outer.len()],
            &mut ws,
            &mut local,
        );
        assert!(local.iter().all(|x| x.is_finite()), "k = {k}: P2L");
        // Unit coefficients evaluated inside and outside.
        let unit = vec![1.0; len];
        for (inside, targets) in [(true, &inner), (false, &outer)] {
            let mut phi = vec![C::new(0.0, 0.0); targets.len()];
            let mut grad = vec![[C::new(0.0, 0.0); 3]; targets.len()];
            let eval = if inside { leaf::l2p } else { leaf::m2p };
            eval(
                p,
                &level,
                &frame(),
                &unit,
                targets,
                &mut ws,
                &mut phi,
                Some(&mut grad),
            );
            assert!(
                phi.iter()
                    .chain(grad.iter().flatten())
                    .all(|c| c.re.is_finite() && c.im.is_finite())
            );
        }
        if k.re == 0.0 {
            let real: Vec<f64> = charges.iter().map(|q| q.re).collect();
            let half = len / 2;
            let (mut m, mut l) = (vec![0.0; half], vec![0.0; half]);
            leaf::p2m_yukawa(
                p,
                &level,
                &frame(),
                &inner,
                &real[..inner.len()],
                &mut ws,
                &mut m,
            );
            leaf::p2l_yukawa(
                p,
                &level,
                &frame(),
                &outer,
                &real[..outer.len()],
                &mut ws,
                &mut l,
            );
            assert!(m.iter().chain(&l).all(|x| x.is_finite()));
            let largest = m.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
            assert!(largest <= total, "Yukawa |M̂ᵞ| = {largest}");
            let unit = vec![1.0; half];
            let mut phi = vec![0.0; inner.len()];
            let mut grad = vec![[0.0; 3]; inner.len()];
            leaf::l2p_yukawa(
                p,
                &level,
                &frame(),
                &unit,
                &inner,
                &mut ws,
                &mut phi,
                Some(&mut grad),
            );
            let mut phi_out = vec![0.0; outer.len()];
            let mut grad_out = vec![[0.0; 3]; outer.len()];
            leaf::m2p_yukawa(
                p,
                &level,
                &frame(),
                &unit,
                &outer,
                &mut ws,
                &mut phi_out,
                Some(&mut grad_out),
            );
            assert!(
                phi.iter()
                    .chain(&phi_out)
                    .chain(grad.iter().flatten())
                    .chain(grad_out.iter().flatten())
                    .all(|x| x.is_finite())
            );
        }
    }
}

/// The series in precision T at p = 8 (s = 1), as complex f64.
fn series_in<T: RealScalar>(
    k: C,
    multipole: bool,
    sources: &[[f64; 3]],
    charges: &[C],
    targets: &[[f64; 3]],
) -> (Vec<C>, Vec<[C; 3]>) {
    let p = 8;
    let to = |x: [f64; 3]| x.map(T::from_f64);
    let kt = nd_fmm_math::Complex::new(T::from_f64(k.re), T::from_f64(k.im));
    let level = Level::new(kt, T::from_f64(HALF));
    let frame = Frame::new(to(CENTRE), T::from_f64(HALF));
    let mut ws = Workspace::new(p);
    let sources: Vec<[T; 3]> = sources.iter().map(|&y| to(y)).collect();
    let targets: Vec<[T; 3]> = targets.iter().map(|&x| to(x)).collect();
    let charges: Vec<_> = charges
        .iter()
        .map(|q| nd_fmm_math::Complex::new(T::from_f64(q.re), T::from_f64(q.im)))
        .collect();
    let mut stored = vec![T::zero(); 2 * (p + 1) * (p + 1)];
    let zero = nd_fmm_math::Complex::new(T::zero(), T::zero());
    let mut phi = vec![zero; targets.len()];
    let mut grad = vec![[zero; 3]; targets.len()];
    if multipole {
        leaf::p2m(p, &level, &frame, &sources, &charges, &mut ws, &mut stored);
        leaf::m2p(
            p,
            &level,
            &frame,
            &stored,
            &targets,
            &mut ws,
            &mut phi,
            Some(&mut grad),
        );
    } else {
        leaf::p2l(p, &level, &frame, &sources, &charges, &mut ws, &mut stored);
        leaf::l2p(
            p,
            &level,
            &frame,
            &stored,
            &targets,
            &mut ws,
            &mut phi,
            Some(&mut grad),
        );
    }
    let back = |c: nd_fmm_math::Complex<T>| C::new(c.re.to_f64(), c.im.to_f64());
    (
        phi.into_iter().map(back).collect(),
        grad.into_iter().map(|g| g.map(back)).collect(),
    )
}

/// f32 at p = 8 and s = 1 (the range CONVENTIONS §3.15 and `radial` check f32 in):
/// P2M → M2P and P2L → L2P in f32 against the same in f64. Error measure: relative ℓ2
/// over the targets; bound 1e-5 (measured below 1e-6 on the M3 Max).
#[test]
fn f32_against_f64() {
    let a = circumradius();
    let mut rng = SplitMix64::new(0x3b_f32);
    let inner = inside(&mut rng, 20, false);
    let outer = shell(&mut rng, 20, 2.5 * a);
    let charges = rng.complex_charges(inner.len());
    let targets_out = shell(&mut rng, 10, 2.5 * a);
    let targets_in = inside(&mut rng, 10, true);
    let mut worst = Worst::default();
    for kind in KINDS {
        let k = wavenumber(kind, 2.0 / a);
        for (multipole, sources, targets) in
            [(true, &inner, &targets_out), (false, &outer, &targets_in)]
        {
            let q = &charges[..sources.len()];
            let (phi32, grad32) = series_in::<f32>(k, multipole, sources, q, targets);
            let (phi64, grad64) = series_in::<f64>(k, multipole, sources, q, targets);
            let name = if multipole { "P2M→M2P" } else { "P2L→L2P" };
            worst.record(rel_l2(&phi32, &phi64), || format!("{kind:?} {name} φ"));
            worst.record(rel_l2_gradient(&grad32, &grad64), || {
                format!("{kind:?} {name} ∇φ")
            });
            // The f64 helper agrees with the test helpers of leaf_common.
            let field = if multipole {
                multipole_series(8, k, sources, q, targets)
            } else {
                local_series(8, k, sources, q, targets)
            };
            assert_eq!(field.potential, phi64);
        }
    }
    worst.check(1e-5, "f32 against f64");
}
