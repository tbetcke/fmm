//! Tests 2–8: the Wronskians, the relations of CONVENTIONS §3.15 (k₀, jₙ(ix), hₙ(ix), the
//! closed forms), finiteness on every operator's range, the scale, the length, f32 and the
//! panics.

use nd_fmm_math::Complex;
use nd_fmm_math::radial::{bessel_j, hankel_h, modified_i, modified_k};

use crate::support::{ANGLES, Family, evaluate, polar, sweep};

/// Bound on the identities, relative to the size of their terms.
const IDENTITY_BOUND: f64 = 1e-13;

/// Values below this are treated as underflowed in the two-way comparison.
const UNDERFLOW: f64 = 1e-280;

/// Test 2: ĵĥ′ − ĵ′ĥ = i/z², ĵₙĥₙ₊₁/s − sĵₙ₊₁ĥₙ = −i/z² and îk̂′ − î′k̂ = −1/x² (the
/// factors cancel), relative to the terms, over the whole range.
#[test]
fn wronskians() {
    let mut worst = [0.0f64; 3];
    for (r, _, top) in sweep() {
        let s = r.min(1.0);
        for theta in ANGLES {
            let z = polar(r, theta);
            let (j, jd) = evaluate(Family::J, z, s, top + 1);
            let (h, hd) = evaluate(Family::H, z, s, top + 1);
            let target = Complex::i() / (z * z);
            for n in 0..=top {
                let (a, b) = (j[n] * hd[n], jd[n] * h[n]);
                let e = (a - b - target).abs() / (a.abs() + b.abs());
                worst[0] = worst[0].max(e);
                assert!(e <= IDENTITY_BOUND, "Wronskian z={z:?} n={n}: {e:e}");
                let (a, b) = (j[n] * h[n + 1] / s, j[n + 1] * h[n] * s);
                let e = (a - b + target).abs() / (a.abs() + b.abs());
                worst[1] = worst[1].max(e);
                assert!(e <= IDENTITY_BOUND, "cross product z={z:?} n={n}: {e:e}");
            }
        }
        let (i, id) = evaluate(Family::I, Complex::real(r), s, top);
        let (k, kd) = evaluate(Family::K, Complex::real(r), s, top);
        for n in 0..=top {
            let (a, b) = (i[n] * kd[n], id[n] * k[n]);
            let e = (a - b + Complex::real(1.0 / (r * r))).abs() / (a.abs() + b.abs());
            worst[2] = worst[2].max(e);
            assert!(e <= IDENTITY_BOUND, "modified Wronskian x={r} n={n}: {e:e}");
        }
    }
    println!(
        "worst: Wronskian {:.1e}, cross product {:.1e}, modified Wronskian {:.1e}",
        worst[0], worst[1], worst[2]
    );
}

/// Test 3a: the real Yukawa path against the complex one at z = ix (§3.15):
/// îₙ = i⁻ⁿ ĵₙ(ix), î′ₙ = i¹⁻ⁿ ĵ′ₙ(ix), k̂ₙ = −iⁿ ĥₙ(ix), k̂′ₙ = −iⁿ⁺¹ ĥ′ₙ(ix).
#[test]
fn yukawa_two_ways() {
    let mut worst = 0.0f64;
    for (x, top_regular, top_singular) in sweep() {
        let s = x.min(1.0);
        let z = Complex::new(0.0, x);
        let (i, id) = evaluate(Family::I, Complex::real(x), s, top_regular);
        let (k, kd) = evaluate(Family::K, Complex::real(x), s, top_singular);
        let (j, jd) = evaluate(Family::J, z, s, top_regular);
        let (h, hd) = evaluate(Family::H, z, s, top_singular);
        let regular = (0..=top_regular).flat_map(|n| {
            let p = n as i64;
            [
                (n, i[n], Complex::i_pow(-p) * j[n]),
                (n, id[n], Complex::i_pow(1 - p) * jd[n]),
            ]
        });
        let singular = (0..=top_singular).flat_map(|n| {
            let p = n as i64;
            [
                (n, k[n], -(Complex::i_pow(p) * h[n])),
                (n, kd[n], -(Complex::i_pow(p + 1) * hd[n])),
            ]
        });
        for (n, real, complex) in regular.chain(singular) {
            if real.abs() < UNDERFLOW && complex.abs() < UNDERFLOW {
                continue;
            }
            let e = (real - complex).abs() / real.abs();
            worst = worst.max(e);
            assert!(
                e <= IDENTITY_BOUND,
                "x={x} n={n}: {real:?} vs {complex:?} ({e:e})"
            );
        }
    }
    println!("worst: {worst:.1e}");
}

fn close(ours: Complex<f64>, exact: Complex<f64>, bound: f64, what: &str) {
    let e = (ours - exact).abs() / exact.abs();
    assert!(e <= bound, "{what}: {ours:?} vs {exact:?} ({e:e})");
}

/// Test 3b: k̂₀(x) = 1/x exactly, and orders 0 to 2 against their closed forms with the
/// factors of §3.15 (at |z| ≥ 1, away from the closed forms' cancellation at small |z|).
#[test]
fn closed_forms() {
    for x in [1e-8, 0.3, 1.0, 7.5, 200.0, 4000.0] {
        let mut k = [0.0; 3];
        modified_k(x, 1.0, &mut k, None);
        assert_eq!(k[0], 1.0 / x, "k̂₀({x})");
    }
    let s = 0.7;
    for z in [
        Complex::<f64>::new(1.0, 0.0),
        Complex::new(7.5, 0.4),
        Complex::new(2.0, 9.0),
        Complex::new(0.0, 4.0),
        Complex::new(40.0, 0.0),
    ] {
        let (sin, cos) = {
            let e = Complex::cis(z.re) * (-z.im).exp(); // e^{iz}
            let f = Complex::cis(-z.re) * z.im.exp(); // e^{−iz}
            ((e - f) / Complex::new(0.0, 2.0), (e + f) * 0.5)
        };
        let damp = (-z.im).exp();
        let z2 = z * z;
        let j = [
            sin / z,
            sin / z2 - cos / z,
            (Complex::real(3.0) / z2 - Complex::one()) * sin / z - cos * 3.0 / z2,
        ];
        let e = Complex::cis(z.re) * (-z.im).exp();
        let h = [
            -(Complex::i() * e) / z,
            -(e * (z + Complex::i())) / z2,
            Complex::i() * e / z
                * (Complex::one() + Complex::new(0.0, 3.0) / z - Complex::real(3.0) / z2),
        ];
        let (v, _) = evaluate(Family::J, z, s, 2);
        let (w, _) = evaluate(Family::H, z, s, 2);
        for n in 0..3 {
            let sn = s.powi(n as i32);
            // jₙ has real zeros: its error is measured against the envelope
            // max(|jₙ|, |hₙ|), as in the fixture test; the closed form itself cancels
            // near them.
            let envelope = j[n].abs().max(h[n].abs()) * damp / sn;
            let e = (v[n] - j[n] * (damp / sn)).abs() / envelope;
            assert!(e <= 1e-14, "ĵ{n}({z:?}): {e:e}");
            close(
                w[n],
                h[n] * (z.im.exp() * sn),
                1e-14,
                &format!("ĥ{n}({z:?})"),
            );
        }
    }
    for x in [1.0, 1.5, 12.0, 200.0] {
        let (i, _) = evaluate(Family::I, Complex::real(x), s, 1);
        let (k, _) = evaluate(Family::K, Complex::real(x), s, 1);
        let em = (-x).exp();
        close(i[0], Complex::real(x.sinh() / x * em), 1e-14, "î₀");
        close(
            i[1],
            Complex::real((x.cosh() / x - x.sinh() / (x * x)) * em / s),
            1e-14,
            "î₁",
        );
        close(k[1], Complex::real((1.0 + x) / (x * x) * s), 1e-15, "k̂₁");
    }
}

/// Test 4: finite on every operator's range, Yukawa to κρ = 4550 included, at the box's
/// own scale and at scales √3 and 8 times smaller than the argument; z = 0 exact.
#[test]
fn finite_everywhere() {
    for (r, top_regular, top_singular) in sweep() {
        for s in [r.min(1.0), (r / 1.733).min(1.0), (r / 8.0).min(1.0)] {
            for theta in ANGLES {
                let z = polar(r, theta);
                let (j, jd) = evaluate(Family::J, z, s, top_regular);
                let (h, hd) = evaluate(Family::H, z, s, top_singular);
                assert!(
                    j.iter().chain(&jd).all(|v| v.is_finite()),
                    "j z={z:?} s={s}"
                );
                assert!(
                    h.iter().chain(&hd).all(|v| v.is_finite()),
                    "h z={z:?} s={s}"
                );
            }
            let (i, id) = evaluate(Family::I, Complex::real(r), s, top_regular);
            let (k, kd) = evaluate(Family::K, Complex::real(r), s, top_singular);
            assert!(i.iter().chain(&id).all(|v| v.is_finite()), "i x={r} s={s}");
            assert!(k.iter().chain(&kd).all(|v| v.is_finite()), "k x={r} s={s}");
        }
    }
    for s in [1e-3, 1.0] {
        for family in [Family::J, Family::I] {
            let (v, d) = evaluate(family, Complex::zero(), s, 5);
            assert_eq!(v[0], Complex::one());
            assert!(v[1..].iter().all(|x| *x == Complex::zero()));
            assert_eq!(d[0], Complex::zero());
            assert!((d[1].re - 1.0 / (3.0 * s)).abs() <= 1e-15 / s && d[1].im == 0.0);
            assert!(d[2..].iter().all(|x| *x == Complex::zero()));
        }
    }
}

/// Test 5: the same argument at two scales differs by (s′/s)ⁿ.
#[test]
fn scale_consistency() {
    let (s1, s2) = (1.0, 0.3);
    for z in [
        Complex::new(2.0, 0.5),
        Complex::new(0.2, 0.0),
        Complex::new(0.0, 1.5),
    ] {
        for family in [Family::J, Family::H, Family::I, Family::K] {
            let z = if matches!(family, Family::I | Family::K) {
                Complex::real(z.abs())
            } else {
                z
            };
            let (a, ad) = evaluate(family, z, s1, 40);
            let (b, bd) = evaluate(family, z, s2, 40);
            let regular = matches!(family, Family::J | Family::I);
            for n in 0..=40 {
                let f = if regular {
                    (s1 / s2).powi(n as i32)
                } else {
                    (s2 / s1).powi(n as i32)
                };
                for (x, y) in [(a[n], b[n]), (ad[n], bd[n])] {
                    let e = (x * f - y).abs() / y.abs();
                    assert!(e <= IDENTITY_BOUND, "{family:?} z={z:?} n={n}: {e:e}");
                }
            }
        }
    }
}

/// Test 6: a shorter request gives the same leading orders: bit for bit for ĥ and k̂
/// (the upward recursion), within 1e-15 for ĵ and î (Miller's start order grows with the
/// length).
#[test]
fn prefixes_of_longer_requests() {
    for z in [
        Complex::<f64>::new(0.01, 0.0),
        Complex::new(3.0, 0.2),
        Complex::new(50.0, 5.0),
        Complex::new(0.0, 20.0),
    ] {
        let s = z.abs().min(1.0);
        for family in [Family::J, Family::H, Family::I, Family::K] {
            let z = if matches!(family, Family::I | Family::K) {
                Complex::real(z.abs())
            } else {
                z
            };
            let (long, long_d) = evaluate(family, z, s, 120);
            for top in [0, 1, 7, 60] {
                let (short, short_d) = evaluate(family, z, s, top);
                for n in 0..=top {
                    for (a, b) in [(short[n], long[n]), (short_d[n], long_d[n])] {
                        if matches!(family, Family::H | Family::K) {
                            assert_eq!(a, b, "{family:?} z={z:?} top={top} n={n}");
                        } else {
                            let e = (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
                            assert!(e <= 1e-15, "{family:?} z={z:?} top={top} n={n}: {e:e}");
                        }
                    }
                }
            }
        }
    }
}

/// Test 7: f32 against f64 at s = 1, |z| from 0.1 to 20, n ≤ 8.
#[test]
fn f32_against_f64() {
    const TOP: usize = 8;
    let mut worst = 0.0f64;
    for r in [0.1, 0.7, 2.0, 9.0, 20.0] {
        for theta in [0.0, 0.3, std::f64::consts::FRAC_PI_2] {
            let z = polar(r, theta);
            let z32 = Complex::<f32>::from_f64(z);
            let z = z32.to_f64();
            for family in [Family::J, Family::H] {
                let (v, d) = evaluate(family, z, 1.0, TOP);
                let mut v32 = [Complex::<f32>::zero(); TOP + 1];
                let mut d32 = [Complex::<f32>::zero(); TOP + 1];
                if family == Family::J {
                    bessel_j(z32, 1.0, &mut v32, Some(&mut d32));
                } else {
                    hankel_h(z32, 1.0, &mut v32, Some(&mut d32));
                }
                for n in 0..=TOP {
                    for (a, b) in [(v32[n], v[n]), (d32[n], d[n])] {
                        // ĵ has real zeros; measure against the order's envelope.
                        let scale = if family == Family::J {
                            v[n].abs().max(d[n].abs()).max(1e-30)
                        } else {
                            b.abs()
                        };
                        let e = (a.to_f64() - b).abs() / scale;
                        worst = worst.max(e);
                        assert!(e <= 1e-5, "{family:?} f32 z={z:?} n={n}: {e:e}");
                    }
                }
            }
            let x32 = r as f32;
            let x = f64::from(x32);
            let (iv, id) = evaluate(Family::I, Complex::real(x), 1.0, TOP);
            let (kv, kd) = evaluate(Family::K, Complex::real(x), 1.0, TOP);
            let (mut i32v, mut i32d) = ([0.0f32; TOP + 1], [0.0f32; TOP + 1]);
            let (mut k32v, mut k32d) = ([0.0f32; TOP + 1], [0.0f32; TOP + 1]);
            modified_i(x32, 1.0, &mut i32v, Some(&mut i32d));
            modified_k(x32, 1.0, &mut k32v, Some(&mut k32d));
            for n in 0..=TOP {
                for (a, b) in [
                    (i32v[n], iv[n]),
                    (i32d[n], id[n]),
                    (k32v[n], kv[n]),
                    (k32d[n], kd[n]),
                ] {
                    let e = (f64::from(a) - b.re).abs() / b.re.abs();
                    worst = worst.max(e);
                    assert!(e <= 1e-5, "f32 x={x} n={n}: {e:e}");
                }
            }
        }
    }
    println!("worst f32: {worst:.1e}");
}

/// Test 8: the panics (arguments that are not finite would loop in Miller's start test).
mod panics {
    use super::*;

    #[test]
    #[should_panic(expected = "Im z ≥ 0")]
    fn j_below_the_real_axis() {
        bessel_j(
            Complex::new(1.0, -0.1),
            1.0,
            &mut [Complex::zero(); 3],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "Im z ≥ 0")]
    fn h_below_the_real_axis() {
        hankel_h(
            Complex::new(1.0, -0.1),
            1.0,
            &mut [Complex::zero(); 3],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "z ≠ 0")]
    fn h_at_zero() {
        hankel_h(Complex::zero(), 1.0, &mut [Complex::<f64>::zero(); 3], None);
    }

    #[test]
    #[should_panic(expected = "x > 0")]
    fn k_at_zero() {
        modified_k(0.0, 1.0, &mut [0.0; 3], None);
    }

    #[test]
    #[should_panic(expected = "x ≥ 0")]
    fn i_negative() {
        modified_i(-1.0, 1.0, &mut [0.0; 3], None);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn infinite_argument() {
        modified_i(f64::INFINITY, 1.0, &mut [0.0; 3], None);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn not_a_number() {
        bessel_j(
            Complex::new(f64::NAN, 0.0),
            1.0,
            &mut [Complex::zero(); 3],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "(0, 1]")]
    fn scale_above_one() {
        modified_i(1.0, 1.5, &mut [0.0; 3], None);
    }

    #[test]
    #[should_panic(expected = "(0, 1]")]
    fn scale_zero() {
        bessel_j(Complex::new(1.0, 0.0), 0.0, &mut [Complex::zero(); 3], None);
    }

    #[test]
    #[should_panic(expected = "length of values")]
    fn derivative_length() {
        modified_k(1.0, 1.0, &mut [0.0; 3], Some(&mut [0.0; 2]));
    }
}
