//! Tests 2 to 6, 9 and 10: the definition pointwise, reality for real k and for Yukawa,
//! the symmetries, rectangular truncation, f32 and the panics.

use nd_fmm_math::coaxial::CoaxialTable;
use nd_fmm_math::orthonormal::harmonics;
use nd_fmm_math::radial::{bessel_j, hankel_h};
use nd_fmm_math::{Complex, Layout};
use num_traits::Zero;

use crate::support::{DISTANCES2, degree, i_pow, m2l_t, m2l_weights, scale, sets, weighted};

/// Fₙᵐ(x) = fₙ(k|x|) Yₙᵐ(x̂), unscaled, for n ≤ n_max in the layout of §3.6; f = j
/// (`singular` false) or h.
fn waves(singular: bool, k: Complex<f64>, n_max: usize, x: [f64; 3]) -> Vec<Complex<f64>> {
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let z = k * r;
    let mut f = vec![Complex::zero(); n_max + 1];
    if singular {
        hankel_h(z, 1.0, &mut f, None);
    } else {
        bessel_j(z, 1.0, &mut f, None);
    }
    // ĵ = e^{−Im z} j, ĥ = e^{Im z} h.
    let undo = if singular { (-z.im).exp() } else { z.im.exp() };
    let layout = Layout::new(n_max);
    let mut y = vec![0.0; layout.len()];
    harmonics(n_max, x, &mut y);
    let mut out = vec![Complex::zero(); layout.len()];
    for (n, fn_) in f.iter().enumerate() {
        for m in -(n as isize)..=n as isize {
            let i = layout.idx(n, m);
            out[i] = *fn_ * (undo * y[i]);
        }
    }
    out
}

/// One case of the addition theorems: its name, whether F and G are singular, the table,
/// the factor that undoes its stored normalisation, and |r|/t.
type Case<'a> = (&'a str, bool, bool, &'a CoaxialTable<f64>, f64, f64);

/// Test 2 (with the symmetries of test 5): Fₙᵐ(r ± t ẑ) = Σ_{n′} (±1)^{n+n′} Eᵐ_{n′n}(t)
/// G_{n′}ᵐ(r) for every slot of n ≤ 6, n′ summed to 46, at three points each with
/// |r| = 0.7 t ((R|R)), 0.12 t ((S|R)) and 8 t ((S|S) = (R|R)), for real, complex and
/// imaginary k: 1e-12 relative to the largest term. The tables are unscaled back from the
/// stored normalisation; the entries with n′ < n are read through the index symmetry,
/// and both slots ±m through one coefficient.
#[test]
fn tables_satisfy_the_addition_theorems() {
    let (n_out, n_in, t) = (6, 46, 0.9);
    let directions: [[f64; 3]; 3] = [[0.3, -0.5, 0.8], [-0.7, 0.2, -0.4], [0.1, 0.9, 0.05]];
    let layout_out = Layout::new(n_out);
    let layout_in = Layout::new(n_in);
    let mut worst = (0.0f64, String::new());
    for k in [
        Complex::new(1.3, 0.0),
        Complex::new(1.3, 0.4),
        Complex::new(0.0, 1.3),
    ] {
        let z = k * t;
        let regular = CoaxialTable::regular(z, n_out, n_in);
        let singular = CoaxialTable::singular_regular(z, 1.0, n_out, n_in);
        // name, F singular, G singular, the table, its unscaling, |r|/t
        let cases: [Case; 3] = [
            ("RR", false, false, &regular, z.im.exp(), 0.7),
            ("SR", true, false, &singular, (-z.im).exp(), 0.12),
            ("SS", true, true, &regular, z.im.exp(), 8.0),
        ];
        for (name, f_out, f_in, table, unscale, radius) in cases {
            for sign in [1.0, -1.0] {
                for d in directions {
                    let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let x = d.map(|v| v * radius * t / norm);
                    let lhs = waves(f_out, k, n_out, [x[0], x[1], x[2] + sign * t]);
                    let basis = waves(f_in, k, n_in, x);
                    for n in 0..=n_out {
                        for slot in -(n as isize)..=n as isize {
                            let m = slot.unsigned_abs();
                            let mut sum = Complex::<f64>::zero();
                            let mut largest = 0.0f64;
                            for n1 in m..=n_in {
                                let mut c = table.get(m, n1, n) * unscale;
                                if sign < 0.0 && (n + n1) % 2 == 1 {
                                    c = -c;
                                }
                                let term = c * basis[layout_in.idx(n1, slot)];
                                largest = largest.max(term.norm());
                                sum += term;
                            }
                            let value = lhs[layout_out.idx(n, slot)];
                            let err = (sum - value).norm() / largest.max(value.norm());
                            if err > worst.0 {
                                worst = (
                                    err,
                                    format!("{name} k={k} t={} n={n} slot={slot}", sign * t),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!("addition theorems: worst {:.2e} ({})", worst.0, worst.1);
    assert!(worst.0 <= 1e-12, "{worst:?}");
}

/// Test 3, the recursion: for real k the (R|R) = (S|S) tables of M2M and L2L are real,
/// every imaginary part exactly zero (the recursion closes over the reals), at every k a
/// where the rule uses it; and Re (S|R) = (R|R) s^{n+n′} in the M2L weights at all 15
/// V-list distances, 1e-11.
#[test]
fn real_k_regular_coefficients_are_real() {
    for ka in [0.01, 1.0, 10.0, 30.0, 40.0] {
        let (p, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
        let m2m = CoaxialTable::m2m(Complex::new(ka, 0.0), 0.5, pc, p);
        let l2l = CoaxialTable::l2l(Complex::new(ka, 0.0), 0.5, p, pc);
        for table in [&m2m, &l2l] {
            for m in 0..table.orders() {
                for n1 in m..=table.p_tgt() {
                    for n in m..=table.p_src() {
                        assert_eq!(table.get(m, n1, n).im, 0.0, "ka={ka} ({m}, {n1}, {n})");
                    }
                }
            }
        }
    }
    let mut worst = 0.0f64;
    for ka in [0.01, 1.0, 10.0, 30.0] {
        let p = degree(ka, 1e-10);
        let s = scale(ka);
        let k = Complex::new(ka, 0.0);
        let w = m2l_weights(k, 1.0, s, p);
        for d2 in DISTANCES2 {
            let z = k * m2l_t(d2);
            let rr = CoaxialTable::regular(z, p, p);
            let sr = CoaxialTable::singular_regular(z, s, p, p);
            let err = weighted(
                p,
                p,
                &w,
                &w,
                |m, a, b| sr.get(m, a, b),
                |m, a, b| {
                    let v = sr.get(m, a, b);
                    let power = s.powi((a + b) as i32);
                    v - Complex::new(v.re - rr.get(m, a, b).re * power, 0.0)
                },
            );
            worst = worst.max(err);
        }
    }
    println!("Re (S|R) against (R|R) in the M2L weights: {worst:.2e}");
    assert!(worst <= 1e-11, "{worst:.2e}");
}

/// Checks that i^{n−n′} Êᵐ_{n′n} has imaginary part exactly zero.
fn assert_yukawa_real(table: &CoaxialTable<f64>, what: &str) {
    for m in 0..table.orders() {
        for n1 in m..=table.p_tgt() {
            for n in m..=table.p_src() {
                let v = i_pow(n as i64 - n1 as i64) * table.get(m, n1, n);
                assert_eq!(v.im, 0.0, "{what} ({m}, {n1}, {n}): {v}");
            }
        }
    }
}

/// The Yukawa tables of a level with κ a = `ka` by the rule: M2M, L2L and the M2L at
/// d² = 12.
fn yukawa_tables(ka: f64) -> [CoaxialTable<f64>; 3] {
    let (p, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
    let k = Complex::new(0.0, ka);
    [
        CoaxialTable::m2m(k, 0.5, pc, p),
        CoaxialTable::l2l(k, 0.5, p, pc),
        CoaxialTable::m2l(k, m2l_t(12), 1.0, scale(ka), p, p),
    ]
}

/// Test 4: for Yukawa (k = iκ), i^{n−n′} (E|F) is real for (R|R) = (S|S) and (S|R): by
/// the rule (the recursion on every Yukawa level) exactly in f64 at κ a = 1 and 30, and
/// to 1e-30 relative in the fixture.
#[test]
fn yukawa_is_real_after_the_powers_of_i() {
    for ka in [1.0, 30.0] {
        for (table, what) in yukawa_tables(ka).iter().zip(["M2M", "L2L", "M2L"]) {
            assert_yukawa_real(table, &format!("κa={ka} {what}"));
        }
    }
    let mut reference = 0.0f64;
    for set in sets().iter().filter(|s| s.kind == "imaginary") {
        for &(_, n1, n, v) in &set.entries {
            let w = i_pow(n as i64 - n1 as i64) * v;
            if w.norm() > 0.0 {
                reference = reference.max(w.im.abs() / w.norm());
            }
        }
    }
    println!("Yukawa: the fixture's imaginary parts after i^(n−n′): {reference:.2e}");
    assert!(reference <= 1e-30, "{reference:.2e}");
}

/// Test 4 at κ a = 300 (p = 371).
#[test]
#[ignore = "release mode: tables to p = 371"]
fn yukawa_is_real_after_the_powers_of_i_at_high_degree() {
    for (table, what) in yukawa_tables(300.0).iter().zip(["M2M", "L2L", "M2L"]) {
        assert_yukawa_real(table, &format!("κa=300 {what}"));
    }
}

/// Asserts that `rect` is the block of `square`, bit for bit.
fn assert_block(rect: &CoaxialTable<f64>, square: &CoaxialTable<f64>, what: &str) {
    for m in 0..rect.orders() {
        for n1 in m..=rect.p_tgt() {
            for n in m..=rect.p_src() {
                assert_eq!(
                    rect.get(m, n1, n),
                    square.get(m, n1, n),
                    "{what} ({m}, {n1}, {n})"
                );
            }
        }
    }
}

/// Test 6 (with test 5's orientations): the table p_src → p_tgt, in both orientations,
/// is the block of the square table at max(p_src, p_tgt) bit for bit, for the recursion
/// and for each projection (with the same nodes); and a table reversed equals the table
/// built with the degrees exchanged.
#[test]
fn rectangular_truncation_is_a_block_of_the_square() {
    let (pc, pp) = (degree(5.0, 1e-6), degree(10.0, 1e-6));
    let k = Complex::new(10.0, 1.0);
    let t = 0.5;
    let z = k * t;
    let nodes = pp + 40;
    let rho_outer = 8.0 * t / 3.0f64.sqrt();
    let t_m2l = m2l_t(4);
    type Builder = Box<dyn Fn(usize, usize) -> CoaxialTable<f64>>;
    let builders: [(&str, Builder); 5] = [
        (
            "recursion (R|R)",
            Box::new(move |a, b| CoaxialTable::regular(z, a, b)),
        ),
        (
            "recursion (S|R)",
            Box::new(move |a, b| CoaxialTable::singular_regular(z, 0.7, a, b)),
        ),
        (
            "outer projection",
            Box::new(move |a, b| {
                CoaxialTable::projection_regular_outer(k, t, rho_outer, a, b, nodes)
            }),
        ),
        (
            "inner projection",
            Box::new(move |a, b| CoaxialTable::projection_regular_inner(k, t, t, a, b, nodes)),
        ),
        (
            "(S|R) projection",
            Box::new(move |a, b| {
                CoaxialTable::projection_singular_regular(k, t_m2l, 1.0, 0.7, a, b, nodes)
            }),
        ),
    ];
    for (what, build) in &builders {
        let square = build(pp, pp);
        for (p_src, p_tgt) in [(pc, pp), (pp, pc)] {
            let rect = build(p_src, p_tgt);
            assert_block(&rect, &square, what);
            assert_eq!(rect.clone().reversed(), build(p_tgt, p_src), "{what}");
        }
    }
}

/// Test 9: the recursion in f32 against f64 at p = 8: 1e-5 relative to the largest entry
/// of each target degree.
#[test]
fn f32_recursion_agrees_with_f64() {
    let p = 8;
    let mut worst = 0.0f64;
    for (z, s) in [
        (Complex::new(0.5, 0.0), 0.5),
        (Complex::new(2.0, 0.3), 1.0),
        (Complex::new(0.0, 1.5), 1.0),
    ] {
        let z32 = Complex::new(z.re as f32, z.im as f32);
        let pairs = [
            (
                CoaxialTable::regular(z, p, p),
                CoaxialTable::<f32>::regular(z32, p, p),
            ),
            (
                CoaxialTable::singular_regular(z, s, p, p),
                CoaxialTable::<f32>::singular_regular(z32, s as f32, p, p),
            ),
        ];
        for (wide, narrow) in &pairs {
            for n1 in 0..=p {
                let mut size = 0.0f64;
                let mut err = 0.0f64;
                for m in 0..=n1 {
                    for n in m..=p {
                        let (a, b) = (wide.get(m, n1, n), narrow.get(m, n1, n));
                        size = size.max(a.norm());
                        err = err.max((a - Complex::new(b.re as f64, b.im as f64)).norm());
                    }
                }
                worst = worst.max(err / size);
            }
        }
    }
    println!("f32 against f64 at p = 8: {worst:.2e}");
    assert!(worst <= 1e-5, "{worst:.2e}");
}

#[test]
#[should_panic(expected = "power scale")]
fn panics_on_a_scale_outside_the_unit_interval() {
    CoaxialTable::singular_regular(Complex::new(1.0, 0.0), 1.5, 4, 4);
}

#[test]
#[should_panic(expected = "Im z ≥ 0")]
fn panics_on_a_decaying_argument() {
    CoaxialTable::regular(Complex::new(1.0, -0.1), 4, 4);
}

#[test]
#[should_panic(expected = "non-zero")]
fn panics_on_a_zero_wavenumber() {
    CoaxialTable::m2m(Complex::new(0.0, 0.0), 0.5, 4, 4);
}

#[test]
#[should_panic(expected = "shift t must be positive")]
fn panics_on_a_non_positive_shift() {
    CoaxialTable::m2m(Complex::new(1.0, 0.0), 0.0, 4, 4);
}

#[test]
#[should_panic(expected = "0 < ρ < t")]
fn panics_on_an_m2l_projection_outside_the_shift() {
    CoaxialTable::projection_singular_regular(Complex::new(1.0, 0.0), 1.0, 1.5, 1.0, 4, 4, 10);
}

#[test]
#[should_panic(expected = "ρ > t")]
fn panics_on_an_outer_projection_inside_the_shift() {
    CoaxialTable::projection_regular_outer(Complex::new(1.0, 0.0), 1.0, 0.5, 4, 4, 10);
}

#[test]
#[should_panic(expected = "0 < a < t")]
fn panics_on_an_m2l_between_overlapping_boxes() {
    CoaxialTable::m2l(Complex::new(1.0, 0.0), 1.0, 1.0, 1.0, 4, 4);
}

#[test]
#[should_panic(expected = "overflows")]
fn panics_when_the_start_column_overflows() {
    // s^l hₗ(z) at l = 400 with s = |k| a: far beyond the f64 range.
    CoaxialTable::singular_regular(Complex::new(0.02, 0.0), 0.01, 200, 200);
}
