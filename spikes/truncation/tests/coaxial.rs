//! Test 5 of docs/helmholtz/tasks/CH0.7.md: the double-double coaxial tables against S2's
//! arbitrary-precision fixture, recursion against projection, and rectangular truncation.
//!
//! The metric is S2's FMM-weighted error (`spikes/coaxial/src/metric.rs`): each entry
//! weighted by the bounds of the coefficients it multiplies and produces,
//! max |ΔE| w / max |E| w.

use nd_fmm_spike_truncation::coaxial::{self, Op, Table};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::radial;
use nd_fmm_spike_truncation::real::{Dd, Real};

type C = Complex<Dd>;

/// Jₙ(R) = max_{0 < r ≤ R} |jₙ(k r)|, n ≤ n_max, sampled at 16 points per period.
fn envelope(k: Complex<f64>, radius: f64, n_max: usize) -> Vec<f64> {
    let samples = ((16.0 * k.abs() * radius / std::f64::consts::PI).ceil() as usize).max(16);
    let mut out = vec![0.0f64; n_max + 1];
    let mut j = vec![Complex::<f64>::ZERO; n_max + 1];
    for i in 1..=samples {
        let r = radius * i as f64 / samples as f64;
        radial::bessel_j(k * r, 1.0, &mut j);
        for (o, v) in out.iter_mut().zip(&j) {
            *o = o.max(v.abs());
        }
    }
    out
}

/// S2's weights for a = 1: M2L uₙ = Jₙ(1) √(2n + 1)/gⁿ on both sides; M2M input
/// Jₙ(1/2) √(2n + 1), output |hₙ′(k 4/√3)| √(2n′ + 1).
fn weights(op: Op, k: Complex<f64>, g: f64, p_src: usize, p_tgt: usize) -> (Vec<f64>, Vec<f64>) {
    match op {
        Op::Singular => {
            let mut s = 1.0;
            let u: Vec<f64> = envelope(k, 1.0, p_src.max(p_tgt))
                .iter()
                .enumerate()
                .map(|(n, v)| {
                    let x = v * ((2 * n + 1) as f64).sqrt() / s;
                    s *= g;
                    x
                })
                .collect();
            (u.clone(), u)
        }
        Op::Regular => {
            let mut h = vec![Complex::<f64>::ZERO; p_tgt + 1];
            radial::hankel(k * (4.0 / 3f64.sqrt()), 1.0, &mut h);
            let input = envelope(k, 0.5, p_src)
                .iter()
                .enumerate()
                .map(|(n, v)| v * ((2 * n + 1) as f64).sqrt())
                .collect();
            let output = h
                .iter()
                .enumerate()
                .map(|(n, v)| v.abs() * ((2 * n + 1) as f64).sqrt())
                .collect();
            (input, output)
        }
    }
}

struct Set {
    name: String,
    op: Op,
    z: C,
    g: f64,
    p_src: usize,
    p_tgt: usize,
    entries: Vec<(usize, usize, usize, Complex<f64>)>,
}

impl Set {
    fn ka(&self) -> f64 {
        let part = self.name.split('-').find(|s| s.starts_with("ka")).unwrap();
        part[2..].parse().unwrap()
    }

    /// k and t for a = 1 (S2's geometry): M2M t = 1/2, M2L t = 2 √d2/√3.
    fn geometry(&self) -> (C, Dd) {
        let t = if self.name.ends_with("m2m") {
            Dd::from_f64(0.5)
        } else {
            let d2 = if self.name.ends_with("d2") { 4.0 } else { 27.0 };
            Dd::from_f64(2.0) * Dd::from_f64(d2).sqrt() / Dd::from_f64(3.0).sqrt()
        };
        let ka = self.ka();
        let (re, im) = match self.name.split('-').next().unwrap() {
            "real" => (1.0, 0.0),
            "complex0.1" => (1.0, 0.1),
            "complex1" => (1.0, 1.0),
            "imaginary" => (0.0, 1.0),
            other => panic!("{other}"),
        };
        let k = C::new(Dd::from_f64(re * ka), Dd::from_f64(ka) * Dd::from_f64(im));
        let z = k * t;
        assert!(
            (z - self.z).abs().to_f64() <= 1e-13 * self.z.abs().to_f64(),
            "{}",
            self.name
        );
        // The reference was computed at the f64 z; the table moves by about l δz/z with
        // it, so k is taken as z/t exactly.
        (self.z / C::real(t), t)
    }

    fn table(&self, recursion: bool) -> Table<Dd> {
        let p = self.p_src.max(self.p_tgt);
        let g = Dd::from_f64(self.g);
        let (k, t) = self.geometry();
        if recursion {
            let gg = if self.op == Op::Singular { g } else { Dd::ONE };
            coaxial::recursion(&coaxial::start_values(self.op, k * t, gg, 2 * p), gg, p)
        } else {
            let rho = if self.op == Op::Singular {
                Dd::ONE
            } else {
                Dd::from_f64(4.0) / Dd::from_f64(3.0).sqrt()
            };
            coaxial::projection(self.op, k, t, rho, g, p, p + 100)
        }
    }

    /// The weighted error of `table` over the sampled entries, with S2's exponential
    /// normalisation applied to the table.
    fn error(&self, table: &Table<Dd>) -> f64 {
        let factor = match self.op {
            Op::Regular => (-self.z.im.abs()).exp(),
            Op::Singular => self.z.im.exp(),
        };
        let (k, _) = self.geometry();
        let (input, output) = weights(self.op, k.to_f64(), self.g, self.p_src, self.p_tgt);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for &(m, n1, n, want) in &self.entries {
            let got = (table.get(m, n1, n) * factor).to_f64();
            let w = output[n1] * input[n];
            num = num.max((got - want).abs() * w);
            den = den.max(want.abs() * w);
        }
        num / den
    }
}

fn pair(v: &serde_json::Value) -> Complex<f64> {
    let p = v.as_array().unwrap();
    Complex::new(
        p[0].as_str().unwrap().parse().unwrap(),
        p[1].as_str().unwrap().parse().unwrap(),
    )
}

fn sets() -> Vec<Set> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../coaxial/fixtures/reference.json"
    ))
    .unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let us = |v: &serde_json::Value| -> Vec<usize> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    };
    doc["sets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let (m, row, col) = (us(&s["m"]), us(&s["row"]), us(&s["col"]));
            let values = s["value"].as_array().unwrap();
            let z = pair(&s["z"]);
            Set {
                name: s["name"].as_str().unwrap().to_string(),
                op: if s["op"] == "RR" {
                    Op::Regular
                } else {
                    Op::Singular
                },
                z: C::from_f64(z.re, z.im),
                g: s["g"].as_str().unwrap().parse().unwrap(),
                p_src: s["p_src"].as_u64().unwrap() as usize,
                p_tgt: s["p_tgt"].as_u64().unwrap() as usize,
                entries: (0..m.len())
                    .map(|i| (m[i], row[i], col[i], pair(&values[i])))
                    .collect(),
            }
        })
        .collect()
}

/// Every fixture set in the spike's range, Im(k) a ≤ 32 (beyond it the spike's tables,
/// without S2's exponential factor, leave the exponent range; §5.5's pruning removes
/// such levels): by the recursion where Re(k) a ≤ 100 (the spike's double-double rule),
/// by projection where k a ≥ 30. The fixture is f64-rounded, so the bound is near its
/// rounding.
#[test]
#[ignore = "tables to p = 371 in double-double; run in release"]
fn against_s2() {
    let mut worst_rec = 0.0f64;
    let mut worst_proj = 0.0f64;
    let mut checked = 0;
    for set in sets() {
        let ka = set.ka();
        let (k, _) = set.geometry();
        if k.im.abs().to_f64() > 32.0 {
            continue;
        }
        if ka <= 100.0 {
            let e = set.error(&set.table(true));
            assert!(e < 1e-14, "{} recursion: {e:e}", set.name);
            worst_rec = worst_rec.max(e);
            checked += 1;
        }
        if ka >= 30.0 {
            let e = set.error(&set.table(false));
            assert!(e < 1e-14, "{} projection: {e:e}", set.name);
            worst_proj = worst_proj.max(e);
            checked += 1;
        }
    }
    println!("checked {checked}: recursion {worst_rec:e}, projection {worst_proj:e}");
    assert!(checked > 80);
}

/// Whole tables: the double-double recursion against the double-double projection, at
/// k a = 90.5 (real and η = 0.1, the top of the recursion's range in double-double) and
/// k a ≤ 32 (every kind), M2L at d = 2 and √27 and M2M.
#[test]
#[ignore = "whole tables to p = 200 in double-double; run in release"]
fn recursion_against_projection() {
    let kinds = [(1.0, 0.0), (1.0, 0.1), (1.0, 1.0), (0.0, 1.0)];
    for &(ka, eps) in &[(4.0, 1e-10), (32.0, 1e-10), (90.5, 1e-10)] {
        for &(re, im) in &kinds {
            if ka > 32.0 && (im >= 1.0 || re == 0.0) {
                continue;
            }
            let p = nd_fmm_spike_truncation::calibrate::formula_p(ka, eps);
            let k = C::new(Dd::from_f64(re * ka), Dd::from_f64(ka) * Dd::from_f64(im));
            let g = if ka < 1.0 { ka } else { 1.0 };
            for (op, d2) in [
                (Op::Singular, 4.0),
                (Op::Singular, 27.0),
                (Op::Regular, 0.0),
            ] {
                let (t, rho, gg) = if op == Op::Singular {
                    let t = Dd::from_f64(2.0) * Dd::from_f64(d2).sqrt() / Dd::from_f64(3.0).sqrt();
                    (t, Dd::ONE, Dd::from_f64(g))
                } else {
                    (
                        Dd::from_f64(0.5),
                        Dd::from_f64(4.0) / Dd::from_f64(3.0).sqrt(),
                        Dd::ONE,
                    )
                };
                let rec = coaxial::recursion(&coaxial::start_values(op, k * t, gg, 2 * p), gg, p);
                let proj = coaxial::projection(op, k, t, rho, gg, p, p + 100);
                let (input, output) = weights(op, k.to_f64(), gg.to_f64(), p, p);
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for m in 0..=p {
                    for (n1, w_out) in output.iter().enumerate().skip(m) {
                        for (n, w_in) in input.iter().enumerate().skip(m) {
                            let (a, b) = (rec.get(m, n1, n), proj.get(m, n1, n));
                            let w = w_out * w_in;
                            num = num.max((a - b).abs().to_f64() * w);
                            den = den.max(b.abs().to_f64() * w);
                        }
                    }
                }
                let e = num / den;
                println!("ka {ka} k ({re}, {im}) {op:?} d2 {d2} p {p}: {e:e}");
                assert!(e < 1e-24, "ka {ka} k ({re}, {im}) {op:?} d2 {d2}: {e:e}");
            }
        }
    }
}

/// A table of degree P read for p_src, p_tgt < P equals the table of degree max(p_src,
/// p_tgt): the leading block, bit for bit for the recursion.
#[test]
fn rectangular_truncation() {
    let z = C::from_f64(7.3, 0.2);
    let g = Dd::ONE;
    let big = coaxial::recursion(&coaxial::start_values(Op::Singular, z, g, 80), g, 40);
    let small = coaxial::recursion(&coaxial::start_values(Op::Singular, z, g, 50), g, 25);
    for m in 0..=25 {
        for n1 in m..=25 {
            for n in m..=25 {
                // hₗ by the upward recursion does not depend on the length, and each entry
                // only on start values below n + n′ + 1.
                assert_eq!(big.get(m, n1, n), small.get(m, n1, n), "{m} {n1} {n}");
            }
        }
    }
}
