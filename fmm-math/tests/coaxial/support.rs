//! Shared helpers: the fixture, the configurations of the octree (spike S2's
//! `src/config.rs`) and the FMM-weighted error of a coaxial table (S2's `src/metric.rs`,
//! SPIKE_REPORT §4.1).
//!
//! A coaxial table only ever multiplies bounded coefficients, so each entry is weighted
//! by the bounds of its input and output:
//!
//! - **M2L, (S|R)**, between boxes of circumradius a: uₙ u_{n′}, uₙ = Jₙ(a) √(2n + 1)/sⁿ
//!   with Jₙ(R) = max_{r ≤ R} |jₙ(k r)|.
//! - **Shifts, (R|R) = (S|S)**, between a child (circumradius t) and its parent: the
//!   child degree weighs Jₙ(t) √(2n + 1), the parent degree |hₙ(k · 2w_parent)| √(2n + 1)
//!   (the parent's nearest M2L distance, 8t/√3). M2M reads the table from the child to
//!   the parent, L2L from the parent to the child; the weights go with the degrees.
//!
//! The weighted error is max |ΔÊ| w / max |Ê| w over the table: the relative error of the
//! field the table produces, up to a factor of order p.

use nd_fmm_math::Complex;
use nd_fmm_math::coaxial::CoaxialTable;
use nd_fmm_math::radial::{bessel_j, hankel_h};
use num_traits::Zero;

/// p of proposal §3.5 (Gumerov–Duraiswami 2009 eq. 18, δ = 2), as S2's `degree`.
pub fn degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// The squared V-list distances in box widths: offsets with largest coordinate 2 or 3.
pub const DISTANCES2: [u32; 15] = [4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27];

/// The M2L distance at squared distance `d2` box widths, for circumradius a = 1:
/// t = d w = d · 2/√3.
pub fn m2l_t(d2: u32) -> f64 {
    2.0 * (d2 as f64).sqrt() / 3.0f64.sqrt()
}

/// The kinds of k by (Re, Im) of k/κ: real, complex with Im k = 0.1 Re k and Im k =
/// Re k, and imaginary (Yukawa).
pub const KINDS: [(&str, f64, f64); 4] = [
    ("real", 1.0, 0.0),
    ("complex0.1", 1.0, 0.1),
    ("complex1", 1.0, 1.0),
    ("imaginary", 0.0, 1.0),
];

/// The power scale of a level with |k| a = ka.
pub fn scale(ka: f64) -> f64 {
    ka.min(1.0)
}

/// iᵏ, exactly.
pub fn i_pow(k: i64) -> Complex<f64> {
    match k.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

/// Jₙ(R) = max_{0 < r ≤ R} |jₙ(k r)| for n ≤ n_max, up to the common factor e^{Im k R},
/// sampled at 16 points per period π/|k| of the oscillation and at least 16 in all.
pub fn envelope(k: Complex<f64>, radius: f64, n_max: usize) -> Vec<f64> {
    let samples = ((16.0 * k.norm() * radius / std::f64::consts::PI).ceil() as usize).max(16);
    let mut out = vec![0.0f64; n_max + 1];
    let mut j = vec![Complex::zero(); n_max + 1];
    for i in 1..=samples {
        let r = radius * i as f64 / samples as f64;
        bessel_j(k * r, 1.0, &mut j, None);
        // ĵ = j e^{−Im k r}; relative to e^{Im k R}: |ĵ| e^{Im k (r − R)}.
        let f = (k.im * (r - radius)).exp();
        for (o, v) in out.iter_mut().zip(&j) {
            *o = o.max(v.norm() * f);
        }
    }
    out
}

/// The M2L weights uₙ = Jₙ(a) √(2n + 1)/sⁿ, n ≤ p, at a level with circumradius a.
pub fn m2l_weights(k: Complex<f64>, a: f64, s: f64, p: usize) -> Vec<f64> {
    let mut power = 1.0;
    envelope(k, a, p)
        .iter()
        .enumerate()
        .map(|(n, v)| {
            let u = v * ((2 * n + 1) as f64).sqrt() / power;
            power *= s;
            u
        })
        .collect()
}

/// The shift weights at shift t: (child degrees n ≤ p_child: Jₙ(t) √(2n + 1), parent
/// degrees n ≤ p_parent: |ĥₙ(k 8t/√3)| √(2n + 1)).
pub fn shift_weights(
    k: Complex<f64>,
    t: f64,
    p_child: usize,
    p_parent: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut h = vec![Complex::zero(); p_parent + 1];
    hankel_h(k * (8.0 * t / 3.0f64.sqrt()), 1.0, &mut h, None);
    let child = envelope(k, t, p_child)
        .iter()
        .enumerate()
        .map(|(n, v)| v * ((2 * n + 1) as f64).sqrt())
        .collect();
    let parent = h
        .iter()
        .enumerate()
        .map(|(n, v)| v.norm() * ((2 * n + 1) as f64).sqrt())
        .collect();
    (child, parent)
}

/// The weighted error of `got` against `exact` over every entry of tables of degrees
/// p_src → p_tgt, with weight input[n] output[n′].
pub fn weighted(
    p_src: usize,
    p_tgt: usize,
    input: &[f64],
    output: &[f64],
    exact: impl Fn(usize, usize, usize) -> Complex<f64>,
    got: impl Fn(usize, usize, usize) -> Complex<f64>,
) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for m in 0..=p_src.min(p_tgt) {
        for (n1, &w_out) in output.iter().enumerate().take(p_tgt + 1).skip(m) {
            for (n, &w_in) in input.iter().enumerate().take(p_src + 1).skip(m) {
                let (e, g) = (exact(m, n1, n), got(m, n1, n));
                let w = w_out * w_in;
                num = num.max((g - e).norm() * w);
                den = den.max(e.norm() * w);
            }
        }
    }
    num / den
}

/// The weighted difference of two tables of the same degrees.
pub fn table_difference(
    a: &CoaxialTable<f64>,
    b: &CoaxialTable<f64>,
    input: &[f64],
    output: &[f64],
) -> f64 {
    assert_eq!((a.p_src(), a.p_tgt()), (b.p_src(), b.p_tgt()));
    weighted(
        a.p_src(),
        a.p_tgt(),
        input,
        output,
        |m, x, y| a.get(m, x, y),
        |m, x, y| b.get(m, x, y),
    )
}

/// Which coefficients a fixture set holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// (R|R) = (S|S), an M2M set.
    Regular,
    /// (S|R), an M2L set.
    SingularRegular,
}

/// One sampled configuration of `fixtures/helmholtz_coaxial.json`.
pub struct Set {
    pub name: String,
    pub kind: String,
    pub op: Op,
    pub ka: f64,
    /// k for a = 1, the shift t and the circumradius a.
    pub k: Complex<f64>,
    pub t: f64,
    pub a: f64,
    /// z = k t and s as the generator used them.
    pub z: Complex<f64>,
    pub s: f64,
    pub p_src: usize,
    pub p_tgt: usize,
    /// (m, n′, n, value)
    pub entries: Vec<(usize, usize, usize, Complex<f64>)>,
    /// (l, f̂ₗ): the start column's radial values.
    pub radial: Vec<(usize, Complex<f64>)>,
}

impl Set {
    /// Whether S2 found the recursion within its bound (SPIKE_REPORT §4.3): below
    /// k a = 70, and every strongly damped and imaginary k.
    pub fn recursion_stable(&self) -> bool {
        self.ka < 70.0 || self.kind == "complex1" || self.kind == "imaginary"
    }

    /// The input and output weights of the set's table.
    pub fn weights(&self) -> (Vec<f64>, Vec<f64>) {
        match self.op {
            Op::Regular => shift_weights(self.k, self.t, self.p_src, self.p_tgt),
            Op::SingularRegular => {
                let w = m2l_weights(self.k, self.a, self.s, self.p_src);
                (w.clone(), w)
            }
        }
    }

    /// The table by the recursion at the stored z.
    pub fn recursion(&self) -> CoaxialTable<f64> {
        match self.op {
            Op::Regular => CoaxialTable::regular(self.z, self.p_src, self.p_tgt),
            Op::SingularRegular => {
                CoaxialTable::singular_regular(self.z, self.s, self.p_src, self.p_tgt)
            }
        }
    }

    /// The table by the rule of §3.15.
    pub fn by_rule(&self) -> CoaxialTable<f64> {
        match self.op {
            Op::Regular => CoaxialTable::m2m(self.k, self.t, self.p_src, self.p_tgt),
            Op::SingularRegular => {
                CoaxialTable::m2l(self.k, self.t, self.a, self.s, self.p_src, self.p_tgt)
            }
        }
    }
}

/// A decimal string of a double, parsed by Rust's correctly rounded `str::parse`.
fn number(v: &serde_json::Value) -> f64 {
    match v {
        serde_json::Value::String(s) => s.parse().unwrap(),
        other => other.as_f64().unwrap(),
    }
}

fn pair(v: &serde_json::Value) -> Complex<f64> {
    let p = v.as_array().unwrap();
    Complex::new(number(&p[0]), number(&p[1]))
}

/// The fixture's sets.
pub fn sets() -> Vec<Set> {
    let text = include_str!("../../fixtures/helmholtz_coaxial.json");
    let doc: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(
        doc["header"]["convention_version"].as_u64().unwrap() as u32,
        nd_fmm_math::CONVENTION_VERSION
    );
    let usizes = |v: &serde_json::Value| -> Vec<usize> {
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
            let (m, row, col) = (usizes(&s["m"]), usizes(&s["row"]), usizes(&s["col"]));
            let values = s["value"].as_array().unwrap();
            let k = s["k"].as_array().unwrap();
            Set {
                name: s["name"].as_str().unwrap().to_string(),
                kind: s["kind"].as_str().unwrap().to_string(),
                op: if s["op"] == "RR" {
                    Op::Regular
                } else {
                    Op::SingularRegular
                },
                ka: number(&s["ka"]),
                k: Complex::new(number(&k[0]), number(&k[1])),
                t: number(&s["t"]),
                a: number(&s["a"]),
                z: pair(&s["z"]),
                s: number(&s["g"]),
                p_src: s["p_src"].as_u64().unwrap() as usize,
                p_tgt: s["p_tgt"].as_u64().unwrap() as usize,
                entries: (0..m.len())
                    .map(|i| (m[i], row[i], col[i], pair(&values[i])))
                    .collect(),
                radial: usizes(&s["radial_l"])
                    .into_iter()
                    .zip(s["radial"].as_array().unwrap().iter().map(pair))
                    .collect(),
            }
        })
        .collect()
}
