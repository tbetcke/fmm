//! The acceptance tests of CH0.2 (docs/helmholtz/tasks/CH0.2.md, "Tests that prove it").
//! The tests that build tables to p = 371 are `#[ignore]`d, for debug builds; run them
//! all with `cargo test -p nd-fmm-spike-coaxial --release -- --include-ignored`.

use nd_fmm_spike_coaxial::coaxial::{Table, Variant, recurse};
use nd_fmm_spike_coaxial::complex::C64;
use nd_fmm_spike_coaxial::config::{degree, m2l_kt, m2m_kt, scale};
use nd_fmm_spike_coaxial::metric::{compare, m2l_weights, m2m_weights};
use nd_fmm_spike_coaxial::projection::{
    Dense, default_nodes, regular_inner, regular_regular, singular_regular,
};
use nd_fmm_spike_coaxial::{Op, start_values};

/// Bound on the FMM-weighted error (src/metric.rs) against the reference.
const WEIGHTED_BOUND: f64 = 1e-11;

/// One sampled configuration of fixtures/reference.json.
struct Set {
    name: String,
    op: Op,
    z: C64,
    g: f64,
    p_src: usize,
    p_tgt: usize,
    /// (m, n′, n, value)
    entries: Vec<(usize, usize, usize, C64)>,
    /// (l, f̂ₗ)
    radial: Vec<(usize, C64)>,
}

impl Set {
    /// k a, parsed from the name (`<kind>-ka<ka>-eps<eps>-<op>`).
    fn ka(&self) -> f64 {
        let part = self.name.split('-').find(|s| s.starts_with("ka")).unwrap();
        part[2..].parse().unwrap()
    }

    /// The wavenumber for a = 1 and the shift t, from the kind and the operator, checked
    /// against the stored z = k t.
    fn geometry(&self) -> (C64, f64) {
        let ka = self.ka();
        let (re, im) = match self.name.split('-').next().unwrap() {
            "real" => (1.0, 0.0),
            "complex0.1" => (1.0, 0.1),
            "complex1" => (1.0, 1.0),
            "imaginary" => (0.0, 1.0),
            other => panic!("unknown kind {other}"),
        };
        let t = if self.name.ends_with("m2m") {
            m2m_kt(1.0)
        } else if self.name.ends_with("d2") {
            m2l_kt(1.0, 4)
        } else {
            m2l_kt(1.0, 27)
        };
        let k = C64::new(re * ka, im * ka);
        let z = k * t;
        assert!((z - self.z).abs() <= 1e-13 * self.z.abs(), "{}", self.name);
        (k, t)
    }

    /// Whether the recursion is expected to meet the bound: SPIKE_REPORT.md §4, every
    /// configuration below k a = 70, and every strongly damped one.
    fn recursion_stable(&self) -> bool {
        self.ka() < 70.0 || self.name.starts_with("complex1-") || self.name.starts_with("imaginary")
    }

    /// The input and output weights of the set's table.
    fn weights(&self, k: C64) -> (Vec<f64>, Vec<f64>) {
        match self.op {
            Op::Regular => m2m_weights(k, 1.0, self.p_src, self.p_tgt),
            Op::Singular => {
                let w = m2l_weights(k, 1.0, self.g, self.p_src);
                (w.clone(), w)
            }
        }
    }

    /// The projection table of the set.
    fn projection(&self, k: C64, t: f64) -> Dense {
        let nodes = default_nodes(self.p_src, self.p_tgt);
        match self.op {
            Op::Regular => {
                regular_regular(k, t, 4.0 / 3.0f64.sqrt(), self.p_src, self.p_tgt, nodes)
            }
            Op::Singular => singular_regular(k, t, 1.0, self.g, self.p_src, self.p_tgt, nodes),
        }
    }
}

fn pair(v: &serde_json::Value) -> C64 {
    let p = v.as_array().unwrap();
    C64::new(
        p[0].as_str().unwrap().parse().unwrap(),
        p[1].as_str().unwrap().parse().unwrap(),
    )
}

fn sets() -> Vec<Set> {
    let text = include_str!("../fixtures/reference.json");
    let doc: serde_json::Value = serde_json::from_str(text).unwrap();
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
            let entries = (0..m.len())
                .map(|i| (m[i], row[i], col[i], pair(&values[i])))
                .collect();
            let radial = usizes(&s["radial_l"])
                .into_iter()
                .zip(s["radial"].as_array().unwrap().iter().map(pair))
                .collect();
            let z = s["z"].as_array().unwrap();
            Set {
                name: s["name"].as_str().unwrap().to_string(),
                op: if s["op"] == "RR" {
                    Op::Regular
                } else {
                    Op::Singular
                },
                z: C64::new(
                    z[0].as_str().unwrap().parse().unwrap(),
                    z[1].as_str().unwrap().parse().unwrap(),
                ),
                g: s["g"].as_str().unwrap().parse().unwrap(),
                p_src: s["p_src"].as_u64().unwrap() as usize,
                p_tgt: s["p_tgt"].as_u64().unwrap() as usize,
                entries,
                radial,
            }
        })
        .collect()
}

/// Entries of the reference below this are rounded to (near) zero in f64 and not
/// compared entrywise.
const TINY: f64 = 1e-290;

/// Test 1a: the spike's start values against the reference, relative 1e-13.
#[test]
fn radial_functions_agree_with_the_reference() {
    let mut worst = (0.0f64, String::new());
    for set in sets() {
        let start = start_values(set.op, set.z, set.g, set.p_src + set.p_tgt);
        for &(l, exact) in &set.radial {
            if exact.abs() < TINY {
                continue;
            }
            let err = (start[l] - exact).abs() / exact.abs();
            if err > worst.0 {
                worst = (err, format!("{} l={l}", set.name));
            }
        }
    }
    println!(
        "start values: worst relative error {:.2e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= 1e-13, "{:?}", worst);
}

/// Errors of `got` at the set's sampled entries: (weighted, entrywise). The weighted
/// error is max over the samples of |Δ| w divided by the largest |Ê| w of the full
/// `scale` table.
fn sampled_errors(
    set: &Set,
    input: &[f64],
    output: &[f64],
    scale: impl Fn(usize, usize, usize) -> C64,
    got: impl Fn(usize, usize, usize) -> C64,
) -> (f64, f64) {
    let mut den = 0.0f64;
    for m in 0..=set.p_src.min(set.p_tgt) {
        for (n1, &w_out) in output.iter().enumerate().take(set.p_tgt + 1).skip(m) {
            for (n, &w_in) in input.iter().enumerate().take(set.p_src + 1).skip(m) {
                den = den.max(scale(m, n1, n).abs() * w_out * w_in);
            }
        }
    }
    let (mut num, mut entry) = (0.0f64, 0.0f64);
    for &(m, n1, n, exact) in &set.entries {
        let v = got(m, n1, n);
        assert!(v.is_finite(), "{} ({m}, {n1}, {n}) not finite", set.name);
        let err = (v - exact).abs();
        num = num.max(err * output[n1] * input[n]);
        if exact.abs() >= TINY {
            entry = entry.max(err / exact.abs());
        }
    }
    (num / den, entry)
}

/// Test 1: every sampled entry against both methods (the recursion from the spike's own
/// start values, and the projection). The projection meets the weighted bound on every
/// configuration; the recursion where SPIKE_REPORT.md §4 says it is stable, and the
/// rest is reported. The per-entry relative error is reported for both.
#[test]
#[ignore = "release mode: tables to p = 371"]
fn agrees_with_the_arbitrary_precision_reference() {
    let mut worst_projection = (0.0f64, String::new());
    let mut worst_recursion = (0.0f64, String::new());
    for set in sets() {
        let (k, t) = set.geometry();
        let (input, output) = set.weights(k);
        let start = start_values(set.op, set.z, set.g, set.p_src + set.p_tgt);
        let table = Table::build(&start, set.g, set.p_src, set.p_tgt);
        let proj = set.projection(k, t);
        let scale = |m, a, b| proj.get(m, a, b);
        let (rw, re) = sampled_errors(&set, &input, &output, scale, |m, a, b| table.get(m, a, b));
        let (pw, pe) = sampled_errors(&set, &input, &output, scale, |m, a, b| proj.get(m, a, b));
        // The normalisation is the projection's full-table maximum; where the recursion is
        // stable, an independent table must give the same maximum.
        if set.recursion_stable() {
            let largest = |f: &dyn Fn(usize, usize, usize) -> C64| {
                let mut out = 0.0f64;
                for m in 0..=set.p_src.min(set.p_tgt) {
                    for (n1, &w_out) in output.iter().enumerate().take(set.p_tgt + 1).skip(m) {
                        for (n, &w_in) in input.iter().enumerate().take(set.p_src + 1).skip(m) {
                            out = out.max(f(m, n1, n).abs() * w_out * w_in);
                        }
                    }
                }
                out
            };
            let (a, b) = (
                largest(&|m, x, y| proj.get(m, x, y)),
                largest(&|m, x, y| table.get(m, x, y)),
            );
            assert!(
                (a - b).abs() <= 1e-10 * a,
                "{}: weighted maxima {a:e} {b:e}",
                set.name
            );
        }
        println!(
            "{:<38} p {:>3} -> {:>3}: recursion weighted {rw:.1e} entrywise {re:.1e}; \
             projection weighted {pw:.1e} entrywise {pe:.1e}",
            set.name, set.p_src, set.p_tgt
        );
        if pw > worst_projection.0 {
            worst_projection = (pw, set.name.clone());
        }
        if set.recursion_stable() && rw > worst_recursion.0 {
            worst_recursion = (rw, set.name.clone());
        }
    }
    println!(
        "worst weighted: projection {:.1e} ({}), recursion where stable {:.1e} ({})",
        worst_projection.0, worst_projection.1, worst_recursion.0, worst_recursion.1
    );
    assert!(worst_projection.0 <= WEIGHTED_BOUND, "{worst_projection:?}");
    assert!(worst_recursion.0 <= WEIGHTED_BOUND, "{worst_recursion:?}");
}

/// Every computed column of a variant, as a map (m, n′, n) → Ê.
fn columns(
    start: &[C64],
    g: f64,
    p_src: usize,
    p_tgt: usize,
    variant: Variant,
) -> std::collections::HashMap<(usize, usize, usize), C64> {
    let mut out = std::collections::HashMap::new();
    recurse(start, g, p_src, p_tgt, variant, |m, n, lo, values| {
        for (i, v) in values.iter().enumerate() {
            if lo + i <= p_tgt.max(p_src) {
                out.insert((m, lo + i, n), *v);
            }
        }
    });
    out
}

/// Test 4: the lower triangle n′ ≥ n is closed under the recursions, so both variants
/// compute it bit for bit alike. The full variant's upper triangle, against the index
/// symmetry, is wrong by O(1) per entry for (R|R) and right for (S|R); it is also
/// measured in the FMM weights. The shift is complex (Im z = 0.3 Re z).
#[test]
fn the_variants_share_the_lower_triangle() {
    for (op, ka) in [(Op::Regular, 3.0), (Op::Singular, 10.0), (Op::Regular, 0.1)] {
        let p = degree(ka, 1e-10);
        let (kt, g) = match op {
            Op::Regular => (m2m_kt(ka), 1.0),
            Op::Singular => (m2l_kt(ka, 4), scale(ka)),
        };
        let z = C64::new(kt, 0.3 * kt);
        let start = start_values(op, z, g, 2 * p);
        let tri = columns(&start, g, p, p, Variant::Triangle);
        let full = columns(&start, g, p, p, Variant::Full);
        let k = C64::new(ka, 0.3 * ka);
        let (input, output) = match op {
            Op::Regular => m2m_weights(k, 1.0, p, p),
            Op::Singular => {
                let w = m2l_weights(k, 1.0, g, p);
                (w.clone(), w)
            }
        };
        let (mut upper, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        for (&(m, n1, n), v) in &full {
            let sym = if n1 >= n {
                assert_eq!(tri[&(m, n1, n)], *v, "({m}, {n1}, {n})");
                *v
            } else {
                let sym = tri[&(m, n, n1)] * if (n + n1) % 2 == 1 { -1.0 } else { 1.0 };
                let size = v.abs().max(sym.abs());
                if size > 0.0 {
                    upper = upper.max((*v - sym).abs() / size);
                }
                sym
            };
            if n1 <= p && n <= p {
                let w = output[n1] * input[n];
                num = num.max((*v - sym).abs() * w);
                den = den.max(sym.abs() * w);
            }
        }
        println!(
            "{op:?} ka={ka} p={p}: full variant's upper triangle against the symmetry {upper:.2e} \
             per entry, {:.2e} weighted",
            num / den
        );
        match op {
            Op::Regular => assert!(upper > 0.5, "{upper:e}"),
            Op::Singular => assert!(upper < 1e-12, "{upper:e}"),
        }
    }
}

/// Test 2: for real k, (R|R) = (S|S) is real. The acceptance item rests on the
/// reference, whose definition check confirms the addition theorems with these
/// coefficients (tools/reference.py); here, in f64:
/// - the (R|R) recursion closes over the reals (imaginary parts exactly zero; structural);
/// - Re (S|R) = (R|R) (hₗ = jₗ + i yₗ, Gaunt phases ±1) in the M2L weights to 1e-11, by the
///   recursion at every V-list distance (k a ≤ 30) and by the inner-sphere projection at
///   k a = 100 and 300; since both methods are linear with real coefficients, this checks
///   that Re ĥₗ and ĵₗ agree as the start values feed them, not the identity itself;
/// - the (S|S) = (R|R) table by projection with hₗ on the outer sphere, whose arithmetic
///   is complex throughout, has imaginary parts below 1e-11 in the M2M weights at
///   k a = 100 and 300: a check that can fail.
#[test]
#[ignore = "release mode: tables to p = 371"]
fn real_k_regular_coefficients_are_real() {
    // Errors of an approximation `got` of the reference values `of`, in the M2L weights.
    let weighted =
        |p: usize,
         w: &[f64],
         of: &dyn Fn(usize, usize, usize) -> C64,
         got: &dyn Fn(usize, usize, usize) -> C64| { compare(p, p, w, w, of, got).0 };
    let mut recursion = 0.0f64;
    for ka in [0.01, 1.0, 10.0, 30.0] {
        let p = degree(ka, 1e-10);
        let g = scale(ka);
        let w = m2l_weights(C64::new(ka, 0.0), 1.0, g, p);
        for d2 in nd_fmm_spike_coaxial::config::DISTANCES2 {
            let z = C64::new(m2l_kt(ka, d2), 0.0);
            let rr = Table::build(&start_values(Op::Regular, z, g, 2 * p), g, p, p);
            let sr = Table::build(&start_values(Op::Singular, z, g, 2 * p), g, p, p);
            for m in 0..rr.orders() {
                for n1 in m..=p {
                    for n in m..=p {
                        assert_eq!(rr.get(m, n1, n).im, 0.0, "ka={ka} d2={d2} ({m}, {n1}, {n})");
                    }
                }
            }
            // `got` differs from (S|R) by Re (S|R) − (R|R).
            let err = weighted(p, &w, &|m, a, b| sr.get(m, a, b), &|m, a, b| {
                sr.get(m, a, b) - C64::new(sr.get(m, a, b).re - rr.get(m, a, b).re, 0.0)
            });
            recursion = recursion.max(err);
        }
    }
    let mut projected = (0.0f64, 0.0f64);
    for ka in [100.0, 300.0] {
        let p = degree(ka, 1e-10);
        let k = C64::new(ka, 0.0);
        let t = m2l_kt(1.0, 4);
        let w = m2l_weights(k, 1.0, 1.0, p);
        let nodes = default_nodes(p, p);
        let sr = singular_regular(k, t, 1.0, 1.0, p, p, nodes);
        let rr = regular_inner(k, t, 1.0, 1.0, p, p, nodes);
        let err = weighted(p, &w, &|m, a, b| sr.get(m, a, b), &|m, a, b| {
            sr.get(m, a, b) - C64::new(sr.get(m, a, b).re - rr.get(m, a, b).re, 0.0)
        });
        // `got` differs from (S|R) by Im (R|R).
        let imag = weighted(p, &w, &|m, a, b| sr.get(m, a, b), &|m, a, b| {
            sr.get(m, a, b) + C64::new(0.0, rr.get(m, a, b).im)
        });
        projected = (projected.0.max(err), projected.1.max(imag));
    }
    let mut outer = 0.0f64;
    for ka in [100.0, 300.0] {
        let (p, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
        let k = C64::new(ka, 0.0);
        let (input, output) = m2m_weights(k, 1.0, pc, p);
        let rr = regular_regular(k, 0.5, 4.0 / 3.0f64.sqrt(), pc, p, default_nodes(pc, p));
        let (err, _) = compare(
            pc,
            p,
            &input,
            &output,
            |m, a, b| rr.get(m, a, b),
            |m, a, b| C64::new(rr.get(m, a, b).re, 0.0),
        );
        outer = outer.max(err);
    }
    println!(
        "Re (S|R) against (R|R), M2L weights: recursion {recursion:.2e}; projection {:.2e}, \
         Im (R|R) {:.2e}; outer-sphere (S|S) imaginary parts, M2M weights {outer:.2e}",
        projected.0, projected.1
    );
    assert!(outer <= 1e-11, "{outer:.2e}");
    assert!(recursion <= 1e-11, "{recursion:.2e}");
    assert!(
        projected.0 <= 1e-11 && projected.1 <= 1e-11,
        "{projected:?}"
    );
}

/// Test 3: for imaginary k, i^{n−n′} Êᵐ_{n′n} is real: exactly in f64 (the arithmetic on
/// purely real and purely imaginary parts produces exact zeros), and to 1e-30 relative in
/// the reference.
#[test]
fn imaginary_k_is_real_after_the_powers_of_i() {
    let mut worst = 0.0f64;
    for (op, ka) in [
        (Op::Regular, 1.0),
        (Op::Regular, 30.0),
        (Op::Singular, 1.0),
        (Op::Singular, 30.0),
    ] {
        let p = degree(ka, 1e-10);
        let (kt, g) = match op {
            Op::Regular => (m2m_kt(ka), 1.0),
            Op::Singular => (m2l_kt(ka, 12), scale(ka)),
        };
        let table = Table::build(&start_values(op, C64::new(0.0, kt), g, 2 * p), g, p, p);
        for m in 0..table.orders() {
            for n1 in m..=p {
                let size = table.degree_max(n1);
                for n in m..=p {
                    let v = C64::i_pow(n as i64 - n1 as i64) * table.get(m, n1, n);
                    worst = worst.max(v.im.abs() / size);
                }
            }
        }
    }
    let mut reference = 0.0f64;
    for set in sets().iter().filter(|s| s.name.starts_with("imaginary")) {
        for &(_, n1, n, v) in &set.entries {
            let w = C64::i_pow(n as i64 - n1 as i64) * v;
            if w.abs() > 0.0 {
                reference = reference.max(w.im.abs() / w.abs());
            }
        }
    }
    println!("imaginary k: f64 {worst:.2e}, reference {reference:.2e}");
    assert!(
        worst == 0.0 && reference <= 1e-30,
        "{worst:.2e} {reference:.2e}"
    );
}

/// Test 5: the rectangular table p_src → p_tgt is a block of the square table at
/// max(p_src, p_tgt), bit for bit, from the same start values.
#[test]
fn rectangular_truncation_is_a_block_of_the_square() {
    for op in [Op::Regular, Op::Singular] {
        let (p_child, p_parent) = (degree(5.0, 1e-6), degree(10.0, 1e-6));
        let z = C64::new(5.0, 0.5);
        let start = start_values(op, z, 1.0, 2 * p_parent);
        let square = Table::build(&start, 1.0, p_parent, p_parent);
        for (p_src, p_tgt) in [(p_child, p_parent), (p_parent, p_child)] {
            let rect = Table::build(&start, 1.0, p_src, p_tgt);
            for m in 0..rect.orders() {
                for n1 in m..=p_tgt {
                    for n in m..=p_src {
                        assert_eq!(rect.get(m, n1, n), square.get(m, n1, n));
                    }
                }
            }
        }
    }
}

/// Test 6: the whole recursion table against the projection at the threshold proposed in
/// SPIKE_REPORT.md §8 (Re(k) a = 40, p = 77), every kind of k, M2M and all 15 V-list
/// distances, in the FMM weights, bound 1e-11. The fixture's samples cannot show this:
/// they miss the entries where the recursion fails.
#[test]
#[ignore = "release mode: 64 tables to p = 77 by projection"]
fn recursion_is_stable_at_the_proposed_threshold() {
    let ka = 40.0;
    let (p, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
    let mut worst = (0.0f64, String::new());
    for (name, re, im) in [
        ("real", 1.0, 0.0),
        ("complex0.1", 1.0, 0.1),
        ("complex1", 1.0, 1.0),
        ("imaginary", 0.0, 1.0),
    ] {
        let k = C64::new(re * ka, im * ka);
        let kt = m2m_kt(ka);
        let rec = Table::build(
            &start_values(Op::Regular, C64::new(re * kt, im * kt), 1.0, pc + p),
            1.0,
            pc,
            p,
        );
        let proj = regular_regular(k, 0.5, 4.0 / 3.0f64.sqrt(), pc, p, default_nodes(pc, p));
        let (input, output) = m2m_weights(k, 1.0, pc, p);
        let (err, _) = compare(
            pc,
            p,
            &input,
            &output,
            |m, a, b| proj.get(m, a, b),
            |m, a, b| rec.get(m, a, b),
        );
        if err > worst.0 {
            worst = (err, format!("{name} M2M"));
        }
        let g = scale(k.abs());
        let w = m2l_weights(k, 1.0, g, p);
        for d2 in nd_fmm_spike_coaxial::config::DISTANCES2 {
            let kt = m2l_kt(ka, d2);
            let rec = Table::build(
                &start_values(Op::Singular, C64::new(re * kt, im * kt), g, 2 * p),
                g,
                p,
                p,
            );
            let proj = singular_regular(k, m2l_kt(1.0, d2), 1.0, g, p, p, default_nodes(p, p));
            let (err, _) = compare(
                p,
                p,
                &w,
                &w,
                |m, a, b| proj.get(m, a, b),
                |m, a, b| rec.get(m, a, b),
            );
            if err > worst.0 {
                worst = (err, format!("{name} M2L d2={d2}"));
            }
        }
    }
    println!(
        "k a = 40, p {pc} -> {p}: worst weighted {:.2e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= 1e-11, "{worst:?}");
}
