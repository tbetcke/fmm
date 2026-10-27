//! The acceptance tests of CH0.3 (docs/helmholtz/tasks/CH0.3.md, "Tests that prove it").
//! The sweeps to order 1460 are `#[ignore]`d, for debug builds; run them all with
//! `cargo test -p nd-fmm-spike-radial --release -- --include-ignored`.

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8};

use nd_fmm_spike_radial::complex::C64;
use nd_fmm_spike_radial::radial::{bessel_j, hankel, modified_i, modified_k};
use nd_fmm_spike_radial::ranges::{regular_top, singular_top};

/// Bound on the error against the reference (relative, or relative to the oscillation
/// envelope for j and j′).
const BOUND: f64 = 1e-13;

/// Bound on the identities, relative to the size of their terms.
const IDENTITY_BOUND: f64 = 1e-13;

/// Reference values below the f64 range are stored as 0; the spike's must be below this.
const UNDERFLOW: f64 = 1e-280;

fn fixture() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}

/// The spike's values and derivatives at orders 0..=top.
fn evaluate(family: &str, z: C64, s: f64, top: usize) -> (Vec<C64>, Vec<C64>) {
    let mut v = vec![C64::ZERO; top + 1];
    let mut d = vec![C64::ZERO; top + 1];
    match family {
        "j" => bessel_j(z, s, &mut v, Some(&mut d)),
        "h" => hankel(z, s, &mut v, Some(&mut d)),
        _ => {
            let mut vr = vec![0.0; top + 1];
            let mut dr = vec![0.0; top + 1];
            if family == "i" {
                modified_i(z.re, s, &mut vr, Some(&mut dr));
            } else {
                modified_k(z.re, s, &mut vr, Some(&mut dr));
            }
            for n in 0..=top {
                v[n] = C64::new(vr[n], 0.0);
                d[n] = C64::new(dr[n], 0.0);
            }
        }
    }
    (v, d)
}

/// The error of one value: 0 if both underflow, else relative to `scale`; `None` if the
/// reference underflows and the spike does not.
fn error(ours: C64, reference: C64, scale: f64) -> Option<f64> {
    if reference == C64::ZERO {
        return (ours.abs() < UNDERFLOW).then_some(0.0);
    }
    Some((ours - reference).abs() / scale)
}

/// Test 1: every fixture entry, the four families and their derivatives, on every
/// operator's range.
#[test]
#[ignore = "orders to 1460; run in release"]
fn reference_on_every_range() {
    let doc = fixture();
    // (operator, family) → (worst value error, worst derivative error, entries, where)
    let mut worst: BTreeMap<(String, String), (f64, f64, usize, String)> = BTreeMap::new();
    let mut failures = Vec::new();
    for set in doc["sets"].as_array().unwrap() {
        let name = set["name"].as_str().unwrap();
        let family = set["family"].as_str().unwrap();
        let z = C64::new(num(&set["z"][0]), num(&set["z"][1]));
        let s = num(&set["s"]);
        let rows = set["rows"].as_array().unwrap();
        let top = rows
            .iter()
            .map(|r| r[0].as_u64().unwrap() as usize)
            .max()
            .unwrap();
        let (v, d) = evaluate(family, z, s, top);
        let op = name.split('-').next().unwrap();
        let op = if op.starts_with("random") {
            "random"
        } else {
            op
        };
        let entry = worst
            .entry((op.to_string(), family.to_string()))
            .or_insert((0.0, 0.0, 0, String::new()));
        for r in rows {
            let n = r[0].as_u64().unwrap() as usize;
            let rv = C64::new(num(&r[1]), num(&r[2]));
            let rd = C64::new(num(&r[3]), num(&r[4]));
            let (sv, sd) = if family == "j" {
                (num(&r[5]), num(&r[6]))
            } else {
                (rv.abs(), rd.abs())
            };
            for (k, ours, reference, scale) in [(0, v[n], rv, sv), (1, d[n], rd, sd)] {
                match error(ours, reference, scale) {
                    Some(e) if e <= BOUND => {
                        let slot = if k == 0 { &mut entry.0 } else { &mut entry.1 };
                        if e > *slot {
                            *slot = e;
                            if k == 0 {
                                entry.3 = format!("{name} n={n}");
                            }
                        }
                    }
                    e => failures.push(format!(
                        "{name} n={n} {}: {ours:?} vs {reference:?} (error {e:?})",
                        if k == 0 { "value" } else { "derivative" }
                    )),
                }
            }
            entry.2 += 1;
        }
    }
    println!("| operator | family | entries | worst value | worst derivative | at |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for ((op, family), (wv, wd, count, at)) in &worst {
        println!("| {op} | {family} | {count} | {wv:.1e} | {wd:.1e} | {at} |");
    }
    assert!(
        failures.is_empty(),
        "{} entries beyond {BOUND:e}:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

/// The arguments of the dense sweeps: |z| from 10⁻⁶ to 3900, with the highest orders the
/// operators need there, of the regular and of the singular functions (src/ranges.rs).
fn sweep() -> Vec<(f64, usize, usize)> {
    (0..=64)
        .map(|i| {
            let r = 10f64.powf(-6.0 + i as f64 * (3900f64.log10() + 6.0) / 64.0);
            (r, regular_top(r), singular_top(r))
        })
        .collect()
}

const ANGLES: [f64; 7] = [
    0.0,
    0.05,
    FRAC_PI_8,
    FRAC_PI_4,
    3.0 * FRAC_PI_8,
    FRAC_PI_2 - 1e-3,
    FRAC_PI_2,
];

/// Test 2: the Wronskians and the cross product, scaled, over the whole range.
#[test]
#[ignore = "orders to 1460; run in release"]
fn identities() {
    let mut worst = [0.0f64; 3];
    for (r, _, top) in sweep() {
        let s = r.min(1.0);
        for theta in ANGLES {
            let z = C64::new(r * theta.cos(), r * theta.sin());
            let (j, jd) = evaluate("j", z, s, top + 1);
            let (h, hd) = evaluate("h", z, s, top + 1);
            let target = C64::I / (z * z);
            for n in 0..=top {
                // ĵ ĥ′ − ĵ′ ĥ = i/z² (the factors cancel for Im z ≥ 0).
                let (a, b) = (j[n] * hd[n], jd[n] * h[n]);
                let e = (a - b - target).abs() / (a.abs() + b.abs());
                worst[0] = worst[0].max(e);
                assert!(e <= IDENTITY_BOUND, "Wronskian z={z:?} n={n}: {e:e}");
                // jₙ hₙ₊₁ − jₙ₊₁ hₙ = −i/z².
                let (a, b) = (j[n] * h[n + 1] / s, j[n + 1] * h[n] * s);
                let e = (a - b + target).abs() / (a.abs() + b.abs());
                worst[1] = worst[1].max(e);
                assert!(e <= IDENTITY_BOUND, "cross product z={z:?} n={n}: {e:e}");
            }
        }
        // î k̂′ − î′ k̂ = −1/x².
        let (i, id) = evaluate("i", C64::new(r, 0.0), s, top);
        let (k, kd) = evaluate("k", C64::new(r, 0.0), s, top);
        for n in 0..=top {
            let (a, b) = (i[n] * kd[n], id[n] * k[n]);
            let e = (a - b + C64::new(1.0 / (r * r), 0.0)).abs() / (a.abs() + b.abs());
            worst[2] = worst[2].max(e);
            assert!(e <= IDENTITY_BOUND, "modified Wronskian x={r} n={n}: {e:e}");
        }
    }
    println!(
        "worst: Wronskian {:.1e}, cross product {:.1e}, modified Wronskian {:.1e}",
        worst[0], worst[1], worst[2]
    );
}

/// Test 3: the real Yukawa path against the complex path at z = i x:
/// îₙ = i⁻ⁿ ĵₙ(ix), î′ₙ = i¹⁻ⁿ ĵ′ₙ(ix), k̂ₙ = −iⁿ ĥₙ(ix), k̂′ₙ = −iⁿ⁺¹ ĥ′ₙ(ix).
#[test]
#[ignore = "orders to 1460; run in release"]
fn yukawa_two_ways() {
    let mut worst = 0.0f64;
    for (x, top_regular, top_singular) in sweep() {
        let s = x.min(1.0);
        let z = C64::new(0.0, x);
        let (i, id) = evaluate("i", C64::new(x, 0.0), s, top_regular);
        let (k, kd) = evaluate("k", C64::new(x, 0.0), s, top_singular);
        let (j, jd) = evaluate("j", z, s, top_regular);
        let (h, hd) = evaluate("h", z, s, top_singular);
        let regular = (0..=top_regular).flat_map(|n| {
            let p = n as i64;
            [
                (n, i[n], C64::i_pow(-p) * j[n]),
                (n, id[n], C64::i_pow(1 - p) * jd[n]),
            ]
        });
        let singular = (0..=top_singular).flat_map(|n| {
            let p = n as i64;
            [
                (n, k[n], -(C64::i_pow(p) * h[n])),
                (n, kd[n], -(C64::i_pow(p + 1) * hd[n])),
            ]
        });
        {
            for (n, real, complex) in regular.chain(singular) {
                if real.abs() < UNDERFLOW && complex.abs() < UNDERFLOW {
                    continue;
                }
                let e = (real - complex).abs() / real.abs();
                worst = worst.max(e);
                assert!(e <= BOUND, "x={x} n={n}: {real:?} vs {complex:?} ({e:e})");
            }
        }
    }
    println!("worst: {worst:.1e}");
}

/// Test 4: finite on every operator's range, Yukawa at κρ = 3900 included, and the exact
/// values at z = 0.
#[test]
#[ignore = "orders to 1460; run in release"]
fn finite_everywhere() {
    for (r, top_regular, top_singular) in sweep() {
        // The scales of the operators: the box's own (P2M, M2M: ρ ≤ a) and a box √3 to 8
        // times smaller than the distance (M2P, M2L).
        for s in [r.min(1.0), (r / 1.733).min(1.0), (r / 8.0).min(1.0)] {
            for theta in ANGLES {
                let z = C64::new(r * theta.cos(), r * theta.sin());
                let (j, jd) = evaluate("j", z, s, top_regular);
                let (h, hd) = evaluate("h", z, s, top_singular);
                for n in 0..=top_regular {
                    assert!(
                        j[n].is_finite() && jd[n].is_finite(),
                        "j z={z:?} s={s} n={n}"
                    );
                }
                for n in 0..=top_singular {
                    assert!(
                        h[n].is_finite() && hd[n].is_finite(),
                        "h z={z:?} s={s} n={n}"
                    );
                }
            }
            let (i, id) = evaluate("i", C64::new(r, 0.0), s, top_regular);
            let (k, kd) = evaluate("k", C64::new(r, 0.0), s, top_singular);
            for n in 0..=top_regular {
                assert!(i[n].is_finite() && id[n].is_finite(), "i x={r} s={s} n={n}");
            }
            for n in 0..=top_singular {
                assert!(k[n].is_finite() && kd[n].is_finite(), "k x={r} s={s} n={n}");
            }
        }
    }
    for s in [1e-3, 1.0] {
        let (j, jd) = evaluate("j", C64::ZERO, s, 5);
        let (i, id) = evaluate("i", C64::ZERO, s, 5);
        for (v, d) in [(&j, &jd), (&i, &id)] {
            assert_eq!(v[0], C64::new(1.0, 0.0));
            assert!(v[1..].iter().all(|x| *x == C64::ZERO));
            assert_eq!(d[0], C64::ZERO);
            assert!((d[1].re - 1.0 / (3.0 * s)).abs() <= 1e-15 / s);
            assert!(d[2..].iter().all(|x| *x == C64::ZERO));
        }
    }
}

/// Test 5: the same argument at two scales differs by (s′/s)ⁿ.
#[test]
fn scale_consistency() {
    let (s1, s2) = (1.0, 0.3);
    for z in [C64::new(2.0, 0.5), C64::new(0.2, 0.0), C64::new(0.0, 1.5)] {
        for family in ["j", "h", "i", "k"] {
            let z = if family == "i" || family == "k" {
                C64::new(z.abs(), 0.0)
            } else {
                z
            };
            let (a, ad) = evaluate(family, z, s1, 40);
            let (b, bd) = evaluate(family, z, s2, 40);
            let regular = family == "j" || family == "i";
            for n in 0..=40 {
                let f = if regular {
                    (s1 / s2).powi(n as i32)
                } else {
                    (s2 / s1).powi(n as i32)
                };
                for (x, y) in [(a[n], b[n]), (ad[n], bd[n])] {
                    let e = (x * f - y).abs() / y.abs();
                    assert!(e <= BOUND, "{family} z={z:?} n={n}: {e:e}");
                }
            }
        }
    }
}
