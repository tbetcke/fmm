//! Tests 1–4 and 7–8 of docs/helmholtz/tasks/CH0.7.md: the double-double arithmetic, the
//! radial functions, the harmonics, the rotations, the geometry and the formula.

use nd_fmm_spike_truncation::calibrate::{formula_p, smallest_passing};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::geometry::{
    Placement, Rng, child_index, children, distances, near_offsets, offsets, points,
};
use nd_fmm_spike_truncation::harmonics::{Harmonics, Polar, idx, len};
use nd_fmm_spike_truncation::radial;
use nd_fmm_spike_truncation::real::{Dd, Real};
use nd_fmm_spike_truncation::rotation::{Direction, Rotation, YBlocks};

type C = Complex<Dd>;

fn rel(a: Dd, b: Dd) -> f64 {
    ((a - b).abs() / b.abs()).to_f64()
}

fn dd(s: &str) -> Dd {
    // A decimal string to double-double: integer digits and fraction, exactly enough for
    // the constants below (40 digits).
    let (int, frac) = s.split_once('.').unwrap();
    let mut v = Dd::from_f64(int.parse::<f64>().unwrap());
    let mut scale = Dd::ONE;
    for ch in frac.chars() {
        scale = scale.div_f64(10.0);
        v += scale * Dd::from_f64(ch.to_digit(10).unwrap() as f64);
    }
    v
}

#[test]
fn double_double_arithmetic() {
    let pi = dd("3.1415926535897932384626433832795028841971");
    let e = dd("2.7182818284590452353602874713526624977572");
    let ln2 = dd("0.6931471805599453094172321214581765680755");
    assert!(rel(Dd::PI, pi) < 1e-31, "{}", rel(Dd::PI, pi));
    assert!(rel(Dd::E, e) < 1e-31);
    assert!(rel(Dd::LN2, ln2) < 1e-31);
    assert!(rel(Dd::ONE.exp(), e) < 1e-30, "{}", rel(Dd::ONE.exp(), e));
    let mut rng = Rng::new(1);
    for _ in 0..2000 {
        let x = Dd::from_f64(rng.range(-50.0, 50.0)) + Dd::from_f64(rng.range(-1e-17, 1e-17));
        let (s, c) = x.sin_cos();
        let one = s * s + c * c;
        assert!((one - Dd::ONE).abs().to_f64() < 1e-30, "sin²+cos² at {x:?}");
        let p = x.exp() * (-x).exp();
        assert!((p - Dd::ONE).abs().to_f64() < 1e-30, "exp at {x:?}");
        let y = x.abs() + Dd::from_f64(1e-3);
        let r = y.sqrt();
        assert!(rel(r * r, y) < 1e-31);
        let q = x / y;
        assert!(rel(q * y, x) < 1e-31);
        // f64 path against std.
        let xf = x.to_f64();
        assert!((s.to_f64() - xf.sin()).abs() < 1e-15);
        assert!((c.to_f64() - xf.cos()).abs() < 1e-15);
    }
    // Large arguments of sin, cos: sin(1000π + π/6) = 1/2.
    let x = Dd::PI * Dd::from_f64(1000.0) + Dd::PI.div_f64(6.0);
    assert!(((x.sin_cos().0 - Dd::from_f64(0.5)).abs().to_f64()) < 1e-28);
}

/// jₙ yₙ₋₁ − jₙ₋₁ yₙ = 1/z² with yₙ = (hₙ − jₙ)/i.
#[test]
fn radial_cross_products() {
    let one = Dd::ONE;
    let zs = [
        C::from_f64(1e-3, 0.0),
        C::from_f64(0.3, 0.0),
        C::from_f64(2.5, 0.0),
        C::from_f64(40.0, 0.0),
        C::from_f64(700.0, 0.0),
        C::from_f64(2000.0, 0.0),
        C::from_f64(30.0, 3.0),
        C::from_f64(30.0, 30.0),
        C::from_f64(0.0, 0.5),
        C::from_f64(0.0, 25.0),
        C::from_f64(0.0, 150.0),
    ];
    for z in zs {
        let n_max = 700usize.min(((z.abs().to_f64() + 40.0) * 2.0) as usize);
        let mut j = vec![C::ZERO; n_max + 1];
        let mut h = vec![C::ZERO; n_max + 1];
        radial::bessel_j(z, one, &mut j);
        radial::hankel(z, one, &mut h);
        let y: Vec<C> = h.iter().zip(&j).map(|(h, j)| (*h - *j) / C::I).collect();
        let target = C::ONE / (z * z);
        let mut worst = 0.0f64;
        for n in 1..=n_max {
            // Below about 10⁻²⁹² the low part of a double-double is subnormal.
            if j[n].abs().to_f64() < 1e-280 || y[n].abs().to_f64() > 1e280 {
                break;
            }
            let w = j[n] * y[n - 1] - j[n - 1] * y[n];
            // Relative to the size of the terms, which cancel.
            let size = (j[n] * y[n - 1]).abs() + (j[n - 1] * y[n]).abs();
            let err = ((w - target).abs() / size).to_f64();
            worst = worst.max(err);
        }
        assert!(worst < 1e-28, "z = {:?}: {worst:e}", z.to_f64());
        // Miller's start raised: the same values.
        let mut j2 = vec![C::ZERO; n_max + 150];
        radial::bessel_j(z, one, &mut j2);
        // Relative to the envelope max_{l ≤ n} |jₗ|, since jₙ has zeros for n < |z|.
        let mut envelope = 0.0f64;
        for n in 0..=n_max {
            if j[n].abs().to_f64() < 1e-280 {
                break;
            }
            envelope = envelope.max(j[n].abs().to_f64());
            let d = (j2[n] - j[n]).abs().to_f64();
            assert!(
                d <= 1e-30 * envelope,
                "Miller at z = {:?}, n = {n}",
                z.to_f64()
            );
        }
    }
}

/// Against S3's arbitrary-precision reference (f64 values).
#[test]
fn radial_against_s3() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../radial/fixtures/reference.json"
    ))
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut checked = 0;
    let mut worst = 0.0f64;
    for set in v["sets"].as_array().unwrap() {
        let family = set["family"].as_str().unwrap();
        if family != "j" && family != "h" {
            continue;
        }
        let z = set["z"].as_array().unwrap();
        let z = C::from_f64(
            z[0].as_str().unwrap().parse().unwrap(),
            z[1].as_str().unwrap().parse().unwrap(),
        );
        let s: f64 = set["s"].as_str().unwrap().parse().unwrap();
        // S3 stores ĵ = e^{−|Im z|} j / sⁿ and ĥ = e^{Im z} h sⁿ.
        if z.im.to_f64().abs() > 200.0 || z.abs().to_f64() > 3000.0 {
            continue;
        }
        let rows = set["rows"].as_array().unwrap();
        let n_max = rows
            .iter()
            .map(|r| r[0].as_u64().unwrap() as usize)
            .max()
            .unwrap();
        let mut f = vec![C::ZERO; n_max + 1];
        let sd = Dd::from_f64(s);
        if family == "j" {
            radial::bessel_j(z, sd, &mut f);
        } else {
            radial::hankel(z, sd, &mut f);
        }
        let factor = if family == "j" {
            (-z.im.abs()).exp()
        } else {
            z.im.exp()
        };
        for r in rows {
            let n = r[0].as_u64().unwrap() as usize;
            let want = Complex::new(r[1].as_f64().unwrap(), r[2].as_f64().unwrap());
            let got = (f[n] * factor).to_f64();
            let size = want.abs();
            if size < 1e-280 || !size.is_finite() {
                continue;
            }
            let err = (got - want).abs() / size;
            worst = worst.max(err);
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked}");
    assert!(worst < 1e-14, "worst {worst:e} over {checked}");
}

/// Orthonormality on the sphere, the addition theorem, and the gradient against central
/// differences.
#[test]
fn harmonics_identities() {
    let p = 24;
    let nq = 40;
    let (xs, ws) = nd_fmm_spike_truncation::coaxial::gauss_legendre::<Dd>(nq);
    let nphi = 2 * p + 2;
    let mut gram = vec![Dd::ZERO; len(p) * len(p)];
    let mut h = Harmonics::<Dd>::new(p);
    let two_pi = Dd::PI * Dd::from_f64(2.0);
    for (x, w) in xs.iter().zip(&ws) {
        let s = (Dd::ONE - *x * *x).sqrt();
        for j in 0..nphi {
            let phi = two_pi * Dd::from_f64(j as f64) / Dd::from_f64(nphi as f64);
            let (sp, cp) = phi.sin_cos();
            let polar = Polar {
                r: Dd::ONE,
                u: *x,
                s,
                cp,
                sp,
            };
            h.eval(&polar, false);
            let wt = *w * two_pi / Dd::from_f64(nphi as f64);
            for a in 0..len(p) {
                for b in 0..len(p) {
                    gram[a * len(p) + b] += h.y[a] * h.y[b] * wt;
                }
            }
        }
    }
    for a in 0..len(p) {
        for b in 0..len(p) {
            let want = if a == b { Dd::ONE } else { Dd::ZERO };
            assert!(
                (gram[a * len(p) + b] - want).abs().to_f64() < 1e-29,
                "gram {a} {b}"
            );
        }
    }
    // Addition theorem.
    let mut rng = Rng::new(7);
    let mut hx = Harmonics::<Dd>::new(p);
    let mut hy = Harmonics::<Dd>::new(p);
    for _ in 0..20 {
        let a = rng.direction();
        let b = rng.direction();
        let pa = Polar::new([Dd::from_f64(a[0]), Dd::from_f64(a[1]), Dd::from_f64(a[2])]);
        let pb = Polar::new([Dd::from_f64(b[0]), Dd::from_f64(b[1]), Dd::from_f64(b[2])]);
        hx.eval(&pa, false);
        hy.eval(&pb, false);
        let cosg = pa.u * pb.u + pa.s * pb.s * (pa.cp * pb.cp + pa.sp * pb.sp);
        let (mut p0, mut p1) = (Dd::ONE, cosg);
        for n in 0..=p {
            let leg = if n == 0 { p0 } else { p1 };
            let mut sum = Dd::ZERO;
            for m in -(n as isize)..=n as isize {
                sum += hx.y[idx(n, m)] * hy.y[idx(n, m)];
            }
            let want = Dd::from_usize(2 * n + 1) * leg / (Dd::from_f64(4.0) * Dd::PI);
            assert!((sum - want).abs().to_f64() < 1e-29, "addition n = {n}");
            if n >= 1 {
                let next = (Dd::from_usize(2 * n + 1) * cosg * p1 - Dd::from_usize(n) * p0)
                    / Dd::from_usize(n + 1);
                p0 = p1;
                p1 = next;
            }
        }
    }
    // Gradients: ∂_θ and (1/sin θ) ∂_φ against central differences in θ and φ.
    let step = 1e-9;
    for _ in 0..10 {
        let theta = rng.range(0.05, 3.1);
        let phi = rng.range(0.0, 6.2);
        // θ ± h and φ ± h formed in double-double, exactly.
        let at = |t: f64, dt: f64, f: f64, df: f64| {
            let (st, ct) = (Dd::from_f64(t) + Dd::from_f64(dt)).sin_cos();
            let (sf, cf) = (Dd::from_f64(f) + Dd::from_f64(df)).sin_cos();
            Polar {
                r: Dd::ONE,
                u: ct,
                s: st,
                cp: cf,
                sp: sf,
            }
        };
        let mut h0 = Harmonics::<Dd>::new(p);
        h0.eval(&at(theta, 0.0, phi, 0.0), true);
        let mut hp = Harmonics::<Dd>::new(p);
        let mut hm = Harmonics::<Dd>::new(p);
        hp.eval(&at(theta, step, phi, 0.0), false);
        hm.eval(&at(theta, -step, phi, 0.0), false);
        for i in 0..len(p) {
            let fd = (hp.y[i] - hm.y[i]).to_f64() / (2.0 * step);
            assert!((fd - h0.dtheta[i].to_f64()).abs() < 1e-12, "dθ {i}");
        }
        hp.eval(&at(theta, 0.0, phi, step), false);
        hm.eval(&at(theta, 0.0, phi, -step), false);
        for i in 0..len(p) {
            let fd = (hp.y[i] - hm.y[i]).to_f64() / (2.0 * step) / theta.sin();
            assert!((fd - h0.dphi[i].to_f64()).abs() < 1e-12, "dφ {i}");
        }
    }
}

fn apply(q: [[Dd; 3]; 3], x: [Dd; 3]) -> [Dd; 3] {
    std::array::from_fn(|i| q[i][0] * x[0] + q[i][1] * x[1] + q[i][2] * x[2])
}

/// Y(Q x) = O(Q) Y(x) for the rotation onto +z of every offset and child direction, and
/// Q t̂ = ẑ.
#[test]
fn rotations_definition() {
    let p = 20;
    let mut dirs: Vec<[f64; 3]> = offsets()
        .iter()
        .map(|o| [o[0] as f64, o[1] as f64, o[2] as f64])
        .collect();
    for c in children([0.0; 3], 1.0) {
        dirs.push(c);
    }
    let mut rng = Rng::new(3);
    let mut hx = Harmonics::<Dd>::new(p);
    let mut hq = Harmonics::<Dd>::new(p);
    for d in dirs {
        let t = [Dd::from_f64(d[0]), Dd::from_f64(d[1]), Dd::from_f64(d[2])];
        let dir = Direction::new(t);
        let q = dir.matrix();
        let qt = apply(q, t);
        let r = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        assert!(qt[0].abs().to_f64() < 1e-30 && qt[1].abs().to_f64() < 1e-30);
        assert!(((qt[2] - r) / r).abs().to_f64() < 1e-30);
        let rot = Rotation::new(dir, p);
        // An expansion c, its rotated coefficients c′ = O(Q) c: c′ · Y(Q x) = c · Y(x).
        let c: Vec<C> = (0..len(p))
            .map(|_| C::from_f64(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)))
            .collect();
        let mut cp = vec![C::ZERO; len(p)];
        let mut work = vec![C::ZERO; len(p)];
        rot.forward(p, &c, &mut cp, &mut work);
        let mut back = vec![C::ZERO; len(p)];
        rot.back(p, &cp, &mut back, &mut work);
        for i in 0..len(p) {
            assert!((back[i] - c[i]).abs().to_f64() < 1e-29, "back {i}");
        }
        for _ in 0..3 {
            let x = rng.direction();
            let x = [Dd::from_f64(x[0]), Dd::from_f64(x[1]), Dd::from_f64(x[2])];
            hx.eval(&Polar::new(x), false);
            hq.eval(&Polar::new(apply(q, x)), false);
            let (mut a, mut b) = (C::ZERO, C::ZERO);
            for i in 0..len(p) {
                a += c[i] * hx.y[i];
                b += cp[i] * hq.y[i];
            }
            assert!(
                (a - b).abs().to_f64() < 1e-28,
                "definition at {d:?}: {:e}",
                (a - b).abs().to_f64()
            );
        }
    }
}

/// Orthogonality of the double-double blocks to degree 300, and S1's fixture.
#[test]
#[ignore = "degree 300 in double-double; run in release"]
fn rotations_large_degree() {
    let p = 300;
    let beta = (1.0f64 / 3.0f64.sqrt()).acos();
    let (s, x) = Dd::from_f64(beta).sin_cos();
    let blocks = YBlocks::new(p, x, s);
    for n in [1, 10, 100, 200, 300] {
        let mut worst = 0.0f64;
        let ni = n as isize;
        for a in -ni..=ni {
            for b in -ni..=ni {
                let mut sum = Dd::ZERO;
                for c in -ni..=ni {
                    sum += blocks.entry(n, a, c) * blocks.entry(n, b, c);
                }
                let want = if a == b { Dd::ONE } else { Dd::ZERO };
                worst = worst.max((sum - want).abs().to_f64());
            }
        }
        assert!(worst < 1e-28, "orthogonality n = {n}: {worst:e}");
    }
}

#[test]
fn rotations_against_s1() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rotation-blocks/fixtures/reference.json"
    ))
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut checked = 0;
    let mut worst = 0.0f64;
    for set in v["sets"].as_array().unwrap() {
        let beta: f64 = set["beta"].as_str().unwrap().parse().unwrap();
        let n = set["n"].as_u64().unwrap() as usize;
        if n > 400 {
            continue;
        }
        let (s, x) = Dd::from_f64(beta).sin_cos();
        let blocks = YBlocks::new(n, x, s);
        let is = set["i"].as_array().unwrap();
        let js = set["j"].as_array().unwrap();
        let vals = set["value"].as_array().unwrap();
        for ((i, j), val) in is.iter().zip(js).zip(vals) {
            let want: f64 = val.as_str().unwrap().parse().unwrap();
            let got = blocks
                .entry(
                    n,
                    i.as_i64().unwrap() as isize,
                    j.as_i64().unwrap() as isize,
                )
                .to_f64();
            worst = worst.max((got - want).abs());
            checked += 1;
        }
    }
    assert!(checked > 100, "{checked}");
    assert!(worst < 1e-14, "worst {worst:e} over {checked}");
}

#[test]
fn geometry() {
    let o = offsets();
    assert_eq!(o.len(), 316);
    assert_eq!(distances().len(), 15);
    assert_eq!(near_offsets().len(), 54);
    let c = [0.5, -1.0, 2.0];
    for (i, ch) in children(c, 1.0).iter().enumerate() {
        assert_eq!(child_index(c, *ch), i);
    }
    let mut rng = Rng::new(11);
    for placement in Placement::ALL {
        for w_leaf in [1.0, 0.5, 0.25] {
            let pts = points(placement, c, 1.0, w_leaf, 500, &mut rng);
            assert_eq!(pts.len(), 500);
            for x in &pts {
                assert!((0..3).all(|d| (x[d] - c[d]).abs() <= 0.5), "{placement:?}");
                if placement == Placement::Corners {
                    // Within w_leaf/8 of a corner of its leaf in every coordinate.
                    for d in 0..3 {
                        let rel = (x[d] - (c[d] - 0.5)) / w_leaf;
                        let frac = rel - rel.floor();
                        assert!(frac <= 0.125 + 1e-12 || frac >= 0.875 - 1e-12, "{frac}");
                    }
                }
            }
        }
    }
}

/// The formula against proposal §3.5's tables (kD = 500, ε = 10⁻⁶; level 2 against kD).
#[test]
fn formula() {
    let ka = |kd: f64, level: i32| kd * 3f64.sqrt() / 2.0 * 0.5f64.powi(level);
    let want = [
        (2, 144),
        (3, 82),
        (4, 50),
        (5, 33),
        (6, 25),
        (7, 23),
        (8, 22),
    ];
    for (level, p) in want {
        assert_eq!(formula_p(ka(500.0, level), 1e-6), p, "level {level}");
    }
    let top = [
        (50.0, [24, 30, 35]),
        (100.0, [37, 43, 48]),
        (200.0, [63, 69, 75]),
        (500.0, [135, 144, 151]),
        (1000.0, [251, 261, 271]),
        (3000.0, [699, 714, 728]),
    ];
    for (kd, ps) in top {
        for (eps, p) in [1e-4, 1e-6, 1e-8].iter().zip(ps) {
            assert_eq!(formula_p(ka(kd, 2), *eps), p, "kD {kd} eps {eps}");
        }
    }
}

/// Test 8: the search returns the smallest p from which on every scanned error passes,
/// also on a curve that dips below the threshold and rises again, and None when the last
/// point fails.
#[test]
fn search() {
    let curve = [
        (10, 1e-2),
        (11, 4e-4),
        (12, 6e-4),
        (13, 3e-4),
        (14, 1e-5),
        (15, 2e-6),
    ];
    assert_eq!(smallest_passing(&curve, 5e-4), Some(13));
    assert_eq!(smallest_passing(&curve, 1e-1), Some(10));
    assert_eq!(smallest_passing(&curve, 1e-5), Some(14));
    assert_eq!(smallest_passing(&curve, 1e-6), None);
    assert_eq!(smallest_passing(&[], 1.0), None);
    // Sparse samples: the first passing sample from which on all pass.
    let sparse = [(20, 1e-3), (24, 1e-7), (28, 1e-9)];
    assert_eq!(smallest_passing(&sparse, 1e-6), Some(24));
}
