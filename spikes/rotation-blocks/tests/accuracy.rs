//! The acceptance tests of CH0.1 (docs/helmholtz/tasks/CH0.1.md, "Tests that prove it").
//! Tests 1, 2, 5 and 6 are ignored in debug builds' default runs; run them all with
//! `cargo test -p nd-fmm-spike-rotation-blocks --release -- --include-ignored`.

use nd_fmm_math::rotation::{block_range, blocks_len, euler_zyz};
use nd_fmm_math::{Layout, harmonics};
use nd_fmm_spike_rotation_blocks::conjugated::{conjugated_blocks, n_ratios};
use nd_fmm_spike_rotation_blocks::entry;
use nd_fmm_spike_rotation_blocks::gd::{
    Scratch, compact_blocks_len, compact_offset, euler_blocks, expand, matmul, y_blocks,
};

/// Compact y-rotation blocks to degree p for the angle β.
fn y(p: usize, beta: f64) -> Vec<f64> {
    let (s, x) = beta.sin_cos();
    let mut out = vec![f64::NAN; compact_blocks_len(p)];
    y_blocks(p, x, s, &mut out, &mut Scratch::new(p));
    out
}

/// Full (2n + 1)² block of degree n from compact blocks.
fn full(n: usize, compact: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; (2 * n + 1) * (2 * n + 1)];
    expand(
        n,
        &compact[compact_offset(n)..compact_offset(n + 1)],
        &mut out,
    );
    out
}

/// The octree's polar angles and two generic ones, as in tools/reference.py.
fn angles() -> Vec<f64> {
    let m2m = (1.0f64 / 3.0f64.sqrt()).acos();
    vec![
        m2m,
        std::f64::consts::PI - m2m,
        (1.0f64 / 3.0).atan(),
        std::f64::consts::FRAC_PI_2,
        1.234,
        1e-3,
    ]
}

/// Test 1: every entry of fixtures/reference.json, absolute error ≤ 1e-12.
#[test]
#[ignore = "release mode: block sets to degree 1000"]
fn agrees_with_the_arbitrary_precision_reference() {
    let text = include_str!("../fixtures/reference.json");
    let doc: serde_json::Value = serde_json::from_str(text).unwrap();
    let mut worst = 0.0f64;
    let mut cache: Option<(f64, Vec<f64>)> = None;
    for set in doc["sets"].as_array().unwrap() {
        let n = set["n"].as_u64().unwrap() as usize;
        let beta: f64 = set["beta"].as_str().unwrap().parse().unwrap();
        // One block set per angle at the largest degree serves every n.
        if cache.as_ref().is_none_or(|(b, _)| *b != beta) {
            cache = Some((beta, y(1000, beta)));
        }
        let block = full(n, &cache.as_ref().unwrap().1);
        let (is, js, vs) = (
            set["i"].as_array().unwrap(),
            set["j"].as_array().unwrap(),
            set["value"].as_array().unwrap(),
        );
        let mut set_worst = 0.0f64;
        for ((i, j), v) in is.iter().zip(js).zip(vs) {
            let (i, j) = (i.as_i64().unwrap() as isize, j.as_i64().unwrap() as isize);
            let expected: f64 = v.as_str().unwrap().parse().unwrap();
            set_worst = set_worst.max((block[entry(n, i, j)] - expected).abs());
        }
        println!(
            "n = {n:4} {:16} max |error| {set_worst:.2e}",
            set["angle"].as_str().unwrap()
        );
        worst = worst.max(set_worst);
    }
    assert!(worst <= 1e-12, "worst {worst:e}");
}

/// ‖O Oᵀ − I‖_max of a full block of width w.
fn orthogonality_error(w: usize, o: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..w {
        for j in 0..w {
            let dot: f64 = o[i * w..(i + 1) * w]
                .iter()
                .zip(&o[j * w..(j + 1) * w])
                .map(|(a, b)| a * b)
                .sum();
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    worst
}

/// Test 2: O Oᵀ = I to 1e-12 at every n ≤ 100 and at selected degrees to 1000; the probe
/// ‖Oᵀ(O v) − v‖_∞ at every degree to 1000.
#[test]
#[ignore = "release mode: O Oᵀ at degrees to 1000"]
fn blocks_are_orthogonal_to_degree_1000() {
    let selected = [150, 200, 300, 500, 700, 1000];
    for beta in angles() {
        let compact = y(1000, beta);
        let mut worst = 0.0f64;
        let mut probe = 0.0f64;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for n in 0..=1000 {
            let w = 2 * n + 1;
            let o = full(n, &compact);
            if n <= 100 || selected.contains(&n) {
                worst = worst.max(orthogonality_error(w, &o));
            }
            let v: Vec<f64> = (0..w)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect();
            let ov: Vec<f64> = (0..w)
                .map(|i| (0..w).map(|j| o[i * w + j] * v[j]).sum())
                .collect();
            for (j, vj) in v.iter().enumerate() {
                let back: f64 = (0..w).map(|i| o[i * w + j] * ov[i]).sum();
                probe = probe.max((back - vj).abs());
            }
        }
        println!("beta = {beta:.6}: max |O Oᵀ − I| {worst:.2e}, probe {probe:.2e}");
        assert!(
            worst <= 1e-12 && probe <= 1e-12,
            "beta = {beta}: {worst:e}, {probe:e}"
        );
    }
}

/// Test 3: the GD blocks of z-y-z rotations against N Dⁿ N⁻¹ from
/// `nd_fmm_math::rotation::blocks`, p ≤ 30, absolute 1e-13.
#[test]
fn agrees_with_conjugated_nd_fmm_math_blocks_to_degree_30() {
    let p = 30;
    let mut triples: Vec<[f64; 3]> = angles().into_iter().map(|b| [0.0, b, 0.0]).collect();
    triples.extend([
        [0.3, 1.1, -0.4],
        [-1.2, 0.8, 2.5],
        [2.9, 2.7, 0.6],
        [0.0, 0.0, 1.0],
        [1.0, std::f64::consts::PI, -2.0],
    ]);
    let mut worst = 0.0f64;
    for [a, b, g] in triples {
        let ours = euler_blocks(p, a, b, g);
        let theirs = conjugated_blocks(p, &euler_zyz(a, b, g));
        assert_eq!(ours.len(), blocks_len(p));
        for (x, y) in ours.iter().zip(&theirs) {
            worst = worst.max((x - y).abs());
        }
    }
    println!("max |GD − N D N⁻¹| at p ≤ 30: {worst:.2e}");
    assert!(worst <= 1e-13, "worst {worst:e}");
}

/// Test 4: Yₙ(Qx) = Oⁿ(Q) Yₙ(x), Yₙ = N Rₙ, at random points, n ≤ 20, relative 1e-13.
#[test]
fn satisfies_the_definition_to_degree_20() {
    let p = 20;
    let layout = Layout::new(p);
    let points = [[0.2, -0.5, 0.7], [-0.9, 0.1, 0.3], [0.4, 0.6, -0.55]];
    for [a, b, g] in [[0.3, 1.1, -0.4], [-1.2, 0.8, 2.5], [0.0, 1.234, 0.0]] {
        let q = euler_zyz(a, b, g);
        let o = euler_blocks(p, a, b, g);
        for x in points {
            let qx = q.map(|row: [f64; 3]| row[0] * x[0] + row[1] * x[1] + row[2] * x[2]);
            let (mut r, mut rq) = (vec![0.0; layout.len()], vec![0.0; layout.len()]);
            harmonics::regular(p, x, &mut r);
            harmonics::regular(p, qx, &mut rq);
            for n in 0..=p {
                // N as ratios to N₀ = n!, so that Yₙ / n! is compared.
                let w = n_ratios(n);
                let ni = n as isize;
                let yn = |src: &[f64]| -> Vec<f64> {
                    (-ni..=ni)
                        .map(|m| w[m.unsigned_abs()] * src[layout.degree(n)][(m + ni) as usize])
                        .collect()
                };
                let (yx, yqx) = (yn(&r), yn(&rq));
                let scale = yqx
                    .iter()
                    .fold(0.0f64, |acc, v| acc.max(v.abs()))
                    .max(f64::MIN_POSITIVE);
                let block = &o[block_range(n)];
                for (i, expected) in yqx.iter().enumerate() {
                    let row = &block[i * (2 * n + 1)..(i + 1) * (2 * n + 1)];
                    let value: f64 = row.iter().zip(&yx).map(|(a, b)| a * b).sum();
                    assert!(
                        (value - expected).abs() <= 1e-13 * scale,
                        "n = {n}, slot {i}: {value} vs {expected}"
                    );
                }
            }
        }
    }
}

/// Test 5: Oⁿ(π − β) = (−1)ⁿ E Oⁿ(β) E F with E = diag((−1)^|m|) and F = diag(+1 on the
/// cos slots, −1 on the sin slots): entry (i, j) gains (−1)^(n + |i| + |j|) on the cos
/// half and −(−1)^(n + |i| + |j|) on the sin half. Checked to round-off (1e-12) with
/// (cos, sin) = (−cos β, sin β) as the second input, at every degree to 1000; the
/// entries that differ in their bits are counted and printed.
#[test]
#[ignore = "release mode: block sets to degree 1000"]
fn supplementary_angle_is_a_sign_change() {
    let p = 1000;
    for beta in angles() {
        let (s, x) = beta.sin_cos();
        let (mut a, mut b) = (
            vec![0.0; compact_blocks_len(p)],
            vec![0.0; compact_blocks_len(p)],
        );
        let mut scratch = Scratch::new(p);
        y_blocks(p, x, s, &mut a, &mut scratch);
        y_blocks(p, -x, s, &mut b, &mut scratch);
        let mut mismatches = 0usize;
        let mut worst = 0.0f64;
        for n in 0..=p {
            let base = compact_offset(n);
            for i in 0..=n {
                for j in 0..=n {
                    let sign = if (n + i + j) % 2 == 0 { 1.0 } else { -1.0 };
                    let (u, v) = (a[base + i * (n + 1) + j], b[base + i * (n + 1) + j]);
                    mismatches += usize::from(v != sign * u);
                    worst = worst.max((v - sign * u).abs());
                }
            }
            let sin = base + (n + 1) * (n + 1);
            for i in 1..=n {
                for j in 1..=n {
                    let sign = if (n + i + j) % 2 == 0 { -1.0 } else { 1.0 };
                    let k = sin + (i - 1) * n + (j - 1);
                    mismatches += usize::from(b[k] != sign * a[k]);
                    worst = worst.max((b[k] - sign * a[k]).abs());
                }
            }
        }
        println!(
            "beta = {beta:.6}: {mismatches} entries not bit for bit, max |difference| {worst:.2e}"
        );
        assert!(worst <= 1e-12, "beta = {beta}: {worst:e}");
    }
}

/// Test 6: Oⁿ(R_y(β₁)) Oⁿ(R_y(β₂)) = Oⁿ(R_y(β₁ + β₂)) at n = 500, 1e-12.
#[test]
#[ignore = "release mode: a product of width 1001"]
fn group_property_at_degree_500() {
    let (n, b1, b2) = (500, 0.7, 1.1);
    let w = 2 * n + 1;
    let (a, b, c) = (
        full(n, &y(n, b1)),
        full(n, &y(n, b2)),
        full(n, &y(n, b1 + b2)),
    );
    let mut ab = vec![0.0; w * w];
    matmul(w, &a, &b, &mut ab);
    let worst = ab
        .iter()
        .zip(&c)
        .fold(0.0f64, |acc, (x, y)| acc.max((x - y).abs()));
    println!("group property at n = 500: {worst:.2e}");
    assert!(worst <= 1e-12, "worst {worst:e}");
}
