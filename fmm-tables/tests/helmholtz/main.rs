//! Acceptance tests of the Helmholtz RCR tables (Helmholtz CH3.1a,
//! `nd_fmm_tables::helmholtz`): M2M and L2L on the 8 octants and M2L on the 316 offsets
//! against `nd_fmm_ref::helmholtz::rcr` at the same frames, for real and complex k, one and
//! two planes, rectangular degrees, and above the projection threshold of CONVENTIONS
//! §3.15; the stored and per-call blocks, several vectors and the key counts.
//!
//! Error measure: the FMM-weighted error of `nd-fmm-ref`'s translation tests,
//! ‖tables − rcr‖_w / ‖rcr‖_w, within [`BOUND`] times max(1, |k̂| a) of the level
//! ([`conditioning`]). The tables take their angles and distances
//! from exact integer classes and `rcr` from the floating-point shift, so the two agree to
//! rounding, not bit for bit; the real shifts of real k also drop the tables' imaginary
//! parts, which are rounding. Stored against per-call blocks and several vectors against
//! one: bit for bit (`==` per value, so a −0 equals a +0). The f32 test names its own.

use nd_fmm_math::radial::{bessel_j, hankel_h};
use nd_fmm_math::{Complex, Layout};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, rcr};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, OCTANT_COUNT, m2l_offsets, octant_direction};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};

/// A deterministic input of `planes` planes at degree p whose degree-n coefficients are of
/// size about 2⁻ⁿ, with a seed.
fn input(p: usize, planes: usize, seed: u64) -> Vec<f64> {
    let len = Layout::new(p).len();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..planes * len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let n = ((i % len) as f64).sqrt() as i32;
            u * 0.5f64.powi(n)
        })
        .collect()
}

/// Two planes from one (Im = 0), as `rcr` takes a real expansion.
fn widened(real: &[f64]) -> Vec<f64> {
    let mut out = real.to_vec();
    out.extend(core::iter::repeat_n(0.0, real.len()));
    out
}

fn assert_same(got: &[f64], expected: &[f64], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: lengths");
    for (i, (a, b)) in got.iter().zip(expected).enumerate() {
        assert!(a == b, "{what}: value {i} is {a:e}, rcr gives {b:e}");
    }
}

/// The wavenumbers of the tests, k̂₀ = k r₀: real and complex, from |k̂| a of about 0.01
/// on level 4 to 30 on level 2.
const K_ROOTS: [(f64, f64); 5] = [
    (0.05, 0.0),
    (2.0, 0.0),
    (40.0, 0.0),
    (8.0, 1.5),
    (40.0, 4.0),
];

/// M2L on every offset of `level` against `rcr` (two planes; and one widened plane where
/// k is real, the real multipoles of real charges): the largest FMM-weighted error.
fn check_m2l(
    tables: &HelmholtzTables<f64>,
    level: usize,
    offsets: impl Iterator<Item = usize>,
) -> f64 {
    let p = tables.degree(level);
    let len = Layout::new(p).len();
    let k_level = Level::new(tables.k_hat(level), 1.0);
    let w = weights(&k_level, false, p);
    let from = Frame::new([0.0; 3], 1.0);
    let mut ws = Workspace::new(p);
    let mut scratch = tables.scratch();
    let mut worst = 0.0f64;
    for d in offsets {
        let to = Frame::new(m2l_offsets()[d].map(|x| 2.0 * x as f64), 1.0);
        let table = rcr::m2l_table(p, p, &k_level, &from, &to);
        let plane_counts: &[usize] = if tables.real_shifts() { &[2, 1] } else { &[2] };
        for &planes in plane_counts {
            let x = input(p, planes, d as u64);
            let mut got = vec![0.5; 2 * len];
            tables.m2l(level, d, &x, planes, &mut got, 1, &mut scratch);
            let mut expected = vec![0.5; 2 * len];
            let x = if planes == 2 { x } else { widened(&x) };
            rcr::m2l(
                p,
                p,
                &k_level,
                &from,
                &to,
                &table,
                &mut ws,
                &x,
                &mut expected,
            );
            worst = worst.max(compare(&got, &expected, 0.5, &w));
        }
    }
    worst
}

/// M2M and L2L of every octant between `level` and `level` + 1 against `rcr`, in the
/// child's units; L2L against ½ `rcr`'s (the change of units, module docs of
/// `nd_fmm_tables::helmholtz`). Complex k: bit for bit. Real k: the real shifts against `rcr`'s complex
/// step in the FMM-weighted measure, returning the largest error ([`compare`]).
fn check_shifts(tables: &HelmholtzTables<f64>, level: usize) -> f64 {
    let (p_parent, p_child) = (tables.degree(level), tables.degree(level + 1));
    let (len_parent, len_child) = (Layout::new(p_parent).len(), Layout::new(p_child).len());
    let k = tables.k_hat(level + 1);
    let child_level = Level::new(k, 1.0);
    let parent_level = Level::new(k, 2.0);
    let parent_frame = Frame::new([0.0; 3], 2.0);
    let m2m_table = rcr::m2m_table(p_child, p_parent, &child_level);
    let l2l_table = rcr::l2l_table(p_parent, p_child, &child_level);
    let mut ws = Workspace::new(p_parent.max(p_child));
    let mut scratch = tables.scratch();
    let mut worst = 0.0f64;
    let plane_counts: &[usize] = if tables.real_shifts() { &[2, 1] } else { &[2] };
    for o in 0..OCTANT_COUNT {
        let child_frame = Frame::new(octant_direction(o).map(|s| s as f64), 1.0);
        for &planes in plane_counts {
            let x = input(p_child, planes, o as u64);
            let mut got = vec![0.25; planes * len_parent];
            tables.m2m(level, o, &x, &mut got, planes, 1, &mut scratch);
            let mut expected = vec![0.25; 2 * len_parent];
            rcr::m2m(
                p_child,
                p_parent,
                &child_level,
                &parent_level,
                &child_frame,
                &parent_frame,
                &m2m_table,
                &mut ws,
                &if planes == 2 { x } else { widened(&x) },
                &mut expected,
            );
            worst = worst.max(compare(
                &got,
                &expected,
                0.25,
                &weights(&parent_level, true, p_parent),
            ));

            let x = input(p_parent, planes, 8 + o as u64);
            // From zero, so that ½ rcr's output is exact.
            let mut got = vec![0.0; planes * len_child];
            tables.l2l(level + 1, o, &x, &mut got, planes, 1, &mut scratch);
            let mut expected = vec![0.0; 2 * len_child];
            rcr::l2l(
                p_parent,
                p_child,
                &parent_level,
                &child_level,
                &parent_frame,
                &child_frame,
                &l2l_table,
                &mut ws,
                &if planes == 2 { x } else { widened(&x) },
                &mut expected,
            );
            let half: Vec<f64> = expected.iter().map(|v| 0.5 * v).collect();
            worst = worst.max(compare(
                &got,
                &half,
                0.0,
                &weights(&child_level, false, p_child),
            ));
        }
    }
    worst
}

/// The outputs started at `initial`: the FMM-weighted error of the reference tests,
/// ‖got − expected‖_w / ‖expected‖_w over both planes, a one-plane result's imaginary plane
/// taken as zero (for real k `rcr`'s imaginary parts there are the tables' rounding).
fn compare(got: &[f64], expected: &[f64], initial: f64, weights: &[f64]) -> f64 {
    let expected: Vec<f64> = expected.iter().map(|v| v - initial).collect();
    let mut got: Vec<f64> = got.iter().map(|v| v - initial).collect();
    got.resize(expected.len(), 0.0);
    let diff: Vec<f64> = got.iter().zip(&expected).map(|(a, b)| a - b).collect();
    weighted_norm(weights, &diff) / weighted_norm(weights, &expected)
}

/// The FMM weights wₙ = √((2n + 1)/(4π)) |fₙ(kR)|, n ≤ p, of an expansion on `level`: a
/// multipole at the nearest points of its V list, R = (4 − √3) r, fₙ = 4π ik ĥₙ(kR)
/// e^{β − Im(k)R}; a local on its sphere, R = a, fₙ = ĵₙ(kR) e^{Im(k)R − β} (as the
/// reference tests).
fn weights(level: &Level<f64>, multipole: bool, p: usize) -> Vec<f64> {
    let k = level.k();
    let mut f = vec![Complex::new(0.0, 0.0); p + 1];
    let e = if multipole {
        let r = (4.0 - 3.0_f64.sqrt()) * level.half_width();
        hankel_h(k * r, level.scale(), &mut f, None);
        (level.damping() - k.im * r).exp() * 4.0 * core::f64::consts::PI * k.norm()
    } else {
        let r = level.circumradius();
        bessel_j(k * r, level.scale(), &mut f, None);
        (k.im * r - level.damping()).exp()
    };
    f.iter()
        .enumerate()
        .map(|(n, x)| x.norm() * e * ((2 * n + 1) as f64 / (4.0 * core::f64::consts::PI)).sqrt())
        .collect()
}

/// ‖C‖_w = √(Σₙ wₙ² ‖Cₙ‖²) over two planes, from terms scaled by the largest.
fn weighted_norm(weights: &[f64], c: &[f64]) -> f64 {
    let len = c.len() / 2;
    let terms: Vec<f64> = (0..weights.len())
        .flat_map(|n| (n * n..(n + 1) * (n + 1)).flat_map(move |i| [(n, i), (n, len + i)]))
        .map(|(n, j)| weights[n] * c[j])
        .collect();
    let largest = terms.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if largest == 0.0 {
        return 0.0;
    }
    largest
        * terms
            .iter()
            .map(|v| (v / largest).powi(2))
            .sum::<f64>()
            .sqrt()
}

/// The bound of every comparison with `rcr`, in the FMM-weighted measure (module
/// documentation), times [`conditioning`]: rounding, with room for other libms.
const BOUND: f64 = 2e-13;

/// max(1, |k̂| a) of `level` (a = √3 in its units): a relative change u of a shift's length
/// or angles, which the tables and `rcr` round differently, moves the phases of the coaxial
/// step by about |k̂| |t| u.
fn conditioning(tables: &HelmholtzTables<f64>, level: usize) -> f64 {
    (tables.k_hat(level).norm() * 3f64.sqrt()).max(1.0)
}

#[test]
fn m2l_on_every_offset_matches_rcr() {
    for (re, im) in K_ROOTS {
        // Levels 2–4; |k̂| a from 0.05 · √3 / 16 to 40 · √3 / 4.
        let tables = HelmholtzTables::<f64>::build(
            Complex::new(re, im),
            &[0, 0, 7, 6, 5],
            DEFAULT_BLOCK_BUDGET,
        );
        let worst = (2..5)
            .map(|level| {
                check_m2l(&tables, level, 0..M2L_OFFSET_COUNT) / conditioning(&tables, level)
            })
            .fold(0.0, f64::max);
        println!("k̂₀ = {re} + {im}i: M2L against rcr, worst {worst:.2e} per unit of conditioning");
        assert!(worst <= BOUND, "k̂₀ = {re} + {im}i: {worst:e}");
    }
}

#[test]
fn m2m_and_l2l_on_every_octant_match_rcr() {
    for (re, im) in K_ROOTS {
        // Rectangular degrees between every pair of levels.
        let tables = HelmholtzTables::<f64>::build(
            Complex::new(re, im),
            &[2, 6, 9, 4, 7],
            DEFAULT_BLOCK_BUDGET,
        );
        let worst = (0..4)
            .map(|level| check_shifts(&tables, level) / conditioning(&tables, level))
            .fold(0.0, f64::max);
        println!(
            "k̂₀ = {re} + {im}i: M2M and L2L against rcr, worst {worst:.2e} per unit of conditioning"
        );
        assert!(worst <= BOUND, "k̂₀ = {re} + {im}i: {worst:e}");
    }
}

/// Above the projection threshold Re(k) a > 40 of §3.15 (M2L on level 2 at Re(k̂) a ≈ 52;
/// M2M and L2L of the pair (2, 3) at Re(k̂) 2a ≈ 52). The degrees do not resolve the
/// fields; the comparison is of the implementations.
#[test]
fn above_the_projection_threshold() {
    for (re, im) in [(120.0, 0.0), (120.0, 6.0)] {
        let tables = HelmholtzTables::<f64>::build(
            Complex::new(re, im),
            &[0, 0, 12, 10],
            DEFAULT_BLOCK_BUDGET,
        );
        let c = conditioning(&tables, 2);
        let m2l = check_m2l(&tables, 2, 0..M2L_OFFSET_COUNT) / c;
        let shifts = check_shifts(&tables, 2) / c;
        println!("k̂₀ = {re} + {im}i: M2L {m2l:.2e}, M2M and L2L {shifts:.2e}");
        assert!(m2l <= BOUND && shifts <= BOUND, "k̂₀ = {re} + {im}i");
    }
}

#[test]
fn stored_and_per_call_blocks_give_the_same_bits() {
    let k = Complex::new(6.0, 0.5);
    let stored = HelmholtzTables::<f64>::build(k, &[0, 0, 9, 8], DEFAULT_BLOCK_BUDGET);
    let per_call = HelmholtzTables::<f64>::build(k, &[0, 0, 9, 8], 0);
    assert!(stored.m2l_blocks_stored(2) && stored.m2l_blocks_stored(3));
    assert!(!per_call.m2l_blocks_stored(2) && !per_call.m2l_blocks_stored(3));
    // A budget that holds the finest level only.
    let one = stored.m2l_polar_angles()
        * nd_fmm_math::orthonormal::rotation::compact_blocks_len(8)
        * size_of::<f64>();
    let finest = HelmholtzTables::<f64>::build(k, &[0, 0, 9, 8], one);
    assert!(!finest.m2l_blocks_stored(2) && finest.m2l_blocks_stored(3));
    assert!(stored.bytes() > per_call.bytes());
    let (mut a, mut b) = (stored.scratch(), per_call.scratch());
    for level in 2..4 {
        let len = 2 * Layout::new(stored.degree(level)).len();
        for d in 0..M2L_OFFSET_COUNT {
            let x = input(stored.degree(level), 2, d as u64);
            let (mut y, mut z) = (vec![0.0; len], vec![0.0; len]);
            stored.m2l(level, d, &x, 2, &mut y, 1, &mut a);
            per_call.m2l(level, d, &x, 2, &mut z, 1, &mut b);
            assert_same(&y, &z, &format!("level {level}, d = {d}"));
        }
    }
}

#[test]
fn each_vector_is_the_one_vector_application() {
    for (re, im) in [(3.0, 0.0), (3.0, 0.7)] {
        let tables = HelmholtzTables::<f64>::build(
            Complex::new(re, im),
            &[0, 0, 6, 8],
            DEFAULT_BLOCK_BUDGET,
        );
        let mut scratch = tables.scratch();
        let (p2, p3) = (tables.degree(2), tables.degree(3));
        let (l2, l3) = (Layout::new(p2).len(), Layout::new(p3).len());
        let planes: &[usize] = if tables.real_shifts() { &[1, 2] } else { &[2] };
        for &planes in planes {
            let vectors = 3;
            // M2M from level 3 into 2, L2L from 2 into 3, M2L on 3.
            let x3 = input(p3, planes * vectors, 1);
            let x2 = input(p2, planes * vectors, 2);
            let (mut m2m, mut l2l, mut m2l) = (
                vec![0.0; vectors * planes * l2],
                vec![0.0; vectors * planes * l3],
                vec![0.0; vectors * 2 * l3],
            );
            tables.m2m(2, 5, &x3, &mut m2m, planes, vectors, &mut scratch);
            tables.l2l(3, 6, &x2, &mut l2l, planes, vectors, &mut scratch);
            tables.m2l(3, 200, &x3, planes, &mut m2l, vectors, &mut scratch);
            for v in 0..vectors {
                let one =
                    |x: &[f64], len: usize| x[v * planes * len..(v + 1) * planes * len].to_vec();
                let mut y = vec![0.0; planes * l2];
                tables.m2m(2, 5, &one(&x3, l3), &mut y, planes, 1, &mut scratch);
                assert_same(
                    &m2m[v * planes * l2..(v + 1) * planes * l2],
                    &y,
                    "M2M vector",
                );
                let mut y = vec![0.0; planes * l3];
                tables.l2l(3, 6, &one(&x2, l2), &mut y, planes, 1, &mut scratch);
                assert_same(
                    &l2l[v * planes * l3..(v + 1) * planes * l3],
                    &y,
                    "L2L vector",
                );
                let mut y = vec![0.0; 2 * l3];
                tables.m2l(3, 200, &one(&x3, l3), planes, &mut y, 1, &mut scratch);
                assert_same(&m2l[v * 2 * l3..(v + 1) * 2 * l3], &y, "M2L vector");
            }
        }
    }
}

#[test]
fn classes_sizes_and_shape() {
    let tables = HelmholtzTables::<f64>::build(
        Complex::new(1.0, 0.0),
        &[0, 0, 4, 4, 4],
        DEFAULT_BLOCK_BUDGET,
    );
    println!(
        "{} distances, {} polar angles, {} bytes",
        tables.m2l_distances(),
        tables.m2l_polar_angles(),
        tables.bytes()
    );
    // The 15 values of |d|² and the 49 polar angles of the geometry, θ = 0 and π included.
    assert_eq!(tables.m2l_distances(), 15);
    assert_eq!(tables.m2l_polar_angles(), 49);
    assert_eq!(tables.nlevels(), 5);
    assert!(tables.real_shifts());
    assert_eq!(tables.k_hat(3), Complex::new(0.125, 0.0));
    assert!(!HelmholtzTables::<f64>::build(Complex::new(1.0, 0.1), &[3], 0).real_shifts());
}

/// f32 tables are the f64 tables rounded: the f32 M2L within 1e-5 of the f64 one,
/// relative to the largest output value of each degree.
#[test]
fn f32_tables() {
    let k = Complex::new(5.0, 0.3);
    let t64 = HelmholtzTables::<f64>::build(k, &[0, 0, 6], DEFAULT_BLOCK_BUDGET);
    let t32 = HelmholtzTables::<f32>::build(k, &[0, 0, 6], DEFAULT_BLOCK_BUDGET);
    let (mut s64, mut s32) = (t64.scratch(), t32.scratch());
    let len = Layout::new(6).len();
    for d in (0..M2L_OFFSET_COUNT).step_by(11) {
        let x = input(6, 2, d as u64);
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let (mut y64, mut y32) = (vec![0.0; 2 * len], vec![0.0f32; 2 * len]);
        t64.m2l(2, d, &x, 2, &mut y64, 1, &mut s64);
        t32.m2l(2, d, &x32, 2, &mut y32, 1, &mut s32);
        for plane in 0..2 {
            for (_, range) in Layout::new(6).degrees() {
                let scale = range
                    .clone()
                    .fold(0.0f64, |m, i| m.max(y64[plane * len + i].abs()));
                for i in range {
                    let diff = (y32[plane * len + i] as f64 - y64[plane * len + i]).abs();
                    assert!(diff <= 1e-5 * scale, "d = {d}: {diff:e} of {scale:e}");
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "Yukawa")]
fn yukawa_is_refused() {
    let _ = HelmholtzTables::<f64>::build(Complex::new(0.0, 1.0), &[0, 0, 4], 0);
}

#[test]
#[should_panic(expected = "planes")]
fn a_complex_step_refuses_one_output_plane() {
    let tables = HelmholtzTables::<f64>::build(Complex::new(1.0, 0.5), &[2, 2], 0);
    let mut scratch = tables.scratch();
    let x = vec![0.0; 9];
    let mut y = vec![0.0; 9];
    tables.m2m(0, 0, &x, &mut y, 1, 1, &mut scratch);
}

/// Below `rcr`'s limit p log₁₀(1/s_P) ≲ 300 the power factors of M2M and L2L overflow:
/// `build` refuses the degrees rather than build non-finite tables (s_P ≈ 3.5e-6 here, so
/// p = 60 asks for about 10³²⁷).
#[test]
#[should_panic(expected = "too high for the level's power scale")]
fn power_factors_beyond_the_f64_range_are_refused() {
    let _ = HelmholtzTables::<f64>::build(Complex::new(2e-6, 0.0), &[60, 60], 0);
}
