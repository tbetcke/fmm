//! S → F and F → R as products over a batch of boxes (Helmholtz CH4.1a,
//! `nd_fmm_tables::helmholtz::diagonal`) against the reference
//! `nd_fmm_ref::helmholtz::diagonal::{s2f, f2r}`, the round trip, and a box's bits in every
//! batch.
//!
//! Error measure: the largest absolute difference of a value over the largest absolute value
//! of the reference's result, within 256 u of the precision (measured at most 16 u in f64, at
//! p = 245 in `helmholtz_conversions`); the round trip
//! f2r(s2f(M̂)) = s²ⁿ M̂ in the same measure. A box converted in a batch against the box alone:
//! bit for bit (`to_bits`).

use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_ref::helmholtz::diagonal::{self, Grid, Workspace};
use nd_fmm_tables::helmholtz::diagonal::{BLOCK_BOXES, ConversionScratch, Conversions};

/// Deterministic values in [−1, 1).
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

fn signatures_of(count: usize, seed: u64) -> Vec<Complex<f64>> {
    let v = values(2 * count, seed);
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| Complex::new(re, im))
        .collect()
}

/// A level with power scale 1 (k a ≥ 1) or below it.
fn level(below_one: bool) -> Level<f64> {
    if below_one {
        Level::new(Complex::new(0.7, 0.0), 0.5) // s = 0.7 √3 / 2 ≈ 0.61
    } else {
        Level::new(Complex::new(20.0, 0.0), 0.5)
    }
}

fn largest(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::max)
}

/// S → F of `boxes` random multipoles against the reference, box by box; the error measure
/// of the module documentation.
fn check_s2f(p: usize, s_below_one: bool, boxes: usize) {
    let level = level(s_below_one);
    let conversions = Conversions::<f64>::new(p, level.scale());
    let mut scratch = ConversionScratch::new(&conversions);
    let (len, samples) = (conversions.expansion_len(), conversions.samples());
    let multipoles = values(boxes * len, p as u64 + 1);
    let mut signatures = vec![Complex::new(f64::NAN, f64::NAN); boxes * samples];
    conversions.s2f(&multipoles, &mut signatures, &mut scratch);

    let grid = Grid::<f64>::new(p);
    let mut ws = Workspace::new(p);
    let mut expected = vec![Complex::new(0.0, 0.0); samples];
    for b in 0..boxes {
        diagonal::s2f(
            &grid,
            &level,
            &multipoles[b * len..(b + 1) * len],
            &mut ws,
            &mut expected,
        );
        let scale = largest(expected.iter().map(|z| z.norm()));
        let error = largest(
            signatures[b * samples..(b + 1) * samples]
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).norm()),
        );
        let bound = 256.0 * f64::EPSILON * scale;
        assert!(
            error <= bound,
            "S → F, p {p}, s {}: box {b} errs {error:e} > {bound:e}",
            level.scale()
        );
    }
}

/// F → R of `boxes` random signatures (every bandwidth of the grid) added to random locals
/// against the reference.
fn check_f2r(p: usize, s_below_one: bool, boxes: usize) {
    let level = level(s_below_one);
    let conversions = Conversions::<f64>::new(p, level.scale());
    let mut scratch = ConversionScratch::new(&conversions);
    let (len, samples) = (conversions.expansion_len(), conversions.samples());
    let signatures = signatures_of(boxes * samples, 100 + p as u64);
    let start = values(boxes * len, 200 + p as u64);
    let mut locals = start.clone();
    conversions.f2r(&signatures, &mut locals, &mut scratch);

    let grid = Grid::<f64>::new(p);
    let mut ws = Workspace::new(p);
    for b in 0..boxes {
        let mut expected = start[b * len..(b + 1) * len].to_vec();
        diagonal::f2r(
            &grid,
            &level,
            &signatures[b * samples..(b + 1) * samples],
            &mut ws,
            &mut expected,
        );
        let scale = largest(expected.iter().map(|v| v.abs()));
        let error = largest(
            locals[b * len..(b + 1) * len]
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).abs()),
        );
        let bound = 256.0 * f64::EPSILON * scale;
        assert!(
            error <= bound,
            "F → R, p {p}, s {}: box {b} errs {error:e} > {bound:e}",
            level.scale()
        );
    }
}

#[test]
fn s2f_against_the_reference() {
    for p in [0, 1, 2, 3, 4, 5, 7, 12, 24] {
        for below_one in [false, true] {
            check_s2f(p, below_one, BLOCK_BOXES + 2);
        }
    }
}

#[test]
fn f2r_against_the_reference() {
    for p in [0, 1, 2, 3, 4, 5, 7, 12, 24] {
        for below_one in [false, true] {
            check_f2r(p, below_one, BLOCK_BOXES + 2);
        }
    }
}

/// The degrees of W1's high-frequency levels at kD = 500 (66–135) against the reference
/// (well under a second in release on the M6).
#[test]
#[ignore = "large degrees: run in release"]
fn against_the_reference_at_large_degrees() {
    for p in [41, 72, 131] {
        check_s2f(p, false, 2);
        check_f2r(p, false, 2);
    }
}

/// F → R of S → F is sⁿ sⁿ times the identity: the grid integrates the products of the
/// harmonics of degrees ≤ p exactly (the grid's bandwidth).
#[test]
fn the_round_trip() {
    for p in [0, 1, 2, 3, 6, 11, 24] {
        for below_one in [false, true] {
            let level = level(below_one);
            let s = level.scale();
            let conversions = Conversions::<f64>::new(p, s);
            let mut scratch = ConversionScratch::new(&conversions);
            let (len, samples) = (conversions.expansion_len(), conversions.samples());
            let boxes = 3;
            let multipoles = values(boxes * len, 7 + p as u64);
            let mut signatures = vec![Complex::new(0.0, 0.0); boxes * samples];
            conversions.s2f(&multipoles, &mut signatures, &mut scratch);
            let mut locals = vec![0.0; boxes * len];
            conversions.f2r(&signatures, &mut locals, &mut scratch);
            let plane = (p + 1) * (p + 1);
            for (at, (&l, &m)) in locals.iter().zip(&multipoles).enumerate() {
                let n = ((at % len % plane) as f64).sqrt().floor() as i32;
                let expected = m * s.powi(2 * n);
                assert!(
                    (l - expected).abs() <= 1e-13,
                    "p {p}, s {s}: value {at} (n {n}) is {l}, expected {expected}"
                );
            }
        }
    }
}

/// Every box of a batch, at every position and batch size, is the box converted alone, bit
/// for bit, both ways; at p = 17 the rings make two blocks, so F → R's chains also cross a
/// block.
#[test]
fn a_box_has_the_same_bits_in_every_batch() {
    let p = 17;
    let level = level(false);
    let conversions = Conversions::<f64>::new(p, level.scale());
    let mut scratch = ConversionScratch::new(&conversions);
    let (len, samples) = (conversions.expansion_len(), conversions.samples());
    let most = 2 * BLOCK_BOXES + 3;
    let multipoles = values(most * len, 31);
    let incoming = signatures_of(most * samples, 32);

    let mut alone_s = vec![Complex::new(0.0, 0.0); most * samples];
    let mut alone_l = vec![0.0; most * len];
    for b in 0..most {
        conversions.s2f(
            &multipoles[b * len..(b + 1) * len],
            &mut alone_s[b * samples..(b + 1) * samples],
            &mut scratch,
        );
        conversions.f2r(
            &incoming[b * samples..(b + 1) * samples],
            &mut alone_l[b * len..(b + 1) * len],
            &mut scratch,
        );
    }
    let bits = |z: &Complex<f64>| (z.re.to_bits(), z.im.to_bits());
    for size in [1, 2, 5, BLOCK_BOXES, BLOCK_BOXES + 1, most] {
        for first in [0, most - size, (most - size) / 2] {
            let range = first..first + size;
            let mut s = vec![Complex::new(0.0, 0.0); size * samples];
            conversions.s2f(
                &multipoles[range.start * len..range.end * len],
                &mut s,
                &mut scratch,
            );
            assert!(
                s.iter()
                    .zip(&alone_s[range.start * samples..range.end * samples])
                    .all(|(a, b)| bits(a) == bits(b)),
                "S → F: batch of {size} from box {first}"
            );
            let mut l = vec![0.0; size * len];
            conversions.f2r(
                &incoming[range.start * samples..range.end * samples],
                &mut l,
                &mut scratch,
            );
            assert!(
                l.iter()
                    .zip(&alone_l[range.start * len..range.end * len])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "F → R: batch of {size} from box {first}"
            );
        }
    }
}

/// f32 tables and transforms against the f64 reference, within 256 u₃₂.
#[test]
fn f32_against_the_reference() {
    for p in [3, 12, 30] {
        let level = level(false);
        let conversions = Conversions::<f32>::new(p, level.scale());
        let mut scratch = ConversionScratch::new(&conversions);
        let (len, samples) = (conversions.expansion_len(), conversions.samples());
        let multipoles = values(len, 41);
        let narrow: Vec<f32> = multipoles.iter().map(|&v| v as f32).collect();
        let mut signature = vec![Complex::new(0.0f32, 0.0); samples];
        conversions.s2f(&narrow, &mut signature, &mut scratch);
        let grid = Grid::<f64>::new(p);
        let mut ws = Workspace::new(p);
        let mut expected = vec![Complex::new(0.0, 0.0); samples];
        let rounded: Vec<f64> = narrow.iter().map(|&v| v as f64).collect();
        diagonal::s2f(&grid, &level, &rounded, &mut ws, &mut expected);
        let scale = largest(expected.iter().map(|z| z.norm()));
        let bound = 256.0 * f32::EPSILON as f64 * scale;
        for (a, e) in signature.iter().zip(&expected) {
            let a = Complex::new(a.re as f64, a.im as f64);
            assert!((a - e).norm() <= bound, "S → F f32, p {p}");
        }

        let mut local = vec![0.0f32; len];
        conversions.f2r(&signature, &mut local, &mut scratch);
        let wide: Vec<Complex<f64>> = signature
            .iter()
            .map(|z| Complex::new(z.re as f64, z.im as f64))
            .collect();
        let mut expected = vec![0.0; len];
        diagonal::f2r(&grid, &level, &wide, &mut ws, &mut expected);
        let scale = largest(expected.iter().map(|v| v.abs()));
        let bound = 256.0 * f32::EPSILON as f64 * scale;
        for (&a, &e) in local.iter().zip(&expected) {
            assert!((a as f64 - e).abs() <= bound, "F → R f32, p {p}");
        }
    }
}

/// The panics on wrong lengths.
#[test]
#[should_panic(expected = "not a multiple of 2(p + 1)²")]
fn refuses_a_partial_expansion() {
    let conversions = Conversions::<f64>::new(4, 1.0);
    let mut scratch = ConversionScratch::new(&conversions);
    let mut signatures = vec![Complex::new(0.0, 0.0); conversions.samples()];
    conversions.s2f(&[0.0; 7], &mut signatures, &mut scratch);
}

#[test]
#[should_panic(expected = "signature samples")]
fn refuses_too_few_signatures() {
    let conversions = Conversions::<f64>::new(4, 1.0);
    let mut scratch = ConversionScratch::new(&conversions);
    let multipoles = vec![0.0; 2 * conversions.expansion_len()];
    let mut signatures = vec![Complex::new(0.0, 0.0); conversions.samples()];
    conversions.s2f(&multipoles, &mut signatures, &mut scratch);
}

#[test]
#[should_panic(expected = "the scratch has degree")]
fn refuses_another_degrees_scratch() {
    let conversions = Conversions::<f64>::new(4, 1.0);
    let mut scratch = ConversionScratch::new(&Conversions::<f64>::new(5, 1.0));
    let mut locals = vec![0.0; conversions.expansion_len()];
    let signatures = vec![Complex::new(0.0, 0.0); conversions.samples()];
    conversions.f2r(&signatures, &mut locals, &mut scratch);
}
