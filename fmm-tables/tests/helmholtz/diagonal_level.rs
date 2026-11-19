//! The diagonal form's level buffers and driver (Helmholtz CH4.1b,
//! `nd_fmm_tables::helmholtz::diagonal::level`) and the reflection classes on the cube's group
//! (review 2026-11-19, C5).
//!
//! Error measure of the level call against the trusted path (the sum over each target's pairs
//! of `nd_fmm_ref::helmholtz::diagonal::m2l`): the largest absolute difference of a local's
//! value over the largest absolute value of the reference's locals, within 256 u. Everything
//! else bit for bit (`to_bits`): the stored conversions against `s2f` and `f2r`, and the level
//! call at every chunk size against the whole level in one chunk.

use std::collections::BTreeSet;
use std::f64::consts::PI;

use nd_fmm_math::Complex;
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_ref::helmholtz::diagonal::{
    self, Grid, SignatureWorkspace, Transfer, azimuth_count, class_of, classes, offsets,
};
use nd_fmm_tables::geometry::m2l_offsets;
use nd_fmm_tables::helmholtz::diagonal::{
    BLOCK_BOXES, ConversionScratch, Conversions, LevelBuffers, SIGNATURE_ALIGN, Signatures,
    m2l_level, scratch_shape, signed_permutation,
};
use nd_fmm_tables::symmetry;

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

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn signature_bits<T: Copy + Into<f64>>(values: &[Complex<T>]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|c| (c.re.into().to_bits(), c.im.into().to_bits()))
        .collect()
}

#[test]
fn every_box_starts_on_a_line() {
    for p in (0..=7).chain([12]) {
        let samples = (2 * p + 1) * azimuth_count(p);
        let check = |store_aligned: bool, stride: usize, line: usize| {
            assert!(store_aligned, "p = {p}: a box off its line");
            assert!(stride >= samples && stride.is_multiple_of(line));
            assert!(stride < samples + line);
        };
        let s = Signatures::<f64>::new(samples, 5);
        check(s.is_aligned(), s.stride(), SIGNATURE_ALIGN / 16);
        for b in 0..5 {
            assert!((s.signature(b).as_ptr() as usize).is_multiple_of(SIGNATURE_ALIGN));
            assert_eq!(s.signature(b).len(), samples);
        }
        assert_eq!(s.bytes(), Signatures::<f64>::bytes_for(samples, 5));
        let s = Signatures::<f32>::new(samples, 3);
        check(s.is_aligned(), s.stride(), SIGNATURE_ALIGN / 8);
        for b in 0..3 {
            assert!((s.signature(b).as_ptr() as usize).is_multiple_of(SIGNATURE_ALIGN));
        }
        assert_eq!(s.bytes(), Signatures::<f32>::bytes_for(samples, 3));
    }
    // f32 at p = 0: 4 samples on a line of 8, padded.
    assert_eq!(Signatures::<f32>::stride_for(4), 8);
    assert_eq!(Signatures::<f64>::stride_for(4), 4);
    assert_eq!(Signatures::<f64>::bytes_for(100, 0), 0);
}

#[test]
fn a_store_holds_any_count_up_to_its_capacity() {
    let mut s = Signatures::<f64>::new(12, 4);
    assert_eq!((s.boxes(), s.capacity()), (4, 4));
    s.signature_mut(3)[0] = Complex::new(1.0, 2.0);
    s.set_boxes(2);
    assert_eq!(s.boxes(), 2);
    assert_eq!(s.boxes_mut().count(), 2);
    s.set_boxes(4);
    assert_eq!(s.signature(3)[0], Complex::new(1.0, 2.0));
    s.zero();
    assert_eq!(s.signature(3)[0], Complex::new(0.0, 0.0));
}

#[test]
#[should_panic(expected = "exceed the capacity")]
fn refuses_more_boxes_than_its_capacity() {
    Signatures::<f64>::new(12, 4).set_boxes(5);
}

/// `s2f_into` and `f2r_from` give each box the bits of `s2f` and `f2r`, at any first box.
fn check_store_conversions(p: usize) {
    let conversions = Conversions::<f64>::new(p, 1.0);
    let mut scratch = ConversionScratch::new(&conversions);
    let (len, samples) = (conversions.expansion_len(), conversions.samples());
    let boxes = BLOCK_BOXES + 3;
    let multipoles = values(boxes * len, p as u64 + 3);
    let mut plain = vec![Complex::new(0.0, 0.0); boxes * samples];
    conversions.s2f(&multipoles, &mut plain, &mut scratch);
    let mut store = Signatures::new(samples, boxes + 2);
    conversions.s2f_into(&multipoles, &mut store, 2, &mut scratch);
    for b in 0..boxes {
        assert_eq!(
            signature_bits(store.signature(b + 2)),
            signature_bits(&plain[b * samples..(b + 1) * samples]),
            "p = {p}, box {b}"
        );
    }
    let start = values(boxes * len, p as u64 + 5);
    let (mut a, mut b) = (start.clone(), start);
    conversions.f2r(&plain, &mut a, &mut scratch);
    conversions.f2r_from(&store, 2, &mut b, &mut scratch);
    assert_eq!(bits(&a), bits(&b), "p = {p}");
}

#[test]
fn the_store_conversions_are_the_plain_ones() {
    for p in [3, 12, 17] {
        check_store_conversions(p);
    }
    // f32 (lines of 8 samples): p = 2 pads (5 × 12 = 60 samples, stride 64), p = 3 does not
    // (7 × 16 = 112).
    assert_eq!(Signatures::<f32>::stride_for(60), 64);
    for p in [2, 3] {
        let conversions = Conversions::<f32>::new(p, 1.0);
        let mut scratch = ConversionScratch::new(&conversions);
        let (len, samples) = (conversions.expansion_len(), conversions.samples());
        let multipoles: Vec<f32> = values(5 * len, 11).iter().map(|&v| v as f32).collect();
        let mut plain = vec![Complex::new(0.0, 0.0); 5 * samples];
        conversions.s2f(&multipoles, &mut plain, &mut scratch);
        let mut store = Signatures::new(samples, 5);
        conversions.s2f_into(&multipoles, &mut store, 0, &mut scratch);
        for b in 0..5 {
            assert_eq!(
                signature_bits(store.signature(b)),
                signature_bits(&plain[b * samples..(b + 1) * samples])
            );
        }
    }
}

#[test]
#[should_panic(expected = "boxes 3..6 of a store of 5")]
fn refuses_boxes_beyond_the_store() {
    let conversions = Conversions::<f64>::new(2, 1.0);
    let mut scratch = ConversionScratch::new(&conversions);
    let mut store = Signatures::new(conversions.samples(), 5);
    let multipoles = vec![0.0; 3 * conversions.expansion_len()];
    conversions.s2f_into(&multipoles, &mut store, 3, &mut scratch);
}

#[test]
fn the_sizes_without_building() {
    for p in [0, 1, 5, 12, 17, 31] {
        let conversions = Conversions::<f64>::new(p, 1.0);
        assert_eq!(
            Conversions::<f64>::bytes_for(p),
            conversions.bytes(),
            "p = {p}"
        );
        let scratch = ConversionScratch::new(&conversions);
        assert_eq!(
            ConversionScratch::<f64>::bytes_for(p),
            scratch.bytes(),
            "p = {p}"
        );
        let conversions = Conversions::<f32>::new(p, 1.0);
        assert_eq!(
            Conversions::<f32>::bytes_for(p),
            conversions.bytes(),
            "p = {p}"
        );
    }
}

/// A small level of unit boxes: the sources on a 6 × 6 × 2 block, the targets five of them,
/// each with a pair from every source at an M2L offset (max |oᵢ| ∈ {2, 3}), in the sources'
/// order; one target with no pair last.
struct Case {
    level: Level<f64>,
    centres: Vec<[f64; 3]>,
    sources: Vec<usize>,
    targets: Vec<usize>,
    rows: Vec<usize>,
    slots: Vec<usize>,
}

fn case() -> Case {
    let centres: Vec<[f64; 3]> = (0..2)
        .flat_map(|z| (0..6).flat_map(move |y| (0..6).map(move |x| [x as f64, y as f64, z as f64])))
        .collect();
    // The sources in an order other than the boxes'.
    let sources: Vec<usize> = (0..centres.len()).rev().collect();
    let targets = vec![14, 0, 35, 50, 21, 7];
    let (mut rows, mut slots) = (vec![0], Vec::new());
    for (i, &t) in targets.iter().enumerate() {
        if i + 1 < targets.len() {
            for (slot, &s) in sources.iter().enumerate() {
                let o: [f64; 3] = core::array::from_fn(|a| centres[t][a] - centres[s][a]);
                let reach = o.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                if reach == 2.0 || reach == 3.0 {
                    slots.push(slot);
                }
            }
        }
        rows.push(slots.len());
    }
    Case {
        level: Level::new(Complex::new(3.0, 0.0), 0.5), // s = 1
        centres,
        sources,
        targets,
        rows,
        slots,
    }
}

/// The level call on `case` at degree p with chunks of `chunk` targets, its transfer functions
/// from ref's `Transfer` (factor 4π ik, real k); returns the locals.
fn level_call(c: &Case, p: usize, chunk: usize, transfers: &[Transfer<f64>]) -> Vec<f64> {
    let conversions = Conversions::<f64>::new(p, c.level.scale());
    let len = conversions.expansion_len();
    let multipoles = values(c.centres.len() * len, p as u64 + 17);
    let mut locals = values(c.centres.len() * len, p as u64 + 19);
    let mut buffers = LevelBuffers::new(&conversions, c.sources.len(), chunk);
    assert_eq!(buffers.chunk(), chunk);
    let factor = Complex::new(0.0, 4.0 * PI * c.level.k().re);
    m2l_level(
        &conversions,
        &multipoles,
        &c.sources,
        &mut locals,
        &c.targets,
        &c.rows,
        &c.slots,
        &mut buffers,
        |_, k, signature, accumulator| {
            for ((a, &g), &t) in accumulator
                .iter_mut()
                .zip(signature)
                .zip(transfers[k].samples())
            {
                *a += t * g * factor;
            }
        },
    );
    locals
}

fn transfers(c: &Case, grid: &Grid<f64>) -> Vec<Transfer<f64>> {
    let frame = |b: usize| Frame::new(c.centres[b], 0.5);
    let mut out = Vec::new();
    for (i, &t) in c.targets.iter().enumerate() {
        for k in c.rows[i]..c.rows[i + 1] {
            let s = c.sources[c.slots[k]];
            out.push(Transfer::new(grid, &c.level, &frame(s), &frame(t)));
        }
    }
    out
}

/// The level call against the sum of ref's `m2l` over each target's pairs, and at every
/// chunk size bit for bit the whole level's.
fn check_level(p: usize) {
    let c = case();
    let grid = Grid::<f64>::new(p);
    let transfers = transfers(&c, &grid);
    let whole = level_call(&c, p, c.targets.len(), &transfers);

    // The trusted path.
    let len = 2 * (p + 1) * (p + 1);
    let multipoles = values(c.centres.len() * len, p as u64 + 17);
    let mut expected = values(c.centres.len() * len, p as u64 + 19);
    let mut ws = SignatureWorkspace::new(p);
    let frame = |b: usize| Frame::new(c.centres[b], 0.5);
    for (i, &t) in c.targets.iter().enumerate() {
        for (k, transfer) in transfers
            .iter()
            .enumerate()
            .take(c.rows[i + 1])
            .skip(c.rows[i])
        {
            let s = c.sources[c.slots[k]];
            diagonal::m2l(
                &grid,
                &c.level,
                &frame(s),
                &frame(t),
                transfer,
                &mut ws,
                &multipoles[s * len..(s + 1) * len],
                &mut expected[t * len..(t + 1) * len],
            );
        }
    }
    let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = whole
        .iter()
        .zip(&expected)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 256.0 * f64::EPSILON * scale,
        "p = {p}: {:.2e} u against the sum of ref's m2l",
        worst / scale / f64::EPSILON
    );
    // Boxes that are no target keep their bits.
    let start = values(c.centres.len() * len, p as u64 + 19);
    for b in (0..c.centres.len()).filter(|b| !c.targets.contains(b)) {
        assert_eq!(
            bits(&whole[b * len..(b + 1) * len]),
            bits(&start[b * len..(b + 1) * len])
        );
    }
    for chunk in [1, 2, 3, BLOCK_BOXES, 5] {
        assert_eq!(
            bits(&level_call(&c, p, chunk, &transfers)),
            bits(&whole),
            "p = {p}, chunks of {chunk}"
        );
    }
}

#[test]
fn the_level_call_against_refs_m2l_at_every_chunk_size() {
    let c = case();
    assert!(c.rows.windows(2).take(5).all(|w| w[1] > w[0]));
    assert_eq!(c.rows[5], c.rows[6], "the last target has no pair");
    check_level(6);
    check_level(12);
}

#[test]
#[ignore = "a larger degree: run in release"]
fn the_level_call_at_a_larger_degree() {
    check_level(17);
}

#[test]
fn the_scratch_under_a_budget() {
    let p = 12;
    let samples = (2 * p + 1) * azimuth_count(p);
    let stride_bytes = Signatures::<f64>::stride_for(samples) * 16;
    let fixed = |sources: usize, threads: usize| {
        Conversions::<f64>::bytes_for(p)
            + threads * (ConversionScratch::<f64>::bytes_for(p) + BLOCK_BOXES * 2 * 169 * 8)
            + Signatures::<f64>::bytes_for(samples, sources)
    };
    // Every target fits.
    let shape = scratch_shape::<f64>(p, 40, 30, 2, usize::MAX).unwrap();
    assert_eq!(shape.chunk, 30);
    assert_eq!(shape.chunks(30), 1);
    assert_eq!(
        shape.bytes(),
        fixed(40, 2) + Signatures::<f64>::bytes_for(samples, 30)
    );
    // Room for exactly 9 targets, and one byte less: 8.
    let budget = fixed(40, 2) + Signatures::<f64>::bytes_for(samples, 9);
    let shape = scratch_shape::<f64>(p, 40, 30, 2, budget).unwrap();
    assert_eq!((shape.chunk, shape.chunks(30)), (9, 4));
    assert!(shape.bytes() <= budget);
    let shape = scratch_shape::<f64>(p, 40, 30, 2, budget - 1).unwrap();
    assert_eq!(shape.chunk, 8);
    // The floor: BLOCK_BOXES targets, else none.
    let floor = fixed(40, 2) + Signatures::<f64>::bytes_for(samples, BLOCK_BOXES);
    assert_eq!(
        scratch_shape::<f64>(p, 40, 30, 2, floor).unwrap().chunk,
        BLOCK_BOXES
    );
    assert_eq!(scratch_shape::<f64>(p, 40, 30, 2, floor - 1), None);
    // Fewer targets than the floor: all of them; none: no accumulators.
    let shape = scratch_shape::<f64>(p, 40, 2, 1, fixed(40, 1) + 2 * stride_bytes + 64).unwrap();
    assert_eq!(shape.chunk, 2);
    let shape = scratch_shape::<f64>(p, 40, 0, 1, fixed(40, 1)).unwrap();
    assert_eq!(
        (shape.chunk, shape.accumulators, shape.chunks(0)),
        (0, 0, 0)
    );
    // The buffers of a shape hold what it counts for one thread.
    let shape = scratch_shape::<f64>(p, 40, 30, 1, budget).unwrap();
    let buffers = LevelBuffers::new(&Conversions::<f64>::new(p, 1.0), 40, shape.chunk);
    assert_eq!(shape.bytes(), shape.tables + buffers.bytes());
}

#[test]
fn the_offsets_are_one_set_and_the_classes_nest() {
    let ours: BTreeSet<[i64; 3]> = offsets().into_iter().map(|o| o.map(i64::from)).collect();
    let tables: BTreeSet<[i64; 3]> = m2l_offsets().into_iter().collect();
    assert_eq!(ours.len(), 316);
    assert_eq!(ours, tables);

    // Each offset onto its representative by the element of O_h.
    for o in offsets() {
        let (representative, s) = class_of(o);
        let g = signed_permutation(s);
        assert_eq!(
            g.apply(o.map(i64::from)),
            representative.map(i64::from),
            "{o:?}"
        );
    }
    // Each of the 34 classes lies in one of the 16 (the 16 elements are a subgroup of O_h).
    let index_of = |o: [i64; 3]| m2l_offsets().iter().position(|&d| d == o).unwrap();
    let mut seen = BTreeSet::new();
    for class in classes() {
        let tables_classes: BTreeSet<usize> = offsets()
            .into_iter()
            .filter(|&o| class_of(o).0 == class.representative)
            .map(|o| symmetry::class_of(index_of(o.map(i64::from))).0)
            .collect();
        assert_eq!(tables_classes.len(), 1, "{:?}", class.representative);
        seen.extend(tables_classes);
    }
    assert_eq!(seen.len(), 16);
}
