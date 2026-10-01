//! `relative_frame` (CONVENTIONS §3.13, "Relative frames"): bit for bit equal, in f64
//! and f32, to (c_s − c_t) / r_t and r_s / r_t computed in f64 in the dyadic domain,
//! where every step is exact; for every pair the U, V, W and X lists can produce and
//! for arbitrary pairs on levels 0–16. Also the self frame, the reversal rule, the
//! composition rule and the canonical frames of §3.12, all exactly.

use nd_fmm_exec::geometry::{integer_centre, relative_frame};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::Frame;
use nd_fmm_tables::geometry::{m2l_offset_index, octant_direction};
use nd_octree::{MortonKey, morton};

use crate::common::{DEEPEST, SplitMix64, dyadic_frame, sample_keys};

/// Asserts that `relative_frame::<T>(s, t)` equals the dyadic f64 reference bit for bit,
/// and that the reference is representable in T.
fn assert_exact_in<T: RealScalar>(s: MortonKey, t: MortonKey) {
    let (centre, radius) = dyadic_frame(s, t);
    let frame = relative_frame::<T>(s, t);
    for (&got, want) in frame.centre.iter().zip(centre) {
        assert_eq!(T::from_f64(want).to_f64(), want, "ĉ not representable");
        assert_eq!(got.to_f64().to_bits(), want.to_bits(), "ĉ({s}|{t})");
    }
    assert_eq!(
        frame.radius.to_f64().to_bits(),
        radius.to_bits(),
        "r̂({s}|{t})"
    );
}

/// Asserts exactness in f64 and f32 and returns the largest |C_L(s) − C_L(t)|.
fn check_pair(s: MortonKey, t: MortonKey) -> i64 {
    assert_exact_in::<f64>(s, t);
    assert_exact_in::<f32>(s, t);
    let reference = morton::level(s).max(morton::level(t));
    let (cs, ct) = (integer_centre(s, reference), integer_centre(t, reference));
    (0..3).map(|k| (cs[k] - ct[k]).abs()).max().unwrap()
}

/// The box at `index` + `offset` on `level`, if it exists.
fn shifted(level: usize, index: [usize; 3], offset: [i64; 3]) -> Option<MortonKey> {
    let n = 1i64 << level;
    let shifted: [i64; 3] = core::array::from_fn(|k| index[k] as i64 + offset[k]);
    shifted
        .iter()
        .all(|&i| (0..n).contains(&i))
        .then(|| morton::from_index_and_level(shifted.map(|i| i as usize), level))
}

fn cube(range: core::ops::RangeInclusive<i64>) -> impl Iterator<Item = [i64; 3]> {
    let r = range.clone();
    range.flat_map(move |x| {
        let r2 = r.clone();
        r.clone()
            .flat_map(move |y| r2.clone().map(move |z| [x, y, z]))
    })
}

#[test]
fn list_pairs_are_exact_in_f64_and_f32() {
    // Error measure: bit-for-bit equality.
    let mut rng = SplitMix64::new(0x7307);
    let (mut pairs, mut largest) = (0usize, 0i64);
    let (mut largest_same, mut largest_family, mut largest_adjacent) = (0i64, 0i64, 0i64);
    for level in 0..=DEEPEST {
        for t in sample_keys(&mut rng, level, 3) {
            let (_, index) = morton::decode(t);
            // U and V lists: the same level, offsets in {−3..3}³.
            for d in cube(-3..=3) {
                if let Some(s) = shifted(level, index, d) {
                    largest_same = largest_same.max(check_pair(s, t));
                    pairs += 1;
                }
            }
            if level == DEEPEST {
                continue;
            }
            // M2M and L2L: parent and child, both ways.
            for child in morton::children(t).unwrap() {
                largest_family = largest_family.max(check_pair(child, t).max(check_pair(t, child)));
                pairs += 2;
            }
            // W and X lists: the children of the neighbours of t, one level finer,
            // both ways: index 2 (i_t + d) + b with d ∈ {−1..1}³ \ {0} and b ∈ {0, 1}³.
            for d in cube(-1..=1).filter(|d| *d != [0, 0, 0]) {
                let Some(neighbour) = shifted(level, index, d) else {
                    continue;
                };
                for s in morton::children(neighbour).unwrap() {
                    largest_adjacent = largest_adjacent.max(check_pair(s, t).max(check_pair(t, s)));
                    pairs += 2;
                }
            }
        }
    }
    largest = largest
        .max(largest_same)
        .max(largest_family)
        .max(largest_adjacent);
    println!(
        "list pairs: {pairs} exact in f64 and f32; largest |C_L(s) - C_L(t)| = {largest} \
         (same level {largest_same}, parent-child {largest_family}, one level apart {largest_adjacent})"
    );
    assert_eq!(
        largest, 6,
        "the V list at |d| = 3 has the largest numerator, 2 · 3"
    );
}

#[test]
fn arbitrary_pairs_are_exact_in_f64_and_f32() {
    // Error measure: bit-for-bit equality.
    let mut rng = SplitMix64::new(0x7308);
    let (mut pairs, mut largest) = (0usize, 0i64);
    for level_s in 0..=DEEPEST {
        for level_t in 0..=DEEPEST {
            let (last_s, last_t) = ((1 << level_s) - 1, (1 << level_t) - 1);
            let mut keys = vec![
                (
                    morton::from_index_and_level([0, 0, 0], level_s),
                    morton::from_index_and_level([last_t; 3], level_t),
                ),
                (
                    morton::from_index_and_level([last_s; 3], level_s),
                    morton::from_index_and_level([0, 0, 0], level_t),
                ),
            ];
            keys.extend((0..8).map(|_| (rng.key(level_s), rng.key(level_t))));
            for (s, t) in keys {
                largest = largest.max(check_pair(s, t));
                pairs += 1;
            }
        }
    }
    println!("arbitrary pairs: {pairs} on levels 0-16 x 0-16, exact; largest |N| = {largest}");
    assert_eq!(
        largest,
        (1 << 17) - 2,
        "opposite corners of level 16 attain 2^17 - 2"
    );
}

fn self_reversal_and_composition<T: RealScalar + core::fmt::Debug>(rng: &mut SplitMix64) -> usize {
    let zero = T::zero();
    let mut checked = 0;
    for _ in 0..400 {
        let levels: [usize; 3] = core::array::from_fn(|_| rng.below(DEEPEST + 1));
        let [s, b, t] = levels.map(|l| rng.key(l));
        // ĉ(t|t) = 0, r̂(t|t) = 1.
        assert_eq!(relative_frame::<T>(t, t), Frame::new([zero; 3], T::one()));
        // Reversal: ĉ(t|s) = −ĉ(s|t) / r̂(s|t), r̂(t|s) = 1 / r̂(s|t).
        let (st, ts) = (relative_frame::<T>(s, t), relative_frame::<T>(t, s));
        assert_eq!(ts.centre, st.centre.map(|c| -c / st.radius), "{s}, {t}");
        assert_eq!(ts.radius, T::one() / st.radius);
        // Composition: ĉ(s|t) = ĉ(s|b) r̂(b|t) + ĉ(b|t), r̂(s|t) = r̂(s|b) r̂(b|t).
        let (sb, bt) = (relative_frame::<T>(s, b), relative_frame::<T>(b, t));
        let composed: [T; 3] = core::array::from_fn(|k| sb.centre[k] * bt.radius + bt.centre[k]);
        assert_eq!(composed, st.centre, "{s} | {b} | {t}");
        assert_eq!(sb.radius * bt.radius, st.radius);
        checked += 1;
    }
    checked
}

#[test]
fn self_frame_reversal_and_composition_are_exact() {
    // Error measure: exact equality, in f64 and f32.
    let mut rng = SplitMix64::new(0x7309);
    let n64 = self_reversal_and_composition::<f64>(&mut rng);
    let n32 = self_reversal_and_composition::<f32>(&mut rng);
    println!("self frame, reversal and composition exact for {n64} (f64) and {n32} (f32) triples");
}

#[test]
fn child_and_v_list_frames_are_the_canonical_frames() {
    // Error measure: exact equality with (½ s_o, ½) and (2d, 1) of CONVENTIONS §3.12.
    let mut rng = SplitMix64::new(0x730a);
    for level in 0..DEEPEST {
        for parent in sample_keys(&mut rng, level, 3) {
            for child in morton::children(parent).unwrap() {
                let o = morton::child_index(child);
                let frame = relative_frame::<f32>(child, parent);
                assert_eq!(frame.centre, octant_direction(o).map(|s| s as f32 / 2.0));
                assert_eq!(frame.radius, 0.5);
            }
        }
    }
    for level in 2..=DEEPEST {
        for source in sample_keys(&mut rng, level, 3) {
            let (_, index) = morton::decode(source);
            for d in cube(-3..=3).filter(|d| m2l_offset_index(*d).is_some()) {
                if let Some(target) = shifted(level, index, d) {
                    let frame = relative_frame::<f32>(target, source);
                    assert_eq!(frame.centre, d.map(|t| 2.0 * t as f32));
                    assert_eq!(frame.radius, 1.0);
                }
            }
        }
    }
    println!("child-parent frames are (s_o / 2, 1/2), V-list frames (2d, 1), exactly");
}
