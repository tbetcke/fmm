//! Box geometry of an octree level: child octants, V-list offsets and the canonical
//! frames at which the tables are built (CONVENTIONS §3.12).
//!
//! The octant order restates that of `nd_octree::morton::child_index`, and the offset
//! order that of `nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS`; this crate is
//! MPI-free and cannot import them. Phase 3 (C3.1) tests that they agree.
//!
//! Tables are looked up by the integer child index o and offset d, never by a
//! floating-point shift: on deep levels a centre difference c′ − c formed in floating
//! point carries a relative error of about ε |c| / r, while the tables are exact in
//! their geometry (CONVENTIONS §3.12, "Canonical frames and level independence").
//!
//! ```
//! use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offset_index, m2l_offsets, octant_direction};
//!
//! assert_eq!(octant_direction(4), [1, -1, -1]);
//! let offsets = m2l_offsets();
//! assert_eq!(offsets.len(), M2L_OFFSET_COUNT);
//! assert_eq!(offsets[0], [-3, -3, -3]);
//! assert_eq!(m2l_offset_index([3, 3, 3]), Some(315));
//! assert_eq!(m2l_offset_index([1, 0, -1]), None); // adjacent, not in the V list
//! ```

use nd_fmm_ref::Frame;

/// Number of child octants of a box, and of matrices in an M2M or L2L table.
pub const OCTANT_COUNT: usize = 8;

/// Number of V-list offsets, 7³ − 3³ = 316, and of matrices in an M2L table
/// (CONVENTIONS §3.12, "V-list offsets").
pub const M2L_OFFSET_COUNT: usize = 316;

/// Returns the sign vector s_o = (2x − 1, 2y − 1, 2z − 1) of child index
/// o = 4x + 2y + z (CONVENTIONS §3.12, "Child octants").
///
/// The child centre is c_parent + r_child s_o with r_child = r_parent / 2. So o = 0 is
/// the child at the lower corner, s₀ = (−1, −1, −1), and s₄ = (+1, −1, −1).
///
/// # Panics
///
/// If `o >= 8`.
pub const fn octant_direction(o: usize) -> [i64; 3] {
    assert!(o < OCTANT_COUNT, "child index must be below 8");
    [bit_sign(o, 2), bit_sign(o, 1), bit_sign(o, 0)]
}

/// +1 if bit `bit` of `o` is set, −1 otherwise.
const fn bit_sign(o: usize, bit: usize) -> i64 {
    if (o >> bit) & 1 == 1 { 1 } else { -1 }
}

/// Returns the 316 V-list offsets 𝒟 = {−3..3}³ \ {−1..1}³ in the lexicographic order
/// of (d_x, d_y, d_z) (CONVENTIONS §3.12, "V-list offsets").
///
/// The position of d in this array is its table index, [`m2l_offset_index`]. An offset
/// is d = index(target) − index(source), in the index units of the common level.
pub const fn m2l_offsets() -> [[i64; 3]; M2L_OFFSET_COUNT] {
    let mut offsets = [[0; 3]; M2L_OFFSET_COUNT];
    let mut next = 0;
    let mut x = -3;
    while x <= 3 {
        let mut y = -3;
        while y <= 3 {
            let mut z = -3;
            while z <= 3 {
                if !(adjacent(x) && adjacent(y) && adjacent(z)) {
                    offsets[next] = [x, y, z];
                    next += 1;
                }
                z += 1;
            }
            y += 1;
        }
        x += 1;
    }
    assert!(next == M2L_OFFSET_COUNT);
    offsets
}

/// Returns the table index of the V-list offset `d`, its position in
/// [`m2l_offsets`], or `None` if d is not in 𝒟 = {−3..3}³ \ {−1..1}³.
///
/// Uses the closed form of CONVENTIONS §3.12, "V-list offsets":
///
/// index(d) = 49(d_x + 3) + 7(d_y + 3) + (d_z + 3) − 9 κ(d_x)
///            − [|d_x| ≤ 1] (3 κ(d_y) + [|d_y| ≤ 1] κ(d_z)),
///
/// with κ(t) = min(max(t + 1, 0), 3) the number of t′ ∈ {−1, 0, 1} below t, and
/// [·] = 1 if the condition holds and 0 otherwise.
pub const fn m2l_offset_index(d: [i64; 3]) -> Option<usize> {
    let [x, y, z] = d;
    if x < -3 || x > 3 || y < -3 || y > 3 || z < -3 || z > 3 {
        return None;
    }
    if adjacent(x) && adjacent(y) && adjacent(z) {
        return None;
    }
    let index = 49 * (x + 3) + 7 * (y + 3) + (z + 3)
        - 9 * kappa(x)
        - iverson(adjacent(x)) * (3 * kappa(y) + iverson(adjacent(y)) * kappa(z));
    Some(index as usize)
}

/// Whether |t| ≤ 1.
const fn adjacent(t: i64) -> bool {
    -1 <= t && t <= 1
}

/// κ(t) = min(max(t + 1, 0), 3), the number of t′ ∈ {−1, 0, 1} with t′ < t; for
/// |t| ≤ 3, where t + 1 cannot overflow.
const fn kappa(t: i64) -> i64 {
    let k = t + 1;
    if k < 0 {
        0
    } else if k > 3 {
        3
    } else {
        k
    }
}

/// The Iverson bracket [b]: 1 if `b` holds and 0 otherwise.
const fn iverson(b: bool) -> i64 {
    if b { 1 } else { 0 }
}

/// Returns the canonical (input, output) frames of the M2M table of child index `o`:
/// the child (½ s_o, ½) and the parent (0, 1) (CONVENTIONS §3.12, "Canonical frames
/// and level independence"). The M2M of §3.11 then has b = ½ s_o and ρ = ½.
///
/// # Panics
///
/// If `o >= 8`.
pub fn m2m_frames(o: usize) -> (Frame<f64>, Frame<f64>) {
    (child_frame(o), unit_frame())
}

/// Returns the canonical (input, output) frames of the L2L table of child index `o`:
/// the parent (0, 1) and the child (½ s_o, ½) (CONVENTIONS §3.12, "Canonical frames
/// and level independence"). The L2L of §3.11 then has t = ½ s_o and σ = ½.
///
/// # Panics
///
/// If `o >= 8`.
pub fn l2l_frames(o: usize) -> (Frame<f64>, Frame<f64>) {
    (unit_frame(), child_frame(o))
}

/// Returns the canonical (input, output) frames of the M2L table with table index
/// `index`, the offset d = [`m2l_offsets`]`()[index]`: the source (0, 1) and the target
/// (2d, 1) (CONVENTIONS §3.12, "Canonical frames and level independence"). The M2L of
/// §3.11 then has b = 2d and σ = 1.
///
/// # Panics
///
/// If `index >= 316`.
pub fn m2l_frames(index: usize) -> (Frame<f64>, Frame<f64>) {
    assert!(
        index < M2L_OFFSET_COUNT,
        "M2L offset index {index} out of range for {M2L_OFFSET_COUNT} offsets"
    );
    let d = m2l_offsets()[index];
    (unit_frame(), Frame::new(d.map(|t| 2.0 * t as f64), 1.0))
}

/// The parent frame (0, 1).
fn unit_frame() -> Frame<f64> {
    Frame::new([0.0; 3], 1.0)
}

/// The child frame (½ s_o, ½) of the parent (0, 1).
fn child_frame(o: usize) -> Frame<f64> {
    Frame::new(octant_direction(o).map(|s| 0.5 * s as f64), 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octant_direction_matches_the_bits_of_o() {
        // Error measure: exact equality (integers).
        for o in 0..OCTANT_COUNT {
            let bits = [(o >> 2) & 1, (o >> 1) & 1, o & 1];
            let want = bits.map(|b| 2 * b as i64 - 1);
            assert_eq!(octant_direction(o), want, "o = {o}");
            // The index is recovered from the signs as o = 4x + 2y + z.
            let [x, y, z] = octant_direction(o).map(|s| ((s + 1) / 2) as usize);
            assert_eq!(4 * x + 2 * y + z, o);
        }
        assert_eq!(octant_direction(0), [-1, -1, -1]);
        assert_eq!(octant_direction(4), [1, -1, -1]);
        assert_eq!(octant_direction(7), [1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "child index must be below 8")]
    fn octant_direction_rejects_index_8() {
        let _ = octant_direction(8);
    }

    #[test]
    fn offsets_are_316_lexicographic_and_not_adjacent() {
        // Error measure: exact equality (integers).
        let offsets = m2l_offsets();
        assert_eq!(offsets.len(), 316);
        assert_eq!(M2L_OFFSET_COUNT, 7 * 7 * 7 - 3 * 3 * 3);
        for d in offsets {
            assert!(d.iter().all(|t| t.abs() <= 3), "{d:?} outside {{-3..3}}^3");
            assert!(d.iter().any(|t| t.abs() > 1), "{d:?} is adjacent");
        }
        for pair in offsets.windows(2) {
            assert!(pair[0] < pair[1], "{:?} !< {:?}", pair[0], pair[1]);
        }
        // Strictly increasing and 316 in number: every element of 𝒟 appears once.
        let all = (-3..=3)
            .flat_map(|x| (-3..=3).flat_map(move |y| (-3..=3).map(move |z| [x, y, z])))
            .filter(|d: &[i64; 3]| d.iter().any(|t| t.abs() > 1))
            .collect::<Vec<_>>();
        assert_eq!(all, offsets);
        assert_eq!(offsets[0], [-3, -3, -3]);
        assert_eq!(offsets[315], [3, 3, 3]);
    }

    #[test]
    fn offset_index_round_trips() {
        // Error measure: exact equality (integers).
        for (index, d) in m2l_offsets().into_iter().enumerate() {
            assert_eq!(m2l_offset_index(d), Some(index), "d = {d:?}");
        }
        // Points outside 𝒟: the adjacent ones and those beyond ±3.
        for x in -5i64..=5 {
            for y in -5i64..=5 {
                for z in -5i64..=5 {
                    let d = [x, y, z];
                    let in_d = d.iter().all(|t| t.abs() <= 3) && d.iter().any(|t| t.abs() > 1);
                    assert_eq!(m2l_offset_index(d).is_some(), in_d, "d = {d:?}");
                }
            }
        }
        for d in [[i64::MIN, 0, 0], [0, i64::MAX, 0], [0, 0, 4]] {
            assert_eq!(m2l_offset_index(d), None);
        }
    }

    #[test]
    fn canonical_frames_are_those_of_section_3_12() {
        // Error measure: exact equality (all values are dyadic).
        for o in 0..OCTANT_COUNT {
            let s = octant_direction(o).map(|t| t as f64);
            let child = Frame::new(s.map(|t| t / 2.0), 0.5);
            let parent = Frame::new([0.0; 3], 1.0);
            assert_eq!(m2m_frames(o), (child, parent), "o = {o}");
            assert_eq!(l2l_frames(o), (parent, child), "o = {o}");
            // The §3.11 parameters: b = (c − c′)/r′ = ½ s_o, ρ = ½ for M2M, and
            // t = (c′ − c)/r = ½ s_o, σ = ½ for L2L.
            let (from, to) = m2m_frames(o);
            assert_eq!(to.scaled(from.centre), s.map(|t| t / 2.0));
            assert_eq!(from.radius / to.radius, 0.5);
            let (from, to) = l2l_frames(o);
            assert_eq!(from.scaled(to.centre), s.map(|t| t / 2.0));
            assert_eq!(to.radius / from.radius, 0.5);
        }
        for (index, d) in m2l_offsets().into_iter().enumerate() {
            let (source, target) = m2l_frames(index);
            assert_eq!(source, Frame::new([0.0; 3], 1.0));
            assert_eq!(target, Frame::new(d.map(|t| 2.0 * t as f64), 1.0));
            // b = (c′ − c)/r = 2d and σ = 1, with 4 ≤ |b| ≤ 6√3 (CONVENTIONS §3.9).
            let b = source.scaled(target.centre);
            let norm2: f64 = b.iter().map(|t| t * t).sum();
            assert!((16.0..=108.0).contains(&norm2), "|b|² = {norm2}");
        }
    }

    #[test]
    #[should_panic(expected = "M2L offset index 316 out of range for 316 offsets")]
    fn m2l_frames_rejects_index_316() {
        let _ = m2l_frames(M2L_OFFSET_COUNT);
    }
}
