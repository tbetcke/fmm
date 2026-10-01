//! Box geometry from Morton keys: the cubic domain, box centres and radii
//! (CONVENTIONS §3.12), integer centres, exact relative frames and leaf-scaled
//! coordinates (CONVENTIONS §3.13).
//!
//! Every operator of this crate takes its geometry from here. The domain enters only
//! twice: when a point is loaded into a leaf ([`leaf_coordinates`]) and when output is
//! scaled back by the target radius ([`radius`]). Between boxes, an operator uses
//! [`relative_frame`], which is computed from the integer key indices alone, does not
//! depend on the domain and is exact in f32 and f64. No function here forms a shift as
//! the difference of two floating-point centres: such a shift carries a relative error
//! of about ε |c| / r_l, up to 6.5e4 ε at level 16 (CONVENTIONS §3.12), while the
//! relative frames carry none.
//!
//! The module is MPI-free.
//!
//! ```
//! use nd_fmm_exec::geometry::{Domain, contains, leaf_coordinates, radius, relative_frame};
//! use nd_octree::{PhysicalBox, morton};
//!
//! let domain = Domain::new(&PhysicalBox::new([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])).unwrap();
//! let t = morton::from_index_and_level([2, 2, 2], 2);
//! let s = morton::from_index_and_level([3, 2, 2], 2);
//!
//! // Box s seen from box t: one box to the right, same size.
//! let frame = relative_frame::<f32>(s, t);
//! assert_eq!((frame.centre, frame.radius), ([2.0, 0.0, 0.0], 1.0));
//!
//! // The centre of box t, x = (0.25, 0.25, 0.25), has the leaf-scaled coordinates
//! // (0, 0, 0); r_t = 2 / 2³.
//! let x = [0.25, 0.25, 0.25];
//! assert!(contains(t, x, &domain));
//! assert_eq!(leaf_coordinates::<f64>(x, t, &domain), [0.0, 0.0, 0.0]);
//! assert_eq!(radius(2, &domain), 0.25);
//! ```

use nd_fmm_math::RealScalar;
use nd_fmm_ref::Frame;
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::{MortonKey, PhysicalBox, morton};
use thiserror::Error;

/// The deepest level of a Morton key, 16 (CONVENTIONS §3.12).
const DEEPEST: usize = DEEPEST_LEVEL as usize;

/// Tolerance on the sides of a cubic domain, in units of ε₆₄ (M + w), with ε₆₄ = 2⁻⁵³,
/// M the largest corner coordinate in magnitude and w the largest side.
///
/// [`Domain::new`] accepts a box whose three sides w_k = fl(max_k − min_k) differ
/// pairwise by at most `SIDE_TOLERANCE` · ε₆₄ (M + w).
///
/// *Why a tolerance.* `nd_octree::octree::compute_global_bounding_box` builds a cube
/// about the centre m_k of the points with one half-side h: min_k = fl(m_k − h) and
/// max_k = fl(m_k + h). Each corner rounds by at most ε₆₄ M, and the side
/// fl(max_k − min_k) rounds once more by at most ε₆₄ w, so to first order
/// |w_k − 2h| ≤ ε₆₄ (2M + w) and |w_j − w_k| ≤ ε₆₄ (4M + 2w). A box built by hand as
/// [a, fl(a + w)] differs by less, ε₆₄ (2M + 2w) pairwise. The constant 4 covers both,
/// with 2 ε₆₄ w to spare for the second-order terms.
///
/// *Why not relative to w.* The mismatch scales with the corner coordinates, not with
/// the side: for points 10⁹ times their extent from the origin the sides differ by
/// about 10⁹ ε₆₄ w, and a fixed relative tolerance would reject the output of
/// `compute_global_bounding_box` itself. The mismatch never costs more than the input's
/// own precision: a double locates a point only to ε₆₄ (|x| + |a|), the same order
/// (docs/phase3/README.md, "Domain"; CONVENTIONS §3.13, "Containment").
pub const SIDE_TOLERANCE: f64 = 4.0;

/// Why a [`PhysicalBox`] is not a valid cubic domain.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum GeometryError {
    /// A corner coordinate is NaN or infinite.
    #[error("the domain corners must be finite, got {coordinates:?}")]
    NonFinite {
        /// The corners `[xmin, ymin, zmin, xmax, ymax, zmax]` of the box.
        coordinates: [f64; 6],
    },
    /// A side max_k − min_k is zero, negative or overflows.
    #[error("the domain sides must be positive and finite, got sides {sides:?}")]
    InvalidSide {
        /// The sides fl(max_k − min_k) along x, y and z.
        sides: [f64; 3],
    },
    /// The sides differ by more than [`SIDE_TOLERANCE`] allows.
    #[error(
        "the domain must be cubic, got sides {sides:?}, which differ by {difference:e}, \
         more than the tolerance {tolerance:e}"
    )]
    NotCubic {
        /// The sides fl(max_k − min_k) along x, y and z.
        sides: [f64; 3],
        /// The largest difference between two sides.
        difference: f64,
        /// The tolerance [`SIDE_TOLERANCE`] · ε₆₄ (M + w) for this box.
        tolerance: f64,
    },
}

/// A validated cubic domain with lower corner a and side w (CONVENTIONS §3.12).
///
/// It keeps the [`PhysicalBox`] it was built from, which `nd_octree::points_to_morton`
/// and [`contains`] use, with its own side per axis. The side w of §3.12 and §3.13 is
/// the largest of the three; they agree up to [`SIDE_TOLERANCE`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain {
    coordinates: [f64; 6],
    lower: [f64; 3],
    side: f64,
}

impl Domain {
    /// Validates `bounding_box` as a cubic domain.
    ///
    /// # Errors
    ///
    /// - [`GeometryError::NonFinite`] if a corner coordinate is not finite;
    /// - [`GeometryError::InvalidSide`] if a side fl(max_k − min_k) is not positive or
    ///   not finite;
    /// - [`GeometryError::NotCubic`] if two sides differ by more than
    ///   [`SIDE_TOLERANCE`] · ε₆₄ (M + w), with M the largest corner coordinate in
    ///   magnitude and w the largest side.
    pub fn new(bounding_box: &PhysicalBox) -> Result<Self, GeometryError> {
        let coordinates = bounding_box.coordinates();
        if !coordinates.iter().all(|c| c.is_finite()) {
            return Err(GeometryError::NonFinite { coordinates });
        }
        let lower = [coordinates[0], coordinates[1], coordinates[2]];
        let sides: [f64; 3] = core::array::from_fn(|k| coordinates[k + 3] - lower[k]);
        if !sides.iter().all(|&w| w > 0.0 && w.is_finite()) {
            return Err(GeometryError::InvalidSide { sides });
        }
        let side = sides[0].max(sides[1]).max(sides[2]);
        let narrowest = sides[0].min(sides[1]).min(sides[2]);
        let largest_corner = coordinates.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        let tolerance = SIDE_TOLERANCE * f64::EPSILON / 2.0 * (largest_corner + side);
        let difference = side - narrowest;
        if difference > tolerance {
            return Err(GeometryError::NotCubic {
                sides,
                difference,
                tolerance,
            });
        }
        Ok(Self {
            coordinates,
            lower,
            side,
        })
    }

    /// The lower corner a = (xmin, ymin, zmin).
    pub fn lower(&self) -> [f64; 3] {
        self.lower
    }

    /// The side w, the largest of the three sides of the box.
    pub fn side(&self) -> f64 {
        self.side
    }

    /// The box the domain was built from, with its own side per axis, as
    /// `nd_octree::points_to_morton` uses it.
    pub fn physical_box(&self) -> PhysicalBox {
        PhysicalBox::new(self.coordinates)
    }
}

/// Returns 2^k, exactly, for −1022 ≤ k ≤ 1023.
#[inline]
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((1023 + k) as u64) << 52)
}

/// Returns the half-width r_l = w / 2^(l+1) of the boxes on `level`
/// (CONVENTIONS §3.12), exactly: dividing by a power of two does not round.
///
/// It is the scaling radius of §3.7, and the r_t by which output leaves the FMM
/// (§3.13, "Output").
///
/// # Panics
///
/// If `level > 16`.
#[inline]
pub fn radius(level: usize, domain: &Domain) -> f64 {
    assert!(level <= DEEPEST, "level {level} is deeper than {DEEPEST}");
    domain.side * pow2(-(level as i32 + 1))
}

/// Returns the centre c = a + (2i + 1) r_l of the box `key` on level l with index i
/// (CONVENTIONS §3.12), which is a + (i + ½) w / 2^l.
///
/// The product (2i + 1) r_l and the sum each round once, so the result is within
/// ε₆₄ (|a| + 2w) of the exact centre per component, and exact when a and w are dyadic
/// with few enough bits, as in the tests. It is for loading and reporting only; the
/// geometry between boxes comes from [`relative_frame`], never from a difference of
/// two centres.
///
/// `key` must be a valid key on a level 0–16.
#[inline]
pub fn centre(key: MortonKey, domain: &Domain) -> [f64; 3] {
    let (level, index) = morton::decode(key);
    let r = radius(level, domain);
    core::array::from_fn(|k| domain.lower[k] + (2 * index[k] + 1) as f64 * r)
}

/// Returns the integer centre C_L = (2i + 1) 2^(L − l) of the box `key` on level l with
/// index i, on the reference level L = `reference_level` (CONVENTIONS §3.13,
/// "Integer centres").
///
/// It is the centre in units of r_L, measured from the lower corner:
/// c = a + C_L w / 2^(L+1) for every L ≥ l. Each component is an odd multiple of
/// 2^(L − l) in [1, 2^(L+1) − 1].
///
/// # Panics
///
/// Unless l ≤ L ≤ 16. `key` must be a valid key.
#[inline]
pub fn integer_centre(key: MortonKey, reference_level: usize) -> [i64; 3] {
    let (level, index) = morton::decode(key);
    assert!(
        level <= reference_level && reference_level <= DEEPEST,
        "reference level {reference_level} must lie between the key's level {level} and {DEEPEST}"
    );
    index.map(|i| (2 * i as i64 + 1) << (reference_level - level))
}

/// Returns the frame (ĉ(s|t), r̂(s|t)) of box `s` in the scaled coordinates of box `t`
/// (CONVENTIONS §3.13, "Relative frames"), from the keys alone.
///
/// With L = max(l_s, l_t),
///
/// ĉ(s|t) = (c_s − c_t) / r_t = (C_L(s) − C_L(t)) 2^(l_t − L),  r̂(s|t) = r_s / r_t = 2^(l_t − l_s).
///
/// Neither the corner nor the side of the domain appears, so the frame is the same in
/// every cubic domain.
///
/// **Exact in f32 and f64.** The numerator N = C_L(s) − C_L(t) is formed in integers;
/// |N| ≤ 2^(L+1) − 2 ≤ 131070 = 2¹⁷ − 2 for keys on levels 0–16, attained by opposite
/// corners of level 16. Its conversion to T is exact (|N| < 2²⁴), and the product with
/// 2^(l_t − L), 0 ≤ L − l_t ≤ 16, is exact. So ĉ has at most 17 significant bits and
/// r̂ = 2^k with |k| ≤ 16: no step rounds, in either precision.
///
/// Consequences, all exact: ĉ(t|t) = 0 and r̂(t|t) = 1; ĉ(t|s) = −ĉ(s|t) / r̂(s|t) and
/// r̂(t|s) = 1 / r̂(s|t); a child of octant o seen from its parent is (½ s_o, ½), and the
/// target of a V-list pair with offset d seen from its source is (2d, 1), the canonical
/// frames of the tables (§3.12).
///
/// Operators pass these frames to `nd_fmm_ref::leaf` as §3.13, "Operators in scaled
/// coordinates", lists: P2L from leaf s into box t at (ĉ(t|s), r̂(t|s)), M2P from box s
/// at leaf t at (ĉ(s|t), r̂(s|t)), and P2P maps a source to ĉ(s|t) + r̂(s|t) u_s.
///
/// `s` and `t` must be valid keys on levels 0–16.
#[inline]
pub fn relative_frame<T: RealScalar>(s: MortonKey, t: MortonKey) -> Frame<T> {
    let (level_s, level_t) = (morton::level(s), morton::level(t));
    let reference = level_s.max(level_t);
    let (cs, ct) = (integer_centre(s, reference), integer_centre(t, reference));
    let scale = T::from_f64(pow2(level_t as i32 - reference as i32));
    Frame {
        centre: core::array::from_fn(|k| T::from_f64((cs[k] - ct[k]) as f64) * scale),
        radius: T::from_f64(pow2(level_t as i32 - level_s as i32)),
    }
}

/// Returns the leaf-scaled coordinates u = (x − a) 2^(l+1) / w − (2i + 1) of the point
/// `x` in the box `leaf` on level l with index i (CONVENTIONS §3.13, "Leaf-scaled
/// coordinates").
///
/// In exact arithmetic u = (x − c_b) / r_b, the scaled coordinates of §3.7 in the
/// frame of the leaf. Each component is evaluated in f64 in the order of §3.13:
/// d = fl(x − a), then fl(d · 2^(l+1) / w), in which the product is exact, then
/// fl(· − (2i + 1)), and finally rounded to T. For T = f32 the result is the cast of
/// the f64 value, never a computation in f32.
///
/// **Error bound.** With u the exact value of the formula for the doubles x, a and w,
/// per component and to first order,
///
/// |ũ − u| ≤ ε₆₄ |x − a| / r_l + ε₆₄ |x − a| / r_l + ε₆₄ |u| + ε_T |u|,
///
/// for the rounding of x − a, the division by w, the subtraction of 2i + 1 and the
/// cast; ε₆₄ = 2⁻⁵³, ε_T = 2⁻²⁴ for f32 and no cast term for f64. For a point of the
/// domain |x − a| ≤ w, so the first two terms are at most 2^(l−51), 2.9e-11 at level
/// 16, wherever the domain lies: in f64 the stored u is within 2^(l−51) + ε₆₄ of the
/// exact one, and in f32 the cast dominates on every level.
///
/// **Containment.** For a point that [`contains`] assigns to the leaf, the exact u lies
/// in [−1 − β_k, 1 + β_k] per component, with β_k = (|w_k − w| + 2 ε₆₄ w) / r_l to
/// first order and w_k the side of the box along axis k (§3.13, "Containment").
///
/// `leaf` must be a valid key on a level 0–16.
#[inline]
pub fn leaf_coordinates<T: RealScalar>(x: [f64; 3], leaf: MortonKey, domain: &Domain) -> [T; 3] {
    let (level, index) = morton::decode(leaf);
    let scale = pow2(level as i32 + 1);
    core::array::from_fn(|k| {
        let d = x[k] - domain.lower[k];
        T::from_f64(d * scale / domain.side - (2 * index[k] + 1) as f64)
    })
}

/// Returns whether `nd_octree::points_to_morton`, with the domain's box, puts `x` into
/// `leaf` or one of its descendants.
///
/// The key is computed with `morton::from_physical_point` on the leaf's level, as
/// `points_to_morton` does on its maximum level. The index on a deeper level L,
/// shifted right by L − l, is the index on level l, because both are the floor of a
/// power-of-two multiple of the same reference coordinate. So the answer is the same
/// for every maximum level from the leaf's level to 16.
///
/// A point not strictly inside the box (outside, on its boundary or with a NaN
/// coordinate) is in no leaf, since `points_to_morton` requires strict containment.
///
/// `leaf` must be a valid key on a level 0–16.
pub fn contains(leaf: MortonKey, x: [f64; 3], domain: &Domain) -> bool {
    let [xmin, ymin, zmin, xmax, ymax, zmax] = domain.coordinates;
    let inside = (0..3).all(|k| [xmin, ymin, zmin][k] < x[k] && x[k] < [xmax, ymax, zmax][k]);
    inside && morton::from_physical_point(x, &domain.physical_box(), morton::level(leaf)) == leaf
}
