//! Rotation and coaxial tables of point-and-shoot translation for the 316 V-list
//! offsets and the 8 child octants, and the table-driven rotation M2L, M2M and L2L
//! (CONVENTIONS §3.8, §3.11, §3.12; C2.3).
//!
//! # Point and shoot
//!
//! [`nd_fmm_ref::rotation`] translates in three steps (CONVENTIONS §3.11, "Rotation of
//! coefficients"; design §3.2): it rotates the input by the proper rotation
//! Q = R_y(−θ) R_z(−φ), with θ and φ the polar angle and azimuth of the shift c′ − c
//! (§3.2), so that the shift lies along +z; it applies the coaxial translation of §3.11
//! ("Coaxial translations") for d = |c′ − c|; and it rotates the result back by Qᵀ.
//! Degree n of multipole coefficients rotates by T_M = K Dⁿ K and of local coefficients
//! by T_L = K S Dⁿ S⁻¹ K, with the blocks Dⁿ of §3.8, S = diag((n − |m|)! (n + |m|)!)
//! and K = diag(+1 for m ≥ 0, −1 for m < 0). Shifts on the z axis are translated by
//! the coaxial form alone, with the signed d, and no rotation.
//!
//! `nd_fmm_ref::rotation` builds the blocks on every call. On a uniform level the
//! shifts are fixed: 2 r d for the V-list offsets d and ∓ r_child s_o for M2M and L2L
//! of child octant o (§3.12). This module precomputes, for every such shift, everything
//! the three steps need, so that the operators build no block per call.
//!
//! # Families and their geometry
//!
//! A [`RotationTables`] holds three [`ShiftTables`], one per [`Operator`], each built
//! at the canonical frames of [`geometry`](crate::geometry):
//!
//! | Family | Entries | Shift direction | Input rule | Output rule |
//! | --- | --- | --- | --- | --- |
//! | M2L | 316 offsets d, table index of §3.12 | d | T_M | T_L |
//! | M2M | 8 octants o, child index | −s_o | T_M | T_M |
//! | L2L | 8 octants o, child index | s_o | T_L | T_L |
//!
//! The angles are those of `nd_fmm_ref::rotation`: with (x, y, z) = c′ − c at the
//! canonical frames, θ = atan2(√(x² + y²), z) and φ = atan2(y, x), and Q = R_y(−θ)
//! R_z(−φ). A shift with x = y = 0 is [`Alignment::Up`] or [`Alignment::Down`] and uses
//! no rotation, as in nd-fmm-ref.
//!
//! Over the 316 offsets there are [`M2L_DISTANCE_COUNT`] = 15 distinct distances |d|,
//! [`M2L_AZIMUTH_COUNT`] = 32 distinct azimuths off the z axis, and
//! [`M2L_POLAR_ANGLE_COUNT`] = 49 distinct polar angles including θ = 0 and θ = π on
//! the axis. The two axis angles need no rotation, so the M2L family stores 47. Each
//! octant family has one distance, √3/2, four azimuths and two polar angles.
//!
//! Distinct values are found from exact integer keys of the direction, not by comparing
//! floating-point angles: |d|² for the distance, (sgn d_z, d_z²/|d|²) as a reduced
//! fraction for θ, and (d_x, d_y) divided by their gcd for φ. Floating point would split
//! one class: θ of (2, 2, 2) and of (3, 3, 3) differ in the last bit, because
//! √(6² + 6²) rounds differently from √(4² + 4²). Values are numbered in order of first
//! appearance in table order, and each is computed, as nd-fmm-ref would, from the shift
//! of the first entry that has it. Every entry has the distance d of its reference
//! computation bit for bit, since x² + y² + z² is exact for these shifts; its angles
//! can differ from its own reference angles in the last bit.
//!
//! # Storage
//!
//! Q is factored as nd-fmm-ref's z-y-z Euler angles suggest: T(Q) = T(R_y(−θ)) T(R_z(−φ))
//! and T(Qᵀ) = T(R_z(φ)) T(R_y(θ)), by the homomorphism of §3.8 and K² = I. Each family
//! stores, in f64 or rounded entry by entry to f32:
//!
//! - **y-rotations**, per distinct polar angle θ, off the axis: the forward blocks
//!   T_in(R_y(−θ)) with the input rule and the backward blocks T_out(R_y(θ)) with the
//!   output rule, `blocks_len(p)` = (p + 1)(2p + 1)(2p + 3)/3 reals each, in the
//!   row-major layout of [`nd_fmm_math::rotation::blocks`] with K and S applied
//!   ([`ShiftTables::forward_blocks`], [`ShiftTables::backward_blocks`]). In real
//!   storage the blocks are not orthogonal (§3.8), so the backward blocks are built from
//!   R_y(θ) = R_y(−θ)ᵀ, not by transposing the forward ones.
//! - **z-rotations**, per distinct azimuth φ: Cₘ and Sₘ for m = 1 to p, 2p reals
//!   ([`ShiftTables::azimuth_factors`]). T(R_z(−φ)) is the same for both rules and acts
//!   on each slot pair (+m, −m) of every degree n ≥ m as the plane rotation
//!
//!   x₊ ↦ Cₘ x₊ − Sₘ x₋,  x₋ ↦ Sₘ x₊ + Cₘ x₋,  Cₘ = cos mφ, Sₘ = sin mφ,
//!
//!   and leaves slot 0 unchanged; T(R_z(φ)) is its transpose. The reason: Dⁿ(R_z(α))
//!   rotates each pair of basis values by mα (§3.8); K turns that into the rotation by
//!   −mα of the conjugated coefficients, here α = −φ; and S is constant on a pair, so
//!   the local rule gives the same. Cₘ and Sₘ are read from the degree-m block of
//!   `blocks` of R_z(−φ) with K applied, so they come from the same recursion as every
//!   other block; the entries of a pair do not change with the degree n ≥ m. The unit
//!   tests check the action on coefficient vectors of both kinds against full blocks,
//!   and its physical meaning on point sources.
//! - **Coaxial factors**, per distinct distance d: for every order i = 0 to p, the real
//!   (p + 1 − i) × (p + 1 − i) matrix, row-major, from input degrees n to output degrees
//!   j, both from i to p, that the coaxial form of §3.11 applies to slot +i and to slot
//!   −i alike ([`ShiftTables::coaxial_factors`]); M2M uses its lower and L2L its upper
//!   triangle, and the other entries are zero. That is
//!   (p + 1)(p + 2)(2p + 3)/6 reals per distance. The column of input degree n and
//!   order i is the `nd_fmm_ref::rotation` operator of the family, at on-axis frames
//!   with the canonical radii and the shift d e_z, applied to the unit vector of slot
//!   (n, +i) into a zeroed output. On the axis nd-fmm-ref applies the coaxial form
//!   alone, so these are its coaxial factors, bit for bit. A shift along −z uses the
//!   factors of |d| with the parity (−1)ⁿ⁺ʲ of the axis values (§3.11), which is exact.
//! - **Shifts**: for each entry, its [`Shift`], the indices of its polar angle, azimuth
//!   and distance.
//!
//! With B = (p + 1)(2p + 1)(2p + 3)/3 and C = (p + 1)(p + 2)(2p + 3)/6, the M2L family
//! stores 94 B + 64 p + 15 C reals and each octant family 4 B + 8 p + C, so a
//! [`RotationTables`] holds 102 B + 80 p + 17 C reals ([`RotationTables::storage_len`]):
//! in f64 about 0.83 MB at p = 8, 5.6 MB at p = 16 and 10.5 MB at p = 20. The
//! alternative, full forward and backward blocks for each of the 312 offsets off the
//! axis, needs 624 B for M2L alone, about 62 MB at p = 20; the dense M2L tables need
//! 492 MB and the 16-class form 34 MB ([`crate::m2l`]).
//!
//! # Cost
//!
//! A table-driven operator rotates by T(R_z(−φ)) and T_in(R_y(−θ)), applies the coaxial
//! factors, and rotates back by T_out(R_y(θ)) and T(R_z(φ)); on the axis it applies the
//! coaxial factors alone. Its multiply-adds, with no block construction at all:
//!
//! - each y-rotation Σₙ (2n + 1)² = B ≈ (4/3) p³;
//! - each z-rotation 4 · p(p + 1)/2 = 2p(p + 1), plus p + 1 additions for slot 0;
//! - the coaxial step (p + 1)(2p² + 4p + 3)/3 ≈ (2/3) p³ for M2L and
//!   (p + 1)(p + 2)(2p + 3)/6 ≈ (1/3) p³ for M2M and L2L.
//!
//! So M2L costs (p + 1)(10p² + 32p + 9)/3 ≈ (10/3) p³ and M2M and L2L cost
//! 3 (p + 1)(2p² + 7p + 2)/2 ≈ 3p³ multiply-adds, plus 2 (p + 1) additions. That is the
//! count of `nd_fmm_ref::rotation`, (p + 1)(10p² + 20p + 9)/3 and
//! (p + 1)(2p + 3)(3p + 2)/2, plus the 4p(p + 1) of the two z-rotations, and matches
//! design §3.2: (10/3) p³ per M2L and 3p³ per M2M or L2L. `nd_fmm_ref::rotation`
//! additionally builds two sets of blocks per call, about as many operations again,
//! which the tables remove. The y-blocks are stored dense, although in the complex
//! basis they are real, so that half of their entries are zero; exploiting that is left
//! to later phases.
//!
//! # Accuracy
//!
//! On the axis the operators equal `nd_fmm_ref::rotation` bit for bit. Off the axis
//! they compose the rotation from two factors instead of building Dⁿ(Q) at once, and
//! may use angles that differ from nd-fmm-ref's in the last bit, so they agree to
//! rounding (the tests assert 1e-14 relative to the terms).
//!
//! ```
//! use nd_fmm_ref::{Workspace, rotation};
//! use nd_fmm_tables::geometry::m2l_frames;
//! use nd_fmm_tables::rotation::{RotationScratch, RotationTables};
//!
//! let p = 6;
//! let tables = RotationTables::<f64>::build(p);
//! let mut scratch = RotationScratch::new(p);
//! let index = tables.index([3, -2, 1]).unwrap();
//! let multipole: Vec<f64> = (0..49).map(|i| 1.0 / (1.0 + i as f64)).collect();
//! let mut from_tables = vec![0.0; 49];
//! tables.m2l(index, &multipole, &mut from_tables, &mut scratch);
//!
//! let (source, target) = m2l_frames(index);
//! let mut from_reference = vec![0.0; 49];
//! rotation::m2l(p, &source, &target, &mut Workspace::new(p), &multipole, &mut from_reference);
//! let largest = from_reference.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
//! for (a, b) in from_tables.iter().zip(&from_reference) {
//!     assert!((a - b).abs() < 1e-14 * largest);
//! }
//! ```

use core::ops::RangeInclusive;

use nd_fmm_math::rotation::{block_range, blocks, blocks_len, euler_zyz, to_irregular};
use nd_fmm_math::{Layout, RealScalar};
use nd_fmm_ref::{Frame, Workspace, rotation};

use crate::geometry::{
    M2L_OFFSET_COUNT, OCTANT_COUNT, l2l_frames, m2l_frames, m2l_offset_index, m2l_offsets,
    m2m_frames, octant_direction,
};
use crate::symmetry::Expansion;

/// Number of distinct distances |d| of the 316 V-list offsets: |d|² takes the values 4,
/// 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22 and 27.
pub const M2L_DISTANCE_COUNT: usize = 15;

/// Number of distinct azimuths of the V-list offsets off the z axis: the directions of
/// the nonzero (d_x, d_y) in {−3..3}².
pub const M2L_AZIMUTH_COUNT: usize = 32;

/// Number of distinct polar angles of the V-list offsets, including θ = 0 and θ = π on
/// the z axis. The M2L family stores blocks for the 47 off the axis.
pub const M2L_POLAR_ANGLE_COUNT: usize = 49;

/// The signature of the `nd_fmm_ref::rotation` operators in f64.
type Translate = fn(usize, &Frame<f64>, &Frame<f64>, &mut Workspace<f64>, &[f64], &mut [f64]);

/// A translation family of point-and-shoot tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    /// M2M from child octant o to its parent: multipole in and out, shift −s_o.
    M2m,
    /// L2L from a parent to its child octant o: local in and out, shift s_o.
    L2l,
    /// M2L across the V-list offset d: multipole in, local out, shift d.
    M2l,
}

impl Operator {
    /// Returns the number of entries: 316 offsets for M2L, 8 octants otherwise.
    pub const fn count(self) -> usize {
        match self {
            Operator::M2l => M2L_OFFSET_COUNT,
            Operator::M2m | Operator::L2l => OCTANT_COUNT,
        }
    }

    /// Returns the rotation rule of the input: T_M for multipoles, T_L for locals.
    pub const fn input(self) -> Expansion {
        match self {
            Operator::M2m | Operator::M2l => Expansion::Multipole,
            Operator::L2l => Expansion::Local,
        }
    }

    /// Returns the rotation rule of the output: T_M for multipoles, T_L for locals.
    pub const fn output(self) -> Expansion {
        match self {
            Operator::M2m => Expansion::Multipole,
            Operator::L2l | Operator::M2l => Expansion::Local,
        }
    }

    /// Returns the integer direction of the shift c′ − c of entry `t`: the offset d for
    /// M2L, −s_o for M2M and s_o for L2L (CONVENTIONS §3.12).
    ///
    /// # Panics
    ///
    /// If `t >= self.count()`.
    pub fn direction(self, t: usize) -> [i64; 3] {
        match self {
            Operator::M2l => {
                assert!(
                    t < M2L_OFFSET_COUNT,
                    "M2L offset index {t} out of range for {M2L_OFFSET_COUNT} offsets"
                );
                m2l_offsets()[t]
            }
            Operator::M2m => octant_direction(t).map(|s| -s),
            Operator::L2l => octant_direction(t),
        }
    }

    /// Returns the canonical (input, output) frames of entry `t`, from
    /// [`geometry`](crate::geometry).
    ///
    /// # Panics
    ///
    /// If `t >= self.count()`.
    pub fn frames(self, t: usize) -> (Frame<f64>, Frame<f64>) {
        match self {
            Operator::M2l => m2l_frames(t),
            Operator::M2m => m2m_frames(t),
            Operator::L2l => l2l_frames(t),
        }
    }

    /// The `nd_fmm_ref::rotation` operator of the family.
    fn oracle(self) -> Translate {
        match self {
            Operator::M2m => rotation::m2m::<f64>,
            Operator::L2l => rotation::l2l::<f64>,
            Operator::M2l => rotation::m2l::<f64>,
        }
    }

    /// The input degrees n of output degree j at order i (i ≤ j ≤ p) with a nonzero
    /// coaxial factor: n ≤ j for M2M, n ≥ j for L2L, every n ≥ i for M2L (§3.11).
    fn inputs(self, i: usize, j: usize, p: usize) -> RangeInclusive<usize> {
        match self {
            Operator::M2m => i..=j,
            Operator::L2l => j..=p,
            Operator::M2l => i..=p,
        }
    }
}

/// How the shift of one entry is brought onto the +z axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alignment {
    /// The shift lies along +z: no rotation.
    Up,
    /// The shift lies along −z: no rotation; the coaxial factors of the distance take
    /// the parity (−1)ⁿ⁺ʲ of input degree n and output degree j.
    Down,
    /// Any other shift: rotation by Q = R_y(−θ) R_z(−φ), with θ the stored polar angle
    /// `polar` and φ the stored azimuth `azimuth`.
    Rotated {
        /// Index of θ, below [`ShiftTables::polar_count`].
        polar: usize,
        /// Index of φ, below [`ShiftTables::azimuth_count`].
        azimuth: usize,
    },
}

/// The geometry of one entry: how its shift is aligned with z, and its distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shift {
    /// The rotation, if any.
    pub alignment: Alignment,
    /// Index of the distance |c′ − c|, below [`ShiftTables::distance_count`].
    pub distance: usize,
}

/// The rotation and coaxial tables of one translation family (see the
/// [module documentation](self)): the y-rotation blocks per distinct polar angle, the
/// z-rotation factors per distinct azimuth, the coaxial factors per distinct distance,
/// and the [`Shift`] of each entry.
///
/// Storage: with B = (p + 1)(2p + 1)(2p + 3)/3 and C = (p + 1)(p + 2)(2p + 3)/6,
/// 2 B per stored polar angle, 2p per azimuth and C per distance
/// ([`ShiftTables::storage_len`]): 94 B + 64 p + 15 C reals for M2L and 4 B + 8 p + C
/// for M2M or L2L.
#[derive(Clone, Debug, PartialEq)]
pub struct ShiftTables<T: RealScalar> {
    pub(crate) operator: Operator,
    pub(crate) p: usize,
    /// θ per stored polar angle, in f64.
    pub(crate) polar_angles: Vec<f64>,
    /// φ per azimuth, in f64.
    pub(crate) azimuth_angles: Vec<f64>,
    /// |c′ − c| per distance, in f64.
    pub(crate) distances: Vec<f64>,
    /// T_in(R_y(−θ)) per polar angle, `blocks_len(p)` each.
    pub(crate) forward: Vec<T>,
    /// T_out(R_y(θ)) per polar angle, `blocks_len(p)` each.
    pub(crate) backward: Vec<T>,
    /// (Cₘ, Sₘ) for m = 1 to p per azimuth, 2p each.
    pub(crate) azimuth: Vec<T>,
    /// The per-order coaxial matrices per distance, `coaxial_len(p)` each.
    pub(crate) coaxial: Vec<T>,
    /// The shift of each entry, in entry order.
    pub(crate) shifts: Vec<Shift>,
}

/// Number of coaxial factors per distance, Σᵢ (p + 1 − i)² = (p + 1)(p + 2)(2p + 3)/6.
pub(crate) const fn coaxial_len(p: usize) -> usize {
    (p + 1) * (p + 2) * (2 * p + 3) / 6
}

/// Index of `key` in `keys`, appending it (and calling `new`) if absent.
fn position_or_push<K: PartialEq>(keys: &mut Vec<K>, key: K, new: impl FnOnce()) -> usize {
    match keys.iter().position(|k| *k == key) {
        Some(i) => i,
        None => {
            keys.push(key);
            new();
            keys.len() - 1
        }
    }
}

/// The greatest common divisor of two nonnegative integers, not both zero.
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The blocks of `q` with the rule of `expansion`, in f64: K Dⁿ K for multipoles and
/// K S Dⁿ S⁻¹ K for locals (CONVENTIONS §3.11, "Rotation of coefficients").
fn rule_blocks(p: usize, q: &[[f64; 3]; 3], expansion: Expansion) -> Vec<f64> {
    let mut d = vec![0.0; blocks_len(p)];
    blocks(p, q, &mut d);
    if expansion == Expansion::Local {
        to_irregular(p, &mut d);
    }
    for n in 1..=p {
        let width = 2 * n + 1;
        for (at, v) in d[block_range(n)].iter_mut().enumerate() {
            // K on both sides negates entry (i, j) when exactly one order is negative;
            // rows and columns run over m = −n..n, so slot t has order t − n.
            if (at / width < n) != (at % width < n) {
                *v = -*v;
            }
        }
    }
    d
}

/// Adds B x to `out` degree by degree, for the row-major blocks `blocks` in the layout
/// of `nd_fmm_math::rotation::blocks`: each row is summed in increasing column order
/// and then added. `blocks_len(p)` multiply-adds.
fn apply_blocks<T: RealScalar>(p: usize, blocks: &[T], x: &[T], out: &mut [T]) {
    for (n, range) in Layout::new(p).degrees() {
        let block = &blocks[block_range(n)];
        let x = &x[range.clone()];
        for (o, row) in out[range].iter_mut().zip(block.chunks_exact(2 * n + 1)) {
            let sum = row
                .iter()
                .zip(x)
                .fold(T::zero(), |acc, (&a, &v)| acc + a * v);
            *o = *o + sum;
        }
    }
}

impl ShiftTables<f64> {
    /// Builds the tables of `operator` at degree `p` in f64 (module documentation).
    fn build(p: usize, operator: Operator) -> Self {
        let mut polar_keys = Vec::new();
        let mut azimuth_keys = Vec::new();
        let mut distance_keys = Vec::new();
        let mut polar_angles = Vec::new();
        let mut azimuth_angles = Vec::new();
        // The distance and the radii of the input and output frames.
        let mut distances: Vec<(f64, f64, f64)> = Vec::new();
        let mut shifts = Vec::with_capacity(operator.count());
        for t in 0..operator.count() {
            let (from, to) = operator.frames(t);
            let [dx, dy, dz] = operator.direction(t);
            // The shift as nd-fmm-ref forms it, and its angles (CONVENTIONS §3.2).
            let [x, y, z]: [f64; 3] = core::array::from_fn(|i| to.centre[i] - from.centre[i]);
            let norm2 = dx * dx + dy * dy + dz * dz;
            let distance = position_or_push(&mut distance_keys, norm2, || {
                distances.push(((x * x + y * y + z * z).sqrt(), from.radius, to.radius));
            });
            let alignment = if dx == 0 && dy == 0 {
                if dz > 0 {
                    Alignment::Up
                } else {
                    Alignment::Down
                }
            } else {
                // cos²θ = d_z²/|d|² as a reduced fraction, with the sign of d_z.
                let g = gcd(dz * dz, norm2);
                let polar_key = (dz.signum(), dz * dz / g, norm2 / g);
                let polar = position_or_push(&mut polar_keys, polar_key, || {
                    polar_angles.push((x * x + y * y).sqrt().atan2(z));
                });
                let g = gcd(dx.abs(), dy.abs());
                let azimuth = position_or_push(&mut azimuth_keys, [dx / g, dy / g], || {
                    azimuth_angles.push(y.atan2(x));
                });
                Alignment::Rotated { polar, azimuth }
            };
            shifts.push(Shift {
                alignment,
                distance,
            });
        }

        let mut forward = Vec::with_capacity(polar_angles.len() * blocks_len(p));
        let mut backward = Vec::with_capacity(polar_angles.len() * blocks_len(p));
        for &theta in &polar_angles {
            let q = euler_zyz(0.0, -theta, 0.0);
            forward.extend(rule_blocks(p, &q, operator.input()));
            let q_transposed: [[f64; 3]; 3] = core::array::from_fn(|i| q.map(|row| row[i]));
            backward.extend(rule_blocks(p, &q_transposed, operator.output()));
        }

        let mut azimuth = Vec::with_capacity(azimuth_angles.len() * 2 * p);
        for &phi in &azimuth_angles {
            // T(R_z(−φ)) is the same for both rules (module documentation).
            let d = rule_blocks(p, &euler_zyz(0.0, 0.0, -phi), Expansion::Multipole);
            for m in 1..=p {
                let block = &d[block_range(m)];
                let width = 2 * m + 1;
                // Column +m (position 2m): row +m holds Cₘ, row −m (position 0) Sₘ.
                azimuth.push(block[2 * m * width + 2 * m]);
                azimuth.push(block[2 * m]);
            }
        }

        let layout = Layout::new(p);
        let mut ws = Workspace::new(p);
        let mut unit = vec![0.0; layout.len()];
        let mut column = vec![0.0; layout.len()];
        let mut coaxial = Vec::with_capacity(distances.len() * coaxial_len(p));
        for &(d, from_radius, to_radius) in &distances {
            // On-axis frames with the canonical radii: nd-fmm-ref applies the coaxial
            // form alone, with the shift d e_z.
            let from = Frame::new([0.0; 3], from_radius);
            let to = Frame::new([0.0, 0.0, d], to_radius);
            for i in 0..=p {
                let width = p + 1 - i;
                let start = coaxial.len();
                coaxial.resize(start + width * width, 0.0);
                for n in i..=p {
                    let k = layout.idx(n, i as isize);
                    unit[k] = 1.0;
                    column.fill(0.0);
                    operator.oracle()(p, &from, &to, &mut ws, &unit, &mut column);
                    unit[k] = 0.0;
                    for j in i..=p {
                        coaxial[start + (j - i) * width + (n - i)] =
                            column[layout.idx(j, i as isize)];
                    }
                }
            }
        }

        Self {
            operator,
            p,
            polar_angles,
            azimuth_angles,
            distances: distances.into_iter().map(|(d, _, _)| d).collect(),
            forward,
            backward,
            azimuth,
            coaxial,
            shifts,
        }
    }
}

impl<T: RealScalar> ShiftTables<T> {
    /// Returns the translation family.
    #[inline]
    pub fn operator(&self) -> Operator {
        self.operator
    }

    /// Returns the degree p of input and output.
    #[inline]
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the number of entries: 316 offsets for M2L, 8 octants otherwise.
    #[inline]
    pub fn count(&self) -> usize {
        self.shifts.len()
    }

    /// Returns the shift of entry `t`: the table index of §3.12 for M2L, the child index
    /// for M2M and L2L.
    ///
    /// # Panics
    ///
    /// If `t >= self.count()`.
    #[inline]
    pub fn shift(&self, t: usize) -> Shift {
        self.shifts[t]
    }

    /// Returns the number of stored polar angles, those off the z axis: 47 for M2L and 2
    /// for M2M and L2L.
    #[inline]
    pub fn polar_count(&self) -> usize {
        self.polar_angles.len()
    }

    /// Returns the number of distinct azimuths: 32 for M2L and 4 for M2M and L2L.
    #[inline]
    pub fn azimuth_count(&self) -> usize {
        self.azimuth_angles.len()
    }

    /// Returns the number of distinct distances: 15 for M2L and 1 for M2M and L2L.
    #[inline]
    pub fn distance_count(&self) -> usize {
        self.distances.len()
    }

    /// Returns the polar angle θ = atan2(√(x² + y²), z) of stored angle `polar`, in f64,
    /// computed from the canonical shift (x, y, z) of the first entry that has it.
    ///
    /// # Panics
    ///
    /// If `polar >= self.polar_count()`.
    #[inline]
    pub fn polar_angle(&self, polar: usize) -> f64 {
        self.polar_angles[polar]
    }

    /// Returns the azimuth φ = atan2(y, x) of azimuth `azimuth`, in f64, computed from
    /// the canonical shift (x, y, z) of the first entry that has it.
    ///
    /// # Panics
    ///
    /// If `azimuth >= self.azimuth_count()`.
    #[inline]
    pub fn azimuth_angle(&self, azimuth: usize) -> f64 {
        self.azimuth_angles[azimuth]
    }

    /// Returns the distance |c′ − c| at the canonical frames of distance `distance`, in
    /// f64: 2|d| for M2L and √3/2 for M2M and L2L.
    ///
    /// # Panics
    ///
    /// If `distance >= self.distance_count()`.
    #[inline]
    pub fn distance(&self, distance: usize) -> f64 {
        self.distances[distance]
    }

    /// Returns the forward y-rotation blocks T_in(R_y(−θ)) of polar angle `polar`, with
    /// the input rule: `blocks_len(p)` reals, per degree row-major in slot order
    /// m = −n to n, in the layout of `nd_fmm_math::rotation::blocks`.
    ///
    /// # Panics
    ///
    /// If `polar >= self.polar_count()`.
    #[inline]
    pub fn forward_blocks(&self, polar: usize) -> &[T] {
        let len = blocks_len(self.p);
        &self.forward[polar * len..(polar + 1) * len]
    }

    /// Returns the backward y-rotation blocks T_out(R_y(θ)) of polar angle `polar`, with
    /// the output rule, in the layout of [`ShiftTables::forward_blocks`].
    ///
    /// # Panics
    ///
    /// If `polar >= self.polar_count()`.
    #[inline]
    pub fn backward_blocks(&self, polar: usize) -> &[T] {
        let len = blocks_len(self.p);
        &self.backward[polar * len..(polar + 1) * len]
    }

    /// Returns the z-rotation factors of azimuth `azimuth`: C₁, S₁, C₂, S₂, …, Cₚ, Sₚ,
    /// 2p reals, with Cₘ = cos mφ and Sₘ = sin mφ as read from the blocks of R_z(−φ)
    /// (module documentation).
    ///
    /// # Panics
    ///
    /// If `azimuth >= self.azimuth_count()`.
    #[inline]
    pub fn azimuth_factors(&self, azimuth: usize) -> &[T] {
        let len = 2 * self.p;
        &self.azimuth[azimuth * len..(azimuth + 1) * len]
    }

    /// Returns the coaxial factors of distance `distance`: for each order i = 0 to p in
    /// turn, the (p + 1 − i)² matrix, row-major, whose entry (j − i, n − i) takes input
    /// degree n to output degree j at orders +i and −i (module documentation);
    /// (p + 1)(p + 2)(2p + 3)/6 reals.
    ///
    /// # Panics
    ///
    /// If `distance >= self.distance_count()`.
    #[inline]
    pub fn coaxial_factors(&self, distance: usize) -> &[T] {
        let len = coaxial_len(self.p);
        &self.coaxial[distance * len..(distance + 1) * len]
    }

    /// Returns the number of reals stored: 2 `blocks_len(p)` per polar angle, 2p per
    /// azimuth and (p + 1)(p + 2)(2p + 3)/6 per distance (type documentation). The
    /// angles, distances and shifts are not counted.
    pub fn storage_len(&self) -> usize {
        self.forward.len() + self.backward.len() + self.azimuth.len() + self.coaxial.len()
    }

    /// Adds the translation of entry `t` of `input` to `output`: rotation by Q with the
    /// input rule, the coaxial step and rotation back by Qᵀ with the output rule, or,
    /// for a shift on the z axis, the coaxial step alone (module documentation). Both
    /// are scaled coefficients of degree p, (p + 1)² reals in the storage of §3.6.
    /// Accumulates, allocates nothing and builds no block.
    ///
    /// Cost: (p + 1)(10p² + 32p + 9)/3 multiply-adds for M2L and
    /// 3 (p + 1)(2p² + 7p + 2)/2 for M2M and L2L, plus 2 (p + 1) additions; on the axis
    /// only the coaxial step, (p + 1)(2p² + 4p + 3)/3 or (p + 1)(p + 2)(2p + 3)/6.
    ///
    /// # Panics
    ///
    /// If `t >= self.count()`, `scratch` is not of degree p, or `input` or `output` does
    /// not have length (p + 1)².
    pub fn apply(&self, t: usize, input: &[T], output: &mut [T], scratch: &mut RotationScratch<T>) {
        self.check(input, output, scratch);
        let Shift {
            alignment,
            distance,
        } = self.shifts[t];
        match alignment {
            Alignment::Up => self.coaxial_step(distance, false, input, output),
            Alignment::Down => self.coaxial_step(distance, true, input, output),
            Alignment::Rotated { polar, azimuth } => {
                let RotationScratch { first, second } = scratch;
                first.fill(T::zero());
                self.rotate_z(azimuth, false, input, first);
                second.fill(T::zero());
                apply_blocks(self.p, self.forward_blocks(polar), first, second);
                first.fill(T::zero());
                self.coaxial_step(distance, false, second, first);
                second.fill(T::zero());
                apply_blocks(self.p, self.backward_blocks(polar), first, second);
                self.rotate_z(azimuth, true, second, output);
            }
        }
    }

    /// Adds T_in(Q) `input` to `output`, the first step of [`ShiftTables::apply`]: the
    /// z-rotation by R_z(−φ), then the forward y-blocks. For a shift on the z axis Q = I
    /// and `input` is added unchanged. Accumulates and allocates nothing.
    ///
    /// # Panics
    ///
    /// As [`ShiftTables::apply`].
    pub fn rotate(
        &self,
        t: usize,
        input: &[T],
        output: &mut [T],
        scratch: &mut RotationScratch<T>,
    ) {
        self.check(input, output, scratch);
        match self.shifts[t].alignment {
            Alignment::Up | Alignment::Down => add(input, output),
            Alignment::Rotated { polar, azimuth } => {
                let first = &mut scratch.first;
                first.fill(T::zero());
                self.rotate_z(azimuth, false, input, first);
                apply_blocks(self.p, self.forward_blocks(polar), first, output);
            }
        }
    }

    /// Adds T_out(Qᵀ) `input` to `output`, the last step of [`ShiftTables::apply`]: the
    /// backward y-blocks, then the z-rotation by R_z(φ). For a shift on the z axis
    /// `input` is added unchanged. Accumulates and allocates nothing.
    ///
    /// # Panics
    ///
    /// As [`ShiftTables::apply`].
    pub fn rotate_back(
        &self,
        t: usize,
        input: &[T],
        output: &mut [T],
        scratch: &mut RotationScratch<T>,
    ) {
        self.check(input, output, scratch);
        match self.shifts[t].alignment {
            Alignment::Up | Alignment::Down => add(input, output),
            Alignment::Rotated { polar, azimuth } => {
                let first = &mut scratch.first;
                first.fill(T::zero());
                apply_blocks(self.p, self.backward_blocks(polar), input, first);
                self.rotate_z(azimuth, true, first, output);
            }
        }
    }

    /// Adds the coaxial step of entry `t` of `input` to `output`, the middle step of
    /// [`ShiftTables::apply`]: the coaxial form of §3.11 for its distance, with the
    /// parity (−1)ⁿ⁺ʲ for a shift along −z. For a shift on the z axis this is the whole
    /// translation. Accumulates and allocates nothing.
    ///
    /// # Panics
    ///
    /// If `t >= self.count()`, or `input` or `output` does not have length (p + 1)².
    pub fn translate_coaxial(&self, t: usize, input: &[T], output: &mut [T]) {
        self.check_coefficients(input, output);
        let Shift {
            alignment,
            distance,
        } = self.shifts[t];
        self.coaxial_step(distance, alignment == Alignment::Down, input, output);
    }

    /// Returns the same tables in precision `U`, every stored real rounded to nearest.
    pub fn cast<U: RealScalar>(&self) -> ShiftTables<U> {
        let cast = |v: &[T]| v.iter().map(|&x| U::from_f64(x.to_f64())).collect();
        ShiftTables {
            operator: self.operator,
            p: self.p,
            polar_angles: self.polar_angles.clone(),
            azimuth_angles: self.azimuth_angles.clone(),
            distances: self.distances.clone(),
            forward: cast(&self.forward),
            backward: cast(&self.backward),
            azimuth: cast(&self.azimuth),
            coaxial: cast(&self.coaxial),
            shifts: self.shifts.clone(),
        }
    }

    /// Adds T(R_z(−φ)) x, or with `inverse` T(R_z(φ)) x, to `out` for azimuth
    /// `azimuth`: slot 0 of each degree unchanged, each pair (+m, −m) rotated by the
    /// factors Cₘ and Sₘ (module documentation). 2p(p + 1) multiply-adds and p + 1
    /// additions.
    fn rotate_z(&self, azimuth: usize, inverse: bool, x: &[T], out: &mut [T]) {
        let factors = self.azimuth_factors(azimuth);
        let layout = Layout::new(self.p);
        for n in 0..=self.p {
            let centre = layout.idx(n, 0);
            out[centre] = out[centre] + x[centre];
            for (m, &[c, s]) in (1..=n).zip(factors.as_chunks::<2>().0) {
                let s = if inverse { -s } else { s };
                let (plus, minus) = (centre + m, centre - m);
                let (u, v) = (x[plus], x[minus]);
                out[plus] = out[plus] + (c * u - s * v);
                out[minus] = out[minus] + (s * u + c * v);
            }
        }
    }

    /// Adds the coaxial step of distance `distance` of x to `out`, with the parity
    /// (−1)ⁿ⁺ʲ if `down`. Each output slot (j, ±i) is accumulated in increasing input
    /// degree n, starting from its incoming value, as in nd-fmm-ref.
    fn coaxial_step(&self, distance: usize, down: bool, x: &[T], out: &mut [T]) {
        let p = self.p;
        let layout = Layout::new(p);
        let factors = self.coaxial_factors(distance);
        let mut start = 0;
        for i in 0..=p {
            let width = p + 1 - i;
            let block = &factors[start..start + width * width];
            start += width * width;
            for j in i..=p {
                let row = &block[(j - i) * width..(j - i + 1) * width];
                let orders: &[isize] = if i == 0 {
                    &[0]
                } else {
                    &[i as isize, -(i as isize)]
                };
                for &m in orders {
                    let y = layout.idx(j, m);
                    let mut sum = out[y];
                    for n in self.operator.inputs(i, j, p) {
                        let a = row[n - i];
                        let a = if down && (n + j) % 2 == 1 { -a } else { a };
                        sum = sum + a * x[layout.idx(n, m)];
                    }
                    out[y] = sum;
                }
            }
        }
    }

    /// Checks the coefficient lengths and the degree of the scratch.
    fn check(&self, input: &[T], output: &[T], scratch: &RotationScratch<T>) {
        self.check_coefficients(input, output);
        assert_eq!(
            scratch.first.len(),
            input.len(),
            "`scratch` must be of degree p = {}",
            self.p
        );
    }

    /// Checks that `input` and `output` have length (p + 1)².
    fn check_coefficients(&self, input: &[T], output: &[T]) {
        let n = Layout::new(self.p).len();
        assert_eq!(input.len(), n, "`input` must have length (p + 1)^2 = {n}");
        assert_eq!(output.len(), n, "`output` must have length (p + 1)^2 = {n}");
    }
}

/// Adds x to `out`.
fn add<T: RealScalar>(x: &[T], out: &mut [T]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = *o + v;
    }
}

/// The rotation and coaxial tables of point-and-shoot translation for the 316 V-list
/// offsets and the 8 child octants, and the table-driven rotation operators
/// (CONVENTIONS §3.11, §3.12; see the [module documentation](self)).
///
/// It holds one [`ShiftTables`] per [`Operator`]. Built in f64 from
/// `nd_fmm_math::rotation` and `nd_fmm_ref::rotation` at the canonical frames, and
/// rounded entry by entry to `T`. By the scaling of §3.7 the same tables serve every
/// level of every cubic domain; they are looked up by table index and child index,
/// never by a floating-point shift.
///
/// Storage: 102 B + 80 p + 17 C reals with B = (p + 1)(2p + 1)(2p + 3)/3 and
/// C = (p + 1)(p + 2)(2p + 3)/6 ([`RotationTables::storage_len`]); in f64 about 0.83 MB
/// at p = 8, 5.6 MB at p = 16 and 10.5 MB at p = 20.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationTables<T: RealScalar> {
    pub(crate) p: usize,
    pub(crate) m2m: ShiftTables<T>,
    pub(crate) l2l: ShiftTables<T>,
    pub(crate) m2l: ShiftTables<T>,
}

impl<T: RealScalar> RotationTables<T> {
    /// Builds the tables of degree `p`: the three families in f64 (module
    /// documentation), then rounded to `T`. Serial and deterministic; allocates.
    ///
    /// Building costs, per family, two `nd_fmm_math::rotation::blocks` per polar angle
    /// and one per azimuth, O(p³) each, and (p + 1)(p + 2)/2 on-axis calls of the
    /// `nd_fmm_ref::rotation` operator per distance, O(p³) each.
    pub fn build(p: usize) -> Self {
        RotationTables {
            p,
            m2m: ShiftTables::build(p, Operator::M2m),
            l2l: ShiftTables::build(p, Operator::L2l),
            m2l: ShiftTables::build(p, Operator::M2l),
        }
        .cast()
    }

    /// Returns the degree p of input and output.
    #[inline]
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the tables of `operator`.
    #[inline]
    pub fn tables(&self, operator: Operator) -> &ShiftTables<T> {
        match operator {
            Operator::M2m => &self.m2m,
            Operator::L2l => &self.l2l,
            Operator::M2l => &self.m2l,
        }
    }

    /// Returns the table index of the V-list offset `offset` = d, or `None` if d is not
    /// in {−3..3}³ \ {−1..1}³ ([`m2l_offset_index`]).
    #[inline]
    pub fn index(&self, offset: [i64; 3]) -> Option<usize> {
        m2l_offset_index(offset)
    }

    /// Returns the number of reals stored by the three families
    /// ([`ShiftTables::storage_len`]).
    pub fn storage_len(&self) -> usize {
        self.m2m.storage_len() + self.l2l.storage_len() + self.m2l.storage_len()
    }

    /// Adds the M2L of the source multipole expansion `multipole` to the local
    /// expansion `local` of the target box at the offset with table index `index`
    /// (CONVENTIONS §3.11, §3.12), by rotation, coaxial translation and rotation back.
    /// Both are scaled coefficients of degree p, (p + 1)² reals in the storage of §3.6.
    /// Equals `nd_fmm_ref::rotation::m2l` at [`m2l_frames`]`(index)` to rounding.
    ///
    /// Accumulates, allocates nothing and builds no block. Cost:
    /// (p + 1)(10p² + 32p + 9)/3 ≈ (10/3) p³ multiply-adds and 2 (p + 1) additions; for
    /// the four offsets on the z axis only the coaxial step,
    /// (p + 1)(2p² + 4p + 3)/3 ≈ (2/3) p³.
    ///
    /// # Panics
    ///
    /// If `index >= 316`, `scratch` is not of degree p, or `multipole` or `local` does
    /// not have length (p + 1)².
    #[inline]
    pub fn m2l(
        &self,
        index: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut RotationScratch<T>,
    ) {
        self.m2l.apply(index, multipole, local, scratch);
    }

    /// Adds the M2M of the child multipole expansion `input`, in child octant `o`, to
    /// the parent multipole expansion `output` (CONVENTIONS §3.11, §3.12), by rotation,
    /// coaxial translation and rotation back. Both are scaled coefficients of degree p.
    /// Equals `nd_fmm_ref::rotation::m2m` at [`m2m_frames`]`(o)` to rounding.
    ///
    /// Accumulates, allocates nothing and builds no block. Cost:
    /// 3 (p + 1)(2p² + 7p + 2)/2 ≈ 3p³ multiply-adds and 2 (p + 1) additions.
    ///
    /// # Panics
    ///
    /// If `o >= 8`, `scratch` is not of degree p, or `input` or `output` does not have
    /// length (p + 1)².
    #[inline]
    pub fn m2m(&self, o: usize, input: &[T], output: &mut [T], scratch: &mut RotationScratch<T>) {
        self.m2m.apply(o, input, output, scratch);
    }

    /// Adds the L2L of the parent local expansion `input` to the local expansion
    /// `output` of child octant `o` (CONVENTIONS §3.11, §3.12), by rotation, coaxial
    /// translation and rotation back. Both are scaled coefficients of degree p. Equals
    /// `nd_fmm_ref::rotation::l2l` at [`l2l_frames`]`(o)` to rounding.
    ///
    /// Accumulates, allocates nothing and builds no block. Cost:
    /// 3 (p + 1)(2p² + 7p + 2)/2 ≈ 3p³ multiply-adds and 2 (p + 1) additions.
    ///
    /// # Panics
    ///
    /// If `o >= 8`, `scratch` is not of degree p, or `input` or `output` does not have
    /// length (p + 1)².
    #[inline]
    pub fn l2l(&self, o: usize, input: &[T], output: &mut [T], scratch: &mut RotationScratch<T>) {
        self.l2l.apply(o, input, output, scratch);
    }

    /// Returns the same tables in precision `U`, every stored real rounded to nearest.
    pub fn cast<U: RealScalar>(&self) -> RotationTables<U> {
        RotationTables {
            p: self.p,
            m2m: self.m2m.cast(),
            l2l: self.l2l.cast(),
            m2l: self.m2l.cast(),
        }
    }
}

/// Caller-owned scratch of the table-driven rotation operators: two coefficient
/// vectors of degree p.
#[derive(Clone, Debug)]
pub struct RotationScratch<T: RealScalar> {
    first: Vec<T>,
    second: Vec<T>,
}

impl<T: RealScalar> RotationScratch<T> {
    /// Allocates the scratch for degree `p`, 2 (p + 1)² reals.
    pub fn new(p: usize) -> Self {
        let n = Layout::new(p).len();
        Self {
            first: vec![T::zero(); n],
            second: vec![T::zero(); n],
        }
    }
}

#[cfg(test)]
mod tests {
    use nd_fmm_ref::leaf;

    use super::*;

    /// A small deterministic generator of values in [−1, 1).
    struct Values(u64);

    impl Values {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    /// The largest absolute difference and the largest absolute value of `want`.
    fn difference(got: &[f64], want: &[f64]) -> (f64, f64) {
        let diff = got
            .iter()
            .zip(want)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let size = want.iter().fold(0.0_f64, |m, b| m.max(b.abs()));
        (diff, size)
    }

    #[test]
    fn azimuth_factors_are_cos_and_sin_of_m_phi() {
        // Error measure: absolute error of Cₘ and Sₘ against cos mφ and sin mφ, bound
        // 1e-14 (the recursion of `blocks` runs over m degrees).
        let p = 20;
        for operator in [Operator::M2l, Operator::M2m, Operator::L2l] {
            let tables = ShiftTables::build(p, operator);
            for a in 0..tables.azimuth_count() {
                let phi = tables.azimuth_angle(a);
                for (m, &[c_m, s_m]) in (1..=p).zip(tables.azimuth_factors(a).as_chunks::<2>().0) {
                    let (s, c) = (m as f64 * phi).sin_cos();
                    assert!((c_m - c).abs() <= 1e-14, "{operator:?} φ = {phi}, m = {m}");
                    assert!((s_m - s).abs() <= 1e-14, "{operator:?} φ = {phi}, m = {m}");
                }
            }
        }
    }

    #[test]
    fn z_rotation_equals_full_blocks_on_coefficient_vectors() {
        // The factorised z-rotation against the full blocks of both rules, T_M = K D K
        // and T_L = K S D S⁻¹ K of R_z(−φ) and of R_z(φ), from `nd_fmm_math::rotation`,
        // on random coefficient vectors (not only basis vectors).
        //
        // Error measure: largest absolute difference relative to the largest absolute
        // entry of the full-block result, bound 1e-14.
        let p = 12;
        let tables = ShiftTables::build(p, Operator::M2l);
        let n = Layout::new(p).len();
        let mut values = Values(0x7a11);
        for a in 0..tables.azimuth_count() {
            let phi = tables.azimuth_angle(a);
            for (inverse, angle) in [(false, -phi), (true, phi)] {
                let q = euler_zyz(0.0, 0.0, angle);
                for expansion in [Expansion::Multipole, Expansion::Local] {
                    let full = rule_blocks(p, &q, expansion);
                    let x: Vec<f64> = (0..n).map(|_| values.next()).collect();
                    let mut want = vec![0.0; n];
                    apply_blocks(p, &full, &x, &mut want);
                    let mut got = vec![0.0; n];
                    tables.rotate_z(a, inverse, &x, &mut got);
                    let (diff, size) = difference(&got, &want);
                    assert!(
                        diff <= 1e-14 * size,
                        "φ = {phi}, inverse = {inverse}, {expansion:?}: {diff:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn z_rotation_rotates_point_source_coefficients() {
        // Physical meaning (CONVENTIONS §3.11, "Rotation of coefficients"): for sources
        // y ↦ R_z(−φ) y about the frame centre, the P2M and P2L coefficients become
        // T(R_z(−φ)) times the old ones. Sources at |u| ≤ √3 for P2M and |u| ≥ 2 for
        // P2L (§3.9).
        //
        // Error measure: largest absolute difference relative to the largest absolute
        // coefficient, bound 1e-13 (the harmonics themselves carry rounding).
        let p = 10;
        let tables = ShiftTables::build(p, Operator::M2l);
        let n = Layout::new(p).len();
        let frame = Frame::new([0.0; 3], 1.0);
        let mut ws = Workspace::new(p);
        let mut values = Values(0x51de);
        for a in 0..tables.azimuth_count() {
            let q = euler_zyz(0.0, 0.0, -tables.azimuth_angle(a));
            let rotate = |y: [f64; 3]| q.map(|row| row[0] * y[0] + row[1] * y[1] + row[2] * y[2]);
            type P2x = fn(usize, &Frame<f64>, &[[f64; 3]], &[f64], &mut Workspace<f64>, &mut [f64]);
            for (name, p2x) in [("P2M", leaf::p2m::<f64> as P2x), ("P2L", leaf::p2l::<f64>)] {
                // |u| ≤ √3/2 for P2M; every component in [2, 10), so |u| ≥ 2√3, for P2L.
                let place = |v: f64| {
                    if name == "P2M" {
                        0.5 * v
                    } else {
                        4.0 * (1.5 + v)
                    }
                };
                let sources: Vec<[f64; 3]> = (0..4)
                    .map(|_| [0; 3].map(|_| place(values.next())))
                    .collect();
                let charges: Vec<f64> = (0..4).map(|_| values.next()).collect();
                let rotated: Vec<[f64; 3]> = sources.iter().map(|&y| rotate(y)).collect();
                let mut before = vec![0.0; n];
                p2x(p, &frame, &sources, &charges, &mut ws, &mut before);
                let mut want = vec![0.0; n];
                p2x(p, &frame, &rotated, &charges, &mut ws, &mut want);
                let mut got = vec![0.0; n];
                tables.rotate_z(a, false, &before, &mut got);
                let (diff, size) = difference(&got, &want);
                assert!(diff <= 1e-13 * size, "azimuth {a}, {name}: {diff:e}");
            }
        }
    }

    #[test]
    fn keys_group_the_shifts_exactly() {
        // Error measure: exact integer and bit comparisons. Entries share a polar angle
        // (azimuth, distance) exactly when their directions have the same θ (φ, |d|),
        // decided in integers: cos θ and the ray of (d_x, d_y) are compared by
        // cross-multiplication.
        let tables = ShiftTables::build(2, Operator::M2l);
        let offsets = m2l_offsets();
        for (s, a) in offsets.iter().enumerate() {
            for (t, b) in offsets.iter().enumerate() {
                let (sa, sb) = (tables.shift(s), tables.shift(t));
                let norm = |d: &[i64; 3]| d.iter().map(|v| v * v).sum::<i64>();
                assert_eq!(sa.distance == sb.distance, norm(a) == norm(b));
                if let (
                    Alignment::Rotated {
                        polar: pa,
                        azimuth: aa,
                    },
                    Alignment::Rotated {
                        polar: pb,
                        azimuth: ab,
                    },
                ) = (sa.alignment, sb.alignment)
                {
                    let same_polar = a[2].signum() == b[2].signum()
                        && a[2] * a[2] * norm(b) == b[2] * b[2] * norm(a);
                    assert_eq!(pa == pb, same_polar, "{a:?}, {b:?}");
                    let same_ray = a[0] * b[1] == a[1] * b[0] && a[0] * b[0] + a[1] * b[1] > 0;
                    assert_eq!(aa == ab, same_ray, "{a:?}, {b:?}");
                }
            }
        }
        // The distance of every entry is that nd-fmm-ref forms from its own shift.
        for (t, d) in offsets.iter().enumerate() {
            let (source, target) = m2l_frames(t);
            let v: [f64; 3] = core::array::from_fn(|i| target.centre[i] - source.centre[i]);
            let own = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let stored = tables.distance(tables.shift(t).distance);
            assert_eq!(own.to_bits(), stored.to_bits(), "{d:?}");
        }
    }
}
