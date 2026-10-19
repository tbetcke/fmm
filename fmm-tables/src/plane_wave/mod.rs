//! The tables of plane-wave M2L (CONVENTIONS §3.14; Phase 6 T13, C6.5): the quadrature,
//! the conversions between expansions and plane waves in six directions, the diagonal
//! shifts of merge and shift, and a pair translation as the trusted slower path.
//!
//! # The representation
//!
//! CONVENTIONS §3.14 derives it; in short, in box widths u = v/2 (half-width 1 in the
//! scaled coordinates v of §3.7) a scaled multipole has M̂ₙᵐ = 2^{−(n+1)} M̃ₙᵐ, and for a
//! target above the source (z > 0) the potential is a sum of plane waves over the nodes
//! (λₖ, αⱼ) of a [`Quadrature`]:
//!
//! X(λₖ, αⱼ) = (wₖ/Nₖ') [Aₖ(0) + Σ_{1≤m<Fₖ'} 2 (−i)ᵐ Re(Aₖ(m) e^{imαⱼ})],
//! Aₖ(m) = Σₙ λₖⁿ M̂ₙᵐ,
//!
//! with the down waves (z < 0) the same with a factor (−1)ⁿ⁻ᵐ on each term. Half of the
//! Nₖ' (even) α nodes carry the waves of real charges: H = Σₖ Nₖ'/2 complex values per
//! direction, stored as 2H reals, the H real parts and then the H imaginary parts, λₖ's
//! values at [`LambdaBlock::start`]. A shift by d (box widths, in the direction's frame)
//! multiplies node (k, j) by e^{∓λₖ d_z + iλₖ(d_x cos αⱼ + d_y sin αⱼ)} (up: −; down: +),
//! and the incoming waves give the local
//!
//! L̃ₙᵐ += 2^{−n} λₖⁿ (∓1)^{n−m} iᵐ Gₖ(m),  Gₖ(m) = Σⱼ V(λₖ, αⱼ) e^{−imαⱼ},
//!
//! over all Nₖ' nodes (up: (−1)^{n−m}; down: 1). ±y and ±x are ±z after the rotations of
//! [`FRAMES`]: the multipole rotates into the frame by K Dⁿ(Q) K and the local back by
//! K S Dⁿ(Qᵀ) S⁻¹ K (§3.8, §3.11 "Rotation of coefficients").
//!
//! # The steps, and who runs them
//!
//! [`PlaneWaveTables`] holds every table in `T` (built in f64, rounded entry by entry) and
//! runs the steps that carry the formulas; the loops over the H values, which dominate,
//! are plain arithmetic that `nd-fmm-simd` runs on vectors, with this module's functions as
//! their definition (the trusted slower path):
//!
//! | Step | Here | Work per box |
//! | --- | --- | --- |
//! | the multipole into a frame | [`rotate_multipole`](PlaneWaveTables::rotate_multipole) | O(p³) |
//! | its Fourier modes Aₖ(m), weighted | [`outgoing_modes`](PlaneWaveTables::outgoing_modes) | O(s p²) |
//! | the waves from the modes | [`waves_from_modes`] | O(H F) |
//! | a shift, an accumulation | [`shift_add`], [`add`], [`shift`] | O(H) |
//! | the sums Σⱼ x cos mαⱼ, Σⱼ x sin mαⱼ | [`sums_from_waves`] | O(H F) |
//! | the local from the sums | [`local_from_sums`](PlaneWaveTables::local_from_sums) | O(s p²) |
//! | the local back from the frame | [`add_local_from_frame`](PlaneWaveTables::add_local_from_frame) | O(p³) |
//!
//! F is the largest number of Fourier modes of a λ, at most p + 1. Each step has a fixed
//! order per element, the order its function documents. The loops a vector kernel runs
//! (the waves, the shifts and accumulations, the sums) use one fused multiply–add per
//! product and sum (`mul_add`), which every ISA computes alike, so the kernels give these
//! functions' bits on every ISA; the others round products and sums apart, so that a build
//! without the `fma` target feature calls no library function for them.
//!
//! [`PlaneWaveTables::m2l`] translates one pair by these steps, directly (one shift from
//! the source to the target): the slower path that the merge and shift of
//! `nd-fmm-exec` equals to rounding. Merge and shift itself (CONVENTIONS §3.14, "Merge
//! and shift") is an order of the same steps over a level; [`geometry`] holds its
//! directions and accumulators, and this type its shifts:
//! [`source_shift`](PlaneWaveTables::source_shift) from a slot to the parent's centre
//! and [`child_shift`](PlaneWaveTables::child_shift) from the parent's centre to a child.
//!
//! These tables are not cached: built in milliseconds (the shifts, 2H values each for
//! the 104 source offsets and 48 child offsets, and two rotations per frame), they are
//! not worth a file.

pub mod geometry;
pub mod quadrature;
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "FMM3D's literals keep their 36 digits, as in the Fortran source"
)]
mod quadrature_data;

use nd_fmm_math::RealScalar;
use nd_fmm_math::rotation::{block_range, blocks, blocks_len, to_irregular};

pub use geometry::{Accumulator, CHILDREN, Direction, MergeGeometry, SLOTS};
pub use quadrature::{MAX_DEGREE, Quadrature, nodes_for_degree};

use crate::geometry::{M2L_OFFSET_COUNT, m2l_offset_index, m2l_offsets};

/// The 316 V-list offsets in table order (CONVENTIONS §3.12).
const OFFSETS: [[i64; 3]; M2L_OFFSET_COUNT] = m2l_offsets();

/// The rotation of each frame, taking its directions to ±z: the identity (±z),
/// R_x(π/2) with (x, y, z) ↦ (x, −z, y) (±y), and R_y(−π/2) with (x, y, z) ↦ (−z, y, x)
/// (±x); [`Direction::rotate`] applies them to vectors.
pub const FRAMES: [[[f64; 3]; 3]; 3] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
];

/// The values of one λ node in the waves and the tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LambdaBlock {
    /// Its first value among the H complex values of a direction.
    pub start: usize,
    /// Its values, Nₖ'/2.
    pub len: usize,
    /// Its Fourier modes Fₖ' = min(Fₖ, p + 1): orders 0 ≤ m < Fₖ'.
    pub modes: usize,
    /// Its first mode among all the modes, Σ_{l<k} F_l'.
    pub mode_start: usize,
    /// Its first value in [`PlaneWaveTables::cos`] and [`PlaneWaveTables::sin`]: Fₖ' rows
    /// of `len` values, row m at `trig + m len`.
    pub trig: usize,
}

/// The plane-wave tables of one degree; see the [module documentation](self).
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveTables<T: RealScalar> {
    p: usize,
    quadrature: Quadrature,
    blocks: Vec<LambdaBlock>,
    /// H.
    h: usize,
    /// Σₖ Fₖ'.
    modes: usize,
    /// Per degree n, s values: (wₖ/Nₖ') λₖⁿ 2^{−(n+1)} by λ node (the powers of
    /// [`outgoing_modes`](Self::outgoing_modes), by degree so that a sweep runs over k).
    outgoing_powers: Vec<T>,
    /// Per λ node, p + 1 values: 2 λₖⁿ 2^{−n} ([`local_from_sums`](Self::local_from_sums)).
    incoming_powers: Vec<T>,
    /// cos mαⱼ and sin mαⱼ per λ: at `blocks[k].trig + m len + j`.
    cos: Vec<T>,
    sin: Vec<T>,
    /// The multipole rotations K Dⁿ(Q) K of frames 1 and 2, per degree, transposed.
    rotate_multipole: [Vec<T>; 2],
    /// The local rotations K S Dⁿ(Qᵀ) S⁻¹ K back from frames 1 and 2, transposed.
    rotate_local: [Vec<T>; 2],
    geometry: MergeGeometry,
    /// The distinct source shifts, 2H values each.
    source_shifts: Vec<T>,
    /// Per direction and slot, the index of its source shift, or `u16::MAX`.
    source_shift_index: Vec<[u16; SLOTS]>,
    /// Per direction and child, its child shift, 2H values each.
    child_shifts: Vec<T>,
}

/// The scratch of [`PlaneWaveTables::m2l`]: a frame's coefficients, the waves and modes of
/// one translation, and the shift factors of its offset. Made by
/// [`PlaneWaveTables::scratch`].
#[derive(Clone, Debug)]
pub struct PlaneWaveScratch<T> {
    frame: Vec<T>,
    up: Vec<T>,
    down: Vec<T>,
    incoming: Vec<T>,
    modes: Vec<T>,
    factors: Vec<T>,
    work: Vec<T>,
}

/// One value of `T` from f64, rounded to nearest.
fn cast<T: RealScalar>(x: f64) -> T {
    T::from_f64(x)
}

/// Per degree, the transposed blocks K B K, or with `irregular` K S B S⁻¹ K, of the
/// rotation `q`: entry (i, j) of degree n at `block_range(n).start + j (2n + 1) + i`.
fn frame_blocks<T: RealScalar>(p: usize, q: &[[f64; 3]; 3], irregular: bool) -> Vec<T> {
    let mut b = vec![0.0; blocks_len(p)];
    blocks(p, q, &mut b);
    if irregular {
        to_irregular(p, &mut b);
    }
    let mut out = vec![T::zero(); b.len()];
    for n in 0..=p {
        let w = 2 * n + 1;
        let range = block_range(n);
        let k = |i: usize| if i < n { -1.0 } else { 1.0 };
        for i in 0..w {
            for j in 0..w {
                out[range.start + j * w + i] = cast(k(i) * k(j) * b[range.start + i * w + j]);
            }
        }
    }
    out
}

/// The transpose of a 3 × 3 matrix.
fn transpose(q: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| q[j][i]))
}

/// out_n += B_n in_n per degree, B stored transposed ([`frame_blocks`]): for every
/// output value the columns j in ascending order, a product and a sum each, rounded
/// apart (no fma: on an x86_64 build without the `fma` feature `mul_add` is a library
/// call).
fn apply_blocks<T: RealScalar>(p: usize, blocks_t: &[T], input: &[T], out: &mut [T]) {
    for n in 0..=p {
        let w = 2 * n + 1;
        let b = &blocks_t[block_range(n)];
        let (start, out) = (n * n, &mut out[n * n..n * n + w]);
        for (j, column) in b.chunks_exact(w).enumerate() {
            let x = input[start + j];
            for (o, &bij) in out.iter_mut().zip(column) {
                *o = *o + bij * x;
            }
        }
    }
}

impl<T: RealScalar> PlaneWaveTables<T> {
    /// The tables of degree `p` with the quadrature of [`nodes_for_degree`]`(p)`.
    ///
    /// # Panics
    ///
    /// If `p` > [`MAX_DEGREE`].
    pub fn build(p: usize) -> Self {
        Self::with_quadrature(p, Quadrature::for_degree(p))
    }

    /// The tables of degree `p` with the quadrature `quadrature` (for tests and studies
    /// of the quadrature; the FMM uses [`build`](Self::build)). Builds every table in
    /// f64 and rounds it to `T`; serial and deterministic.
    pub fn with_quadrature(p: usize, quadrature: Quadrature) -> Self {
        let even = quadrature.even_alpha_nodes();
        let s = quadrature.nodes();
        let mut blocks_ = Vec::with_capacity(s);
        let (mut start, mut mode_start, mut trig) = (0, 0, 0);
        for (k, &nodes) in even.iter().enumerate() {
            let (len, modes) = (nodes / 2, quadrature.modes(k, p));
            blocks_.push(LambdaBlock {
                start,
                len,
                modes,
                mode_start,
                trig,
            });
            start += len;
            mode_start += modes;
            trig += modes * len;
        }
        let (h, modes) = (start, mode_start);
        let mut outgoing_powers = vec![T::zero(); s * (p + 1)];
        let mut incoming_powers = vec![T::zero(); s * (p + 1)];
        for (k, &lambda) in quadrature.lambdas.iter().enumerate() {
            let weight = quadrature.weights[k] / even[k] as f64;
            for n in 0..=p {
                let power = lambda.powi(n as i32);
                outgoing_powers[n * s + k] = cast(weight * power / 2f64.powi(n as i32 + 1));
                incoming_powers[k * (p + 1) + n] = cast(2.0 * power / 2f64.powi(n as i32));
            }
        }
        let (mut cos, mut sin) = (Vec::with_capacity(trig), Vec::with_capacity(trig));
        for (block, &nodes) in blocks_.iter().zip(&even) {
            for m in 0..block.modes {
                for j in 0..block.len {
                    let angle = core::f64::consts::TAU * (m * j) as f64 / nodes as f64;
                    cos.push(cast(angle.cos()));
                    sin.push(cast(angle.sin()));
                }
            }
        }
        let mut pairs = Vec::with_capacity(2 * trig);
        for block in &blocks_ {
            for j in 0..block.len {
                for m in (0..block.modes)
                    .step_by(2)
                    .chain((1..block.modes).step_by(2))
                {
                    let row = block.trig + m * block.len + j;
                    pairs.push(cos[row]);
                    pairs.push(sin[row]);
                }
            }
        }
        let rotate_multipole = [1, 2].map(|f| frame_blocks(p, &FRAMES[f], false));
        let rotate_local = [1, 2].map(|f| frame_blocks(p, &transpose(&FRAMES[f]), true));

        let mut tables = Self {
            p,
            quadrature,
            blocks: blocks_,
            h,
            modes,
            outgoing_powers,
            incoming_powers,
            cos,
            sin,
            rotate_multipole,
            rotate_local,
            geometry: MergeGeometry::new(),
            source_shifts: Vec::new(),
            source_shift_index: vec![[u16::MAX; SLOTS]; 6],
            child_shifts: Vec::with_capacity(6 * CHILDREN * 2 * h),
        };
        // The source shifts, by their offset in the frame in half box widths: from the
        // slot's centre to the parent's, 5 − 2a per axis (the parent's centre 2P + 1, the
        // slot's 2P − 2 + a + ½, in box widths).
        let mut keys: Vec<[i64; 3]> = Vec::new();
        let mut shifts = Vec::new();
        for direction in Direction::ALL {
            for accumulator in tables.geometry.accumulators(direction) {
                for &slot in &accumulator.slots {
                    let a = geometry::slot_coordinates(slot as usize);
                    let key = direction.rotate(a.map(|x| 5 - 2 * x));
                    let index = keys.iter().position(|&k| k == key).unwrap_or_else(|| {
                        keys.push(key);
                        shifts.extend(tables.factors_f64(key.map(|x| x as f64 / 2.0), direction));
                        keys.len() - 1
                    });
                    tables.source_shift_index[direction.index()][slot as usize] =
                        u16::try_from(index).expect("fewer than 2¹⁶ shifts");
                }
            }
        }
        tables.source_shifts = shifts.into_iter().map(cast).collect();
        // The child shifts: from the parent's centre to the child's, 2b − 1 half widths
        // per axis.
        for direction in Direction::ALL {
            for child in 0..CHILDREN {
                let b = geometry::child_bits(child);
                let offset = direction.rotate(b.map(|x| (2 * x - 1) as f64 / 2.0));
                let factors = tables.factors_f64(offset, direction);
                tables
                    .child_shifts
                    .extend(factors.into_iter().map(cast::<T>));
            }
        }
        tables
    }

    /// The shift factors of the offset `offset` (box widths, in the frame of `direction`):
    /// e^{∓λ v + iλ(a cos α + b sin α)} for (a, b, v) = `offset`, − for an up direction
    /// and + for a down one, as 2H values in f64.
    fn factors_f64(&self, offset: [f64; 3], direction: Direction) -> Vec<f64> {
        let mut t = vec![0.0; 2 * self.h];
        let even = self.quadrature.even_alpha_nodes();
        let sign = if direction.is_up() { -1.0 } else { 1.0 };
        for (k, &lambda) in self.quadrature.lambdas.iter().enumerate() {
            let decay = (sign * lambda * offset[2]).exp();
            let block = self.blocks[k];
            for j in 0..block.len {
                let alpha = core::f64::consts::TAU * j as f64 / even[k] as f64;
                let phase = lambda * (offset[0] * alpha.cos() + offset[1] * alpha.sin());
                t[block.start + j] = decay * phase.cos();
                t[self.h + block.start + j] = decay * phase.sin();
            }
        }
        t
    }

    /// The degree p.
    pub fn p(&self) -> usize {
        self.p
    }

    /// The quadrature.
    pub fn quadrature(&self) -> &Quadrature {
        &self.quadrature
    }

    /// H, the complex values of the waves of one direction.
    pub fn h(&self) -> usize {
        self.h
    }

    /// The values of the waves of one direction, 2H: the H real parts, then the H
    /// imaginary parts.
    pub fn wave_len(&self) -> usize {
        2 * self.h
    }

    /// The Fourier modes of all the λ nodes, Σₖ Fₖ': the modes of
    /// [`outgoing_modes`](Self::outgoing_modes) and the sums of [`sums_from_waves`] hold
    /// four values per mode.
    pub fn mode_count(&self) -> usize {
        self.modes
    }

    /// The λ nodes' blocks.
    pub fn blocks(&self) -> &[LambdaBlock] {
        &self.blocks
    }

    /// cos mαⱼ per λ node, `blocks()[k].trig + m len + j`.
    pub fn cos(&self) -> &[T] {
        &self.cos
    }

    /// sin mαⱼ per λ node, as [`cos`](Self::cos).
    pub fn sin(&self) -> &[T] {
        &self.sin
    }

    /// The accumulators of merge and shift.
    pub fn geometry(&self) -> &MergeGeometry {
        &self.geometry
    }

    /// The shift of the waves of `slot` in `direction` to the parent's centre (2H values),
    /// or `None` if no pair of the slot has that direction.
    pub fn source_shift(&self, direction: Direction, slot: usize) -> Option<&[T]> {
        let index = self.source_shift_index[direction.index()][slot];
        (index != u16::MAX).then(|| {
            let len = self.wave_len();
            &self.source_shifts[index as usize * len..(index as usize + 1) * len]
        })
    }

    /// The distinct source shifts, 2H values each:
    /// [`source_shift_offset`](Self::source_shift_offset) gives the first of each.
    pub fn source_shifts(&self) -> &[T] {
        &self.source_shifts
    }

    /// The first value of the source shift of `slot` in `direction` in
    /// [`source_shifts`](Self::source_shifts), or `None` if no pair of the slot has that
    /// direction.
    pub fn source_shift_offset(&self, direction: Direction, slot: usize) -> Option<usize> {
        let index = self.source_shift_index[direction.index()][slot];
        (index != u16::MAX).then(|| index as usize * self.wave_len())
    }

    /// The number of distinct source shifts.
    pub fn source_shift_count(&self) -> usize {
        self.source_shifts.len() / self.wave_len().max(1)
    }

    /// The shift of waves in `direction` from the parent's centre to the child `child`
    /// (2H values).
    pub fn child_shift(&self, direction: Direction, child: usize) -> &[T] {
        let len = self.wave_len();
        let i = direction.index() * CHILDREN + child;
        &self.child_shifts[i * len..(i + 1) * len]
    }

    /// The multipole `multipole` ((p + 1)² scaled coefficients, §3.6) in `frame`, written
    /// to `out`: per degree n and output slot, the input slots in ascending order, a
    /// product and a sum each, from zero. Frame 0 copies.
    ///
    /// # Panics
    ///
    /// If `frame` ≥ 3 or a slice does not hold (p + 1)² values.
    pub fn rotate_multipole(&self, frame: usize, multipole: &[T], out: &mut [T]) {
        let n = (self.p + 1).pow(2);
        assert!(
            multipole.len() == n && out.len() == n,
            "(p + 1)² coefficients"
        );
        if frame == 0 {
            out.copy_from_slice(multipole);
        } else {
            out.fill(T::zero());
            apply_blocks(self.p, &self.rotate_multipole[frame - 1], multipole, out);
        }
    }

    /// Adds the local `local` in `frame` ((p + 1)² scaled coefficients), rotated back to
    /// the box's frame, to `out`: frame 0 adds it as it is, the others per degree n and
    /// output slot over the input slots in ascending order, a product and a sum each.
    ///
    /// # Panics
    ///
    /// If `frame` ≥ 3 or a slice does not hold (p + 1)² values.
    pub fn add_local_from_frame(&self, frame: usize, local: &[T], out: &mut [T]) {
        let n = (self.p + 1).pow(2);
        assert!(local.len() == n && out.len() == n, "(p + 1)² coefficients");
        if frame == 0 {
            for (o, &l) in out.iter_mut().zip(local) {
                *o = *o + l;
            }
        } else {
            apply_blocks(self.p, &self.rotate_local[frame - 1], local, out);
        }
    }

    /// The values of the work buffer of [`outgoing_modes`](Self::outgoing_modes):
    /// 4 (p + 1) s.
    pub fn work_len(&self) -> usize {
        4 * (self.p + 1) * self.blocks.len()
    }

    /// The weighted Fourier modes of the multipole `multipole`, already in its frame, for
    /// the up and down waves of the frame: four values per mode (k, m), at
    /// 4 (`mode_start` + m), the coefficients (a↑, b↑, a↓, b↓) of
    /// [`waves_from_modes`]: the waves of node (k, j) are Σₘ a cos mαⱼ + b sin mαⱼ over the
    /// even m into the real parts and over the odd m into the imaginary parts
    /// (CONVENTIONS §3.14, "Multipole to waves"). `work` holds at least
    /// [`work_len`](Self::work_len) values; its contents do not matter.
    ///
    /// Each Aₖ(m) sums its degrees n ascending, even and odd n − m apart, from zero, a
    /// product and a sum each (rounded apart, as every step of this type outside the vector
    /// kernels' loops), as sweeps over the λ nodes for each (m, n).
    ///
    /// # Panics
    ///
    /// If `multipole` does not hold (p + 1)² values, `modes` 4 [`mode_count`](Self::mode_count)
    /// or `work` fewer than [`work_len`](Self::work_len).
    pub fn outgoing_modes(&self, multipole: &[T], modes: &mut [T], work: &mut [T]) {
        let (p, s) = (self.p, self.blocks.len());
        assert_eq!(multipole.len(), (p + 1) * (p + 1), "(p + 1)² coefficients");
        assert_eq!(modes.len(), 4 * self.modes, "4 values per mode");
        // A[q][m][k], q = even real, even imaginary, odd real, odd imaginary parts.
        let plane = (p + 1) * s;
        let a = &mut work[..4 * plane];
        a.fill(T::zero());
        let (even, odd) = a.split_at_mut(2 * plane);
        for m in 0..=p {
            for (parts, first) in [(&mut *even, m), (&mut *odd, m + 1)] {
                let (re, im) = parts.split_at_mut(plane);
                let (re, im) = (&mut re[m * s..(m + 1) * s], &mut im[m * s..(m + 1) * s]);
                for n in (first..=p).step_by(2) {
                    let powers = &self.outgoing_powers[n * s..(n + 1) * s];
                    let x = multipole[n * n + n + m];
                    for (a, &power) in re.iter_mut().zip(powers) {
                        *a = *a + power * x;
                    }
                    if m > 0 {
                        // The slot −m only for m > 0: A's imaginary part is zero at m = 0.
                        let x = multipole[n * n + n - m];
                        for (a, &power) in im.iter_mut().zip(powers) {
                            *a = *a + power * x;
                        }
                    }
                }
            }
        }
        let two = cast::<T>(2.0);
        for (k, block) in self.blocks.iter().enumerate() {
            for m in 0..block.modes {
                let at = m * s + k;
                let (er, ei, or, oi) = (even[at], even[plane + at], odd[at], odd[plane + at]);
                let out = &mut modes[4 * (block.mode_start + m)..4 * (block.mode_start + m + 1)];
                for (pair, (ar, ai)) in out
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip([(er + or, ei + oi), (er - or, ei - oi)])
                {
                    if m == 0 {
                        pair[0] = ar;
                        pair[1] = T::zero();
                        continue;
                    }
                    // 2 (−i)ᵐ Re(A e^{imα}) = 2 (−i)ᵐ (A_r cos − A_i sin): into the real
                    // part for even m, the imaginary part for odd m, with the sign of (−i)ᵐ.
                    let sign = if m % 4 < 2 { two } else { -two };
                    let (a, b) = if m % 2 == 0 {
                        (sign * ar, -(sign * ai))
                    } else {
                        (-(sign * ar), sign * ai)
                    };
                    pair[0] = a;
                    pair[1] = b;
                }
            }
        }
    }

    /// Writes the local of the incoming sums `sums` (four per mode, [`sums_from_waves`])
    /// into `local`, in the frame of the waves (CONVENTIONS §3.14, "Waves to local"):
    /// every slot (n, ±m) from zero, then for each λ node and mode in order the degrees
    /// n ≥ m ascending, a product and a sum each into the slots of (n, ±m).
    ///
    /// # Panics
    ///
    /// If `sums` does not hold 4 [`mode_count`](Self::mode_count) values or `local`
    /// (p + 1)².
    pub fn local_from_sums(&self, sums: &[T], local: &mut [T]) {
        let p = self.p;
        assert_eq!(sums.len(), 4 * self.modes, "4 sums per mode");
        assert_eq!(local.len(), (p + 1) * (p + 1), "(p + 1)² coefficients");
        local.fill(T::zero());
        for (k, block) in self.blocks.iter().enumerate() {
            let power = &self.incoming_powers[k * (p + 1)..(k + 1) * (p + 1)];
            for m in 0..block.modes {
                let s = &sums[4 * (block.mode_start + m)..4 * (block.mode_start + m + 1)];
                // G(m) / 2 over all Nₖ' nodes from half of them: (C, −S) for even m from
                // the real parts, (S, C) for odd m from the imaginary parts.
                let g = |c: T, s: T| if m % 2 == 0 { (c, -s) } else { (s, c) };
                let (up, down) = (g(s[0], s[1]), g(s[2], s[3]));
                // n − m even: down + up; odd: down − up (the up waves' (−1)^{n−m}); then
                // times iᵐ.
                let rot = |(a, b): (T, T)| match m % 4 {
                    0 => (a, b),
                    1 => (-b, a),
                    2 => (-a, -b),
                    _ => (b, -a),
                };
                let even = rot((up.0 + down.0, up.1 + down.1));
                let odd = rot((down.0 - up.0, down.1 - up.1));
                for (n, &power) in power.iter().enumerate().skip(m) {
                    let (re, im) = if (n - m) % 2 == 0 { even } else { odd };
                    let slot = n * n + n + m;
                    local[slot] = local[slot] + power * re;
                    if m > 0 {
                        let slot = n * n + n - m;
                        local[slot] = local[slot] + power * im;
                    }
                }
            }
        }
    }

    /// The scratch of [`m2l`](Self::m2l).
    pub fn scratch(&self) -> PlaneWaveScratch<T> {
        let (n, w) = ((self.p + 1).pow(2), self.wave_len());
        PlaneWaveScratch {
            frame: vec![T::zero(); n],
            up: vec![T::zero(); w],
            down: vec![T::zero(); w],
            incoming: vec![T::zero(); w],
            modes: vec![T::zero(); 4 * self.modes],
            factors: vec![T::zero(); w],
            work: vec![T::zero(); self.work_len()],
        }
    }

    /// Adds the plane-wave M2L of the V pair with offset index `offset_index` (§3.12)
    /// along `direction` from `multipole` to `local`: the source's waves in the
    /// direction's frame, one shift by the pair's offset, the local, and back to the box's
    /// frame. The trusted slower path of merge and shift, which assigns the pair the
    /// direction of [`Direction::for_slot`]; any direction along which the offset is 2 or
    /// 3 boxes lies in the quadrature's region. Allocates nothing; the shift factors are
    /// computed in f64 per call.
    ///
    /// # Panics
    ///
    /// If `offset_index` ≥ 316, the offset is not 2 or 3 boxes along `direction`, a slice
    /// does not hold (p + 1)² values, or `scratch` was not made by these tables.
    pub fn m2l(
        &self,
        offset_index: usize,
        direction: Direction,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut PlaneWaveScratch<T>,
    ) {
        let d = OFFSETS[offset_index];
        let offset = direction.rotate(d);
        let along = if direction.is_up() {
            offset[2]
        } else {
            -offset[2]
        };
        assert!(
            (2..=3).contains(&along),
            "the offset {d:?} is not 2 or 3 boxes along {direction:?}"
        );
        let PlaneWaveScratch {
            frame,
            up,
            down,
            incoming,
            modes,
            factors,
            work,
        } = scratch;
        self.rotate_multipole(direction.frame(), multipole, frame);
        self.outgoing_modes(frame, modes, work);
        waves_from_modes(self, modes, up, down);
        let wave = if direction.is_up() { &*up } else { &*down };
        for (f, x) in factors
            .iter_mut()
            .zip(self.factors_f64(offset.map(|x| x as f64), direction))
        {
            *f = cast(x);
        }
        incoming.fill(T::zero());
        shift_add(incoming, wave, factors);
        up.fill(T::zero());
        let (up_in, down_in): (&[T], &[T]) = if direction.is_up() {
            (incoming, up)
        } else {
            (up, incoming)
        };
        sums_from_waves(self, up_in, down_in, modes);
        self.local_from_sums(modes, frame);
        self.add_local_from_frame(direction.frame(), frame, local);
    }

    /// [`m2l`](Self::m2l) along [`Direction::of`] the offset: the first axis of z, y, x
    /// along which the target lies two or three boxes from the source.
    ///
    /// # Panics
    ///
    /// As [`m2l`](Self::m2l).
    pub fn m2l_of_offset(
        &self,
        offset_index: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut PlaneWaveScratch<T>,
    ) {
        let direction =
            Direction::of(OFFSETS[offset_index]).expect("a V-list offset has a direction");
        self.m2l(offset_index, direction, multipole, local, scratch);
    }

    /// The direction merge and shift gives the V pair with offset index `offset_index` to
    /// a target with child index `child` ([`Direction::for_slot`]).
    ///
    /// # Panics
    ///
    /// If `offset_index` ≥ 316 or `child` ≥ 8.
    pub fn pair_direction(offset_index: usize, child: usize) -> Direction {
        let d = OFFSETS[offset_index];
        debug_assert_eq!(m2l_offset_index(d), Some(offset_index));
        let slot = geometry::slot_of(child, d).expect("a V pair lies in the parent's slots");
        Direction::for_slot(slot, child).expect("a V pair has a direction")
    }
}

/// The up and down waves (2H values each) from the modes `modes` of
/// [`PlaneWaveTables::outgoing_modes`] (CONVENTIONS §3.14, "Multipole to waves"): for each
/// λ node k and value j, up and down from zero, then for every mode m ascending
/// x ← fma(a, cos mαⱼ, x), x ← fma(b, sin mαⱼ, x) into the real part for even m and the
/// imaginary part for odd m. Writes every value of `up` and `down`.
///
/// # Panics
///
/// If a slice does not have its length.
pub fn waves_from_modes<T: RealScalar>(
    tables: &PlaneWaveTables<T>,
    modes: &[T],
    up: &mut [T],
    down: &mut [T],
) {
    let h = tables.h;
    assert!(
        modes.len() == 4 * tables.modes && up.len() == 2 * h && down.len() == 2 * h,
        "4 values per mode and 2H per direction"
    );
    up.fill(T::zero());
    down.fill(T::zero());
    for block in &tables.blocks {
        let len = block.len;
        for m in 0..block.modes {
            let c = &tables.cos[block.trig + m * len..block.trig + (m + 1) * len];
            let s = &tables.sin[block.trig + m * len..block.trig + (m + 1) * len];
            let k = 4 * (block.mode_start + m);
            let half = if m % 2 == 0 { 0 } else { h };
            for (wave, (a, b)) in [
                (&mut *up, (modes[k], modes[k + 1])),
                (&mut *down, (modes[k + 2], modes[k + 3])),
            ] {
                let x = &mut wave[half + block.start..half + block.start + len];
                for ((x, &c), &s) in x.iter_mut().zip(c).zip(s) {
                    *x = a.mul_add(c, *x);
                    *x = b.mul_add(s, *x);
                }
            }
        }
    }
}

/// The incoming sums of the up and down waves `up`, `down` (2H values each): four values
/// per mode (k, m), (Σⱼ xⱼ↑ cos mαⱼ, Σⱼ xⱼ↑ sin mαⱼ, Σⱼ xⱼ↓ cos mαⱼ, Σⱼ xⱼ↓ sin mαⱼ) with
/// x the real parts for even m and the imaginary parts for odd m, each over j ascending
/// from zero, one fma per term (CONVENTIONS §3.14, "Waves to local"). Writes every value
/// of `sums`.
///
/// # Panics
///
/// If a slice does not have its length.
pub fn sums_from_waves<T: RealScalar>(
    tables: &PlaneWaveTables<T>,
    up: &[T],
    down: &[T],
    sums: &mut [T],
) {
    let h = tables.h;
    assert!(
        sums.len() == 4 * tables.modes && up.len() == 2 * h && down.len() == 2 * h,
        "4 sums per mode and 2H values per direction"
    );
    for block in &tables.blocks {
        let len = block.len;
        for m in 0..block.modes {
            let c = &tables.cos[block.trig + m * len..block.trig + (m + 1) * len];
            let s = &tables.sin[block.trig + m * len..block.trig + (m + 1) * len];
            let half = if m % 2 == 0 { 0 } else { h };
            let k = 4 * (block.mode_start + m);
            for (q, wave) in [up, down].into_iter().enumerate() {
                let x = &wave[half + block.start..half + block.start + len];
                let (mut sc, mut ss) = (T::zero(), T::zero());
                for ((&x, &c), &s) in x.iter().zip(c).zip(s) {
                    sc = x.mul_add(c, sc);
                    ss = x.mul_add(s, ss);
                }
                sums[k + 2 * q] = sc;
                sums[k + 2 * q + 1] = ss;
            }
        }
    }
}

/// acc += x ⊙ t, complex, over waves of 2H values (the real parts, then the imaginary
/// parts): per value, re ← fma(x_r, t_r, re), re ← fma(−x_i, t_i, re), im ← fma(x_r, t_i,
/// im), im ← fma(x_i, t_r, im).
///
/// # Panics
///
/// If the slices differ in length or have an odd one.
pub fn shift_add<T: RealScalar>(acc: &mut [T], x: &[T], t: &[T]) {
    assert!(acc.len() == x.len() && x.len() == t.len() && acc.len().is_multiple_of(2));
    let h = acc.len() / 2;
    let (ar, ai) = acc.split_at_mut(h);
    let ((xr, xi), (tr, ti)) = (x.split_at(h), t.split_at(h));
    for i in 0..h {
        ar[i] = xr[i].mul_add(tr[i], ar[i]);
        ar[i] = (-xi[i]).mul_add(ti[i], ar[i]);
        ai[i] = xr[i].mul_add(ti[i], ai[i]);
        ai[i] = xi[i].mul_add(tr[i], ai[i]);
    }
}

/// v ← v ⊙ t, complex, over waves of 2H values: per value, re ← v_r t_r − v_i t_i with the
/// product v_r t_r rounded and the rest fused (fma(−v_i, t_i, v_r t_r)), im ← fma(v_r,
/// t_i, v_i t_r).
///
/// # Panics
///
/// If the slices differ in length or have an odd one.
pub fn shift<T: RealScalar>(v: &mut [T], t: &[T]) {
    assert!(v.len() == t.len() && v.len().is_multiple_of(2));
    let h = v.len() / 2;
    let (vr, vi) = v.split_at_mut(h);
    let (tr, ti) = t.split_at(h);
    for i in 0..h {
        let (a, b) = (vr[i], vi[i]);
        vr[i] = (-b).mul_add(ti[i], a * tr[i]);
        vi[i] = a.mul_add(ti[i], b * tr[i]);
    }
}

/// v += a, value by value.
///
/// # Panics
///
/// If the slices differ in length.
pub fn add<T: RealScalar>(v: &mut [T], a: &[T]) {
    assert_eq!(v.len(), a.len());
    for (v, &a) in v.iter_mut().zip(a) {
        *v = *v + a;
    }
}

#[cfg(test)]
mod tests;
