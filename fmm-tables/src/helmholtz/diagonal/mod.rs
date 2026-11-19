//! S → F and F → R of the diagonal form as products over a batch of boxes (proposal §5.4,
//! §6.7; CONVENTIONS §3.15 "Signatures"; Helmholtz CH4.1a).
//!
//! [`nd_fmm_ref::helmholtz::diagonal`] converts one box at a time, ring by ring: per order a
//! Legendre sum, then the discrete Fourier transform along the ring written as a product. The
//! reference is O(p³) per box with a large constant (proposal §6.7: about 2P³ complex
//! multiply-adds for the Legendre step and 16P³ for the φ step, P = p + 1). [`Conversions`]
//! computes the same transforms, to rounding, with less work and in the shape of a product:
//!
//! - **Legendre step, halved by parity.** The rings are the Gauss–Legendre nodes, symmetric
//!   about the equator: ring 2p − i has u = −uᵢ, and P̄ₙᵐ(−u) = (−1)^{n+m} P̄ₙᵐ(u). Per order
//!   μ the sums over n with n + μ even (E) and odd (O) are formed on the rings 0 ≤ i ≤ p only;
//!   ring i takes E + O and ring 2p − i takes E − O. The θ-factors are real, so the four real
//!   planes of a box (the real and imaginary parts of the coefficients +μ and −μ) share them:
//!   per order a real matrix of rings × degrees times a matrix of degrees × (4 × boxes).
//! - **φ step, quartered by the azimuths' symmetries.** With aᵤ and bᵤ the ring sums of the
//!   coefficients +μ and −μ, F(φ) = Σᵤ aᵤ cos μφ + bᵤ sin μφ (the reference's form), a real
//!   transform of each complex part. N_φ is a multiple of 4, so every azimuth is
//!   φ_J = σ φⱼ + q π/2 with σ = ±1, q ∈ {0, 1, 2, 3} and a base azimuth 0 ≤ j ≤ N_φ/8. For
//!   the orders μ ≡ r (mod 4) the shift by q π/2 turns cos μφ and sin μφ into ± one of them,
//!   by the residue t = r q mod 4. So the sums Ca = Σ aᵤ cos μφⱼ, Sa = Σ aᵤ sin μφⱼ, Cb and Sb
//!   per residue at the base azimuths give every azimuth by signs and one add:
//!
//!   ```text
//!   t = 0: Ca + σ Sb    t = 1: Cb − σ Sa    t = 2: −(Ca + σ Sb)    t = 3: −(Cb − σ Sa)
//!   F(φ_J) = ((X₀ + X₁) + X₂) + X₃    (Xᵣ the residue's term; residues without orders left out)
//!   ```
//!
//!   per residue one real product [cos; sin] (2 × bases × orders) times (orders × columns).
//!   That is about a quarter of the reference's φ work: N_φ/8 + 1 base azimuths, each with
//!   four products per order, against N_φ azimuths with two.
//! - **F → R** is the transpose: the samples folded onto the base azimuths with the same
//!   signs, per residue the transposed product, the rings folded into E and O sums, the
//!   transposed Legendre product with the quadrature weights folded into its table.
//!
//! Per box and conversion this is about 2P³ real multiply-adds for the Legendre step and 8P³
//! for the φ step ([`Conversions::multiply_adds`]), 10P³ in all, against about 68P³ real ones in
//! proposal §6.7's count (2P³ complex-by-real for the Legendre step, 16P³ complex for the φ
//! step).
//!
//! # Tables
//!
//! Built in f64 from `nd_fmm_ref::helmholtz::diagonal::Grid` (its θ-factors
//! [`Grid::theta`], weights [`Grid::weight`] and azimuth tables [`Grid::azimuth_tables`]) and
//! the level's power scale s of §3.15, each entry rounded once to `T`: the Legendre tables
//! hold Θₙᵘ(uᵢ) sⁿ (S → F) and Θₙᵘ(uᵢ) sⁿ wᵢ 2π/N_φ (F → R), the φ tables cos μφⱼ and
//! sin μφⱼ at the base azimuths, cos and sin of the grid's table at (μ j) mod N_φ. Nothing is
//! derived here that the reference does not hold; the reference multiplies by sⁿ and the
//! weight in the working precision instead, so the two agree to rounding, not bit for bit.
//!
//! # Determinism
//!
//! Every product goes through one column kernel (`gemm_acc`): each element of the result is
//! the chain c ← c + a·x over the inner index in ascending order, the product and the sum
//! rounded apart (Rust never contracts them), from its own row and column only. The folds
//! and combinations are per element in a fixed order. So a box's signature and local depend
//! on its own input alone, bit for bit: not on the batch it is converted in, its position in
//! it, the block of boxes or the thread (tested). The rings are processed in blocks of
//! [`RING_BLOCK`]; F → R's Legendre chain continues across the blocks in ring order, so the
//! block size does not change a bit either.
//!
//! # Storage
//!
//! Multipoles and locals are §3.15's stored complex expansions of degree p, two real planes
//! of (p + 1)² (real, then imaginary) at idx(n, m) = n² + n + m, box after box. A signature
//! is the grid's (2p + 1) N_φ complex samples at i N_φ + j (CONVENTIONS §3.15), interleaved,
//! box after box. The level's buffers ([`Signatures`]: each box on its own 64-byte line), the
//! level driver ([`m2l_level`]) and its scratch under a budget ([`scratch_shape`]) are the
//! module [`level`]'s (Helmholtz CH4.1b), the layout fixed there.
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::helmholtz::diagonal::{self, Grid};
//! use nd_fmm_ref::helmholtz::Level;
//! use nd_fmm_tables::helmholtz::diagonal::{ConversionScratch, Conversions};
//!
//! let p = 6;
//! let level = Level::new(Complex::new(20.0_f64, 0.0), 0.5); // s = 1
//! let conversions = Conversions::<f64>::new(p, level.scale());
//! let mut scratch = ConversionScratch::new(&conversions);
//! let len = 2 * (p + 1) * (p + 1);
//! let multipoles: Vec<f64> = (0..3 * len).map(|i| ((i * 7919) % 101) as f64 / 101.0 - 0.5).collect();
//! let samples = conversions.samples();
//! let mut signatures = vec![Complex::new(0.0, 0.0); 3 * samples];
//! conversions.s2f(&multipoles, &mut signatures, &mut scratch);
//!
//! // Box 1 against the reference.
//! let grid = Grid::<f64>::new(p);
//! let mut expected = vec![Complex::new(0.0, 0.0); samples];
//! diagonal::s2f(&grid, &level, &multipoles[len..2 * len], &mut diagonal::SignatureWorkspace::new(p), &mut expected);
//! for (a, b) in signatures[samples..2 * samples].iter().zip(&expected) {
//!     assert!((a - b).norm() < 1e-13);
//! }
//!
//! // F → R of S → F is the identity at s = 1 (the grid's quadrature is exact to degree 2p).
//! let mut locals = vec![0.0; 3 * len];
//! conversions.f2r(&signatures, &mut locals, &mut scratch);
//! for (a, b) in locals.iter().zip(&multipoles) {
//!     assert!((a - b).abs() < 1e-13);
//! }
//! ```

use nd_fmm_math::orthonormal::legendre;
use nd_fmm_math::{Complex, RealScalar};
use nd_fmm_ref::helmholtz::diagonal::{Grid, azimuth_count};

pub mod level;
mod symmetry;

pub use level::{
    DEFAULT_SIGNATURE_NODE_BUDGET, LevelBuffers, SIGNATURE_ALIGN, ScratchShape, Signatures,
    m2l_level, scratch_shape,
};
pub use symmetry::signed_permutation;

/// The number of boxes converted together: the columns of the Legendre product are the four
/// real planes of this many boxes. A box's bits do not depend on it.
pub const BLOCK_BOXES: usize = 4;

/// The number of rings 0 ≤ i ≤ p (each with its partner 2p − i) per block of the transforms.
/// A box's bits do not depend on it.
pub const RING_BLOCK: usize = 16;

/// Rows and columns of the column kernel's register tile.
const TILE_ROWS: usize = 4;
const TILE_COLUMNS: usize = 8;

/// The tables of S → F and F → R for expansions of degree p on one level, at the grid of
/// `nd_fmm_ref::helmholtz::diagonal::Grid` (module documentation). Immutable once built; one
/// [`ConversionScratch`] per thread.
#[derive(Clone, Debug)]
pub struct Conversions<T> {
    p: usize,
    azimuths: usize,
    /// The base azimuths 0 ≤ j ≤ N_φ/8.
    bases: usize,
    /// Per order μ the offsets in `legendre` of [S → F even, S → F odd, F → R even, F → R odd].
    legendre_at: Vec<[usize; 4]>,
    legendre: Vec<T>,
    /// Per residue the offsets in `phi` of the forward table (2 bases × orders, row-major)
    /// and the transposed one (orders × 2 bases).
    phi_at: [[usize; 2]; 4],
    phi: Vec<T>,
    /// Per azimuth J: its base, whether it is reflected (σ = −1) and its quarter q.
    map: Vec<Azimuth>,
}

/// An azimuth φ_J = σ φⱼ + q π/2 by its base azimuth j.
#[derive(Clone, Copy, Debug)]
struct Azimuth {
    base: u32,
    reflected: bool,
    quarter: u8,
}

impl<T: RealScalar> Conversions<T> {
    /// The tables for degree p and the level's power scale s in f64 (`Level::<f64>::scale` of
    /// `nd_fmm_ref::helmholtz`, 1 on every high-frequency level), built in f64 and rounded
    /// once to `T` (module documentation, "Tables").
    ///
    /// # Panics
    ///
    /// If s is not finite and positive.
    pub fn new(p: usize, scale: f64) -> Self {
        let s = scale;
        assert!(
            s.is_finite() && s > 0.0,
            "the power scale must be finite and positive, not {s}"
        );
        let grid = Grid::<f64>::new(p);
        let azimuths = grid.azimuths();
        debug_assert_eq!(azimuths, azimuth_count(p));
        let bases = azimuths / 8 + 1;
        let half = p + 1;
        let round = T::from_f64;

        let mut powers = vec![1.0f64; p + 1];
        for n in 1..=p {
            powers[n] = powers[n - 1] * s;
        }
        let mut legendre_at = Vec::with_capacity(p + 1);
        let mut legendre = Vec::new();
        for mu in 0..=p {
            let mut at = [0; 4];
            for (slot, parity) in [(0, 0), (1, 1)] {
                at[slot] = legendre.len();
                for i in 0..half {
                    let theta = grid.theta(i);
                    for n in (mu + parity..=p).step_by(2) {
                        legendre.push(round(theta[legendre::index(n, mu)] * powers[n]));
                    }
                }
            }
            for (slot, parity) in [(2, 0), (3, 1)] {
                at[slot] = legendre.len();
                for n in (mu + parity..=p).step_by(2) {
                    for i in 0..half {
                        legendre.push(round(
                            grid.theta(i)[legendre::index(n, mu)] * powers[n] * grid.weight(i),
                        ));
                    }
                }
            }
            legendre_at.push(at);
        }

        let (cos, sin) = grid.azimuth_tables();
        let mut phi_at = [[0; 2]; 4];
        let mut phi = Vec::new();
        for (r, at) in phi_at.iter_mut().enumerate() {
            let orders: Vec<usize> = (r..=p).step_by(4).collect();
            at[0] = phi.len();
            for table in [cos, sin] {
                for j in 0..bases {
                    phi.extend(orders.iter().map(|&mu| round(table[(mu * j) % azimuths])));
                }
            }
            at[1] = phi.len();
            for &mu in &orders {
                for table in [cos, sin] {
                    phi.extend((0..bases).map(|j| round(table[(mu * j) % azimuths])));
                }
            }
        }

        let quarter = azimuths / 4;
        let map = (0..azimuths)
            .map(|at| {
                let (q, j) = (at / quarter, at % quarter);
                if 8 * j <= azimuths {
                    Azimuth {
                        base: j as u32,
                        reflected: false,
                        quarter: q as u8,
                    }
                } else {
                    Azimuth {
                        base: (quarter - j) as u32,
                        reflected: true,
                        quarter: ((q + 1) % 4) as u8,
                    }
                }
            })
            .collect();
        Self {
            p,
            azimuths,
            bases,
            legendre_at,
            legendre,
            phi_at,
            phi,
            map,
        }
    }

    /// The degree p.
    pub fn degree(&self) -> usize {
        self.p
    }

    /// The number of azimuths N_φ (`nd_fmm_ref::helmholtz::diagonal::azimuth_count`).
    pub fn azimuths(&self) -> usize {
        self.azimuths
    }

    /// The number of samples of a signature, (2p + 1) N_φ.
    pub fn samples(&self) -> usize {
        (2 * self.p + 1) * self.azimuths
    }

    /// The length of a box's expansion, 2(p + 1)² (two real planes).
    pub fn expansion_len(&self) -> usize {
        2 * (self.p + 1) * (self.p + 1)
    }

    /// The real multiply-adds of one box's S → F (or F → R) as computed: (Legendre step, φ
    /// step). The Legendre step forms four real planes of every order on the rings
    /// 0 ≤ i ≤ p, (p + 1) · (p + 1)(p + 2)/2 · 4; the φ step, per residue, 2(N_φ/8 + 1) base
    /// rows times the residue's orders times 4 planes of 2(p + 1) ring slots (the equator's
    /// partner slot is computed and discarded).
    pub fn multiply_adds(&self) -> (u64, u64) {
        let half = (self.p + 1) as u64;
        let legendre = half * (half * (half + 1) / 2) * 4;
        let orders = half; // Σ over the residues of their orders
        let phi = 2 * self.bases as u64 * orders * 4 * 2 * half;
        (legendre, phi)
    }

    /// The bytes of the tables.
    pub fn bytes(&self) -> usize {
        (self.legendre.len() + self.phi.len()) * size_of::<T>()
            + self.map.len() * size_of::<Azimuth>()
            + self.legendre_at.len() * size_of::<[usize; 4]>()
    }

    /// S → F of a batch of boxes: writes the signature F̂ = Σₙ (−i)ⁿ sⁿ Σₘ M̂ₙᵐ Yₙᵐ of every
    /// box's stored multipole (`nd_fmm_ref::helmholtz::diagonal::s2f`, to rounding) into its
    /// [`samples`](Self::samples) of `signatures`, box after box. Overwrites `signatures`.
    /// Allocates nothing; each box's bits depend on its own multipole only.
    ///
    /// # Panics
    ///
    /// If `multipoles` is not a whole number of expansions
    /// ([`expansion_len`](Self::expansion_len)), `signatures` not as many signatures, or the
    /// scratch was made for another degree.
    pub fn s2f(
        &self,
        multipoles: &[T],
        signatures: &mut [Complex<T>],
        scratch: &mut ConversionScratch<T>,
    ) {
        self.check(multipoles.len(), signatures.len(), scratch, "multipoles");
        self.s2f_strided(multipoles, signatures, self.samples(), scratch);
    }

    /// F → R of a batch of boxes: adds sⁿ iⁿ Σᵢⱼ wᵢ (2π/N_φ) Yₙᵐ(ŝᵢⱼ) G(ŝᵢⱼ) of every box's
    /// signature G to its stored local (`nd_fmm_ref::helmholtz::diagonal::f2r`, to rounding),
    /// box after box. Applies no factor 4π ik. Allocates nothing; each box's bits depend on its
    /// own signature and local only.
    ///
    /// # Panics
    ///
    /// As [`s2f`](Self::s2f), with `locals` in place of `multipoles`.
    pub fn f2r(
        &self,
        signatures: &[Complex<T>],
        locals: &mut [T],
        scratch: &mut ConversionScratch<T>,
    ) {
        self.check(locals.len(), signatures.len(), scratch, "locals");
        self.f2r_strided(signatures, self.samples(), locals, scratch);
    }

    /// S → F of a batch of boxes into a stride of signatures: box b's signature at
    /// `b * stride` (stride ≥ [`samples`](Self::samples)), the padding between untouched. The
    /// caller has checked the lengths.
    pub(super) fn s2f_strided(
        &self,
        multipoles: &[T],
        signatures: &mut [Complex<T>],
        stride: usize,
        scratch: &mut ConversionScratch<T>,
    ) {
        let (len, samples) = (self.expansion_len(), self.samples());
        let boxes = multipoles.len() / len;
        for first in (0..boxes).step_by(BLOCK_BOXES) {
            let nb = BLOCK_BOXES.min(boxes - first);
            self.s2f_block(
                &multipoles[first * len..(first + nb) * len],
                &mut signatures[first * stride..(first + nb - 1) * stride + samples],
                stride,
                nb,
                scratch,
            );
        }
    }

    /// F → R of a batch of boxes from a stride of signatures (as
    /// [`s2f_strided`](Self::s2f_strided)). The caller has checked the lengths.
    pub(super) fn f2r_strided(
        &self,
        signatures: &[Complex<T>],
        stride: usize,
        locals: &mut [T],
        scratch: &mut ConversionScratch<T>,
    ) {
        let (len, samples) = (self.expansion_len(), self.samples());
        let boxes = locals.len() / len;
        for first in (0..boxes).step_by(BLOCK_BOXES) {
            let nb = BLOCK_BOXES.min(boxes - first);
            self.f2r_block(
                &signatures[first * stride..(first + nb - 1) * stride + samples],
                stride,
                &mut locals[first * len..(first + nb) * len],
                nb,
                scratch,
            );
        }
    }

    fn check(
        &self,
        expansions: usize,
        signatures: usize,
        scratch: &ConversionScratch<T>,
        name: &str,
    ) -> usize {
        assert_eq!(
            scratch.p, self.p,
            "the scratch has degree {}, the conversions {}",
            scratch.p, self.p
        );
        let len = self.expansion_len();
        assert!(
            expansions.is_multiple_of(len),
            "{name} hold {expansions} values, not a multiple of 2(p + 1)² = {len}"
        );
        let boxes = expansions / len;
        assert_eq!(
            signatures,
            boxes * self.samples(),
            "{boxes} boxes need {} signature samples, not {signatures}",
            boxes * self.samples()
        );
        boxes
    }

    /// The orders of residue r: μ = r, r + 4, … ≤ p.
    fn orders(&self, r: usize) -> usize {
        if r > self.p { 0 } else { (self.p - r) / 4 + 1 }
    }

    /// The row offset of order μ in the scratch's expansion buffer (rows of W columns: the
    /// even degrees, then the odd).
    fn expansion_row(&self, mu: usize) -> usize {
        // Σ_{ν < μ} (p + 1 − ν) rows.
        mu * (self.p + 1) - mu * mu.saturating_sub(1) / 2
    }

    fn s2f_block(
        &self,
        multipoles: &[T],
        signatures: &mut [Complex<T>],
        stride: usize,
        nb: usize,
        scratch: &mut ConversionScratch<T>,
    ) {
        let p = self.p;
        let (len, samples, azimuths, bases) = (
            self.expansion_len(),
            self.samples(),
            self.azimuths,
            self.bases,
        );
        let w = 4 * nb;
        let ConversionScratch {
            expansions,
            e,
            o,
            y,
            z,
            ..
        } = scratch;

        // Gather: row (μ, n) of the four planes [Re +μ, Im +μ, Re −μ, Im −μ] times (−i)ⁿ.
        let half = (p + 1) * (p + 1);
        for mu in 0..=p {
            let mut row = self.expansion_row(mu);
            for parity in 0..2 {
                for n in (mu + parity..=p).step_by(2) {
                    let out = &mut expansions[row * w..(row + 1) * w];
                    for b in 0..nb {
                        let m = &multipoles[b * len..(b + 1) * len];
                        let plus = times_minus_i_pow(m[index(n, mu)], m[half + index(n, mu)], n);
                        let minus = if mu == 0 {
                            (T::zero(), T::zero())
                        } else {
                            let at = index_negative(n, mu);
                            times_minus_i_pow(m[at], m[half + at], n)
                        };
                        out[b] = plus.0;
                        out[nb + b] = plus.1;
                        out[2 * nb + b] = minus.0;
                        out[3 * nb + b] = minus.1;
                    }
                    row += 1;
                }
            }
        }

        for i0 in (0..=p).step_by(RING_BLOCK) {
            let h = RING_BLOCK.min(p + 1 - i0);
            let columns = 2 * h * w;
            // The Legendre step: per order, rings i and 2p − i of the block into residue r's
            // row t of y (rows of all residues: r's rows start at Σ_{r' < r} orders).
            for mu in 0..=p {
                let at = self.legendre_at[mu];
                let (even, odd) = ((p - mu) / 2 + 1, (p + 1 - mu) / 2);
                let row = self.expansion_row(mu);
                e[..h * w].fill(T::zero());
                gemm_acc(
                    h,
                    w,
                    even,
                    &self.legendre[at[0] + i0 * even..],
                    even,
                    &expansions[row * w..],
                    w,
                    e,
                    w,
                );
                o[..h * w].fill(T::zero());
                if odd > 0 {
                    gemm_acc(
                        h,
                        w,
                        odd,
                        &self.legendre[at[1] + i0 * odd..],
                        odd,
                        &expansions[(row + even) * w..],
                        w,
                        o,
                        w,
                    );
                }
                let target = self.residue_row(mu % 4) + mu / 4;
                let out = &mut y[target * columns..(target + 1) * columns];
                for hh in 0..h {
                    for v in 0..w {
                        let (ev, od) = (e[hh * w + v], o[hh * w + v]);
                        out[2 * hh * w + v] = ev + od;
                        out[(2 * hh + 1) * w + v] = ev - od;
                    }
                }
            }
            // The φ step per residue, then the azimuths from the base ones.
            for r in 0..4 {
                let orders = self.orders(r);
                if orders == 0 {
                    continue;
                }
                let rows = 2 * bases;
                z[..rows * columns].fill(T::zero());
                gemm_acc(
                    rows,
                    columns,
                    orders,
                    &self.phi[self.phi_at[r][0]..],
                    orders,
                    &y[self.residue_row(r) * columns..],
                    columns,
                    z,
                    columns,
                );
                for b in 0..nb {
                    let signature = &mut signatures[b * stride..b * stride + samples];
                    for hh in 0..h {
                        let i = i0 + hh;
                        for side in 0..2 {
                            if side == 1 && i == p {
                                continue;
                            }
                            let ring = if side == 0 { i } else { 2 * p - i };
                            let slot = (2 * hh + side) * w;
                            let out = &mut signature[ring * azimuths..(ring + 1) * azimuths];
                            for (sample, azimuth) in out.iter_mut().zip(&self.map) {
                                let j = azimuth.base as usize;
                                let t = (r * azimuth.quarter as usize) % 4;
                                let part = |c: usize| {
                                    let (a, bb) = (slot + c * nb + b, slot + (2 + c) * nb + b);
                                    let (ca, cb) = (z[j * columns + a], z[j * columns + bb]);
                                    let (sa, sb) = (
                                        z[(bases + j) * columns + a],
                                        z[(bases + j) * columns + bb],
                                    );
                                    combine(t, azimuth.reflected, ca, sa, cb, sb)
                                };
                                let value = Complex::new(part(0), part(1));
                                *sample = if r == 0 { value } else { *sample + value };
                            }
                        }
                    }
                }
            }
        }
    }

    fn f2r_block(
        &self,
        signatures: &[Complex<T>],
        stride: usize,
        locals: &mut [T],
        nb: usize,
        scratch: &mut ConversionScratch<T>,
    ) {
        let p = self.p;
        let (len, samples, azimuths, bases) = (
            self.expansion_len(),
            self.samples(),
            self.azimuths,
            self.bases,
        );
        let w = 4 * nb;
        let ConversionScratch {
            expansions,
            e,
            o,
            y,
            z,
            ..
        } = scratch;
        let rows_total = self.expansion_row(p + 1);
        expansions[..rows_total * w].fill(T::zero());

        for i0 in (0..=p).step_by(RING_BLOCK) {
            let h = RING_BLOCK.min(p + 1 - i0);
            let columns = 2 * h * w;
            for r in 0..4 {
                let orders = self.orders(r);
                if orders == 0 {
                    continue;
                }
                // Fold the samples of the block's rings onto the base azimuths: rows j hold
                // the coefficients of Ca (a planes) and Cb (b planes), rows bases + j those of
                // Sa and Sb.
                let rows = 2 * bases;
                z[..rows * columns].fill(T::zero());
                for b in 0..nb {
                    let signature = &signatures[b * stride..b * stride + samples];
                    for hh in 0..h {
                        let i = i0 + hh;
                        for side in 0..2 {
                            if side == 1 && i == p {
                                continue;
                            }
                            let ring = if side == 0 { i } else { 2 * p - i };
                            let slot = (2 * hh + side) * w;
                            let g = &signature[ring * azimuths..(ring + 1) * azimuths];
                            for (&value, azimuth) in g.iter().zip(&self.map) {
                                let j = azimuth.base as usize;
                                let t = (r * azimuth.quarter as usize) % 4;
                                for (c, v) in [(0, value.re), (1, value.im)] {
                                    let (a, bb) = (slot + c * nb + b, slot + (2 + c) * nb + b);
                                    let (cos_row, sin_row) = (j * columns, (bases + j) * columns);
                                    let reflected = if azimuth.reflected { -v } else { v };
                                    match t {
                                        0 => {
                                            z[cos_row + a] = z[cos_row + a] + v;
                                            z[sin_row + bb] = z[sin_row + bb] + reflected;
                                        }
                                        1 => {
                                            z[cos_row + bb] = z[cos_row + bb] + v;
                                            z[sin_row + a] = z[sin_row + a] - reflected;
                                        }
                                        2 => {
                                            z[cos_row + a] = z[cos_row + a] - v;
                                            z[sin_row + bb] = z[sin_row + bb] - reflected;
                                        }
                                        _ => {
                                            z[cos_row + bb] = z[cos_row + bb] - v;
                                            z[sin_row + a] = z[sin_row + a] + reflected;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // The transposed φ product: row t of y holds ā (a planes) and b̄ (b planes) of
                // order r + 4t on every ring slot of the block.
                y[..orders * columns].fill(T::zero());
                gemm_acc(
                    orders,
                    columns,
                    rows,
                    &self.phi[self.phi_at[r][1]..],
                    rows,
                    z,
                    columns,
                    y,
                    columns,
                );
                // Fold rings i and 2p − i into E and O sums, then the transposed Legendre
                // product of the block's rings, continuing each degree's chain.
                for t in 0..orders {
                    let mu = r + 4 * t;
                    let at = self.legendre_at[mu];
                    let (even, odd) = ((p - mu) / 2 + 1, (p + 1 - mu) / 2);
                    let source = &y[t * columns..(t + 1) * columns];
                    for hh in 0..h {
                        let equator = i0 + hh == p;
                        for v in 0..w {
                            let near = source[2 * hh * w + v];
                            let (ev, od) = if equator {
                                (near, near)
                            } else {
                                let far = source[(2 * hh + 1) * w + v];
                                (near + far, near - far)
                            };
                            e[hh * w + v] = ev;
                            o[hh * w + v] = od;
                        }
                    }
                    let row = self.expansion_row(mu);
                    gemm_acc(
                        even,
                        w,
                        h,
                        &self.legendre[at[2] + i0..],
                        p + 1,
                        e,
                        w,
                        &mut expansions[row * w..],
                        w,
                    );
                    if odd > 0 {
                        gemm_acc(
                            odd,
                            w,
                            h,
                            &self.legendre[at[3] + i0..],
                            p + 1,
                            o,
                            w,
                            &mut expansions[(row + even) * w..],
                            w,
                        );
                    }
                }
            }
        }

        // Scatter: L̂ₙ^{±μ} += iⁿ (the accumulated sum).
        let half = (p + 1) * (p + 1);
        for mu in 0..=p {
            let mut row = self.expansion_row(mu);
            for parity in 0..2 {
                for n in (mu + parity..=p).step_by(2) {
                    let acc = &expansions[row * w..(row + 1) * w];
                    for b in 0..nb {
                        let l = &mut locals[b * len..(b + 1) * len];
                        let (re, im) = times_i_pow(acc[b], acc[nb + b], n);
                        let at = index(n, mu);
                        l[at] = l[at] + re;
                        l[half + at] = l[half + at] + im;
                        if mu > 0 {
                            let (re, im) = times_i_pow(acc[2 * nb + b], acc[3 * nb + b], n);
                            let at = index_negative(n, mu);
                            l[at] = l[at] + re;
                            l[half + at] = l[half + at] + im;
                        }
                    }
                    row += 1;
                }
            }
        }
    }

    /// The first row of residue r in the φ step's input (the orders of the residues before).
    fn residue_row(&self, r: usize) -> usize {
        (0..r).map(|q| self.orders(q)).sum()
    }
}

/// The scratch of [`Conversions::s2f`] and [`Conversions::f2r`] for one degree, sized at
/// construction for [`BLOCK_BOXES`] boxes and [`RING_BLOCK`] rings: with W = 4 · BLOCK_BOXES
/// and C = 2 · RING_BLOCK · W, (p + 1)(p + 2)/2 · W + 2 · RING_BLOCK · W + (p + 1) C +
/// 2 (⌊N_φ/8⌋ + 1) C values, 2.2 MB at p = 131 in f64 ([`bytes`](Self::bytes)). One per
/// thread.
#[derive(Clone, Debug)]
pub struct ConversionScratch<T> {
    p: usize,
    /// The gathered multipoles of a block (S → F) or the locals' Legendre sums (F → R):
    /// (p + 1)(p + 2)/2 rows of 4 · BLOCK_BOXES.
    expansions: Vec<T>,
    /// E and O sums of a block of rings.
    e: Vec<T>,
    o: Vec<T>,
    /// The φ step's orders × ring slots (S → F: all residues; F → R: one).
    y: Vec<T>,
    /// The φ step's base azimuths × ring slots.
    z: Vec<T>,
}

impl<T: RealScalar> ConversionScratch<T> {
    /// The scratch for `conversions`' degree.
    pub fn new(conversions: &Conversions<T>) -> Self {
        let p = conversions.p;
        let w = 4 * BLOCK_BOXES;
        let columns = 2 * RING_BLOCK * w;
        let zero = T::zero();
        Self {
            p,
            expansions: vec![zero; conversions.expansion_row(p + 1) * w],
            e: vec![zero; RING_BLOCK * w],
            o: vec![zero; RING_BLOCK * w],
            y: vec![zero; (p + 1) * columns],
            z: vec![zero; 2 * conversions.bases * columns],
        }
    }

    /// The bytes the scratch holds.
    pub fn bytes(&self) -> usize {
        (self.expansions.len() + self.e.len() + self.o.len() + self.y.len() + self.z.len())
            * size_of::<T>()
    }
}

/// One azimuth's term of a residue from its base sums (module documentation).
#[inline(always)]
fn combine<T: RealScalar>(t: usize, reflected: bool, ca: T, sa: T, cb: T, sb: T) -> T {
    match (t, reflected) {
        (0, false) => ca + sb,
        (0, true) => ca - sb,
        (1, false) => cb - sa,
        (1, true) => cb + sa,
        (2, false) => -(ca + sb),
        (2, true) => -(ca - sb),
        (3, false) => -(cb - sa),
        _ => -(cb + sa),
    }
}

/// c[i·ldc + j] ← c[i·ldc + j] + Σₗ a[i·lda + l] x[l·ldx + j] for i < m, j < n, each element
/// the chain over l = 0, 1, …, k − 1 in turn, the product and the sum rounded apart: the same
/// bits in the register tile and the tails, whatever the other rows and columns.
#[allow(clippy::too_many_arguments)]
fn gemm_acc<T: RealScalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &[T],
    ldx: usize,
    c: &mut [T],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut i = 0;
    while i + TILE_ROWS <= m {
        let rows: [&[T]; TILE_ROWS] = core::array::from_fn(|r| &a[(i + r) * lda..][..k]);
        let mut j = 0;
        while j + TILE_COLUMNS <= n {
            let mut acc = [[T::zero(); TILE_COLUMNS]; TILE_ROWS];
            for (r, accr) in acc.iter_mut().enumerate() {
                accr.copy_from_slice(&c[(i + r) * ldc + j..][..TILE_COLUMNS]);
            }
            for l in 0..k {
                let xl: &[T; TILE_COLUMNS] = x[l * ldx + j..][..TILE_COLUMNS]
                    .try_into()
                    .expect("a tile of columns");
                for (accr, row) in acc.iter_mut().zip(&rows) {
                    let al = row[l];
                    for (cv, &xv) in accr.iter_mut().zip(xl) {
                        *cv = *cv + al * xv;
                    }
                }
            }
            for (r, accr) in acc.iter().enumerate() {
                c[(i + r) * ldc + j..][..TILE_COLUMNS].copy_from_slice(accr);
            }
            j += TILE_COLUMNS;
        }
        for (r, row) in rows.iter().enumerate() {
            tail(row, x, ldx, &mut c[(i + r) * ldc..], j, n);
        }
        i += TILE_ROWS;
    }
    for r in i..m {
        tail(&a[r * lda..][..k], x, ldx, &mut c[r * ldc..], 0, n);
    }
}

/// One row of [`gemm_acc`] for the columns from..n, in the same chain.
#[inline(always)]
fn tail<T: RealScalar>(row: &[T], x: &[T], ldx: usize, c: &mut [T], from: usize, n: usize) {
    for j in from..n {
        let mut acc = c[j];
        for (l, &al) in row.iter().enumerate() {
            acc = acc + al * x[l * ldx + j];
        }
        c[j] = acc;
    }
}

/// idx(n, m) of §3.6 for m ≥ 0.
fn index(n: usize, m: usize) -> usize {
    n * n + n + m
}

/// idx(n, −m) of §3.6.
fn index_negative(n: usize, m: usize) -> usize {
    n * n + n - m
}

/// (re + i im) iⁿ, exactly.
fn times_i_pow<T: RealScalar>(re: T, im: T, n: usize) -> (T, T) {
    match n % 4 {
        0 => (re, im),
        1 => (-im, re),
        2 => (-re, -im),
        _ => (im, -re),
    }
}

/// (re + i im) (−i)ⁿ, exactly.
fn times_minus_i_pow<T: RealScalar>(re: T, im: T, n: usize) -> (T, T) {
    times_i_pow(re, im, (3 * (n % 4)) % 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The column kernel against a plain triple loop, bit for bit, over the tile and tails.
    #[test]
    fn gemm_acc_is_the_chain() {
        let value = |i: usize| ((i * 2654435761) % 1000) as f64 / 997.0 - 0.5;
        for m in [1, 3, 4, 5, 9] {
            for n in [1, 7, 8, 9, 17] {
                for k in [1, 2, 5] {
                    let (lda, ldx, ldc) = (k + 1, n + 2, n + 3);
                    let a: Vec<f64> = (0..m * lda).map(value).collect();
                    let x: Vec<f64> = (0..k * ldx).map(|i| value(i + 5)).collect();
                    let mut c: Vec<f64> = (0..m * ldc).map(|i| value(i + 11)).collect();
                    let mut expected = c.clone();
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = expected[i * ldc + j];
                            for l in 0..k {
                                acc += a[i * lda + l] * x[l * ldx + j];
                            }
                            expected[i * ldc + j] = acc;
                        }
                    }
                    gemm_acc(m, n, k, &a, lda, &x, ldx, &mut c, ldc);
                    assert!(
                        c.iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "m {m} n {n} k {k}"
                    );
                }
            }
        }
    }

    /// The azimuth map: every azimuth from its base by σ and q, exactly in integers.
    #[test]
    fn azimuths_from_their_bases() {
        for p in 0..20 {
            let conversions = Conversions::<f64>::new(p, 1.0);
            let n = conversions.azimuths as i64;
            for (at, azimuth) in conversions.map.iter().enumerate() {
                let j = azimuth.base as i64;
                assert!(8 * j <= n, "p {p}: base {j} above N/8");
                let sigma = if azimuth.reflected { -1 } else { 1 };
                let angle = (sigma * j + azimuth.quarter as i64 * n / 4).rem_euclid(n);
                assert_eq!(angle, at as i64, "p {p}");
            }
        }
    }

    #[test]
    fn expansion_rows() {
        for p in 0..12 {
            let conversions = Conversions::<f64>::new(p, 1.0);
            let mut row = 0;
            for mu in 0..=p + 1 {
                assert_eq!(conversions.expansion_row(mu), row);
                row += p + 1 - mu.min(p + 1);
            }
        }
    }
}
