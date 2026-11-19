//! The transfer function of an M2L offset on a grid, in double-double rounded once
//! (proposal §3.6 and §5.4; S6's `transfer.rs`).

use nd_fmm_math::{Complex, RealScalar};

use super::dd::{Cdd, Dd, hankel};
use super::{Grid, Symmetry};
use crate::Frame;
use crate::helmholtz::Level;
use crate::helmholtz::leaf::check_frame;

/// The transfer function of the shift t = c_t − c_s between two boxes of a level, sampled
/// on a [`Grid`] of degree p, in the stored normalisation
///
/// ```text
/// T̂(ŝ) = e^{Im z} Σ_{l ≤ 2p} (2l + 1) iˡ hₗ(z) Pₗ(ŝ · t̂),   z = k|t|,
/// ```
///
/// the bandwidth 2P − 2 = 2p of proposal §5.4 (P = p + 1), with the factor e^{Im z} of the
/// stored (S|R) table of CONVENTIONS §3.15 ("Coaxial translations"), so that [`m2l`](super::m2l)
/// applies 4π ik e^{2β − Im(k)|t|} as RCR does. Each sample is computed in double-double
/// (hₗ by the upward recursion, Pₗ by the forward three-term recurrence, ŝ · t̂ from the
/// grid's double-double directions) and rounded once to `T`, never summed in the working
/// precision (proposal §3.6: a T summed in f64 exceeds the bound by up to 47×).
///
/// A transfer function records its shift; [`m2l`](super::m2l) checks it against the frames.
#[derive(Clone, Debug)]
pub struct Transfer<T> {
    p: usize,
    shift: [f64; 3],
    samples: Vec<Complex<T>>,
}

impl<T: RealScalar> Transfer<T> {
    /// The transfer function from the box `from` to the box `to` of `level`, on `grid`.
    /// Its cost is O(p) double-double operations per sample, O(p³) in all.
    ///
    /// # Panics
    ///
    /// If a frame's radius is not the level's half-width, or the shift is zero.
    pub fn new(grid: &Grid<T>, level: &Level<T>, from: &Frame<T>, to: &Frame<T>) -> Self {
        check_frame(level, from);
        check_frame(level, to);
        let shift: [f64; 3] = core::array::from_fn(|i| (to.centre[i] - from.centre[i]).to_f64());
        assert!(
            shift.iter().any(|&x| x != 0.0),
            "the M2L needs distinct centres"
        );
        let k = level.k();
        Self {
            p: grid.degree(),
            shift,
            samples: sample(grid, Cdd::from_f64(k.re.to_f64(), k.im.to_f64()), shift),
        }
    }

    /// The transfer function of the offset S⁻¹ t, from this one of the shift t (a class
    /// representative's) and the symmetry S of the offset ([`class_of`](super::class_of)):
    /// sample (i, j) is this one's sample [`Symmetry::sample`]`(i, j)`. Exact: no sample is
    /// recomputed.
    ///
    /// # Panics
    ///
    /// If `grid` is not of this transfer function's degree.
    pub fn permuted(&self, grid: &Grid<T>, symmetry: Symmetry) -> Self {
        assert_eq!(grid.degree(), self.p, "the grid has another degree");
        let (rings, azimuths) = (grid.rings(), grid.azimuths());
        let mut samples = Vec::with_capacity(self.samples.len());
        for i in 0..rings {
            for j in 0..azimuths {
                let (a, b) = symmetry.sample(rings, azimuths, i, j);
                samples.push(self.samples[a * azimuths + b]);
            }
        }
        Self {
            p: self.p,
            shift: symmetry.apply_inverse(self.shift),
            samples,
        }
    }

    /// The degree p of the grid.
    pub fn degree(&self) -> usize {
        self.p
    }

    /// The shift t = c_t − c_s, in f64.
    pub fn shift(&self) -> [f64; 3] {
        self.shift
    }

    /// The samples T̂(ŝᵢⱼ) at the grid's indices i N_φ + j.
    pub fn samples(&self) -> &[Complex<T>] {
        &self.samples
    }
}

/// T̂ at every sample of `grid` for wavenumber k and shift t.
fn sample<T: RealScalar>(grid: &Grid<T>, k: Cdd, t: [f64; 3]) -> Vec<Complex<T>> {
    let l_max = 2 * grid.degree();
    let tv = t.map(Dd::from_f64);
    let r = (tv[0] * tv[0] + tv[1] * tv[1] + tv[2] * tv[2]).sqrt();
    let that = tv.map(|x| x / r);
    let z = k.scale(r);
    let mut h = vec![Cdd::ZERO; l_max + 1];
    hankel(z, &mut h);
    let damping = z.im.exp();
    // cₗ = e^{Im z} (2l + 1) iˡ hₗ(z); αₗ = (2l + 1)/(l + 1), βₗ = l/(l + 1).
    let coeffs: Vec<Cdd> = h
        .iter()
        .enumerate()
        .map(|(l, &hl)| hl.times_i_pow(l).scale(Dd::from_usize(2 * l + 1) * damping))
        .collect();
    let alpha: Vec<Dd> = (0..=l_max)
        .map(|l| Dd::from_usize(2 * l + 1) / Dd::from_usize(l + 1))
        .collect();
    let beta: Vec<Dd> = (0..=l_max)
        .map(|l| Dd::from_usize(l) / Dd::from_usize(l + 1))
        .collect();
    let mut out = Vec::with_capacity(grid.samples());
    for i in 0..grid.rings() {
        for j in 0..grid.azimuths() {
            let s = grid.direction_dd(i, j);
            let mu = s[0] * that[0] + s[1] * that[1] + s[2] * that[2];
            let (mut p0, mut p1) = (Dd::ONE, mu);
            let mut acc = coeffs[0];
            for l in 1..=l_max {
                let c = coeffs[l];
                acc = Cdd::new(acc.re + c.re * p1, acc.im + c.im * p1);
                let p2 = alpha[l] * mu * p1 - beta[l] * p0;
                p0 = p1;
                p1 = p2;
            }
            out.push(Complex::new(
                T::from_f64(acc.re.to_f64()),
                T::from_f64(acc.im.to_f64()),
            ));
        }
    }
    out
}
