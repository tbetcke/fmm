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
/// It does not record k: one built for another wavenumber gives a wrong result without a
/// panic. Where hₗ(k|t|) leaves the f64 range (k|t| small against the bandwidth 2p, far
/// outside the diagonal form's regime, proposal §3.6), the samples are not finite; the
/// bound E ([`bound`](super::bound)) is then +∞ or far above any tolerance.
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
        let shift = Self::shift_of(level, from, to);
        let k = level.k();
        Self {
            p: grid.degree(),
            shift,
            samples: sample(
                grid,
                Cdd::from_f64(k.re.to_f64(), k.im.to_f64()),
                shift,
                0..grid.rings(),
            ),
        }
    }

    /// The samples of ring `ring` of the transfer function from `from` to `to`, bit for bit
    /// those of [`new`](Self::new) at indices `ring` N_φ + j: the same series, the same
    /// operations in the same order, so that a build may cut a transfer function into rings
    /// (Helmholtz CH4.2c1). Its cost is O(p) double-double operations per sample, plus the
    /// series' set-up, O(p).
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and if `ring` is not below the grid's rings.
    pub fn ring_samples(
        grid: &Grid<T>,
        level: &Level<T>,
        from: &Frame<T>,
        to: &Frame<T>,
        ring: usize,
    ) -> Vec<Complex<T>> {
        assert!(
            ring < grid.rings(),
            "ring {ring} of a grid of {} rings",
            grid.rings()
        );
        let shift = Self::shift_of(level, from, to);
        let k = level.k();
        sample(
            grid,
            Cdd::from_f64(k.re.to_f64(), k.im.to_f64()),
            shift,
            ring..ring + 1,
        )
    }

    /// The shift t = c_t − c_s in f64, the frames checked against the level.
    pub(super) fn shift_of(level: &Level<T>, from: &Frame<T>, to: &Frame<T>) -> [f64; 3] {
        check_frame(level, from);
        check_frame(level, to);
        let shift: [f64; 3] = core::array::from_fn(|i| (to.centre[i] - from.centre[i]).to_f64());
        assert!(
            shift.iter().any(|&x| x != 0.0),
            "the M2L needs distinct centres"
        );
        shift
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

/// The series T̂(μ) = Σ_{l ≤ L} cₗ Pₗ(μ) of a shift t at wavenumber k, in double-double, with
/// cₗ = e^{Im z} (2l + 1) iˡ hₗ(z), z = k|t|, and t̂ (shared by [`Transfer`] and
/// [`AngleTable`](super::AngleTable), so that both sum the same terms in the same order).
pub(super) struct Series {
    /// t̂ in double-double.
    pub(super) t_hat: [Dd; 3],
    coeffs: Vec<Cdd>,
    /// αₗ = (2l + 1)/(l + 1), βₗ = l/(l + 1).
    alpha: Vec<Dd>,
    beta: Vec<Dd>,
}

impl Series {
    /// The series of bandwidth `l_max` for wavenumber k and shift t.
    pub(super) fn new(k: Cdd, t: [f64; 3], l_max: usize) -> Self {
        let tv = t.map(Dd::from_f64);
        let r = (tv[0] * tv[0] + tv[1] * tv[1] + tv[2] * tv[2]).sqrt();
        let t_hat = tv.map(|x| x / r);
        let z = k.scale(r);
        let mut h = vec![Cdd::ZERO; l_max + 1];
        hankel(z, &mut h);
        let damping = z.im.exp();
        let coeffs = h
            .iter()
            .enumerate()
            .map(|(l, &hl)| hl.times_i_pow(l).scale(Dd::from_usize(2 * l + 1) * damping))
            .collect();
        let alpha = (0..=l_max)
            .map(|l| Dd::from_usize(2 * l + 1) / Dd::from_usize(l + 1))
            .collect();
        let beta = (0..=l_max)
            .map(|l| Dd::from_usize(l) / Dd::from_usize(l + 1))
            .collect();
        Self {
            t_hat,
            coeffs,
            alpha,
            beta,
        }
    }

    /// T̂(μ) in double-double: Pₗ by the forward three-term recurrence.
    pub(super) fn at(&self, mu: Dd) -> Cdd {
        let (mut p0, mut p1) = (Dd::ONE, mu);
        let mut acc = self.coeffs[0];
        for l in 1..self.coeffs.len() {
            let c = self.coeffs[l];
            acc = Cdd::new(acc.re + c.re * p1, acc.im + c.im * p1);
            let p2 = self.alpha[l] * mu * p1 - self.beta[l] * p0;
            p0 = p1;
            p1 = p2;
        }
        acc
    }
}

/// T̂ at the samples of `grid`'s rings `rings` for wavenumber k and shift t, ring after ring.
fn sample<T: RealScalar>(
    grid: &Grid<T>,
    k: Cdd,
    t: [f64; 3],
    rings: core::ops::Range<usize>,
) -> Vec<Complex<T>> {
    let series = Series::new(k, t, 2 * grid.degree());
    let that = series.t_hat;
    let mut out = Vec::with_capacity(rings.len() * grid.azimuths());
    for i in rings {
        for j in 0..grid.azimuths() {
            let s = grid.direction_dd(i, j);
            let mu = s[0] * that[0] + s[1] * that[1] + s[2] * that[2];
            let acc = series.at(mu);
            out.push(Complex::new(
                T::from_f64(acc.re.to_f64()),
                T::from_f64(acc.im.to_f64()),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every ring of `ring_samples` is `new`'s ring bit for bit, real and damped k.
    #[test]
    fn rings_are_the_transfer_functions_rings() {
        let p = 5;
        let grid = Grid::<f64>::new(p);
        let from = Frame::new([0.0; 3], 1.0);
        for (k, to) in [
            (Complex::new(4.0, 0.0), [6.0, 2.0, -2.0]),
            (Complex::new(4.0, 0.4), [0.0, 0.0, 4.0]),
        ] {
            let level = Level::new(k, 1.0);
            let to = Frame::new(to, 1.0);
            let whole = Transfer::new(&grid, &level, &from, &to);
            let n = grid.azimuths();
            for ring in 0..grid.rings() {
                let part = Transfer::ring_samples(&grid, &level, &from, &to, ring);
                let bits = |v: &[Complex<f64>]| -> Vec<(u64, u64)> {
                    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
                };
                assert_eq!(
                    bits(&part),
                    bits(&whole.samples()[ring * n..(ring + 1) * n]),
                    "ring {ring}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "ring 11 of a grid of 11 rings")]
    fn a_ring_beyond_the_grid_panics() {
        let grid = Grid::<f64>::new(5);
        let level = Level::new(Complex::new(4.0, 0.0), 1.0);
        let from = Frame::new([0.0; 3], 1.0);
        let to = Frame::new([4.0, 0.0, 0.0], 1.0);
        Transfer::ring_samples(&grid, &level, &from, &to, 11);
    }
}
