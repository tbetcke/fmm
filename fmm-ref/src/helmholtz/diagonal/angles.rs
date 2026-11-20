//! The transfer function on a uniform grid in the angle ψ between ŝ and t̂, and the samples'
//! positions on it, in double-double (proposal §5.4: T "interpolated in ŝ · t̂ from an
//! extended-precision table"; S6's `Interp` and `angle`, `spikes/diagonal/src/transfer.rs`;
//! Helmholtz CH4.2c1).

use nd_fmm_math::{Complex, RealScalar};

use super::Grid;
use super::dd::{Cdd, Dd};
use super::transfer::{Series, Transfer};
use crate::Frame;
use crate::helmholtz::Level;

/// The stored transfer function T̂ of a shift t ([`Transfer`]'s series: bandwidth 2p, the
/// factor e^{Im z}) at the angles ψₖ = kπ/M, k = 0..=M, between a direction and t̂, each value
/// computed in double-double (μ = cos ψₖ in double-double, the series as [`Transfer`] sums it)
/// and rounded once to f64; and the position of each grid sample on that grid, in
/// double-double.
///
/// T̂ depends on ŝ through μ = ŝ · t̂ = cos ψ alone, and as a function of ψ it is a cosine
/// polynomial of degree 2p, so a grid of M > 2p steps on [0, π] determines it: a builder reads
/// it by interpolation in ψ (`nd_fmm_tables`' interpolated build, S6's Gaussian-regularised
/// sinc series), O(M p) double-double operations for the table and O(1) per sample, against
/// O(p) per sample for [`Transfer`]. This type holds the double-double parts only; the
/// interpolation sums in f64 and is the builder's (this crate's rule: nothing summed in the
/// working precision here).
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_ref::Frame;
/// use nd_fmm_ref::helmholtz::Level;
/// use nd_fmm_ref::helmholtz::diagonal::{AngleTable, Grid};
///
/// let p = 6;
/// let level = Level::new(Complex::new(3.0, 0.0), 1.0);
/// let (from, to) = (Frame::new([0.0; 3], 1.0), Frame::new([4.0, 2.0, 0.0], 1.0));
/// let table = AngleTable::new(p, &level, &from, &to, 6 * p + 1);
/// assert_eq!(table.values().len(), 6 * p + 2);
/// let grid = Grid::<f64>::new(p);
/// let (n, f) = table.position(&grid, 3, 5);
/// assert!(n >= 0 && n as usize <= table.steps() && (0.0..1.0).contains(&f));
/// ```
#[derive(Clone, Debug)]
pub struct AngleTable {
    p: usize,
    steps: usize,
    shift: [f64; 3],
    t_hat: [Dd; 3],
    /// M/π in double-double.
    per_radian: Dd,
    values: Vec<Complex<f64>>,
}

impl AngleTable {
    /// The table of the transfer function of degree p from the box `from` to the box `to` of
    /// `level` on `steps` = M steps of [0, π] (M + 1 values).
    ///
    /// # Panics
    ///
    /// If a frame's radius is not the level's half-width, the shift is zero, or `steps` is 0.
    pub fn new<T: RealScalar>(
        p: usize,
        level: &Level<T>,
        from: &Frame<T>,
        to: &Frame<T>,
        steps: usize,
    ) -> Self {
        assert!(steps > 0, "the angle grid needs a step");
        let shift = Transfer::shift_of(level, from, to);
        let k = level.k();
        let series = Series::new(Cdd::from_f64(k.re.to_f64(), k.im.to_f64()), shift, 2 * p);
        let step = Dd::PI / Dd::from_usize(steps);
        let values = (0..=steps)
            .map(|i| {
                let (_, mu) = (step * Dd::from_usize(i)).sin_cos();
                let v = series.at(mu);
                Complex::new(v.re.to_f64(), v.im.to_f64())
            })
            .collect();
        Self {
            p,
            steps,
            shift,
            t_hat: series.t_hat,
            per_radian: Dd::from_usize(steps) / Dd::PI,
            values,
        }
    }

    /// The degree p (the series' bandwidth is 2p).
    pub fn degree(&self) -> usize {
        self.p
    }

    /// The steps M of [0, π].
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The shift t = c_t − c_s, in f64.
    pub fn shift(&self) -> [f64; 3] {
        self.shift
    }

    /// T̂(ψₖ) at ψₖ = kπ/M, k = 0..=M, each rounded once from double-double.
    pub fn values(&self) -> &[Complex<f64>] {
        &self.values
    }

    /// The position of sample (i, j) of `grid` on the angle grid: (n, f) with
    /// ψ M/π = n + f and 0 ≤ f < 1, ψ the angle between ŝᵢⱼ and t̂. ψ is S6's: atan2 in f64 of
    /// the double-double |ŝ × t̂| and ŝ · t̂, corrected once in double-double by
    /// sin(ψ − ψ₀) = sin ψ cos ψ₀ − cos ψ sin ψ₀; the position is formed in double-double and f
    /// rounded once. (ψ in f64 alone would cost T about 2p u |T̂|, |dT̂/dψ| ≈ 2p |T̂|, and arccos of
    /// a rounded ŝ · t̂ far more near ψ = 0 and π.)
    ///
    /// # Panics
    ///
    /// If `grid` is not of this table's degree, or (i, j) is not a sample.
    pub fn position<T: RealScalar>(&self, grid: &Grid<T>, i: usize, j: usize) -> (isize, f64) {
        assert_eq!(grid.degree(), self.p, "the grid has another degree");
        assert!(
            i < grid.rings() && j < grid.azimuths(),
            "({i}, {j}) is not a sample"
        );
        let s = grid.direction_dd(i, j);
        let t = self.t_hat;
        let c = [
            s[1] * t[2] - s[2] * t[1],
            s[2] * t[0] - s[0] * t[2],
            s[0] * t[1] - s[1] * t[0],
        ];
        let sin = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let cos = s[0] * t[0] + s[1] * t[1] + s[2] * t[2];
        let psi0 = sin.to_f64().atan2(cos.to_f64());
        let (s0, c0) = Dd::from_f64(psi0).sin_cos();
        let psi = Dd::from_f64(psi0) + (sin * c0 - cos * s0);
        let x = psi * self.per_radian;
        let mut n = x.hi.floor();
        let mut f = (x - Dd::from_f64(n)).to_f64();
        if f < 0.0 {
            n -= 1.0;
            f += 1.0;
        }
        // A tiny negative f rounds to f + 1 = 1 (ψ at a node from below).
        if f >= 1.0 {
            n += 1.0;
            f -= 1.0;
        }
        (n as isize, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(k: Complex<f64>, to: [f64; 3], p: usize, steps: usize) -> AngleTable {
        let level = Level::new(k, 1.0);
        AngleTable::new(
            p,
            &level,
            &Frame::new([0.0; 3], 1.0),
            &Frame::new(to, 1.0),
            steps,
        )
    }

    /// Every sample's position against ŝ · t̂ in f64: cos(π (n + f)/M) within 4 ε, 0 ≤ f < 1,
    /// 0 ≤ n ≤ M; on the class representatives' directions and an off-axis shift.
    #[test]
    fn positions_are_the_angles() {
        let p = 7;
        let grid = Grid::<f64>::new(p);
        let steps = 6 * p + 1;
        for to in [
            [6.0, 0.0, 0.0],
            [6.0, 6.0, 6.0],
            [4.0, -2.0, 6.0],
            [0.0, 0.0, -4.0],
        ] {
            let table = table(Complex::new(3.0, 0.0), to, p, steps);
            let r = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt();
            let t_hat = to.map(|x| x / r);
            for i in 0..grid.rings() {
                for j in 0..grid.azimuths() {
                    let (n, f) = table.position(&grid, i, j);
                    assert!((0.0..1.0).contains(&f), "{to:?} ({i}, {j}): f = {f}");
                    assert!(n >= 0 && n as usize <= steps, "{to:?} ({i}, {j}): n = {n}");
                    let psi = core::f64::consts::PI * (n as f64 + f) / steps as f64;
                    let s = grid.direction(i, j);
                    let mu = s[0] * t_hat[0] + s[1] * t_hat[1] + s[2] * t_hat[2];
                    assert!(
                        (psi.cos() - mu).abs() <= 4.0 * f64::EPSILON,
                        "{to:?} ({i}, {j}): cos ψ {} against μ {mu}",
                        psi.cos()
                    );
                }
            }
        }
    }

    /// The first value is the series at μ = 1 bit for bit, and every value the series at
    /// cos ψₖ rounded once.
    #[test]
    fn values_are_the_series() {
        let (p, steps) = (6, 37);
        let k = Complex::new(4.0, 0.3);
        let to = [4.0, 2.0, 0.0];
        let t = table(k, to, p, steps);
        let series = Series::new(Cdd::from_f64(k.re, k.im), to, 2 * p);
        let step = Dd::PI / Dd::from_usize(steps);
        for (i, v) in t.values().iter().enumerate() {
            let mu = if i == 0 {
                Dd::ONE
            } else {
                (step * Dd::from_usize(i)).sin_cos().1
            };
            let s = series.at(mu);
            assert_eq!(
                (v.re.to_bits(), v.im.to_bits()),
                (s.re.to_f64().to_bits(), s.im.to_f64().to_bits()),
                "value {i}"
            );
        }
        assert_eq!((t.degree(), t.steps(), t.shift()), (p, steps, to));
    }

    #[test]
    #[should_panic(expected = "another degree")]
    fn a_grid_of_another_degree_panics() {
        let t = table(Complex::new(3.0, 0.0), [4.0, 0.0, 0.0], 5, 31);
        t.position(&Grid::<f64>::new(6), 0, 0);
    }
}
