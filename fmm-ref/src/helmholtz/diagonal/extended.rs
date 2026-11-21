//! The diagonal form's chain in double-double: the rounding-free reference of the M2L on
//! high-frequency levels (Helmholtz CH4.3c1; review 2026-11-19, M12).

use nd_fmm_math::orthonormal::legendre;
use nd_fmm_math::{Layout, RealScalar};

use super::Grid;
use super::dd::{Cdd, Dd};
use super::transfer::sample_dd;
use crate::Frame;
use crate::helmholtz::Level;
use crate::helmholtz::leaf::check_coefficients;

/// The M2L of [`m2l`](super::m2l), S → F, the diagonal step and F → R, computed in
/// double-double (about 32 significant digits) from the same f64 inputs: the stored
/// multipole, the level's k, s and β, and the frames. Everything the f64 chain derives from
/// them is derived here in double-double, never rounded to f64: the grid's nodes, weights and
/// azimuths, the θ-factors √((2 − δ_{m0})(2n + 1)/(4π)) P̄ₙᵐ(uᵢ) (by the three-term recursion
/// in n, with a binary exponent per order, so that P̄ₘᵐ ~ sinᵐ θ stays in range near the
/// poles), the transfer function T̂ ([`Transfer`](super::Transfer)'s series before its rounding), the factor
/// 4π ik e^{2β − Im(k)|t|} and the powers sⁿ.
///
/// In exact arithmetic the chain is the truncated M2L that `rcr::m2l` applies (the module's
/// identity: the grid's quadrature is exact for the product's degree). In double-double its
/// own rounding lies about 10⁻¹⁶ below the f64 chain's, so the difference between an f64
/// M2L of this pair and [`ExtendedLocal`] is that M2L's rounding alone: the reference that
/// resolves the floor C₀ u of the bound E ([`bound`](super::bound)) at large degrees, where the
/// f64 references (RCR, the projection of `direct::m2l`) round more than the diagonal form.
///
/// It is a reference, not an operator: O(p³) double-double operations per pair, and a
/// transfer function of (2p + 1) N_φ double-double samples (16 MB at p = 251).
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_ref::Frame;
/// use nd_fmm_ref::helmholtz::diagonal::{ExtendedM2l, Grid, SignatureWorkspace, Transfer, m2l};
/// use nd_fmm_ref::helmholtz::Level;
///
/// let p = 10;
/// let level = Level::new(Complex::new(9.0_f64, 0.0), 0.5);
/// let (from, to) = (Frame::new([0.0; 3], 0.5), Frame::new([2.0, 1.0, -1.0], 0.5));
/// let grid = Grid::new(p);
/// let multipole: Vec<f64> = (0..2 * (p + 1) * (p + 1)).map(|i| (i as f64).sin()).collect();
///
/// let reference = ExtendedM2l::new(&grid, &level);
/// let exact = reference.apply(&reference.transfer(&grid, &from, &to), &multipole);
///
/// let transfer = Transfer::new(&grid, &level, &from, &to);
/// let mut local = vec![0.0; multipole.len()];
/// m2l(&grid, &level, &from, &to, &transfer, &mut SignatureWorkspace::new(p), &multipole,
///     &mut local);
/// let error = exact.difference(&local);
/// let size = exact.hi().iter().fold(0.0f64, |a, &b| a.max(b.abs()));
/// assert!(error.iter().all(|e| e.abs() < 1e-12 * size));
/// ```
#[derive(Clone, Debug)]
pub struct ExtendedM2l {
    p: usize,
    rings: usize,
    azimuths: usize,
    /// The level's k, s, β and half-width, as given (exact).
    k: Cdd,
    s: Dd,
    damping: Dd,
    half_width: f64,
    /// Per ring, the θ-factors at `legendre::index(n, m)`.
    theta: Vec<Dd>,
    /// wᵢ 2π/N_φ per ring.
    weights: Vec<Dd>,
    /// cos φⱼ, sin φⱼ.
    cos: Vec<Dd>,
    sin: Vec<Dd>,
}

/// A transfer function of [`ExtendedM2l`]: the samples of T̂ in double-double and the factor
/// 4π ik e^{2β − Im(k)|t|} of its shift.
#[derive(Clone, Debug)]
pub struct ExtendedTransfer {
    p: usize,
    shift: [f64; 3],
    samples: Vec<Cdd>,
    factor: Cdd,
}

impl ExtendedTransfer {
    /// The shift t = c_t − c_s, in f64.
    pub fn shift(&self) -> [f64; 3] {
        self.shift
    }
}

/// A stored complex local (the layout of [`m2l`](super::m2l)'s) in double-double, as the
/// unevaluated sum of two f64 vectors: [`hi`](Self::hi), each entry the f64 nearest to the
/// value, and [`lo`](Self::lo), the remainder.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedLocal {
    hi: Vec<f64>,
    lo: Vec<f64>,
}

impl ExtendedLocal {
    /// The f64 nearest to each entry.
    pub fn hi(&self) -> &[f64] {
        &self.hi
    }

    /// Each entry minus its [`hi`](Self::hi), rounded to f64.
    pub fn lo(&self) -> &[f64] {
        &self.lo
    }

    /// `local` minus this local, entry by entry, formed in double-double and rounded once:
    /// an f64 M2L's error against this reference, to about 10⁻¹⁶ of itself.
    ///
    /// # Panics
    ///
    /// If `local` has another length.
    pub fn difference(&self, local: &[f64]) -> Vec<f64> {
        assert_eq!(
            local.len(),
            self.hi.len(),
            "the local has length {}, the reference {}",
            local.len(),
            self.hi.len()
        );
        local
            .iter()
            .zip(self.hi.iter().zip(&self.lo))
            .map(|(&v, (&hi, &lo))| {
                (Dd::from_f64(v) - Dd::from_f64(hi) - Dd::from_f64(lo)).to_f64()
            })
            .collect()
    }
}

impl ExtendedM2l {
    /// The reference for the grid's degree p on `level`: its θ-factors, weights and
    /// azimuths in double-double, O(p³) operations.
    pub fn new<T: RealScalar>(grid: &Grid<T>, level: &Level<T>) -> Self {
        let p = grid.degree();
        let rings = grid.rings();
        let len = legendre::len(p);
        let mut theta = vec![Dd::ZERO; rings * len];
        for i in 0..rings {
            let (u, s) = grid.ring_dd(i);
            theta_factors(p, u, s, &mut theta[i * len..(i + 1) * len]);
        }
        let (cos, sin) = grid.azimuth_tables_dd();
        let k = level.k();
        Self {
            p,
            rings,
            azimuths: grid.azimuths(),
            k: Cdd::from_f64(k.re.to_f64(), k.im.to_f64()),
            s: Dd::from_f64(level.scale().to_f64()),
            damping: Dd::from_f64(level.damping().to_f64()),
            half_width: level.half_width().to_f64(),
            theta,
            weights: (0..rings).map(|i| grid.weight_dd(i)).collect(),
            cos: cos.to_vec(),
            sin: sin.to_vec(),
        }
    }

    /// The degree p.
    pub fn degree(&self) -> usize {
        self.p
    }

    /// The transfer function from the box `from` to the box `to` of the level, on `grid`
    /// (the grid of [`new`](Self::new)): [`Transfer`](super::Transfer)'s samples before their rounding, and
    /// the factor 4π ik e^{2β − Im(k)|t|}.
    ///
    /// # Panics
    ///
    /// If the grid has another degree, if a frame's radius is not the level's half-width, or
    /// if the shift is zero.
    pub fn transfer<T: RealScalar>(
        &self,
        grid: &Grid<T>,
        from: &Frame<T>,
        to: &Frame<T>,
    ) -> ExtendedTransfer {
        assert_eq!(grid.degree(), self.p, "the grid has another degree");
        for frame in [from, to] {
            assert!(
                frame.radius.to_f64() == self.half_width,
                "the frame's radius {} is not the level's half-width {}",
                frame.radius.to_f64(),
                self.half_width
            );
        }
        let shift: [f64; 3] = core::array::from_fn(|i| (to.centre[i] - from.centre[i]).to_f64());
        assert!(
            shift.iter().any(|&x| x != 0.0),
            "the M2L needs distinct centres"
        );
        let t = shift.map(Dd::from_f64);
        let distance = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        let four_pi = Dd::PI * Dd::from_f64(4.0);
        let four_pi_ik = Cdd::new(-(four_pi * self.k.im), four_pi * self.k.re);
        let factor = four_pi_ik.scale((self.damping + self.damping - self.k.im * distance).exp());
        ExtendedTransfer {
            p: self.p,
            shift,
            samples: sample_dd(grid, self.k, shift, 0..grid.rings()),
            factor,
        }
    }

    /// The stored local L̂ = f2r(4π ik e^{2β − Im(k)|t|} T̂ · s2f(M̂)) of the stored complex
    /// multipole M̂ (degree p), in double-double.
    ///
    /// # Panics
    ///
    /// If the transfer function has another degree, or `multipole` is not 2(p + 1)² long.
    pub fn apply<T: RealScalar>(
        &self,
        transfer: &ExtendedTransfer,
        multipole: &[T],
    ) -> ExtendedLocal {
        let p = self.p;
        assert_eq!(
            transfer.p, p,
            "the transfer function has another degree than the reference"
        );
        check_coefficients(p, multipole, 2, "multipole");
        let m: Vec<Dd> = multipole
            .iter()
            .map(|&v| Dd::from_f64(v.to_f64()))
            .collect();
        let mut signature = self.signature(&m);
        for (g, &t) in signature.iter_mut().zip(&transfer.samples) {
            *g = t * *g * transfer.factor;
        }
        let local = self.local(&signature);
        let hi: Vec<f64> = local.iter().map(|v| v.to_f64()).collect();
        let lo = local
            .iter()
            .zip(&hi)
            .map(|(&v, &h)| (v - Dd::from_f64(h)).to_f64())
            .collect();
        ExtendedLocal { hi, lo }
    }

    /// S → F in double-double (`signature_of` of the f64 chain).
    fn signature(&self, multipole: &[Dd]) -> Vec<Cdd> {
        let p = self.p;
        let len = Layout::new(p).len();
        let (re, im) = multipole.split_at(len);
        let tlen = legendre::len(p);
        let mut out = vec![Cdd::ZERO; self.rings * self.azimuths];
        let (mut a, mut b) = (vec![Cdd::ZERO; p + 1], vec![Cdd::ZERO; p + 1]);
        let mut powers = vec![Dd::ONE; p + 1];
        for n in 1..=p {
            powers[n] = powers[n - 1] * self.s;
        }
        for i in 0..self.rings {
            let theta = &self.theta[i * tlen..(i + 1) * tlen];
            for mu in 0..=p {
                let (mut sa, mut sb) = (Cdd::ZERO, Cdd::ZERO);
                for n in mu..=p {
                    let th = theta[legendre::index(n, mu)] * powers[n];
                    let plus = index(n, mu as isize);
                    // (−i)ⁿ = i^{3n}.
                    sa = sa
                        + Cdd::new(re[plus], im[plus])
                            .times_i_pow(3 * (n % 4))
                            .scale(th);
                    if mu > 0 {
                        let minus = index(n, -(mu as isize));
                        sb = sb
                            + Cdd::new(re[minus], im[minus])
                                .times_i_pow(3 * (n % 4))
                                .scale(th);
                    }
                }
                a[mu] = sa;
                b[mu] = sb;
            }
            for (j, f) in out[i * self.azimuths..(i + 1) * self.azimuths]
                .iter_mut()
                .enumerate()
            {
                let mut acc = a[0];
                for mu in 1..=p {
                    let at = mu * j % self.azimuths;
                    acc = acc + a[mu].scale(self.cos[at]) + b[mu].scale(self.sin[at]);
                }
                *f = acc;
            }
        }
        out
    }

    /// F → R in double-double (`local_of` of the f64 chain), into a zero local.
    fn local(&self, signature: &[Cdd]) -> Vec<Dd> {
        let p = self.p;
        let len = Layout::new(p).len();
        let tlen = legendre::len(p);
        let mut acc = vec![Cdd::ZERO; len];
        let (mut c, mut sn) = (vec![Cdd::ZERO; p + 1], vec![Cdd::ZERO; p + 1]);
        for i in 0..self.rings {
            let theta = &self.theta[i * tlen..(i + 1) * tlen];
            let g = &signature[i * self.azimuths..(i + 1) * self.azimuths];
            for mu in 0..=p {
                let (mut sc, mut ss) = (Cdd::ZERO, Cdd::ZERO);
                for (j, &gj) in g.iter().enumerate() {
                    let at = mu * j % self.azimuths;
                    sc = sc + gj.scale(self.cos[at]);
                    ss = ss + gj.scale(self.sin[at]);
                }
                c[mu] = sc.scale(self.weights[i]);
                sn[mu] = ss.scale(self.weights[i]);
            }
            for n in 0..=p {
                let i0 = index(n, 0);
                acc[i0] = acc[i0] + c[0].scale(theta[legendre::index(n, 0)]);
                for mu in 1..=n {
                    let th = theta[legendre::index(n, mu)];
                    let (plus, minus) = (index(n, mu as isize), index(n, -(mu as isize)));
                    acc[plus] = acc[plus] + c[mu].scale(th);
                    acc[minus] = acc[minus] + sn[mu].scale(th);
                }
            }
        }
        let mut out = vec![Dd::ZERO; 2 * len];
        let mut power = Dd::ONE;
        for n in 0..=p {
            for m in -(n as isize)..=n as isize {
                let i = index(n, m);
                let v = acc[i].times_i_pow(n % 4).scale(power);
                out[i] = v.re;
                out[len + i] = v.im;
            }
            power = power * self.s;
        }
        out
    }
}

/// idx(n, m) of §3.6.
fn index(n: usize, m: isize) -> usize {
    (n * n + n).wrapping_add_signed(m)
}

/// 2²⁰⁰, the step of the binary exponents of [`theta_factors`].
const STEP: i32 = 200;

/// x 2ᵉ, exactly unless it leaves f64's range (then it under- or overflows like f64).
fn scaled(x: Dd, mut e: i32) -> Dd {
    let mut x = x;
    while e < -1000 {
        x = Dd {
            hi: x.hi * 2f64.powi(-1000),
            lo: x.lo * 2f64.powi(-1000),
        };
        e += 1000;
    }
    let f = 2f64.powi(e);
    Dd {
        hi: x.hi * f,
        lo: x.lo * f,
    }
}

/// The θ-factors √((2 − δ_{m0})(2n + 1)/(4π)) P̄ₙᵐ(u) at `legendre::index(n, m)`,
/// 0 ≤ m ≤ n ≤ p, for u = cos θ and s = sin θ ≥ 0 in double-double, with P̄ₙᵐ normalised as
/// in `nd_fmm_math::orthonormal::legendre` (no Condon–Shortley phase):
///
/// P̄ₘᵐ = Πᵢ₌₁..ₘ √((2i − 1)/(2i)) sᵐ, P̄ₘ₊₁ᵐ = √(2m + 1) u P̄ₘᵐ,
/// P̄ₙᵐ = ((2n − 1) u P̄ₙ₋₁ᵐ − cₙ₋₁ P̄ₙ₋₂ᵐ)/cₙ, cₙ = √((n − m)(n + m)).
///
/// Each order's values carry a binary exponent of their own (a multiple of 2²⁰⁰), so that
/// P̄ₘᵐ ~ sᵐ stays in range; values below f64's range are stored as they underflow.
fn theta_factors(p: usize, u: Dd, s: Dd, out: &mut [Dd]) {
    let four_pi = Dd::PI * Dd::from_f64(4.0);
    let norm = |n: usize, m: usize| {
        (Dd::from_usize(if m == 0 { 1 } else { 2 } * (2 * n + 1)) / four_pi).sqrt()
    };
    let big = 2f64.powi(STEP);
    let (mut diag, mut diag_exp) = (Dd::ONE, 0i32);
    for m in 0..=p {
        if m > 0 {
            diag = diag * (Dd::from_usize(2 * m - 1) / Dd::from_usize(2 * m)).sqrt() * s;
            if diag.hi != 0.0 && diag.hi.abs() < 1.0 / big {
                diag = scaled(diag, STEP);
                diag_exp -= STEP;
            }
        }
        let mut e = diag_exp;
        let (mut p0, mut p1) = (Dd::ZERO, diag);
        out[legendre::index(m, m)] = scaled(p1, e) * norm(m, m);
        for n in m + 1..=p {
            let next = if n == m + 1 {
                (Dd::from_usize(2 * m + 1)).sqrt() * u * p1
            } else {
                let c = |n: usize| Dd::from_usize((n - m) * (n + m)).sqrt();
                (Dd::from_usize(2 * n - 1) * u * p1 - c(n - 1) * p0) / c(n)
            };
            p0 = p1;
            p1 = next;
            if p1.hi.abs() > big {
                p0 = scaled(p0, -STEP);
                p1 = scaled(p1, -STEP);
                e += STEP;
            }
            out[legendre::index(n, m)] = scaled(p1, e) * norm(n, m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helmholtz::diagonal::dd::gauss_legendre;

    /// The θ-factors in double-double against the f64 grid's (`legendre::normalised`) on the
    /// Gauss–Legendre rings of p = 300, the rings nearest the poles included, and finite
    /// where P̄ₘᵐ ~ sᵐ falls below f64's range.
    #[test]
    fn the_theta_factors_against_the_f64_table() {
        let p = 300;
        let (nodes, _) = gauss_legendre(2 * p + 1);
        let len = legendre::len(p);
        let mut dd = vec![Dd::ZERO; len];
        let mut f = vec![0.0f64; len];
        let mut worst = 0.0f64;
        for i in [0, 1, 7, p / 2, p, 2 * p - 3, 2 * p] {
            let u = nodes[i];
            let s = (Dd::ONE - u * u).sqrt();
            theta_factors(p, u, s, &mut dd);
            legendre::normalised(p, u.to_f64(), s.to_f64(), &mut f);
            for n in 0..=p {
                for m in 0..=n {
                    let c = (if m == 0 { 1.0 } else { 2.0 }) * (2 * n + 1) as f64
                        / (4.0 * core::f64::consts::PI);
                    let want = c.sqrt() * f[legendre::index(n, m)];
                    let got = dd[legendre::index(n, m)].to_f64();
                    assert!(got.is_finite(), "ring {i}, ({n}, {m})");
                    worst = worst.max((got - want).abs() / c.sqrt());
                }
            }
        }
        // f64's own error (the node rounded to f64 included): 8.4·10⁻¹⁶ on the M6; about
        // 10× that as the bound (glibc's last bits).
        assert!(worst < 1e-14, "{worst:e}");
    }

    /// S → F → R without the diagonal step (T̂ ≡ 1 and no factor) gives back s²ⁿ M̂ₙᵐ
    /// (the grid's quadrature is exact to degree 4p; the f64 chain's identity test is
    /// `s2f_then_f2r_is_the_identity`): to 10⁻²⁶ of the largest coefficient in
    /// double-double, at s = 1 and s < 1. Measured 3.6·10⁻³¹ and 2.2·10⁻³² on the M6.
    #[test]
    fn s2f_then_f2r_is_the_identity_in_double_double() {
        use nd_fmm_math::Complex;
        for (p, k, half) in [(8, 9.0, 0.5), (16, 0.7, 0.5)] {
            let grid = Grid::<f64>::new(p);
            let level = Level::new(Complex::new(k, 0.0), half);
            let reference = ExtendedM2l::new(&grid, &level);
            let len = Layout::new(p).len();
            let m: Vec<Dd> = (0..2 * len)
                .map(|i| Dd::from_f64(((i * 7 + 3) as f64).sin()))
                .collect();
            let signature = reference.signature(&m);
            let local = reference.local(&signature);
            let mut worst = 0.0f64;
            for n in 0..=p {
                let mut s2n = Dd::ONE;
                for _ in 0..2 * n {
                    s2n = s2n * reference.s;
                }
                for mm in -(n as isize)..=n as isize {
                    for plane in [0, len] {
                        let i = plane + index(n, mm);
                        worst = worst.max((local[i] - m[i] * s2n).to_f64().abs());
                    }
                }
            }
            assert!(worst < 1e-26, "p = {p}, s = {:?}: {worst:e}", reference.s);
        }
    }

    /// As [`s2f_then_f2r_is_the_identity_in_double_double`] at the degree of level 2 at
    /// kD = 1000 and 10⁻⁸ (p = 251), where CH4.3c1 reads the floor: the reference's own rounding
    /// there, to 10⁻²⁶ of the largest coefficient (the θ-factors' scaled recursion near the
    /// poles included). Measured 1.0·10⁻²⁹ on the M6. Release only.
    #[test]
    #[ignore = "a large degree: run in release by hand"]
    fn the_identity_in_double_double_at_p_251() {
        use nd_fmm_math::Complex;
        let p = 251;
        let grid = Grid::<f64>::new(p);
        let level = Level::new(Complex::new(216.5 / 3f64.sqrt(), 0.0), 1.0);
        let reference = ExtendedM2l::new(&grid, &level);
        let len = Layout::new(p).len();
        let m: Vec<Dd> = (0..2 * len)
            .map(|i| Dd::from_f64(((i * 7 + 3) as f64).sin()))
            .collect();
        let local = reference.local(&reference.signature(&m));
        let worst = m
            .iter()
            .zip(&local)
            .map(|(&a, &b)| (b - a).to_f64().abs())
            .fold(0.0f64, f64::max);
        println!("p = {p}: {worst:e}");
        assert!(worst < 1e-26, "{worst:e}");
    }

    /// Exact scaling by powers of two, through the subnormal range to zero.
    #[test]
    fn scaling_by_powers_of_two() {
        let x = Dd::ONE / Dd::from_f64(3.0);
        assert_eq!(scaled(scaled(x, -600), 600), x);
        assert_eq!(scaled(x, -1200).hi, 0.0);
        assert!(scaled(x, -1060).hi > 0.0);
    }
}
