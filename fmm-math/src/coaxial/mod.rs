//! Coaxial translation coefficients of the Helmholtz family (CONVENTIONS §3.15,
//! "Coaxial translations").
//!
//! With Fₙᵐ(r) = fₙ(k|r|) Yₙᵐ(r̂), f = j (regular, R) or h = h⁽¹⁾ (singular, S), Yₙᵐ the
//! real orthonormal harmonics of [`crate::orthonormal`], and a shift t = t ẑ along the
//! axis (t > 0):
//!
//! ```text
//! Fₙᵐ(r + t) = Σ_{n′} (E|F)ᵐ_{n′n}(t) G_{n′}ᵐ(r)
//! ```
//!
//! This defines (R|R) at any r, (S|R) for |r| < t and (S|S) for |r| > t; (S|S) = (R|R)
//! as functions of t. The slots +m and −m share one coefficient. The translations of
//! §3.15 in coaxial form (after the rotation that takes the shift onto +ẑ) are M2M with
//! (S|S) from p_child to p_parent, L2L with (R|R) from p_parent to p_child, both at the
//! shift t = a_child (the child's circumradius), and M2L with (S|R) between two boxes of
//! one level.
//!
//! # Storage
//!
//! A [`CoaxialTable`] holds, per order m, the triangle n′ ≥ n of the table of degrees
//! max(p_src, p_tgt) by min(p_src, p_tgt), and reads the other entries through the index
//! symmetry (E|F)ᵐ_{nn′} = (−1)^{n+n′} (E|F)ᵐ_{n′n}. One stored table therefore serves
//! both orientations: [`CoaxialTable::reversed`] reads the M2M table of a level pair as
//! its L2L table. The values carry the stored normalisation of §3.15, with z = k t:
//!
//! ```text
//! (R|R) = (S|S):  Êᵐ_{n′n} = e^{−Im z} Eᵐ_{n′n}
//! (S|R):          Êᵐ_{n′n} = e^{Im z} s^{n+n′} Eᵐ_{n′n}
//! ```
//!
//! with s the power scale of the M2L's level (the same at both ends). M2M and L2L apply
//! the power scales of the two levels themselves.
//!
//! # Generators
//!
//! - **The recursion** ([`CoaxialTable::regular`], [`CoaxialTable::singular_regular`]),
//!   spike S2 (`spikes/coaxial/SPIKE_REPORT.md` §2.1): the start column
//!   E⁰_{n′0}(t) = (−1)^{n′} √(2n′ + 1) f_{n′}(k t) from [`crate::radial`], then the
//!   recursion in m on the sectorial column n = m and the recursion in n towards the
//!   diagonal, both of §3.15, over the triangle n′ ≥ n, which they keep closed. O(p³)
//!   operations; the start column needs degrees to p_src + p_tgt. Accurate where
//!   Re(k) a ≤ 40 (S2 §4.3: the recursion in n amplifies rounding like e^{0.2 k a} for
//!   real and weakly damped k; strongly damped k and Yukawa are stable to k a = 300).
//! - **The projections**, S2 §2.2 (FMM3D's stabilised "point and shoot"): the translated
//!   wave is evaluated at Gauss–Legendre nodes in the polar angle on a sphere of radius ρ
//!   about the new centre and projected onto the orthonormal Legendre functions of each
//!   order. [`CoaxialTable::projection_singular_regular`] ((S|R), ρ < t) and
//!   [`CoaxialTable::projection_regular_inner`] ((R|R), any ρ) solve each target degree
//!   by least squares from the value and the radial derivative, since jₙ(kρ) has zeros;
//!   [`CoaxialTable::projection_regular_outer`] ((S|S), ρ > t) divides by hₙ(kρ).
//!   O(N p³) for N nodes. Each projection computes only the half of the table it resolves
//!   (S7, `spikes/truncation/SPIKE_REPORT.md` §2.4): the outer one the entries with
//!   target degree ≥ source degree (M2M's orientation, low-degree waves projected onto
//!   high degrees), the inner ones those with target degree ≤ source degree.
//!
//! # The rule
//!
//! [`CoaxialTable::m2l`], [`CoaxialTable::m2m`] and [`CoaxialTable::l2l`] choose by
//! §3.15's rule: the recursion where Re(k) a ≤ [`RECURSION_LIMIT`] (a the circumradius of
//! the M2L's level, or of the shift's parent; [`uses_recursion`]), and above it the
//! projection on S2's spheres: M2L on the target box's sphere ρ = a, M2M on the sphere of
//! the parent's nearest M2L distance ρ = 2 w_parent = 8t/√3, L2L on the child's sphere
//! ρ = t (inner, §3.15: "an inner-sphere projection is tested for the rest").
//!
//! # Tested ranges
//!
//! `tests/coaxial/` checks f64 (CH1.2c) against S2's arbitrary-precision fixture (k a
//! from 0.01 to 300, p to 371, real, complex and imaginary k) in S2's FMM-weighted error
//! (the field a table produces, SPIKE_REPORT §4.1), and the definition pointwise at
//! n ≤ 6. Per-entry relative accuracy is not a property of these tables at large p: they
//! span hundreds of orders of magnitude, and their smallest entries only ever multiply
//! coefficients that the geometry makes equally small (S2 §4.2). The (S|R) start column
//! s^l hₗ(k t) leaves the f64 range near l ≈ 175 when s = |k| a < 1 (S7 §2.3), so
//! p_src + p_tgt is limited to about 170 on low-frequency levels; the builders panic on
//! a non-finite start value. f32 is checked at p ≤ 8 only (§3.15 reserves it).
//!
//! # Panics
//!
//! Every builder panics unless k ≠ 0 with Re k ≥ 0 and Im k ≥ 0 (z = k t likewise),
//! t > 0, s in (0, 1], and the projection radius on its side of t; and if the start
//! column is not finite.

mod projection;
mod recursion;

use crate::{Complex, RealScalar};

/// The largest Re(k) a at which the rule of §3.15 builds a table by the recursion, a the
/// circumradius of the M2L's level or of the shift's parent (spike S2, signed off
/// 2026-10-27).
pub const RECURSION_LIMIT: f64 = 40.0;

/// Whether the rule of §3.15 builds the tables of a level by the recursion:
/// Re(k) a ≤ [`RECURSION_LIMIT`]. Yukawa (Re k = 0) always uses it.
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_math::coaxial::uses_recursion;
///
/// assert!(uses_recursion(Complex::new(10.0_f64, 0.0), 4.0));
/// assert!(!uses_recursion(Complex::new(10.0_f64, 0.0), 4.1));
/// assert!(uses_recursion(Complex::new(0.0_f64, 1e3), 1.0));
/// ```
pub fn uses_recursion<T: RealScalar>(k: Complex<T>, a: T) -> bool {
    k.re.to_f64() * a.to_f64() <= RECURSION_LIMIT
}

/// Which coefficients a table holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coefficients {
    /// (R|R) = (S|S), from jₗ.
    Regular,
    /// (S|R), from hₗ.
    SingularRegular,
}

/// A coaxial table (R|R) = (S|S) or (S|R) for source degrees n ≤ p_src and target degrees
/// n′ ≤ p_tgt, in the stored normalisation of §3.15 (the [module documentation](self)).
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_math::coaxial::CoaxialTable;
///
/// // (R|R) at z = k t = 0.5: E⁰₀₀ = j₀(0.5) and E⁰₁₀ = −√3 j₁(0.5).
/// let table = CoaxialTable::regular(Complex::new(0.5_f64, 0.0), 4, 4);
/// let j0 = 0.5_f64.sin() / 0.5;
/// assert!((table.get(0, 0, 0).re - j0).abs() < 1e-15);
/// // The index symmetry: E⁰₀₁ = −E⁰₁₀.
/// assert_eq!(table.get(0, 0, 1), -table.get(0, 1, 0));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CoaxialTable<T> {
    p_src: usize,
    p_tgt: usize,
    /// max(p_src, p_tgt): the last row of the stored triangle.
    rows: usize,
    /// min(p_src, p_tgt): the last column, and the last order.
    cols: usize,
    /// Where each order's triangle starts in `data`, cols + 2 values.
    order_start: Vec<usize>,
    data: Vec<Complex<T>>,
}

impl<T: RealScalar> CoaxialTable<T> {
    /// A zero table of the given degrees.
    fn zeros(p_src: usize, p_tgt: usize) -> Self {
        let rows = p_src.max(p_tgt);
        let cols = p_src.min(p_tgt);
        let mut order_start = Vec::with_capacity(cols + 2);
        let mut total = 0;
        for m in 0..=cols {
            order_start.push(total);
            total += triangle_len(rows - m, cols - m);
        }
        order_start.push(total);
        Self {
            p_src,
            p_tgt,
            rows,
            cols,
            order_start,
            data: vec![Complex::new(T::zero(), T::zero()); total],
        }
    }

    /// The position of the stored entry (m, r, c), m ≤ c ≤ min(r, cols), r ≤ rows.
    #[inline]
    fn at(&self, m: usize, r: usize, c: usize) -> usize {
        // Rows m..r of order m hold min(row, cols) − m + 1 entries each.
        let (rr, cc) = (r - m, self.cols - m);
        let before = if rr <= cc + 1 {
            rr * (rr + 1) / 2
        } else {
            (cc + 1) * (cc + 2) / 2 + (rr - cc - 1) * (cc + 1)
        };
        self.order_start[m] + before + (c - m)
    }

    /// Sets the stored entry Êᵐ_{rc}, r ≥ c.
    #[inline]
    fn set(&mut self, m: usize, r: usize, c: usize, value: Complex<T>) {
        let i = self.at(m, r, c);
        self.data[i] = value;
    }

    /// Êᵐ_{n′n}, the coefficient from source degree `n_src` = n to target degree
    /// `n_tgt` = n′ of order m (for the slots +m and −m alike), m ≤ n′ ≤ p_tgt and
    /// m ≤ n ≤ p_src.
    ///
    /// # Panics
    ///
    /// If the indices are out of range.
    #[inline]
    pub fn get(&self, m: usize, n_tgt: usize, n_src: usize) -> Complex<T> {
        assert!(
            m <= n_tgt.min(n_src) && n_tgt <= self.p_tgt && n_src <= self.p_src,
            "coaxial index (m, n′, n) = ({m}, {n_tgt}, {n_src}) out of range for degrees \
             {} → {}",
            self.p_src,
            self.p_tgt
        );
        if n_tgt >= n_src {
            self.data[self.at(m, n_tgt, n_src)]
        } else {
            let v = self.data[self.at(m, n_src, n_tgt)];
            if (n_src + n_tgt) % 2 == 1 { -v } else { v }
        }
    }

    /// The largest source degree.
    pub fn p_src(&self) -> usize {
        self.p_src
    }

    /// The largest target degree.
    pub fn p_tgt(&self) -> usize {
        self.p_tgt
    }

    /// The number of orders, min(p_src, p_tgt) + 1; orders above have no coefficient.
    pub fn orders(&self) -> usize {
        self.cols + 1
    }

    /// The number of stored complex values, Σₘ of the triangle of order m: about p³/6
    /// for a square table of degree p.
    pub fn stored(&self) -> usize {
        self.data.len()
    }

    /// The same coefficients with source and target degrees exchanged: the table
    /// p_tgt → p_src at the same shift. The M2M table of a level pair (p_child →
    /// p_parent), reversed, is its L2L table (p_parent → p_child). Nothing is recomputed.
    ///
    /// ```
    /// use nd_fmm_math::Complex;
    /// use nd_fmm_math::coaxial::CoaxialTable;
    ///
    /// let m2m = CoaxialTable::regular(Complex::new(0.8_f64, 0.0), 3, 6);
    /// let l2l = m2m.clone().reversed();
    /// assert_eq!((l2l.p_src(), l2l.p_tgt()), (6, 3));
    /// assert_eq!(l2l.get(1, 2, 5), CoaxialTable::regular(Complex::new(0.8, 0.0), 6, 3).get(1, 2, 5));
    /// ```
    pub fn reversed(mut self) -> Self {
        std::mem::swap(&mut self.p_src, &mut self.p_tgt);
        self
    }

    /// (R|R) = (S|S) at z = k t by the recursion, unscaled (Êᵐ_{n′n} = e^{−Im z} Eᵐ_{n′n}).
    ///
    /// # Panics
    ///
    /// Unless z ≠ 0 with Re z ≥ 0 and Im z ≥ 0, finite.
    pub fn regular(z: Complex<T>, p_src: usize, p_tgt: usize) -> Self {
        check_wavenumber(z, "z");
        recursion::build(Coefficients::Regular, z, T::one(), p_src, p_tgt)
    }

    /// (S|R) at z = k t by the recursion, scaled (Êᵐ_{n′n} = e^{Im z} s^{n+n′} Eᵐ_{n′n}).
    ///
    /// # Panics
    ///
    /// Unless z ≠ 0 with Re z ≥ 0 and Im z ≥ 0, finite, and s in (0, 1]; if the start
    /// column s^l hₗ(z), l ≤ p_src + p_tgt, is not finite.
    pub fn singular_regular(z: Complex<T>, s: T, p_src: usize, p_tgt: usize) -> Self {
        check_wavenumber(z, "z");
        check_scale(s);
        recursion::build(Coefficients::SingularRegular, z, s, p_src, p_tgt)
    }

    /// (S|R) for wavenumber k and shift t by projection on the sphere of radius
    /// 0 < ρ < t about the target centre, least squares from the value and the radial
    /// derivative per target degree, with `nodes` Gauss–Legendre nodes; scaled as
    /// [`Self::singular_regular`].
    ///
    /// # Panics
    ///
    /// Unless k is a valid wavenumber, 0 < ρ < t, s in (0, 1] and `nodes` ≥ 1.
    pub fn projection_singular_regular(
        k: Complex<T>,
        t: T,
        rho: T,
        s: T,
        p_src: usize,
        p_tgt: usize,
        nodes: usize,
    ) -> Self {
        check_wavenumber(k, "k");
        check_shift(t);
        check_scale(s);
        assert!(
            rho > T::zero() && rho < t,
            "the (S|R) projection needs 0 < ρ < t"
        );
        projection::build(
            projection::Sphere {
                kind: projection::Kind::SingularRegular,
                k,
                t,
                rho,
                s,
                nodes,
            },
            p_src,
            p_tgt,
        )
    }

    /// (R|R) = (S|S) for wavenumber k and shift t by projection of hₙ onto hₙ on the
    /// sphere of radius ρ > t about the target centre, with `nodes` Gauss–Legendre nodes;
    /// unscaled as [`Self::regular`]. Accurate in M2M's orientation, which it computes
    /// (target degree ≥ source degree).
    ///
    /// # Panics
    ///
    /// Unless k is a valid wavenumber, 0 < t < ρ and `nodes` ≥ 1.
    pub fn projection_regular_outer(
        k: Complex<T>,
        t: T,
        rho: T,
        p_src: usize,
        p_tgt: usize,
        nodes: usize,
    ) -> Self {
        check_wavenumber(k, "k");
        check_shift(t);
        assert!(
            rho > t && rho.is_finite(),
            "the outer projection needs ρ > t"
        );
        projection::build(
            projection::Sphere {
                kind: projection::Kind::RegularOuter,
                k,
                t,
                rho,
                s: T::one(),
                nodes,
            },
            p_src,
            p_tgt,
        )
    }

    /// (R|R) for wavenumber k and shift t by projection of jₙ onto jₙ on the sphere of
    /// radius ρ > 0 about the target centre, least squares from the value and the radial
    /// derivative per target degree, with `nodes` Gauss–Legendre nodes; unscaled as
    /// [`Self::regular`]. Computes L2L's orientation (target degree ≤ source degree).
    ///
    /// # Panics
    ///
    /// Unless k is a valid wavenumber, t > 0, ρ > 0 and `nodes` ≥ 1.
    pub fn projection_regular_inner(
        k: Complex<T>,
        t: T,
        rho: T,
        p_src: usize,
        p_tgt: usize,
        nodes: usize,
    ) -> Self {
        check_wavenumber(k, "k");
        check_shift(t);
        assert!(
            rho > T::zero() && rho.is_finite(),
            "the inner projection needs ρ > 0"
        );
        projection::build(
            projection::Sphere {
                kind: projection::Kind::RegularInner,
                k,
                t,
                rho,
                s: T::one(),
                nodes,
            },
            p_src,
            p_tgt,
        )
    }

    /// The M2L table (S|R) between two boxes of a level with circumradius a and power
    /// scale s, at the distance t of their centres, from p_src to p_tgt, by the rule of
    /// §3.15: the recursion at z = k t where Re(k) a ≤ [`RECURSION_LIMIT`], else the
    /// projection on the target box's sphere ρ = a with max(p_src, p_tgt) + 40 nodes.
    ///
    /// # Panics
    ///
    /// As the generator it uses; also unless a < t.
    pub fn m2l(k: Complex<T>, t: T, a: T, s: T, p_src: usize, p_tgt: usize) -> Self {
        check_shift(t);
        assert!(a > T::zero() && a < t, "the M2L needs 0 < a < t");
        if uses_recursion(k, a) {
            check_wavenumber(k, "k");
            Self::singular_regular(k * t, s, p_src, p_tgt)
        } else {
            let nodes = p_src.max(p_tgt) + 40;
            Self::projection_singular_regular(k, t, a, s, p_src, p_tgt, nodes)
        }
    }

    /// The M2M table (S|S) from a child of degree p_child to its parent of degree
    /// p_parent, at the shift t = a_child (the parent's circumradius is 2t), by the rule
    /// of §3.15: the recursion at z = k t where Re(k) 2t ≤ [`RECURSION_LIMIT`], else the
    /// outer projection on the sphere of the parent's nearest M2L distance,
    /// ρ = 2 w_parent = 8t/√3, with max(p_child, p_parent) + 40 nodes. Unscaled.
    ///
    /// # Panics
    ///
    /// As the generator it uses.
    pub fn m2m(k: Complex<T>, t: T, p_child: usize, p_parent: usize) -> Self {
        check_shift(t);
        if uses_recursion(k, t + t) {
            check_wavenumber(k, "k");
            Self::regular(k * t, p_child, p_parent)
        } else {
            let rho = T::from_f64(8.0 / 3f64.sqrt()) * t;
            let nodes = p_child.max(p_parent) + 40;
            Self::projection_regular_outer(k, t, rho, p_child, p_parent, nodes)
        }
    }

    /// The L2L table (R|R) from a parent of degree p_parent to its child of degree
    /// p_child, at the shift t = a_child, by the rule of §3.15: the recursion at z = k t
    /// where Re(k) 2t ≤ [`RECURSION_LIMIT`], else the inner projection on the child's
    /// sphere ρ = t, with max(p_child, p_parent) + 40 + ⌈|k| t⌉ nodes. Unscaled.
    ///
    /// # Panics
    ///
    /// As the generator it uses.
    pub fn l2l(k: Complex<T>, t: T, p_parent: usize, p_child: usize) -> Self {
        check_shift(t);
        if uses_recursion(k, t + t) {
            check_wavenumber(k, "k");
            Self::regular(k * t, p_parent, p_child)
        } else {
            let nodes = p_child.max(p_parent) + 40 + (k.norm() * t).to_f64().ceil() as usize;
            Self::projection_regular_inner(k, t, t, p_parent, p_child, nodes)
        }
    }
}

/// Entries of a triangle with rows 0..=r and columns 0..=min(row, c), c ≤ r.
#[inline]
fn triangle_len(r: usize, c: usize) -> usize {
    (c + 1) * (c + 2) / 2 + (r - c) * (c + 1)
}

fn check_wavenumber<T: RealScalar>(k: Complex<T>, name: &str) {
    assert!(
        k.re.is_finite()
            && k.im.is_finite()
            && k.re >= T::zero()
            && k.im >= T::zero()
            && (k.re > T::zero() || k.im > T::zero()),
        "{name} must be finite and non-zero with Re {name} ≥ 0 and Im {name} ≥ 0"
    );
}

fn check_shift<T: RealScalar>(t: T) {
    assert!(
        t > T::zero() && t.is_finite(),
        "the shift t must be positive"
    );
}

fn check_scale<T: RealScalar>(s: T) {
    assert!(
        s > T::zero() && s <= T::one(),
        "the power scale must lie in (0, 1]"
    );
}
