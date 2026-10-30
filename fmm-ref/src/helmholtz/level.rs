//! The constants of an octree level for the Helmholtz family (CONVENTIONS §3.15,
//! "Radial functions and scaling").

use nd_fmm_math::{Complex, RealScalar};

/// The wavenumber and the constants of one octree level that the Helmholtz operators
/// apply (CONVENTIONS §3.15): for boxes of half-width r (width w = 2r),
///
/// - the circumradius a = (√3/2) w = √3 r,
/// - the power scale s = min(|k| a, 1),
/// - the damping exponent β = Im(k) a (zero for real k, κ a for Yukawa).
///
/// The stored coefficients of a box on this level are M̂ = e^{−β} M / sⁿ and
/// L̂ = e^β L sⁿ. A level is **Yukawa** when Re k = 0, k = iκ; the operators of Yukawa's
/// real path accept only such a level.
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_ref::helmholtz::Level;
///
/// let level = Level::new(Complex::new(0.0, 4.0_f64), 0.25);
/// assert!(level.is_yukawa());
/// assert_eq!(level.circumradius(), 3.0_f64.sqrt() * 0.25);
/// assert_eq!(level.scale(), 1.0); // |k| a ≈ 1.73
/// assert_eq!(level.damping(), 4.0 * level.circumradius());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level<T: RealScalar> {
    k: Complex<T>,
    half_width: T,
    circumradius: T,
    scale: T,
    damping: T,
}

impl<T: RealScalar> Level<T> {
    /// The level of boxes of half-width `half_width` for the wavenumber `k`.
    ///
    /// # Panics
    ///
    /// Unless k is finite and nonzero with Re k ≥ 0 and Im k ≥ 0 (the range of CONVENTIONS
    /// §3.15; k = 0 is Laplace), and the half-width is finite and positive.
    pub fn new(k: Complex<T>, half_width: T) -> Self {
        assert!(
            k.re.is_finite() && k.im.is_finite(),
            "the wavenumber must be finite"
        );
        assert!(
            k.re >= T::zero() && k.im >= T::zero() && (k.re > T::zero() || k.im > T::zero()),
            "the wavenumber must be nonzero with Re k ≥ 0 and Im k ≥ 0"
        );
        assert!(
            half_width.is_finite() && half_width > T::zero(),
            "the half-width must be finite and positive"
        );
        let circumradius = T::from_f64(3.0).sqrt() * half_width;
        Self {
            k,
            half_width,
            circumradius,
            scale: (k.norm() * circumradius).min(T::one()),
            damping: k.im * circumradius,
        }
    }

    /// The wavenumber k.
    pub fn k(&self) -> Complex<T> {
        self.k
    }

    /// The half-width r of the level's boxes, the radius of their [`Frame`](crate::Frame).
    pub fn half_width(&self) -> T {
        self.half_width
    }

    /// The circumradius a = √3 r.
    pub fn circumradius(&self) -> T {
        self.circumradius
    }

    /// The power scale s = min(|k| a, 1).
    pub fn scale(&self) -> T {
        self.scale
    }

    /// The damping exponent β = Im(k) a.
    pub fn damping(&self) -> T {
        self.damping
    }

    /// Whether k = iκ (Re k = 0): Yukawa, the modified Helmholtz kernel e^{−κr}/r.
    pub fn is_yukawa(&self) -> bool {
        self.k.re == T::zero()
    }
}
