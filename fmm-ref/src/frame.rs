//! Expansion frames: centre and scaling radius (CONVENTIONS §3.7).

use nd_fmm_math::RealScalar;

/// The centre and scaling radius of an expansion (CONVENTIONS §3.7).
///
/// A multipole or local expansion about `centre` with scaling radius `radius` = r
/// stores the scaled coefficients M̃ = M / rⁿ and L̃ = L · rⁿ⁺¹, and is evaluated at
/// the scaled coordinates (x − c) / r, which [`Frame::scaled`] computes. In an octree
/// the radius is the box half-width. Operators take the frame of their input and of
/// their output expansion instead of a shift vector.
///
/// Build frames with [`Frame::new`], which checks that the radius is positive; the
/// fields are public for reading, and a frame built directly must keep `radius > 0`.
///
/// ```
/// use nd_fmm_ref::Frame;
///
/// let frame = Frame::new([1.0, -2.0, 0.5], 0.25);
/// assert_eq!(frame.scaled([1.5, -2.0, 0.0]), [2.0, 0.0, -2.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame<T: RealScalar> {
    /// Expansion centre c.
    pub centre: [T; 3],
    /// Scaling radius r > 0 of CONVENTIONS §3.7.
    pub radius: T,
}

impl<T: RealScalar> Frame<T> {
    /// Creates the frame with the given centre and scaling radius.
    ///
    /// # Panics
    ///
    /// Panics unless `radius > 0`; in particular for a zero, negative or NaN radius.
    pub fn new(centre: [T; 3], radius: T) -> Self {
        assert!(
            radius > T::zero(),
            "Frame radius must be positive, got {}",
            radius.to_f64()
        );
        Self { centre, radius }
    }

    /// Returns the scaled coordinates (x − c) / r of the point `x` in this frame
    /// (CONVENTIONS §3.7).
    ///
    /// Each component is computed as one subtraction followed by one division, so for
    /// a power-of-two radius the result is the correctly rounded x − c, divided
    /// exactly.
    #[inline]
    pub fn scaled(&self, x: [T; 3]) -> [T; 3] {
        core::array::from_fn(|i| (x[i] - self.centre[i]) / self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn new_keeps_centre_and_radius() {
        // Error measure: exact equality (no arithmetic is involved).
        let frame = Frame::new([0.1, -0.2, 0.3], 0.7);
        assert_eq!(frame.centre, [0.1, -0.2, 0.3]);
        assert_eq!(frame.radius, 0.7);
        let frame = Frame::new([0.1_f32, -0.2, 0.3], 0.7);
        assert_eq!(frame.centre, [0.1_f32, -0.2, 0.3]);
        assert_eq!(frame.radius, 0.7_f32);
    }

    #[test]
    fn new_accepts_tiny_positive_radius() {
        let frame = Frame::new([0.0; 3], f64::MIN_POSITIVE);
        assert_eq!(frame.radius, f64::MIN_POSITIVE);
    }

    #[test]
    #[should_panic(expected = "Frame radius must be positive")]
    fn new_rejects_zero_radius() {
        let _ = Frame::new([0.0; 3], 0.0_f64);
    }

    #[test]
    #[should_panic(expected = "Frame radius must be positive")]
    fn new_rejects_negative_zero_radius() {
        let _ = Frame::new([0.0; 3], -0.0_f64);
    }

    #[test]
    #[should_panic(expected = "Frame radius must be positive")]
    fn new_rejects_negative_radius() {
        let _ = Frame::new([1.0; 3], -0.5_f64);
    }

    #[test]
    #[should_panic(expected = "Frame radius must be positive")]
    fn new_rejects_nan_radius() {
        let _ = Frame::new([1.0; 3], f64::NAN);
    }

    #[test]
    #[should_panic(expected = "Frame radius must be positive")]
    fn new_rejects_zero_radius_f32() {
        let _ = Frame::new([0.0_f32; 3], 0.0);
    }

    #[test]
    fn scaled_examples() {
        // Error measure: exact equality; all values and the radii are dyadic, so every
        // operation is exact.
        let frame = Frame::new([1.0, -2.0, 0.5], 0.25);
        assert_eq!(frame.scaled([1.0, -2.0, 0.5]), [0.0; 3]);
        assert_eq!(frame.scaled([1.5, -2.0, 0.0]), [2.0, 0.0, -2.0]);
        assert_eq!(frame.scaled([1.25, -1.75, 0.75]), [1.0; 3]);

        let frame = Frame::new([0.5_f32, 0.5, 0.5], 2.0);
        assert_eq!(frame.scaled([2.5, -1.5, 0.5]), [1.0, -1.0, 0.0]);

        // The unit frame at the origin is the identity.
        let unit = Frame::new([0.0; 3], 1.0);
        let x = [0.3, -0.7, 0.45];
        assert_eq!(unit.scaled(x), x);
    }

    #[test]
    fn scaled_box_corners_have_norm_sqrt3() {
        // A box of half-width r maps its corners to |u| = √3 (CONVENTIONS §3.9).
        // Error measure: absolute error of |u|², which is exact here (dyadic data).
        let frame = Frame::new([0.375, -1.25, 2.0], 0.125);
        for corner in 0..8 {
            let sign = |bit: usize| if corner >> bit & 1 == 1 { 1.0 } else { -1.0 };
            let x: [f64; 3] = core::array::from_fn(|i| frame.centre[i] + sign(i) * 0.125);
            let u = frame.scaled(x);
            assert_eq!(u.iter().map(|c| c * c).sum::<f64>(), 3.0);
        }
    }

    fn coordinate() -> impl Strategy<Value = f64> {
        -1.0e3..1.0e3
    }

    fn point() -> impl Strategy<Value = [f64; 3]> {
        [coordinate(), coordinate(), coordinate()]
    }

    proptest! {
        #[test]
        fn scaled_matches_definition(
            centre in point(),
            radius in 1.0e-3..1.0e3,
            x in point(),
        ) {
            // Error measure: exact equality with (x − c) / r evaluated in the same
            // order.
            let u = Frame::new(centre, radius).scaled(x);
            for i in 0..3 {
                prop_assert_eq!(u[i], (x[i] - centre[i]) / radius);
            }
        }

        #[test]
        fn scaled_inverts_unscaling(
            centre in point(),
            radius in 1.0e-3..1.0e3,
            x in point(),
        ) {
            // Error measure: componentwise absolute error of c + r u against x,
            // relative to |x| + |c|; bound 4 ε (one rounding each in −, /, · and +).
            let frame = Frame::new(centre, radius);
            let u = frame.scaled(x);
            for i in 0..3 {
                let back = centre[i] + radius * u[i];
                let scale = x[i].abs() + centre[i].abs();
                prop_assert!(
                    (back - x[i]).abs() <= 4.0 * f64::EPSILON * scale,
                    "component {i}: {back} vs {}", x[i]
                );
            }
        }

        #[test]
        fn scaled_by_power_of_two_is_exact_shift(
            centre in point(),
            level in -20i32..20,
            x in point(),
        ) {
            // Error measure: exact equality; dividing by a power of two is exact, so
            // r u reproduces the rounded x − c bit for bit.
            let radius = 2.0_f64.powi(level);
            let u = Frame::new(centre, radius).scaled(x);
            for i in 0..3 {
                prop_assert_eq!(u[i] * radius, x[i] - centre[i]);
            }
        }

        #[test]
        fn scaled_f32_agrees_with_f64(
            centre in [-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0],
            radius in 1.0e-2f32..1.0e2,
            x in [-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0],
        ) {
            // Error measure: componentwise absolute error of the f32 result against f64
            // on the same (f32-representable) inputs, relative to (|x| + |c|) / r;
            // bound 2 ε_f32 (one rounding each in − and /, plus f64 rounding).
            let single = Frame::new(centre, radius).scaled(x);
            let frame64 = Frame::new(centre.map(f64::from), f64::from(radius));
            let double = frame64.scaled(x.map(f64::from));
            for i in 0..3 {
                let scale = (f64::from(x[i]).abs() + f64::from(centre[i]).abs())
                    / f64::from(radius);
                let err = (f64::from(single[i]) - double[i]).abs();
                prop_assert!(
                    err <= 2.0 * f64::from(f32::EPSILON) * scale,
                    "component {i}: {} vs {}", single[i], double[i]
                );
            }
        }
    }
}
