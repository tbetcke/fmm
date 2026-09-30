//! Caller-owned scratch memory for the operators.

use nd_fmm_math::rotation::blocks_len;
use nd_fmm_math::{Layout, RealScalar};

/// Scratch memory for every operator of this crate, built once for a maximum degree.
///
/// Operators never allocate: they take the temporaries they need from a `Workspace`
/// that the caller builds with [`Workspace::new`] and passes to every call. A workspace
/// built for degree `p` serves every call with degree at most `p`; an operator called
/// with a larger degree panics. Its contents between calls are unspecified, and one
/// workspace can be reused for any sequence of operators.
///
/// It holds the buffers of the leaf operators in [`leaf`](crate::leaf): one set of
/// solid-harmonic values and their three gradient components, each of length (p + 1)²
/// in the real storage of CONVENTIONS §3.6. For the translations in
/// [`direct`](crate::direct) it holds the solid harmonics of the shift vector, up to
/// degree 2p ((2p + 1)² reals), because M2L needs irregular harmonics up to degree 2p
/// (CONVENTIONS §3.11). For the translations in [`rotation`](crate::rotation) it holds
/// the rotation blocks Dⁿ, n ≤ p, of CONVENTIONS §3.8
/// ([`blocks_len`](nd_fmm_math::rotation::blocks_len)`(p)` reals), two coefficient
/// vectors of length (p + 1)² in the rotated frame, and the 2p + 1 axis values of the
/// coaxial translation (§3.11, "Coaxial translations").
///
/// ```
/// use nd_fmm_ref::{Frame, Workspace, leaf};
///
/// let p = 4;
/// let mut ws = Workspace::new(p);
/// assert_eq!(ws.p(), p);
///
/// let frame = Frame::new([0.0; 3], 1.0);
/// let mut multipole = vec![0.0; (p + 1) * (p + 1)];
/// leaf::p2m(p, &frame, &[[0.1, 0.2, 0.3]], &[1.0], &mut ws, &mut multipole);
/// // The same workspace serves any smaller degree.
/// let mut small = vec![0.0; 4];
/// leaf::p2m(1, &frame, &[[0.1, 0.2, 0.3]], &[1.0], &mut ws, &mut small);
/// assert_eq!(small[..], multipole[..4]);
/// ```
#[derive(Clone, Debug)]
pub struct Workspace<T: RealScalar> {
    /// Largest degree the buffers hold.
    p: usize,
    /// Solid-harmonic values, (p + 1)² in real storage.
    harmonics: Vec<T>,
    /// ∂x, ∂y and ∂z of the solid harmonics, (p + 1)² each in real storage.
    gradient: [Vec<T>; 3],
    /// Solid harmonics of a translation's shift vector, (2p + 1)² in real storage.
    shift: Vec<T>,
    /// Rotation blocks Dⁿ, n ≤ p (CONVENTIONS §3.8), `blocks_len(p)` reals.
    blocks: Vec<T>,
    /// The input rotated into the coaxial frame, and the coaxial output, (p + 1)² each.
    rotated: [Vec<T>; 2],
    /// Solid harmonics of order 0 of a coaxial shift, degrees 0 to 2p.
    axis: Vec<T>,
}

/// The buffers of a rotation-based translation of degree p, borrowed from a
/// [`Workspace`] by [`Workspace::rotation`].
pub(crate) struct RotationBuffers<'a, T> {
    /// Rotation blocks Dⁿ, n ≤ p, `blocks_len(p)` reals.
    pub blocks: &'a mut [T],
    /// Input coefficients in the rotated frame, (p + 1)² reals.
    pub rotated_in: &'a mut [T],
    /// Output coefficients in the rotated frame, (p + 1)² reals.
    pub rotated_out: &'a mut [T],
    /// Order-0 harmonics of the coaxial shift, 2p + 1 reals (degrees 0 to 2p).
    pub axis: &'a mut [T],
}

impl<T: RealScalar> Workspace<T> {
    /// Allocates a workspace for operators of degree at most `p`.
    pub fn new(p: usize) -> Self {
        let len = Layout::new(p).len();
        Self {
            p,
            harmonics: vec![T::zero(); len],
            gradient: core::array::from_fn(|_| vec![T::zero(); len]),
            shift: vec![T::zero(); Layout::new(2 * p).len()],
            blocks: vec![T::zero(); blocks_len(p)],
            rotated: core::array::from_fn(|_| vec![T::zero(); len]),
            axis: vec![T::zero(); 2 * p + 1],
        }
    }

    /// Largest degree p this workspace serves.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Panics unless this workspace serves degree `p`.
    fn check(&self, p: usize) {
        assert!(
            p <= self.p,
            "Workspace built for p = {} is too small for p = {p}",
            self.p
        );
    }

    /// The solid-harmonic buffer for degree `p`, of length (p + 1)².
    ///
    /// # Panics
    ///
    /// If `p` exceeds [`p`](Self::p).
    pub(crate) fn harmonics(&mut self, p: usize) -> &mut [T] {
        self.check(p);
        &mut self.harmonics[..Layout::new(p).len()]
    }

    /// The solid-harmonic buffer and the three gradient buffers for degree `p`, each of
    /// length (p + 1)².
    ///
    /// # Panics
    ///
    /// If `p` exceeds [`p`](Self::p).
    pub(crate) fn harmonics_and_gradient(&mut self, p: usize) -> (&mut [T], [&mut [T]; 3]) {
        self.check(p);
        let len = Layout::new(p).len();
        let [gx, gy, gz] = &mut self.gradient;
        (
            &mut self.harmonics[..len],
            [&mut gx[..len], &mut gy[..len], &mut gz[..len]],
        )
    }

    /// The buffer for the solid harmonics of a translation's shift vector up to degree
    /// `degree`, of length (degree + 1)², for a translation of degree `p`: `degree` is
    /// p for M2M and L2L and 2p for M2L (CONVENTIONS §3.11).
    ///
    /// # Panics
    ///
    /// If `p` exceeds [`p`](Self::p), or if `degree` exceeds 2p.
    pub(crate) fn shift(&mut self, p: usize, degree: usize) -> &mut [T] {
        self.check(p);
        assert!(
            degree <= 2 * p,
            "shift harmonics of degree {degree} exceed 2p = {}",
            2 * p
        );
        &mut self.shift[..Layout::new(degree).len()]
    }

    /// The buffers of a rotation-based translation of degree `p`: the rotation blocks
    /// of degrees 0 to p, two coefficient vectors of length (p + 1)², and 2p + 1 axis
    /// values.
    ///
    /// # Panics
    ///
    /// If `p` exceeds [`p`](Self::p).
    pub(crate) fn rotation(&mut self, p: usize) -> RotationBuffers<'_, T> {
        self.check(p);
        let len = Layout::new(p).len();
        let [rotated_in, rotated_out] = &mut self.rotated;
        RotationBuffers {
            blocks: &mut self.blocks[..blocks_len(p)],
            rotated_in: &mut rotated_in[..len],
            rotated_out: &mut rotated_out[..len],
            axis: &mut self.axis[..2 * p + 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocates_for_p() {
        let mut ws = Workspace::<f64>::new(5);
        assert_eq!(ws.p(), 5);
        assert_eq!(ws.harmonics(5).len(), 36);
        assert_eq!(ws.harmonics(2).len(), 9);
        let (values, [gx, gy, gz]) = ws.harmonics_and_gradient(3);
        assert_eq!([values.len(), gx.len(), gy.len(), gz.len()], [16; 4]);
        assert_eq!(Workspace::<f32>::new(0).harmonics(0).len(), 1);
        assert_eq!(ws.shift(5, 10).len(), 121);
        assert_eq!(ws.shift(5, 5).len(), 36);
        assert_eq!(ws.shift(2, 4).len(), 25);
        assert_eq!(Workspace::<f32>::new(0).shift(0, 0).len(), 1);
        for (built, p) in [(5, 5), (5, 3), (0, 0)] {
            let mut ws = Workspace::<f64>::new(built);
            let buffers = ws.rotation(p);
            assert_eq!(buffers.blocks.len(), blocks_len(p));
            assert_eq!(buffers.rotated_in.len(), (p + 1) * (p + 1));
            assert_eq!(buffers.rotated_out.len(), (p + 1) * (p + 1));
            assert_eq!(buffers.axis.len(), 2 * p + 1);
        }
    }

    #[test]
    #[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
    fn rotation_rejects_larger_p() {
        let _ = Workspace::<f64>::new(3).rotation(4);
    }

    #[test]
    #[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
    fn shift_rejects_larger_p() {
        let _ = Workspace::<f64>::new(3).shift(4, 4);
    }

    #[test]
    #[should_panic(expected = "shift harmonics of degree 5 exceed 2p = 4")]
    fn shift_rejects_degree_above_2p() {
        let _ = Workspace::<f64>::new(3).shift(2, 5);
    }

    #[test]
    #[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
    fn harmonics_rejects_larger_p() {
        let _ = Workspace::<f64>::new(3).harmonics(4);
    }

    #[test]
    #[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
    fn harmonics_and_gradient_rejects_larger_p() {
        let _ = Workspace::<f32>::new(3).harmonics_and_gradient(4);
    }
}
