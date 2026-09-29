//! Rotation-based O(p³) translations M2M, L2L and M2L (CONVENTIONS §3.11).
//!
//! Each operator translates by "point and shoot" (CONVENTIONS §3.11, "Rotation of
//! coefficients"; design document §3.2), in three steps:
//!
//! 1. Rotate the input about its frame centre by the proper rotation Q with
//!    Q(c′ − c) = |c′ − c| e_z, so that the shift lies along +z. Q is the z-y-z Euler
//!    rotation R_y(−θ) R_z(−φ) of [`euler_zyz`], with θ and φ the angles of c′ − c
//!    (§3.2). Degree n of the coefficients becomes K Dⁿ(Q) K M̃ₙ for multipoles and
//!    K S Dⁿ(Q) S⁻¹ K L̃ₙ for locals, with the blocks Dⁿ of [`blocks`], S Dⁿ S⁻¹ from
//!    [`to_irregular`], S = diag((n − |m|)! (n + |m|)!) (§3.8) and
//!    K = diag(+1 for m ≥ 0, −1 for m < 0), the conjugation in real storage (§3.6).
//! 2. Apply the coaxial translation of §3.11 ("Coaxial translations") with
//!    d = |c′ − c|. It keeps the order fixed and transforms the slots +i and −i by the
//!    same real factors, the order-0 harmonics of the shift on the z axis:
//!    Rₖ⁰(s e_z) = sᵏ/k! and Iₖ⁰(s e_z) = (sgn s)ᵏ k!/|s|ᵏ⁺¹.
//! 3. Rotate the result back about the output frame centre with the rule of its kind,
//!    using the blocks of Qᵀ. In real storage the blocks are not orthogonal (§3.8), so
//!    the inverse Dⁿ(Q)⁻¹ = Dⁿ(Qᵀ) is built from Qᵀ by [`blocks`], not by transposing
//!    the blocks of Q.
//!
//! Shifts on the z axis are handled without rotation: for c′ − c = d e_z with d of
//! either sign, including d = 0, the coaxial form of §3.11 applies directly with that
//! d. This covers the zero shift, where M2M and L2L reduce to the radius rescaling,
//! and the shift along −z, where the z-y-z Euler angles are singular (θ = π leaves φ
//! undefined).
//!
//! The rotation blocks are built per call, into the [`Workspace`], by the recursion of
//! [`blocks`], which is O(p³) as well; Phase 2 precomputes them. The multiply-add
//! counts in the operator documentation exclude that construction. They compare with
//! the (10/3) p³ per M2L and 3 p³ per M2M or L2L of design §3.2: each rotation costs
//! Σₙ (2n + 1)² = (p + 1)(2p + 1)(2p + 3)/3 ≈ (4/3) p³, and the coaxial step about
//! (1/3) p³ for M2M and L2L and (2/3) p³ for M2L.
//!
//! The operators take exactly the arguments of their [`direct`](crate::direct)
//! counterparts: the degree `p`, the [`Frame`] of the input and of the output
//! expansion, the workspace, the input and the output coefficients. Input and output
//! have the same degree p, (p + 1)² reals each in the real storage of §3.6, and are
//! the scaled coefficients of §3.7. Every operator adds (+=) into its output and none
//! applies 1/(4π) (§3.1). None checks the relative position of the frames: M2M and
//! L2L are exact for any two frames, and the convergence condition of M2L is the
//! caller's to meet (see [`m2l`]).
//!
//! # Panics
//!
//! Every operator panics if the workspace was built for a smaller degree than `p`, or
//! if a coefficient slice does not have length (p + 1)².
//!
//! # Example
//!
//! The rotation-based and the direct M2L agree for a V-list neighbour.
//!
//! ```
//! use nd_fmm_ref::{Frame, Workspace, direct, rotation};
//!
//! let p = 8;
//! let mut ws = Workspace::new(p);
//! let source = Frame::new([0.0, 0.0, 0.0], 0.5);
//! let target = Frame::new([2.0, -1.0, 1.5], 0.5);
//! let multipole: Vec<f64> = (0..(p + 1) * (p + 1))
//!     .map(|i| 1.0 / (1.0 + i as f64))
//!     .collect();
//!
//! let mut by_rotation = vec![0.0; (p + 1) * (p + 1)];
//! rotation::m2l(p, &source, &target, &mut ws, &multipole, &mut by_rotation);
//! let mut by_direct = vec![0.0; (p + 1) * (p + 1)];
//! direct::m2l(p, &source, &target, &mut ws, &multipole, &mut by_direct);
//! let largest = by_direct.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
//! for (a, b) in by_rotation.iter().zip(&by_direct) {
//!     assert!((a - b).abs() < 1e-14 * largest);
//! }
//! ```
//!
//! [`blocks`]: nd_fmm_math::rotation::blocks
//! [`to_irregular`]: nd_fmm_math::rotation::to_irregular
//! [`euler_zyz`]: nd_fmm_math::rotation::euler_zyz

use nd_fmm_math::rotation::{block_range, blocks, euler_zyz, to_irregular};
use nd_fmm_math::{Layout, RealScalar};

use crate::leaf::check_coefficients;
use crate::workspace::RotationBuffers;
use crate::{Frame, Workspace};

/// M2M: adds the multipole expansion `multipole_in` in the frame `from` = (c, r),
/// translated to the frame `to` = (c′, r′), to `multipole_out` (CONVENTIONS §3.11,
/// "M2M"), by rotation, coaxial translation and rotation back (see the
/// [module documentation](self)).
///
/// With ρ = r/r′ and d = |c′ − c|, the coaxial step in the rotated frame is
///
/// M̃′ⱼⁱ = Σₖ₌|ᵢ|ʲ ρᵏ M̃ₖⁱ (−d/r′)ʲ⁻ᵏ / (j − k)!,
///
/// the M2M sum for the shift b = (c − c′)/r′ = −(d/r′) e_z. Both rotations use the
/// multipole rule K Dⁿ K. The result equals [`direct::m2m`](crate::direct::m2m), and
/// like it the translation is exact to degree p for any two frames.
///
/// Cost: (p + 1)(2p + 3)(3p + 2)/2 ≈ 3p³ multiply-adds, of which
/// 2 (p + 1)(2p + 1)(2p + 3)/3 are the two rotations and (p + 1)(p + 2)(2p + 3)/6 the
/// coaxial step; for a shift along the z axis only the coaxial step runs. Building
/// the rotation blocks is not counted.
///
/// # Panics
///
/// If `ws.p() < p`, or `multipole_in` or `multipole_out` does not have length
/// (p + 1)².
pub fn m2m<T: RealScalar>(
    p: usize,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    multipole_in: &[T],
    multipole_out: &mut [T],
) {
    check_coefficients(p, multipole_in, "multipole_in");
    check_coefficients(p, multipole_out, "multipole_out");
    let rho = from.radius / to.radius;
    let buffers = ws.rotation(p);
    // The coaxial shift b = (c − c′)/r′ along z: −d/r′.
    translate(
        p,
        Shift::new(from, to, |d| -d / to.radius),
        [Kind::Multipole, Kind::Multipole],
        buffers,
        multipole_in,
        multipole_out,
        |s, axis, input, output| coaxial_m2m(p, s, rho, axis, input, output),
    );
}

/// L2L: adds the local expansion `local_in` in the frame `from` = (c, r), translated
/// to the frame `to` = (c′, r′), to `local_out` (CONVENTIONS §3.11, "L2L"), by
/// rotation, coaxial translation and rotation back (see the
/// [module documentation](self)).
///
/// With σ = r′/r and d = |c′ − c|, the coaxial step in the rotated frame is
///
/// L̃′ⱼⁱ = σʲ⁺¹ Σₙ₌ⱼᵖ L̃ₙⁱ (d/r)ⁿ⁻ʲ / (n − j)!,
///
/// the L2L sum for the shift t = (c′ − c)/r = (d/r) e_z. Both rotations use the local
/// rule K S Dⁿ S⁻¹ K. The result equals [`direct::l2l`](crate::direct::l2l), and like
/// it the translation is exact for any two frames.
///
/// Cost: (p + 1)(2p + 3)(3p + 2)/2 ≈ 3p³ multiply-adds, of which
/// 2 (p + 1)(2p + 1)(2p + 3)/3 are the two rotations and (p + 1)(p + 2)(2p + 3)/6 the
/// coaxial step; for a shift along the z axis only the coaxial step runs. Building
/// the rotation blocks is not counted.
///
/// # Panics
///
/// If `ws.p() < p`, or `local_in` or `local_out` does not have length (p + 1)².
pub fn l2l<T: RealScalar>(
    p: usize,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    local_in: &[T],
    local_out: &mut [T],
) {
    check_coefficients(p, local_in, "local_in");
    check_coefficients(p, local_out, "local_out");
    let sigma = to.radius / from.radius;
    let buffers = ws.rotation(p);
    // The coaxial shift t = (c′ − c)/r along z: d/r.
    translate(
        p,
        Shift::new(from, to, |d| d / from.radius),
        [Kind::Local, Kind::Local],
        buffers,
        local_in,
        local_out,
        |s, axis, input, output| coaxial_l2l(p, s, sigma, axis, input, output),
    );
}

/// M2L: adds the local expansion, in the frame `target` = (c′, r′), of the multipole
/// expansion `multipole` in the frame `source` = (c, r) to `local`
/// (CONVENTIONS §3.11, "M2L"), by rotation, coaxial translation and rotation back (see
/// the [module documentation](self)).
///
/// With σ = r′/r and d = |c′ − c|, the coaxial step in the rotated frame is
///
/// L̃′ⱼⁱ = (−1)ʲ⁺ⁱ σʲ⁺¹ Σₙ₌|ᵢ|ᵖ M̃ₙⁱ (n + j)! (r/d)ⁿ⁺ʲ⁺¹,
///
/// the M2L sum for the shift b = (c′ − c)/r = (d/r) e_z, with the axis values
/// Iₙ₊ⱼ⁰(b) up to degree 2p. The input rotates by the multipole rule K Dⁿ K, the
/// output back by the local rule K S Dⁿ S⁻¹ K. The result equals
/// [`direct::m2l`](crate::direct::m2l).
///
/// Convergence: as for [`direct::m2l`](crate::direct::m2l), the expansion converges
/// when √3 (r + r′) < |c′ − c|, that is |b| > √3 (1 + σ); this function does not
/// check it. For c′ = c the output is not finite.
///
/// Cost: (p + 1)(10p² + 20p + 9)/3 ≈ (10/3) p³ multiply-adds, of which
/// 2 (p + 1)(2p + 1)(2p + 3)/3 are the two rotations and (p + 1)(2p² + 4p + 3)/3 the
/// coaxial step; for a shift along the z axis only the coaxial step runs. Building
/// the rotation blocks is not counted.
///
/// # Panics
///
/// If `ws.p() < p`, or `multipole` or `local` does not have length (p + 1)².
pub fn m2l<T: RealScalar>(
    p: usize,
    source: &Frame<T>,
    target: &Frame<T>,
    ws: &mut Workspace<T>,
    multipole: &[T],
    local: &mut [T],
) {
    check_coefficients(p, multipole, "multipole");
    check_coefficients(p, local, "local");
    let sigma = target.radius / source.radius;
    let buffers = ws.rotation(p);
    // The coaxial shift b = (c′ − c)/r along z: d/r.
    translate(
        p,
        Shift::new(source, target, |d| d / source.radius),
        [Kind::Multipole, Kind::Local],
        buffers,
        multipole,
        local,
        |s, axis, input, output| coaxial_m2l(p, s, sigma, axis, input, output),
    );
}

/// The rotation rule of a coefficient vector (CONVENTIONS §3.11, "Rotation of
/// coefficients").
#[derive(Clone, Copy)]
enum Kind {
    /// Conjugates of regular harmonics: degree n rotates by K Dⁿ K.
    Multipole,
    /// Conjugates of irregular harmonics: degree n rotates by K S Dⁿ S⁻¹ K.
    Local,
}

/// The shift c′ − c of a translation, how it is brought onto the z axis, and the
/// scaled coaxial shift s of the operator (s e_z is the shift vector of its §3.11 sum
/// in the coaxial frame).
enum Shift<T> {
    /// c′ − c = d e_z, with d of either sign or zero: no rotation.
    Axis { s: T },
    /// Any other shift: Q(c′ − c) = d e_z with d = |c′ − c| > 0.
    Rotated { q: [[T; 3]; 3], s: T },
}

impl<T: RealScalar> Shift<T> {
    /// The shift from the centre of `from` to the centre of `to`; `scaled` maps the
    /// signed distance d along the coaxial z axis to s.
    fn new(from: &Frame<T>, to: &Frame<T>, scaled: impl Fn(T) -> T) -> Self {
        let [x, y, z]: [T; 3] = core::array::from_fn(|i| to.centre[i] - from.centre[i]);
        let zero = T::zero();
        if x == zero && y == zero {
            return Shift::Axis { s: scaled(z) };
        }
        let planar = (x * x + y * y).sqrt();
        let d = (x * x + y * y + z * z).sqrt();
        // Polar angle θ and azimuth φ of c′ − c (CONVENTIONS §3.2);
        // Q = R_y(−θ) R_z(−φ) maps it to d e_z (§3.11).
        let theta = planar.atan2(z);
        let phi = y.atan2(x);
        let q = euler_zyz(zero, -theta, -phi);
        Shift::Rotated { q, s: scaled(d) }
    }
}

/// Runs the three steps of a rotation-based translation (see the
/// [module documentation](self)): rotates `input` by Q with the rule `kinds[0]`,
/// applies `coaxial` to the rotated coefficients, and adds the result, rotated back by
/// Qᵀ with the rule `kinds[1]`, to `output`. For a shift on the z axis, `coaxial` adds
/// into `output` directly.
///
/// `coaxial(s, axis, input, output)` adds the coaxial translation of `input` for the
/// scaled shift s e_z to `output`, using `axis` for its axis values.
fn translate<T: RealScalar>(
    p: usize,
    shift: Shift<T>,
    kinds: [Kind; 2],
    buffers: RotationBuffers<'_, T>,
    input: &[T],
    output: &mut [T],
    coaxial: impl Fn(T, &mut [T], &[T], &mut [T]),
) {
    let RotationBuffers {
        blocks: d_blocks,
        rotated_in,
        rotated_out,
        axis,
    } = buffers;
    match shift {
        Shift::Axis { s } => coaxial(s, axis, input, output),
        Shift::Rotated { q, s } => {
            rotation_blocks(p, &q, kinds[0], d_blocks);
            rotated_in.fill(T::zero());
            rotate(p, d_blocks, input, rotated_in);

            rotated_out.fill(T::zero());
            coaxial(s, axis, rotated_in, rotated_out);

            let q_transposed: [[T; 3]; 3] = core::array::from_fn(|i| q.map(|row| row[i]));
            rotation_blocks(p, &q_transposed, kinds[1], d_blocks);
            rotate(p, d_blocks, rotated_out, output);
        }
    }
}

/// Writes the blocks that rotate coefficients of the given kind by `q` to `out`: Dⁿ(Q)
/// for multipoles and S Dⁿ(Q) S⁻¹ for locals (CONVENTIONS §3.8). The conjugation K is
/// applied by [`rotate`].
fn rotation_blocks<T: RealScalar>(p: usize, q: &[[T; 3]; 3], kind: Kind, out: &mut [T]) {
    blocks(p, q, out);
    if let Kind::Local = kind {
        to_irregular(p, out);
    }
}

/// Adds K Bⁿ K xₙ to `out` for every degree n ≤ p, with Bⁿ the degree-n block of
/// `blocks` (row-major, rows and columns in the order m = −n to n) and
/// K = diag(+1 for m ≥ 0, −1 for m < 0) (CONVENTIONS §3.11, "Rotation of
/// coefficients"). (2n + 1)² multiply-adds per degree.
fn rotate<T: RealScalar>(p: usize, blocks: &[T], x: &[T], out: &mut [T]) {
    let layout = Layout::new(p);
    for (n, range) in layout.degrees() {
        let block = &blocks[block_range(n)];
        let x = &x[range.clone()];
        let rows = block.chunks_exact(2 * n + 1);
        for (i, (row, y)) in rows.zip(&mut out[range]).enumerate() {
            let mut sum = T::zero();
            for (j, (&b, &x)) in row.iter().zip(x).enumerate() {
                // K on the columns: slots m < 0 come first.
                sum = if j < n { sum - b * x } else { sum + b * x };
            }
            // K on the rows.
            *y = if i < n { *y - sum } else { *y + sum };
        }
    }
}

/// The coaxial M2M of CONVENTIONS §3.11 for the shift b = s e_z (s = −d/r′) and
/// ρ = r/r′: adds M̃′ⱼⁱ = Σₖ₌|ᵢ|ʲ ρᵏ M̃ₖⁱ Rⱼ₋ₖ⁰(b) to `output`, with
/// Rₖ⁰(s e_z) = sᵏ/k!. conj Rⱼ₋ₖ⁰(b) = Rⱼ₋ₖ⁰(b), since it is real.
fn coaxial_m2m<T: RealScalar>(
    p: usize,
    s: T,
    rho: T,
    axis: &mut [T],
    input: &[T],
    output: &mut [T],
) {
    let axis = &mut axis[..=p];
    axis_regular(s, axis);
    let layout = Layout::new(p);
    let mut rho_k = T::one();
    for k in 0..=p {
        let input = &input[layout.degree(k)];
        for j in k..=p {
            let factor = rho_k * axis[j - k];
            // Orders |i| ≤ k of degree j.
            let centre = layout.idx(j, 0);
            let output = &mut output[centre - k..=centre + k];
            for (y, &x) in output.iter_mut().zip(input) {
                *y = *y + factor * x;
            }
        }
        rho_k = rho_k * rho;
    }
}

/// The coaxial L2L of CONVENTIONS §3.11 for the shift t = s e_z (s = d/r) and
/// σ = r′/r: adds L̃′ⱼⁱ = σʲ⁺¹ Σₙ₌ⱼᵖ L̃ₙⁱ Rₙ₋ⱼ⁰(t) to `output`, with
/// Rₖ⁰(s e_z) = sᵏ/k!.
fn coaxial_l2l<T: RealScalar>(
    p: usize,
    s: T,
    sigma: T,
    axis: &mut [T],
    input: &[T],
    output: &mut [T],
) {
    let axis = &mut axis[..=p];
    axis_regular(s, axis);
    let layout = Layout::new(p);
    let mut sigma_j = sigma;
    for j in 0..=p {
        let output = &mut output[layout.degree(j)];
        for n in j..=p {
            let factor = sigma_j * axis[n - j];
            // Orders |i| ≤ j of degree n.
            let centre = layout.idx(n, 0);
            let input = &input[centre - j..=centre + j];
            for (y, &x) in output.iter_mut().zip(input) {
                *y = *y + factor * x;
            }
        }
        sigma_j = sigma_j * sigma;
    }
}

/// The coaxial M2L of CONVENTIONS §3.11 for the shift b = s e_z (s = d/r) and
/// σ = r′/r: adds L̃′ⱼⁱ = (−1)ʲ⁺ⁱ σʲ⁺¹ Σₙ₌|ᵢ|ᵖ M̃ₙⁱ Iₙ₊ⱼ⁰(b) to `output`, with
/// Iₖ⁰(s e_z) = (sgn s)ᵏ k!/|s|ᵏ⁺¹. Slot −i takes the sign (−1)ʲ⁺ⁱ of slot +i.
fn coaxial_m2l<T: RealScalar>(
    p: usize,
    s: T,
    sigma: T,
    axis: &mut [T],
    input: &[T],
    output: &mut [T],
) {
    axis_irregular(s, axis);
    let layout = Layout::new(p);
    let mut sigma_j = sigma;
    for j in 0..=p {
        let centre_out = layout.idx(j, 0);
        for n in 0..=p {
            // (−1)ʲ σʲ⁺¹ Iₙ₊ⱼ⁰(b); the factor (−1)ⁱ follows per order.
            let factor = if j % 2 == 0 { sigma_j } else { -sigma_j } * axis[n + j];
            let centre_in = layout.idx(n, 0);
            for i in 0..=j.min(n) {
                let signed = if i % 2 == 0 { factor } else { -factor };
                let (y, x) = (centre_out + i, centre_in + i);
                output[y] = output[y] + signed * input[x];
                if i > 0 {
                    let (y, x) = (centre_out - i, centre_in - i);
                    output[y] = output[y] + signed * input[x];
                }
            }
        }
        sigma_j = sigma_j * sigma;
    }
}

/// Writes Rₖ⁰(s e_z) = sᵏ/k!, k < `out.len()`, to `out` (CONVENTIONS §3.11, "Coaxial
/// translations"), by Rₖ⁰ = Rₖ₋₁⁰ s/k.
fn axis_regular<T: RealScalar>(s: T, out: &mut [T]) {
    let mut value = T::one();
    for (k, slot) in out.iter_mut().enumerate() {
        if k > 0 {
            value = value * s / T::from_f64(k as f64);
        }
        *slot = value;
    }
}

/// Writes Iₖ⁰(s e_z) = (sgn s)ᵏ k!/|s|ᵏ⁺¹, k < `out.len()`, to `out` (CONVENTIONS
/// §3.11, "Coaxial translations"), by I₀⁰ = 1/|s| and Iₖ⁰ = Iₖ₋₁⁰ k/s. Not finite
/// for s = 0.
fn axis_irregular<T: RealScalar>(s: T, out: &mut [T]) {
    let mut value = T::one() / s.abs();
    for (k, slot) in out.iter_mut().enumerate() {
        if k > 0 {
            value = value * T::from_f64(k as f64) / s;
        }
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_values_match_the_closed_forms() {
        // Error measure: relative error against sᵏ/k! and (sgn s)ᵏ k!/|s|ᵏ⁺¹ evaluated
        // with powi and a product of integers; a few roundings per degree.
        for s in [0.75_f64, -1.5, 4.0, -6.0] {
            let mut regular = [0.0; 21];
            let mut irregular = [0.0; 41];
            axis_regular(s, &mut regular);
            axis_irregular(s, &mut irregular);
            let factorial = |k: usize| (1..=k).map(|j| j as f64).product::<f64>();
            for (k, &value) in regular.iter().enumerate() {
                let want = s.powi(k as i32) / factorial(k);
                assert!(
                    (value - want).abs() <= 1e-14 * want.abs(),
                    "R, s = {s}, k = {k}"
                );
            }
            for (k, &value) in irregular.iter().enumerate() {
                let want = factorial(k) / s.powi(k as i32) / s.abs();
                assert!(
                    (value - want).abs() <= 1e-14 * want.abs(),
                    "I, s = {s}, k = {k}"
                );
            }
        }
        // Zero shift: R₀⁰ = 1 and Rₖ⁰ = 0 for k > 0.
        let mut regular = [f64::NAN; 4];
        axis_regular(0.0, &mut regular);
        assert_eq!(regular, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn shift_classifies_axis_and_general_shifts() {
        // Error measure: exact classification; for the general shift, the absolute
        // error of Q(c′ − c) against d e_z, relative to d, bound 4 ε.
        let from = Frame::new([0.5, -0.25, 1.0], 0.5);
        for (dz, want) in [(0.0, 0.0), (1.5, 1.5), (-2.0, -2.0)] {
            let to = Frame::new([0.5, -0.25, 1.0 + dz], 0.5);
            match Shift::new(&from, &to, |d| d) {
                Shift::Axis { s } => assert_eq!(s, want),
                Shift::Rotated { .. } => panic!("dz = {dz}: expected an axis shift"),
            }
        }
        for t in [[1.0, 2.0, -0.5], [1e-9, 0.0, -1.0], [0.0, -3.0, 0.0]] {
            let to = Frame::new(core::array::from_fn(|i| from.centre[i] + t[i]), 0.5);
            let Shift::Rotated { q, s: d } = Shift::new(&from, &to, |d| d) else {
                panic!("{t:?}: expected a rotated shift");
            };
            let qt: [f64; 3] = core::array::from_fn(|i| (0..3).map(|j| q[i][j] * t[j]).sum());
            let want = [0.0, 0.0, d];
            for i in 0..3 {
                assert!(
                    (qt[i] - want[i]).abs() <= 4.0 * f64::EPSILON * d,
                    "{t:?}: {qt:?}"
                );
            }
        }
    }
}
