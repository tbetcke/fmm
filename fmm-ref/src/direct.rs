//! Direct O(p⁴) translations M2M, L2L and M2L (CONVENTIONS §3.11).
//!
//! Each operator evaluates the solid harmonics of its shift vector once per call, into
//! the [`Workspace`], and then forms the sums of CONVENTIONS §3.11 term by term on real
//! storage (§3.6): a factor of any order q is read from its stored order |q| by the
//! (−1)ᵐ rule, and the products are formed in real arithmetic. Only the outputs with
//! 0 ≤ i ≤ j are formed; the others follow from the (−1)ᵐ rule, which every operator
//! preserves. This is the oracle for the rotation-based translations and for the
//! operator tables of later phases, so the sums are written for clarity, not speed.
//!
//! All three take the degree `p` first, then the [`Frame`] of the input and of the
//! output expansion, the workspace, the input and the output coefficients. Input and
//! output have the same degree p, (p + 1)² reals each in the real storage of §3.6, and
//! are the scaled coefficients of §3.7. Every operator adds (+=) into its output and
//! none applies 1/(4π) (§3.1). None checks the relative position of the frames:
//! M2M and L2L are exact for any two frames, and the convergence condition of M2L is
//! the caller's to meet (see [`m2l`]).
//!
//! # Panics
//!
//! Every operator panics if the workspace was built for a smaller degree than `p`, or
//! if a coefficient slice does not have length (p + 1)².
//!
//! # Example
//!
//! M2M from a child box to its parent gives the multipole expansion that P2M forms at
//! the parent directly.
//!
//! ```
//! use nd_fmm_ref::{Frame, Workspace, direct, leaf};
//!
//! let p = 10;
//! let mut ws = Workspace::new(p);
//! let child = Frame::new([0.25, 0.25, 0.25], 0.25);
//! let parent = Frame::new([0.0, 0.0, 0.0], 0.5);
//! let sources: [[f64; 3]; 2] = [[0.3, 0.1, 0.4], [0.2, 0.45, 0.05]];
//! let charges = [1.0, -0.5];
//!
//! let mut at_child = vec![0.0; (p + 1) * (p + 1)];
//! leaf::p2m(p, &child, &sources, &charges, &mut ws, &mut at_child);
//! let mut translated = vec![0.0; (p + 1) * (p + 1)];
//! direct::m2m(p, &child, &parent, &mut ws, &at_child, &mut translated);
//!
//! let mut at_parent = vec![0.0; (p + 1) * (p + 1)];
//! leaf::p2m(p, &parent, &sources, &charges, &mut ws, &mut at_parent);
//! for (a, b) in translated.iter().zip(&at_parent) {
//!     assert!((a - b).abs() < 1e-14);
//! }
//! ```

use nd_fmm_math::{Layout, RealScalar, harmonics};

use crate::leaf::check_coefficients;
use crate::{Frame, Workspace};

/// M2M: adds the multipole expansion `multipole_in` in the frame `from` = (c, r),
/// translated to the frame `to` = (c′, r′), to `multipole_out` (CONVENTIONS §3.11,
/// "M2M").
///
/// With the shift b = (c − c′)/r′ and the radius factor ρ = r/r′,
///
/// M̃′ⱼⁱ += Σₖ₌₀ʲ Σₗ₌₋ₖᵏ ρᵏ M̃ₖˡ conj(Rⱼ₋ₖⁱ⁻ˡ(b)),  0 ≤ j ≤ p, |i| ≤ j,
///
/// where terms with |i − l| > j − k vanish. The regular harmonics of b up to degree p
/// come from [`harmonics::regular`] (§3.5).
///
/// Output degree j uses input degrees k ≤ j only, so the translation is exact to
/// degree p for any two frames; there is no convergence condition. In use the output
/// sphere contains the input one: on a uniform octree r′ = 2r (child to parent).
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
    let layout = Layout::new(p);
    // b = (c − c′)/r′, formed as `Frame::scaled` forms it.
    let b = to.scaled(from.centre);
    let rho = from.radius / to.radius;
    let shift = ws.shift(p, p);
    harmonics::regular(p, b, shift);
    for j in 0..=p {
        for i in 0..=j as isize {
            let mut sum = Complex::zero();
            for k in 0..=j {
                let mut orders = Complex::zero();
                for l in -(k as isize)..=k as isize {
                    let a = read(layout, multipole_in, k, l);
                    let r = read(layout, shift, j - k, i - l).conj();
                    orders = orders + a * r;
                }
                sum = sum + orders.scale(power(rho, k));
            }
            add(layout, multipole_out, j, i, sum);
        }
    }
}

/// L2L: adds the local expansion `local_in` in the frame `from` = (c, r), translated
/// to the frame `to` = (c′, r′), to `local_out` (CONVENTIONS §3.11, "L2L").
///
/// With the shift t = (c′ − c)/r and the radius factor σ = r′/r,
///
/// L̃′ⱼⁱ += σʲ⁺¹ Σₙ₌ⱼᵖ Σₘ₌₋ₙⁿ L̃ₙᵐ Rₙ₋ⱼᵐ⁻ⁱ(t),  0 ≤ j ≤ p, |i| ≤ j,
///
/// where terms with |m − i| > n − j vanish. The regular harmonics of t up to degree p
/// come from [`harmonics::regular`] (§3.5).
///
/// A local expansion of degree p is a polynomial of degree p, and L2L re-expands it
/// exactly in the output frame; there is no convergence condition. In use the output
/// sphere lies inside the input one: on a uniform octree r′ = r/2 (parent to child).
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
    let layout = Layout::new(p);
    // t = (c′ − c)/r, formed as `Frame::scaled` forms it.
    let t = from.scaled(to.centre);
    let sigma = to.radius / from.radius;
    let shift = ws.shift(p, p);
    harmonics::regular(p, t, shift);
    for j in 0..=p {
        for i in 0..=j as isize {
            let mut sum = Complex::zero();
            for n in j..=p {
                for m in -(n as isize)..=n as isize {
                    let a = read(layout, local_in, n, m);
                    let r = read(layout, shift, n - j, m - i);
                    sum = sum + a * r;
                }
            }
            add(layout, local_out, j, i, sum.scale(power(sigma, j + 1)));
        }
    }
}

/// M2L: adds the local expansion, in the frame `target` = (c′, r′), of the multipole
/// expansion `multipole` in the frame `source` = (c, r) to `local`
/// (CONVENTIONS §3.11, "M2L").
///
/// With the shift b = (c′ − c)/r and the radius factor σ = r′/r,
///
/// L̃′ⱼⁱ += (−1)ʲ⁺ⁱ σʲ⁺¹ Σₙ₌₀ᵖ Σₘ₌₋ₙⁿ M̃ₙᵐ Iₙ₊ⱼᵐ⁻ⁱ(b),  0 ≤ j ≤ p, |i| ≤ j.
///
/// No term vanishes. The irregular harmonics of b up to degree 2p come from
/// [`harmonics::irregular`] (§3.5); input and output have the same degree p.
///
/// Convergence: with the sources within √3 r of c and the targets within √3 r′ of c′,
/// the expansion converges when √3 (r + r′) < |c′ − c|, that is |b| > √3 (1 + σ). On a
/// uniform octree level (σ = 1, V list) 4 ≤ |b| ≤ 6√3. The error of truncating the
/// input at degree p is bounded in CONVENTIONS §3.11. This function does not check the
/// separation; for c′ = c the harmonics, and hence the output, are not finite.
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
    let layout = Layout::new(p);
    let shift_layout = Layout::new(2 * p);
    // b = (c′ − c)/r, formed as `Frame::scaled` forms it.
    let b = source.scaled(target.centre);
    let sigma = target.radius / source.radius;
    let shift = ws.shift(p, 2 * p);
    harmonics::irregular(2 * p, b, shift);
    for j in 0..=p {
        for i in 0..=j as isize {
            let mut sum = Complex::zero();
            for n in 0..=p {
                for m in -(n as isize)..=n as isize {
                    let a = read(layout, multipole, n, m);
                    let irregular = read(shift_layout, shift, n + j, m - i);
                    sum = sum + a * irregular;
                }
            }
            let sign = if (j as isize + i) % 2 == 0 {
                T::one()
            } else {
                -T::one()
            };
            add(layout, local, j, i, sum.scale(sign * power(sigma, j + 1)));
        }
    }
}

/// A complex number as a pair of reals, for the products of CONVENTIONS §3.11
/// ("Real storage"). Only what the sums need.
#[derive(Clone, Copy)]
struct Complex<T> {
    re: T,
    im: T,
}

impl<T: RealScalar> Complex<T> {
    fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn scale(self, s: T) -> Self {
        Self {
            re: s * self.re,
            im: s * self.im,
        }
    }
}

impl<T: RealScalar> core::ops::Add for Complex<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl<T: RealScalar> core::ops::Mul for Complex<T> {
    type Output = Self;

    /// Re = Re·Re − Im·Im, Im = Re·Im + Im·Re (CONVENTIONS §3.11, "Real storage").
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Reads the value of degree `n` and order `q` from real storage (CONVENTIONS §3.6,
/// §3.11 "Real storage"): order m ≥ 0 is (slot m, slot −m), with imaginary part 0 for
/// m = 0; order −m < 0 is (−1)ᵐ (slot m, −slot −m), which is (−1)ᵐ conj of order m;
/// orders with |q| > n are 0. Valid for every quantity that satisfies
/// X⁻ᵐ = (−1)ᵐ conj(Xᵐ): scaled coefficients and the solid harmonics R and I.
fn read<T: RealScalar>(layout: Layout, values: &[T], n: usize, q: isize) -> Complex<T> {
    let m = q.unsigned_abs();
    if m > n {
        return Complex::zero();
    }
    let re = values[layout.idx(n, m as isize)];
    let im = if m == 0 {
        T::zero()
    } else {
        values[layout.idx(n, -(m as isize))]
    };
    let value = Complex { re, im };
    match (q < 0, m % 2 == 1) {
        (false, _) => value,
        (true, false) => value.conj(),
        (true, true) => value.conj().scale(-T::one()),
    }
}

/// Adds the output of degree `j` and order i ≥ 0 into real storage: Re into slot +i
/// (slot 0 for i = 0, where the imaginary part vanishes), Im into slot −i
/// (CONVENTIONS §3.11, "Real storage").
fn add<T: RealScalar>(layout: Layout, out: &mut [T], j: usize, i: isize, value: Complex<T>) {
    let re = layout.idx(j, i);
    out[re] = out[re] + value.re;
    if i > 0 {
        let im = layout.idx(j, -i);
        out[im] = out[im] + value.im;
    }
}

/// xᵏ by repeated multiplication.
fn power<T: RealScalar>(x: T, k: usize) -> T {
    (0..k).fold(T::one(), |acc, _| acc * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_applies_the_order_rule() {
        // Error measure: exact equality; all values are small integers.
        // Degree 1 in real storage: slots (1, −1), (1, 0), (1, 1) at indices 1, 2, 3.
        let layout = Layout::new(1);
        let values = [7.0, 2.0, 5.0, 3.0];
        let at = |q| {
            let c: Complex<f64> = read(layout, &values, 1, q);
            (c.re, c.im)
        };
        assert_eq!(at(0), (5.0, 0.0));
        assert_eq!(at(1), (3.0, 2.0));
        // X⁻¹ = −conj(X¹).
        assert_eq!(at(-1), (-3.0, 2.0));
        assert_eq!(at(2), (0.0, 0.0));
        assert_eq!(at(-2), (0.0, 0.0));

        // Degree 2, order −2: X⁻² = conj(X²).
        let layout = Layout::new(2);
        let mut values = [0.0; 9];
        values[layout.idx(2, 2)] = 4.0;
        values[layout.idx(2, -2)] = -6.0;
        let c: Complex<f64> = read(layout, &values, 2, -2);
        assert_eq!((c.re, c.im), (4.0, 6.0));
    }

    #[test]
    fn add_writes_real_and_imaginary_slots() {
        // Error measure: exact equality; all values are small integers.
        let layout = Layout::new(1);
        let mut out = [1.0; 4];
        add(layout, &mut out, 1, 1, Complex { re: 2.0, im: 3.0 });
        add(layout, &mut out, 0, 0, Complex { re: 5.0, im: 9.0 });
        assert_eq!(out, [6.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn power_matches_repeated_product() {
        // Error measure: exact equality; powers of two are exact.
        assert_eq!(power(2.0_f64, 0), 1.0);
        assert_eq!(power(2.0_f64, 10), 1024.0);
        assert_eq!(power(0.5_f32, 3), 0.125);
    }
}
