//! Normalised associated Legendre functions to large degree (CONVENTIONS §3.15).
//!
//! P̄ₙᵐ(x) = √((n − m)!/(n + m)!) Pₙᵐ(x), 0 ≤ m ≤ n, with Pₙᵐ as in CONVENTIONS §3.3
//! (no Condon–Shortley phase), for x = cos θ and s = sin θ ≥ 0, by the recursion in n
//! for each order m:
//!
//! P̄ₘᵐ = Πᵢ₌₁..ₘ √((2i − 1)/(2i)) · sᵐ,  P̄ₘ₊₁ᵐ = √(2m + 1) x P̄ₘᵐ,
//! P̄ₙᵐ = ((2n − 1) x P̄ₙ₋₁ᵐ − cₙ₋₁ P̄ₙ₋₂ᵐ) / cₙ,  cₙ = √((n − m)(n + m)).
//!
//! In this form the recursion's own rounding grows like n² ε near x = ±1, where Pₙ moves
//! by about n²/2 per unit of x and the coefficients nearly cancel (spike S1,
//! `spikes/rotation-blocks/SPIKE_REPORT.md` §2.3: 2.75·10⁻¹² in P̄₁₀₀₀⁰ at θ = 10⁻³). The
//! recursion therefore runs on |x|, with u = 1 − |x| = s²/(1 + |x|), in telescoped form:
//! with Eₙ = cₙ (P̄ₙᵐ − P̄ₙ₋₁ᵐ),
//!
//! Eₙ = Eₙ₋₁ + (rₙ − (2n − 1) u) P̄ₙ₋₁ᵐ,  P̄ₙᵐ = P̄ₙ₋₁ᵐ + Eₙ / cₙ,
//!
//! where rₙ = 2n − 1 − cₙ − cₙ₋₁ = m²/(n + cₙ) + m²/(n − 1 + cₙ₋₁) is formed without
//! cancellation. x < 0 follows from P̄ₙᵐ(−x) = (−1)ⁿ⁺ᵐ P̄ₙᵐ(x).
//!
//! The start value falls like sᵐ and leaves the floating-point range for m of a few
//! hundred when s is small, while P̄ₙᵐ at n ≫ m can still be of order one. Each column
//! therefore runs with a binary exponent of its own, as in the X-numbers of Fukushima
//! (2012), and is scaled back when stored.
//!
//! [`normalised_over_sin`] runs the same recursion from the start value divided by s, so
//! that P̄ₙᵐ / sin θ (m ≥ 1), which the φ-derivative of the harmonics needs, is regular
//! at the poles.

use crate::RealScalar;

/// Number of values of degrees 0 to `nmax`, (nmax + 1)(nmax + 2)/2.
///
/// ```
/// use nd_fmm_math::orthonormal::legendre;
///
/// assert_eq!(legendre::len(0), 1);
/// assert_eq!(legendre::len(2), 6);
/// ```
#[inline]
pub const fn len(nmax: usize) -> usize {
    (nmax + 1) * (nmax + 2) / 2
}

/// Index of P̄ₙᵐ, 0 ≤ m ≤ n, in a table of [`len`] values: n(n + 1)/2 + m.
#[inline]
pub const fn index(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

/// Writes P̄ₙᵐ(x), 0 ≤ m ≤ n ≤ `nmax`, to `out` at [`index`]`(n, m)`, for x = cos θ and
/// s = sin θ ≥ 0.
///
/// ```
/// use nd_fmm_math::orthonormal::legendre;
///
/// let (x, s) = (0.6_f64, 0.8_f64);
/// let mut out = vec![0.0; legendre::len(2)];
/// legendre::normalised(2, x, s, &mut out);
/// // P̄₁¹ = P₁¹/√2 = s/√2 and P̄₂⁰ = P₂ = (3x² − 1)/2
/// assert!((out[legendre::index(1, 1)] - s / 2f64.sqrt()).abs() < 1e-15);
/// assert!((out[legendre::index(2, 0)] - (3.0 * x * x - 1.0) / 2.0).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// If `out.len() != len(nmax)`.
pub fn normalised<T: RealScalar>(nmax: usize, x: T, s: T, out: &mut [T]) {
    table(nmax, x, s, false, out);
}

/// Writes P̄ₙᵐ(x) / s, 1 ≤ m ≤ n ≤ `nmax`, to `out` at [`index`]`(n, m)`, and zero at
/// m = 0, for x = cos θ and s = sin θ ≥ 0. The values are finite at s = 0: there
/// P̄ₙ¹ / s is the limit (±1)ⁿ⁺¹ √(n(n + 1))/2 and every other order is zero.
///
/// ```
/// use nd_fmm_math::orthonormal::legendre;
///
/// let mut out = vec![0.0; legendre::len(1)];
/// legendre::normalised_over_sin(1, 1.0_f64, 0.0, &mut out);
/// // P̄₁¹ / sin θ = 1/√2 for every θ
/// assert!((out[legendre::index(1, 1)] - 0.5f64.sqrt()).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// If `out.len() != len(nmax)`.
pub fn normalised_over_sin<T: RealScalar>(nmax: usize, x: T, s: T, out: &mut [T]) {
    table(nmax, x, s, true, out);
}

fn table<T: RealScalar>(nmax: usize, x: T, s: T, over_sin: bool, out: &mut [T]) {
    assert_eq!(
        out.len(),
        len(nmax),
        "`out` must have length len(nmax) = {} for nmax = {nmax}",
        len(nmax)
    );
    let mut columns = Columns::new(x, s, over_sin);
    for _ in 0..=nmax {
        let m = columns.order();
        columns.next(nmax, |n, value| out[index(n, m)] = value);
    }
}

/// The columns of P̄ₙᵐ (or P̄ₙᵐ / s), one order m at a time from m = 0, each over
/// n = m to nmax, with the recursion of the [module documentation](self).
pub(crate) struct Columns<T> {
    /// |cos θ|.
    x: T,
    /// 1 − |cos θ|, formed as s²/(1 + |x|).
    u: T,
    s: T,
    /// cos θ < 0: P̄ₙᵐ(−x) = (−1)ⁿ⁺ᵐ P̄ₙᵐ(x).
    flip: bool,
    over_sin: bool,
    /// The order of the next column.
    m: usize,
    /// P̄ₘᵐ (or P̄ₘᵐ / s) of the last column as a mantissa and a binary exponent.
    diag: T,
    diag_exp: i32,
    span: i32,
    big: T,
    small: T,
}

impl<T: RealScalar> Columns<T> {
    /// Columns for x = cos θ and s = sin θ ≥ 0, of P̄ₙᵐ or, with `over_sin`, of P̄ₙᵐ / s.
    pub(crate) fn new(x: T, s: T, over_sin: bool) -> Self {
        // Mantissas stay within 2^±span: 2^±256 in f64, 2^±32 in f32, far inside each
        // range, so that the recursion's terms neither overflow nor underflow.
        let span = if T::max_value().to_f64() > 1e300 {
            256
        } else {
            32
        };
        let ax = x.abs();
        Self {
            x: ax,
            u: s * s / (T::one() + ax),
            s,
            flip: x < T::zero(),
            over_sin,
            m: 0,
            diag: T::one(),
            diag_exp: 0,
            span,
            big: T::from_f64(2f64.powi(span)),
            small: T::from_f64(2f64.powi(-span)),
        }
    }

    /// The order of the next column.
    pub(crate) fn order(&self) -> usize {
        self.m
    }

    /// Calls `emit(n, value)` for n = m to `nmax` with the values of the next column m,
    /// then moves to order m + 1. Does nothing once m > `nmax`. With `over_sin`, the
    /// column m = 0 is zero.
    pub(crate) fn next(&mut self, nmax: usize, mut emit: impl FnMut(usize, T)) {
        let m = self.m;
        if m > nmax {
            return;
        }
        self.m += 1;
        if m > 0 {
            // P̄ₘᵐ = √((2m − 1)/(2m)) s P̄ₘ₋₁ᵐ⁻¹; divided by s, the first step omits s.
            let factor = int::<T>(2 * m - 1) / int::<T>(2 * m);
            self.diag = self.diag * factor.sqrt();
            if !(self.over_sin && m == 1) {
                self.diag = self.diag * self.s;
            }
            if self.diag != T::zero() && self.diag.abs() < self.small {
                self.diag = self.diag * self.big;
                self.diag_exp -= self.span;
            }
        } else if self.over_sin {
            for n in 0..=nmax {
                emit(n, T::zero());
            }
            return;
        }
        let (x, u) = (self.x, self.u);
        let sign = |n: usize| {
            if self.flip && (n + m) % 2 == 1 {
                -T::one()
            } else {
                T::one()
            }
        };
        let (diag, diag_exp) = (self.diag, self.diag_exp);
        emit(m, scaled(diag, diag_exp));
        if m == nmax {
            return;
        }
        // P̄ₘ₊₁ᵐ = √(2m + 1) x P̄ₘᵐ, and E_{m+1} = c_{m+1} (P̄ₘ₊₁ᵐ − P̄ₘᵐ) with
        // c_{m+1} = √(2m + 1).
        let root = int::<T>(2 * m + 1).sqrt();
        let (mut cur, mut exp) = (root * x * diag, diag_exp);
        let mut diff = root * ((root - T::one()) - root * u) * diag;
        emit(m + 1, sign(m + 1) * scaled(cur, exp));
        let mm = int::<T>(m * m);
        let mut c_prev = root;
        for n in m + 2..=nmax {
            let c = int::<T>((n - m) * (n + m)).sqrt();
            let r = mm / (int::<T>(n) + c) + mm / (int::<T>(n - 1) + c_prev);
            diff = diff + (r - int::<T>(2 * n - 1) * u) * cur;
            cur = cur + diff / c;
            c_prev = c;
            if cur.abs() > self.big {
                diff = diff * self.small;
                cur = cur * self.small;
                exp += self.span;
            }
            emit(n, sign(n) * scaled(cur, exp));
        }
    }
}

/// `value` · 2^`exponent`, in two steps so that a large mantissa with a very negative
/// exponent lands in the subnormal range rather than at zero early.
#[inline]
fn scaled<T: RealScalar>(value: T, exponent: i32) -> T {
    if exponent == 0 {
        return value;
    }
    let half = exponent / 2;
    value * T::from_f64(2f64.powi(half)) * T::from_f64(2f64.powi(exponent - half))
}

/// An integer as `T`.
#[inline(always)]
fn int<T: RealScalar>(n: usize) -> T {
    T::from_f64(n as f64)
}
