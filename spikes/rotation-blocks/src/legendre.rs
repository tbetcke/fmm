//! Normalised associated Legendre functions to large degree.
//!
//! P̄ₙᵐ(x) = √((n − m)!/(n + m)!) Pₙᵐ(x), 0 ≤ m ≤ n, with Pₙᵐ as in CONVENTIONS §3.3
//! (no Condon–Shortley phase), by the recursion in n for each m:
//!
//! P̄ₘᵐ = Πᵢ₌₁..ₘ √((2i − 1)/(2i)) · sᵐ,  P̄ₘ₊₁ᵐ = √(2m + 1) x P̄ₘᵐ,
//! P̄ₙᵐ = ((2n − 1) x P̄ₙ₋₁ᵐ − √((n + m − 1)(n − m − 1)) P̄ₙ₋₂ᵐ) / √((n − m)(n + m)),
//!
//! with x = cos β and s = sin β ≥ 0. Near β = 0, Pₙ(x) moves by about n²/2 per unit of
//! x, and in this form the recursion's own rounding near x = 1 grows like n² ε (the
//! homogeneous coefficients nearly cancel: cₙ + cₙ₋₁ ≈ 2n − 1 with cₙ = √((n − m)(n + m))
//! and √((n + m − 1)(n − m − 1)) = cₙ₋₁). At n = 1000 and β = 10⁻³ this cost
//! 2.75·10⁻¹² in P̄₁₀₀₀⁰ itself with x, and 3.8·10⁻¹³ with x = 1 − u, in the spike's
//! first versions.
//! The recursion therefore runs on |x|, with u = 1 − |x| = s²/(1 + |x|), in telescoped
//! form: with Eₙ = cₙ (P̄ₙᵐ − P̄ₙ₋₁ᵐ),
//!
//! Eₙ = Eₙ₋₁ + (rₙ − (2n − 1) u) P̄ₙ₋₁ᵐ,  P̄ₙᵐ = P̄ₙ₋₁ᵐ + Eₙ / cₙ,
//!
//! where rₙ = 2n − 1 − cₙ − cₙ₋₁ = m²/(n + cₙ) + m²/(n − 1 + cₙ₋₁) is formed without
//! cancellation. x < 0 follows from P̄ₙᵐ(−x) = (−1)ⁿ⁺ᵐ P̄ₙᵐ(x).
//!
//! The start value falls like sᵐ and leaves the f64
//! range for m of a few hundred when s is small, while P̄ₙᵐ at n ≫ m can still be of
//! order one. Each column therefore runs with a binary exponent of its own, as in the
//! X-numbers of Fukushima (2012), and is scaled back when stored.

/// Number of values of degrees 0 to `nmax`.
pub const fn len(nmax: usize) -> usize {
    (nmax + 1) * (nmax + 2) / 2
}

/// Index of P̄ₙᵐ, n(n + 1)/2 + m.
#[inline]
pub const fn index(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

/// Mantissas are kept within 2^±[`SPAN`].
const SPAN: i32 = 256;

/// `value` · 2^`exponent`, in two steps so that a large mantissa with a very negative
/// exponent still lands in the subnormal range rather than at zero early.
#[inline]
fn scaled(value: f64, exponent: i32) -> f64 {
    let half = exponent / 2;
    value * 2f64.powi(half) * 2f64.powi(exponent - half)
}

/// Writes P̄ₙᵐ(x), 0 ≤ m ≤ n ≤ `nmax`, to `out` at [`index`]`(n, m)`, for x = cos β and
/// s = sin β ≥ 0.
///
/// # Panics
///
/// If `out.len() != len(nmax)`.
pub fn normalised(nmax: usize, x: f64, s: f64, out: &mut [f64]) {
    assert_eq!(out.len(), len(nmax), "`out` must have length len(nmax)");
    let flip = x < 0.0;
    let x = x.abs();
    let u = s * s / (1.0 + x);
    // (−1)ⁿ⁺ᵐ when x < 0.
    let sign = |n: usize, m: usize| if flip && (n + m) % 2 == 1 { -1.0 } else { 1.0 };
    let big = 2f64.powi(SPAN);
    let small = 2f64.powi(-SPAN);
    // P̄ₘᵐ as a mantissa and an exponent, carried from m − 1 to m.
    let (mut diag, mut diag_exp) = (1.0f64, 0i32);
    for m in 0..=nmax {
        if m > 0 {
            diag *= ((2 * m - 1) as f64 / (2 * m) as f64).sqrt() * s;
            if diag != 0.0 && diag.abs() < small {
                diag *= big;
                diag_exp -= SPAN;
            }
        }
        out[index(m, m)] = scaled(diag, diag_exp);
        if m == nmax {
            break;
        }
        // P̄ₘ₊₁ᵐ = √(2m + 1) x P̄ₘᵐ, and E_{m+1} = c_{m+1} (P̄ₘ₊₁ᵐ − P̄ₘᵐ), c_{m+1} = √(2m + 1).
        let root = ((2 * m + 1) as f64).sqrt();
        let (mut cur, mut exp) = (root * x * diag, diag_exp);
        let mut diff = root * ((root - 1.0) - root * u) * diag;
        out[index(m + 1, m)] = sign(m + 1, m) * scaled(cur, exp);
        let mm = (m * m) as f64;
        let mut c_prev = root;
        for n in m + 2..=nmax {
            let c = (((n - m) * (n + m)) as f64).sqrt();
            let r = mm / (n as f64 + c) + mm / ((n - 1) as f64 + c_prev);
            diff += (r - (2 * n - 1) as f64 * u) * cur;
            cur += diff / c;
            c_prev = c;
            if cur.abs() > big {
                diff *= small;
                cur *= small;
                exp += SPAN;
            }
            out[index(n, m)] = sign(n, m) * scaled(cur, exp);
        }
    }
}
