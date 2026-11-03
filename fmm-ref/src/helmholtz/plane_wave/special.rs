//! Bessel functions Jₙ of real argument and Gauss–Legendre rules, the generator's own
//! (S4's `bessel::j_all` and `gauss::legendre`, kept in their order of operations so
//! that the generator gives the spike's values bit for bit).

/// J₀(x), …, J_top(x) for real x ≥ 0: Miller's backward recurrence from an order well
/// above max(top, x), normalised by J₀ + 2 Σₖ J₂ₖ = 1, rescaled against overflow.
pub(super) fn j_all(x: f64, top: usize) -> Vec<f64> {
    assert!(x >= 0.0, "x = {x} < 0");
    let mut out = vec![0.0; top + 1];
    if x == 0.0 {
        out[0] = 1.0;
        return out;
    }
    let start = top.max(x.ceil() as usize) + 20 + (40.0 * x.max(1.0)).sqrt().ceil() as usize;
    let start = start + start % 2;
    let (mut above, mut cur) = (0.0f64, 1e-300f64);
    let mut sum = 0.0;
    for n in (1..=start).rev() {
        // cur = J_n, above = J_{n+1} (unnormalised); J_{n−1} = (2n/x) J_n − J_{n+1}.
        let below = 2.0 * n as f64 / x * cur - above;
        if n <= top {
            out[n] = cur;
        }
        if n % 2 == 0 {
            sum += 2.0 * cur;
        }
        above = cur;
        cur = below;
        if cur.abs() > 1e250 {
            let s = 1e-250;
            above *= s;
            cur *= s;
            sum *= s;
            for v in out.iter_mut() {
                *v *= s;
            }
        }
    }
    out[0] = cur;
    sum += cur;
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

/// The n-point Gauss–Legendre rule on [a, b]: nodes and weights, by Newton's method on
/// Pₙ from the Chebyshev-like initial guesses. (`direct`'s rule on [−1, 1] stops its
/// iteration by a relative step, this one by an absolute one, so the two can differ in
/// the last bits.)
pub(super) fn legendre(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let (mut xs, mut ws) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
    for i in 0..n {
        let mut x = (core::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            // P_n(x) and P_n'(x) by the three-term recurrence.
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let p = if n == 0 { 1.0 } else { p1 };
            let pm1 = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (x * p - pm1) / (x * x - 1.0);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        xs.push(mid - half * x);
        ws.push(half * 2.0 / ((1.0 - x * x) * dp * dp));
    }
    (xs, ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the trapezoidal sum Jₙ(x) = (1/N) Σⱼ cos(x sin αⱼ − n αⱼ), exact to
    /// round-off for N > x + n + 40; absolute error.
    #[test]
    fn bessel_against_the_trapezoid() {
        for &x in &[0.3, 1.0, 7.5, 30.0, 120.0] {
            let top = 160;
            let j = j_all(x, top);
            let m = (x as usize) + top + 64;
            for n in [0, 1, 5, 40, 100, 160] {
                let t: f64 = (0..m)
                    .map(|i| {
                        let a = core::f64::consts::TAU * i as f64 / m as f64;
                        (x * a.sin() - n as f64 * a).cos()
                    })
                    .sum::<f64>()
                    / m as f64;
                assert!((j[n] - t).abs() <= 1e-14, "J_{n}({x}) = {} vs {t}", j[n]);
            }
        }
    }

    /// Exact on the monomials of degree below 2n on [−1, 2]; error relative to
    /// max(1, |the integral|).
    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        for n in [1, 2, 5, 16, 30] {
            let (x, w) = legendre(n, -1.0, 2.0);
            for d in 0..2 * n {
                let q: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(d as i32)).sum();
                let exact = (2f64.powi(d as i32 + 1) - (-1f64).powi(d as i32 + 1)) / (d + 1) as f64;
                assert!(
                    (q - exact).abs() <= 1e-13 * exact.abs().max(1.0),
                    "n {n} d {d}"
                );
            }
        }
    }
}
