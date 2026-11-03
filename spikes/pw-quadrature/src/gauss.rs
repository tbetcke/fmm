//! Gauss–Legendre rules.

/// The n-point Gauss–Legendre rule on [a, b]: nodes and weights, by Newton's method on
/// Pₙ from the Chebyshev-like initial guesses.
pub fn legendre(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
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

    #[test]
    fn integrates_polynomials_exactly() {
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
