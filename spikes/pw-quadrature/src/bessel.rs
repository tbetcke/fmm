//! Bessel functions Jₙ of real argument, by Miller's backward recurrence.

/// J₀(x), …, J_top(x) for real x ≥ 0: Miller's backward recurrence from an order well
/// above max(top, x), normalised by J₀ + 2 Σₖ J₂ₖ = 1, rescaled against overflow.
pub fn j_all(x: f64, top: usize) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the trapezoidal sum Jₙ(x) = (1/N) Σⱼ cos(x sin αⱼ − n αⱼ), exact to
    /// round-off for N > x + n + 40.
    #[test]
    fn against_the_trapezoid() {
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
}
