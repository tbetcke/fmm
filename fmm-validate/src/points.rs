//! Seeded point distributions and charges.
//!
//! Every function draws from a [`SplitMix64`] passed by the caller, so a distribution
//! is fixed by the seed and by the order of the calls. Points are f64; callers that
//! run in another precision round them first (and pass the rounded points to the
//! oracle, so that the input rounding is not counted as an operator error).
//!
//! Phase 1 needs uniform points in a cube and in a ball, and on a sphere surface. The
//! Plummer and Gaussian-cluster distributions of design §8.2 follow in Phase 3.
//!
//! ```
//! use nd_fmm_validate::{SplitMix64, points};
//!
//! let mut rng = SplitMix64::new(3);
//! let cube = points::cube(&mut rng, 100, [0.5, 0.5, 0.5], 0.5);
//! assert!(cube.iter().flatten().all(|&c| (0.0..=1.0).contains(&c)));
//! let charges = points::charges(&mut rng, 100);
//! assert!(charges.iter().all(|q| (-1.0..=1.0).contains(q)));
//! ```

use crate::SplitMix64;

/// `n` points uniform in the cube centre + [−h, h]³ with half-width `half_width` = h.
///
/// Each coordinate is drawn independently, x, y, z in turn.
pub fn cube(rng: &mut SplitMix64, n: usize, centre: [f64; 3], half_width: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| core::array::from_fn(|i| centre[i] + half_width * rng.range(-1.0, 1.0)))
        .collect()
}

/// `n` points uniform in the closed ball of radius `radius` about `centre`, by
/// rejection sampling in the enclosing cube.
pub fn ball(rng: &mut SplitMix64, n: usize, centre: [f64; 3], radius: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let v = in_unit_ball(rng, 0.0);
            core::array::from_fn(|i| centre[i] + radius * v[i])
        })
        .collect()
}

/// `n` points uniform on the sphere of radius `radius` about `centre`: uniform
/// directions, by rejection sampling in the unit ball (points with norm below 10⁻³
/// are also rejected) and normalisation.
pub fn sphere(rng: &mut SplitMix64, n: usize, centre: [f64; 3], radius: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let v = in_unit_ball(rng, 1e-3);
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            core::array::from_fn(|i| centre[i] + radius * (v[i] / norm))
        })
        .collect()
}

/// `n` charges uniform in [−1, 1), of both signs, so potentials can cancel.
pub fn charges(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
}

/// A point uniform in the unit ball, with norm at least `min_norm`.
fn in_unit_ball(rng: &mut SplitMix64, min_norm: f64) -> [f64; 3] {
    loop {
        let v: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
        let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if norm_sq <= 1.0 && norm_sq >= min_norm * min_norm {
            return v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Distribution = fn(&mut SplitMix64, usize, [f64; 3], f64) -> Vec<[f64; 3]>;

    const DISTRIBUTIONS: [(&str, Distribution); 3] =
        [("cube", cube), ("ball", ball), ("sphere", sphere)];

    const CENTRE: [f64; 3] = [0.3, -1.2, 2.5];

    fn distance(x: [f64; 3], c: [f64; 3]) -> f64 {
        (0..3).map(|i| (x[i] - c[i]).powi(2)).sum::<f64>().sqrt()
    }

    #[test]
    fn deterministic_for_a_seed() {
        // Error measure: exact equality of the points.
        for (name, draw) in DISTRIBUTIONS {
            let a = draw(&mut SplitMix64::new(11), 200, CENTRE, 0.7);
            let b = draw(&mut SplitMix64::new(11), 200, CENTRE, 0.7);
            assert_eq!(a, b, "{name}");
            let c = draw(&mut SplitMix64::new(12), 200, CENTRE, 0.7);
            assert_ne!(a, c, "{name}");
        }
        let a = charges(&mut SplitMix64::new(5), 100);
        assert_eq!(a, charges(&mut SplitMix64::new(5), 100));
    }

    #[test]
    fn inside_their_domains() {
        // Error measure: absolute distance, with 4 ε of rounding in centre + h v.
        let h = 0.7;
        let tol = 4.0 * f64::EPSILON * (h + 3.0);
        let mut rng = SplitMix64::new(21);
        for x in cube(&mut rng, 5000, CENTRE, h) {
            for i in 0..3 {
                assert!((x[i] - CENTRE[i]).abs() <= h + tol, "{x:?}");
            }
        }
        for x in ball(&mut rng, 5000, CENTRE, h) {
            assert!(distance(x, CENTRE) <= h + tol, "{x:?}");
        }
        for x in sphere(&mut rng, 5000, CENTRE, h) {
            assert!((distance(x, CENTRE) - h).abs() <= tol, "{x:?}");
        }
        for q in charges(&mut rng, 5000) {
            assert!((-1.0..1.0).contains(&q));
        }
    }

    #[test]
    fn fill_their_domains() {
        // Loose statistical checks for n = 20 000: the mean is near the centre, the
        // cube reaches near its faces, and the ball has about 1/8 of its points within
        // half its radius (volume ratio). Error measure: absolute deviation, many
        // standard deviations wide.
        let n = 20_000;
        let h = 2.0;
        let mut rng = SplitMix64::new(33);
        for (name, draw) in DISTRIBUTIONS {
            let pts = draw(&mut rng, n, CENTRE, h);
            for i in 0..3 {
                let mean = pts.iter().map(|x| x[i]).sum::<f64>() / n as f64;
                assert!((mean - CENTRE[i]).abs() < 0.05 * h, "{name}: mean {mean}");
            }
        }
        let pts = cube(&mut rng, n, CENTRE, h);
        let reach = pts
            .iter()
            .map(|x| (x[0] - CENTRE[0]).abs())
            .fold(0.0, f64::max);
        assert!(reach > 0.99 * h);
        let pts = ball(&mut rng, n, CENTRE, h);
        let inner = pts
            .iter()
            .filter(|&&x| distance(x, CENTRE) < h / 2.0)
            .count();
        assert!((inner as f64 / n as f64 - 0.125).abs() < 0.02, "{inner}");
        let q = charges(&mut rng, n);
        let negative = q.iter().filter(|&&q| q < 0.0).count();
        assert!((negative as f64 / n as f64 - 0.5).abs() < 0.02);
    }
}
