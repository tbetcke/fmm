//! Seeded point distributions and charges.
//!
//! Every function draws from a [`SplitMix64`] passed by the caller, so a distribution
//! is fixed by the seed and by the order of the calls. Points are f64; callers that
//! run in another precision round them first (and pass the rounded points to the
//! oracle, so that the input rounding is not counted as an operator error).
//!
//! Phase 1 needs uniform points in a cube and in a ball, and on a sphere surface. Phase
//! 3 (T11) adds the clustered distributions of design §8.2: the [`plummer`] sphere and
//! a few tight [`gaussian_clusters`]. Both are truncated at a documented radius, so the
//! domain stays bounded.
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

/// The truncation radius of [`plummer`], in units of its scale a: 10 a, which holds
/// 98.5% of the mass of the untruncated sphere.
pub const PLUMMER_TRUNCATION: f64 = 10.0;

/// The truncation radius of [`gaussian_clusters`], in units of its width σ: 4 σ, which
/// holds 99.89% of the mass of an untruncated three-dimensional Gaussian.
pub const GAUSSIAN_TRUNCATION: f64 = 4.0;

/// `n` points of a Plummer sphere of scale `scale` = a about `centre`, truncated at
/// radius [`PLUMMER_TRUNCATION`] · a.
///
/// The density is ρ(r) ∝ (1 + r²/a²)^(−5/2), with cumulative mass
/// M(r)/M = (r/a)³ / (1 + r²/a²)^(3/2). Each point draws a mass fraction
/// u = M(r_max)/M · U for U uniform in [0, 1), so that r ≤ r_max = 10 a, and inverts
/// the cumulative mass, r = a / √(u^(−2/3) − 1), which is exact for the truncated
/// distribution, without rejection. Its direction is then uniform, as in [`sphere`].
/// The radius comes first, then the direction.
///
/// The mean radius is a (2 − 3 S^(−1/2) + S^(−3/2)) / (M(r_max)/M) with S = 1 + 10²,
/// about 1.728 a; the median radius is about 1.29 a.
pub fn plummer(rng: &mut SplitMix64, n: usize, centre: [f64; 3], scale: f64) -> Vec<[f64; 3]> {
    let x = PLUMMER_TRUNCATION;
    let mass = x.powi(3) / (1.0 + x * x).powf(1.5);
    (0..n)
        .map(|_| {
            let u = mass * rng.uniform();
            let r = if u > 0.0 {
                scale / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
            } else {
                0.0
            };
            let v = in_unit_ball(rng, 1e-3);
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            core::array::from_fn(|i| centre[i] + r * (v[i] / norm))
        })
        .collect()
}

/// `n` points spread evenly over Gaussian clusters of width `width` = σ at `centres`:
/// cluster k gets ⌊n / K⌋ points, and the first n mod K clusters one more, in the order
/// of `centres`, cluster after cluster.
///
/// Each point is c_k + σ z for z a standard normal three-vector (Box–Muller, one
/// uniform pair per coordinate), redrawn while |z| > [`GAUSSIAN_TRUNCATION`], so every
/// point lies within 4 σ of its centre. The mean radius |z| of the truncated
/// distribution is about 1.5928 (1.5958 untruncated, 2 √(2/π)).
///
/// # Panics
///
/// If `centres` is empty and `n > 0`.
pub fn gaussian_clusters(
    rng: &mut SplitMix64,
    n: usize,
    centres: &[[f64; 3]],
    width: f64,
) -> Vec<[f64; 3]> {
    assert!(
        n == 0 || !centres.is_empty(),
        "points but no cluster centres"
    );
    let k = centres.len().max(1);
    let mut points = Vec::with_capacity(n);
    for (j, c) in centres.iter().enumerate() {
        let count = n / k + usize::from(j < n % k);
        points.extend((0..count).map(|_| {
            let z = loop {
                let z: [f64; 3] = core::array::from_fn(|_| normal(rng));
                if z[0] * z[0] + z[1] * z[1] + z[2] * z[2]
                    <= GAUSSIAN_TRUNCATION * GAUSSIAN_TRUNCATION
                {
                    break z;
                }
            };
            core::array::from_fn(|i| c[i] + width * z[i])
        }));
    }
    points
}

/// `n` charges uniform in [−1, 1), of both signs, so potentials can cancel.
pub fn charges(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
}

/// A standard normal number: Box–Muller from a uniform pair, the cosine branch only.
fn normal(rng: &mut SplitMix64) -> f64 {
    // 1 − U lies in (0, 1], so the logarithm is finite.
    let radius = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
    radius * (2.0 * std::f64::consts::PI * rng.uniform()).cos()
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

    /// Cluster centres for the tests.
    const CENTRES: [[f64; 3]; 3] = [[0.3, -1.2, 2.5], [-2.0, 0.5, 0.0], [1.0, 1.0, -1.5]];

    #[test]
    fn clustered_distributions_are_deterministic_for_a_seed() {
        // Error measure: exact equality of the points.
        let a = plummer(&mut SplitMix64::new(11), 300, CENTRE, 0.4);
        assert_eq!(a, plummer(&mut SplitMix64::new(11), 300, CENTRE, 0.4));
        assert_ne!(a, plummer(&mut SplitMix64::new(12), 300, CENTRE, 0.4));
        let a = gaussian_clusters(&mut SplitMix64::new(11), 301, &CENTRES, 0.05);
        assert_eq!(
            a,
            gaussian_clusters(&mut SplitMix64::new(11), 301, &CENTRES, 0.05)
        );
        assert_ne!(
            a,
            gaussian_clusters(&mut SplitMix64::new(12), 301, &CENTRES, 0.05)
        );
        assert!(gaussian_clusters(&mut SplitMix64::new(1), 0, &[], 0.1).is_empty());
    }

    /// The sample mean and standard error of `values`.
    fn mean_and_standard_error(values: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
        let n = values.clone().count() as f64;
        let mean = values.clone().sum::<f64>() / n;
        let variance = values.map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (variance / n).sqrt())
    }

    /// Composite Simpson's rule for f on [a, b] with 2 m intervals.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, m: usize) -> f64 {
        let h = (b - a) / (2 * m) as f64;
        let inner: f64 = (1..2 * m)
            .map(|i| f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
            .sum();
        (f(a) + inner + f(b)) * h / 3.0
    }

    #[test]
    fn plummer_matches_its_mean_and_radius() {
        // n = 20 000. The sample mean of each coordinate against the centre, and the mean
        // radius against its analytic value for the truncated sphere,
        // a (2 − 3 S^(−1/2) + S^(−3/2)) / M(r_max) with S = 1 + 10² (the integral of
        // r dM). Error measure: the deviation in units of the sample's standard error,
        // at most 5. Every point lies within the truncation radius (up to the rounding
        // of centre + r v, 4 ε (a r_max + |c|)).
        let (n, a) = (20_000, 0.4);
        let pts = plummer(&mut SplitMix64::new(41), n, CENTRE, a);
        assert_eq!(pts.len(), n);
        for i in 0..3 {
            let (mean, se) = mean_and_standard_error(pts.iter().map(|x| x[i]));
            assert!(
                (mean - CENTRE[i]).abs() < 5.0 * se,
                "axis {i}: {mean} ± {se}"
            );
        }
        let x = PLUMMER_TRUNCATION;
        let s = 1.0 + x * x;
        let exact = a * (2.0 - 3.0 / s.sqrt() + s.powf(-1.5)) / (x.powi(3) / s.powf(1.5));
        assert!((exact / a - 1.728_07).abs() < 1e-5, "{exact}");
        let (mean, se) = mean_and_standard_error(pts.iter().map(|&p| distance(p, CENTRE)));
        assert!((mean - exact).abs() < 5.0 * se, "{mean} ± {se} vs {exact}");
        let r_max = PLUMMER_TRUNCATION * a;
        let tol = 4.0 * f64::EPSILON * (r_max + 3.0);
        let reach = pts.iter().map(|&p| distance(p, CENTRE)).fold(0.0, f64::max);
        assert!(reach <= r_max + tol, "{reach}");
        // Not concentrated at the centre only: the outer tail is populated.
        assert!(reach > 0.5 * r_max, "{reach}");
    }

    #[test]
    fn gaussian_clusters_match_their_means_and_radii() {
        // n = 20 002 over three clusters: 6,668 points in the first, 6,667 in the
        // others. Per cluster, the sample mean of each coordinate against its centre, and
        // the mean of |x − c| / σ against the mean radius of the truncated Gaussian,
        // ∫ r³ e^(−r²/2) / ∫ r² e^(−r²/2) over [0, 4] (Simpson's rule). Error measure:
        // the deviation in units of the sample's standard error, at most 5. Every point
        // lies within 4 σ of its centre (up to the rounding of c + σ z).
        let (n, width) = (20_002, 0.05);
        let pts = gaussian_clusters(&mut SplitMix64::new(43), n, &CENTRES, width);
        assert_eq!(pts.len(), n);
        let radius_cubed = simpson(|r| r.powi(3) * (-r * r / 2.0).exp(), 0.0, 4.0, 2000);
        let radius_squared = simpson(|r| r.powi(2) * (-r * r / 2.0).exp(), 0.0, 4.0, 2000);
        let exact = radius_cubed / radius_squared;
        assert!((exact - 1.592_757).abs() < 1e-5, "{exact}");
        let tol = 4.0 * f64::EPSILON * (GAUSSIAN_TRUNCATION * width + 3.0);
        let mut start = 0;
        for (k, c) in CENTRES.iter().enumerate() {
            let count = if k == 0 { 6668 } else { 6667 };
            let cluster = &pts[start..start + count];
            start += count;
            for i in 0..3 {
                let (mean, se) = mean_and_standard_error(cluster.iter().map(|x| x[i]));
                assert!((mean - c[i]).abs() < 5.0 * se, "cluster {k}, axis {i}");
            }
            let radii = cluster.iter().map(|&p| distance(p, *c) / width);
            let (mean, se) = mean_and_standard_error(radii.clone());
            assert!(
                (mean - exact).abs() < 5.0 * se,
                "cluster {k}: {mean} ± {se}"
            );
            let reach = radii.fold(0.0, f64::max);
            assert!(
                reach * width <= GAUSSIAN_TRUNCATION * width + tol,
                "{reach}"
            );
            assert!(reach > 3.0, "cluster {k}: the tail is populated, {reach}");
        }
        assert_eq!(start, n);
    }
}
