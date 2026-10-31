//! The reference workloads of the Helmholtz FMM (docs/design/helmholtz-fmm-proposal.md
//! §9.1, decision D12), in the root cube [0, 1]³, so D = 1 and kD is k.
//!
//! Ported from the census spike (`spikes/census/src/workloads.rs`, Helmholtz CH0.8a) in
//! Helmholtz CH3.0b, point for point: the same seed, margin and constructions, so the
//! census's trees are these workloads' trees.
//!
//! | ID | Points | N |
//! | --- | --- | --- |
//! | W1 | a sphere surface of diameter D, quasi-uniform (Fibonacci), ten per wavelength | ⌈100 (kD)²/(4π)⌉ ≈ 7.96 (kD)² |
//! | W2 | W1 with the polar cap of half-angle 30° about +z at ten times the density | about 1.6 N(W1) |
//! | W3 | uniform random points in the cube | `n`; [`Workload::w3_resolved`] at ten per wavelength |
//! | W4 | a distribution of the Laplace gates ([`Distribution`]) | `n` |
//!
//! Charges are not part of a workload yet. The module is MPI-free.
//!
//! ```
//! use nd_fmm_validate::helmholtz_workloads::{Workload, w1_count};
//!
//! let points = Workload::W1.points(100.0);
//! assert_eq!(points.len(), w1_count(100.0));
//! assert_eq!(points.len(), 79_578);
//! ```

use std::f64::consts::PI;

use crate::SplitMix64;
use crate::fmm_accuracy::Distribution;

/// The margin that keeps every point strictly inside the root cube.
pub const MARGIN: f64 = 1e-9;

/// The seed of the random workloads (W3, W4).
pub const SEED: u64 = 0x5_8C0;

/// The polar cap of W2: half-angle 30° about +z.
pub const CAP_HALF_ANGLE: f64 = PI / 6.0;

/// The root cube [0, 1]³ as `[xmin, ymin, zmin, xmax, ymax, zmax]`, the domain of every
/// workload (D = 1).
pub const DOMAIN: [f64; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

/// One reference workload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Workload {
    /// W1: a sphere surface of diameter D, quasi-uniform (Fibonacci) at ten points per
    /// wavelength, N = ⌈100 (kD)²/(4π)⌉ ≈ 7.96 (kD)².
    W1,
    /// W2: W1 with the polar cap of half-angle 30° at ten times the density.
    W2,
    /// W3: uniform random points in the cube, `n` of them.
    W3 {
        /// The number of points.
        n: usize,
    },
    /// W4: a distribution of the Laplace gates (laplace-fmm-plan §8.2) at fixed `n`, with
    /// this module's seed.
    W4 {
        /// The distribution.
        distribution: Distribution,
        /// The number of points.
        n: usize,
    },
}

impl Workload {
    /// W3 at ten points per wavelength, 1000 per λ³: N = ⌈1000 (kD/2π)³⌉.
    pub fn w3_resolved(kd: f64) -> Self {
        Workload::W3 {
            n: (1000.0 * (kd / (2.0 * PI)).powi(3)).ceil() as usize,
        }
    }

    /// A short name, e.g. "W1", "W3 (N = 1000000)", "W4 plummer (N = 100000)".
    pub fn name(&self) -> String {
        match self {
            Workload::W1 => "W1".into(),
            Workload::W2 => "W2".into(),
            Workload::W3 { n } => format!("W3 (N = {n})"),
            Workload::W4 { distribution, n } => format!("W4 {} (N = {n})", distribution.name()),
        }
    }

    /// The points at wavenumber `kd` (W1 and W2 depend on it; W3 and W4 do not), each
    /// strictly inside [`DOMAIN`].
    pub fn points(&self, kd: f64) -> Vec<[f64; 3]> {
        match *self {
            Workload::W1 => {
                let n = w1_count(kd);
                fibonacci(n).into_iter().map(on_sphere).collect()
            }
            Workload::W2 => {
                let n = w1_count(kd);
                let cos_cap = CAP_HALF_ANGLE.cos();
                let outside = fibonacci(n).into_iter().filter(|p| p[2] <= cos_cap);
                let cap = fibonacci(10 * n).into_iter().filter(|p| p[2] > cos_cap);
                outside.chain(cap).map(on_sphere).collect()
            }
            Workload::W3 { n } => {
                let mut rng = SplitMix64::new(SEED);
                (0..n)
                    .map(|_| {
                        let mut p = [0.0; 3];
                        for c in &mut p {
                            *c = rng.range(MARGIN, 1.0 - MARGIN);
                        }
                        p
                    })
                    .collect()
            }
            Workload::W4 { distribution, n } => {
                let mut rng = SplitMix64::new(SEED);
                // [−1, 1]³ to the root cube.
                distribution
                    .points(&mut rng, n)
                    .into_iter()
                    .map(|p| p.map(|c| MARGIN + (1.0 - 2.0 * MARGIN) * 0.5 * (c + 1.0)))
                    .collect()
            }
        }
    }
}

/// N of W1: 100 points per λ² on a sphere of diameter 1, λ = 2π/kD.
pub fn w1_count(kd: f64) -> usize {
    (100.0 * kd * kd / (4.0 * PI)).ceil() as usize
}

/// `n` quasi-uniform points on the unit sphere (the Fibonacci lattice).
fn fibonacci(n: usize) -> Vec<[f64; 3]> {
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let phi = golden * i as f64;
            [r * phi.cos(), r * phi.sin(), z]
        })
        .collect()
}

/// A point of the unit sphere on the sphere of diameter 1 − 2·MARGIN centred in the cube.
fn on_sphere(p: [f64; 3]) -> [f64; 3] {
    p.map(|c| 0.5 + (0.5 - MARGIN) * c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside(points: &[[f64; 3]]) -> bool {
        points.iter().flatten().all(|&c| c > 0.0 && c < 1.0)
    }

    #[test]
    fn w1_has_ten_points_per_wavelength_on_its_sphere() {
        for kd in [1.0, 10.0, 100.0] {
            let points = Workload::W1.points(kd);
            assert_eq!(points.len(), w1_count(kd));
            assert!(inside(&points));
            for p in &points {
                let r = p.iter().map(|c| (c - 0.5) * (c - 0.5)).sum::<f64>().sqrt();
                assert!((r - (0.5 - MARGIN)).abs() < 1e-15, "radius {r}");
            }
        }
        // 7.96 (kD)²: 80 thousand at kD = 100, 2.0 million at 500 (§9.1).
        assert_eq!(w1_count(100.0), 79_578);
        assert_eq!(w1_count(500.0), 1_989_437);
    }

    #[test]
    fn w2_has_the_dense_cap() {
        let kd = 30.0;
        let w1 = w1_count(kd);
        let points = Workload::W2.points(kd);
        assert!(inside(&points));
        // The cap holds (1 − cos 30°)/2 of the sphere; at ten times the density the total is
        // (1 + 9 · 0.067) N(W1) ≈ 1.60 N(W1).
        let ratio = points.len() as f64 / w1 as f64;
        assert!((ratio - 1.603).abs() < 0.01, "N(W2)/N(W1) = {ratio}");
        let cos_cap = CAP_HALF_ANGLE.cos();
        let in_cap = points
            .iter()
            .filter(|p| (p[2] - 0.5) / (0.5 - MARGIN) > cos_cap)
            .count();
        let expected = 10.0 * w1 as f64 * (1.0 - cos_cap) / 2.0;
        assert!(
            (in_cap as f64 / expected - 1.0).abs() < 0.01,
            "{in_cap} cap points, expected about {expected}"
        );
    }

    #[test]
    fn w3_is_uniform_and_seeded() {
        let a = Workload::W3 { n: 10_000 }.points(500.0);
        let b = Workload::W3 { n: 10_000 }.points(1.0);
        assert_eq!(a, b, "W3 does not depend on kD and is seeded");
        assert!(inside(&a));
        let mean: [f64; 3] = core::array::from_fn(|k| a.iter().map(|p| p[k]).sum::<f64>() / 1e4);
        for m in mean {
            assert!((m - 0.5).abs() < 0.01, "mean {m}");
        }
        // 4.0 million at kD = 100 (§9.1).
        assert_eq!(Workload::w3_resolved(100.0), Workload::W3 { n: 4_031_442 });
    }

    #[test]
    fn w4_maps_each_distribution_into_the_cube() {
        for distribution in [
            Distribution::Cube,
            Distribution::Sphere,
            Distribution::Plummer,
            Distribution::Clusters,
        ] {
            let workload = Workload::W4 {
                distribution,
                n: 2_000,
            };
            let points = workload.points(1.0);
            assert_eq!(points.len(), 2_000);
            assert!(inside(&points), "{}", workload.name());
            assert_eq!(points, workload.points(1.0), "seeded");
        }
    }

    #[test]
    fn names() {
        assert_eq!(Workload::W1.name(), "W1");
        assert_eq!(Workload::W3 { n: 7 }.name(), "W3 (N = 7)");
        let w4 = Workload::W4 {
            distribution: Distribution::Plummer,
            n: 100_000,
        };
        assert_eq!(w4.name(), "W4 plummer (N = 100000)");
    }
}
