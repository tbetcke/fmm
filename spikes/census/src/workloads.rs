//! The reference workloads of proposal §9.1, in the root cube [0, 1]³ (D = 1).

use std::f64::consts::PI;

use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::fmm_accuracy::Distribution;

/// The margin that keeps every point strictly inside the root cube.
const MARGIN: f64 = 1e-9;

/// The seed of the random workloads (W3, W4).
const SEED: u64 = 0x5_8C0;

/// The polar cap of W2: half-angle 30° about +z.
pub const CAP_HALF_ANGLE: f64 = PI / 6.0;

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
    /// this spike's seed.
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

    /// The points at wavenumber `kd` (W1 and W2 depend on it; W3 and W4 do not).
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
