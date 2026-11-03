//! The target region and its samples, and the error norms.
//!
//! The region of a direction (CONVENTIONS §3.14): offsets u = target − source in box
//! widths with 1 ≤ z ≤ 4 and |x|, |y| ≤ 4, the differences of points of a source box
//! and a target box whose centres are 2 or 3 boxes apart along the direction and at most
//! 3 across it.

use crate::C;

/// The grid of the region: z ∈ {1, 1.25, …, 4}, x, y ∈ {−4, −3.5, …, 4}.
#[derive(Clone, Debug)]
pub struct Grid {
    /// The x (and y) values.
    pub xy: Vec<f64>,
    /// The z values.
    pub z: Vec<f64>,
}

impl Grid {
    /// The grid of the module documentation.
    pub fn region() -> Self {
        Self {
            xy: (0..17).map(|i| -4.0 + 0.5 * i as f64).collect(),
            z: (0..13).map(|i| 1.0 + 0.25 * i as f64).collect(),
        }
    }

    /// The points, z slowest, then y, then x.
    pub fn points(&self) -> impl Iterator<Item = [f64; 3]> + '_ {
        self.z.iter().flat_map(move |&z| {
            self.xy
                .iter()
                .flat_map(move |&y| self.xy.iter().map(move |&x| [x, y, z]))
        })
    }
}

/// The chain's geometry: source points in the source box [−½, ½]³ (its corners, its
/// centre and seeded interior points), and target points on the tensor grid
/// x, y ∈ {−3.5, −3, …, 3.5}, z ∈ {1.5, 2, …, 3.5}, which holds the corners, edge and
/// face midpoints and centres of the target boxes of the up direction (centred at
/// (dx, dy, dz) with dz ∈ {2, 3} and |dx|, |dy| ≤ 3).
#[derive(Clone, Debug)]
pub struct Pairs {
    /// The source points.
    pub sources: Vec<[f64; 3]>,
    /// The targets' x (and y) values.
    pub target_xy: Vec<f64>,
    /// The targets' z values.
    pub target_z: Vec<f64>,
}

impl Pairs {
    /// The geometry of the type documentation with `interior` seeded source points.
    pub fn up(interior: usize) -> Self {
        let mut sources = vec![[0.0; 3]];
        for b in 0..8 {
            let h = |bit: usize| if b & bit != 0 { 0.5 } else { -0.5 };
            sources.push([h(4), h(2), h(1)]);
        }
        let mut rng = SplitMix(0x5eed_c404);
        for _ in 0..interior {
            sources.push([rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5]);
        }
        Self {
            sources,
            target_xy: (0..15).map(|i| -3.5 + 0.5 * i as f64).collect(),
            target_z: (0..5).map(|i| 1.5 + 0.5 * i as f64).collect(),
        }
    }
}

/// A seeded generator (SplitMix64), so every sample is reproducible.
#[derive(Clone, Debug)]
pub struct SplitMix(pub u64);

impl SplitMix {
    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform value in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The error of an approximation of the kernel over a sample, in three norms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Errors {
    /// max |Q − G|, G in box widths: FMM3D's norm (its tables' stated errors, and the
    /// Laplace tables of CONVENTIONS §3.14).
    pub absolute: f64,
    /// max |Q − G| / max |G|: relative to the largest kernel value of the sample.
    pub relative_to_max: f64,
    /// max |Q − G| / |G|, pointwise.
    pub pointwise: f64,
}

/// Accumulates [`Errors`] over (approximation, exact) pairs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Accumulator {
    abs: f64,
    max_g: f64,
    point: f64,
}

impl Accumulator {
    /// Adds one sample.
    pub fn add(&mut self, approx: C, exact: C) {
        let e = (approx - exact).norm();
        self.abs = self.abs.max(e);
        self.max_g = self.max_g.max(exact.norm());
        self.point = self.point.max(e / exact.norm());
    }

    /// Merges another accumulator.
    pub fn merge(mut self, other: Self) -> Self {
        self.abs = self.abs.max(other.abs);
        self.max_g = self.max_g.max(other.max_g);
        self.point = self.point.max(other.point);
        self
    }

    /// The norms.
    pub fn errors(&self) -> Errors {
        Errors {
            absolute: self.abs,
            relative_to_max: self.abs / self.max_g,
            pointwise: self.point,
        }
    }
}
