//! Boxes, offsets and point placements.
//!
//! Lengths are in units of the width w = 1 of the boxes at the M2L level; children have
//! width 1/2, grandchildren 1/4. A box is given by its centre and width.

use crate::complex::Complex;
use crate::real::Real;

/// The 316 V-list offsets: (i, j, k) ∈ [−3, 3]³ with max |·| ≥ 2, in a fixed order.
pub fn offsets() -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(316);
    for i in -3i32..=3 {
        for j in -3i32..=3 {
            for k in -3i32..=3 {
                if i.abs().max(j.abs()).max(k.abs()) >= 2 {
                    out.push([i, j, k]);
                }
            }
        }
    }
    out
}

/// |t|² of an offset.
pub fn dist2(o: [i32; 3]) -> i32 {
    o[0] * o[0] + o[1] * o[1] + o[2] * o[2]
}

/// The distinct |t|² of the offsets, ascending (15 values).
pub fn distances() -> Vec<i32> {
    let mut d: Vec<i32> = offsets().into_iter().map(dist2).collect();
    d.sort_unstable();
    d.dedup();
    d
}

/// The offsets with |t|² ≤ 6 (54), where the search for p runs.
pub fn near_offsets() -> Vec<[i32; 3]> {
    offsets().into_iter().filter(|&o| dist2(o) <= 6).collect()
}

/// The centres of the eight children of a box of width `w` at `c`, in the order of the
/// sign bits (x, y, z) = bit 0, 1, 2 set meaning +.
pub fn children(c: [f64; 3], w: f64) -> [[f64; 3]; 8] {
    let mut out = [[0.0; 3]; 8];
    for (i, o) in out.iter_mut().enumerate() {
        for d in 0..3 {
            let sign = if i >> d & 1 == 1 { 1.0 } else { -1.0 };
            o[d] = c[d] + sign * w / 4.0;
        }
    }
    out
}

/// The child of a box at `c` that contains x (ties to the + side).
pub fn child_index(c: [f64; 3], x: [f64; 3]) -> usize {
    (0..3).map(|d| usize::from(x[d] >= c[d]) << d).sum()
}

/// A small deterministic generator (SplitMix64), so the spike needs no dependency.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator from a seed.
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform below n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n
    }

    /// A uniformly random unit vector.
    pub fn direction(&mut self) -> [f64; 3] {
        let z = self.range(-1.0, 1.0);
        let phi = self.range(0.0, 2.0 * std::f64::consts::PI);
        let s = (1.0 - z * z).sqrt();
        [s * phi.cos(), s * phi.sin(), z]
    }
}

/// How the points fill a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Uniform in the box.
    Uniform,
    /// In the eight corner cubes (side w_leaf/8) of the leaf boxes, the farthest points
    /// from the leaf centres, where the expansions converge slowest.
    Corners,
    /// On a sphere of radius 2w or 8w through the box, at a random orientation: a
    /// surface patch of W1 at levels 2 and 4.
    Patch,
}

impl Placement {
    /// All three.
    pub const ALL: [Placement; 3] = [Placement::Uniform, Placement::Corners, Placement::Patch];

    /// A short name.
    pub fn name(self) -> &'static str {
        match self {
            Placement::Uniform => "uniform",
            Placement::Corners => "corners",
            Placement::Patch => "patch",
        }
    }
}

/// `n` points in the box of width `w` at `c` by `placement`, whose leaves have width
/// `w_leaf` (w, w/2 or w/4).
pub fn points(
    placement: Placement,
    c: [f64; 3],
    w: f64,
    w_leaf: f64,
    n: usize,
    rng: &mut Rng,
) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(n);
    match placement {
        Placement::Uniform => {
            for _ in 0..n {
                out.push(std::array::from_fn(|d| c[d] + rng.range(-w / 2.0, w / 2.0)));
            }
        }
        Placement::Corners => {
            let per_side = (w / w_leaf).round() as usize;
            let side = w_leaf / 8.0;
            for _ in 0..n {
                // A leaf, a corner of it, a point in that corner's cube.
                let leaf: [usize; 3] = std::array::from_fn(|_| rng.below(per_side));
                let corner = rng.below(8);
                out.push(std::array::from_fn(|d| {
                    let lo = c[d] - w / 2.0 + leaf[d] as f64 * w_leaf;
                    let at = if corner >> d & 1 == 1 {
                        lo + w_leaf - side
                    } else {
                        lo
                    };
                    at + rng.range(0.0, side)
                }));
            }
        }
        Placement::Patch => {
            let radius = if rng.below(2) == 0 { 2.0 * w } else { 8.0 * w };
            let normal = rng.direction();
            let q: [f64; 3] = std::array::from_fn(|d| c[d] + rng.range(-w / 4.0, w / 4.0));
            let centre: [f64; 3] = std::array::from_fn(|d| q[d] - radius * normal[d]);
            // Every point of the box is within 2a of q (a = √3 w/2); sample the cap of
            // the sphere within that chord of q, then keep the points in the box.
            let chord = 3f64.sqrt() * w;
            let half = if chord >= 2.0 * radius {
                std::f64::consts::PI
            } else {
                2.0 * (chord / (2.0 * radius)).asin()
            };
            let zmin = half.cos();
            // An orthonormal frame (e1, e2, normal).
            let helper = if normal[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let e1 = normalise(cross(normal, helper));
            let e2 = cross(normal, e1);
            while out.len() < n {
                let z = rng.range(zmin, 1.0);
                let phi = rng.range(0.0, 2.0 * std::f64::consts::PI);
                let s = (1.0 - z * z).sqrt();
                let x: [f64; 3] = std::array::from_fn(|d| {
                    centre[d]
                        + radius * (z * normal[d] + s * phi.cos() * e1[d] + s * phi.sin() * e2[d])
                });
                if (0..3).all(|d| (x[d] - c[d]).abs() <= w / 2.0) {
                    out.push(x);
                }
            }
        }
    }
    out
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(a: [f64; 3]) -> [f64; 3] {
    let r = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / r, a[1] / r, a[2] / r]
}

/// `n` random complex charges with parts uniform in [−1, 1] (proposal §9.1).
pub fn charges<T: Real>(n: usize, rng: &mut Rng) -> Vec<Complex<T>> {
    (0..n)
        .map(|_| Complex::from_f64(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)))
        .collect()
}

/// Points converted exactly to T.
pub fn to_real<T: Real>(points: &[[f64; 3]]) -> Vec<[T; 3]> {
    points
        .iter()
        .map(|x| [T::from_f64(x[0]), T::from_f64(x[1]), T::from_f64(x[2])])
        .collect()
}
