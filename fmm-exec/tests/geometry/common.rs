//! Helpers shared by the geometry tests: a seeded random generator, the test domains,
//! random keys, the padding formula of `compute_global_bounding_box` without MPI, the
//! double-double reference for leaf-scaled coordinates, the error bounds of
//! CONVENTIONS §3.13 and a recorder for worst errors.

use nd_fmm_exec::geometry::{Domain, centre, radius};
use nd_octree::{MortonKey, PhysicalBox, morton};

/// Unit roundoff of f64, ε₆₄ = 2⁻⁵³.
pub const EPS64: f64 = f64::EPSILON / 2.0;

/// Unit roundoff of f32, ε₃₂ = 2⁻²⁴.
pub const EPS32: f64 = f32::EPSILON as f64 / 2.0;

/// The deepest level of a Morton key.
pub const DEEPEST: usize = 16;

/// SplitMix64: a small, deterministic generator, so the tests need no extra dependency.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
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

    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A uniformly random key on `level`.
    pub fn key(&mut self, level: usize) -> MortonKey {
        let n = 1 << level;
        morton::from_index_and_level([self.below(n), self.below(n), self.below(n)], level)
    }

    /// A point uniformly distributed in the physical box of `key`.
    pub fn in_box(&mut self, key: MortonKey, domain: &Domain) -> [f64; 3] {
        let [x0, y0, z0, x1, y1, z1] =
            morton::physical_box(key, &domain.physical_box()).coordinates();
        [self.range(x0, x1), self.range(y0, y1), self.range(z0, z1)]
    }
}

/// The corner keys and `count` random keys of `level`.
pub fn sample_keys(rng: &mut SplitMix64, level: usize, count: usize) -> Vec<MortonKey> {
    let last = (1 << level) - 1;
    let mut keys = vec![
        morton::from_index_and_level([0, 0, 0], level),
        morton::from_index_and_level([last, last, last], level),
        morton::from_index_and_level([last, 0, last], level),
    ];
    keys.extend((0..count).map(|_| rng.key(level)));
    keys
}

/// The dyadic domain of Phase 2, a = (−1.25, 0.5, 2) and w = 3: every centre, centre
/// difference and half-width on levels 0–16 is exact in f64.
pub fn dyadic_domain() -> Domain {
    Domain::new(&PhysicalBox::new([-1.25, 0.5, 2.0, 1.75, 3.5, 5.0])).unwrap()
}

/// The generic domain, a = (0.1, −2.3, 7.9) and w = 0.37, built by hand as
/// [a, fl(a + w)], so its three sides differ by rounding.
pub fn generic_domain() -> Domain {
    let (a, w) = ([0.1, -2.3, 7.9], 0.37);
    Domain::new(&PhysicalBox::new([
        a[0],
        a[1],
        a[2],
        a[0] + w,
        a[1] + w,
        a[2] + w,
    ]))
    .unwrap()
}

/// The padded cubic box of `nd_octree::octree::compute_global_bounding_box` for the
/// points on one rank, operation by operation, without MPI.
pub fn padded_box(points: &[[f64; 3]]) -> PhysicalBox {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for x in points {
        for k in 0..3 {
            lo[k] = f64::min(lo[k], x[k]);
            hi[k] = f64::max(hi[k], x[k]);
        }
    }
    let diam: [f64; 3] = core::array::from_fn(|k| hi[k] - lo[k]);
    let mean: [f64; 3] = core::array::from_fn(|k| lo[k] + 0.5 * diam[k]);
    let deepest_box_diam = 1.0 / (1 << DEEPEST) as f64;
    let max_diam = diam.into_iter().reduce(f64::max).unwrap();
    let max_diam = max_diam * (1.0 + deepest_box_diam);
    PhysicalBox::new([
        mean[0] - 0.5 * max_diam,
        mean[1] - 0.5 * max_diam,
        mean[2] - 0.5 * max_diam,
        mean[0] + 0.5 * max_diam,
        mean[1] + 0.5 * max_diam,
        mean[2] + 0.5 * max_diam,
    ])
}

/// The sides fl(max_k − min_k) of a box.
pub fn sides(bounding_box: &PhysicalBox) -> [f64; 3] {
    let c = bounding_box.coordinates();
    [c[3] - c[0], c[4] - c[1], c[5] - c[2]]
}

/// The midpoint and half-side of `morton::physical_box(key)`, per axis.
pub fn physical_midpoint(key: MortonKey, domain: &Domain) -> ([f64; 3], [f64; 3]) {
    let c = morton::physical_box(key, &domain.physical_box()).coordinates();
    (
        core::array::from_fn(|k| (c[k] + c[k + 3]) / 2.0),
        core::array::from_fn(|k| (c[k + 3] - c[k]) / 2.0),
    )
}

/// The relative frame (c_s − c_t) / r_t and r_s / r_t, computed in f64 from the
/// absolute centres and radii of the dyadic domain, where every step is exact.
pub fn dyadic_frame(s: MortonKey, t: MortonKey) -> ([f64; 3], f64) {
    let domain = dyadic_domain();
    let (cs, ct) = (centre(s, &domain), centre(t, &domain));
    let (rs, rt) = (
        radius(morton::level(s), &domain),
        radius(morton::level(t), &domain),
    );
    (core::array::from_fn(|k| (cs[k] - ct[k]) / rt), rs / rt)
}

/// Error-free sum: a + b = s + e exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// The exact u = (x − a) 2^(l+1) / w − (2i + 1) of one component, in double-double
/// (hi + lo), for the doubles x, a and w. x − a and (2i + 1) w are formed exactly, the
/// numerator rounds at about 2⁻¹⁰⁶ of its terms and the division is compensated, so
/// the reference is good to far below ε₆₄ |u|.
pub fn exact_u(x: f64, a: f64, w: f64, level: usize, index: usize) -> (f64, f64) {
    let scale = (1u64 << (level + 1)) as f64;
    let (s, e) = two_sum(x, -a);
    let (s, e) = (s * scale, e * scale);
    let m = (2 * index + 1) as f64;
    let p = m * w;
    let pe = m.mul_add(w, -p);
    let (h, l) = two_sum(s, -p);
    let (h, l) = two_sum(h, l + (e - pe));
    let q = h / w;
    let r = (-q).mul_add(w, h) + l;
    (q, r / w)
}

/// |ũ − u| for a stored value ũ and the double-double u = hi + lo.
pub fn error(stored: f64, (hi, lo): (f64, f64)) -> f64 {
    ((stored - hi) - lo).abs()
}

/// The f64 error bound of §3.13 for one component, in the rigorous form of
/// `check_leaf_geometry.py`: (2ε + ε²) |x − a| / r_l for the rounding of x − a and the
/// scaling, plus ε (|u| + that) for the subtraction of 2i + 1.
pub fn bound64(x_minus_a: f64, r: f64, u: f64) -> f64 {
    let scaling = (2.0 * EPS64 + EPS64 * EPS64) * x_minus_a.abs() / r;
    scaling + EPS64 * (u.abs() + scaling)
}

/// The f32 bound: the f64 bound plus the cast, ε₃₂ (|u| + bound64).
pub fn bound32(bound64: f64, u: f64) -> f64 {
    bound64 + EPS32 * (u.abs() + bound64)
}

/// β_k of §3.13, "Containment", with a margin for the second-order terms:
/// 2^(l+1) (|w_k − w| / w + 3 ε₆₄).
pub fn containment_bound(side_k: f64, side: f64, level: usize) -> f64 {
    (1u64 << (level + 1)) as f64 * ((side_k - side).abs() / side + 3.0 * EPS64)
}

/// Records the worst value of an error measure and prints it.
pub struct Worst {
    name: &'static str,
    value: f64,
}

impl Worst {
    pub fn new(name: &'static str) -> Self {
        Self { name, value: 0.0 }
    }

    pub fn record(&mut self, value: f64) {
        assert!(!value.is_nan(), "{}: NaN", self.name);
        self.value = self.value.max(value);
    }

    /// Prints the worst value.
    pub fn report(&self) {
        println!("{}: worst {:.3e}", self.name, self.value);
    }

    /// Prints the worst value and asserts it is at most `tolerance`.
    pub fn check(&self, tolerance: f64) {
        println!(
            "{}: worst {:.3e} (tolerance {:.1e})",
            self.name, self.value, tolerance
        );
        assert!(
            self.value <= tolerance,
            "{}: worst {:e} exceeds {:e}",
            self.name,
            self.value,
            tolerance
        );
    }
}
