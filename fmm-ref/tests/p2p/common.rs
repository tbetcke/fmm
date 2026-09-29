//! Helpers shared by the P2P tests: a seeded random generator, the naive double loop,
//! double-double arithmetic and the double-double reference sum, term magnitudes, the
//! relative error measure and a recorder for worst errors.

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

    /// Uniform in the cube `centre` + [−half, half]³.
    pub fn in_cube(&mut self, centre: [f64; 3], half: f64) -> [f64; 3] {
        centre.map(|c| c + self.range(-half, half))
    }

    /// Random charge in [−1, 1], bounded away from zero.
    pub fn charge(&mut self) -> f64 {
        let q = self.range(0.1, 1.0);
        if self.uniform() < 0.5 { -q } else { q }
    }

    /// `n` points in the cube `centre` + [−half, half]³ and `n` charges.
    pub fn charged(&mut self, n: usize, centre: [f64; 3], half: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        (0..n)
            .map(|_| (self.in_cube(centre, half), self.charge()))
            .unzip()
    }
}

/// Point charges and the targets they are evaluated at.
pub struct Set {
    pub sources: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub targets: Vec<[f64; 3]>,
}

/// The geometries of the random tests, as (centre, half-width) of a cube: the unit
/// cube, a small cube far from the origin, where x − y cancels in the leading digits,
/// and a large cube.
pub const CUBES: [([f64; 3], f64); 3] = [
    ([0.0, 0.0, 0.0], 1.0),
    ([1000.0, -250.0, 64.0], 1e-3),
    ([-3.0, 7.0, 2.0], 100.0),
];

/// The naive double loop, written out independently of the crate: for each target and
/// each source in input order, skip exact coincidence, else add q / r and subtract
/// q dₖ / (r² r) with r² = (d₀² + d₁²) + d₂². `potential` and `gradient` are
/// accumulated.
pub fn naive(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    potential: &mut [f64],
    gradient: &mut [[f64; 3]],
) {
    for i in 0..targets.len() {
        for j in 0..sources.len() {
            let (x, y, q) = (targets[i], sources[j], charges[j]);
            if x[0] == y[0] && x[1] == y[1] && x[2] == y[2] {
                continue;
            }
            let dx = x[0] - y[0];
            let dy = x[1] - y[1];
            let dz = x[2] - y[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            potential[i] += q / r;
            gradient[i][0] -= q * dx / (r2 * r);
            gradient[i][1] -= q * dy / (r2 * r);
            gradient[i][2] -= q * dz / (r2 * r);
        }
    }
}

/// A double-double number hi + lo with |lo| ≤ ulp(hi)/2: about 106 significant bits.
///
/// The algorithms are the standard error-free transformations (Knuth's TwoSum, TwoProd
/// with a fused multiply-add) and the accurate double-double addition, multiplication,
/// division and square root of Hida, Li and Bailey's QD library. Their relative error
/// is a small multiple of 2⁻¹⁰⁴ ≈ 5e-32.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

/// s + e = a + b exactly, with s = fl(a + b) (Knuth).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// s + e = a + b exactly, with s = fl(a + b), assuming |a| ≥ |b| (Dekker).
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// p + e = a · b exactly, with p = fl(a · b).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Dd {
    pub fn new(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// The exact difference a − b.
    pub fn diff(a: f64, b: f64) -> Self {
        let (hi, lo) = two_sum(a, -b);
        Self { hi, lo }
    }

    /// Rounds to the nearest f64 (up to one ulp).
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    pub fn add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.hi, other.hi);
        let (t, f) = two_sum(self.lo, other.lo);
        let (s, e) = quick_two_sum(s, e + t);
        let (hi, lo) = quick_two_sum(s, e + f);
        Self { hi, lo }
    }

    pub fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    pub fn mul(self, other: Self) -> Self {
        let (p, e) = two_prod(self.hi, other.hi);
        let (hi, lo) = quick_two_sum(p, e + (self.hi * other.lo + self.lo * other.hi));
        Self { hi, lo }
    }

    pub fn div(self, other: Self) -> Self {
        let q1 = self.hi / other.hi;
        let r = self.sub(other.mul(Self::new(q1)));
        let q2 = r.hi / other.hi;
        let r = r.sub(other.mul(Self::new(q2)));
        let q3 = r.hi / other.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Self { hi, lo }.add(Self::new(q3))
    }

    /// √self for self > 0, by one Newton step on 1/√hi (Karp's method).
    pub fn sqrt(self) -> Self {
        let x = 1.0 / self.hi.sqrt();
        let ax = self.hi * x;
        let (p, e) = two_prod(ax, ax);
        let correction = self.sub(Self { hi: p, lo: e }).hi * (x * 0.5);
        let (hi, lo) = two_sum(ax, correction);
        Self { hi, lo }
    }
}

/// Potentials and gradients computed with double-double arithmetic throughout, from
/// the exact input coordinates, and rounded to f64 at the end. Pairs with x == y
/// exactly are skipped, as in the crate.
///
/// This is the reference of the cancellation and gradient tests. Each term and each
/// sum carries about 106 bits, so the reference is accurate to far below 1e-15
/// relative to the sum of term magnitudes; the final rounding to f64 adds at most half
/// an ulp of the result.
pub fn dd_reference(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
) -> (Vec<f64>, Vec<[f64; 3]>) {
    targets
        .iter()
        .map(|&x| {
            let mut phi = Dd::default();
            let mut grad = [Dd::default(); 3];
            for (&y, &q) in sources.iter().zip(charges) {
                if x == y {
                    continue;
                }
                let d = [0, 1, 2].map(|k| Dd::diff(x[k], y[k]));
                let r2 = d[0].mul(d[0]).add(d[1].mul(d[1])).add(d[2].mul(d[2]));
                let r = r2.sqrt();
                let q = Dd::new(q);
                phi = phi.add(q.div(r));
                let r3 = r2.mul(r);
                for k in 0..3 {
                    grad[k] = grad[k].sub(q.mul(d[k]).div(r3));
                }
            }
            (phi.to_f64(), grad.map(Dd::to_f64))
        })
        .unzip()
}

/// Per target, the sums of term magnitudes Σⱼ |qⱼ| / rᵢⱼ (potential) and
/// Σⱼ |qⱼ| / rᵢⱼ² (gradient, the length of each gradient term), over the pairs with
/// xᵢ ≠ yⱼ. They are the scales of the relative error measure.
pub fn magnitudes(
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
) -> (Vec<f64>, Vec<f64>) {
    targets
        .iter()
        .map(|&x| {
            let (mut potential, mut gradient) = (0.0, 0.0);
            for (&y, &q) in sources.iter().zip(charges) {
                if x == y {
                    continue;
                }
                let r2: f64 = (0..3).map(|k| (x[k] - y[k]).powi(2)).sum();
                potential += q.abs() / r2.sqrt();
                gradient += q.abs() / r2;
            }
            (potential, gradient)
        })
        .unzip()
}

/// |got − reference| / scale, and 0 when the values are equal, also for scale 0 (a
/// target without contributing pairs). A difference at scale 0 and a NaN are infinite,
/// so that no failure can hide in `f64::max`.
fn relative(got: f64, reference: f64, scale: f64) -> f64 {
    let e = (got - reference).abs();
    if e == 0.0 {
        0.0
    } else if e.is_nan() {
        f64::INFINITY
    } else {
        e / scale
    }
}

/// The largest relative error of `got` against `reference`, per target, relative to
/// the target's sum of term magnitudes `scale`.
pub fn potential_error(got: &[f64], reference: &[f64], scale: &[f64]) -> f64 {
    got.iter()
        .zip(reference)
        .zip(scale)
        .map(|((&g, &r), &s)| relative(g, r, s))
        .fold(0.0, f64::max)
}

/// The largest relative error of any gradient component of `got` against `reference`,
/// per target, relative to the target's sum of gradient term magnitudes `scale`.
pub fn gradient_error(got: &[[f64; 3]], reference: &[[f64; 3]], scale: &[f64]) -> f64 {
    got.iter()
        .zip(reference)
        .zip(scale)
        .flat_map(|((g, r), &s)| (0..3).map(move |k| relative(g[k], r[k], s)))
        .fold(0.0, f64::max)
}

/// Bit patterns, for bit-for-bit comparisons that also distinguish −0 from +0.
pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Bit patterns of gradient components.
pub fn gradient_bits(values: &[[f64; 3]]) -> Vec<[u64; 3]> {
    values.iter().map(|g| g.map(f64::to_bits)).collect()
}

/// The largest error seen in one test, printed when the test ends
/// (`cargo test -- --nocapture` shows it).
pub struct Worst {
    name: &'static str,
    value: f64,
}

impl Worst {
    pub fn new(name: &'static str) -> Self {
        Self { name, value: 0.0 }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.value = self.value.max(value);
        value
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        eprintln!("worst {}: {:.3e}", self.name, self.value);
    }
}

#[test]
fn double_double_is_accurate() {
    // Error measure: absolute error of double-double identities, far below f64
    // precision; this checks the reference arithmetic itself.
    let two = Dd::new(2.0);
    let root = two.sqrt();
    assert!(root.mul(root).sub(two).to_f64().abs() < 1e-30);
    let third = Dd::new(1.0).div(Dd::new(3.0));
    assert!(third.mul(Dd::new(3.0)).sub(Dd::new(1.0)).to_f64().abs() < 1e-31);
    // 0.1 + 0.2 − 0.3 of the f64 inputs is exactly 2⁻⁵⁵; plain f64 gives 2⁻⁵⁴.
    let sum = Dd::new(0.1).add(Dd::new(0.2)).sub(Dd::new(0.3));
    assert_eq!(sum.to_f64(), 2f64.powi(-55));
    // The difference is exact, although it is not representable in f64.
    let d = Dd::diff(1.0, 2f64.powi(-60));
    assert_eq!((d.hi, d.lo), (1.0, -(2f64.powi(-60))));
}
