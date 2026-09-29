//! Helpers shared by the harmonics tests: a seeded random generator, random points,
//! access to complex values in real storage and the degree-relative error measure.

use nd_fmm_math::Layout;

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

    /// Uniformly distributed direction, by rejection sampling in the unit ball.
    pub fn unit_vector(&mut self) -> [f64; 3] {
        loop {
            let v = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let r = norm(v);
            if (1e-3..=1.0).contains(&r) {
                return v.map(|c| c / r);
            }
        }
    }

    /// Random point with radius uniform in [rmin, rmax] and a uniform direction.
    pub fn point_in_shell(&mut self, rmin: f64, rmax: f64) -> [f64; 3] {
        let r = self.range(rmin, rmax);
        self.unit_vector().map(|c| r * c)
    }
}

pub fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Complex value of Xₙᵐ for any integer m, read from real storage (CONVENTIONS §3.6),
/// with Xₙ⁻ᵐ = (−1)ᵐ conj(Xₙᵐ) (§3.3) and Xₙᵐ = 0 for |m| > n.
pub fn complex_at(values: &[f64], n: usize, m: isize) -> (f64, f64) {
    let k = m.unsigned_abs();
    if k > n {
        return (0.0, 0.0);
    }
    let layout = Layout::new(n);
    if k == 0 {
        return (values[layout.idx(n, 0)], 0.0);
    }
    let re = values[layout.idx(n, k as isize)];
    let im = values[layout.idx(n, -(k as isize))];
    if m > 0 {
        (re, im)
    } else if k.is_multiple_of(2) {
        (re, -im)
    } else {
        (-re, im)
    }
}

/// Largest, over degrees n ≤ p, of max_m |computed − reference| / max_m |reference|.
///
/// Single components can be arbitrarily close to zero, so errors are measured relative
/// to the largest reference value of the same degree, as in tools/fixtures/README.md.
/// A degree whose reference values are all zero must be matched exactly.
pub fn degree_relative_error(p: usize, computed: &[f64], reference: &[f64]) -> f64 {
    let layout = Layout::new(p);
    assert_eq!(computed.len(), layout.len());
    assert_eq!(reference.len(), layout.len());
    let mut worst = 0.0_f64;
    for (_, range) in layout.degrees() {
        let scale = reference[range.clone()]
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let err = computed[range.clone()]
            .iter()
            .zip(&reference[range])
            .fold(0.0_f64, |acc, (c, r)| acc.max((c - r).abs()));
        let rel = if scale > 0.0 {
            err / scale
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        worst = worst.max(rel);
    }
    worst
}
