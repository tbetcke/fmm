//! Helpers shared by the dipole tests: a seeded random generator, the test frames,
//! vector arithmetic and the per-degree coefficient error.

use nd_fmm_math::Layout;
use nd_fmm_ref::Frame;

/// √3: scaled radius of the sphere around a box (CONVENTIONS §3.9).
pub const SQRT3: f64 = 1.732_050_807_568_877_2;

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

    /// Random vector with length uniform in [rmin, rmax] and a uniform direction.
    pub fn in_shell(&mut self, rmin: f64, rmax: f64) -> [f64; 3] {
        let v = loop {
            let v = [0; 3].map(|_| self.range(-1.0, 1.0));
            let r = norm(v);
            if (1e-3..=1.0).contains(&r) {
                break v.map(|c| c / r);
            }
        };
        let r = self.range(rmin, rmax);
        v.map(|c| r * c)
    }

    /// A moment with components uniform in [−1, 1).
    pub fn moment(&mut self) -> [f64; 3] {
        [0; 3].map(|_| self.range(-1.0, 1.0))
    }
}

/// The test frames: radii {1, 0.37, 2⁻¹⁶}, at centres away from the origin.
pub fn frames() -> Vec<Frame<f64>> {
    vec![
        Frame::new([0.3, -1.1, 0.7], 1.0),
        Frame::new([-2.5, 1.75, 3.25], 0.37),
        Frame::new([0.3, -1.1, 0.7], 2f64.powi(-16)),
    ]
}

/// c + r u.
pub fn place(frame: &Frame<f64>, u: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|k| frame.centre[k] + frame.radius * u[k])
}

pub fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// (p + 1)².
pub fn len(p: usize) -> usize {
    Layout::new(p).len()
}

/// The largest error per degree, relative to the largest magnitude of `want` in that
/// degree (the measure of tools/fixtures: single slots can be arbitrarily close to
/// zero), over every degree n ≤ p; a degree whose `want` vanishes (degree 0 of a dipole
/// multipole) is measured absolutely.
pub fn degree_error(p: usize, got: &[f64], want: &[f64]) -> f64 {
    let layout = Layout::new(p);
    let mut worst: f64 = 0.0;
    for n in 0..=p {
        let slots = (-(n as isize)..=n as isize).map(|m| layout.idx(n, m));
        let scale = slots.clone().map(|i| want[i].abs()).fold(0.0, f64::max);
        let scale = if scale > 0.0 { scale } else { 1.0 };
        for i in slots {
            worst = worst.max((got[i] - want[i]).abs() / scale);
        }
    }
    worst
}
