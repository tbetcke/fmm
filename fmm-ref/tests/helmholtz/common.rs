//! Helpers shared by the Helmholtz P2P tests: a seeded generator, the term magnitudes,
//! the error measures and a recorder for worst errors.

use nd_fmm_math::Complex;

pub type C = Complex<f64>;

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

    /// `n` points uniform in the cube `centre` + [−half, half]³.
    pub fn points(&mut self, n: usize, centre: [f64; 3], half: f64) -> Vec<[f64; 3]> {
        (0..n)
            .map(|_| centre.map(|c| c + self.range(-half, half)))
            .collect()
    }

    /// `n` real charges uniform in [−1, 1].
    pub fn real_charges(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.range(-1.0, 1.0)).collect()
    }

    /// `n` complex charges, each part uniform in [−1, 1] (proposal §9.1).
    pub fn complex_charges(&mut self, n: usize) -> Vec<C> {
        (0..n)
            .map(|_| C::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0)))
            .collect()
    }
}

/// The real charges as complex numbers with zero imaginary parts.
pub fn complexify(q: &[f64]) -> Vec<C> {
    q.iter().map(|&q| C::new(q, 0.0)).collect()
}

/// The term magnitudes of a target x: (Σⱼ |qⱼ| |e^{ikr}|/r, Σⱼ |qⱼ| |ikr − 1| |e^{ikr}|/r²),
/// exact coincidences skipped, in f64.
pub fn magnitudes(k: C, sources: &[[f64; 3]], charges: &[C], x: [f64; 3]) -> (f64, f64) {
    let (mut potential, mut gradient) = (0.0, 0.0);
    for (&y, &q) in sources.iter().zip(charges) {
        if x == y {
            continue;
        }
        let r = distance(x, y);
        let decay = (-k.im * r).exp();
        potential += q.norm() * decay / r;
        gradient += q.norm() * (C::new(0.0, 1.0) * k * r - 1.0).norm() * decay / (r * r);
    }
    (potential, gradient)
}

pub fn distance(x: [f64; 3], y: [f64; 3]) -> f64 {
    ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
}

/// The Euclidean norm of the difference of two complex 3-vectors.
pub fn gradient_error(a: [C; 3], b: [C; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).norm_sqr()).sum::<f64>().sqrt()
}

/// The relative ℓ2 error ‖a − b‖₂ / ‖b‖₂ of proposal §9.1.
pub fn rel_l2(a: &[C], b: &[C]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum();
    let den: f64 = b.iter().map(|y| y.norm_sqr()).sum();
    (num / den).sqrt()
}

/// The relative ℓ2 error of gradient fields.
pub fn rel_l2_gradient(a: &[[C; 3]], b: &[[C; 3]]) -> f64 {
    let flat = |g: &[[C; 3]]| g.iter().flatten().copied().collect::<Vec<_>>();
    rel_l2(&flat(a), &flat(b))
}

/// The largest of a sequence of errors, with a label for the worst one.
#[derive(Default)]
pub struct Worst {
    pub value: f64,
    pub label: String,
}

impl Worst {
    pub fn record(&mut self, value: f64, label: impl FnOnce() -> String) {
        assert!(!value.is_nan(), "a NaN error");
        if value > self.value {
            self.value = value;
            self.label = label();
        }
    }

    pub fn check(&self, tol: f64, what: &str) {
        println!("{what}: worst {:.2e} at {}", self.value, self.label);
        assert!(
            self.value <= tol,
            "{what}: {:.3e} > {tol:.1e} at {}",
            self.value,
            self.label
        );
    }
}
