//! Helpers shared by the leaf-operator tests: a seeded random generator, test frames,
//! vector arithmetic, the truncated Legendre series and its gradient, the exact kernel,
//! term magnitudes of an expansion and the per-degree weighted coefficient error.

use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::Frame;

/// Largest degree tested in f64 (CONVENTIONS §3.9).
pub const P_MAX: usize = 30;

/// Largest degree tested in f32 (CONVENTIONS §3.9).
pub const P_MAX_F32: usize = 8;

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

    /// Uniformly distributed direction, by rejection sampling in the unit ball.
    pub fn unit_vector(&mut self) -> [f64; 3] {
        loop {
            let v = [0; 3].map(|_| self.range(-1.0, 1.0));
            let r = norm(v);
            if (1e-3..=1.0).contains(&r) {
                return v.map(|c| c / r);
            }
        }
    }

    /// Random vector with length uniform in [rmin, rmax] and a uniform direction.
    pub fn in_shell(&mut self, rmin: f64, rmax: f64) -> [f64; 3] {
        let r = self.range(rmin, rmax);
        self.unit_vector().map(|c| r * c)
    }

    /// Uniform in the cube [−1, 1]³.
    pub fn in_cube(&mut self) -> [f64; 3] {
        [0; 3].map(|_| self.range(-1.0, 1.0))
    }

    /// Random charge in [−1, 1], bounded away from zero.
    pub fn charge(&mut self) -> f64 {
        let q = self.range(0.1, 1.0);
        if self.uniform() < 0.5 { -q } else { q }
    }
}

/// The test frames: radii {1, 0.37, 2⁻¹⁶}, each at two centres away from the origin.
pub fn frames() -> Vec<Frame<f64>> {
    let centres = [[0.3, -1.1, 0.7], [-2.5, 1.75, 3.25]];
    let radii = [1.0, 0.37, 2f64.powi(-16)];
    radii
        .iter()
        .flat_map(|&r| centres.iter().map(move |&c| Frame::new(c, r)))
        .collect()
}

/// The point c + r u of `frame` at scaled coordinates `u`, rounded to f64. Tests use
/// the rounded point as the true one; its scaled coordinates are `frame.scaled(..)`.
pub fn place(frame: &Frame<f64>, u: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| frame.centre[i] + frame.radius * u[i])
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// `error / scale`, where a zero scale (for example the gradient of a degree-0 local
/// expansion) demands a zero error: then the result is 0 or infinite.
pub fn relative(error: f64, scale: f64) -> f64 {
    if scale > 0.0 {
        error / scale
    } else if error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

pub fn len(p: usize) -> usize {
    Layout::new(p).len()
}

/// Legendre polynomials Pₙ(t) and derivatives Pₙ′(t), n ≤ p, from the recursions
/// (n + 1) Pₙ₊₁ = (2n + 1) t Pₙ − n Pₙ₋₁ and Pₙ₊₁′ = Pₙ₋₁′ + (2n + 1) Pₙ.
pub fn legendre(p: usize, t: f64) -> (Vec<f64>, Vec<f64>) {
    let mut value = vec![1.0, t];
    let mut derivative = vec![0.0, 1.0];
    for n in 1..p {
        let nf = n as f64;
        value.push(((2.0 * nf + 1.0) * t * value[n] - nf * value[n - 1]) / (nf + 1.0));
        derivative.push(derivative[n - 1] + (2.0 * nf + 1.0) * value[n]);
    }
    value.truncate(p + 1);
    derivative.truncate(p + 1);
    (value, derivative)
}

/// Which of the two points of a Legendre series is the target.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// The target is the farther point (multipole expansion, M2P).
    Far,
    /// The target is the nearer point (local expansion, L2P).
    Near,
}

/// A value and gradient together with the sums of the magnitudes of their terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reference {
    pub value: f64,
    pub value_scale: f64,
    pub gradient: [f64; 3],
    pub gradient_scale: f64,
}

impl Reference {
    /// Relative potential error of `value` against this reference, relative to the sum
    /// of term magnitudes.
    pub fn value_error(&self, value: f64) -> f64 {
        relative((value - self.value).abs(), self.value_scale)
    }

    /// Euclidean norm of the gradient error of `gradient`, relative to the sum of the
    /// Euclidean norms of the terms.
    pub fn gradient_error(&self, gradient: [f64; 3]) -> f64 {
        relative(norm(sub(gradient, self.gradient)), self.gradient_scale)
    }
}

/// The truncated Legendre series Σⱼ Σₙ≤ₚ qⱼ aⁿ / dⁿ⁺¹ Pₙ(cos γⱼ) at `target`, with its
/// gradient in the target, and the sums of term magnitudes.
///
/// For each source, a < d are the distances of the nearer and farther of source and
/// target from `centre`, and cos γ is the cosine of the angle between them at the
/// centre. The difference vectors are computed as `Frame::scaled` computes them,
/// before its division. With x̂ the direction of the target, ŷ that of the source and
/// t = cos γ, ∇t = (ŷ − t x̂) / |x − c|, and the gradient terms are
///
/// - `Target::Far` (|x − c| = d): q aⁿ/dⁿ⁺² [−(n + 1) Pₙ x̂ + Pₙ′ (ŷ − t x̂)];
/// - `Target::Near` (|x − c| = a): q aⁿ⁻¹/dⁿ⁺¹ [n Pₙ x̂ + Pₙ′ (ŷ − t x̂)].
///
/// The gradient scale is the sum over terms of their Euclidean norms.
pub fn legendre_series(
    p: usize,
    centre: [f64; 3],
    sources: &[[f64; 3]],
    charges: &[f64],
    target: [f64; 3],
    which: Target,
) -> Reference {
    let mut out = Reference::default();
    let xs = sub(target, centre);
    let dx = norm(xs);
    let xhat = xs.map(|c| c / dx);
    for (&y, &q) in sources.iter().zip(charges) {
        let ys = sub(y, centre);
        let dy = norm(ys);
        let yhat = ys.map(|c| c / dy);
        let t = dot(xhat, yhat);
        let (legendre, derivative) = legendre(p, t);
        let tangent: [f64; 3] = core::array::from_fn(|i| yhat[i] - t * xhat[i]);
        let (a, d) = match which {
            Target::Far => (dy, dx),
            Target::Near => (dx, dy),
        };
        assert!(a < d, "the series needs a < d, got a = {a}, d = {d}");
        for n in 0..=p {
            let nf = n as f64;
            let power = (a / d).powi(n as i32) / d;
            let value = q * power * legendre[n];
            let gradient: [f64; 3] = match which {
                Target::Far => core::array::from_fn(|i| {
                    q * power / d
                        * (-(nf + 1.0) * legendre[n] * xhat[i] + derivative[n] * tangent[i])
                }),
                Target::Near if n == 0 => [0.0; 3],
                Target::Near => core::array::from_fn(|i| {
                    q * power / a * (nf * legendre[n] * xhat[i] + derivative[n] * tangent[i])
                }),
            };
            out.value += value;
            out.value_scale += value.abs();
            for (total, term) in out.gradient.iter_mut().zip(gradient) {
                *total += term;
            }
            out.gradient_scale += norm(gradient);
        }
    }
    out
}

/// The exact Σⱼ qⱼ / |x − yⱼ| and its gradient −Σⱼ qⱼ (x − yⱼ) / |x − yⱼ|³ in x, with
/// the scales Σⱼ |qⱼ| / |x − yⱼ| and Σⱼ |qⱼ| / |x − yⱼ|².
pub fn exact(sources: &[[f64; 3]], charges: &[f64], target: [f64; 3]) -> Reference {
    let mut out = Reference::default();
    for (&y, &q) in sources.iter().zip(charges) {
        let diff = sub(target, y);
        let r = norm(diff);
        out.value += q / r;
        out.value_scale += q.abs() / r;
        for (total, component) in out.gradient.iter_mut().zip(diff) {
            *total -= q * component / (r * r * r);
        }
        out.gradient_scale += q.abs() / (r * r);
    }
    out
}

/// The solid harmonics an expansion is evaluated in.
#[derive(Clone, Copy, Debug)]
pub enum Basis {
    /// Rₙᵐ: local expansions.
    Regular,
    /// Iₙᵐ: multipole expansions.
    Irregular,
}

/// Sums of term magnitudes of the evaluation (1/r) Σₙ Σₘ Cₙᵐ Xₙᵐ(v) at `target`
/// and of its gradient, from real storage with the doubling rule of CONVENTIONS §3.6:
/// Σᵢ wᵢ |Cᵢ Xᵢ| / r and Σᵢ wᵢ |Cᵢ| |∇Xᵢ| / r², with wᵢ = 1 for m = 0 and 2
/// otherwise, and |∇Xᵢ| the Euclidean norm of the gradient of slot i.
pub fn term_magnitudes(
    basis: Basis,
    p: usize,
    frame: &Frame<f64>,
    coefficients: &[f64],
    target: [f64; 3],
) -> (f64, f64) {
    let layout = Layout::new(p);
    let v = frame.scaled(target);
    let mut values = vec![0.0; layout.len()];
    let [mut gx, mut gy, mut gz] = [0; 3].map(|_| vec![0.0; layout.len()]);
    let grad = [&mut gx[..], &mut gy[..], &mut gz[..]];
    match basis {
        Basis::Regular => harmonics::regular_grad(p, v, &mut values, grad),
        Basis::Irregular => harmonics::irregular_grad(p, v, &mut values, grad),
    }
    let (mut value, mut gradient) = (0.0, 0.0);
    for i in 0..layout.len() {
        let w = if layout.nm(i).1 == 0 { 1.0 } else { 2.0 };
        value += w * (coefficients[i] * values[i]).abs();
        gradient += w * coefficients[i].abs() * norm([gx[i], gy[i], gz[i]]);
    }
    let r = frame.radius;
    (value / r, gradient / (r * r))
}

/// Which weight of the orthonormal basis of CONVENTIONS §3.8 a coefficient vector
/// takes (docs/phase1/README.md, "Error measures").
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// Conjugates of regular harmonics (multipoles): weight Nₘ.
    Multipole,
    /// Conjugates of irregular harmonics (locals): weight Nₘ/Sₘ.
    Local,
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// Weight of slot m of degree n: Nₘ = √((n + |m|)! (n − |m|)!) cₘ, or Nₘ/Sₘ =
/// cₘ / √((n + |m|)! (n − |m|)!), with c₀ = 1 and cₘ = √2 otherwise (§3.8).
pub fn weight(kind: Kind, n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    let c = if k == 0 { 1.0 } else { 2f64.sqrt() };
    let f = (factorial(n + k) * factorial(n - k)).sqrt();
    match kind {
        Kind::Multipole => f * c,
        Kind::Local => c / f,
    }
}

/// Per-degree weighted norms ‖W xₙ‖₂, n ≤ p.
pub fn degree_norms(kind: Kind, p: usize, x: &[f64]) -> Vec<f64> {
    Layout::new(p)
        .degrees()
        .map(|(n, range)| {
            range
                .map(|i| {
                    let m = Layout::new(p).nm(i).1;
                    (weight(kind, n, m) * x[i]).powi(2)
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// Worst per-degree error of `got` against `want`: the largest over n of
/// ‖W (got − want)ₙ‖₂ / Σₛ ‖W sₙ‖₂, weighted as in [`weight`], with the vectors s in
/// `scales` giving the magnitude of degree n.
pub fn degree_error(kind: Kind, p: usize, got: &[f64], want: &[f64], scales: &[&[f64]]) -> f64 {
    let diff: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    let errors = degree_norms(kind, p, &diff);
    let mut denominators = vec![0.0; p + 1];
    for s in scales {
        for (d, v) in denominators.iter_mut().zip(degree_norms(kind, p, s)) {
            *d += v;
        }
    }
    errors
        .iter()
        .zip(&denominators)
        .map(|(e, d)| {
            assert!(
                *d > 0.0 || *e == 0.0,
                "degree with zero scale but error {e}"
            );
            if *d > 0.0 { e / d } else { 0.0 }
        })
        .fold(0.0, f64::max)
}

/// Tracks and prints the worst value of an error measure.
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

    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        eprintln!("worst {}: {:.3e}", self.name, self.value);
    }
}
