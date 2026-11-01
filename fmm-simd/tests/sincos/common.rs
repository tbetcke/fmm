//! Shared helpers: the available ISAs, runs of `sincos_slice`, the error oracles, the
//! domain and a seeded generator.

use nd_fmm_simd::{Isa, SimdScalar, sincos::sincos_slice};

/// The contract of design §4.11, in units of u_T, absolute.
pub const CONTRACT: f64 = 4.0;
/// The end of the domain, θ_max = 2¹⁰.
pub const THETA_MAX: f64 = 1024.0;
/// The number of values per call of `sincos_slice` in the large checks.
pub const BATCH: usize = 1 << 16;

/// The ISAs this machine cannot run, for the line every test prints: their tests skip,
/// they do not fail.
pub fn unavailable() -> String {
    let names: Vec<String> = Isa::all()
        .filter(|isa| !isa.is_available())
        .map(|isa| isa.to_string())
        .collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// The ISAs this machine offers, after printing them under the test's name.
pub fn isas<T: SimdScalar>(test: &str) -> Vec<Isa> {
    let isas: Vec<Isa> = Isa::available().collect();
    let names: Vec<String> = isas.iter().map(|isa| isa.to_string()).collect();
    println!(
        "{test} ({}): ISAs run: {}; not available here: {}",
        std::any::type_name::<T>(),
        names.join(", "),
        unavailable()
    );
    isas
}

/// `sincos_slice(isa, theta)` into two new vectors, which start as NaN so that an
/// unwritten value shows.
pub fn sincos<T: SimdScalar>(isa: Isa, theta: &[T]) -> (Vec<T>, Vec<T>) {
    let mut sin = vec![T::nan(); theta.len()];
    let mut cos = vec![T::nan(); theta.len()];
    sincos_slice(isa, theta, &mut sin, &mut cos);
    (sin, cos)
}

/// The bits of `x`, widened to f64 (exact and injective).
pub fn bits<T: SimdScalar>(x: T) -> u64 {
    x.to_f64().to_bits()
}

/// The precisions, with the unit roundoff and the error oracle of each.
pub trait Precision: SimdScalar {
    /// The unit roundoff u_T: 2⁻²⁴ or 2⁻⁵³.
    const U: f64;

    /// The absolute errors |s − sin θ| and |c − cos θ|; infinite unless finite.
    fn errors(theta: Self, s: Self, c: Self) -> (f64, f64);
}

/// |y − v|, infinite unless y is finite.
fn error(y: f64, v: f64) -> f64 {
    if y.is_finite() {
        (y - v).abs()
    } else {
        f64::INFINITY
    }
}

impl Precision for f32 {
    const U: f64 = 5.960464477539063e-8;

    fn errors(theta: f32, s: f32, c: f32) -> (f64, f64) {
        let theta = f64::from(theta);
        (
            error(f64::from(s), theta.sin()),
            error(f64::from(c), theta.cos()),
        )
    }
}

impl Precision for f64 {
    const U: f64 = 1.1102230246251565e-16;

    fn errors(theta: f64, s: f64, c: f64) -> (f64, f64) {
        let (sin, cos) = sincos_dd(theta);
        // (y − hi) is exact for y within a few units of hi (Sterbenz) or both small.
        let dd_error = |y: f64, (hi, lo): (f64, f64)| {
            if y.is_finite() {
                ((y - hi) - lo).abs()
            } else {
                f64::INFINITY
            }
        };
        (dd_error(s, sin), dd_error(c, cos))
    }
}

/// a + b = s + e exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// a · b = p + e exactly (by fma).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// The sum of two double-doubles.
fn dd_add((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
    let (s, e) = two_sum(ah, bh);
    two_sum(s, e + al + bl)
}

/// The product of two double-doubles.
fn dd_mul((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
    let (p, e) = two_prod(ah, bh);
    two_sum(p, e + (ah * bl + al * bh))
}

/// The double-double quotient a / d for a small integer d, one correction step.
fn dd_div_int((ah, al): (f64, f64), d: f64) -> (f64, f64) {
    let q = ah / d;
    let (p, e) = two_prod(q, d);
    two_sum(q, ((ah - p) - e + al) / d)
}

/// π as a triple-double, hi + mid + lo.
const PI: [f64; 3] = [
    core::f64::consts::PI,
    1.2246467991473532e-16,
    -2.9947698097183397e-33,
];

/// (sin θ, cos θ) as double-doubles, for |θ| ≤ 2¹⁰: n = round(θ / π), the reduced
/// argument t = θ − nπ with n π formed exactly from the triple-double π (|n| < 2¹⁰), the
/// Taylor series of sin t and cos t in double-double to t⁴¹, and the sign (−1)ⁿ.
pub fn sincos_dd(theta: f64) -> ((f64, f64), (f64, f64)) {
    let n = (theta / PI[0]).round();
    let mut t = (theta, 0.0);
    for c in PI {
        let (p, e) = two_prod(n, c);
        t = dd_add(t, (-p, -e));
    }
    let t2 = dd_mul(t, t);
    let minus_t2 = (-t2.0, -t2.1);
    let mut sin = (0.0, 0.0);
    let mut term = t;
    let mut cos = (0.0, 0.0);
    let mut cos_term = (1.0, 0.0);
    for i in 0..21 {
        sin = dd_add(sin, term);
        cos = dd_add(cos, cos_term);
        let j = f64::from(2 * i + 1);
        term = dd_div_int(dd_div_int(dd_mul(term, minus_t2), j + 1.0), j + 2.0);
        cos_term = dd_div_int(dd_div_int(dd_mul(cos_term, minus_t2), j), j + 1.0);
    }
    if n.rem_euclid(2.0) == 1.0 {
        sin = (-sin.0, -sin.1);
        cos = (-cos.0, -cos.1);
    }
    (sin, cos)
}

/// The largest errors seen, in units of u_T, and where.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxError {
    /// The largest error of the sine, in units of u_T.
    pub sin: f64,
    /// The input that attained it.
    pub sin_at: f64,
    /// The largest error of the cosine, in units of u_T.
    pub cos: f64,
    /// The input that attained it.
    pub cos_at: f64,
}

impl MaxError {
    /// Records the errors of `sin` and `cos` on `theta`.
    pub fn record<T: Precision>(&mut self, theta: &[T], sin: &[T], cos: &[T]) {
        for ((&x, &s), &c) in theta.iter().zip(sin).zip(cos) {
            let (es, ec) = T::errors(x, s, c);
            self.note(x, es / T::U, ec / T::U);
        }
    }

    /// Records the errors of `sin` and `cos` on `theta` against `reference`, the exact
    /// (sin θ, cos θ) of each value to well below u_T (the f32 check's shared oracle).
    pub fn record_against<T: Precision>(
        &mut self,
        theta: &[T],
        sin: &[T],
        cos: &[T],
        reference: &[(f64, f64)],
    ) {
        for (((&x, &s), &c), &(rs, rc)) in theta.iter().zip(sin).zip(cos).zip(reference) {
            self.note(
                x,
                error(s.to_f64(), rs) / T::U,
                error(c.to_f64(), rc) / T::U,
            );
        }
    }

    /// Keeps the larger of each error of `self` and `other`.
    pub fn merge(&mut self, other: &MaxError) {
        self.note_at(other.sin, other.sin_at, other.cos, other.cos_at);
    }

    /// Notes the errors `es` and `ec`, in units of u_T, of the value `x`.
    fn note<T: Precision>(&mut self, x: T, es: f64, ec: f64) {
        self.note_at(es, x.to_f64(), ec, x.to_f64());
    }

    /// Notes a sine error `es` at `sin_at` and a cosine error `ec` at `cos_at`.
    fn note_at(&mut self, es: f64, sin_at: f64, ec: f64, cos_at: f64) {
        if es.is_nan() || es > self.sin {
            self.sin = es;
            self.sin_at = sin_at;
        }
        if ec.is_nan() || ec > self.cos {
            self.cos = ec;
            self.cos_at = cos_at;
        }
    }

    /// Prints the errors under `what` and asserts the contract.
    pub fn check(&self, what: &str, isa: Isa) {
        println!(
            "{what}: {isa} max error sin {:.3} u (at θ = {:e}), cos {:.3} u (at θ = {:e})",
            self.sin, self.sin_at, self.cos, self.cos_at
        );
        assert!(
            self.sin <= CONTRACT && self.cos <= CONTRACT,
            "{what}: {isa} max error sin {} u at θ = {:e}, cos {} u at θ = {:e}, > {CONTRACT} u",
            self.sin,
            self.sin_at,
            self.cos,
            self.cos_at
        );
    }
}

/// SplitMix64, seeded.
pub struct Rng(pub u64);

impl Rng {
    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 2f64.powi(-53)
    }

    /// A value of [0, θ_max]: uniform in it, or (every other draw) 2^e with e uniform in
    /// [−30, 10), for small arguments, rounded to `T`.
    pub fn domain<T: Precision>(&mut self) -> T {
        let x = if self.next_u64().is_multiple_of(2) {
            THETA_MAX * self.unit()
        } else {
            (40.0 * self.unit() - 30.0).exp2()
        };
        T::from_f64(x)
    }
}
