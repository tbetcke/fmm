//! Shared helpers: the available ISAs, runs of `exp_slice`, the error oracles, the domain
//! and a seeded generator.

use nd_fmm_simd::{Isa, SimdScalar, exp::exp_slice};

/// The contract of design §4.9, in units of u_T.
pub const CONTRACT: f64 = 4.0;
/// The number of values per call of `exp_slice` in the large checks.
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

/// `exp_slice(isa, x)` into a new vector, which starts as NaN so that an unwritten value
/// shows.
pub fn exp<T: SimdScalar>(isa: Isa, x: &[T]) -> Vec<T> {
    let mut out = vec![T::nan(); x.len()];
    exp_slice(isa, x, &mut out);
    out
}

/// The bits of `x`, widened to f64 (exact and injective).
pub fn bits<T: SimdScalar>(x: T) -> u64 {
    x.to_f64().to_bits()
}

/// The precisions, with the unit roundoff, the domain and the error oracle of each.
pub trait Precision: SimdScalar {
    /// The unit roundoff u_T: 2⁻²⁴ or 2⁻⁵³.
    const U: f64;
    /// The lower end of the domain, (e_min + 1) ln 2 rounded towards zero
    /// (`nd_fmm_simd::exp`).
    const X_MIN: Self;
    /// The bound on the result below the domain, 2^(e_min + 2).
    const BELOW: f64;

    /// The relative error |y − e^x| / e^x of `y`, for x in the domain; infinite unless y
    /// is finite and positive.
    fn exp_error(x: Self, y: Self) -> f64;
}

impl Precision for f32 {
    const U: f64 = 5.960464477539063e-8;
    const X_MIN: f32 = -86.643_394;
    const BELOW: f64 = 4.701_977_403_289_551e-38;

    fn exp_error(x: f32, y: f32) -> f64 {
        if !(y.is_finite() && y > 0.0) {
            return f64::INFINITY;
        }
        let reference = f64::from(x).exp();
        ((f64::from(y) - reference) / reference).abs()
    }
}

impl Precision for f64 {
    const U: f64 = 1.1102230246251565e-16;
    const X_MIN: f64 = -707.703_271_351_704_1;
    const BELOW: f64 = 8.900_295_434_028_806e-308;

    fn exp_error(x: f64, y: f64) -> f64 {
        if !(y.is_finite() && y > 0.0) {
            return f64::INFINITY;
        }
        let (hi, lo) = exp_dd(x);
        // (y − hi) is exact for y within a few units of hi (Sterbenz).
        (((y - hi) - lo) / hi).abs()
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
    let (s, e2) = two_sum(s, e + al + bl);
    (s, e2)
}

/// The product of two double-doubles.
fn dd_mul((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
    let (p, e) = two_prod(ah, bh);
    two_sum(p, e + (ah * bl + al * bh))
}

/// ln 2 as a triple-double, hi + mid + lo.
const LN2: [f64; 3] = [
    core::f64::consts::LN_2,
    2.3190468138462996e-17,
    5.707708438416212e-34,
];

/// e^x as a double-double hi + lo, for x in the f64 domain: n = round(x / ln 2), the
/// reduced argument r = x − n ln 2 with n ln 2 formed exactly from the triple-double ln 2
/// (|n| < 2¹¹), e^r by its Taylor series to degree 30 in double-double, and 2ⁿ applied
/// exactly (the result is normal).
pub fn exp_dd(x: f64) -> (f64, f64) {
    let n = (x / LN2[0]).round();
    // x − n ln 2: each n · ln2ᵢ as an exact product, subtracted in double-double.
    let mut r = (x, 0.0);
    for c in LN2 {
        let (p, e) = two_prod(n, c);
        r = dd_add(r, (-p, -e));
    }
    let mut sum = (1.0, 0.0);
    let mut term = (1.0, 0.0);
    for i in 1..=30 {
        term = dd_mul(term, r);
        // term / i: the double-double quotient, one correction step.
        let q = term.0 / f64::from(i);
        let (p, e) = two_prod(q, f64::from(i));
        let correction = ((term.0 - p) - e + term.1) / f64::from(i);
        term = two_sum(q, correction);
        sum = dd_add(sum, term);
    }
    let scale = 2f64.powi(n as i32);
    (sum.0 * scale, sum.1 * scale)
}

/// The largest error seen, in units of u_T, and where.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxError {
    /// The largest error, in units of u_T.
    pub ulps: f64,
    /// The input that attained it.
    pub at: f64,
}

impl MaxError {
    /// Records the errors of `y` on `x`.
    pub fn record<T: Precision>(&mut self, x: &[T], y: &[T]) {
        for (&xi, &yi) in x.iter().zip(y) {
            let ulps = T::exp_error(xi, yi) / T::U;
            if ulps.is_nan() || ulps > self.ulps {
                self.ulps = ulps;
                self.at = xi.to_f64();
            }
        }
    }

    /// Prints the error under `what` and asserts the contract.
    pub fn check(&self, what: &str, isa: Isa) {
        println!(
            "{what}: {isa} max error {:.3} u (at x = {:e})",
            self.ulps, self.at
        );
        assert!(
            self.ulps <= CONTRACT,
            "{what}: {isa} max error {} u > {CONTRACT} u at x = {:e}",
            self.ulps,
            self.at
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

    /// A value of the domain [x_min, 0]: uniform in it, or (every other draw) −2^t with t
    /// uniform in [−200, log₂ |x_min|), for small arguments, rounded to `T`.
    pub fn domain<T: Precision>(&mut self) -> T {
        let x_min = T::X_MIN.to_f64();
        let x = if self.next_u64().is_multiple_of(2) {
            x_min * self.unit()
        } else {
            -((x_min.abs().log2() + 200.0) * self.unit() - 200.0).exp2()
        };
        T::from_f64(x).max(T::X_MIN)
    }
}
