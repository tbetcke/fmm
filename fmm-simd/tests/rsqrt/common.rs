//! Shared helpers: the available ISAs, runs of `rsqrt_slice`, the error oracles, the
//! kernel domain and a seeded generator.

use nd_fmm_simd::{Isa, SimdScalar, rsqrt::rsqrt_slice};

/// The smallest nonzero r² of the kernel domain, 2⁻¹⁰⁸ (CONVENTIONS §3.13).
pub const DOMAIN_MIN: f64 = 3.0814879110195774e-33;
/// The largest r² of the kernel domain, 2⁷ (CONVENTIONS §3.13).
pub const DOMAIN_MAX: f64 = 128.0;
/// The contract of design §4.3, in units of u_T.
pub const CONTRACT: f64 = 4.0;
/// The number of values per call of `rsqrt_slice` in the large checks.
pub const BATCH: usize = 1 << 16;

/// The ISAs this machine offers, after printing them under the test's name.
pub fn isas<T: SimdScalar>(test: &str) -> Vec<Isa> {
    let isas: Vec<Isa> = Isa::available().collect();
    let names: Vec<String> = isas.iter().map(|isa| isa.to_string()).collect();
    println!(
        "{test} ({}): ISAs run: {}",
        std::any::type_name::<T>(),
        names.join(", ")
    );
    isas
}

/// `rsqrt_slice(isa, x)` into a new vector, which starts as NaN so that an unwritten
/// value shows.
pub fn rsqrt<T: SimdScalar>(isa: Isa, x: &[T]) -> Vec<T> {
    let mut out = vec![T::nan(); x.len()];
    rsqrt_slice(isa, x, &mut out);
    out
}

/// The bits of `x`, widened to f64 (exact and injective).
pub fn bits<T: SimdScalar>(x: T) -> u64 {
    x.to_f64().to_bits()
}

/// The precisions, with the unit roundoff and the error oracle of each.
pub trait Precision: SimdScalar {
    /// The unit roundoff u_T: 2⁻²⁴ or 2⁻⁵³.
    const U: f64;

    /// The relative error |y √x − 1| of `y` as 1/√x, for x > 0; infinite unless y is
    /// finite and positive.
    fn rsqrt_error(x: Self, y: Self) -> f64;
}

impl Precision for f32 {
    const U: f64 = 5.960464477539063e-8;

    fn rsqrt_error(x: f32, y: f32) -> f64 {
        if !(y.is_finite() && y > 0.0) {
            return f64::INFINITY;
        }
        // 1/√x in f64: its own error, about 2⁻⁵², is negligible against u₃₂.
        let reference = 1.0 / f64::from(x).sqrt();
        ((f64::from(y) - reference) / reference).abs()
    }
}

impl Precision for f64 {
    const U: f64 = 1.1102230246251565e-16;

    fn rsqrt_error(x: f64, y: f64) -> f64 {
        if !(y.is_finite() && y > 0.0) {
            return f64::INFINITY;
        }
        // y² = p + pe and x p = a + ae exactly (error-free products by fma); then
        // t = x y² − 1 = (a − 1) + (ae + x pe), where a − 1 is exact (Sterbenz) when
        // y is close, and the rest is a correction of order u.
        let p = y * y;
        let pe = y.mul_add(y, -p);
        let a = x * p;
        let ae = x.mul_add(p, -a);
        let t = (a - 1.0) + (ae + x * pe);
        // x y² = (1 + e)², so e = √(1 + t) − 1 = t/2 − t²/8 + O(t³).
        if t.abs() < 1e-6 {
            (t / 2.0 - t * t / 8.0).abs()
        } else {
            ((1.0 + t).sqrt() - 1.0).abs()
        }
    }
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
            let ulps = T::rsqrt_error(xi, yi) / T::U;
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

    /// Log-uniform on the kernel domain [2⁻¹⁰⁸, 2⁷), rounded to `T`.
    pub fn domain<T: SimdScalar>(&mut self) -> T {
        T::from_f64((115.0 * self.unit() - 108.0).exp2())
    }

    /// Uniform in [1, 4), rounded to `T`.
    pub fn period<T: SimdScalar>(&mut self) -> T {
        let x = T::from_f64(1.0 + 3.0 * self.unit());
        if x < T::from_f64(4.0) { x } else { T::one() }
    }
}
