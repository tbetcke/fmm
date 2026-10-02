//! The contract of design §4.3: within 4 u_T relative on the kernel domain, and +0 at
//! r² = 0. f32 over one period [1, 4), which by scale invariance (`invariance`) covers
//! every normal input; f64 on seeded log-uniform samples; both at the powers of two,
//! their neighbours and the domain ends.

use nd_fmm_simd::Isa;

use crate::common::{BATCH, DOMAIN_MAX, DOMAIN_MIN, MaxError, Precision, Rng, bits, isas, rsqrt};

/// Every `stride`-th f32 in [1, 4), from 1.
fn f32_period(test: &str, stride: usize) {
    let first = 1.0f32.to_bits();
    let end = 4.0f32.to_bits();
    let what = if stride == 1 {
        "f32, every float in [1, 4)".to_string()
    } else {
        format!("f32, every {stride}th float in [1, 4)")
    };
    for isa in isas::<f32>(test) {
        let mut max = MaxError::default();
        let mut count = 0usize;
        let mut x = Vec::with_capacity(BATCH);
        let mut bits = (first..end).step_by(stride).peekable();
        while bits.peek().is_some() {
            x.clear();
            x.extend(bits.by_ref().take(BATCH).map(f32::from_bits));
            max.record(&x, &rsqrt(isa, &x));
            count += x.len();
        }
        max.check(&format!("{what} ({count} values)"), isa);
    }
}

#[test]
fn f32_contract_strided() {
    f32_period("f32_contract_strided", 97);
}

#[test]
#[ignore = "exhaustive; run in release with --ignored"]
fn f32_contract_exhaustive() {
    f32_period("f32_contract_exhaustive", 1);
}

/// `n` seeded log-uniform samples of the kernel domain.
fn f64_sampled(test: &str, n: usize) {
    for isa in isas::<f64>(test) {
        let mut rng = Rng(0x0005_eed0_f64d_0a1e);
        let mut max = MaxError::default();
        let mut remaining = n;
        let mut x = Vec::with_capacity(BATCH);
        while remaining > 0 {
            let len = remaining.min(BATCH);
            x.clear();
            x.extend((0..len).map(|_| rng.domain::<f64>()));
            max.record(&x, &rsqrt(isa, &x));
            remaining -= len;
        }
        max.check(&format!("f64, {n} log-uniform samples of the domain"), isa);
    }
}

#[test]
fn f64_contract_sampled() {
    f64_sampled("f64_contract_sampled", 100_000);
}

#[test]
#[ignore = "10⁷ samples; run in release with --ignored"]
fn f64_contract_sampled_large() {
    f64_sampled("f64_contract_sampled_large", 10_000_000);
}

/// The powers of two of the kernel domain, their neighbours inside it, and its ends.
fn special_inputs<T: Precision>() -> Vec<T> {
    let mut x = Vec::new();
    for k in -108..=7 {
        let p = T::from_f64(2f64.powi(k));
        // The neighbours one unit in the last place below and above.
        let below = T::from_f64(p.to_f64() * (1.0 - T::U));
        let above = T::from_f64(p.to_f64() * (1.0 + 2.0 * T::U));
        x.extend([below, p, above]);
    }
    let (lo, hi) = (T::from_f64(DOMAIN_MIN), T::from_f64(DOMAIN_MAX));
    x.retain(|&v| lo <= v && v <= hi);
    x.extend([lo, hi]);
    x
}

/// The special inputs within 4 u_T, and r² = 0 giving +0, alone and among them.
fn special_values<T: Precision>(test: &str) {
    let x = special_inputs::<T>();
    assert!(x.len() > 3 * 115, "{} special inputs", x.len());
    for isa in isas::<T>(test) {
        let mut max = MaxError::default();
        max.record(&x, &rsqrt(isa, &x));
        max.check(
            &format!(
                "{}, powers of two, neighbours and domain ends",
                std::any::type_name::<T>()
            ),
            isa,
        );
        check_zero::<T>(isa, &x);
    }
}

/// r² = 0 gives +0 alone and between other values; the other values are unchanged.
fn check_zero<T: Precision>(isa: Isa, x: &[T]) {
    let zero = rsqrt(isa, &[T::zero()]);
    assert_eq!(bits(zero[0]), bits(T::zero()), "{isa} rsqrt(0)");
    let mut mixed: Vec<T> = x.iter().take(20).copied().collect();
    for i in (0..mixed.len()).step_by(3) {
        mixed[i] = T::zero();
    }
    let without: Vec<T> = mixed
        .iter()
        .map(|&v| if v == T::zero() { T::one() } else { v })
        .collect();
    let (got, other) = (rsqrt(isa, &mixed), rsqrt(isa, &without));
    for i in 0..mixed.len() {
        let want = if mixed[i] == T::zero() {
            T::zero()
        } else {
            other[i]
        };
        assert_eq!(
            bits(got[i]),
            bits(want),
            "{isa} rsqrt with zeros, index {i}"
        );
    }
}

#[test]
fn special_values_f32() {
    special_values::<f32>("special_values_f32");
}

#[test]
fn special_values_f64() {
    special_values::<f64>("special_values_f64");
}
