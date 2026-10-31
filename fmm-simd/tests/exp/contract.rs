//! The contract of design §4.9: within 4 u_T relative on the domain [x_min, 0], exactly 1
//! at ±0, and below the domain the value at x_min. f32 over every float of the domain
//! (strided in debug); f64 on seeded samples; both at the reduction's boundaries
//! (n + ½) ln 2, their neighbours and the domain end.

use nd_fmm_simd::Isa;

use crate::common::{BATCH, MaxError, Precision, Rng, bits, exp, exp_dd, isas};

/// Every `stride`-th f32 of the domain, from −0 down to x_min.
fn f32_domain(test: &str, stride: usize) {
    // The negative floats in order of magnitude: their bits grow from −0's.
    let first = (-0.0f32).to_bits();
    let end = f32::X_MIN.to_bits() + 1;
    let what = if stride == 1 {
        "f32, every float in [x_min, 0]".to_string()
    } else {
        format!("f32, every {stride}th float in [x_min, 0]")
    };
    for isa in isas::<f32>(test) {
        let mut max = MaxError::default();
        let mut count = 0usize;
        let mut x = Vec::with_capacity(BATCH);
        let mut bits = (first..end).step_by(stride).peekable();
        while bits.peek().is_some() {
            x.clear();
            x.extend(bits.by_ref().take(BATCH).map(f32::from_bits));
            max.record(&x, &exp(isa, &x));
            count += x.len();
        }
        max.check(&format!("{what} ({count} values)"), isa);
    }
}

#[test]
fn f32_contract_strided() {
    f32_domain("f32_contract_strided", 9973);
}

#[test]
#[ignore = "exhaustive; run in release with --ignored"]
fn f32_contract_exhaustive() {
    f32_domain("f32_contract_exhaustive", 1);
}

/// `n` seeded samples of the domain, half uniform and half of small magnitude.
fn f64_sampled(test: &str, n: usize) {
    for isa in isas::<f64>(test) {
        let mut rng = Rng(0x0e8b_0f64_5eed);
        let mut max = MaxError::default();
        let mut remaining = n;
        let mut x = Vec::with_capacity(BATCH);
        while remaining > 0 {
            let len = remaining.min(BATCH);
            x.clear();
            x.extend((0..len).map(|_| rng.domain::<f64>()));
            max.record(&x, &exp(isa, &x));
            remaining -= len;
        }
        max.check(&format!("f64, {n} samples of the domain"), isa);
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

/// The reduction's boundaries x = (n + ½) ln 2 for every n of the domain, with the
/// neighbours a few units in the last place on each side, the domain end and its
/// neighbours inside the domain.
fn special_inputs<T: Precision>() -> Vec<T> {
    let x_min = T::X_MIN.to_f64();
    let mut x = Vec::new();
    let mut n = 0.0;
    while (n + 0.5) * core::f64::consts::LN_2 <= -x_min {
        let b = T::from_f64(-(n + 0.5) * core::f64::consts::LN_2);
        let mut below = b;
        let mut above = b;
        x.push(b);
        for _ in 0..3 {
            below = T::from_f64(below.to_f64() * (1.0 + 2.0 * T::U));
            above = T::from_f64(above.to_f64() * (1.0 - T::U));
            x.extend([below, above]);
        }
        n += 1.0;
    }
    let mut inside = T::X_MIN;
    for _ in 0..4 {
        x.push(inside);
        inside = T::from_f64(inside.to_f64() * (1.0 - T::U));
    }
    x.retain(|&v| T::X_MIN <= v && v <= T::zero());
    x
}

/// The special inputs within 4 u_T; ±0 exactly 1; below the domain the value at x_min.
fn special_values<T: Precision>(test: &str) {
    let x = special_inputs::<T>();
    assert!(x.len() > 7 * 120, "{} special inputs", x.len());
    for isa in isas::<T>(test) {
        let mut max = MaxError::default();
        max.record(&x, &exp(isa, &x));
        max.check(
            &format!(
                "{}, reduction boundaries, neighbours and domain end",
                std::any::type_name::<T>()
            ),
            isa,
        );
        check_zero_and_below::<T>(isa);
    }
}

/// e^{±0} = 1 exactly; every input below x_min (−∞ included) gives the result at x_min,
/// positive and at most 2^(e_min + 2), while the first float above x_min differs from it.
fn check_zero_and_below<T: Precision>(isa: Isa) {
    for zero in [T::zero(), -T::zero()] {
        assert_eq!(
            bits(exp(isa, &[zero])[0]),
            bits(T::one()),
            "{isa} exp({:e})",
            zero.to_f64()
        );
    }
    let at_min = exp(isa, &[T::X_MIN])[0];
    assert!(
        at_min > T::zero() && at_min.to_f64() <= T::BELOW,
        "{isa} exp(x_min) = {:e}",
        at_min.to_f64()
    );
    let below = [
        T::from_f64(T::X_MIN.to_f64() * (1.0 + 2.0 * T::U)),
        T::X_MIN - T::one(),
        T::from_f64(-1e30),
        T::min_value(),
        T::neg_infinity(),
    ];
    for (&xi, yi) in below.iter().zip(exp(isa, &below)) {
        assert_eq!(
            bits(yi),
            bits(at_min),
            "{isa} exp({:e}) below the domain",
            xi.to_f64()
        );
    }
    let inside = T::from_f64(T::X_MIN.to_f64() * (1.0 - T::U));
    assert_ne!(
        bits(exp(isa, &[inside])[0]),
        bits(at_min),
        "{isa}: x_min is the clamp's value"
    );
}

#[test]
fn special_values_f32() {
    special_values::<f32>("special_values_f32");
}

#[test]
fn special_values_f64() {
    special_values::<f64>("special_values_f64");
}

/// The f64 oracle: `exp_dd` at three f64 values against e^x 2⁻ⁿ to 90 digits (Python's
/// `decimal`), to a relative 2⁻¹⁰⁰, with the n of the reduction; and on samples of the
/// domain within one unit in the last place of `f64::exp`.
#[test]
fn f64_oracle() {
    let known = [
        (-1.0, -1, 0.7357588823428847, -2.4857507345576725e-17),
        (-0.3, 0, 0.7408182206817179, -1.805530505953e-18),
        (-700.0, -1010, 1.081828175944821, 9.312926389907246e-18),
    ];
    for (x, n, hi, lo) in known {
        let (h, l, m) = exp_dd(x);
        assert_eq!(m, n, "exp_dd({x}): the reduction's n");
        let error = ((h - hi) + (l - lo)).abs() / hi;
        assert!(error <= 2f64.powi(-100), "exp_dd({x}): {error:e}");
    }
    let mut rng = Rng(0x0dd0_0dd0);
    for _ in 0..10_000 {
        let x: f64 = rng.domain();
        let (h, _, n) = exp_dd(x);
        let h = h * 2f64.powi(n);
        let libm = x.exp();
        assert!(
            (h - libm).abs() <= 2.0 * f64::U * h,
            "exp_dd({x:e}) = {h:e}, libm {libm:e}"
        );
    }
}
