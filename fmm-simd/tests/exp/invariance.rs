//! Bit identity across ISAs, lane independence and argument checks.

use nd_fmm_simd::{Isa, exp::exp_slice};

use crate::common::{Precision, Rng, bits, exp, isas};

/// Sampled values per test and precision.
const SAMPLES: usize = 20_000;

/// Every ISA gives the scalar ISA's result bit for bit (the formulation uses only
/// operations every ISA rounds as IEEE 754 does; `nd_fmm_simd::exp`), on samples of the
/// domain and below it.
fn same_bits_on_every_isa<T: Precision>(test: &str) {
    let mut rng = Rng(0xb175_0000);
    let mut x: Vec<T> = (0..SAMPLES).map(|_| rng.domain()).collect();
    x.extend([
        T::zero(),
        -T::zero(),
        T::X_MIN,
        T::X_MIN - T::one(),
        T::neg_infinity(),
    ]);
    let scalar = exp(Isa::Scalar, &x);
    for isa in isas::<T>(test) {
        let got = exp(isa, &x);
        for (i, (&g, &s)) in got.iter().zip(&scalar).enumerate() {
            assert_eq!(
                bits(g),
                bits(s),
                "{isa} against scalar at x = {:e} (index {i})",
                x[i].to_f64()
            );
        }
        println!("{test}: {isa}: {} values bit for bit", x.len());
    }
}

#[test]
fn same_bits_on_every_isa_f32() {
    same_bits_on_every_isa::<f32>("same_bits_on_every_isa_f32");
}

#[test]
fn same_bits_on_every_isa_f64() {
    same_bits_on_every_isa::<f64>("same_bits_on_every_isa_f64");
}

/// A value's result depends neither on its lane nor on the slice length: every value of
/// slices of every length 0..=3W, with zeros among them, equals its result alone.
fn lane_independence<T: Precision>(test: &str) {
    let mut rng = Rng(0x1a7e_e000);
    for isa in isas::<T>(test) {
        let w = isa.lanes::<T>();
        for len in 0..=3 * w {
            for _ in 0..20 {
                let x: Vec<T> = (0..len)
                    .map(|_| {
                        if rng.next_u64().is_multiple_of(5) {
                            T::zero()
                        } else {
                            rng.domain()
                        }
                    })
                    .collect();
                let got = exp(isa, &x);
                for (i, (&xi, &gi)) in x.iter().zip(&got).enumerate() {
                    let alone = exp(isa, &[xi])[0];
                    assert_eq!(
                        bits(gi),
                        bits(alone),
                        "{isa} length {len}, index {i}, x = {:e}",
                        xi.to_f64()
                    );
                }
            }
        }
    }
}

#[test]
fn lane_independence_f32() {
    lane_independence::<f32>("lane_independence_f32");
}

#[test]
fn lane_independence_f64() {
    lane_independence::<f64>("lane_independence_f64");
}

#[test]
#[should_panic(expected = "`x` and `out` must have the same length, got 3 and 2")]
fn mismatched_lengths_panic() {
    exp_slice(Isa::Scalar, &[-1.0f64; 3], &mut [0.0; 2]);
}

#[test]
fn unavailable_isa_panics() {
    for isa in Isa::all().filter(|isa| !isa.is_available()) {
        let result = std::panic::catch_unwind(|| exp_slice(isa, &[-1.0f32; 4], &mut [0.0; 4]));
        let message = result.expect_err("an unavailable ISA must panic");
        let message = message
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert_eq!(
            message,
            format!("instruction set `{isa}` is not available on this machine")
        );
        println!("unavailable_isa_panics: {isa} panics");
    }
}
