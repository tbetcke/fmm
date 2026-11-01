//! Bit identity across ISAs, lane independence and argument checks.

use nd_fmm_simd::{Isa, sincos::sincos_slice};

use crate::common::{Precision, Rng, bits, isas, sincos};

/// Sampled values per test and precision.
const SAMPLES: usize = 20_000;

/// A sample of the domain, negated every third draw.
fn signed_sample<T: Precision>(rng: &mut Rng) -> T {
    let x: T = rng.domain();
    if rng.next_u64().is_multiple_of(3) {
        -x
    } else {
        x
    }
}

/// Every ISA gives the scalar ISA's results bit for bit (the formulation uses only
/// operations every ISA rounds as IEEE 754 does; `nd_fmm_simd::sincos`), on samples of the
/// domain of both signs.
fn same_bits_on_every_isa<T: Precision>(test: &str) {
    let mut rng = Rng(0xb175_51c0);
    let mut theta: Vec<T> = (0..SAMPLES).map(|_| signed_sample(&mut rng)).collect();
    theta.extend([T::zero(), -T::zero(), T::from_f64(1024.0)]);
    let scalar = sincos(Isa::Scalar, &theta);
    for isa in isas::<T>(test) {
        let got = sincos(isa, &theta);
        for (i, &x) in theta.iter().enumerate() {
            for (what, g, s) in [
                ("sin", got.0[i], scalar.0[i]),
                ("cos", got.1[i], scalar.1[i]),
            ] {
                assert_eq!(
                    bits(g),
                    bits(s),
                    "{isa} {what} against scalar at θ = {:e} (index {i})",
                    x.to_f64()
                );
            }
        }
        println!("{test}: {isa}: {} values bit for bit", theta.len());
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

/// A value's results depend neither on its lane nor on the slice length: every value of
/// slices of every length 0..=3W, with zeros among them, equals its results alone.
fn lane_independence<T: Precision>(test: &str) {
    let mut rng = Rng(0x1a7e_51c0);
    for isa in isas::<T>(test) {
        let w = isa.lanes::<T>();
        for len in 0..=3 * w {
            for _ in 0..20 {
                let theta: Vec<T> = (0..len)
                    .map(|_| {
                        if rng.next_u64().is_multiple_of(5) {
                            T::zero()
                        } else {
                            signed_sample(&mut rng)
                        }
                    })
                    .collect();
                let got = sincos(isa, &theta);
                for (i, &x) in theta.iter().enumerate() {
                    let alone = sincos(isa, &[x]);
                    assert_eq!(
                        (bits(got.0[i]), bits(got.1[i])),
                        (bits(alone.0[0]), bits(alone.1[0])),
                        "{isa} length {len}, index {i}, θ = {:e}",
                        x.to_f64()
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
#[should_panic(expected = "`theta`, `sin` and `cos` must have the same length, got 3, 3 and 2")]
fn mismatched_lengths_panic() {
    sincos_slice(Isa::Scalar, &[1.0f64; 3], &mut [0.0; 3], &mut [0.0; 2]);
}

#[test]
fn unavailable_isa_panics() {
    for isa in Isa::all().filter(|isa| !isa.is_available()) {
        let result = std::panic::catch_unwind(|| {
            sincos_slice(isa, &[1.0f32; 4], &mut [0.0; 4], &mut [0.0; 4]);
        });
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
