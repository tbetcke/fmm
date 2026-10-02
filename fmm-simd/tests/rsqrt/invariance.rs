//! Scale invariance, lane independence and argument checks, bit for bit.

use nd_fmm_simd::{Isa, rsqrt::rsqrt_slice};

use crate::common::{DOMAIN_MAX, DOMAIN_MIN, Precision, Rng, bits, isas, rsqrt};

/// Sampled x in [1, 4) per test and precision.
const SCALE_SAMPLES: usize = 2000;

/// rsqrt(4ᵏ x) = 2⁻ᵏ rsqrt(x) bit for bit, for sampled x in [1, 4) and every k with
/// 4ᵏ x in the kernel domain. This is what lets the exhaustive f32 check cover one
/// period only (design §4.3).
fn scale_invariance<T: Precision>(test: &str) {
    let mut rng = Rng(0x5ca1_e000);
    let mut x: Vec<T> = (0..SCALE_SAMPLES).map(|_| rng.period()).collect();
    x.extend([T::one(), T::from_f64(2.0), T::from_f64(4.0 - 4.0 * T::U)]);
    for isa in isas::<T>(test) {
        let base = rsqrt(isa, &x);
        let mut checked = 0usize;
        for k in -54..=3 {
            let scale = T::from_f64(4f64.powi(k));
            let unscale = T::from_f64(2f64.powi(-k));
            let (kept, scaled): (Vec<usize>, Vec<T>) = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| (i, xi * scale))
                .filter(|&(_, s)| DOMAIN_MIN <= s.to_f64() && s.to_f64() <= DOMAIN_MAX)
                .unzip();
            let got = rsqrt(isa, &scaled);
            for (&i, &g) in kept.iter().zip(&got) {
                assert_eq!(
                    bits(g),
                    bits(base[i] * unscale),
                    "{isa} rsqrt(4^{k} · {:e}) against 2^{} · rsqrt({:e})",
                    x[i].to_f64(),
                    -k,
                    x[i].to_f64()
                );
            }
            checked += kept.len();
        }
        println!("{test}: {isa}: {checked} scaled values bit for bit");
    }
}

#[test]
fn scale_invariance_f32() {
    scale_invariance::<f32>("scale_invariance_f32");
}

#[test]
fn scale_invariance_f64() {
    scale_invariance::<f64>("scale_invariance_f64");
}

/// A value's result depends neither on its lane nor on the slice length: every value
/// of slices of every length 0..=3W, with zeros among them, equals its result alone.
fn lane_independence<T: Precision>(test: &str) {
    let mut rng = Rng(0x1a7e_0000);
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
                let got = rsqrt(isa, &x);
                for (i, (&xi, &gi)) in x.iter().zip(&got).enumerate() {
                    let alone = rsqrt(isa, &[xi])[0];
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
    rsqrt_slice(Isa::Scalar, &[1.0f64; 3], &mut [0.0; 2]);
}

#[test]
fn unavailable_isa_panics() {
    for isa in Isa::all().filter(|isa| !isa.is_available()) {
        let result = std::panic::catch_unwind(|| rsqrt_slice(isa, &[1.0f32; 4], &mut [0.0; 4]));
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
