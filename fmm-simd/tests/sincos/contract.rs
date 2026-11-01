//! The contract of design §4.11: within 4 u_T absolute on |θ| ≤ 2¹⁰ for both the sine and
//! the cosine; odd and even bit for bit for θ ≠ 0; at ±0 the sine +0, the cosine exactly 1.
//! f32 over every float of [0, 2¹⁰] (strided in debug), with its negation, on all the
//! machine's threads, the oracle formed once per value for every ISA (CI's x86_64 job has
//! five minutes); f64 on seeded samples; both at the reduction's boundaries (n + ½) π, the
//! multiples n π, their neighbours and the domain end.

use nd_fmm_simd::Isa;

use crate::common::{BATCH, MaxError, Precision, Rng, THETA_MAX, bits, isas, sincos, sincos_dd};

/// Asserts sin(−θ) = −sin θ and cos(−θ) = cos θ bit for bit, given both runs, for θ ≠ 0
/// (at ±0 the sine is +0 on both sides; `special_values`).
fn check_symmetry<T: Precision>(isa: Isa, theta: &[T], plus: &(Vec<T>, Vec<T>)) {
    let negated: Vec<T> = theta.iter().map(|&x| -x).collect();
    let minus = sincos(isa, &negated);
    for (i, &x) in theta.iter().enumerate() {
        let odd = if x == T::zero() {
            plus.0[i]
        } else {
            -plus.0[i]
        };
        assert_eq!(
            bits(minus.0[i]),
            bits(odd),
            "{isa} sin(−θ) ≠ −sin θ at θ = {:e}",
            x.to_f64()
        );
        assert_eq!(
            bits(minus.1[i]),
            bits(plus.1[i]),
            "{isa} cos(−θ) ≠ cos θ at θ = {:e}",
            x.to_f64()
        );
    }
}

/// Every `stride`-th f32 of [0, θ_max], and its negation for the symmetries: the values
/// split into one contiguous range per thread, each batch's oracle (`f64::sin`, `f64::cos`)
/// formed once and every ISA checked against it.
fn f32_domain(test: &str, stride: usize) {
    let count = ((THETA_MAX as f32).to_bits() as usize + 1).div_ceil(stride);
    let what = if stride == 1 {
        "f32, every float in [0, 2¹⁰]".to_string()
    } else {
        format!("f32, every {stride}th float in [0, 2¹⁰]")
    };
    let isas = isas::<f32>(test);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per_thread = count.div_ceil(threads);
    let maxima: Vec<Vec<MaxError>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                let isas = &isas;
                scope.spawn(move || {
                    let (first, last) = (t * per_thread, ((t + 1) * per_thread).min(count));
                    let mut max = vec![MaxError::default(); isas.len()];
                    let mut theta = Vec::with_capacity(BATCH);
                    let mut reference = Vec::with_capacity(BATCH);
                    let mut i = first;
                    while i < last {
                        let len = BATCH.min(last - i);
                        theta.clear();
                        theta.extend((i..i + len).map(|j| {
                            f32::from_bits(u32::try_from(j * stride).expect("an f32's bits"))
                        }));
                        reference.clear();
                        reference.extend(theta.iter().map(|&x| {
                            let x = f64::from(x);
                            (x.sin(), x.cos())
                        }));
                        for (&isa, max) in isas.iter().zip(&mut max) {
                            let result = sincos(isa, &theta);
                            max.record_against(&theta, &result.0, &result.1, &reference);
                            check_symmetry(isa, &theta, &result);
                        }
                        i += len;
                    }
                    max
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    for (k, &isa) in isas.iter().enumerate() {
        let mut max = MaxError::default();
        for thread in &maxima {
            max.merge(&thread[k]);
        }
        max.check(
            &format!("{what} ({count} values, and their negations)"),
            isa,
        );
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

/// `n` seeded samples of the domain, half uniform and half of small magnitude, and their
/// negations for the symmetries.
fn f64_sampled(test: &str, n: usize) {
    for isa in isas::<f64>(test) {
        let mut rng = Rng(0x51c0_5f64_5eed);
        let mut max = MaxError::default();
        let mut remaining = n;
        let mut theta = Vec::with_capacity(BATCH);
        while remaining > 0 {
            let len = remaining.min(BATCH);
            theta.clear();
            theta.extend((0..len).map(|_| rng.domain::<f64>()));
            let result = sincos(isa, &theta);
            max.record(&theta, &result.0, &result.1);
            check_symmetry(isa, &theta, &result);
            remaining -= len;
        }
        max.check(
            &format!("f64, {n} samples of the domain (and their negations)"),
            isa,
        );
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

/// The reduction's boundaries θ = (n + ½) π and the multiples n π in the domain, each with
/// the neighbours a few units in the last place on each side, and the domain end with its
/// neighbours below it.
pub fn special_inputs<T: Precision>() -> Vec<T> {
    let mut theta = Vec::new();
    let mut push_with_neighbours = |v: f64| {
        let b = T::from_f64(v);
        let mut below = b;
        let mut above = b;
        theta.push(b);
        for _ in 0..3 {
            below = T::from_f64(below.to_f64() * (1.0 - T::U));
            above = T::from_f64(above.to_f64() * (1.0 + 2.0 * T::U));
            theta.extend([below, above]);
        }
    };
    let pi = core::f64::consts::PI;
    let mut n = 0.0;
    while n * pi <= THETA_MAX {
        push_with_neighbours((n + 0.5) * pi);
        if n > 0.0 {
            push_with_neighbours(n * pi);
        }
        n += 1.0;
    }
    let mut end = T::from_f64(THETA_MAX);
    for _ in 0..4 {
        theta.push(end);
        end = T::from_f64(end.to_f64() * (1.0 - T::U));
    }
    theta.retain(|&v| v.to_f64() <= THETA_MAX);
    theta
}

/// The special inputs within 4 u_T, odd and even; sin(±0) = +0 and cos(±0) = 1 exactly
/// (the sine's last step adds the correction t s P(s), a zero of the opposite sign at
/// t = −0, to t: design §4.11).
fn special_values<T: Precision>(test: &str) {
    let theta = special_inputs::<T>();
    assert!(theta.len() > 7 * 600, "{} special inputs", theta.len());
    for isa in isas::<T>(test) {
        let mut max = MaxError::default();
        let result = sincos(isa, &theta);
        max.record(&theta, &result.0, &result.1);
        check_symmetry(isa, &theta, &result);
        max.check(
            &format!(
                "{}, reduction boundaries, multiples of π, neighbours and domain end",
                std::any::type_name::<T>()
            ),
            isa,
        );
        for zero in [T::zero(), -T::zero()] {
            let (s, c) = sincos(isa, &[zero]);
            assert_eq!(
                bits(s[0]),
                bits(T::zero()),
                "{isa} sin({:e})",
                zero.to_f64()
            );
            assert_eq!(bits(c[0]), bits(T::one()), "{isa} cos({:e})", zero.to_f64());
        }
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

/// A double-double, hi + lo.
type DoubleDouble = (f64, f64);

/// The f64 oracle: `sincos_dd` at six f64 values against sin θ and cos θ computed once in
/// 80-digit decimal arithmetic (Python's `decimal`: the reduction by a 100-digit π, then
/// the Taylor series), to an absolute 2⁻¹⁰⁰; and on samples of the domain within 2 u₆₄
/// absolute of `f64::sin` and `f64::cos`.
#[test]
fn f64_oracle() {
    #[expect(clippy::approx_constant, reason = "the f64 nearest π/2 as an input")]
    let known: [(f64, DoubleDouble, DoubleDouble); 6] = [
        (
            1.0,
            (0.8414709848078965, 1.776845092935536e-18),
            (0.5403023058681398, -4.760954612604417e-17),
        ),
        (
            1e-3,
            (0.0009999998333333417, 5.670638989736153e-21),
            (0.9999995000000417, -7.831485455398128e-18),
        ),
        (
            1.5707963267948966,
            (1.0, -1.874699728327322e-33),
            (6.123233995736766e-17, -1.4973849048591698e-33),
        ),
        (
            100.0,
            (-0.5063656411097588, -3.050947053792115e-18),
            (0.8623188722876839, 4.334809858136501e-17),
        ),
        (
            1023.75,
            (-0.3978801555122713, 2.6189437164266964e-17),
            (0.9174373994172741, -4.505747600932243e-17),
        ),
        (
            942.4777960769379,
            (-5.09502586896226e-14, -3.8346802804659844e-31),
            (1.0, -1.2979644302697316e-27),
        ),
    ];
    for (theta, sin, cos) in known {
        let (s, c) = sincos_dd(theta);
        for (what, got, want) in [("sin", s, sin), ("cos", c, cos)] {
            let error = ((got.0 - want.0) + (got.1 - want.1)).abs();
            assert!(
                error <= 2f64.powi(-100),
                "sincos_dd({theta}) {what}: {error:e}"
            );
        }
    }
    let mut rng = Rng(0x0dd0_51c0);
    for _ in 0..10_000 {
        let theta: f64 = rng.domain();
        let (s, c) = sincos_dd(theta);
        for (what, got, libm) in [("sin", s.0, theta.sin()), ("cos", c.0, theta.cos())] {
            assert!(
                (got - libm).abs() <= 2.0 * f64::U,
                "sincos_dd({theta:e}) {what} = {got:e}, libm {libm:e}"
            );
        }
    }
}
