//! The vector layer of every ISA this machine offers against the scalar implementation
//! [`Scalar`] (Phase 3S T4), on seeded random vectors.
//!
//! Error measure: exact equality of bits, lane by lane, for every method but the two
//! inverse square roots. The arithmetic is IEEE 754 on both sides, each operation
//! rounded once, and the inputs are finite and moderate, so no NaN with an
//! ISA-dependent payload arises (NaN only passes through the and-not unchanged). The
//! kernel's inverse square root equals the scalar path bit for bit where both use
//! `sqrt` and a division (NEON); on AVX2 it is checked against 1/√x within its
//! contract here, and exhaustively in `tests/rsqrt`. The estimate is checked against
//! its documented bound.
//!
//! The P2P kernel body is checked here where it is generic: every K gives the same
//! bits on every ISA, and the body on the scalar layer (W = 1), which the scalar ISA
//! does not run, is within the term and sum tolerances of the reference. The public
//! kernel is tested in `tests/p2p`.
//!
//! Every test prints the ISAs it ran; run with `--show-output` to see it.

use std::panic::{AssertUnwindSafe, catch_unwind};

#[cfg(target_arch = "x86_64")]
use super::avx2::Avx2;
#[cfg(target_arch = "aarch64")]
use super::neon::Neon;
use super::{Simd, p2p::p2p_body, scalar::Scalar};
use crate::{Isa, SimdScalar};

/// Random vectors per ISA, precision and test.
const TRIALS: usize = 2000;

/// Runs the generic check `$check::<$t, _>(token)` on the token of every available ISA
/// and prints the ISAs it ran under the test's name.
macro_rules! on_every_isa {
    ($test:expr, $t:ty, $check:ident) => {{
        let mut ran = vec![Isa::Scalar.to_string()];
        $check::<$t, _>(Scalar);
        #[cfg(target_arch = "aarch64")]
        if let Some(s) = Neon::try_new() {
            $check::<$t, _>(s);
            ran.push(Isa::Neon.to_string());
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(s) = Avx2::try_new() {
            $check::<$t, _>(s);
            ran.push(Isa::Avx2.to_string());
        }
        println!(
            "{} ({}): ISAs run: {}",
            $test,
            stringify!($t),
            ran.join(", ")
        );
    }};
}

/// SplitMix64, seeded.
struct Rng(u64);

impl Rng {
    /// The next 64 random bits.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 2f64.powi(-53)
    }

    /// A finite value of either sign with a random significand, of magnitude in
    /// [2⁻⁸, 2⁸), or ±0 one time in eight.
    fn value<T: SimdScalar>(&mut self) -> T {
        let bits = self.next_u64();
        if bits.is_multiple_of(8) {
            return T::from_f64(if bits & 8 == 0 { 0.0 } else { -0.0 });
        }
        let magnitude = (16.0 * self.unit() - 8.0).exp2();
        T::from_f64(if bits & 16 == 0 {
            magnitude
        } else {
            -magnitude
        })
    }

    /// A value in the kernel domain 2⁻¹⁰⁸ ≤ x ≤ 2⁷, log-uniform.
    fn domain<T: SimdScalar>(&mut self) -> T {
        T::from_f64((115.0 * self.unit() - 108.0).exp2())
    }

    /// `n` random values.
    fn values<T: SimdScalar>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.value()).collect()
    }
}

/// The unit roundoff of `T`.
fn unit_roundoff<T: SimdScalar>() -> f64 {
    T::epsilon().to_f64() / 2.0
}

/// The bits of `x`, widened to f64 (exact and injective, NaN payloads included).
fn bits<T: SimdScalar>(x: T) -> u64 {
    x.to_f64().to_bits()
}

/// The lanes of `v`.
fn lanes<T: SimdScalar, S: Simd<T>>(s: S, v: S::V) -> Vec<T> {
    let mut out = vec![T::zero(); S::W];
    s.store(v, &mut out);
    out
}

/// Asserts that `got` and `want` agree bit for bit, lane by lane.
fn assert_bits<T: SimdScalar, S: Simd<T>>(what: &str, got: &[T], want: &[T]) {
    let got: Vec<u64> = got.iter().map(|&x| bits(x)).collect();
    let want: Vec<u64> = want.iter().map(|&x| bits(x)).collect();
    assert_eq!(got, want, "{} {what}", S::ISA);
}

/// Splat, load and store, and the arithmetic, lane by lane against the scalar path.
fn arithmetic<T: SimdScalar, S: Simd<T>>(s: S) {
    let mut rng = Rng(0x5eed_0001);
    let w = S::W;
    for _ in 0..TRIALS {
        let a: Vec<T> = rng.values(w);
        let b: Vec<T> = rng.values(w);
        let c: Vec<T> = rng.values(w);
        let (va, vb, vc) = (s.load(&a), s.load(&b), s.load(&c));
        let x = a[0];
        assert_bits::<T, S>("splat", &lanes(s, s.splat(x)), &vec![x; w]);
        assert_bits::<T, S>("load and store", &lanes(s, va), &a);
        let lane_wise = |f: &dyn Fn(usize) -> T| (0..w).map(f).collect::<Vec<T>>();
        let sc = Scalar;
        assert_bits::<T, S>(
            "add",
            &lanes(s, s.add(va, vb)),
            &lane_wise(&|i| sc.add(a[i], b[i])),
        );
        assert_bits::<T, S>(
            "sub",
            &lanes(s, s.sub(va, vb)),
            &lane_wise(&|i| sc.sub(a[i], b[i])),
        );
        assert_bits::<T, S>(
            "mul",
            &lanes(s, s.mul(va, vb)),
            &lane_wise(&|i| sc.mul(a[i], b[i])),
        );
        assert_bits::<T, S>(
            "fma",
            &lanes(s, s.fma(va, vb, vc)),
            &lane_wise(&|i| sc.fma(a[i], b[i], c[i])),
        );
        assert_bits::<T, S>(
            "fnma",
            &lanes(s, s.fnma(va, vb, vc)),
            &lane_wise(&|i| sc.fnma(a[i], b[i], c[i])),
        );
    }
}

/// Load and store touch exactly the first W values of a longer slice.
fn load_store_prefix<T: SimdScalar, S: Simd<T>>(s: S) {
    let mut rng = Rng(0x5eed_0002);
    let w = S::W;
    for _ in 0..TRIALS / 10 {
        let long: Vec<T> = rng.values(w + 3);
        assert_bits::<T, S>("load of a prefix", &lanes(s, s.load(&long)), &long[..w]);
        let mut out = vec![T::from_f64(7.0); w + 3];
        s.store(s.load(&long), &mut out);
        assert_bits::<T, S>("store of a prefix", &out[..w], &long[..w]);
        assert_bits::<T, S>("store past the prefix", &out[w..], &[T::from_f64(7.0); 3]);
    }
}

/// Compare-equal and and-not against the scalar path, with equal lanes, ±0 and
/// non-finite values behind the mask.
fn masks<T: SimdScalar, S: Simd<T>>(s: S) {
    let mut rng = Rng(0x5eed_0003);
    let w = S::W;
    let specials = [T::infinity(), T::neg_infinity(), T::nan()];
    for _ in 0..TRIALS {
        let a: Vec<T> = rng.values(w);
        let mut b: Vec<T> = rng.values(w);
        let mut c: Vec<T> = rng.values(w);
        for i in 0..w {
            match rng.next_u64() % 4 {
                0 => b[i] = a[i],
                1 => b[i] = -a[i],
                _ => {}
            }
            if rng.next_u64().is_multiple_of(4) {
                c[i] = specials[(rng.next_u64() % 3) as usize];
            }
        }
        let got = lanes(s, s.and_not(s.eq(s.load(&a), s.load(&b)), s.load(&c)));
        let want: Vec<T> = (0..w)
            .map(|i| Scalar.and_not(Scalar.eq(a[i], b[i]), c[i]))
            .collect();
        assert_bits::<T, S>("and_not(eq(a, b), c)", &got, &want);
        for i in 0..w {
            if a[i] == b[i] {
                assert_eq!(bits(got[i]), bits(T::zero()), "{} masked lane", S::ISA);
            }
        }
    }
}

/// The relative error of `y` as 1/√x, against 1/√x in f64. For f64 inputs this
/// reference has its own error of up to 1.5 u₆₄, which the callers allow for.
fn rsqrt_error<T: SimdScalar>(x: T, y: T) -> f64 {
    let reference = 1.0 / x.to_f64().sqrt();
    ((y.to_f64() - reference) / reference).abs()
}

/// The kernel's inverse square root (unmasked and masked) and the estimate, on the
/// kernel domain and at 0.
fn inverse_square_roots<T: SimdScalar, S: Simd<T>>(s: S) {
    let mut rng = Rng(0x5eed_0004);
    let w = S::W;
    let u = unit_roundoff::<T>();
    // The contract (4 u_T), plus the f64 reference's own error for f64 inputs.
    let rsqrt_bound = if u < 1e-10 { 5.5 * u } else { 4.0 * u };
    let estimate_bound = match S::ISA {
        Isa::Neon => 2f64.powi(-8),
        // `vrsqrtps` (Intel Intrinsics Guide), plus the conversion of f64 to f32.
        Isa::Avx2 => 1.5 * 2f64.powi(-12) + 2f64.powi(-24),
        _ => rsqrt_bound,
    };
    for _ in 0..TRIALS {
        let mut x: Vec<T> = (0..w).map(|_| rng.domain()).collect();
        let v = s.load(&x);
        let rsqrt = lanes(s, s.rsqrt(v));
        let estimate = lanes(s, s.rsqrt_estimate(v));
        for i in 0..w {
            let e = rsqrt_error(x[i], rsqrt[i]);
            assert!(
                e <= rsqrt_bound,
                "{} rsqrt({:e}): {e:e}",
                S::ISA,
                x[i].to_f64()
            );
            let e = rsqrt_error(x[i], estimate[i]);
            assert!(
                e <= estimate_bound,
                "{} estimate({:e}): {e:e}",
                S::ISA,
                x[i].to_f64()
            );
        }
        if S::ISA != Isa::Avx2 {
            // `sqrt` and a division, as on the scalar path.
            let want: Vec<T> = x.iter().map(|&xi| Scalar.rsqrt(xi)).collect();
            assert_bits::<T, S>("rsqrt", &rsqrt, &want);
        }
        // Zeros in random lanes: masked to +0, the other lanes unchanged.
        for xi in x.iter_mut() {
            if rng.next_u64().is_multiple_of(3) {
                *xi = T::zero();
            }
        }
        let rsqrt = lanes(s, s.rsqrt(s.load(&x)));
        let masked = lanes(s, s.rsqrt_masked(s.load(&x)));
        for i in 0..w {
            let want = if x[i] == T::zero() {
                T::zero()
            } else {
                rsqrt[i]
            };
            assert_eq!(bits(masked[i]), bits(want), "{} rsqrt_masked", S::ISA);
        }
    }
}

/// The interleaved block load and store against the triples, touching exactly the
/// first W of a longer slice, and the broadcast of one source.
fn blocks<T: SimdScalar, S: Simd<T>>(s: S) {
    let mut rng = Rng(0x5eed_0005);
    let w = S::W;
    for _ in 0..TRIALS / 10 {
        let points: Vec<[T; 3]> = (0..w + 2)
            .map(|_| [rng.value(), rng.value(), rng.value()])
            .collect();
        let v = s.load3(&points);
        for (k, vk) in v.iter().enumerate() {
            let want: Vec<T> = points[..w].iter().map(|p| p[k]).collect();
            assert_bits::<T, S>("load3", &lanes(s, *vk), &want);
        }
        let sentinel = [T::from_f64(7.0); 3];
        let mut out = vec![sentinel; w + 2];
        s.store3(v, &mut out);
        let flat = |p: &[[T; 3]]| p.iter().flatten().copied().collect::<Vec<T>>();
        assert_bits::<T, S>("store3", &flat(&out[..w]), &flat(&points[..w]));
        assert_bits::<T, S>(
            "store3 past the block",
            &flat(&out[w..]),
            &flat(&[sentinel; 2]),
        );

        let point = points[w + 1];
        let charge = rng.value();
        let (vp, vq) = s.broadcast(&point, charge);
        for (k, vk) in vp.iter().enumerate() {
            assert_bits::<T, S>("broadcast point", &lanes(s, *vk), &vec![point[k]; w]);
        }
        assert_bits::<T, S>("broadcast charge", &lanes(s, vq), &vec![charge; w]);
    }
}

/// Every load and store panics, without touching memory, on a slice shorter than W.
fn short_slices_panic<T: SimdScalar, S: Simd<T>>(s: S) {
    let w = S::W;
    let values = vec![T::one(); w - 1];
    let mut out = vec![T::one(); w - 1];
    let points = vec![[T::one(); 3]; w - 1];
    let mut out_points = vec![[T::one(); 3]; w - 1];
    let v = s.splat(T::one());
    let panics = |f: &mut dyn FnMut()| catch_unwind(AssertUnwindSafe(f)).is_err();
    assert!(
        panics(&mut || {
            let _ = s.load(&values);
        }),
        "{} load",
        S::ISA
    );
    assert!(panics(&mut || s.store(v, &mut out)), "{} store", S::ISA);
    assert!(
        panics(&mut || {
            let _ = s.load3(&points);
        }),
        "{} load3",
        S::ISA
    );
    assert!(
        panics(&mut || s.store3([v; 3], &mut out_points)),
        "{} store3",
        S::ISA
    );
}

#[test]
fn layer_arithmetic_matches_scalar() {
    on_every_isa!("layer_arithmetic_matches_scalar", f32, arithmetic);
    on_every_isa!("layer_arithmetic_matches_scalar", f64, arithmetic);
}

#[test]
fn layer_load_and_store_touch_one_vector() {
    on_every_isa!(
        "layer_load_and_store_touch_one_vector",
        f32,
        load_store_prefix
    );
    on_every_isa!(
        "layer_load_and_store_touch_one_vector",
        f64,
        load_store_prefix
    );
}

#[test]
fn layer_masks_match_scalar() {
    on_every_isa!("layer_masks_match_scalar", f32, masks);
    on_every_isa!("layer_masks_match_scalar", f64, masks);
}

#[test]
fn layer_inverse_square_roots() {
    on_every_isa!("layer_inverse_square_roots", f32, inverse_square_roots);
    on_every_isa!("layer_inverse_square_roots", f64, inverse_square_roots);
}

#[test]
fn layer_blocks_match_triples() {
    on_every_isa!("layer_blocks_match_triples", f32, blocks);
    on_every_isa!("layer_blocks_match_triples", f64, blocks);
}

#[test]
fn layer_rejects_short_slices() {
    on_every_isa!("layer_rejects_short_slices", f32, short_slices_panic);
    on_every_isa!("layer_rejects_short_slices", f64, short_slices_panic);
}

/// A random P2P problem: `n_s` sources and `n_t` targets in [−1, 1)³, charges and
/// initial outputs in [−1, 1).
struct P2pProblem<T> {
    sources: Vec<[T; 3]>,
    charges: Vec<T>,
    targets: Vec<[T; 3]>,
    potential: Vec<T>,
    gradient: Vec<[T; 3]>,
}

impl Rng {
    /// Uniform in [−1, 1), rounded to `T`.
    fn signed<T: SimdScalar>(&mut self) -> T {
        T::from_f64(2.0 * self.unit() - 1.0)
    }

    /// A random P2P problem.
    fn p2p_problem<T: SimdScalar>(&mut self, n_s: usize, n_t: usize) -> P2pProblem<T> {
        let point = |rng: &mut Rng| [0; 3].map(|_| rng.signed::<T>());
        P2pProblem {
            sources: (0..n_s).map(|_| point(self)).collect(),
            charges: (0..n_s).map(|_| self.signed()).collect(),
            targets: (0..n_t).map(|_| point(self)).collect(),
            potential: (0..n_t).map(|_| self.signed()).collect(),
            gradient: (0..n_t).map(|_| point(self)).collect(),
        }
    }
}

/// The generic body with K blocks on `s`, from the initial outputs of `p`; the bits of
/// the potential and, if `G`, the gradient.
fn body<T: SimdScalar, S: Simd<T>, const K: usize, const G: bool>(
    s: S,
    p: &P2pProblem<T>,
) -> (Vec<T>, Vec<[T; 3]>) {
    let mut potential = p.potential.clone();
    let mut gradient = if G { p.gradient.clone() } else { Vec::new() };
    p2p_body::<T, S, K, G>(
        s,
        &p.sources,
        &p.charges,
        &p.targets,
        &mut potential,
        &mut gradient,
    );
    (potential, gradient)
}

/// The bits of P2P outputs.
fn output_bits<T: SimdScalar>((potential, gradient): &(Vec<T>, Vec<[T; 3]>)) -> Vec<u64> {
    potential
        .iter()
        .chain(gradient.iter().flatten())
        .map(|&x| bits(x))
        .collect()
}

/// Every K (1, 2, 4) gives the same bits, at every n_t from 0 to 3 · 4 · W + 1: the
/// per-lane instruction sequence does not depend on K, which is what lets the tail of
/// a call run smaller blocks than the whole ones.
fn p2p_k_invariance<T: SimdScalar, S: Simd<T>>(s: S) {
    let mut rng = Rng(0x5eed_0006);
    for n_t in 0..=3 * 4 * S::W + 1 {
        let p = rng.p2p_problem::<T>(23, n_t);
        let k1 = output_bits(&body::<T, S, 1, false>(s, &p));
        assert_eq!(
            k1,
            output_bits(&body::<T, S, 2, false>(s, &p)),
            "{} K = 2",
            S::ISA
        );
        assert_eq!(
            k1,
            output_bits(&body::<T, S, 4, false>(s, &p)),
            "{} K = 4",
            S::ISA
        );
        let k1 = output_bits(&body::<T, S, 1, true>(s, &p));
        assert_eq!(
            k1,
            output_bits(&body::<T, S, 2, true>(s, &p)),
            "{} K = 2, gradient",
            S::ISA
        );
        assert_eq!(
            k1,
            output_bits(&body::<T, S, 4, true>(s, &p)),
            "{} K = 4, gradient",
            S::ISA
        );
    }
}

#[test]
fn p2p_body_is_independent_of_k() {
    on_every_isa!("p2p_body_is_independent_of_k", f32, p2p_k_invariance);
    on_every_isa!("p2p_body_is_independent_of_k", f64, p2p_k_invariance);
}

/// The generic body on the scalar layer (W = 1), which the scalar ISA does not run
/// (it runs the reference's loop): one pair within 8 u_T (potential) and 16 u_T
/// relative to |q| / r² (gradient) of the reference's term; sums of 64 sources within
/// 1e-6 (f32) or 1e-14 (f64) of the reference's relative to the term magnitudes. It
/// is not the reference bit for bit: it forms r² with fma and the gradient as
/// (q ρ) ρ² d.
fn p2p_body_at_w1<T: SimdScalar>() {
    let mut rng = Rng(0x5eed_0007);
    let u = unit_roundoff::<T>();
    let sum_tol = if u < 1e-10 { 1e-14 } else { 1e-6 };
    let mut differing = 0usize;
    for trial in 0..TRIALS {
        let p = rng.p2p_problem::<T>(if trial % 2 == 0 { 1 } else { 64 }, 3);
        let (potential, gradient) = body::<T, Scalar, 1, true>(Scalar, &p);
        let mut ref_potential = p.potential.clone();
        let mut ref_gradient = p.gradient.clone();
        nd_fmm_ref::p2p::p2p(
            &p.sources,
            &p.charges,
            &p.targets,
            &mut ref_potential,
            Some(&mut ref_gradient),
        );
        for (i, x) in p.targets.iter().enumerate() {
            let (mut sp, mut sg) = (p.potential[i].to_f64().abs(), 0.0f64);
            for g in p.gradient[i] {
                sg = sg.max(g.to_f64().abs());
            }
            for (y, &q) in p.sources.iter().zip(&p.charges) {
                let r2: f64 = (0..3)
                    .map(|k| (x[k].to_f64() - y[k].to_f64()).powi(2))
                    .sum();
                sp += q.to_f64().abs() / r2.sqrt();
                sg += q.to_f64().abs() / r2;
            }
            let (tol_p, tol_g) = if p.sources.len() == 1 {
                (8.0 * u, 16.0 * u)
            } else {
                (sum_tol, sum_tol)
            };
            let ep = (potential[i].to_f64() - ref_potential[i].to_f64()).abs() / sp;
            assert!(ep <= tol_p, "W = 1 potential: {ep:e}");
            for k in 0..3 {
                let eg = (gradient[i][k].to_f64() - ref_gradient[i][k].to_f64()).abs() / sg;
                assert!(eg <= tol_g, "W = 1 gradient: {eg:e}");
            }
            if bits(potential[i]) != bits(ref_potential[i]) {
                differing += 1;
            }
        }
    }
    println!(
        "p2p_body_at_w1 ({}): {differing} of {} potentials differ from the reference's bits",
        std::any::type_name::<T>(),
        3 * TRIALS
    );
}

#[test]
fn p2p_body_at_w1_is_within_tolerance() {
    println!("p2p_body_at_w1_is_within_tolerance: ISAs run: scalar (the layer at W = 1)");
    p2p_body_at_w1::<f32>();
    p2p_body_at_w1::<f64>();
}
