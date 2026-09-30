//! The coefficient transforms T_M(P) and T_L(P) (CONVENTIONS §3.12, "Coefficients
//! under improper P" and "The z-axis elements"): the action on the harmonics for all 48
//! elements, inverses and products, −I, and the structure of the z-axis elements.

use nd_fmm_math::Layout;
use nd_fmm_math::harmonics::{irregular, regular};
use nd_fmm_tables::symmetry::{CoefficientTransform, Expansion, SignedPermutation};

use crate::common::{Kind, P_MAX_M2L, SplitMix64, Worst, len, transform, weight};

/// K = diag(+1 for m ≥ 0, −1 for m < 0), conjugation in real storage (§3.11).
fn conjugate(p: usize, x: &mut [f64]) {
    let layout = Layout::new(p);
    for (i, v) in x.iter_mut().enumerate() {
        if layout.nm(i).1 < 0 {
            *v = -*v;
        }
    }
}

/// A random point with radius uniform in [lo, hi) and a direction uniform on the
/// sphere.
fn point_in_shell(rng: &mut SplitMix64, lo: f64, hi: f64) -> [f64; 3] {
    loop {
        let v = rng.in_cube();
        let r2: f64 = v.iter().map(|t| t * t).sum();
        if (1e-4..=1.0).contains(&r2) {
            let r = rng.range(lo, hi) / r2.sqrt();
            return v.map(|t| t * r);
        }
    }
}

/// The degree-relative error of Phase 0 / T5: per degree, the largest |got − want|
/// divided by the largest |want|, the worst over the degrees.
fn degree_relative_error(p: usize, got: &[f64], want: &[f64]) -> f64 {
    Layout::new(p)
        .degrees()
        .map(|(_, range)| {
            let scale = want[range.clone()]
                .iter()
                .fold(0.0_f64, |a, v| a.max(v.abs()));
            let err = got[range.clone()]
                .iter()
                .zip(&want[range])
                .fold(0.0_f64, |a, (g, w)| a.max((g - w).abs()));
            if scale > 0.0 { err / scale } else { err }
        })
        .fold(0.0, f64::max)
}

/// Tolerance of the harmonic identities, as in Phase 0 / T5 for p ≤ 20.
const HARMONIC_TOL: f64 = 1e-13;

/// Tolerance of T(P) T(Pᵀ) = I and of the homomorphism, in the orthonormal weighting.
const INVERSE_TOL: f64 = 1e-14;

/// Tolerance of the z-axis structure.
const Z_AXIS_TOL: f64 = 1e-15;

#[test]
fn multipole_transform_maps_regular_harmonics_for_all_48_elements() {
    // Rₙ(Px) = Dⁿ(P) Rₙ(x), in the coefficient form T_M(P) K Rₙ(x) = K Rₙ(Px), for all
    // 48 P (proper and improper), n ≤ 20, at 8 random x with 0.1 ≤ |x| ≤ 1.8
    // (CONVENTIONS §3.9). Px is exact, as P has entries 0 and ±1.
    //
    // Error measure: the degree-relative error of Phase 0 / T5, per degree the largest
    // |got − want| over the largest |want|, in real storage.
    let p = P_MAX_M2L;
    let mut rng = SplitMix64::new(0x5e70_0001);
    let worst = Worst::new("T_M(P) K R(x) vs K R(Px), n ≤ 20, degree-relative");
    let mut kr = vec![0.0; len(p)];
    let mut want = vec![0.0; len(p)];
    for e in SignedPermutation::all() {
        let t = CoefficientTransform::<f64>::multipole(p, e);
        for _ in 0..8 {
            let x = point_in_shell(&mut rng, 0.1, 1.8);
            regular(p, x, &mut kr);
            conjugate(p, &mut kr);
            regular(p, e.apply_f64(x), &mut want);
            conjugate(p, &mut want);
            let got = transform(&t, &kr);
            let err = worst.update(degree_relative_error(p, &got, &want));
            assert!(err <= HARMONIC_TOL, "g = {}: {err:.3e}", e.index());
        }
    }
}

#[test]
fn local_transform_maps_irregular_harmonics_for_all_48_elements() {
    // Iₙ(Px) = S Dⁿ(P) S⁻¹ Iₙ(x), in the coefficient form T_L(P) K Iₙ(x) = K Iₙ(Px),
    // for all 48 P, n ≤ 20, at 8 random x with 2 ≤ |x| ≤ 8 (CONVENTIONS §3.9).
    //
    // Error measure: the degree-relative error of Phase 0 / T5 in the orthonormal
    // weighting Nₘ/Sₘ, in which the transform is orthogonal (the raw measure is
    // ill-conditioned; see the Phase 0 rotation tests).
    let p = P_MAX_M2L;
    let layout = Layout::new(p);
    let mut rng = SplitMix64::new(0x5e70_0002);
    let worst = Worst::new("T_L(P) K I(x) vs K I(Px), n ≤ 20, degree-relative (Nₘ/Sₘ)");
    let mut ki = vec![0.0; len(p)];
    let mut want = vec![0.0; len(p)];
    for e in SignedPermutation::all() {
        let t = CoefficientTransform::<f64>::local(p, e);
        for _ in 0..8 {
            let x = point_in_shell(&mut rng, 2.0, 8.0);
            irregular(p, x, &mut ki);
            conjugate(p, &mut ki);
            irregular(p, e.apply_f64(x), &mut want);
            conjugate(p, &mut want);
            let mut got = transform(&t, &ki);
            for (i, (g, w)) in got.iter_mut().zip(&mut want).enumerate() {
                let (n, m) = layout.nm(i);
                *g *= weight(Kind::Local, n, m);
                *w *= weight(Kind::Local, n, m);
            }
            let err = worst.update(degree_relative_error(p, &got, &want));
            assert!(err <= HARMONIC_TOL, "g = {}: {err:.3e}", e.index());
        }
    }
}

/// The block product A B of two row-major blocks of order w.
fn block_product(a: &[f64], b: &[f64], w: usize) -> Vec<f64> {
    let mut c = vec![0.0; w * w];
    for i in 0..w {
        for k in 0..w {
            for j in 0..w {
                c[i * w + j] += a[i * w + k] * b[k * w + j];
            }
        }
    }
    c
}

/// The largest entry of W (A − B) W⁻¹ over the degree-n blocks, n ≤ p, with W the
/// orthonormal weight of `kind`.
fn weighted_distance(
    kind: Kind,
    p: usize,
    a: impl Fn(usize) -> Vec<f64>,
    b: impl Fn(usize) -> Vec<f64>,
) -> f64 {
    let mut worst: f64 = 0.0;
    for n in 0..=p {
        let w = 2 * n + 1;
        let (a, b) = (a(n), b(n));
        for i in 0..w {
            for j in 0..w {
                let (mi, mj) = (i as isize - n as isize, j as isize - n as isize);
                let scale = weight(kind, n, mi) / weight(kind, n, mj);
                worst = worst.max(scale * (a[i * w + j] - b[i * w + j]).abs());
            }
        }
    }
    worst
}

fn identity_block(n: usize) -> Vec<f64> {
    let w = 2 * n + 1;
    (0..w * w)
        .map(|k| if k % (w + 1) == 0 { 1.0 } else { 0.0 })
        .collect()
}

fn kind_of(expansion: Expansion) -> Kind {
    match expansion {
        Expansion::Multipole => Kind::Multipole,
        Expansion::Local => Kind::Local,
    }
}

#[test]
fn transform_times_transform_of_transpose_is_the_identity() {
    // T(P) T(Pᵀ) = I for all 48 P, both kinds, n ≤ 20 (CONVENTIONS §3.12).
    //
    // Error measure: the largest entry of W (T(P) T(Pᵀ) − I) W⁻¹, with W the
    // orthonormal weight (Nₘ for T_M, Nₘ/Sₘ for T_L), in which T is orthogonal.
    let p = P_MAX_M2L;
    let worst = Worst::new("T(P) T(Pᵀ) − I, n ≤ 20, orthonormal weighting");
    for expansion in [Expansion::Multipole, Expansion::Local] {
        for e in SignedPermutation::all() {
            let t = CoefficientTransform::<f64>::new(p, e, expansion);
            let inverse = CoefficientTransform::<f64>::new(p, e.transpose(), expansion);
            let err = worst.update(weighted_distance(
                kind_of(expansion),
                p,
                |n| block_product(t.block(n), inverse.block(n), 2 * n + 1),
                identity_block,
            ));
            assert!(
                err <= INVERSE_TOL,
                "{expansion:?}, g = {}: {err:.3e}",
                e.index()
            );
        }
    }
}

#[test]
fn transforms_are_homomorphisms() {
    // T(P Q) = T(P) T(Q) for all 48 × 48 pairs, both kinds, n ≤ 8 (CONVENTIONS §3.12).
    //
    // Error measure: as for the inverse, the largest entry of W (T(PQ) − T(P) T(Q)) W⁻¹.
    let p = 8;
    let worst = Worst::new("T(PQ) − T(P) T(Q), n ≤ 8, orthonormal weighting");
    for expansion in [Expansion::Multipole, Expansion::Local] {
        let all = SignedPermutation::all();
        let t: Vec<_> = all
            .iter()
            .map(|&e| CoefficientTransform::<f64>::new(p, e, expansion))
            .collect();
        for e in &all {
            for f in &all {
                let product = &t[e.compose(f).index()];
                let err = worst.update(weighted_distance(
                    kind_of(expansion),
                    p,
                    |n| product.block(n).to_vec(),
                    |n| block_product(t[e.index()].block(n), t[f.index()].block(n), 2 * n + 1),
                ));
                assert!(err <= INVERSE_TOL, "{expansion:?}: {err:.3e}");
            }
        }
    }
}

#[test]
fn inversion_gives_the_parity_exactly() {
    // T_M(−I) = T_L(−I) = diag((−1)ⁿ) (CONVENTIONS §3.12). Error measure: exact
    // equality of every entry (signed zeros compare equal).
    let p = P_MAX_M2L;
    let minus_identity = SignedPermutation::from_index(7);
    assert_eq!(minus_identity.apply([1, 2, 3]), [-1, -2, -3]);
    for expansion in [Expansion::Multipole, Expansion::Local] {
        let t = CoefficientTransform::<f64>::new(p, minus_identity, expansion);
        for n in 0..=p {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let want: Vec<f64> = identity_block(n).iter().map(|v| sign * v).collect();
            assert_eq!(t.block(n), want, "{expansion:?}, n = {n}");
        }
        // The identity is exact as well.
        let t = CoefficientTransform::<f64>::new(p, SignedPermutation::IDENTITY, expansion);
        for n in 0..=p {
            assert_eq!(t.block(n), identity_block(n), "{expansion:?}, n = {n}");
        }
    }
}

/// How a block of degree n acts on slot 0 and on each slot pair (+m, −m), m ≥ 1, if it
/// is a signed permutation within tolerance: `Some(false)` for a pair mapped diagonally,
/// `Some(true)` for one whose two slots are exchanged, `None` otherwise.
fn pair_structure(block: &[f64], n: usize) -> Vec<Option<bool>> {
    let w = 2 * n + 1;
    let at = |m: isize, k: isize| block[(m + n as isize) as usize * w + (k + n as isize) as usize];
    let unit = |v: f64| (v.abs() - 1.0).abs() <= Z_AXIS_TOL;
    let zero = |v: f64| v.abs() <= Z_AXIS_TOL;
    let row_ok = |m: isize, k: isize| {
        (-(n as isize)..=n as isize).all(|j| {
            if j == k {
                unit(at(m, j))
            } else {
                zero(at(m, j))
            }
        })
    };
    let mut out = vec![if row_ok(0, 0) { Some(false) } else { None }];
    for m in 1..=n as isize {
        out.push(if row_ok(m, m) && row_ok(-m, -m) {
            Some(false)
        } else if row_ok(m, -m) && row_ok(-m, m) {
            Some(true)
        } else {
            None
        });
    }
    out
}

#[test]
fn z_axis_elements_are_signed_slot_permutations() {
    // CONVENTIONS §3.12, "The z-axis elements": for the 16 elements with P e_z = ±e_z,
    // T_L(P) = T_M(P); both are signed permutations mapping slot 0 to ± itself and each
    // pair (+m, −m) to itself; for g < 8 they are diagonal, for 16 ≤ g < 24 they
    // exchange the slots of the pairs with odd m and are diagonal on those with even m.
    //
    // Error measure: absolute, every entry within 1e-15 of 0 or ±1 and of the other
    // kind's entry. The test also reports whether the entries are exactly 0 and ±1.
    let p = P_MAX_M2L;
    let mut exact = true;
    let mut worst: f64 = 0.0;
    let z_axis: Vec<SignedPermutation> = SignedPermutation::all()
        .into_iter()
        .filter(|e| e.permutation()[2] == 2)
        .collect();
    assert_eq!(
        z_axis.iter().map(|e| e.index()).collect::<Vec<_>>(),
        (0..8).chain(16..24).collect::<Vec<_>>()
    );
    for e in z_axis {
        let g = e.index();
        let m = CoefficientTransform::<f64>::multipole(p, e);
        let l = CoefficientTransform::<f64>::local(p, e);
        for (a, b) in m.blocks().iter().zip(l.blocks()) {
            worst = worst.max((a - b).abs());
            assert!((a - b).abs() <= Z_AXIS_TOL, "g = {g}: T_M {a} vs T_L {b}");
            exact &= [0.0, 1.0, -1.0].contains(a) && a == b;
        }
        for n in 0..=p {
            let structure = pair_structure(m.block(n), n);
            for (k, s) in structure.iter().enumerate() {
                let swap = g >= 16 && k % 2 == 1;
                assert_eq!(*s, Some(swap), "g = {g}, n = {n}, m = {k}");
            }
        }
    }
    eprintln!(
        "z-axis elements: entries exactly 0 and ±1 with T_L = T_M: {exact}; \
         worst |T_M − T_L| {worst:.3e}"
    );
}

#[test]
fn apply_accumulates_and_checks_lengths() {
    // Error measure: exact equality. T(I) is the identity, so applying it to x adds x.
    let p = 4;
    let t = CoefficientTransform::<f64>::multipole(p, SignedPermutation::IDENTITY);
    assert_eq!(t.p(), p);
    assert_eq!(t.blocks().len(), 1 + 9 + 25 + 49 + 81);
    let x: Vec<f64> = (0..len(p)).map(|i| i as f64 + 0.5).collect();
    let mut y: Vec<f64> = (0..len(p)).map(|i| -(i as f64)).collect();
    t.apply(&x, &mut y);
    assert!(y.iter().all(|&v| v == 0.5));
    // cast rounds every entry.
    let single =
        CoefficientTransform::<f64>::local(p, SignedPermutation::from_index(29)).cast::<f32>();
    let double = CoefficientTransform::<f64>::local(p, SignedPermutation::from_index(29));
    for (s, d) in single.blocks().iter().zip(double.blocks()) {
        assert_eq!(s.to_bits(), (*d as f32).to_bits());
    }
    assert_eq!(
        CoefficientTransform::<f32>::local(p, SignedPermutation::from_index(29)),
        single
    );
}

#[test]
#[should_panic(expected = "`out` must have length (p + 1)^2 = 16")]
fn apply_rejects_wrong_output_length() {
    let t = CoefficientTransform::<f64>::local(3, SignedPermutation::from_index(5));
    t.apply(&[0.0; 16], &mut [0.0; 9]);
}

/// Extension used by the tests: P applied to a real point, exactly.
trait ApplyF64 {
    fn apply_f64(&self, x: [f64; 3]) -> [f64; 3];
}

impl ApplyF64 for SignedPermutation {
    fn apply_f64(&self, x: [f64; 3]) -> [f64; 3] {
        let (s, pi) = (self.signs(), self.permutation());
        core::array::from_fn(|a| s[a] as f64 * x[pi[a]])
    }
}
