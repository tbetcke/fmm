//! The 16-class M2L form against the dense T4 tables: expanded and applied without
//! expanding, the identity M2L(P d₀) = T_L(P) M2L(d₀) T_M(Pᵀ) for every P, f32,
//! determinism, and the large-p checks with the memory and timings of the report.

use std::time::Instant;

use nd_fmm_math::rotation::blocks_len;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offset_index};
use nd_fmm_tables::m2l::build_matrices;
use nd_fmm_tables::symmetry::{
    CoefficientTransform, M2L_CLASS_COUNT, SignedPermutation, class_of, class_representatives,
};
use nd_fmm_tables::{L2lTables, M2lClasses, M2lScratch, M2lTables, M2mTables, MatrixSet};

use crate::common::{
    CLASS_TOL, DEBUG_DEGREES, Kind, P_DEBUG, P_MAX_F32, P_MAX_M2L, SplitMix64, Worst, apply,
    block_error, classes, degree_error, dense, len, product, random_coefficients, terms, transform,
    transform_terms,
};
use crate::octants::check_octants;

/// Bit patterns of every entry.
fn bits<T: nd_fmm_math::RealScalar>(x: &[T]) -> Vec<u64> {
    x.iter()
        .map(|&v| nd_fmm_math::RealScalar::to_f64(v).to_bits())
        .collect()
}

/// The terms |T_L(P)| (|A| (|T_M(Pᵀ)| |x|)) of the class form applied to `x` at
/// `index`.
fn class_terms(classes: &M2lClasses<f64>, index: usize, x: &[f64]) -> Vec<f64> {
    let e = classes.element(index);
    let t1 = transform_terms(classes.multipole_transform(e.transpose()), x);
    let t2 = terms(classes.matrices(), classes.class(index), &t1);
    transform_terms(classes.local_transform(e), &t2)
}

/// Compares every matrix of `expanded` with the dense matrix at the same index, block by
/// block, relative to the terms of the class form (`block_error`); returns the worst
/// error.
fn check_expansion(
    classes: &M2lClasses<f64>,
    expanded: &M2lTables<f64>,
    dense: &MatrixSet<f64>,
) -> f64 {
    let p = classes.p();
    let mut worst: f64 = 0.0;
    for index in 0..M2L_OFFSET_COUNT {
        let e = classes.element(index);
        let (_, terms) = product(
            classes.local_transform(e),
            classes.matrices(),
            classes.class(index),
            classes.multipole_transform(e.transpose()),
        );
        let got = expanded.matrices().matrix(index);
        let err = block_error(
            Kind::Local,
            Kind::Multipole,
            p,
            got,
            dense.matrix(index),
            &terms,
        );
        assert!(err <= CLASS_TOL, "p = {p}, index {index}: {err:.3e}");
        worst = worst.max(err);
    }
    worst
}

#[test]
fn class_matrices_are_the_dense_matrices_of_the_representatives() {
    // Error measure: bit-for-bit equality with the T4 matrices, and exact equality of
    // the class and element of every offset with `class_of`.
    for p in [0, 3, P_DEBUG] {
        let c = classes(p);
        assert_eq!(c.p(), p);
        assert_eq!(
            (c.matrices().n(), c.matrices().count()),
            (len(p), M2L_CLASS_COUNT)
        );
        for (class, rep) in class_representatives().into_iter().enumerate() {
            let index = m2l_offset_index(rep).unwrap();
            assert_eq!(
                bits(c.matrices().matrix(class)),
                bits(dense(p).matrices().matrix(index))
            );
        }
        for index in 0..M2L_OFFSET_COUNT {
            assert_eq!((c.class(index), c.element(index)), class_of(index));
        }
        let d = [-3, 1, 2];
        assert_eq!(c.index(d), m2l_offset_index(d));
        for e in SignedPermutation::all() {
            assert_eq!(
                c.multipole_transform(e),
                &CoefficientTransform::multipole(p, e)
            );
            assert_eq!(c.local_transform(e), &CoefficientTransform::local(p, e));
        }
    }
}

#[test]
fn expand_reproduces_the_dense_tables() {
    // The C2.2 symmetry criterion: expand() reproduces all 316 dense T4 matrices, p ≤ 8.
    //
    // Error measure: per block of output and input degree, in the orthonormal weighting
    // (Nₘ/Sₘ for the output, Nₘ for the input), relative to the terms
    // |T_L(P)| |A| |T_M(Pᵀ)| of the three products (`block_error`). The representatives
    // are bit-identical (their element is I).
    let worst = Worst::new("expand() vs dense M2L, p ≤ 8, per block (terms)");
    for p in DEBUG_DEGREES {
        let c = classes(p);
        let expanded = c.expand();
        assert_eq!(expanded.p(), p);
        worst.update(check_expansion(c, &expanded, dense(p).matrices()));
        for rep in class_representatives() {
            let index = m2l_offset_index(rep).unwrap();
            assert_eq!(
                bits(expanded.matrices().matrix(index)),
                bits(dense(p).matrices().matrix(index))
            );
        }
    }
}

#[test]
fn apply_equals_the_dense_application() {
    // apply() without expanding, all 316 offsets, three random inputs each, p ≤ 8.
    //
    // Error measure: local coefficients per degree in the orthonormal weighting
    // Nₘ/Sₘ, relative to the terms |T_L(P)| (|A| (|T_M(Pᵀ)| |x|)) of the three
    // products.
    let mut rng = SplitMix64::new(0x5e70_0101);
    let worst = Worst::new("class apply vs dense apply, p ≤ 8, per degree (terms)");
    for p in DEBUG_DEGREES {
        let c = classes(p);
        let mut scratch = M2lScratch::new(p);
        for index in 0..M2L_OFFSET_COUNT {
            for _ in 0..3 {
                let x = random_coefficients(Kind::Multipole, p, &mut rng);
                let mut got = vec![0.0; len(p)];
                c.apply(index, &x, &mut got, &mut scratch);
                let want = apply(dense(p).matrices(), index, &x);
                let e = worst.update(degree_error(
                    Kind::Local,
                    p,
                    &got,
                    &want,
                    &class_terms(c, index, &x),
                ));
                assert!(e <= CLASS_TOL, "p = {p}, index {index}: {e:.3e}");
            }
        }
    }
}

/// Checks M2L(P d₀) = T_L(P) M2L(d₀) T_M(Pᵀ) for the representative of every class and
/// the elements `elements`, applied to random inputs, against the dense matrices of the
/// offsets P d₀: `dense` maps a table index to its position in a matrix set. Returns
/// the worst error.
fn check_identity<'a>(
    classes: &M2lClasses<f64>,
    elements: &[SignedPermutation],
    dense: impl Fn(usize) -> (usize, &'a MatrixSet<f64>),
    rng: &mut SplitMix64,
) -> f64 {
    let p = classes.p();
    let mut worst: f64 = 0.0;
    for (class, rep) in class_representatives().into_iter().enumerate() {
        for &e in elements {
            let index = m2l_offset_index(e.apply(rep)).unwrap();
            let (position, set) = dense(index);
            let x = random_coefficients(Kind::Multipole, p, rng);
            let rotated = transform(classes.multipole_transform(e.transpose()), &x);
            let got = transform(
                classes.local_transform(e),
                &apply(classes.matrices(), class, &rotated),
            );
            let tau = transform_terms(
                classes.local_transform(e),
                &terms(
                    classes.matrices(),
                    class,
                    &transform_terms(classes.multipole_transform(e.transpose()), &x),
                ),
            );
            let want = apply(set, position, &x);
            let err = degree_error(Kind::Local, p, &got, &want, &tau);
            assert!(
                err <= CLASS_TOL,
                "p = {p}, class {class}, g = {}: {err:.3e}",
                e.index()
            );
            worst = worst.max(err);
        }
    }
    worst
}

#[test]
fn identity_holds_for_every_group_element() {
    // CONVENTIONS §3.12, "Operator identities", for all 48 P and not only the element
    // that class_of picks: every class representative d₀, every P, p ≤ 8.
    //
    // Error measure: as for apply, per degree relative to the terms of the three
    // products.
    let mut rng = SplitMix64::new(0x5e70_0102);
    let worst =
        Worst::new("T_L(P) M2L(d₀) T_M(Pᵀ) vs M2L(P d₀), all 48 P, p ≤ 8, per degree (terms)");
    for p in DEBUG_DEGREES {
        let all = SignedPermutation::all();
        worst.update(check_identity(
            classes(p),
            &all,
            |i| (i, dense(p).matrices()),
            &mut rng,
        ));
    }
}

#[test]
fn class_apply_accumulates_and_uses_no_state() {
    // Error measure: bit-for-bit equality. apply adds to a nonzero output exactly what
    // it gives from zero (plus rounding of that one sum), and a reused scratch gives the
    // same result as a fresh one.
    let p = 5;
    let c = classes(p);
    let mut rng = SplitMix64::new(0x5e70_0103);
    let x = random_coefficients(Kind::Multipole, p, &mut rng);
    let y0 = random_coefficients(Kind::Local, p, &mut rng);
    let mut scratch = M2lScratch::new(p);
    for index in [0, 17, 158, 315] {
        let mut from_zero = vec![0.0; len(p)];
        c.apply(index, &x, &mut from_zero, &mut scratch);
        let mut again = vec![0.0; len(p)];
        c.apply(index, &x, &mut again, &mut M2lScratch::new(p));
        assert_eq!(bits(&from_zero), bits(&again));
        let mut acc = y0.clone();
        c.apply(index, &x, &mut acc, &mut scratch);
        let want: Vec<f64> = y0.iter().zip(&from_zero).map(|(a, b)| a + b).collect();
        let err = degree_error(Kind::Local, p, &acc, &want, &class_terms(c, index, &x));
        assert!(err <= 1e-15, "index {index}: {err:.3e}");
    }
}

#[test]
#[should_panic(expected = "`scratch` must be of degree p = 3")]
fn class_apply_rejects_scratch_of_another_degree() {
    let c = classes(3);
    c.apply(0, &[0.0; 16], &mut [0.0; 16], &mut M2lScratch::new(2));
}

#[test]
fn f32_class_form_expanded_in_f32_matches_f64() {
    // The class form cast to f32 (equal to the one built in f32), expanded in f32,
    // against the dense f64 tables, p ≤ 8.
    //
    // Error measure: per block of output and input degree, as for the f64 expansion,
    // relative to the terms of the f64 class form's three products.
    const F32_TOL: f64 = 1e-5;
    let worst = Worst::new("f32 expand() vs dense f64 M2L, p ≤ 8, per block (terms)");
    for p in DEBUG_DEGREES.into_iter().filter(|&p| p <= P_MAX_F32) {
        let c = classes(p);
        let single = c.cast::<f32>();
        let built = M2lClasses::<f32>::build(p);
        assert_eq!(
            bits(single.matrices().as_slice()),
            bits(built.matrices().as_slice())
        );
        for e in SignedPermutation::all() {
            assert_eq!(single.multipole_transform(e), built.multipole_transform(e));
            assert_eq!(single.local_transform(e), built.local_transform(e));
        }
        let expanded = single.expand();
        for index in 0..M2L_OFFSET_COUNT {
            let e = c.element(index);
            let (_, tau) = product(
                c.local_transform(e),
                c.matrices(),
                c.class(index),
                c.multipole_transform(e.transpose()),
            );
            let got: Vec<f64> = expanded
                .matrices()
                .matrix(index)
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let err = worst.update(block_error(
                Kind::Local,
                Kind::Multipole,
                p,
                &got,
                dense(p).matrices().matrix(index),
                &tau,
            ));
            assert!(err <= F32_TOL, "p = {p}, index {index}: {err:.3e}");
        }
    }
}

#[test]
fn building_twice_is_bit_identical() {
    // Error measure: bit-for-bit equality of the class matrices and every transform,
    // and exact equality of the classes and elements; in f64 and f32.
    for p in [0, 3, 6] {
        let (a, b) = (M2lClasses::<f64>::build(p), M2lClasses::<f64>::build(p));
        assert_eq!(bits(a.matrices().as_slice()), bits(b.matrices().as_slice()));
        for e in SignedPermutation::all() {
            assert_eq!(
                bits(a.multipole_transform(e).blocks()),
                bits(b.multipole_transform(e).blocks())
            );
            assert_eq!(
                bits(a.local_transform(e).blocks()),
                bits(b.local_transform(e).blocks())
            );
        }
        for index in 0..M2L_OFFSET_COUNT {
            assert_eq!(
                (a.class(index), a.element(index)),
                (b.class(index), b.element(index))
            );
        }
        assert_eq!(
            bits(a.expand().matrices().as_slice()),
            bits(b.expand().matrices().as_slice())
        );
    }
    let (a, b) = (M2lClasses::<f32>::build(4), M2lClasses::<f32>::build(4));
    assert_eq!(bits(a.matrices().as_slice()), bits(b.matrices().as_slice()));
}

/// Storage of the class form in bytes: the class matrices and the 96 transforms.
fn class_bytes<T: nd_fmm_math::RealScalar>(c: &M2lClasses<T>) -> usize {
    let transforms: usize = SignedPermutation::all()
        .iter()
        .map(|&e| c.multipole_transform(e).blocks().len() + c.local_transform(e).blocks().len())
        .sum();
    assert_eq!(transforms, 96 * blocks_len(c.p()));
    core::mem::size_of_val(c.matrices().as_slice()) + transforms * core::mem::size_of::<T>()
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-tables --release -- --ignored`"]
fn class_form_at_large_p() {
    // As the debug tests, at large p:
    // - expand() against the dense T4 tables, all 316 offsets, p = 16 and 20;
    // - apply() for the 16 representatives with three group elements each at p = 20
    //   (proper, improper and a z-axis element), against the dense matrices of those
    //   offsets, and the identity with those elements;
    // - the octant identities at p = 20.
    // Error measures as in the debug tests. Also reports the storage of the dense and
    // class forms at p = 8, 16 and 20, the build times of both and the time of
    // expand().
    let mut rng = SplitMix64::new(0x5e70_0201);
    for p in [8, 16, P_MAX_M2L] {
        let start = Instant::now();
        let c = M2lClasses::<f64>::build(p);
        let class_time = start.elapsed();
        let start = Instant::now();
        let expanded = c.expand();
        let expand_time = start.elapsed();
        let start = Instant::now();
        let dense = M2lTables::<f64>::build(p);
        let dense_time = start.elapsed();
        let dense_bytes = core::mem::size_of_val(dense.matrices().as_slice());
        let bytes = class_bytes(&c);
        let matrix_bytes = core::mem::size_of_val(c.matrices().as_slice());
        eprintln!(
            "p = {p}: dense {:.1} MB built in {dense_time:.3?}; class form {:.2} MB \
             ({:.2} MB matrices + {:.2} MB transforms) built in {class_time:.3?}; \
             expand() {expand_time:.3?}",
            dense_bytes as f64 / 1e6,
            bytes as f64 / 1e6,
            matrix_bytes as f64 / 1e6,
            (bytes - matrix_bytes) as f64 / 1e6,
        );
        if p >= 16 {
            let worst = Worst::new(if p == 16 {
                "expand() vs dense M2L, p = 16, per block (terms)"
            } else {
                "expand() vs dense M2L, p = 20, per block (terms)"
            });
            worst.update(check_expansion(&c, &expanded, dense.matrices()));
        }
        drop(expanded);
        drop(dense);
        if p == P_MAX_M2L {
            // g = 29 is proper, g = 43 improper, neither maps z to ±z; g = 19 is an
            // improper z-axis element.
            let elements = [29, 43, 19].map(SignedPermutation::from_index);
            assert!(
                elements[0].is_proper() && !elements[1].is_proper() && !elements[2].is_proper()
            );
            assert!(elements[..2].iter().all(|e| e.permutation()[2] != 2));
            assert_eq!(elements[2].permutation()[2], 2);
            let offsets: Vec<usize> = class_representatives()
                .iter()
                .flat_map(|&rep| elements.map(|e| m2l_offset_index(e.apply(rep)).unwrap()))
                .collect();
            let subset = build_matrices::<f64>(p, &offsets);
            let position = |index: usize| offsets.iter().position(|&i| i == index).unwrap();

            let worst = Worst::new(
                "class apply vs dense apply, p = 20, 16 × 3 offsets, per degree (terms)",
            );
            let mut scratch = M2lScratch::new(p);
            for &index in &offsets {
                for _ in 0..3 {
                    let x = random_coefficients(Kind::Multipole, p, &mut rng);
                    let mut got = vec![0.0; len(p)];
                    c.apply(index, &x, &mut got, &mut scratch);
                    let want = apply(&subset, position(index), &x);
                    let e = worst.update(degree_error(
                        Kind::Local,
                        p,
                        &got,
                        &want,
                        &class_terms(&c, index, &x),
                    ));
                    assert!(e <= CLASS_TOL, "index {index}: {e:.3e}");
                }
            }
            let worst = Worst::new(
                "T_L(P) M2L(d₀) T_M(Pᵀ) vs M2L(P d₀), p = 20, 3 elements, per degree (terms)",
            );
            worst.update(check_identity(
                &c,
                &elements,
                |i| (position(i), &subset),
                &mut rng,
            ));

            let (m, l) = check_octants(p, &M2mTables::build(p), &L2lTables::build(p));
            eprintln!(
                "worst octant identities at p = 20, per block (terms): M2M {m:.3e}, L2L {l:.3e}"
            );
        }
    }
}
