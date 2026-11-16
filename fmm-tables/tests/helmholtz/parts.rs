//! The parts of a table build on a caller's threads (Helmholtz CH4.0h): every table built
//! with a map that runs the parts out of order on several threads is the serial build's bit
//! for bit, compared through the applications (every octant's M2M and L2L, every offset's
//! M2L, on every level) and the sizes.

use nd_fmm_math::{Complex, Layout};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, OCTANT_COUNT};
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::ClassM2l;
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::plane_wave::merge::{PlaneWaveM2l, select};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables, TableChoice};
use nd_fmm_tables::parts::{Parts, Serial};

use super::{assert_same, input};

/// A map that computes the parts in reverse order, spread over three threads, and puts
/// them back in order: a build on it is the serial build's only if no part depends on
/// another, on the thread or on the order.
struct Threads;

impl Parts for Threads {
    fn map<R: Send>(&self, n: usize, f: &(dyn Fn(usize) -> R + Sync)) -> Vec<R> {
        let mut out: Vec<Option<R>> = (0..n).map(|_| None).collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|t| {
                    scope.spawn(move || {
                        (0..n)
                            .rev()
                            .filter(|i| i % 3 == t)
                            .map(|i| (i, f(i)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for handle in handles {
                for (i, r) in handle.join().expect("a part panicked") {
                    out[i] = Some(r);
                }
            }
        });
        out.into_iter()
            .map(|r| r.expect("every part computed"))
            .collect()
    }
}

/// k̂₀ as (Re, Im), the degrees and the M2L degrees.
type Case = ((f64, f64), [usize; 5], [usize; 5]);

/// k̂₀ as (Re, Im), the degrees and the levels with matrices.
type MatrixCase = ((f64, f64), [usize; 5], [bool; 5]);

/// The cases: above the projection threshold on level 2 (complex and real k), and below it;
/// the degrees given and the M2L degrees, raised on some levels.
pub(super) const CASES: [Case; 3] = [
    ((120.0, 6.0), [0, 0, 12, 10, 8], [0, 0, 14, 10, 9]),
    ((120.0, 0.0), [0, 0, 12, 10, 8], [0, 0, 12, 11, 8]),
    ((6.0, 0.5), [0, 0, 9, 8, 7], [0, 0, 9, 8, 7]),
];

/// Every application of `tables` on fixed inputs, one after the other: per level with a
/// child level the M2M and L2L of every octant, then per level from 2 the M2L of every
/// offset, two planes each.
pub(super) fn applications(tables: &HelmholtzTables<f64>) -> Vec<f64> {
    let mut scratch = tables.scratch();
    let mut out = Vec::new();
    let len = |l: usize| 2 * Layout::new(tables.m2l_degree(l)).len();
    for level in 0..tables.nlevels() {
        if level + 1 < tables.nlevels() {
            for o in 0..OCTANT_COUNT {
                let child = input(tables.m2l_degree(level + 1), 2, o as u64);
                let mut parent = vec![0.0; len(level)];
                tables.m2m(level, o, &child, &mut parent, 2, 1, &mut scratch);
                out.extend(parent);
                let parent = input(tables.m2l_degree(level), 2, 100 + o as u64);
                let mut child = vec![0.0; len(level + 1)];
                tables.l2l(level + 1, o, &parent, &mut child, 2, 1, &mut scratch);
                out.extend(child);
            }
        }
    }
    for level in 2..tables.nlevels() {
        out.extend(m2l_applications(tables, level));
    }
    out
}

/// The M2L of every offset on `level`, two planes each, on fixed inputs.
pub(super) fn m2l_applications(tables: &HelmholtzTables<f64>, level: usize) -> Vec<f64> {
    let mut scratch = tables.scratch();
    let q = tables.m2l_degree(level);
    let mut out = Vec::new();
    for d in 0..M2L_OFFSET_COUNT {
        let x = input(q, 2, 1000 + d as u64);
        let mut y = vec![0.0; 2 * Layout::new(q).len()];
        tables.m2l(level, d, &x, 2, &mut y, 1, &mut scratch);
        out.extend(y);
    }
    out
}

/// The matrices' cases: k̂₀, the degrees and the levels with matrices.
pub(super) const MATRIX_CASES: [MatrixCase; 2] = [
    (
        (6.0, 0.5),
        [0, 0, 9, 8, 7],
        [false, false, false, true, true],
    ),
    (
        (120.0, 0.0),
        [0, 0, 12, 10, 8],
        [false, false, true, false, true],
    ),
];

/// Every matrix of Dense, Batched and Classes on the chosen levels, one after the other.
pub(super) fn matrices(
    dense: &DenseM2l<f64>,
    batched: &BatchedM2l<f64>,
    classes: &ClassM2l<f64>,
    levels: &[bool],
) -> Vec<f64> {
    let mut out = Vec::new();
    for level in (0..levels.len()).filter(|&l| levels[l]) {
        for d in 0..M2L_OFFSET_COUNT {
            out.extend_from_slice(dense.matrix(level, d));
            let (a_r, a_i) = batched.planes(level, d);
            out.extend_from_slice(a_r);
            out.extend_from_slice(a_i);
        }
        for class in 0..16 {
            out.extend_from_slice(classes.matrix(level, class));
        }
    }
    out
}

/// The exponential M2L's tables of the k̂₀ and degrees, every level the rule gives a
/// quadrature at 10⁻⁶ and 10⁻⁸, as their `Debug` text (every value's shortest round trip).
pub(super) fn plane_wave_text(k_root: Complex<f64>, degrees: &[usize], epsilon: f64) -> String {
    let precisions = select(k_root, degrees, Some(epsilon));
    format!(
        "{:?}",
        PlaneWaveM2l::<f64>::build(k_root, degrees, &precisions)
    )
}

#[test]
fn the_tables_on_threads_are_the_serial_tables() {
    for ((re, im), p, q) in CASES {
        let k = Complex::new(re, im);
        for budget in [DEFAULT_BLOCK_BUDGET, 0] {
            let serial = HelmholtzTables::<f64>::build_with_m2l_degrees(k, &p, &q, budget);
            let choice = TableChoice {
                m2l_degrees: Some(&q),
                ..TableChoice::new(budget)
            };
            let threads = HelmholtzTables::<f64>::build_by(k, &p, &choice, &Threads);
            let what = format!("k̂₀ = {re} + {im}i, budget {budget}");
            assert_eq!(serial.bytes(), threads.bytes(), "{what}: bytes");
            assert!(threads.has_translations(), "{what}");
            for l in 0..p.len() {
                assert_eq!(
                    serial.m2l_blocks_stored(l),
                    threads.m2l_blocks_stored(l),
                    "{what}"
                );
                assert_eq!(threads.has_m2l(l), l >= 2, "{what}, level {l}");
            }
            assert_same(&applications(&threads), &applications(&serial), &what);
        }
    }
}

#[test]
fn the_m2l_tables_of_some_levels_alone() {
    for ((re, im), p, q) in CASES {
        let k = Complex::new(re, im);
        let full = HelmholtzTables::<f64>::build_with_m2l_degrees(k, &p, &q, DEFAULT_BLOCK_BUDGET);
        let levels = [false, false, true, false, true];
        let choice = TableChoice {
            m2l_degrees: Some(&q),
            translations: false,
            m2l_levels: Some(&levels),
            ..TableChoice::new(DEFAULT_BLOCK_BUDGET)
        };
        let some = HelmholtzTables::<f64>::build_by(k, &p, &choice, &Threads);
        assert!(!some.has_translations());
        assert!(some.bytes() < full.bytes());
        for l in 0..p.len() {
            assert_eq!(some.has_m2l(l), levels[l], "level {l}");
            assert_eq!(some.m2l_degree(l), q[l]);
            if levels[l] {
                assert_same(
                    &m2l_applications(&some, l),
                    &m2l_applications(&full, l),
                    &format!("k̂₀ = {re} + {im}i, level {l}"),
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "no M2L tables built on level 3")]
fn m2l_on_a_level_without_tables_is_refused() {
    let levels = [false, false, true, false];
    let choice = TableChoice {
        m2l_levels: Some(&levels),
        ..TableChoice::new(DEFAULT_BLOCK_BUDGET)
    };
    let tables =
        HelmholtzTables::<f64>::build_by(Complex::new(3.0, 0.0), &[0, 0, 4, 3], &choice, &Serial);
    m2l_applications(&tables, 3);
}

#[test]
#[should_panic(expected = "built without translations")]
fn m2m_without_translations_is_refused() {
    let choice = TableChoice {
        translations: false,
        ..TableChoice::new(DEFAULT_BLOCK_BUDGET)
    };
    let tables =
        HelmholtzTables::<f64>::build_by(Complex::new(3.0, 0.0), &[0, 0, 4, 3], &choice, &Serial);
    applications(&tables);
}

/// The blocks of the levels named alone: the others store none and spend none of the
/// budget, so that a coarser level fits; `set_m2l_blocks` gives the same blocks from any
/// earlier ones; the applications unchanged.
#[test]
fn the_m2l_blocks_of_some_levels_alone() {
    let ((re, im), p, q) = CASES[2];
    let k = Complex::new(re, im);
    let full = HelmholtzTables::<f64>::build(k, &p, DEFAULT_BLOCK_BUDGET);
    // A budget that holds the finest level's blocks only.
    let finest = full.m2l_polar_angles()
        * nd_fmm_math::orthonormal::rotation::compact_blocks_len(p[4])
        * size_of::<f64>();
    let only_finest = HelmholtzTables::<f64>::build(k, &p, finest);
    assert_eq!(
        (2..5)
            .map(|l| only_finest.m2l_blocks_stored(l))
            .collect::<Vec<_>>(),
        [false, false, true]
    );
    // Without the finest level the default budget stores levels 2 and 3.
    let levels = [false, false, true, true, false];
    let choice = TableChoice {
        block_levels: Some(&levels),
        ..TableChoice::new(DEFAULT_BLOCK_BUDGET)
    };
    let named = HelmholtzTables::<f64>::build_by(k, &p, &choice, &Threads);
    assert_eq!(
        (2..5)
            .map(|l| named.m2l_blocks_stored(l))
            .collect::<Vec<_>>(),
        [true, true, false]
    );
    assert_same(&applications(&named), &applications(&full), "named levels");
    assert_eq!(q, p);
    // `set_m2l_blocks` from every block stored, and from none, to the named levels.
    for mut tables in [full, HelmholtzTables::<f64>::build(k, &p, 0)] {
        tables.set_m2l_blocks(&levels, DEFAULT_BLOCK_BUDGET, &Threads);
        assert_eq!(tables.bytes(), named.bytes());
        for l in 0..p.len() {
            assert_eq!(tables.m2l_blocks_stored(l), named.m2l_blocks_stored(l));
        }
        assert_same(&applications(&tables), &applications(&named), "set");
    }
}

#[test]
fn the_matrices_on_threads_are_the_serial_matrices() {
    // The first case alone in debug builds (the second's p = 12 matrices take seconds).
    let cases = if cfg!(debug_assertions) {
        &MATRIX_CASES[..1]
    } else {
        &MATRIX_CASES[..]
    };
    for &((re, im), p, levels) in cases {
        let k = Complex::new(re, im);
        let serial = matrices(
            &DenseM2l::build(k, &p, &levels),
            &BatchedM2l::build(k, &p, &levels),
            &ClassM2l::build(k, &p, &levels),
            &levels,
        );
        let threads = matrices(
            &DenseM2l::build_on(k, &p, &levels, &Threads),
            &BatchedM2l::build_on(k, &p, &levels, &Threads),
            &ClassM2l::build_on(k, &p, &levels, &Threads),
            &levels,
        );
        assert_same(&threads, &serial, &format!("k̂₀ = {re} + {im}i"));
    }
}

#[test]
fn the_plane_wave_tables_on_threads_are_the_serial_tables() {
    let k = Complex::new(20.0, 2.0);
    let degrees = [0, 0, 20, 12, 9, 7];
    let precisions = select(k, &degrees, Some(1e-6));
    assert!(precisions.iter().filter(|p| p.is_some()).count() >= 2);
    let serial = PlaneWaveM2l::<f64>::build(k, &degrees, &precisions);
    let threads = PlaneWaveM2l::<f64>::build_with_reach_on(
        k,
        &degrees,
        &precisions,
        nd_fmm_tables::helmholtz::plane_wave::quadrature::RHO_MAX,
        &Threads,
    );
    assert_eq!(format!("{serial:?}"), format!("{threads:?}"));
}

/// FNV-1a over the values' bits.
fn hash(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Prints the applications' hash of each case, for a comparison of builds across versions
/// by hand (`--ignored --nocapture`).
#[test]
#[ignore = "prints the tables' hashes"]
fn print_the_hashes() {
    for ((re, im), p, q) in CASES {
        let tables = HelmholtzTables::<f64>::build_with_m2l_degrees(
            Complex::new(re, im),
            &p,
            &q,
            DEFAULT_BLOCK_BUDGET,
        );
        let values = applications(&tables);
        println!(
            "k̂₀ = {re} + {im}i: {:016x} ({} values)",
            hash(&values),
            values.len()
        );
        assert_same(&values, &values, "itself");
    }
    for ((re, im), p, levels) in MATRIX_CASES {
        let k = Complex::new(re, im);
        let values = matrices(
            &DenseM2l::build(k, &p, &levels),
            &BatchedM2l::build(k, &p, &levels),
            &ClassM2l::build(k, &p, &levels),
            &levels,
        );
        println!(
            "matrices k̂₀ = {re} + {im}i: {:016x} ({} values)",
            hash(&values),
            values.len()
        );
    }
    for (k, p) in [
        ((40.0, 0.0), [0, 0, 30, 20, 12, 9]),
        ((20.0, 2.0), [0, 0, 20, 12, 9, 7]),
    ] {
        for epsilon in [1e-6, 1e-8] {
            let text = plane_wave_text(Complex::new(k.0, k.1), &p, epsilon);
            let h = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
            println!(
                "plane waves k̂₀ = {} + {}i, ε = {epsilon:e}: {h:016x} ({} bytes)",
                k.0,
                k.1,
                text.len()
            );
        }
    }
}
