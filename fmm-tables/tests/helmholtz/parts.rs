//! The parts of a table build on a caller's threads (Helmholtz CH4.0h): every table built
//! with a map that runs the parts out of order on several threads is the serial build's bit
//! for bit, compared through the applications (every octant's M2M and L2L, every offset's
//! M2L, on every level) and the sizes; and the exponential M2L's tables in parts within a
//! level (Helmholtz CH4.0j), against the serial build and against each level formed from
//! the public steps one by one.

use nd_fmm_math::{Complex, Layout};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, OCTANT_COUNT};
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::ClassM2l;
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use std::time::Instant;

use nd_fmm_tables::helmholtz::plane_wave::merge::{
    MergeLevel, PlaneWaveM2l, plane_wave_degree, select,
};
use nd_fmm_tables::helmholtz::plane_wave::quadrature::RHO_MAX;
use nd_fmm_tables::helmholtz::plane_wave::{PlaneWaveLevel, Precision, Quadrature};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables, TableChoice};
use nd_fmm_tables::parts::{Parts, Serial};
use nd_fmm_tables::plane_wave::geometry::{
    CHILDREN, Direction, MergeGeometry, child_bits, slot_coordinates,
};

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

/// k̂₀ as (Re, Im), the degrees given, the tolerance and the reach.
type PlaneWaveCase = ((f64, f64), [usize; 6], f64, f64);

/// The exponential M2L's cases: k̂₀ (real and complex), the degrees given, the tolerance
/// and the reach. 10⁻³ takes iprec 1 at the level's degree, 10⁻⁶ iprec 2, 10⁻⁸ iprec 3 at
/// the raised degree max(p_l, 32) (`plane_wave_degree`), as the operator builds them.
const PLANE_WAVE_CASES: [PlaneWaveCase; 5] = [
    ((20.0, 2.0), [0, 0, 20, 12, 9, 7], 1e-6, RHO_MAX),
    ((40.0, 0.0), [0, 0, 30, 20, 12, 9], 1e-3, RHO_MAX),
    ((40.0, 0.0), [0, 0, 30, 20, 12, 9], 1e-8, RHO_MAX),
    ((6.0, 0.5), [0, 0, 9, 8, 7, 6], 1e-6, 4.0),
    ((60.0, 0.0), [0, 0, 40, 25, 15, 10], 1e-3, 4.0),
];

/// The precisions and the degrees the operator builds a case's tables with.
fn plane_wave_inputs(
    k: Complex<f64>,
    degrees: &[usize],
    epsilon: f64,
) -> (Vec<Option<Precision>>, Vec<usize>) {
    let precisions = select(k, degrees, Some(epsilon));
    let raised = degrees
        .iter()
        .zip(&precisions)
        .map(|(&p, q)| q.map_or(p, |_| plane_wave_degree(p, epsilon)))
        .collect();
    (precisions, raised)
}

#[test]
fn the_plane_wave_tables_on_threads_are_the_serial_tables() {
    for ((re, im), degrees, epsilon, reach) in PLANE_WAVE_CASES {
        let k = Complex::new(re, im);
        let (precisions, degrees) = plane_wave_inputs(k, &degrees, epsilon);
        assert!(
            precisions.iter().filter(|p| p.is_some()).count() >= 2,
            "k̂₀ = {k}, ε = {epsilon:e}: two levels with waves"
        );
        let serial = PlaneWaveM2l::<f64>::build_with_reach(k, &degrees, &precisions, reach);
        let threads =
            PlaneWaveM2l::<f64>::build_with_reach_on(k, &degrees, &precisions, reach, &Threads);
        assert_eq!(
            format!("{serial:?}"),
            format!("{threads:?}"),
            "k̂₀ = {k}, ε = {epsilon:e}, reach {reach}"
        );
    }
}

/// k̂_l = k̂₀ 2⁻ˡ, as `HelmholtzTables` and the merge's tables form it.
fn level_k_hat(k: Complex<f64>, level: usize) -> Complex<f64> {
    k * (0.5f64).powi(level as i32)
}

/// Every level of a build in parts (Helmholtz CH4.0j) against the level formed from the
/// public steps one by one, independently of the parts: the quadrature, the
/// `PlaneWaveLevel` (its `Debug` text) and every slot's source shift and every child shift,
/// each `shift_factors` of its offset, bit for bit.
#[test]
fn every_level_in_parts_is_the_level_by_its_steps() {
    let geometry = MergeGeometry::new();
    for ((re, im), degrees, epsilon, reach) in PLANE_WAVE_CASES {
        let k = Complex::new(re, im);
        let (precisions, degrees) = plane_wave_inputs(k, &degrees, epsilon);
        let m2l =
            PlaneWaveM2l::<f64>::build_with_reach_on(k, &degrees, &precisions, reach, &Threads);
        for (l, &precision) in precisions.iter().enumerate() {
            let name = format!("k̂₀ = {k}, ε = {epsilon:e}, reach {reach}, level {l}");
            let Some(precision) = precision else {
                assert!(m2l.level(l).is_none(), "{name}: no tables");
                continue;
            };
            let level = m2l.level(l).expect("a level with a precision has tables");
            let k_hat = level_k_hat(k, l);
            let quadrature = Quadrature::with_reach(k_hat * 2.0, precision, reach)
                .expect("the rule names levels in the tables");
            let tables = PlaneWaveLevel::<f64>::new(k_hat, degrees[l], &quadrature);
            assert_eq!(
                format!("{:?}", level.tables()),
                format!("{tables:?}"),
                "{name}: the level's tables"
            );
            assert_eq!(level.precision(), precision, "{name}");
            let mut factors = vec![0.0; tables.wave_len()];
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            for direction in Direction::ALL {
                for accumulator in geometry.accumulators(direction) {
                    for &slot in &accumulator.slots {
                        let a = slot_coordinates(slot as usize);
                        let d = direction.rotate(a.map(|x| (5 - 2 * x) as f64 / 2.0));
                        tables.shift_factors(d, direction.is_up(), &mut factors);
                        let shift = level
                            .source_shift(direction, slot as usize)
                            .expect("a slot of an accumulator has a shift");
                        assert_eq!(
                            bits(shift),
                            bits(&factors),
                            "{name}: {direction:?}, slot {slot}"
                        );
                    }
                }
                for child in 0..CHILDREN {
                    let b = child_bits(child);
                    let d = direction.rotate(b.map(|x| (2 * x - 1) as f64 / 2.0));
                    tables.shift_factors(d, direction.is_up(), &mut factors);
                    assert_eq!(
                        bits(level.child_shift(direction, child)),
                        bits(&factors),
                        "{name}: {direction:?}, child {child}"
                    );
                }
            }
        }
    }
}

/// The least of five runs of `f`, in milliseconds.
fn best_ms(mut f: impl FnMut()) -> f64 {
    (0..5)
        .map(|_| {
            let clock = Instant::now();
            f();
            clock.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

/// Times one level's parts of the exponential M2L's tables (Helmholtz CH4.0j) on one thread,
/// over the range of κ = 2k̂ and p the levels with waves take, at iprec 1–3: the quadrature
/// (`Quadrature::with_reach`, its counts most of it), the `PlaneWaveLevel`, the 152 shift
/// tables (the level less the two) and the level (`MergeLevel::with_reach`), with the parts
/// of each map (nodes; nodes and frames; shifts). The time on many threads is the FMM's, on
/// its pool: `nd-fmm-validate`'s `helmholtz_accuracy`, the build's part "plane-wave tables"
/// (this crate has no thread pool, and a map that starts threads per call times their start).
/// Prints a table; asserts no time (`--ignored --nocapture`, release).
#[test]
#[ignore = "prints a level's build by part (release, by hand on the machines)"]
fn time_a_levels_parts() {
    let geometry = MergeGeometry::new();
    println!(
        "| iprec | κ | p | nodes | waves | quadrature ms | level ms | shifts ms | in all ms \
         | parts (nodes; nodes + frames; shifts) |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
    for precision in [Precision::Iprec1, Precision::Iprec2, Precision::Iprec3] {
        for kappa in [0.5, 2.0, 6.0, 12.0, 25.0, 45.0] {
            for p in [8, 16, 32] {
                let k_hat = Complex::new(kappa / 2.0, 0.0);
                let quadrature = Quadrature::with_reach(k_hat * 2.0, precision, RHO_MAX)
                    .expect("κ in the tables");
                let quad = best_ms(|| {
                    std::hint::black_box(Quadrature::with_reach(k_hat * 2.0, precision, RHO_MAX));
                });
                let level = best_ms(|| {
                    std::hint::black_box(PlaneWaveLevel::<f64>::new(k_hat, p, &quadrature));
                });
                let all = best_ms(|| {
                    std::hint::black_box(MergeLevel::<f64>::with_reach(
                        k_hat, p, precision, &geometry, RHO_MAX,
                    ));
                });
                let merge = MergeLevel::<f64>::with_reach(k_hat, p, precision, &geometry, RHO_MAX)
                    .expect("κ in the tables");
                let s = quadrature.nodes();
                println!(
                    "| {} | {kappa} | {p} | {s} | {} | {quad:.3} | {level:.3} | {:.3} | {all:.3} \
                     | {s}; {}; {} |",
                    precision.iprec(),
                    quadrature.wave_count(),
                    all - quad - level,
                    s + 2,
                    merge.source_shift_count() + 6 * CHILDREN,
                );
            }
        }
    }
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
