//! The accuracy harness of the Helmholtz FMM (Helmholtz CH3.1c,
//! `nd_fmm_validate::helmholtz_accuracy`) on small cases: the checksum independent of the
//! partition and sensitive to one bit, the blocks a partition, the sharded oracle the
//! unsharded one bit for bit at any thread count, and the FMM within ε of the oracle with
//! the same checksum at 1 and 2 threads and, on several ranks, on one rank. A smoke run of
//! the calibration's plateau scan (Helmholtz CH3.3, `nd_fmm_validate::helmholtz_calibration`):
//! the errors fall with the degree, the smallest degree at ε/2 meets it, and the default
//! rule's degrees are at most S7's at the levels' circumradii. The FMM3D comparison's
//! measures (Helmholtz CH3.6, `nd_fmm_validate::helmholtz_fmm3d`) on a stand-in for FMM3D,
//! the direct sum at every point in the driver's files: its sampled errors zero, our
//! FMM's full-target error its error at the samples (which are every point), the same on
//! one rank and on several. The M2L rule per level (Helmholtz CH3.2d): `Auto` at a named
//! crossover reported in the row, within ε at 1 and 2 threads and on one rank; below every
//! degree Rotation's checksum, above every degree Batched on every level with tables. The
//! exponential M2L (Helmholtz CH6.1e): under the precision rule within
//! [`PLANE_WAVE_RATIO`] of Rotation's errors on small cases of CH3.1c's matrix, each level's
//! precision in the row, within ε at 1 and 2 threads and on one rank, and Rotation's output
//! below the rule's tolerance.
//!
//! One test initialises MPI (at `Funneled`); it passes on any number of ranks, each check
//! on every rank or on none.

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use nd_fmm_exec::helmholtz::{
    ChargeKind, Degrees, Densities, HelmholtzM2l, Output, PlaneWavePrecision, PlaneWaveSplit,
    Wavenumber, WavenumberKind, auto_m2l_mul_add, s7_degree,
};
use nd_fmm_exec::operator::Isa;
use nd_fmm_tables::helmholtz::plane_wave::merge::precision_for_tolerance;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::RHO_MAX;
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{
    Case, Oracle, Problem, Row, checksum, complex_k, run, run_output, sweep_crossovers,
};
use nd_fmm_validate::helmholtz_calibration::Curve;
use nd_fmm_validate::helmholtz_fmm3d::{
    Grid, Values, crosscheck, crosscheck_values, full_bytes, parse_full, parse_sampled,
    sample_errors, sampled_bytes,
};
use nd_fmm_validate::helmholtz_workloads::Workload;

/// An output of `n` targets, 2 complex vectors with gradients, filled with random values.
fn random_output(n: usize, rng: &mut SplitMix64) -> Output<f64> {
    let densities = Densities::new(WavenumberKind::Real, ChargeKind::Complex, 2);
    let mut output = Output::new(&densities, n, true);
    for v in &mut output.potential {
        *v = rng.range(-1.0, 1.0);
    }
    for g in output.gradient.as_mut().expect("gradients") {
        *g = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
    }
    output
}

/// Targets `range` of `output` as an output of their own.
fn part(output: &Output<f64>, range: std::ops::Range<usize>) -> Output<f64> {
    let densities = Densities::new(WavenumberKind::Real, ChargeKind::Complex, 2);
    let n = output.ntargets();
    let mut part = Output::new(&densities, range.len(), true);
    let columns = output.potential.len() / n;
    for o in 0..columns {
        let (from, to) = (o * n, o * range.len());
        part.potential[to..to + range.len()]
            .copy_from_slice(&output.potential[from + range.start..from + range.end]);
        part.gradient.as_mut().expect("gradients")[to..to + range.len()].copy_from_slice(
            &output.gradient.as_ref().expect("gradients")[from + range.start..from + range.end],
        );
    }
    part
}

/// The checksum, the blocks and the oracle (no FMM).
fn check_harness(world: &SimpleCommunicator) {
    let mut rng = SplitMix64::new(11);
    let output = random_output(37, &mut rng);
    let whole = checksum(&output, 0);
    let split =
        checksum(&part(&output, 0..20), 0).wrapping_add(checksum(&part(&output, 20..37), 20));
    assert_eq!(split, whole, "the checksum depends on the partition");
    // The same values at shifted positions, and one changed bit, change it.
    assert_ne!(checksum(&output, 1), whole);
    let mut changed = output.clone();
    changed.gradient.as_mut().expect("gradients")[40][2] =
        f64::from_bits(changed.gradient.as_ref().expect("gradients")[40][2].to_bits() ^ 1);
    assert_ne!(checksum(&changed, 0), whole);

    let case = Case::new(
        Workload::W4 {
            distribution: Distribution::Plummer,
            n: 500,
        },
        (10.0, 2.0),
        Degrees::Tolerance(1e-3),
    );
    let problem = Problem::new(&case);
    assert_eq!(problem.sample.len(), 500, "at most N samples");
    let mut sorted = problem.sample.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 500, "distinct samples");
    for size in [1, 3, 7] {
        let blocks: Vec<_> = (0..size).map(|r| problem.block(r, size)).collect();
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[size - 1].end, 500);
        for pair in blocks.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert!(pair[0].len().abs_diff(pair[1].len()) <= 1);
        }
    }
    let k = complex_k(&case);
    let all: Vec<usize> = (0..problem.sample.len()).collect();
    let serial = Oracle::of_samples(k, &problem, &all, true, 1);
    assert_eq!(Oracle::of_samples(k, &problem, &all, true, 3), serial);
    assert_eq!(Oracle::sharded(k, &problem, true, 2, world), serial);
}

/// The per-level columns of a row (Helmholtz CH6.2b): an entry per level, no V pairs on
/// levels 0 and 1, no M2L time on level 0 (the downward pass calls M2L from level 1), a
/// time on every level with V pairs; the exponential M2L's two passes (Helmholtz CH6.2d)
/// timed on every level with waves and V pairs, each within the level's M2L time, and zero
/// on the others.
fn check_levels(row: &Row) {
    assert_eq!(row.m2l_seconds.len(), row.levels);
    assert_eq!(row.plane_wave_seconds.len(), row.levels);
    assert_eq!(row.plane_wave_parts.len(), row.levels);
    for (l, &[waves, parents]) in row.plane_wave_seconds.iter().enumerate() {
        let [busy, shifts, children, conversions] = row.plane_wave_parts[l];
        if row.plane_wave[l].is_some() && row.v_pairs[l] > 0 {
            assert!(
                waves > 0.0 && parents > 0.0,
                "level {l}: passes {waves} s, {parents} s"
            );
            // The parts (CH6.2f), each rank's within its busy time, the slowest of each.
            assert!(
                busy > 0.0 && conversions > 0.0 && shifts.max(children).max(conversions) <= busy,
                "level {l}: parts {:?}",
                row.plane_wave_parts[l]
            );
            assert!(
                waves <= row.m2l_seconds[l] && parents <= row.m2l_seconds[l],
                "level {l}: passes {waves} s, {parents} s within {} s",
                row.m2l_seconds[l]
            );
        } else {
            assert_eq!(
                [waves, parents],
                [0.0, 0.0],
                "level {l}: no merge and shift"
            );
            assert_eq!(row.plane_wave_parts[l], [0.0; 4], "level {l}: no parts");
        }
    }
    assert_eq!(row.v_pairs.len(), row.levels);
    assert_eq!(&row.v_pairs[..2], &[0, 0], "no V pairs on levels 0 and 1");
    assert_eq!(row.m2l_seconds[0], 0.0, "no M2L on level 0");
    assert!(
        row.v_pairs.iter().any(|&n| n > 0),
        "V pairs: {:?}",
        row.v_pairs
    );
    for (l, (&seconds, &pairs)) in row.m2l_seconds.iter().zip(&row.v_pairs).enumerate() {
        assert!(
            pairs == 0 || seconds > 0.0,
            "level {l}: {pairs} V pairs in {seconds} s"
        );
    }
}

/// The FMM within ε, at 1 and 2 threads with one checksum, and on several ranks the
/// checksum of one rank; the per-level columns, the same V pairs on one rank, and two
/// evaluations of one build the same checksum.
fn check_fmm(world: &SimpleCommunicator, case: &Case) {
    let problem = Problem::new(case);
    let oracle = Oracle::sharded(complex_k(case), &problem, case.gradients, 2, world);
    let rows: Vec<_> = [1, 2]
        .into_iter()
        .map(|threads| run(case, &problem, &oracle, threads, world).expect("the FMM runs"))
        .collect();
    let Degrees::Tolerance(eps) = case.degrees else {
        panic!("a tolerance")
    };
    for row in &rows {
        for (v, errors) in row.errors.iter().enumerate() {
            assert!(
                errors[0] <= eps && errors[1] <= eps,
                "{:?} k = {:?}, vector {v}: errors {errors:?} above ε = {eps}",
                case.workload,
                case.k,
            );
        }
        assert_eq!(row.errors.len(), case.vectors + 1);
        check_levels(row);
        assert_eq!(row.first_evaluate, row.evaluate, "one evaluation");
    }
    assert_eq!(
        rows[0].checksum, rows[1].checksum,
        "threads change the output"
    );
    let twice = Case {
        evaluations: 2,
        ..case.clone()
    };
    let row = run(&twice, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(row.checksum, rows[0].checksum, "the second evaluation");
    assert_eq!(row.errors, rows[0].errors);
    assert!(row.first_evaluate > 0.0);
    check_levels(&row);
    if world.size() > 1 {
        let one = SimpleCommunicator::self_comm();
        let row = run(case, &problem, &oracle, 1, &one).expect("the one-rank FMM runs");
        assert_eq!(row.checksum, rows[0].checksum, "ranks change the output");
        assert_eq!(row.v_pairs, rows[0].v_pairs, "the V pairs of one rank");
        // The same output; the error sums are reduced in another order (relative 1e-12).
        for (one, many) in row.errors.iter().zip(&rows[0].errors) {
            for (a, b) in one.iter().zip(many) {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs(),
                    "errors {one:?} and {many:?}"
                );
            }
        }
    }
}

/// Auto on W1 at kD = 10: at a crossover below every degree the row reports it and the
/// output is Rotation's bit for bit (the checksum); above every degree every level with M2L
/// runs Batched; at either, `check_fmm`'s checks.
fn check_auto(world: &SimpleCommunicator) {
    let mut case = Case::new(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    let problem = Problem::new(&case);
    let oracle = Oracle::sharded(complex_k(&case), &problem, case.gradients, 2, world);
    let rotation = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(rotation.m2l_max_degree, None);
    let smallest = rotation.degrees[2..]
        .iter()
        .min()
        .copied()
        .expect("levels with M2L");
    case.m2l = HelmholtzM2l::Auto;
    case.m2l_max_degree = Some(smallest - 1);
    let below = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(below.m2l, HelmholtzM2l::Auto);
    assert_eq!(below.m2l_max_degree, Some(smallest - 1));
    assert!(!below.dense_levels.contains(&true));
    assert_eq!(below.checksum, rotation.checksum, "Auto below every degree");
    check_fmm(world, &case);
    case.m2l_max_degree = Some(usize::MAX);
    let every = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
    assert!(every.dense_levels[2..].iter().all(|&on| on));
    assert_eq!(every.m2l_mul_add, auto_m2l_mul_add(Isa::detect()));
    check_fmm(world, &case);
    // Above every degree: Batched in Auto's mode, bit for bit.
    let mut batched = case.clone();
    (batched.m2l, batched.m2l_mul_add, batched.m2l_max_degree) =
        (HelmholtzM2l::Batched, Some(every.m2l_mul_add), None);
    let fused = run(&batched, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(every.checksum, fused.checksum, "Auto above every degree");
    // The sweep's crossovers: every choice once, from every level to none.
    let sweep = sweep_crossovers(&rotation.degrees);
    assert_eq!(sweep.first(), Some(&usize::MAX));
    assert_eq!(sweep.last(), Some(&(smallest - 1)));
    assert_eq!(
        sweep_crossovers(&[0, 0, 30, 18, 11, 7]),
        [usize::MAX, 18, 11, 7, 6]
    );
    assert_eq!(sweep_crossovers(&[0, 0, 15, 15, 15]), [usize::MAX, 14]);
    assert_eq!(sweep_crossovers(&[0, 0, 12, 16, 8]), [usize::MAX, 12, 8, 7]);
    assert_eq!(sweep_crossovers(&[0, 0, 0]), [usize::MAX]);
    assert!(sweep_crossovers(&[0, 0]).is_empty());
}

/// The largest ratio of the exponential M2L's errors to Rotation's under the precision rule
/// (CH6.1's acceptance, "within about 1.1× Rotation's").
const PLANE_WAVE_RATIO: f64 = 1.1;

/// The exponential M2L under the precision rule on small cases of CH3.1c's matrix (W1 at
/// kD = 10 at ε = 10⁻³, 10⁻⁶ and 10⁻⁸; W4's clusters at N = 2,000, complex k, real
/// charges, ε = 10⁻³): φ and ∇φ within [`PLANE_WAVE_RATIO`] of Rotation's errors, the row's
/// precisions the rule's (`precision_for_tolerance` of ε on every level with waves), and
/// named precisions as named; at ε = 10⁻⁶ also `check_fmm`'s checks. Below the rule's
/// tolerance (ε = 10⁻⁸) no level has waves and the output is Rotation's (the checksum). The
/// counts' reach (Helmholtz CH6.2e): the corner's by default, with waves exactly on the
/// levels with a precision; at reach 4 at most the corner's waves per level and fewer in all,
/// and on W1 within [`PLANE_WAVE_RATIO`] of Rotation's errors. The split by child
/// (Helmholtz CH6.2f) at 2 threads the default's checksum.
fn check_plane_wave(world: &SimpleCommunicator) {
    let clusters = Workload::W4 {
        distribution: Distribution::Clusters,
        n: 2000,
    };
    for (workload, k, eps) in [
        (Workload::W1, (10.0, 0.0), 1e-3),
        (Workload::W1, (10.0, 0.0), 1e-6),
        (Workload::W1, (10.0, 0.0), 1e-8),
        (clusters, (10.0, 1.0), 1e-3),
    ] {
        let mut case = Case::new(workload, k, Degrees::Tolerance(eps));
        if let Workload::W4 { .. } = workload {
            case.charges = ChargeKind::Real;
        }
        let problem = Problem::new(&case);
        let oracle = Oracle::sharded(complex_k(&case), &problem, case.gradients, 2, world);
        let rotation = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
        assert!(rotation.plane_wave.iter().all(Option::is_none));
        case.m2l = HelmholtzM2l::PlaneWave;
        let row = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
        assert_eq!(row.m2l, HelmholtzM2l::PlaneWave);
        check_levels(&row);
        for (l, precision) in row.plane_wave.iter().enumerate() {
            assert_eq!(row.dense_levels[l], precision.is_some(), "level {l}");
        }
        let Some(want) = precision_for_tolerance(eps) else {
            // Below the rule's tolerance: Rotation on every level, bit for bit.
            assert!(row.plane_wave.iter().all(Option::is_none), "ε = {eps}");
            assert_eq!(
                row.checksum, rotation.checksum,
                "ε = {eps}: Rotation's output"
            );
            continue;
        };
        assert!(
            row.plane_wave.iter().any(Option::is_some),
            "{workload:?} at ε = {eps}: a level with waves, {:?}",
            row.plane_wave
        );
        assert!(
            row.plane_wave.iter().flatten().all(|&q| q == want),
            "{workload:?} at ε = {eps}: the rule's precision, {:?}",
            row.plane_wave
        );
        let (all, reference) = (row.errors.last().unwrap(), rotation.errors.last().unwrap());
        for j in 0..2 {
            assert!(
                all[j] <= PLANE_WAVE_RATIO * reference[j],
                "{workload:?} at ε = {eps}, precisions {:?}: errors {all:?} against \
                 Rotation's {reference:?}",
                row.plane_wave
            );
        }
        if eps == 1e-6 {
            check_fmm(world, &case);
        }
        // The counts' reach (Helmholtz CH6.2e): the corner's by default; at 4 at most the
        // corner's waves on every level with waves, fewer in all, within PLANE_WAVE_RATIO of
        // Rotation's errors on W1.
        assert_eq!(row.plane_wave_reach, RHO_MAX);
        for (l, &w) in row.plane_wave_waves.iter().enumerate() {
            assert_eq!(w > 0, row.plane_wave[l].is_some(), "level {l}");
        }
        case.plane_wave_reach = Some(4.0);
        let reach = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
        case.plane_wave_reach = None;
        assert_eq!(reach.plane_wave_reach, 4.0);
        assert_eq!(reach.plane_wave, row.plane_wave);
        assert!(
            reach
                .plane_wave_waves
                .iter()
                .zip(&row.plane_wave_waves)
                .all(|(a, b)| a <= b)
                && reach.plane_wave_waves.iter().sum::<usize>()
                    < row.plane_wave_waves.iter().sum::<usize>(),
            "{workload:?} at ε = {eps}: waves at reach 4 {:?}, at the corner {:?}",
            reach.plane_wave_waves,
            row.plane_wave_waves
        );
        if workload == Workload::W1 {
            let all = reach.errors.last().unwrap();
            for j in 0..2 {
                assert!(
                    all[j] <= PLANE_WAVE_RATIO * reference[j],
                    "{workload:?} at ε = {eps}, reach 4: errors {all:?} against Rotation's \
                     {reference:?}"
                );
            }
        }
        // The split by child (CH6.2f): the default's output at 2 threads.
        assert_eq!(row.plane_wave_split, PlaneWaveSplit::Auto);
        case.plane_wave_split = Some(PlaneWaveSplit::Children);
        let children = run(&case, &problem, &oracle, 2, world).expect("the FMM runs");
        case.plane_wave_split = None;
        assert_eq!(children.plane_wave_split, PlaneWaveSplit::Children);
        assert_eq!(
            children.checksum, row.checksum,
            "{workload:?} at ε = {eps}: the split by child"
        );
        // Named: one precision on the finest level alone.
        let finest = row.levels - 1;
        let mut named = vec![None; row.levels];
        named[finest] = row.plane_wave[finest];
        case.plane_wave = PlaneWavePrecision::PerLevel(named.clone());
        let one = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
        assert_eq!(one.plane_wave, named);
    }
}

/// The plateau scan of the calibration on W1 at kD = 10: one degree p = 1..=8 on every
/// level ≥ 2, as the `helmholtz_calibrate` example runs it.
fn check_calibration_scan(world: &SimpleCommunicator) {
    let mut case = Case::new(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    let problem = Problem::new(&case);
    let oracle = Oracle::sharded(complex_k(&case), &problem, true, 2, world);
    let default = run(&case, &problem, &oracle, 2, world).expect("the FMM runs");
    // The circumradii are √3 r_l of the run's domain, halving per level.
    let k = Wavenumber::new(10.0, 0.0).unwrap();
    assert_eq!(default.circumradii.len(), default.levels);
    for l in 1..default.levels {
        assert_eq!(default.circumradii[l], default.circumradii[l - 1] / 2.0);
    }
    // The default rule never above S7's.
    for (l, &p) in default.degrees.iter().enumerate().skip(2) {
        assert!(
            p <= s7_degree(&k, default.circumradii[l], 1e-3),
            "level {l}: {p}"
        );
    }
    let mut curve = Curve::new(1);
    while curve.next_degree() <= 8 {
        let mut degrees = vec![curve.next_degree(); 64];
        degrees[..2].fill(0);
        case.degrees = Degrees::PerLevel(degrees);
        let row = run(&case, &problem, &oracle, 2, world).expect("the FMM runs");
        curve.push(*row.errors.last().unwrap());
    }
    let largest = |e: &[f64; 2]| e[0].max(e[1]);
    assert!(
        largest(&curve.errors[7]) < 1e-3 * largest(&curve.errors[0]),
        "{curve:?}"
    );
    let p = curve.smallest_degree(0.5e-3).expect("p ≤ 8 meets 5e-4");
    for e in &curve.errors[p - 1..] {
        assert!(largest(e) <= 0.5e-3, "{curve:?}");
    }
}

/// The FMM3D comparison's measures on W1 at kD = 10 with every point sampled, the
/// direct sum standing in for FMM3D's output (module documentation).
fn check_fmm3d_measures(world: &SimpleCommunicator) {
    let mut case = Case::new(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    case.samples = usize::MAX;
    let problem = Problem::new(&case);
    assert_eq!(problem.sample.len(), problem.len());
    let oracle = Oracle::sharded(complex_k(&case), &problem, true, 2, world);
    // The direct sum at every point, in point order (the samples are a permutation).
    let zero = nd_fmm_math::Complex::new(0.0, 0.0);
    let mut direct = Values {
        potential: vec![zero; problem.len()],
        gradient: vec![[zero; 3]; problem.len()],
    };
    for (s, &i) in problem.sample.iter().enumerate() {
        direct.potential[i] = oracle.potential[0][s];
        direct.gradient[i] = oracle.gradient[0][s];
    }
    let path = std::path::Path::new("stand-in");
    let full = parse_full(path, &full_bytes(&direct)).unwrap();
    let sampled = parse_sampled(path, &sampled_bytes(&full.at(&problem.sample))).unwrap();
    assert_eq!(full, direct);
    assert_eq!(sample_errors(&sampled, &oracle, 0), [0.0, 0.0]);

    let ours = run_output(&case, &problem, &oracle, 2, world).expect("the FMM runs");
    let grid = Grid {
        domain: ours.domain,
        level: ours.row.levels - 1,
    };
    let block = problem.block(world.rank() as usize, world.size() as usize);
    let check = crosscheck(&ours.output, 0, &problem, &block, &full, &grid, world);
    let at_samples = ours.row.errors[0];
    for j in 0..2 {
        assert!(
            (check.errors[j] - at_samples[j]).abs() <= 1e-12 * at_samples[j],
            "{:?} against {at_samples:?}",
            check.errors
        );
        assert!(check.errors[j] > 0.0 && check.errors[j] <= 1e-3);
    }
    // One entry per box that holds a point; the worst at least the ℓ2 error.
    let mut occupied: Vec<u64> = problem.points.iter().map(|&x| grid.box_of(x)).collect();
    occupied.sort_unstable();
    occupied.dedup();
    let keys: Vec<u64> = check.boxes.iter().map(|&(k, _)| k).collect();
    assert_eq!(keys, occupied);
    let worst = check.worst();
    assert!(worst[0] >= check.errors[0] && worst[1] >= check.errors[1]);
    assert_eq!(
        crosscheck_values(&full, &full, &problem.points, &grid).worst(),
        [0.0, 0.0]
    );
    if world.size() > 1 {
        let one = SimpleCommunicator::self_comm();
        let single = run_output(&case, &problem, &oracle, 2, &one).expect("the one-rank FMM runs");
        let all = 0..problem.len();
        let alone = crosscheck(&single.output, 0, &problem, &all, &full, &grid, &one);
        assert_eq!(alone.boxes, check.boxes, "ranks change the box errors");
        for j in 0..2 {
            assert!((alone.errors[j] - check.errors[j]).abs() <= 1e-12 * alone.errors[j]);
        }
    }
}

#[test]
fn helmholtz_accuracy_harness() {
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let world = universe.world();
    check_harness(&world);

    // W1 at kD = 10 (796 points), two complex vectors with gradients.
    let mut w1 = Case::new(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    w1.vectors = 2;
    check_fmm(&world, &w1);
    // W4's clusters, complex k, real charges, no gradients.
    let mut w4 = Case::new(
        Workload::W4 {
            distribution: Distribution::Clusters,
            n: 2000,
        },
        (10.0, 1.0),
        Degrees::Tolerance(1e-3),
    );
    w4.charges = ChargeKind::Real;
    w4.gradients = false;
    check_fmm(&world, &w4);
    check_auto(&world);
    check_plane_wave(&world);
    check_calibration_scan(&world);
    check_fmm3d_measures(&world);
}
