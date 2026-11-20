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
//! degree Rotation's checksum, above every degree Batched on every level with tables (the
//! term of the levels' V lists off, Helmholtz CH4.0; with it on, reported, and the levels'
//! boxes, estimate and visits consistent with their pairs). The
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
    ChargeKind, Degrees, Densities, HelmholtzDiagonal, HelmholtzM2l, LevelShape, Output,
    PlaneWavePrecision, PlaneWaveRaise, PlaneWaveSplit, RcrSplit, TransferBuild, VNear, Wavenumber,
    WavenumberKind, WxNear, auto_m2l_mul_add, s7_degree,
};
use nd_fmm_exec::operator::Isa;
use nd_fmm_tables::helmholtz::plane_wave::merge::{plane_wave_degree, precision_for_tolerance};
use nd_fmm_tables::helmholtz::plane_wave::quadrature::RHO_MAX;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{
    BUILD_PARTS, Case, Density, FORM_COSTS, Oracle, PLANE_WAVE_DIRECTION, Problem, RCR_COSTS, Row,
    checksum, complex_k, run, run_output, sweep_crossovers,
};
use nd_fmm_validate::helmholtz_boxes::{Errors, Reference, box_sample};
use nd_fmm_validate::helmholtz_calibration::Curve;
use nd_fmm_validate::helmholtz_fmm3d::{
    Grid, Values, crosscheck, crosscheck_values, full_bytes, parse_full, parse_sampled,
    sample_errors, sampled_bytes,
};
use nd_fmm_validate::helmholtz_workloads::Workload;
use nd_fmm_validate::{SplitMix64, points};

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

    let case = case(
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

/// `Case::new` on the tree before Helmholtz CH4.0d3's defaults (64 points per leaf, D11's
/// c_leaf), on which these checks were written; `check_leaves` checks the library's.
fn case(workload: Workload, k: (f64, f64), degrees: Degrees) -> Case {
    Case {
        max_points_per_leaf: Some(64),
        c_leaf: Some(nd_fmm_exec::helmholtz::C_LEAF),
        ..Case::new(workload, k, degrees)
    }
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
    // The diagonal form's scratch (Helmholtz CH4.1b): targets with a V row and their sources
    // on a level with V pairs, a source per pair at most, every rank within its budget on
    // these small trees, all of a rank's targets in one chunk.
    assert_eq!(row.signature_scratch.len(), row.levels);
    for (l, s) in row.signature_scratch.iter().enumerate() {
        if row.v_pairs[l] == 0 {
            assert_eq!((s.targets, s.sources, s.bytes), (0, 0, 0), "level {l}");
            continue;
        }
        assert!(
            s.targets > 0 && s.targets <= row.v_targets[l],
            "level {l}: {s:?}"
        );
        assert!(
            s.sources > 0 && s.sources <= row.v_pairs[l],
            "level {l}: {s:?}"
        );
        assert_eq!(s.over_budget, 0, "level {l}: {s:?}");
        assert!(
            s.source_bytes < s.bytes && s.bytes <= s.budget,
            "level {l}: {s:?}"
        );
        assert!(s.chunk > 0 && s.chunk <= s.targets, "level {l}: {s:?}");
        assert_eq!(s.chunks, 1, "level {l}: {s:?}");
    }
    // The shapes (Helmholtz CH4.0): a box per target, a visit for at most 64 pairs (a block
    // of 64 targets) and at least one pair, m̂ from the shape.
    assert_eq!(row.v_targets.len(), row.levels);
    assert_eq!(row.block_visits.len(), row.levels);
    assert_eq!(row.batched_columns.len(), row.levels);
    assert_eq!(row.v_targets[0], 1, "the root");
    for l in 0..row.levels {
        let (pairs, visits) = (row.v_pairs[l], row.block_visits[l]);
        assert!(
            visits <= pairs && pairs <= 64 * visits,
            "level {l}: {pairs} pairs in {visits} visits"
        );
        assert!(
            pairs <= 189 * row.v_targets[l],
            "level {l}: V rows of at most 189"
        );
        assert_eq!(
            row.batched_columns[l],
            LevelShape {
                pairs: pairs as u64,
                targets: row.v_targets[l] as u64
            }
            .batched_columns()
        );
    }
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
/// runs Batched; at either, `check_fmm`'s checks. With the exponential M2L (Helmholtz
/// CH6.2h) from degree 0 and no Batched level, the exponential M2L's checksum, the agreed
/// degree reported and the errors within [`PLANE_WAVE_RATIO`] of Rotation's; off, none.
fn check_auto(world: &SimpleCommunicator) {
    let mut case = case(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
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
    // CH3.2d's rule: the exponential M2L, the default since D8, named off (with it below).
    assert!(case.auto_plane_wave);
    case.auto_plane_wave = false;
    // And without the term of the levels' V lists (Helmholtz CH4.0), so that every level
    // runs Batched above every degree; with it, below.
    assert!(case.batched_term);
    case.batched_term = false;
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
    assert_eq!(every.m2l_plane_wave_min_degree, None, "off");
    assert!(!every.batched_term);
    // The term on: reported, and at a named block width and two threads the same output.
    let mut term = case.clone();
    term.batched_term = true;
    let on = run(&term, &problem, &oracle, 1, world).expect("the FMM runs");
    assert!(on.batched_term);
    term.m2l_block_targets = Some(3);
    let narrow = run(&term, &problem, &oracle, 2, world).expect("the FMM runs");
    assert_eq!(narrow.m2l_block_targets, Some(3));
    assert_eq!(narrow.checksum, on.checksum, "blocks of 3 at 2 threads");
    // With waves from degree 0 and no Batched level: PlaneWave's output.
    let mut plane_wave = case.clone();
    (plane_wave.m2l, plane_wave.m2l_max_degree) = (HelmholtzM2l::PlaneWave, None);
    let plane_wave = run(&plane_wave, &problem, &oracle, 1, world).expect("the FMM runs");
    let mut waves = case.clone();
    waves.m2l_max_degree = Some(0);
    waves.auto_plane_wave = true;
    waves.m2l_plane_wave_min_degree = Some(0);
    let row = run(&waves, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(row.m2l_plane_wave_min_degree, Some(0));
    assert_eq!(row.plane_wave, plane_wave.plane_wave);
    assert!(row.plane_wave.iter().any(Option::is_some));
    assert_eq!(row.checksum, plane_wave.checksum, "Auto with every wave");
    check_levels(&row);
    let ([phi, grad], [rotation_phi, rotation_grad]) = (
        *row.errors.last().expect("errors"),
        *rotation.errors.last().expect("errors"),
    );
    assert!(
        phi <= PLANE_WAVE_RATIO * rotation_phi && grad <= PLANE_WAVE_RATIO * rotation_grad,
        "Auto with waves: {phi:e}, {grad:e} against Rotation's {rotation_phi:e}, \
         {rotation_grad:e}"
    );
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
/// named precisions as named; at ε = 10⁻⁶ also `check_fmm`'s checks. At ε = 10⁻⁸ the rule
/// takes iprec 3 at the resolving degree (Helmholtz CH6.1g): each level with waves at
/// max(p_l, 32), the others at Rotation's degrees, the translations at Rotation's (the
/// default `PlaneWaveRaise::Rule`, `Leaves` there; Helmholtz CH6.1h); `Level` within
/// [`PLANE_WAVE_RATIO`] of Rotation's errors, `M2l` printed. The counts' reach (Helmholtz CH6.2e): the corner's by default, with waves exactly on the
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
        let mut case = case(workload, k, Degrees::Tolerance(eps));
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
        let want = precision_for_tolerance(eps).expect("a precision at every ε of the list");
        for (l, precision) in row.plane_wave.iter().enumerate() {
            let degree = match precision {
                Some(_) => plane_wave_degree(rotation.degrees[l], eps),
                None => rotation.degrees[l],
            };
            assert_eq!(row.degrees[l], degree, "ε = {eps}, level {l}");
        }
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
        // The raised levels' translations at the degrees given (CH6.1h): the rule's form, and
        // each named one.
        assert_eq!(row.plane_wave_raise, PlaneWaveRaise::Rule);
        assert_eq!(row.translation_degrees, rotation.degrees);
        let resolved = PlaneWaveRaise::Rule.resolved(Some(eps));
        for raise in [
            PlaneWaveRaise::Level,
            PlaneWaveRaise::Leaves,
            PlaneWaveRaise::M2l,
        ] {
            if eps >= 1e-6 {
                break;
            }
            case.plane_wave_raise = Some(raise);
            let named = run(&case, &problem, &oracle, 1, world).expect("the FMM runs");
            case.plane_wave_raise = None;
            assert_eq!(named.plane_wave_raise, raise);
            assert_eq!(named.degrees, row.degrees);
            let translations = match raise {
                PlaneWaveRaise::Level => &row.degrees,
                _ => &rotation.degrees,
            };
            assert_eq!(&named.translation_degrees, translations);
            if raise == resolved {
                assert_eq!(named.checksum, row.checksum, "{raise:?}: the rule's form");
            }
            let all = named.errors.last().unwrap();
            println!(
                "{workload:?} at ε = {eps}, {raise:?}: errors {all:?}, Rotation's {reference:?}"
            );
            // M2l outside the rule's band (10⁻⁷ ≤ ε < 10⁻⁶) is not held to the ratio.
            if raise == PlaneWaveRaise::M2l && eps < 1e-7 {
                continue;
            }
            for j in 0..2 {
                assert!(
                    all[j] <= PLANE_WAVE_RATIO * reference[j],
                    "{workload:?} at ε = {eps}, raised by {raise:?}: errors {all:?} against \
                     Rotation's {reference:?}"
                );
            }
        }
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
    let mut case = case(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
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
    let mut case = case(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
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
    let mut w1 = case(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    w1.vectors = 2;
    check_fmm(&world, &w1);
    // W4's clusters, complex k, real charges, no gradients.
    let mut w4 = case(
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
    check_densities(&world);
    check_box_study(&world);
    check_empty_boxes(&world);
    check_near_v(&world);
    check_leaves(&world);
    check_near_wx(&world);
    check_block_budget(&world);
    check_rcr_split(&world);
    check_diagonal(&world);
}

/// Helmholtz CH4.0e: the M2L's rotation blocks formed per V pair (a block budget of 0)
/// against the stored ones (the library's budget), on Rotation: the same bits, and the
/// stored levels reported; and (Helmholtz CH4.0h) the build by part, `Row::build_parts`.
fn check_block_budget(world: &SimpleCommunicator) {
    let stored = case(Workload::W1, (30.0, 0.0), Degrees::Tolerance(1e-6));
    assert_eq!(stored.block_budget, None, "Case::new's");
    assert_eq!(stored.m2l, HelmholtzM2l::Rotation);
    let per_pair = Case {
        block_budget: Some(0),
        ..stored.clone()
    };
    let problem = Problem::new(&stored);
    let oracle = Oracle::sharded(complex_k(&stored), &problem, stored.gradients, 2, world);
    let row_stored = run(&stored, &problem, &oracle, 2, world).expect("the FMM runs");
    let row_per_pair = run(&per_pair, &problem, &oracle, 2, world).expect("the FMM runs");
    let with_pairs: Vec<usize> = (0..row_stored.levels)
        .filter(|&l| row_stored.v_pairs[l] > 0)
        .collect();
    assert!(!with_pairs.is_empty());
    assert!(with_pairs.iter().all(|&l| row_stored.m2l_blocks_stored[l]));
    assert!(
        with_pairs
            .iter()
            .all(|&l| !row_per_pair.m2l_blocks_stored[l])
    );
    assert_eq!(row_stored.m2l_blocks_stored.len(), row_stored.levels);
    assert_eq!(row_per_pair.checksum, row_stored.checksum);
    assert_eq!(row_per_pair.errors, row_stored.errors);
    // The build by part (Helmholtz CH4.0h): the octree, the routes, the plan, the operator
    // and its RCR tables timed; on one rank the tables within the operator and the steps
    // within the build (on several, each part is the slowest rank's, maybe not one rank's).
    let parts = |name: &str| {
        let i = BUILD_PARTS.iter().position(|&p| p == name).expect("a part");
        row_stored.build_parts[i]
    };
    for name in [
        "octree",
        "routes",
        "plan",
        "operator",
        "RCR tables",
        "evaluator",
    ] {
        assert!(parts(name) > 0.0, "{name}: {:?}", row_stored.build_parts);
    }
    let tables = [
        "RCR tables",
        "matrices",
        "plane-wave tables",
        "diagonal tables",
        "scratch",
    ];
    if world.size() == 1 {
        let in_tables: f64 = tables.iter().map(|&n| parts(n)).sum();
        assert!(in_tables <= parts("operator") * (1.0 + 1e-9));
        let steps: f64 = BUILD_PARTS
            .iter()
            .filter(|n| !tables.contains(n))
            .map(|&n| parts(n))
            .sum();
        assert!(
            steps <= row_stored.build * 1.000_001,
            "{steps} > {}",
            row_stored.build
        );
    }
}

/// The Rotation M2L by pairs (Helmholtz CH4.0i) on W1 at kD = 30: `None` in `Case::new` (the
/// library's `Auto`); by pairs at 2 threads the checksum and errors of by boxes, every level
/// with V pairs reported by pairs, none by boxes.
fn check_rcr_split(world: &SimpleCommunicator) {
    let auto = case(Workload::W1, (30.0, 0.0), Degrees::Tolerance(1e-3));
    assert_eq!(auto.rcr_split, None, "Case::new's");
    let boxes = Case {
        rcr_split: Some(RcrSplit::Boxes),
        ..auto.clone()
    };
    let pairs = Case {
        rcr_split: Some(RcrSplit::Pairs),
        ..auto.clone()
    };
    let problem = Problem::new(&auto);
    let oracle = Oracle::sharded(complex_k(&auto), &problem, auto.gradients, 2, world);
    let row_boxes = run(&boxes, &problem, &oracle, 2, world).expect("the FMM runs");
    let row_pairs = run(&pairs, &problem, &oracle, 2, world).expect("the FMM runs");
    let row_auto = run(&auto, &problem, &oracle, 2, world).expect("the FMM runs");
    assert_eq!(row_auto.rcr_split, RcrSplit::Auto);
    assert_eq!(row_boxes.rcr_split, RcrSplit::Boxes);
    assert_eq!(row_pairs.rcr_split, RcrSplit::Pairs);
    for l in 0..row_pairs.levels {
        assert!(!row_boxes.rcr_by_pairs[l], "level {l} by pairs");
        assert_eq!(
            row_pairs.rcr_by_pairs[l],
            row_pairs.v_pairs[l] > 0,
            "level {l}"
        );
    }
    for row in [&row_pairs, &row_auto] {
        assert_eq!(row.checksum, row_boxes.checksum);
        assert_eq!(row.errors, row_boxes.errors);
    }
}

/// The W and X pairs as near pairs (Helmholtz CH4.0d3) on W4's clusters (N = 2,000, kD = 10;
/// an adaptive tree): off in `Case::new` with nothing offered; at `inf` every W and X pair
/// between leaves converted, the P2L and M2P interactions fewer and the near interactions
/// more than off, the errors within 1.1× the rule off's.
fn check_near_wx(world: &SimpleCommunicator) {
    let off = case(
        Workload::W4 {
            distribution: Distribution::Clusters,
            n: 2000,
        },
        (10.0, 0.0),
        Degrees::Tolerance(1e-3),
    );
    assert_eq!(off.near_wx, WxNear::Off, "Case::new's");
    let on = Case {
        near_wx: WxNear::Crossovers {
            w: f64::INFINITY,
            x: f64::INFINITY,
        },
        max_points_per_leaf: Some(16),
        ..off.clone()
    };
    let off = Case {
        max_points_per_leaf: Some(16),
        ..off
    };
    let problem = Problem::new(&off);
    let oracle = Oracle::sharded(complex_k(&off), &problem, off.gradients, 2, world);
    let row_off = run(&off, &problem, &oracle, 1, world).expect("the FMM runs");
    let row_on = run(&on, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(row_off.near_wx, WxNear::Off);
    assert_eq!(row_on.near_wx, on.near_wx);
    let [w, w_converted, x, x_converted] = row_on.wx_near;
    assert!(
        w + x > 0 && w_converted == w && x_converted == x,
        "{:?}",
        row_on.wx_near
    );
    assert_eq!(row_off.wx_near[1] + row_off.wx_near[3], 0);
    let far = |row: &Row| row.far_interactions.iter().flatten().sum::<u64>();
    assert!(
        far(&row_on) < far(&row_off),
        "{} {}",
        far(&row_on),
        far(&row_off)
    );
    let near = |row: &Row| row.near_interactions.iter().sum::<u64>();
    assert!(near(&row_on) > near(&row_off));
    for (on, off) in row_on.errors.iter().zip(&row_off.errors) {
        assert!(
            on[0] <= 1.1 * off[0] && on[1] <= 1.1 * off[1],
            "{on:?}, {off:?}"
        );
    }
}

/// The leaf size and c_leaf (Helmholtz CH4.0d3) on W1 at kD = 100 (leaves on the leaf
/// rule's level at 64 points per leaf): the library's as built by default (the leaf size
/// rule's and `DEFAULT_C_LEAF`), named values as built, the leaves per level summing to the
/// leaves and the occupied ones to at most those, the phase bound within 22 c_leaf without V
/// pairs made near and twice that with them, and c_leaf = 2 putting the leaf rule's level
/// one level coarser than 1.
fn check_leaves(world: &SimpleCommunicator) {
    let default = Case::new(Workload::W1, (100.0, 0.0), Degrees::Tolerance(1e-3));
    assert_eq!(default.max_points_per_leaf, None, "the library's");
    assert_eq!(default.c_leaf, None, "the library's");
    let problem = Problem::new(&default);
    let oracle = Oracle::sharded(complex_k(&default), &problem, default.gradients, 2, world);
    let row = run(&default, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(row.c_leaf, nd_fmm_exec::helmholtz::DEFAULT_C_LEAF);
    assert_eq!(
        row.max_points_per_leaf,
        nd_fmm_exec::helmholtz::default_max_points_per_leaf(Some(1e-3), Isa::detect())
    );
    let narrow = Case {
        max_points_per_leaf: Some(64),
        c_leaf: Some(nd_fmm_exec::helmholtz::C_LEAF),
        ..default.clone()
    };
    let narrow = run(&narrow, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!((narrow.max_points_per_leaf, narrow.c_leaf), (64, 1.0));
    let named = Case {
        max_points_per_leaf: Some(256),
        c_leaf: Some(2.0),
        ..default.clone()
    };
    let wide = run(&named, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!((wide.max_points_per_leaf, wide.c_leaf), (256, 2.0));
    assert_eq!(
        wide.min_leaf_level + 1,
        narrow.min_leaf_level,
        "one level coarser"
    );
    let near = Case {
        near_v: VNear::Crossover(f64::INFINITY),
        ..default.clone()
    };
    let converted = run(&near, &problem, &oracle, 1, world).expect("the FMM runs");
    for (row, span) in [(&row, 2.0), (&narrow, 2.0), (&wide, 2.0), (&converted, 4.0)] {
        assert_eq!(row.leaves_by_level.len(), row.levels);
        let leaves: usize = row.leaves_by_level.iter().map(|c| c[0]).sum();
        assert_eq!(leaves, row.leaves);
        assert!(row.leaves_by_level.iter().all(|c| c[1] <= c[0]));
        assert_eq!(row.kind_seconds.len(), row.levels);
        assert_eq!(row.far_leaf_entries.len(), row.levels);
        // Every level with leaves holding points ran L2P (its fourth kind).
        assert!(
            (0..row.levels).all(|l| row.leaves_by_level[l][1] == 0 || row.kind_seconds[l][3] > 0.0),
            "{:?}",
            row.kind_seconds
        );
        assert!(row.leaves_by_level.iter().map(|c| c[1]).sum::<usize>() > 0);
        let phase = row.phase_bound(100.0);
        assert!(
            phase > 0.0 && phase <= span * 11.0 * row.c_leaf,
            "{phase} at c_leaf {}",
            row.c_leaf
        );
        for e in row.errors.last().expect("an entry over all vectors") {
            assert!(*e < 1e-3, "{e:e}");
        }
    }
    assert!(converted.v_near[1] > 0);
}

/// The V pairs as near pairs (Helmholtz CH4.0d2) on W1 at kD = 30 under Rotation: off by
/// default with nothing offered; every V pair between leaves converted at `inf`, fewer V
/// pairs and more near interactions than off, the errors within 1.1× the rule off's, and
/// the P2P timed on every level with interactions.
fn check_near_v(world: &SimpleCommunicator) {
    let off = case(Workload::W1, (30.0, 0.0), Degrees::Tolerance(1e-3));
    assert_eq!(off.near_v, VNear::Off, "Case::new's");
    let on = Case {
        near_v: VNear::Crossover(f64::INFINITY),
        ..off.clone()
    };
    let problem = Problem::new(&off);
    let oracle = Oracle::sharded(complex_k(&off), &problem, off.gradients, 2, world);
    let row_off = run(&off, &problem, &oracle, 1, world).expect("the FMM runs");
    let row_on = run(&on, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(row_off.near_v, VNear::Off);
    assert_eq!(row_off.v_near, [0, 0]);
    assert_eq!(row_on.near_v, on.near_v);
    assert!(
        row_on.v_near[0] > 0 && row_on.v_near[1] == row_on.v_near[0],
        "{:?}",
        row_on.v_near
    );
    let pairs = |row: &Row| row.v_pairs.iter().sum::<usize>();
    assert_eq!(
        pairs(&row_on) + row_on.v_near[1],
        pairs(&row_off),
        "V pairs"
    );
    let interactions = |row: &Row| row.near_interactions.iter().sum::<u64>();
    assert!(interactions(&row_on) > interactions(&row_off));
    for (on, off) in row_on.errors.iter().zip(&row_off.errors) {
        assert!(
            on[0] <= 1.1 * off[0] && on[1] <= 1.1 * off[1],
            "{on:?}, {off:?}"
        );
    }
    for row in [&row_on, &row_off] {
        assert_eq!(row.p2p_seconds.len(), row.levels);
        assert!(
            (0..row.levels).all(|l| row.near_interactions[l] == 0 || row.p2p_seconds[l] > 0.0),
            "{:?}",
            row.p2p_seconds
        );
        // Summed over the ranks, every point sent is received.
        assert_eq!(row.source_points[0], row.source_points[1]);
    }
    if world.size() == 1 {
        assert_eq!(row_on.source_points, [0, 0], "one rank exchanges no point");
    }
    assert!(row_on.remote_multipoles <= row_off.remote_multipoles);
}

/// The empty-box rule (Helmholtz CH4.0d1) on W1 at kD = 30 under Rotation (leaves on
/// level 3, most of it empty): on by default, with entries left out and fewer V pairs than
/// off, the same errors, and nothing left out with it off.
fn check_empty_boxes(world: &SimpleCommunicator) {
    let on = case(Workload::W1, (30.0, 0.0), Degrees::Tolerance(1e-3));
    assert!(on.empty_boxes, "the library's default");
    let off = Case {
        empty_boxes: false,
        ..on.clone()
    };
    let problem = Problem::new(&on);
    let oracle = Oracle::sharded(complex_k(&on), &problem, on.gradients, 2, world);
    let row_on = run(&on, &problem, &oracle, 1, world).expect("the FMM runs");
    let row_off = run(&off, &problem, &oracle, 1, world).expect("the FMM runs");
    assert!(row_on.empty_boxes && !row_off.empty_boxes);
    assert!(
        row_on.left_out[0] > 0 && row_on.left_out[5] > 0,
        "{:?}",
        row_on.left_out
    );
    assert_eq!(row_off.left_out, [0; 8]);
    let pairs = |row: &Row| row.v_pairs.iter().sum::<usize>();
    assert!(pairs(&row_on) < pairs(&row_off), "V pairs");
    assert_eq!(
        row_on.errors, row_off.errors,
        "the rule leaves out exact zeros"
    );
    if world.size() == 1 {
        assert_eq!(
            row_on.remote_multipoles, 0,
            "one rank reads no other's multipole"
        );
    }
    assert!(row_on.remote_multipoles <= row_off.remote_multipoles);
    check_levels(&row_on);
}

/// The smooth densities (Helmholtz CH4.0b) on W1 at kD = 10: the plane wave's trace
/// q_j = e^{ik d·x_j} through `check_fmm`, the constant density (real charges) within ε;
/// §9.1's random charges the same problem with the density named; on W1 at kD = 20 a named
/// domain with a margin and a tree of one leaf level (no leaf size bound) within ε, the
/// domain and the leaves as named.
fn check_densities(world: &SimpleCommunicator) {
    let case = case(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    // §9.1's charges as before the field: Re q then Im q from the generator of seed + 1.
    let random = Problem::new(&case);
    let mut rng = SplitMix64::new(case.seed + 1);
    assert_eq!(random.re[0], points::charges(&mut rng, random.len()));
    assert_eq!(random.im[0], points::charges(&mut rng, random.len()));
    let wave = Case {
        density: Density::PlaneWave,
        ..case.clone()
    };
    let problem = Problem::new(&wave);
    for (i, x) in problem.points.iter().enumerate() {
        let phase = 10.0 * (0..3).map(|a| PLANE_WAVE_DIRECTION[a] * x[a]).sum::<f64>();
        assert_eq!(problem.re[0][i], phase.cos());
        assert_eq!(problem.im[0][i], phase.sin());
    }
    check_fmm(world, &wave);
    let mut constant = Case {
        density: Density::Constant,
        ..case.clone()
    };
    constant.charges = ChargeKind::Real;
    let problem = Problem::new(&constant);
    assert!(problem.re[0].iter().all(|&q| q == 1.0));
    let oracle = Oracle::sharded(complex_k(&constant), &problem, true, 2, world);
    let row = run(&constant, &problem, &oracle, 2, world).expect("the FMM runs");
    let one = run(&constant, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(one.checksum, row.checksum, "threads change the output");
    assert!(
        row.errors[0][0] <= 1e-3 && row.errors[0][1] <= 1e-3,
        "{:?}",
        row.errors
    );
    for name in ["random", "planewave", "constant"] {
        assert_eq!(Density::from_name(name).map(|d| d.name()), Some(name));
    }

    // At kD = 20 the leaf rule puts the leaves on level 2 of the domain of side 1.2.
    let named = Case {
        k: (20.0, 0.0),
        domain: Some([-0.1, -0.1, -0.1, 1.1, 1.1, 1.1]),
        max_points_per_leaf: Some(usize::MAX),
        ..case.clone()
    };
    let problem = Problem::new(&named);
    let oracle = Oracle::sharded(complex_k(&named), &problem, true, 2, world);
    let run = run_output(&named, &problem, &oracle, 1, world).expect("the FMM runs");
    assert!(run.row.errors[0][0] <= 1e-3 && run.row.errors[0][1] <= 1e-3);
    check_levels(&run.row);
    assert_eq!(run.domain.lower(), [-0.1; 3]);
    assert!((run.domain.side() - 1.2).abs() < 1e-15);
    // One leaf level: every box above the deepest level L that holds points is refined, so
    // the leaves, empty ones included, are Σ_{l<L} 8 occ(l) − Σ_{0<l<L} occ(l); a bound of 16
    // points per leaf refines further.
    let deepest = run.row.levels - 1;
    let occupied = |level: usize| {
        let grid = Grid {
            domain: run.domain,
            level,
        };
        let mut keys: Vec<u64> = problem.points.iter().map(|&x| grid.box_of(x)).collect();
        keys.sort_unstable();
        keys.dedup();
        keys.len()
    };
    let children: usize = (0..deepest).map(|l| 8 * occupied(l)).sum();
    let refined: usize = (1..deepest).map(occupied).sum();
    assert_eq!(run.row.leaves, children - refined);
    let fine = Case {
        max_points_per_leaf: Some(16),
        ..named.clone()
    };
    let refined = run_output(&fine, &problem, &oracle, 1, world).expect("the FMM runs");
    assert!(refined.row.leaves > run.row.leaves);
}

/// The box study's measures (Helmholtz CH4.0b) on W1 at kD = 10 on one rank (each rank its
/// own): the reference at every point equals the sharded oracle at the samples bit for bit,
/// and a per-box sample the reference at every point at its targets; the box measure is the
/// FMM3D comparison's `crosscheck` against the same reference bit for bit, its ℓ2 errors to
/// rounding.
fn check_box_study(world: &SimpleCommunicator) {
    let case = case(Workload::W1, (10.0, 0.0), Degrees::Tolerance(1e-3));
    let problem = Problem::new(&case);
    let k = complex_k(&case);
    let oracle = Oracle::sharded(k, &problem, true, 2, world);
    let every = Reference::every(k, &problem, 0, 3);
    assert_eq!(every.targets, (0..problem.len()).collect::<Vec<_>>());
    assert_eq!(every, Reference::every(k, &problem, 0, 1), "threads");
    for (s, &i) in problem.sample.iter().enumerate() {
        assert_eq!(every.potential[i], oracle.potential[0][s]);
        assert_eq!(every.gradient[i], oracle.gradient[0][s]);
    }

    let one = SimpleCommunicator::self_comm();
    let run = run_output(&case, &problem, &oracle, 2, &one).expect("the FMM runs");
    let grid = Grid {
        domain: run.domain,
        level: run.row.levels - 1,
    };
    let targets = box_sample(&problem.points, &problem.sample[..10], &grid, 2, 7);
    let sampled = Reference::at(k, &problem, 0, targets.clone(), 2);
    assert_eq!(sampled.targets, targets);
    for (j, &i) in targets.iter().enumerate() {
        assert_eq!(sampled.potential[j], every.potential[i]);
        assert_eq!(sampled.gradient[j], every.gradient[i]);
    }

    let errors = Errors::of(&run.output, 0, &every);
    let boxes = errors.boxes(&every, &problem.points, &grid);
    let full = Values {
        potential: every.potential.clone(),
        gradient: every.gradient.clone(),
    };
    let all = 0..problem.len();
    let check = crosscheck(&run.output, 0, &problem, &all, &full, &grid, &one);
    let ours: Vec<(u64, [f64; 2])> = boxes.iter().map(|b| (b.key, b.error)).collect();
    assert_eq!(ours, check.boxes);
    let l2 = errors.l2(&every);
    for (ours, theirs) in l2.iter().zip(check.errors) {
        assert!((ours - theirs).abs() <= 1e-12 * theirs);
    }
    assert!(boxes.iter().all(|b| b.targets > 0));
    assert_eq!(
        boxes.iter().map(|b| b.targets).sum::<usize>(),
        problem.len()
    );
}

/// The diagonal M2L (Helmholtz CH4.2b2) on W1 at kD = 30, ε = 10⁻⁴: off in `Case::new`, no
/// level and no columns; by the rule some level takes it, the row's columns the operator's
/// counts and times (pairs on the level's V pairs, signatures and conversions back, the parts
/// timed, the multiply-adds and build), the errors within 1.1× Rotation's without it, and the
/// same checksum at 1 and 2 threads; the same with interpolated transfer functions (Helmholtz
/// CH4.2c2), its classes at most the direct build's per level.
fn check_diagonal(world: &SimpleCommunicator) {
    let off = case(Workload::W1, (30.0, 0.0), Degrees::Tolerance(1e-4));
    assert_eq!(off.diagonal, HelmholtzDiagonal::Off, "Case::new's");
    let on = Case {
        diagonal: HelmholtzDiagonal::Rule,
        ..off.clone()
    };
    let problem = Problem::new(&off);
    let oracle = Oracle::sharded(complex_k(&off), &problem, off.gradients, 2, world);
    let row_off = run(&off, &problem, &oracle, 1, world).expect("the FMM runs");
    let row_on = run(&on, &problem, &oracle, 1, world).expect("the FMM runs");
    let row_two = run(&on, &problem, &oracle, 2, world).expect("the FMM runs");
    assert_eq!(row_off.diagonal, HelmholtzDiagonal::Off);
    assert_eq!(row_on.diagonal, HelmholtzDiagonal::Rule);
    assert!(row_off.diagonal_classes.iter().all(|&c| c == 0));
    assert!(row_off.diagonal_seconds.iter().flatten().all(|&t| t == 0.0));
    let levels: Vec<usize> = (0..row_on.levels)
        .filter(|&l| row_on.diagonal_classes[l] > 0)
        .collect();
    assert!(!levels.is_empty(), "no level took the diagonal form");
    for &l in &levels {
        let [diagonal, rcr] = row_on.diagonal_pairs[l];
        let [forward, back] = row_on.diagonal_conversions[l];
        let [signatures, targets, s2f, pairs, f2r, rcr_s] = row_on.diagonal_seconds[l];
        // On one rank the level call's pairs are the plan's (on several, a replicated top
        // box's rows run on every rank).
        if world.size() == 1 {
            assert_eq!(diagonal + rcr, row_on.v_pairs[l], "level {l}");
            assert!(back <= row_on.v_targets[l], "level {l}");
        }
        assert!(diagonal > 0 && forward > 0 && back > 0, "level {l}");
        assert!(forward <= diagonal, "level {l}");
        assert!(signatures > 0.0 && targets > 0.0 && s2f > 0.0 && pairs > 0.0 && f2r > 0.0);
        assert_eq!(rcr_s > 0.0, rcr > 0, "level {l}");
        // Each column is the slowest rank's, so on several ranks the passes' sum may exceed the
        // M2L's maximum.
        if world.size() == 1 {
            assert!(signatures + targets <= row_on.m2l_seconds[l], "level {l}");
        }
        assert!(row_on.diagonal_multiply_adds[l] > 0, "level {l}");
        assert!(
            row_on.diagonal_build[l].iter().all(|&t| t > 0.0),
            "level {l}"
        );
    }
    for l in (0..row_on.levels).filter(|l| !levels.contains(l)) {
        assert_eq!(row_on.diagonal_pairs[l], [0, 0], "level {l}");
        assert_eq!(row_on.diagonal_multiply_adds[l], 0, "level {l}");
    }
    let diagonal = BUILD_PARTS
        .iter()
        .position(|&p| p == "diagonal tables")
        .expect("a part");
    // Off, the rule's choice of no level is timed too (a fraction of a microsecond).
    assert!(row_on.build_parts[diagonal] > row_off.build_parts[diagonal]);
    for (got, want) in row_on.errors.iter().zip(&row_off.errors) {
        assert!(
            got[0] <= 1.1 * want[0] && got[1] <= 1.1 * want[1],
            "{got:?} against {want:?}"
        );
    }
    assert_eq!(row_two.checksum, row_on.checksum);
    if world.rank() == 0 {
        println!(
            "diagonal on W1 at kD = 30, 1e-4: levels {levels:?}, classes {:?}, errors {:?} \
             against {:?}",
            row_on.diagonal_classes, row_on.errors, row_off.errors
        );
    }
    // The interpolated build (Helmholtz CH4.2c2): reported, its classes at most the direct
    // build's per level, within 1.1× Rotation's errors, 1 and 2 threads the same checksum.
    assert_eq!(off.diagonal_build, TransferBuild::Direct, "Case::new's");
    assert_eq!(row_on.diagonal_build_kind, TransferBuild::Direct);
    let interpolated = Case {
        diagonal_build: TransferBuild::Interpolated,
        ..on.clone()
    };
    let row_i = run(&interpolated, &problem, &oracle, 1, world).expect("the FMM runs");
    let row_i_two = run(&interpolated, &problem, &oracle, 2, world).expect("the FMM runs");
    assert_eq!(row_i.diagonal_build_kind, TransferBuild::Interpolated);
    assert!(
        row_i
            .diagonal_classes
            .iter()
            .zip(&row_on.diagonal_classes)
            .all(|(i, d)| i <= d),
        "{:?} against {:?}",
        row_i.diagonal_classes,
        row_on.diagonal_classes
    );
    assert!(row_i.diagonal_classes.iter().any(|&c| c > 0));
    for (got, want) in row_i.errors.iter().zip(&row_off.errors) {
        assert!(
            got[0] <= 1.1 * want[0] && got[1] <= 1.1 * want[1],
            "interpolated: {got:?} against {want:?}"
        );
    }
    assert_eq!(row_i_two.checksum, row_i.checksum);
    if world.rank() == 0 {
        println!(
            "interpolated: classes {:?}, errors {:?}",
            row_i.diagonal_classes, row_i.errors
        );
    }
    // The time rule (Helmholtz CH4.3b1): reported, its levels a subset of the rule's with the
    // same classes, within 1.1× Rotation's errors; the model's columns on every level: k a
    // from the level's circumradius, the bound's classes the rule's where it took a level,
    // the modelled seconds positive where the level has V pairs.
    assert_eq!(off.diagonal_costs, None, "Case::new's");
    let auto = Case {
        diagonal: HelmholtzDiagonal::Auto,
        ..on.clone()
    };
    let row_a = run(&auto, &problem, &oracle, 1, world).expect("the FMM runs");
    assert_eq!(row_a.diagonal, HelmholtzDiagonal::Auto);
    for l in 0..row_a.levels {
        let taken = row_a.diagonal_classes[l];
        assert!(
            taken == 0 || taken == row_on.diagonal_classes[l],
            "level {l}"
        );
        let [ka, bound, diagonal, rcr, waves] = row_a.diagonal_model[l];
        let expected = 30.0 * row_a.circumradii[l];
        assert!((ka - expected).abs() <= 1e-12 * expected, "level {l}");
        if l >= 2 && row_on.diagonal_classes[l] > 0 {
            assert_eq!(bound as usize, row_on.diagonal_classes[l], "level {l}");
        }
        if l >= 2 && row_a.v_pairs[l] > 0 {
            assert!(diagonal > 0.0 && rcr > 0.0 && waves > 0.0, "level {l}");
        }
    }
    for (got, want) in row_a.errors.iter().zip(&row_off.errors) {
        assert!(
            got[0] <= 1.1 * want[0] && got[1] <= 1.1 * want[1],
            "time rule: {got:?} against {want:?}"
        );
    }
    if world.rank() == 0 {
        println!(
            "time rule: classes {:?}, model {:?}, errors {:?}",
            row_a.diagonal_classes, row_a.diagonal_model, row_a.errors
        );
    }
    // The time rule at matched accuracy (Helmholtz CH4.3b2a): under Auto at 10⁻⁷, where the
    // waves raise their levels, at the costs that force the form and RCR. The costs reported,
    // the degrees the defaults', some level taken from the waves, within 1.1× the defaults'
    // errors.
    let epsilon = 1e-7;
    let defaults = Case {
        m2l: HelmholtzM2l::Auto,
        degrees: Degrees::Tolerance(epsilon),
        diagonal_build: TransferBuild::Interpolated,
        ..off.clone()
    };
    let problem = Problem::new(&defaults);
    let oracle = Oracle::sharded(complex_k(&defaults), &problem, defaults.gradients, 2, world);
    let row_d = run(&defaults, &problem, &oracle, 1, world).expect("the FMM runs");
    assert!(
        row_d.plane_wave.iter().any(Option::is_some),
        "the defaults run waves"
    );
    for (name, costs) in [("form", FORM_COSTS), ("rcr", RCR_COSTS)] {
        let forced = Case {
            diagonal: HelmholtzDiagonal::Auto,
            diagonal_costs: Some(costs),
            ..defaults.clone()
        };
        let row = run(&forced, &problem, &oracle, 1, world).expect("the FMM runs");
        assert_eq!(row.diagonal_costs, costs, "{name}");
        assert_eq!(row.degrees, row_d.degrees, "{name}: the defaults' degrees");
        assert!(
            row.taken_from_waves.iter().any(|&t| t),
            "{name}: no level taken from the waves"
        );
        for l in 0..row.levels {
            if row.taken_from_waves[l] {
                assert!(row_d.plane_wave[l].is_some(), "{name}, level {l}");
                assert!(row.plane_wave[l].is_none(), "{name}, level {l}");
                assert_eq!(
                    row.diagonal_classes[l] > 0,
                    name == "form",
                    "{name}, level {l}: the form at form costs, RCR at RCR costs"
                );
            }
        }
        for (got, want) in row.errors.iter().zip(&row_d.errors) {
            let within = |g: f64, w: f64| g <= 1.1 * w;
            assert!(
                within(got[0], want[0]) && within(got[1], want[1]),
                "{name}: {got:?} against the defaults' {want:?}"
            );
        }
        if world.rank() == 0 {
            println!(
                "time rule at {name} costs, ε {epsilon:e}: taken from the waves {:?}, classes \
                 {:?}, errors {:?} against the defaults' {:?}",
                row.taken_from_waves, row.diagonal_classes, row.errors, row_d.errors
            );
        }
    }
}
