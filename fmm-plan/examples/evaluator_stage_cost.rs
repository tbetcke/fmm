//! Measure the wall time per stage of the old and the new evaluator with the index FMM,
//! on one rank.
//!
//! The tree holds uniform random points, about 10⁴ leaves by default. The index FMM
//! stores one count per leaf in every coefficient and target point, so each evaluator
//! needs about boxes × leaves × 4 bytes for multipoles and locals each, about 1.3 GB at
//! 10⁴ leaves (design §11.4); the evaluators are built one after the other. Every
//! evaluator is checked once, then each stage is timed, best of `repeats` runs. Nothing
//! is asserted. Run in release mode:
//! `cargo run --release -p nd-fmm-plan --example evaluator_stage_cost [npoints [capacity]]`.

use std::time::Instant;

use mpi::traits::CommunicatorCollectives;
use nd_fmm_plan::{
    fmm::{
        evaluator::FmmEvaluator,
        index_fmm::{IndexFmm as OldIndexFmm, check_targets, global_leaf_indices},
    },
    interaction_manager::InteractionManager,
    v2::{
        evaluator::Evaluator,
        index_fmm::{BatchedIndexFmm, GlobalLeaves, IndexFmm, Walk, check_counts},
        operator::{FmmOperator, PerPair},
        plan::Plan,
    },
};
use nd_octree::{Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rlst::rlst_dynamic_array;

/// The stages of both evaluators, in order.
const STAGES: [&str; 7] = [
    "reset",
    "exchange_sources",
    "upward_local",
    "upward_global",
    "exchange_multipoles",
    "downward",
    "evaluate_leaves",
];

/// Time `f` in ms.
fn time(f: impl FnOnce()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1e3
}

/// The best of `repeats` timings of every stage of the new evaluator with `operator`,
/// and the construction time.
fn new_evaluator<C: CommunicatorCollectives, Op: FmmOperator<Value = u32>>(
    plan: &Plan,
    comm: &C,
    operator: Op,
    repeats: usize,
) -> (f64, [f64; 7]) {
    let numbering = GlobalLeaves::new(plan, comm);
    let ones = vec![1; plan.index().leaves().nlocal()];
    let mut evaluator = None;
    let build = time(|| evaluator = Some(Evaluator::new(plan, comm, operator, &ones, &ones)));
    let mut evaluator = evaluator.unwrap().unwrap();
    numbering.fill_sources(evaluator.local_sources_mut());
    evaluator.evaluate();
    let global = numbering.gather_counts(comm, &ones);
    check_counts(plan.index(), evaluator.target_output_store(), &global).unwrap();

    let mut best = [f64::INFINITY; 7];
    for _ in 0..repeats {
        let times = [
            time(|| evaluator.reset()),
            time(|| evaluator.exchange_sources()),
            time(|| evaluator.upward_local()),
            time(|| evaluator.upward_global()),
            time(|| evaluator.exchange_multipoles()),
            time(|| evaluator.downward()),
            time(|| evaluator.evaluate_leaves()),
        ];
        for (b, t) in best.iter_mut().zip(times) {
            *b = b.min(t);
        }
    }
    (build, best)
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let mut args = std::env::args().skip(1);
    let npoints: usize = args.next().map_or(26_500, |a| a.parse().unwrap());
    let capacity: usize = args.next().map_or(8, |a| a.parse().unwrap());
    let repeats = 2;

    let mut points = rlst_dynamic_array!(f64, [3, npoints]);
    points.fill_from_equally_distributed(&mut ChaCha8Rng::seed_from_u64(0));
    let bounding_box = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let fine_keys = points_to_morton(&points, DEEPEST_LEVEL as usize, &bounding_box);
    let options = OctreeOptions::new()
        .with_max_level(DEEPEST_LEVEL as usize)
        .with_max_fine_keys(capacity)
        .with_ghost_children(true);
    let octree = Octree::new(&fine_keys, options, &comm);
    let nleaves = octree.leaf_keys().len();
    println!(
        "{npoints} uniform points, {capacity} per leaf: {nleaves} leaves, {} boxes held, {} levels",
        octree.all_keys().len(),
        octree.global_max_level() + 1
    );

    // The old evaluator. It has no public reset; `evaluate` clears first, and the stages
    // are timed after it, on uncleared data, which costs the same.
    let mut rows = Vec::new();
    {
        let lists = InteractionManager::new(&octree);
        let (n, indices) = global_leaf_indices(&octree);
        let mut evaluator = None;
        let build = time(|| {
            evaluator = Some(FmmEvaluator::new(&octree, &lists, OldIndexFmm::new(n)));
        });
        let mut evaluator = evaluator.unwrap();
        for (&leaf, &index) in octree.leaf_keys().iter().zip(&indices) {
            evaluator.sources_mut(leaf)[0] = index;
        }
        let reset = time(|| evaluator.evaluate());
        check_targets(&evaluator).unwrap();
        let mut best = [f64::INFINITY; 7];
        for _ in 0..repeats {
            let times = [
                f64::NAN,
                time(|| evaluator.exchange_sources()),
                time(|| evaluator.upward_local()),
                time(|| evaluator.upward_global()),
                time(|| evaluator.exchange_multipoles()),
                time(|| evaluator.downward()),
                time(|| evaluator.evaluate_leaves()),
            ];
            for (b, t) in best.iter_mut().zip(times) {
                *b = b.min(t);
            }
        }
        best[0] = f64::NAN;
        println!("old: full evaluate (with clear) {reset:.1} ms");
        rows.push(("old `FmmEvaluator`", build, best));
    }

    let plan = Plan::new(&octree).unwrap();
    rows.push({
        let (build, best) = new_evaluator(&plan, &comm, PerPair(IndexFmm::new(nleaves)), repeats);
        ("new `Evaluator`, `PerPair<IndexFmm>`", build, best)
    });
    rows.push({
        let (build, best) = new_evaluator(
            &plan,
            &comm,
            BatchedIndexFmm::new(nleaves, Walk::Rows),
            repeats,
        );
        ("new `Evaluator`, `BatchedIndexFmm` (rows)", build, best)
    });
    rows.push({
        let (build, best) = new_evaluator(
            &plan,
            &comm,
            BatchedIndexFmm::new(nleaves, Walk::Groupings),
            repeats,
        );
        (
            "new `Evaluator`, `BatchedIndexFmm` (groupings)",
            build,
            best,
        )
    });

    println!("\nWall time in ms, best of {repeats}:\n");
    println!("| Evaluator | new | {} | total |", STAGES.join(" | "));
    println!("| --- | --- |{}", " --- |".repeat(STAGES.len() + 1));
    for (name, build, best) in rows {
        let total: f64 = best.iter().filter(|t| !t.is_nan()).sum();
        let cells: Vec<String> = best
            .iter()
            .map(|t| {
                if t.is_nan() {
                    "–".to_string()
                } else {
                    format!("{t:.1}")
                }
            })
            .collect();
        println!(
            "| {name} | {build:.1} | {} | {total:.1} |",
            cells.join(" | ")
        );
    }
}
