//! The stress case of the overlap protocol (Phase 5 T9; docs/phase5/T9-overlap.md): the
//! graded scenario of `tests/mpi_regressions.rs` with the last rank empty, evaluated 100
//! times by `Evaluator::evaluate_overlapped`, each time bit for bit the blocking
//! evaluation, without a hang. Ignored: by hand at 8 ranks, under an external timeout
//! (fmm-plan/CLAUDE.md, "Multi-rank runs"):
//!
//! ```sh
//! cargo test -p nd-fmm-plan --release --test overlap_stress --no-run
//! RUST_MIN_STACK=8388608 timeout 600 mpirun -n 8 target/release/deps/overlap_stress-<hash> \
//!   --ignored --test-threads=1 --nocapture
//! ```
//!
//! Its own executable, since it owns MPI initialisation. Error measure: exact equality of
//! every store, agreed on every rank.

use mpi::{collective::SystemOperation, traits::*};
use nd_fmm_plan::{
    evaluator::Evaluator,
    operator::{FmmSizes, PairOperator, PerPair},
    plan::Plan,
    store::{LeafStore, LevelBuffers},
};
use nd_octree::{Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rlst::rlst_dynamic_array;

/// Evaluations of the stress case.
const EVALUATIONS: usize = 100;

/// The Morton keys of `points` in the unit cube.
fn keys(points: &[[f64; 3]]) -> Vec<u64> {
    let mut array = rlst_dynamic_array!(f64, [3, points.len()]);
    for (j, point) in points.iter().enumerate() {
        for (i, &value) in point.iter().enumerate() {
            array[[i, j]] = value;
        }
    }
    let bounds = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    points_to_morton(&array, DEEPEST_LEVEL as usize, &bounds)
}

/// A per-pair operator whose values do not matter: every call adds a hash of all its
/// inputs and of the pair's keys to its output, so a stale, missing or doubled ghost
/// value shows (as `Mixer` of `tests/mpi_regressions.rs`).
struct Mix;

fn mix(inputs: [&[u32]; 2], keys: [u64; 2], output: &mut [u32]) {
    let hash = inputs.iter().flat_map(|input| input.iter()).fold(
        0xcbf2_9ce4_8422_2325 ^ keys[0] ^ keys[1].rotate_left(32),
        |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x0100_0000_01b3),
    );
    for (k, value) in output.iter_mut().enumerate() {
        *value = value.wrapping_add((hash >> (k % 32)) as u32);
    }
}

impl FmmSizes for Mix {
    type Value = u32;

    fn multipole_size(&self, level: usize) -> usize {
        1 + level % 3
    }

    fn local_size(&self, _level: usize) -> usize {
        2
    }

    fn source_point_size(&self) -> usize {
        2
    }

    fn target_input_point_size(&self) -> usize {
        1
    }

    fn target_output_point_size(&self) -> usize {
        2
    }
}

impl PairOperator for Mix {
    fn p2m(&mut self, leaf: u64, sources: &[u32], multipole: &mut [u32]) {
        mix([sources, &[]], [leaf, leaf], multipole);
    }

    fn m2m(&mut self, child: u64, parent: u64, _: usize, input: &[u32], output: &mut [u32]) {
        mix([input, &[]], [child, parent], output);
    }

    fn m2l(&mut self, source: u64, target: u64, _: usize, input: &[u32], output: &mut [u32]) {
        mix([input, &[]], [source, target], output);
    }

    fn p2l(&mut self, source: u64, target: u64, sources: &[u32], local: &mut [u32]) {
        mix([sources, &[]], [source, target], local);
    }

    fn l2l(&mut self, parent: u64, child: u64, _: usize, input: &[u32], output: &mut [u32]) {
        mix([input, &[]], [parent, child], output);
    }

    fn l2p(&mut self, leaf: u64, local: &[u32], target_input: &[u32], output: &mut [u32]) {
        mix([local, target_input], [leaf, leaf], output);
    }

    fn m2p(
        &mut self,
        source: u64,
        target: u64,
        multipole: &[u32],
        target_input: &[u32],
        output: &mut [u32],
    ) {
        mix([multipole, target_input], [source, target], output);
    }

    fn p2p(
        &mut self,
        source: u64,
        target: u64,
        sources: &[u32],
        target_input: &[u32],
        output: &mut [u32],
    ) {
        mix([sources, target_input], [source, target], output);
    }
}

/// Every store an evaluation writes.
type Stores = (
    LeafStore<u32>,
    LevelBuffers<u32>,
    LevelBuffers<u32>,
    LeafStore<u32>,
);

fn stores<C: CommunicatorCollectives>(evaluator: &Evaluator<'_, C, PerPair<Mix>>) -> Stores {
    (
        evaluator.target_output_store().clone(),
        evaluator.multipoles().clone(),
        evaluator.locals().clone(),
        evaluator.source_store().clone(),
    )
}

#[test]
#[ignore = "the overlap stress case: by hand at 8 ranks, release"]
fn overlap_stress() {
    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    // The graded corner blob of `tests/mpi_regressions.rs`: a sparse cloud and a dense
    // blob in a different level-4 cell on every rank; the last rank passes no point.
    let cloud: Vec<[f64; 3]> = (0..48)
        .map(|i| {
            let j = i + 17 * rank;
            [
                ((13 * j + 7) % 97) as f64 / 97.0,
                ((29 * j + 3) % 89) as f64 / 89.0,
                ((11 * j + 5) % 83) as f64 / 83.0,
            ]
        })
        .collect();
    let corner = |n: usize| 0.005 + (n % 8) as f64 / 8.0;
    let origin = [corner(rank), corner(3 * rank + 1), corner(5 * rank + 2)];
    let blob: Vec<[f64; 3]> = (0..64usize)
        .map(|i| {
            let coordinate = |k: usize, d: usize| origin[d] + (k % 4) as f64 / 64.0;
            [
                coordinate(i, 0),
                coordinate(i / 4, 1),
                coordinate(i / 16, 2),
            ]
        })
        .collect();
    let empty = size > 1 && rank + 1 == size;
    let mut fine_keys = Vec::new();
    if !empty {
        fine_keys = keys(&cloud);
        fine_keys.extend(keys(&blob));
        let mut reversed = cloud.clone();
        reversed.reverse();
        fine_keys.extend(keys(&reversed));
    }
    let options = OctreeOptions::new()
        .with_max_level(6)
        .with_max_fine_keys(1)
        .with_ghost_children(true);
    let octree = Octree::new(&fine_keys, options, &comm);
    let plan = Plan::new(&octree).expect("the plan builds");

    // Seeded counts per leaf (zeros included) and values, from the key.
    let leaves = plan.index().leaves();
    let count =
        |key: u64, salt: u64| ((key ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize % 5;
    let sources: Vec<usize> = (0..leaves.nlocal())
        .map(|j| count(leaves.key(j), 0))
        .collect();
    let targets: Vec<usize> = (0..leaves.nlocal())
        .map(|j| count(leaves.key(j), 0x5bd1_e995))
        .collect();
    let mut evaluator = Evaluator::new(&plan, &comm, PerPair(Mix), &sources, &targets)
        .expect("the evaluator builds");
    for (j, chunk) in evaluator.local_sources_mut().chunks_mut().enumerate() {
        for (k, value) in chunk.iter_mut().enumerate() {
            *value = (leaves.key(j).wrapping_mul(64) + k as u64) as u32;
        }
    }
    for (j, chunk) in evaluator.local_target_inputs_mut().chunks_mut().enumerate() {
        for (k, value) in chunk.iter_mut().enumerate() {
            *value = (leaves.key(j).wrapping_mul(32) + k as u64) as u32;
        }
    }
    evaluator.evaluate();
    let blocking = stores(&evaluator);

    let mut differing = 0usize;
    let start = std::time::Instant::now();
    for _ in 0..EVALUATIONS {
        evaluator.evaluate_overlapped();
        differing += usize::from(stores(&evaluator) != blocking);
    }
    let elapsed = start.elapsed();
    let mut total = 0usize;
    comm.all_reduce_into(&differing, &mut total, SystemOperation::sum());
    let mut ghosts = [0usize; 2];
    let local = [
        leaves.ghosts().len(),
        (0..plan.nlevels())
            .map(|l| evaluator.multipole_exchange().receive_boxes(l).len())
            .sum(),
    ];
    comm.all_reduce_into(&local[..], &mut ghosts[..], SystemOperation::sum());
    if rank == 0 {
        println!(
            "overlap stress on {size} ranks ({}): {EVALUATIONS} overlapped \
             evaluations in {:.2} s, {} ghost leaves and {} ghost boxes over all ranks, {total} \
             evaluations differing from the blocking one on some rank",
            if size > 1 {
                format!("rank {} empty", size - 1)
            } else {
                "no empty rank on one rank".to_owned()
            },
            elapsed.as_secs_f64(),
            ghosts[0],
            ghosts[1]
        );
    }
    assert_eq!(
        total, 0,
        "rank {rank}: overlapped evaluations differ from the blocking one"
    );
}
