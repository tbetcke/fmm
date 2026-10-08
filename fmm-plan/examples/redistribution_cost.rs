//! Measure the time and traffic of `Redistribution::new` and of its `forward` and
//! `backward`, on any number of ranks.
//!
//! N points uniform in the unit cube (point i from a hash of i, so that every rank can
//! generate its own share), held either all on rank 0 or by a seeded random share. The
//! octree is built from the held points with leaves of at most 64 distinct keys, as
//! `Fmm` does. Reports per input distribution the items that change rank, the bytes
//! `new` sends (key and position) and `forward` sends (three `f64` per item), and the
//! median over the repeats of each call, the slowest rank's. Nothing is asserted. Run
//! in release mode, under an external timeout, for example:
//! `cargo build --release -p nd-fmm-plan --example redistribution_cost` and then
//! `timeout 600 mpirun -n 4 target/release/examples/redistribution_cost 1000000 5`
//! (on macOS with the loopback flags of the root `CLAUDE.md`).

use std::time::Instant;

use mpi::{collective::SystemOperation, traits::*};
use nd_fmm_plan::{plan::Plan, redistribute::Redistribution};
use nd_octree::{Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rlst::rlst_dynamic_array;

/// Point i of the cube: three coordinates from a hash of i, in (0, 1).
fn point(i: usize) -> [f64; 3] {
    let mut state = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x2545_f491_4f6c_dd1d;
    [0, 1, 2].map(|_| {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    })
}

/// The rank of point i in the seeded share.
fn share_rank(i: usize, nranks: usize) -> usize {
    ((i as u64 ^ 0x5bd1_e995).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize % nranks
}

fn keys(points: &[[f64; 3]]) -> Vec<u64> {
    let mut array = rlst_dynamic_array!(f64, [3, points.len()]);
    for (j, point) in points.iter().enumerate() {
        for i in 0..3 {
            array[[i, j]] = point[i];
        }
    }
    points_to_morton(
        &array,
        DEEPEST_LEVEL as usize,
        &PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
}

/// The median over `repeats` runs of `f` of the slowest rank's time, in ms; every rank
/// starts each run together.
fn timed<C: CommunicatorCollectives, T>(comm: &C, repeats: usize, mut f: impl FnMut() -> T) -> f64 {
    let mut times: Vec<f64> = (0..repeats)
        .map(|_| {
            comm.barrier();
            let start = Instant::now();
            let result = f();
            let local = start.elapsed().as_secs_f64() * 1e3;
            drop(result);
            let mut slowest = 0.0;
            comm.all_reduce_into(&local, &mut slowest, SystemOperation::max());
            slowest
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn main() {
    let universe = mpi::initialize().expect("MPI is initialised once");
    let comm = universe.world();
    let rank = comm.rank() as usize;
    let nranks = comm.size() as usize;
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(1_000_000, |a| a.parse().expect("N"));
    let repeats: usize = args.next().map_or(5, |a| a.parse().expect("repeats"));

    if rank == 0 {
        println!("redistribution cost: cube, N = {n}, {nranks} ranks, median of {repeats}");
        println!(
            "| input | moved items | moved per rank (max) | new: MB sent | new (ms) | forward: MB sent | forward, 3 f64 (ms) | backward, 1 f64 (ms) |"
        );
        println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    }
    for (label, all_on_zero) in [("all on rank 0", true), ("random share", false)] {
        let held: Vec<[f64; 3]> = (0..n)
            .filter(|&i| {
                if all_on_zero {
                    rank == 0
                } else {
                    share_rank(i, nranks) == rank
                }
            })
            .map(point)
            .collect();
        let keys = keys(&held);
        let options = OctreeOptions::new()
            .with_max_fine_keys(64)
            .with_ghost_children(true);
        let octree = Octree::new(&keys, options, &comm);
        let plan = Plan::new(&octree).expect("a plan of the octree");

        let new_ms = timed(&comm, repeats, || {
            Redistribution::new(&octree, &plan, &keys).expect("valid keys")
        });
        let redistribution = Redistribution::new(&octree, &plan, &keys).expect("valid keys");
        let coordinates: Vec<f64> = held.iter().flatten().copied().collect();
        let forward_ms = timed(&comm, repeats, || redistribution.forward(&coordinates, 3));
        let results = vec![1.0f64; redistribution.nreceived()];
        let backward_ms = timed(&comm, repeats, || redistribution.backward(&results, 1));

        // Items that change rank: received from another rank.
        let moved = redistribution
            .origins()
            .iter()
            .filter(|&&(origin, _)| origin as usize != rank)
            .count();
        let mut total = 0usize;
        let mut most = 0usize;
        comm.all_reduce_into(&moved, &mut total, SystemOperation::sum());
        comm.all_reduce_into(&moved, &mut most, SystemOperation::max());
        if rank == 0 {
            let mb = |bytes_per_item: usize| (total * bytes_per_item) as f64 / 1e6;
            println!(
                "| {label} | {total} | {most} | {:.1} | {new_ms:.1} | {:.1} | {forward_ms:.1} | {backward_ms:.1} |",
                mb(12),
                mb(24)
            );
        }
    }
}
