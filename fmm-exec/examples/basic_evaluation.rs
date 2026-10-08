//! How to call the FMM: build it once from source and target points, evaluate several
//! charge vectors, and check a few potentials against the direct sum.
//!
//! ```text
//! cargo run --release -p nd-fmm-exec --example basic_evaluation
//! mpirun -n 3 target/release/examples/basic_evaluation
//! ```
//!
//! Each rank passes its own points and charges, and gets the potentials of its own
//! targets back, in its own order. Until points are redistributed (C5.1), a build on
//! several ranks whose points lie in leaves other ranks own is refused on every rank with
//! `FmmError::PointsNotOwned`; the example reports that and exits normally, so that it
//! runs on any number of ranks (the weekly `run-examples` job uses three).

use std::f64::consts::PI;

use mpi::Threading;
use mpi::traits::Communicator;
use nd_fmm_exec::fmm::{FmmBuilder, FmmError, Output, StageTimings};

/// Points per rank.
const N: usize = 100_000;

fn main() {
    // MPI is required, also on one rank. `Funneled` lets the FMM use threads.
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    let (rank, size) = (comm.rank(), comm.size());

    // This rank's points in the unit cube, used as sources and targets, and its charges.
    // Coordinates are always f64, whatever the FMM's precision.
    let mut uniform = xorshift(0x2545_f491_4f6c_dd1d ^ rank as u64);
    let points: Vec<[f64; 3]> = (0..N).map(|_| [uniform(), uniform(), uniform()]).collect();
    let charges: Vec<f64> = (0..N).map(|_| uniform() - 0.5).collect();

    // Ranks × threads within the cores (the threading rules of `nd_fmm_exec::threading`).
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads = (cores / size as usize).clamp(1, 4);

    // Build once: degree 8, with gradients; every other setting at its default (the host
    // backend, 64 points per leaf, the M2L strategy chosen from p). `build` is collective.
    let built = FmmBuilder::<f64>::new(8)
        .gradients(true)
        .threads(threads)
        .build(&points, &points, &comm);
    let mut fmm = match built {
        Ok(fmm) => fmm,
        Err(FmmError::PointsNotOwned { count }) => {
            if rank == 0 {
                println!(
                    "{size} ranks: {count} points lie in leaves other ranks own; the FMM does \
                     not redistribute points yet (C5.1), so every rank stops here"
                );
            }
            return;
        }
        Err(error) => panic!("rank {rank}: build failed: {error}"),
    };
    if rank == 0 {
        println!(
            "built on {size} rank(s), {threads} thread(s) each: {} leaves on {} levels, \
             M2L {:?}, in {:?}",
            fmm.nleaves(),
            fmm.nlevels(),
            fmm.strategy(),
            fmm.build_timings().total()
        );
    }

    // Evaluate (collective): φ and ∇φ at this rank's targets, in the order of `points`.
    let output = fmm
        .evaluate(&charges)
        .unwrap_or_else(|error| panic!("rank {rank}: evaluate failed: {error}"));
    let gradient = output.gradient.as_ref().expect("built with gradients");
    println!(
        "rank {rank}: φ(x₀) = {:.6e}, ∇φ(x₀) = {:?}",
        output.potential[0], gradient[0]
    );

    // On one rank, check a few targets against φ(x) = Σ q / (4π |x − y|); a source at the
    // target itself does not act on it.
    if size == 1 {
        for i in [0, N / 2, N - 1] {
            let direct = direct_potential(&points, &charges, i);
            let error = (output.potential[i] - direct).abs() / direct.abs();
            println!(
                "target {i}: fmm {:.9e}, direct {direct:.9e}, relative error {error:.1e}",
                output.potential[i]
            );
            assert!(error < 1e-3, "target {i}: relative error {error:.1e}");
        }
    }

    // Further charge vectors reuse the tree and the tables; `evaluate_into` also reuses
    // the output's buffers and gives the bits `evaluate` would.
    let mut reused = Output {
        potential: Vec::new(),
        gradient: None,
        timings: StageTimings::default(),
    };
    for k in 1..=3 {
        let scaled: Vec<f64> = charges.iter().map(|q| q * k as f64).collect();
        fmm.evaluate_into(&scaled, &mut reused)
            .unwrap_or_else(|error| panic!("rank {rank}: evaluate failed: {error}"));
        if rank == 0 {
            let t = &reused.timings;
            println!(
                "charges × {k}: φ(x₀) = {:.6e}; downward {:?}, leaves {:?}, output {:?}",
                reused.potential[0], t.downward, t.evaluate_leaves, t.output
            );
        }
    }
}

/// A uniform generator on [0, 1) (xorshift64), so that the example needs no dependency.
fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// φ at point `i` from every other point: Σ_{j ≠ i} q_j / (4π |x_i − x_j|).
fn direct_potential(points: &[[f64; 3]], charges: &[f64], i: usize) -> f64 {
    let x = points[i];
    points
        .iter()
        .zip(charges)
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(_, (y, q))| {
            let r = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt();
            q / (4.0 * PI * r)
        })
        .sum()
}
