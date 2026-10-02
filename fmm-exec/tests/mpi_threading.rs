//! `FmmBuilder::threads` against the MPI threading level (C3.5): with MPI initialised at
//! `Threading::Single`, `threads(2)` gives `FmmError::MpiThreading` on every rank, and
//! `threads(1)` still builds and evaluates.
//!
//! Its own executable, because its one test initialises MPI at a level below the
//! `Threading::Funneled` of `tests/mpi_exec.rs`, and MPI is initialised once per process.
//! Error measure: exact equality of the errors and of the output bits.

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::fmm::{FmmBuilder, FmmError};
use nd_fmm_exec::threading::REQUIRED_MPI_THREADING;

#[test]
fn threads_need_funneled_mpi() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Single)
        .expect("this test owns MPI initialization");
    let comm = universe.world();
    assert!(
        provided < REQUIRED_MPI_THREADING,
        "MPI provides {provided:?} when asked for Single; the error path cannot be checked"
    );
    assert_eq!(mpi::environment::threading_support(), provided);

    // Every rank passes the same points and gets the same error: the check is local and
    // the same on every rank.
    let points: Vec<[f64; 3]> = (0..200)
        .map(|i| {
            let t = i as f64;
            [(0.37 * t).sin(), (0.71 * t).cos(), (0.13 * t).sin()]
        })
        .collect();
    let charges: Vec<f64> = (0..points.len())
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    for n in [2, 8] {
        let error = FmmBuilder::<f64>::new(4)
            .threads(n)
            .build(&points, &points, &comm)
            .err();
        assert_eq!(
            error,
            Some(FmmError::MpiThreading {
                required: Threading::Funneled,
                provided,
            }),
            "threads({n})"
        );
    }

    // One thread needs no more than Single.
    if comm.size() == 1 {
        let mut fmm = FmmBuilder::<f64>::new(4)
            .threads(1)
            .build(&points, &points, &comm)
            .expect("threads(1) builds with MPI at Single");
        assert_eq!(fmm.threading().threads, 1);
        assert_eq!(fmm.threading().mpi, provided);
        assert!(
            fmm.threading().warnings().is_empty(),
            "one thread warns of nothing"
        );
        let output = fmm.evaluate(&charges).expect("the FMM evaluates");
        assert_eq!(output.potential.len(), points.len());
        assert!(output.potential.iter().all(|v| v.is_finite()));
    }
    eprintln!(
        "rank {}: MPI at {provided:?}: threads(2) and threads(8) give MpiThreading, \
         threads(1) evaluates",
        comm.rank()
    );
}
