//! Threads, MPI and BLAS: the rules of the host threading (C3.5) and the
//! [`ThreadingReport`] of an [`Fmm`](crate::fmm::Fmm).
//!
//! # Threads
//!
//! [`FmmBuilder::threads(n)`](crate::fmm::FmmBuilder::threads) runs every level call of
//! the operator on n threads of a rayon pool that the `Fmm` owns
//! ([`LaplaceOperator::with_pool`](crate::operator::LaplaceOperator::with_pool)). The
//! default is one thread and no pool, so that several MPI ranks per node do not
//! oversubscribe the cores by default. The global rayon pool is never configured or
//! used, and `RAYON_NUM_THREADS` is not read. The output is bit-identical for every n.
//!
//! # MPI
//!
//! Worker threads never call MPI: the evaluator runs every exchange on the calling
//! thread, between level calls, and a level call only computes. MPI must therefore
//! provide at least [`Threading::Funneled`] when n > 1, and
//! [`FmmBuilder::build`](crate::fmm::FmmBuilder::build) returns
//! [`FmmError::MpiThreading`](crate::fmm::FmmError::MpiThreading) otherwise. Initialise
//! MPI with
//!
//! ```no_run
//! use mpi::Threading;
//!
//! let (universe, provided) =
//!     mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
//! assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
//! ```
//!
//! `mpi::initialize()` asks for [`Threading::Single`], with which only `threads(1)` builds.
//!
//! # BLAS
//!
//! A matrix product called from inside a rayon worker would start its own BLAS threads
//! on every worker, and two nested pools that both fill the machine oversubscribe it. The
//! rule, for this and every later phase (docs/phase3/README.md, "Threads and BLAS"):
//!
//! - ranks × rayon threads × BLAS threads (and any other pool) stay at or below the
//!   physical cores;
//! - a matrix product inside a rayon worker runs single-threaded. Large products outside
//!   rayon may use BLAS threads.
//!
//! Phases 3 and 3S call no BLAS or LAPACK routine from a worker, or anywhere in a level call:
//! the tables are applied by the hand-written loops of `nd-fmm-tables`, the leaf
//! operators are the plain Rust of `nd-fmm-ref`, P2P is the `core::arch` code of
//! `nd-fmm-simd` (or `nd-fmm-ref`'s, with `P2pChoice::Reference`), and `nd-fmm-plan`
//! uses `rlst` only in its distributed tools, outside the level calls.
//!
//! BLAS threads are set by the launcher, before the process starts, through
//! [`BLAS_VARIABLES`]: OpenBLAS reads them once at initialisation, so setting them later
//! has no effect, and Rust 2024 makes `std::env::set_var` unsafe. This crate never sets
//! an environment variable; it reads them once in `build` and reports them
//! ([`ThreadingReport`]). With more than one rayon thread, a variable that is unset or not
//! 1 is a [warning](ThreadingReport::warnings), not an error, because no BLAS routine runs
//! in a worker. The first task that calls BLAS inside a worker must turn the warning into
//! an error or force one BLAS thread (`rlst::threading::set_blas_threads`).
//!
//! A launch with every variable at 1, with `cargo run`:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example fmm_accuracy -- --threads 4
//! ```
//!
//! and with Open MPI, which passes each variable to every rank with `-x`, here two ranks
//! of four threads each:
//!
//! ```text
//! mpirun -n 2 -x OPENBLAS_NUM_THREADS=1 -x OMP_NUM_THREADS=1 -x MKL_NUM_THREADS=1 \
//!     -x BLIS_NUM_THREADS=1 -x VECLIB_MAXIMUM_THREADS=1 ./my_fmm_program --threads 4
//! ```

use std::fmt;

use mpi::Threading;

/// The MPI threading level that more than one thread requires: the pool's threads never
/// call MPI.
pub const REQUIRED_MPI_THREADING: Threading = Threading::Funneled;

/// The environment variables that set the number of BLAS threads: OpenBLAS, OpenMP
/// (also OpenBLAS built with OpenMP), MKL, BLIS, and Accelerate on macOS.
pub const BLAS_VARIABLES: [&str; 5] = [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
];

/// One of [`BLAS_VARIABLES`] and its value when it was read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlasVariable {
    /// The variable's name.
    pub name: &'static str,
    /// Its value, or `None` if it is unset. A value that is not Unicode is converted
    /// lossily.
    pub value: Option<String>,
}

impl BlasVariable {
    /// Returns whether the variable is set to 1 (surrounding whitespace ignored).
    pub fn is_one(&self) -> bool {
        self.value
            .as_deref()
            .is_some_and(|value| value.trim().parse::<u64>() == Ok(1))
    }
}

/// How an [`Fmm`](crate::fmm::Fmm) runs: its rayon threads, the MPI threading level
/// provided, and the BLAS thread variables, read once by
/// [`FmmBuilder::build`](crate::fmm::FmmBuilder::build); see the [module
/// documentation](self).
///
/// Its `Display` is one line, for reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadingReport {
    /// The number of rayon threads of the level calls; 1 means no pool.
    pub threads: usize,
    /// The threading level MPI provides (`mpi::environment::threading_support`).
    pub mpi: Threading,
    /// The [`BLAS_VARIABLES`], in that order.
    pub blas: [BlasVariable; 5],
}

impl ThreadingReport {
    /// Reads the [`BLAS_VARIABLES`] from this process's environment, for `threads` rayon
    /// threads and the MPI level `mpi`. Reads only; never sets a variable.
    pub fn read(threads: usize, mpi: Threading) -> Self {
        Self {
            threads,
            mpi,
            blas: BLAS_VARIABLES.map(|name| BlasVariable {
                name,
                value: std::env::var_os(name).map(|value| value.to_string_lossy().into_owned()),
            }),
        }
    }

    /// Returns the variables that would let BLAS oversubscribe the cores: with more than
    /// one rayon thread, every variable that is unset or not 1; with one thread, none.
    ///
    /// They are warnings, not errors: Phase 3 calls no BLAS routine inside a worker
    /// ([module documentation](self#blas)).
    pub fn warnings(&self) -> Vec<&BlasVariable> {
        if self.threads <= 1 {
            return Vec::new();
        }
        self.blas
            .iter()
            .filter(|variable| !variable.is_one())
            .collect()
    }
}

impl fmt::Display for ThreadingReport {
    /// For example `rayon threads 4; MPI Funneled; OPENBLAS_NUM_THREADS=1,
    /// OMP_NUM_THREADS unset (warning), …`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rayon threads {}; MPI {:?}; ", self.threads, self.mpi)?;
        let warnings = self.warnings();
        for (k, variable) in self.blas.iter().enumerate() {
            if k > 0 {
                f.write_str(", ")?;
            }
            match &variable.value {
                Some(value) => write!(f, "{}={value}", variable.name)?,
                None => write!(f, "{} unset", variable.name)?,
            }
            if warnings.contains(&variable) {
                f.write_str(" (warning)")?;
            }
        }
        Ok(())
    }
}
