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
//! # Ranks × threads on a node (Phase 5N)
//!
//! On a node with many cores, run **a few ranks with threads**: one rank per socket or
//! NUMA domain, or a few per socket, with ranks × threads × BLAS threads within the
//! physical cores (docs/design/distributed-fmm.md §14.4, S6, measured in Phase 5N T3,
//! docs/design/node-m2l.md §7). Threads share their rank's tables, tree and buffers;
//! ranks each hold their own, so every replicated per-rank structure (the M2L tables,
//! the top tree, distributed-fmm.md §9.2) grows with the ranks, not the cores. Measured
//! at N = 10⁶, f64 p = 8, the cube (release, every BLAS variable at 1):
//!
//! | machine | best split | its time | one rank per core |
//! | --- | --- | ---: | ---: |
//! | Apple M3 Max, 12 performance cores | 1 × 12 | 607 ms | 12 × 1: 613 ms |
//! | locust (GH200), 72 cores, one NUMA domain | 8 × 9 | 143 ms | 72 × 1: 160 ms |
//! | Kathleen, 2 × 20 Cascade Lake cores | 4 × 10 | 573 ms | 40 × 1: 638 ms |
//!
//! (Rotation in every cell, the strategy [`M2lStrategy::Auto`](crate::tables::M2lStrategy)
//! now resolves to at f64 p = 8.) With Dense, which `Auto` keeps where the tables are small
//! (f32 p = 8, f64 p ≤ 6), the gap is larger, since the threads of a rank share one copy
//! of the tables: 481 ms at 1 × 12 against 1,268 at 12 × 1 on the M3 Max, 108 against 388
//! at 8 × 9 and 72 × 1 on locust, 392 against 788 at 4 × 10 and 40 × 1 on Kathleen
//! (node-m2l.md §7). With one rank per core the strategy rule
//! ([`M2lStrategy::resolve_on_node`](crate::tables::M2lStrategy::resolve_on_node)) keeps
//! the default within 10 % of the best strategy at p = 8, but not the tree and buffers per
//! rank. Kathleen's step from 20 to 40 ranks (1.4× in every stage, phase5n-kathleen.md
//! §1) is the all-core clock and is paid by every split. [`ThreadingReport`] gives the
//! ranks per node and the strategy's reason for every report.
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

use crate::operator::{Isa, MulAdd};
use crate::tables::StrategyChoice;

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
/// [`FmmBuilder::build`](crate::fmm::FmmBuilder::build), and the node context of the M2L
/// strategy (Phase 5N T4): the ranks per node and the strategy with its reason; see the
/// [module documentation](self).
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
    /// The ranks per node, the largest over the nodes of the run, which every rank agrees
    /// (one `split_shared` and the all-reduce of `build`'s step 7); 1 from
    /// [`read`](Self::read).
    pub ranks_per_node: usize,
    /// The M2L strategy of the build and why it was chosen
    /// ([`M2lStrategy::choose`](crate::tables::M2lStrategy::choose)); `None` from
    /// [`read`](Self::read), set by `build`.
    pub m2l: Option<StrategyChoice>,
    /// The ISA and multiply–add mode of the batched M2L's column kernel under
    /// `M2lStrategy::Batched` (Phase 6 T2; on a device backend it runs only where M2L is
    /// on the host fallback); `None` under every other strategy and from
    /// [`read`](Self::read).
    pub m2l_kernel: Option<(Isa, MulAdd)>,
}

impl ThreadingReport {
    /// Reads the [`BLAS_VARIABLES`] from this process's environment, for `threads` rayon
    /// threads and the MPI level `mpi`, with one rank per node and no strategy. Reads
    /// only; never sets a variable.
    pub fn read(threads: usize, mpi: Threading) -> Self {
        Self {
            threads,
            mpi,
            blas: BLAS_VARIABLES.map(|name| BlasVariable {
                name,
                value: std::env::var_os(name).map(|value| value.to_string_lossy().into_owned()),
            }),
            ranks_per_node: 1,
            m2l: None,
            m2l_kernel: None,
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
    /// OMP_NUM_THREADS unset (warning), …; ranks per node 12; M2L Auto → Rotation: 12
    /// ranks per node × 8.7 MB > 64 MB` (the M2L part only once `build` set it), or
    /// `…; M2L Batched (named), batched on neon, separate`.
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
        write!(f, "; ranks per node {}", self.ranks_per_node)?;
        if let Some(choice) = &self.m2l {
            write!(f, "; M2L {choice}")?;
        }
        if let Some((isa, mul_add)) = self.m2l_kernel {
            write!(f, ", batched on {isa}, {mul_add}")?;
        }
        Ok(())
    }
}
