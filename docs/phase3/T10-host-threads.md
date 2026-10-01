# Phase 3 / T10 — nd-fmm-exec: host threading with rayon (C3.5, new)

Until now the Laplace host path runs serially on each rank, so the only parallelism is
across MPI ranks. This task threads it with rayon inside each rank. It is opt-in, and
the output is bit-identical to the serial path for every thread count.

The redesign makes this safe without `unsafe` and without atomics:
- The plan hands every level call a target-centric view (requirement 5 of
  docs/phase3/README.md), and T8 already executes each level target by target.
- Each target's output slice can therefore go to one thread, borrowed mutably.
- Each target's contributions keep the same order as in the serial loop, whichever
  thread computes them.

Read first: root CLAUDE.md, fmm-exec/CLAUDE.md, docs/phase3/README.md ("Requirements"
5, 6 and 8; "Design decisions": "Host execution", "Threading (T10)"),
docs/design/fmm-plan-redesign.md §6 (the host-path check), the T8 and T9 code, the docs
of `rayon::{ThreadPoolBuilder, ThreadPool::install, current_thread_index}` and of
rsmpi's threading levels (`mpi::Threading`, `mpi::initialize_with_threading`, and the
query for the level actually provided).

Do:
- Add `rayon = "1"` to [workspace.dependencies] and to nd-fmm-exec (`workspace = true`).
  No other crate gets rayon.
- `FmmBuilder::threads(n)`:
  - n ≥ 1, default 1; n = 0 is an error;
  - with n > 1, `build` creates a `rayon::ThreadPool` of n threads that the `Fmm` owns,
    and every level call runs inside `install`. The global pool is never configured or
    used;
  - with n = 1, no pool is built and the same loop bodies run on the calling thread.
- MPI threading:
  - worker threads never call MPI; all communication stays on the calling thread,
    between level calls, as the evaluator does it;
  - with n > 1, `build` checks that MPI provides at least `Threading::Funneled`, and
    otherwise returns `FmmError::MpiThreading { required, provided }` on every rank.
    The check is local and the same on every rank, so it needs no collective; say why
    in the code;
  - document in the crate docs that callers initialise MPI with
    `mpi::initialize_with_threading(Threading::Funneled)` to use threads.
- `LaplaceOperator`:
  - each level call iterates its targets in parallel, through the target-centric view;
  - level buffers are split with `par_chunks_mut(size)` (M2L, L2L and P2L into locals;
    M2M into parent multipoles; P2M per leaf);
  - leaf output chunks are split into disjoint mutable slices by the CSR offsets
    (L2P, M2P and P2P), for example by recursive `split_at_mut` under `rayon::join`.
    Allocate nothing per pair. If a level call needs one small allocation (a vector of
    slices), document it and measure it;
  - the body of the per-target loop is exactly the serial T8 body, so the order of
    contributions per target is unchanged;
  - per-thread scratch (`Workspace`, table scratch, the P2P mapping buffer), one set
    per pool thread, created at construction and selected by `current_thread_index()`.
    Use a safe container (for example a `Mutex` per set, uncontended by construction),
    never `unsafe`. Sizes come from the global maximum leaf occupancy, as in T8.
- Load balance: leaves differ widely in cost (P2P scales with the product of the two
  point counts). Use rayon's work stealing as it is. Report the balance, and leave
  `with_min_len` or cost-based splitting alone unless a measurement shows it is
  needed. If so, record it in the PR.
- Threads and BLAS (docs/phase3/README.md, "Threads and BLAS"):
  - Audit that no code reachable from a rayon worker calls BLAS or LAPACK: read the
    call paths of every level call (nd-fmm-exec, nd-fmm-tables apply functions,
    nd-fmm-ref), and check the release test binary and the `fmm_accuracy` binary for
    undefined BLAS symbols (`nm -u`, for example `dgemm_`, `sgemm_`, `cblas_dgemm`).
    Report both results in the PR.
  - `Fmm::threading()` returns a `ThreadingReport`: the rayon thread count, the MPI
    threading level provided, and the value (or absence) of `OPENBLAS_NUM_THREADS`,
    `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` and
    `VECLIB_MAXIMUM_THREADS`, read once in `build`. With more than one rayon thread, an
    unset variable or one above 1 is listed as a warning, not an error, because Phase 3
    calls no BLAS. Document that the first task that calls BLAS inside a worker turns
    the warning into an error or forces one BLAS thread.
  - Never set an environment variable from library or test code.
  - Crate docs: the rule, and a launch example with every variable set to 1, for
    `cargo run` and for `mpirun -x`.
- tests/mpi_exec.rs initialises MPI with `Threading::Funneled`. This is the only change
  to its initialisation.
- nd-fmm-validate: the `fmm_accuracy` example takes `--threads n`, initialises MPI
  with `Threading::Funneled`, and prints the `ThreadingReport` with its results.

Tests that define done:
- Bit-identity, the C3.5 criterion: every scenario of tests/mpi_exec.rs (T8, T9 and,
  once merged, T11) runs at 1, 2, 4 and 8 threads, in f32 and f64, every strategy,
  with gradients on and off. Every output is bit-identical to the 1-thread output.
- The ignored release accuracy test of T9 at 4 threads equals 1 thread bit for bit.
- No allocation per pair: capacity checks on every per-thread scratch set before and
  after a threaded evaluation, as in T8.
- Errors: `threads(0)` is rejected. In a separate test executable,
  tests/mpi_threading.rs, with its one MPI-initialising test at `Threading::Single`:
  `threads(2)` gives `FmmError::MpiThreading`, and `threads(1)` still works.
- `ThreadingReport`: with a variable unset, set to 1 and set to 4 (each in a child
  process started by the test with its own environment, since the test must not set
  variables in its own process), the report shows the value and the warning rule.
- Repeatability: two threaded evaluations are bit-identical, and so is a threaded
  evaluation after a serial one on the same `Fmm`.
- Multi-rank, by hand, under an external timeout: tests/mpi_exec.rs on 2 ranks with
  2 threads per rank neither hangs nor diverges.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug run under a minute;
  keep thread counts above 4 to a few small scenarios if needed);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- `cargo run --release -p nd-fmm-validate --example fmm_accuracy -- --threads n` for
  n = 1, 2, 4 and 8, and the number of physical cores if larger;
- the 2-rank run above;
- `cargo clippy -p nd-fmm-exec -p nd-fmm-validate --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-exec --no-deps` without warnings;
- the root checks.

Run every timing with all five BLAS variables set to 1 in the launching shell, and say
so. Report, from the example: the machine (CPU model, physical and logical cores), the
`ThreadingReport`, the BLAS audit, and for
N = 10⁵ on the uniform cube at p = 3, 8 and 18 the time per stage and in total at each
thread count, with speed-up and parallel efficiency against 1 thread. Note which
stages scale poorly and why (for example memory bandwidth in dense M2L at high p, or
load imbalance in P2P). Do not assert timings.

Do not:
- use `unsafe`, atomics or locks shared between targets;
- call BLAS or LAPACK inside a rayon worker, or set an environment variable;
- configure or use the global rayon pool, or read `RAYON_NUM_THREADS`;
- change the accumulation order or the serial results of T8 and T9;
- thread anything in nd-fmm-plan, nd-fmm-tables or nd-fmm-ref, or make worker threads
  call MPI;
- change the default of one thread.

If bit-identity fails, find the operator whose order differs, report it and stop.
