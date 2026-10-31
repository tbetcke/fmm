# CH2.0 Laplace's real host drivers made generic over the kernel

Started 2026-10-30 on `tbetcke/h-ch2-0`. Roadmap: "CH2.0 Laplace's real host drivers made
generic over the kernel" (H2; sixth on the priority list). Proposal: §10 CH2.0; §2
(constraint 1), §6.1, §6.2, §6.3; D9 (signed off 2026-10-26, clarified the same day).

## Goal

The host level calls of `LaplaceOperator` (the batch walking, the threading and scratch,
the charge vectors, the batched M2L's blocks) moved into drivers generic over a kernel
trait, with Laplace as the first kernel and its outputs and benchmarks unchanged, so that
the Helmholtz-family operators (CH2.1 Yukawa, CH3.1 complex) can run on them.

## Acceptance (proposal §10, verbatim)

> constraint 1 of Section 2: no new dispatch below the level call, layouts and tables
> unchanged, Laplace outputs bit for bit unchanged, Laplace benchmarks within noise on the
> M3 Max, locust and Kathleen; the pull request states how p reaches Laplace's inner loops
> today and how the trait supplies it; else dropped and Yukawa gets copies

## Read first

- `fmm-exec/src/operator.rs`: `LaplaceOperator`, `Kernels`, `Scratch`, `Execution`,
  `BoxCall`, `LeafCall`, the `*_target` bodies, `m2l_block`, the `FmmOperator` impl.
- `fmm-exec/src/operator_plane_wave.rs` (the plane-wave level call, which stays Laplace's).
- `fmm-exec/src/tables.rs` (`Tables`, the per-pair dispatch over `Families` that exists
  today), `fmm-simd/src/matrix.rs` (`MatrixKernel::accumulate_columns`).
- `fmm-exec/CLAUDE.md` ("Several charge vectors", "Threading", the batched M2L).

## How p reaches Laplace's inner loops today

At run time, never as a compile-time constant. `Kernels::p` (a `usize` field, fixed at
construction) goes to `nd_fmm_ref::leaf::*_many(p, …)` for the leaf operators. It sets
`each_vector`'s chunk length (p + 1)². The tables carry their own p: `M2mTables`,
`L2lTables`, `M2lTables`, `M2lClasses` and `RotationTables` apply with n = (p + 1)² or p
read from themselves, and `MatrixKernel::accumulate_columns` takes n as an argument. No
host loop of Laplace is specialised on p (no const generic over p in `nd-fmm-exec`,
`nd-fmm-tables`, `nd-fmm-ref`'s leaf operators or `nd-fmm-simd`'s matrix kernel; checked
by grep). The strategy is also a run-time choice: `Tables` matches on its `Families` enum
per pair. That dispatch exists today and is not new.

**How the trait supplies it:** the trait's methods take the level call's `level`. Laplace
ignores it and keeps reading its one p from `Kernels` and the tables, so its arithmetic
gets the same runtime values as today. A Helmholtz kernel reads p_l and its tables from
`level`. No const generic is introduced.

## Changes

`nd-fmm-exec` only, crate-private. No public API change, no manifest change, no
convention, table, layout or `nd-fmm-plan` change.

- `src/driver.rs` (new, `pub(crate)`):
  - `trait HostKernel<T: SimdScalar>: Sync` with `type Scratch: Send`. Its methods are the
    per-box and per-leaf bodies the drivers call today, each with the level call's level:
    - `p2m(level, sources, multipole, scratch)`;
    - `m2m(level, o, child, parent, scratch)`;
    - `m2l(level, d, child, multipole, local, scratch)`;
    - `l2l(level, o, parent, child, scratch)`;
    - `p2l(level, frame, sources, local, scratch)`;
    - `l2p(level, local, target_input, target_output, scratch)`;
    - `m2p(level, frame, multipole, target_input, target_output, scratch)`;
    - `p2p(level, frame, sources, target_input, target_output, scratch)`.

    Plus the batched M2L's hooks: `batched_m2l(level)` returns the matrix set and column
    kernel, or `None`; `columns(scratch)` returns the column lists and packing buffer.
    Plus `vectors()`.
  - `Execution<S>`, `split_leaves`, `split_parents`, `with_scratch`, moved from
    `operator.rs` and generic over the scratch type `S` instead of `Scratch<T>`. Bodies
    unchanged.
  - `BoxCall`, `LeafCall` and the target bodies `p2m_target` … `p2p_target`, `m2l_block`:
    moved, generic over `K: HostKernel<T>`, calling `K`'s methods with `call.level`.
    Bodies unchanged otherwise.
  - The level calls as generic functions `driver::{p2m, m2m, m2l, p2l, l2l, l2p, m2p,
    p2p}(kernel, execution, batch)`. Each is today's `FmmOperator` method body for that
    kind; M2L picks blocks (when `batched_m2l` is `Some`) or target by target, as now.
- `src/operator.rs`:
  - `Kernels<T>` implements `HostKernel<T>` with `Scratch = Scratch<T>`: each trait
    method ignores the level and calls the inherent method of the same name, which is
    unchanged (as built; the per-pair path keeps calling the inherent methods).
  - `LaplaceOperator<T>` holds `Execution<Scratch<T>>`. Its `FmmOperator` methods call the
    drivers. The `PlaneWave` M2L branch stays in `LaplaceOperator::m2l`, before the
    driver call: proposal §6.2 shares no plane-wave or diagonal form.
  - The module documentation gains a short section on the drivers.
- `src/operator_plane_wave.rs`: imports only (`BoxCall` from `driver`).
- `fmm-exec/CLAUDE.md`: one rule on the drivers (where they live, what the trait is,
  that Laplace's bits and benchmarks guard them).

Not in CH2.0:
- `PLANES` and `REAL_SHIFTS` of proposal §6.2's sketch: these are CH3.5's, the complex
  case.
- A Yukawa kernel: CH2.1 builds one on these drivers, with its per-level tables.
  Rule of two: the trait is shaped by CH2.1's and CH3.1's needs (per-level degree, sizes
  and tables through `level`), and its second user comes with CH3.1 on the priority list
  (CH3.5, then CH3.1), or with CH2.1.
- The device path: `device.rs` uses `LaplaceOperator`'s public API (host fallback) and
  `Tables`, both unchanged.

## Tests that prove it

Laplace bit for bit (constraint 1, condition 3):
- every existing `nd-fmm-exec` test, unchanged:
  - the operator tests against `nd-fmm-ref` (`tests/operator/`);
  - `tests/mpi_exec.rs`: threads 1, 2, 4 and 8 bit for bit, overlap, `Batched` against
    `Dense` bit for bit, every ISA against `Reference`, vectors, dipoles, the one-rank
    reference, at 1, 2, 4 and 8 ranks;
  - `tests/invariance.rs` at 1, 2, 4 and 8 ranks;
  - the ignored `tests/multi_rank.rs` at 2, 4 and 8 ranks in release;
  - the ignored `accuracy.rs` and `adaptive.rs` gates in release.
- The outputs of `main` and the branch compared byte for byte, on each machine: `nd-fmm-bench
  --write-problem` once, then `--write-output` from both binaries, `cmp`. The comparison
  covers host, f32 and f64, p = 3, 6 and 8, and every strategy (dense, classes, rotation,
  batched, planewave, auto). This is the direct check of "outputs unchanged" on each ISA
  (NEON on the M3 Max and locust, AVX-512 on Kathleen).

No new test of the trait itself: its only implementor in this PR is Laplace, which every
test above runs through it.

## Checks

- `/verify` (format, Clippy default, examples and workspace all-targets, the default and
  workspace tests, `cargo doc`, the MPI tests at 2 and 4 ranks, the Laplace protection).
- `nd-fmm-exec`'s own: `cargo clippy -p nd-fmm-exec --all-targets -- -D warnings`;
  `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings` and
  `cargo check -p nd-fmm-exec --features cuda` (the device path builds on the changed
  operator); `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`;
  `tests/mpi_exec.rs` at 8 ranks and `tests/multi_rank.rs` at 2, 4 and 8 ranks in release,
  by hand under a timeout.

## Machine runs

D9 on each machine, host only. Five runs per side in the order A B B A A B B A A B, `main`
(A) in a detached worktree with its own target directory. Each run is `tools/bench/run.sh`
at N = 10⁶ with 10 timed evaluations and 2 warm-ups. Per benchmark, the spread is
max − min of `main`'s five run medians. The branch fails only if its median of medians is
slower than `main`'s by more than that spread. Faster passes when the bits are identical.

Benchmarks (12 per machine):
- `--strategy auto`, f32 and f64, p = 3, 6, 8 (Batched at 3 and 6, PlaneWave at 8);
- `--strategy dense`, `classes`, `rotation`, f32 and f64, p = 6.

Machines:
- M3 Max: outside the sandbox, on a quiet machine, with the user's go-ahead.
- locust: `tools/gh200/sync.sh`, then `remote.sh`. Check idle first; detached worktree
  `detached-<main sha>`.
- Kathleen: one `singlenode` job that builds both trees and alternates them.

On each machine, the `cmp` of the outputs above, from the same binaries.

Not re-measured:
- the device gates and the device benchmarks (no device code changes; the default device
  evaluation does not call the host drivers);
- the C3S.4 P2P throughput gate (`nd-fmm-simd` unchanged);
- the tuner's and node-m2l.md's cost constants (re-measured only if a benchmark moves).

## Constraints and risks

Constraint 1 of proposal §2 binds every line:
1. No new runtime dispatch: `K` is a type parameter, monomorphised. The level argument is
   a dead value for Laplace. Strides and sizes stay what they are (the runtime p of today).
2. Layouts and tables: untouched.
3. Bits: the bodies move verbatim, and the order of operations is unchanged. Tested as
   above.
4. Benchmarks: D9 as above.

Risks:
- Inlining across the new generic boundary: today's `#[inline]` on the bodies is kept. A
  regression would show in the benchmarks (the M2L and P2P per-kind columns of the bench
  report localise it).
- If condition 4 fails and cannot be fixed: the roadmap's fallback, Laplace left as it
  was (this PR closed, not merged) and the Helmholtz operators get their own drivers.

Constraint 2 (determinism, rank invariance): the drivers are the same code, so it holds
as before. Constraints 3–9: not affected.

## Pending runs

(filled in step 6)

## Open questions

None open. The user, 2026-10-30: "Proceed as planned" (Laplace the only kernel in this
PR) and "12 host benchmarks" (the set above, host only).
