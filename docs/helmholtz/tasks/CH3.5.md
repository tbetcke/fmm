# CH3.5 The shared drivers extended to complex expansions

Started 2026-11-01 on `tbetcke/h-ch3-5`. Roadmap: "CH3.5 The shared drivers extended to
complex expansions" (H3; ninth on the priority list). Proposal: §10 CH3.5; §2
(constraint 1), §3.1, §3.3, §5.2, §6.2, §6.3.

Closed as a record, without a code change (the user, 2026-11-01: "Close as a record").
The finding is below. The batched M2L's two-plane case moves to CH3.2, its first user.

## Goal

Have the shared host drivers of CH2.0 (`fmm-exec/src/driver.rs`) carry the complex
expansions of the Helmholtz operator: two planes, real shifts for real k, and the
epilogue. Laplace's paths must stay as constraint 1 requires.

## Acceptance (proposal §10, verbatim)

> constraint 1 of Section 2, as CH2.0; else the complex operator gets its own drivers

## Read first

- `fmm-exec/src/driver.rs`: `HostKernel`, `Execution`, the level calls and the target
  bodies, and `m2l_block`.
- `fmm-exec/src/operator.rs`: `Kernels`' `HostKernel` impl and `LaplaceOperator`'s
  `FmmSizes`.
- `fmm-plan/src/evaluator.rs`: `level_sizes`, `LevelBuffers::from_index` and the
  exchanges' `multipole_sizes`.
- `fmm-ref/src/helmholtz/rcr.rs`: the RCR operators a Helmholtz kernel's bodies are
  built from.
- `fmm-exec/src/helmholtz/densities.rs`: the box and chunk layouts of CH3.0c.

## The finding

What the proposal's three items need from the drivers, item by item:

1. **Two planes.** The target-by-target drivers know nothing of a box's layout. A level
   call takes its chunk size from the plan's store (`LevelSliceMut::size`), and that
   store is sized by the operator's `multipole_size(level)` and `local_size(level)`
   (`nd_fmm_plan::evaluator::level_sizes`). Each kernel body then receives the whole
   block of one box: `m2m(level, o, child, parent, …)`, `m2l(level, d, child, multipole,
   local, …)`, and so on. So a Helmholtz box block (m complex expansions of two planes
   each; for real k with real charges, multipoles of (p + 1)² m and locals of
   2 (p + 1)² m) passes through the drivers as Laplace's does. The plan has supported
   multipole and local sizes that differ from each other and by level since its redesign,
   and tests them: `fmm-plan/src/evaluator_tests.rs` (sizes 1 + l and 2 + l mod 3),
   `operator_tests.rs` and `tests/mpi_regressions.rs`. The exchanges use the multipole
   sizes per level.
2. **Real shifts for real k.** In RCR, M2M and L2L are a real rotation, a real coaxial
   step (for real k, §3.1) and the rotation back, applied to both planes of each
   expansion. That all happens inside the kernel's `m2m` and `l2l` bodies, on the block
   the driver passes. Laplace's Rotation likewise lives in its kernel (`Tables`), not in
   the driver. Nothing in the drivers changes.
3. **The epilogue.** Proposal §6.2 places it in Dense and Batched. There the products with
   [A_r; A_i] give four real products per pair, which are accumulated over a target's
   offsets and combined once per target. Target by target (Dense, Rotation), a kernel's
   `m2l` body applies the complex step per pair, into the target's two planes, and needs
   no hook. Only the batched block (`driver::m2l_block`) shares the arithmetic across
   pairs, and it is the one driver code that assumes Laplace's layout. It takes square
   n × n tables (`MatrixSet::n`) and m expansions of n values per box, multipole and
   local alike (the column offsets `(s m + v) n`, the block end `locals.len() / (m n)`).
   A kernel with another layout returns `None` from `HostKernel::batched_m2l`, and its
   M2L then runs target by target.

So the complex RCR operator of CH3.1, the first deliverable of H3, runs on the unchanged
drivers. Constraint 1 holds trivially: no Laplace code, layout, table or output changes,
so no D9 runs are needed. The two-plane case of the batched block has no user until
CH3.2 (Dense, Classes and Batched for Helmholtz), which is off the priority list. Under
§6.2's rule of two it is built there: `PLANES` (an associated constant, 1 for Laplace),
rectangular real-to-complex tables, and the per-target epilogue. That PR carries
constraint 1 as CH2.0 did, with D9 on the three machines for the Batched rows.

## Changes

Documentation only:
- `fmm-exec/src/driver.rs`: the docs of `HostKernel` (a box block's layout is the
  kernel's; the batched M2L's assumptions; `None` for any other layout) and of `Columns`
  (unused by a kernel without a batched M2L). No code line changes.
- `fmm-exec/CLAUDE.md`: the same in the rule on the shared drivers.
- `docs/helmholtz/roadmap.md`: CH3.5 ticked with this record; the batched two-plane case
  added to CH3.2's *Done when*.
- `docs/helmholtz/progress.md`: the entry.

No public API, manifest, convention, table, layout or `nd-fmm-plan` change.

## Tests that prove it

No new test: no code changes. The evidence for the finding:
- the plan's tests with differing and level-dependent multipole and local sizes (above);
- CH3.1, whose Helmholtz kernel runs on these drivers and is checked against the direct
  sum (its acceptance). If CH3.1 finds a driver change it needs, the change is made there
  under constraint 1.

## Checks

`/verify` (the branch changes documentation and a Rust file's doc comments): format;
Clippy default, examples and workspace all-targets; `cargo test`; `cargo doc`. The
comment-only change to `driver.rs` cannot change Laplace's bits or timings.

## Machine runs

None.

## Constraints and risks

- Constraint 1: Laplace untouched. Its compiled code is unchanged (comments only).
- Risk: CH3.1 meets a driver assumption missed here. For example, P2L, M2P and P2P give
  the kernel the target's level and a frame. A Helmholtz body needs the source leaf's
  radius for k̂ = k r_s, and it takes that from the frame's radius r̂, as the trait's docs
  say. Any such change is CH3.1's, under constraint 1.

## Pending runs

None.

## Open questions

None open. The scope was decided by the user on 2026-11-01 ("Close as a record").
