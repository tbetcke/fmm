# CH4.2b1 The diagonal M2L in the operator: the level call and the rule

Started 2026-11-20 on `tbetcke/h-ch4-2b1`. Roadmap: CH4.2b1 (CH4.2b split on 2026-11-20 into
CH4.2b1, the operator, and CH4.2b2, W1 measured on Kathleen and locust, by the size rule of
`/next-task`: a new operator plus measurements on several machines). Proposal: §10 CH4.2; §3.6
(the bound and the rule), §5.4 (signatures, "Scratch and chunking"), §6.3, §6.4, §7, §2
(constraints 1, 2, 7, 9); D3 (f32 scope).

## Goal
The high-frequency levels' M2L of `HelmholtzOperator` by CH4.2a's transfer functions, opt-in:
each offset class on the diagonal form where E(l, c) ≤ θ ε, RCR on the rest, in a threaded
level call whose output is the same bits at every thread count, chunk size and rank count.

## Acceptance (proposal §10 CH4.2, verbatim)
"against RCR on the same class within E(l, c); determinism with the RCR pairs added after
F → R". CH4.2a showed both at the table level. The roadmap's CH4.2b: "the FMM against the direct
sum on W1 within RCR's errors (1.1×); every output bit for bit at every thread count, chunk size
and rank count (constraint 2)"; CH4.2b1 shows the FMM part on the test trees CI runs (mixed
levels: some classes diagonal, some RCR), CH4.2b2 on W1.

## Read first
- `fmm-tables/src/helmholtz/diagonal/{mod,level,transfer}.rs` (`Conversions`, `Signatures`,
  `m2l_level`'s order, `scratch_shape`, `DiagonalM2l`, `class_bounds`), ref's
  `diagonal::{bound, BoundConstants, THETA}`.
- `fmm-exec/src/helmholtz/operator.rs` ("M2L", "The tables' build", "Execution";
  `select_tables`, `Kernel::m2l`, `impl FmmOperator::m2l`, `Scratch`), `operator_rcr.rs` (a
  level call of its own on `Execution::chunks`), `fmm.rs` (`build`: the agreed settings, the
  node budgets over `ranks_per_node`).
- `fmm-exec/tests/helmholtz_fmm.rs` (`check_auto`, `check_rcr_split`: the shape of a check),
  `tests/helmholtz/` (operator tests without MPI).

## Changes
- `nd-fmm-exec`:
  - `HelmholtzDiagonal` { `Off` (default), `Rule`, `Tolerance(ε)` }: `Rule` takes the class c of
    a level to the diagonal form where `class_bounds(p_l, k̂_l, constants)[c] ≤ THETA · ε` at the
    FMM's tolerance (none with named degrees: no level), `Tolerance(ε)` at a named ε (tests,
    measurements). Constants `BoundConstants::F64` in f64; in f32 `F32`, and only at ε ≥ 10⁻⁴
    (D3). Candidate levels: l ≥ 2 whose M2L under the current setting is RCR's (no waves, no
    Dense, Classes or Batched tables), so the setting composes with every `HelmholtzM2l`; a
    level with no class passing runs as before. A function of k, the level's width, p_l, ε and
    T (constraint 9).
  - The scratch budget: per rank min(named, `DEFAULT_SIGNATURE_NODE_BUDGET` / ranks per node).
    A level runs diagonal only where `scratch_shape(p_l, S_l · m, …)` fits, S_l the largest
    count over the ranks of the level's distinct V sources (own and ghost; the builder's one
    all-reduce max, `with_diagonal_sources`): the whole-level fallback to RCR, the same on every
    rank (moved from CH4.3's Also). The threads' scratch is counted at one thread in the
    decision (the decision reads no thread count), reported in full by `diagonal_bytes`.
  - `select_tables` builds per diagonal level `DiagonalM2l::build_on` (one part per class on
    the pool) and `Conversions::new(p_l, scale_l)`; `TableTimings::diagonal`.
  - The level call (new private `operator_diagonal.rs`), on a level with transfer functions:
    1. the slot of each distinct source of a diagonal pair, ascending source index;
       S → F of every (slot, vector) into the level's `Signatures` (sources × m), blocks of
       `BLOCK_BOXES` signatures on the pool, each block's multipoles gathered (a real
       multipole widened to two planes, the imaginary plane zero) into the thread's scratch;
    2. the targets in chunks of consecutive rows on the pool (`Execution::chunks` over the
       locals; the chunk's width named by `with_diagonal_chunk` or ⌈rows/threads⌉ within 1 and
       `BLOCK_BOXES`): per target with a diagonal pair and vector, the accumulator zeroed, its
       diagonal pairs in row order (`DiagonalM2l::pair`), F → R added into its local; then its
       RCR pairs in row order (`HelmholtzTables::m2l`, Rotation's body). Each local's bits
       depend on its row alone: the same at every chunk, thread count and rank count.
    The accumulators are each thread's (a chunk of targets × m), `Signatures` as signed off.
  - The pair path (`PairOperator::m2l`, per pair): S → F of the source, the pair, F → R: the
    level call to rounding (a target's diagonal pairs summed in signature space there).
  - Reports: `diagonal_classes(level)`, `diagonal_levels`, `diagonal_bytes`, the setting.
  - `HelmholtzFmmBuilder::{diagonal, diagonal_budget}` (per node; default
    `DEFAULT_SIGNATURE_NODE_BUDGET`), `HelmholtzFmm::diagonal_classes`.
- `nd-fmm-tables` (Helmholtz module only): `Signatures::strides_mut` (the held boxes' strides as
  one slice, for a caller that splits blocks over threads) and `Conversions::s2f_strides` (S → F
  of a block into such a slice, `s2f`'s bits per box), so the pool writes disjoint blocks.
- Docs: the operator's module docs ("M2L"), `fmm-exec/CLAUDE.md`, `fmm-tables/CLAUDE.md`.
- No convention, no dependency, no Laplace file (`driver.rs`, `pipeline.rs`, `nd-fmm-plan`,
  `nd-octree`, `nd-fmm-simd` untouched), no `nd-fmm-plan` interface change, no default changed.

## Tests that prove it
- `fmm-exec/tests/helmholtz/diagonal.rs` (no MPI), f64 unless said:
  - the rule per level: the classes `diagonal_classes` reports are those with
    E ≤ θ ε (`class_bounds`); `Off` none; `Rule` with named degrees none; f32 below 10⁻⁴ none;
    levels with waves or batched tables none; the fallback at a budget below a level's need;
  - the pair path on every diagonal offset against Rotation's pair within 2E (CH1.3d1's
    measure, as CH4.2a), and an RCR class's pair Rotation's bits;
  - `strides_mut`/`s2f_strides` (in `fmm-tables/tests/helmholtz/diagonal_level.rs`): each box
    `s2f`'s bits at every block split.
- `fmm-exec/tests/helmholtz_fmm.rs`, `check_diagonal` (the MPI test, CI at 1, 2 and 4 ranks),
  on two scenarios with a mixed level (the cube at kD ≈ 40 at ε = 10⁻⁴, a sphere), one and two
  multipole planes:
  - some level diagonal with some class on RCR (asserted), the setting off Rotation's bits;
  - the FMM against the direct sum within `DIAGONAL_RATIO` = 1.1 of Rotation's error, under
    Rotation and under the default Auto;
  - 2 threads (and 4 in release) bit for bit 1 thread; chunk widths 1, 2, 3, 5, 7 bit for bit
    the default; on P ranks the one-rank FMM bit for bit; two vectors each the one-vector FMM;
  - the budget's floor: every level RCR, Rotation's bits.

## Checks
`/verify`: fmt; Clippy (default, workspace all-targets); rustdoc `-D warnings`; default and
workspace tests at opt-level 1; the MPI list at 2 and 4 ranks (`helmholtz_fmm` changes);
`cargo test -p nd-fmm-tables` (the debug budget of a minute); `cargo deny` not needed (no
manifest). Laplace: no Laplace file changes, so its fixtures and bit-identity tests in the
workspace run cover constraint 1; no benchmark needed (D9 applies to shared code changes).

## Machine runs
None here: correctness on the M6. The times, W1's errors and the build time are CH4.2b2's
(Kathleen, locust).

## Constraints and risks
- Constraint 1: Laplace untouched (Helmholtz modules; `Execution` used, not changed).
- Constraint 2: each local's chain is its row alone (diagonal pairs in row order, F → R, RCR
  pairs in row order); the source signatures are functions of their multipoles alone.
- Constraint 7: the scratch within the node budget over the ranks per node.
- Constraint 9: the classes from k, w_l, p_l, ε and T; the fallback from the agreed largest
  source count; the chunk width an execution split only.
- Risk: the debug cost of `helmholtz_fmm` (CI's 900 s timeout at 4 ranks; the one-rank debug
  run about 12 minutes on a loaded M6): keep the scenarios small, release-only extras.
- Risk: f32's bound passes few classes at test sizes; f32 tested on the rule's table only.

## Pending runs
None.

## Open questions
1. The whole-level fallback built here (from CH4.3's Also), since the opt-in on several ranks
   needs it for constraint 2. Recommended: accept.
