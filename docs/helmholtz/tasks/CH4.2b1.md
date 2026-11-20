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
  - Reports: `diagonal_classes(level)`, `diagonal_counts`, `diagonal_bytes`, the setting.
  - `HelmholtzFmmBuilder::{diagonal, diagonal_budget}` (per node; default
    `DEFAULT_SIGNATURE_NODE_BUDGET`); the FMM's through `HelmholtzFmm::operator()`.
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
  - the FMM against the direct sum within `DIAGONAL_RATIO` = 1.1 of the setting's error
    without it, under Rotation and under the default Auto;
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

## Results
- Built as planned, with these deviations: no `HelmholtzFmm::diagonal_classes` or
  `diagonal_levels` (the FMM's reports through `HelmholtzFmm::operator()`); `Signatures::relayout`
  and `Signatures::fits` added to `nd-fmm-tables` besides `strides_mut` and `s2f_strides` (one
  store for every level's degree); the chunk widths 2, 5 and 7 and 4 threads in release builds
  only (CI's opt-level 1 keeps debug assertions, so CI runs widths 1 and 3 at one thread and
  width 3 at 2 threads); the scenarios the cube at kD = 40 + 2i and the clusters at kD = 20, both
  at ε = 10⁻⁴ (at the calibrated degrees the bound takes every class down to k a ≈ 2 at 10⁻⁴, so
  a mixed level needs the clusters' level 4, k a 1.1, 27 classes).
- The rule at k = 40 on the unit cube, ε = 10⁻⁴, S7's degrees (p = 22): level 2 all 34 classes,
  level 3 (k a 4.3) the 9 farthest; damped (40 + 4i) 32 and 5.
- The pair body against Rotation's within 2E on every offset of levels 2 and 3 (CH1.3d1's
  measure, the M6, debug): at most 0.335 E (level 2, real k, complex charges), 0.269 E (one
  plane, two vectors), 0.091 E (damped); level 3 0.126, 0.179, 0.117 E. The RCR classes
  Rotation's bits.
- The FMM (`check_diagonal`, one rank, the M6): the cube's levels 2 and 3 diagonal (all 34
  classes; 3,096 and 48,080 pairs), φ 4.777·10⁻⁶ and ∇φ 1.322·10⁻⁵, 1.000× Rotation's; under
  Auto no RCR level (waves and Batched), the output unchanged. The clusters' levels 2, 3 and 4
  (34, 34, 27 classes; level 4 96 diagonal and 192 RCR pairs), φ 5.898·10⁻⁶ against
  5.897·10⁻⁶, 1.000×, under Rotation and under Auto alike. Debug cost about 110 s at opt-level 0
  on a loaded M6 (load about 23), 8 s at opt-level 1. A first opt-level 0 run went past 30
  minutes and was stopped by its time limit; its output was buffered by a pipe, so where it
  stood is unknown; two runs since (opt-level 1 and 0) took 8 s and 110 s.
- Self-review (a subagent, read-only): one fix (the operator's signature store read its fit
  from the last layout, not its allocation, so it was reallocated between two levels on every
  evaluation; now `Signatures::fits`, and `check_diagonal` evaluates twice and checks the
  buffer's peak); documented the one-off growth without source counts and of the slots, and
  that the levels depend on the ranks per node where the budget binds (as the dense tables').
  Left as notes: the fallback in the FMM is tested at the budget's floor (every level) and the
  partial fallback by the rule's unit test; f32 tested on the rule alone (no f32 FMM test);
  the setters on a built operator rebuild every table.

## Open questions
1. The whole-level fallback built here (from CH4.3's Also), since the opt-in on several ranks
   needs it for constraint 2. Recommended: accept.
