# CH3.2a Dense

Started 2026-11-02 on `tbetcke/h-ch3-2a`. Roadmap: "CH3.2a Dense" (H3; CH3.2 split on
2026-11-02 into a/b/c; off the priority list, named by the user). Proposal: §10 CH3.2;
§5.3 (Dense), §6.2, §6.3, §6.4, §7; §2 (constraints 1, 2, 5, 6, 7, 9).

## Goal

A Dense M2L for the Helmholtz operator: per level and V-list offset the complex M2L table,
built in f64 from the level's RCR and rounded to T, chosen per level where its tables fit a
byte budget and applied pair by pair on the unchanged shared drivers.

## Acceptance (proposal §10, verbatim)

CH3.2: "measured against Rotation per p, including the cost of the cube-group transforms
that move the z-axis". The roadmap's *Done when* of CH3.2a: every offset against Rotation
to rounding on several levels, real and complex k, one and two multipole planes; the FMM
with Dense against the direct sum and against Rotation's output, the same at every thread
count and on P ranks the one-rank FMM bit for bit; the tables' bytes and build time, and
the M2L cost per pair against Rotation per p on the M6. (The cube-group transforms are
CH3.2b's; the three machines CH3.2c's.)

## Read first

- `fmm-tables/src/helmholtz.rs`: `HelmholtzTables::{build, m2l}`, `translate`, the module
  docs ("Levels and units", "Application"); `fmm-tables/tests/helmholtz/main.rs` (the
  FMM-weighted measure, `BOUND`, `K_ROOTS`).
- `fmm-tables/src/matrix_set.rs` (`MatrixSet::apply`: the chain of Laplace's Dense).
- `fmm-exec/src/helmholtz/operator.rs` (`Kernel::m2l`, `HostKernel for Kernel`,
  constructors); `fmm-exec/src/helmholtz/fmm.rs` (the builder, `block_budget`).
- `fmm-exec/src/driver.rs` (`m2l_target`; `batched_m2l` stays `None`).
- `fmm-exec/tests/helmholtz/operator.rs`, `fmm-exec/tests/helmholtz_fmm.rs`;
  `fmm-validate/src/helmholtz_accuracy.rs` and its example.
- docs/helmholtz/tasks/CH3.5.md (the finding and question 2).

## Changes

`nd-fmm-tables` (the `helmholtz` module), `nd-fmm-exec` (the Helmholtz modules only) and
`nd-fmm-validate` (an example, an option). No Laplace file, no driver, no `nd-fmm-plan`, no
convention, no manifest dependency.

- `nd_fmm_tables::helmholtz::dense` (new file `fmm-tables/src/helmholtz/dense.rs`, a
  submodule of `helmholtz.rs`): `DenseM2l<T>`.
  - Layout: per level l with Dense and offset index d one 2n × n column-major matrix,
    n = (p_l + 1)², rows 0..n the real part A_r and rows n..2n the imaginary part A_i of
    the complex M2L from the multipole coefficients (CONVENTIONS §3.6 order) to the local's;
    so column j is the local of the unit multipole e_j in two planes (Re, then Im), the
    layout of a box's complex local. The stored form's factors are those of
    `HelmholtzTables` (its "The stored form, folded in").
  - Build: `DenseM2l::build(k_root, degrees, levels: &[bool])`, from a
    `HelmholtzTables::<f64>::build(k_root, degrees, budget)` of the same inputs, column j of
    offset d on level l = `m2l(l, d, e_j, 1 plane, …)` into zeros, rounded to T entry by
    entry; serial and deterministic. Levels 0 and 1 and the levels not chosen hold none.
    `DenseM2l::level_bytes(p)` = 316 · 2 (p + 1)⁴ · size_of T, and `select(degrees,
    budget)`: the levels ≥ 2 from the finest up while the running total fits the budget
    (a level that does not fit is skipped, a coarser one may still fit; as the y-blocks of
    `HelmholtzTables::build`). Constraint 9: a function of the degrees and the budget only.
  - Application `m2l(level, d, multipole, planes, local, vectors)`, adding (+=) per vector:
    one plane (real multipole): L[r] ← L[r] + A[r, k] M[k], k ascending, for the 2n rows;
    two planes: first the real plane as above, then the imaginary plane, k ascending,
    L_r[r] ← L_r[r] − A_i[r, k] M_i[k] and L_i[r] ← L_i[r] + A_r[r, k] M_i[k]; each product
    and sum rounded apart, as `MatrixSet::apply`. Each vector gets the one-vector chain.
  - Accessors: `is_dense(level)`, `matrix(level, d)`, `bytes()`, `degree(level)`.
- `nd_fmm_exec::helmholtz`:
  - `HelmholtzM2l` (public enum): `Rotation` (the default) and `Dense`; and
    `DEFAULT_DENSE_BUDGET` (1 GiB per rank; opt-in, so no Auto rule yet: the rule is a
    question for CH3.2c, which has the measurements of all four).
  - `HelmholtzOperator::with_m2l(choice, dense_budget)` (or a builder-style setter after
    `new`; the tables built once): under `Dense`, `DenseM2l` on the levels of `select`,
    `Kernel::m2l` dispatching per level by `is_dense` (one branch per pair, Helmholtz only;
    Rotation keeps `HelmholtzTables::m2l`). `m2l_levels()` reports the choice per level
    (`Rotation` or `Dense`), `dense_bytes()` the tables' bytes.
  - `HelmholtzFmmBuilder::{m2l, dense_budget}`, `HelmholtzFmm::m2l_levels`; the build time
    of the tables in what the FMM reports (or measured by the example).
  - Docs: the operator module's table and "Execution"; `fmm-exec/CLAUDE.md`'s Helmholtz
    operator rule.
- `nd-fmm-validate`:
  - the `helmholtz_m2l` example (MPI-free; registered `[[example]]` without
    templated-examples, as `helmholtz_calibrate`): per p (5 to 30 by the degree, or named),
    real and complex k, one and two multipole planes, on one level: the 316 offsets
    through Rotation (`HelmholtzTables::m2l`) and Dense (`DenseM2l::m2l`), the median of
    repeated passes, ns per pair; Dense's bytes per level and build time; prints Markdown.
  - `helmholtz_accuracy::Case::m2l` and the example's `--m2l dense` (end to end on CH3.1c
    cases, time and error against Rotation's).
  - Report `fmm-validate/results/helmholtz-ch3-2a.md`.

## Tests that prove it

- `fmm-tables/tests/helmholtz/` (new module `dense.rs`):
  - every one of the 316 offsets on levels 2 and 4 (and one deeper level) of the K_ROOTS
    (real and complex k̂), one and two planes, two vectors: `DenseM2l::m2l` against
    `HelmholtzTables::<f64>::m2l` in the FMM-weighted measure within the existing `BOUND`
    (2e-13 · max(1, |k̂| a)), and against `rcr::m2l` for a sample of offsets;
  - each vector the one-vector application bit for bit; a matrix's column j equals Rotation
    applied to e_j in f64 bit for bit (the build's definition);
  - f32 tables are the f64 tables rounded entry by entry; `select` against a hand count
    (finest first, a skipped level, budget 0 gives none); refusals (level < 2, a level
    without Dense, wrong lengths).
- `fmm-exec/tests/helmholtz/operator.rs`: the operator under `Dense` pair by pair against
  `Rotation` on every offset of a level within the same bound, real and complex k and
  charges; `m2l_levels` against `select`.
- `fmm-exec/tests/helmholtz_fmm.rs` (in CI's MPI list): on two scenarios (one real k with
  real charges, one complex k with complex charges) the FMM under `Dense` against the
  direct sum within the scenario's tolerance and against `Rotation`'s output within 1e-12
  relative ℓ2; at 1, 2 (and 4 in debug only if cheap) threads and pair by pair bit for bit;
  on P ranks the one-rank FMM bit for bit; a budget that leaves the finest level to
  Rotation still correct. Keep the one-rank debug run's growth small (report it).
- f64 throughout for the FMM (D3); f32 only for the tables' rounding.

## Checks

`/verify`: format; Clippy default, examples and workspace all-targets; `cargo test` and
`--workspace`; `cargo doc`; the MPI list at 2 and 4 ranks (`helmholtz_fmm` changes);
`cargo test -p nd-fmm-tables` and `-p nd-fmm-exec`; the licence check is not needed (no
dependency change; `fmm-validate/Cargo.toml` gains an `[[example]]` only).

## Machine runs

The M6 only (release, BLAS threads 1, one thread unless named): `helmholtz_m2l` per p, and
`helmholtz_accuracy --m2l dense` on CH3.1c cases (W1 at kD = 10, 30, 100; W4 cube at
kD = 10) at ε = 10⁻³, 10⁻⁶, 10⁻⁸, against Rotation, both orders. They decide nothing for
the defaults (Rotation stays the default), and feed CH3.2c's rule. locust and Kathleen are
CH3.2c's.

## Constraints and risks

- Constraint 1: Laplace untouched (no Laplace file, no driver change; `batched_m2l` stays
  `None`). The per-level branch is in the Helmholtz kernel, above the pair arithmetic.
- Constraint 2: the per-pair order is the driver's; Dense's chain is fixed per element, so
  threads and ranks give the same bits.
- Constraint 5: tables built in f64, rounded once; not cached (CH2.2).
- Constraint 7 and 9: the budget is per rank, the choice a function of the degrees and the
  budget; the node budget across ranks (node-m2l.md) is left to the Auto rule (CH3.2c).
- Memory: 316 · 2 (p + 1)⁴ values per level, 21 MB at p = 7, 331 MB at p = 15, 983 MB at
  p = 20 in f64; the M2L y-blocks of a Dense level are still stored by `HelmholtzTables`
  (wasted; noted, a follow-up if it matters).
- Build cost: n columns of an RCR per offset, O(p⁵) per offset; measured.
- Batched (CH3.2c) will want Dense's chain: with two planes the imaginary plane's pass reads
  the halves swapped and one negated, which a column kernel does not do as it stands. Not
  this task's; noted for CH3.2c.

## Pending runs

None yet.

## Open questions

None yet.
