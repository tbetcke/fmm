# nd-fmm-tables

Purpose: precomputed, level-independent operator tables, built in f64 from nd-fmm-ref,
and a versioned on-disk cache. The tables are M2M and L2L for the 8 child octants, M2L
for the 316 V-list offsets and their 16 symmetry classes, and the rotation and coaxial
tables of point-and-shoot translation.
Phase and components: Phase 2, C2.1–C2.4 (tasks T1 and T3–T7 in docs/phase2/); Phase 6
C6.5 (task T13: `plane_wave`, the tables of plane-wave M2L, CONVENTIONS §3.14);
Helmholtz CH3.1a (`helmholtz`, the per-level RCR tables of the Helmholtz family,
CONVENTIONS §3.15); Helmholtz CH3.2a (`helmholtz::dense`, the dense M2L tables); Helmholtz
CH3.2b (`helmholtz::classes`, the class M2L tables and the cube-group transforms); Helmholtz
CH3.2c (`helmholtz::batched`, Dense's matrices in split planes for the batched M2L); Helmholtz
CH6.1a (`helmholtz::plane_wave::quadrature`, the quadratures of the exponential M2L); Helmholtz
CH6.1c (`helmholtz::plane_wave::pair`, the exponential M2L's conversions, shifts and pair
translation on one level, CONVENTIONS §3.15 "Exponential M2L"); Helmholtz CH6.1d
(`helmholtz::plane_wave::merge`, the merge's shifts and each level's quadrature), the
rule of which Helmholtz CH6.1e measured; Helmholtz CH4.0h (`parts`, the parts of a build on
a caller's threads); Helmholtz CH4.0j (the exponential M2L's tables in parts within a
level); Helmholtz CH4.1a (`helmholtz::diagonal`, S → F and F → R of the diagonal form as
products over a batch of boxes); Helmholtz CH4.1b (`helmholtz::diagonal::level`, the level's
signature buffers, scratch and driver, and `signed_permutation`); Helmholtz CH4.2a
(`helmholtz::diagonal::transfer`, the transfer functions per (level, class) and the pair).

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  - Geometry, table layout and symmetry follow §3.12.
  - The translations are those of §3.11, reached only through nd-fmm-ref.
- Tables are built in f64 by applying nd-fmm-ref operators at the canonical frames of
  §3.12. An f32 table is the f64 table rounded entry by entry. Never build in f32, and
  never re-derive an operator formula here.
- The exception, `plane_wave` (Phase 6 T13): the plane-wave M2L has no operator in
  nd-fmm-ref, so its tables follow the formulas of CONVENTIONS §3.14 (built in f64, rounded
  to T) and are checked against `nd_fmm_ref::direct::m2l` (levels 2, 9 and 16, within the
  quadrature's bound; the ignored `pair_m2l_every_degree` at every p ≤ 20). FMM3D's
  quadratures (`quadrature_data.rs`, NIST's notice kept) are regenerated only by
  `tools/fixtures/plane_wave_quadrature.sh`. The degree's quadrature
  (`nodes_for_degree`) is the study of the ignored `level_every_degree` (the spike's
  matched rule on level 3); a change to it changes the FMM's errors, so rerun the gates of
  nd-fmm-exec. The functions that a `nd-fmm-simd` `WaveKernel` loop restates
  (`waves_from_modes`, `sums_from_waves`, `shift_add`, `shift`, `add`) define its chains
  with `mul_add`; keep the two in step. The other steps round products and sums apart: an
  x86_64 build without `fma` turns `mul_add` into a library call. The plane-wave tables
  are not cached (milliseconds to build). The accessors `outgoing_powers`,
  `incoming_powers`, `multipole_rotation`, `local_rotation` and `child_shifts` (Helmholtz
  CH7.5a) give `nd-fmm-kernels`' device kernels the arrays behind the steps, in their
  storage; `arrays_match_the_steps` checks them against the steps.
- The second exception, `helmholtz` (Helmholtz CH3.1a; its module docs): the coaxial
  tables come from `nd_fmm_ref::helmholtz::rcr::{m2m_table, l2l_table, m2l_table}`, but
  the stored form's factors folded into them and the three steps of the application
  restate `rcr`'s private `translate` in its order of operations; keep the two in step.
  Its tables are keyed by exact integer classes of the shifts, as `rotation`'s are (15
  values of |d|², 49 polar angles, the azimuth per offset from (d_x, d_y)/gcd), their
  values formed from the integers by correctly rounded square roots and quotients, so the
  classes are the same on every machine; never key a table by a floating-point value.
  `rcr` forms its angles with `hypot`, so the two agree to rounding: `tests/helmholtz/`
  compares them in the FMM-weighted measure within 2e-13 times max(1, |k̂| a) (1.7e-14
  measured on the M6). Its y-blocks are built in T, as `rcr`
  builds them per call, so that stored and per-call blocks agree; the M2L blocks are
  stored from the finest level up within a byte budget. Not cached (CH2.2).
  Since Helmholtz CH4.0i `HelmholtzTables::m2l_contribution` writes the terms `m2l` adds
  (step 3 stores instead of adding) and `add_m2l_contribution` adds them, the two in turn
  `m2l` bit for bit (`to_bits`), so a caller may compute a level's pairs in any order and
  add them in each row's order; `m2l_classes(d)` gives an offset's (angle, distance), and a
  `HelmholtzScratch` keeps the per-call y-blocks it formed last with their degree and
  (cos θ, sin θ), forming them again only for another (the same function of the same
  inputs). `tests/helmholtz/contribution.rs` checks the split, a row added in order, and
  kept blocks against fresh ones in three orders, bit for bit.
  Since Helmholtz CH6.1h `HelmholtzTables::build_with_m2l_degrees` gives each level a
  second degree q_l ≥ p_l (`m2l_degree`): its expansions and M2L at q_l, the M2M into it
  from the children's first p_{l+1} degrees into q_l, the L2L from q_l into the children's
  first p_{l+1}; `translate` reads and writes the first degrees of each plane of a larger
  expansion. `build` is it with q = p (bit for bit as before). `tests/helmholtz/main.rs`
  checks the translations on the first degrees against `rcr` (the degrees above NaN, or
  left untouched) and M2L bit for bit the tables built at q.
  `helmholtz::dense` (Helmholtz CH3.2a; its module docs): `DenseM2l`, per level and offset
  the complex M2L as one 2n × n column-major matrix [A_r; A_i], column j Rotation's M2L of
  e_j built by `HelmholtzTables<f64>` and rounded to T (so Dense equals Rotation to
  rounding: 4e-14 in the FMM-weighted measure on the M6); levels chosen within a byte
  budget from the finest up (`select`, a function of the degrees and the budget). Its
  application's chain (one plane: k ascending over 2n rows; two planes: one pass,
  L_r + A_r M_r − A_i M_i and L_i + A_i M_r + A_r M_i per k) is what a batched form must
  equal bit for bit, or say it does not (CH3.2c). Not cached.
  `helmholtz::batched` (Helmholtz CH3.2c; its module docs): `BatchedM2l`, Dense's matrices
  re-laid as A_r then A_i, each n × n column-major, bit for bit Dense's entries; Dense's
  `select` and `level_bytes`; `select_within` (Helmholtz CH3.2d) the same choice among the
  levels of degree at most a crossover only (`nd-fmm-exec`'s `HelmholtzM2l::Auto`). Its
  application is `nd-fmm-exec`'s (the column kernel takes
  square matrices): one plane Dense's chain bit for bit, two planes the real plane's
  products of a column before the imaginary plane's, Dense to rounding. Not cached.
  `helmholtz::classes` (Helmholtz CH3.2b; its module docs): `ClassM2l`, per level the 16
  class matrices of §3.12 built by `DenseM2l`'s rule (bit for bit its matrices of the
  representatives), and `CubeTransforms`, Oⁿ(P) of the orthonormal coefficients of §3.15
  for the 48 elements (Oⁿ(P) = (−1)ⁿ Oⁿ(−P) for improper P; T_M = T_L = Oⁿ): rotations
  factored into quarter turns about z and y, the y-block of π/2 from
  `orthonormal::rotation::y_blocks` at cos 0 and sin 1 exactly, so the 16 elements that fix
  the z-axis are exact signed permutations and the 32 others dense per-degree blocks. Per
  pair O(Pᵀ), the class matrix by Dense's chain into zeroed scratch, then O(P) added: Dense to
  rounding (3.7e-14 in the FMM-weighted measure on the M6), a representative into a zero
  local Dense bit for bit. `select` charges the transforms of the largest chosen degree. The
  transforms follow from §3.12 and §3.15 as they stand (no convention of their own). Not
  cached.
  `helmholtz::plane_wave` (Helmholtz CH6.1a; its `quadrature` module docs): the quadratures
  of the Helmholtz exponential M2L per level, FMM3D's tables for iprec 1–3
  (`hwts_iprec{1,2,3}.txt` beside the module, NIST's notice and a dated change note kept,
  FMM3D's literals with D written as e), regenerated only by
  `tools/fixtures/helmholtz_pw_quadrature.sh` and parsed on first use; `hwts3e`'s lookup
  (`domain_of`, `Precision::for_tolerance`) and node maps; the α and Fourier counts not
  FMM3D's (`hnumphys` misses the stated error at the region's corner) but a Bessel bound per
  node with the share e/(4s) of the stated error, |J_m| of complex argument by Miller's
  recurrence normalised by e^{iz} at Im z ≤ 0. A function of κ and the precision only.
  `Quadrature::with_reach`, `MergeLevel::with_reach` and `PlaneWaveM2l::build_with_reach`
  (Helmholtz CH6.2e, for the measurements) take the α counts for lateral offsets ρ ≤ a
  reach in place of the corner's `RHO_MAX` = 4√2 (the Fourier counts unchanged; the corner's
  reach is `new` and `build` bit for bit); `the_counts_for_a_smaller_reach` checks the kernel
  on ρ ≤ 4 within 2× at reach 4 on the fixture's domains, the ignored
  `the_counts_for_a_smaller_reach_on_every_domain` (reaches 4 and 4.5, about 8 minutes in
  release) on every domain, with the waves and the corner's error printed
  (fmm-validate/results/helmholtz-ch6-2e.md §1). The corner stays the default (Helmholtz
  CH6.2g, fmm-validate/results/helmholtz-ch6-2g.md): at the largest V offsets the FMM's
  chain errs by up to 2·10⁴× the stated error with the counts for reach 4.5 at high κ,
  nd-fmm-exec's `tests/helmholtz/plane_wave_reach.rs`. The
  trusted path is the kernel e^{iκ|u|}/|u| on the region: `tests/helmholtz/` checks the
  kernel within 2× and the point-to-point chain within 3× the stated error (on every domain
  in the ignored `the_counts_on_every_domain`, about 6 minutes in release). A change to
  the counts changes the waves of every level, so rerun that test and the spike's
  `examples/counts.rs` (the wave counts against FMM3D's, docs/helmholtz/tasks/CH6.1a.md).
  Not cached.
  `helmholtz::plane_wave::pair` (Helmholtz CH6.1c; its module docs): `PlaneWaveLevel`, one
  level's factors Pₖ(n, m) (solid normalised Legendre polynomials at (s/κ)(iλₖ, tₖ),
  `harmonic_factors`), the cos and sin of mαⱼ, and the frames' Oⁿ(Q) (`classes`'
  `rotation_blocks`, bit for bit `CubeTransforms`' blocks), built in f64 from either
  quadrature (CH6.1a's `Quadrature`, or any `Nodes`, `nd_fmm_ref`'s generated ones
  included) and rounded to T; the steps of CONVENTIONS §3.15 "Exponential M2L" and the
  pair `m2l` in `HelmholtzTables`' units and slices (one or two multipole planes, two
  local planes). Real and complex k only (Yukawa's real path is refused, as
  `HelmholtzTables`). The trusted path is RCR: `tests/helmholtz/plane_wave_pair.rs`
  checks every V offset along every direction merge and shift gives it within 3× the
  stated error (FMM3D's) or 3ε (generated), at degrees that resolve the multipoles of
  in-box charges (`resolved`: base 20, 36, 60 for iprec 1–3, plus ⌈0.9 |κ|⌉); below
  them the chain also errs on the truncated multipole's tail, which the ignored
  `degree_study` records (docs/helmholtz/tasks/CH6.1c.md) and CH6.1d's choice of the
  quadrature per degree must answer. Every product and sum rounded apart (no `mul_add`
  yet: no SIMD kernel restates these steps). Not cached. Helmholtz CH6.1f
  (fmm-validate/results/helmholtz-ch6-1f.md) added
  `against_rcr_with_degree_matched_quadratures` (`nd_fmm_ref`'s `generate_for_degree` through
  `PlaneWaveLevel` against RCR at its degree, within 3ε; its ignored `…_at_the_fmms_degrees`)
  and the ignored `degree_matched_study` (about 35 minutes in release): the four candidates of
  CH6.1f at the FMM's degrees of 10⁻⁷ and 10⁻⁸, each level between children at p as in the
  FMM, against the direct sum over RCR's (FMM3D's tables at p and at a raised degree, the
  generator at a chosen ε, the degree-matched quadrature).
  `helmholtz::plane_wave::merge` (Helmholtz CH6.1d; its module docs): `MergeLevel`, a
  level's `PlaneWaveLevel` with the merge's shift factors of CONVENTIONS §3.14's geometry
  (`crate::plane_wave::geometry`, read only): the source shifts slot → parent, one table
  per distinct (rotated offset, up or down), and the child shifts parent → child, each
  `shift_factors`' 2W reals; `PlaneWaveM2l`, a tree's levels; `shift` and `add`, the
  merge's complex steps (rounded apart, `shift` is `shift_add` from zero bit for bit but
  for the sign of a zero);
  `precision_for_tolerance`, `precision_for_degree` and `select`, each level's precision
  where κ_l lies in the tables (a function of k, w_l, p_l and ε: constraint 9), measured
  in the FMM by Helmholtz CH6.1e (fmm-validate/results/helmholtz-ch6-1e.md): iprec 2 at
  the FMM's tolerance ε ≥ `PLANE_WAVE_MIN_TOLERANCE` (10⁻⁶) at the level's degree; below,
  since Helmholtz CH6.1g (CH6.1f's resolving degree, fmm-validate/results/helmholtz-ch6-1g.md),
  iprec 3 to `RESOLVING_MIN_TOLERANCE` (10⁻⁸) at `plane_wave_degree` = max(p_l,
  `resolving_degree`): 28 to `RESOLVING_STEP_TOLERANCE` (10⁻⁷), 32 below (iprec 2's own
  error lies above Rotation's at 10⁻⁷ whatever the degree), which nd-fmm-exec's operator
  raises the level to, and none below 10⁻⁸; without a
  tolerance (named degrees) iprec 2 to `MAX_PLANE_WAVE_DEGREE` (15, the calibrated plateau
  at 10⁻⁶), none above and never raised. iprec 1 misses 1.1× Rotation's error from ε = 10⁻³, iprec 3 at the
  level's degree on the clustered trees' deep levels at 10⁻⁷ and 10⁻⁸. A change to the rule changes the FMM's
  errors under `HelmholtzM2l::PlaneWave`: rerun the report's verification
  (`tools/helmholtz/ch6-1e-runs.sh verify`) and nd-fmm-exec's `tests/helmholtz_fmm.rs`.
  `tests/helmholtz/plane_wave_merge.rs` checks slot shift × child shift against the pair's
  shift on every direction, slot and child within 1e-12 relative (7.5e-14 measured on the
  M6), the shared tables bit for bit each slot's own, and the rule. Not cached
  (about 50 ms a level in debug builds).
- The third exception, `helmholtz::diagonal` (Helmholtz CH4.1a; its module docs):
  `Conversions`, the tables of S → F and F → R of a level's degree and power scale, built in
  f64 from `nd_fmm_ref::helmholtz::diagonal::Grid`'s θ-factors, weights and azimuth tables
  (`Grid::theta`, `Grid::weight`, `Grid::azimuth_tables`) times sⁿ (and the weight for F → R),
  rounded once to T; nothing derived that the reference does not hold. The transforms are not
  the reference's order of operations (the Legendre step by the rings' parity, the φ step per
  residue μ mod 4 at the base azimuths 0 ≤ j ≤ N_φ/8 and the others by signs), so they agree
  with `nd_fmm_ref::helmholtz::diagonal::{s2f, f2r}` to rounding: `tests/helmholtz/diagonal.rs`
  checks 256 u of the largest value (p ≤ 24; p = 41, 72, 131 ignored, in release), the
  round trip s²ⁿ M̂, f32, and every box of a batch bit for bit the box alone. Every product is
  one column kernel (`gemm_acc`: each element its chain over the inner index ascending, product
  and sum rounded apart, from its own row and column), so a box's bits never depend on its
  batch, position, `BLOCK_BOXES` or `RING_BLOCK`; keep it so (no `mul_add`, no reassociation,
  no sum across boxes). Matrices here are row-major (the products' natural form); the
  expansions are §3.15's two planes, the signatures §3.15's samples, interleaved complex, box
  after box (the level's buffers are CH4.1b's). `s2f` overwrites its signatures, as the
  reference does (the one exception to "every application accumulates"); `f2r` adds. Not cached.
  Since Helmholtz CH4.1b (`diagonal::level`, its module docs; docs/helmholtz/tasks/CH4.1b.md):
  `Signatures`, a level's signatures box after box, each box's first sample on a 64-byte line
  (`SIGNATURE_ALIGN`; the stride the samples rounded up to whole lines, the padding zero and
  never read; aligned from the allocation's address in safe Rust, `is_aligned`), interleaved
  complex in §3.15's sample order; this layout is the crate's, not a convention.
  `Conversions::{s2f_into, f2r_from}` convert into and out of it, each box `s2f`'s and `f2r`'s
  bits. `m2l_level` is the serial level driver (S → F of every source once per call, then
  the targets in chunks of `LevelBuffers::chunk`: each accumulator zeroed, its pairs in its
  row's order by the caller's closure, F → R of the chunk by blocks, the locals gathered and
  scattered by copies), so every local is the same bits at every chunk size; nd-fmm-exec's
  threaded level call (CH4.2) keeps that order. `scratch_shape` sizes it under a byte budget
  per rank (`DEFAULT_SIGNATURE_NODE_BUDGET` 8 GiB per node, divided by the caller over the
  node's ranks): the tables (`Conversions::bytes_for`, the Legendre tables (p + 1)²(p + 2)
  values), a `ConversionScratch` and gather buffer per thread, the sources' signatures whole,
  the largest chunk of accumulators that fits, `None` below `BLOCK_BOXES` targets (CH4.3's
  fallback to RCR). `signed_permutation` names ref's `diagonal::Symmetry` (flips, then the
  x ↔ y swap) as the element of O_h acting the same on offsets (review 2026-11-19, C5).
  `tests/helmholtz/diagonal_level.rs` checks the alignment (f64 and f32, padded and not), the
  store conversions bit for bit, the level call against the sum of `nd_fmm_ref`'s
  `diagonal::m2l` over each target's pairs (256 u; p = 6 and 12, p = 17 ignored in release),
  chunks of 1, 2, 3, 4, 5, 7 and all bit for bit (f64, and a padded f32 stride), clones aligned,
  the shapes by hand, the sizes without building,
  and ref's 316 offsets the tables' set with each of the 34 classes inside one of the 16.
  Since Helmholtz CH4.2a (`diagonal::transfer`, its module docs; docs/helmholtz/tasks/CH4.2a.md):
  `DiagonalM2l`, one level's transfer functions in level units (k̂_l, half-width 1, |t| = 2|d|)
  for the classes a caller names (indices into ref's `classes()`), each stored once as
  W = 4π ik̂ e^{2β − Im(k̂)|t|} T̂_rep: T̂ from ref's `Transfer::<f64>` (double-double, rounded
  once), the factor formed in f64 as `HelmholtzTables` forms the RCR M2L's and multiplied in
  f64, then rounded to T; built one part per class (`build_on`). Each V offset reads its class
  through an `OffsetMap` (the ring flip and the azimuth map j ↦ (start ± j) mod N_φ, derived
  from ref's `Symmetry::sample`), so `pair` (g = g + w · f per sample, product and sum rounded
  apart, two contiguous runs per ring) needs no table per offset; `offset_samples`
  materialises an offset's W. `class_bounds` is ref's `bound` per class at |t| = 2|rep|, a = √3:
  the rule's input, not the rule (CH4.2b, CH4.3). `tests/helmholtz/diagonal_transfer.rs`
  checks the maps against ref's symmetries on every sample, the stored samples and every
  offset's bit for bit ref's `Transfer` (and `permuted`) times the factor, the pair bit for bit
  its naive loop (f64, f32), the build in parts bit for bit, and the acceptance: the tables'
  level call against ref's RCR within 2E (CH1.3d1's measure) on the 34 classes (real and
  damped k), the 316 offsets and f32 (k a = 16 ignored, in release), and the ignored large
  degrees (k a 32–108; there, at E's floor, an offset above 2E against RCR passes if the
  diagonal form lies no farther than max(2E, RCR's error) from ref's projection); and
  a level with each target's RCR pairs added after F → R in row order, bit for bit at every
  chunk size and within 2E of the level with every pair RCR.
- Matrices are column-major with output = A · input. Every application accumulates
  (+=), and none applies 1/(4π) (§3.1).
- Builds are deterministic: the same inputs give bit-identical tables and cache files.
  This crate runs no thread. Since Helmholtz CH4.0h (`parts`; amended from "serial", signed
  off by the user on #221, 2026-11-16) a
  builder may cut its work into independent parts and hand them to a caller's
  `parts::Parts` (`Serial`, the calling thread, or `nd-fmm-exec`'s pool): each part a
  function of the build's inputs and its number alone, the results put together in part
  order, so the tables are the serial build's bit for bit on any map. So far
  `HelmholtzTables::build_by` (with a `TableChoice`: the M2L degrees, the block budget,
  translations or none, the levels with M2L tables and with stored blocks; and
  `set_m2l_blocks`), `DenseM2l`, `BatchedM2l` and `ClassM2l::build_on` (their f64 tables
  now hold the M2L of their own levels alone, no translations),
  `DiagonalM2l::build_on` (Helmholtz CH4.2a, one part per class) and
  `PlaneWaveM2l::build_with_reach_on` (since Helmholtz CH4.0j within a level too, every
  level's parts in three maps: each node's quadrature counts (`QuadratureNodes`), each
  node's values and each frame's rotation blocks (`LevelLayout`), each shift table, the
  shift tables kept one allocation each; `merge`'s module docs, "The build in parts").
  `tests/helmholtz/parts.rs` checks every build on a map that runs the parts in reverse on
  three threads against the serial build, bit for bit
  through every application (and the matrices, and the plane-wave tables' `Debug` text),
  the restricted builds' M2L against the full build's, the refusals, and the blocks of
  named levels; its ignored `print_the_hashes` prints the applications' hashes, which were
  the same before and after the change on the M6 (docs/helmholtz/tasks/CH4.0h.md; again at
  CH4.0j, docs/helmholtz/tasks/CH4.0j.md). Since CH4.0j it also checks every level of the
  exponential M2L built in parts against the level formed from the public steps one by one
  (`every_level_in_parts_is_the_level_by_its_steps`; the steps run the same node and level
  parts, so the hashes are what hold them to the build before), and its ignored
  `time_a_levels_parts` prints a level's build by part on one thread (no time asserted; the time on many threads
  is the FMM's, `helmholtz_accuracy`'s build part "plane-wave tables").
- Generic over T: RealScalar. Allocate only in builders and constructors, never in
  apply functions: their temporaries come from caller-owned scratch.
- The octant and offset order are restated here from §3.12, not imported from
  nd-octree or nd-fmm-plan. Phase 3 (C3.1) tests that they agree.
- Tests name their error measure (docs/phase1/README.md, "Error measures", and
  docs/phase2/README.md). A table is compared with nd-fmm-ref on other levels in a
  dyadic domain.
- Keep the debug-mode test run under a minute. Large-p checks go into `#[ignore]` tests.
- Before finishing: `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings` and
  `cargo test -p nd-fmm-tables` must pass. Tasks that add ignored tests must also pass
  `cargo test -p nd-fmm-tables --release -- --ignored`.

## Allowed dependencies
nd-fmm-math, nd-fmm-ref, num-traits; thiserror (from T7, for cache errors);
dev-dependencies: proptest.
Not allowed:
- MPI, nd-octree, nd-fmm-plan or CubeCL;
- rlst, which waits for SVD compression (C6.2). When it comes, it is used without
  its `mpi` feature, so `cargo test -p nd-fmm-tables` still needs no MPI;
- serde or another serialiser;
- rayon;
- nd-fmm-validate, not even as a dev-dependency.

Anything else needs a note in the PR.

## Test oracle
- `nd_fmm_ref::direct` for every table, and `nd_fmm_ref::rotation` for the rotation
  tables, at the canonical frames and on other levels.
- The Phase 1 physical chains (M2M after P2M, L2L exact on local polynomials).
- The symmetry identities of §3.12: the class form against the dense tables.
- A cold build for the cache.
