# CH4.2a The transfer functions per (level, class) and the pair

Started 2026-11-19 on `tbetcke/h-ch4-2a`. Roadmap: CH4.2a "The transfer functions per (level,
class) and the pair" (CH4.2 split on 2026-11-19 into CH4.2a, the tables, and CH4.2b, the
operator, by the size rule), with CH4.2's Alsos that concern the tables: the trusted path
sample by sample, 2E at E's floor, review 2026-11-19 M11. Proposal: §10 CH4.2; §3.6 (the bound,
the rule), §5.4 ("Diagonal translation", "Reflection classes", "Scratch and chunking"), §2
(constraints 1, 2, 5, 6, 9).

## Goal
A level's transfer functions of the diagonal form, stored once per offset class and read by
every offset of the class through its symmetry, and the pair step that the level driver of
CH4.1b applies, shown against RCR within E(l, c) and with the RCR pairs added after F → R bit
for bit at every chunk size.

## Acceptance (proposal §10 CH4.2, verbatim)
"against RCR on the same class within E(l, c); determinism with the RCR pairs added after
F → R". CH4.2a shows both at the table level (the tables' level call, serial, every chunk size);
CH4.2b shows them in the FMM (threads and ranks). "Within E" is measured ≤ 2E = E/θ, as
CH1.3d1 and S6 measure it (S6 fitted C_T so that the measured rounding stays within 2E); at E's
floor f64 RCR rounds up to 2.4× more than the diagonal form, which this reading covers.

## Read first
- `fmm-ref/src/helmholtz/diagonal/{mod,transfer,classes,bound}.rs` (`Transfer`, `Symmetry`,
  `class_of`, `classes`, `bound`), `fmm-ref/tests/helmholtz/diagonal.rs` (the error measure:
  P2M of complex charges in the source box, L2P at points of the target box, the larger of the
  relative ℓ2 errors of φ̃ and ∇φ̃).
- `fmm-tables/src/helmholtz/diagonal/{mod,level}.rs` (CH4.1a, CH4.1b: `Conversions`,
  `Signatures`, `m2l_level`), `fmm-tables/src/helmholtz.rs` (level units: `Level::new(k̂_l, 1)`,
  |t| = 2|d|; the factor 4π ik e^{2β − Im(k)|t|} folded into the RCR M2L tables),
  `fmm-tables/src/parts.rs`, `fmm-tables/src/geometry.rs` (`m2l_offsets`).
- CONVENTIONS §3.15 "Signatures" (the classes and the sample maps; "Not fixed here": how T̂ is
  computed and stored is CH4.2's crate documentation).

## Changes
- `nd-fmm-tables`, new `helmholtz::diagonal::transfer` (re-exported from `diagonal`):
  - `DiagonalM2l<T>`: one level's transfer functions in level units (k̂_l, half-width 1,
    |t| = 2|d|) at degree p, for the classes a caller names (indices into
    `nd_fmm_ref::helmholtz::diagonal::classes()`, ordered by |t|² then representative), each
    stored once: sample (i, j) holds W = 4π ik̂ e^{2β − Im(k̂)|t|} T̂_rep(ŝᵢⱼ), T̂_rep from ref's
    `Transfer::<f64>` of the representative (double-double, rounded once), the factor formed in
    f64 as `HelmholtzTables` forms the RCR M2L's and multiplied in f64, then rounded to T (an f32
    table is the f64 table rounded). The factor folded in, as the RCR tables fold it: one complex
    multiply-add per sample in the pair, not two multiplies.
  - Per V offset (the 316 of `m2l_offsets`, by table index) an `OffsetMap`: its class, the ring
    flip (z → −z) and the azimuth map j ↦ (start ± j) mod N_φ, the composition of CONVENTIONS
    §3.15's reflections, derived from ref's `Symmetry::sample` and checked against it.
  - `pair(offset, signature, accumulator)`: accumulator += W_σ · signature per sample, ring by
    ring over at most two contiguous runs per ring (forward or reversed), no index table; each
    sample's chain is the accumulator plus one product (complex, product and sum rounded apart,
    no `mul_add`), so a target's bits depend on its row alone.
  - `offset_samples(offset, out)`: the offset's W materialised (the trusted path's comparison,
    and CH7.4's per-offset materialisation on the device).
  - `build(p, k̂, classes)` and `build_on(…, parts)` (one part per class, put together in class
    order: bit for bit on any map), `bytes`, `bytes_for(p, classes)`, accessors.
  - `class_bounds(p, k̂, constants)`: E(l, c) of proposal §3.6 per class (ref's `bound` at
    |t| = 2|rep|, a = √3), the input of CH4.2b's and CH4.3's rule (constraint 9: k, the level,
    p and the class alone). No rule here.
- `nd-fmm-ref`, tests only: review M11's ignored release test, `diagonal::m2l` against
  `rcr::m2l` within 2E at k a = 108 (p = 131) and 216 (p = 245, CH4.4's level 2 of W1 at
  kD = 1000) on a few classes (the nearest, a middle one, the farthest).
- Docs: `fmm-tables/CLAUDE.md`, `fmm-ref/CLAUDE.md` (the M11 test), the module docs (the
  storage of W, the maps, the pair's order; CONVENTIONS' "Not fixed here" items CH4.2 fixes).
- No convention, no dependency, no Laplace file, no `nd-fmm-plan` interface, no nd-fmm-exec
  change (CH4.2b).

## Tests that prove it
`fmm-tables/tests/helmholtz/diagonal_transfer.rs`, f64 unless said:
- The maps: for p = 3 and 8, every offset of `m2l_offsets` and every sample (i, j), the map's
  sample equals ref's `class_of(o).1.sample(…)`, and its class the position of the
  representative in `classes()`.
- The trusted path sample by sample: each stored class bit for bit ref's `Transfer` of the
  representative times the factor, rounded; every offset's `offset_samples` bit for bit ref's
  `Transfer::permuted` times the factor, and within 4 u of the largest sample of ref's
  `Transfer::new` of the offset itself (real and damped k̂, p = 6; f32 the f64 table rounded).
- The pair: bit for bit the naive loop acc[s] += W_o[s] · F[s] over `offset_samples`, every
  offset, p = 5, f64 and f32; a class not stored panics.
- Acceptance, against RCR within E(l, c): the tables' path (`Conversions` S → F, the pair,
  F → R through `m2l_level`, one source and one target) against ref's `rcr::m2l` in CH1.3d1's
  measure, ≤ 2E, on the 34 classes at k a = 2.83 (p = 7), 8 and 16 (the formula's p at 10⁻⁶),
  weakly damped k a = 8 (Im k = Re k / 10); the 316 offsets at k a = 5, p = 16; f32 at
  k a ≈ 16, p = 30 within the f32 bound (C_T = 0.18, C₀ = 50, u = 2⁻²⁴). Ignored, in release:
  k a = 32 (the formula at 10⁻⁸), 54 (p = 72 and 82) and 108 (p = 131) on the 34 classes.
- Determinism with the RCR pairs added after F → R: a small level of unit boxes (as CH4.1b's
  `case`, in level units) whose targets' V rows mix diagonal classes and RCR classes; the
  diagonal pairs through `m2l_level`, then each target's RCR pairs through `HelmholtzTables`'
  M2L in row order: every chunk size bit for bit the whole level, and the result against
  the level with every pair RCR, in the FMM measure (P2M of charges in each source box, L2P
  at points of each target box), within 2 max E of the diagonal classes used.
- The build in parts: `build_on` on a map that runs the parts in reverse on three threads, bit
  for bit `build`; the bytes without building; `class_bounds` equal to ref's `bound` per class.
- ref (M11): `m2l_against_rcr_at_the_degrees_of_kd_1000` (ignored, release): ref's diagonal
  `m2l` against `rcr::m2l` within 2E at (k a, p) = (108, 131) and (216, 245) on the classes
  (0, 0, 2), (1, 2, 2) and (3, 3, 3).

## Checks
`/verify` (fmt, Clippy default and workspace all-targets, rustdoc `-D warnings`, the default and
workspace tests at opt-level 1); `cargo clippy -p nd-fmm-tables --all-targets`, `cargo test -p
nd-fmm-tables` and `-p nd-fmm-ref` (the tables' Helmholtz test executable under a minute in
debug), the new ignored tests in release. No MPI crate's code changes.

## Machine runs
None: correctness on the M6 (no timing; the build and level times are CH4.2b's).

## Constraints and risks
- Constraint 1: Laplace untouched (Helmholtz modules of `nd-fmm-tables`, a ref test).
- Constraint 2: the pair's per-sample chain is fixed and per target; the level's order is
  CH4.1b's; the RCR pairs after F → R in row order (tested bit for bit at every chunk size).
- Constraint 5: tables in f64 from ref, rounded once to T after the factor.
- Constraint 6: the trusted path is ref's `Transfer` and `diagonal::m2l`, and ref's RCR.
- Constraint 9: the tables and `class_bounds` are functions of k̂, p and the classes alone.
- Risk: the build is O(p³) double-double per class (S6: about 5 s per class at p = 256, so
  about 0.7 s at p = 131, 24 s for 34 classes serially); one part per class bounds the
  parallel build at 34 parts per level. CH4.2b measures it and may need parts by rings or the
  interpolated build (proposal §5.4, C_T = 0.23).
- Risk: the debug budget of the tables' Helmholtz tests (a minute): the acceptance cases are
  ref's own, whose sizes run in debug there.

## Pending runs
None planned.

## Open questions
None yet.
