# CH3.1a The Helmholtz operator

Started 2026-11-01 on `tbetcke/h-ch3-1a`. Roadmap: "CH3.1a The Helmholtz operator" (H3;
the first sub-task of CH3.1, tenth on the priority list). Proposal: §10 CH3.1; §2
(constraints 1, 3, 5, 6, 9), §3.3, §3.4, §3.5, §5.1–§5.3, §5.7, §6.1–§6.3. S7:
`spikes/truncation/SPIKE_REPORT.md` §3 and §8 (question 1, signed off 2026-10-29).

CH3.1 was split on 2026-11-01 into a/b/c (the user: "a/b/c"). The user also decided, on
2026-11-01 ("Yes, both as recommended"):
- the degree is a rule behind a setting: S7's closed form at a tolerance by default, degrees
  named per level as the override (this task);
- CH3.1b extracts the operator-independent point pipeline of `fmm.rs` into crate-private
  code that `Fmm` and the Helmholtz FMM share, with Laplace's outputs bit for bit and
  "minimal timing runs to check the impact" for D9 (CH3.1b, not this task).

## Goal

The Helmholtz-family operator of the plan, for real and complex k with real and complex
charges, on the unchanged shared drivers: RCR M2M, L2L and M2L on every level with
per-level degrees and tables, and the leaf operators and P2P on §3.15's chunks.

## Acceptance (proposal §10, verbatim)

CH3.1: "FMM against the direct sum on W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500;
deterministic". That is CH3.1c's run. CH3.1a's own *Done when* (roadmap): every body
against `nd_fmm_ref::helmholtz` (the translations against `rcr` to rounding on the 8
octants and the 316 offsets, with rectangular degrees, below and above the projection
threshold; the leaf operators and P2P against `leaf` and `p2p`); the chains against the
direct sum within the truncation; vector v the one-vector operator bit for bit; real
charges as complex ones with Im q = 0; Yukawa refused (its path is CH2.1's); Laplace
untouched.

## Read first

- `fmm-ref/src/helmholtz/rcr.rs` (the three steps, the stored form, the table builders),
  `leaf.rs` (`p2l`, `m2p`, `check_frame`), `p2p.rs`, `level.rs`.
- `fmm-math/src/coaxial/mod.rs` (`CoaxialTable`, the rule of §3.15), `orthonormal/rotation.rs`
  (`y_blocks`, `compact_offset`).
- `fmm-tables/src/rotation.rs` (Laplace's table-driven point and shoot: the shape to follow).
- `fmm-exec/src/driver.rs` (`HostKernel`, `Execution`), `operator.rs` (`Kernels`,
  `LaplaceOperator`: the bodies, scratch, `PairOperator`), `helmholtz/densities.rs`.
- `docs/helmholtz/tasks/CH3.5.md` ("Constraints and risks": the levels of P2L, M2P, P2P).

## Changes

Two crates; no Laplace file changes.

1. **`nd-fmm-tables`: `helmholtz` (new module).** `HelmholtzTables<T>`, the RCR tables of
   an octree's levels and their application:
   - Built in f64 from a root wavenumber k̂₀ = k r₀ (r₀ the root's half-width) and the
     degrees p_l; level l works in its own units, k̂_l = k̂₀ / 2^l (exact), with
     `nd_fmm_ref::helmholtz::Level::new(k̂_l, 1)`. Entries rounded to T.
   - M2M from level l + 1 to l and L2L from l to l + 1: the coaxial tables of
     `rcr::{m2m_table, l2l_table}` with the power ratios of the stored form folded in (the
     `rcr` module docs, "Stored form"); real for real k (the real part, §3.1), complex
     otherwise. M2L on levels ≥ 2: `rcr::m2l_table` per distinct |t| of the 316 shifts
     with the factor 4π ik e^{2β − Im(k)|t|} folded in.
   - As built: the tables are keyed by exact integer classes (|d|², 15; the polar angle by
     (sgn d_z, d_z²/|d|²) reduced, 49; the azimuth per offset by (d_x, d_y)/gcd), their
     values formed from the integers by correctly rounded square roots, the same on every
     machine. (The first version keyed them by the bits of `rcr`'s floating-point angles
     and distances, 21 and 53 on the M6, for bit-identity with `rcr`; the user found that
     brittle and chose exact classes on 2026-11-01, "switch to A in this PR".) The M2L
     coaxial tables come from `CoaxialTable::m2l` at |t| = 2√|d|².
   - As built (found by the chain tests): a box of level l holds r_l φ̃ in its own units,
     as a chunk does. A multipole's factor 4π ik̂_l carries the change of units, so M2M and
     M2L need none; a local has no such factor, so the L2L tables carry ½ (r_{l+1}/r_l)
     and P2L scales its charges by r̂(t|s). Both are exact (powers of two).
   - The y-blocks: M2M and L2L of a level pair share two polar angles (cos θ = ±1/√3),
     always stored; M2L's per distinct (cos θ, sin θ) of the 316 shifts (keyed by their
     bits, so each block is the one `rcr` forms for that shift), stored level by level from
     the finest where they fit in the remaining byte budget (default 512 MiB), else formed
     per call into scratch. As built, the blocks (and angles) are formed in T as `rcr` forms
     them, so that stored and per-call blocks agree: for f32 that departs from constraint 5
     (a question for sign-off).
   - Application `m2m`, `l2l`, `m2l` for m vectors per box, the input and output planes
     (1 or 2) of each expansion, accumulated (+=): rotate (z by −φ, the transposed
     y-block), the coaxial step per order, rotate back. Real entries act on each plane
     (the real shifts of real k); complex entries take one or two input planes to two
     output planes. Scratch caller-owned (`HelmholtzScratch`), no allocation in
     application.
   - The three steps and the folded factors restate `rcr`'s private `translate` and its
     stored-form factors (CONVENTIONS §3.15; the module docs say so); the coaxial tables
     come from `nd_fmm_ref::helmholtz::rcr`. This departs from the crate rule "never
     re-derive an operator formula here", as Laplace's `rotation` and `plane_wave` do;
     the tests pin it to `rcr` bit for bit. `fmm-tables/CLAUDE.md` records it.
   - No on-disk cache (CH2.2's per-level cache keys).
2. **`nd-fmm-exec`: `helmholtz::degree` (new).** `degree(k, circumradius, tolerance)`: S7's
   closed form p = max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉) at
   Re(k) a (complex k provisional until CH5.2; S7 §3, η = 0.1 within 10 degrees).
   `Degrees::{Tolerance(ε), PerLevel(Vec<usize>)}` and `Degrees::resolve(k, domain,
   nlevels)`: the rule on levels ≥ 2, degree 0 on levels 0 and 1, which carry no
   expansion that is read (no V list above level 2, no X list into them, no W box on
   them). Inputs k, the domain and ε only (constraint 9).
3. **`nd-fmm-exec`: `helmholtz::operator` (new).** `HelmholtzOperator<T>`:
   - `new(k, densities, domain, degrees, gradients, max_leaf_points)`, `with_pool`;
     `OperatorError` for Yukawa (CH2.1's path), a `Densities` of another kind, no levels.
   - `FmmSizes`: the `Densities` sizes at p_l; `FmmOperator` through `driver::*` (the
     unchanged shared drivers; `batched_m2l` `None`, an empty `Columns`); `PairOperator`
     for the pair tests and `PerPair`.
   - Bodies: P2M and L2P by `Densities::{p2m, l2p}`; M2M, L2L, M2L by `HelmholtzTables`;
     P2L and M2P by `nd_fmm_ref::helmholtz::leaf::{p2l, m2p}` at the plan's frames, the
     `Level` in the points' units (k̂ of the source leaf for P2L, of the target leaf for
     M2P, the frame's radius as half-width); a real multipole (real k, real charges)
     widened to two planes in scratch for M2P; P2P by `nd_fmm_ref::helmholtz::p2p::p2p`
     (the reference kernel until CH3.4), the sources mapped into the target leaf's units
     as Laplace does. Every vector gets the one-vector operator.
   - `fmm-exec/CLAUDE.md`'s rules updated.

No `nd-fmm-plan`, `nd-octree`, `nd-fmm-simd`, manifest, dependency, convention or Laplace
change. The public API grows by the modules above.

## Tests that prove it

All f64 but one f32 check, one thread, no MPI. As built (the plan's first list had per-pair
bit-for-bit checks and a level-independence test; the chain tests below replace them, the
low-degree one in absolute units, which also pins the change of units).

`nd-fmm-tables`, `tests/helmholtz/` (new executable, 0.6 s in debug):
- Every comparison with `rcr` is in the FMM-weighted error of `nd-fmm-ref`'s translation
  tests, within 2e-13 times max(1, |k̂| a) of the level (the tables' and `rcr`'s angles
  and distances round apart, and a relative change u moves the coaxial phases by about
  |k̂| |t| u); 1.7e-14 measured on the M6.
- M2L on all 316 offsets on levels 2–4 for five k̂₀ (real and complex, |k̂| a from about
  0.005 to 17), two planes and for real k one plane widened by Im = 0.
- M2M and L2L on the 8 octants between every pair of five levels with rectangular degrees
  (L2L against ½ `rcr`).
- Above the projection threshold (Re(k̂) a ≈ 52): M2L on all 316 offsets, M2M and L2L.
- Stored and per-call blocks bit for bit (budgets 0, finest level only, default); several
  vectors bit for bit one at a time; the class counts (15 and 49); f32 within 1e-5 of
  f64; Yukawa and a one-plane complex step refused.

`nd-fmm-exec`, `tests/helmholtz/` (`degree.rs`, `operator.rs`; 5.7 s in debug with the
existing tests):
- `degree`: S7's closed form equals the table's closed-form column and is ≥ p* at seven
  rows of k a × eight ε; complex k takes Re k; `resolve` against degrees computed by hand,
  levels 0 and 1 at 0, named degrees, the errors.
- Chains by pair calls (far: P2M → M2M → M2M → M2L on level 2 → L2L → L2L → L2P; v4: M2L
  on level 4; x: P2L from a level-3 leaf; w: M2P from a level-4 box; near: P2P from a
  neighbour and the self pair) against the direct sum, relative L2 of φ and ∇φ below ε at
  ε = 10⁻⁶ and 10⁻¹⁰ (P2P below 1e-13), real k at kD = 1, 20, 100 and complex k 20 + 2i,
  real and complex charges.
- At named low degrees (p = 5–8), where truncation dominates (10⁻⁵ to 10⁻¹ against the
  direct sum): the far, v4, x and w chains against the same chains of
  `nd_fmm_ref::helmholtz` (`leaf`, `rcr`) in absolute units, within 2e-13 (1.2e-14).
- P2P bit for bit against `nd_fmm_ref::helmholtz::p2p::p2p` on the mapped sources.
- Vector v of three the one-vector operator bit for bit; real charges bit for bit complex
  ones with Im q = 0 (asserted within 4 ε); sizes and errors.

The level calls through the drivers (`FmmOperator`) need a plan, so MPI; they are tested in
CH3.1b (the level calls against `PerPair` bit for bit, and the FMM against the direct sum).

## Checks

`/verify`: format; Clippy default, examples and workspace all-targets; `cargo test`
(default) and `--workspace`; `cargo doc`; `cargo test -p nd-fmm-tables` and
`-p nd-fmm-exec` and their Clippy; the ignored large-p table tests in release if any are
added. No manifest change, so no licence check.

## Machine runs

None (no Laplace change, no timing).

## Constraints and risks

- Constraint 1: nothing of Laplace changes (new modules only; `driver.rs` untouched).
- Constraints 3, 5, 9: `Value` real, complex quantities in split planes; tables in f64,
  rounded when stored; degrees and tables from k, the domain and ε only.
- Constraint 6: every fast path against `nd_fmm_ref::helmholtz` and the direct sum.
- Risk: the coaxial (S|R) start column leaves the f64 range near p_src + p_tgt ≈ 170 where
  s < 1 (`coaxial` docs); the closed form keeps low-frequency p ≤ 58 at 10⁻¹⁰.
- Risk: the M2M and L2L power factors (s_c/s_P)ⁿ s_P^{n−n′} overflow where p log₁₀(1/s_P)
  exceeds about 300 (`rcr`'s documented limit): at ε = 10⁻¹⁰ from about level 14 at
  kD = 0.1. `HelmholtzTables::build` panics with a message there rather than build
  non-finite tables.
- Risk: memory of the top levels at kD = 500 (p ≈ 135: 15 M2L tables of about 13 MB, y-blocks
  of 13 MB per angle); the block budget bounds the latter. CH3.1c measures.
- Risk: per-call y-blocks double the M2L's cost where the budget binds; CH3.2 and CH3.6
  measure.

## Pending runs

None. `/verify` ran on the M6 on 2026-11-01, all passed (progress entry).

## Open questions

None blocking. For sign-off, from the PR:
1. The y-blocks (and the angles they come from) are built in T, as `rcr` builds them per
   call, so that stored and per-call blocks agree bit for bit. In f64 nothing changes; in
   f32 this departs from constraint 5 ("tables in f64, rounded when stored") and the crate
   rule, which `fmm-tables/CLAUDE.md` now names as an exception. *Recommended*: accept,
   since f32 is reserved until H7 (D3), and revisit there.
2. The level units: a box of level l holds r_l φ̃ in its own units, as §3.13's chunks do.
   So the L2L tables carry ½ and P2L scales its charges by r̂(t|s). This is a scaling
   convention of the stored expansions. *Recommended*: record it in the module docs now,
   and add it to CONVENTIONS §3.15 at its next revision (no version bump now; nothing on
   disk depends on it).
3. Complex k's degree is the closed form at Re(k) a, provisionally until CH5.2.
   *Recommended*: accept.

The question on keys by bits was settled on 2026-11-01: exact integer classes ("switch to
A in this PR").
