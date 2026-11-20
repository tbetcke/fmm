# CH4.2c1 The transfer functions interpolated, and built by ring

Started 2026-11-20 on `tbetcke/h-ch4-2c1`. Roadmap: CH4.2c "The transfer functions
interpolated, and built in parts finer than a class" (added at CH4.2b2, its question 2, signed
off 2026-11-20 on #239), split on 2026-11-20 by the size rule of `/next-task` (four crates and
measurements on Kathleen) into CH4.2c1 (the build, `nd-fmm-ref` and `nd-fmm-tables`) and CH4.2c2
(the operator's setting and rule, the harness, W1's tables timed on Kathleen). Proposal: §10
CH4.2 (CH4.2c is a follow-up of it); §5.4 (the transfer functions "per sample in O(p³) per class
at build time, or interpolated in ŝ · t̂ from an extended-precision table in O(p²) per class with
C_T = 0.23", amended at CH0.6); §3.6 (the bound); §2 (constraints 1, 2, 5, 6, 9).

## Goal
A second build of `DiagonalM2l`'s transfer functions, T̂ interpolated in ψ = arccos(ŝ · t̂) from
a double-double table on a uniform grid in ψ (S6's `Interp`), shown within the bound at
C_T = 0.23 against the direct build and RCR; and both builds in parts of one ring of one class,
bit for bit the build by class.

## Acceptance (the roadmap's CH4.2c, the part of it this task covers)
CH4.2c's *Done when*: "T interpolated in ŝ · t̂ from an extended-precision table (proposal §5.4;
S6: 11× faster at p = 256, C_T = 0.23) against the direct build within the bound at C_T = 0.23
on every class CI runs; the build in parts by ring, bit for bit the build by class; W1 at
kD = 500 and 1000's diagonal tables timed at 1 × 40 and 40 × 1 against CH4.2b2's (2.3 s and
61 s at p = 131)."

CH4.2c1 takes the first two clauses; CH4.2c2 the third (the operator chooses the build, and W1
is timed there). Proposal §10 has no row for CH4.2c; CH4.2's row ("against RCR on the same class
within E(l, c)") holds for the interpolated tables with E at C_T = 0.23. "Within the bound" is
read as CH4.2a read it: measured ≤ 2E = E/θ (S6 fitted C_T so that the measured rounding stays
within 2E).

## Read first
- `spikes/diagonal/src/transfer.rs` (`Interp`, `angle`) and `SPIKE_REPORT.md` §1.3 and §7: the
  table of M = ⌈3L⌉ + 1 points on [0, π] (L = 2p, so M = 6p + 1), double-double rounded to
  f64, read by Gaussian-regularised sinc interpolation (Qian 2003) with 36 terms each side, at
  a position computed in double-double; ratio to the bound 0.45 (C_T = 0.23), rounding 0.3–6×
  that of T rounded per sample.
- `fmm-ref/src/helmholtz/diagonal/{transfer,grid,dd,bound}.rs` (`Transfer`, `Grid`'s
  double-double directions, `BoundConstants::F64_INTERPOLATED`, "not built here").
- `fmm-tables/src/helmholtz/diagonal/transfer.rs` (`DiagonalM2l::build_on`, one part per class),
  `fmm-tables/src/parts.rs`, `fmm-tables/tests/helmholtz/{diagonal_transfer,parts}.rs`.
- `fmm-exec/src/helmholtz/operator_diagonal.rs` (`select`: `DiagonalM2l::build_on` on the
  pool; unchanged here).

## Changes
- `nd-fmm-ref` (`helmholtz::diagonal`; the oracle computes what is double-double, nothing is
  summed in the working precision, its crate rule):
  - `Transfer::ring_samples(grid, level, from, to, ring)`: one ring's samples, bit for bit
    `Transfer::new`'s (the series' set-up and the per-sample loop shared, the same operations in
    the same order), so a build may cut a class into rings.
  - `AngleTable` (new, `diagonal/angles.rs`): for a shift t and degree p, T̂ at ψ_k = kπ/M,
    k = 0..=M, each in double-double (the same series as `Transfer`: bandwidth 2p, e^{Im z})
    rounded once to f64; and `position(grid, i, j)`: the sample's ψ M/π = n + f in double-double
    (S6's `angle`: atan2 in f64 of the double-double cross and dot products, corrected once in
    double-double), returned as (n, f) with 0 ≤ f < 1 in f64. f64 only (tables are built in f64).
- `nd-fmm-tables` (`helmholtz::diagonal::transfer`):
  - `TransferBuild { Direct, Interpolated }` with `bound_constants(single)`: Direct's f64 S6
    constants (`BoundConstants::F64`), Interpolated's `F64_INTERPOLATED` (C_T = 0.23), and
    `F32` for an f32 table either way (an f32 table is the f64 table rounded; the
    interpolation's f64 error lies about 2²⁹ below f32's rounding; tested in f32 below).
  - `DiagonalM2l::build_by(p, k̂, classes, build, parts)`: Interpolated maps one part per class
    (its `AngleTable`), then both builds map one part per (class, ring), put together in
    (class, ring) order, so the tables are the serial build's bit for bit on any map.
    `build` and `build_on` are `build_by` with `Direct` (CH4.2a's tables bit for bit, now in parts
    by ring, which `nd-fmm-exec`'s `select` gets through `build_on` without a change there).
  - The interpolation (private): S6's sum, 2·36 terms of the table's even, 2π-periodic
    extension, the Gaussian window r² = (h − 2)/(π − δ), δ = Lπ/M, in f64; then the factor
    4π ik̂ e^{2β − Im(k̂)|t|} in f64 as the direct build, rounded to T. Constants
    `INTERPOLATION_OVERSAMPLING` (3) and `INTERPOLATION_TERMS` (36), S6's.
- No convention (CONVENTIONS §3.15 "Not fixed here" leaves how T̂ is computed to CH4.2), no
  dependency, no interface of `nd-fmm-plan`, no Laplace path.

## Tests that prove it
In `fmm-tables/tests/helmholtz/diagonal_transfer.rs` (the measure of CH4.2a: CH1.3d1's, the larger
of the relative ℓ2 errors of φ̃ and ∇φ̃ after L2P in the target box, P2M of complex charges in the
source box) and `parts.rs`; one thread unless said:
1. **Per sample**: every class's interpolated W against the direct W, at p = 6 (k a = 5, real and
   η = 0.1) and p = 16: the largest difference within 64 u of the class's largest sample
   (recorded; S6: 0.3–6× the per-sample rounding).
2. **The acceptance against RCR** (CH4.2a's `against_rcr` with the build named): the
   interpolated tables within 2E at C_T = 0.23 (`F64_INTERPOLATED`) on the 34 classes at
   k a = 2.83 (p = 7), 8 and damped 8 (η = 0.1); the 316 offsets at k a = 5 (p = 16); f32 at
   k a = 8 within 2E of `F32`. Ignored, in release: k a = 16 and the large degrees (k a 32–108,
   CH4.2a's floor reading against the projection).
3. **The acceptance against the direct build**: the interpolated tables' M2L against the direct
   tables' on the same cases (34 classes at k a 2.83, 8, damped 8; 316 offsets at k a 5) within
   2E at C_T = 0.23.
4. **The build in parts by ring**: `build_by` of both builds on `parts.rs`' map that runs the parts
   in reverse on three threads bit for bit `Serial`; `build_on` bit for bit CH4.2a's tables
   (`Transfer::new` times the factor, the existing `the_samples_are_refs_transfer_functions`) and
   `build_by(Direct)`; f32 the f64 tables rounded.
5. `nd-fmm-ref` unit tests: `ring_samples` bit for bit `Transfer::new`'s rings; `AngleTable`'s
   positions: cos(π(n + f)/M) against ŝ · t̂ in f64 within 4 ε, 0 ≤ f < 1; its values at the nodes
   against `Transfer` where a sample falls on a node is not available, so: its first value T̂(0)
   against Σₗ cₗ (the series at μ = 1) within 4 ε of it, and the table's values against
   `Transfer`'s samples through the interpolation (test 1).
6. Ignored, printing (no assert of a time): one class's build per p (41, 72, 131, 245) on one
   thread, direct against interpolated (`time_one_class`), for CH4.2c2's report and the measured
   speed-up against S6's 5.4× at p = 128 and 11× at p = 256.

The tables' Helmholtz test executable is at its minute in debug (CH4.2b1: 67 s at load 22–42):
the new debug tests stay at p ≤ 16, the rest ignored.

## Checks
`/verify` (fmt; Clippy default, workspace and all targets; rustdoc `-D warnings`; the default and
workspace tests at opt-level 1; `nd-fmm-tables`' and `nd-fmm-ref`'s ignored tests in release;
the tables' and exec's Helmholtz executables at opt-level 0 against their minute). No MPI code
changes (exec's `select` unchanged; its build now in parts by ring through `build_on`, bit for
bit); `helmholtz_fmm`'s `check_diagonal` covers it.

## Machine runs
The ignored timing test (6) on locust, one Grace core (`taskset -c 8`), release, idle (load
0.0–0.2 before and after), 2026-11-20, through `tools/gh200/sync.sh` and `remote.sh`: no
decision rests on it (CH4.2c2 times W1 on Kathleen); it checks S6's expected speed-up and found
where the time goes. Not on the M6 (the user runs other jobs there).

## Constraints and risks
- Constraint 2 and 9: each part a function of p, k̂, the class and the ring; libm's `atan2`,
  `sin` and `exp` in f64 make the interpolated tables a function of the machine's libm too, the
  same on every rank and thread of a run (one binary, one libm); not bit for bit between macOS
  and glibc (as nothing that calls libm is).
- Constraint 5: both builds in f64 from double-double values, rounded once to T.
- Constraint 6: the interpolated tables against the direct tables, RCR and the projection.
- Constraint 1: Laplace not touched (Helmholtz modules of `nd-fmm-ref` and `nd-fmm-tables`).
- Risk: the interpolation's error near ψ = 0 and π (where |dT/dψ| is largest and T peaks):
  S6 handled it by the double-double position; test 1 covers every sample of every class.
- Risk: the debug executable's minute; heavy cases ignored.

## Results

**Accuracy** (the M6, debug unless said; the tests' printed figures):

| check | case | measured |
| --- | --- | --- |
| interpolated W against direct W, per sample | p = 6, 16; k a 5, 8; real and η = 0.1 | 3.3–4.1 ε of the class's largest (bound 32 ε) |
| interpolated M2L against RCR, measured/E at C_T = 0.23 | 34 classes, k a 2.83, p = 7 | 1.35 (S6's worst case; the direct build 0.55 at C_T = 0.15) |
| | 34 classes, k a 8, p = 27; damped | 0.33; 0.03 |
| | 316 offsets, k a 5, p = 16 | 0.31 |
| | f32, k a 8, p = 16 (`F32`) | 0.21 |
| interpolated M2L against the direct build's, measured/E | the same cases | 0.27, 0.08, 0.02, 0.75 |
| release, ignored | k a 16 (p = 36) against RCR; against direct | 0.59; 0.04 |
| | k a 32 (p = 61), 54 (p = 72, 82) against RCR | 1.10, 1.79, 2.07 (at E's floor (3, 3, 3) at p = 82: 6.8·10⁻¹⁴ from the projection, RCR 1.4·10⁻¹³) |
| | k a 108 (p = 131), three classes | 3.04 against RCR (direct 3.72); nearer the projection than RCR |

A mutation (4 interpolation terms in place of 36) fails all three interpolated tests.

**Bits.** Both builds on the reversed three-thread map are the serial build bit for bit; the
direct build by ring is CH4.2a's by class (ref's `Transfer::new` samples hashed on `main` and on
the branch at p = 7, 16 and 24 over the 34 classes: `2407511671127d4a` both).

**Positions.** S6's `angle` costs one double-double sine and cosine per sample: on locust it was
53 % of the interpolated build at p = 131 (0.090 of 0.171 s) and 49 % at 245. `position` now
takes δ from the nearest node (the table's node sines and cosines, an f64 arcsine correction of
size ≤ (π/2M)²/6): the same (n, f) as S6's method on every sample tested (p = 1–40, four shifts;
largest difference 0, the test's bound 4 u), at about a fifth of the cost.

**Time** (`time_one_class`, locust, one core, class (0, 0, 2), one build each):

| p | k a | direct s | interpolated s | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 41 | 27 | 0.020 | 0.008 | 2.3 |
| 72 | 54 | 0.096 | 0.026 | 3.7 |
| 131 | 108 | 0.538 | 0.089 | 6.1 |
| 245 | 216 | 3.427 | 0.358 | 9.6 |

Before the positions' change: 0.017, 0.051, 0.171 and 0.644 s (1.2–5.3×). S6 measured 5.4× at
p = 128 and 11× at 256 against its own direct build, which was about 1.6× slower than ref's. The
rest of the interpolated build is the 72 `exp`s per sample (about 40 % at p = 131) and the sums.

## Pending runs
None.

## Open questions
1. The acceptance as read (the PR's question 1): signed off 2026-11-20 with the merge request,
   the user, 2026-11-20: "Merge the PR" (no separate answer to the question; accepted as recommended).
