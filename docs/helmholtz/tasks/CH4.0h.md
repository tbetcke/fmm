# CH4.0h The Helmholtz build by part, and parallel

Started 2026-11-16 on `tbetcke/h-ch4-0h`. Roadmap: "CH4.0h The Helmholtz build by part, and
parallel" (H4; priority list item 6, added on 2026-11-16 by CH4.0e's sign-off). Proposal: no
§10 row (CH4.0e's report §3.2 and §7 question 1, signed off as recommended on #220); §2
(constraints 1, 2, 9), §6.4 (the M2L's tables per level).

## Goal

Know where the Helmholtz FMM's build spends its time, and make it use the node: time the
build by part, then remove the serial work that is repeated or never read and run the
remaining per-level table work on the FMM's threads, the output bit for bit.

## Acceptance (roadmap *Done when*; no proposal §10 row)

W1 at kD = 500's build on the node is at most half its one-core time, the output bit for
bit; Laplace unchanged.

Read as: at each of 10⁻³, 10⁻⁶ and 10⁻⁸, the build at the node's best split (Kathleen 1 × 40
and the other splits of CH4.0e; locust 1 × 72 and its splits) at most half the build on one
core of the same machine, both after this task; also reported against CH4.0e's one-core
builds (Kathleen 5.5 s, locust 4.5 s at 10⁻⁶). "Bit for bit": every checksum of the runs
equal to CH4.0e's launches of the same case, split and machine, and the tests below.

## Read first

- CH4.0e's report §3.2 ("The build": 5.5 s on one Kathleen core, 5.7, 4.6, 4.0, 4.0 s at
  1 × 40 … 8 × 5; it follows the degrees, not N) and §7 question 1.
- `fmm-exec/src/helmholtz/fmm.rs` `HelmholtzFmmBuilder::build` (steps 1–8);
  `fmm-exec/src/helmholtz/operator.rs` `HelmholtzOperator::with_block_budget`, the `with_*`
  setters and `select_tables`.
- `fmm-tables/src/helmholtz.rs` `HelmholtzTables::build_with_m2l_degrees` (per level: M2M and
  L2L coaxial tables, the octant y-blocks, 15 M2L coaxial tables, then the M2L y-blocks of 49
  angles from the finest level within the block budget); `helmholtz/dense.rs` and
  `batched.rs` (`DenseM2l::build`, `BatchedM2l::build`: each builds a whole
  `HelmholtzTables<f64>` of every level at `DEFAULT_BLOCK_BUDGET`, then 316 matrices per
  chosen level); `helmholtz/plane_wave/merge.rs` `PlaneWaveM2l::build_with_reach`.
- `fmm-tables/CLAUDE.md` ("Builds are serial and deterministic"; rayon not allowed) and
  `fmm-exec/CLAUDE.md` (rayon only in nd-fmm-exec; the Helmholtz rules).

Found reading the code, before any run (to be confirmed by the timings):
1. Under the builder the operator's RCR tables are built at the degrees given in
   `with_block_budget`, then `with_m2l(Auto, …)` builds them again where Auto's exponential
   M2L raises a level (every ε below 10⁻⁶).
2. `BatchedM2l::build` (and `DenseM2l::build`, `ClassM2l::build`) builds a whole second set of
   RCR tables in f64, every level's, the coarse levels at p = 131 included, to form the
   matrices of the few fine levels it batches.
3. The M2L y-blocks are stored for every level within the budget from the finest up, also on
   levels whose M2L is the exponential one or Batched and never reads them (CH4.0e's column
   "blocks" says "stored" on the levels with waves).
4. Every table is built on one thread; each level, and within a level each coaxial table and
   each angle's blocks, is independent of the others.

## Changes

Step 1, measure (the roadmap's "time the build by part in the harness"):
- `nd-fmm-exec`: `HelmholtzBuildTimings` (pub, documented), `HelmholtzFmm::build_timings`:
  the settings and domain (steps 1–3), the octree and its index (step 4), the routes (steps
  5–6), the counts and the plan (step 7's first part), the operator (its tables; with the
  operator's own parts: RCR tables, Batched/Dense/Classes, the exponential M2L's tables, the
  scratch sets), the evaluator, the coordinates (step 8). Clock reads only, no value changed;
  the agreeing all-reduces stay where they are.
- `HelmholtzOperator::table_timings` (the parts of the last `select_tables` and of
  construction).
- `nd-fmm-validate`: `Row::build_parts` and a table "Build" in `helmholtz_accuracy`'s output
  (a row per run: the parts, the slowest rank's each).

Step 2, the fixes, each bit for bit:
- Build the operator's RCR tables once, at the degrees the setting runs (findings 1): the
  builder names every setting before the tables are chosen. Design: the operator's
  constructor keeps its contract (it builds the tables at the degrees given, so a caller who
  never names a setting sees no change); the builder uses a new constructor that takes the
  M2L settings first and builds the tables once. The output the same (the tables are a
  function of the degrees and the budget).
- Batched, Dense and Classes from the operator's levels (finding 2): `nd-fmm-tables` gains a
  build of only the levels chosen (`HelmholtzTables::<f64>` restricted to the chosen levels'
  M2L tables), so the matrices are formed from the same per-level values bit for bit (the
  M2L of a level reads only that level's M2L shifts, its blocks and the offsets' keys; stored
  and per-call blocks agree by construction).
- The M2L y-blocks only for the levels whose M2L reads them (finding 3): the operator passes
  the levels that run RCR's M2L; the others store none and spend none of the budget, so
  another level may now fit (the output does not depend on the budget: CH4.0e's
  `check_block_budget`).
- In parallel (finding 4): `nd-fmm-tables` stays free of rayon and its builds deterministic;
  its builders take a caller's map over independent parts (a trait in `nd-fmm-tables`, a
  serial implementation there, a pool implementation in `nd-fmm-exec` on the FMM's pool),
  each part computed by the same function from the same inputs whichever thread runs it, the
  results kept in part order. Parts: per level the M2M/L2L pair and each of the 15 M2L
  coaxial tables; per level and angle the M2L y-blocks; per batched/dense level the 316
  matrices by offset; the exponential M2L's levels. `fmm-tables/CLAUDE.md`'s rule "Builds are
  serial and deterministic" amended to "deterministic; the parts of a build may run on a
  caller's threads, the same bits" (a crate rule, not a convention).
- Whatever the timings show next (the plan, the routes) only if it is Helmholtz-only code; a
  Laplace path (`nd-fmm-plan`, `pipeline.rs`, `nd-octree`) is out of scope here.

Public API: `HelmholtzBuildTimings`, `HelmholtzFmm::build_timings`,
`HelmholtzOperator::table_timings`, the operator's new constructor, the tables' map trait and
the level-restricted build. No convention changes; no dependency changes; `nd-fmm-plan`
untouched.

## Tests that prove it

- `nd-fmm-tables` (`tests/helmholtz/`): the tables built with a parallel map (a test map that
  runs the parts in reverse order and on threads of `std::thread::scope`) bit for bit the
  serial build, every table and block of every level (compare by the applications: M2M, L2L,
  M2L of every offset of every level on random inputs, bit for bit), at q = p and raised q;
  the level-restricted build's M2L on the chosen levels bit for bit the full build's; Dense,
  Batched and Classes from the restricted build bit for bit the full one's matrices; the
  blocks for a level mask: the masked levels store none, the others' applications unchanged.
- `nd-fmm-exec` `tests/helmholtz/`: the operator's new constructor against the old chain of
  setters: the same tables per level (degrees, `m2l_on`, bytes of the batched and plane-wave
  tables, blocks stored only where Rotation runs).
- `nd-fmm-exec` `tests/helmholtz_fmm.rs`: as built, no `check_build` scenario and no
  `serial_tables` switch: the builder makes no pool at one thread, so every check's 2-thread
  run (tables built on the pool) against its 1-thread run (tables built serially) holds the
  parallel tables to the serial ones bit for bit, under every setting the checks cover (Auto,
  raised at 10⁻⁸, PlaneWave, Dense, Batched, Classes, the defaults); `check_build_timings` (in
  `check_defaults`) builds once more under a wall clock: the parts timed, the tables within
  the operator, the steps within the wall time, M2L blocks only on RCR levels, every rank.
- `nd-fmm-validate` `tests/helmholtz_accuracy.rs`: the "Build" table's columns on W1 at kD = 30.
- Machine runs (below): checksums equal to CH4.0e's on every case, split and machine.

## Checks

`/verify` (fmt, Clippy default/examples/workspace, docs, default and workspace tests at
opt-level 1, the MPI list at 2 and 4 ranks); `cargo test -p nd-fmm-tables` and its Clippy;
`cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`; `cargo deny` not needed (no
manifest change).

## Machine runs

- Kathleen (one node, `singlenode`): W1 at kD = 500 at 10⁻³, 10⁻⁶, 10⁻⁸, the build by part at
  1 × 1 and at CH4.0e's four splits, `main` (CH4.0e's numbers and a profile of the parts, run
  with this branch's timings and the fixes switched off where a switch exists) and this
  branch, ABBA; then CH3.6's matrix at the best splits once more for the build column and the
  checksums. Job ids in "Pending runs".
- locust: the same at 1 × 1 and 1 × 72, 2 × 36, 4 × 18, 8 × 9 (check `uptime` and
  `nvidia-smi` first).
- Decides: the acceptance (node ≤ ½ one core) and the "bit for bit" (checksums).

## Constraints and risks

- Constraint 1 (Laplace unchanged): no Laplace path changes. `nd-fmm-tables`' Helmholtz
  modules only (Laplace's `rotation`, `plane_wave`, `cache` untouched); `nd-fmm-exec`'s
  `helmholtz` module only (`driver.rs`, `pipeline.rs`, `fmm.rs` untouched). `nd-fmm-plan`,
  `nd-octree`, `nd-fmm-simd` untouched. Laplace's tests and the MPI list run.
- Constraint 2 (determinism) and 9 (rank-independent choices): every part is a function of
  k, the degrees, the budget and the level masks, all agreed before the tables; no thread
  count reaches a value.
- Memory: parallel parts hold their results at once (the same total as serial); the
  scratch per part (a `BlockScratch` at the level's degree) times the threads: at p = 131
  small against the blocks.
- Risk: rayon only in `nd-fmm-exec` (`fmm-exec/CLAUDE.md`); the tables' map trait keeps it so.
- Risk: the exponential M2L's raise (CH6.1g) rebuilds tables when a later setting removes the
  waves; the new constructor must give the same final tables as the setter chain under every
  setting the builder passes.

## Results

- The table hashes (`tests/helmholtz/parts.rs`, `print_the_hashes`, release, the M6) on `main`'s
  tables (before any change) and the branch's, equal: RCR k̂₀ = 120 + 6i d99e3221b99a61e3,
  120 71223dfc2b0eba4f, 6 + 0.5i 1520f647ee9b88bd; matrices 6 + 0.5i e8f14c144df269a6, 120
  9218603869fc1abd; plane waves 40 at 10⁻⁶ bc43acd5206e0682 and 10⁻⁸ 2b62dea0a7d6d982, 20 + 2i
  38d11660252af3c9 and b9419e1bdc4071ca. The self-review repeated them on a copy of `main`.
- The machine runs: `fmm-validate/results/helmholtz-ch4-0h.md` (Kathleen job 634392; locust
  2026-11-16). W1 at kD = 500 at the best split 0.15–0.23 of one core, at one rank per node
  0.44–0.66; every checksum CH4.0e's.

## Checks (run)

`/verify` on the M6: fmt, Clippy (default, examples, workspace), docs; the default and
workspace tests at opt-level 1 (1,270 and 1,384 passed); the MPI list at 2 and 4 ranks (24 of
24; `helmholtz_fmm` 26.7 and 27.1 s); the one-rank opt-level 0 runs: `helmholtz_fmm` 510 s
(load 2–26; CH4.0d3 666 s at 15–19), `nd-fmm-validate`'s `helmholtz_accuracy` 252 s (load
2–4; CH4.0e 246 s at 2–3), `nd-fmm-exec`'s `tests/helmholtz/` 44 s (load 8; `build.rs` about
9 s of it), `nd-fmm-tables`' `tests/helmholtz/` 35.9 s (`main` 35.5 s). After the
self-review's fixes the four touched test executables again at opt-level 1 and Clippy.

## Pending runs

None.

## Open questions

The PR's (report §8): the acceptance as read (at the node's best split), the follow-ups'
order (CH4.0j, CH4.0k), and `nd-fmm-tables`' amended rule.
