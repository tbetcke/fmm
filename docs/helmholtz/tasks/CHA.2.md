# CHA.2 `unsafe` enforced

Started 2026-11-19 on `tbetcke/h-cha-2`. Roadmap: "CHA.2 `unsafe` enforced. Review C1. *Done
when:* `#![forbid(unsafe_code)]` in nd-fmm-math, -ref, -tables, -exec, -validate and -bench,
`#![deny(unsafe_code)]` in nd-fmm-plan and nd-octree; `octree/src/octree.rs:1411` made safe, or
kept with its `// SAFETY:` comment if the safe form is measurably slower; the counting allocator
of `fmm-plan/examples/plan_build_cost.rs` allowed in that file with its reason; the root
CLAUDE.md's rule names the exceptions." Proposal: none (section HA: the roadmap's *Done when*
with constraint 1 of §2).

## Goal
Make the root CLAUDE.md's `unsafe` rule a compiler check in every crate it covers, with its two
unrecorded exceptions (review 2026-11-12-H6, C1) resolved or recorded.

## Acceptance (roadmap *Done when*, verbatim)
`#![forbid(unsafe_code)]` in nd-fmm-math, -ref, -tables, -exec, -validate and -bench,
`#![deny(unsafe_code)]` in nd-fmm-plan and nd-octree; `octree/src/octree.rs:1411` made safe, or
kept with its `// SAFETY:` comment if the safe form is measurably slower; the counting allocator
of `fmm-plan/examples/plan_build_cost.rs` allowed in that file with its reason; the root
CLAUDE.md's rule names the exceptions. With constraint 1 (the HA preamble): Laplace's outputs
bit for bit, its benchmarks within D9's spread wherever a host or device path's code moves.

## Read first
- `docs/reviews/2026-11-12-H6.md` C1 (checked against the tree on 2026-11-19: the two sites are
  still the only `unsafe` in the eight crates, `octree/src/octree.rs:1411` and
  `fmm-plan/examples/plan_build_cost.rs:29-42`; no crate has an `unsafe_code` lint).
- Root CLAUDE.md, "Working rules" (the `unsafe` rule); `docs/design/simd-p2p.md` §5.4.
- `octree/src/octree.rs` `points_to_morton`; rlst 0.12.0 `traits/accessors.rs` (`get_value`).

## Changes
- **Where the lint goes.** A crate-root `#![forbid(unsafe_code)]` binds the library target only;
  tests, examples, benches and binaries are crates of their own. The *Done when*'s allowance for
  an example (`plan_build_cost.rs`) presumes the lint reaches examples, so the proposal is the
  package-level form: `[lints.rust] unsafe_code = "forbid"` in the `Cargo.toml` of nd-fmm-math,
  -ref, -tables, -exec, -validate and -bench, `"deny"` in nd-fmm-plan's and nd-octree's (octree
  already has a `[lints.clippy]` table). Every target of the package is checked. (Question 1.)
- `octree/src/octree.rs` `points_to_morton`: the three `get_value_unchecked` reads become
  `[0, 1, 2].map(|i| point.get_value([i]).unwrap())`, after the function's existing
  `assert_eq!(points.shape()[0], 3)`. The same values are read, so the keys are the same bit for
  bit; the cost is three bounds checks per point beside the Morton encoding. Kept unsafe, with
  `#[allow(unsafe_code)]` and a `// SAFETY:` comment, only if the timing below shows it
  measurably slower. (It did; kept, "Results".)
- `fmm-plan/examples/plan_build_cost.rs`: `#[allow(unsafe_code)]` on the `GlobalAlloc` impl,
  with the reason in a comment next to its existing `// SAFETY:` comment.
- Root CLAUDE.md's rule: the lint and its levels, and the exceptions named (the example's
  allocator; octree's block only if kept). `fmm-plan/CLAUDE.md` and `octree/CLAUDE.md` where
  they describe the crate's rules; `docs/design/workspace-structure.md` if it states the rule.
- No public API, behaviour, default, dependency or pin changes. Conventions: none.

## Tests that prove it
- The lint itself: `cargo clippy --workspace --all-targets -- -D warnings` (and the default
  checks) pass, so no target of the eight crates holds an unallowed `unsafe`. A deliberate
  `unsafe {}` added by hand to a test of a forbid crate and of nd-octree fails to compile
  (checked once, not committed; recorded in the PR).
- `points_to_morton`'s keys: the existing octree tests and every Laplace test that builds a tree
  (fixtures, `mpi_exec`, `mpi_threading`, `invariance` at 2 and 4 ranks); the reads are the same
  `f64` values, so every key and output is `main`'s bit for bit.

## Checks
`/verify`: fmt, Clippy (default, examples, workspace), docs, the default and workspace tests at
opt-level 1, the MPI list at 2 and 4 ranks, `cargo check -p nd-fmm-exec --features cuda`,
`cargo clippy -p nd-fmm-kernels --all-targets --features cpu,metal` (untouched, unaffected).

## Machine runs
- `points_to_morton`, the safe form against the unchecked one: a throwaway, uncommitted release
  example timing both at N = 10⁶ and 10⁷ uniform points, alternated, best and median of ten,
  BLAS threads at 1; it decides "measurably slower". Machine: question 2.
- D9's Laplace benchmarks (`tools/bench/run.sh`) time the evaluation only; `points_to_morton`
  runs in the build (`fmm-exec/src/pipeline.rs`), and the lint attributes generate no code. So
  no D9 run is planned; the timing above stands in for it. (Question 3.)

## Results (2026-11-19)
- **The timing (M6, load 1.6–3.2, one thread, release, no LTO).** A throwaway example (not
  committed) timed `points_to_morton` from the library against an unchecked copy in the
  example, at N = 10⁶ and 10⁷ uniform points, 20 alternated calls each, three runs per build.
  `points_to_morton` is generic, so it is compiled into the calling crate: one build that also
  timed a local safe copy moved the library's own call from 26.6 to 59.5 ms at 10⁷ (codegen,
  not the reads), so the forms were compared as three builds of the library itself, each against
  the same unchecked copy (library / copy, medians, and bests in brackets):

  | Library form | N = 10⁶ | N = 10⁷ |
  | --- | --- | --- |
  | unchecked (`main`) | 1.002–1.010 (0.971–0.993) | 0.998–1.003 (1.000–1.003) |
  | `get_value([i]).unwrap()` | 1.064–1.072 (1.022–1.044) | 1.068–1.072 (1.040–1.045) |
  | the column's `iter_value()` zipped into `[f64; 3]` | 1.020–1.070 (1.047–1.071) | 1.013–1.030 (1.081–1.088) |

  At 10⁷ points the unchecked form takes 24.3–26.0 ms; the safe forms 1–2 ms more, 1–9 % of the
  function. That is measurably slower by the *Done when*, so the block is kept: `#[allow(unsafe_code)]`
  on the statement, its `// SAFETY:` comment (the asserted `[3, n]` shape) and its reason. A safe
  form without the checks (contiguous `chunks_exact(3)`) would need a raw-data bound on the
  public signature; not in scope. Raw outputs: `~/projects/fmm-archive/m6/logs/cha-2/`.
- **The lint's reach**, checked by hand and not committed: `unsafe {}` in a test of nd-fmm-math
  (`tests/convention_version.rs`) fails (`usage of an unsafe block`), an
  `#[allow(unsafe_code)]` there is E0453 (incompatible with the `forbid`), and `unsafe {}` in
  nd-octree's library tests (`src/tools.rs`) fails under `deny`.

## Constraints and risks
- Constraint 1: nd-octree and nd-fmm-plan are Laplace's paths. Neither crate's code changes:
  the lint levels, the two allowances and comments only, so the generated code is `main`'s.
- A future dependency's derive macro that expands to `unsafe` would trip `forbid` (rsmpi's
  `Equivalence` derive is not used in these crates today); `forbid` cannot be allowed locally,
  so such a crate would move to `deny`. Recorded, not expected.

## Pending runs
None.

## Open questions
Answered 2026-11-19 at the plan (the user, all three as recommended):
1. "[lints.rust] per crate (Recommended)": the lint in each `Cargo.toml`, every target.
2. "M6, now (Recommended)": the `points_to_morton` timing on the M6.
3. "Skip D9 runs (Recommended)": no D9 benchmark run; Laplace bit for bit through the tests
   and the MPI list, the build-side timing in their place.
