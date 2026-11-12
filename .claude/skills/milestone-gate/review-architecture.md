# Brief: the review of the architecture

The brief of the second review pass of `/milestone-gate` (its step 3), given to one subagent
as written, after the header the gate fills in (the milestone, `main`'s commit, the earlier
review files).

You review the Rust workspace in the working directory (nd-project/fmm: fast multipole
methods; the root CLAUDE.md is the map). Do the work yourself; start no subagents.

## Limits

- Read only. Change, create or delete no file. No git command that changes state;
  `git log`, `git show` and `git diff --stat` are fine.
- Compile and run nothing: no `cargo` of any kind (`metadata` and `tree` included; read the
  manifests and `Cargo.lock`), no `mpirun`, no script of `tools/`, no `ssh`.

## Scope

The architecture of the code. A second reviewer covers the mathematics and the benchmarks;
do not judge formulas.

## Read

The root CLAUDE.md, `docs/design/workspace-structure.md`, the root `Cargo.toml` and every
crate's manifest and CLAUDE.md; then breadth first (each `lib.rs`, the module trees, line
counts), then depth in the two or three largest or most central crates and in what changed
since the last review (`git log --stat <its commit>..HEAD`). Start the depth with what the
earlier reviews list as not reached (their "Coverage"). Read the code for what you assert;
do not infer structure from documents.

## Check

1. **The crate split.** The dependency graph from the manifests against
   `workspace-structure.md`; the layering; crates too thin or doing several jobs (lines per
   crate and for the largest files). The rules of the root CLAUDE.md: nd-fmm-math, -ref and
   -tables free of MPI and the octree; `unsafe` only where it says; CubeCL only in
   nd-fmm-kernels and `spikes/`.
2. **The kernel families.** How each kernel (Laplace, Helmholtz, Yukawa) sits on the shared
   code: a clean abstraction, or parallel near-copies? Name duplication with both paths and
   rough line counts, what a shared form would look like, and whether it is worth it.
3. **The public API.** The main traits and builders: coherent, minimal, at the right level?
   Generic parameters with one instantiation; options left by experiments (settings for
   candidates the record calls not adopted or superseded); feature flags.
4. **Bloat.** Dead code and `allow`s; superseded paths still compiled; very large files and
   functions; helpers copied across crates; overlapping examples and tests; stale spikes,
   tools and documents; TODO markers; names tied to task IDs. Quantify, and rank by payoff
   against risk.
5. **The tests' structure.** Every fast path against a slower trusted path; tests in the
   right crates; copied scaffolding; oversized files. Run none.
6. **Hot paths, as structure.** Where the structure forces allocation or dynamic dispatch
   in inner loops, or generic bloat; whether the choice of host, SIMD, device and MPI paths
   is in one place.
7. **The build.** Members against default members, the feature matrix, `[patch]`, pins:
   whatever is fragile.
8. **The earlier findings.** For each finding of the earlier reviews in this pass's scope:
   resolved (where), open (which task has it), or no longer applies.

## Calibrate

A dozen findings with their evidence beat fifty guesses. Before you report a duplication or
dead code, read both sites or search for the uses; what you could not verify is a
*suspicion*, and labelled so. Say briefly what is well designed and should be left alone.

## Report

Your last message is the report, for the session that drives the gate. Markdown: a table of
the crates (path, lines of src, tests and examples, workspace dependencies, role) and the
graph; the findings by group (crate structure, kernel families, API, bloat, tests, build),
each with a severity (high, medium, low), files with line numbers, the evidence, and a
concrete change with its effort and risk; the ten changes of best payoff; the earlier
findings; what you read and what you did not reach.
