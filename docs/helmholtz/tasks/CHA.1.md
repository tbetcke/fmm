# CHA.1 The spikes out of the workspace

Started 2026-11-19 on `tbetcke/h-cha-1`. Roadmap: "CHA.1 The spikes out of the workspace"
(HA; priority list item 7). Review: `docs/reviews/2026-11-12-H6.md` K3, U1. No proposal §10
row (an HA task): the acceptance is the roadmap's *Done when* with constraint 1 of §2.

## Goal

Take the twelve spikes out of the root workspace, so that the strict checks, `Cargo.lock`,
the root `[patch]` and `[workspace.dependencies]` hold the product crates alone, while every
spike that a test, a fixture tool, a report or C3S.4 (D9) names still builds by hand.

## Acceptance (roadmap *Done when*, verbatim)

The finished spikes are outside `members` (excluded, each with what it needs to build by
hand, or archived to `../fmm-archive` with their reports kept in the tree); `spikes/p2p-simd`
still builds and runs as C3S.4's reference (D9), and every spike a test or a report's
reproduction names still builds by hand, each said in its README; the `[patch]`, the second
RLST and the spike-only entries of `[workspace.dependencies]` gone from the root if no member
needs them; `deny.toml` and the root CLAUDE.md's "Layout" (no crate-level lock files) brought
in line. *Sign-off:* which spikes are archived, and the lock files of the kept ones.

With constraint 1 (the HA section's preamble): Laplace's outputs bit for bit, its benchmarks
within D9's spread wherever a host or device path's code moves (none moves here), `/verify`'s
Laplace protection run. No behaviour or default changes.

## Read first

- Root `Cargo.toml` (members, `[workspace.dependencies]`, `[patch]`), `deny.toml`,
  `Cargo.lock` (578 packages; rlst 0.10.2 and 0.12.0; green-kernels from git).
- Each spike's `Cargo.toml`. Workspace libraries used: census (nd-octree, nd-fmm-plan, mpi,
  nd-fmm-validate), cubecl-gemm (cubecl, cubek-*), device-arith (cubecl, math, kernels, ref,
  tables, simd, validate, rayon), download-path (cubecl, kernels, rayon, validate), p2p-simd
  (math, ref, simd, validate, green-kernels, pulp), plane-wave (math, ref, tables, simd),
  pw-quadrature (tables, ref, rayon; dev: plane-wave), rotation-blocks (math), diagonal
  (truncation); coaxial, radial and truncation none.
- What names them outside `spikes/`: C3S.4's reference `spikes/p2p-simd/results-m6.md`
  (D9, `tools/m6/machine.md`); the fixture tools (`tools/fixtures/`: `helmholtz_pw_quadrature.sh`
  takes `spikes/pw-quadrature/tools/fetch.sh`, the S1–S3 generators cite the spikes'
  `tools/reference.py` and `fixtures/reference.json`); `fmm-ref/CLAUDE.md` (the bit-for-bit
  check against `spikes/pw-quadrature/examples/generated.rs`, by hand); `fmm-validate/CLAUDE.md`
  (`spikes/census/tests/ported_workloads.rs` checks `helmholtz_workloads`); the S6 prototype
  `spikes/diagonal`, rerun at CH4.0i; rustdoc and comments citing reports and sources of
  census, coaxial, cubecl-gemm, device-arith, plane-wave, pw-quadrature, radial,
  rotation-blocks, truncation.

## Changes

Proposed (for sign-off, questions below):

- A second, separate workspace `spikes/Cargo.toml` with every kept spike as a member, its own
  `[workspace.dependencies]` (path entries for the workspace libraries, the spikes' CubeCL pins
  repeated from the root, green-kernels and pulp), the green-kernels `[patch]`, and one lock
  file `spikes/Cargo.lock`, seeded from the root's so the shared crates keep their versions.
  Its target directory `target/spikes` at the root (`spikes/.cargo/config.toml`), so it never
  clobbers the root's builds. Every spike's own `Cargo.toml` unchanged apart from what the move
  forces.
- Root `Cargo.toml`: the spikes out of `members`, `exclude = ["spikes"]`; green-kernels, pulp
  and the `[patch]` removed; `Cargo.lock` regenerated (rlst 0.10.2, green-kernels, pulp and the
  spikes leave it; no product crate's version changes, checked by diff).
- `deny.toml`: the five spike exclusions removed (nd-fmm-kernels stays).
- Root `CLAUDE.md`: "Layout" (`Cargo.lock` at the root for the product, `spikes/Cargo.lock` for
  the spikes workspace), "spikes/ is excluded …", the CubeCL and `unsafe` rules (spikes/ kept),
  the rlst note under "Navigating the code"; `spikes/README.md` (new): how to build and run each
  spike, which test or report names it; `fmm-validate/CLAUDE.md`, `docs/design/simd-p2p.md`
  §8.4's member sentence, `docs/design/workspace-structure.md` where they say "member".
- No Rust source of any product crate changes.

## Tests that prove it

- Root: `cargo metadata` lists no spike; `cargo tree -i rlst@0.10.2` and `-i green-kernels`
  fail (absent); the diff of `Cargo.lock` removes packages only.
- Spikes workspace: `cargo check --all-targets` and `cargo test --no-run` for every kept spike
  (`cd spikes`); `cargo clippy --all-targets` reported, not gated (spikes are not held to it).
- C3S.4: `spikes/p2p-simd`'s `compare` example builds and runs (`cargo run --release --example
  compare`), its output against `results-m6.md` for the checksums/accuracy columns, not timed
  (no quiet-machine run asked for).
- `spikes/census`'s `ported_workloads` test passes (the check of `helmholtz_workloads`);
  `spikes/pw-quadrature`'s `generated` example builds.
- Laplace: no product source changes; `/verify`'s full set, and every product crate's tests
  pass with the regenerated lock (same versions, so same bits).

## Checks

`/verify` (fmt, Clippy default/examples/workspace, docs, default and workspace tests, the MPI
list at 2 and 4 ranks, the licence check `cargo deny --locked check licenses`); the spikes
workspace's checks above.

## Machine runs

None. (`spikes/p2p-simd` on x86 builds the same code; Kathleen's earlier failures of its tests
are recorded in `tools/kathleen/README.md` and are not this task's.)

## Constraints and risks

- Constraint 1: no product code moves; the lock's product versions must not change (diffed).
- A dependency and pin move (the CubeCL pins repeated in the spikes workspace, green-kernels,
  pulp and the `[patch]` moved there): asked for sign-off.
- Risk: the spikes stop being compiled by the strict checks, so a later change to a workspace
  library may break a spike unseen. `spikes/README.md` says to rebuild the spikes workspace
  when a task reruns one; the census check of `helmholtz_workloads` becomes by hand.

## Pending runs

None.

## Open questions

None open. Signed off 2026-11-19 (the user, at the plan, before the branch's first commit):

1. Which spikes are archived: "Keep all, none archived (Recommended)".
2. The kept spikes' lock files: "One spikes workspace (Recommended)": `spikes/Cargo.toml`
   with one `spikes/Cargo.lock`, its own `[workspace.dependencies]`, `[patch]` and pins.
