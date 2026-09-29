# nd-project/fmm

Fast multipole methods in Rust. Workspace crates: octree (distributed octree),
fmm-plan (nd-fmm-plan), and the nd-fmm-* transfer-operator crates.

## Source of truth
- docs/CONVENTIONS.md defines every basis function, phase, storage layout and scaling.
  Never change a convention in code; propose changes in the PR description instead.
- Cite conventions in doc comments as `CONVENTIONS §3.x`.
- CONVENTION_VERSION in nd-fmm-math must match the file.
- Current phase and task briefs: docs/phase0/README.md.
- Background: docs/design/laplace-fmm-plan.md (operators, plan) and
  docs/design/workspace-structure.md (crate layout). CONVENTIONS.md wins on any conflict.

## Working rules
- Run `cargo fmt` after every Rust edit.
- Before finishing any task:
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace`.
- One task per branch and PR; stay inside the crates the task names.
- Tests first: write the acceptance tests from the task brief, then the implementation.
- Every fast path is tested against a slower trusted path (fixtures, identities, nd-fmm-ref).
- Numeric code is generic over `T: RealScalar`; no allocation in hot loops.
- New dependencies only through [workspace.dependencies]; CubeCL only in nd-fmm-kernels
  and spikes/.

## Layout
- Directories fmm-<name>, packages nd-fmm-<name>, library names nd_fmm_<name>.
- Fixtures live in <crate>/fixtures/ and are regenerated only by tools/fixtures/.
- spikes/ is excluded from default members and never built in CI.

## Commands
- Test one crate: cargo test -p nd-fmm-<name>
- Regenerate fixtures: see tools/fixtures/README.md
