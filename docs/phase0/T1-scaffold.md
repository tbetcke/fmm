# Phase 0 / T1 — workspace scaffold

Read first: root CLAUDE.md, the existing CLAUDE.md files in octree/ and fmm-plan/,
root Cargo.toml, fmm-math/CLAUDE.md (already committed with this brief),
docs/design/workspace-structure.md §1 and §5.2.

Starting point: the root Cargo.toml has only `members = ["fmm-plan", "octree"]` and
`resolver = "2"`. There is no [workspace.package], [workspace.dependencies] or
default-members yet. Both existing crates declare their package fields directly.

Do:
- Check that root CLAUDE.md does not contradict the existing crate CLAUDE.md files;
  if it does, report the conflict instead of editing either. Known discrepancies are
  listed in docs/design/workspace-structure.md §6; confirm them and add any others.
- Add [workspace.package] with edition = "2024", license = "MIT OR Apache-2.0",
  repository = "https://codeberg.org/nd-project/fmm". Only fmm-math inherits it.
- Add [workspace.dependencies]: num-traits 0.2, thiserror 2, proptest 1, approx 0.5,
  serde_json 1, and nd-fmm-math = { path = "fmm-math" }.
- Create the crate in fmm-math/ (package nd-fmm-math, lib nd_fmm_math) with an empty
  lib.rs and a one-line README; keep the existing fmm-math/CLAUDE.md.
- Add fmm-math to members, and add default-members = ["octree", "fmm-plan", "fmm-math"]
  so that T6 can add the spike to members only. Keep resolver = "2".

Must pass, run exactly as CI does:
- cargo fmt -- --check
- cargo clippy --workspace --all-targets -- -D warnings (stricter than CI's
  `cargo clippy -- -D warnings` plus `--examples`)
- RUST_MIN_STACK=8388608 cargo test --workspace (needs an MPI runtime, because octree
  and fmm-plan tests initialise MPI)
- cargo doc --no-deps
- The Forgejo run-tests workflow stays green.
Also: cargo test -p nd-fmm-math builds and runs without touching MPI.

Do not touch: any file under octree/ or fmm-plan/. Do not migrate their manifests to
workspace inheritance; that is a separate decision.
