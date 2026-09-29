# Phase 1 / T1 — nd-fmm-ref scaffold

Read first: root CLAUDE.md, fmm-ref/CLAUDE.md (already committed with this brief), root
Cargo.toml, fmm-math/Cargo.toml and fmm-math/src/lib.rs (the template to follow),
docs/phase1/README.md ("Design decisions for this phase"),
docs/design/workspace-structure.md §3 and §5.

Starting point: the workspace has `members = ["fmm-plan", "octree", "fmm-math",
"spikes/cubecl-gemm"]` and `default-members = ["octree", "fmm-plan", "fmm-math"]`.
The directory fmm-ref/ contains only its CLAUDE.md.

Do:
- Create the crate in fmm-ref/ (package nd-fmm-ref, lib nd_fmm_ref, version
  0.1.0-dev, edition, licence and repository inherited from [workspace.package]), with
  a one-line README. Keep fmm-ref/CLAUDE.md.
- Dependencies: nd-fmm-math and num-traits (`workspace = true`); dev-dependencies:
  proptest. Add `nd-fmm-ref = { path = "fmm-ref" }` to [workspace.dependencies].
- Add fmm-ref to `members` and `default-members`.
- Crate-level doc comment: purpose, the conventions the crate relies on (§3.1, §3.6,
  §3.7), the rules "scaled coefficients only", "every operator accumulates" and "no
  1/(4π)", and the error measures of docs/phase1/README.md.
- `pub struct Frame<T: RealScalar> { pub centre: [T; 3], pub radius: T }`: the centre
  and scaling radius of an expansion (CONVENTIONS §3.7). Add `Frame::new`, which
  asserts `radius > 0`, and `Frame::scaled(x)`, which returns (x − centre) / radius.
  Test both.
- Declare no operator modules yet; T3 to T6 add them.
- In root CLAUDE.md, change "Current phase and task briefs: docs/phase0/README.md"
  to docs/phase1/README.md. Change nothing else there.

Must pass, and report which ran:
- The CI commands of root CLAUDE.md, and the stricter
  `cargo clippy --workspace --all-targets -- -D warnings` and
  `RUST_MIN_STACK=8388608 cargo test --workspace` (needs MPI).
- `cargo test -p nd-fmm-ref` builds and runs without touching MPI.
- `cargo doc -p nd-fmm-ref --no-deps` without warnings.

Do not: touch octree/, fmm-plan/ or fmm-math/; add any operator; add rayon or any other
dependency not listed above.
