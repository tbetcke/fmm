# Phase 2 / T1 — nd-fmm-tables scaffold

Read first: root CLAUDE.md, fmm-tables/CLAUDE.md (already committed with this brief),
root Cargo.toml, fmm-ref/Cargo.toml and fmm-ref/src/lib.rs (the template to follow),
docs/phase2/README.md ("Design decisions for this phase"),
docs/design/workspace-structure.md §3 and §5.

Starting point: with Phase 1 T7 merged, `members` lists "fmm-plan", "octree",
"fmm-math", "fmm-ref", "fmm-validate" and "spikes/cubecl-gemm", and `default-members`
lists the same without the spike. The directory fmm-tables/ contains only its CLAUDE.md.
If fmm-validate is not a member, T7 is not merged: stop and report.

Do:
- Create the crate in fmm-tables/: package nd-fmm-tables, lib nd_fmm_tables, version
  0.1.0-dev, with edition, licence and repository inherited from [workspace.package].
  Add a one-line README and keep fmm-tables/CLAUDE.md.
- Dependencies: nd-fmm-math, nd-fmm-ref and num-traits (`workspace = true`);
  dev-dependencies: proptest. Add `nd-fmm-tables = { path = "fmm-tables" }` to
  [workspace.dependencies]. `thiserror` comes with T7.
- Add fmm-tables to `members` and `default-members`.
- Crate-level doc comment:
  - the purpose;
  - the conventions the crate relies on (§3.6, §3.7, §3.11; T3 adds §3.12 once it is
    signed off);
  - the rules of docs/phase2/README.md: built from the oracle in f64, canonical frames,
    column-major layout, every application accumulates, no 1/(4π) (§3.1);
  - the error measures.
- `pub struct MatrixSet<T: RealScalar>`: `count` square matrices of order `n`, stored
  column-major and contiguously, matrix after matrix. Provide:
  - `MatrixSet::zeros(n, count)`;
  - `n()`, `count()`, `as_slice()`;
  - `matrix(i)` and `matrix_mut(i)`, each (n² reals); `column_mut(i, k)` (n reals);
  - `apply(i, x, y)`, which adds Aᵢ x to y. It loops over the columns k in increasing
    order and adds xₖ times column k into y, allocates nothing, and panics on a slice
    of the wrong length or an index out of range. Document the summation order.
  - `cast::<U: RealScalar>()`, which rounds every entry through `to_f64` and
    `from_f64` (round to nearest).
- Tests:
  - `apply` equals a naive double loop in the same order, bit for bit, on random
    matrices, n ∈ {1, 4, 81};
  - it accumulates onto a nonzero y;
  - entry (r, c) of matrix i sits at i n² + r + c n;
  - `cast` from f64 to f32 equals `as f32` entry by entry;
  - `apply` panics on a wrong length;
  - proptest, 64 cases.
- Declare no table modules yet; T3 to T7 add them.
- In root CLAUDE.md, change "Current phase and task briefs: docs/phase1/README.md" to
  docs/phase2/README.md. Change nothing else there.

Must pass, and report which ran:
- The CI commands of root CLAUDE.md, and the stricter
  `cargo clippy --workspace --all-targets -- -D warnings` and
  `RUST_MIN_STACK=8388608 cargo test --workspace` (needs MPI).
- `cargo test -p nd-fmm-tables` builds and runs without touching MPI.
- `cargo doc -p nd-fmm-tables --no-deps` without warnings.

Do not: touch octree/, fmm-plan/, fmm-math/, fmm-ref/ or fmm-validate/; add any table
or geometry; add a dependency not listed above.
