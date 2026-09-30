# Phase 2: operator tables

Phase 2 builds `nd-fmm-tables`, the precomputed, level-independent operator data that
the FMM applies on every level:

- 8 M2M and 8 L2L matrices, one per child octant;
- 316 M2L matrices, one per V-list offset, and their reduction to 16 symmetry classes;
- the rotation and coaxial tables of point-and-shoot translation for the uniform V list
  and the octants;
- a versioned on-disk cache.

Every table is built in f64 from the `nd-fmm-ref` oracle and checked against it on
several levels. First, the box geometry that the tables encode goes into
`docs/CONVENTIONS.md`: the child octant index, the sign and order of the V-list offsets,
the shift 2r · d, the matrix layout and the action of the cube symmetry group. That
closes the open question in workspace-structure §6. The phase ends when every table
reproduces `nd-fmm-ref` on all levels and a cache round trip is bit-identical.

Companion documents: [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md)
(Sections 2.4, 2.5, 3.2, 3.6, 4 and 5.3; components C2.1–C2.4 in Section 7) and
[docs/design/workspace-structure.md](../design/workspace-structure.md) (Section 3,
`nd-fmm-tables`; Section 4; Section 5, crate template; Section 6, "Still open").

Prerequisite: Phase 1 is complete. T7 (`nd-fmm-validate` and the design-document update
with the Phase 1 outcome; reviewed in PR #13) reached `main` through PR #14. The briefs
assume the workspace as it is after that merge.

## Scope

In scope:
- `docs/CONVENTIONS.md`: a new section §3.12 on box geometry and operator tables, and
  an extension of the §3.10 versioning rule to cover it.
- `tools/fixtures/`: a numerical check of the §3.12 symmetry rules.
- `nd-fmm-tables`: a dense matrix set, the octant and offset geometry, M2M and L2L
  tables (C2.1), dense M2L tables and their symmetry-reduced form (C2.2), rotation
  and coaxial tables with table-driven rotation operators (C2.3), and the versioned
  cache (C2.4). It is generic over `T: RealScalar`, always builds in f64, and stores
  f64 or f32.
- `nd-fmm-validate`: a report of build times, memory and time per application, and the
  accuracy sweep run through the tables.

Out of scope:
- SVD or other low-rank compression of M2L (C6.2), and with it `rlst`.
- Batched application (GEMM over a level), device upload and CubeCL (Phase 4);
  parallel table building (rayon).
- Morton keys, `nd-octree`, `nd-fmm-plan` and MPI. Box geometry from keys, and the
  cross-check of the tables' octant and offset order against `morton::child_index`
  and `V_LIST_DIRECTIONS`, belong to C3.1.
- Tables for the leaf operators (P2M, L2P, P2L, M2P). The W and X lists apply them
  per particle, so they need no tables.
- Plane-wave tables (C6.5).

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **Built from the oracle.** Column k of a translation matrix is the `nd-fmm-ref`
  operator applied to the k-th unit vector, at the canonical frames below, into a
  zeroed output. No new formula enters the code, so the tables inherit §3.11 and the
  Phase 1 tests. Tables are always built in f64. An f32 table is the f64 table rounded
  entry by entry, never a table built in f32.
- **Canonical frames.** By §3.7 and §3.11 a table depends only on its octant or offset.
  Each table is built at these frames, with s_o = (2x − 1, 2y − 1, 2z − 1) for child
  index o = 4x + 2y + z:
  - M2L: source frame (0, 1) and target frame (2d, 1);
  - M2M: child (½ s_o, ½) to parent (0, 1);
  - L2L: parent (0, 1) to child (½ s_o, ½).

  The same matrix then serves every level and every cubic domain.
- **Matrix layout.**
  - Each operator is a real (p + 1)² × (p + 1)² matrix A with output = A · input. Rows
    index output slots and columns input slots, both in the real storage of §3.6.
  - Storage is column-major: entry (i, k) sits at i + k (p + 1)². A table is then a
    GEMM operand for `nd-fmm-plan`'s column-major `LevelData`.
  - The matrices of one family are contiguous: M2M and L2L in child-index order, M2L
    in the lexicographic (x, y, z) order of `V_LIST_DIRECTIONS`.
  - Applying a table accumulates (+=), as every operator does.
- **One degree per table.** p_in = p_out = p. The entries do not depend on p, so a
  table at p is the leading block of one at a larger p. The tests check this; no code
  relies on it.
- **MPI-free.** `nd-fmm-tables` depends on `nd-fmm-math` and `nd-fmm-ref`, plus
  `thiserror` for the cache. It restates the octant and offset order of §3.12 instead
  of importing them from `nd-octree` or `nd-fmm-plan`, which both need MPI. C3.1 tests
  that they agree.
- **No new numeric dependencies.** Building a table needs only operator application,
  and applying one needs only a matrix–vector product. `rlst` (for its LAPACK SVD,
  without its `mpi` feature) waits for SVD compression (C6.2). The cache uses a small
  hand-written little-endian format with a checksum, not serde. This refines
  workspace-structure §3, which lists `rlst` and "a binary serialiser" from the start;
  the design documents are updated at the end of the phase.
- **Serial and deterministic.** Building the same table twice gives bit-identical
  results, and the cache relies on this. There is no rayon. T8 measures build times,
  and a parallel build waits until those justify it.
- **Error measures.** Phase 1's measures carry over:
  - per degree, in the §3.8 weighting, relative to the term magnitudes (for a table
    application the terms are |Aᵢₖ xₖ|);
  - the result's own degree norm is printed but not asserted.

  One more lesson applies in this phase. A table is exact in its geometry, but a
  direct operator on level l takes its shift from the centre difference c′ − c. At
  deep levels in a domain away from the origin, that difference carries a relative
  error of about ε |c| / r.
  - Tests that compare a table with `nd-fmm-ref` on level l use a dyadic domain for the
    tight tolerance. Its corner and side are exactly representable, so every centre
    and every centre difference is exact.
  - A generic domain is used only with a documented tolerance that accounts for that
    rounding.
  - The same argument tells `nd-fmm-exec` (Phase 3) to look tables up by integer
    offset and octant, never by a floating-point shift.
- **Test time.** Keep the debug-mode run of `cargo test -p nd-fmm-tables` under a
  minute. Checks at large p go into `#[ignore]` tests, which the PR runs with
  `cargo test -p nd-fmm-tables --release -- --ignored` and reports. From the Phase 1
  T7 timing, a dense M2L table at p = 20 takes 316 · 441 direct M2L calls of about
  0.2 ms each, which is tens of seconds in release mode.

## Exit gate
- Every acceptance test in the task briefs passes in CI. `nd-fmm-tables` is a default
  member, is MPI-free and has no CubeCL.
- CONVENTIONS §3.12 and the §3.10 change are reviewed and signed off by hand before T3
  starts.
- C2.1: the M2M and L2L tables reproduce `direct::m2m` and `direct::l2l` on parent
  levels 0 to 15, to 1e-14 (terms, per degree, dyadic domain), for p ≤ 30.
- C2.2: the M2L tables reproduce `direct::m2l` for all 316 offsets on levels 2, 9 and
  16, to 1e-14, for p ≤ 20. The 16-class form reconstructs all 316 matrices to 1e-13.
- C2.3: the table-driven rotation M2L, M2M and L2L agree with `nd_fmm_ref::rotation` to
  1e-14 and with `direct` to 1e-13, for p ≤ 20.
- C2.4: a cache load is bit-identical to a cold build for every family and both
  precisions. The cache rejects a stale convention or format version, a different p,
  precision or kind, and a corrupt file.
- The T8 PR contains the build times, the memory per family and form, and the time per
  application; they are copied into the design document. T8 recommends the CPU M2L
  strategy for Phase 3.

## Tasks

One pull request each.
- T2 needs no crate: it touches only the docs and `tools/fixtures/`. It can start at
  once, alongside T1.
- T3 needs T1 and T2, with §3.12 signed off.
- T4 and T6 need T3 and can run in parallel. T5 needs T4.
- T7 needs T3 to T6, and T8 needs T7.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-scaffold.md](T1-scaffold.md) | `fmm-tables` crate skeleton, `MatrixSet`, workspace entries, phase pointer in root `CLAUDE.md` | — | Phase 1 complete |
| T2 | [T2-geometry-conventions.md](T2-geometry-conventions.md) | CONVENTIONS §3.12 draft, §3.10 change, `check_symmetry.py` | prerequisite of C2.1–C2.3 | none |
| T3 | [T3-octant-tables.md](T3-octant-tables.md) | `geometry` module, M2M and L2L tables | C2.1 | T1, T2 (signed off) |
| T4 | [T4-m2l-tables.md](T4-m2l-tables.md) | dense M2L tables for the 316 offsets | C2.2 | T3 |
| T5 | [T5-m2l-symmetry.md](T5-m2l-symmetry.md) | cube symmetry group, coefficient transforms, 16-class M2L tables | C2.2 | T4 |
| T6 | [T6-rotation-tables.md](T6-rotation-tables.md) | rotation and coaxial tables, table-driven rotation M2M, L2L, M2L | C2.3 | T3 |
| T7 | [T7-cache.md](T7-cache.md) | versioned on-disk cache for every table family | C2.4 | T3–T6 |
| T8 | [T8-validate.md](T8-validate.md) | tables report and table-path accuracy in `nd-fmm-validate` | supports C2.1–C2.4; input to C3.1 | T7 |

Review T2 yourself before T3 starts. The later tasks encode §3.12 in tests and in the
cache format, so a rule changed afterwards means revisiting T3 to T7.

T4 and T6 both add a module to `fmm-tables/src/lib.rs`. Merge one, then rebase the
other; the conflict is one line.

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase2/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [x] T1 merged: `nd-fmm-tables` skeleton, CI green, root `CLAUDE.md` points to Phase 2
- [x] T2 merged: `check_symmetry.py` passes; §3.12 drafted
- [x] CONVENTIONS §3.12 and the §3.10 change reviewed and signed off; `CONVENTION_VERSION` decision recorded
- [x] T3 merged: M2M and L2L tables equal `direct` to 1e-14 on parent levels 0–15, p ≤ 30 (ignored release test for p > 12)
- [x] T4 merged: M2L tables equal `direct` to 1e-14 for all 316 offsets on levels 2, 9 and 16, p ≤ 20
- [x] T5 merged: 16 classes; the class form reconstructs all 316 matrices to 1e-13; memory reported
- [ ] T6 merged: table-driven rotation equals `nd_fmm_ref::rotation` to 1e-14 and `direct` to 1e-13, p ≤ 20
- [ ] T7 merged: cache round trip bit-identical; stale, mismatched and corrupt files rejected
- [ ] T8 merged: tables report in the PR; CPU M2L strategy for Phase 3 recommended
- [ ] Design documents updated: Section 7 Phase 2 status and measured numbers; Section 9.2 box-geometry question answered; workspace-structure §3 (dependencies of `nd-fmm-tables`), §3.1 (the built surface) and §6 ("Still open") match the result
