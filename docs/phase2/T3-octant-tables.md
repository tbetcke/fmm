# Phase 2 / T3 — nd-fmm-tables: geometry, M2M and L2L tables (C2.1)

Starts only after CONVENTIONS §3.12 (T2) is signed off.

Read first: docs/CONVENTIONS.md §3.6, §3.7, §3.11 and §3.12;
docs/phase2/README.md ("Design decisions"); fmm-tables/CLAUDE.md; the docs of
`nd_fmm_ref::{Frame, Workspace, direct}`; the T1 code; fmm-ref/tests/direct/common.rs
(the weighted per-degree measure to mirror in this crate's tests).

Do:
- Module `geometry` (§3.12):
  - `octant_direction(o) -> [i64; 3]`, the vector s_o of child index o.
  - `M2L_OFFSET_COUNT = 316`, `m2l_offsets() -> [[i64; 3]; 316]` in the order of
    §3.12, and `m2l_offset_index(d) -> Option<usize>` in the closed form of §3.12. Use
    the element type of `V_LIST_DIRECTIONS`.
  - The canonical frames: `m2m_frames(o)`, `l2l_frames(o)` and `m2l_frames(index)`,
    each returning the (input, output) `Frame<f64>` pair.
- Module `octant`, with `M2mTables<T>` and `L2lTables<T>`. They may share a private
  implementation, but keep the two public types. Each provides:
  - `build(p)`: for each octant o and column k, applies `direct::m2m` (`direct::l2l`)
    at the canonical frames to the k-th unit vector, into a zeroed column of a
    `MatrixSet<f64>`, then casts to T. It uses one `Workspace` and may allocate.
  - `p()`, `matrices() -> &MatrixSet<T>`, and `apply(o, input, output)`, which
    accumulates and does not allocate.
- Doc comments cite §3.11 (M2M, L2L) and §3.12 (octants, canonical frames, layout).
  They give the storage, 8 (p + 1)⁴ reals per family, and the build cost.
- Crate-level doc: add §3.12 to the conventions relied on.

Tests that define done. f64 unless stated. Coefficients are compared per degree, in
the §3.8 weighting (Nₘ for multipoles, Nₘ/Sₘ for locals), relative to the term
magnitudes |Aᵢₖ xₖ| of the table application ("terms").
- Geometry:
  - `octant_direction` matches the bits of o;
  - the offsets are 316 in number, in lexicographic order, not adjacent, and the index
    round-trips;
  - the canonical frames are those of §3.12.
- Canonical columns: column k of octant o equals the `direct` operator applied to the
  k-th unit vector at the canonical frames, bit for bit.
- Level independence, the C2.1 criterion:
  - Use a dyadic domain, a = (−1.25, 0.5, 2) and w = 3, so every centre and every
    centre difference is exact.
  - On parent levels l ∈ {0, 1, 4, 8, 12, 15}, pick random parent boxes and each
    octant. Apply the table to random coefficients, and compare with `direct::m2m`
    (`direct::l2l`) at the actual child and parent frames, to 1e-14 (terms).
  - p ≤ 12 in the debug run; p = 20 and p = 30 for all octants in an `#[ignore]`
    release test.
- Generic domain: a = (0.1, −2.3, 7.9), w = 0.37, parent level 15.
  - Derive a tolerance from the rounding of c − c′ relative to r, and document it in
    the test's doc comment.
  - Report the measured error next to the dyadic one. This test documents the geometry
    lesson of docs/phase2/README.md.
- Physical chains, as in Phase 1 T5, to 1e-13 (terms):
  - P2M at a child box, then the M2M table, equals P2M at the parent.
  - L2P after the L2L table equals L2P of the original local, at points inside the
    child.
- Composition: M2M table (o₂) after M2M table (o₁) equals `direct::m2m` from the
  grandchild to the grandparent, to 1e-13; the same for L2L.
- Structure:
  - M2M output degree j uses only input degrees k ≤ j, and L2L uses only n ≥ j. The
    blocks outside those ranges are exactly zero.
  - At p = 0, the M2M table is [1] and the L2L table is [½]. Check p = 1 against
    values worked out by hand in the test's doc comment.
- Leading block: the table at p equals the leading block of the table at p + 3. Report
  whether it is bit-identical.
- Determinism: building twice gives bit-identical tables.
- f32, p ≤ 8: the cast table applied in f32 matches f64 to 1e-5 (terms).
- proptest for random levels, boxes, octants and coefficients, 64 cases per property.

Must pass:
- `cargo test -p nd-fmm-tables` (debug run under a minute);
- `cargo test -p nd-fmm-tables --release -- --ignored`;
- `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-tables --no-deps` without warnings;
- the root checks.

Report the measured worst error of every test and the release build time of both
families at p = 8, 20 and 30.

Do not: add M2L tables (T4), rotation tables (T6) or the cache (T7); depend on
nd-octree or nd-fmm-plan; derive table entries from a formula instead of `direct`;
optimise the build beyond clarity. If a table disagrees with `direct`, stop and report.
