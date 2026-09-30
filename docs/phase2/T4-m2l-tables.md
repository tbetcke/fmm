# Phase 2 / T4 — nd-fmm-tables: dense M2L tables for the 316 offsets (C2.2)

Read first: docs/CONVENTIONS.md §3.9, §3.11 (M2L, including the truncation bound) and
§3.12; docs/phase2/README.md ("Design decisions"); the T3 code and tests; the docs of
`nd_fmm_ref::direct::m2l`; fmm-ref/tests/direct/m2l.rs (the Phase 1 M2L tests and
their bound).

Do:
- Module `m2l`, with `M2lTables<T>`: 316 matrices in the offset order of §3.12. It
  provides:
  - `build(p)`: applies `direct::m2l` at the canonical frames (source (0, 1), target
    (2d, 1)) to each unit vector, in f64, then casts to T;
  - `p()` and `matrices()`;
  - `apply(index, multipole, local)`, which accumulates and does not allocate, and
    `index(offset) -> Option<usize>`, which delegates to `geometry`.
- Doc comments cite §3.11 (M2L) and §3.12. They state:
  - the convergence condition that holds for every V-list offset (4 ≤ |b| ≤ 6√3);
  - the storage, 316 (p + 1)⁴ reals: in f64 about 17 MB at p = 8, 211 MB at p = 16
    and 492 MB at p = 20;
  - that T5 provides the 16-class form.

Tests that define done. f64 unless stated; the "terms" measure of T3.
- Canonical columns: column k of offset d equals `direct::m2l` of the k-th unit vector
  at the canonical frames, bit for bit.
- Levels, the C2.2 criterion:
  - Use the dyadic domain of T3, levels 2, 9 and 16, and random target boxes with a
    source box at each offset that lies inside the domain.
  - Apply the table to random multipoles and compare with `direct::m2l` at the actual
    frames, to 1e-14 (terms).
  - All 316 offsets at p ≤ 8 in the debug run, plus the 16 class representatives of
    §3.12 at p = 12. All 316 offsets at p = 16 and p = 20 in an `#[ignore]` release
    test.
- Chain:
  - P2M, then the table, then L2P equals the same chain with `direct::m2l`, to 1e-13
    (terms).
  - Against `direct_sum`, for the offsets (2, 0, 0), (2, 2, 2) and (3, 3, 3) at
    p ∈ {4, 8, 12}, within the §3.11 truncation bound, as the Phase 1 T5 test measures
    it.
- Structure:
  - Column 0 (a unit monopole) equals P2L of a unit charge at the source centre, in the
    target frame, to 1e-13 (terms).
  - The table at p is the leading block of the table at p + 5. Report whether it is
    bit-identical.
  - Inversion: the table of −d equals diag((−1)ʲ) · table(d) · diag((−1)ⁿ), for all 158
    pairs ±d at p ≤ 8. Report whether it holds bit for bit.
- Determinism: building twice gives bit-identical tables.
- f32, p ≤ 8: the cast table applied in f32 matches f64 to 1e-5 (terms).
- proptest for random levels, boxes, offsets and multipoles, 64 cases per property.

Must pass:
- `cargo test -p nd-fmm-tables` (debug run under a minute);
- `cargo test -p nd-fmm-tables --release -- --ignored`;
- `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-tables --no-deps` without warnings;
- the root checks.

Report the measured worst error of every test, and the release build time and memory
at p = 8, 12, 16 and 20.

Do not: add the symmetry-reduced form (T5), rotation tables (T6) or the cache (T7);
build tables for adaptive (non-V-list) offsets; build or apply in parallel.
