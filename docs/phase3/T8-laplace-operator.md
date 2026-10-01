# Phase 3 / T8 — nd-fmm-exec: the Laplace operator (C3.1)

This task implements the batched operator interface of the new `nd-fmm-plan` (T6) for
the Laplace kernel, on the host:
- the translations apply the Phase 2 tables, chosen by the batch's octant or offset;
- the leaf operators and P2P call `nd-fmm-ref` in the leaf-scaled coordinates of §3.13.

Every batch is executed pair by pair, serially. No operator formula is new. The
per-pair kernels are inherent methods, so they can be tested one pair at a time.

Read first: docs/CONVENTIONS.md §3.1, §3.7, §3.11 (L2P and M2P), §3.12 and §3.13;
docs/phase3/README.md ("Design decisions", all); fmm-exec/CLAUDE.md; the T3 code; the
operator traits, the per-pair adapter, the evaluator and the leaf stores of the new
nd-fmm-plan (T5, T6) and their docs; docs/design/fmm-plan-redesign.md §6; the docs of
`nd_fmm_ref::{leaf, p2p, Workspace}` and of `nd_fmm_tables::{M2mTables, L2lTables,
M2lTables, M2lClasses, RotationTables, TableCache, CacheOutcome}`; fmm-tables/tests (the
"terms" measure to mirror).

Do:
- Add mpi (`workspace = true`) to nd-fmm-exec.
- `M2lStrategy::{Dense, Classes, Rotation, Auto}` and its resolution for a given p, as
  docs/phase3/README.md ("Strategy") states. `Auto` resolves to `Dense` for p ≤ 8 and
  `Rotation` for p ≥ 9; document the source of that rule (design §7, Phase 2
  recommendation).
- `Tables<T>`: the translation tables a resolved strategy needs, and nothing else.
  - `Tables::build(p, strategy)`, and `Tables::load_or_build(p, strategy, &TableCache)
    -> (Tables<T>, Vec<(TableKind, CacheOutcome)>)`, which reports every family.
  - `m2m(o, …)`, `l2l(o, …)` and `m2l(offset_index, …)` dispatch to the chosen family,
    with the scratch the family needs.
- `LaplaceOperator<T: RealScalar + Equivalence>`:
  - built from p, the tables, whether gradients are wanted, and the global maximum
    number of points per leaf;
  - owns an `nd_fmm_ref::Workspace` for p, the table scratch and one buffer for mapped
    P2P sources, all sized at construction. The interface gives `&mut self`, so there
    is no `RefCell`. If the signed-off design chose otherwise, follow it and say how
    scratch is held;
  - coefficient sizes (p + 1)² on every level. Per point: 4 source values, 3 target
    input values (the leaf-scaled position), and 1 target output value, or 4 with
    gradients (§3.13).
- Per-pair kernels, as inherent methods taking the two keys and the chunks, each
  accumulating, none allocating, none applying 1/(4π):
  - M2M and L2L by `morton::child_index(child)`; check in debug builds that the pair
    is parent and child;
  - M2L by `m2l_offset_index(index(target) − index(source))`; panic with both keys if
    the offset is not a V-list offset or the levels differ;
  - P2M, P2L, L2P, M2P and P2P with the frames of §3.13, from
    `geometry::relative_frame`:
    - source and target chunks are split with `as_chunks::<3>()`, without copying;
    - P2P maps the sources of a different leaf into the target's coordinates in the
      scratch buffer; the self pair uses the chunk directly;
  - a leaf with no points is a no-op.
- The batched trait implemented target by target, through the target-centric view
  of each level call: for every target, its contributions in CSR order, each by a
  per-pair kernel, with the table chosen by the entry's octant or offset index (debug
  builds check that it agrees with the keys). Write the per-target body as a function
  of one target's output slice, the shared inputs and one scratch set, so T10 can run
  it from a parallel iterator unchanged. Document the order. Also implement the plan's
  per-pair trait (or adapter), so tests can compare the two paths.
- Doc comments cite §3.11, §3.12 and §3.13 and give, for each operator, the frames it
  uses and its cost.

Tests that define done. f64 unless stated. The per-pair kernels are called directly,
without an octree, on keys of a dyadic domain (a = (−1.25, 0.5, 2), w = 3) unless
stated. The reference is `nd-fmm-ref` at the absolute frames of §3.12, with the
absolute coordinates the points were generated in. Its potentials and gradients are
converted to leaf-scaled units (§3.13) for the comparison.
- Translations, the C3.1 criterion for M2M, L2L and M2L:
  - on levels 2, 9 and 16, random boxes, every octant and every V-list offset that lies
    inside the domain;
  - random scaled coefficients, compared with `direct::m2m`, `direct::l2l` and
    `direct::m2l` at the actual frames;
  - to 1e-14 (terms, per degree) with `Dense`, and to 1e-13 with `Classes` and
    `Rotation`;
  - all offsets at p ≤ 6 in the debug run; p = 8 for every strategy and p = 16 for
    `Rotation` in an `#[ignore]` release test.
- Leaf operators and P2P, the C3.1 criterion for the rest, to 1e-13 (terms), p ≤ 12,
  potentials and gradients:
  - P2M and L2P at leaves on levels 2, 9 and 16;
  - P2L for X-list geometry: a source leaf on a coarser level that touches the
    target's parent but not the target (state the pairs used);
  - M2P for W-list geometry: a source box on a finer level whose parent touches the
    target leaf and which itself does not;
  - P2P for a leaf with itself (coincident sources and targets excluded), with each of
    its 26 neighbours on the same level, and with neighbours one level coarser and
    finer.
- Chains against `direct_sum` (φ̂ converted to 1/|x − y| units; 4π not involved):
  P2M → M2L → L2P for (2, 0, 0), (2, 2, 2) and (3, 3, 3) at p ∈ {4, 8, 12}, within
  the §3.11 bound as the Phase 1 T5 test measures it.
- Strategy: `Dense`, `Classes` and `Rotation` give the same locals to 1e-13 (terms);
  `Auto` resolves as documented for p = 0..=20.
- Tables: `load_or_build` with a cache directory under `env!("CARGO_TARGET_TMPDIR")`
  reports `Built`, then `Loaded`, with bit-identical results; `build` without a cache
  works. Keep p ≤ 4 here.
- Generic domain (a = (0.1, −2.3, 7.9), w = 0.37), level 16: the operator against
  `nd-fmm-ref` at absolute frames. Report the error and explain it in the test's doc
  comment (the reference, not the operator, carries ε |c| / r); assert only a
  documented tolerance.
- f32, p ≤ 8: each operator in f32 against f64, to 1e-5 (terms).
- Panics: a non-V-list offset, a non-parent pair, a chunk whose length is not a
  multiple of the size per point.
- No allocation: the operator's buffers are not reallocated across a run of every
  operator on the largest leaf (check capacity before and after).
- proptest for random keys, levels, offsets and points, 64 cases per property.
- tests/mpi_exec.rs: the one MPI-initialising test of this executable, with a `cases`
  list that T9, T10 and T11 extend, as in fmm-plan/tests/mpi_regressions.rs. Scenarios, on
  one rank in CI:
  - table order against the plan: for a uniform level-3 tree in a dyadic domain, every
    (level, offset) batch's offset index is the position of its direction in
    `V_LIST_DIRECTIONS` and `m2l_offset_index` of it; every octant batch's o is
    `morton::child_index` of its children; and the centre difference of every V pair,
    from `morton::physical_box`, is 2 r_l d exactly;
  - batched against per-pair: on a small adaptive tree with points loaded by hand,
    the evaluator with the batched operator and with the per-pair adapter give
    bit-identical target output.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug run under a minute);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`;
- `cargo clippy -p nd-fmm-exec --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-exec --no-deps` without warnings;
- the root checks.

Report the measured worst error of every test, and the table build or load time per
strategy at p = 8 and 16.

Do not: add `FmmBuilder` or point binning (T9); add rayon or GEMM paths; re-derive a
translation instead of using the tables; change nd-fmm-plan, nd-fmm-tables or
nd-fmm-ref. If an operator disagrees with nd-fmm-ref beyond its tolerance, stop and
report.
