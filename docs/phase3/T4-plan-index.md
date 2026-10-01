# Phase 3 / T4 — nd-fmm-plan: box index and index-based interaction lists (C3.0)

Starts only after `docs/design/fmm-plan-redesign.md` (T1) is signed off. That document
is the specification; this brief fixes only the scope and the acceptance tests. Where
the two disagree, the signed-off design wins. Report the difference.

This is the first of four tasks that rebuild `nd-fmm-plan` (T4–T7). It adds the box
index and the interaction lists in their new form, beside the old code. Nothing old is
changed or removed.

This task works inside fmm-plan/ and follows fmm-plan/CLAUDE.md, including its
multi-rank runs.

Read first: root CLAUDE.md, fmm-plan/CLAUDE.md, docs/phase3/README.md ("Requirements
on the new nd-fmm-plan", "Design decisions": "Old code as reference"),
docs/design/fmm-plan-redesign.md (all, closely §3, §4 and §11),
fmm-plan/src/interaction_manager.rs and its tests, fmm-plan/tests/mpi_regressions.rs,
docs/CONVENTIONS.md §3.12 (octant and offset order).

Do:
- Under the new module path of the design (for example `nd_fmm_plan::v2`, or the name
  it chose), add:
  - the per-level box index of design §3: every box a rank holds (local, `Global` and
    ghost) numbered in Morton order, the key ↔ index maps, and leaf numbering;
  - the interaction lists of design §4 in index form: U, W and X as CSR per level, V
    grouped per (level, offset) in the order of `V_LIST_DIRECTIONS`, and the M2M and
    L2L groupings per (level, octant);
  - the target-centric view of requirement 5: per level, CSR from each target to its
    V-list sources with each entry's offset index, from each parent to its children
    with their octants, and from each child to its parent, in the order of design §4;
  - whatever the design says about reusing today's list logic.
- Keep `V_LIST_DIRECTIONS` with its current path and order. nd-fmm-exec (T3) and
  CONVENTIONS §3.12 depend on it.
- Construction stays local (no collectives), unless the design states otherwise and
  says why.
- Docs: module docs in the style of `interaction_manager.rs`, with the guarantees
  (entries, sortedness, the at-most-once invariant per offset batch, what is a ghost)
  and their cost.

Tests that define done:
- Serial (`src/*_tests.rs`, no MPI), on hand-built key maps as today's
  `interaction_manager_tests.rs` uses:
  - the numbering is Morton order per level, dense (0..n) and deterministic across
    `HashMap` orders (build the input map in several insertion orders);
  - the index-form lists equal today's `InteractionManager` lists, mapped through the
    index, on every existing serial case;
  - every V batch has each target at most once; the batches of a level partition its
    V-list pairs; each pair's offset is index(target) − index(source);
  - every octant batch pairs each child with its parent, with
    o = `morton::child_index(child)`, and the eight batches of a level partition its
    parent–child pairs;
  - the target-centric view holds exactly the pairs of the groupings, with the same
    offset indices and octants, in its documented per-target order;
  - its CSR rows cover the targets of the level once each, so a level buffer split
    with `chunks_mut(size)` lines up with the rows (the property T10's threading
    relies on).
- tests/mpi_regressions.rs, in the existing `cases` loop (no new `#[test]`): every
  scenario also checks the index-form lists of every non-ghost box against the
  brute-force oracle, through the key ↔ index maps, and checks the invariants above.
  The existing checks stay as they are.
- Multi-rank, by hand, under an external timeout (fmm-plan/CLAUDE.md, "Multi-rank
  runs"; on macOS with the loopback flags of root CLAUDE.md): the `mpi_regressions`
  binary on 1, 2 and 4 ranks.

Must pass, and report which ran:
- the root CI commands and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` (debug run under a minute);
- `cargo clippy -p nd-fmm-plan --all-targets -- -D warnings`;
- `RUSTDOCFLAGS="-D warnings" cargo doc -p nd-fmm-plan --no-deps`;
- the multi-rank runs above, with their output.

Report the time and memory of building the index and lists against today's
`InteractionManager::new`, on the "graded corner blob" scenario and on a 10⁵-point
uniform tree on one rank (release; an example or an ignored test, not asserted).

Do not: change or remove any existing public item; add data stores, exchange or
operators (T5, T6); change nd-octree; put a collective in a rank-dependent branch. If a
list disagrees with the oracle, stop and report.
