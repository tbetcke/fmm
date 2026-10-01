# Phase 3 / T1 — design of the new nd-fmm-plan (design for C3.0 and C4.0)

`nd-fmm-plan` predates the library design. Its algorithms are sound and well tested,
but its data model is not what the rest of the library needs:
- `HashMap` lookups by Morton key on every operator call;
- level columns in insertion order;
- a fixed amount of data per leaf;
- one pair of boxes per call, through `&self`;
- no place for target input.

The crate may be rewritten completely. This task writes the design of its replacement
as `docs/design/fmm-plan-redesign.md`, for sign-off by hand before any code changes. It
is the Phase 3 counterpart of the conventions tasks of Phases 1 and 2, and it writes
no Rust.

Read first:
- root CLAUDE.md, fmm-plan/CLAUDE.md and octree/CLAUDE.md;
- docs/phase3/README.md, closely: "Requirements on the new nd-fmm-plan" is what this
  design answers; also "Design decisions";
- every source file of fmm-plan/ (`src/lib.rs`, `src/fmm.rs`, `src/fmm/*.rs`,
  `src/interaction_manager.rs`, `src/ghost_communicator.rs`, their `*_tests.rs`,
  `tests/mpi_regressions.rs`, `examples/test_index_fmm.rs`), in full;
- the public API of nd-octree (`src/octree.rs`, `src/morton.rs`), and the rlst 0.8.0
  sources of `distributed_tools::{ghost_communicator, array_tools}` in
  `~/.cargo/registry/src/*/rlst-0.8.0/`;
- docs/design/laplace-fmm-plan.md §5 (architecture, interfaces, data layout), §6.4–§6.6
  (kernel mapping, batching, avoiding atomics) and §9;
- docs/design/workspace-structure.md §1.1 and §4;
- docs/CONVENTIONS.md §3.12 (octant and offset order);
- docs/phase3/T2-leaf-data-conventions.md (what the Laplace operator will store per
  leaf).

Write `docs/design/fmm-plan-redesign.md` with these sections:

1. **Assessment.** For each module of today's crate:
   - what it does, how it is tested, and what to keep: algorithms, invariants, tests
     and oracles;
   - what to replace, and why, tied to the requirements of docs/phase3/README.md.

   Be concrete, with file and function names. Name any defect or fragile spot found on
   the way, for example a `HashMap` iteration order that leaks into a result.
2. **Requirements.** Restate requirements 1–10 of docs/phase3/README.md. For each,
   say how the design meets it. If one should change, say why and propose the change
   as a sign-off question.
3. **Box index.**
   - The numbering of the boxes a rank holds on every level (local, `Global`, ghost)
     in Morton order. How it is built from `Octree::all_keys`, and what it costs.
   - Key ↔ index lookups, and which of them are allowed off hot paths.
   - Leaf numbering, and how leaves map to level positions.
   - What changes when the tree is rebuilt.
4. **Interaction lists.**
   - The index form of U, V, W and X (CSR per level), and the groupings: V per
     (level, offset) in the order of `V_LIST_DIRECTIONS`, M2M and L2L per
     (level, octant).
   - Whether the list rules are computed from today's `InteractionManager` logic or
     rewritten. Either way, the brute-force oracle is the specification.
   - The invariant that each target appears at most once per (level, offset) batch,
     and the analogous invariant for octant batches.
   - The target-centric view of requirement 5 (V per target with offset indices,
     children per parent with octants, parent per child), its order, and how it relates
     to the groupings: the same pairs, and the same per-target order whichever view an
     operator uses, or a stated difference.
   - Memory per box.
5. **Data stores.**
   - Level buffers for multipoles and locals (size per level, possibly varying with
     the level).
   - The three leaf stores of requirement 4: layout, counts, zero counts, and how each
     is filled and read.
   - A single `Value` type for all data, against separate types for source data,
     target input, target output and coefficients. Recommend one with reasons; mixed
     precision (f64 positions with f32 coefficients) is the case to weigh. §3.13 makes
     it unnecessary for Laplace.
6. **Operator interface.**
   - The trait or traits, as Rust signatures with doc comments: the batched calls of
     requirement 6, what each receives (level, octant or offset, index arrays, the keys,
     buffer views), and the accumulation rule.
   - `&mut self` access, so operators own their scratch.
   - The per-pair adapter, a trait with one method per pair and a blanket or wrapper
     implementation of the batched trait, used by `IndexFmm`.
   - **GEMM check.** Sketch how a Phase 4 M2L would run one (level, offset) batch:
     gather source columns, one GEMM with the table, scatter-add into target columns,
     no atomics (design §5.3, §6.4–§6.6). Show that the interface hands it everything
     it needs without a `HashMap`. Do the same for one M2M octant batch and for P2P
     over a level.
   - **Host-parallel check.** Sketch how a host operator runs one level's M2L, M2M,
     L2L and P2P with rayon (T10): targets in parallel through the target-centric view
     (requirement 5), each target's output slice borrowed mutably with safe Rust
     (`par_chunks_mut` on a level buffer, disjoint leaf chunks by CSR offsets), each
     target's contributions in CSR order. Show that this needs no `unsafe`, no atomics
     and no change to the plan, and that the serial loop is the same code with a
     sequential iterator.
7. **Evaluator.**
   - The pass order, unchanged from `src/fmm.rs` (compute graph steps 1–6).
   - Where each collective sits, and how ranks with empty input reach all of them.
   - The global coarse levels.
   - The order in which contributions are accumulated into each target, which
     requirement 8 needs.
   - Public stage methods and a `reset()`, so a caller can time each stage.
8. **Ghost exchange.**
   - Variable-size source chunks per key (rlst's `ChunkSizes::PerIndex` allows sizes of
     zero); multipoles per level.
   - Flat buffers in index order, so that staging to a device is one copy (requirement
     7).
   - Whether `FmmGhostCommunicator` is kept, adapted or replaced.
9. **Redistribution (designed here, implemented in C5.1).**
   - An API that takes, on every rank, items with a finest-level key and a payload, and
     returns, on each owning rank, the items in its leaves, grouped by leaf, with
     their origin (rank and position), so that results can be sent back.
   - The collectives it uses (for example `Octree::lookup_leaves`, then an
     all-to-all), and their cost.
   - The return path for results.
10. **Device and overlap compatibility.** What Phase 4 (device-resident level buffers,
    uploaded index arrays) and C5.2 (overlap of exchange with P2P and the local upward
    pass) need from this design, and why it does not preclude them. Do not design them.
11. **Migration and tests.**
    - The split into T4 (index and lists), T5 (data stores and exchange), T6 (operator
      interface, evaluator, `IndexFmm`) and T7 (removal and docs), with each task's new
      modules and the module path they live under beside the old code.
    - Which existing tests carry over unchanged, which are ported, and what each task
      adds. In particular: the list oracle on the index form; `IndexFmm` with variable
      counts, where every target point of leaf t holds, at index j, the number of source
      points of leaf j; and the old evaluator as the reference until T7.
    - The multi-rank runs each task needs.
12. **Questions for sign-off.** At least:
    1. the requirements, and any change this design proposes to them;
    2. one `Value` type or several;
    3. the trait shape: batched only with a per-pair adapter (recommended), or both as
       peers;
    4. whether the interaction-list logic is reused or rewritten;
    5. moving nd-fmm-plan to `[workspace.dependencies]` and `[workspace.package]`, and
       fixing its stale metadata (licence form `"MIT / Apache-2.0"`, the
       `codeberg.com.com` homepage, the old `repository`). Recommend yes, in T7;
    6. any change to nd-octree that the design would like. Propose it as a separate
       decision, with a workaround for Phase 3;
    7. whether redistribution should move from C5.1 into Phase 3.

Keep the document in the style of the other design documents: short sections, tables
where they help, Rust signatures where they decide something, no code beyond sketches.

The PR description must contain a one-page summary of the design, the sign-off
questions with a recommendation for each, and the measured size of today's crate
(lines per module, tests per module).

Must pass: nothing builds differently; run `cargo fmt -- --check` and
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` once, to confirm the starting point,
and report the result.

Do not: write or change any Rust; change docs/phase3/ briefs other than to fix a factual
error (say which); change nd-octree. If a requirement cannot be met without changing
nd-octree or rlst, say so in the document and in the PR, and propose the smallest change.
