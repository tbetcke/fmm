# Phase 5 / T4 — nd-octree adaptations for the distributed FMM (part of C5.1)

Until now `nd-octree` stayed as it was: Phases 3 and 4 worked around it, and the only
change was a documentation and test PR (fmm-plan-redesign §12, decision 6). The
distributed FMM needs more from it:
- `nd-octree` balances distinct finest keys, not points or work, so coincident points
  weigh as one;
- a heavy coarse block cannot be split;
- construction panics when the coarse tree has fewer blocks than ranks, which a small
  `Fmm` on 8 ranks can reach;
- `Redistribution` may need a lookup the octree does not offer.

Changes to `nd-octree` are allowed where the distributed FMM needs them (decided on
2026-10-04, docs/phase5/README.md decision 6). T1 lists them, and the sign-off accepts
or rejects each one. **This task implements exactly the changes the sign-off accepted,
as `docs/design/distributed-fmm.md` §3 specifies them.** If the sign-off accepted none,
this task is not needed. Record that in the exit checklist.

Read first:
- root CLAUDE.md, octree/CLAUDE.md (all of it: the construction contract, ownership and
  ghosts, MPI discipline, debug against release, the rustdoc shape, the MPI examples);
- docs/phase5/README.md ("Requirements" 4 and 8, "Design decisions");
- docs/design/distributed-fmm.md, signed off: §3, §9 (collectives and memory), §10
  (testing) and §12 (the task check for T4);
- octree/src/octree.rs (`Octree::new`, `OctreeOptions`, the lookups) and
  octree/src/octree/implementation.rs (`compute_coarse_tree`,
  `compute_coarse_tree_weights`, `load_balance`, `partition_blocks`, the redistribution
  of keys, `balance`, `generate_all_keys`), with their unit tests;
- octree/examples/test_mpi_*.rs and octree/tests/.

Do, as the signed-off design specifies, for example (each only if accepted):
- **Weighted load balance.** Block weights from a weight per finest key: point
  multiplicities, so that duplicates count, or a caller-supplied cost. It enters through
  `OctreeOptions` or a new constructor, as the design fixes. The weights travel with the
  keys through the parallel sort and the redistribution of keys. The partition cuts at
  the weighted targets. A heavy block still cannot be split; the design says what
  happens to it.
- **Finer coarse partitions.** An option that refines the coarse tree until every rank
  can receive at least the design's number of blocks, so that the balance is not
  limited by a few heavy blocks. The design states its effect on the replicated data
  (each block is held on every rank with the ghost-children layer).
- **An agreed error instead of a panic** where a distributed `Fmm` can reach the panic
  (fewer blocks than ranks; any other input the design names). The error is returned on
  every rank, decided by data every rank holds or by an agreement in an existing
  collective, never by a rank-dependent branch around a collective.
  `Octree::new` returns `Result` if the design says so; update `nd-fmm-plan`'s tests and
  `nd-fmm-exec`'s `build` to the new signature in this PR (they are the only callers).
- **Lookups** that `Redistribution` needs (for example a batched `owner_rank`), local and
  communication-free.

Whatever is added:
- The defaults keep today's behaviour unless the design says otherwise. A tree built
  with default options has the same leaves and the same partition as before on every
  existing example and test.
- No collective is skipped or added in a rank-dependent branch. Every new collective is
  in the design's table, in its stated order, and ranks with no keys enter it.
- The construction contract and the ghost-children guarantees of octree/CLAUDE.md still
  hold, and its docs ("Construction contract", "Ownership and ghosts", the
  `OctreeOptions` docs, the crate-level operation and communication table) describe the
  change.
- `#![warn(missing_docs)]`; the rustdoc shape `# Parameters`, `# Returns`, `# Examples`,
  `# Collective operation`; no wildcard imports.

Tests that define done:
- Unit tests next to the serial helpers (`partition_blocks` with weights, refinement to a
  minimum block count, error cases), with `tools::seeded_rng`. The lib test binary's one
  MPI initialisation is spent; a new MPI unit test reuses it or moves to `examples/`.
- A new MPI example `examples/test_mpi_weighted_partition.rs` (or extended existing ones,
  as the design's task check says), registered with templated-examples. It covers:
  - weighted balance on uniform, clustered and duplicate-heavy keys: per-rank weight
    within the design's bound of the mean, the bound derived from the largest block;
  - a rank with no keys; every key on one rank;
  - the former panic case returning the agreed error on every rank;
  - every existing invariant: completeness, linearity and 2:1 balance, ghosts and
    neighbours, coarse blocks replicated with the ghost-children layer.
- The existing examples unchanged at default options. Run them at 1 and 3 ranks, and
  `test_mpi_construction_edge_cases` also at 2 and 4.
- `nd-fmm-plan`'s `tests/mpi_regressions.rs` on 1, 2 and 4 ranks, and
  `nd-fmm-exec`'s `tests/mpi_exec.rs` on 1 and 2 ranks, pass. With a signature change
  they are adapted; otherwise they are untouched.
- The measured imbalance on the workloads of the design §3 (cube, Plummer, clusters,
  duplicate-heavy; N = 10⁵ and 10⁶; 2, 4 and 8 ranks; release) before and after, by the
  design's weight. Reported, not asserted.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- octree/CLAUDE.md's crate checks: `cargo clippy -p nd-octree --all-targets --features
  strict -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-octree`,
  `cargo build -p nd-octree --examples`;
- every `nd-octree` MPI example at 1 and 3 ranks (and the new one at 2, 4 and 8) by hand,
  under an external timeout with the macOS loopback flags;
- the multi-rank CI job, if T3 kept it.

Report:
- each change, with its API and its collectives;
- the imbalance table before and after;
- construction time before and after at 1, 4 and 8 ranks (release; reported, not
  asserted);
- any existing test whose trees changed, and why that is expected.

Do not:
- add a change the sign-off did not accept, or store points or application data in the
  octree (octree/CLAUDE.md, "Project");
- change Morton key layout, the levels, the neighbour rules or 2:1 balance;
- change the defaults' partition, unless the design says so;
- touch `nd-fmm-plan` or `nd-fmm-exec` beyond adapting to a changed signature, or set
  the FMM's weights (T6 does that).
