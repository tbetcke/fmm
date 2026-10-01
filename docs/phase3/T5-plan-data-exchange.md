# Phase 3 / T5 — nd-fmm-plan: data stores and variable-size ghost exchange (C3.0)

The second of the four rebuild tasks. On top of the T4 index, it adds the data stores
of the new `nd-fmm-plan`:
- level buffers for multipoles and locals;
- leaf stores for source data, target input and target output, with a varying number
  of points per leaf.

It also adds the ghost exchange that fills them: variable-size source chunks for the U
and X lists, per-level multipoles for V and W. Like T4, it changes nothing old.

This task works inside fmm-plan/ and follows fmm-plan/CLAUDE.md. The signed-off
`docs/design/fmm-plan-redesign.md` is the specification; where it and this brief
disagree, the design wins. Report the difference.

Read first: root CLAUDE.md, fmm-plan/CLAUDE.md, docs/phase3/README.md (requirements 3,
4, 7 and 8), docs/design/fmm-plan-redesign.md (§5, §8, §10 and §11), the T4 code,
fmm-plan/src/ghost_communicator.rs and src/fmm/evaluator.rs (`LevelData`, `exchange`,
`upward_global`), and the rlst 0.9.0 sources of `distributed_tools::ghost_communicator`
(`ChunkSizes::PerIndex`, `receive_chunk_sizes`, `receive_offsets`, `send_chunk_sizes`,
`send_offsets`) in `~/.cargo/registry/src/*/rlst-0.9.0/`, not in any `../rlst`
checkout.

Do:
- Level buffers (design §5): one contiguous buffer per level and kind, box i's
  coefficients at i · size, sizes per level from the operator.
- Leaf stores (design §5, requirement 4):
  - per leaf, a count of points and a fixed number of values per point, stored CSR;
    counts may be zero;
  - sources for the local leaves and the ghost leaves the U and X lists need;
    target input and target output for the local leaves only;
  - a constructor from per-leaf counts in leaf order, mutable and immutable access per
    leaf index, `clear` for outputs.
- Ghost exchange (design §8):
  - variable-size source chunks, sized by the owner's count (rlst
    `ChunkSizes::PerIndex`; zero allowed), received into the source store;
  - per-level multipoles for the ghost boxes of the V and W lists, received into the
    level buffers;
  - flat send and receive buffers in index order (requirement 7);
  - validation of sizes and counts agreed on all ranks before any level communicator
    is built, as `FmmGhostCommunicator::new` does today.
- The gather of the coarse-tree multipoles that the global upward pass needs
  (today `upward_global` uses `gather_to_all`), in the new index form, without the
  M2M itself (T6 calls it).
- Docs: module docs with the layouts, the collectives and their order, and the
  guarantees.

Tests that define done:
- Serial (no MPI):
  - leaf stores: offsets and lengths for given counts, zero counts, disjoint chunks,
    `clear`, a leaf with no chunk of a kind;
  - level buffers: box i's chunk is at i · size, levels independent;
  - the ghost bucketing with per-key sizes.
- tests/mpi_regressions.rs, in the existing `cases` loop:
  - a forward exchange of seeded source chunks whose counts derive from a hash of the
    key (for example `hash(key) % 5`, so zeros occur and every rank can recompute any
    owner's count): every received chunk has the owner's length and values;
  - the same for multipoles with per-level sizes, checked against today's
    `check_ghost_exchange` values;
  - the coarse-tree gather: every rank holds every coarse leaf's chunk, in a
    deterministic order;
  - a scenario with one leaf at `max_level` holding far more points than
    `max_fine_keys` (for example 1,000 duplicate points), so one chunk dwarfs the
    others;
  - "empty input ranks" and "uneven rank populations" pass.
- Multi-rank, by hand, under an external timeout: the `mpi_regressions` binary on 1, 2
  and 4 ranks.

Must pass, and report which ran:
- the root CI commands and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` (debug run under a minute);
- `cargo clippy -p nd-fmm-plan --all-targets -- -D warnings`;
- `RUSTDOCFLAGS="-D warnings" cargo doc -p nd-fmm-plan --no-deps`;
- the multi-rank runs above, with their output.

Report the number of ghost keys and values exchanged per kind on 2 and 4 ranks in the
"graded corner blob" scenario, old against new.

Do not: change or remove any existing public item; add the operator interface or the
evaluator (T6); add overlap of exchange and computation (C5.2) or device buffers; put
communication inside a branch only some ranks take; change rlst. If the per-key
exchange needs an rlst change, stop and report.
