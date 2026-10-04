# Phase 5 / T5 — nd-fmm-plan: redistribution of points to their owning ranks (part of C5.1)

`Fmm::build` bins each rank's points into finest keys, and the octree partitions the keys
into Morton ranges. A point therefore usually lies in a leaf that another rank owns, and
today `build` stops with `PointsNotOwned`. fmm-plan-redesign §9 designed the fix, a
`Redistribution` in `nd-fmm-plan` that moves per-item payloads to the ranks that own
their leaves and the results back. T1 refined it in `docs/design/distributed-fmm.md` §4.
This task builds it, kernel-agnostically. T6 uses it in `Fmm`.

Read first:
- root CLAUDE.md, fmm-plan/CLAUDE.md;
- docs/phase5/README.md ("Requirements" 1–5 and 8, "Design decisions", "Workloads");
- docs/design/distributed-fmm.md, signed off: §4, §9 and §10, and §12 for T5;
- docs/design/fmm-plan-redesign.md §3.2 (lookups), §3.3 (leaf numbering) and §9;
- fmm-plan/src/{index, plan, exchange}.rs (`BoxIndex::find_leaf`,
  `local_leaf_containing`, the validation-and-agreement pattern of the exchanges);
- `nd_octree::Octree::{owner_rank, local_leaf, lookup_leaves}` (communication-free
  lookups, the `i32` overflow agreement of `lookup_leaves`), and T4's additions if any;
- fmm-plan/tests/mpi_regressions.rs (the `cases` loop and the scenarios).

Do:
- A module `redistribute` with `Redistribution` and `RedistributionError`, with the API
  of the signed-off design. The starting point is fmm-plan-redesign §9:
  - `new(octree, plan, keys)`: collective. Each local item's finest key gives its owner
    (`owner_rank`, no communication). It makes one all-to-all of counts, one agreement
    that every count fits MPI's `i32`, and one all-to-all-v of the routing data. On the
    owner, each item gets its local leaf through the octree and the `BoxIndex`;
  - `counts()`: received items per local leaf, in leaf order, for `Evaluator::new`;
  - `origins()`: the origin (rank, position) of every received item, grouped by leaf;
  - `forward(payload, per_item)`: collective; `per_item` values of each local item to
    its owner, returned grouped by leaf, in the order the design fixes within a leaf;
  - `backward(results, per_item)`: collective; values of each received item back to its
    origin, returned in the original item order;
  - the payload type is independent of the plan's `Value`: generic over
    `T: Equivalence + Copy + Default`.
- Invalid input (a key not at the finest level, or invalid; a payload of the wrong
  length) is an error on every rank, agreed before the first collective that depends
  on it (requirement 4). A rank with no items calls everything with empty slices.
- The order within a leaf, exactly as the design fixes it. If the design adopted a
  rank-count-independent order (for example by finest key, then by the f64
  coordinates), `new` takes what it needs to apply it, and the docs state which
  differences across rank counts remain.
- Memory and traffic linear in the items moved. No allocation in `forward` or
  `backward` beyond the returned vector and one send buffer. A `_into` variant that
  writes into a caller's buffer is allowed if the design asks for one: T6 forwards
  charges on every evaluation.
- Docs: what is moved and when, the collectives in order (`# Collective operation`),
  determinism (fixed for fixed input and ranks; what changes with the distribution),
  memory, and an example (`no_run`, it initialises MPI). Update fmm-plan/CLAUDE.md
  (code map; "not implemented (C5.1)" goes) and the crate docs.

Tests that define done:
- Serial (`src/redistribute_tests.rs`, no MPI): the local parts. Grouping by leaf, the
  order within a leaf, count and offset arithmetic, and the inverse permutation of
  `backward`, on hand-made routing data.
- `tests/mpi_regressions.rs`: in every scenario (the `cases` loop; the scenario set does
  not shrink), with the source and the target points of the scenario:
  - for each input distribution of the README ("Workloads": all on rank 0, a seeded
    random share, the octree's own partition, one empty rank), built from the scenario's
    global point set the same way on every rank:
    - every item arrives exactly once, on the rank that owns its leaf, in that leaf.
      Check it against `octree.local_leaf` on the owner and against the counts the
      `Plan`'s leaves imply;
    - `counts()` sums to the items this rank owns, and the global sum is the total;
    - `forward` then `backward` of a payload that encodes (origin rank, position) is
      the identity, for `f64` with `per_item` 3 and for `u32` with `per_item` 1;
    - `origins()` agrees with the forwarded payload;
    - the order within each leaf is the design's;
  - two `forward` calls on the same `Redistribution` are bit-identical;
  - invalid keys on one rank give the agreed error on every rank, with no hang.
- A new scenario if the design asks for one, for example many duplicate keys spread over
  every rank, or a point set whose keys all belong to one rank's range.
- The existing plan, list, exchange and recording-operator checks unchanged.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-fmm-plan`;
- `tests/mpi_regressions.rs` on 1, 2 and 4 ranks by hand, under an external timeout with
  the macOS loopback flags, and on 8 ranks once;
- the multi-rank CI job, if T3 kept it.

Report:
- the collectives of `new`, `forward` and `backward`, with their counts per rank;
- the time and traffic of `new` and `forward` for the cube at N = 10⁶ from all points on
  rank 0 and from a random share, on 2, 4 and 8 ranks (release; reported, not asserted).

Do not:
- change `Fmm` or anything in `nd-fmm-exec` (T6);
- put kernel- or Laplace-specific code in `nd-fmm-plan` (no leaf scaling, no charges as
  such: payloads are opaque values);
- change the plan, the lists, the exchanges or the evaluator;
- use the index FMM (removed in T2).
