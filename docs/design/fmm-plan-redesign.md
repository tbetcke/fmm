# nd-fmm-plan redesign: box index, batched operators, variable leaf data

As of 2026-10-01. Phase 3 / T1. Status: **signed off** on 2026-10-01 by Timo Betcke, with
every recommendation accepted. The decisions are recorded in Section 12. This document is
the specification for T4–T7.

This document designs the replacement of `nd-fmm-plan` (`fmm-plan/`). It answers the
ten requirements of `docs/phase3/README.md` ("Requirements on the new nd-fmm-plan") and
follows its "Design decisions for this phase". T4–T7 implement it, in that order, beside
the old code. Companion documents: `docs/design/laplace-fmm-plan.md` (§5, §6.4–§6.6,
§9) and `docs/design/workspace-structure.md` (§1.1, §4). `docs/CONVENTIONS.md` wins on
any conflict; this design changes no convention. References to §3.13 (Laplace leaf data)
are to the section T2 drafts; the plan itself is independent of that layout.

In one paragraph: every box a rank holds gets a dense, Morton-ordered `u32` index per
level; the U, V, W and X lists become index arrays, held twice, once per target (CSR)
and once grouped by V-list offset or child octant, with the same per-target order in
both; leaf data lives in three CSR stores with a varying number of points per leaf;
operators receive one call per level and kind, with `&mut self`, the level's buffers
and both views; and the evaluator keeps today's pass order, collectives and global
coarse levels. A per-pair adapter keeps `IndexFmm` and other simple operators at one
method per pair.

## 1. Assessment

### 1.1 Size of today's crate

Measured on `fecdf7c`. "Code" excludes blank and `//` lines; `interaction_manager.rs`
includes the 316-row `V_LIST_DIRECTIONS` table (about 318 of its 506 code lines).

| File | Lines | Code | Tests |
| --- | --- | --- | --- |
| `src/lib.rs` | 9 | 3 | – |
| `src/fmm.rs` | 45 | 3 | – |
| `src/fmm/operator.rs` | 96 | 53 | – |
| `src/fmm/evaluator.rs` | 429 | 312 | – |
| `src/fmm/index_fmm.rs` | 186 | 127 | – |
| `src/interaction_manager.rs` | 730 | 506 | – |
| `src/ghost_communicator.rs` | 324 | 171 | – |
| `src/interaction_manager_tests.rs` | 507 | 411 | 9 |
| `src/ghost_communicator_tests.rs` | 82 | 72 | 5 |
| `src/fmm/evaluator_tests.rs` | 52 | 46 | 4 |
| `src/fmm/index_fmm_tests.rs` | 44 | 39 | 4 |
| `tests/mpi_regressions.rs` | 399 | 349 | 1 test, 11 scenarios |
| `examples/test_index_fmm.rs` | 48 | 37 | – |
| **Total** | 2951 | 2129 | 22 unit + 1 integration, no doctests |

Baseline on this branch: `cargo fmt -- --check` clean; `RUST_MIN_STACK=8388608 cargo
test -p nd-fmm-plan` passes 22 unit tests and the integration test (one rank).

### 1.2 `interaction_manager.rs`

**What it does.** `InteractionManager::new(&octree)` calls the private
`from_key_types(all_keys)`, which visits every non-ghost key of `Octree::all_keys` and
derives its lists from neighbour classifications only:
- U and W from the 26 same-level neighbour cells of a leaf: a leaf cell goes to U; an
  interior cell contributes its adjacent children to U and the others to W; a missing
  cell contributes its parent (a coarser leaf, by 2:1 balance) to U;
- V and X from the neighbour cells of the parent: children of interior neighbours not
  adjacent to the key go to V; leaf neighbours not adjacent to the key go to X.

Lists are `HashMap<MortonKey, Vec<MortonKey>>`, sorted and deduplicated, with an entry
for every non-ghost key. `new` asserts that every entry is a key of the tree when the
tree carries the ghost-children layer. `v_list_by_direction(level)` groups the V pairs
of a level by offset into a `HashMap<[i64; 3], Vec<(target, source)>>`.
`V_LIST_DIRECTIONS` lists the 316 offsets in table order (CONVENTIONS §3.12).

**Tests.** Nine serial tests on hand-built key maps (`interaction_manager_tests.rs`):
hand counts on a uniform level-2 tree, W and X on a refined octant, the brute-force
`Oracle` on three adaptive trees, invariants, `is_adjacent`, ghost classification, the
completeness of `V_LIST_DIRECTIONS` and of `v_list_by_direction`. On every rank count,
`tests/mpi_regressions.rs::check_interaction_lists` compares all four lists of every
non-ghost key with `oracle_lists` over the gathered global tree.

**Keep.** The per-key rule (the loop body of `from_key_types`), which the oracle has
checked on every scenario; `is_adjacent`; `V_LIST_DIRECTIONS` at its current path
(CONVENTIONS §3.12 and nd-fmm-exec cite it); both oracles and every test tree; the
statement of what the rule relies on (completeness, 26-direction 2:1 balance, the
same-level halo, the ghost-children layer).

**Replace.**
- The map-based storage, because the evaluator looks lists up by key on every target
  (requirement 2) and they have no index form (requirement 5).
- `v_list_by_direction`, because it allocates a `HashMap` of 316 vectors per call, is
  keyed by the offset vector instead of its table index, and is not used by the
  evaluator (requirement 5).

**Fragile spots.**
- The assertion in `new` and the `expect`s in `from_key_types` panic on one rank only.
  Any other rank then blocks in its next collective. The new plan validates locally and
  agrees the outcome with an all-reduce before it returns (Section 3.6).
- No `HashMap` iteration order leaks into a result: every list is sorted, and
  `v_list_by_direction` sorts each vector.

### 1.3 `ghost_communicator.rs`

**What it does.** `FmmGhostCommunicator<T>` builds one rlst `GhostCommunicator<MortonKey>`
per level from the ghost keys among the keys it is given (local and `Global` keys
skipped, owner from `KeyType::ghost_rank`), with `LevelChunkSizes::{Uniform, PerLevel}`.
It owns a send and a receive buffer per level and offers `forward`, `forward_all`,
`backward`, chunk iterators and a key → receive-position `HashMap`.

**Tests.** Five serial tests of `bucket_ghosts` and `LevelChunkSizes`; in
`mpi_regressions.rs`, `check_ghost_exchange` forwards and returns a value derived from
each key for every list ghost, with uniform and per-level chunk sizes.

**Keep.** Validation agreed on all ranks before any communicator is built (`new`, the
`all_reduce_into` of `local_valid`); skipping local and `Global` keys; sorted,
deduplicated ghost buckets; one neighbourhood collective per level, entered by every
rank; the forward/backward check of `mpi_regressions.rs` as a pattern.

**Replace.**
- A fixed chunk size per level, because source chunks vary per leaf (requirement 4).
- The key → position `HashMap` and the chunk iterators by key, because exchange buffers
  must map to index positions without lookups (requirements 2 and 7).

**Property worth keeping explicitly.** `bucket_ghosts` sorts by key, and rlst sorts the
ghosts stably by owner. Every non-`Global` key lies in its owner's Morton range
(`Octree::coarse_tree_bounds`), so owners increase with the key, and the receive order
is the Morton order of the ghosts. Section 8 relies on this.

### 1.4 `fmm/operator.rs`

**What it does.** `FmmOperator` has `type Value: Equivalence + Copy + Default`, sizes
(`multipole_size(level)`, `local_size(level)`, `source_size()`, `target_size()`), and
eight methods on one pair of boxes given by their keys, all `&self`, all accumulating.

**Keep.** The accumulation rule (every operator adds); keys as the geometry input; one
`Value` type; sizes per level for coefficients.

**Replace.**
- One pair per call, because the GEMM path of Phase 4 needs a whole (level, offset) or
  (level, octant) batch (requirement 6).
- `&self`, because operators need scratch (requirement 6; T8 and T10).
- One size per leaf, because counts vary (requirement 4).
- The missing target input: L2P, M2P and P2P see only the output slice, so target
  positions have nowhere to live (requirement 4).

### 1.5 `fmm/evaluator.rs`

**What it does.** `LevelData<T>` stores a fixed chunk per key, one buffer per level,
with a `HashMap<MortonKey, usize>` of positions. `FmmEvaluator::new` collects the keys
per level, the local M2M pairs and the `Global` keys (sorted), builds two
`FmmGhostCommunicator`s (sources for U and X, multipoles for V and W), and allocates
four `LevelData`. `evaluate` clears and runs `exchange_sources`, `upward_local`,
`upward_global`, `exchange_multipoles`, `downward`, `evaluate_leaves`.

**Tests.** Four serial tests of `LevelData`; the whole evaluator through
`run_index_fmm` in every `mpi_regressions.rs` scenario.

**Keep.** The pass order (compute graph steps 1–6 of `src/fmm.rs`); the global upward
pass (coarse-tree multipoles gathered to all ranks, `Global` keys recomputed
redundantly on every rank); which list feeds which operator; public stage methods;
"every rank enters every exchange, every level".

**Replace.**
- `HashMap` positions and per-pair `HashMap` lookups in every loop
  (`self.lists.v_list()[&key]`, `self.multipoles.get(source)`, two per chunk in
  `exchange`) (requirement 2).
- Fixed source and target chunks (requirement 4).

**Defects and fragile spots.**
- *Multipole columns follow `HashMap` order.* `LevelData::new(all_keys.keys(), …)` for
  the multipoles assigns positions in the iteration order of `all_keys`, which uses
  `std`'s randomly seeded hasher. The column order therefore changes from run to run
  and from rank to rank. Results do not depend on it today, because every access goes
  through the key. A GEMM plan built on these positions would. Locals are sorted,
  sources are the leaves followed by the received ghosts, and targets are the leaves,
  so only the multipoles are affected (laplace-fmm-plan §5.3 says "partly").
- *The root multipole is formed only on more than one rank.* On one rank the root is
  `Global` and the only coarse block. `local_m2m` skips level-1 children because their
  parent is not `LocalInterior`, and `upward_global` starts at level 1. On several ranks
  the root is formed from its children. No list names the root, so nothing observes
  the difference. The new global pass forms it on every rank count.
- *Redundant collectives.* `global_max_level` (an all-reduce) runs three times in `new`
  (directly and in each `FmmGhostCommunicator::new`). `upward_global` gathers the
  static coarse-block keys again on every evaluation: two `gather_to_all`, each two
  validity all-reduces, an all-gather of counts and an all-gather of data.
- *`IndexFmm` scales as boxes × leaves.* Its coefficient and target sizes are the
  global number of leaves. At 10⁴ leaves a level buffer holds about 10⁸ `u32`, so one
  evaluator needs over 1 GB (see Section 11.4).

### 1.6 `fmm/index_fmm.rs` and `fmm.rs`

`IndexFmm` propagates global leaf indices (`global_leaf_indices`: Morton order across
ranks), and `check_targets` requires a count of one at every index on every leaf.
`run_index_fmm` agrees the outcome with an all-reduce. `fmm.rs` documents the compute
graph and lists the "Future extensions" this redesign delivers.

**Keep.** The operator semantics (P2M, P2L, P2P add one per source point at the
source's leaf index; the others add their input), the global leaf numbering, the
collective pass/fail agreement, the compute-graph documentation.
**Replace.** The check for counts of one by the check for variable counts (Section
11.2).

### 1.7 `tests/mpi_regressions.rs` and the example

One MPI-owning test runs 11 scenarios: unequal and identical populations, refinement
caps, large leaves, duplicates, single points, no sources, no targets, uneven ranks, a
graded corner blob (leaf level 6 against coarse remote neighbours) and empty input
ranks. Each checks tree balance, all lists against the oracle, a forward/backward
exchange and the index FMM. The example runs the index FMM on 200 random points per
rank. **Keep all of it**; the new checks are added to the same `cases` loop (Section
11).

## 2. Requirements

| # | Requirement (short) | How the design meets it | Change? |
| --- | --- | --- | --- |
| 1 | Kernel-agnostic; `IndexFmm` exact | No arithmetic outside `IndexFmm`; `Value` is any `Equivalence + Copy + Default + Send + Sync` (§5.3, §6) | no |
| 2 | Morton-ordered dense index per level, deterministic, no `HashMap` on hot paths | `BoxIndex` (§3): sorted keys per level; all views carry indices | no |
| 3 | One buffer per level and kind, box i at i · size | `LevelBuffers` (§5.1); one allocation per kind | no; ghost boxes get a (zero) local slot, see §5.1 |
| 4 | Variable-size leaf data, three CSR stores | `LeafStore` (§5.2): sources (local + ghost), target input and output (local) | no |
| 5 | Index lists: U/W/X CSR, V per (level, offset), M2M/L2L per (level, octant), target-centric views | §4 | refinement: the P2P list is U plus the leaf itself (§4.2) |
| 6 | One call per level and kind, both views, keys, `&mut self`, per-pair adapter | §6 | refinement: M2M is called once per level and **pass** (local, then global) |
| 7 | Same pass order, collectives, global levels; every rank in every collective; flat exchange buffers per level in index order | §7, §8 | refinement: source data use one exchange for all levels (leaf data are not level data), contiguous in leaf order |
| 8 | Bit-identical for fixed tree and ranks; fixed accumulation order | §7.4: every row has a fixed order, every stage a fixed place | no |
| 9 | Redistribution designed | §9 | no |
| 10 | Tests carried over and extended; old evaluator as reference | §11 | note on `IndexFmm` memory (§11.4) |

The three refinements are sign-off question 1 (Section 12). None weakens a
requirement; each states how a requirement applies where its wording leaves room.

## 3. Box index

### 3.1 Numbering

`BoxIndex` numbers, on every level l in 0..nlevels, every key of `Octree::all_keys` on
that level: `LocalLeaf`, `LocalInterior`, `Global`, `GhostLeaf` and `GhostInterior`.
Box i of level l is the i-th smallest key of the level. nlevels is
`global_max_level() + 1`, the same on every rank (one all-reduce, as today).

```rust
/// Every box this rank holds, numbered 0..len(level) in Morton order on each level.
pub struct BoxIndex {
    keys: Vec<Vec<MortonKey>>,  // [level][box], ascending
    kinds: Vec<Vec<KeyType>>,   // [level][box], from Octree::all_keys
    first_child: Vec<Vec<u32>>, // [level][box]: index of child 0 on level + 1, or NONE
    box_leaf: Vec<Vec<u32>>,    // [level][box]: leaf index (§3.3), or NONE
    leaves: LeafNumbering,
}
```

Build: one pass over `all_keys` into per-level vectors, then a sort per level. Cost
O(K log K) for K held keys, deterministic for any `HashMap` order. Memory: 8 (key)
+ 16 (`KeyType`) + 4 + 4 = 32 bytes per box.

**Why not the Morton index itself.** The box index *is* Morton order, with the positions
a rank does not hold left out. On a uniform tree on one rank it equals the interleaved
Morton index of the level. The raw index cannot be the storage slot:
- Level l has 8^l positions (2^48 at level 16), of which an adaptive, distributed tree
  holds a small, scattered subset.
- A rank's local boxes form one Morton range, but its ghosts and `Global` boxes lie
  outside it, so no offset makes the raw index dense.

A map from key to slot is therefore needed in any case. Today it is a `HashMap` consulted
on every pair. Here it is computed once, at build time, and stored in the views as dense
`u32` slots. That makes each level a dense size × n matrix (requirement 3), keeps hot
paths free of lookups (requirement 2), and halves the size of index arrays against `u64`
keys. The keys stay available in O(1) from the index (§3.2). The cost is that an index,
unlike a key, changes when the tree is rebuilt (§3.4).

`first_child` exists because the eight children of a box whose children are all held
are eight consecutive indices on the next level, in octant order: no other key of that
level lies between them in Morton order. That holds for every non-ghost interior box
and every interior neighbour of one (ghost-children layer), but not for the ghost
interiors that the layer adds; those get `NONE`.

### 3.2 Lookups

| Lookup | Cost | Where allowed |
| --- | --- | --- |
| index → key: `keys(level)[i]`, `leaf_key(j)` | O(1) | everywhere, including operators (geometry) |
| index → kind, leaf, first child | O(1) | everywhere |
| key → index: `find(key) -> Option<(level, u32)>` | O(log n), binary search | build, loading points, reading results, tests |
| key → leaf: `find_leaf(key)`; finest-level key → local leaf: `local_leaf_containing(fine)` | O(L log n) | loading points, redistribution (§9), tests |

No `HashMap` is part of the index. Hot paths take every index from a view (Section 4),
so they never search.

### 3.3 Leaves

Leaves have their own numbering, because leaf data are not level data:
- local leaves first, as indices 0..n_local, ordered by (level, key);
- then the ghost leaves that some U or X row names, as n_local..n_leaves, ordered by
  key.

| Choice | Reason |
| --- | --- |
| local leaves by (level, key) | the local leaves of level l are one range `local_leaves(l)`, so a per-level leaf call (L2P, M2P, P2P) receives one contiguous slice of each target store, which `split_at_mut` divides by CSR offsets (§6.5) |
| ghost leaves by key | key order is owner order (§1.3), so the receive buffer of the source exchange *is* the ghost tail of the source store: received with no copy (§8.1) |
| only ghost leaves that a list names | the same ghost set as today's source exchange; other ghost leaves have a box index but no leaf index |
| locals before ghosts | a near or X row sorted by leaf index lists local sources before ghost sources, which leaves room for overlap (§10) |

`LeafNumbering` holds `leaf_key`, `leaf_level` and `leaf_box` per leaf (13 bytes), the
per-level local ranges and `box_leaf` per box (in §3.1). On a uniform tree (level-major
= Morton) the local leaves are in the order of `Octree::leaf_keys`.

### 3.4 Rebuilding the tree

The index is a pure function of `all_keys` and nlevels. A new tree means a new
`Plan`, new stores and new exchanges; there is no incremental update. A box keeps its
key across rebuilds but not its index, which depends on every other box held. `Plan`
owns its arrays and holds no borrow of the octree, so an application can drop the old
tree and plan independently and map results across rebuilds by key.

### 3.5 Coarse blocks

The global upward pass (Section 7.3) writes the multipoles of every rank's coarse
blocks into this rank's level buffers. With the ghost-children layer, `nd-octree`
advertises every coarse block to every rank (`generate_all_keys`, private module
`octree/src/octree/implementation.rs`: "it advertises every local coarse block to all
ranks"), so every block, and hence every child of every `Global` box, has a box index
on every rank. The plan checks this at construction. The public docs of
`OctreeOptions::with_ghost_children` do not state it; Section 12, question 6 proposes
that they do.

### 3.6 Construction and validation

```rust
impl Plan {
    /// Build the box index and the interaction lists of `octree`.
    ///
    /// # Collective operation
    /// One all-reduce for the global level count and one for the validity check, in that
    /// order, on every rank. The lists themselves are built without communication.
    ///
    /// # Errors
    /// Returned on every rank if, on any rank, the tree lacks the ghost-children layer,
    /// a list entry or a child of a `Global` box is not held, or a level has 2^32 boxes
    /// or more.
    pub fn new<C: CommunicatorCollectives>(octree: &Octree<'_, C>) -> Result<Self, PlanError>;

    /// The same, from a key classification; local, for tests on hand-built maps.
    pub fn from_key_types(all_keys: &HashMap<MortonKey, KeyType>, nlevels: usize,
                          own_coarse_blocks: &[MortonKey]) -> Result<Self, PlanError>;
}
```

## 4. Interaction lists

### 4.1 Rules: reused, not rewritten

The per-key rule of `from_key_types` is moved, unchanged, into a private function
`key_lists(key, key_type, all_keys) -> [Vec<MortonKey>; 4]`. The old map builder and the
new plan both call it, so the old tests guard the move. The plan translates each entry
with `BoxIndex::find`, at build time only. The brute-force oracles of
`interaction_manager_tests.rs` and `mpi_regressions.rs` stay the specification and are
applied to the index form through the key ↔ index maps.

A rewrite in index arithmetic (neighbours of the parent, then `first_child` + 0..8) would
avoid the O(log n) translation per entry. It is a later optimisation, to be taken only
if T4's measured build time calls for it.

### 4.2 What each list holds

Facts from 2:1 balance that the views use (and T4 asserts):

| List of box t on level l | Entries | Level of entries | Stored as |
| --- | --- | --- | --- |
| U (t a leaf) | adjacent leaves | l − 1, l or l + 1 | leaf indices |
| V | children of the parent's neighbours, not adjacent to t | l | box indices on l, with offset index |
| W (t a leaf) | children of interior neighbours, not adjacent to t | l + 1 | box indices on l + 1 |
| X | leaf neighbours of the parent, not adjacent to t | l − 1 | leaf indices |

The P2P row of a local leaf t is U(t) ∪ {t}, the "near list". Holding the self pair in
the row lets the batched P2P and the per-pair adapter treat it like any other pair; the
operator recognises it by source = target.

### 4.3 Views per level

Every view of level l has one row per **target**. Rows of a level-buffer view cover all
boxes of the level, ghosts included with empty rows, so `chunks_mut(size)` of the
level's output buffer lines up with them. Rows of a leaf view cover the local leaves of
the level, in leaf order, so the CSR offsets of the target stores line up with them.

| View (level l = target level) | Rows | Row entries, in this order | Grouping |
| --- | --- | --- | --- |
| `v` | boxes of l | (source box on l, offset index), by offset index | per offset index d: (targets, sources), by target |
| `x` | boxes of l | source leaf, by leaf index | – |
| `m2m_local` | boxes of l (parents) | (child box on l + 1, octant), by octant; rows only for `LocalInterior` | per octant o: (parents, children), by parent |
| `m2m_global` | boxes of l (parents) | the same; rows only for `Global` | per octant o |
| `l2l` | boxes of l (children) | (parent box on l − 1, octant), at most one; rows for non-ghost boxes, l ≥ 1 | per octant o: (children, parents), by child |
| `p2m` | boxes of l | the local leaf of the box, if any | – |
| `near` | local leaves of l | source leaf, by leaf index, self included | – |
| `w` | local leaves of l | source box on l + 1, by box index | – |
| `l2p` | local leaves of l | the box of the leaf | – |

The offset index is the position of d = index(target) − index(source) in
`V_LIST_DIRECTIONS` (CONVENTIONS §3.12), stored as `u16`; the octant is
`morton::child_index(child)`, stored as `u8`.

```rust
/// The V-list pairs of one level, by target and by offset.
pub struct VList { /* row_offsets: Vec<u32>, sources: Vec<u32>, offsets: Vec<u16>,
                      batch_offsets: [u32; 317], batch_targets: Vec<u32>, batch_sources: Vec<u32> */ }

impl VList {
    /// The sources of target box `t` and their offset indices, by offset index.
    pub fn row(&self, t: usize) -> (&[u32], &[u16]);
    /// The pairs of offset index `d`: targets ascending, each at most once.
    pub fn batch(&self, d: usize) -> (&[u32], &[u32]);
    /// The raw arrays, for upload to a device.
    pub fn row_offsets(&self) -> &[u32];
    pub fn sources(&self) -> &[u32];
    pub fn offset_indices(&self) -> &[u16];
}
```

`Children` (M2M) and `Parents` (L2L) have the same shape with octants instead of
offsets; `Csr` (X, near, W) has `row(t) -> &[u32]` and the raw arrays.

### 4.4 Invariants

- **At most once per offset batch.** For fixed d the source is t − d, a function of the
  target, so each target appears at most once in batch (l, d), and so does each source.
  The 316 batches of a level partition its V pairs.
- **Octant batches.** A parent has one child per octant, so it appears at most once in
  batch (l, o) of `m2m_*`; each child appears in exactly one batch, its own octant. In
  `l2l` each child appears once in total, and each parent at most once per batch. The
  eight batches of a level partition its parent–child pairs.
- **Both views hold the same pairs, in the same per-target order.** A V row is ordered
  by offset index, and an M2M row by octant, which is the order in which an operator
  that walks the batches 0, 1, 2, … meets that target. An operator that executes the
  groupings in index order and one that walks the target-centric rows therefore add
  each target's contributions in the same order. L2L rows have one entry, and X, near
  and W have no grouping.
- U, V, W, X equal today's lists mapped through the index (T4 tests this on every
  serial tree), and the oracle on every scenario.

### 4.5 Memory

Per target box on a uniform tree, with `u32` indices: V has up to 189 entries at 6 bytes
in the rows and 8 bytes in the groupings, about 2.6 kB. Everything else (row offsets of
five box views, M2M and L2L entries, 27 near entries per leaf) adds about 0.25 kB. For
comparison:
- today's lists take about 1.9 kB per box (eight-byte keys, four map entries);
- the coefficients of one box take 1.3 kB at p = 8 and 5.8 kB at p = 18 (f64,
  multipole and local).

The duplicate V grouping is the price of requirement 5. If memory becomes the limit, a
plan option can build the groupings only for operators that use them (Phase 4
decision).

## 5. Data stores

### 5.1 Level buffers

```rust
/// One contiguous buffer per kind; box i of level l at offset(l) + i * size(l).
pub struct LevelBuffers<T> { data: Vec<T>, offsets: Vec<usize>, sizes: Vec<usize> }

impl<T: Copy + Default> LevelBuffers<T> {
    pub fn level(&self, l: usize) -> LevelSlice<'_, T>;
    pub fn level_mut(&mut self, l: usize) -> LevelSliceMut<'_, T>;
    /// Level `coarse` mutably and level `coarse + 1` immutably (M2M), or the reverse (L2L).
    pub fn parent_child_mut(&mut self, coarse: usize) -> (LevelSliceMut<'_, T>, LevelSlice<'_, T>);
    pub fn child_parent_mut(&mut self, coarse: usize) -> (LevelSliceMut<'_, T>, LevelSlice<'_, T>);
    pub fn as_slice(&self) -> &[T];  // the whole buffer, for one device upload
    pub fn clear(&mut self);
}
```

- Sizes per level come from `multipole_size(level)` and `local_size(level)`, so they may
  vary with the level. They must be positive.
- Multipoles and locals both have a slot for every box of the level, as requirement 3
  states. Ghost locals are never written. The cost is (p + 1)² values per ghost box,
  none on one rank. Numbering locals over non-ghost boxes only would save it, at the
  price of a second index space; not recommended.
- One allocation per kind, levels in order, so `split_at_mut` gives two levels at once
  and a device upload is one copy.

### 5.2 Leaf stores

```rust
/// Per-leaf data: leaf j holds counts[j] points of `point_size` values, CSR.
pub struct LeafStore<T> { data: Vec<T>, point_offsets: Vec<usize>, point_size: usize }

impl<T: Copy + Default> LeafStore<T> {
    pub fn new(counts: &[usize], point_size: usize) -> Self;
    pub fn count(&self, leaf: usize) -> usize;
    pub fn chunk(&self, leaf: usize) -> &[T];
    pub fn chunk_mut(&mut self, leaf: usize) -> &mut [T];
    /// Leaves `range` as one slice with their offsets, for a level call.
    pub fn range(&self, range: Range<usize>) -> LeafSlice<'_, T>;
    pub fn range_mut(&mut self, range: Range<usize>) -> LeafSliceMut<'_, T>;
    pub fn clear(&mut self);
}
```

| Store | Leaves | Counts from | Values per point (Laplace, §3.13) | Exchanged |
| --- | --- | --- | --- | --- |
| sources | local, then ghost (§3.3) | caller for local leaves; the owner's count for ghosts, delivered by the exchange build (§8.1) | 4 (u, then q) | ghost tail, forward |
| target input | local | caller | 3 (u) | never |
| target output | local | same counts as target input | 1 or 4 | never |

- Zero counts are allowed everywhere; such a leaf has an empty chunk, and operators
  treat it as a no-op.
- Sources and targets are separate populations with separate counts per leaf.
- The caller fills local sources and target input through `Evaluator::sources_mut(leaf)`
  and `target_input_mut(leaf)` (or whole-range slices), and reads
  `target_output(leaf)`. The layout inside a chunk is the operator's (CONVENTIONS
  §3.13 for Laplace).
- New counts mean a new `Evaluator` on the same `Plan`; the exchange layout depends on
  them.

### 5.3 One `Value` type

Recommended: **one** `Value` type for coefficients, source data, target input and target
output, as today.

| | One `Value` | Separate types |
| --- | --- | --- |
| Laplace (f32 or f64) | fits: §3.13 stores leaf-scaled positions in [−1, 1]³, so f32 positions lose nothing to the domain offset | no benefit |
| Mixed precision (f64 positions, f32 coefficients) | not expressible | expressible, but §3.13 removes the reason for it |
| `IndexFmm` (u32) | fits | fits |
| Exchange, stores, views | one generic parameter, one MPI datatype per call | a parameter per store, four datatypes, four sets of view types |

A later split (C6.4 dipoles need no new type; a kernel-independent FMM might want one)
changes the trait while the crate is at 0.1.0-dev and touches only the store and batch
types; the index and lists are type-free.

## 6. Operator interface

### 6.1 Traits

```rust
/// The data sizes of an FMM, shared by the batched and the per-pair interface.
pub trait FmmSizes {
    /// The scalar type of every buffer.
    type Value: Equivalence + Copy + Default + Send + Sync;
    /// Values of a multipole expansion on `level`; positive.
    fn multipole_size(&self, level: usize) -> usize;
    /// Values of a local expansion on `level`; positive.
    fn local_size(&self, level: usize) -> usize;
    /// Values per source point.
    fn source_point_size(&self) -> usize;
    /// Values per target point in the target input; may be zero.
    fn target_input_point_size(&self) -> usize;
    /// Values per target point in the target output.
    fn target_output_point_size(&self) -> usize;
}

/// Level-batched operators.
///
/// The evaluator calls each method once per level (M2M once per level and pass), in the
/// order of design §7. Every method adds into its output. It may execute the targets of a
/// call in any order and on any number of threads, provided it adds the contributions of
/// each target in the order of that target's row in the target-centric view. Operators
/// document the order they use.
pub trait FmmOperator: FmmSizes {
    fn p2m(&mut self, batch: P2m<'_, Self::Value>);
    fn m2m(&mut self, batch: M2m<'_, Self::Value>);
    fn m2l(&mut self, batch: M2l<'_, Self::Value>);
    fn p2l(&mut self, batch: P2l<'_, Self::Value>);
    fn l2l(&mut self, batch: L2l<'_, Self::Value>);
    fn l2p(&mut self, batch: L2p<'_, Self::Value>);
    fn m2p(&mut self, batch: M2p<'_, Self::Value>);
    fn p2p(&mut self, batch: P2p<'_, Self::Value>);
}
```

### 6.2 What each call receives

Every batch carries `level` (the target level), `index: &BoxIndex` (keys of every box and
leaf, for geometry), its views, its inputs (shared) and its output (exclusive).

| Call (target level l) | Views | Inputs | Output |
| --- | --- | --- | --- |
| `p2m` | `p2m` (box → local leaf) | sources (all leaves) | multipoles of l |
| `m2m` | `m2m_local` or `m2m_global` (rows and octant batches) | multipoles of l + 1 | multipoles of l |
| `m2l` | `v` (rows and offset batches) | multipoles of l | locals of l |
| `p2l` | `x` | sources | locals of l |
| `l2l` | `l2l` (rows and octant batches) | locals of l − 1 | locals of l |
| `l2p` | `l2p` (leaf → box) | locals of l; target input of the level's leaves | target output of the level's leaves |
| `m2p` | `w` | multipoles of l + 1; target input | target output |
| `p2p` | `near` | sources; target input | target output |

For example:

```rust
/// M2L on one level: every V-list pair whose target lies on `level`.
pub struct M2l<'a, T> {
    pub level: usize,
    pub index: &'a BoxIndex,
    /// By target (offset order) and by offset index.
    pub pairs: &'a VList,
    /// Multipoles of the level, box i at i * size.
    pub multipoles: LevelSlice<'a, T>,
    /// Locals of the level; the call adds into them.
    pub locals: LevelSliceMut<'a, T>,
}

/// P2P for the local leaves of one level.
pub struct P2p<'a, T> {
    pub level: usize,
    pub index: &'a BoxIndex,
    /// Row r is the near list (U and self) of local leaf `local_leaves(level).start + r`.
    pub near: &'a Csr,
    pub sources: LeafSlice<'a, T>,        // every leaf, by leaf index
    pub target_input: LeafSlice<'a, T>,   // the level's local leaves, by row
    pub target_output: LeafSliceMut<'a, T>,
}
```

`LevelSlice`/`LevelSliceMut` wrap a slice with its chunk size (`chunk(i)`,
`chunks_mut()`, `as_mut_slice()`); `LeafSlice`/`LeafSliceMut` wrap a slice with its
point offsets and point size (`chunk(j)`, `count(j)`, `split_mut()` into per-row
slices).

No finer calls (per octant, per offset) are added. A level call already holds every
grouping, so a device operator can issue one launch per batch, or one grouped launch,
from inside it (§6.4).

### 6.3 `&mut self`, and the per-pair adapter

`&mut self` lets an operator own its scratch (T8: `Workspace`, table scratch, the P2P
mapping buffer; T10: one set per pool thread). The evaluator holds the operator by value
next to its stores, so it can pass `&mut self.operator` together with borrows of
`self.multipoles` and `self.locals` (disjoint fields).

```rust
/// One method per pair. `PerPair<P>` turns it into an `FmmOperator`.
pub trait PairOperator: FmmSizes {
    fn p2m(&mut self, leaf: MortonKey, sources: &[Self::Value], multipole: &mut [Self::Value]);
    fn m2m(&mut self, child: MortonKey, parent: MortonKey, octant: usize,
           child_multipole: &[Self::Value], parent_multipole: &mut [Self::Value]);
    fn m2l(&mut self, source: MortonKey, target: MortonKey, offset_index: usize,
           source_multipole: &[Self::Value], target_local: &mut [Self::Value]);
    fn p2l(&mut self, source: MortonKey, target: MortonKey,
           sources: &[Self::Value], target_local: &mut [Self::Value]);
    fn l2l(&mut self, parent: MortonKey, child: MortonKey, octant: usize,
           parent_local: &[Self::Value], child_local: &mut [Self::Value]);
    fn l2p(&mut self, leaf: MortonKey, local: &[Self::Value],
           target_input: &[Self::Value], target_output: &mut [Self::Value]);
    fn m2p(&mut self, source: MortonKey, target: MortonKey, source_multipole: &[Self::Value],
           target_input: &[Self::Value], target_output: &mut [Self::Value]);
    fn p2p(&mut self, source: MortonKey, target: MortonKey, sources: &[Self::Value],
           target_input: &[Self::Value], target_output: &mut [Self::Value]);
}

/// Runs a `PairOperator` serially, target by target, each target's row in order.
pub struct PerPair<P>(pub P);
impl<P: PairOperator> FmmSizes for PerPair<P> { /* delegates */ }
impl<P: PairOperator> FmmOperator for PerPair<P> { /* walks the target-centric views */ }
```

A wrapper, not a blanket `impl<P: PairOperator> FmmOperator for P`: with a blanket impl,
a type could not implement both traits, and T8's `LaplaceOperator` needs both to compare
its batched path with its per-pair path. `IndexFmm` implements `PairOperator`; T6 adds a
batched test operator that implements `FmmOperator` directly.

### 6.4 GEMM check (Phase 4)

**M2L, one (level, offset) batch.** With `(targets, sources) = batch.pairs.batch(d)` and
k = `targets.len()`:

```text
X[:, j]           = M_l[:, sources[j]]        gather, (p+1)² × k, from batch.multipoles
Y                 = A_d · X                   one GEMM with table d (index in V_LIST_DIRECTIONS)
L_l[:, targets[j]] += Y[:, j]                 scatter-add into batch.locals
```

The targets of a batch are distinct, so the scatter-add has no write conflict and needs
no atomics (laplace-fmm-plan §5.3, §6.6). Batches of different offsets write the same
targets, so they run in sequence on one stream, or in one grouped launch with offset as
a grid dimension (§6.5). Run in index order, the batches add each target's
contributions in its row order, as on the host. The "stacked" variant of §6.5
(Z = [A_0; …; A_315] · M_l, then a reduction per target) uses the rows instead: entry
(s, d) of row t says which block and column of Z to add. Everything comes from the
batch: index arrays, level buffers, the offset index for the table. No key and no
`HashMap` is involved; keys are needed only by geometry-dependent operators.

**M2M, one (level, octant) batch.** With `(parents, children) = batch.children.batch(o)`:
gather `M_{l+1}[:, children[j]]`, multiply by M2M(o), scatter-add into
`M_l[:, parents[j]]`. Parents are distinct within the batch. L2L mirrors it: gather
`L_{l−1}[:, parents[j]]`, multiply by L2L(o), scatter-add into `L_l[:, children[j]]`.

**P2P over a level.** One work item (a cube on a device) per row r: it owns
`target_output` chunk r, reads `target_input` chunk r, and walks `near.row(r)`, reading
each source chunk by leaf index from `sources` (tiled through shared memory on a
device). No two work items write the same chunk. The CSR arrays and the point offsets
are plain `u32`/`usize` arrays, uploaded once per plan and evaluator.

### 6.5 Host-parallel check (T10)

M2L on level l, serially and with rayon:

```rust
fn m2l(&mut self, b: M2l<'_, T>) {
    // One target: its row, in order, with one thread's scratch set.
    let body = |t: usize, local: &mut [T], scratch: &mut Scratch<T>| {
        let (sources, offsets) = b.pairs.row(t);
        for (&s, &d) in sources.iter().zip(offsets) {
            kernel_m2l(&self.tables, d as usize, b.multipoles.chunk(s as usize), local, scratch);
        }
    };
    // serial:
    //   for (t, local) in b.locals.chunks_mut().enumerate() { body(t, local, &mut serial_scratch) }
    // threaded, inside the operator's own pool:
    //   b.locals.as_mut_slice().par_chunks_mut(size).enumerate()
    //       .for_each(|(t, local)| body(t, local, &mut this_threads_scratch()));
}
```

- M2L, P2L and L2L split the locals of l, and M2M the multipoles of l, with
  `par_chunks_mut(size)`. The rows cover every box of the level (§4.3), so chunk t is
  row t. Inputs are other levels or other stores, borrowed shared.
- P2M splits the multipoles of l the same way and reads the box's leaf from `p2m`.
- L2P, M2P and P2P split `target_output` into per-row slices by the CSR offsets: a
  recursive `split_at_mut` under `rayon::join` (no allocation), or one `Vec<&mut [T]>`
  per level call. Both are safe Rust.
- Each target's body walks its row in order, whatever thread runs it, so the output is
  the same for every thread count. Ghost rows are empty.
- Scratch per thread is the operator's business (`&mut self`; T10 uses one set per
  pool thread). Batch inputs are `Sync` because `Value: Sync` and the views are plain
  arrays.

None of this needs `unsafe`, atomics or a plan change. The serial loop is the same body
under `chunks_mut().enumerate()`.

## 7. Evaluator

### 7.1 API

```rust
pub struct Evaluator<'p, C: CommunicatorCollectives, Op: FmmOperator> { /* plan, comm, operator,
    multipoles, locals: LevelBuffers; sources, target_input, target_output: LeafStore;
    source_exchange, multipole_exchange, coarse_exchange */ }

impl<'p, C: CommunicatorCollectives, Op: FmmOperator> Evaluator<'p, C, Op> {
    /// Collective. `source_counts` and `target_counts` have one entry per local leaf, in
    /// leaf order (§3.3). Validates sizes and counts on every rank and agrees the result
    /// before building any exchange.
    pub fn new(plan: &'p Plan, comm: &'p C, operator: Op,
               source_counts: &[usize], target_counts: &[usize]) -> Result<Self, EvaluatorError>;

    pub fn plan(&self) -> &'p Plan;
    pub fn operator(&self) -> &Op;
    pub fn operator_mut(&mut self) -> &mut Op;
    pub fn sources_mut(&mut self, leaf: usize) -> &mut [Op::Value];        // local leaves
    pub fn target_input_mut(&mut self, leaf: usize) -> &mut [Op::Value];
    pub fn target_output(&self, leaf: usize) -> &[Op::Value];
    pub fn multipoles(&self) -> &LevelBuffers<Op::Value>;
    pub fn locals(&self) -> &LevelBuffers<Op::Value>;

    /// Zero multipoles, locals and target output.
    pub fn reset(&mut self);
    /// `reset`, then every stage in order. Collective.
    pub fn evaluate(&mut self);

    pub fn exchange_sources(&mut self);     // 1, collective
    pub fn upward_local(&mut self);         // 2
    pub fn upward_global(&mut self);        // 3, collective
    pub fn exchange_multipoles(&mut self);  // 4, collective
    pub fn downward(&mut self);             // 5
    pub fn evaluate_leaves(&mut self);      // 6
}
```

The stages are public so a caller can time each one. Called out of order they give
wrong results, but every rank still enters the same collectives as long as all ranks
call the same stages; debug builds check the order.

### 7.2 Pass order

Unchanged from `src/fmm.rs`; L is the deepest level (nlevels − 1).

| Step | Stage | Calls, in order |
| --- | --- | --- |
| 1 | `exchange_sources` | one neighbourhood exchange (§8.1) |
| 2 | `upward_local` | for l = L down to 0: `p2m(l)`, then `m2m(l − 1)` with `m2m_local` (l ≥ 1) |
| 3 | `upward_global` | coarse gather (§8.3); for l = L − 1 down to 0: `m2m(l)` with `m2m_global` |
| 4 | `exchange_multipoles` | one neighbourhood exchange per level, l = 0..=L (§8.2) |
| 5 | `downward` | for l = 1..=L: `l2l(l)`, `m2l(l)`, `p2l(l)` |
| 6 | `evaluate_leaves` | for l = 0..=L: `l2p(l)`, `m2p(l)`, `p2p(l)` |

Every call is made on every rank for every level, also when its view is empty; the
operator returns at once.

### 7.3 Collectives

| Where | Collectives, in order | Ranks with no points |
| --- | --- | --- |
| `Plan::new` | all-reduce (nlevels), all-reduce (validity) | take part |
| `Evaluator::new` | all-reduce (validity); source exchange build (rlst: three validity all-reduces, an all-to-all, two graph creates, two neighbour all-to-alls); one multipole exchange build per level, the same each; coarse-key gather (`gather_to_all`) | take part with empty slices |
| `exchange_sources` | one neighbour all-to-all | send and receive nothing, still call |
| `upward_global` | one all-gather-v of coarse multipoles, with counts fixed at construction | contribute their blocks (every rank owns at least one) |
| `exchange_multipoles` | one neighbour all-to-all per level, in level order | still call on every level |

No collective sits in a branch that only some ranks take: the level loops run over the
global nlevels, and validation is agreed before any communicator is built. Compared with
today, `upward_global` drops the per-evaluation key gather, and `Evaluator::new`
computes the global level count once (in `Plan::new`) instead of three times.

### 7.4 Global coarse levels

As today: every rank gathers the multipoles of all coarse blocks and computes the
multipoles of the `Global` boxes itself. The difference is where the gathered values go.
Every block has a box index on every rank (§3.5), so the gather writes each remote
block's multipole into its slot of the level buffer, and the `Global` parents are then
formed by the ordinary `m2m` call with the `m2m_global` view, deepest level first. The
downward pass forms the locals of `Global` boxes on every rank from identical inputs in
identical order, so they are bit-identical across ranks.

### 7.5 Accumulation order (requirement 8)

| Value | Contributions, in this order |
| --- | --- |
| multipole of a local leaf | `p2m` |
| multipole of a `LocalInterior` box | `m2m` (local pass), children by octant |
| multipole of a `Global` box | `m2m` (global pass), children by octant |
| multipole of a ghost box | overwritten by the coarse gather or the exchange |
| local of a non-ghost box on level ≥ 1 | `l2l` from the parent; then V by offset index; then X by leaf index |
| target output of a local leaf | `l2p`; then W by box index; then the near list (U and self) by leaf index |

Every contribution adds into the value; `reset` zeroes them first. With a fixed tree,
ranks and counts, this fixes every floating-point sum, so two evaluations are
bit-identical. Operators that follow §6.1 keep this for any number of threads.

Compared with today: V was in source-key order, now offset order; the self P2P was
last, now in leaf-index order within the near list. `IndexFmm` is exact in either
order.

## 8. Ghost exchange

`FmmGhostCommunicator` is **replaced** by three small types on rlst's
`GhostCommunicator<MortonKey>`. Its validation pattern, its skipping of local and
`Global` keys and its sorted ghost buckets are kept.

### 8.1 Sources: one exchange, variable chunks, no receive copy

- Ghosts: the ghost leaves of the leaf numbering (those named by a U or X row), in leaf
  order, each with its owner.
- Chunk sizes: `ChunkSizes::PerIndex(|key| count(key) * source_point_size)`, evaluated
  by the owner, with zero allowed. The owner finds the count with `BoxIndex::find_leaf`,
  at build time. The build delivers the sizes to the receivers (`receive_chunk_sizes`),
  which gives the ghost counts and hence the offsets of the source store's ghost tail.
- Receive: the receive order is the ghost-leaf order (§1.3), and the receive layout is
  the tail's layout, so `forward_send_values(send, &mut tail)` writes straight into the
  store.
- Send: an owned send buffer, filled before each exchange by gathering `send_leaves`
  (the leaf index of every entry of `send_indices`; a leaf sent to two ranks appears
  twice).
- One exchange for all levels: leaf data are not level data, and one neighbourhood
  all-to-all replaces nlevels.

### 8.2 Multipoles: one exchange per level

- Ghosts: per level l, the ghost boxes named by a V row of level l or a W row of level
  l − 1, in box order; `ChunkSizes::Uniform(multipole_size(l))`.
- Flat receive buffer per level, chunks in box order; one scatter by `receive_boxes`
  (box index per chunk) into the level buffer. Flat send buffer, filled by gathering
  `send_boxes`. Both index arrays are built once.
- Ghost boxes are interleaved with local and `Global` boxes in Morton order, so the
  receive buffer cannot be the level buffer. The scatter costs one copy of the ghost
  data; the send side needs a gather anyway.
- Per level, as today, so a device can stage one level at a time and C5.2 can exchange a
  level as soon as it is complete.

### 8.3 Coarse gather

Built in `Evaluator::new`: gather every rank's coarse-block keys once (`gather_to_all`),
map each to its (level, box) on this rank, and fix the per-rank value counts. On each
evaluation: gather the own blocks' multipoles into a flat buffer in key order, one
all-gather, then scatter every remote block into its slot. Ranks own contiguous key
ranges, so the gathered buffer is in global key order.

## 9. Redistribution (designed here, implemented in C5.1)

Moves per-item payloads to the ranks that own their leaves, collectively, and results
back. In `nd-fmm-plan`, because it needs only the octree and the leaf numbering.

```rust
/// Routing of items (points) to the ranks that own their leaves, and back.
pub struct Redistribution { /* per sent item: destination; per received item: origin;
                               received items grouped by local leaf (CSR) */ }

impl Redistribution {
    /// Collective. `keys[i]` is the finest-level Morton key of local item i (for example
    /// from `points_to_morton` with the FMM's domain).
    pub fn new<C: CommunicatorCollectives>(octree: &Octree<'_, C>, plan: &Plan,
                                           keys: &[MortonKey]) -> Result<Self, RedistributionError>;
    /// Received items per local leaf, in leaf order: the counts for `Evaluator::new`.
    pub fn counts(&self) -> &[usize];
    /// Origin (rank, position) of every received item, grouped by leaf.
    pub fn origins(&self) -> &[(u32, u32)];
    /// Collective. `per_item` values of each local item to its owner; returned grouped
    /// by leaf, items within a leaf ordered by (origin rank, origin position).
    pub fn forward<T: Equivalence + Copy + Default>(&self, payload: &[T], per_item: usize) -> Vec<T>;
    /// Collective. `per_item` values of each received item back to its origin; returned
    /// in the original item order.
    pub fn backward<T: Equivalence + Copy + Default>(&self, results: &[T], per_item: usize) -> Vec<T>;
}
```

- Routing is local: `Octree::owner_rank(key)` costs O(log P) without communication. On
  the owner, `Octree::local_leaf(key)` and `BoxIndex::find_leaf` give the leaf index.
- `new`: one all-to-all of counts, one all-reduce that agrees whether every count fits
  MPI's `i32` (as `lookup_leaves` does), and one all-to-all-v of (key, origin position).
  `forward`: one all-to-all-v of payload. `backward`: one all-to-all-v with the counts
  reversed. Memory and traffic are linear in the items moved.
- `Octree::lookup_leaves` would add a query–reply round trip before the move. It is
  only worth it if the sender needs the leaf, for example to send leaf-scaled f32
  coordinates instead of f64 ones.
- The payload type is independent of `Value`. For Laplace the payload is the f64
  coordinates and charges, converted to leaf-scaled values on the owner (§3.13).
  Sources and targets are two `Redistribution`s.
- Determinism: for fixed inputs and ranks the received order is fixed. With a different
  distribution of the input over ranks, the order within a leaf changes, and so do the
  P2P sums in the last bits.
- Every rank calls `new`, `forward` and `backward` in the same order, with empty slices
  if it has no items.

## 10. Device and overlap compatibility

**Phase 4 (device-resident buffers, uploaded index arrays).**
- Every view is a set of flat `u32`/`u16`/`u8` arrays with raw accessors, uploaded once
  per plan. Store offsets are uploaded once per evaluator.
- Level buffers are one allocation per kind, leaf stores one per store, so each is one
  upload.
- No step of an evaluation looks up a key.
- The evaluator moves data itself in exactly four ways:
  - zeroing (`reset`);
  - a gather by an index array into a flat send buffer (both exchanges, the coarse
    gather);
  - a flat receive buffer copied or scattered by an index array;
  - a scatter of the gathered coarse blocks.

  Phase 4 can route these four through a device backend (gather and scatter kernels,
  staging copies) without changing the plan or the operator interface. Designing that
  hook is Phase 4's job.

**C5.2 (overlap of exchange with P2P and the local upward pass).**
- The stages are separate methods, and each exchange's buffers are owned per exchange,
  so splitting an exchange into start and finish is local to `exchange`.
- rlst 0.9.0 only offers the blocking `MPI_Neighbor_alltoallv`. A non-blocking version
  can use the graph communicator rlst exposes (`GhostCommunicator::forward_comm`) and
  its counts, or an rlst addition. That is C5.2's decision.
- Near and X rows list local sources before ghost sources (§3.3). So P2P and P2L can
  process the local part of every row before the source exchange completes and the
  ghost part after, keeping each row's order.
- Moving P2P before L2P (to overlap the multipole exchange) changes the stage order and
  hence the documented accumulation order (§7.5), though it stays deterministic. C5.2
  documents the new order.

## 11. Migration and tests

### 11.1 Tasks and modules

T4–T6 add the new modules under `nd_fmm_plan::v2`; T7 moves them to the crate root.

| Task | New modules (`v2::…`, final path after T7) | Old code |
| --- | --- | --- |
| T4 | `index` (`BoxIndex`, leaf numbering), `lists` (`VList`, `Children`, `Parents`, `Csr`, per-level views), `plan` (`Plan`, `PlanError`) | per-key rule moved to a private `key_lists`, called by `from_key_types` (no public change) |
| T5 | `store` (`LevelBuffers`, `LeafStore`, slice types), `exchange` (`SourceExchange`, `MultipoleExchange`, `CoarseExchange`) | unchanged |
| T6 | `operator` (`FmmSizes`, `FmmOperator`, `PairOperator`, `PerPair`, batch types), `evaluator` (`Evaluator`), `index_fmm` (`IndexFmm` on `PairOperator`, a batched test variant, the variable-count check) | unchanged; the reference |
| T7 | `v2` contents moved to `nd_fmm_plan::{index, lists, plan, store, exchange, operator, evaluator, index_fmm}` | removed: `fmm` (`FmmOperator`, `FmmEvaluator`, `LevelData`, `IndexFmm`, `run_index_fmm`), `ghost_communicator`, `InteractionManager`; kept: `interaction_manager::V_LIST_DIRECTIONS` and the private rule |
| C5.1 | `redistribute` (`Redistribution`) | – |

CONVENTIONS §3.12 defines the offset "as `InteractionManager::v_list_by_direction`
computes it". T7 removes that method, so the sentence needs a new reference (for
example the `v2` V-list view). That is a wording change, not a convention change;
question 8 asks for it.

### 11.2 Tests

**Carried over unchanged until T7:** all 22 unit tests, the 11 scenarios with their
oracle, exchange and index-FMM checks, and the example.

**Added by T4.**
- Serial, on the existing hand-built trees, each map built in several insertion
  orders:
  - numbering dense, Morton per level and independent of the insertion order;
  - index lists equal to the old lists mapped through the index;
  - every invariant of §4.4, and the entry levels of §4.2;
  - rows cover the targets once each, aligned with `chunks_mut(size)`.
- `mpi_regressions.rs`: the index lists of every non-ghost box against
  `oracle_lists`, and the same invariants, in every scenario.

**Added by T5.**
- Serial: leaf-store offsets, zero counts, disjoint chunks, `clear`; level-buffer
  layout; ghost bucketing with per-key sizes.
- `mpi_regressions.rs`:
  - a source exchange with counts `hash(key) % 5` (zeros included, so every rank can
    recompute an owner's count);
  - multipoles with per-level sizes, checked with today's `ghost_value`;
  - the coarse gather on every rank;
  - a new scenario with one leaf of 1,000 duplicate points.

**Added by T6.**
- Serial: the adapter on hand-made batches; the batched `IndexFmm` variant on chunks
  with several points per leaf.
- `mpi_regressions.rs`:
  - new evaluator against old, counts of one, equal on every leaf;
  - variable counts `hash(key) % 5`, through the adapter and the batched variant;
  - a recording operator that checks every batch's grouping and that every list pair is
    issued once;
  - target-centric walk against grouping walk;
  - two evaluations bit-identical.

**`IndexFmm` with variable counts.** Every source point of leaf j carries the global
index of leaf j. P2M, P2L and P2P add one at that index per source point; the other
operators add their input; target input has zero values per point, and target output
nleaves per point. The check: every target point of local leaf t holds, at index j, the
number of source points of leaf j. With counts of one this is today's check. A leaf
with zero sources contributes zero everywhere, so a missed interaction with it cannot be
seen. The docs say so, and the test counts are mostly nonzero.

**Removed by T7:** the old-against-new comparison and the tests of removed items
(`LevelData`, `bucket_ghosts`, `LevelChunkSizes`, old `IndexFmm` units). Every oracle
and `IndexFmm` check stays on the new API. The scenario set does not shrink: 11 today,
12 after T5.

### 11.3 Multi-rank runs

| Task | By hand, under an external timeout (macOS: loopback flags of root `CLAUDE.md`) |
| --- | --- |
| T4, T5 | `mpi_regressions` on 1, 2 and 4 ranks |
| T6 | `mpi_regressions` on 1, 2 and 4 ranks; `test_index_fmm` on 2, 3 and 4 ranks |
| T7 | `mpi_regressions` on 1, 2 and 4 ranks; `test_index_fmm` |

### 11.4 Risk: `IndexFmm` memory in T6's timing

T6 asks for stage timings of both evaluators with `IndexFmm` on a 10⁴-leaf tree. Each
evaluator then holds about 1.2·10⁴ boxes × 10⁴ `u32` for each of multipoles and locals,
plus 10⁴ × 10⁴ for target output: about 1.3 GB, about 2.6 GB for both. If that does not
fit, time a 3·10³-leaf tree and say so. The brief's figure is not wrong, only
expensive; it is left as it is.

## 12. Questions for sign-off

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | The requirements, and the refinements of Section 2 | Accept 1–10 as stated, with three refinements: (a) the P2P list is U plus the leaf itself; (b) M2M is called once per level and pass; (c) sources use one exchange for all levels, contiguous in leaf order, multipoles one per level |
| 2 | One `Value` type or several | One (§5.3) |
| 3 | Trait shape | Batched `FmmOperator` only, with the `PairOperator` trait and the `PerPair` wrapper (§6.3); no finer per-offset calls |
| 4 | List logic reused or rewritten | Reused: move the per-key rule into a private function shared by old and new code; translate keys to indices at build time (§4.1) |
| 5 | Move nd-fmm-plan to `[workspace.dependencies]`/`[workspace.package]`, fix licence form, `homepage`, `repository` | Yes, in T7 |
| 6 | Changes to nd-octree | None needed. Separately, document in `OctreeOptions::with_ghost_children` (public) that every coarse block is then held on every rank, which the global pass relies on, and add a test there. Phase 3 workaround: `Plan::new` checks it and fails on every rank if it ever breaks |
| 7 | Redistribution in Phase 3 instead of C5.1 | No: Phase 3 gates on one rank; the API (§9) keeps `Fmm`'s interface stable. If multi-rank Laplace is wanted early, it is a small task after T6 |
| 8 | CONVENTIONS §3.12 cites `InteractionManager::v_list_by_direction`, which T7 removes | Reword in T7 to cite the new V-list view; no change to the convention |

Further choices made in this design, listed so they can be overruled:
- local leaves ordered by (level, key), ghost leaves by key (§3.3);
- V rows in offset order, not source order (§4.4);
- a local slot for ghost boxes (§5.1);
- the root multipole formed on every rank count (§1.5).

### Recorded decisions

Signed off by Timo Betcke on 2026-10-01 (PR #25). Every recommendation above is accepted
as written:

| # | Decision | Carried out in |
| --- | --- | --- |
| 1 | Requirements 1–10 accepted, with refinements (a) P2P list = U plus the leaf itself, (b) M2M called once per level and pass, (c) one source exchange for all levels, multipole exchange per level | T4 (a, b), T5 (c), T6 (b) |
| 2 | One `Value` type | T5, T6 |
| 3 | Batched `FmmOperator` only, with `PairOperator` and the `PerPair` wrapper; no finer per-offset calls | T6 |
| 4 | Per-key list rule reused, moved into a private function shared by old and new code | T4 |
| 5 | nd-fmm-plan moves to `[workspace.dependencies]` and `[workspace.package]`; licence form, `homepage` and `repository` fixed | T7 |
| 6 | No nd-octree change inside the Phase 3 tasks; `Plan::new` checks that every coarse block is held. Also approved, as a separate decision: document in `OctreeOptions::with_ghost_children` (public docs) that with the layer every coarse block is held on every rank, and add an nd-octree test for it | T4 (check); a separate nd-octree PR (docs and test) |
| 7 | Redistribution stays in C5.1, with the API of §9 | C5.1 |
| 8 | CONVENTIONS §3.12's reference to `InteractionManager::v_list_by_direction` is reworded to the new V-list view; the convention is unchanged | T7 |

The further choices stand as designed:
- local leaves ordered by (level, key), ghost leaves by key (§3.3);
- V rows in offset order (§4.4);
- a local slot for ghost boxes (§5.1);
- the root multipole formed on every rank count (§1.5).
