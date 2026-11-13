# CH4.0d1b The empty-box rule's build cost

Started 2026-11-13 on `tbetcke/h-ch4-0d1b`. Roadmap: "CH4.0d1b The empty-box rule's build
cost" (H4; priority list item 4, after CH4.0d1, before CH4.0d2; added at CH4.0d1, #216's
question 2, signed off 2026-11-13). Proposal: no §10 row (CH4.0d was added by the roadmap
after the H6 gate); §2 (constraints 1, 2, 9). Sources: CH4.0d1's report §0 and §5
(`fmm-validate/results/helmholtz-ch4-0d1.md`) and record (`docs/helmholtz/tasks/CH4.0d1.md`).

## Goal

Build the empty-box rule's plan once, without the first `Plan::new` it now needs for the
counts' exchange lists and the routes, so that the build with the rule costs no more than
the build without it.

## Acceptance (the roadmap's *Done when*; no proposal §10 row)

"The build with the rule within 1.1× the rule off's on W3 at N = 4·10⁶ and on the
under-resolved W3 at kD = 500 (one rank × 40 threads, Kathleen), the plans and outputs
CH4.0d1's bit for bit, Laplace unchanged."

## What the record shows

- `HelmholtzFmmBuilder::build`, step 4 (`fmm-exec/src/helmholtz/fmm.rs`): `Plan::new`, the
  routes (`Points::route`, two `Redistribution::new`), then with the rule
  `LeafCounts::exchange_with_boxes(&plan, …)` and `Plan::with_rules`: the plan is built
  twice. A plan is built serially at about 17 µs per box on Kathleen (the per-key rule
  `key_lists` and the index-form rows), so the build with the rule is 1.1–1.7× the rule off's
  (W3 at N = 4·10⁶: 24.5 s against 15.6 s; the under-resolved W3 at kD = 500: 139 s against
  93 s).
- What the first plan is used for:
  - the routes: `Redistribution::new` and `Points::route` read only `plan.index()`'s local
    leaves (keys, levels, `local_leaf_containing`), which the rule does not change (the local
    leaves are numbered by (level, key); the ghost tail comes after them);
  - the counts: `exchange_with_boxes` reads the index (the local boxes' subtree sums), the
    coarse blocks and the top-tree keys (both from the partition map), and the V and W rows'
    ghost boxes (which boxes to request from their owners). Only the last needs rows.
- Every held ghost box that is not a top-tree key lies below another rank's coarse block, so
  it is a key of that rank's tree and a local box there (`nd-octree`: H = N(S) ∪ C(N(S)),
  every key of it a key of the tree; a ghost's rank is the owner of its block). So every held
  ghost can be counted by its owner, and requesting all of them needs no rows.

## Design (to be approved: it changes `nd-fmm-plan`'s interface)

1. **The index without a plan.** `BoxIndex::of_octree(octree) -> Result<BoxIndex,
   PlanError>` (collective: the global level count, as `Plan::new`, and one all-reduce that
   agrees the outcome): the index `Plan::new` builds, before its ghost leaves are numbered
   (local leaves only). `from_key_types_with_rules` is split into the private
   `index_from_keys` (the key checks, unchanged in order) and the row building, so the two
   share the code.
2. **The plan on a given index.** `Plan::with_rules_on(octree, index, &PlanRules)`: the plan
   of `with_rules`, built on `index` (moved in) instead of a new one; an index of another
   octree (key count or a key not in `all_keys`) is a new `PlanError::IndexMismatch`.
   Collective as `with_rules` (the level count and the agreement). `with_rules` and `new` keep
   their code path: `index_from_keys` then the rows, as now.
3. **The counts without a plan.** `LeafCounts::exchange_with_boxes(octree, &index, sources,
   targets)` replaces `exchange_with_boxes(&plan, comm, sources, targets)`: the same counts
   for every local box and every top-tree key (the partition map's blocks and owners, as the
   plan took them), and the source count of **every held ghost box** that is not a top-tree
   key (and of every ghost leaf, for the near rule), not only those a V or W row names. Counts
   only (the chunk sizes of one ghost communicator, as now), so the traffic grows by the held
   ghosts no V or W row names, a count each. A superset of CH4.0d1's counts, so every plan
   reads the same values.
4. **The routes on the index.** `Redistribution::with_index(octree, &index, keys)`;
   `Redistribution::new(octree, &plan, keys)` calls it with `plan.index()` (unchanged for
   every caller). `PlanMismatch` names the index's leaves.
5. **`nd-fmm-exec`.** `pipeline::Points::route` takes the `BoxIndex` (crate-private; `Fmm`
   passes `plan.index()`, the same bits). `HelmholtzFmmBuilder::build`, step 4, with the
   rule: `BoxIndex::of_octree`, the routes on it, the counts, `Plan::with_rules_on`; without
   the rule `Plan::new` as now.

Alternatives, not proposed:
- **The rule applied to the first plan's rows** (a `Plan::apply_empty_boxes`): keeps
  `Plan::new` and the V/W-only requests, but shortening the rows afterwards means renumbering
  the ghost leaf tail (X entries of boxes without targets drop ghost leaves, so the near and X
  views in leaf indices change), rebuilding the grouped views' octant batches and offsets and
  the near rule's order; more code on the plan's internals, and the first plan's cost stays
  (only the second goes).
- **Keep `exchange_with_boxes(&plan, …)` beside a plan-free one.** Two count paths to keep
  equal; the old one has one caller (the Helmholtz builder). Replaced instead; the tests
  compare against brute-force counts and CH4.0d1's plan (below).

## Changes

`nd-fmm-plan` (interface; the stop-and-ask of `/next-task` step 3):
- `index.rs`: `BoxIndex::of_octree`; `plan.rs`: `index_from_keys` (private),
  `Plan::with_rules_on`, `PlanError::IndexMismatch`; `near.rs`: `exchange_with_boxes`'s new
  signature and ghost set; `redistribute.rs`: `Redistribution::with_index`.
- `fmm-plan/CLAUDE.md` (code map, "The interaction lists"), the `empty` module docs.

`nd-fmm-exec`: `pipeline.rs` (`Points::route` on the index), `fmm.rs` (the call),
`helmholtz/fmm.rs` (step 4), `tests/shadow/mod.rs` if needed; `fmm-exec/CLAUDE.md`'s
Helmholtz FMM bullet.

`nd-fmm-validate`: none expected (the harness prints the build time); a runs script
`tools/helmholtz/ch4-0d1b-runs.sh`; the report `fmm-validate/results/helmholtz-ch4-0d1b.md`.

Conventions: none. Dependencies: none.

## Tests that prove it

Serial (`fmm-plan/src/plan_tests.rs`, `empty_tests.rs`, no MPI):
- `with_rules_on`'s equivalent on `from_key_types`' maps: the plan on a prebuilt index equals
  `from_key_types_with_rules`' bit for bit (with and without each rule);
- an index of another key map is `IndexMismatch`.

Multi-rank (`fmm-plan/tests/mpi_regressions.rs`, every scenario, at 1, 2, 4 and 8 ranks):
- `BoxIndex::of_octree` equals `Plan::new`'s index on its keys, kinds and local leaves;
- the counts of the new exchange equal the brute-force subtree counts (`EmptyOracle`, widened
  to every held ghost box) for every held box, and are a superset of CH4.0d1's;
- **the plan bit for bit CH4.0d1's**: `Plan::with_rules_on` on the index with the new counts
  equals `Plan::with_rules` with the oracle's counts (`Plan: PartialEq`: index with its ghost
  leaves, every view, `EmptyBoxes`, `NearPairs`), and with the near rule as well;
- `Redistribution::with_index` on the index delivers what `Redistribution::new` on the plan
  does, bit for bit (the forward of the scenario's payloads);
- the rest of `check_empty_boxes` (exchanges, recorder, one-rank plan bit for bit) on the new
  plan.

The FMM (`fmm-exec/tests/helmholtz_fmm.rs`, `check_empty_boxes`, at 1, 2 and 4 ranks): the
checks of CH4.0d1 unchanged on the new build (the rule off's values under Rotation, Batched,
PlaneWave and Auto, 2 threads and the one-rank FMM bit for bit). The outputs CH4.0d1's bit for
bit: the plan is (above) and nothing else in the build changes; on Kathleen the checksums
equal job 633596's for the same cases.

Laplace: every existing test unchanged (`Plan::new`, `Redistribution::new` and `Fmm`'s routes
the same code on the same index).

## Checks

`/verify` (fmt, Clippy default, examples and workspace, docs, the workspace tests at opt-level
1, the MPI list at 2 and 4 ranks); `mpi_regressions` at 8 ranks by hand; `cargo deny` not
needed (no manifest change).

## Machine runs

Kathleen, `singlenode`, one rank × 40 threads, `tools/helmholtz/ch4-0d1b-runs.sh`:
- `build`: W3 at N = 4·10⁶, kD = 100 and the under-resolved W3 (N = 10⁶) at kD = 500, 10⁻⁶,
  the default, `--empty-boxes on,off` and `off,on` (ABBA), one evaluation: the build with the
  rule over the rule off's (*Done when*: at most 1.1), and the checksums against job 633596
  (and 633566 for the under-resolved W3);
- `large`: W1 at kD = 500 at 10⁻⁶ and 10⁻⁸, the default, ABBA: the sparse case's build and
  checksums;
- `traffic`: W3 (N = 10⁶) at kD = 500, 4 ranks × 10 threads: the build on several ranks and
  the counts' wider ghost set;
- `plan_build_cost` (Laplace), `main` and the branch alternately three times, as CH4.0d1's
  job 633567: `Plan::new` within the spread.
Decides: the *Done when*.

## Constraints and risks

- Constraint 1: `nd-fmm-plan` and the pipeline are on Laplace's path. `Plan::new` keeps its
  code path (the split of `from_key_types_with_rules` moves code, changes no order),
  `Redistribution::new` calls `with_index` with the plan's index, `Points::route` reads the
  same index. Shown by every Laplace test and `plan_build_cost`; no D9 evaluation runs (no
  evaluation code changes).
- Constraint 2: the counts are by key and of whole subtrees, as before; the plan equals
  CH4.0d1's (tested), so the one-rank comparison holds.
- Risk: a held ghost whose owner does not hold it as a local box would be
  `ExchangeError::UnheldKey` on every rank; by `nd-octree`'s H every non-top ghost is a key
  of its owner's tree. The multi-rank scenarios (empty and uneven ranks, the ghost-children
  layer) check it.
- Risk: the wider ghost request costs more than the second plan saves on many ranks (counts
  only, one ghost communicator either way); `traffic` measures it at 4 ranks.

## Pending runs

## Open questions

Asked before the code (`/next-task` step 3: the plan changes `nd-fmm-plan`'s interface),
answered on 2026-11-13 by the user:

1. The design above (the `nd-fmm-plan` interface): "Approve (Recommended)".
