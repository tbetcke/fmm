# CH4.0d1 Empty boxes left out of the lists

Started 2026-11-13 on `tbetcke/h-ch4-0d1`. Roadmap: "CH4.0d The Helmholtz leaf size" (H4;
priority list item 4), split on 2026-11-13 into CH4.0d1 (this task: the empty boxes), CH4.0d2
(V pairs between leaves of few points as near pairs) and CH4.0d3 (the leaf size and c_leaf
sweep, the default). Proposal: no §10 row (CH4.0d was added by the roadmap after the H6 gate);
§2 (constraints 1, 2, 9); §6.1–§6.3 (the builder); §9.1 (the workloads). Sources: CH4.0c's
report §3.2 (`fmm-validate/results/helmholtz-ch4-0c.md`), the design the user agreed to on
#214 ("Yes. Please.", 2026-11-13; the roadmap's CH4.0d text).

## Goal

An opt-in rule of `nd-fmm-plan` that leaves out of the lists the work of boxes whose subtree
holds no target or no source point, on by default under `HelmholtzFmmBuilder`, with the output
the rule off's bit for bit (−0 and +0 equal) and the gain measured per case.

## Acceptance (the roadmap's *Done when* for this part; no proposal §10 row)

From CH4.0d's *Done when*: "The empty-box rule on under the Helmholtz builder with its gain per
case", and from the design: "Only exact zeros go: the output is the rule off's bit for bit (−0
and +0 equal), on P ranks the one-rank FMM's, and the traffic counters show the multipoles no
longer sent." With constraint 1: `Plan::new` and Laplace unchanged.

## What the record shows

- nd-octree's tree is complete (every interior box has eight children) and 2:1 balanced; the
  Helmholtz leaf rule (`min_leaf_level`, c_leaf = 1) forces every leaf holding a point down to
  the level whose boxes are at most λ wide, so the complete tree has empty leaves where the
  points are sparse at that level.
- The plan's lists come from the keys alone (`interaction_manager::key_lists`), so an empty box
  has every row a full one has: its V row runs M2L into a local no point reads, and the V rows of
  its neighbours run M2L from its zero multipole.
- The under-resolved W3 (N = 10⁶, kD = 500; CH4.0c §3.2): 2.06 million leaves, 61 % empty, 0.62
  million of one point; about 38 % of level 7's V pairs have a source holding a point, 14 % a
  source and a target; level 7's M2L is 1,056 of 1,232 s under Rotation at 10⁻⁸ (247 of 340 s
  under the default). W1 at kD = 500 (points on a sphere, every leaf forced to level 7) is
  likely the same in kind (not counted).
- The near rule (Phase 6 T12, `nd_fmm_plan::near`) is the pattern: an operator-supplied input
  (`LeafCounts`, fetched by `LeafCounts::exchange` from the owners as a ghost communicator's
  chunk sizes), applied locally in `plan::first_pass`, `Plan::new` unchanged bit for bit.

## The rule

Counts by key. For every box b the rank has rows for, s(b) and t(b), the source and target
points of b's subtree.

- **No target (t(b) = 0)**: b's V, X and W rows are emptied, and the L2L row into b. Its local is
  then zero and no point reads it; its L2P (zero targets) and near row stay.
- **No source (s(b) = 0)**: b leaves every V and W row that names it, and the M2M row of its
  parent (local, or global on the parent's owner). Its multipole is zero (P2M of no point, M2M of zeros). Its P2M row (no point), the
  X and U rows naming it (P2L and P2P of no source) stay.
- **Every box**, `Global` boxes included (amended 2026-11-13, below): the counts of a top-tree
  key (a coarse block or an ancestor of one) are the sums of every rank's blocks below it, so
  the rule leaves out the same entries on every rank count, and the global M2M rows, the
  reduction and the replicated top follow the rows. The design of #214 exempted the top-tree
  keys; which keys those are depends on the rank count (on one rank the root alone), so the
  P-rank plan would keep zero terms the one-rank plan drops, and the one-rank comparison would
  hold for values only.
- Every row keeps its order; rows are only shortened, never reordered, so every remaining sum is
  the rule off's sum without terms that are exact zeros. A sum s + (±0) is s, except −0 + (+0),
  so the output equals the rule off's with −0 and +0 equal (compared as values, not bits).
- The exchanges follow the rows: the multipole exchange requests what V and W rows name
  (`multipole_ghosts`), so the empty boxes' multipoles are not sent; the source exchange follows
  the U and X rows, so a ghost leaf named only by an X entry of a box without targets is no
  longer sent.
- The counts: s(b) and t(b) of every local box from the local leaves' counts summed up the tree
  (every descendant of a local box is local); of every top-tree key from the coarse blocks'
  counts, all-gathered once; s(b) of every other ghost box a V or W row names, fetched from its
  owner (its `ghost_rank`, which holds b as a local box) in the same ghost communicator as the
  near rule's W-row leaves, so a build with both rules makes one exchange.

## As built (2026-11-13)

- The top-tree keys are not exempt (the user, 2026-11-13, asked when the one-rank comparison
  showed the exemption rank-dependent: "Global counts (Recommended)"). `LeafCounts::
  exchange_with_boxes` all-gathers (block, s, t) of every rank's coarse blocks
  (`exchange::gather_all`) and sums them into every ancestor; the `Global` M2M rows drop the
  children without sources as the local ones do.
- `LeafCounts`' setters are `set_box_sources` and `set_box_targets` (a ghost has a source
  count only). `ghost_source_counts` is now a length check around `exchange::ghost_key_counts`,
  which takes the owners' answer as a closure: Laplace's `LeafCounts::exchange` sends the same
  keys and counts as before.
- An M2M view's octant batches are no longer complete under the rule (a parent without one of
  its children has no entry in that octant's batch). No driver reads them for M2M (the drivers
  walk rows); `mpi_regressions`' recorder asserted completeness and now asserts that batch o
  holds the parents whose rows have an octant-o child, which is the old check without the rule.
- `HelmholtzFmm` has no build timings; the harness's build time includes the counts and the
  second plan. The harness reports what the rule left out and the multipoles read from other
  ranks (`Row::{left_out, remote_multipoles}`), not empty leaves per level; V pairs per level
  with the rule on and off are in `--levels`.
- `check_plane_wave` in `tests/helmholtz_fmm.rs` now names the rule off: with the rule the
  clusters' finest level has fewer V sources than the waves buffer's floor holds, so the floor
  no longer chunked it and the test's "the floor bound chunks a level call" failed. Its tree
  is CH6.2a's again; `check_empty_boxes` runs PlaneWave with the rule.
- No M6 count run: the Kathleen launches print the same table ("Empty boxes").
- Under the default (Auto with the term) the rule changed a level's M2L on both test cases:
  the term reads the rule's V pairs. The test compares values there only where every level's
  M2L agrees, and the four named settings always.

## Read first

- `fmm-plan/src/near.rs`, `plan.rs` (`build`, `from_key_types_with`, `first_pass`),
  `exchange.rs` (`ghost_source_counts`, `owned_chunk_size`, `multipole_ghosts`), `index.rs`.
- `fmm-plan/tests/mpi_regressions.rs` (`check_near_rule`, `near_oracle`, `check_plan`,
  `check_against_one_rank`); `src/near_tests.rs`.
- `fmm-exec/src/helmholtz/fmm.rs` (`build`, step 4; the term's shapes), `src/fmm.rs`
  (`near_plan`), `src/cost.rs` (`leaf_point_counts`).
- `fmm-exec/tests/helmholtz_fmm.rs` (the scenarios and the one-rank check).

## Changes

`nd-fmm-plan` (interface; the stop-and-ask of `/next-task` step 3):

- `near::LeafCounts` gains box counts: `set_box_counts(key, sources, targets)`,
  `box_sources(key)`, `box_targets(key)`; and `LeafCounts::exchange_with_boxes(plan, comm,
  sources, targets)`: `exchange`'s counts plus every local box's subtree counts and the subtree
  source counts of the non-top ghost boxes the V and W rows name, in one ghost communicator
  (`ghost_source_counts` generalised from local leaves to local boxes; `owned_chunk_size` reads a
  box's subtree count). `LeafCounts::exchange` keeps its keys and traffic (Laplace's `Fmm` calls
  it).
- `plan::PlanRules<'a> { near: Option<&'a NearRule<'a>>, empty_boxes: Option<&'a LeafCounts> }`
  (`Default`: neither), `Plan::with_rules(octree, &PlanRules)` and
  `Plan::from_key_types_with_rules`; `new`, `with_near_rule`, `from_key_types` and
  `from_key_types_with_near_rule` call them. A missing count is `PlanError::MissingCount`.
- `Plan::empty_boxes() -> EmptyBoxes` (per rank: the rows emptied for no target, by view; the
  entries left out for no source, by view), like `near_pairs`.
- `first_pass` takes the counts; with `None` every row is built as now.
- `fmm-plan/CLAUDE.md`: the code map and "The interaction lists" (the rule).

`nd-fmm-exec` (the Helmholtz builder only):

- `HelmholtzFmmBuilder::empty_boxes(bool)` (default `true`), step 4: the leaf counts where the
  points are (`cost::leaf_point_counts`, used, not changed; the keys of `pipeline::octree` are
  the sources' then the targets'), `LeafCounts::exchange_with_boxes`, `Plan::with_rules`.
  `HelmholtzFmm::empty_boxes()` reports the setting and the plan's `EmptyBoxes`; the build's
  timings include the counts and the second plan.
- The term's shapes (CH4.0) are counted on the plan the FMM runs, so with the rule on a level
  may count fewer V pairs and its method under `Auto` may change; the bit-for-bit checks below
  name the M2L. Reported per case in the measurements (which levels change method, if any).
- `fmm-exec/CLAUDE.md` (the Helmholtz FMM's bullet), the `helmholtz::fmm` module docs.

`nd-fmm-validate`: the harness prints the empty boxes per level and `EmptyBoxes`, and takes
`--empty-boxes on|off`; `tools/helmholtz/ch4-0d1-runs.sh`; the report
`fmm-validate/results/helmholtz-ch4-0d1.md`.

Conventions: none. Dependencies: none. `FmmBuilder` (Laplace): unchanged (CH4.0g decides).

## Tests that prove it

Serial (`fmm-plan/src/empty_tests.rs`, on `plan_tests.rs`'s key maps, no MPI):
- with counts that empty some boxes: exactly the rows and entries of the rule emptied, every
  other row equal to `from_key_types`' row entry for entry and in order (`Global` boxes
  included, since the amendment of 2026-11-13); `EmptyBoxes` the counts by brute force;
- counts with no zero give `from_key_types`' plan; a missing count is `MissingCount`;
- with the near rule as well, the near rule's conversions as without the empty-box rule on the
  boxes it keeps.

Multi-rank (`fmm-plan/tests/mpi_regressions.rs`, a `check_empty_boxes` in every scenario, at 1,
2 and 4 ranks; the scenarios with empty source or target populations and uneven ranks matter
here):
- the box counts the rule reads equal the brute-force subtree counts over the gathered tree
  (ghosts' from their owners);
- the plan passes `check_plan` against an `empty_oracle` (the oracle lists with the rule
  applied);
- the multipole exchange's receives are exactly the V- and W-row boxes of the rule's plan that
  other ranks compute, minus the reduction and the replicated top; its traffic not above the rule
  off's;
- `check_against_one_rank` with the one-rank plan of the same rule, bit for bit (the `Mixer`
  operator); the recorder sees no pair the rule left out.

The FMM (`fmm-exec/tests/helmholtz_fmm.rs`, `check_empty_boxes`, on the scenarios with empty
boxes, at 1, 2 and 4 ranks; trusted path: the same build with the rule off):
- rule on against off, equal as values (−0 = +0), under Rotation, Batched, PlaneWave and the
  default at a named M2L per level, at 1 and 2 threads;
- on P ranks the one-rank FMM with the rule, bit for bit;
- `EmptyBoxes` non-zero on a scenario built to have empty boxes (a sphere surface under the leaf
  rule), and the multipole traffic (`Traffic`) lower than the rule off's on several ranks;
- the default builder's `empty_boxes()` is on.

Laplace: every existing test, unchanged (`Plan::new` is `with_rules` with no rule).

## Checks

`/verify` (fmt, Clippy default, examples and workspace, docs, the workspace tests at opt-level
1, the MPI list at 2 and 4 ranks: `mpi_regressions`, `mpi_exec`, `invariance`,
`helmholtz_fmm`); `mpi_regressions` at 8 ranks by hand (fmm-plan/CLAUDE.md); the one-rank debug
time of `helmholtz_fmm` and `mpi_regressions` before and after.

## Machine runs

- The M6 (no timing): the empty boxes per level and `EmptyBoxes` on CH3.6's matrix (W1, W3 and
  the four W4 distributions at 10⁻³, 10⁻⁶, 10⁻⁸), W1 at kD = 500, W3 at kD = 100, 200 and 500.
- Kathleen, 1 node, one rank × 40 threads (`singlenode`): build and evaluation with the rule on
  and off on the cases with empty boxes (W3 N = 10⁶ at kD = 200 and 500, W1 at kD = 500, the W4
  cases where the M6 count is non-trivial), the default and Rotation at 10⁻⁸ and 10⁻⁶, the output
  compared on and off; 4 ranks × 10 threads on W3 at kD = 500 for the traffic.
  Decides: the gain per case for the sign-off of the default.

## Constraints and risks

- Constraint 1: `nd-fmm-plan` is on Laplace's path. `Plan::new` is `with_rules` with no rule and
  builds every row as before; Laplace's evaluation code is not touched. Shown by every Laplace
  test bit for bit and `plan_build_cost` (the plan's build time) on Kathleen before and after;
  no evaluation benchmark changes (question for the user, before the code).
- Constraint 2: the counts are by key, so every rank leaves out what the one-rank plan leaves
  out of its rows; checked bit for bit against one rank.
- Constraint 9: the rule reads the global tree's occupancy (points per box), not the charges or
  a rank's share; the decisions it changes (the term's shapes) are summed over the ranks.
- Risk: a zero that is not exact (a NaN or ∞ in a table times 0); none expected, the tests
  compare as values.
- Risk: the source-empty rule on a ghost needs its owner's subtree count; a ghost whose owner
  does not hold it as a local box would be a defect, reported as `ExchangeError::UnheldKey` on
  every rank.

## Pending runs

Kathleen, branch tree synced at 8631669 (`tools/kathleen/sync.sh`), outputs in
`/scratch/scratch/ucahtbe/fmm/logs/ch4-0d1/<stage>/`, one node each (`singlenode`, 6 h):

- job 633564: the release build, `tools/helmholtz/ch4-0d1-runs.sh … matrix`, then `traffic`;
- job 633565: `large`;
- job 633566: `w3`;
- job 633567: `plan_build_cost` of `main` (f32e5ba, `src/base-ch4-0d1`) and of the branch,
  alternately three times each on one node (constraint 1, the user's "Tests + build time").

## Open questions

Asked before the code (`/next-task` step 3: the plan changes `nd-fmm-plan`'s interface), all
answered on 2026-11-13 by the user:

1. The split: "d1/d2/d3 (Recommended)" (CH4.0d1 the empty boxes, CH4.0d2 the V pairs of few
   points, CH4.0d3 the sweep; the roadmap split in this task's first commit).
2. The `nd-fmm-plan` interface of "Changes": "Approve (Recommended)".
3. The W rows of a leaf with no target, beyond #214's V, X and L2L: "Also drop W
   (Recommended)".
4. Constraint 1's evidence: "Tests + build time (Recommended)": every Laplace test bit for bit,
   `Plan::new`'s views unchanged, `plan_build_cost` on Kathleen before and after; no D9
   evaluation runs.
