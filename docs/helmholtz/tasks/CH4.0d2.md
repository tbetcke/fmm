# CH4.0d2 V pairs between leaves of few points as near pairs

Started 2026-11-13 on `tbetcke/h-ch4-0d2`. Roadmap: "CH4.0d2 V pairs between leaves of few
points as near pairs" (H4; priority list item 4, after CH4.0d1 and CH4.0d1b). Proposal: no §10
row (CH4.0d was added by the roadmap after the H6 gate); §2 (constraints 1, 2, 9); §6.1–§6.3
(the builder); §9.1 (the workloads, the tolerance). Sources: CH4.0c's report §3.2, CH4.0d1's
and CH4.0d1b's reports and records, Phase 6 T12's near rule (`nd_fmm_plan::near`,
`docs/design/optimisation.md` §8a, `fmm-validate/results/phase6-near.md`).

## Goal

An opt-in V conversion of `nd-fmm-plan`'s near rule (a V pair between two leaves becomes a near
pair where the operator's predicate holds), offered by `HelmholtzFmmBuilder` with a crossover
from a cost model, measured, and proposed as a default or not.

## Acceptance (the roadmap's *Done when*; no proposal §10 row)

"By the near rule's mechanism (n_s n_t below a crossover of the cost model; a Rotation pair at
p = 24 costs about 110 µs of thread time on Kathleen, CH4.0c report §3.2): decided on the
target's rank from the owners' counts; a converted ghost leaf named in the leaf numbering, so
the source exchange delivers its points (a halo of up to three leaf boxes, not one); the near
rows in (level, key) order. *Done when:* the points sent and the multipoles no longer sent
counted; the errors within 1.1× the rule off's (the near rule's bound); the one-rank FMM of the
same rule bit for bit; rank-count invariance with the rule on; proposed or not on its measured
gain (*Sign-off*)." With constraint 1: `Plan::new`, `Plan::with_near_rule` with Laplace's
rule, and Laplace's `Fmm` unchanged.

## What the record shows

- A V pair (t, s) of a local leaf t with a leaf s of the same level adds M2L(s → t) to t's
  local. t is a leaf, so that local feeds only t's L2P; s's multipole also feeds its parent's
  M2M. A near pair instead adds P2P(s → t), exact. Only the pair's M2L goes; P2M of s and L2P
  of t stay (other pairs read them). Where every V pair of t converts, t's local still receives
  its parent's L2L.
- Thread time per Rotation pair at p = 24 on Kathleen: 1,056 s × 40 / 383 M ≈ 110 µs (CH4.0c
  §3.2); the default's waves on the same level 247 s, about 26 µs per pair, but much of the
  exponential M2L's cost is per box (the waves of each source and target box), not per pair, so
  removing one pair saves only its shift.
- The Helmholtz P2P (CH3.4b §5) costs 3.8–5.9× Laplace's per pair: about 2.3–3.5 ns of thread
  time in f64 with gradients on AVX-512 (Laplace 0.60 ns). A pair of leaves of n_s and n_t
  points: n_s n_t of those.
- So under Rotation at p = 24 the crossover is n_s n_t of order 3·10⁴ (every pair of leaves of
  64 points converts), under the waves or Batched far less. The crossover depends on the
  level's M2L, which Auto chooses after the plan from the V rows (CH4.0's term): circular if the
  rule reads the choice.
- The under-resolved W3 at kD = 500 has 0.62 million leaves of one point and 0.17 million more
  (CH4.0c §3.2): every V pair there is far below any crossover. Uniform trees at 64 points per
  leaf: n_s n_t ≈ 4,096 against crossovers of order 10³–10⁴.
- The near rule of Phase 6 T12 converts W and X pairs only; the Helmholtz builder does not use
  it at all (only the empty-box rule, CH4.0d1). `LeafCounts::exchange_with_boxes` (CH4.0d1b)
  already holds the source count of every held ghost leaf, every V entry's included, and the
  target count of every local leaf: the V rule needs no new exchange.

## Design (for approval: it adds to `nd-fmm-plan`'s interface)

### In `nd-fmm-plan` (kernel-agnostic, opt-in)

- `near::VPair { target, source, sources, targets }`: a V entry `source` (a leaf of the
  same level, local or ghost) of the local leaf `target`, with the source points of `source`
  and the target points of `target` (both from `LeafCounts`). The level is the keys'.
- `NearRule::with_v_pairs(self, predicate: &'a dyn Fn(VPair) -> bool) -> Self`: the rule also
  offers V pairs. `NearRule::new` is unchanged and offers none, so Laplace's rule, its
  predicate (an exhaustive match on `LeafPair`) and its plan are untouched: no new `LeafPair`
  variant.
- In `plan::first_pass`, after the empty-box rule has kept an entry: for a local leaf t with
  a V predicate, each V entry s that is a leaf is offered once; a converted s leaves t's V row
  and joins t's near row (sorted by (level, key) in `finish`, P1), and a ghost s is pushed to
  the ghost leaves, so the source exchange delivers its points. Interior V entries are never
  offered (their points lie in leaves no list names). A missing count is `MissingCount`.
- `NearPairs` gains `v_leaf_pairs` and `v_converted` (no struct literal of it outside the
  crate).
- The exchanges follow the rows as for W: a multipole no remaining row reads is not sent;
  the source exchange's halo grows from U's and X's leaves to the V leaves that converted
  (up to three leaf widths).

### In `nd-fmm-exec` (the Helmholtz builder only)

- `HelmholtzFmmBuilder::near_v(VNear)`: `VNear::Off`, `VNear::Crossover(n)` (n_s n_t < n on
  every level, for the measurements) and `VNear::CostModel` (per level, below).
- The cost model's crossover per level: N_V(l) = α_M2L(l) / α_P2P, with α_P2P the Helmholtz
  P2P's thread time per pair (planes and gradients as built) and α_M2L(l) the thread time a
  level saves per V pair removed, at the level's degree, for the **cheapest** M2L per pair
  that Auto may choose on the level (the exponential M2L where D8's conditions hold, else
  Batched where p_l is at most its crossover, else Rotation; a named M2L its own). Taking
  the cheapest makes the rule conservative: after the conversions Auto's term reads the pairs
  that remain, and if it then picks a dearer M2L, more pairs would have paid, none that
  converted loses. No circularity: the degrees need only the octree's level count, known
  before the plan.
- The constants per machine class (the ISA, as Auto's constants), from measurements of this
  task on Kathleen and locust: per (kind, degree) the level's M2L time saved per pair
  removed (from the level times with the named crossover against the rule off, CH6.2b's
  per-level timings), and the P2P per pair. Inputs that do not depend on the data
  (constraint 9): degrees, kinds, ISA, the counts by key; the same on every rank count.
- The rule is built with the empty-box rule in the one plan (`PlanRules { near,
  empty_boxes }` on `Plan::with_rules_on`), from the counts the builder already exchanges.
- Off by default in this PR's code until the sign-off; the PR proposes `CostModel` as the
  default or not, on the gain measured.
- As built (after open question 3 and the first rounds of runs): on a level whose M2L is the
  exponential one the rule decides per parent P and source s, not per pair (`v_group`:
  every child of P reading s converts if all are leaves and n_s n_P < n m_s), the builder
  predicting those levels before the plan and asserting them against the operator's; the
  cost model's constants are per machine class (`VCosts`: x86 from Kathleen, NEON from
  locust, by the ISA as Auto's), the crossovers agreed over the ranks (all-reduce min).
- W and X conversions stay off under Helmholtz (the predicate's `LeafPair` arms false): out of
  this task's scope; a follow-up line if the measurements show them worth it.

### In `nd-fmm-validate`

- `helmholtz_accuracy::Case::near_v` and `--near-v off|cost|<n>,…`; `Row` gains the V pairs
  offered and converted, the source points the source exchange receives (summed over ranks)
  and, as now, `remote_multipoles`. `tools/helmholtz/ch4-0d2-runs.sh` and
  `ch4-0d2-tables.py`.

## Tests that prove it

- `fmm-plan/src/near_tests.rs` (serial, the key maps of `plan_tests.rs`): exactly the V pairs
  the predicate names convert (leaf to leaf only; interior V entries never offered), each
  candidate offered once with both counts, the V rows lose them and the near rows gain them in
  (level, key) order, the ghost leaves are U's and X's plus the converted ghost W and V
  entries; a V predicate that converts nothing gives the rule's plan without V; with the
  empty-box rule, V conversions only on the entries it keeps; a missing count is an error.
- `fmm-plan/tests/mpi_regressions.rs`, `check_near_rule` extended (1, 2, 4, 8 ranks): a
  kernel-agnostic V predicate on every scenario; `check_plan` against the oracle with the V
  conversions applied; the source exchange delivers every converted ghost V leaf with its
  count; the multipole exchange follows the rows (no multipole sent that no row reads); the
  recorder sees each converted V pair as P2P once and no M2L for it; `check_against_one_rank`
  with the one-rank plan of the same rule bit for bit; over the scenarios some converted V
  entry is a ghost on several ranks.
- `fmm-exec/tests/helmholtz_fmm.rs` (MPI, 1, 2, 4 ranks): with `VNear::Crossover(n)` at a
  crossover that converts some and keeps some pairs, and at one that converts every V leaf
  pair: the output against the rule off's within 1.1× of its error against the direct sum
  (the near rule's bound; near pairs are exact), under Rotation and the default, 2 threads;
  the one-rank FMM of the same rule bit for bit; the converted pairs and points received
  reported.
- `fmm-exec/tests/invariance.rs` or `helmholtz_fmm.rs`: rank-count invariance with the rule on
  (P ranks against P/2 and 1, bit for bit).
- Laplace: every existing test unchanged; `Plan::with_near_rule` with a W/X rule bit for bit
  `main`'s (the existing near tests).

## Checks

`/verify` (fmt, Clippy default, examples, workspace; docs; the default and workspace tests at
opt-level 1; the MPI list at 2 and 4 ranks; `mpi_regressions` at 8), `cargo deny` not needed
(no manifest change).

## Machine runs

- Kathleen (1 node, `singlenode`, 1 × 40): the named crossover swept (n = 0, 10², 10³, 3·10³,
  10⁴, 3·10⁴, ∞) on the under-resolved W3 at kD = 500 (10⁻⁶, 10⁻⁸), W1 at kD = 500, W3 at
  N = 10⁶, kD = 100 and 200, W4's cube and Plummer sphere, at 10⁻³, 10⁻⁶, 10⁻⁸, Rotation
  and the default, with `--levels`: the per-level M2L times give α_M2L per (kind, degree), the
  P2P time α_P2P. Then the cost model on CH3.6's matrix and those cases, rule on against off
  (ABBA): build, evaluation, errors, pairs converted, points received, multipoles no longer
  read. And 4 × 10 for the traffic and rank-count invariance at 40 against 10 and 1 ranks.
- locust (1 × 72): the same constants and the cost model's gain on a subset (the other
  machine class: NEON).
- The M6: none unless the user gives the go-ahead for a timing run.
- What it decides: the cost model's constants, and whether `CostModel` is proposed as the
  default.

## Constraints and risks

- Constraint 1: `nd-fmm-plan` is on Laplace's path. The V rule is opt-in, a new builder method
  and struct; `NearRule::new`, `LeafPair`, `Plan::new` and `with_near_rule` unchanged;
  `NearPairs` gains two fields (zero without V). Laplace's plan bit for bit, `plan_build_cost`
  on Kathleen within `main`'s spread (one `if` per V entry of a local leaf when no V
  predicate: a field test).
- Constraint 2 (determinism) and 9: the rule decides each pair on the target's rank from counts
  by key and constants agreed over the ranks: every rank count makes the one-rank rows.
- Risk: the near field grows sharply where pairs convert (up to 189 more source boxes per
  leaf), with the P2P's memory traffic and the source exchange's points; the model's per-pair
  constant for the exponential M2L is a marginal cost and may not hold where a box loses every
  V pair (its waves then not needed, not modelled). The measurement decides.
- Risk: scope. Three crates (plan, exec, validate) as CH4.0d1. If the cost model's calibration
  needs a second round of runs, it is split off as CH4.0d2b (the default), with this PR the
  mechanism, the named crossover and the measurements.

## Pending runs

Kathleen, `singlenode` (6 h cap per job), one rank × 40 threads, tree e6ea63e, outputs in
`/scratch/scratch/ucahtbe/fmm/logs/ch4-0d2/<job>/`, `tools/helmholtz/ch4-0d2-runs.sh`:
- 633813 (a): `matrix`, W1 and W3; 633814 (b): `matrix`, W4's cube and sphere; 633815 (c):
  `matrix`, W4's Plummer sphere and clusters: the named crossovers swept, both orders;
- 633811 (d): `large` (W1 at kD = 500, W3 at kD = 100);
- 633812 (e): `w3` (the under-resolved W3 at kD = 500) and `traffic` (4 × 10 against 1 × 40);
- 633816: `plan_build_cost`, `main` (7f4466c, `src/main`) and the branch alternately three
  times (constraint 1).
They decide the cost model's constants (`ch4-0d2-tables.py levels`), then a second round runs
the cost model on the same cases (`custom`), which decides the default's proposal.

## Open questions

1. The interface (above): `VPair` and `NearRule::with_v_pairs`, opt-in, no new `LeafPair`
   variant. Approved by the user on 2026-11-13 before the code: "Approve (Recommended)".
2. The crossover: the cheapest M2L Auto may choose per level (conservative, no circularity),
   with a named crossover for the measurements; off by default until the sign-off. Approved by
   the user on 2026-11-13: "As planned (Recommended)".
3. Found in Kathleen job 633811 (W1 at kD = 500 under the default, a partial crossover: φ
   4·10⁵× the rule off's) and reproduced in `check_near_v` under PlaneWave (1.8·10⁻² against
   1.2·10⁻⁵): on a level with waves merge and shift serves each source of a parent's slots
   to every child in its geometric mask, so a pair converted while a sibling keeps its source
   is counted twice. Rotation, Batched, Dense and Classes follow the rows; the empty-box rule
   leaves out exact zeros only. Decided by the user on 2026-11-13: "Group per (parent,
   source) (Recommended)": on a level with waves the rule decides once per parent P and
   source s for every child of P reading s, converting if all are leaves and
   n_s n_P < n m_s (`v_group`), with the builder's levels with waves asserted to be the
   operator's. The matrix jobs 633813–633815 ran the per-pair rule: their wave levels'
   errors at partial crossovers are not usable, their times are (the work is the same).
