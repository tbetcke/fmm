# CH0.8a S8: workload census, first pass

Started 2026-10-28 on `tbetcke/h-ch0-8a`. Roadmap: "CH0.8a S8: workload census, first
pass" (H0). Proposal: §10 CH0.8; §3.5 (p), §3.6 (the bound, the rule, the regimes), §3.7
(leaves), §5.2–§5.4 (the operators counted), §6.6 (the replicated top), §6.7 (grid sizes,
batch sizes), §7 (the model this replaces), §9.1 (W1–W4), §12 D11 and D12.

## Goal

Build the trees of W1–W4 over the kD sweep from `nd-octree` and `nd-fmm-plan` with a
prototype maximum leaf width, count per level what the Helmholtz FMM would do on them
(boxes, lists, p, regime, memory, operations) from the formula p and the bound of §3.6,
and from that propose c_leaf (D11) and the workloads (D12). No FMM is run.

## Acceptance (proposal §10, verbatim)

> trees of W1–W4 over the kD sweep from `nd-octree` with a prototype maximum leaf width
> (c_leaf from 0.25 to 2) and the points-per-leaf rule. Per level: occupied boxes; U, V
> (by distance class), W and X list sizes; p_l (formula and S7); regime and per-class
> choice (S6's bound); the memory model of Section 7 (multipoles, locals, signatures,
> tables, replicated top per rank and per node) and an operation count per operator.
> Outputs: c_leaf (D11); confirmation that no leaf is high frequency; the numbers that
> replace Section 7's; batch sizes for S5 and S9. No FMM is run

Dependency column: "none for a first pass with the formula p; CH0.6 and CH0.7 for the final
pass". So this first pass uses the formula p of §3.5 (no S7) and the bound of §3.6 with
C_T = 2, θ = 1 (no S6); CH0.8b repeats it with S7's p and S6's constants.

Roadmap *Done when*: per level the occupied boxes, U, V (by distance class), W and X sizes,
p_l, regime and per-class choice, the memory model of §7 and the operation counts;
c_leaf proposed (D11); no high-frequency leaf; the workloads proposed (D12); batch sizes for
S5 and S9.

## Read first

- Proposal §3.5–§3.7, §5.2–§5.4, §6.6, §6.7 ("Grid sizes", "Plans per box"), §7, §9.1,
  §12 (D11, D12).
- `octree/src/octree.rs` (`OctreeOptions`, `Octree::new`, `points_to_morton`),
  `fmm-plan/src/plan.rs`, `lists.rs`, `index.rs`, `interaction_manager.rs` (the list
  rule; the lists hold empty boxes), `fmm-exec/src/fmm.rs` step 4 (how the Laplace FMM
  builds its tree: `points_to_morton` at level 16, `with_max_fine_keys`, ghost children).
- `fmm-validate/src/fmm_accuracy.rs` (`Distribution`, W4's four distributions).
- `spikes/radial/` (CH0.3): `ranges::degree` (the formula p), the h reference sets of
  `fixtures/reference.json`.

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- `spikes/census/` (new, package `nd-fmm-spike-census`), a workspace member outside
  `default-members` like CH0.1–CH0.3's spikes. Dependencies, all already in
  `[workspace.dependencies]` and the lock file: `nd-octree`, `nd-fmm-plan`, `mpi`,
  `nd-fmm-validate` (W4's generators; `morton::from_physical_point` replaces
  `points_to_morton`, so no `rlst`); dev-dependency `serde_json` (the radial fixture). No new
  dependency; the root `Cargo.toml` gains only the member entry, `Cargo.lock` the
  package entry.
  - `src/workloads.rs`: W1 (Fibonacci sphere of diameter D, N = ⌈7.96 (kD)²⌉, i.e.
    100 per λ²), W2 (W1 with the polar cap of half-angle 30° replaced by the same
    construction at ten times the density), W3 (uniform random in the cube, 1000 per λ³
    up to kD = 100; 10⁶ at kD = 200 and 500), W4 (the four `Distribution`s at N = 10⁵ and
    10⁶, the gates' generators with this spike's seed). Each in a root cube of side D, given explicitly, so kD is the root's side times k.
  - `src/tree.rs`: the prototype maximum leaf width. A serial refinement in Morton keys:
    a box is refined if it holds more than n distinct level-16 keys (nd-octree's rule) or
    if it holds a point and is wider than c_leaf λ. The leaves of that complete tree are
    handed to `Octree::new` as one level-16 key each (the first descendant) with
    `max_fine_keys = 1`, which reproduces the tree and lets nd-octree 2:1 balance it as it
    balances the Laplace trees; ghost children on, as `Fmm` builds them; one rank. With
    no width (c_leaf = ∞) this must equal `Octree::new` on the points with
    `max_fine_keys = n` (a test).
  - `src/census.rs`: per level, from `Plan::new`: boxes held and occupied (a point in
    the box), leaves (occupied, empty), points per occupied leaf; U pairs and point
    pairs (P2P), V pairs per distance class |t|²/w² (15 classes) and per reflection class
    (34), W and X entries and their point counts; every count both as the plan holds it
    (empty boxes included) and over occupied boxes only (the work an FMM that skips empty
    boxes does).
  - `src/model.rs`: the formula p of §3.5 at the circumradius (copied from
    `spikes/radial`, spikes do not depend on each other); max_j (2j+1)|h_j(x)| by the
    upward recurrence; the bound E(l, c) of §3.6 per (level, distance class); the regime
    rule (H if some class passes and the level's hybrid M2L count is below RCR's, else
    L); N_φ per level (host: the multiple of 4 ≥ 4P − 3 with prime factors ≤ 7; device:
    a power of two); the memory model of §7 (multipoles and locals at (p+1)² complex
    f64 per occupied box, real multipoles half; signature scratch (2P − 1) N_φ per source
    and target box of a diagonal pair; transfer functions per passing reflection class;
    the (S|R) tables per distance of the RCR classes; rotation block sets; the replicated
    top per rank, Phase 5S's cap of 4,096 boxes and §6.6's levels 2–4, and per node at 40
    ranks (Kathleen) and 72 (locust)); operation counts per operator in complex
    multiply-adds with §7's conventions (an RCR translation 2P³, P1³ + P2³ when
    rectangular; a diagonal pair (2P − 1) N_φ; a conversion 2P³ + 16P³ as a product,
    as in §6.7; P2M, L2P, M2P, P2L (p + 1)² per point and expansion; P2P per point pair).
  - `examples/census.rs`: the sweep; prints Markdown tables (per-level tables, sweep
    summaries, c_leaf study) to `spikes/census/results/` (committed; Markdown only, no
    CSV).
  - `tests/census.rs`: the tests below; one MPI-owning test executable.
  - `SPIKE_REPORT.md`: the report.

The sweep: W1 at kD = 10, 20, 50, 100, 200, 500, 1000; W2 to 500; W3 at 10, 20, 50, 100
(resolved) and 200, 500 (10⁶); W4 at kD = 1, 10, 100. ε = 10⁻⁴, 10⁻⁶, 10⁻⁸ for p and
regimes (D1 is open; §7 used 10⁻⁶). c_leaf = 0.25, 0.5, 0.75, 1, 1.5, 2 and none; n = 128
(Laplace's host default from p = 12), with n = 64 as a sensitivity row in the c_leaf study.

**D11's rule (proposal §12)**: the largest c_leaf at which P2P and leaf costs (P2P, P2M,
L2P, M2P, P2L) stay within 10 % of their best on W1 and W3. A P2P pair is weighted as
w complex multiply-adds, w = 4 (a Helmholtz pair's sincos and divide against one complex
multiply-add), with w = 2 and 8 as sensitivity; the far-field total is reported beside it.

## Tests that prove it

All in release, one rank, one thread (`tests/census.rs`, one MPI initialisation):

1. **The prototype tree is nd-octree's tree.** With no width limit, the leaf keys equal
   `Octree::new` on the points with `max_fine_keys = n`, bit for bit, on W1 (kD = 50),
   W2 (kD = 50), W3 (kD = 20) and the four W4 distributions (N = 10⁴).
2. **The width rule holds.** With c_leaf = 0.5 and 2: every occupied leaf is at most
   c_leaf λ wide; every occupied leaf with more than n distinct keys is at level 16; the
   tree refines the tree without the rule (every leaf of it is a leaf or an ancestor of
   leaves of the new one).
3. **The counts against brute force.** On small trees (W1 kD = 20 at n = 16, W2 kD = 20
   at n = 16 and c_leaf = 1, Clusters and Plummer N = 3·10³ at n = 8, the Plummer sphere at
   c_leaf = 0.5): boxes, occupied boxes and leaves per level from the points' ancestors;
   V pairs per distance and reflection class, the source and target masks, U, W and X
   entries, with and without empty boxes, recomputed from the list definitions of
   `interaction_manager` (adjacency of closed cubes, O(n²) over the boxes) and equal to
   the census; point pairs of P2P from the U lists equal to the brute-force count of
   pairs of points in adjacent or equal leaves.
4. **The offsets.** The 316 V offsets give 15 distance classes and 34 reflection classes
   (x → −x, y → −y, z → −z, x ↔ y).
5. **The formula p** reproduces §3.5's tables (kD = 500, ε = 10⁻⁶: 144, 82, 50, 33, 25,
   23, 22 on levels 2–8; level 2 against kD at three ε).
6. **The Hankel magnitudes** against `spikes/radial/fixtures/reference.json`'s h sets of
   real argument (unscaled by their s), relative 10⁻¹².
7. **The bound** reproduces §3.6: the "classes on RCR" of kD = 500, ε = 10⁻⁶ (level 2
   none; 3: 4; 4: 4, 5; 5: 4, 5, 6) and the "nearest class has E ≤ ε from" column
   (1.8 λ, 17 λ, 42 λ, to the stated digits).
8. **The cost and memory formulas** reproduce §7's worked numbers: rotation block sets
   (0.09 MiB at p = 20, 31 MiB at p = 144), an (S|R) table (about 3 MiB at p = 82),
   level 2 of W1 at kD = 500 from §7's own boxes and mean V list (6.6·10⁹, signature
   scratch 0.28 GiB), N_φ as §6.7's examples (580 → 588, 1048 → 1080, 204 → 216 at
   p = 50).

The census's own W1 numbers are compared with §7's (sampled sphere) in the report, not
asserted.

## Checks

`/verify` (a new spike member: format, Clippy default and workspace all-targets, examples,
`cargo doc`, default and workspace tests, licence check, no fftw-sys); the spike's own
`cargo test -p nd-fmm-spike-census --release -- --include-ignored`; `cargo clippy -p
nd-fmm-spike-census --all-targets -- -D warnings`.

## Machine runs

None: the census is counting on one rank of the M3 Max; the largest tree is W1 at
kD = 1000 (8.0 million points). Time and peak memory per tree are reported.

## Constraints and risks

- Constraint 1: no Laplace code changes; nd-octree and nd-fmm-plan are used through their
  public API. The prototype width rule lives in the spike (CH3.0 adds the real one to
  nd-octree, behind its own Laplace check).
- Constraint 9: every decision the census models (p_l, regime, per-class choice, N_φ) is a
  function of k, the level width and ε only.
- Risk: the leaf key trick (one key per leaf, `max_fine_keys = 1`) relies on nd-octree
  refining exactly the boxes with more than one key; test 1 checks the equivalence on the
  real rule, test 2 the width rule.
- Risk: memory at kD = 1000 (8 million keys, the plan with ghost children on one rank);
  if it does not fit, W1 at kD = 1000 is counted at reduced N with the levels above the
  leaves unchanged, and the report says so.
- The memory and operation models are §7's first-pass conventions, not measurements;
  CH0.8b replaces p and the constants, CH2/CH3/CH4 measure.

## Pending runs

None.

## Open questions

The sign-off questions of `spikes/census/SPIKE_REPORT.md` §9: D11 (c_leaf = 1 by a revised
rule; §12's rule is degenerate), D12, the H leaf levels at 10⁻⁴, and the proposal's
corrections.
