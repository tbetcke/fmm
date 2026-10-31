# CH3.0a nd-octree's maximum leaf width

Started 2026-10-31 on `tbetcke/h-ch3-0a`. Roadmap: "CH3.0a nd-octree's maximum leaf width"
(H3; the first third of CH3.0, eighth on the priority list). Proposal: §10 CH3.0; §2
(constraint 1), §3.7.

## The split of CH3.0

CH3.0 has three independent parts: a tree option in `nd-octree`, the Helmholtz builder's
leaf rule on the reference workloads, and the complex densities. As one PR it would span
`nd-octree`, `nd-fmm-validate` and `nd-fmm-exec`, with a constraint-1 check on Laplace's
trees. Proposed split (roadmap), each one reviewable PR:

- **CH3.0a** (this): `nd-octree`'s maximum leaf width, as an option that defaults to none.
- **CH3.0b**: the Helmholtz builder's leaf rule. k's classification, the leaf level from
  (k, D, c_leaf), and the reference workloads W1–W4 in `nd-fmm-validate`, ported from
  `spikes/census`. No leaf wider than c_leaf λ on W1–W4.
- **CH3.0c**: complex densities. Complex charges in and split planes out (§3.15 "Inputs
  and outputs"), the leaf chunks fixed, one complex expansion per complex charge vector.

## Goal

An `OctreeOptions` setting that refines every box that holds a key and lies above a given
level. It defaults to none, so every existing tree is unchanged key for key.

## Acceptance (proposal §10, verbatim)

> no leaf wider than c_leaf λ on W1–W4; if `nd-octree` gains the maximum leaf width,
> Laplace's trees and outputs bit for bit unchanged (constraint 1); complex charges in
> and potentials and gradients out as split planes (§3.15 item 9), one complex expansion
> per complex charge vector

CH3.0a's share is the second clause. Its *Done when* (roadmap): with the option set,
every leaf holding a key lies at or below the given level (capped by `max_level`), on any
number of ranks, and the leaves equal the one-rank leaves and a brute-force refinement.
With the option unset, every tree and Laplace's outputs are bit for bit unchanged.

## Read first

- `octree/CLAUDE.md` (the construction contract, O1: blocks are nodes of the one-rank tree).
- `octree/src/octree.rs` (`OctreeOptions`, `Octree::new`'s docs);
  `octree/src/octree/implementation.rs` (`create_local_tree`, `refine_heavy_blocks`,
  `coarse_tree`, the local balance).
- `spikes/census/src/tree.rs` (the prototype rule `refine_box`, `width_level`).
- Proposal §3.7.

## Changes

Only `nd-octree`, plus docs (`octree/CLAUDE.md`). No manifest, dependency, convention or
`nd-fmm-plan` change. Nothing in `nd-fmm-exec` uses the option yet; CH3.0b will.

- `OctreeOptions::with_min_leaf_level(level: usize)` and `min_leaf_level()`. The name is
  for sign-off: in `nd-octree`'s key space the maximum leaf width w_max of the proposal is
  a level, the shallowest whose boxes are at most w_max wide. The default is 0, which adds
  no condition. The doc states the rule: every box that holds a key and lies above
  `min(min_leaf_level, max_level)` is refined, in addition to the `max_fine_keys` rule.
  Empty boxes stay as coarse as completeness and the 2:1 balance allow (proposal §3.7:
  "every box that contains points").
- `create_local_tree` splits a box when `current.len() > max_keys` or when `!current.is_empty()`
  and its level is below `min_leaf_level`. Everything else is unchanged (the coarse tree by
  weight, the cut, the balance).
  - The blocks stay nodes of the one-rank tree, because the one-rank tree splits a
    superset of the boxes it splits now. The leaves therefore stay the one-rank leaves on
    every rank count (O1).
  - With the default 0 the added condition is false for every box, so the trees are
    unchanged.
- `Octree::options()` reports the new field. The `Hash` and `Eq` derives still hold.

## Tests that prove it

- **Unit tests** (`octree/src/octree.rs` doctests and the serial tests in
  `implementation.rs`, which take no new MPI initialisation):
  - `create_local_tree` against a brute-force recursion (the census spike's `refine_box`)
    on seeded random keys and clustered keys, for min_leaf_level 0–6 and `max_fine_keys`
    1, 8 and 64, key for key;
  - the default 0 key for key equal to the call without the option;
  - min_leaf_level above `max_level` capped at `max_level`;
  - the builder methods and the `Default`.
- **Distributed** (`examples/test_mpi_weighted_partition.rs`, a new scenario in the
  existing executable): at min_leaf_level 4 and 6, on uniform and clustered keys, the
  leaves on P ranks equal the one-rank leaves and the brute force (gathered). Every leaf
  holding a key lies at or below the level, and the tree is complete, linear and 2:1
  balanced (`is_complete_linear_and_balanced`). This runs at 1–4 ranks in CI, and by hand
  at 1–8 ranks, the latter under a timeout.
  - As built: levels 0, 4 and 5 on the first 2,000 points of the cube and the Plummer
    sphere. The Gaussian clusters were tried first, but their leaves already lie below
    level 5, so the rule never acted on them. Every check of `check_tree` (all invariants,
    the one-rank leaves) runs from the share and from rank 0; level 0 reproduces the tree
    without the option. The rule takes the cube's 64 leaves to 4,012 at level 4 and
    15,282 at level 5, and the sphere's 232 to 1,184 and 3,872.
  - In the serial test of the coarse tree by weight
    (`test_coarse_tree_by_weight_keeps_the_one_rank_leaves`), three cases with the level
    set: the leaves from the blocks equal the one-rank leaves for every refinement factor.
- **Laplace unchanged:**
  - the existing `nd-octree`, `nd-fmm-plan` and `nd-fmm-exec` tests and the MPI list;
  - the full FMM outputs (CH2.0's cases, as built 36) of `main` and the branch compared byte for byte on the M6,
    with CH2.0's `bits.sh` recreated in the scratchpad.

## Checks

`/verify`, which includes `nd-octree`'s crate checks (Clippy with `strict`, `cargo doc` with
`-D warnings`, the examples built) and the MPI examples at 1–8 ranks.

## Machine runs

None. The change is in tree construction only. The evaluate path's code and the trees of
every existing caller are unchanged, which is checked bit for bit. The plan therefore
proposes no D9 timing runs on locust and Kathleen (question 2).

## Constraints and risks

- Constraint 1 binds (`nd-octree` is a Laplace path):
  1. no dispatch below the level call (none is touched);
  2. layouts and tables unchanged;
  3. outputs bit for bit (the trees key for key with the default, and the 36 outputs);
  4. benchmarks within noise (question 2).
- Constraint 2: the leaves on P ranks are the one-rank leaves, as the test above checks.
- Risk: a deep `min_leaf_level` on a sparse input makes many leaves. This is the
  intended cost of §3.7, and S8 measured it.

## Results

- Unit tests: `test_create_local_tree_min_leaf_level_against_brute_force` (max_keys 1, 8
  and 64; max_level 3, 5 and 16; levels 0–6; from the root and from the octants) and the
  extended coarse-tree test pass.
- `test_mpi_weighted_partition` passes at 1, 2, 3, 4 and 8 ranks on the M6 (debug,
  about 8 s at 2 and 4 ranks).
- Laplace's outputs: `nd-fmm-bench --write-output` from `main` (e876cbc, a detached
  worktree with its own target directory) and from the branch, on the M6. The inputs were
  the cube and the Plummer sphere (N = 10⁵, 2 vectors, f64, 4 threads), at p = 3, 6
  and 8, with Dense, Classes, Rotation, Batched, PlaneWave and Auto. 36 of 36 outputs are
  identical byte for byte.

## Pending runs

None.

## Open questions

Answered 2026-10-31 by the user, each as recommended:

1. The split into a, b and c and the change to `nd-octree`: "Split a/b/c, proceed".
2. Constraint 1's fourth condition by the byte comparison of the outputs (CH2.0's 18 cases, as built 36) and the
   tree tests, without D9 timing runs: "Bytes only, no D9".
3. The name: `with_min_leaf_level`.

D1 was signed off the same day: kD ≤ 500 at 10⁻³ to 10⁻⁸, as recommended.
