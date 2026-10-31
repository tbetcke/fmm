# CH3.0b The Helmholtz builder's leaf rule

Started 2026-10-31 on `tbetcke/h-ch3-0b`. Roadmap: "CH3.0b The Helmholtz builder's leaf
rule" (H3; the second third of CH3.0, eighth on the priority list). Proposal: §10 CH3.0;
§2, §3.7, §6.1, §6.3, §9 item 8, §9.1; CONVENTIONS §3.15 ("Kernel and factors", "Inputs and
outputs").

## Goal

The leaf rule of proposal §3.7 as library code: k classified as §3.15 states, and the
level that keeps every occupied leaf at most c_leaf λ wide, from (k, the domain, c_leaf).
The reference workloads W1–W4 move from the census spike into `nd-fmm-validate`, and a
test shows the rule on their trees.

## Acceptance (proposal §10, verbatim)

> no leaf wider than c_leaf λ on W1–W4; if `nd-octree` gains the maximum leaf width,
> Laplace's trees and outputs bit for bit unchanged (constraint 1); complex charges in
> and potentials and gradients out as split planes (§3.15 item 9), one complex expansion
> per complex charge vector

CH3.0b's share is the first clause. Its *Done when* (roadmap): no leaf wider than
c_leaf λ on W1–W4 over D1's kD range (kD ≤ 500); no rule for Yukawa. CH3.0a did the
second clause; CH3.0c does the third.

## Read first

- `docs/helmholtz/tasks/CH3.0a.md`; `OctreeOptions::with_min_leaf_level` in
  `octree/src/octree.rs`.
- `spikes/census/src/{workloads.rs, tree.rs}` (`Workload`, `width_level`) and its report
  (`spikes/census/SPIKE_REPORT.md`: D11 = 1, the leaf levels per workload).
- `fmm-exec/src/geometry.rs` (`Domain`), `fmm-exec/src/fmm.rs` step 4 (the tree's
  options).
- CONVENTIONS §3.15 (the three kinds of k and the rejected ones).

## Changes

Two crates, no convention, no `nd-fmm-plan` or Laplace code.

- **`nd-fmm-exec`**, a new module `helmholtz` (`src/helmholtz.rs`), proposal §6.1's home
  of the builder:
  - `Wavenumber::new(re, im) -> Result<Wavenumber, WavenumberError>` and
    `WavenumberKind { Real, Complex, Yukawa }`, §3.15's classification: Yukawa for
    Re k = 0 and Im k > 0, real for Im k = 0 and Re k > 0, complex for both positive.
    Rejected, with a `thiserror` error each: a part not finite; k = 0 (Laplace, not a
    Helmholtz case); Re k < 0 (incoming waves); Im k < 0 (growing kernels). A signed zero
    counts as zero. Accessors `re`, `im`, `kind`, and `wavelength()`, 2π/Re k, `None` for
    Yukawa.
  - `C_LEAF = 1.0`, D11's value (signed off 2026-10-28).
  - `min_leaf_level(k, domain, c_leaf) -> usize`: the shallowest level l whose boxes are
    at most c_leaf λ wide, w 2⁻ˡ ≤ c_leaf λ with w the domain's side, capped at level 16;
    0 (no rule, `OctreeOptions`' default) for Yukawa. It is the value the builder passes
    to `OctreeOptions::with_min_leaf_level`. For complex k, λ = 2π/Re k: the
    oscillation's wavelength, which the phase range of P2P (k r below about 22 c_leaf,
    §3.7) and the leaf degrees follow; Im k only damps. Panics unless c_leaf is positive
    and finite. It depends on k, the domain and c_leaf only (constraint 9), so every
    rank finds the same level.
  - Nothing in `fmm.rs` changes: `FmmBuilder` is Laplace's builder and gets no new
    setting here. CH3.1's Helmholtz FMM passes the level to its tree.
- **`nd-fmm-validate`**, a new module `workloads` (`src/workloads.rs`): proposal §9.1's W1–W4
  in the root cube [0, 1]³ (D = 1), the census spike's `Workload` ported unchanged
  (points, seeds, margin, `w1_count`, `w3_resolved`), plus the domain as a constant.
  Charges stay out (CH3.0c and CH3.1 add them).
  - **Dependency (question 1, answered: add it):** the test below builds trees with `nd_octree::Octree`,
    so `nd-fmm-validate` needs `nd-octree` as a direct dependency (`workspace = true`;
    already in the graph through `nd-fmm-exec`, so no new package and no licence change,
    one new edge in `Cargo.lock`). `fmm-validate/CLAUDE.md`'s allowed list gains it.
- `fmm-exec/CLAUDE.md` and `fmm-validate/CLAUDE.md` describe the additions.

## Tests that prove it

- **`nd-fmm-exec`, unit tests of `helmholtz`** (no MPI):
  - the classification of each kind, and each rejection (0, NaN, ±∞, Re k < 0,
    Im k < 0), with signed zeros;
  - `min_leaf_level` against the census spike's `width_level` (its search, ported into
    the test as the trusted path) for kD over 10⁻² to 10⁵ and c_leaf 0.25 to 2: equal;
  - minimality: at the level, w 2⁻ˡ ≤ c_leaf λ; one level up, wider (for l > 0);
  - complex k at the level of real Re k; Yukawa 0; the cap at 16; the side's scaling
    (k w the same, the level the same); panics for c_leaf ≤ 0 or not finite.
- **`nd-fmm-validate`, unit tests of `workloads`**: the counts (W1 7.96 (kD)², W2 about
  1.6 N(W1), W3 resolved), every point strictly inside the root cube, W1 on its sphere,
  the seeds deterministic; the points equal the census spike's bit for bit (checked
  once against the spike's crate, which depends on `nd-fmm-validate`, by a test in
  `spikes/census/tests/`).
- **`nd-fmm-validate`, `tests/helmholtz_leaves.rs`** (its one MPI test, one rank in CI):
  the tree of `Fmm`'s step 4 (`max_level` 16, 64 points per leaf, ghost children) with
  `with_min_leaf_level(min_leaf_level(k, domain, C_LEAF))`, and for every point its leaf
  (`Octree::local_leaf`) no wider than c_leaf λ. In debug: W1 and W2 at kD = 10, 30 and
  100; W3 at N = 10⁵ (kD = 10, 100, 500); W4's four distributions at N = 10⁴, kD = 1,
  10, 100, 500; complex k (Im k = Re k / 10) with the same leaves as real k; Yukawa the
  tree without the option, key for key.
  - An ignored release test, the D1 range in full, run by hand: W1 and W2 at kD = 10,
    30, 100, 200, 300 and 500 (W1 2.0 million and W2 3.2 million points at 500); W3
    resolved at kD = 10, 30 and 100 (4.0 million) and at N = 10⁶ for kD = 200 and 500;
    W4 at N = 10⁵ over kD = 1 to 500. It prints per tree the leaf level of the rule, the
    occupied leaves and the widest occupied leaf in λ, for the record.

## Checks

`/verify`; `cargo deny --locked check licenses` (a manifest changes); the ignored
release test of `tests/helmholtz_leaves.rs` on the M6.

## Machine runs

None besides the M6: tree construction only, no FMM and no timing. The record is this
plan's "Results" (the printed table); no separate report.

## Constraints and risks

- Constraint 1: no Laplace path changes (no change to `nd-octree`, `nd-fmm-plan`,
  `fmm.rs`, the drivers, the tables or `nd-fmm-simd`). The trees of every existing caller
  are untouched, so no byte comparison or D9 run is needed.
- Constraint 9: the level is a function of k, the domain and c_leaf only.
- Risk: the ignored test's largest trees (W2 at kD = 500, 3.2 million points) take a
  few seconds each in release (the census built W1 at kD = 1000 in 13 s).

## Results

<filled when run>

## Pending runs

None planned.

## Open questions

1. `nd-octree` as a direct dependency of `nd-fmm-validate` (for the tree test).
   Answered 2026-10-31 by the user: "Add nd-octree" (as recommended).
2. λ = 2π/Re k for complex k (above). Recorded in the PR as a sign-off point.
