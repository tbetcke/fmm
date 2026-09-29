# Phase 1 / T7 — nd-fmm-validate: metrics, accuracy and timing reports

Read first: root CLAUDE.md, fmm-validate/CLAUDE.md (already committed with this brief),
docs/design/laplace-fmm-plan.md §4 ("Starting points for p"), §7 (C3.2, C3.4) and §8;
docs/design/workspace-structure.md §3; the nd-fmm-ref API.

This crate holds tooling that measures rather than tests. Its accuracy table is the
"single-translation prediction" against which C3.2 (Phase 3) is accepted. Its timing
report confirms the O(p³) cost of the rotation operators and locates the
direct/rotation crossover in `nd-fmm-ref`.

Do:
- Create the crate in fmm-validate/ (package nd-fmm-validate, lib nd_fmm_validate,
  `publish = false`, fields inherited from [workspace.package]). Add it to `members`
  and `default-members`. Dependencies: nd-fmm-math and nd-fmm-ref
  (`workspace = true`). No others.
- Library:
  - `SplitMix64`: a seeded generator, the same algorithm as the fmm-math tests, so no
    rand dependency.
  - `points`: uniform in a cube and in a ball, and on a sphere surface. The Plummer
    and Gaussian-cluster distributions wait for Phase 3.
  - `metrics`:
    - relative L2 and max error of potentials and gradients;
    - error relative to the sum of term magnitudes;
    - the per-degree coefficient error in the orthonormal weighting of §3.8 (Nₘ and
      Nₘ/Sₘ).

    Doc comments say when to use which (docs/phase1/README.md, "Error measures").
- Example `accuracy`:
  - Chains: P2M → M2P, P2L → L2P, P2M → M2M → M2P, P2M → M2L → L2P and
    P2M → M2L → L2L → L2P.
  - Geometry: the standard one-box separation, with the worst offset and all 316
    offsets.
  - Charges: 1,000 uniform sources with random charges; 1,000 targets.
  - Reported: relative L2 and max error of potential and gradient against
    `direct_sum`, for p = 1..=20 in f64 and p = 1..=8 in f32.
  - Output: a Markdown table on stdout.
- Example `timing`:
  - Time per call of direct and rotation M2M, L2L and M2L for
    p ∈ {2, 4, 6, 8, 10, 12, 16, 20}, plus 30 for M2M and L2L.
  - Run in release mode; report the median of repeated batches, timed with
    `std::time::Instant`.
  - Report a fitted exponent for p ≥ 8 and the crossover p above which rotation is
    faster.
  - Print the machine (CPU model, core count) with the table.
- Tests: metrics on hand-made examples (zero error, a known error); distributions
  deterministic for a seed and inside their domains; a smoke run of the accuracy
  sweep at p ≤ 4 with few points.

Must pass: the root CI commands and the stricter workspace checks;
`cargo test -p nd-fmm-validate`;
`cargo run --release -p nd-fmm-validate --example accuracy` and `--example timing`
each finish within a few minutes. Paste both tables into the PR.

Do not:
- assert timings in tests;
- commit CSV or other output files (`.gitignore` does not cover `*.csv`);
- add criterion or any other dependency without asking;
- make any library crate depend on nd-fmm-validate, not even as a dev-dependency.
