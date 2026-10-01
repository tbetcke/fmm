# Phase 3 / T11 — adaptive trees: W and X lists on clustered distributions (C3.3)

T8 implemented M2P (W list) and P2L (X list), and T9 ran them only where uniform
trees leave those lists empty. Clustered points give adaptive trees whose leaves meet
neighbours on other levels, so W and X are where the remaining errors hide (design
§9.1, "Adaptive-list edge cases"). This task adds the clustered distributions of
design §8.2 and checks the FMM on them against the direct sum.

Read first: docs/phase3/README.md ("Exit gate", C3.3); design §2.5 (the lists) and
§8.2 (distributions); the docs of the new nd-fmm-plan interaction lists (the U, V,
W and X rules and their guarantees); fmm-plan/tests/mpi_regressions.rs (the
"graded corner blob" scenario); the T9 code and its `fmm_accuracy` example;
nd_fmm_validate::points.

Do:
- nd-fmm-validate `points`:
  - `plummer(rng, n, centre, scale)`: the Plummer sphere, by inverse transform of
    its cumulative mass, truncated at a documented radius so the domain stays bounded;
  - `gaussian_clusters(rng, n, centres, width)`: n points spread evenly over a few
    tight Gaussian clusters at the given centres, with documented truncation;
  - tests for both: seeded determinism, the sample mean and radius against their
    analytic values within a few standard errors, and no point beyond the truncation.
- Extend `fmm_accuracy` with `--distribution {cube, sphere, plummer, clusters}` (or
  print all four), using `max_level` 16 and a fixed `max_points_per_leaf` (for example
  64) for the adaptive ones. Print per distribution:
  - the error table of T9, with the uniform-cube error at the same p and N and the
    ratio to it;
  - the tree: leaf levels (min, max, histogram), points per leaf (min, max, mean), and
    the total sizes of the U, V, W and X lists.
- nd-fmm-exec tests/mpi_exec.rs scenarios (small N, debug, against `direct_sum` over
  all targets):
  - each clustered distribution at N = 2,000, p ∈ {4, 8}, every strategy, with
    assertions that W and X are non-empty and that the leaves span at least three
    levels;
  - a strongly graded tree (a dense blob next to a sparse cloud, as in
    mpi_regressions), checked at p = 8;
  - a leaf at level 16 holding many coincident and nearly coincident points
    (duplicates beyond `max_points_per_leaf`), sources equal to targets: coincident
    pairs are excluded, the rest matches the direct sum;
  - leaves with sources but no targets and the reverse, next to each other;
  - points on box faces and corners of the domain.
- If T9's ignored accuracy test is parametrised, add the Plummer case at N = 10⁵, p = 8.
- If T10 (threads) is merged first, the new scenarios also run at its thread counts,
  bit-identical to one thread, and `fmm_accuracy` keeps its `--threads` option. If this
  task merges first, T10 picks the scenarios up instead.

Tests that define done: the scenarios above; the C3.3 gate from the example:
- for the sphere surface, Plummer and Gaussian clusters at N = 10⁵ and p ∈ {3, 8, 18}
  in f64, the relative L2 error of φ is at most twice the uniform-cube error at the
  same p and N;
- the W and X lists are non-empty for all three, with their sizes reported;
- f32 at p ∈ {3, 8} reported next to f64.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug run under a minute);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- `cargo run --release -p nd-fmm-validate --example fmm_accuracy` for all four
  distributions; paste the output into the PR;
- `cargo clippy -p nd-fmm-exec -p nd-fmm-validate --all-targets -- -D warnings`;
- the root checks.

Report the C3.3 table, the tree statistics per distribution, and the worst error per
list type where it can be separated (for example by disabling M2P or P2L in a test
operator wrapper and comparing with the direct sum restricted to the same pairs). Put
such wrappers in tests, not in the library.

Do not: change the interaction lists or nd-fmm-plan (if a list looks wrong, reproduce
it against the brute-force oracle of mpi_regressions and stop); tune
`max_points_per_leaf` per distribution to pass the gate; add dependencies.
