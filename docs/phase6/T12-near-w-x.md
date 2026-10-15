# Phase 6 / T12 — M2P and P2L only where they pay (C6.9)

The W and X lists apply a multipole at a target point (M2P) or form a local from a source
point (P2L) whatever the box holds. Phase 5S T9 found the cost on the half-refined cube at
2 × 10⁶ points (`fmm-validate/results/phase5s-kathleen.md` §11); T1 measured it as the general
case of an adaptive tree: M2P and P2L take 28–74 % of an evaluation on the Plummer sphere at
N = 10⁶ on every machine and split, and 71–73 % on that cube at one rank, more than P2P in
every cell at p ≥ 6 (docs/design/optimisation.md §1.2, §8a.2). One M2P target point costs
46–963 P2P pairs and one P2L source point 15–460, by machine and p, while a leaf holds at
most 64 points. This task sends such pairs to P2P, exactly, by a rule the operator supplies
to the plan, as `docs/design/optimisation.md` §8a designs it and as signed off on
2026-10-15 (decision 1 and question 7 of optimisation.md §10, the `nd-fmm-plan` interface
change approved).

Written on 2026-10-15, after T1's sign-off.

Read first:
- root CLAUDE.md, fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-validate/CLAUDE.md;
- docs/phase6/README.md (requirements 1–6, C6.9's criterion);
- docs/design/optimisation.md §8a (the pairs, the crossovers, the rule, accuracy and order),
  §1.2 (the shares) and §10;
- docs/design/fmm-plan-redesign.md §4 (the lists and views), §7.5 (the accumulation order);
  docs/design/scale-out.md §6 (the work-weighted cut and its counts);
- the code: `nd_fmm_plan::interaction_manager` (`key_lists`: U, V, W, X), `plan`
  (`Plan::new`, `Plan::from_key_types`, the leaf numbering and its ghost leaves named by U
  or X), `lists` (the near, W and X views); `nd_fmm_exec::cost` (`CostModel`, `calibrated`,
  `leaf_point_counts`, `work_counts`), `nd_fmm_exec::fmm` (`build`, step 4: the counts and
  the second cut), the `scaling` harness (`--kinds`, the fit of the constants).

Do:
- **The near rule in `nd-fmm-plan`**: an optional, kernel-agnostic predicate given to
  `Plan::new` (and `from_key_types`) with the point counts of the leaves it reads, deciding
  for each W entry s of a leaf t whose box s is a leaf whether it becomes a near entry of t
  (t's near row gains s, the W entry goes), and for each X entry t of a box s (t a leaf)
  whether it becomes a near entry of s (s's near row gains t, the X entry goes). The rule
  stays local (no collective) and gives the same rows on every rank; the near rows keep
  their (level, key) order; a converted ghost leaf is named in the leaf numbering as a
  U-list leaf is, so the source exchange delivers its points. Without a rule the plan is
  unchanged bit for bit.
- **The rule in `nd-fmm-exec`**: from the `CostModel`'s constants at the run's p and
  precision, N_W = α_M2P / α_P2P and N_X = α_P2L / α_P2P (optimisation.md §8a.2): a W pair
  converts when the source leaf holds fewer than N_W points, an X pair when the target leaf
  holds fewer than N_X; the counts are those `build` already gathers for the cut (every local
  leaf's source and target counts, the ghost leaves' source counts), agreed on every rank.
  Opt-in by a builder setting (named in this task, off by default until T11), reported by
  `Fmm` (the rule, N_W, N_X, the pairs converted).
- **The cost model and the cut**: converted pairs count as P2P, not M2P or P2L, in
  `cost::work_counts`, so the cut weighs the converted plan.
- **The harness**: `scaling` reports the converted W and X pairs and the setting.
- **Measurements** (release, the machine named): the leaves and downward stages and the
  evaluation with the rule off and on, f64 p = 3 and 8 and f32 p = 3 and 8, the Plummer
  sphere and the Gaussian clusters at N = 10⁶ and the cube at 2 × 10⁶, at one rank and at
  the node's ranks on the M3 Max, locust and one Kathleen node; the cube at 2 × 10⁶ on 80
  Kathleen ranks (2 nodes), C6.9's criterion; the share of W boxes that are leaves and
  their point counts; the errors against the direct sum. Report in
  `fmm-validate/results/phase6-near.md`.

Tests that define done:
- `nd-fmm-plan`: the serial tests of the rule on the synthetic trees of
  `interaction_manager_tests.rs` (converted entries exactly those the predicate names, the
  duality of W and X kept for the pairs not converted, rows ordered); `tests/mpi_regressions.rs`
  scenarios with a rule (counts per leaf from the scenario's points), at 1, 2 and 4 ranks:
  the lists against the brute-force oracle with the conversions applied, the source
  exchange delivering every converted ghost leaf, and the plan without a rule unchanged.
- `nd-fmm-exec`: on every `tests/mpi_exec.rs` scenario with W or X lists, at 1, 2 and 4
  ranks, the rule on against the direct sum within requirement 1's bounds and with errors
  at most the rule-off path's (P2P is exact); bit-identical across thread counts, two
  evaluations and two builds, and on P ranks the one-rank `Fmm` of the same rule bit for
  bit; the rule off unchanged bit for bit.
- The C5.1 host gate (`tests/multi_rank.rs`) with the rule on at 2 and 4 ranks;
  `tests/invariance.rs` (rank-count invariance) with the rule on.

Must pass: `cargo fmt --all`, the root checks and the stricter workspace checks;
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan` and `-p nd-fmm-exec`;
`.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks; `mpi_regressions` and `mpi_exec` at 8
ranks by hand under an external timeout; Kathleen's `correctness.sbatch` at 2 nodes.

Do not: put kernel arithmetic or the cost constants into `nd-fmm-plan` (it receives a
predicate and counts only); convert interior W boxes (their points are not delivered);
make the rule a default (T11 decides); assert a timing.
