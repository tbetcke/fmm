# Phase 6 T12: M2P and P2L only where they pay (C6.9)

Phase 6, task T12 ([docs/phase6/T12-near-w-x.md](../../docs/phase6/T12-near-w-x.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §8a, signed off on
2026-10-15 (decision 1, question 7: the `nd-fmm-plan` interface change; §8a.6 records what
was built). Measured on 2026-10-19 from branch `tbetcke/phase6_t12` (main at 0a0f5fb plus
this task's changes, uncommitted), release builds, rustc 1.99.0, every BLAS thread variable
at 1, one thread per rank, the default strategy (`auto`, resolved by the ranks per node),
the blocking path. Kathleen is usable (docs/phase6/README.md): one node in `singlenode`
(job 605533) and two in `small` (job 605535) and `test` (job 605534, the correctness job),
nodes exclusive, so the path "with Kathleen" of the README; locust idle (load average
0.01 before the sweep, no other user's process above 20 % CPU before any launch, GPU
unused); the M3 Max behind the sweep's gate (other processes at most 194 % CPU, two cores'
worth, before every launch; 30 s of waiting in all), run outside the sandbox.

## Summary

1. **The near rule** (§1): an optional, kernel-agnostic rule in `nd-fmm-plan`'s list
   construction (`nd_fmm_plan::near`, `Plan::with_near_rule`): the operator supplies a
   predicate over a W or X pair between two leaves with the fine leaf's point count, and
   the counts; a W entry that is a leaf, or an X entry of a leaf, that the predicate names
   moves to the leaf's near row (P2P, exact) in (level, key) order, and a converted ghost W
   entry is numbered as a ghost leaf so the source exchange delivers its points.
   `Plan::new` is unchanged bit for bit. In `nd-fmm-exec`,
   `FmmBuilder::near_rule(NearRuleChoice::CostModel)` takes the crossovers of the cost
   model, N_W = α_M2P / α_P2P and N_X = α_P2L / α_P2P (107 and 35 points at f64 p = 3
   `Dense`, 421 and 184 at f64 p = 8 `Rotation`, Kathleen's constants), from the counts the
   cut already gathers plus the W rows' ghost leaves' (which the cut's source exchange does
   not name, a correction to §8a.3); the cut weighs the converted plan. **Off by default**
   until T11.
2. **Correct and deterministic** (§2): every test of the brief passes: the plan's lists
   against the per-key rule and the brute-force oracle with the conversions applied, at 1,
   2, 4 and 8 ranks; every `tests/mpi_exec.rs` scenario with W or X lists with the rule on,
   at 1, 2, 4 and 8 ranks, two evaluations, threads and overlap bit for bit, on P ranks the
   one-rank `Fmm` of the same rule bit for bit; rank-count invariance (`tests/invariance.rs`
   and at 80 Kathleen ranks against 40, 20 and 10); the C5.1 host gate at 1, 2, 4 and 8
   ranks; Kathleen's correctness job at 2 nodes.
3. **The W boxes** (§3): 96.6–100 % of the W entries are leaves (the cube at 2 × 10⁶ 100 %,
   the Plummer sphere 99.7 %, the clusters 96.6 %), and those hold 1–32 points in 84–99.99 %
   of the entries (none above 64, the leaf size): every W pair between leaves converts at
   every p and precision measured, every X pair at p = 8 and at f32 p = 3, 97–100 % at f64
   p = 3.
4. **Faster everywhere measured** (§4): the evaluation with the rule against without,
   one rank / the node's ranks:

   | problem | M3 Max | locust | Kathleen |
   | --- | --- | --- | --- |
   | the cube, N = 2 × 10⁶ | 2.46–4.04× | 1.79–4.14× | 3.06–5.89× |
   | the Plummer sphere, N = 10⁶ | 1.42–1.93× | 1.32–1.89× | 1.48–2.71× |
   | the Gaussian clusters, N = 10⁶ | 1.37–1.96× | 1.34–1.91× | 1.51–2.74× |

   M2P and P2L fall to 0–6 % of their time (the interior W boxes keep theirs); P2P grows
   by up to 1.8×. The model of §8a.2 expected 1.15–2.2×.
5. **C6.9's criterion is met** (§5): the cube at 2 × 10⁶ on 80 Kathleen ranks (2 nodes,
   job 605535) spends 1.9× (f64 p = 8) and 2.1× (f32 p = 3) the uniform tree's leaf time
   (10⁶ points on 40 ranks), from 15× and 12× without the rule; its evaluation is 3.28×
   (1,970 → 601 ms) and 5.77× (424 → 73 ms) faster, and its errors against the direct sum
   fall (φ 0.995–0.997, ∇φ 0.942).
6. **Accuracy, a sign-off question** (§6): a converted pair is exact, and the errors
   against the direct sum fall on C6.9's problems: the C3.2 cube has no W or X pair (the
   same bits), the C3.3 Plummer sphere 0.995–0.998 at every gate point, the cube at 2 × 10⁶
   0.995–0.998 (∇φ 0.94). But the error is a sum over pairs whose truncation errors partly
   cancel, so converting some pairs can raise it: on the clusters at 10⁶ by up to 0.13 %
   (φ, f32 p = 3; ∇φ falls), and on a small test tree with leaves of hundreds of points by
   2.1 % (p = 5, the cost model's crossovers) and 1.2 % with every pair converted. The
   change from the rule off falls with p as the FMM's error does (a tenth of it), so it is
   cancellation, not a defect. **Proposed**: C6.9's accuracy criterion as "within 1.1× the
   existing path's errors against the direct sum" (the form of C6.5's budget), which the
   tests assert; "at most" holds on C6.9's named problems.
7. **The build** (§7) grows by the counts and a second plan: by 0.6–2.5 s at one rank
   (51–70 %), by up to 0.2 s at the node's ranks; the first evaluation repays it in 67 of
   the 72 configurations, the second in all.
8. **For T11** (§8): make the cost model's rule the default (every measured configuration
   gains 1.3–5.9×); refit the cost model with it (P2P's share grows); build the counts
   without the first plan.

## 1. What T12 adds

- **`nd-fmm-plan`, module `near`** (`docs/design/optimisation.md` §8a.6):
  - `LeafPair::W { target, source, sources }` (the W entry `source`, a leaf one level
    finer, of the leaf `target`, with the source points of `source`) and
    `LeafPair::X { target, source, targets }` (the X entry `source` of the leaf `target`,
    with the target points of `target`): the fine leaf's count in both, which decides
    (§8a.1: per pair M2P costs α_M2P n_t against α_P2P n_t n_s for P2P, so n_t cancels).
  - `LeafCounts`: source counts of the local leaves and of every leaf a W row names, target
    counts of the local leaves, by key. `LeafCounts::exchange(plan, comm, sources,
    targets)` (collective) fetches the W rows' ghost leaves' counts from their owners as
    the chunk sizes of one rlst ghost communicator (no value moves): those leaves are not
    always among the ghost leaves the source exchange of the cut names, U and X name those.
  - `NearRule::new(&counts, &predicate)`; `Plan::with_near_rule(octree, &rule)` and
    `Plan::from_key_types_with_near_rule` (local) apply it in `first_pass`: the predicate is
    called once per W entry of a local leaf that is a leaf and once per X entry of a local
    leaf; a converted entry joins the leaf's U-entries, so its near row in (level, key)
    order (P1), and a ghost W entry joins the leaf numbering's ghost leaves. An interior W
    box keeps its M2P, the X entries of an interior box their P2L. A missing count is
    `PlanError::MissingCount` (agreed by `Plan::new`'s validity all-reduce).
    `Plan::near_pairs` reports the candidates and the conversions (`NearPairs`).
  - `Plan::new` and `Plan::from_key_types` keep their signatures and build the plan of
    before, bit for bit: a sibling constructor, not a changed `Plan::new` (the brief's
    wording), so that the call sites in twenty-odd files of other crates stay as they are.
  - Everything downstream follows the rows unchanged: the source exchange delivers the
    converted ghost leaves (they are ghost leaves of the numbering), the multipole exchange
    no longer moves a multipole only a converted W pair read, P2P runs host and device as
    for any near pair (the frames between any two leaves are exact, CONVENTIONS §3.13).
- **`nd-fmm-exec`**: `FmmBuilder::near_rule(NearRuleChoice)` (`Off` by default; `CostModel`;
  `Crossovers { w, x }` for tests and sweeps, negative or NaN refused,
  `SettingsError::NearCrossover`), `CostModel::near_crossovers`, `Fmm::near_rule`
  (`NearRuleReport`: the choice, [N_W, N_X], the plan's `NearPairs`). In `build`, step 4: the
  points of every local leaf where they are (`cost::leaf_point_counts`, the second cut's
  all-to-all-v, run once for both), `LeafCounts::exchange`, the plan again with the rule;
  the cut weighs that plan (`cost::work_counts` counts its views, so the converted pairs as
  P2P), and after a recut the counts and the rule's plan are made again for the new owners.
  `BuildTimings::plan` holds the rule's part (`cut` after a recut).
- **The harness** (`nd-fmm-validate`'s `scaling`): `--near off|cost|W,X`, the crossovers and
  the pairs converted in the report and the summary (`near`, `near_crossovers`,
  `w_leaf_pairs`, `w_converted`, `x_leaf_pairs`, `x_converted`), and the W entries of the
  local leaves by kind and points (`w_entries`, §3); `tools/scaling/run.sh` sweeps
  `t12-near` and `t12-weak`.

## 2. Correctness

| check | where | result |
| --- | --- | --- |
| the rule on the synthetic trees: converted entries exactly the predicate's, each candidate offered once with its count, every other view and the local leaves unchanged, the ghost leaves those of U and X plus the converted ghost W entries; W–X duality for the pairs kept; rows in (level, key) order; a rule converting nothing gives `Plan::new`'s views; a missing count is an error | `nd-fmm-plan` `src/near_tests.rs`, 5 tests | pass |
| every `mpi_regressions` scenario with counts from its points: the counts read (ghost W leaves' from their owners), the lists against the brute-force oracle with the conversions, `NearPairs`, the source exchange delivering every ghost leaf with its count, the reduction, top and multipole exchange on the converted rows, every converted pair issued once as P2P and never as M2P or P2L, one rank's bits, a never-converting rule equal to `Plan::new` | `tests/mpi_regressions.rs`, `check_near_rule` | pass at 1, 2, 4, 8 ranks (M3 Max) and 80 (Kathleen); at 8 ranks 22,717 of 29,614 W and 22,263 of 29,614 X pairs converted, 1,278 ghost W leaves numbered |
| every `mpi_exec` scenario with W or X lists, rule of the cost model and crossovers W 12, X 6: errors within 1.1× the rule off's and changed by at most the rule-off error, two evaluations, a second build at 8 threads overlapped, the one-rank `Fmm` of the same rule bit for bit | `tests/mpi_exec.rs`, `check_near_rule` | pass at 1, 2, 4, 8 ranks (M3 Max) and 80 (Kathleen); largest error ratio 1.021 (f64), 0.995 (f32) |
| the work-weighted cut with the rule: cut off against on bit for bit, the same pairs converted, counts by kind summed over the ranks those of one rank | `tests/mpi_exec.rs`, **work-weighted cut** | pass at 2, 4, 8 |
| rank-count invariance with the rule: the Plummer sphere and the graded tree, f64 p = 8 `Rotation` and f32 p = 3 `Dense` | `tests/invariance.rs` | pass at 1, 2, 4 ranks (M3 Max) |
| rank-count invariance at scale, the cube at 2 × 10⁶, 80 ranks against 40, 20, 10, f64 p = 8 and f32 p = 3 `Rotation`, overlap both | `scaling --invariance 2,4,8 --near cost`, job 605535 | 0 of 1.6 × 10⁷ values differ in each |
| the C5.1 host gate with the variant "auto, near rule" (N = 10⁵, five workload points) and the C6.9 ratio | `tests/multi_rank.rs`, release | pass at 1, 2, 4, 8 ranks |
| Metal f32 with the rule (the Plummer sphere at 2 × 10⁵, p = 8): the same pairs converted, φ error 3.629e-5 against the host's 3.647e-5 with the rule (3.639e-5 / 3.658e-5 without) | `scaling --backend metal` | as the host |

## 3. The W boxes

The W entries of the local leaves on one rank (the rule off; `w_entries`), by box and
source points:

| problem | W entries | interior | leaves with 0 | 1–8 | 9–16 | 17–32 | 33–64 | > 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the cube, N = 2 × 10⁶ | 1,015,057 | 0 | 78 | 497,800 | 511,247 | 5,932 | 0 | 0 |
| the Plummer sphere, N = 10⁶ | 201,566 | 598 (0.3 %) | 23 | 64,642 | 111,854 | 21,555 | 2,894 | 0 |
| the Gaussian clusters, N = 10⁶ | 308,917 | 10,353 (3.4 %) | 26,405 | 84,848 | 131,376 | 44,352 | 11,583 | 0 |

A W leaf is a child of a refined neighbour (whose parent held more than 64 points), so it
holds a few to a few dozen; none holds more than 32 on the cube. The X pairs between leaves
are as many as the W pairs (duality). With every crossover of the cost model above 64 every
W pair between leaves converts; only N_X at f64 p = 3 (35 points) keeps some X pairs (1–3 %).

## 4. Measurements

Evaluation (the slowest rank, median of three), the leaves and downward stages, M2P + P2L
and P2P kind times (`KindTiming::Synchronous`), in ms, with the rule off → on (`--near
cost`), and the errors against the direct sum of one charge vector at 1,000 sampled
targets (relative L2). The full tables per machine are in §10.

### 4.1 Kathleen, one node (job 605533; 1 × 1 and 40 × 1, AVX2 P2P)

| problem | f64 p = 3 | f64 p = 8 | f32 p = 3 | f32 p = 8 |
| --- | --- | --- | --- | --- |
| cube 2 × 10⁶, 1 × 1 | 23,994 → 7,852 (3.06×) | 111,203 → 33,391 (3.33×) | 21,411 → 3,936 (5.44×) | 101,710 → 21,362 (4.76×) |
| cube 2 × 10⁶, 40 × 1 | 941 → 297 (3.17×) | 4,209 → 1,223 (3.44×) | 871 → 148 (5.89×) | 3,837 → 1,017 (3.77×) |
| Plummer 10⁶, 1 × 1 | 5,891 → 3,246 (1.81×) | 37,671 → 24,830 (1.52×) | 4,643 → 1,716 (2.71×) | 29,913 → 16,374 (1.83×) |
| Plummer 10⁶, 40 × 1 | 229 → 133 (1.71×) | 1,322 → 894 (1.48×) | 187 → 71 (2.63×) | 1,216 → 761 (1.60×) |
| clusters 10⁶, 1 × 1 | 6,096 → 3,312 (1.84×) | 40,904 → 27,091 (1.51×) | 4,936 → 1,803 (2.74×) | 32,409 → 17,893 (1.81×) |
| clusters 10⁶, 40 × 1 | 244 → 132 (1.84×) | 1,467 → 975 (1.51×) | 194 → 75 (2.57×) | 1,359 → 841 (1.62×) |

### 4.2 locust (1 × 1 and 72 × 1, NEON P2P)

| problem | f64 p = 3 | f64 p = 8 | f32 p = 3 | f32 p = 8 |
| --- | --- | --- | --- | --- |
| cube 2 × 10⁶, 1 × 1 | 14,810 → 8,153 (1.82×) | 55,710 → 20,690 (2.69×) | 11,127 → 3,357 (3.31×) | 47,008 → 11,343 (4.14×) |
| cube 2 × 10⁶, 72 × 1 | 212 → 118 (1.79×) | 794 → 304 (2.62×) | 159 → 50 (3.18×) | 714 → 218 (3.27×) |
| Plummer 10⁶, 1 × 1 | 4,118 → 3,122 (1.32×) | 19,687 → 13,882 (1.42×) | 2,727 → 1,445 (1.89×) | 13,904 → 7,944 (1.75×) |
| Plummer 10⁶, 72 × 1 | 74 → 50 (1.48×) | 297 → 214 (1.39×) | 42 → 25 (1.70×) | 261 → 169 (1.54×) |
| clusters 10⁶, 1 × 1 | 4,098 → 3,065 (1.34×) | 21,056 → 14,974 (1.41×) | 2,844 → 1,491 (1.91×) | 15,031 → 8,630 (1.74×) |
| clusters 10⁶, 72 × 1 | 71 → 49 (1.43×) | 318 → 226 (1.41×) | 44 → 24 (1.83×) | 276 → 190 (1.46×) |

### 4.3 The M3 Max (1 × 1 and 12 × 1, NEON P2P)

| problem | f64 p = 3 | f64 p = 8 | f32 p = 3 | f32 p = 8 |
| --- | --- | --- | --- | --- |
| cube 2 × 10⁶, 1 × 1 | 9,145 → 3,432 (2.66×) | 41,875 → 12,911 (3.24×) | 8,175 → 2,141 (3.82×) | 38,551 → 9,546 (4.04×) |
| cube 2 × 10⁶, 12 × 1 | 861 → 350 (2.46×) | 3,907 → 1,345 (2.90×) | 743 → 204 (3.64×) | 3,676 → 1,092 (3.37×) |
| Plummer 10⁶, 1 × 1 | 2,430 → 1,521 (1.60×) | 14,369 → 9,470 (1.52×) | 2,058 → 1,066 (1.93×) | 12,102 → 7,194 (1.68×) |
| Plummer 10⁶, 12 × 1 | 241 → 167 (1.44×) | 1,373 → 967 (1.42×) | 200 → 113 (1.77×) | 1,278 → 852 (1.50×) |
| clusters 10⁶, 1 × 1 | 2,528 → 1,563 (1.62×) | 15,551 → 10,318 (1.51×) | 2,188 → 1,118 (1.96×) | 13,129 → 7,875 (1.67×) |
| clusters 10⁶, 12 × 1 | 248 → 162 (1.53×) | 1,413 → 1,033 (1.37×) | 209 → 110 (1.90×) | 1,345 → 915 (1.47×) |

### 4.4 Where the time goes

Without the rule M2P and P2L (their kind times, the slowest rank's) take 32–87 % of an
evaluation on these trees (Kathleen at one rank, f64 p = 8: the cube 72 %, the Plummer
sphere 36 %, the clusters 36 %; f32 p = 3: the cube 87 %, the Plummer sphere 68 %, the
clusters 70 %); with it nothing on the cube and at most 9 % elsewhere (the interior W boxes
of the clusters and the Plummer sphere; 9 % in the clusters' 75 ms at f32 p = 3 on 40
ranks, below 2 % at one rank). P2P, which takes over the pairs, grows by up to 1.8× (on
the cube at one rank on Kathleen 3,730 → 6,173 ms at f64 p = 8), a tenth or less of what
M2P and P2L cost. At p = 8 the downward pass (M2L) is what remains: the Plummer sphere's
evaluation at f64 p = 8 is 78–86 % downward with the rule. More than §8a.2's model gains because the model assumed only W converts at
p = 3 and leaves at the 64-point maximum: X converts too (N_X = 35 to 92 points against
leaves of 1–32), and the W leaves hold 1–32 points.

## 5. C6.9's criterion: the cube at 2 × 10⁶ on 80 Kathleen ranks

Job 605535 (`small`, 2 nodes, node-c11b-[021-022], 40 ranks per node, one thread each,
`t12-weak`): Phase 5S T9's weak points, 2.5 × 10⁴ points per rank; the uniform tree at
N = 10⁶ on 40 ranks (here spread over both nodes, 20 per node) and the half-refined tree at
2 × 10⁶ on 80.

| run | evaluation | leaves | downward | M2P + P2L kinds | P2P kind | φ error | ∇φ error |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| f64 p = 8, 10⁶ on 40, rule off | 500 | 75 | 404 | 0 | 51 | 1.828e-5 | 5.271e-5 |
| f64 p = 8, 10⁶ on 40, rule on | 498 | 74 | 402 | 0 | 51 | 1.828e-5 | 5.271e-5 |
| f64 p = 8, 2 × 10⁶ on 80, rule off | 1,970 | 1,119 | 901 | 1,489 | 71 | 2.361e-5 | 1.433e-5 |
| f64 p = 8, 2 × 10⁶ on 80, rule on | 601 | 144 | 439 | 0 | 115 | 2.354e-5 | 1.350e-5 |
| f32 p = 3, 10⁶ on 40, rule off | 40 | 27 | 9 | 0 | 21 | 2.924e-3 | 4.082e-3 |
| f32 p = 3, 10⁶ on 40, rule on | 41 | 27 | 9 | 0 | 21 | 2.924e-3 | 4.082e-3 |
| f32 p = 3, 2 × 10⁶ on 80, rule off | 424 | 323 | 108 | 385 | 29 | 3.698e-3 | 1.144e-3 |
| f32 p = 3, 2 × 10⁶ on 80, rule on | 73 | 56 | 12 | 0 | 50 | 3.680e-3 | 1.078e-3 |

The leaves at 2 × 10⁶ on 80 ranks against the uniform tree's on 40: **15× → 1.9× at f64
p = 8 and 12× → 2.1× at f32 p = 3** (Phase 5S T9's figures were 1,068–1,083 ms against
70–89 ms at 40 and 160 ranks at f64 p = 8). The rest of the difference is P2P (115 against
51 ms): the 2 × 10⁶ tree mixes leaves of about 60 points with refined ones of a few to 32,
so it has more near pairs per point, not the expansions. The uniform tree has no W or X pair, so the rule changes nothing there (the
same errors, times within noise). With the rule the 80-rank evaluation is 1.2× (f64 p = 8)
and 1.8× (f32 p = 3) the 40-rank one, from 3.9× and 10.5×. Rank-count invariance with the
rule at 80 ranks against 40, 20 and 10: no value differs (§2).

## 6. Accuracy

The φ error with the rule over the error without it (∇φ in brackets), one charge vector
at 1,000 targets; the same on every machine and rank count of §4 to the digits shown but
for the strategy, which `auto` resolves by the ranks per node:

| problem | f64 p = 3 | f64 p = 8 | f32 p = 3 | f32 p = 8 |
| --- | --- | --- | --- | --- |
| cube 2 × 10⁶ | 0.9951 (0.942) | 0.9970 (0.942) | 0.9951 (0.942) | 0.9975–0.9987 (0.946) |
| Plummer 10⁶ | 0.9977 (0.997) | 0.9987 (0.998) | 0.9977 (0.997) | 0.9990–0.9993 (0.998) |
| clusters 10⁶ | 1.0006 (0.995) | 1.0000 (0.993) | 1.0013 (0.995) | 0.9990–1.0010 (0.993) |

The C5.1 gate's problems at N = 10⁵ (8 charge vectors, `tests/multi_rank.rs`, every rank
count): the C3.2 cube 1.00000 (its uniform level-4 tree has no W or X pair: the same bits),
the C3.3 Plummer sphere 0.99749 / 0.99522 / 0.99826 at f64 p = 3 / 8 / 18 and 0.99728 /
0.99519–0.99590 at f32 p = 3 / 8.

So on the problems C6.9 names (§8a.4: the C3.2 cube, the C3.3 Plummer sphere, the cube at
2 × 10⁶ on 80 ranks) the errors are at most the existing path's. They are not in general:
the Gaussian clusters' φ error grows by up to 0.13 % (f32 p = 3; ∇φ falls 0.5 %), and in
`tests/mpi_exec.rs` a tree of 3,000 points with leaves of hundreds of points
(**work-weighted cut**, p = 5, `Rotation`) by 2.1 % with the cost model's crossovers. On
that tree at one rank (a by-hand diagnostic, not committed):

| p | φ error, rule off | cost model | W only (N_W 187) | X only (N_X 82) | every pair | change, cost model |
| ---: | --- | --- | --- | --- | --- | --- |
| 3 | 1.5603e-4 | 1.5577e-4 | 1.5377e-4 | 1.5580e-4 | 1.5015e-4 | 8.1e-6 |
| 5 | 2.2141e-5 | 2.2607e-5 | 2.2627e-5 | 2.2125e-5 | 2.2398e-5 | 2.0e-6 |
| 8 | 1.3183e-6 | 1.2598e-6 | 1.2972e-6 | 1.3174e-6 | 1.2567e-6 | 2.1e-7 |
| 12 | 5.6525e-8 | 5.5242e-8 | 5.5730e-8 | 5.6454e-8 | 5.5242e-8 | 3.7e-9 |
| 16 | 3.4772e-9 | 3.4979e-9 | 3.4911e-9 | 3.4766e-9 | 3.4979e-9 | 1.4e-10 |

The change from the rule off (relative L2, last column) is a tenth of the error and falls
with p as the error does: the converted pairs lose their truncation error exactly, and where
that error was anti-correlated with the rest the sum grows. Even converting every pair
raises it at p = 5 (1.2 %) and p = 16 (0.6 %). On the strongly graded tree of
`tests/mpi_exec.rs` (p = 8, crossovers 12 and 6) φ grows 1.5 %. The tests therefore bound
the ratio at 1.1 (`NEAR_ERROR_BOUND`) and the change at the rule-off error; requirement 1's
"at most today's" for C6.9 (signed off at T1) is the sign-off question of §8.

## 7. The build

The rule builds the plan twice (once to number the leaves and find the W rows' ghost
leaves, once with the rule) and adds the counts' exchange. At one rank the build grows by
51–70 %: from 1.0–1.9 s to 1.6–3.0 s on the M3 Max, from 1.4–2.8 s to 2.3–4.3 s on
locust and from 2.3–4.5 s to 3.8–7.0 s on Kathleen (the plan part 1.35 → 2.90 s for the
Plummer sphere at f64 p = 8 on Kathleen). At the node's ranks it grows by 22–44 ms on
locust, 6–151 ms on Kathleen and up to 218 ms on the M3 Max, whose 12-rank builds also
vary by ±25 ms between launches (a recut makes the rule's plan again,
`BuildTimings::cut`). The first evaluation repays it in 67 of the 72 configurations of
§4; the other five are the small problems at the node's ranks (the Plummer sphere and the
clusters at p = 3 on 72 locust ranks: 22–27 ms more build against 18–24 ms saved per
evaluation; the clusters at f64 p = 3 on 12 M3 Max ranks, 117 against 86 ms), which the
second evaluation repays.

## 8. For sign-off and T11

- **Question 1, C6.9's accuracy criterion.** As signed off (requirement 1): "errors against
  the direct sum at most the existing path's". Measured: met on C6.9's three problems, not
  in general (+0.13 % on the clusters, +2.1 % on a small test tree, §6), since removing
  some truncation errors can undo cancellation. *Proposed*: "within 1.1× the existing
  path's errors against the direct sum" (C6.5's form), as the tests assert, with "at most"
  kept as the measured outcome on the C3.2 cube, the C3.3 Plummer sphere and the cube at
  2 × 10⁶.
- **Question 2, the interface.** `Plan::with_near_rule` beside an unchanged `Plan::new`
  (the brief says "given to `Plan::new`"): keeps the call sites of `nd-fmm-kernels`,
  `nd-fmm-validate` and the spikes as they are. *Proposed*: keep it.
- **For T11**: the cost model's rule as the default (1.3–5.9× on every configuration
  measured; the one-rank outputs change, requirement 6); the cost model refitted with it
  (P2P's share of the leaves grows; the second cut weighs the converted plan already); the
  counts without the first plan (number the leaves and the W rows' ghost leaves in one
  pass) to halve the rule's build cost.

## 9. The checks

On the M3 Max: `cargo fmt --all` (and `-- --check`); `cargo clippy -- -D warnings`,
`cargo clippy --examples -- -D warnings`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings`,
`cargo check -p nd-fmm-exec --features cuda`, `cargo doc --no-deps`: clean.
`RUST_MIN_STACK=8388608 cargo test --workspace`: pass (it includes `-p nd-fmm-plan` and
`-p nd-fmm-exec`; `mpi_exec` 171 s at one rank in debug, of which T12's checks 31 s;
`invariance` 69 s, 39 s before). `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks:
every executable and example passes (`mpi_exec` 230 s and 193 s, `invariance` 56 s and
69 s; CI's limit is 600 s per launch); `mpi_regressions` and `mpi_exec` at 8 ranks: pass
(11 s, 179 s). `tests/multi_rank.rs` in release at 1, 2, 4 and 8 ranks: pass
(34, 46, 41 and 39 min). `cargo test -p nd-fmm-validate --test scaling` at 1 and 2 ranks: pass.
On Kathleen: `tools/kathleen/submit.sh correctness.sbatch` at 2 nodes (job 605534,
node-c11b-[015-016], 39 min): the MPI test list at 2, 4, 8 and 80 ranks, invariance at 80
against 40, 20 and 10, and the smoke run: every step passes.
Not run: the device test suites (no device code changed; Metal checked by hand, §2), CUDA
on locust.

## 10. The raw data

One file per launch, in `bench-results/t12/` on the M3 Max (ignored by git): `kathleen-near-605533/`
(48 launches), `kathleen-weak-605535/` (10), `locust-near/` (48), `m3max-near/` (48);
on Kathleen also in the root's `logs/scaling-605533/`, `logs/scaling-605535/`. The full
tables of §4 (the summary fields of each pair of launches):

### M3 Max

| problem | ranks | strategy | eval off → on (ms) | × | leaves off → on | downward off → on | M2P + P2L off → on | P2P off → on | φ error off → on (ratio) | ∇φ ratio | W conv / pairs | X conv / pairs | N_W / N_X | build off → on (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clusters-n1000000-f32-p3 | r12x1 | Dense | 209 → 110 | 1.90 | 135 → 52 | 82 → 51 | 129 → 2 | 32 → 45 | 1.582e-3 → 1.584e-3 (1.0013) | 0.9953 | 298564 / 298564 | 298564 / 298564 | 187.7 / 91.7 | 297 → 274 |
| clusters-n1000000-f32-p3 | r1x1 | Dense | 2188 → 1118 | 1.96 | 1337 → 534 | 811 → 545 | 1221 → 12 | 321 → 458 | 1.582e-3 → 1.584e-3 (1.0013) | 0.9953 | 298564 / 298564 | 298564 / 298564 | 187.7 / 91.7 | 1092 → 1842 |
| clusters-n1000000-f32-p8 | r12x1 | Rotation | 1345 → 915 | 1.47 | 488 → 86 | 950 → 808 | 578 → 10 | 30 → 48 | 1.033e-5 → 1.033e-5 (1.0000) | 0.9927 | 298564 / 298564 | 298564 / 298564 | 945.2 / 445.3 | 281 → 436 |
| clusters-n1000000-f32-p8 | r1x1 | Dense | 13129 → 7875 | 1.67 | 4586 → 813 | 8333 → 6874 | 5429 → 51 | 320 → 458 | 1.070e-5 → 1.071e-5 (1.0009) | 0.9927 | 298564 / 298564 | 298564 / 298564 | 950.3 / 464.3 | 1323 → 2062 |
| clusters-n1000000-f64-p3 | r12x1 | Dense | 248 → 162 | 1.53 | 151 → 87 | 96 → 65 | 130 → 3 | 57 → 79 | 1.582e-3 → 1.583e-3 (1.0006) | 0.9951 | 298564 / 298564 | 289696 / 298564 | 107.5 / 35.5 | 304 → 421 |
| clusters-n1000000-f64-p3 | r1x1 | Dense | 2528 → 1563 | 1.62 | 1588 → 888 | 895 → 629 | 1225 → 19 | 570 → 813 | 1.582e-3 → 1.583e-3 (1.0006) | 0.9951 | 298564 / 298564 | 289696 / 298564 | 107.5 / 35.5 | 1089 → 1838 |
| clusters-n1000000-f64-p8 | r12x1 | Rotation | 1413 → 1033 | 1.37 | 538 → 132 | 986 → 867 | 602 → 11 | 53 → 91 | 1.034e-5 → 1.034e-5 (1.0000) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 421.1 / 184.1 | 279 → 431 |
| clusters-n1000000-f64-p8 | r1x1 | Rotation | 15551 → 10318 | 1.51 | 4894 → 1185 | 10362 → 8920 | 5457 → 52 | 570 → 823 | 1.034e-5 → 1.034e-5 (1.0000) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 421.1 / 184.1 | 1086 → 1826 |
| cube-n2000000-f32-p3 | r12x1 | Dense | 743 → 204 | 3.64 | 545 → 133 | 178 → 51 | 588 → 0 | 73 → 121 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 384 → 583 |
| cube-n2000000-f32-p3 | r1x1 | Dense | 8175 → 2141 | 3.82 | 6120 → 1493 | 1975 → 563 | 6577 → 0 | 819 → 1360 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 1681 → 2719 |
| cube-n2000000-f32-p8 | r12x1 | Rotation | 3676 → 1092 | 3.37 | 2104 → 186 | 1533 → 854 | 2715 → 0 | 74 → 124 | 2.360e-5 → 2.357e-5 (0.9987) | 0.9464 | 1015057 / 1015057 | 1015057 / 1015057 | 945.2 / 445.3 | 401 → 590 |
| cube-n2000000-f32-p8 | r1x1 | Dense | 38551 → 9546 | 4.04 | 23005 → 2002 | 15171 → 7173 | 29535 → 0 | 821 → 1359 | 2.365e-5 → 2.360e-5 (0.9979) | 0.9464 | 1015057 / 1015057 | 1015057 / 1015057 | 950.3 / 464.3 | 1919 → 2959 |
| cube-n2000000-f64-p3 | r12x1 | Dense | 861 → 350 | 2.46 | 628 → 252 | 203 → 66 | 614 → 0 | 144 → 239 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 107.5 / 35.5 | 297 → 393 |
| cube-n2000000-f64-p3 | r1x1 | Dense | 9145 → 3432 | 2.66 | 6858 → 2684 | 2124 → 650 | 6629 → 0 | 1570 → 2552 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 107.5 / 35.5 | 1667 → 2745 |
| cube-n2000000-f64-p8 | r12x1 | Rotation | 3907 → 1345 | 2.90 | 2255 → 328 | 1597 → 948 | 2791 → 0 | 149 → 263 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 379 → 597 |
| cube-n2000000-f64-p8 | r1x1 | Rotation | 41875 → 12911 | 3.24 | 24029 → 3196 | 17460 → 9303 | 29988 → 0 | 1569 → 2551 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 1679 → 2724 |
| plummer-n1000000-f32-p3 | r12x1 | Dense | 200 → 113 | 1.77 | 119 → 51 | 69 → 46 | 107 → 1 | 33 → 44 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 200968 / 200968 | 187.7 / 91.7 | 292 → 292 |
| plummer-n1000000-f32-p3 | r1x1 | Dense | 2058 → 1066 | 1.93 | 1290 → 532 | 731 → 494 | 1114 → 3 | 347 → 462 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 200968 / 200968 | 187.7 / 91.7 | 970 → 1632 |
| plummer-n1000000-f32-p8 | r12x1 | Rotation | 1278 → 852 | 1.50 | 437 → 75 | 863 → 743 | 505 → 2 | 34 → 44 | 3.045e-5 → 3.043e-5 (0.9993) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 945.2 / 445.3 | 275 → 290 |
| plummer-n1000000-f32-p8 | r1x1 | Dense | 12102 → 7194 | 1.68 | 4299 → 787 | 7588 → 6226 | 4997 → 13 | 347 → 460 | 3.076e-5 → 3.073e-5 (0.9990) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 950.3 / 464.3 | 1201 → 1851 |
| plummer-n1000000-f64-p3 | r12x1 | Dense | 241 → 167 | 1.44 | 147 → 91 | 81 → 56 | 111 → 1 | 61 → 85 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 198841 / 200968 | 107.5 / 35.5 | 292 → 281 |
| plummer-n1000000-f64-p3 | r1x1 | Dense | 2430 → 1521 | 1.60 | 1574 → 911 | 814 → 567 | 1123 → 6 | 636 → 842 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 198841 / 200968 | 107.5 / 35.5 | 1001 → 1601 |
| plummer-n1000000-f64-p8 | r12x1 | Rotation | 1373 → 967 | 1.42 | 459 → 117 | 886 → 818 | 508 → 2 | 62 → 84 | 3.045e-5 → 3.041e-5 (0.9987) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 421.1 / 184.1 | 230 → 274 |
| plummer-n1000000-f64-p8 | r1x1 | Rotation | 14369 → 9470 | 1.52 | 4674 → 1181 | 9497 → 8097 | 5071 → 14 | 639 → 850 | 3.045e-5 → 3.041e-5 (0.9987) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 421.1 / 184.1 | 962 → 1623 |

### locust

| problem | ranks | strategy | eval off → on (ms) | × | leaves off → on | downward off → on | M2P + P2L off → on | P2P off → on | φ error off → on (ratio) | ∇φ ratio | W conv / pairs | X conv / pairs | N_W / N_X | build off → on (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clusters-n1000000-f32-p3 | r1x1 | Dense | 2844 → 1491 | 1.91 | 1909 → 938 | 872 → 491 | 1626 → 16 | 584 → 840 | 1.582e-3 → 1.584e-3 (1.0013) | 0.9953 | 298564 / 298564 | 298564 / 298564 | 187.7 / 91.7 | 1651 → 2628 |
| clusters-n1000000-f32-p3 | r72x1 | Dense | 44 → 24 | 1.83 | 32 → 16 | 19 → 9 | 32 → 1 | 12 → 15 | 1.582e-3 → 1.584e-3 (1.0013) | 0.9953 | 298564 / 298564 | 298564 / 298564 | 187.7 / 91.7 | 79 → 101 |
| clusters-n1000000-f32-p8 | r1x1 | Dense | 15031 → 8630 | 1.74 | 5899 → 1281 | 8876 → 7091 | 6717 → 65 | 584 → 838 | 1.070e-5 → 1.071e-5 (1.0009) | 0.9927 | 298564 / 298564 | 298564 / 298564 | 950.3 / 464.3 | 1843 → 2848 |
| clusters-n1000000-f32-p8 | r72x1 | Rotation | 276 → 190 | 1.46 | 120 → 25 | 238 → 164 | 142 → 6 | 9 → 16 | 1.036e-5 → 1.035e-5 (0.9990) | 0.9927 | 298564 / 298564 | 298564 / 298564 | 945.2 / 445.3 | 67 → 98 |
| clusters-n1000000-f64-p3 | r1x1 | Dense | 4098 → 3065 | 1.34 | 2926 → 2317 | 1096 → 675 | 1713 → 26 | 1556 → 2218 | 1.582e-3 → 1.583e-3 (1.0006) | 0.9951 | 298564 / 298564 | 289696 / 298564 | 107.5 / 35.5 | 1627 → 2641 |
| clusters-n1000000-f64-p3 | r72x1 | Dense | 71 → 49 | 1.43 | 43 → 36 | 28 → 15 | 33 → 2 | 32 → 35 | 1.582e-3 → 1.583e-3 (1.0006) | 0.9951 | 298564 / 298564 | 289696 / 298564 | 107.5 / 35.5 | 81 → 108 |
| clusters-n1000000-f64-p8 | r1x1 | Rotation | 21056 → 14974 | 1.41 | 6918 → 2686 | 13860 → 12012 | 6923 → 66 | 1558 → 2245 | 1.034e-5 → 1.034e-5 (1.0000) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 421.1 / 184.1 | 1625 → 2623 |
| clusters-n1000000-f64-p8 | r72x1 | Rotation | 318 → 226 | 1.41 | 134 → 52 | 258 → 183 | 146 → 6 | 26 → 43 | 1.034e-5 → 1.034e-5 (1.0000) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 421.1 / 184.1 | 67 → 100 |
| cube-n2000000-f32-p3 | r1x1 | Dense | 11127 → 3357 | 3.31 | 8413 → 2710 | 2585 → 508 | 8787 → 0 | 1533 → 2538 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 2642 → 4102 |
| cube-n2000000-f32-p3 | r72x1 | Dense | 159 → 50 | 3.18 | 120 → 39 | 39 → 8 | 127 → 0 | 23 → 36 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 80 → 123 |
| cube-n2000000-f32-p8 | r1x1 | Dense | 47008 → 11343 | 4.14 | 29180 → 3337 | 17315 → 7503 | 36659 → 0 | 1539 → 2545 | 2.365e-5 → 2.360e-5 (0.9979) | 0.9464 | 1015057 / 1015057 | 1015057 / 1015057 | 950.3 / 464.3 | 2848 → 4312 |
| cube-n2000000-f32-p8 | r72x1 | Rotation | 714 → 218 | 3.27 | 424 → 49 | 308 → 163 | 532 → 0 | 23 → 37 | 2.361e-5 → 2.355e-5 (0.9975) | 0.9464 | 1015057 / 1015057 | 1015057 / 1015057 | 945.2 / 445.3 | 77 → 117 |
| cube-n2000000-f64-p3 | r1x1 | Dense | 14810 → 8153 | 1.82 | 11604 → 7280 | 3042 → 711 | 9336 → 0 | 4428 → 7111 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 107.5 / 35.5 | 2633 → 4111 |
| cube-n2000000-f64-p3 | r72x1 | Dense | 212 → 118 | 1.79 | 165 → 102 | 46 → 12 | 135 → 0 | 66 → 99 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 107.5 / 35.5 | 80 → 124 |
| cube-n2000000-f64-p8 | r1x1 | Rotation | 55710 → 20690 | 2.69 | 32235 → 7881 | 22924 → 12267 | 37652 → 0 | 4420 → 7092 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 2621 → 4085 |
| cube-n2000000-f64-p8 | r72x1 | Rotation | 794 → 304 | 2.62 | 471 → 117 | 338 → 184 | 552 → 0 | 68 → 105 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 78 → 117 |
| plummer-n1000000-f32-p3 | r1x1 | Dense | 2727 → 1445 | 1.89 | 1867 → 941 | 798 → 443 | 1496 → 4 | 640 → 852 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 200968 / 200968 | 187.7 / 91.7 | 1440 → 2305 |
| plummer-n1000000-f32-p3 | r72x1 | Dense | 42 → 25 | 1.70 | 30 → 16 | 16 → 8 | 29 → 1 | 16 → 14 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 200968 / 200968 | 187.7 / 91.7 | 64 → 89 |
| plummer-n1000000-f32-p8 | r1x1 | Dense | 13904 → 7944 | 1.75 | 5571 → 1258 | 8080 → 6432 | 6196 → 17 | 639 → 850 | 3.076e-5 → 3.073e-5 (0.9990) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 950.3 / 464.3 | 1643 → 2542 |
| plummer-n1000000-f32-p8 | r72x1 | Rotation | 261 → 169 | 1.54 | 110 → 24 | 225 → 147 | 134 → 3 | 13 → 16 | 3.045e-5 → 3.043e-5 (0.9993) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 945.2 / 445.3 | 64 → 86 |
| plummer-n1000000-f64-p3 | r1x1 | Dense | 4118 → 3122 | 1.32 | 3029 → 2414 | 1015 → 632 | 1578 → 8 | 1759 → 2325 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 198841 / 200968 | 107.5 / 35.5 | 1432 → 2321 |
| plummer-n1000000-f64-p3 | r72x1 | Dense | 74 → 50 | 1.48 | 54 → 38 | 21 → 14 | 31 → 1 | 46 → 36 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 198841 / 200968 | 107.5 / 35.5 | 67 → 94 |
| plummer-n1000000-f64-p8 | r1x1 | Rotation | 19687 → 13882 | 1.42 | 6736 → 2739 | 12686 → 10879 | 6376 → 17 | 1762 → 2334 | 3.045e-5 → 3.041e-5 (0.9987) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 421.1 / 184.1 | 1439 → 2327 |
| plummer-n1000000-f64-p8 | r72x1 | Rotation | 297 → 214 | 1.39 | 130 → 53 | 236 → 167 | 142 → 3 | 39 → 47 | 3.045e-5 → 3.041e-5 (0.9987) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 421.1 / 184.1 | 61 → 87 |

### Kathleen, one node (job 605533)

| problem | ranks | strategy | eval off → on (ms) | × | leaves off → on | downward off → on | M2P + P2L off → on | P2P off → on | φ error off → on (ratio) | ∇φ ratio | W conv / pairs | X conv / pairs | N_W / N_X | build off → on (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clusters-n1000000-f32-p3 | r1x1 | Dense | 4936 → 1803 | 2.74 | 3388 → 1127 | 1403 → 538 | 3435 → 33 | 641 → 922 | 1.582e-3 → 1.584e-3 (1.0013) | 0.9953 | 298564 / 298564 | 298564 / 298564 | 187.7 / 91.7 | 2554 → 4319 |
| clusters-n1000000-f32-p3 | r40x1 | Dense | 194 → 75 | 2.57 | 144 → 47 | 64 → 28 | 157 → 7 | 36 → 38 | 1.582e-3 → 1.584e-3 (1.0013) | 0.9953 | 298564 / 298564 | 298564 / 298564 | 187.7 / 91.7 | 256 → 322 |
| clusters-n1000000-f32-p8 | r1x1 | Dense | 32409 → 17893 | 1.81 | 11666 → 1847 | 20170 → 15481 | 14959 → 142 | 653 → 938 | 1.070e-5 → 1.071e-5 (1.0009) | 0.9927 | 298564 / 298564 | 298564 / 298564 | 950.3 / 464.3 | 3157 → 4881 |
| clusters-n1000000-f32-p8 | r40x1 | Rotation | 1359 → 841 | 1.62 | 550 → 81 | 1163 → 752 | 693 → 24 | 24 → 42 | 1.033e-5 → 1.034e-5 (1.0010) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 945.2 / 445.3 | 210 → 313 |
| clusters-n1000000-f64-p3 | r1x1 | Dense | 6096 → 3312 | 1.84 | 4168 → 2185 | 1768 → 969 | 3425 → 52 | 1391 → 1986 | 1.582e-3 → 1.583e-3 (1.0006) | 0.9951 | 298564 / 298564 | 289696 / 298564 | 107.5 / 35.5 | 2547 → 4309 |
| clusters-n1000000-f64-p3 | r40x1 | Dense | 244 → 132 | 1.84 | 163 → 85 | 110 → 55 | 159 → 8 | 81 → 76 | 1.582e-3 → 1.583e-3 (1.0006) | 0.9951 | 298564 / 298564 | 289696 / 298564 | 107.5 / 35.5 | 267 → 341 |
| clusters-n1000000-f64-p8 | r1x1 | Rotation | 40904 → 27091 | 1.51 | 12296 → 2901 | 27951 → 23511 | 14629 → 138 | 1391 → 2008 | 1.034e-5 → 1.034e-5 (1.0000) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 421.1 / 184.1 | 2540 → 4329 |
| clusters-n1000000-f64-p8 | r40x1 | Rotation | 1467 → 975 | 1.51 | 568 → 124 | 1220 → 845 | 672 → 23 | 52 → 92 | 1.034e-5 → 1.034e-5 (1.0000) | 0.9925 | 298564 / 298564 | 298564 / 298564 | 421.1 / 184.1 | 207 → 306 |
| cube-n2000000-f32-p3 | r1x1 | Dense | 21411 → 3936 | 5.44 | 15850 → 3098 | 5281 → 554 | 18661 → 0 | 1550 → 2736 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 3961 → 6419 |
| cube-n2000000-f32-p3 | r40x1 | Dense | 871 → 148 | 5.89 | 633 → 111 | 216 → 27 | 754 → 0 | 59 → 98 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 248 → 392 |
| cube-n2000000-f32-p8 | r1x1 | Dense | 101710 → 21362 | 4.76 | 58634 → 4396 | 41980 → 15842 | 81567 → 0 | 1550 → 2768 | 2.365e-5 → 2.361e-5 (0.9983) | 0.9464 | 1015057 / 1015057 | 1015057 / 1015057 | 950.3 / 464.3 | 4491 → 6959 |
| cube-n2000000-f32-p8 | r40x1 | Rotation | 3837 → 1017 | 3.77 | 2094 → 155 | 1759 → 808 | 2931 → 0 | 60 → 101 | 2.360e-5 → 2.357e-5 (0.9987) | 0.9458 | 1015057 / 1015057 | 1015057 / 1015057 | 945.2 / 445.3 | 243 → 393 |
| cube-n2000000-f64-p3 | r1x1 | Dense | 23994 → 7852 | 3.06 | 18194 → 6521 | 5455 → 985 | 18588 → 0 | 3725 → 6173 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 107.5 / 35.5 | 3941 → 6409 |
| cube-n2000000-f64-p3 | r40x1 | Dense | 941 → 297 | 3.17 | 700 → 236 | 230 → 47 | 739 → 0 | 137 → 223 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 107.5 / 35.5 | 245 → 251 |
| cube-n2000000-f64-p8 | r1x1 | Rotation | 111203 → 33391 | 3.33 | 61133 → 7764 | 48777 → 24338 | 79936 → 0 | 3730 → 6173 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 3923 → 6320 |
| cube-n2000000-f64-p8 | r40x1 | Rotation | 4209 → 1223 | 3.44 | 2324 → 281 | 1856 → 887 | 3082 → 0 | 141 → 228 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 248 → 392 |
| plummer-n1000000-f32-p3 | r1x1 | Dense | 4643 → 1716 | 2.71 | 3220 → 1096 | 1284 → 483 | 3159 → 9 | 679 → 910 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 200968 / 200968 | 187.7 / 91.7 | 2294 → 3804 |
| plummer-n1000000-f32-p3 | r40x1 | Dense | 187 → 71 | 2.63 | 130 → 44 | 54 → 26 | 142 → 2 | 38 → 36 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 200968 / 200968 | 187.7 / 91.7 | 211 → 289 |
| plummer-n1000000-f32-p8 | r1x1 | Dense | 29913 → 16374 | 1.83 | 10916 → 1770 | 18417 → 14049 | 13824 → 38 | 702 → 931 | 3.075e-5 → 3.073e-5 (0.9993) | 0.9979 | 200968 / 200968 | 200968 / 200968 | 950.3 / 464.3 | 2828 → 4358 |
| plummer-n1000000-f32-p8 | r40x1 | Rotation | 1216 → 761 | 1.60 | 497 → 73 | 978 → 684 | 641 → 7 | 31 → 40 | 3.047e-5 → 3.045e-5 (0.9993) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 945.2 / 445.3 | 192 → 290 |
| plummer-n1000000-f64-p3 | r1x1 | Dense | 5891 → 3246 | 1.81 | 4113 → 2224 | 1617 → 866 | 3162 → 16 | 1538 → 2043 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 198841 / 200968 | 107.5 / 35.5 | 2316 → 3799 |
| plummer-n1000000-f64-p3 | r40x1 | Dense | 229 → 133 | 1.71 | 154 → 84 | 80 → 48 | 141 → 5 | 81 → 77 | 4.706e-3 → 4.695e-3 (0.9977) | 0.9967 | 200968 / 200968 | 198841 / 200968 | 107.5 / 35.5 | 237 → 300 |
| plummer-n1000000-f64-p8 | r1x1 | Rotation | 37671 → 24830 | 1.52 | 11656 → 2878 | 25369 → 21296 | 13409 → 37 | 1536 → 2052 | 3.045e-5 → 3.041e-5 (0.9987) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 421.1 / 184.1 | 2283 → 3816 |
| plummer-n1000000-f64-p8 | r40x1 | Rotation | 1322 → 894 | 1.48 | 521 → 120 | 1028 → 769 | 629 → 7 | 69 → 89 | 3.045e-5 → 3.041e-5 (0.9987) | 0.9982 | 200968 / 200968 | 200968 / 200968 | 421.1 / 184.1 | 190 → 285 |

### Kathleen, two nodes (job 605535)

| problem | ranks | strategy | eval off → on (ms) | × | leaves off → on | downward off → on | M2P + P2L off → on | P2P off → on | φ error off → on (ratio) | ∇φ ratio | W conv / pairs | X conv / pairs | N_W / N_X | build off → on (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cube-n1000000-f32-p3 | r40x1 | Dense | 40 → 41 | 1.00 | 27 → 27 | 9 → 9 | 0 → 0 | 21 → 21 | 2.924e-3 → 2.924e-3 (1.0000) | 1.0000 | 0 / 0 | 0 / 0 | 187.7 / 91.7 | 100 → 152 |
| cube-n1000000-f64-p8 | r40x1 | Rotation | 500 → 498 | 1.00 | 75 → 74 | 404 → 402 | 0 → 0 | 51 → 51 | 1.828e-5 → 1.828e-5 (1.0000) | 1.0000 | 0 / 0 | 0 / 0 | 421.1 / 184.1 | 129 → 158 |
| cube-n2000000-f32-p3 | r80x1 | Dense | 424 → 73 | 5.77 | 323 → 56 | 108 → 12 | 385 → 0 | 29 → 50 | 3.698e-3 → 3.680e-3 (0.9951) | 0.9423 | 1015057 / 1015057 | 1015057 / 1015057 | 187.7 / 91.7 | 157 → 240 |
| cube-n2000000-f64-p8 | r80x1 | Rotation | 1970 → 601 | 3.28 | 1119 → 144 | 901 → 439 | 1489 → 0 | 71 → 115 | 2.361e-5 → 2.354e-5 (0.9970) | 0.9421 | 1015057 / 1015057 | 1015057 / 1015057 | 421.1 / 184.1 | 149 → 226 |
