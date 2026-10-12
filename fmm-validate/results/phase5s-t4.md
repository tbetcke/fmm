# The plan over the locally essential set (Phase 5S T4)

S1 in `nd-fmm-plan`, as docs/phase5s/T4-plan-locally-essential.md asks and
docs/design/scale-out.md §3.4–§3.7 (signed off) specifies: `Plan::new` over the octree's
locally essential set H and the top-tree keys T, the held-key check in place of "every
coarse block held", and the exchanges built from it. Run on 2026-10-12 on the M3 Max, locust
(GH200) and Kathleen.

**Measured** means run here and read from the output; timings are reported, never asserted.
locust is a shared node: its load is in each output file before and after every launch.
Kathleen nodes are exclusive; every job is named.

## Summary

- **Every output bit is Phase 5's.** `multi_rank` gives 0 values differing from the
  one-rank `Fmm` in all 30 rows at 2, 4 and 8 ranks (M3 Max and locust); T2's invariance
  check gives 0 differing values at 1, 2, 4 and 8 ranks (`invariance`), at 512 against 64
  on locust and at 160 against 80, 40 and 20 on Kathleen. The list oracle, the exchange
  checks and the recording operator of `mpi_regressions` pass unchanged on every scenario,
  and on every timed sweep the lists, the exchange traffic and the work per rank are
  identical before and after (0 of 825 rows differ).
- **The plan holds H ∪ T, exactly.** A new check in `mpi_regressions` compares the held
  keys on every rank with H ∪ T computed by brute force from the gathered tree and
  blocks, kinds included (Phase 5's kinds: every ancestor of a block `Global`), at 1–8
  ranks: nothing is held that no list, halo or parent-child step names, apart from the
  top-tree keys the design keeps until T5 (§3.7).
- **Held boxes per rank** (2.5 × 10⁴ points per rank): H stays at 3,585–7,025 (cube),
  6,569–9,177 (Plummer) and 6,497–7,825 (clusters) from 40 to 640 Kathleen ranks, at most
  1.96 times its 40-rank value (T1's bound is 2), while Phase 5's set grows to 42,417,
  50,377 and 59,097. The plan's total H ∪ T still grows with P through T, as the design
  accepted for T4: at 512 locust ranks 39,865 → 23,833 (cube), 43,377 → 24,057 (Plummer),
  54,753 → 26,657 (clusters); on Phase 5 T10's own case (the N = 10⁶ cube on 512 ranks)
  37,449 → 20,897, with H 1,481. On the cube below 512 ranks H ∪ T equals Phase 5's set, as
  §3.7 predicted: there the top tree is every block's ancestors only.
- **`Plan::new`** is faster on every timed run: at 72 locust ranks 13.6 → 9.9 ms (cube),
  34.1 → 27.5 ms (Plummer), 37.0 → 28.4 ms (clusters); at 160 Kathleen ranks 71.8 → 62.1,
  83.8 → 60.1 and 95.6 → 67.2 ms. (On `main` it still rebuilt Phase 5's key set with two
  all-gathers and an all-to-all-v; T4 adds no collective.) Its arrays shrink with the held
  set: at 512 locust ranks 4.4 → 3.5, 9.4 → 8.3 and 8.8 → 7.2 MB per rank (max).
- **The evaluation does not change**, as it should not before T5: the lists, exchanges
  and work are identical, and the timed difference is an order effect of the sweep (the
  second launch of each pair is 1–3% slower, whichever binary it is; §5).
- `CoarseExchange` and the global upward pass are untouched: every block stays held (T),
  so the gather fills every block's slot as before.

## 1. The change

`fmm-plan/src/plan.rs` (`Plan::new`, `held_keys`, `from_key_types`, `validate_tree`,
`first_pass`, the module docs), `fmm-plan/src/phase5_keys.rs` (removed),
`fmm-plan/tests/mpi_regressions.rs`, `fmm-plan/src/plan_tests.rs`, docs in
`lib.rs`, `index.rs`, `exchange.rs` and `fmm-plan/CLAUDE.md`; `fmm-validate/src/scaling.rs`.
`nd-octree` and `nd-fmm-exec` are unchanged.

1. **The held keys, H ∪ T** (scale-out.md §3.7). `Plan::new` takes the octree's
   `all_keys` (H, since T3) and adds, from the replicated `PartitionMap`, every coarse
   block with the kind `PartitionMap::top_kind` gives it (own block local, another rank's
   block a ghost of its owner) and every ancestor of a block as `Global`. An ancestor of
   other ranks' blocks only, which H holds as a ghost of its owner, is retyped `Global`:
   those are Phase 5's kinds, and the global pass forms the multipole of every `Global`
   box until T5. Local, O(B) for B blocks; T3's transitional adapter (`phase5_keys`,
   which rebuilt Phase 5's set with two all-gathers and an all-to-all-v) is removed, so
   `Plan::new` is back to its two all-reduces.
2. **P2's pruned rows are no longer resolved.** A `Global` box that is not an ancestor of
   an own block gets empty V, X and L2L rows (as before) and now no list resolution and no
   validation of its neighbourhood: H does not hold it. Its global M2M row stays (its
   children are top-tree keys, all in T). Every other row is built as before, so every
   list entry, as a key, and every row order are Phase 5's (§3.6).
3. **The held-key check.** Every list entry of a box with rows, its parent, every child of
   a non-ghost interior box and every own coarse block must be held, else a `PlanError`
   agreed on every rank by the existing all-reduce (`UnheldListEntry`, `MalformedTree`,
   `MissingChild`, `CoarseBlock`). "Every coarse block of every rank held" (redesign §3.5,
   decision 6) is no longer a guarantee of the octree's key set; it holds because T holds
   every block, which `CoarseExchange::new` still checks with its closing all-reduce.
4. **The exchanges** are built from the plan as before: the source exchange over the
   ghost leaves the U- and X-rows name, the multipole exchange over the ghosts of the V-
   and W-rows, the coarse gather over every block. Their neighbours and sizes are
   reported as before (`exchange_traffic`, `exchange_sizes`), and they move exactly what
   they moved before (§6).
5. **`scaling`** (`nd_fmm_validate::scaling`): the held boxes per rank are split into own,
   halo (ghosts below other ranks' blocks), shared ancestors, the other top-tree keys of
   H, and the keys held outside H (`RankRow::held`, `HELD_PARTS`); the first four sum to
   |H|. `RankRow::plan_bytes` counts the plan's arrays from their lengths (box index, leaf
   numbering, every view with its row offsets and batch arrays). For the "before" runs the
   same `scaling.rs` was built into a detached worktree of `main` (bf4f537) as a
   measurement patch, so both sides report the same quantities; there the keys outside H
   are Phase 5's whole replication beyond H.

## 2. Correctness

### 2.1 Tests

- `mpi_regressions` (`check_held_keys`, new): on every rank and scenario, the held keys
  equal H ∪ T computed by brute force (S from the own blocks, same-level neighbours by
  index arithmetic, the children of interior ones, every gathered block and its
  ancestors), and every kind is the one derived independently (own local, every ancestor
  of a block `Global`, on one rank the root `Global` unless a leaf, everything else a
  ghost of its block's rank). The list oracle, the exchange checks, the recording
  operator (every pair once, the same calls and groupings, blocking and overlapped) and
  the redistribution checks are unchanged and pass. On the graded scenario rank 0 prints
  the maxima of |H ∪ T|, |H| and |T|: 705/705/1 at 1 rank, 1,185/1,185/281 at 2,
  1,273/1,225/401 at 3, 1,793/1,753/537 at 4, 2,177/1,873/985 at 8 (M3 Max).
- `plan_tests` (new): blocks on levels 1–4 over a uniform level-4 tree, with `Global`
  boxes down to level 3; for five owners, the plan over the hand-built H ∪ T has every
  row of the plan over the whole map, key for key and in row order, and the same ghost
  leaves. A mutant that resolves the pruned `Global` boxes again fails it
  (`UnheldListEntry` on the rank of octant 0, whose H does not hold a pruned box's
  V-list); with `Global` boxes only to level 2 the ghost-children layer happened to cover
  them, so the test was deepened until the mutant failed.

### 2.2 Runs

| Where | Ranks | What | Result |
| --- | --- | --- | --- |
| M3 Max | 1, 2, 4, 8 | `run-mpi-tests.sh`: `mpi_regressions`, `mpi_exec`, `mpi_threading`, `invariance` at 1, 2, 4, 8; the seven `nd-octree` examples at 2 and 4 | pass |
| M3 Max | 3 | `mpi_regressions` | pass |
| M3 Max | 2, 4, 8 | `multi_rank` (release, ignored) | pass; 0 values differ in all 30 rows |
| M3 Max | 2, 4 | `device_ranks` (release, ignored, CPU runtime) | pass |
| locust | 1, 2, 4, 8 | `run-mpi-tests.sh` (44 launches) | pass |
| locust | 2, 4, 8 | `multi_rank` | pass; 0 values differ in all 30 rows |
| locust | 512 against 64, oversubscribed | `scaling --invariance 8`, the N = 10⁷ cube, f64 p = 8, Rotation | 0 of 4 × 10⁷ values differ |
| Kathleen 521278 | 2 nodes | T2's correctness job: the MPI list at 2, 4, 8 and 80 ranks; invariance at 80 against 40, 20, 10; the smoke run | 8 of 8 steps pass |
| Kathleen 521595 | 160 (4 × 40) | `scaling --invariance 2,4,8` (N = 10⁶): the cube f64 p = 8 and f32 p = 3, the Plummer sphere f64 p = 8 | 0 of 8 × 10⁶ values differ in each of the 9 checks |

## 3. Held boxes per rank

The `scaling` example, release, 2.5 × 10⁴ points per rank (`--n-per-rank 25000`), the
random share, f64 p = 8, Rotation, overlap off; held boxes of the plan per rank, max / mean
over the ranks, `main` → T4, and the T4 run's H.

locust (Open MPI 5.0.10; up to 72 ranks `--map-by slot:PE=1 --bind-to core`, above
`--map-by :OVERSUBSCRIBE --bind-to none`, counted, not timed; the second sweep, 2026-10-12
23:00):

| Ranks | cube: Phase 5 → H ∪ T | H | Plummer: Phase 5 → H ∪ T | H | clusters: Phase 5 → H ∪ T | H |
| ---: | --- | --- | --- | --- | --- | --- |
| 2 | 3,017 / 3,013 → 3,017 / 3,013 | 3,017 / 3,013 | 3,353 / 3,333 → 3,209 / 3,101 | 3,209 / 3,101 | 4,249 / 4,121 → 3,857 / 3,677 | 3,761 / 3,553 |
| 4 | 1,961 / 1,961 → 1,961 / 1,961 | 1,961 / 1,961 | 4,713 / 4,443 → 4,345 / 3,905 | 4,345 / 3,905 | 5,801 / 5,499 → 4,729 / 4,339 | 4,249 / 3,819 |
| 8 | 1,753 / 1,683 → 1,633 / 1,543 | 1,529 / 1,396 | 6,561 / 5,665 → 5,857 / 4,801 | 5,857 / 4,801 | 7,721 / 6,996 → 5,937 / 4,982 | 5,129 / 4,057 |
| 16 | 5,753 / 4,893 → 5,753 / 4,893 | 5,505 / 4,568 | 8,065 / 7,681 → 6,129 / 5,404 | 6,017 / 5,100 | 10,057 / 9,528 → 7,025 / 5,893 | 5,993 / 4,294 |
| 32 | 3,817 / 3,253 → 3,817 / 3,253 | 3,521 / 2,891 | 10,161 / 9,562 → 6,985 / 6,126 | 6,217 / 5,200 | 14,329 / 13,168 → 8,457 / 7,694 | 6,521 / 5,356 |
| 64 | 6,697 / 6,169 → 4,897 / 4,264 | 2,889 / 2,085 | 14,345 / 13,040 → 8,937 / 7,798 | 7,177 / 5,894 | 18,129 / 17,279 → 10,265 / 9,143 | 7,217 / 5,456 |
| 72 | 7,537 / 6,984 → 7,537 / 6,984 | 3,745 / 2,977 | 14,809 / 13,474 → 8,977 / 7,958 | 7,153 / 5,912 | 18,665 / 17,822 → 10,689 / 9,348 | 7,425 / 5,406 |
| 128 | 11,497 / 9,978 → 11,497 / 9,978 | 7,609 / 5,824 | 18,105 / 16,869 → 10,433 / 9,397 | 7,257 / 5,951 | 26,289 / 25,390 → 13,257 / 11,203 | 7,801 / 5,303 |
| 256 | 9,009 / 8,055 → 9,009 / 8,055 | 5,017 / 3,836 | 27,897 / 25,833 → 15,337 / 13,407 | 8,369 / 6,016 | 41,185 / 38,785 → 19,241 / 17,262 | 8,593 / 6,218 |
| 512 | 39,865 / 39,239 → 23,833 / 23,095 | 3,793 / 2,710 | 43,377 / 40,775 → 24,057 / 21,681 | 9,129 / 6,340 | 54,753 / 53,232 → 26,657 / 24,628 | 8,729 / 6,105 |

Kathleen (job 521595, node-c11a-[031,048,057,073]; Open MPI 4.1.6 over PSM2,
`PSM2_TID=0`; 40, 80 and 160 ranks as 40 × 1 per node, `--map-by ppr:20:socket:PE=1
--bind-to core`; 320 and 640 oversubscribed on the 4 nodes, `--oversubscribe --map-by
ppr:80:node` / `ppr:160:node --bind-to none -x PSM2_SHAREDCONTEXTS=1`, counted, not timed):

| Ranks | cube: Phase 5 → H ∪ T | H | Plummer: Phase 5 → H ∪ T | H | clusters: Phase 5 → H ∪ T | H |
| ---: | --- | --- | --- | --- | --- | --- |
| 40 | 3,881 / 3,317 → 3,881 / 3,317 | 3,585 / 2,972 | 11,561 / 10,700 → 7,521 / 6,772 | 6,569 / 5,548 | 16,145 / 14,712 → 9,337 / 8,275 | 6,497 / 5,324 |
| 80 | 9,361 / 8,329 → 9,361 / 8,329 | 5,537 / 4,295 | 15,497 / 14,121 → 9,457 / 8,307 | 7,433 / 6,024 | 19,121 / 18,145 → 10,841 / 9,433 | 6,905 / 5,270 |
| 160 | 10,977 / 9,771 → 10,977 / 9,771 | 7,025 / 5,624 | 20,105 / 18,598 → 11,401 / 10,008 | 7,425 / 5,855 | 31,265 / 29,289 → 14,489 / 12,750 | 7,569 / 5,593 |
| 320 | 8,737 / 7,888 → 8,737 / 7,888 | 4,713 / 3,664 | 32,561 / 29,922 → 17,921 / 15,633 | 8,889 / 6,148 | 45,305 / 43,223 → 21,393 / 19,637 | 8,409 / 6,322 |
| 640 | 42,417 / 41,470 → 42,417 / 41,470 | 6,273 / 4,958 | 50,377 / 47,859 → 27,809 / 25,530 | 9,177 / 6,359 | 59,097 / 57,442 → 28,465 / 26,890 | 7,825 / 5,896 |

- **Against T1's bound** (scale-out.md §3.1: at fixed N/P, H's maximum at most twice its
  40-rank value up to 640 Kathleen ranks; met by T4 on H, by T5/T6 on the total, §12
  question 1): the largest ratio is 1.96 (cube at 160 ranks), Plummer 1.40 and clusters
  1.20 at 640. On locust H's maximum grows 2.5× (cube), 1.56× and 1.70× from 8 to 512
  ranks; the cube's H follows the level of its uniform blocks (1,529 at 8, 7,609 at 128,
  3,793 at 512) rather than P. The Phase 5 set grows 10.9×, 4.4× and 3.7× from 40 to 640
  Kathleen ranks. These H values equal T3's octree counts (phase5s-t3.md §3) key for key,
  as they must: the plan holds the octree's H.
- **H ∪ T** still grows with P, through T (about 1.15 B keys for B blocks, O(P)): from
  40 to 640 Kathleen ranks 3,881 → 42,417 (cube), 7,521 → 27,809 and 9,337 → 28,465. On the
  cube below 512 ranks it equals Phase 5's set: with uniform blocks every key Phase 5 held
  beyond H is a top-tree key, so T4 removes nothing there and T5 all of it (§3.7); on
  Kathleen this holds up to 640 ranks. Where Phase 5 replicated neighbourhoods, T4 removes
  35–52% of the held boxes at 40–640 Kathleen ranks (Plummer, clusters) and 40–51% at
  512 locust ranks (all three).
- **Phase 5 T10's case** (the N = 10⁶ cube on 512 locust ranks, oversubscribed, the
  37,449 every rank held in phase5-gh200.md §10): 37,449 → 20,897 per rank (max), H
  1,481 (own 80, halo 432, shared ancestors 14, other top keys 1,051), plan arrays 2.6 →
  1.7 MB.

The split of H ∪ T, max over the ranks (each part's own maximum): own / halo / shared
ancestors / other top-tree keys in H / top-tree keys outside H:

| Ranks | cube | Plummer | clusters |
| ---: | --- | --- | --- |
| locust 72 | 993 / 1,984 / 13 / 820 / 4,328 | 2,227 / 4,136 / 21 / 1,247 / 2,464 | 2,890 / 3,752 / 77 / 1,034 / 4,144 |
| locust 512 | 739 / 2,112 / 13 / 1,023 / 20,792 | 2,541 / 5,032 / 15 / 1,823 / 16,216 | 2,907 / 4,496 / 81 / 1,718 / 18,960 |
| Kathleen 160 | 1,898 / 4,424 / 10 / 750 / 4,400 | 2,030 / 4,344 / 18 / 1,416 / 4,744 | 2,587 / 4,160 / 71 / 1,348 / 7,496 |
| Kathleen 640 | 1,779 / 3,504 / 15 / 1,297 / 37,040 | 2,540 / 5,024 / 15 / 1,885 / 20,088 | 2,420 / 4,480 / 64 / 1,725 / 21,448 |

The last column is what T5 removes (it keeps only the replicated top R, at most 4,096
boxes).

## 4. `Plan::new`: time and memory

The slowest rank's `Plan::new`, the median of 3 builds per launch (release), `main` → T4,
and the plan's arrays per rank (`RankRow::plan_bytes`: box index, leaf numbering, every
view with its offsets and batches), max / mean. On `main` `Plan::new` still contained T3's
adapter (two all-gathers and an all-to-all-v rebuilding Phase 5's key set, which Phase 5
itself did in `Octree::new`), so the sum of `Octree::new` and `Plan::new` is given as well.

locust (second sweep; oversubscribed ranks not timed):

| Ranks | cube: ms | cube: MB | Plummer: ms | Plummer: MB | clusters: ms | clusters: MB |
| ---: | --- | --- | --- | --- | --- | --- |
| 2 | 22.4 → 21.5 | 5.0 / 5.0 → 5.0 / 5.0 | 19.0 → 17.5 | 3.8 / 3.6 → 3.8 / 3.6 | 24.8 → 22.8 | 4.7 / 4.3 → 4.7 / 4.3 |
| 8 | 6.4 → 5.8 | 1.4 / 1.3 → 1.4 / 1.3 | 20.4 → 18.0 | 4.0 / 3.7 → 3.9 / 3.7 | 26.5 → 23.5 | 5.0 / 4.3 → 4.9 / 4.2 |
| 32 | 15.9 → 14.4 | 3.3 / 3.0 → 3.3 / 3.0 | 23.8 → 19.9 | 4.8 / 4.0 → 4.6 / 3.8 | 30.7 → 25.6 | 6.2 / 4.7 → 5.8 / 4.4 |
| 72 | 13.6 → 9.9 | 2.2 / 1.9 → 2.2 / 1.9 | 34.1 → 27.5 | 6.5 / 4.6 → 6.2 / 4.3 | 37.0 → 28.4 | 6.2 / 4.8 → 5.8 / 4.3 |
| 128 | not timed | 7.5 / 6.7 → 7.5 / 6.7 | not timed | 6.0 / 4.8 → 5.6 / 4.4 | not timed | 6.5 / 4.8 → 5.8 / 4.0 |
| 256 | not timed | 4.1 / 3.6 → 4.1 / 3.6 | not timed | 8.2 / 5.3 → 7.5 / 4.6 | not timed | 8.8 / 6.4 → 7.6 / 5.2 |
| 512 | not timed | 4.4 / 4.0 → 3.5 / 3.1 | not timed | 9.4 / 6.5 → 8.3 / 5.4 | not timed | 8.8 / 7.2 → 7.2 / 5.6 |

`Octree::new` + `Plan::new`, slowest rank, ms (locust, second sweep): at 2 ranks 26.2 →
25.2 / 23.9 → 22.4 / 30.7 → 28.7; at 8 9.4 → 8.9 / 26.3 → 23.8 / 33.5 → 30.3; at 72 20.5 →
16.7 / 44.4 → 37.7 / 48.9 → 40.4 (cube / Plummer / clusters).

Kathleen (times from job 521595, `main` and T4 interleaved; the "before" plan bytes from job
522457, which reran only `main`'s side because its first build had not picked up the
measurement patch, §7; its held boxes equal 521595's on every run):

| Ranks | cube: ms | cube: MB | Plummer: ms | Plummer: MB | clusters: ms | clusters: MB |
| ---: | --- | --- | --- | --- | --- | --- |
| 40 | 30.8 → 29.2 | 2.8 / 2.5 → 2.8 / 2.5 | 59.5 → 51.2 | 5.4 / 4.3 → 5.1 / 4.0 | 66.5 → 54.4 | 5.8 / 4.9 → 5.5 / 4.5 |
| 80 | 45.0 → 34.8 | 3.3 / 2.8 → 3.3 / 2.8 | 82.4 → 65.3 | 6.7 / 4.7 → 6.3 / 4.4 | 77.9 → 61.4 | 6.0 / 4.7 → 5.6 / 4.2 |
| 160 | 71.8 → 62.1 | 6.0 / 5.5 → 6.0 / 5.5 | 83.8 → 60.1 | 6.6 / 4.8 → 6.1 / 4.3 | 95.6 → 67.2 | 7.7 / 5.3 → 6.7 / 4.4 |
| 320 | not timed | 3.2 / 3.0 → 3.2 / 3.0 | not timed | 9.0 / 5.7 → 8.2 / 4.9 | not timed | 8.7 / 6.8 → 7.4 / 5.4 |
| 640 | not timed | 5.7 / 5.2 → 5.7 / 5.2 | not timed | 9.9 / 7.0 → 8.6 / 5.7 | not timed | 9.1 / 7.2 → 7.4 / 5.5 |

`Octree::new` + `Plan::new` (Kathleen, job 521595): at 40 ranks 43.9 → 41.4 / 77.8 → 71.5 /
88.1 → 74.9 ms, at 80 63.1 → 53.0 / 106.7 → 90.9 / 103.2 → 86.6 ms, at 160 95.4 → 86.7 /
115.2 → 87.4 ms and, for the clusters, 133.3 → 255.2 ms, where T4's launch hit the
150 ms `Octree::new` plateau at 160 ranks that T3 reported (phase5s-t3.md §4.3; 186 ms
against 36 ms; T4 does not touch `Octree::new`). At 320 and 640 oversubscribed ranks `Plan::new` is not timed.

The plan's memory falls less than its held boxes: most of its bytes are the views of the
own rows (V above all), which T4 keeps unchanged, and a held box costs 32 B in the index
plus 4 B of row offsets in each of six box views. What T removes in T5 is mostly index and
offsets, plus 2 n_c s per box of multipole and local slots in the evaluator (the memory
model's coefficients: at 512 locust ranks, p = 8, f64, 51.7 → 30.9 MB per rank on the
cube).

## 5. The evaluation

T4 changes no list, no exchange and no work: the evaluation should take the same time
until T5 removes the coarse gather and the global M2M. The evaluation wall time (median of
5, max over the ranks), locust, `main` → T4, in the two sweeps: the first ran `main`
first in every pair, the second T4 first:

| Ranks | cube, first | cube, second | Plummer, first | Plummer, second | clusters, first | clusters, second |
| ---: | --- | --- | --- | --- | --- | --- |
| 2 | 395.4 → 409.3 | 396.5 → 389.8 | 560.0 → 579.0 | 578.1 → 564.3 | 552.4 → 580.7 | 565.4 → 553.4 |
| 8 | 216.3 → 220.5 | 219.7 → 216.3 | 513.9 → 527.7 | 525.7 → 515.6 | 573.0 → 589.3 | 583.4 → 572.6 |
| 32 | 287.4 → 293.2 | 289.9 → 285.7 | 542.4 → 560.3 | 556.4 → 543.8 | 611.0 → 627.2 | 628.6 → 616.7 |
| 72 | 446.1 → 459.2 | 453.6 → 444.4 | 616.4 → 641.6 | 630.1 → 618.1 | 690.7 → 701.5 | 706.1 → 691.3 |

(ms.) The second launch of each pair is 1–4% slower, whichever binary it is, even on the
cube at 2 ranks, where both plans hold the same keys. Taken over both orders the two
builds agree within 1%. Kathleen (job 521595, `main` first in every pair) shows the same
pattern, 0–3% slower for T4 at 40 and 80 ranks (cube 638.8 → 648.2 and 2,167.9 → 2,169.2
ms at 40 and 80), except the Plummer sphere at 160 ranks (1,762.0 → 1,631.8 ms), where
`main`'s launch spent 178 ms in the global upward pass against 25 ms (the coarse gather;
the same launch effect at 160 ranks as T3's `Octree::new` plateau, phase5s-t3.md §4.3,
which this job also hit in T4's clusters launch: `Octree::new` 186 ms against 36 ms).
At 160 Kathleen ranks the evaluations take 1.24–1.97 s (cube, Plummer, clusters), of
which the global pass is 19–39 ms when no plateau hits.

## 6. The exchanges

`SourceExchange` and `MultipoleExchange` are built from the plan's rows as before;
`CoarseExchange` from every rank's own blocks, all of which T keeps held. In every sweep
(locust 2–512, Kathleen 40–640, three workloads) the reduced rows of the near, V, W and X
pairs, the leaves, the ghost sources, the source exchange's messages and bytes, the
multipole exchange's messages and bytes and the coarse gather's bytes are identical
before and after (0 of 825 rows differ): T4 exchanges exactly what Phase 5 exchanged.

## 7. Checks

M3 Max (sandboxed): `cargo fmt --all`; the root checks (`cargo fmt -- --check`, `cargo
clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `RUST_MIN_STACK=8388608
cargo test`: 719 passed, `cargo doc --no-deps`) and the stricter workspace checks (`cargo
clippy --workspace --all-targets -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test
--workspace`: 764 passed, 0 failed); `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p
nd-fmm-plan -p nd-fmm-validate`; `cargo clippy -p nd-fmm-validate --all-targets -- -D
warnings`, `cargo test -p nd-fmm-validate --test scaling`. MPI runs as in §2.2.

Kathleen (2026-10-12):

| Job | Nodes | What | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 521187 | 1 (node-c11a-031) | release builds of `scaling`, branch and `main` | 1 s | 1:59 | 0.03 | 2 of 2 steps pass |
| 521270 | 4 (node-c11a-[011,031,057,087]) | the sweep, first attempt | 6 s | 4:43 | 0.31 | the 40-rank launches pass; every launch across nodes fails in `MPI_Init`: node-c11a-011's HFI contexts busy (tools/kathleen/README.md: the node of T2's first attempt) |
| 521278 | 2 (node-c11a-[048,073]) | T2's correctness job (`test`) | 2 s | 21:40 | 0.72 | 8 of 8 steps pass |
| 521595 | 4 (node-c11a-[031,048,057,073]), `--exclude=node-c11a-011` | the sweep (40–640 ranks, before and after) and invariance at 160 | 17 min 14 s | 17:02 | 1.14 | 36 of 36 steps pass |
| 522457 | 4 (node-c11a-[031,048,073,087]), `--exclude=node-c11a-011` | `main`'s side of the sweep again, rebuilt (§4) | 21 s | 8:49 | 0.59 | 33 of 33 steps pass |

2.79 node-hours in all, of T4's 20 (README, "Node-hours"). The first build of `main`'s
tree on Kathleen raced the measurement patch: the patched `scaling.rs` arrived by rsync
with a modification time older than the binary job 521187 had just built, so cargo did not
rebuild it, and 521595's `main` runs lack the plan-bytes line; 522457 touched the file and
reran that side. locust's trees were built after the patch and are not affected.

locust (2026-10-12, Open MPI 5.0.10; one node, no other user's process in any load
check; the 1-minute load before the timed launches, up to 44, came from the sweep's own
preceding launches, run one at a time): the first sweep 22:02–22:08 (`main` first in every
pair), the correctness runs 22:08–22:50, the second sweep 22:58–23:04 (T4 first), Phase 5
T10's case at 23:05.
