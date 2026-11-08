# Helmholtz CH6.2f: the coarse levels' merge and shift

Helmholtz task CH6.2f ([docs/helmholtz/tasks/CH6.2f.md](../../docs/helmholtz/tasks/CH6.2f.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.2 (split on this branch: the coarse levels here, D8's rule in CH6.2h), §2
(constraints 1, 2, 7, 9), §5.3, §6.4. It answers CH6.2d's open question
([helmholtz-ch6-2d.md](helmholtz-ch6-2d.md) §3: why a coarse level's merge and shift cost
ours 2.4–3.8× FMM3D's parents' loop) and CH6.2c's 4 × 10 anomaly
([helmholtz-ch6-2b.md](helmholtz-ch6-2b.md) §4). Measured on:

- **locust** (NVIDIA GH200, 72 Neoverse-V2 cores, `tools/gh200/`): one rank × 72 threads,
  2026-11-08 16:40–16:51 (source eae2f01, the second `Auto` rule; §8), the load below 2 and
  no gridcl test, another user's Python process on about three quarters of one core
  throughout; the first pass (source 41698f9, the first rule, 2026-11-08 afternoon) is a
  record (§8).
- **Kathleen** (two Xeon Gold 6248 sockets, 2 × 20 Cascade Lake cores, AVX-512,
  `tools/kathleen/`): job 630738 (node-c11a-050, `singlenode`, exclusive, 2026-11-08
  16:58–17:50, source eae2f01), one rank × 40 threads and 4 × 10 (`--map-by slot:PE=10
  --bind-to core`), Open MPI 4.1.6, the P2P kernel `Auto` (AVX-512).
  Job 630756 (node-c11a-050, 17:50–18:26, source d6b5cea, the third rule): W1 and W3 at
  kD = 100 at 1 × 40 and 4 × 10, W1 at kD = 500 at 4 × 10 (§5).
- **the M6** (Apple M6, `tools/m6/machine.md`): one rank × 6 threads, W1 and W3 at kD = 100,
  2026-11-08 17:42–18:07 (source d6b5cea, the third rule), with the user's go-ahead ("the m6
  is quiet now"), under the run script's quiet gate (§6).

Release builds, rustc 1.99.0, BLAS threads 1, the default waves budget (8 GiB per node).
Commands: §10.

## Summary

- **Merge and shift ran level 2 on one thread** (§2). The shared drivers' split by parents
  (`Execution::parents`, Laplace's, Phase 6 T13) never cuts a piece below `PARENT_GRAIN` = 64
  boxes; level 2 has 64, so it ran whole on one thread (1.0 thread busy on both machines),
  and level 3 (512 boxes) on 8 pieces (7.8–7.9 busy). CH6.2d's "level 2 holds both codes to
  8 threads" was an assumption; ours held it to one. Per parent, ours was not slow: on
  Kathleen level 2 took 0.12–0.14 thread-seconds for its 8 parents (about 15–18 ms each)
  against FMM3D's about 23–38 ms per parent.
- **Where the thread time goes** (§2): on levels 2–3 the conversions (each child's sums from
  waves, local from sums, rotation back) take 76–90 % of merge and shift's thread-seconds,
  the accumulators' shifts 6–17 % and the children's sums 4–7 %. On the finer levels the
  shifts grow (Kathleen up to 30 %, locust up to 58 % on W3 at kD = 100 level 6 at 10⁻³).
- **The split by child** (`PlaneWaveSplit::Children`, §3), one task per box, each computed
  child running its parent's call for itself alone (the same arithmetic in the same order,
  so the same bits), takes level 2 to 0.05–0.07× the split by parents' time on locust and
  0.11–0.14× on Kathleen, level 3 to 0.31–0.38× and 0.49–0.53×. It repeats the shared
  accumulators' shifts (3–14× the parents' shift time) and spends 1.2–3.0× their
  thread-seconds, so on levels the split by parents already spreads (level 4 and finer) it is
  1.08–2.39× slower.
- **`Auto`** (the default) takes the split by child on a chunk where the split by parents
  makes at most half as many pieces holding a computed box as the pool has threads, the
  split by parents elsewhere. On one rank it chose the child split on levels 2–3 only, on
  every case and both machines: there the exponential M2L now takes 0.38–0.73× Rotation's
  time on locust and 0.43–0.88× on Kathleen (before: 1.2–12.3× and 0.88–7.8×), and on the
  finer levels 0.87–1.03× the split by parents' time (run-to-run noise). Summed over the
  levels, `Auto` takes W1 at kD = 100 to 0.42–0.47× the split by parents' M2L on locust and
  0.59–0.62× on Kathleen, W3 at kD = 100 to 0.95–0.98×; W1 at kD = 500 has no waves on
  levels 2–3 and is unchanged (1.00–1.02×).
- **Beside FMM3D's parents' loop** (§4): our merge and shift on level 2 now takes 0.24–0.37×
  FMM3D's loop on Kathleen and 0.06–0.12× on locust (before 3.6–5.1× and 2.3–3.5×); on
  level 3 0.90–1.72× and 0.42–1.32× (before 2.2–4.0× and 1.8–5.4×), the larger ratios at
  10⁻³, where FMM3D runs iprec 1. The finer levels are unchanged (0.37–2.32×).
- **The 4 × 10 anomaly** (§5): under the split by parents level 2 runs on one thread per
  rank, so 4 × 10 puts four threads on it where 1 × 40 puts one, while Rotation's level 2 is
  1.7–1.9× slower at 4 × 10: CH6.2c's 1.3–1.8× against 5.4–8.1× is the same defect seen from
  two sides. By child, level 2 takes 0.63–0.64× the parents' time at 4 × 10, and `Auto` (after
  its third rule, §8) takes levels 2 and 3 by child there too, level 3 0.67–0.74× the parents'.
- **The M6** (§6, 1 × 6, with the user's go-ahead): `Auto` takes the child split on level 2
  only (0.42–0.46× the parents' time, 0.44–0.69× Rotation's); level 3's eight pieces already
  fill six threads, where the child split is 1.08–1.38× slower and `Auto` keeps the parents'.
- **Bits** (§7): the three settings give the same checksum on every case, ε and split (49
  groups), and 4 × 10 gives 1 × 40's; the tests check `Children` and `Parents` against the
  default bit for bit at 1 and 2 threads, at the buffer's floor and on 2 and 4 ranks against
  the one-rank FMM.

## 1. Method

- **Problems**: W1 and W3 at kD = 100 and W1 at kD = 500, ε = 10⁻³ and 10⁻⁶ (CH6.2b's
  problems), `tools/helmholtz/ch6-2f-runs.sh coarse`: per case one launch of
  `helmholtz_accuracy --m2l rotation,planewave --split parents,children,auto`, then one with
  both lists reversed (ABBA), `--evaluations 2 --levels`, the second evaluation's times, the
  mean of the two launches (± half their range where the range exceeds 10 % of the mean).
- **Per level**: the M2L's seconds (`m2l_times`), the two passes (CH6.2d: the waves, merge
  and shift), and this task's parts of merge and shift (`PlaneWaveTimes::{busy, shifts,
  children, conversions}`): each worker's own clock, summed over threads, so `busy` ÷ the
  pass's wall time is the threads it kept busy. On P ranks the slowest rank's.
- **Settings**: `planewave/parents` (`PlaneWaveSplit::Parents`, as before this task),
  `planewave/children` (`Children`), `planewave` (`Auto`), and Rotation. All three
  exponential-M2L settings take the rule's levels and quadrature (iprec 2, CH6.1e).
- **FMM3D's parents' loop**: CH6.2d's FMM3D runs (the `timed` build, the archived
  `times-<case>-timed-{a,b}.log`, Kathleen job 630125 and locust 2026-11-07), not run again:
  the loop's wall time (`gather`) and its threads busy ((plane-wave + list-3
  thread-seconds) ÷ wall time); levels matched by κ as CH6.2d §1.
- `tools/helmholtz/ch6-2f-tables.py` prints §3's tables and the appendix.

## 2. Where merge and shift's time goes

The split by parents' threads busy and its thread-seconds by part, 10⁻⁶ (10⁻³ alike;
appendix):

| machine | case | level | V pairs | threads busy | busy s | shifts | children's sums | conversions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locust 1 × 72 | w1-100 | 2 | 3,096 | 1.0 | 0.059 | 6 % | 4 % | 90 % |
| locust 1 × 72 | w1-100 | 3 | 53,352 | 7.9 | 0.225 | 14 % | 5 % | 81 % |
| locust 1 × 72 | w1-100 | 4 | 385,344 | 50.3 | 0.999 | 30 % | 7 % | 63 % |
| locust 1 × 72 | w3-100 | 2 | 3,096 | 1.0 | 0.061 | 6 % | 4 % | 90 % |
| locust 1 × 72 | w3-100 | 3 | 53,352 | 7.9 | 0.229 | 14 % | 5 % | 80 % |
| locust 1 × 72 | w3-100 | 5 | 5,398,920 | 67.3 | 10.14 | 37 % | 6 % | 57 % |
| locust 1 × 72 | w3-100 | 6 | 46,298,376 | 71.0 | 74.15 | 37 % | 6 % | 57 % |
| Kathleen 1 × 40 | w1-100 | 2 | 3,096 | 1.0 | 0.141 | 6 % | 5 % | 89 % |
| Kathleen 1 × 40 | w1-100 | 3 | 53,352 | 7.8 | 0.585 | 10 % | 5 % | 85 % |
| Kathleen 1 × 40 | w1-100 | 4 | 385,344 | 31.0 | 2.634 | 15 % | 6 % | 78 % |
| Kathleen 1 × 40 | w3-100 | 2 | 3,096 | 1.0 | 0.142 | 7 % | 4 % | 89 % |
| Kathleen 1 × 40 | w3-100 | 3 | 53,352 | 7.9 | 0.589 | 11 % | 5 % | 84 % |
| Kathleen 1 × 40 | w3-100 | 6 | 46,298,376 | 39.7 | 168.2 | 17 % | 6 % | 77 % |
| Kathleen 1 × 40 | w1-500 | 4 | 511,200 | 30.8 | 30.04 | 9 % | 5 % | 86 % |
| Kathleen 1 × 40 | w1-500 | 7 | 14,492,220 | 39.2 | 113.8 | 10 % | 6 % | 84 % |

Reading it:

- **Level 2 ran on one thread, level 3 on eight** under the split by parents, on both
  machines: the 64-box grain of the shared drivers, not a per-parent cost. Level 2's
  thread-seconds (0.059 s on locust, 0.141 s on Kathleen for 8 parents and 3 frames) are
  its wall time.
- **The conversions dominate the coarse levels** (76–90 % on levels 2–3, every case, ε and
  machine): each child's sums from waves (its waves to its local's modes at the level's
  degree, p = 30–35 on level 2) and its rotation back. The shifts (each accumulator over its
  slots' waves) grow with the waves per box and the boxes per parent: on the finer levels
  16–58 % of the thread-seconds on locust and 7–30 % on Kathleen, the conversions 33–78 %
  and 61–87 %.
- The children's sums take 4–11 % everywhere.

## 3. The split by child, and `Auto`

Per level, the exponential M2L's seconds under each setting and Rotation's, one rank, the
threads busy in merge and shift in brackets; full tables in the appendix. Levels 2 and 3:

| machine | case | ε | level | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | Auto ÷ Rotation | Auto ÷ parents |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locust | w1-100 | 1e-3 | 2 | 0.0043 | 0.0526 (1.0) | 0.0027 (52.6) | 0.0031 (45.1) | 0.73 | 0.06 |
| locust | w1-100 | 1e-3 | 3 | 0.0157 | 0.0268 (7.9) | 0.0094 (54.8) | 0.0088 (59.4) | 0.56 | 0.33 |
| locust | w1-100 | 1e-6 | 2 | 0.0067 | 0.0612 (1.0) | 0.0032 (49.8) | 0.0031 (52.9) | 0.46 | 0.05 |
| locust | w1-100 | 1e-6 | 3 | 0.0274 | 0.0322 (7.9) | 0.0101 (59.2) | 0.0104 (56.8) | 0.38 | 0.32 |
| locust | w3-100 | 1e-3 | 2 | 0.0043 | 0.0528 (1.0) | 0.0033 (47.7) | 0.0031 (51.8) | 0.70 | 0.06 |
| locust | w3-100 | 1e-3 | 3 | 0.0163 | 0.0274 (7.9) | 0.0094 (57.7) | 0.0091 (60.1) | 0.56 | 0.33 |
| locust | w3-100 | 1e-6 | 2 | 0.0070 | 0.0631 (1.0) | 0.0041 (48.5) | 0.0041 (41.1) | 0.58 | 0.06 |
| locust | w3-100 | 1e-6 | 3 | 0.0280 | 0.0326 (7.9) | 0.0123 (51.5) | 0.0115 (52.9) | 0.41 | 0.35 |
| Kathleen | w1-100 | 1e-3 | 2 | 0.0174 | 0.1301 (1.0) | 0.0145 (29.9) | 0.0142 (30.3) | 0.82 | 0.11 |
| Kathleen | w1-100 | 1e-3 | 3 | 0.0571 | 0.0743 (7.8) | 0.0382 (37.3) | 0.0374 (37.4) | 0.66 | 0.50 |
| Kathleen | w1-100 | 1e-6 | 2 | 0.0259 | 0.1493 (1.0) | 0.0162 (29.8) | 0.0159 (29.6) | 0.61 | 0.11 |
| Kathleen | w1-100 | 1e-6 | 3 | 0.0994 | 0.0875 (7.8) | 0.0440 (37.2) | 0.0431 (37.2) | 0.43 | 0.49 |
| Kathleen | w3-100 | 1e-3 | 2 | 0.0170 | 0.1326 (1.0) | 0.0170 (30.5) | 0.0150 (30.1) | 0.88 | 0.11 |
| Kathleen | w3-100 | 1e-3 | 3 | 0.0574 | 0.0764 (7.8) | 0.0401 (37.7) | 0.0396 (37.6) | 0.69 | 0.52 |
| Kathleen | w3-100 | 1e-6 | 2 | 0.0266 | 0.1506 (1.0) | 0.0198 (31.3) | 0.0213 (30.6) | 0.80 | 0.14 |
| Kathleen | w3-100 | 1e-6 | 3 | 0.1006 | 0.0888 (7.9) | 0.0465 (36.8) | 0.0452 (37.8) | 0.45 | 0.51 |

The levels summed (the M2L's seconds; the evaluation's in the appendix):

| machine | case | ε | Rotation | parents | child | Auto | Auto ÷ parents | Auto ÷ Rotation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| locust | w1-100 | 1e-3 | 0.0545 | 0.1219 | 0.0643 | 0.0511 | 0.42 | 0.94 |
| locust | w1-100 | 1e-6 | 0.1350 | 0.1498 | 0.0794 | 0.0710 | 0.47 | 0.53 |
| locust | w3-100 | 1e-3 | 1.385 | 1.412 | 3.145 | 1.335 | 0.95 | 0.96 |
| locust | w3-100 | 1e-6 | 8.594 | 1.888 | 3.578 | 1.813 | 0.96 | 0.21 |
| locust | w1-500 | 1e-3 | 5.382 | 3.466 | 5.161 | 3.457 | 1.00 | 0.64 |
| locust | w1-500 | 1e-6 | 9.599 | 4.326 | 5.945 | 4.398 | 1.02 | 0.46 |
| Kathleen | w1-100 | 1e-3 | 0.2041 | 0.3646 | 0.2534 | 0.2137 | 0.59 | 1.05 |
| Kathleen | w1-100 | 1e-6 | 0.5065 | 0.4727 | 0.3325 | 0.2952 | 0.62 | 0.58 |
| Kathleen | w3-100 | 1e-3 | 5.194 | 4.362 | 9.531 | 4.210 | 0.97 | 0.81 |
| Kathleen | w3-100 | 1e-6 | 32.02 | 7.362 | 12.25 | 7.204 | 0.98 | 0.22 |
| Kathleen | w1-500 | 1e-3 | 20.28 | 13.78 | 18.83 | 13.86 | 1.01 | 0.68 |
| Kathleen | w1-500 | 1e-6 | 38.71 | 18.05 | 22.60 | 18.35 | 1.02 | 0.47 |

Reading it:

- **By child, level 2 uses the pool** (30 of 40 threads busy on Kathleen, 41–53 of 72 on
  locust) and level 3 too (37–38 and 51–60): level 2 0.05–0.07× the split by parents' time
  on locust, 0.11–0.14× on Kathleen; level 3 0.31–0.38× and 0.49–0.53×.
- **It costs thread time**: the child split's thread-seconds are 1.5–1.9× the parents' on
  locust's levels 2–3 and 2.0–2.4× on Kathleen's (1.2–3.0× on all levels); its shifts 3–14×
  the parents' (a shared accumulator is computed once per child it serves, up to 8; level 2's
  ratios above 8, 10–14×, are the per-call overheads of its short shifts, not separated).
  On the levels the split by parents already spreads over 30 threads or more it is
  1.08–2.39× slower, so it is not a default by itself.
- **`Auto` takes the better of the two on every level** at one rank (its busy threads show
  which it ran): the child split on levels 2–3, the parents' on the others, where it is
  0.87–1.03× the split by parents' time.
- **Against Rotation, levels 2–3 now gain** (0.38–0.73× on locust, 0.43–0.88× on Kathleen),
  where under the split by parents level 2 took 9.0–12.3× and 5.7–7.8× Rotation's time and
  level 3 1.2–1.7× and 0.88–1.3×. Summed, W1 at kD = 100
  at 10⁻³ on Kathleen is the one case where Rotation stays ahead (`Auto` 1.05×), from level
  5 (degree 7, 2.9× Rotation's), a fine level the rule's D8 decides (CH6.2h).

## 4. Beside FMM3D's parents' loop

Our merge and shift's wall time (the split by parents, as CH6.2d measured, and `Auto`) over
FMM3D's parents' loop (CH6.2d's `timed` runs; its threads busy in brackets), one rank:

| machine | case | ε | level | FMM3D boxes | FMM3D loop s (busy) | ours parents s | ours Auto s | parents ÷ FMM3D | Auto ÷ FMM3D |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kathleen | w1-100 | 1e-3 | 2 | 64 | 0.0244 (7.6) | 0.1236 | 0.0085 | 5.07 | 0.35 |
| Kathleen | w1-100 | 1e-3 | 3 | 512 | 0.0158 (27.4) | 0.0636 | 0.0272 | 4.02 | 1.72 |
| Kathleen | w1-100 | 1e-6 | 2 | 64 | 0.0395 (7.7) | 0.1416 | 0.0093 | 3.58 | 0.24 |
| Kathleen | w1-100 | 1e-6 | 3 | 512 | 0.0336 (27.2) | 0.0746 | 0.0304 | 2.22 | 0.90 |
| Kathleen | w1-100 | 1e-6 | 4 | 1600 | 0.0906 (25.3) | 0.0851 | 0.0866 | 0.94 | 0.96 |
| Kathleen | w3-100 | 1e-3 | 2 | 64 | 0.0245 (7.6) | 0.1248 | 0.0092 | 5.10 | 0.37 |
| Kathleen | w3-100 | 1e-3 | 3 | 512 | 0.0193 (28.1) | 0.0655 | 0.0284 | 3.39 | 1.47 |
| Kathleen | w3-100 | 1e-6 | 2 | 64 | 0.0396 (7.7) | 0.1421 | 0.0104 | 3.59 | 0.26 |
| Kathleen | w3-100 | 1e-6 | 3 | 512 | 0.0326 (30.5) | 0.0750 | 0.0321 | 2.30 | 0.98 |
| Kathleen | w3-100 | 1e-6 | 4 | 4096 | 0.1333 (37.3) | 0.1146 | 0.1140 | 0.86 | 0.86 |
| Kathleen | w3-100 | 1e-6 | 5 | 32768 | 0.7328 (38.3) | 0.5728 | 0.5780 | 0.78 | 0.79 |
| Kathleen | w1-500 | 1e-6 | 4 | 3840 | 0.9059 (25.5) | 0.9763 | 0.9690 | 1.08 | 1.07 |
| Kathleen | w1-500 | 1e-6 | 6 | 34432 | 2.4994 (20.4) | 1.9907 | 1.9912 | 0.80 | 0.80 |
| locust | w1-100 | 1e-3 | 2 | 64 | 0.0148 (7.9) | 0.0507 | 0.0018 | 3.43 | 0.12 |
| locust | w1-100 | 1e-3 | 3 | 512 | 0.0061 (38.0) | 0.0238 | 0.0058 | 3.90 | 0.96 |
| locust | w1-100 | 1e-6 | 2 | 64 | 0.0260 (7.8) | 0.0594 | 0.0017 | 2.28 | 0.06 |
| locust | w1-100 | 1e-6 | 3 | 512 | 0.0161 (30.6) | 0.0283 | 0.0068 | 1.76 | 0.42 |
| locust | w3-100 | 1e-3 | 2 | 64 | 0.0146 (8.0) | 0.0512 | 0.0015 | 3.50 | 0.11 |
| locust | w3-100 | 1e-3 | 3 | 512 | 0.0045 (54.2) | 0.0241 | 0.0059 | 5.37 | 1.32 |
| locust | w3-100 | 1e-6 | 2 | 64 | 0.0256 (8.0) | 0.0608 | 0.0024 | 2.38 | 0.09 |
| locust | w3-100 | 1e-6 | 3 | 512 | 0.0100 (56.7) | 0.0290 | 0.0077 | 2.89 | 0.77 |
| locust | w3-100 | 1e-6 | 5 | 32768 | 0.2330 (69.3) | 0.1507 | 0.1498 | 0.65 | 0.64 |
| locust | w1-500 | 1e-6 | 6 | 34432 | 0.9463 (35.1) | 0.4571 | 0.4609 | 0.48 | 0.49 |

(The other matched levels, every case and ε, by `ms.py`'s query of §10: unchanged by `Auto`,
0.37–2.32× FMM3D's loop.)

Reading it:

- **CH6.2d's question answered**: our level-2 merge and shift was 2.3–5.1× FMM3D's loop
  because it ran on one thread where FMM3D's ran on 7.6–8.0; per thread, ours took 15–18 ms
  per parent on Kathleen against FMM3D's 23–38 ms. On level 2 ours by child now takes
  0.24–0.37× FMM3D's loop on Kathleen and 0.06–0.12× on locust, with up to 72 threads where
  FMM3D's split by parent keeps 8.
- **Level 3** at 10⁻⁶ is near par (0.90–0.98× on Kathleen, 0.42–0.77× on locust); at 10⁻³,
  where FMM3D runs iprec 1 (fewer waves; it misses ε on W1 and W3 at kD = 100, CH6.2d), FMM3D
  stays ahead on Kathleen (1.47–1.72×) and on locust's W3 (1.32×).
- **The finer levels** are this task's no-change: at 10⁻⁶ ours at par or faster (0.48–1.08×),
  at 10⁻³ FMM3D's iprec 1 ahead (1.5–2.3× on Kathleen; CH6.2d §2).

## 5. Kathleen 4 × 10: CH6.2c's anomaly

CH6.2c saw level 2 on W3 at kD = 100 take 5.4–8.1× Rotation's time at 1 × 40 but 1.3–1.8×
at 4 × 10. Job 630756 (node-c11a-050, 2026-11-08 17:50–18:26, source d6b5cea, the third
rule; the build recompiled nd-fmm-exec) ran W1 and W3 at kD = 100 at 1 × 40 and 4 × 10 and
W1 at kD = 500 at 4 × 10. Levels 2 and 3 at 4 × 10 (the slowest rank's):

| case | ε | level | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | Auto ÷ Rotation | Auto ÷ parents | Auto ÷ parents, second rule (job 630738) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w1-100 | 1e-3 | 2 | 0.0326 | 0.0533 (1.0) | 0.0324 (4.3) | 0.0333 (4.3) | 1.02 | 0.63 | 0.62 |
| w1-100 | 1e-3 | 3 | 0.0558 | 0.0720 (2.9) | 0.0473 (9.4) | 0.0486 (9.2) | 0.87 | 0.67 | 0.97 (3.0 busy) |
| w1-100 | 1e-6 | 2 | 0.0499 | 0.0604 (1.0) | 0.0375 (4.3) | 0.0380 (4.3) | 0.76 | 0.63 | 0.61 |
| w1-100 | 1e-6 | 3 | 0.1009 | 0.0805 (3.0) | 0.0576 (9.3) | 0.0579 (9.4) | 0.57 | 0.72 | 0.99 (3.0 busy) |
| w3-100 | 1e-3 | 2 | 0.0317 | 0.0545 (1.0) | 0.0342 (4.4) | 0.0349 (4.5) | 1.10 | 0.64 | 0.65 |
| w3-100 | 1e-3 | 3 | 0.0573 | 0.0666 (3.1) | 0.0487 (9.3) | 0.0492 (9.4) | 0.86 | 0.74 | 1.00 (3.1 busy) |
| w3-100 | 1e-6 | 2 | 0.0497 | 0.0605 (1.0) | 0.0402 (4.6) | 0.0387 (4.5) | 0.78 | 0.64 | 0.63 |
| w3-100 | 1e-6 | 3 | 0.1012 | 0.0792 (3.1) | 0.0595 (9.3) | 0.0576 (9.4) | 0.57 | 0.73 | 0.97 (3.1 busy) |

Summed over the levels at 4 × 10, `Auto` takes 0.82–0.91× the split by parents' M2L on W1 at
kD = 100 (0.2515 and 0.3647 s against 0.3076 and 0.4029), 0.99–1.00× on W3 at kD = 100 and
1.00–1.01× on W1 at kD = 500 (levels 4–7 only). The 1 × 40 runs of the same job repeat
job 630738's to within its spread (`Auto` on levels 2–3 0.44–0.95× Rotation's, 0.11–0.53×
the parents').

Reading it:

- **The anomaly is the split by parents seen from two sides.** Under it level 2 runs on one
  thread per rank: at 1 × 40 one thread does all 8 parents (0.12–0.14 thread-seconds), at
  4 × 10 each rank does its share (the slowest rank 0.034–0.039 s, about a quarter), so four
  threads work. Rotation's level 2 meanwhile takes 1.7–1.9× its 1 × 40 time at 4 × 10
  (0.032–0.050 s against 0.018–0.027 s; not separated here). Both move the ratio: here
  5.9–7.5× at 1 × 40 and 1.2–1.7× at 4 × 10 on W3, CH6.2c's figures. Nothing in the 4 × 10
  split itself is better for merge and shift.
- **By child the anomaly is gone**: level 2 takes 0.11–0.14× the parents' time at 1 × 40 and
  0.63–0.64× at 4 × 10 (each rank's 16 boxes keep 4.3–4.6 of its 10 threads busy); `Auto`
  takes it on both levels 2 and 3 at 4 × 10, level 3 0.67–0.74× the parents'.
- **The second rule's defect** (§8): at 4 × 10 it kept level 3 by parents on 3 busy threads,
  since a rank's level-3 locals hold boxes the rank does not compute, whose pieces have no
  work; counting the pieces with a computed box fixed it (the last column against the
  `Auto ÷ parents` column).

## 6. The M6

One rank × 6 threads, W1 and W3 at kD = 100 (W1 at kD = 500 needs about 15 GB and runs
Rotation on levels 2–3), the third rule. The run script's gate held every launch to other
processes below 150 % of a core before it began; W3's first reverse launch saw 366 % during
and was repeated (84 %), W3's forward launch saw 190 % (below the 200 % repeat threshold).

| case | ε | level | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | Auto ÷ Rotation | Auto ÷ parents |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w1-100 | 1e-3 | 2 | 0.0268 | 0.0409 (1.0) | 0.0178 (5.5) | 0.0184 (5.3) | 0.69 | 0.45 |
| w1-100 | 1e-3 | 3 | 0.1171 | 0.0533 (4.3) | 0.0595 (5.7) | 0.0488 (4.7) | 0.42 | 0.92 |
| w1-100 | 1e-6 | 2 | 0.0497 | 0.0470 (1.0) | 0.0203 (5.7) | 0.0217 (4.9) | 0.44 | 0.46 |
| w1-100 | 1e-6 | 3 | 0.2120 | 0.0596 (4.6) | 0.0739 (5.5) | 0.0603 (4.7) | 0.28 | 1.01 |
| w3-100 | 1e-3 | 2 | 0.0321 | 0.0478 (1.0) | 0.0220 (5.4) | 0.0200 (5.8) | 0.62 | 0.42 |
| w3-100 | 1e-3 | 3 | 0.1545 | 0.0523 (4.7) | 0.0721 (5.2) | 0.0524 (4.8) | 0.34 | 1.00 |
| w3-100 | 1e-6 | 2 | 0.0495 | 0.0576 (1.0) | 0.0237 (5.5) | 0.0253 (5.0) | 0.51 | 0.44 |
| w3-100 | 1e-6 | 3 | 0.2292 | 0.0693 (4.6) | 0.0749 (6.0) | 0.0664 (4.7) | 0.29 | 0.96 |

Summed over the levels, `Auto` takes 0.88–0.96× the split by parents' M2L (W1 0.88 and
0.96, W3 0.98 and 0.98) and 0.16–0.66× Rotation's.

Reading it:

- **At 6 threads the rule takes the child split on level 2 only** (5 busy threads, 0.42–0.46×
  the parents' time, 0.44–0.69× Rotation's); level 3's eight pieces already fill 4.3–4.7 of
  six threads, and the child split is 1.08–1.38× slower there, so `Auto` keeps the parents'
  (0.92–1.01×).
- Under the split by parents level 2 took 0.95–1.53× Rotation's time here (CH6.2b: 0.9–3.3×
  on the M6) and level 3 already 0.28–0.46×: with six threads Rotation's level is held back
  as well. Level 2 now gains on every case.
- The parts as elsewhere: conversions 79–92 % of levels 2–3's thread-seconds, the child
  split's thread time 1.4–1.7× the parents' there.

## 7. Tests and checks

- `check_plane_wave` (`fmm-exec/tests/helmholtz_fmm.rs`, two scenarios, 1 rank and the MPI
  list's 2 and 4): the default is `Auto`; at 1 thread it never splits by child, at 2 it does
  on some rank (`PlaneWaveCounts::by_child`); `Parents` at 2 threads, `Children` at 1 and 2
  threads, each the default's output bit for bit, `by_child` equal to the chunks or 0;
  `Children` on P ranks against the one-rank FMM (`check_one_rank`); the buffer's floor at 2
  threads runs `Children` (`check_plane_wave_buffer`).
- `check_plane_wave_times`: the parts on every level with waves and V pairs, `busy` and the
  conversions positive, shifts + children + conversions within `busy`; every rank or none.
- `tests/helmholtz/plane_wave.rs` (`the_split_is_a_setting`): the default and the setters.
- The unit test `pieces_count_the_split_by_parents_with_work` (`operator_plane_wave.rs`):
  `pieces` against the driver's halving, pieces without a computed box not counted.
- `fmm-validate/tests/helmholtz_accuracy.rs`: the parts' columns on the exponential-M2L rows,
  `Children` at 2 threads with the default's checksum.
- In the runs: the three settings' checksums equal on every case, ε and split (49 groups, all
  machines), 4 × 10's equal to 1 × 40's.

`/verify`: PENDING (§9 of the PR).

## 8. Runs, records and defects

- **`Auto`'s rule changed twice on this branch.**
  1. First rule (source 41698f9): by child on a chunk of at most `PARENT_GRAIN` × threads
     boxes. locust's first pass (`bench-results/ch6-2f/locust-first`, a record) showed it
     took the child split on level 4 of W1 and W3 at kD = 100 at 72 threads (1,600 and
     4,096 boxes), 1.24–1.32× slower than the parents', which already kept about 50 threads
     busy there.
  2. Second rule (eae2f01, the runs of §2–§4): by child where the split by parents makes at
     most half as many pieces as the pool has threads (`pieces` restates the driver's
     halving; the half from the child split's measured 1.5–1.9× thread time). On one rank it
     chose as §3 shows. At 4 × 10 on Kathleen it kept level 3 by parents (3.0 threads busy,
     0.97–1.00× the parents' time, where the child split took 0.71–0.76×): a rank's level
     holds boxes it does not compute (the locally essential tree's), whose pieces have no
     work but counted.
  3. Third rule (4228f48, the default now): the pieces holding a computed box. On one rank
     every piece has work on these cases, so §2–§4 stand; 4 × 10 again in §5.
- **locust**: the second pass's launches 4–152 s each; the Python process of another user
  (about 76 % of one core) ran throughout both passes.
- **Kathleen**: job 630737 (the first rule) cancelled before it ran; 630738 ran the second.

## 9. Follow-ups

- **CH6.2h (D8)**: levels 2–3 under `Auto` now gain over Rotation on both machines at one
  rank (0.38–0.88×), so D8's rule need not take Rotation on levels of few boxes on that
  account; the remaining losses against Rotation are fine levels of low degree (W1 at
  kD = 100 level 5, p = 7: 1.7–3.0×, the M6 1.7×).
- **Laplace**: its plane-wave M2L (Phase 6 T13) runs on the same `Execution::parents` with
  the same 64-box grain, so its coarse levels are likely held to few threads too; not
  measured or changed here (constraint 1). A question for the PR.
- **The repeated shifts**: a split in two phases (each parent's accumulators once, on the
  pool, then the children) would remove the child split's 1.2–3.0× thread time; `Auto`
  avoids the loss where it matters now, so it is noted, not scheduled.

## 10. Reproducing

```sh
# Build (each machine), then the runs:
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
# locust (quiet; tools/gh200/sync.sh first)
tools/gh200/remote.sh 'THREADS=72 tools/helmholtz/ch6-2f-runs.sh /data/ucahtbe/logs/ch6-2f-b coarse'
# Kathleen (tools/kathleen/sync.sh first)
tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch \
  'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
  'THREADS=40 SPLITS="1x40 4x10" tools/helmholtz/ch6-2f-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2f coarse'
# Kathleen, the re-run with the third rule (job 630756)
tools/kathleen/submit.sh --time=02:00:00 -- run.sbatch \
  'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
  "THREADS=40 SPLITS=\"1x40 4x10\" ONLY='w1-100|w3-100|w1-500-4x10' tools/helmholtz/ch6-2f-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2f-b coarse"
# the M6 (quiet gate built in)
THREADS=6 ONLY='w1-100|w3-100' tools/helmholtz/ch6-2f-runs.sh bench-results/ch6-2f/m6 coarse
# The tables (outputs fetched to bench-results/ch6-2f/<machine>/)
tools/helmholtz/ch6-2f-tables.py locust=bench-results/ch6-2f/locust \
  kathleen=bench-results/ch6-2f/kathleen kathleen-b=bench-results/ch6-2f/kathleen-b \
  m6=bench-results/ch6-2f/m6
```

The raw outputs: `~/projects/fmm-archive/locust/logs/ch6-2f-{b,first}`, `kathleen/logs/ch6-2f{,-b}` and
`m6/logs/ch6-2f` (copies of `bench-results/ch6-2f/`, which git ignores).

## Appendix: the tables script's output

Each machine, case and split (ranks × threads): per level, the M2L's seconds with the
threads busy in merge and shift in brackets; the levels summed and the evaluation; the split
by parents' parts and the child split's thread time. locust's and job 630738's `Auto` ran
the second rule, job 630756's and the M6's the third (the same choices on one rank; at
4 × 10 they differ on level 3, §5). The means of the two launches, the spread (half the
range over the mean) where above 5 %.

#### locust, w1-100, 1 × 72

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0043 | 0.0526 (1.0) | 0.0027 (52.6) | 0.0031 (±11 %) (45.1) | 0.05 | 0.73 | 0.06 |
| 1e-3 | 3 | 18 | 53352 | 0.0157 | 0.0268 (7.9) | 0.0094 (54.8) | 0.0088 (59.4) | 0.35 | 0.56 | 0.33 |
| 1e-3 | 4 | 11 | 385344 | 0.0278 | 0.0267 (±12 %) (43.1) | 0.0311 (63.1) | 0.0232 (50.9) | 1.17 | 0.84 | 0.87 |
| 1e-3 | 5 | 7 | 244944 | 0.0067 | 0.0158 (52.9) | 0.0211 (58.7) | 0.0160 (53.2) | 1.33 | 2.38 | 1.01 |
| 1e-6 | 2 | 35 | 3096 | 0.0067 | 0.0612 (1.0) | 0.0032 (±7 %) (49.8) | 0.0031 (52.9) | 0.05 | 0.46 | 0.05 |
| 1e-6 | 3 | 22 | 53352 | 0.0274 | 0.0322 (7.9) | 0.0101 (59.2) | 0.0104 (56.8) | 0.31 | 0.38 | 0.32 |
| 1e-6 | 4 | 15 | 385344 | 0.0616 | 0.0288 (50.3) | 0.0351 (66.8) | 0.0295 (50.4) | 1.22 | 0.48 | 1.03 |
| 1e-6 | 5 | 15 | 244944 | 0.0393 | 0.0275 (53.1) | 0.0310 (63.5) | 0.0279 (52.4) | 1.12 | 0.71 | 1.01 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 0.0545 | 0.0862 |
| 1e-3 | parents | 0.1219 | 0.1568 |
| 1e-3 | child | 0.0643 | 0.0985 |
| 1e-3 | Auto | 0.0511 | 0.0841 |
| 1e-6 | Rotation | 0.1350 | 0.2031 |
| 1e-6 | parents | 0.1498 | 0.2171 |
| 1e-6 | child | 0.0794 | 0.1481 |
| 1e-6 | Auto | 0.0710 | 0.1403 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0507 | 7 % | 5 % | 88 % | 0.0789 | 1.56 | 7.47 |
| 1e-3 | 3 | 0.1883 | 17 % | 6 % | 77 % | 0.3465 | 1.84 | 5.19 |
| 1e-3 | 4 | 0.8160 | 37 % | 8 % | 54 % | 1.5188 | 1.86 | 3.18 |
| 1e-3 | 5 | 0.5477 | 29 % | 11 % | 59 % | 0.9043 | 1.65 | 3.07 |
| 1e-6 | 2 | 0.0594 | 6 % | 4 % | 90 % | 0.0886 | 1.49 | 7.55 |
| 1e-6 | 3 | 0.2246 | 14 % | 5 % | 81 % | 0.3881 | 1.73 | 5.25 |
| 1e-6 | 4 | 0.9990 | 30 % | 7 % | 63 % | 1.7326 | 1.73 | 3.23 |
| 1e-6 | 5 | 0.9345 | 16 % | 7 % | 78 % | 1.3176 | 1.41 | 3.35 |

#### locust, w1-500, 1 × 72

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 36 | 511200 | 0.9680 | 0.2973 (53.6) | 0.4065 (70.8) | 0.2947 (53.6) | 1.37 | 0.30 | 0.99 |
| 1e-3 | 5 | 21 | 2935296 | 1.2082 | 0.3853 (67.4) | 0.7409 (71.5) | 0.3858 (67.3) | 1.92 | 0.32 | 1.00 |
| 1e-3 | 6 | 13 | 10439280 | 1.1661 | 0.6035 (70.5) | 1.2938 (71.5) | 0.5948 (70.5) | 2.14 | 0.51 | 0.99 |
| 1e-3 | 7 | 8 | 14492220 | 0.5055 | 0.6524 (69.8) | 1.1871 (70.5) | 0.6533 (69.6) | 1.82 | 1.29 | 1.00 |
| 1e-6 | 4 | 41 | 511200 | 1.4039 | 0.3206 (53.8) | 0.4171 (71.0) | 0.3254 (53.5) | 1.30 | 0.23 | 1.01 |
| 1e-6 | 5 | 25 | 2935296 | 1.9344 | 0.4352 (67.2) | 0.7701 (71.5) | 0.4437 (67.0) | 1.77 | 0.23 | 1.02 |
| 1e-6 | 6 | 16 | 10439280 | 1.9924 | 0.6732 (70.5) | 1.3644 (71.5) | 0.6761 (70.6) | 2.03 | 0.34 | 1.00 |
| 1e-6 | 7 | 15 | 14492220 | 2.3276 | 0.9391 (69.8) | 1.4120 (71.2) | 0.9564 (69.6) | 1.50 | 0.41 | 1.02 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 5.3818 | 5.9795 |
| 1e-3 | parents | 3.4657 | 4.0461 |
| 1e-3 | child | 5.1611 | 5.7506 |
| 1e-3 | Auto | 3.4567 | 4.0517 |
| 1e-6 | Rotation | 9.5990 | 10.3979 |
| 1e-6 | parents | 4.3260 | 5.1250 |
| 1e-6 | child | 5.9447 | 6.7292 |
| 1e-6 | Auto | 4.3979 | 5.1957 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 10.6481 | 17 % | 8 % | 75 % | 21.7853 | 2.05 | 5.80 |
| 1e-3 | 5 | 17.1440 | 25 % | 8 % | 67 % | 43.5048 | 2.54 | 6.83 |
| 1e-3 | 6 | 29.9833 | 37 % | 9 % | 54 % | 79.7649 | 2.66 | 5.46 |
| 1e-3 | 7 | 29.5957 | 36 % | 9 % | 54 % | 67.2252 | 2.27 | 4.47 |
| 1e-6 | 4 | 11.5158 | 16 % | 7 % | 77 % | 21.9105 | 1.90 | 5.60 |
| 1e-6 | 5 | 19.0628 | 21 % | 6 % | 72 % | 44.2767 | 2.32 | 6.87 |
| 1e-6 | 6 | 32.2473 | 31 % | 7 % | 61 % | 81.8733 | 2.54 | 5.79 |
| 1e-6 | 7 | 41.9722 | 22 % | 6 % | 71 % | 76.2260 | 1.82 | 4.51 |

#### locust, w3-100, 1 × 72

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0043 | 0.0528 (1.0) | 0.0033 (47.7) | 0.0031 (±9 %) (51.8) | 0.06 | 0.70 | 0.06 |
| 1e-3 | 3 | 18 | 53352 | 0.0163 | 0.0274 (7.9) | 0.0094 (±8 %) (57.7) | 0.0091 (60.1) | 0.34 | 0.56 | 0.33 |
| 1e-3 | 4 | 11 | 584136 | 0.0431 | 0.0310 (53.7) | 0.0447 (66.7) | 0.0304 (54.0) | 1.44 | 0.71 | 0.98 |
| 1e-3 | 5 | 7 | 5398920 | 0.1393 | 0.1566 (67.9) | 0.3541 (70.9) | 0.1588 (67.3) | 2.26 | 1.14 | 1.01 |
| 1e-3 | 6 | 7 | 46298376 | 1.1822 | 1.1439 (70.8) | 2.7336 (71.6) | 1.1334 (71.0) | 2.39 | 0.96 | 0.99 |
| 1e-6 | 2 | 35 | 3096 | 0.0070 | 0.0631 (1.0) | 0.0041 (±23 %) (48.5) | 0.0041 (±21 %) (41.1) | 0.07 | 0.58 | 0.06 |
| 1e-6 | 3 | 22 | 53352 | 0.0280 | 0.0326 (7.9) | 0.0123 (±9 %) (51.5) | 0.0115 (±7 %) (52.9) | 0.38 | 0.41 | 0.35 |
| 1e-6 | 4 | 15 | 584136 | 0.0950 | 0.0382 (54.1) | 0.0503 (68.3) | 0.0386 (53.6) | 1.32 | 0.41 | 1.01 |
| 1e-6 | 5 | 15 | 5398920 | 0.8766 | 0.2173 (67.3) | 0.3985 (70.9) | 0.2200 (67.2) | 1.83 | 0.25 | 1.01 |
| 1e-6 | 6 | 15 | 46298376 | 7.5870 | 1.5367 (71.0) | 3.1126 (71.2) | 1.5391 (71.0) | 2.03 | 0.20 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 1.3852 | 2.1142 |
| 1e-3 | parents | 1.4118 | 2.1395 |
| 1e-3 | child | 3.1451 | 3.8762 |
| 1e-3 | Auto | 1.3348 | 2.0629 |
| 1e-6 | Rotation | 8.5936 | 9.6334 |
| 1e-6 | parents | 1.8880 | 2.9229 |
| 1e-6 | child | 3.5779 | 4.6173 |
| 1e-6 | Auto | 1.8132 | 2.8522 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0512 | 7 % | 5 % | 88 % | 0.0944 | 1.84 | 11.63 |
| 1e-3 | 3 | 0.1904 | 17 % | 6 % | 76 % | 0.3523 | 1.85 | 5.18 |
| 1e-3 | 4 | 1.1587 | 39 % | 9 % | 51 % | 2.3586 | 2.04 | 3.47 |
| 1e-3 | 5 | 7.7167 | 56 % | 9 % | 34 % | 21.9857 | 2.85 | 4.25 |
| 1e-3 | 6 | 57.8353 | 58 % | 8 % | 33 % | 172.3496 | 2.98 | 4.37 |
| 1e-6 | 2 | 0.0608 | 6 % | 4 % | 90 % | 0.1031 | 1.70 | 10.19 |
| 1e-6 | 3 | 0.2288 | 14 % | 5 % | 80 % | 0.4089 | 1.79 | 5.51 |
| 1e-6 | 4 | 1.3997 | 33 % | 7 % | 60 % | 2.6360 | 1.88 | 3.46 |
| 1e-6 | 5 | 10.1377 | 37 % | 6 % | 57 % | 23.5771 | 2.33 | 4.52 |
| 1e-6 | 6 | 74.1460 | 37 % | 6 % | 57 % | 186.5180 | 2.52 | 5.01 |

#### Kathleen job 630738 (second rule), w1-100, 1 × 40

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0174 | 0.1301 (1.0) | 0.0145 (29.9) | 0.0142 (30.3) | 0.11 | 0.82 | 0.11 |
| 1e-3 | 3 | 18 | 53352 | 0.0571 | 0.0743 (7.8) | 0.0382 (37.3) | 0.0374 (37.4) | 0.51 | 0.66 | 0.50 |
| 1e-3 | 4 | 11 | 385344 | 0.1059 | 0.0922 (30.9) | 0.1240 (39.1) | 0.0929 (30.9) | 1.34 | 0.88 | 1.01 |
| 1e-3 | 5 | 7 | 244944 | 0.0237 | 0.0679 (33.9) | 0.0766 (39.3) | 0.0691 (34.4) | 1.13 | 2.92 | 1.02 |
| 1e-6 | 2 | 35 | 3096 | 0.0259 | 0.1493 (1.0) | 0.0162 (29.8) | 0.0159 (29.6) | 0.11 | 0.61 | 0.11 |
| 1e-6 | 3 | 22 | 53352 | 0.0994 | 0.0875 (7.8) | 0.0440 (37.2) | 0.0431 (37.2) | 0.50 | 0.43 | 0.49 |
| 1e-6 | 4 | 15 | 385344 | 0.2336 | 0.1166 (31.0) | 0.1440 (38.8) | 0.1202 (30.6) | 1.24 | 0.51 | 1.03 |
| 1e-6 | 5 | 15 | 244944 | 0.1477 | 0.1193 (34.5) | 0.1283 (37.8) | 0.1160 (34.6) | 1.08 | 0.79 | 0.97 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 0.2041 | 0.2872 |
| 1e-3 | parents | 0.3646 | 0.4456 |
| 1e-3 | child | 0.2534 | 0.3419 |
| 1e-3 | Auto | 0.2137 | 0.2990 |
| 1e-6 | Rotation | 0.5065 | 0.7066 |
| 1e-6 | parents | 0.4727 | 0.6683 |
| 1e-6 | child | 0.3325 | 0.5330 |
| 1e-6 | Auto | 0.2952 | 0.4985 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.1234 | 7 % | 5 % | 88 % | 0.2598 | 2.11 | 10.05 |
| 1e-3 | 3 | 0.4986 | 12 % | 6 % | 82 % | 1.0195 | 2.04 | 6.29 |
| 1e-3 | 4 | 2.1420 | 19 % | 8 % | 72 % | 3.9610 | 1.85 | 4.91 |
| 1e-3 | 5 | 1.6833 | 12 % | 10 % | 77 % | 2.3020 | 1.37 | 3.73 |
| 1e-6 | 2 | 0.1414 | 6 % | 5 % | 89 % | 0.2835 | 2.01 | 9.85 |
| 1e-6 | 3 | 0.5846 | 10 % | 5 % | 85 % | 1.1531 | 1.97 | 6.44 |
| 1e-6 | 4 | 2.6337 | 15 % | 6 % | 78 % | 4.3379 | 1.65 | 4.75 |
| 1e-6 | 5 | 2.7805 | 7 % | 6 % | 87 % | 3.4536 | 1.24 | 3.74 |

#### Kathleen job 630738 (second rule), w1-100, 4 × 10

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0315 | 0.0530 (1.0) | 0.0331 (4.3) | 0.0326 (4.3) | 0.62 | 1.03 | 0.62 |
| 1e-3 | 3 | 18 | 53352 | 0.0579 | 0.0679 (3.0) | 0.0480 (9.3) | 0.0662 (3.0) | 0.71 | 1.14 | 0.97 |
| 1e-3 | 4 | 11 | 385344 | 0.1058 | 0.1029 (7.4) | 0.1386 (8.8) | 0.1036 (7.5) | 1.35 | 0.98 | 1.01 |
| 1e-3 | 5 | 7 | 244944 | 0.0240 | 0.0723 (8.1) | 0.0788 (9.9) | 0.0719 (7.8) | 1.09 | 3.00 | 0.99 |
| 1e-6 | 2 | 35 | 3096 | 0.0496 | 0.0610 (1.0) | 0.0381 (4.3) | 0.0374 (4.3) | 0.62 | 0.75 | 0.61 |
| 1e-6 | 3 | 22 | 53352 | 0.1024 | 0.0804 (3.0) | 0.0571 (9.4) | 0.0796 (3.0) | 0.71 | 0.78 | 0.99 |
| 1e-6 | 4 | 15 | 385344 | 0.2417 | 0.1343 (7.5) | 0.1599 (9.3) | 0.1372 (7.4) | 1.19 | 0.57 | 1.02 |
| 1e-6 | 5 | 15 | 244944 | 0.1532 | 0.1348 (8.2) | 0.1473 (9.9) | 0.1360 (8.1) | 1.09 | 0.89 | 1.01 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 0.2192 | 0.2888 |
| 1e-3 | parents | 0.2962 | 0.3663 |
| 1e-3 | child | 0.2985 | 0.3607 |
| 1e-3 | Auto | 0.2743 | 0.3442 |
| 1e-6 | Rotation | 0.5469 | 0.7369 |
| 1e-6 | parents | 0.4104 | 0.5937 |
| 1e-6 | child | 0.4025 | 0.5954 |
| 1e-6 | Auto | 0.3902 | 0.5810 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0336 | 8 % | 7 % | 86 % | 0.0592 | 1.76 | 5.99 |
| 1e-3 | 3 | 0.1347 | 12 % | 6 % | 82 % | 0.2349 | 1.74 | 5.32 |
| 1e-3 | 4 | 0.5175 | 15 % | 8 % | 77 % | 0.8942 | 1.73 | 5.36 |
| 1e-3 | 5 | 0.4037 | 10 % | 10 % | 79 % | 0.5391 | 1.34 | 4.36 |
| 1e-6 | 2 | 0.0382 | 7 % | 6 % | 87 % | 0.0646 | 1.69 | 5.78 |
| 1e-6 | 3 | 0.1579 | 11 % | 5 % | 84 % | 0.2676 | 1.69 | 5.42 |
| 1e-6 | 4 | 0.6717 | 12 % | 6 % | 82 % | 1.0328 | 1.54 | 5.38 |
| 1e-6 | 5 | 0.7111 | 6 % | 6 % | 89 % | 0.8898 | 1.25 | 4.71 |

#### Kathleen job 630738 (second rule), w1-500, 1 × 40

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 36 | 511200 | 3.6570 | 1.3175 (30.8) | 1.6701 (39.2) | 1.2984 (30.8) | 1.27 | 0.36 | 0.99 |
| 1e-3 | 5 | 21 | 2935296 | 4.4658 | 1.7312 (38.4) | 2.7034 (39.9) | 1.7144 (38.8) | 1.56 | 0.38 | 0.99 |
| 1e-3 | 6 | 13 | 10439280 | 4.3863 | 2.3635 (39.5) | 4.2101 (39.9) | 2.3950 (39.6) | 1.78 | 0.55 | 1.01 |
| 1e-3 | 7 | 8 | 14492220 | 1.8838 | 2.5759 (39.2) | 4.2767 (37.8) | 2.5606 (39.2) | 1.66 | 1.36 | 0.99 |
| 1e-6 | 4 | 41 | 511200 | 5.6545 | 1.4467 (30.8) | 1.7139 (39.2) | 1.4370 (30.5) | 1.18 | 0.25 | 0.99 |
| 1e-6 | 5 | 25 | 2935296 | 8.0378 (±7 %) | 1.9927 (38.5) | 2.9528 (38.0) | 2.0003 (38.5) | 1.48 | 0.25 | 1.00 |
| 1e-6 | 6 | 16 | 10439280 | 8.3391 (±10 %) | 2.8378 (39.5) | 4.6135 (39.0) | 2.8429 (39.5) | 1.63 | 0.34 | 1.00 |
| 1e-6 | 7 | 15 | 14492220 | 8.8234 | 4.2610 (39.2) | 5.8831 (39.1) | 4.2731 (39.2) | 1.38 | 0.48 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 20.2782 | 22.0599 |
| 1e-3 | parents | 13.7820 | 15.6294 |
| 1e-3 | child | 18.8264 | 20.5525 |
| 1e-3 | Auto | 13.8623 | 15.6313 |
| 1e-6 | Rotation | 38.7055 | 41.5946 |
| 1e-6 | parents | 18.0544 | 20.9506 |
| 1e-6 | child | 22.5980 | 25.5108 |
| 1e-6 | Auto | 18.3522 | 21.1298 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 27.7952 | 10 % | 5 % | 84 % | 48.8898 | 1.76 | 7.40 |
| 1e-3 | 5 | 45.5553 | 12 % | 6 % | 81 % | 84.8437 | 1.86 | 7.79 |
| 1e-3 | 6 | 66.7339 | 16 % | 7 % | 77 % | 140.2528 | 2.10 | 7.40 |
| 1e-3 | 7 | 71.4308 | 17 % | 9 % | 74 % | 132.4463 | 1.85 | 5.61 |
| 1e-6 | 4 | 30.0398 | 9 % | 5 % | 86 % | 48.3358 | 1.61 | 6.44 |
| 1e-6 | 5 | 51.6930 | 11 % | 5 % | 84 % | 87.7547 | 1.70 | 7.18 |
| 1e-6 | 6 | 78.7510 | 13 % | 6 % | 81 % | 145.5286 | 1.85 | 6.81 |
| 1e-6 | 7 | 113.8156 | 10 % | 6 % | 84 % | 173.0978 | 1.52 | 5.31 |

#### Kathleen job 630738 (second rule), w1-500, 4 × 10

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 36 | 511200 | 3.8924 | 1.6155 (±6 %) (8.2) | 1.8908 (9.8) | 1.5691 (8.2) | 1.17 | 0.40 | 0.97 |
| 1e-3 | 5 | 21 | 2935296 | 5.5150 (±16 %) | 1.9196 (9.6) | 2.5782 (10.0) | 1.8938 (9.6) | 1.34 | 0.34 | 0.99 |
| 1e-3 | 6 | 13 | 10439280 | 4.4742 | 2.5305 (9.9) | 4.1106 (9.6) | 2.5585 (9.9) | 1.62 | 0.57 | 1.01 |
| 1e-3 | 7 | 8 | 14492220 | 1.9471 | 2.6143 (9.9) | 4.1589 (9.9) | 2.6651 (9.9) | 1.59 | 1.37 | 1.02 |
| 1e-6 | 4 | 41 | 511200 | 5.8806 | 1.8748 (±10 %) (8.2) | 2.2016 (9.9) | 1.9030 (8.1) | 1.17 | 0.32 | 1.02 |
| 1e-6 | 5 | 25 | 2935296 | 7.7105 | 2.1965 (9.6) | 2.8776 (10.0) | 2.1937 (9.6) | 1.31 | 0.28 | 1.00 |
| 1e-6 | 6 | 16 | 10439280 | 7.7120 | 3.0347 (9.9) | 4.5024 (9.8) | 3.0260 (9.9) | 1.48 | 0.39 | 1.00 |
| 1e-6 | 7 | 15 | 14492220 | 10.3752 (±14 %) | 4.4599 (9.9) | 6.0597 (9.9) | 4.5167 (9.9) | 1.36 | 0.44 | 1.01 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 23.2132 | 24.4219 |
| 1e-3 | parents | 16.2257 | 17.4870 |
| 1e-3 | child | 20.3246 | 21.7351 |
| 1e-3 | Auto | 16.0921 | 17.5566 |
| 1e-6 | Rotation | 41.3120 | 43.4400 |
| 1e-6 | parents | 21.0415 | 23.0683 |
| 1e-6 | child | 25.2059 | 27.2297 |
| 1e-6 | Auto | 20.9889 | 23.2220 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 7.2733 | 8 % | 5 % | 87 % | 10.4443 | 1.44 | 5.90 |
| 1e-3 | 5 | 11.5996 | 11 % | 6 % | 83 % | 18.6485 | 1.61 | 5.96 |
| 1e-3 | 6 | 17.0851 | 14 % | 7 % | 79 % | 30.9227 | 1.81 | 6.24 |
| 1e-3 | 7 | 18.0322 | 15 % | 9 % | 76 % | 32.0819 | 1.78 | 5.73 |
| 1e-6 | 4 | 7.8220 | 8 % | 4 % | 88 % | 10.9335 | 1.40 | 6.13 |
| 1e-6 | 5 | 13.2584 | 10 % | 5 % | 85 % | 20.0853 | 1.51 | 5.97 |
| 1e-6 | 6 | 20.4117 | 12 % | 6 % | 82 % | 33.4577 | 1.64 | 5.99 |
| 1e-6 | 7 | 29.2288 | 9 % | 6 % | 85 % | 42.5866 | 1.46 | 5.41 |

#### Kathleen job 630738 (second rule), w3-100, 1 × 40

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0170 | 0.1326 (1.0) | 0.0170 (30.5) | 0.0150 (30.1) | 0.13 | 0.88 | 0.11 |
| 1e-3 | 3 | 18 | 53352 | 0.0574 | 0.0764 (7.8) | 0.0401 (37.7) | 0.0396 (37.6) | 0.53 | 0.69 | 0.52 |
| 1e-3 | 4 | 11 | 584136 | 0.1607 | 0.1253 (30.2) | 0.1760 (39.4) | 0.1271 (30.7) | 1.40 | 0.79 | 1.01 |
| 1e-3 | 5 | 7 | 5398920 | 0.5339 | 0.4830 (38.3) | 1.0627 (39.2) | 0.4833 (38.4) | 2.20 | 0.91 | 1.00 |
| 1e-3 | 6 | 7 | 46298376 | 4.4245 | 3.5449 (39.7) | 8.2350 (39.1) | 3.5453 (39.7) | 2.32 | 0.80 | 1.00 |
| 1e-6 | 2 | 35 | 3096 | 0.0266 | 0.1506 (1.0) | 0.0198 (±17 %) (31.3) | 0.0213 (±10 %) (30.6) | 0.13 | 0.80 | 0.14 |
| 1e-6 | 3 | 22 | 53352 | 0.1006 | 0.0888 (7.9) | 0.0465 (36.8) | 0.0452 (37.8) | 0.52 | 0.45 | 0.51 |
| 1e-6 | 4 | 15 | 584136 | 0.3529 | 0.1566 (30.5) | 0.2061 (38.4) | 0.1564 (30.8) | 1.32 | 0.44 | 1.00 |
| 1e-6 | 5 | 15 | 5398920 | 3.2867 | 0.8316 (38.5) | 1.4068 (39.0) | 0.8242 (38.5) | 1.69 | 0.25 | 0.99 |
| 1e-6 | 6 | 15 | 46298376 | 28.2577 | 6.1344 (39.7) | 10.5667 (39.6) | 6.1569 (39.7) | 1.72 | 0.22 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 5.1936 | 6.4877 |
| 1e-3 | parents | 4.3623 | 5.6534 |
| 1e-3 | child | 9.5309 | 10.9139 |
| 1e-3 | Auto | 4.2103 | 5.6499 |
| 1e-6 | Rotation | 32.0246 | 34.7138 |
| 1e-6 | parents | 7.3621 | 10.0524 |
| 1e-6 | child | 12.2459 | 14.9097 |
| 1e-6 | Auto | 7.2040 | 9.7396 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.1246 | 7 % | 5 % | 87 % | 0.2767 | 2.22 | 9.90 |
| 1e-3 | 3 | 0.5099 | 13 % | 7 % | 80 % | 1.0544 | 2.07 | 6.12 |
| 1e-3 | 4 | 2.8448 | 21 % | 8 % | 71 % | 5.7338 | 2.02 | 5.48 |
| 1e-3 | 5 | 13.7023 | 28 % | 9 % | 62 % | 36.5715 | 2.67 | 6.52 |
| 1e-3 | 6 | 103.7007 | 30 % | 9 % | 61 % | 285.6903 | 2.75 | 6.53 |
| 1e-6 | 2 | 0.1419 | 7 % | 4 % | 89 % | 0.3370 | 2.38 | 14.35 |
| 1e-6 | 3 | 0.5892 | 11 % | 5 % | 84 % | 1.1746 | 1.99 | 6.55 |
| 1e-6 | 4 | 3.4914 | 17 % | 6 % | 77 % | 6.3523 | 1.82 | 5.44 |
| 1e-6 | 5 | 22.0314 | 16 % | 6 % | 78 % | 45.2182 | 2.05 | 6.79 |
| 1e-6 | 6 | 168.1961 | 17 % | 6 % | 77 % | 342.0311 | 2.03 | 6.57 |

#### Kathleen job 630738 (second rule), w3-100, 4 × 10

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0326 | 0.0536 (1.0) | 0.0354 (4.6) | 0.0346 (4.7) | 0.66 | 1.06 | 0.65 |
| 1e-3 | 3 | 18 | 53352 | 0.0584 | 0.0676 (3.1) | 0.0501 (9.3) | 0.0677 (3.1) | 0.74 | 1.16 | 1.00 |
| 1e-3 | 4 | 11 | 584136 | 0.1606 | 0.1304 (8.3) | 0.1832 (9.8) | 0.1312 (8.4) | 1.40 | 0.82 | 1.01 |
| 1e-3 | 5 | 7 | 5398920 | 0.5215 | 0.5109 (9.8) | 1.0536 (9.9) | 0.5059 (9.8) | 2.06 | 0.97 | 0.99 |
| 1e-3 | 6 | 7 | 46298376 | 5.1358 (±12 %) | 3.7252 (9.9) | 7.9028 (9.7) | 3.7097 (9.9) | 2.12 | 0.72 | 1.00 |
| 1e-6 | 2 | 35 | 3096 | 0.0493 | 0.0620 (1.0) | 0.0411 (±5 %) (4.5) | 0.0392 (4.4) | 0.66 | 0.79 | 0.63 |
| 1e-6 | 3 | 22 | 53352 | 0.0989 | 0.0820 (3.0) | 0.0619 (±5 %) (8.8) | 0.0797 (3.1) | 0.76 | 0.81 | 0.97 |
| 1e-6 | 4 | 15 | 584136 | 0.4542 (±20 %) | 0.1691 (8.6) | 0.2578 (±12 %) (8.6) | 0.1698 (8.5) | 1.52 | 0.37 | 1.00 |
| 1e-6 | 5 | 15 | 5398920 | 3.4108 | 0.9320 (9.8) | 1.4869 (9.7) | 0.9343 (9.8) | 1.60 | 0.27 | 1.00 |
| 1e-6 | 6 | 15 | 46298376 | 29.3334 | 6.6145 (9.9) | 10.6503 (9.7) | 6.6406 (9.9) | 1.61 | 0.23 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 5.9089 (±10 %) | 6.8287 (±8 %) |
| 1e-3 | parents | 4.4877 | 5.4483 |
| 1e-3 | child | 9.2251 | 10.1462 |
| 1e-3 | Auto | 4.4491 | 5.4036 |
| 1e-6 | Rotation | 33.3467 | 35.3616 |
| 1e-6 | parents | 7.8597 | 9.8870 |
| 1e-6 | child | 12.4981 | 14.4913 |
| 1e-6 | Auto | 7.8635 | 9.8965 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0345 | 9 % | 7 % | 85 % | 0.0637 | 1.85 | 5.07 |
| 1e-3 | 3 | 0.1378 | 12 % | 7 % | 82 % | 0.2383 | 1.73 | 5.10 |
| 1e-3 | 4 | 0.7037 | 17 % | 8 % | 76 % | 1.3091 | 1.86 | 5.93 |
| 1e-3 | 5 | 3.4316 | 25 % | 10 % | 65 % | 8.8516 | 2.58 | 6.83 |
| 1e-3 | 6 | 25.8956 | 26 % | 10 % | 64 % | 65.4397 | 2.53 | 6.40 |
| 1e-6 | 2 | 0.0388 | 7 % | 7 % | 87 % | 0.0697 | 1.80 | 6.20 |
| 1e-6 | 3 | 0.1571 | 11 % | 6 % | 83 % | 0.2711 | 1.73 | 5.06 |
| 1e-6 | 4 | 0.8999 | 13 % | 6 % | 81 % | 1.5214 | 1.69 | 5.54 |
| 1e-6 | 5 | 5.9352 | 14 % | 6 % | 80 % | 10.7531 | 1.81 | 6.10 |
| 1e-6 | 6 | 42.9874 | 15 % | 6 % | 79 % | 79.3031 | 1.84 | 6.15 |

#### Kathleen job 630756 (third rule), w1-100, 1 × 40

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0182 | 0.1292 (1.0) | 0.0140 (30.4) | 0.0142 (29.7) | 0.11 | 0.78 | 0.11 |
| 1e-3 | 3 | 18 | 53352 | 0.0563 | 0.0737 (7.8) | 0.0374 (37.3) | 0.0371 (37.5) | 0.51 | 0.66 | 0.50 |
| 1e-3 | 4 | 11 | 385344 | 0.1135 (±7 %) | 0.0931 (30.5) | 0.1208 (39.5) | 0.0928 (31.2) | 1.30 | 0.82 | 1.00 |
| 1e-3 | 5 | 7 | 244944 | 0.0259 (±8 %) | 0.0666 (33.7) | 0.0783 (37.8) | 0.0679 (34.0) | 1.18 | 2.62 | 1.02 |
| 1e-6 | 2 | 35 | 3096 | 0.0267 | 0.1474 (1.0) | 0.0157 (30.1) | 0.0159 (30.1) | 0.11 | 0.60 | 0.11 |
| 1e-6 | 3 | 22 | 53352 | 0.0993 | 0.0875 (7.8) | 0.0433 (37.6) | 0.0438 (37.4) | 0.49 | 0.44 | 0.50 |
| 1e-6 | 4 | 15 | 385344 | 0.2333 | 0.1179 (30.4) | 0.1461 (38.3) | 0.1188 (30.6) | 1.24 | 0.51 | 1.01 |
| 1e-6 | 5 | 15 | 244944 | 0.1473 | 0.1159 (34.5) | 0.1226 (39.5) | 0.1164 (34.8) | 1.06 | 0.79 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 0.2140 | 0.2973 |
| 1e-3 | parents | 0.3626 | 0.4501 |
| 1e-3 | child | 0.2504 | 0.3314 |
| 1e-3 | Auto | 0.2122 | 0.2965 |
| 1e-6 | Rotation | 0.5066 | 0.7127 |
| 1e-6 | parents | 0.4686 | 0.6635 |
| 1e-6 | child | 0.3276 | 0.5308 |
| 1e-6 | Auto | 0.2948 | 0.4989 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.1229 | 7 % | 5 % | 88 % | 0.2537 | 2.06 | 9.61 |
| 1e-3 | 3 | 0.4953 | 13 % | 5 % | 82 % | 1.0090 | 2.04 | 6.18 |
| 1e-3 | 4 | 2.1203 | 20 % | 7 % | 73 % | 3.8369 | 1.81 | 4.63 |
| 1e-3 | 5 | 1.6298 | 13 % | 9 % | 78 % | 2.2610 | 1.39 | 3.70 |
| 1e-6 | 2 | 0.1401 | 6 % | 4 % | 90 % | 0.2776 | 1.98 | 9.77 |
| 1e-6 | 3 | 0.5810 | 11 % | 4 % | 85 % | 1.1366 | 1.96 | 6.31 |
| 1e-6 | 4 | 2.6019 | 16 % | 6 % | 79 % | 4.3599 | 1.68 | 4.66 |
| 1e-6 | 5 | 2.7409 | 7 % | 5 % | 87 % | 3.4156 | 1.25 | 3.71 |

#### Kathleen job 630756 (third rule), w1-100, 4 × 10

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0326 | 0.0533 (1.0) | 0.0324 (4.3) | 0.0333 (4.3) | 0.61 | 1.02 | 0.63 |
| 1e-3 | 3 | 18 | 53352 | 0.0558 | 0.0720 (±7 %) (2.9) | 0.0473 (9.4) | 0.0486 (9.2) | 0.66 | 0.87 | 0.67 |
| 1e-3 | 4 | 11 | 385344 | 0.1081 | 0.1070 (7.2) | 0.1273 (9.9) | 0.0991 (7.7) | 1.19 | 0.92 | 0.93 |
| 1e-3 | 5 | 7 | 244944 | 0.0237 | 0.0753 (7.7) | 0.0765 (9.8) | 0.0704 (8.0) | 1.02 | 2.98 | 0.93 |
| 1e-6 | 2 | 35 | 3096 | 0.0499 | 0.0604 (1.0) | 0.0375 (4.3) | 0.0380 (4.3) | 0.62 | 0.76 | 0.63 |
| 1e-6 | 3 | 22 | 53352 | 0.1009 | 0.0805 (3.0) | 0.0576 (9.3) | 0.0579 (9.4) | 0.72 | 0.57 | 0.72 |
| 1e-6 | 4 | 15 | 385344 | 0.2418 | 0.1286 (7.8) | 0.1547 (9.9) | 0.1340 (7.5) | 1.20 | 0.55 | 1.04 |
| 1e-6 | 5 | 15 | 244944 | 0.1520 | 0.1334 (8.0) | 0.1462 (9.9) | 0.1348 (8.0) | 1.10 | 0.89 | 1.01 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 0.2202 | 0.2892 |
| 1e-3 | parents | 0.3076 | 0.3700 |
| 1e-3 | child | 0.2836 | 0.3582 |
| 1e-3 | Auto | 0.2515 | 0.3237 |
| 1e-6 | Rotation | 0.5446 | 0.7368 |
| 1e-6 | parents | 0.4029 | 0.5890 |
| 1e-6 | child | 0.3960 | 0.5895 |
| 1e-6 | Auto | 0.3647 | 0.5583 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0335 | 8 % | 6 % | 87 % | 0.0583 | 1.74 | 5.56 |
| 1e-3 | 3 | 0.1440 | 12 % | 5 % | 83 % | 0.2348 | 1.63 | 5.22 |
| 1e-3 | 4 | 0.5261 | 15 % | 7 % | 78 % | 0.8951 | 1.70 | 5.47 |
| 1e-3 | 5 | 0.4072 | 10 % | 9 % | 81 % | 0.5295 | 1.30 | 4.21 |
| 1e-6 | 2 | 0.0382 | 7 % | 6 % | 87 % | 0.0647 | 1.69 | 5.98 |
| 1e-6 | 3 | 0.1552 | 11 % | 5 % | 85 % | 0.2640 | 1.70 | 5.42 |
| 1e-6 | 4 | 0.6684 | 12 % | 5 % | 83 % | 1.0261 | 1.54 | 5.34 |
| 1e-6 | 5 | 0.7049 | 6 % | 5 % | 89 % | 0.8976 | 1.27 | 4.89 |

#### Kathleen job 630756 (third rule), w1-500, 4 × 10

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 36 | 511200 | 4.7080 (±16 %) | 1.5928 (8.1) | 1.9811 (9.9) | 1.5463 (8.1) | 1.24 | 0.33 | 0.97 |
| 1e-3 | 5 | 21 | 2935296 | 5.5902 (±17 %) | 1.8828 (9.6) | 2.7544 (9.2) | 1.8952 (9.6) | 1.46 | 0.34 | 1.01 |
| 1e-3 | 6 | 13 | 10439280 | 4.4050 | 2.5003 (9.9) | 4.1550 (9.4) | 2.5039 (9.9) | 1.66 | 0.57 | 1.00 |
| 1e-3 | 7 | 8 | 14492220 | 1.8900 | 2.6141 (9.9) | 4.0777 (9.8) | 2.6054 (9.9) | 1.56 | 1.38 | 1.00 |
| 1e-6 | 4 | 41 | 511200 | 5.9145 | 1.7983 (8.1) | 2.1540 (±5 %) (9.9) | 1.9399 (8.1) | 1.20 | 0.33 | 1.08 |
| 1e-6 | 5 | 25 | 2935296 | 7.7555 | 2.2204 (9.6) | 2.8626 (10.0) | 2.1828 (9.6) | 1.29 | 0.28 | 0.98 |
| 1e-6 | 6 | 16 | 10439280 | 7.7889 | 2.9991 (9.9) | 4.5159 (9.8) | 3.0267 (9.9) | 1.51 | 0.39 | 1.01 |
| 1e-6 | 7 | 15 | 14492220 | 9.0167 | 4.4372 (9.9) | 6.0960 (9.6) | 4.4327 (9.9) | 1.37 | 0.49 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 24.2004 | 25.2079 |
| 1e-3 | parents | 16.1301 | 17.6526 |
| 1e-3 | child | 21.2295 | 22.0686 |
| 1e-3 | Auto | 16.1321 | 17.6266 |
| 1e-6 | Rotation | 40.0066 | 42.3386 |
| 1e-6 | parents | 21.1660 | 23.4087 |
| 1e-6 | child | 25.3911 | 27.3432 |
| 1e-6 | Auto | 21.3312 | 23.3079 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 4 | 7.0589 | 8 % | 4 % | 87 % | 10.6406 | 1.51 | 6.01 |
| 1e-3 | 5 | 11.5082 | 11 % | 6 % | 83 % | 18.2887 | 1.59 | 5.91 |
| 1e-3 | 6 | 16.8714 | 14 % | 6 % | 80 % | 30.8582 | 1.83 | 6.17 |
| 1e-3 | 7 | 17.7174 | 15 % | 8 % | 77 % | 31.7321 | 1.79 | 5.75 |
| 1e-6 | 4 | 7.7962 | 8 % | 4 % | 89 % | 11.2712 | 1.45 | 6.12 |
| 1e-6 | 5 | 12.9305 | 10 % | 5 % | 86 % | 20.2137 | 1.56 | 6.18 |
| 1e-6 | 6 | 19.9165 | 12 % | 5 % | 83 % | 33.8549 | 1.70 | 6.20 |
| 1e-6 | 7 | 28.7690 | 9 % | 5 % | 86 % | 42.1298 | 1.46 | 5.41 |

#### Kathleen job 630756 (third rule), w3-100, 1 × 40

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0180 | 0.1342 (1.0) | 0.0171 (31.2) | 0.0172 (±13 %) (31.5) | 0.13 | 0.95 | 0.13 |
| 1e-3 | 3 | 18 | 53352 | 0.0551 | 0.0770 (7.8) | 0.0405 (37.7) | 0.0406 (37.4) | 0.53 | 0.74 | 0.53 |
| 1e-3 | 4 | 11 | 584136 | 0.1607 | 0.1276 (30.5) | 0.1822 (38.5) | 0.1288 (30.6) | 1.43 | 0.80 | 1.01 |
| 1e-3 | 5 | 7 | 5398920 | 0.5040 | 0.4833 (38.5) | 1.1218 (39.8) | 0.4802 (38.5) | 2.32 | 0.95 | 0.99 |
| 1e-3 | 6 | 7 | 46298376 | 4.4050 | 3.4967 (39.7) | 8.4172 (39.8) | 3.5497 (39.7) | 2.41 | 0.81 | 1.02 |
| 1e-6 | 2 | 35 | 3096 | 0.0267 | 0.1562 (1.0) | 0.0258 (±10 %) (31.7) | 0.0217 (±8 %) (29.9) | 0.17 | 0.81 | 0.14 |
| 1e-6 | 3 | 22 | 53352 | 0.1006 | 0.0923 (7.8) | 0.0463 (38.0) | 0.0469 (±5 %) (37.8) | 0.50 | 0.47 | 0.51 |
| 1e-6 | 4 | 15 | 584136 | 0.3536 | 0.1589 (30.5) | 0.2067 (38.8) | 0.1559 (30.6) | 1.30 | 0.44 | 0.98 |
| 1e-6 | 5 | 15 | 5398920 | 3.4607 (±5 %) | 0.8096 (38.5) | 1.4019 (39.0) | 0.8139 (38.5) | 1.73 | 0.24 | 1.01 |
| 1e-6 | 6 | 15 | 46298376 | 28.2291 | 6.1208 (39.7) | 10.5255 (39.6) | 6.1029 (39.7) | 1.72 | 0.22 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 5.1428 | 6.4773 |
| 1e-3 | parents | 4.3187 | 5.6848 |
| 1e-3 | child | 9.7789 | 11.0989 |
| 1e-3 | Auto | 4.2165 | 5.5476 |
| 1e-6 | Rotation | 32.1707 | 34.8372 |
| 1e-6 | parents | 7.3378 | 9.9017 |
| 1e-6 | child | 12.2062 | 14.8078 |
| 1e-6 | Auto | 7.1413 | 9.6448 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.1247 | 8 % | 5 % | 87 % | 0.2879 | 2.31 | 10.83 |
| 1e-3 | 3 | 0.5124 | 13 % | 7 % | 80 % | 1.0559 | 2.06 | 6.26 |
| 1e-3 | 4 | 2.9636 | 24 % | 9 % | 67 % | 5.8197 | 1.96 | 4.81 |
| 1e-3 | 5 | 13.6946 | 29 % | 8 % | 62 % | 39.2624 | 2.87 | 7.00 |
| 1e-3 | 6 | 101.8569 | 30 % | 8 % | 61 % | 296.4135 | 2.91 | 6.93 |
| 1e-6 | 2 | 0.1420 | 7 % | 5 % | 88 % | 0.3438 | 2.42 | 13.77 |
| 1e-6 | 3 | 0.5880 | 11 % | 5 % | 84 % | 1.2284 | 2.09 | 7.23 |
| 1e-6 | 4 | 3.5205 | 18 % | 6 % | 76 % | 6.4015 | 1.82 | 5.08 |
| 1e-6 | 5 | 21.7824 | 17 % | 5 % | 78 % | 44.9304 | 2.06 | 6.55 |
| 1e-6 | 6 | 166.9641 | 18 % | 5 % | 77 % | 340.5941 | 2.04 | 6.41 |

#### Kathleen job 630756 (third rule), w3-100, 4 × 10

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0317 | 0.0545 (1.0) | 0.0342 (4.4) | 0.0349 (4.5) | 0.63 | 1.10 | 0.64 |
| 1e-3 | 3 | 18 | 53352 | 0.0573 | 0.0666 (3.1) | 0.0487 (9.3) | 0.0492 (9.4) | 0.73 | 0.86 | 0.74 |
| 1e-3 | 4 | 11 | 584136 | 0.1605 | 0.1305 (8.4) | 0.1827 (9.9) | 0.1291 (8.4) | 1.40 | 0.80 | 0.99 |
| 1e-3 | 5 | 7 | 5398920 | 0.5259 | 0.4968 (9.8) | 1.0471 (10.0) | 0.5113 (9.8) | 2.11 | 0.97 | 1.03 |
| 1e-3 | 6 | 7 | 46298376 | 4.5918 | 3.6569 (9.9) | 7.9315 (9.6) | 3.6884 (9.9) | 2.17 | 0.80 | 1.01 |
| 1e-6 | 2 | 35 | 3096 | 0.0497 | 0.0605 (1.0) | 0.0402 (4.6) | 0.0387 (4.5) | 0.66 | 0.78 | 0.64 |
| 1e-6 | 3 | 22 | 53352 | 0.1012 | 0.0792 (3.1) | 0.0595 (9.3) | 0.0576 (9.4) | 0.75 | 0.57 | 0.73 |
| 1e-6 | 4 | 15 | 584136 | 0.3624 | 0.1672 (8.4) | 0.2332 (±8 %) (9.0) | 0.1693 (8.3) | 1.40 | 0.47 | 1.01 |
| 1e-6 | 5 | 15 | 5398920 | 3.9648 (±15 %) | 0.9310 (9.8) | 1.4562 (9.9) | 0.9346 (9.8) | 1.56 | 0.24 | 1.00 |
| 1e-6 | 6 | 15 | 46298376 | 29.3657 | 6.5931 (9.9) | 10.5252 (9.7) | 6.5458 (9.9) | 1.60 | 0.22 | 0.99 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 5.3671 | 6.3522 |
| 1e-3 | parents | 4.4054 | 5.3783 |
| 1e-3 | child | 9.2441 | 10.1670 |
| 1e-3 | Auto | 4.4129 | 5.3757 |
| 1e-6 | Rotation | 33.8437 | 35.7862 |
| 1e-6 | parents | 7.8311 | 9.8715 |
| 1e-6 | child | 12.3144 | 14.4042 |
| 1e-6 | Auto | 7.7461 | 9.8178 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0340 | 8 % | 7 % | 87 % | 0.0621 | 1.83 | 6.39 |
| 1e-3 | 3 | 0.1364 | 13 % | 6 % | 81 % | 0.2369 | 1.74 | 4.81 |
| 1e-3 | 4 | 0.6919 | 17 % | 7 % | 76 % | 1.2956 | 1.87 | 5.72 |
| 1e-3 | 5 | 3.3646 | 25 % | 9 % | 66 % | 8.8420 | 2.63 | 6.84 |
| 1e-3 | 6 | 25.4908 | 27 % | 9 % | 64 % | 65.0556 | 2.55 | 6.37 |
| 1e-6 | 2 | 0.0386 | 7 % | 7 % | 87 % | 0.0702 | 1.82 | 5.98 |
| 1e-6 | 3 | 0.1576 | 11 % | 5 % | 85 % | 0.2660 | 1.69 | 5.07 |
| 1e-6 | 4 | 0.8643 | 14 % | 5 % | 81 % | 1.4697 | 1.70 | 5.52 |
| 1e-6 | 5 | 5.8326 | 14 % | 5 % | 80 % | 10.8228 | 1.86 | 6.25 |
| 1e-6 | 6 | 42.2467 | 15 % | 5 % | 80 % | 78.5482 | 1.86 | 6.13 |

#### the M6, w1-100, 1 × 6

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0268 | 0.0409 (1.0) | 0.0178 (5.5) | 0.0184 (5.3) | 0.44 | 0.69 | 0.45 |
| 1e-3 | 3 | 18 | 53352 | 0.1171 | 0.0533 (±9 %) (4.3) | 0.0595 (5.7) | 0.0488 (4.7) | 1.12 | 0.42 | 0.92 |
| 1e-3 | 4 | 11 | 385344 | 0.2428 | 0.1426 (5.8) | 0.2020 (6.0) | 0.1291 (5.8) | 1.42 | 0.53 | 0.91 |
| 1e-3 | 5 | 7 | 244944 | 0.0574 | 0.0972 (±6 %) (5.9) | 0.1337 (6.0) | 0.0970 (5.9) | 1.38 | 1.69 | 1.00 |
| 1e-6 | 2 | 35 | 3096 | 0.0497 (±10 %) | 0.0470 (1.0) | 0.0203 (5.7) | 0.0217 (4.9) | 0.43 | 0.44 | 0.46 |
| 1e-6 | 3 | 22 | 53352 | 0.2120 | 0.0596 (4.6) | 0.0739 (±9 %) (5.5) | 0.0603 (4.7) | 1.24 | 0.28 | 1.01 |
| 1e-6 | 4 | 15 | 385344 | 0.5452 | 0.1765 (5.8) | 0.2580 (±5 %) (6.0) | 0.1822 (5.8) | 1.46 | 0.33 | 1.03 |
| 1e-6 | 5 | 15 | 244944 | 0.3971 (±10 %) | 0.1973 (5.9) | 0.2325 (6.0) | 0.1968 (5.9) | 1.18 | 0.50 | 1.00 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 0.4441 | 0.5739 |
| 1e-3 | parents | 0.3340 | 0.4637 |
| 1e-3 | child | 0.4131 | 0.5398 |
| 1e-3 | Auto | 0.2934 | 0.4195 |
| 1e-6 | Rotation | 1.2040 | 1.5267 |
| 1e-6 | parents | 0.4804 | 0.7929 |
| 1e-6 | child | 0.5846 | 0.8980 |
| 1e-6 | Auto | 0.4610 | 0.7766 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0321 | 6 % | 4 % | 90 % | 0.0497 | 1.55 | 6.50 |
| 1e-3 | 3 | 0.1459 | 14 % | 6 % | 79 % | 0.2313 | 1.59 | 5.09 |
| 1e-3 | 4 | 0.4778 | 20 % | 9 % | 70 % | 0.8934 | 1.87 | 5.24 |
| 1e-3 | 5 | 0.3396 | 16 % | 13 % | 71 % | 0.5457 | 1.61 | 4.83 |
| 1e-6 | 2 | 0.0375 | 5 % | 3 % | 92 % | 0.0582 | 1.55 | 8.15 |
| 1e-6 | 3 | 0.1680 | 11 % | 5 % | 83 % | 0.2696 | 1.60 | 5.67 |
| 1e-6 | 4 | 0.6008 | 16 % | 7 % | 77 % | 1.0445 | 1.74 | 5.50 |
| 1e-6 | 5 | 0.6469 | 8 % | 7 % | 85 % | 0.8678 | 1.34 | 5.01 |

#### the M6, w3-100, 1 × 6

| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) | Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 30 | 3096 | 0.0321 | 0.0478 (±6 %) (1.0) | 0.0220 (5.4) | 0.0200 (5.8) | 0.46 | 0.62 | 0.42 |
| 1e-3 | 3 | 18 | 53352 | 0.1545 (±14 %) | 0.0523 (4.7) | 0.0721 (5.2) | 0.0524 (4.8) | 1.38 | 0.34 | 1.00 |
| 1e-3 | 4 | 11 | 584136 | 0.4230 (±6 %) | 0.1961 (5.8) | 0.3469 (5.7) | 0.1934 (5.8) | 1.77 | 0.46 | 0.99 |
| 1e-3 | 5 | 7 | 5398920 | 1.3523 | 0.9599 (6.0) | 2.0074 (6.0) | 0.9406 (±5 %) (6.0) | 2.09 | 0.70 | 0.98 |
| 1e-3 | 6 | 7 | 46298376 | 12.0620 | 6.9273 (6.0) | 14.0163 (5.8) | 6.8370 (6.0) | 2.02 | 0.57 | 0.99 |
| 1e-6 | 2 | 35 | 3096 | 0.0495 | 0.0576 (±8 %) (1.0) | 0.0237 (5.5) | 0.0253 (5.0) | 0.41 | 0.51 | 0.44 |
| 1e-6 | 3 | 22 | 53352 | 0.2292 | 0.0693 (±6 %) (4.6) | 0.0749 (6.0) | 0.0664 (4.7) | 1.08 | 0.29 | 0.96 |
| 1e-6 | 4 | 15 | 584136 | 0.9256 | 0.2643 (5.9) | 0.3731 (6.0) | 0.2562 (±6 %) (5.8) | 1.41 | 0.28 | 0.97 |
| 1e-6 | 5 | 15 | 5398920 | 8.7524 | 1.6557 (±8 %) (6.0) | 2.6281 (5.7) | 1.6746 (6.0) | 1.59 | 0.19 | 1.01 |
| 1e-6 | 6 | 15 | 46298376 | 76.6215 | 11.9010 (6.0) | 18.7226 (5.7) | 11.7151 (6.0) | 1.57 | 0.15 | 0.98 |

| ε | run | M2L Σ s | evaluate s |
| --- | --- | ---: | ---: |
| 1e-3 | Rotation | 14.0239 | 16.5304 |
| 1e-3 | parents | 8.1833 | 10.7142 |
| 1e-3 | child | 16.4647 | 19.0905 |
| 1e-3 | Auto | 8.0433 | 10.6884 |
| 1e-6 | Rotation | 86.5783 | 91.0154 |
| 1e-6 | parents | 13.9479 | 18.4253 |
| 1e-6 | child | 21.8225 | 26.3552 |
| 1e-6 | Auto | 13.7375 | 18.2396 |

| ε | level | parents: busy s | shifts | children | conversions | child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2 | 0.0370 | 7 % | 5 % | 88 % | 0.0626 | 1.69 | 6.31 |
| 1e-3 | 3 | 0.1523 | 14 % | 7 % | 79 % | 0.2629 | 1.73 | 5.43 |
| 1e-3 | 4 | 0.6816 | 23 % | 9 % | 68 % | 1.4811 | 2.17 | 5.80 |
| 1e-3 | 5 | 3.7143 | 34 % | 11 % | 54 % | 9.7824 | 2.63 | 5.60 |
| 1e-3 | 6 | 26.4135 | 35 % | 11 % | 54 % | 65.9968 | 2.50 | 5.18 |
| 1e-6 | 2 | 0.0459 | 6 % | 4 % | 90 % | 0.0678 | 1.48 | 6.06 |
| 1e-6 | 3 | 0.2012 | 11 % | 8 % | 81 % | 0.2899 | 1.44 | 5.10 |
| 1e-6 | 4 | 0.9230 | 17 % | 7 % | 75 % | 1.6018 | 1.74 | 5.39 |
| 1e-6 | 5 | 6.0777 | 20 % | 7 % | 73 % | 11.3637 | 1.87 | 5.34 |
| 1e-6 | 6 | 42.5077 | 21 % | 7 % | 72 % | 79.9897 | 1.88 | 5.16 |
