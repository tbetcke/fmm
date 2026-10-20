# Phase 6 T11: the final benchmarks and the defaults

Phase 6, task T11 ([docs/phase6/T11-benchmarks.md](../../docs/phase6/T11-benchmarks.md)): the
components of Phase 6 measured together on the three machines, the defaults revisited with
those numbers (decision 7 of [docs/phase6/README.md](../../docs/phase6/README.md), **signed
off by the user on 2026-10-20, "approved as written"**), and the multi-node runs of Phase 5S
T9 repeated with the new defaults. Measured on 2026-10-19 and 2026-10-20 from branch
`tbetcke/phase6_t11` (main at 4bac1ae, T12 and T13 merged, plus this task's changes), release
builds, rustc 1.99.0, every BLAS thread variable at 1 (no BLAS is linked). The path taken:
**with Kathleen** (docs/phase6/README.md, "Kathleen is usable"): one node for the strategy
matrix, the leaf study and the calibration, 1, 2 and 4 nodes for the scaling runs, 2 for
the correctness job; no job used more than four nodes.

**Every figure is measured** (the machine, ranks × threads and build stated with each
table) unless labelled *model*. The one-node "before" figures are the library's defaults
before T11 (`Auto` under the rule of Phase 5N T4, the separate multiply–add, the near rule
off, `Isa::detect`'s AVX2 on Kathleen, 64 points per leaf), measured in the same sweep, on
the same day and machine, as the "after"; the multi-node "before" is Phase 5S's report
([phase5s-kathleen.md](phase5s-kathleen.md) §2, §3), on other days and nodes (Phase 5S T9
found the same code 3–6 % apart on different 4-node sets).

At the user's request the runs were trimmed on 2026-10-20 (the M3 Max and locust were
shared with another session): the strategy matrix stopped where only the Plummer sphere at
N = 10⁷ was missing (§2), the leaf study ran at N = 10⁶ for the winning strategy per p on the
M3 Max and locust, the Metal and CUDA device sweeps were cut down, the multi-node runs are
N = 10⁷ at 40 × 1 only (no N = 10⁸), and overlap's default was not revisited.

## Summary

- **New defaults** (decision 7, signed off on 2026-10-20; §1):

  | default | before T11 | since T11 |
  | --- | --- | --- |
  | M2L strategy of `Auto` on the host | `Dense` while the dense tables are ≤ 9 MB per rank and ranks per node × those ≤ the node budget, `Rotation` otherwise (Phase 5N T4) | **`Batched` at f64 p ≤ 6 and f32 p ≤ 7, `PlaneWave` above**, whatever the ranks per node |
  | multiply–add of `Batched` | separate | **fused from p = 6**, separate below |
  | near rule (C6.9) | off | **from the cost model** |
  | P2P ISA (`Isa::detect`) | AVX2 on an AVX-512 machine | **AVX-512** where the CPU has AVX-512F |
  | leaf size (`max_points_per_leaf`) | 64 | **host: 128 from p = 12**; Metal: 128 at p = 6–7, 256 from p = 8; 64 otherwise |
  | cost-model constants | Phase 5S (AVX2, Dense/Rotation) and T13 | **refitted** with the new operators and AVX-512 |
  | overlap, device strategies, CUDA gather and layout | – | unchanged (§10, §11) |

- **One node, the new defaults against the old** (same sweep, same day; §2):

  | machine | ranks × threads | f64 p = 3 | f64 p = 6 | f64 p = 8 | f64 p = 12 | f64 p = 18 | f32 p = 3–8 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | M3 Max | 1 × 12, 12 × 1 | 1.08–1.99× | 1.77–3.50× | 2.70–6.30× | 4.11–8.24× | 5.41–9.42× | 1.13–7.74× |
  | locust | 1 × 72, 8 × 9, 72 × 1 | 0.92–1.49× | 1.33–2.45× | 1.45–3.35× | 2.54–5.24× | 3.79–6.16× | 0.97–5.45× |
  | Kathleen | 1 × 40, 4 × 10, 40 × 1 | 1.36–3.09× | 2.21–5.50× | 3.11–6.74× | 4.50–9.76× | 6.14–11.45× | 1.14–8.19× |

  (the cube and the Plummer sphere at N = 10⁶ and 10⁷; the time of the old defaults over the
  new; the new leaf size of p = 12 and 18 not included, which gains another 6–21 %, §5.)
  The largest gains are where both new operators meet: high p (PlaneWave) and the Plummer
  sphere (the near rule). The only cells slower than before are on locust at p = 3 (§3).
- **Multi-node** (Kathleen, 40 × 1 per node, N = 10⁷ strong and 2.5 × 10⁴ points per rank
  weak; §6): **1.86–7.99× faster than Phase 5S** at 1, 2 and 4 nodes in strong scaling, 1.2–10.6×
  in weak scaling. Strong-scaling efficiency from 1 to 4 nodes falls from Phase 5S's
  0.93–1.00 to 0.79–0.93, as each rank has 2–8× less work; the parts that shrank least from 1
  to 4 nodes are PlaneWave's downward pass (2.95× for 4×) and the output's move back (16–30 %
  of the four-node time; flat on the Plummer sphere at f64 p = 8).
- **Accuracy**: every variant's errors against the direct sum within 0.92–1.05× of the old
  defaults' on all three machines (bounds 1.1×: C6.5's for PlaneWave, C6.9's restated one for
  the near rule; §4). One-rank outputs change, as requirement 6 foresees (§1).
- **The leaf size** (Phase 3S T7's rule on each backend, §5): the host keeps 64 to p = 11 and
  takes 128 from p = 12 (6–21 % faster); at p = 6–8 Kathleen alone would gain (5–11 % at
  128), the NEON machines not. Metal f32 takes 128 at p = 6 (1.15×) and 256 at p = 8 (1.30×).
- **AVX-512** (§3): faster than or equal to AVX2 (within run-to-run variation) in every cell,
  1.0–1.8× for the new strategies inside the FMM on a fully loaded node.
- **Not decided here**: CUDA's gather and layout per p and CUDA's leaf size (§10: the H100
  was held by another user's job throughout), overlap's default (§11).

## 1. What changed

Code (decision 7; the PR's description lists the files):
- `nd_fmm_exec::tables::M2lStrategy::resolve_on_node`: `Auto` resolves on the host to
  `Batched` below `auto_plane_wave_min_p` (`AUTO_PLANE_WAVE_MIN_P_F64` = 7,
  `AUTO_PLANE_WAVE_MIN_P_F32` = 8) and to `PlaneWave` from it. The ranks per node and the
  node budget (`FmmBuilder::dense_node_budget`) are still agreed and reported, no longer
  read; `StrategyChoice`'s reason reads `Auto → PlaneWave: f64 p = 8 ≥ 7`. The device path
  keeps its own rule (on a device `Batched` and `PlaneWave` run as `Dense`).
- `nd_fmm_exec::fmm::default_m2l_mul_add(p)` (`FUSED_MUL_ADD_MIN_P` = 6): the builder's mode
  when it names none.
- `NearRuleChoice::CostModel` is the default (`FmmBuilder::near_rule`).
- `nd_fmm_simd::Isa::detect` returns `Isa::Avx512` where the CPU has AVX-512F, AVX2 and FMA.
- `nd_fmm_exec::fmm::default_max_points_per_leaf(p, backend)` when the builder names no leaf
  size; `Fmm::max_points_per_leaf` reports the one used.
- `nd_fmm_exec::cost::CostModel::calibrated`: 13 calibrations (§7), `Batched` its own,
  `Auto` that of its host rule's strategy.
- The `scaling` harness takes the library's defaults (`--mul-add auto`, the near rule, the
  leaf size; `--leaf-size n` new) and reports the leaf size (`leaf=`); `tools/scaling/run.sh`
  has the sweeps `t11-host`, `t11-leaf`, `t11-calibrate` and `t11-device`, and its quiet gate
  works on locust too (`QUIET_LINUX`) and voids and repeats a launch during which other
  processes took more than `QUIET_DURING` (§2).

**One-rank outputs change** (requirement 6) wherever `Auto` resolves to another strategy
(`Batched` separate gives `Dense`'s bits, so only where `Auto` gave `Rotation`, or the mode is
fused, or `PlaneWave`), wherever the near rule converts a pair (trees with W or X lists
between leaves), on AVX-512 machines (P2P's terms, within C3S.5's contract), and at p ≥ 12 on
the host and p ≥ 6 on Metal (the tree). The errors stay within §4's ratios. A strategy named
with the near rule on gets the crossovers of its own cost model, so two strategies no longer
convert the same pairs; the tests that compare strategies name the same crossovers or the
rule off (§12).

## 2. Setup

| machine | node | runs | load |
| --- | --- | --- | --- |
| M3 Max | Apple M3 Max, 12 performance + 4 efficiency cores, 64 GB, macOS; Metal (40-core GPU) | `tools/scaling/run.sh t11-host` (337 launches, 2026-10-20 05:56–08:25), the trimmed `t11-leaf` (54) and `t11-device` (24), `nd-fmm-bench --backend host,metal` (to 08:43); unsandboxed | the quiet gate: other processes below 200 % CPU before and every 5 s during each launch (else void and repeated: 41 repeats over 33 launches of `t11-host`, 18 in the leaf and Metal sweeps); the first attempt of 2026-10-19 (contended by another session) discarded |
| locust | GH200: 72 Neoverse-V2 cores, one NUMA node; H100 | `t11-host` (518 launches, 2026-10-20 05:57–08:28), the trimmed `t11-leaf` (54, 08:31–08:35) | another user's vLLM held the H100 (80 GB, 100 %) and one core throughout; the gate as on the M3 Max (`QUIET_LINUX=1`), other processes at most 1.2 cores during any launch |
| Kathleen | 2 × Xeon Gold 6248 (Cascade Lake, AVX-512F), 40 cores, 192 GB, Omni-Path; `--exclude=node-c11a-011` | jobs of §13 | exclusive nodes |

The splits: 1 × cores, Phase 5N's recommendation (8 × 9 on locust, 4 × 10 on Kathleen; 1 × 12
on the M3 Max is both) and cores × 1. Every launch with `--kinds`, overlap off, the second cut
on, the errors of one charge vector at 1,000 targets (N = 10⁶; at N = 10⁷ only with eight ranks
or more), three timed evaluations at N = 10⁶ (two at p = 12, one at p = 18) and two at 10⁷
(one at p ≥ 12); the time is the median over the evaluations of the slowest rank's
evaluation. Every variant ran with the near rule of the cost model, `default` with the old
defaults (above). The sweeps of §2a–§5 ran
before the refit of §7, so their cut and their near rule used the constants before T11 (at
f64 p = 8 under PlaneWave the crossovers 399 and 194 points, at f64 p = 3 under Batched
Dense's 107 and 35; the shipped model gives 853 and 410, 217 and 73); the multi-node runs of
§6 used the shipped ones. **Re-run with the final code** (Kathleen job 606302, 2026-10-20,
node-c11b-040; the cube and the Plummer sphere at N = 10⁶, the new default strategy at f64
and f32 p = 3 and 8, 4 × 10 and 40 × 1; ms, the sweep's time → the final code's):

| problem | 4 × 10 | 40 × 1 |
| --- | --- | --- |
| cube f64 p = 3 (Batched) | 65.6 → 65.2 | 70.0 → **52.1** |
| cube f64 p = 8 (PlaneWave) | 150.2 → 145.5 | 160.6 → 163.1 |
| cube f32 p = 3 (Batched) | 47.7 → 46.1 | 36.5 → 36.9 |
| cube f32 p = 8 (PlaneWave) | 117.2 → 116.6 | 129.6 → 127.9 |
| Plummer f64 p = 3 (Batched) | 86.8 → 86.7 | 67.9 → 66.6 |
| Plummer f64 p = 8 (PlaneWave) | 208.4 → 211.8 | 253.9 → 238.4 |
| Plummer f32 p = 3 (Batched) | 62.6 → 63.6 | 51.2 → 49.7 |
| Plummer f32 p = 8 (PlaneWave) | 167.3 → 165.7 | 185.1 → 182 (repeats; 317 once) |

The same W and X pairs convert under both constants (every W and X pair between leaves on the
Plummer sphere, 200,968 each on these runs; the cube has none at N = 10⁶), so the near rule's
output and cost are those of the sweep; the times agree within run-to-run variation, except
the cube at f64 p = 3 on 40 × 1, 1.34× faster: the cut there now weighs Batched's M2L with
its own constant (48 ns per pair against Dense's 117). The Plummer sphere at f32 p = 8 on
40 × 1 took 317 ms once in 606302 with one rank 1.85× slower than the mean in every stage, and
182–183 ms in two repeats on the same node (jobs 606303, 606304): a slow rank, not the
defaults.

## 2a. One node: the new defaults against the old (measured)

Evaluation time in ms, old defaults → new defaults (the ratio in bold), leaf size 64 in both
(the new leaf size at p = 12 and 18 in §5). The new defaults' strategy per cell: `Batched`
separate at p ≤ 5, fused at p = 6 (and f32 p = 7), `PlaneWave` from f64 p = 7 and f32 p = 8; on
Kathleen with AVX-512.

**M3 Max**

| problem | 1 × 12 | 12 × 1 |
| --- | ---: | ---: |
| cube 10⁶ f64 p = 3 | 120 → 102 (**1.18×**) | 122 → 102 (**1.20×**) |
| cube 10⁶ f64 p = 6 | 360 → 170 (**2.12×**) | 372 → 178 (**2.09×**) |
| cube 10⁶ f64 p = 8 | 662 → 183 (**3.61×**) | 638 → 221 (**2.88×**) |
| cube 10⁶ f64 p = 12 | 1,605 → 290 (**5.54×**) | 1,497 → 361 (**4.14×**) |
| cube 10⁶ f64 p = 18 | 5,640 → 677 (**8.33×**) | 4,967 → 918 (**5.41×**) |
| cube 10⁶ f32 p = 3 | 76.9 → 61.4 (**1.25×**) | 80.0 → 63.1 (**1.27×**) |
| cube 10⁶ f32 p = 6 | 220 → 106 (**2.07×**) | 223 → 106 (**2.11×**) |
| cube 10⁶ f32 p = 8 | 478 → 129 (**3.71×**) | 531 → 148 (**3.60×**) |
| cube 10⁷ f64 p = 3 | 1,558 → 1,364 (**1.14×**) | 1,324 → 1,228 (**1.08×**) |
| cube 10⁷ f64 p = 6 | 3,672 → 1,956 (**1.88×**) | 3,458 → 1,956 (**1.77×**) |
| cube 10⁷ f64 p = 8 | 5,759 → 1,976 (**2.91×**) | 5,644 → 2,094 (**2.70×**) |
| cube 10⁷ f64 p = 12 | 13,528 → 3,025 (**4.47×**) | 13,618 → 3,316 (**4.11×**) |
| cube 10⁷ f64 p = 18 | 39,684 → 6,293 (**6.31×**) | 46,247 → 7,876 (**5.87×**) |
| cube 10⁷ f32 p = 3 | 901 → 799 (**1.13×**) | 777 → 653 (**1.19×**) |
| cube 10⁷ f32 p = 6 | 2,172 → 1,223 (**1.78×**) | 2,071 → 1,113 (**1.86×**) |
| cube 10⁷ f32 p = 8 | 4,576 → 1,343 (**3.41×**) | 4,740 → 1,357 (**3.49×**) |
| plummer 10⁶ f64 p = 3 | 259 → 130 (**1.99×**) | 243 → 133 (**1.83×**) |
| plummer 10⁶ f64 p = 6 | 827 → 236 (**3.50×**) | 833 → 245 (**3.39×**) |
| plummer 10⁶ f64 p = 8 | 1,520 → 241 (**6.30×**) | 1,424 → 308 (**4.63×**) |
| plummer 10⁶ f64 p = 12 | 3,305 → 402 (**8.23×**) | 3,480 → 563 (**6.18×**) |
| plummer 10⁶ f64 p = 18 | 9,249 → 982 (**9.42×**) | 9,924 → 1,485 (**6.68×**) |
| plummer 10⁶ f32 p = 3 | 232 → 82.6 (**2.81×**) | 203 → 83.2 (**2.44×**) |
| plummer 10⁶ f32 p = 6 | 644 → 149 (**4.33×**) | 647 → 147 (**4.39×**) |
| plummer 10⁶ f32 p = 8 | 1,347 → 174 (**7.74×**) | 1,303 → 207 (**6.30×**) |
| plummer 10⁷ f64 p = 3 | 2,315 → 1,396 (**1.66×**) | 2,224 → 1,330 (**1.67×**) |
| plummer 10⁷ f64 p = 6 | 8,456 → 2,507 (**3.37×**) | 7,606 → 2,480 (**3.07×**) |
| plummer 10⁷ f64 p = 8 | 13,498 → 2,434 (**5.55×**) | 13,144 → 2,832 (**4.64×**) |
| plummer 10⁷ f64 p = 12 | 32,818 → 3,983 (**8.24×**) | 32,964 → 4,369 (**7.54×**) |

**locust**

| problem | 1 × 72 | 8 × 9 | 72 × 1 |
| --- | ---: | ---: | ---: |
| cube 10⁶ f64 p = 3 | 80.0 → 78.7 (**1.02×**) | 47.4 → 42.9 (**1.11×**) | 79.1 → 53.3 (**1.49×**) |
| cube 10⁶ f64 p = 6 | 142 → 99.5 (**1.43×**) | 109 → 64.1 (**1.70×**) | 149 → 83.5 (**1.78×**) |
| cube 10⁶ f64 p = 8 | 187 → 107 (**1.75×**) | 160 → 77.9 (**2.05×**) | 249 → 172 (**1.45×**) |
| cube 10⁶ f64 p = 12 | 395 → 130 (**3.04×**) | 365 → 109 (**3.34×**) | 660 → 218 (**3.03×**) |
| cube 10⁶ f64 p = 18 | 1,017 → 248 (**4.10×**) | 1,210 → 233 (**5.20×**) | 2,519 → 462 (**5.46×**) |
| cube 10⁶ f32 p = 3 | 44.7 → 45.8 (**0.97×**) | 21.9 → 20.5 (**1.07×**) | 31.5 → 27.9 (**1.13×**) |
| cube 10⁶ f32 p = 6 | 79.9 → 59.8 (**1.34×**) | 53.7 → 32.2 (**1.67×**) | 115 → 42.8 (**2.69×**) |
| cube 10⁶ f32 p = 8 | 136 → 63.5 (**2.15×**) | 113 → 40.0 (**2.82×**) | 189 → 59.4 (**3.19×**) |
| cube 10⁷ f64 p = 3 | 951 → 940 (**1.01×**) | 589 → 568 (**1.04×**) | 647 → 705 (**0.92×**) |
| cube 10⁷ f64 p = 6 | 1,490 → 1,121 (**1.33×**) | 1,106 → 738 (**1.50×**) | 1,340 → 881 (**1.52×**) |
| cube 10⁷ f64 p = 8 | 1,879 → 1,175 (**1.60×**) | 1,583 → 832 (**1.90×**) | 1,896 → 1,005 (**1.89×**) |
| cube 10⁷ f64 p = 12 | 3,563 → 1,402 (**2.54×**) | 3,292 → 1,107 (**2.97×**) | 4,932 → 1,559 (**3.16×**) |
| cube 10⁷ f64 p = 18 | 8,862 → 2,341 (**3.79×**) | 8,910 → 2,063 (**4.32×**) | 18,542 → 3,296 (**5.63×**) |
| cube 10⁷ f32 p = 3 | 521 → 538 (**0.97×**) | 262 → 258 (**1.02×**) | 376 → 299 (**1.26×**) |
| cube 10⁷ f32 p = 6 | 851 → 644 (**1.32×**) | 552 → 358 (**1.54×**) | 898 → 514 (**1.75×**) |
| cube 10⁷ f32 p = 8 | 1,340 → 681 (**1.97×**) | 1,074 → 435 (**2.47×**) | 1,532 → 588 (**2.60×**) |
| plummer 10⁶ f64 p = 3 | 115 → 98.2 (**1.17×**) | 80.3 → 55.1 (**1.46×**) | 102 → 81.6 (**1.25×**) |
| plummer 10⁶ f64 p = 6 | 255 → 128 (**2.00×**) | 216 → 91.1 (**2.37×**) | 301 → 147 (**2.06×**) |
| plummer 10⁶ f64 p = 8 | 368 → 135 (**2.73×**) | 339 → 101 (**3.35×**) | 434 → 230 (**1.88×**) |
| plummer 10⁶ f64 p = 12 | 808 → 176 (**4.59×**) | 835 → 159 (**5.24×**) | 1,095 → 333 (**3.29×**) |
| plummer 10⁶ f64 p = 18 | 2,023 → 351 (**5.77×**) | 2,155 → 377 (**5.71×**) | 4,039 → 826 (**4.89×**) |
| plummer 10⁶ f32 p = 3 | 80.6 → 62.4 (**1.29×**) | 52.9 → 27.7 (**1.91×**) | 95.5 → 39.5 (**2.42×**) |
| plummer 10⁶ f32 p = 6 | 178 → 78.2 (**2.27×**) | 153 → 45.2 (**3.38×**) | 258 → 71.3 (**3.61×**) |
| plummer 10⁶ f32 p = 8 | 306 → 83.4 (**3.67×**) | 295 → 54.1 (**5.45×**) | 400 → 111 (**3.61×**) |
| plummer 10⁷ f64 p = 3 | 1,113 → 951 (**1.17×**) | 728 → 546 (**1.33×**) | 802 → 663 (**1.21×**) |
| plummer 10⁷ f64 p = 6 | 2,380 → 1,237 (**1.92×**) | 2,041 → 832 (**2.45×**) | 2,276 → 1,088 (**2.09×**) |
| plummer 10⁷ f64 p = 8 | 3,361 → 1,327 (**2.53×**) | 3,184 → 1,026 (**3.10×**) | 3,474 → 1,337 (**2.60×**) |
| plummer 10⁷ f64 p = 12 | 7,512 → 1,755 (**4.28×**) | 7,747 → 1,538 (**5.04×**) | 8,711 → 2,157 (**4.04×**) |
| plummer 10⁷ f64 p = 18 | 19,884 → 3,442 (**5.78×**) | 21,375 → 3,471 (**6.16×**) | – |

**Kathleen**

| problem | 1 × 40 | 4 × 10 | 40 × 1 |
| --- | ---: | ---: | ---: |
| cube 10⁶ f64 p = 3 | 141 → 104 (**1.36×**) | 105 → 65.6 (**1.59×**) | 96.7 → 70.0 (**1.38×**) |
| cube 10⁶ f64 p = 6 | 352 → 144 (**2.45×**) | 304 → 106 (**2.86×**) | 332 → 98.7 (**3.36×**) |
| cube 10⁶ f64 p = 7 | 536 → 186 (**2.88×**) | 432 → 124 (**3.48×**) | 439 → 133 (**3.30×**) |
| cube 10⁶ f64 p = 8 | 627 → 202 (**3.11×**) | 564 → 150 (**3.76×**) | 584 → 161 (**3.64×**) |
| cube 10⁶ f64 p = 12 | 1,519 → 310 (**4.89×**) | 1,428 → 247 (**5.79×**) | 1,578 → 300 (**5.26×**) |
| cube 10⁶ f64 p = 18 | 4,047 → 659 (**6.14×**) | 4,130 → 573 (**7.21×**) | 5,365 → 815 (**6.58×**) |
| cube 10⁶ f32 p = 3 | 84.8 → 73.5 (**1.15×**) | 55.6 → 47.7 (**1.17×**) | 49.9 → 36.5 (**1.37×**) |
| cube 10⁶ f32 p = 6 | 211 → 112 (**1.89×**) | 171 → 83.5 (**2.04×**) | 321 → 77.4 (**4.14×**) |
| cube 10⁶ f32 p = 7 | 281 → 131 (**2.13×**) | 228 → 84.8 (**2.68×**) | 419 → 84.9 (**4.94×**) |
| cube 10⁶ f32 p = 8 | 451 → 154 (**2.92×**) | 372 → 117 (**3.18×**) | 584 → 130 (**4.50×**) |
| cube 10⁷ f64 p = 3 | 1,692 → 1,226 (**1.38×**) | 1,230 → 776 (**1.59×**) | 1,076 → 553 (**1.95×**) |
| cube 10⁷ f64 p = 6 | 3,529 → 1,598 (**2.21×**) | 3,226 → 1,141 (**2.83×**) | 3,420 → 1,014 (**3.37×**) |
| cube 10⁷ f64 p = 7 | 4,621 → 1,755 (**2.63×**) | 4,070 → 1,290 (**3.16×**) | 3,989 → 1,159 (**3.44×**) |
| cube 10⁷ f64 p = 8 | 5,876 → 1,882 (**3.12×**) | 5,291 → 1,455 (**3.64×**) | 5,302 → 1,649 (**3.21×**) |
| cube 10⁷ f64 p = 12 | 12,939 → 2,875 (**4.50×**) | 12,727 → 2,392 (**5.32×**) | 13,932 → 2,500 (**5.57×**) |
| cube 10⁷ f64 p = 18 | 37,782 → 5,811 (**6.50×**) | 36,869 → 5,373 (**6.86×**) | 46,781 → 6,172 (**7.58×**) |
| cube 10⁷ f32 p = 3 | 1,007 → 883 (**1.14×**) | 659 → 523 (**1.26×**) | 540 → 368 (**1.47×**) |
| cube 10⁷ f32 p = 6 | 1,981 → 1,194 (**1.66×**) | 1,726 → 837 (**2.06×**) | 2,819 → 752 (**3.75×**) |
| cube 10⁷ f32 p = 7 | 2,571 → 1,245 (**2.07×**) | 2,477 → 892 (**2.78×**) | 3,681 → 850 (**4.33×**) |
| cube 10⁷ f32 p = 8 | 3,692 → 1,452 (**2.54×**) | 3,552 → 1,149 (**3.09×**) | 5,036 → 1,098 (**4.59×**) |
| plummer 10⁶ f64 p = 3 | 336 → 136 (**2.48×**) | 251 → 86.8 (**2.89×**) | 210 → 67.9 (**3.09×**) |
| plummer 10⁶ f64 p = 6 | 818 → 209 (**3.92×**) | 795 → 147 (**5.42×**) | 763 → 139 (**5.50×**) |
| plummer 10⁶ f64 p = 7 | 1,113 → 234 (**4.76×**) | 1,059 → 177 (**5.97×**) | 979 → 200 (**4.89×**) |
| plummer 10⁶ f64 p = 8 | 1,473 → 265 (**5.56×**) | 1,405 → 208 (**6.74×**) | 1,328 → 254 (**5.23×**) |
| plummer 10⁶ f64 p = 12 | 3,228 → 429 (**7.52×**) | 3,355 → 364 (**9.21×**) | 3,491 → 468 (**7.45×**) |
| plummer 10⁶ f64 p = 18 | 8,742 → 1,003 (**8.72×**) | 9,750 → 851 (**11.45×**) | 10,474 → 1,213 (**8.63×**) |
| plummer 10⁶ f32 p = 3 | 260 → 98.5 (**2.64×**) | 201 → 62.6 (**3.22×**) | 173 → 51.2 (**3.37×**) |
| plummer 10⁶ f32 p = 6 | 665 → 156 (**4.27×**) | 606 → 110 (**5.48×**) | 762 → 103 (**7.43×**) |
| plummer 10⁶ f32 p = 7 | 789 → 169 (**4.67×**) | 801 → 114 (**7.01×**) | 1,011 → 129 (**7.85×**) |
| plummer 10⁶ f32 p = 8 | 1,138 → 217 (**5.24×**) | 1,165 → 167 (**6.97×**) | 1,341 → 185 (**7.24×**) |
| plummer 10⁷ f64 p = 3 | 2,509 → 1,388 (**1.81×**) | 2,052 → 875 (**2.35×**) | 1,766 → 646 (**2.73×**) |
| plummer 10⁷ f64 p = 6 | 7,408 → 1,956 (**3.79×**) | 6,869 → 1,405 (**4.89×**) | 6,943 → 1,333 (**5.21×**) |
| plummer 10⁷ f64 p = 7 | 9,975 → 2,248 (**4.44×**) | 9,848 → 1,663 (**5.92×**) | 9,264 → 1,643 (**5.64×**) |
| plummer 10⁷ f64 p = 8 | 12,924 → 2,503 (**5.16×**) | 12,625 → 1,905 (**6.63×**) | 12,098 → 1,880 (**6.44×**) |
| plummer 10⁷ f64 p = 12 | 31,160 → 3,875 (**8.04×**) | 31,848 → 3,262 (**9.76×**) | 31,519 → 3,459 (**9.11×**) |
| plummer 10⁷ f64 p = 18 | 86,981 → 8,748 (**9.94×**) | 84,709 → 7,858 (**10.78×**) | 99,251 → 9,315 (**10.66×**) |
| plummer 10⁷ f32 p = 3 | 1,903 → 1,019 (**1.87×**) | 1,550 → 619 (**2.51×**) | 1,399 → 446 (**3.13×**) |
| plummer 10⁷ f32 p = 6 | 5,165 → 1,462 (**3.53×**) | 4,861 → 1,043 (**4.66×**) | 6,878 → 978 (**7.03×**) |
| plummer 10⁷ f32 p = 7 | 6,935 → 1,525 (**4.55×**) | 6,869 → 1,148 (**5.99×**) | 9,323 → 1,183 (**7.88×**) |
| plummer 10⁷ f32 p = 8 | 10,162 → 1,916 (**5.30×**) | 9,967 → 1,506 (**6.62×**) | 12,362 → 1,510 (**8.19×**) |

(The M3 Max lacks the Plummer sphere at N = 10⁷ in f32 and at f64 p = 18, locust its 72 × 1 run at f64 p = 18: the trimmed runs.)

## 3. The strategy rule and the ISA (measured)

Against the fastest variant measured in each cell (Batched separate and fused, Rotation,
PlaneWave, and at N = 10⁶ and p ≤ 12 Dense and Classes; on Kathleen each on AVX-512, and
Batched fused and PlaneWave also on AVX2), the new defaults' time over the best:

| machine | f32 p = 3 | f32 p = 6 | f32 p = 7 | f32 p = 8 | f64 p = 3 | f64 p = 6 | f64 p = 7 | f64 p = 8 | f64 p = 12 | f64 p = 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 Max | 1.03–1.04 | 1.00–1.13 | – | 1.00 | 1.01–1.06 | 1.04–1.28 | – | 1.00 | 1.00 | 1.00 |
| locust | 1.00–1.15 | 1.00–1.09 | – | 1.00 | 1.00–1.04 | 1.00–1.14 | – | 1.00 | 1.00 | 1.00 |
| Kathleen | 1.00–1.03 | 1.00–1.04 | 1.00–1.04 | 1.00–1.05 | 1.00–1.36 | 1.00 | 1.00–1.01 | 1.00 | 1.00 | 1.00–1.01 |

- **PlaneWave** is the fastest at f64 p ≥ 8 in every cell of every machine, and at f64 p = 7 in
  11 of 12 on Kathleen (a tie in the twelfth); at f64 p = 6 it is 4–28 % faster than Batched on
  the M3 Max, mixed on locust (Batched over PlaneWave 0.81–1.14) and 0.81–1.00 on Kathleen,
  so the threshold stays
  at 7 for every machine (the alternative, NEON from 6, was offered and not taken).
- **Batched fused** is the fastest or within 1.05 at f64 p = 6 and f32 p = 3–7 on Kathleen;
  on the NEON machines PlaneWave is up to 1.28× faster at f64 p = 6 (M3 Max) and the best of
  the others up to 1.09–1.15× at f32 p = 3–6 (the table above); at f32 p = 8
  PlaneWave and Batched fused are within 10 % of each other on Kathleen, PlaneWave 1.2–1.6×
  ahead on NEON.
- **The multiply–add mode at p = 3** is where the machines disagree: on locust at 72 ranks ×
  1 thread f64 p = 3 fused took 82 against 53 ms separate (the cube, 10⁶) and 766 against
  705 ms (10⁷); on Kathleen with AVX-512 fused was the faster at f64 p = 3 (51.5 against 70.0 ms
  on the cube at 40 × 1, up to 1.36× in the sweep), and on the M3 Max by 1.4–6 %. The signed-off
  rule (fused from p = 6) takes the separate mode there, so the new defaults are within 1.04
  (locust) and 1.06 (M3 Max) of the best at f64 p = 3, and within 1.36 on Kathleen in the
  sweep; with the final cost model (§2) that cell takes 52.1 ms separate, level with the
  sweep's fused 51.5 ms (the 1.36 was mostly the cut weighing Batched with Dense's constants). On locust at f64 p = 3 they are 0.92–1.49×
  the old (the 0.92: the cube at 10⁷ on 72 × 1, Dense's 647 ms against Batched's 705 ms; T2
  measured Batched never more than 1.1 % slower than Dense, and one launch each does not tell
  this apart from run-to-run variation at 72 ranks).
- **Rotation, Dense and Classes were never the fastest.** Dense at one rank with many threads
  (1 × 40 on Kathleen) is far behind Batched (766 against 252 ms at f64 p = 8, the same 17.4 MB
  of tables; not profiled here, T2 traced Dense's cost to its per-target table reads).
- **AVX-512 against AVX2** (Kathleen, the time on AVX2 over the time on AVX-512, 12 cells each):

  | strategy | f32 p = 3 | 6 | 7 | 8 | f64 p = 3 | 6 | 7 | 8 | 12 | 18 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | Batched fused | 1.07–1.29 | 1.01–1.15 | 1.03–1.22 | 1.11–1.21 | 1.21–1.73 | 1.19–1.48 | 0.99–1.48 | 1.19–1.36 | 1.35–1.51 | 1.32–1.79 |
  | PlaneWave | 1.02–1.29 | 0.98–1.13 | 1.01–1.15 | 0.99–1.10 | 1.19–1.63 | 1.11–1.32 | 1.10–1.36 | 1.10–1.27 | 1.06–1.75 | 0.99–1.15 |

  Never more than 2 % slower (within run-to-run variation), up to 1.79× faster. Only Cascade
  Lake was timed; later AVX-512 cores (Intel, AMD Zen 4 and 5, CI's runners) are untimed.

## 4. Accuracy (measured)

The errors against the direct sum (one charge vector, 1,000 sampled targets, relative L2) of
each variant over those of the old defaults, over every cell of §2a (N = 10⁶ on every split,
N = 10⁷ at ≥ 8 ranks):

| variant (near rule on) | M3 Max φ / ∇φ | locust φ / ∇φ | Kathleen φ / ∇φ |
| --- | --- | --- | --- |
| Batched (separate and fused), f64 | 0.998–1.001 / 0.976–1.002 | 0.998–1.001 / 0.976–1.002 | 0.998–1.001 / 0.976–1.002 |
| Batched, f32 | 0.998–1.010 / 0.997–1.000 | 0.998–1.010 / 0.997–1.000 | 0.998–1.043 / 0.988–1.001 |
| PlaneWave, f64 | 0.945–1.030 / 0.957–1.050 | 0.945–1.030 / 0.921–1.050 | 0.945–1.030 / 0.921–1.050 |
| PlaneWave, f32 | 0.992–1.007 / 0.995–1.011 | 0.992–1.007 / 0.995–1.011 | 0.992–1.008 / 0.987–1.011 |

Within 1.1× everywhere (C6.5's bound for PlaneWave; the near rule's 1.1×, T12's question 1),
and 0.92–1.05× in fact. In the multi-node runs (§6) the errors at four nodes are those of the
same problems on one node to the digits printed, but one (the Plummer sphere at f32 p = 8:
1.424e-5 at one node, 1.426e-5 at four).

## 5. The leaf size per p (measured)

Phase 3S T7's rule, applied on each backend: per p the geometric mean of the evaluation
times over the cube and the Plummer sphere, N and precision; a size other than 64 only where
it is at least 5 % faster, at errors no worse (they were equal or better with larger leaves
except at f64 p = 18, N = 10⁷ at 40 × 1, where φ rose by up to 0.3 % at 128 and 512). The
variant is the new default's at each p (Batched fused at p ≤ 6, PlaneWave
above, both measured at p = 6 and 8). Geometric means in ms:

M3 Max (1 × 12, N = 10⁶):

| p | variant | 64 | 128 | 256 | 512 | best (64 / best) |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 3 | Batched fused | 87 | 100 | 257 | – | 64 (1.00) |
| 6 | Batched fused | 159 | 157 | 298 | – | 128 (1.01) |
| 8 | PlaneWave | 211 | 216 | 442 | 599 | 64 (1.00) |
| 12 | PlaneWave | 345 | 327 | 523 | 659 | 128 (1.06) |
| 18 | PlaneWave | 816 | 698 | 752 | 867 | 128 (1.17) |

locust (8 × 9, N = 10⁶):

| p | variant | 64 | 128 | 256 | 512 | best (64 / best) |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 3 | Batched fused | 33 | 40 | 102 | – | 64 (1.00) |
| 6 | Batched fused | 54 | 57 | 116 | – | 64 (1.00) |
| 8 | PlaneWave | 88 | 92 | 197 | 277 | 64 (1.00) |
| 12 | PlaneWave | 138 | 125 | 225 | 307 | 128 (1.10) |
| 18 | PlaneWave | 307 | 256 | 293 | 375 | 128 (1.20) |

Kathleen (4 × 10, AVX-512, N = 10⁶ and 10⁷):

| p | variant | 16 | 32 | 64 | 128 | 256 | 512 | best (64 / best) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | Batched fused | 412 | 311 | 210 | 215 | 319 | – | 64 (1.00) |
| 6 | Batched fused | 884 | 582 | 346 | 331 | 399 | – | 128 (1.05) |
| 8 | PlaneWave | 1,321 | 868 | 486 | 438 | 496 | 640 | 128 (1.11) |
| 12 | PlaneWave | 2,829 | 1,743 | 922 | 805 | 854 | 1,023 | 128 (1.14) |
| 18 | PlaneWave | 7,873 | 4,664 | 2,155 | 1,784 | 1,615 | 1,570 | 512 (1.37) |

- **The host**: 64 stays best or within 1 % at p = 3 everywhere and at p = 6–8 on the NEON
  machines; on Kathleen 128 is 5–11 % faster at p = 6–8 with the default's variant (4 × 10, and
  7.5–8 % at 40 × 1, jobs 606238–606239; 14 % only with Batched fused at p = 8, not the default
  there).
  From p = 12 every machine gains from 128 (1.06–1.20 on NEON, 1.14–1.21 on Kathleen; 512 at
  p = 18 on Kathleen 1.37–1.55, 128 and 256 on NEON). **Signed off**: 64 to p = 11, 128 from
  p = 12, on every host (the x86-only 128 from p = 6 was the alternative, not taken). p = 9–11
  were not timed.
- **Metal** (f32, one rank on the device, every core for the host's parts, N = 10⁶; ms):

  | p | 64 | 128 | 256 | rule |
  | ---: | ---: | ---: | ---: | --- |
  | 3 | 19.0 | 18.6 | 28.6 | 64 (128 only 2 % faster) |
  | 6 | 36.1 | 31.5 | 35.5 | **128** (1.15×) |
  | 8 | 65.5 | 53.1 | 50.4 | **256** (1.30×) |

  The Plummer sphere gains most (84 → 52 ms at p = 8); the cube's tree at 64 and 128 is the
  same one. The device keeps the device path's strategy rule (Dense in f32; Rotation was
  1.37–1.55× slower in every cell).
- **CUDA and the CubeCL CPU runtime** keep 64 (Phase 4S T8's rule), not re-measured (§10).

## 6. Kathleen across nodes (measured)

`tools/scaling/run.sh t9-strong t9-weak` with the new defaults (`--strategy auto`: Batched at
p = 3, PlaneWave at p = 8), 40 ranks × 1 thread per node, N = 10⁷ (strong) and 2.5 × 10⁴
points per rank (weak), both paths from one build, ten timed evaluations, the errors of one
charge vector; jobs 606248 (1 node), 606249 (2), 606250 (4), 2026-10-20 10:19. The blocking
path's wall time (ms) beside Phase 5S's (phase5s-kathleen.md §2, §3; 40 × 1, Dense at p = 3,
Rotation at p = 8):

Strong scaling, N = 10⁷:

| problem | strategy (5S → T11) | Phase 5S at 1, 2, 4 nodes | T11 at 1, 2, 4 nodes | 5S / T11 at 1, 2, 4 | 1→4 (eff.) T11 |
| --- | --- | --- | --- | --- | --- |
| cube, f64 p = 3 | Dense → Batched | 1,051, 520, 263 | 529, 265, 142 | **1.99**, **1.96**, **1.86** | 3.74 (0.93) |
| cube, f64 p = 8 | Rotation → PlaneWave | 5,488, 2,771, 1,396 | 1,332, 731, 401 | **4.12**, **3.79**, **3.48** | 3.32 (0.83) |
| cube, f32 p = 8 | Rotation → PlaneWave | 4,642, 2,298, 1,167 | 1,041, 555, 304 | **4.46**, **4.14**, **3.84** | 3.43 (0.86) |
| plummer, f64 p = 3 | Dense → Batched | 1,811, 932, 484 | 649, 342, 185 | **2.79**, **2.72**, **2.61** | 3.51 (0.88) |
| plummer, f64 p = 8 | Rotation → PlaneWave | 12,889, 6,662, 3,454 | 1,878, 1,050, 592 | **6.86**, **6.35**, **5.84** | 3.18 (0.79) |
| plummer, f32 p = 8 | Rotation → PlaneWave | 11,841, 6,099, 3,134 | 1,482, 828, 457 | **7.99**, **7.36**, **6.85** | 3.24 (0.81) |

Weak scaling, 2.5 × 10⁴ points per rank (N = 10⁶, 2 × 10⁶, 4 × 10⁶):

| problem | strategy (T11) | Phase 5S at 1, 2, 4 nodes | T11 at 1, 2, 4 nodes | 5S / T11 at 1, 2, 4 | errors φ T11 at 4 nodes |
| --- | --- | --- | --- | --- | --- |
| cube, f64 p = 3 | Batched | 95.7, 431.5, 106.3 | 54.5, 78.8, 61.5 | **1.76**, **5.48**, **1.73** | 3.049e-3 |
| cube, f64 p = 8 | PlaneWave | 614.7, 1,962.2, 1,174.7 | 164.9, 238.0, 264.5 | **3.73**, **8.24**, **4.44** | 1.909e-5 |
| cube, f32 p = 8 | PlaneWave | 533.2, 1,898.2, 1,040.0 | 135.3, 179.5, 203.1 | **3.94**, **10.57**, **5.12** | 1.910e-5 |
| cube, f32 p = 3 | Batched | 47.8, 381.7, 57.0 | 40.8, 57.1, 47.6 | **1.17**, **6.68**, **1.20** | 3.049e-3 |
| plummer, f64 p = 3 | Batched | 218.0, 207.9, 191.6 | 64.4, 67.3, 68.7 | **3.39**, **3.09**, **2.79** | 2.829e-3 |
| plummer, f64 p = 8 | PlaneWave | 1,347.5, 1,371.7, 1,278.5 | 237.4, 261.8, 264.0 | **5.68**, **5.24**, **4.84** | 1.985e-5 |
| plummer, f32 p = 8 | PlaneWave | 1,266.7, 1,268.9, 1,153.6 | 183.9, 195.3, 198.2 | **6.89**, **6.50**, **5.82** | 1.982e-5 |
| plummer, f32 p = 3 | Batched | 180.4, 174.9, 148.6 | 51.8, 50.9, 83.0 | **3.48**, **3.44**, **1.79** | 2.829e-3 |
| clusters, f64 p = 3 | Batched | 229.8, 198.2, 228.6 | 67.6, 66.6, 73.9 | **3.40**, **2.98**, **3.09** | 2.217e-3 |
| clusters, f64 p = 8 | PlaneWave | 1,501.0, 1,311.2, 1,427.8 | 253.6, 250.6, 263.9 | **5.92**, **5.23**, **5.41** | 1.613e-5 |
| clusters, f32 p = 8 | PlaneWave | 1,394.5, 1,199.1, 1,293.7 | 198.4, 192.0, 199.5 | **7.03**, **6.25**, **6.48** | 1.616e-5 |
| clusters, f32 p = 3 | Batched | 204.9, 160.1, 189.1 | 55.9, 51.0, 54.3 | **3.67**, **3.14**, **3.48** | 2.217e-3 |

- **The gain carries over**: 1.99–7.99× at one node, 1.86–6.85× at four in strong scaling, the
  ratio a little lower at four nodes as the per-rank work shrinks.
- **Strong scaling** from 1 to 4 nodes: efficiency 0.79–0.93 (Phase 5S 0.93–1.00). The
  exposed communication stays small (3–17 ms at four nodes); what scales less than the rest is
  (a) PlaneWave's downward pass (the Plummer sphere at f64 p = 8: 1,080 → 366 ms, 2.95× for
  4×; T13 found each rank forming its ghost boxes' waves too, not measured separately here)
  and (b) the backward move of the output, 16–30 % of the four-node time: it halves or better
  from 1 to 4 nodes on the cube (85 → 23 ms at f64 p = 3, 176 → 92 ms at p = 8) but stays
  flat on the Plummer sphere at f64 p = 8 (148 → 133 ms). Both are open (§11).
- **Weak scaling**: the cube at two nodes is 5.5–10.6× faster than Phase 5S: its tree is half
  refined to a seventh level there (Phase 5S §11), whose W and X lists the near rule now turns
  into P2P (C6.9's case). One cell, the Plummer sphere at f32 p = 3 on four nodes, took 83 ms
  against 51 at one and two: 44 ms of it in the backward move.

## 7. The cost model refitted (measured)

`t11-calibrate` (job 606208): 40 ranks × 1 thread on one node, the first cut, the near rule
off, the cube, the Plummer sphere and the clusters at N = 10⁷, three timed evaluations, the
constants fitted to the kind times (ns per counted unit; M2L per V pair, under PlaneWave per
box with a V row); per kind the median over the three workloads, now `CALIBRATIONS` in
`nd_fmm_exec::cost`:

| strategy, precision, p (AVX-512) | P2M | M2M | M2L | P2L | L2L | L2P | M2P | P2P | model within |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Batched fused, f64, 3 | 111.37 | 144.14 | 48.37 | 110.59 | 151.81 | 248.10 | 327.13 | 1.5080 | 3.8–4.4 % |
| Batched fused, f64, 6 | 353.85 | 1,045.83 | 228.45 | 349.31 | 1,181.14 | 671.82 | 818.92 | 1.5085 | 2.9–3.4 % |
| Batched fused, f64, 8 | 617.72 | 2,733.97 | 630.01 | 642.61 | 3,182.59 | 1,083.07 | 1,288.92 | 1.5083 | 2.3–3.5 % |
| Batched fused, f32, 3 | 122.23 | 83.10 | 40.71 | 114.08 | 93.07 | 251.10 | 324.89 | 0.9604 | 2.6–4.3 % |
| Batched fused, f32, 8 | 671.12 | 1,423.56 | 291.82 | 676.87 | 1,612.63 | 1,104.98 | 1,314.02 | 0.9617 | 2.4–3.5 % |
| Rotation, f64, 8 | 606.49 | 2,624.05 | 2,908.61 | 600.89 | 2,684.00 | 1,078.04 | 1,289.11 | 1.5109 | 2.4–3.5 % |
| Rotation, f64, 18 | 3,470.95 | 21,418.10 | 30,999.40 | 3,569.59 | 23,020.98 | 4,739.32 | 5,266.74 | 1.5181 | 3.9–5.0 % |
| PlaneWave, f64, 8 | 607.34 | 2,619.66 | 59,420.48 | 619.66 | 2,929.18 | 1,080.42 | 1,289.28 | 1.5114 | 4.5–10.3 % |
| PlaneWave, f64, 18 | 3,466.81 | 21,409.11 | 349,691.74 | 3,563.71 | 24,144.15 | 4,696.45 | 5,233.04 | 1.5087 | 4.2–9.1 % |
| PlaneWave, f32, 8 | 666.62 | 2,686.17 | 44,310.57 | 659.80 | 2,852.22 | 1,101.70 | 1,313.04 | 0.9585 | 2.8–7.2 % |

(The same fits on AVX2: P2P 3.1 ns in f64 and 1.4 in f32, Batched's M2L 0.92–1.38× AVX-512's; the
raw data keeps them.) Batched's M2L per V pair at f64 p = 8 is 630 ns, against the 3,009 of
`Dense`'s constants scaled from p = 3, which the cut used for it before (T2's open item). P2P
halves with AVX-512, so the near rule's crossovers (α_M2P / α_P2P, α_P2L / α_P2P) double
(f64 p = 8: N_W 853, N_X 410 points under PlaneWave; 399 and 194 before): nearly every W and X
pair between leaves converts. `Dense` and f32 `Rotation` keep Phase 5S's constants (named
strategies only).

## 8. The device: Metal (measured)

`nd-fmm-bench --backend host,metal --precision f32 --degree 3,6,8 --n 1000000` (the uniform
cube, one rank, every core, gradients; the old defaults, before the change): the host
70.3 / 193.4 / 434.3 ms and Metal 16.8 / 30.0 / 51.7 ms at p = 3 / 6 / 8 (medians of ten). With
the new defaults the host at 1 × 12 takes 61.4 / 106.4 / 128.8 ms (§2a, `t11-host`, the same
problem through the harness at 12 threads), and the device 16.7 / 29.5 / 48.5 ms with the new
leaf size (64, 128, 256; §5; the cube's gain is small, the Plummer sphere's 1.34× at p = 6 and
1.61× at p = 8). The host closes most
of the gap to Metal at p = 8 (2.5× instead of 8.4×). The device defaults of T7 (decision 8)
were not revisited beyond the leaf size: `METAL_SHARED_MAX_ORDER` stays 9 (T7's medium-
confidence default, §11).

## 9. Several charge vectors, dipoles, FMM3D

Not re-measured: T5's throughput per vector, T8's dipole costs and T9's FMM3D comparison
ran on the operators that are now the defaults (Batched, the AVX-512 P2P on Kathleen) or named
them, and their reports stand ([phase6-rhs.md](phase6-rhs.md),
[phase6-dipoles.md](phase6-dipoles.md), [phase6-fmm3d.md](phase6-fmm3d.md)). PlaneWave with
several vectors and dipoles is T13's (every source type, every m, tested).

## 10. CUDA

**Not measured in T11.** The CUDA benchmarks of the brief (the gather and layout per p that
T7 left, `gemm_variants` at f32 p = 2–10 and f64 p = 2–20, and a device sweep of the
strategies and leaf sizes on CUDA) were queued on locust behind a GPU-idle gate (no other GPU
compute process and utilisation below 5 % for five minutes before a run, a sampler every 2 s
during it). Another user's vLLM engine held the H100 (80 GB, 100 %) from 03:30 on 2026-10-20
throughout the task, and the user dropped the CUDA benchmarks on 2026-10-20 at 12:05. The one
sweep that ran (f32, the cube at N = 10⁶, 2026-10-19 22:38–22:50) ran under the gate's first
version, which counted only other users' processes as foreign while another session of this
user was building and testing GPU code on locust: its timings are discarded.

So CUDA's defaults stay as signed off in Phase 4S and T7 (decision 8): Phase 4S's gather and
layout, the reduction by chunk, one vector's 2 GB chunk budget, 64 points per leaf, and the
device path's strategy rule (Dense at every p). The CUDA tests ran on a Myriad A100 instead (§12).

## 11. Open

- **Overlap's default** stays off; the longer A/B with `PSM2_RCVTHREAD=0` that the brief asks
  for first was not run (trimmed). With the new operators the compute is 2–8× smaller, so the
  communication it would hide weighs more than in Phase 5S (§6: 3–17 ms exposed at four nodes).
- **The backward move of the output** is 16–30 % of a four-node evaluation and stays flat from 1
  to 4 nodes on the Plummer sphere at f64 p = 8 (§6); worth a look before the next scaling
  phase.
- **PlaneWave's downward pass across nodes** (2.95× for 4×, §6; T13's §6 found the ghost boxes'
  waves a cost at one rank per core).
- **The x86 gap of PlaneWave's scalar steps** (T13's §10) keeps the f64 threshold at 7 on
  Kathleen, where the NEON machines would gain from 6.
- **Leaf sizes at p = 9–11** were not timed; the rule switches to 128 at 12. A leaf size per
  machine class (128 from p = 6 on x86) is the measured further gain on Kathleen.
- **Other AVX-512 CPUs** (later Intel, AMD Zen 4 and 5) are untimed; the rule picks AVX-512
  there too.
- **`METAL_SHARED_MAX_ORDER`** (T7) and CUDA's gather and layout per p (§10).

## 12. The checks

Every check of the brief and the root CLAUDE.md, on the final tree unless stated (the test
changes of §1's last paragraph included):

- **The M3 Max** (macOS, the loopback flags, every MPI run under `timeout`):
  - `cargo fmt --all` and `cargo fmt -- --check`; `cargo clippy -- -D warnings`, `cargo
    clippy --examples -- -D warnings`, `cargo clippy --workspace --all-targets -- -D warnings`,
    `cargo doc --no-deps`: clean; `RUST_MIN_STACK=8388608 cargo test --workspace`: 75 of 75
    test executables pass;
  - `nd-fmm-simd`: `cargo test` and `--release -- --ignored` pass (NEON and scalar; AVX2 and
    AVX-512 run on Kathleen and CI's x86_64 runners, below), and the cross clippy
    `--target x86_64-apple-darwin`;
  - `nd-fmm-kernels` (unchanged): clippy `--features cpu,metal`, `cargo check --features
    cuda`; `nd-fmm-exec`: clippy `--features cpu,metal`, `cargo check --features cuda` (and
    `--tests`);
  - `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks: every executable and example
    passes; `mpi_exec` at 4 ranks 244 s on the final tests against 224 s on main (CI's timeout:
    below);
  - the CPU runtime: `cargo test -p nd-fmm-exec --features cpu --release` (every `mpi_exec`
    scenario repeated on the device) passes, and `mpi_exec` and the ignored `device_ranks`
    with `--features cpu,metal` at 2 and 4 ranks (the CPU runtime and Metal, each scenario on
    both) pass; the ignored gates with `--features cpu,metal`: `accuracy` (C3.2), `adaptive`
    (C3.3), `device_fmm` (C4.8), `dipoles`, `device_metal` and the operator sweeps pass
    (outside the sandbox for Metal);
  - `tests/scaling.rs` of `nd-fmm-validate` at 1 and 2 ranks (with the new leaf and mode
    fields).
- **locust** (the GH200's Grace host; the H100 held by another user's job): the MPI list at 2,
  4 and 8 ranks (on the tree of 10:13, before the last test changes), the ignored
  `accuracy` and `adaptive` gates, `mpi_exec` at 2 and 8 ranks (520 and 371 s, debug) and
  `tests/multi_rank.rs` (the C5.1 host gate, release, unbound) at 1, 2, 4 and 8 ranks (41, 62,
  53 and 50 min): every one passes. The CUDA tests did not run there (the H100 held by another
  user's job all day).
- **Myriad** (UCL's cluster, SGE; job 9998740 on node-l00a-005: one NVIDIA A100-PCIE-40GB,
  compute capability 8.0, driver 550.127.05 with CUDA 12.4, the CUDA 12.2.2 toolkit, 16 cores;
  2026-10-20 17:44–19:10; Rust 1.99.1, CubeCL's LLVM from the gridcl project's Spack build,
  the same bundle version 23.1.0-3), the CUDA tests of the root CLAUDE.md in place of
  locust's H100: `cargo test -p nd-fmm-kernels --release --features cpu,cuda --
  --include-ignored` (159 and 37 tests), `cargo clippy -p nd-fmm-exec --all-targets --features
  cpu,cuda -- -D warnings`, `cargo test -p nd-fmm-exec --features cuda --release` (every
  `mpi_exec` scenario repeated on CUDA) and the ignored `device_cuda`, `device_fmm` (C4.8),
  the operator sweeps on CUDA and the CUDA blocks of `accuracy` (C3.2), `adaptive` (C3.3) and
  `dipoles`: every one passes (75 min for the ignored ones); every kind on the host fallback
  bit for bit the host, the default placement within the device bounds (f32 φ within
  2.6–5.1e-7 of the host on `device_cuda`'s scenarios). An A100, not the H100 of Phase 4S's
  CUDA tests: the defaults are by backend, not by device, so the same code paths run.
- **Kathleen**: `correctness.sbatch` at 2 nodes (job 606247, `test`, 47 min, on the tree of
  10:13 with the new defaults): the MPI list at 2, 4, 8 and 80 ranks, rank-count invariance
  at 80 ranks against 40, 20 and 10 (`scaling --invariance`, 0 values differing), and the
  smoke run at 4 × 20 over two nodes (the one-rank reference within 1.0e-15): every step
  passes. AVX-512 is `Isa::detect`'s choice there, so every launch of it ran the new default
  ISA.
- **Not run**: the 4-node correctness job (2 nodes cover the brief's "2 Kathleen nodes");
  `tests/multi_rank.rs` on Kathleen (run on locust instead); the M3 Max's MPI list at 8 ranks
  and `multi_rank` there (run on locust, at the user's request to free the M3 Max).

Changes to the tests (the defaults they pinned): `tests/operator/strategy.rs` (the rule of
§1), the cost model's unit test (13 calibrations), `nd-fmm-simd`'s dispatch tests (AVX-512 where
the CPU has it), the `nd-fmm-bench` golden report (the leaf size's default), and in
`nd-fmm-exec`'s MPI tests the comparisons the new defaults make unequal by design: a strategy
compared with another, or the device with the host, now names the same near pairs
(`near_as`, `device_common::same_near_pairs`) or the rule off, the device's strategy and
leaf size on the host reference, and the device's cost model where boxes are compared level
by level; `check_batched`, `check_plane_wave` and `check_near_rule` build their own `Dense`,
`Rotation` or rule-off references where a scenario now resolves `Batched`, `PlaneWave` or the
rule, reusing the scenario's `Fmm` where they can; the C3.3 per-list test names the C3.3
problems' 64 points per leaf and the rule off. CI's `run-tests-mpi` timeout per launch rose
from 600 to 900 s (its steps from 20 to 25 min, the job from 50 to 60 min).

## 13. Jobs and raw data

Raw output stays outside the repository (`bench-results/t11/`, git-ignored: `m3max/`,
`locust/`, `kathleen/scaling-<job>/`; locust `/data/ucahtbe/logs/p6t11/`; Kathleen
`$FMM_KATHLEEN_ROOT/logs/scaling-<job>`).

| machine | runs | when |
| --- | --- | --- |
| M3 Max | `t11-host` (337 launches), `t11-leaf` (54, trimmed), `t11-device` (24, Metal), `nd-fmm-bench --backend host,metal`; the checks | 2026-10-20 05:56–08:43; checks to 13:00 |
| locust | `t11-host` (518), `t11-leaf` (54, trimmed); the checks | 2026-10-20 05:57–08:35; checks 09:45–14:45 |
| Kathleen | the jobs below | 2026-10-19 22:28 – 2026-10-20 16:40 |

Kathleen (every job with `--exclude=node-c11a-011`; `singlenode` unless stated):

| job | what | nodes | elapsed | node-hours |
| --- | --- | ---: | ---: | ---: |
| 606208 | `t11-calibrate` (§7) | 1 | 1:06:46 | 1.11 |
| 606209–606211, 606213–606215, 606217–606219, 606221–606223 | `t11-host`: the cube and the Plummer sphere, N = 10⁶ and 10⁷, 1 × 40, 4 × 10, 40 × 1 (§2a) | 1 each | 0:09–1:10 | 5.58 |
| 606212, 606216, 606220, 606224 | `t11-leaf` at 4 × 10 (§5) | 1 each | 0:15–0:55 | 2.25 |
| 606237 | `t11-host` at f64 and f32 p = 7 | 1 | 0:55:06 | 0.92 |
| 606238, 606239 | `t11-leaf` at 40 × 1 | 1 each | 0:26, 0:28 | 0.90 |
| 606247 | `correctness.sbatch`, `test` (§12) | 2 | 0:47:02 | 1.57 |
| 606248, 606249, 606250 | `t9-strong`, `t9-weak` at N = 10⁷, 40 × 1, with the new defaults (§6); `small` at 2 and 4 nodes | 1, 2, 4 | 0:07, 0:05, 0:04 | 0.55 |
| 606302, 606303, 606304 | the final code's cost model on 16 cells of `t11-host`, and two repeats of one (§2) | 1 each | 0:01:27, 0:00:19, 0:00:18 | 0.03 |

In all **12.9 node-hours** (with 606207, a duplicate cancelled after 22 s); no job used more
than four nodes. Queue waits: the singlenode jobs ran two at a time (the per-user limit), the
2- and 4-node jobs of 2026-10-20 started within seconds.
