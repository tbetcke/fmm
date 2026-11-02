# Helmholtz CH3.2c: the Batched M2L, and the M2L per p on three machines

Helmholtz task CH3.2c ([docs/helmholtz/tasks/CH3.2c.md](../../docs/helmholtz/tasks/CH3.2c.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§5.3 (Batched), §6.2, §6.5 and §10 CH3.2. Measured on 2026-11-02 on three machines: the M6
(Mac mini, Apple M6, 24 GB; [tools/m6/machine.md](../../tools/m6/machine.md)), locust (one
Neoverse-V2 core of the GH200; [tools/gh200/machine.md](../../tools/gh200/machine.md)) and
Kathleen (one Cascade Lake core, Xeon Gold 6248, node-c11a-011, job 618018;
[tools/kathleen/machine.md](../../tools/kathleen/machine.md)). Release builds, rustc 1.99.0,
one rank, one thread, f64, BLAS threads 1; locust and Kathleen at 64b737e, the M6 at a27a78e
(the commits between change tests and documentation only). The column kernel ran on each
machine's default ISA: NEON on the M6 and locust, AVX-512 on Kathleen. The times are
indicative (§3); nothing here is timed for a gate, and nothing changes a default.

## Summary

- **Batched equals Dense.** With real multipoles (real k, real charges) Batched in its
  default separate mode is Dense bit for bit: pair by pair on every offset of levels 2–4,
  and in the FMM, whose output is Dense's bit for bit. With complex multipoles each column
  sums the real plane's products before the imaginary plane's, where Dense alternates them
  per k, so it is Dense to rounding: at most 7.4e-16 relative L2 per local pair by pair,
  3.3e-16 in the FMM's output; the fused mode 6.9e-16 and 3.4e-16. Against Rotation as Dense
  is (7.6e-16 in the FMM). The level call is the pair body bit for bit, with one and two
  vectors, at 1 and 2 threads and on 2 and 4 ranks the one-rank FMM; the same bits on every
  ISA. Every end-to-end run below has Rotation's errors to the printed digits.
- **Per pair (§1)**, a block of 32 pairs per offset, two multipole planes (complex
  multipoles; one plane is faster still):

  | machine | fused / Rotation's speed at p = 7 | p where fused Batched stops beating Rotation | separate Batched / Dense's speed |
  | --- | ---: | --- | ---: |
  | M6 (NEON) | 1.84 | 10 (1.01), 12 (0.85) | 0.99–1.25 |
  | locust (NEON) | 1.77 | 10 (1.01), 12 (0.77) | 1.0–2.9 |
  | Kathleen (AVX-512) | 4.52 | 15 (1.25), 18 (0.76) | 2.8–8.5 |

  The fused mode is 1.2–1.8× the separate one. On the M6 the separate kernel only ties
  Dense's per-pair chain with two planes, which the compiler vectorises as well there: both
  are bound by their separate multiplies and adds, as CH3.2b found for Dense. On Kathleen
  Dense's chain takes 4–7× the M6's time while the column kernel takes about the M6's, so
  Batched is far ahead there.
- **End to end (§2)**, fused Batched against Rotation: faster on every machine where the
  levels with tables have low p. On the W4 cube at 10⁻³ (p = 7) and W3 (p = 7–12):
  0.72–0.74× Rotation's evaluation time on the M6, 0.78–0.82× on locust, 0.39–0.42× on
  Kathleen; on W1 at kD = 30 and 10⁻³ (p = 12, 8, a sparse tree) 0.93×, 1.10× and 0.57×.
  Where the levels with tables carry p = 15–35 (10⁻⁶; W1 at kD = 100) it is slower on the M6
  (1.27–1.76×) and locust (1.70–2.76×) and 0.80–1.13× on Kathleen. It takes 0.12–0.71× Dense's
  time everywhere.
- **So a per-level rule by p** would take the fused Batched M2L below a machine's crossover
  degree (about 10 on the M6 and locust, about 15 on Kathleen) and Rotation above. That rule
  is not built here (Batched is opt-in, Rotation the default; the question for sign-off).

## 1. The M2L per pair

`helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20`: one level (level 2 of three, k̂ = 1 and
1 + 0.1i on the level), the 316 offsets per pass. Rotation, Dense and Classes take each
offset with its own multipole and local; Batched takes per offset a block of 32 pairs (a
block of 64 targets on a uniform level has about 38 per offset), 32 multipoles into 32
locals, through the operator's calls restated in the example: two calls of the column kernel
per offset with one plane, four with two. Before timing, its block is compared with Dense
applied pair by pair to the same pairs. ns per pair, the mean of two launches; the rows of
k̂ = 1; the rows of k̂ = 1 + 0.1i (two planes) are within 2 % of real k̂'s two planes on
the M6, 4 % on Kathleen and, but for Dense's noisy column at p = 15–20, 3 % on locust.
"spread" is the largest difference between the two launches over the five strategies'
columns.

### M6

| p | planes | Rotation | Dense | Classes | Batched | fused | Rotation / Batched | Rotation / fused | Dense / Batched | Dense / fused | spread |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 161 | 64 | 126 | 32 | 19 | 5.03 | 8.47 | 2.00 | 3.37 | 2 % |
| 3 | 2 | 198 | 72 | 155 | 66 | 40 | 3.02 | 4.94 | 1.11 | 1.81 | 2 % |
| 4 | 1 | 257 | 148 | 217 | 76 | 46 | 3.38 | 5.59 | 1.94 | 3.21 | 1 % |
| 4 | 2 | 316 | 192 | 279 | 154 | 95 | 2.05 | 3.33 | 1.25 | 2.03 | 3 % |
| 5 | 1 | 388 | 298 | 378 | 152 | 90 | 2.54 | 4.31 | 1.95 | 3.31 | 1 % |
| 5 | 2 | 474 | 316 | 475 | 310 | 183 | 1.53 | 2.59 | 1.02 | 1.73 | 2 % |
| 6 | 1 | 560 | 548 | 680 | 284 | 166 | 1.97 | 3.39 | 1.93 | 3.31 | 1 % |
| 6 | 2 | 696 | 710 | 880 | 574 | 335 | 1.21 | 2.08 | 1.24 | 2.12 | 3 % |
| 7 | 1 | 782 | 876 | 1106 | 476 | 262 | 1.64 | 2.99 | 1.84 | 3.35 | 1 % |
| 7 | 2 | 969 | 986 | 1290 | 960 | 526 | 1.01 | 1.84 | 1.03 | 1.87 | 1 % |
| 8 | 1 | 1030 | 1415 | 1704 | 790 | 446 | 1.30 | 2.31 | 1.79 | 3.17 | 1 % |
| 8 | 2 | 1280 | 1812 | 2196 | 1574 | 893 | 0.81 | 1.43 | 1.15 | 2.03 | 2 % |
| 9 | 1 | 1336 | 2174 | 2516 | 1202 | 720 | 1.11 | 1.86 | 1.81 | 3.02 | 1 % |
| 9 | 2 | 1661 | 2370 | 2832 | 2386 | 1387 | 0.70 | 1.20 | 0.99 | 1.71 | 1 % |
| 10 | 1 | 1684 | 3227 | 3614 | 1858 | 1132 | 0.91 | 1.49 | 1.74 | 2.85 | 1 % |
| 10 | 2 | 2115 | 4020 | 4635 | 3586 | 2100 | 0.59 | 1.01 | 1.12 | 1.91 | 1 % |
| 12 | 1 | 2636 | 6376 | 6830 | 3556 | 2083 | 0.74 | 1.27 | 1.79 | 3.06 | 0 % |
| 12 | 2 | 3318 | 7838 | 8860 | 6882 | 3902 | 0.48 | 0.85 | 1.14 | 2.01 | 1 % |
| 15 | 1 | 4656 | 14714 | 14934 | 8538 | 4992 | 0.55 | 0.93 | 1.72 | 2.95 | 0 % |
| 15 | 2 | 5881 | 16416 | 17534 | 16062 | 9022 | 0.37 | 0.65 | 1.02 | 1.82 | 0 % |
| 18 | 1 | 7338 | 29780 | 30226 | 18543 | 11266 | 0.40 | 0.65 | 1.61 | 2.64 | 0 % |
| 18 | 2 | 9346 | 36418 | 38618 | 33486 | 19298 | 0.28 | 0.48 | 1.09 | 1.89 | 1 % |
| 20 | 1 | 9660 | 44226 | 44870 | 27814 | 16518 | 0.35 | 0.58 | 1.59 | 2.68 | 0 % |
| 20 | 2 | 12294 | 53986 | 56992 | 50088 | 28522 | 0.25 | 0.43 | 1.08 | 1.89 | 1 % |

Batched against Dense, two planes, at most 6.3e-15; fused at most 5.9e-15 (one plane: 0 separate).

### locust

| p | planes | Rotation | Dense | Classes | Batched | fused | Rotation / Batched | Rotation / fused | Dense / Batched | Dense / fused | spread |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 295 | 90 | 178 | 58 | 40 | 5.09 | 7.38 | 1.54 | 2.24 | 1 % |
| 3 | 2 | 355 | 122 | 246 | 120 | 84 | 2.96 | 4.23 | 1.01 | 1.45 | 1 % |
| 4 | 1 | 464 | 205 | 348 | 133 | 90 | 3.49 | 5.16 | 1.54 | 2.28 | 2 % |
| 4 | 2 | 566 | 299 | 488 | 270 | 184 | 2.09 | 3.08 | 1.11 | 1.62 | 1 % |
| 5 | 1 | 697 | 405 | 615 | 266 | 175 | 2.63 | 3.98 | 1.53 | 2.31 | 2 % |
| 5 | 2 | 856 | 582 | 850 | 524 | 348 | 1.63 | 2.46 | 1.11 | 1.67 | 0 % |
| 6 | 1 | 999 | 766 | 1137 | 477 | 304 | 2.09 | 3.28 | 1.60 | 2.51 | 6 % |
| 6 | 2 | 1242 | 1136 | 1534 | 958 | 608 | 1.30 | 2.04 | 1.19 | 1.87 | 0 % |
| 7 | 1 | 1388 | 1342 | 1871 | 785 | 490 | 1.77 | 2.83 | 1.71 | 2.73 | 2 % |
| 7 | 2 | 1752 | 2138 | 2549 | 1597 | 991 | 1.10 | 1.77 | 1.34 | 2.16 | 7 % |
| 8 | 1 | 1854 | 2402 | 2908 | 1320 | 828 | 1.40 | 2.24 | 1.82 | 2.90 | 2 % |
| 8 | 2 | 2346 | 4562 | 4270 | 2617 | 1643 | 0.90 | 1.43 | 1.74 | 2.78 | 1 % |
| 9 | 1 | 2408 | 5364 | 4218 | 2121 | 1346 | 1.14 | 1.79 | 2.53 | 3.98 | 2 % |
| 9 | 2 | 3062 | 8053 | 6554 | 4114 | 2534 | 0.74 | 1.21 | 1.96 | 3.18 | 0 % |
| 10 | 1 | 3082 | 9205 | 5818 | 3338 | 2134 | 0.92 | 1.44 | 2.76 | 4.31 | 2 % |
| 10 | 2 | 3940 | 11264 | 9603 | 6192 | 3884 | 0.64 | 1.01 | 1.82 | 2.90 | 1 % |
| 12 | 1 | 4824 | 21104 | 10664 | 7276 | 4614 | 0.66 | 1.05 | 2.90 | 4.57 | 1 % |
| 12 | 2 | 6238 | 34207 | 16127 | 12961 | 8127 | 0.48 | 0.77 | 2.64 | 4.21 | 1 % |
| 15 | 1 | 8632 | 51844 | 23887 | 15224 | 12300 | 0.57 | 0.70 | 3.41 | 4.22 | 7 % |
| 15 | 2 | 11094 | 82954 | 33971 | 28945 | 21467 | 0.38 | 0.52 | 2.87 | 3.86 | 27 % |
| 18 | 1 | 13860 | 107130 | 49849 | 47038 | 39574 | 0.29 | 0.35 | 2.28 | 2.71 | 6 % |
| 18 | 2 | 17892 | 157482 | 72058 | 86708 | 69119 | 0.21 | 0.26 | 1.82 | 2.28 | 12 % |
| 20 | 1 | 18462 | 160178 | 78452 | 73022 | 61507 | 0.25 | 0.30 | 2.19 | 2.60 | 19 % |
| 20 | 2 | 23956 | 246614 | 114548 | 129690 | 106732 | 0.18 | 0.22 | 1.90 | 2.31 | 15 % |

Batched against Dense, two planes, at most 6.1e-15; fused at most 5.8e-15 (one plane: 0 separate).

### Kathleen

| p | planes | Rotation | Dense | Classes | Batched | fused | Rotation / Batched | Rotation / fused | Dense / Batched | Dense / fused | spread |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 499 | 214 | 336 | 40 | 36 | 12.32 | 13.86 | 5.27 | 5.93 | 2 % |
| 3 | 2 | 580 | 269 | 448 | 88 | 78 | 6.62 | 7.43 | 3.07 | 3.45 | 1 % |
| 4 | 1 | 826 | 550 | 589 | 80 | 71 | 10.32 | 11.63 | 6.87 | 7.74 | 3 % |
| 4 | 2 | 960 | 698 | 869 | 164 | 146 | 5.83 | 6.55 | 4.25 | 4.77 | 2 % |
| 5 | 1 | 1250 | 1084 | 1010 | 182 | 146 | 6.85 | 8.59 | 5.94 | 7.45 | 2 % |
| 5 | 2 | 1441 | 1274 | 1440 | 364 | 291 | 3.96 | 4.95 | 3.50 | 4.38 | 2 % |
| 6 | 1 | 1807 | 2032 | 1728 | 260 | 204 | 6.96 | 8.84 | 7.83 | 9.93 | 2 % |
| 6 | 2 | 2092 | 2413 | 2549 | 507 | 397 | 4.13 | 5.27 | 4.76 | 6.08 | 2 % |
| 7 | 1 | 2532 | 3832 | 2782 | 468 | 361 | 5.41 | 7.01 | 8.19 | 10.62 | 26 % |
| 7 | 2 | 2932 | 4538 | 3824 | 862 | 648 | 3.40 | 4.52 | 5.26 | 7.00 | 18 % |
| 8 | 1 | 3448 | 7966 | 4695 | 819 | 647 | 4.21 | 5.33 | 9.73 | 12.31 | 4 % |
| 8 | 2 | 3954 | 9705 | 6518 | 1514 | 1185 | 2.61 | 3.34 | 6.41 | 8.19 | 12 % |
| 9 | 1 | 4562 | 12774 | 7536 | 1428 | 1152 | 3.20 | 3.96 | 8.95 | 11.09 | 3 % |
| 9 | 2 | 5215 | 16119 | 9582 | 2562 | 2040 | 2.04 | 2.56 | 6.29 | 7.90 | 6 % |
| 10 | 1 | 5920 | 18977 | 11894 | 1968 | 1597 | 3.01 | 3.71 | 9.64 | 11.88 | 2 % |
| 10 | 2 | 6767 | 29269 | 15380 | 3452 | 2735 | 1.96 | 2.47 | 8.48 | 10.70 | 2 % |
| 12 | 1 | 9360 | 43216 | 25386 | 4139 | 3142 | 2.26 | 2.98 | 10.44 | 13.75 | 3 % |
| 12 | 2 | 10922 | 57998 | 31752 | 6896 | 5184 | 1.58 | 2.11 | 8.41 | 11.19 | 5 % |
| 15 | 1 | 16708 | 85142 | 58951 | 12380 | 9278 | 1.35 | 1.80 | 6.88 | 9.18 | 2 % |
| 15 | 2 | 19042 | 152125 | 65579 | 20526 | 15243 | 0.93 | 1.25 | 7.41 | 9.98 | 2 % |
| 18 | 1 | 27286 | 169216 | 134746 | 29854 | 23140 | 0.91 | 1.18 | 5.67 | 7.31 | 2 % |
| 18 | 2 | 31238 | 230468 | 154282 | 53524 | 41024 | 0.58 | 0.76 | 4.31 | 5.62 | 3 % |
| 20 | 1 | 36222 | 252624 | 218654 | 65094 | 47554 | 0.56 | 0.76 | 3.88 | 5.31 | 2 % |
| 20 | 2 | 41848 | 304472 | 245974 | 110554 | 79841 | 0.38 | 0.52 | 2.75 | 3.81 | 3 % |

Batched against Dense, two planes, at most 6.1e-15; fused at most 5.8e-15 (one plane: 0 separate).

- The error columns compare Batched's block with Dense's pairs in this example's measure
  (relative L2 of a local of uniform coefficients, which weighs the high degrees): 0 with
  one plane in the separate mode, at most 6.3e-15 with two; the fused mode at most 5.9e-15.
- Rotation, Dense and Classes on the M6 are CH3.2b's within 6 % (for example 969 against
  978 ns for Rotation and 986 against 1,045 ns for Dense at p = 7 with two planes).
- The spreads above 8 % are in Dense's and Classes' columns (locust at p ≥ 15, Dense 27 % at
  p = 15 and Classes 12–19 % at p = 18–20; Kathleen's Dense 12–26 % at p = 7–8); Rotation's
  and fused Batched's columns agree within 7 % between the launches everywhere.
- Batched's tables are Dense's (21, 331 and 983 MB a level at p = 7, 15 and 20) and build
  in Dense's time plus the re-layout (0.02, 0.4 and 1.4 s on the M6).

## 2. End to end

`helmholtz_accuracy` (one rank, one thread, the default P2P kernel of each machine, the
default degree rule, the default budget of 1 GiB for the tables) on CH3.2a's cases, each
with `--m2l rotation,dense,classes,batched,batched-fused` and again in the reverse order:
the evaluation's wall time, the mean of the two launches. Dense, Classes and Batched ran on
the levels named (Classes on more where its smaller tables fit; CH3.2b's report), Rotation
on the others. "fused" is Batched in the fused mode.

### M6

| Case | ε | p by level | Batched levels (MB) | evaluate s, Rotation | Dense | Classes | Batched | fused | Batched / Rotation | fused / Rotation | fused / Dense |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 cube (N = 100000), kD = 10 | 1e-3 | 0×2, 7×3 | 2,3,4 (62 MB) | 0.94 | 0.95 | 1.16 | 0.96 | 0.68 | 1.02 | 0.72 | 0.71 |
| W4 cube (N = 100000), kD = 10 | 1e-6 | 0×2, 15×3 | 2,3,4 (994 MB) | 4.36 | 11.16 | 11.96 | 11.04 | 6.51 | 2.54 | 1.49 | 0.58 |
| W1, kD = 30 | 1e-3 | 0×2, 12, 8 | 2,3 (178 MB) | 0.08 | 0.11 | 0.12 | 0.10 | 0.07 | 1.33 | 0.93 | 0.64 |
| W1, kD = 30 | 1e-6 | 0×2, 16, 15 | 2,3 (754 MB) | 0.25 | 0.64 | 0.68 | 0.69 | 0.44 | 2.74 | 1.76 | 0.69 |
| W1, kD = 100 | 1e-3 | 0×2, 30, 18, 11, 7 | 3,4,5 (784 MB) | 2.49 | 4.79 | 5.99 | 4.79 | 3.15 | 1.93 | 1.27 | 0.66 |
| W1, kD = 100 | 1e-6 | 0×2, 35, 22, 15×2 | 4,5 (663 MB) | 6.31 | 12.88 | 18.39 | 13.32 | 9.02 | 2.11 | 1.43 | 0.70 |
| W3 (N = 108849), kD = 30 | 1e-3 | 0×2, 12, 8, 7 | 2,3,4 (198 MB) | 1.00 | 1.05 | 1.26 | 1.04 | 0.74 | 1.04 | 0.74 | 0.70 |

Errors the same for every strategy: True; each strategy's checksum the same in both orders: True; the two orders differ by at most 14 %.

### locust

| Case | ε | p by level | Batched levels (MB) | evaluate s, Rotation | Dense | Classes | Batched | fused | Batched / Rotation | fused / Rotation | fused / Dense |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, kD = 30 | 1e-3 | 0×2, 12, 8 | 2,3 (178 MB) | 0.15 | 0.24 | 0.24 | 0.20 | 0.16 | 1.38 | 1.10 | 0.67 |
| W1, kD = 30 | 1e-6 | 0×2, 16, 15 | 2,3 (754 MB) | 0.46 | 2.56 | 1.33 | 1.52 | 1.29 | 3.26 | 2.76 | 0.50 |
| W1, kD = 100 | 1e-3 | 0×2, 30, 18, 11, 7 | 3,4,5 (784 MB) | 4.87 | 16.18 | 13.17 | 10.43 | 8.30 | 2.14 | 1.70 | 0.51 |
| W1, kD = 100 | 1e-6 | 0×2, 35, 22, 15×2 | 4,5 (663 MB) | 12.36 | 53.23 | 40.63 | 26.50 | 22.38 | 2.14 | 1.81 | 0.42 |
| W4 cube (N = 100000), kD = 10 | 1e-3 | 0×2, 7×3 | 2,3,4 (62 MB) | 2.02 | 2.33 | 2.58 | 1.94 | 1.57 | 0.96 | 0.78 | 0.67 |
| W4 cube (N = 100000), kD = 10 | 1e-6 | 0×2, 15×3 | 2,3,4 (994 MB) | 8.21 | 51.99 | 24.18 | 21.02 | 16.59 | 2.56 | 2.02 | 0.32 |
| W3 (N = 108849), kD = 30 | 1e-3 | 0×2, 12, 8, 7 | 2,3,4 (198 MB) | 2.21 | 2.55 | 2.86 | 2.21 | 1.81 | 1.00 | 0.82 | 0.71 |

Errors the same for every strategy: True; each strategy's checksum the same in both orders: True; the two orders differ by at most 7 %.

### Kathleen

| Case | ε | p by level | Batched levels (MB) | evaluate s, Rotation | Dense | Classes | Batched | fused | Batched / Rotation | fused / Rotation | fused / Dense |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, kD = 30 | 1e-3 | 0×2, 12, 8 | 2,3 (178 MB) | 0.21 | 0.43 | 0.38 | 0.14 | 0.12 | 0.64 | 0.57 | 0.28 |
| W1, kD = 30 | 1e-6 | 0×2, 16, 15 | 2,3 (754 MB) | 0.81 | 5.15 | 2.71 | 1.14 | 0.91 | 1.42 | 1.13 | 0.18 |
| W1, kD = 100 | 1e-3 | 0×2, 30, 18, 11, 7 | 3,4,5 (784 MB) | 8.21 | 29.30 | 24.41 | 7.80 | 6.60 | 0.95 | 0.80 | 0.23 |
| W1, kD = 100 | 1e-6 | 0×2, 35, 22, 15×2 | 4,5 (663 MB) | 21.45 | 103.07 | 78.38 | 24.45 | 20.93 | 1.14 | 0.98 | 0.20 |
| W4 cube (N = 100000), kD = 10 | 1e-3 | 0×2, 7×3 | 2,3,4 (62 MB) | 2.52 | 3.38 | 3.46 | 1.09 | 0.99 | 0.43 | 0.39 | 0.29 |
| W4 cube (N = 100000), kD = 10 | 1e-6 | 0×2, 15×3 | 2,3,4 (994 MB) | 13.91 | 95.16 | 45.14 | 14.75 | 11.38 | 1.06 | 0.82 | 0.12 |
| W3 (N = 108849), kD = 30 | 1e-3 | 0×2, 12, 8, 7 | 2,3,4 (198 MB) | 2.69 | 3.67 | 3.76 | 1.21 | 1.12 | 0.45 | 0.42 | 0.31 |

Errors the same for every strategy: True; each strategy's checksum the same in both orders: True; the two orders differ by at most 8 %.

- The two orders differ most on the shortest runs (W1 at kD = 30, 0.07–0.25 s).
- W1's levels are sparse (points on a sphere): a block of 64 targets meets fewer pairs per
  offset than the uniform cube's, so each table is read for fewer columns and Batched gains
  less than its per-pair figures suggest (on Kathleen at kD = 30 and 10⁻⁶, p = 15–16: 1.25×
  Rotation's speed per pair, 0.88× end to end).
- At 10⁻⁶ and on W1 at kD = 100, the levels with tables carry p = 15–35; the default budget
  keeps Batched off the coarsest of them only by bytes. A rule by p would leave them to
  Rotation (§3 of the summary).

## 3. How it was run

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    target/release/examples/helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20      # twice
target/release/examples/helmholtz_accuracy --workload w1 --kd 30,100 --eps 1e-3,1e-6 \
    --threads 1 --m2l rotation,dense,classes,batched,batched-fused
target/release/examples/helmholtz_accuracy --workload w4-cube:100000 --kd 10 --eps 1e-3,1e-6 \
    --threads 1 --m2l rotation,dense,classes,batched,batched-fused
target/release/examples/helmholtz_accuracy --workload w3 --kd 30 --eps 1e-3 --threads 1 \
    --m2l rotation,dense,classes,batched,batched-fused
```

and the three `helmholtz_accuracy` launches again with `--m2l
batched-fused,batched,classes,dense,rotation`. Both examples were built immediately before on
each machine.

- **M6**: outside the sandbox, after the user confirmed the machine quiet, each launch behind
  CH3.2a's gate (`HWLOC_COMPONENTS=-darwin`; the other processes' CPU from `ps` at most 150 %
  before a launch, sampled every 5 s during it, a launch void and repeated above 200 %). One
  launch was void (the second `helmholtz_m2l`, 229 %) and repeated; the others saw 8–75 %
  before and 13–189 % during. macOS binds no thread.
- **locust**: `tools/gh200/sync.sh`, then the runs through `tools/gh200/remote.sh`, one after
  the other; another user's single-core Python job ran throughout (load average about 1 of
  72 cores), and the GPU was idle.
- **Kathleen**: job 618018 in `singlenode` on node-c11a-011 (`tools/kathleen/submit.sh
  --time=04:00:00 -- run.sbatch`): the build, then the runs in the batch step, bound to the
  node's 40 physical cores (one thread, unpinned within them).
