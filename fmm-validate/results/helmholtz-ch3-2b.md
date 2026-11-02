# Helmholtz CH3.2b: the Classes M2L

Helmholtz task CH3.2b ([docs/helmholtz/tasks/CH3.2b.md](../../docs/helmholtz/tasks/CH3.2b.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§5.3 (Classes) and §10 CH3.2. Measured on 2026-11-02 on the M6 (Mac mini, Apple M6, 24 GB;
[tools/m6/machine.md](../../tools/m6/machine.md)) from branch `tbetcke/h-ch3-2b` at b8b8301,
release build, rustc 1.99.0, BLAS threads 1, one thread, f64, every launch behind a quiet gate
(§3; the user confirmed the machine quiet first). The times are indicative only (the M6's
launches vary by a few per cent at one thread); nothing here is timed for a gate, and nothing
decides a default.

## Summary

- **Classes agrees with Dense, and so with Rotation, to rounding.** Every one of the 316
  offsets on levels 2–4, for five k̂ (real and complex), one and two multipole planes: at most
  3.7e-14 per unit of conditioning in the FMM-weighted measure against Dense (`nd-fmm-tables`'s
  `tests/helmholtz/classes.rs`, bound 2e-13), the size of Dense's own distance from Rotation.
  The class matrices are Dense's of the representatives bit for bit, and a representative's
  M2L into a zero local is Dense's bit for bit. Pair by pair in the operator at most 5.6e-15
  relative L2 against Rotation (Dense: 7.2e-16); in the FMM at most 2.4e-16 (φ) and 1.3e-16
  (∇φ) of Rotation's output, with the same errors against the direct sum, the same bits at
  1 and 2 threads, pair by pair and on P ranks the one-rank FMM's (`tests/helmholtz_fmm.rs`
  at 1, 2 and 4 ranks).
- **The transforms.** Oⁿ(P) of the orthonormal coefficients for the 48 elements map the
  harmonics, **Y**_n(P x̂) = Oⁿ(P) **Y**_n(x̂), to 2.2e-15 at p = 12. The 16 elements that fix
  the z-axis are exact signed permutations with §3.12's structure. The 32 others are dense
  per-degree blocks of the Gumerov–Duraiswami y-block at π/2, with exact quarter turns about
  z on either side.
- **Per pair, Classes costs Dense's chain plus the transforms** (§1). It is faster than
  Rotation only to p = 4 (1.27–1.31× at p = 3), level at p = 5, and 0.71–0.75× Rotation's
  speed at p = 7, 0.38× at p = 12 and 0.22× at p = 20. Against Dense: 0.48–0.51× its speed at
  p = 3, where the transforms are about half the pair, and 0.86–1.00× from p = 12, where they
  are 6–11 %. The offsets whose element fixes the z-axis (148 of 316) take Dense's time
  within 5 % from p = 6 (1.02–1.5× at p ≤ 5). The others (168) cost 1.1–1.7× the z-axis
  offsets, the dense transforms included: these are the transforms that move the z-axis.
- **Tables**: 16 · 2 (p + 1)⁴ values per level plus the transforms. That is 1.2 MB at p = 7,
  18 MB at p = 15 and 53 MB at p = 20 in f64, against Dense's 21, 331 and 983 MB (about 18×
  less). They build in 0.00, 0.02 and 0.07 s against 0.02, 0.40 and 1.41 s.
- **End to end, Classes is slower than Rotation everywhere measured** (§2): 1.23–1.25× its
  evaluation time on the W4 cube at 10⁻³ and W3 (p = 7–12), 1.7–3.0× on W1 and at 10⁻⁶; and
  1.07–1.42× Dense's.
  So Classes stays opt-in with Rotation the default, as Dense does.
- **What this means for CH3.2c.** Applied pair by pair, neither table form pays above
  p ≈ 5 on the M6. The per-pair chain is bound by its arithmetic, not by reading the table:
  at p = 10 the z-axis offsets read 4 MB of class tables, which stay in cache, and take
  within 4 % of Dense's time from 74 MB. So the case for tables rests on the batched M2L (CH3.2c), which
  turns the 2n² multiply–adds per pair into matrix–matrix products. Classes' small tables
  could carry that product to degrees where Dense's do not fit, if the transforms can be
  batched too.

## 1. The M2L per pair

`helmholtz_m2l` (`--ps 3,4,5,6,7,8,9,10,12,15,18,20`): one level (level 2 of three,
k̂ = 1 and 1 + 0.1i on the level), each pass the offsets in turn, each with its own multipole
and local, the median over 15 batches; Dense's and Classes' locals of one pass first compared
with Rotation's (the largest relative L2 of a local, last two columns). ns per pair, the mean
of two launches (they differ by at most 3.6 % at ≥ 100 ns); build seconds the larger and
errors the worse of the two. "Classes z" is over the 148 offsets whose element fixes the
z-axis (signed permutations), "Classes moving" over the 168 others (dense blocks). The
"transforms" column times them alone, over all 316 offsets: O(Pᵀ) on each multipole plane
and O(P) on two local planes, without `m2l`'s two scratch fills.

| p | k̂ | planes | Rotation | Dense | Classes | Classes z | Classes moving | transforms | Rotation / Classes | Dense / Classes | Dense MB | Classes MB | Dense build s | Classes build s | Dense against Rotation | Classes against Rotation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 1 | 164 | 64 | 125 | 96 | 150 | 60 | 1.31 | 0.51 | 1 | 0.1 | 0.00 | 0.00 | 4.8e-16 | 1.9e-15 |
| 3 | 1 | 2 | 200 | 76 | 156 | 114 | 191 | 79 | 1.27 | 0.49 | 1 | 0.1 | 0.00 | 0.00 | 6.5e-16 | 2.2e-15 |
| 3 | 1 + 0.1i | 2 | 200 | 75 | 155 | 114 | 192 | 78 | 1.29 | 0.48 | 1 | 0.1 | 0.00 | 0.00 | 6.1e-16 | 2.3e-15 |
| 4 | 1 | 1 | 261 | 151 | 221 | 173 | 260 | 86 | 1.18 | 0.68 | 3 | 0.2 | 0.00 | 0.00 | 4.3e-16 | 2.4e-15 |
| 4 | 1 | 2 | 318 | 202 | 281 | 226 | 330 | 113 | 1.13 | 0.72 | 3 | 0.2 | 0.00 | 0.00 | 5.1e-16 | 2.5e-15 |
| 4 | 1 + 0.1i | 2 | 319 | 202 | 281 | 224 | 330 | 113 | 1.14 | 0.72 | 3 | 0.2 | 0.00 | 0.00 | 5.1e-16 | 2.7e-15 |
| 5 | 1 | 1 | 391 | 298 | 386 | 304 | 438 | 116 | 1.01 | 0.77 | 7 | 0.4 | 0.01 | 0.00 | 5.6e-16 | 2.8e-15 |
| 5 | 1 | 2 | 480 | 329 | 487 | 396 | 560 | 154 | 0.98 | 0.68 | 7 | 0.4 | 0.01 | 0.00 | 5.0e-16 | 3.1e-15 |
| 5 | 1 + 0.1i | 2 | 480 | 332 | 489 | 399 | 564 | 154 | 0.98 | 0.68 | 7 | 0.4 | 0.00 | 0.00 | 6.0e-16 | 3.4e-15 |
| 6 | 1 | 1 | 566 | 548 | 687 | 542 | 760 | 162 | 0.82 | 0.80 | 12 | 0.7 | 0.01 | 0.00 | 5.8e-16 | 3.8e-15 |
| 6 | 1 | 2 | 702 | 736 | 905 | 742 | 1006 | 214 | 0.78 | 0.81 | 12 | 0.7 | 0.01 | 0.00 | 5.0e-16 | 3.9e-15 |
| 6 | 1 + 0.1i | 2 | 701 | 733 | 904 | 743 | 1002 | 214 | 0.78 | 0.81 | 12 | 0.7 | 0.01 | 0.00 | 5.1e-16 | 4.1e-15 |
| 7 | 1 | 1 | 789 | 876 | 1109 | 877 | 1238 | 226 | 0.71 | 0.79 | 21 | 1.2 | 0.02 | 0.00 | 5.7e-16 | 5.0e-15 |
| 7 | 1 | 2 | 978 | 1045 | 1307 | 1110 | 1486 | 303 | 0.75 | 0.80 | 21 | 1.2 | 0.02 | 0.00 | 5.4e-16 | 4.7e-15 |
| 7 | 1 + 0.1i | 2 | 977 | 1035 | 1321 | 1114 | 1490 | 302 | 0.74 | 0.78 | 21 | 1.2 | 0.02 | 0.00 | 5.3e-16 | 4.6e-15 |
| 8 | 1 | 1 | 1030 | 1416 | 1708 | 1436 | 1889 | 277 | 0.60 | 0.83 | 33 | 1.9 | 0.03 | 0.00 | 5.8e-16 | 6.2e-15 |
| 8 | 1 | 2 | 1288 | 1853 | 2234 | 1906 | 2461 | 371 | 0.58 | 0.83 | 33 | 1.9 | 0.03 | 0.00 | 5.6e-16 | 6.0e-15 |
| 8 | 1 + 0.1i | 2 | 1293 | 1853 | 2234 | 1900 | 2466 | 370 | 0.58 | 0.83 | 33 | 1.9 | 0.03 | 0.00 | 5.6e-16 | 5.9e-15 |
| 9 | 1 | 1 | 1331 | 2189 | 2526 | 2266 | 2740 | 355 | 0.53 | 0.87 | 51 | 2.9 | 0.05 | 0.00 | 9.2e-16 | 6.0e-15 |
| 9 | 1 | 2 | 1674 | 2470 | 2873 | 2580 | 3188 | 478 | 0.58 | 0.86 | 51 | 2.9 | 0.05 | 0.00 | 5.3e-16 | 6.1e-15 |
| 9 | 1 + 0.1i | 2 | 1669 | 2479 | 2892 | 2584 | 3184 | 475 | 0.58 | 0.86 | 51 | 2.9 | 0.04 | 0.00 | 5.8e-16 | 6.0e-15 |
| 10 | 1 | 1 | 1699 | 3250 | 3614 | 3271 | 3898 | 452 | 0.47 | 0.90 | 74 | 4.2 | 0.07 | 0.00 | 6.4e-16 | 7.1e-15 |
| 10 | 1 | 2 | 2136 | 4098 | 4718 | 4260 | 5114 | 606 | 0.45 | 0.87 | 74 | 4.2 | 0.07 | 0.00 | 7.1e-16 | 7.2e-15 |
| 10 | 1 + 0.1i | 2 | 2140 | 4096 | 4715 | 4288 | 5104 | 610 | 0.45 | 0.87 | 74 | 4.2 | 0.07 | 0.00 | 6.3e-16 | 7.4e-15 |
| 12 | 1 | 1 | 2673 | 6465 | 6826 | 6267 | 7298 | 718 | 0.39 | 0.95 | 144 | 8.1 | 0.15 | 0.01 | 1.1e-15 | 8.7e-15 |
| 12 | 1 | 2 | 3348 | 8016 | 8875 | 8174 | 9500 | 964 | 0.38 | 0.90 | 144 | 8.1 | 0.15 | 0.01 | 5.7e-16 | 8.8e-15 |
| 12 | 1 + 0.1i | 2 | 3342 | 7984 | 8875 | 8170 | 9514 | 960 | 0.38 | 0.90 | 144 | 8.1 | 0.15 | 0.01 | 6.5e-16 | 8.7e-15 |
| 15 | 1 | 1 | 4731 | 14927 | 14922 | 13898 | 15792 | 1290 | 0.32 | 1.00 | 331 | 18.2 | 0.40 | 0.02 | 8.5e-16 | 1.1e-14 |
| 15 | 1 | 2 | 5958 | 16938 | 18152 | 16824 | 19272 | 1728 | 0.33 | 0.93 | 331 | 18.2 | 0.40 | 0.02 | 6.2e-16 | 1.0e-14 |
| 15 | 1 + 0.1i | 2 | 5920 | 16940 | 17996 | 16648 | 19142 | 1719 | 0.33 | 0.94 | 331 | 18.2 | 0.39 | 0.02 | 6.8e-16 | 1.0e-14 |
| 18 | 1 | 1 | 7452 | 30219 | 30128 | 28176 | 31712 | 2090 | 0.25 | 1.00 | 659 | 35.8 | 0.88 | 0.05 | 7.7e-16 | 1.3e-14 |
| 18 | 1 | 2 | 9440 | 36958 | 42995 | 40868 | 45504 | 2834 | 0.22 | 0.86 | 659 | 35.8 | 0.88 | 0.05 | 6.6e-16 | 1.2e-14 |
| 18 | 1 + 0.1i | 2 | 9364 | 36896 | 43150 | 40644 | 45426 | 2836 | 0.22 | 0.86 | 659 | 35.8 | 0.85 | 0.04 | 7.2e-16 | 1.2e-14 |
| 20 | 1 | 1 | 9824 | 45192 | 44968 | 42376 | 46920 | 2819 | 0.22 | 1.00 | 983 | 53.0 | 1.41 | 0.07 | 7.4e-16 | 1.5e-14 |
| 20 | 1 | 2 | 12466 | 55306 | 57198 | 54434 | 60707 | 3888 | 0.22 | 0.97 | 983 | 53.0 | 1.41 | 0.07 | 6.3e-16 | 1.4e-14 |
| 20 | 1 + 0.1i | 2 | 12404 | 55022 | 57703 | 54453 | 60534 | 3885 | 0.21 | 0.95 | 983 | 53.0 | 1.37 | 0.07 | 7.0e-16 | 1.4e-14 |

- **The transforms.** Their share of a Classes pair falls from about half at p = 3 to
  20–23 % at p = 7 and 6–7 % at p = 20. Measured over all offsets, about half of which apply
  a permutation, they take 0.08 µs at p = 3 and 3.9 µs at p = 20 with two planes. The gap
  between "Classes moving" and "Classes z" (0.08 µs at p = 3, 0.38 µs at p = 7, 6.3 µs at
  p = 20) is the extra cost of the dense blocks over the permutations. A z-moving block has
  only (n + 1)² + n² non-zero entries of (2n + 1)², because the y-block keeps the cos and sin
  slots apart and the z-rotations only pick entries. The dense application does not exploit
  this, so storing the compact y-block with its two signed permutations would roughly halve
  the dense transforms; it is not done here.
- **Arithmetic, not reads.** From p = 6 to 10, "Classes z" is within 5 % of Dense though it
  reads 16 class tables instead of 316 (4.2 MB, held in cache, against 74 MB at p = 10); at
  p ≤ 5 its permutations and two scratch fills cost it 1.02–1.5× Dense. From p = 12, where
  neither form's tables stay in cache, it is between 7 % faster and 11 % slower than Dense.
  So per pair the chain's 2n² multiply–adds bound both forms on the M6, and reading the table
  from memory adds at most a few per cent. CH3.2a's report described Dense per pair as bound
  by memory; this measurement corrects that reading.
- **Errors.** Classes' locals sit 1.9e-15 to 1.5e-14 from Rotation's in this example's
  measure (relative L2 of a local of uniform coefficients, which weighs the high degrees),
  growing with p. They come from the transforms' sums of 2n + 1 terms. In the FMM-weighted
  measure of the tests they are Dense's size (Summary).

## 2. End to end

`helmholtz_accuracy` (one rank, one thread, the P2P kernel NEON, the default degree rule,
the default budget of 1 GiB for the dense or class tables): each case with
`--m2l rotation,dense,classes`, then again with `--m2l classes,dense,rotation`; the
evaluation's wall time, the mean of the two launches (they differ by at most 0.11 s, 1 %).
Dense and Classes ran on the levels named, chosen from the finest up within the budget; the
other levels ran Rotation.

| Case | ε | p by level (2, 3, …) | Dense levels (MB) | Classes levels (MB) | build s, R / D / C | evaluate s, Rotation | Dense | Classes | Dense / Rotation | Classes / Rotation | Classes / Dense |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, kD = 30 (7,162 points) | 10⁻³ | 12, 8 | 2, 3 (178) | 2, 3 (10) | 0.00 / 0.18 / 0.01 | 0.07 | 0.11 | 0.12 | 1.6 | 1.7 | 1.09 |
| W1, kD = 30 | 10⁻⁶ | 16, 15 | 2, 3 (754) | 2, 3 (40) | 0.00 / 0.92 / 0.05 | 0.24 | 0.64 | 0.69 | 2.7 | 2.9 | 1.07 |
| W1, kD = 100 (79,578 points) | 10⁻³ | 30, 18, 11, 7 | 3, 4, 5 (784) | 2, 3, 4, 5 (286) | 0.06 / 1.05 / 0.57 | 2.45 | 4.83 | 6.01 | 2.0 | 2.5 | 1.25 |
| W1, kD = 100 | 10⁻⁶ | 35, 22, 15, 15 | 4, 5 (663) | 2, 3, 4, 5 (551) | 0.06 / 1.22 / 1.15 | 6.21 | 12.99 | 18.51 | 2.1 | 3.0 | 1.42 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻³ | 7, 7, 7 | 2, 3, 4 (62) | 2, 3, 4 (3) | 0.04 / 0.09 / 0.04 | 0.93 | 0.95 | 1.14 | 1.02 | 1.23 | 1.20 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻⁶ | 15, 15, 15 | 2, 3, 4 (994) | 2, 3, 4 (52) | 0.04 / 1.23 / 0.10 | 4.29 | 11.46 | 12.27 | 2.7 | 2.9 | 1.07 |
| W3, kD = 30 (108,849 points) | 10⁻³ | 12, 8, 7 | 2, 3, 4 (198) | 2, 3, 4 (11) | 0.04 / 0.24 / 0.05 | 1.02 | 1.08 | 1.27 | 1.06 | 1.25 | 1.17 |

- The errors against the sampled direct sum are Rotation's to the printed three digits in
  every case and for every strategy (for instance 4.15e-5 and 1.36e-4 on W1 at kD = 100,
  ε = 10⁻³; 6.08e-8 and 2.69e-7 on the cube at 10⁻⁶). Each strategy's checksum is the same in
  both launches; the strategies' checksums differ, as they agree to rounding only.
- Rotation's and Dense's times are CH3.2a's within 3 % (for example 6.18 against 6.14 s and
  12.99 against 12.49 s on W1 at kD = 100, 10⁻⁶).
- On W1 at kD = 100, Classes runs on more levels than Dense, because its tables fit the
  budget where Dense's do not: level 2 at p = 30 and 35, and level 3 at 10⁻⁶. Those are the
  levels where a table form is slowest per pair (§1), which accounts for the larger Classes /
  Dense ratios there (1.25 and 1.42). Where both run on the same levels, Classes costs
  1.07–1.20× Dense's evaluation: most at p = 7 (the cube at 10⁻³), where the transforms are a
  quarter of a pair, least at p = 15–16.

## 3. How it was run

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    target/release/examples/helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20
HWLOC_COMPONENTS=-darwin OPENBLAS_NUM_THREADS=1 … target/release/examples/helmholtz_accuracy \
    --workload w1 --kd 30,100 --eps 1e-3,1e-6 --threads 1 --m2l rotation,dense,classes
```

`helmholtz_m2l` ran twice. `helmholtz_accuracy` also ran with `--m2l classes,dense,rotation`,
and for `--workload w4-cube:100000 --kd 10 --eps 1e-3,1e-6` and `--workload w3 --kd 30
--eps 1e-3`. Both examples were built at b8b8301 immediately before. They ran outside the
sandbox, one launch at a time, after the user confirmed that the machine was quiet, each
behind the gate of CH3.2a: the other processes' CPU (`ps`) at most 150 % before each launch,
sampled every 5 s during it, and a launch void and repeated if they rose above 200 %. No
launch was void. The other processes used 8–87 % before a launch and 32–81 % during one.
macOS binds no thread, so a launch may run on a Super or a Performance core.
