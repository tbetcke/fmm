# Helmholtz CH3.6: the gate against FMM3D

Helmholtz task CH3.6 ([docs/helmholtz/tasks/CH3.6.md](../../docs/helmholtz/tasks/CH3.6.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH3.6, §9.1 and §9.2. Measured on 2026-11-02 on Kathleen from branch `tbetcke/h-ch3-6`
(commit a6b3aa7). Each job had one exclusive node (2 × 20 Cascade Lake cores, Xeon Gold
6248, `singlenode`): job 617650 on node-c11a-108 ran W1 and the one-thread run, and job 617651
on node-c11a-123 ran W3 and W4. Both used `tools/kathleen/jobs/fmm3d.sbatch helmholtz`, that is
`tools/fmm3d/run.sh helmholtz`, with a release build, rustc 1.99.0 and BLAS threads 1. FMM3D
v2.1.0 (commit `70455e0`, T9's pin) was built with gcc 12.3.0 and `-O3 -march=native`, in two
variants: `default` (FAST_KER=OFF) and `fastker` (FAST_KER=ON). The driver was relinked for its
full output in job 617649.

## Summary

- **Our FMM meets ε in every case.** The gate covers W1 at kD = 10, 30, 100 and 500, W3 at
  kD = 10, 30 and 100, and W4's cube, sphere, Plummer sphere and clusters (N = 10⁵) at
  kD = 1, 10 and 100. Each case ran at ε = 10⁻³, 10⁻⁶ and 10⁻⁸, giving 57 case × ε points,
  60 with the one-thread run. At every point our relative ℓ2 errors of φ and ∇φ are within
  ε at the 1,000 sampled targets. The largest is 0.63 ε (∇φ, W1 at kD = 500, 10⁻⁸). That is CH3.3's value on the M6, and W1 at kD = 100
  matches the M6 smoke run to the digits shown. The errors over all N targets (the
  crosscheck against FMM3D at 10⁻¹²) are within ε as well, the largest 0.43 ε. That meets
  the acceptance: "both codes against the same sampled direct sum on W1, W3 and W4 for
  kD ≤ 100, and on W1 at kD = 500 with time reported; the full-target crosscheck".
- **FMM3D misses its eps at 10⁻³ at kD = 100, and over-delivers at tight eps.** It misses
  on W1 (1.23 ε), W3 (2.03 ε) and the W4 cube (1.99 ε φ, 2.12 ε ∇φ), in both variants. At
  10⁻⁸ its errors are at most 0.04 ε (W1 at kD = 500: 0.006 ε), where ours are 0.001 to
  0.63 ε. FMM3D is reported, not gated (§9.2 uses it as a crosscheck).
- **Times on one node.** Ours ran on 4 ranks × 10 threads and FMM3D on 40 OpenMP threads.
  For ours the time is build plus evaluation on the slowest rank; for FMM3D it is the whole
  call. At the same requested ε (the cases where ours takes more than 0.25 s), ours takes:
  - at 10⁻³, 0.29 to 0.65 of FMM3D default's time and 0.55 to 1.31 of fastker's;
  - at 10⁻⁶, 0.42 to 1.45 of default's and 0.83 to 2.4 of fastker's;
  - at 10⁻⁸, 0.93 to 3.4 of default's and 1.65 to 5.5 of fastker's.

  W1 at kD = 500 (N = 1,989,437) takes 32, 52 and 100 s for ours, against 49, 63 and 72 s
  (default) and 33, 44 and 49 s (fastker). Equal ε is not equal accuracy, since FMM3D is up
  to 100 times more accurate at 10⁻⁸ and misses at 10⁻³. The comparison at matched accuracy
  is out of scope here (proposal §9.2: after CH4.5 and CH7.6).
- **One thread, W1 at kD = 100.** Ours takes 7.7, 19.8 and 57.9 s, against 11.1, 14.9 and
  23.1 s (default) and 4.9, 6.6 and 8.9 s (fastker). The checksums equal the 4 × 10 run's
  bit for bit.
- **Our errors are more concentrated than FMM3D's.** In the full-target crosscheck, the
  worst box's error (the boxes of the deepest level) is this many times the all-target ℓ2
  error:

  | ε | ours | FMM3D |
  | --- | --- | --- |
  | 10⁻³ | 8 to 25× | 5 to 14× |
  | 10⁻⁶ | 15 to 51× | 7 to 26× |
  | 10⁻⁸ | 13 to 146× | 6 to 37× |

  Pointwise, relative to the reference's RMS over all targets, our worst box reaches 57 ε
  (∇φ, W1 at kD = 500, 10⁻⁸). FMM3D's worst box at 10⁻⁸ stays below ε, because its ℓ2
  error is far below ε. On W1 at 10⁻⁶ and 10⁻⁸, most of our worst boxes lie where the sphere
  touches its bounding cube, at kD = 30, 100 and 500 alike. On the M6, at kD = 100, a
  margin around the domain moved them and left the ratio unchanged (the plan's
  experiment), so this is not a boundary effect. §9.1's criterion is the ℓ2 norm, which this meets; whether a pointwise criterion
  should follow is question 1 of the PR, a follow-up.

## 1. Method

- **Problems.** `helmholtz_fmm3d problem` writes each case of CH3.1c's harness
  (`nd_fmm_validate::helmholtz_accuracy`, D = 1, real k) to FMM3D's problem file
  (`NDFMMHP1`): the points, one complex charge vector (§9.1's, seeded), and the indices of
  the 1,000 sampled targets. `compare` evaluates the f64 direct sum at those targets,
  `nd_fmm_ref::helmholtz::p2p::direct_sum` × 1/(4π), which CH1.4 showed equal to FMM3D's
  `h3ddirectcg` to round-off. §9.1 asks for eight charge vectors, as in T9. This gate uses
  one, as CH3.1c and CH3.3 did (`Case::vectors`), and the PR asks about it. W1 at kD = 10 has 796 points, all sampled.
- **FMM3D.** `fmm3d_driver helmholtz` calls `hfmm3d_s_c_g` once per eps, with no warm-up,
  on 40 threads. It writes φ and ∇φ at the samples (`NDFMMHO1`) and, with FULL = 1, at every
  point (`NDFMMHF1`). Both variants ran at ε, and `default` also at the reference eps
  10⁻¹².
- **Ours.** `helmholtz_fmm3d compare` reads the problem file and FMM3D's outputs; it
  refuses a problem file that differs byte for byte from its own case, and a full output
  that differs at the samples from the sampled one. It builds and evaluates our FMM once per
  ε: the default calibrated degree rule (CH3.3), CH3.4's vectorised P2P (`Auto`, AVX-512
  here), on 4 ranks × 10 threads bound by socket (`--map-by ppr:2:socket:PE=10`).
- **Errors.** §9.1's relative ℓ2 of φ and ∇φ over the samples against the direct sum, for
  both codes.
- **The full-target crosscheck** (§9.2). Each output (ours, and each FMM3D variant at ε) is
  compared at every point against FMM3D at 10⁻¹², giving two measures:
  - the relative ℓ2 over all N points;
  - the largest error per box of the FMM's deepest level, the error at a target over the
    reference's RMS over all targets (|Δ∇φ| over the RMS of |∇φ|), so that an error confined
    to a few boxes shows.

  The reference's own error at the samples is at most 2.3·10⁻¹³ (φ) and 1.05·10⁻¹² (∇φ,
  the W4 sphere). That is about 10⁻⁴ of the smallest ε. The reference took 0.01 to 111 s
  (W1 at kD = 500).
- **Not in scope** (the plan): performance at matched accuracy, and the mapping of box
  errors to offset classes (H4).

The W4 sphere's ∇φ errors are small, below 0.01 ε for both codes. Its 10⁵ points lie on
a surface, so close pairs (computed directly by both codes) presumably dominate ∇φ's
norm.

## 2. Results

Errors are relative ℓ2: "φ, ∇φ" at the 1,000 samples against the direct sum, "all" over
every point against FMM3D at 10⁻¹². "Worst box / ε" is the largest per-box error (as
above) over ε. FMM3D's errors are the default variant's. Fastker's are within a factor of 3 of them,
mostly smaller. On the smallest problems FMM3D evaluates directly, with errors of 10⁻¹⁵:
fastker on W1 at kD = 10 at every ε and, from 10⁻⁶, on W1 at kD = 30 and W3 at
kD = 10; default on W1 at kD = 10 from 10⁻⁶. p by level is our degree from level 0 down (0 on levels 0 and 1,
which have no far field). Times are single launches in seconds. Each case's table on Kathleen
(`helmholtz-<case>.md`, §3) also gives FMM3D's all-target errors and worst-box ratios for
both variants, our checksums, and our build and evaluation times apart. The FMM3D column of
the worst-box ratio table in the Summary, the variant comparison and the one-thread
checksums come from those tables.

### W1 (a sphere surface of diameter D, quasi-uniform, ten points per wavelength)

| Case | ε | p by level | ours build + run s | FMM3D default s | FMM3D fastker s | ours φ, ∇φ | FMM3D φ, ∇φ | ours all φ, ∇φ | ours worst box / ε | FMM3D worst box / ε |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| W1, kD = 10 | 10⁻³ | 0×2, 7 | 0.10 | 0.01 | 0.00 | 4.40e-5, 1.53e-4 | 1.54e-5, 3.95e-5 | 4.40e-5, 1.53e-4 | 0.46, 1.41 | 0.09, 0.20 |
|  | 10⁻⁶ | 0×2, 15 | 0.02 | 0.01 | 0.01 | 5.62e-8, 4.11e-7 | 1.21e-15, 1.30e-15 | 5.62e-8, 4.11e-7 | 0.95, 5.98 | 0.00, 0.00 |
|  | 10⁻⁸ | 0×2, 24 | 0.03 | 0.01 | 0.02 | 7.81e-11, 9.22e-10 | 1.21e-15, 1.30e-15 | 7.81e-11, 9.22e-10 | 0.10, 1.26 | 0.00, 0.00 |
| W1, kD = 30 | 10⁻³ | 0×2, 12, 8 | 0.02 | 0.06 | 0.03 | 2.66e-5, 7.15e-5 | 7.50e-5, 7.42e-5 | 2.92e-5, 7.58e-5 | 0.50, 0.99 | 0.35, 0.36 |
|  | 10⁻⁶ | 0×2, 16, 15 | 0.04 | 0.08 | 0.06 | 1.12e-7, 2.97e-7 | 1.05e-8, 1.19e-8 | 1.52e-7, 4.32e-7 | 3.99, 16.69 | 0.08, 0.16 |
|  | 10⁻⁸ | 0×2, 24×2 | 0.13 | 0.11 | 0.06 | 2.62e-10, 8.32e-10 | 5.67e-12, 7.75e-12 | 4.11e-10, 1.76e-9 | 1.65, 6.82 | 0.00, 0.01 |
| W1, kD = 100 | 10⁻³ | 0×2, 30, 18, 11, 7 | 0.42 | 0.93 | 0.42 | 4.15e-5, 1.36e-4 | 1.23e-3, 1.20e-3 | 4.27e-5, 1.34e-4 | 0.57, 1.63 | 6.73, 5.60 |
|  | 10⁻⁶ | 0×2, 35, 22, 15×2 | 0.82 | 0.92 | 0.50 | 9.61e-8, 2.94e-7 | 3.25e-7, 4.40e-7 | 9.97e-8, 3.22e-7 | 3.08, 11.41 | 4.31, 6.08 |
|  | 10⁻⁸ | 0×2, 38, 24×3 | 2.18 | 1.52 | 0.67 | 1.39e-9, 3.40e-9 | 1.33e-10, 2.09e-10 | 1.65e-9, 3.71e-9 | 9.59, 21.89 | 0.21, 0.41 |
| W1, kD = 500 | 10⁻³ | 0×2, 123, 66, 36, 21, 13, 8 | 32.17 | 49.28 | 33.46 | 3.24e-5, 8.71e-5 | 1.87e-4, 1.75e-4 | 3.31e-5, 9.31e-5 | 0.84, 1.81 | 1.72, 1.55 |
|  | 10⁻⁶ | 0×2, 131, 72, 41, 25, 16, 15 | 52.18 | 62.86 | 43.94 | 1.49e-7, 3.79e-7 | 6.21e-8, 7.19e-8 | 1.22e-7, 3.42e-7 | 5.26, 14.73 | 1.47, 1.45 |
|  | 10⁻⁸ | 0×2, 135, 76, 44, 27, 24×2 | 100.49 | 71.72 | 49.37 | 2.72e-9, 6.33e-9 | 5.19e-11, 6.33e-11 | 2.08e-9, 4.23e-9 | 29.16, 56.61 | 0.17, 0.21 |

### W3 (uniform random points in the cube, ten points per wavelength: N = ⌈1000 (kD/2π)³⌉)

| Case | ε | p by level | ours build + run s | FMM3D default s | FMM3D fastker s | ours φ, ∇φ | FMM3D φ, ∇φ | ours all φ, ∇φ | ours worst box / ε | FMM3D worst box / ε |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| W3 (N = 4,032), kD = 10 | 10⁻³ | 0×2, 7×2 | 0.03 | 0.03 | 0.02 | 4.05e-5, 8.10e-5 | 1.73e-5, 2.27e-5 | 4.01e-5, 7.63e-5 | 0.38, 0.58 | 0.11, 0.13 |
|  | 10⁻⁶ | 0×2, 15×2 | 0.05 | 0.04 | 0.02 | 4.49e-8, 1.70e-7 | 9.66e-9, 2.09e-8 | 5.07e-8, 1.97e-7 | 0.98, 4.38 | 0.07, 0.16 |
|  | 10⁻⁸ | 0×2, 24×2 | 0.12 | 0.07 | 0.02 | 9.83e-11, 8.92e-10 | 7.52e-12, 1.85e-11 | 2.71e-10, 1.76e-9 | 1.05, 7.20 | 0.01, 0.02 |
| W3 (N = 108,849), kD = 30 | 10⁻³ | 0×2, 12, 8, 7 | 0.15 | 0.35 | 0.22 | 4.60e-5, 1.31e-4 | 1.01e-4, 8.82e-5 | 4.25e-5, 4.40e-5 | 0.51, 0.51 | 0.87, 0.22 |
|  | 10⁻⁶ | 0×2, 16, 15×2 | 0.50 | 0.94 | 0.30 | 1.46e-7, 4.17e-7 | 1.58e-8, 4.34e-8 | 1.32e-7, 1.44e-7 | 4.11, 4.13 | 0.16, 0.15 |
|  | 10⁻⁸ | 0×2, 24×3 | 1.66 | 1.23 | 0.46 | 7.34e-10, 2.96e-9 | 5.81e-12, 1.46e-11 | 4.50e-10, 1.13e-9 | 3.25, 13.27 | 0.00, 0.01 |
| W3 (N = 4,031,442), kD = 100 | 10⁻³ | 0×2, 30, 18, 11, 7×2 | 9.55 | 22.09 | 7.27 | 4.86e-5, 1.08e-4 | 2.03e-3, 1.31e-3 | 4.90e-5, 7.04e-5 | 1.15, 1.32 | 19.47, 6.96 |
|  | 10⁻⁶ | 0×2, 35, 22, 15×3 | 38.00 | 31.32 | 22.47 | 1.01e-7, 2.50e-7 | 5.45e-7, 5.17e-7 | 1.07e-7, 1.66e-7 | 5.35, 7.19 | 12.06, 6.53 |
|  | 10⁻⁸ | 0×2, 38, 24×4 | 128.78 | 37.83 | 27.60 | 2.60e-9, 4.10e-9 | 1.89e-10, 2.16e-10 | 2.30e-9, 2.37e-9 | 18.98, 28.30 | 0.57, 0.58 |

At kD = 100 (N = 4,031,442) FMM3D's reference took 54 s.

### W4 (N = 10⁵)

| Case | ε | p by level | ours build + run s | FMM3D default s | FMM3D fastker s | ours φ, ∇φ | FMM3D φ, ∇φ | ours all φ, ∇φ | ours worst box / ε | FMM3D worst box / ε |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| W4 cube, kD = 1 | 10⁻³ | 0×2, 7×3 | 0.13 | 0.25 | 0.16 | 4.07e-5, 8.96e-5 | 1.63e-4, 2.15e-4 | 4.06e-5, 3.28e-5 | 0.51, 0.35 | 1.52, 0.65 |
|  | 10⁻⁶ | 0×2, 15×3 | 0.50 | 0.83 | 0.22 | 6.40e-8, 2.65e-7 | 5.66e-8, 1.18e-7 | 7.24e-8, 1.04e-7 | 2.68, 5.33 | 1.07, 0.83 |
|  | 10⁻⁸ | 0×2, 24×3 | 1.66 | 0.86 | 0.33 | 2.57e-10, 1.73e-9 | 2.71e-11, 4.41e-11 | 4.48e-10, 1.27e-9 | 5.31, 18.49 | 0.03, 0.05 |
| W4 cube, kD = 10 | 10⁻³ | 0×2, 7×3 | 0.13 | 0.25 | 0.16 | 5.00e-5, 9.95e-5 | 7.49e-5, 9.23e-5 | 5.03e-5, 3.72e-5 | 0.73, 0.37 | 0.59, 0.31 |
|  | 10⁻⁶ | 0×2, 15×3 | 0.50 | 0.91 | 0.23 | 6.08e-8, 2.69e-7 | 2.28e-8, 4.86e-8 | 6.99e-8, 1.07e-7 | 2.34, 5.34 | 0.20, 0.22 |
|  | 10⁻⁸ | 0×2, 24×3 | 1.66 | 0.97 | 0.35 | 2.29e-10, 1.71e-9 | 1.74e-11, 2.60e-11 | 3.99e-10, 1.27e-9 | 4.63, 18.48 | 0.01, 0.02 |
| W4 cube, kD = 100 | 10⁻³ | 0×2, 30, 18, 11 | 0.35 | 0.69 | 0.40 | 1.85e-5, 4.04e-5 | 1.99e-3, 2.12e-3 | 1.80e-5, 2.86e-5 | 0.40, 0.61 | 15.64, 8.10 |
|  | 10⁻⁶ | 0×2, 35, 22, 15 | 0.64 | 1.54 | 0.53 | 9.63e-8, 3.04e-7 | 5.24e-7, 7.90e-7 | 1.03e-7, 2.27e-7 | 4.45, 8.45 | 8.23, 9.55 |
|  | 10⁻⁸ | 0×2, 38, 24×2 | 1.69 | 1.42 | 0.72 | 2.11e-9, 4.52e-9 | 1.99e-10, 3.98e-10 | 2.37e-9, 4.13e-9 | 8.75, 18.47 | 0.32, 0.67 |
| W4 sphere, kD = 1 | 10⁻³ | 0×2, 7×4 | 0.20 | 0.37 | 0.17 | 2.30e-5, 8.09e-7 | 7.15e-5, 1.05e-6 | 2.34e-5, 4.35e-7 | 0.29, 0.01 | 0.59, 0.00 |
|  | 10⁻⁶ | 0×2, 15×4 | 0.71 | 0.49 | 0.31 | 4.24e-8, 2.36e-9 | 2.10e-8, 4.74e-10 | 4.47e-8, 1.19e-9 | 1.52, 0.05 | 0.36, 0.00 |
|  | 10⁻⁸ | 0×2, 24×4 | 2.25 | 1.07 | 0.68 | 1.31e-10, 1.20e-11 | 1.26e-11, 1.08e-12 | 1.16e-10, 9.37e-12 | 0.72, 0.14 | 0.02, 0.00 |
| W4 sphere, kD = 10 | 10⁻³ | 0×2, 7×4 | 0.20 | 0.38 | 0.15 | 2.85e-5, 8.42e-7 | 3.62e-5, 7.42e-7 | 2.84e-5, 4.49e-7 | 0.39, 0.01 | 0.20, 0.00 |
|  | 10⁻⁶ | 0×2, 15×4 | 0.72 | 0.53 | 0.30 | 4.36e-8, 2.40e-9 | 1.06e-8, 2.41e-10 | 4.69e-8, 1.21e-9 | 1.60, 0.05 | 0.09, 0.00 |
|  | 10⁻⁸ | 0×2, 24×4 | 2.24 | 1.14 | 0.41 | 1.27e-10, 1.21e-11 | 7.65e-12, 1.06e-12 | 1.15e-10, 9.39e-12 | 0.68, 0.14 | 0.01, 0.00 |
| W4 sphere, kD = 100 | 10⁻³ | 0×2, 30, 18, 11, 7 | 0.38 | 0.97 | 0.45 | 3.06e-5, 2.04e-6 | 9.08e-4, 1.83e-5 | 3.05e-5, 1.08e-6 | 0.41, 0.02 | 5.53, 0.07 |
|  | 10⁻⁶ | 0×2, 35, 22, 15×2 | 0.93 | 1.05 | 0.59 | 6.24e-8, 4.06e-9 | 2.25e-7, 6.41e-9 | 6.66e-8, 2.41e-9 | 2.69, 0.09 | 3.14, 0.05 |
|  | 10⁻⁸ | 0×2, 38, 24×3 | 2.32 | 1.79 | 0.76 | 8.86e-10, 4.86e-11 | 9.57e-11, 3.40e-12 | 1.17e-9, 2.77e-11 | 7.34, 0.17 | 0.16, 0.00 |
| W4 plummer, kD = 1 | 10⁻³ | 0×2, 7×7 | 0.28 | 0.71 | 0.40 | 6.47e-5, 9.67e-5 | 1.50e-4, 1.63e-4 | 6.40e-5, 5.97e-5 | 0.84, 1.52 | 1.45, 1.37 |
|  | 10⁻⁶ | 0×2, 15×7 | 0.97 | 1.56 | 0.98 | 1.11e-7, 3.33e-7 | 7.37e-8, 7.04e-8 | 1.10e-7, 1.72e-7 | 3.56, 8.36 | 0.54, 0.92 |
|  | 10⁻⁸ | 0×2, 24×7 | 3.00 | 2.91 | 1.50 | 5.90e-10, 2.41e-9 | 2.12e-11, 5.71e-11 | 5.41e-10, 1.26e-9 | 2.37, 10.33 | 0.04, 0.10 |
| W4 plummer, kD = 10 | 10⁻³ | 0×2, 7×7 | 0.28 | 0.73 | 0.41 | 5.01e-5, 9.69e-5 | 1.56e-4, 2.17e-4 | 4.88e-5, 5.98e-5 | 0.63, 1.52 | 2.04, 1.51 |
|  | 10⁻⁶ | 0×2, 15×7 | 0.97 | 1.61 | 0.99 | 8.31e-8, 3.33e-7 | 5.55e-8, 1.36e-7 | 8.23e-8, 1.72e-7 | 2.64, 8.36 | 0.89, 1.41 |
|  | 10⁻⁸ | 0×2, 24×7 | 2.89 | 2.96 | 1.52 | 4.34e-10, 2.41e-9 | 3.98e-11, 7.84e-11 | 3.99e-10, 1.26e-9 | 1.73, 10.33 | 0.04, 0.15 |
| W4 plummer, kD = 100 | 10⁻³ | 0×2, 30, 18, 11, 7×4 | 0.54 | 1.20 | 0.77 | 7.95e-5, 1.18e-4 | 2.87e-4, 1.70e-4 | 7.68e-5, 7.47e-5 | 1.22, 1.52 | 4.14, 0.82 |
|  | 10⁻⁶ | 0×2, 35, 21, 15×5 | 1.34 | 2.30 | 1.36 | 1.53e-7, 3.58e-7 | 5.87e-8, 6.39e-8 | 1.56e-7, 1.93e-7 | 3.88, 8.28 | 1.56, 0.47 |
|  | 10⁻⁸ | 0×2, 37, 24×6 | 3.95 | 3.90 | 2.02 | 1.27e-9, 2.59e-9 | 2.35e-11, 5.09e-11 | 1.50e-9, 1.42e-9 | 5.80, 10.26 | 0.02, 0.05 |
| W4 clusters, kD = 1 | 10⁻³ | 0×2, 7×7 | 0.30 | 0.91 | 0.46 | 5.07e-5, 1.01e-4 | 1.13e-4, 1.59e-4 | 4.97e-5, 5.33e-5 | 0.79, 0.77 | 1.54, 0.97 |
|  | 10⁻⁶ | 0×2, 15×7 | 1.15 | 1.88 | 1.17 | 9.45e-8, 4.03e-7 | 4.30e-8, 4.60e-8 | 8.41e-8, 1.66e-7 | 2.04, 7.00 | 0.35, 0.50 |
|  | 10⁻⁸ | 0×2, 24×7 | 3.72 | 3.57 | 1.96 | 5.37e-10, 5.00e-9 | 1.53e-11, 4.35e-11 | 4.35e-10, 1.43e-9 | 4.22, 11.54 | 0.03, 0.05 |
| W4 clusters, kD = 10 | 10⁻³ | 0×2, 7×7 | 0.30 | 0.91 | 0.45 | 5.05e-5, 1.01e-4 | 1.32e-4, 1.30e-4 | 4.95e-5, 5.33e-5 | 0.79, 0.77 | 1.41, 0.84 |
|  | 10⁻⁶ | 0×2, 15×7 | 1.15 | 1.89 | 1.18 | 9.42e-8, 4.03e-7 | 4.06e-8, 6.88e-8 | 8.37e-8, 1.66e-7 | 2.03, 7.00 | 0.63, 0.69 |
|  | 10⁻⁸ | 0×2, 24×7 | 3.52 | 3.54 | 1.95 | 5.35e-10, 5.00e-9 | 2.71e-11, 6.81e-11 | 4.33e-10, 1.43e-9 | 4.19, 11.54 | 0.05, 0.10 |
| W4 clusters, kD = 100 | 10⁻³ | 0×2, 21, 13, 8, 7×4 | 0.34 | 1.16 | 0.62 | 5.26e-5, 1.03e-4 | 6.49e-5, 7.92e-5 | 5.19e-5, 5.46e-5 | 0.77, 0.78 | 0.46, 0.41 |
|  | 10⁻⁶ | 0×2, 25, 16, 15×5 | 1.17 | 2.24 | 1.41 | 9.08e-8, 4.03e-7 | 2.70e-8, 4.19e-8 | 8.14e-8, 1.67e-7 | 1.91, 6.99 | 0.28, 0.37 |
|  | 10⁻⁸ | 0×2, 27, 24×6 | 3.70 | 3.97 | 2.24 | 5.01e-10, 4.98e-9 | 1.11e-11, 3.34e-11 | 4.09e-10, 1.43e-9 | 3.92, 11.53 | 0.02, 0.03 |

### One thread: W1 at kD = 100

Both codes on one core (ours 1 rank × 1 thread, FMM3D `OMP_NUM_THREADS=1`), no crosscheck:

| Case | ε | p by level | ours build + run s | FMM3D default s | FMM3D fastker s | ours φ, ∇φ | FMM3D φ, ∇φ |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| W1, kD = 100 | 10⁻³ | 0×2, 30, 18, 11, 7 | 7.70 | 11.09 | 4.94 | 4.15e-5, 1.36e-4 | 1.23e-3, 1.20e-3 |
|  | 10⁻⁶ | 0×2, 35, 22, 15×2 | 19.78 | 14.87 | 6.63 | 9.61e-8, 2.94e-7 | 3.25e-7, 4.40e-7 |
|  | 10⁻⁸ | 0×2, 38, 24×3 | 57.91 | 23.12 | 8.94 | 1.39e-9, 3.40e-9 | 1.33e-10, 2.09e-10 |

Our checksums at 10⁻³, 10⁻⁶ and 10⁻⁸ are `f103f07e20546c85`, `9263f17ea109543d` and
`01c022d6d6a5c984`, the same on 1 × 1 and 4 × 10.

## 3. Reproduce

```sh
tools/kathleen/sync.sh
tools/kathleen/submit.sh --time=00:20:00 -- fmm3d.sbatch driver
tools/kathleen/submit.sh '--export=ALL,OUT=/scratch/scratch/ucahtbe/fmm/logs/ch3-6-w1,H_CASES=w1@10 w1@30 w1@100 w1@500' -- fmm3d.sbatch helmholtz
tools/kathleen/submit.sh '--export=ALL,OUT=/scratch/scratch/ucahtbe/fmm/logs/ch3-6-w34,H_T1_CASES=,H_CASES=w3@10 w3@30 w3@100 w4-cube:100000@1 …' -- fmm3d.sbatch helmholtz
```

The run.sh defaults (`H_CASES`, `H_T1_CASES`) are the gate's cases in one job. Each case
writes `helmholtz-<case>.md` (the table) and `.log` (the launch, the bindings, and the
five worst boxes of each field with their indices). The raw output stayed on Kathleen in
`logs/ch3-6-w1` and `logs/ch3-6-w34`.
