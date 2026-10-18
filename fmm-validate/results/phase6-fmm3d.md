# Phase 6 T9: the FMM3D comparison (C6.8)

Phase 6, task T9 ([docs/phase6/T9-fmm3d-comparison.md](../../docs/phase6/T9-fmm3d-comparison.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §8.1, signed off on
2026-10-15 (decision 5). Measured on 2026-10-18 from branch `tbetcke/phase6_t9` (main at
a8d9037 plus this task's changes), release builds, rustc 1.99.0, every BLAS thread
variable at 1. Kathleen is usable (docs/phase6/README.md): every Kathleen job used one node
(`singlenode`, nodes exclusive), so the path "with Kathleen" of the README. The scripts are
[tools/fmm3d/](../../tools/fmm3d/README.md) (FMM3D's side) and the sweep `t9-fmm3d` of
`tools/scaling/run.sh` (ours); nothing of FMM3D is in the repository.

## Summary

1. **FMM3D v2.1.0** (the latest release, 2026-01-19; commit 70455e0, the tarball's SHA-256
   checked) built with OpenMP outside the workspace on all three machines, in its two
   variants: `make lib` (the default, its "easy-to-install version") and `make lib
   FAST_KER=ON` (its "high-performance mode": vectorised C++ direct kernels and leaves of
   300–1,000 points). Both pass FMM3D's own test suites on Kathleen. Compilers: gcc 12.3.0
   (Kathleen's module, `-O3 -march=native`, built on a compute node), the system's GCC
   11.4.1 on locust, Homebrew's GCC 16.2.0 on the M3 Max (FMM3D's
   `make.inc.macos_arm64.gnu`) (§1).
2. **The normalisation**: FMM3D v2.1.0 computes q / (4π r), the potential of CONVENTIONS
   §3.1; every release up to v2.0.0 computed q / r. The brief and optimisation.md §8.1
   ("FMM3D's Laplace kernel is 1/r ... divided by 4π") describe those. The driver checks
   the kernel on two unit charges at distance 1 before anything else (φ = ∂φ/∂x =
   0.0795775 = 1/(4π) on every machine) and scales by what it finds: 1 (§2).
3. **Matched accuracy** (§3): the same problems (the `scaling` harness's cube and Plummer
   sphere, N = 10⁵, 10⁶, 10⁷, f64), the same 8 charge vectors, the same 1,000 sampled
   targets and the same direct sum for both codes. FMM3D's error does not follow `eps`
   smoothly: it moves in steps of its expansion order and its plane-wave tables, and no
   `eps` of the grid (1-2-5 per decade) gives an error between about 4 × 10⁻⁵ and
   8 × 10⁻⁸. The closest steps to our p = 3, 6, 8, 12 are `eps` = 2 × 10⁻¹, 2 × 10⁻²,
   2 × 10⁻³ and 2 × 10⁻⁴ on every problem, variant and machine. At p = 3 the two errors
   agree within 13 %; at p = 6 FMM3D is 1.7–2.1× more accurate than ours, at p = 8
   1.8–2.2× less accurate, at p = 12 7.8–10× more accurate (its next step down). The time
   comparisons at p = 6 and 12 therefore favour us, at p = 8 FMM3D.
4. **The outputs against each other** (§3.1, Kathleen, N = 10⁵ and 10⁶, at user request):
   over every point FMM3D's output differs from ours by at most the sum of the two codes'
   errors at the sampled targets (at most 0.99 of it); at p = 12, where FMM3D is 8–10× more
   accurate, the difference over all N points equals our error at the 1,000 samples, so the
   sample represents the field. At p = 3 the two codes' errors are correlated (FMM3D's
   default build differs from ours by 9 × 10⁻⁵ – 7 × 10⁻⁴ while each is 3–4 × 10⁻³ off).
5. **Every core, one charge vector** (§4), ours (the batched M2L, Phase 6 T2, separate or
   fused multiply–add; Phase 5N's split of ranks × threads) against the faster of FMM3D's
   two builds, per evaluation, "FMM3D's time over ours":

   | machine (ours; FMM3D threads) | p = 3 | p = 6 | p = 8 | p = 12 |
   | --- | --- | --- | --- | --- |
   | Kathleen, 40 cores (4 × 10, AVX-512; 40) | 3.5–10.8 | 1.7–7.9 | 1.1–4.6 | 0.61–1.59 |
   | locust, 72 cores (8 × 9, NEON; 72) | 6.4–12.4 | 3.0–9.7 | 1.9–4.0 | 0.99–2.5 |
   | M3 Max, 12 performance cores (1 × 12, NEON; 12) | 4.3–11.6 | 1.7–7.4 | 1.3–2.5 | 0.73–1.55 |

   Ours is faster at p ≤ 8 on every problem and machine. At p = 12 FMM3D (mostly
   `FAST_KER`) wins on the Plummer sphere on Kathleen (all N) and the M3 Max (10⁶, 10⁷) and
   on Kathleen's cube at 10⁵, with errors 8–10× smaller than ours; ours wins the rest (one
   tie each on locust and the M3 Max). FMM3D's call includes its setup (tree and sort,
   19–42 % of the call at 10⁷), which it repeats every call; ours is the evaluation of a
   built FMM.
6. **Eight charge vectors in one call** (§5): FMM3D's vectorised interface amortises much
   more over the densities than our `--rhs 8` (Phase 6 T5). Ours stays faster at p = 3
   (1.6–6.9×) and mostly at p = 6 (0.71–3.5); at p = 8 it depends on the machine and the
   problem (0.48–2.1); at p = 12 FMM3D `FAST_KER` is faster everywhere but locust's cube at
   10⁷ (0.30–1.05).
7. **FMM3D's default build cannot evaluate 8 densities at N = 10⁷ in a Kathleen node's
   188 GB** at the coarse `eps` (out of memory at 2 × 10⁻¹, 2 × 10⁻² and 2 × 10⁻⁴ on the
   cube, 2 × 10⁻² on the Plummer sphere): with leaves of 40–200 points it keeps plane-wave
   expansions for every density in about 10⁶ boxes. One density at a time fits; so does
   `FAST_KER`, and locust's 480 GB. On the M3 Max (64 GB) that call was not attempted (§5).
8. **Precomputation** (§6): at N ≥ 10⁶ and p ≤ 6 our build costs 0.75–5× one FMM3D setup
   (most on the M3 Max, whose one rank builds the whole tree: 8.4–13 s at 10⁷); the M2L
   tables add 0.2–0.6 s at p = 8 and 1.8–4.9 s at p = 12, which the table cache
   (`FmmBuilder::table_cache`) removes from every build after the first. FMM3D repeats its
   setup in every call, so ours pays off after a few evaluations.
9. **Not measured, and why**: one-thread comparisons (the user stopped them on
   2026-10-18 as not necessary; our one-thread runs on Kathleen and locust had finished, §7);
   our CUDA path (the H100 on locust was held by another user's job, 80 GB and 100 %
   utilisation, from 02:16 to 09:37, §7). On the M3 Max, to save its time at the user's
   request, no FMM3D sweep of its own at 10⁷ and no errors of ours at 10⁷ (Kathleen's,
   identical to the printed digits on all three machines where measured, §3).

## 1. What T9 adds, and the builds

- `nd-fmm-bench --write-problem FILE --dist D --n N` (`fmm-bench/src/problem.rs`): the
  points and 8 charge vectors of a `scaling` workload (`nd_fmm_validate::scaling::Workload`,
  generated by index from the `scaling` example's seed), the 1,000 targets of the harness's
  errors (`Workload::sample`) and the f64 direct sum there (`fmm_accuracy::Oracle`, divided
  by 4π; `Oracle::of_targets` made public to split the targets over threads), in one
  little-endian file (0.88 GB at N = 10⁷). `--check-output OUT --problem FILE` measures an
  output file (φ and ∇φ at those targets) with the harness's metrics. Both run without MPI.
  For the cross-check (§3.1): `--write-output OUT --problem FILE --degree p` (our FMM on a
  problem file, one rank, φ and ∇φ at every point in a full output file), `--compare
  REF,OUT,…` (full outputs against the first, over every point), and `--check-output`
  takes a full output at the sampled targets.
- `tools/fmm3d/build.sh`: fetches FMM3D v2.1.0 at its tag (commit 70455e0, SHA-256 of the
  tarball checked), builds both variants with OpenMP under `FMM3D_ROOT` (on Kathleen
  `$FMM_KATHLEEN_ROOT/fmm3d`, on locust `/data/ucahtbe/fmm3d`, on the M3 Max
  `~/projects/fmm3d`) and links the driver against each; `build-info.txt` records the
  compilers and the compile lines. `tools/fmm3d/driver.f90`: reads a problem file, checks
  the normalisation, times FMM3D's setup on its own (`lndiv`, `pts_tree_mem`,
  `pts_tree_build`, `pts_tree_sort`, called as `lfmm3d` calls them) and each call of
  `lfmm3d_s_c_g` (one density) and `lfmm3d_s_c_g_vec` (8 densities) with `omp_get_wtime`,
  and writes φ and ∇φ at the sampled targets (with a 7th argument, at every point).
- `tools/fmm3d/run.sh`: `problems`, `sweep` (FMM3D's errors over `eps` = 5 × 10⁻¹ …
  10⁻¹⁰, 1-2-5 per decade, on every core), `match` (per problem, variant and p the `eps`
  whose relative L2 error of φ is closest to ours in its logarithm) and `time` (FMM3D at
  the matched `eps`: 8 calls of one density, one per vector, after one untimed call, and
  the vectorised call in a launch of its own) and `crosscheck` (§3.1).
  `tools/kathleen/jobs/fmm3d.sbatch` runs them on Kathleen.
- `tools/scaling/run.sh t9-fmm3d`: ours on the same problems, `--strategy batched` with
  `--mul-add separate` (bit for bit `Dense`) and `fused` (opt-in, T2), the P2P kernel
  `auto` (NEON on the aarch64 machines) and on Kathleen AVX-512 (`T9_P2P=avx512`, opt-in,
  T6), overlap off: at Phase 5N's split (docs/design/node-m2l.md §7: 4 × 10 on Kathleen,
  8 × 9 on locust, 1 × 12 on the M3 Max) the median of 10 evaluations after one warm-up,
  with the errors of 8 vectors; the same with `--rhs 8` (3 evaluations); and at 1 × 1 on
  Kathleen and locust (§7).

| machine | FMM3D compilers and flags | ours |
| --- | --- | --- |
| Kathleen, node of 2 × Xeon Gold 6248 (40 cores) | gcc/gfortran/g++ 12.3.0 (Spack, the `compilers/gcc/12.3.0` module), `-O3 -march=native -funroll-loops -std=legacy -fopenmp`, built on a compute node (node-c11a-106) | 4 × 10, AVX-512 P2P and M2L kernel |
| locust, 72 Neoverse-V2 cores (GH200) | system GCC 11.4.1 (RHEL 9), the same flags | 8 × 9, NEON |
| M3 Max, 12 performance cores | Homebrew GCC 16.2.0, `-O3 -arch arm64 -std=legacy -mno-outline-atomics -fopenmp` (FMM3D's `make.inc.macos_arm64.gnu`, no `-mcpu=native`) | 1 × 12, NEON |

FMM3D runs with its defaults except `OMP_NUM_THREADS` (the node's cores: 40, 72, 12); no
binding variable is set (on Kathleen the job binds the batch shell, and so FMM3D's threads,
to the 40 physical cores). FMM3D's tests on Kathleen: every suite of `make test` passes in
both variants (Laplace: 5/5 `laprouts3d`, 27/27 `lfmm3d`, 2/2 scale, 27/27 vec).

## 2. The normalisation

The driver evaluates two unit charges at (0, 0, 0) and (1, 0, 0) with `eps` = 10⁻⁶ before
anything else. On all three machines FMM3D returns φ = 7.957747154594767 × 10⁻² and
∂φ/∂x = 7.957747154594767 × 10⁻² at the first point: 1/(4π) and +1/(4π), the potential and
the gradient (∇φ = −q (x − y)/(4π r³)) of CONVENTIONS §3.1. FMM3D v2.1.0 moved to
1/(4π r): its `lfmm3d.f` says "We use 1/(4\pi)*(1/r) for the Green's function", and its
documentation's Laplace kernel is c_j / (4π ‖x − x_j‖); v1.0.4 and v2.0.0 (released the
same day as v2.1.0, 2026-01-19) still say "We use (1/r) for the Green's function, without
the 1/(4\pi) scaling". The brief's "FMM3D's 1/r divided by 4π" is those releases; the
driver scales by the factor it finds (1) and stops on any other. The tests of
`fmm-bench/src/problem.rs` check that an output in the older normalisation would show a
relative error of 4π − 1 = 11.6.

## 3. Matched accuracy

Both codes against the same direct sum at the same 1,000 targets, the root mean square
over the 8 vectors of the relative L2 errors of φ and ∇φ. Ours from the `scaling` runs at
the node's split, FMM3D's from its sweep; the timed FMM3D runs give the same errors, and
Kathleen's, locust's and the M3 Max's tables (its N ≤ 10⁶ for ours; for FMM3D but the
default build at 10⁷) agree to the printed digits, so the table is Kathleen's. "order, leaf" is FMM3D's
expansion order (`l3dterms`) and its leaf size (`lndiv`), both chosen from `eps`.

| problem | p | ours φ / ∇φ | FMM3D eps: φ / ∇φ (order, leaf) | FMM3D FAST_KER eps: φ / ∇φ (order, leaf) |
| --- | ---: | --- | --- | --- |
| cube 1e5 | 3 | 3.0e-03 / 4.5e-03 | 2e-1: 3.0e-03 / 4.5e-03 (3, 40) | 2e-1: 2.7e-03 / 2.7e-03 (3, 300) |
| cube 1e5 | 6 | 1.3e-04 / 3.1e-04 | 2e-2: 7.1e-05 / 1.9e-04 (7, 40) | 2e-2: 6.1e-05 / 1.1e-04 (7, 300) |
| cube 1e5 | 8 | 1.9e-05 / 6.0e-05 | 2e-3: 4.1e-05 / 8.9e-05 (11, 100) | 2e-3: 3.5e-05 / 5.0e-05 (11, 300) |
| cube 1e5 | 12 | 7.6e-07 / 3.1e-06 | 2e-4: 8.3e-08 / 3.2e-07 (15, 200) | 2e-4: 7.8e-08 / 2.6e-07 (15, 1000) |
| cube 1e6 | 3 | 2.9e-03 / 4.7e-03 | 2e-1: 2.9e-03 / 4.8e-03 (3, 40) | 2e-1: 2.7e-03 / 3.0e-03 (3, 300) |
| cube 1e6 | 6 | 1.3e-04 / 3.1e-04 | 2e-2: 6.9e-05 / 2.0e-04 (7, 40) | 2e-2: 6.5e-05 / 1.3e-04 (7, 300) |
| cube 1e6 | 8 | 1.9e-05 / 5.9e-05 | 2e-3: 4.1e-05 / 9.6e-05 (11, 100) | 2e-3: 3.8e-05 / 6.0e-05 (11, 300) |
| cube 1e6 | 12 | 6.9e-07 / 3.0e-06 | 2e-4: 8.2e-08 / 4.0e-07 (15, 200) | 2e-4: 7.8e-08 / 2.7e-07 (15, 1000) |
| cube 1e7 | 3 | 2.7e-03 / 2.7e-03 | 2e-1: 2.7e-03 / 3.2e-03 (3, 40) | 2e-1: 2.7e-03 / 2.4e-03 (3, 300) |
| cube 1e7 | 6 | 1.1e-04 / 1.8e-04 | 2e-2: 6.6e-05 / 1.4e-04 (7, 40) | 2e-2: 6.5e-05 / 1.0e-04 (7, 300) |
| cube 1e7 | 8 | 1.7e-05 / 3.5e-05 | 2e-3: 3.8e-05 / 5.6e-05 (11, 100) | 2e-3: 3.7e-05 / 5.0e-05 (11, 300) |
| cube 1e7 | 12 | 6.5e-07 / 1.8e-06 | 2e-4: 7.7e-08 / 2.5e-07 (15, 200) | 2e-4: 7.5e-08 / 1.5e-07 (15, 1000) |
| plummer 1e5 | 3 | 3.5e-03 / 1.1e-03 | 2e-1: 3.6e-03 / 1.3e-03 (3, 40) | 2e-1: 3.3e-03 / 8.3e-04 (3, 300) |
| plummer 1e5 | 6 | 1.7e-04 / 7.4e-05 | 2e-2: 9.3e-05 / 5.3e-05 (7, 40) | 2e-2: 9.0e-05 / 3.5e-05 (7, 300) |
| plummer 1e5 | 8 | 2.6e-05 / 1.4e-05 | 2e-3: 4.7e-05 / 2.0e-05 (11, 100) | 2e-3: 4.7e-05 / 1.7e-05 (11, 300) |
| plummer 1e5 | 12 | 9.8e-07 / 6.6e-07 | 2e-4: 1.0e-07 / 7.1e-08 (15, 200) | 2e-4: 9.5e-08 / 4.4e-08 (15, 1000) |
| plummer 1e6 | 3 | 4.0e-03 / 3.4e-03 | 2e-1: 4.0e-03 / 3.8e-03 (3, 40) | 2e-1: 3.9e-03 / 2.7e-03 (3, 300) |
| plummer 1e6 | 6 | 1.9e-04 / 2.2e-04 | 2e-2: 9.7e-05 / 1.5e-04 (7, 40) | 2e-2: 9.6e-05 / 1.1e-04 (7, 300) |
| plummer 1e6 | 8 | 2.8e-05 / 4.1e-05 | 2e-3: 5.1e-05 / 6.3e-05 (11, 100) | 2e-3: 5.1e-05 / 5.5e-05 (11, 300) |
| plummer 1e6 | 12 | 1.0e-06 / 1.9e-06 | 2e-4: 1.2e-07 / 2.3e-07 (15, 200) | 2e-4: 1.2e-07 / 1.6e-07 (15, 1000) |
| plummer 1e7 | 3 | 3.6e-03 / 5.6e-03 | 2e-1: 3.6e-03 / 6.1e-03 (3, 40) | 2e-1: 3.6e-03 / 4.3e-03 (3, 300) |
| plummer 1e7 | 6 | 1.6e-04 / 3.7e-04 | 2e-2: 8.8e-05 / 2.6e-04 (7, 40) | 2e-2: 8.7e-05 / 1.8e-04 (7, 300) |
| plummer 1e7 | 8 | 2.4e-05 / 7.2e-05 | 2e-3: 4.8e-05 / 1.1e-04 (11, 100) | 2e-3: 4.8e-05 / 9.0e-05 (11, 300) |
| plummer 1e7 | 12 | 8.7e-07 / 3.5e-06 | 2e-4: 1.1e-07 / 4.1e-07 (15, 200) | 2e-4: 1.1e-07 / 2.7e-07 (15, 1000) |

FMM3D's error over `eps`, the default build on the cube at N = 10⁷ (Kathleen; the other
problems and the `FAST_KER` build step at the same places):

| eps | 5e-1 | 2e-1 | 1e-1 | 5e-2 | 2e-2 | 1e-2 | 5e-3 | 2e-3 | 1e-3 | 5e-4 | 2e-4 | 1e-4 … 5e-7 | 2e-7 … 5e-10 | 2e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| φ, rel. L2 | 9.5e-3 | 2.7e-3 | 9.1e-4 | 3.2e-4 | 6.6e-5 | 4.5e-5 | 3.9e-5 | 3.8e-5 | 3.8e-5 | 3.8e-5 | 7.7e-8 | 5.2e-8 … 4.9e-8 | 3.0e-11 | 4.0e-14 |

Our errors fall between FMM3D's steps at p = 6, 8 and 12: the matched `eps` is the nearest
step, so FMM3D is 1.7–2.1× more accurate at p = 6, 1.8–2.2× less accurate at p = 8 and
7.8–10× more accurate at p = 12 (it has no step between 3.8 × 10⁻⁵ and 7.7 × 10⁻⁸). The
time ratios below are at these accuracies; at p = 12 FMM3D is doing a more accurate
evaluation than ours, at p = 8 a less accurate one.

### 3.1 The outputs against each other (Kathleen, job 604824)

Added at the user's request (2026-10-18): the two codes' outputs compared at every point,
not only their errors at the sampled targets, on Kathleen at N = 10⁵ and 10⁶. Ours
(`nd-fmm-bench --write-output`: Batched, separate, one rank, 40 threads) and FMM3D at the
matched `eps` (both builds, 8 single-density calls) wrote φ and ∇φ at every point;
`nd-fmm-bench --compare` gives the relative L2 difference of FMM3D's output from ours over
all N points (the root mean square over the 8 vectors). The errors recomputed from these
files at the sampled targets equal §3's to the printed digits. "bound" is the sum of the
two codes' errors at the sampled targets, which the difference cannot exceed by much if
both errors hold over the whole field.

| problem | p | errors φ: ours / FMM3D / FAST_KER | FMM3D − ours: φ / ∇φ (bound) | FAST_KER − ours: φ / ∇φ (bound) |
| --- | ---: | --- | --- | --- |
| cube 1e5 | 3 | 3.0e-03 / 3.0e-03 / 2.7e-03 | 9.3e-05 / 1.1e-04 (6.0e-03 / 9.1e-03) | 1.3e-03 / 1.1e-03 (5.6e-03 / 7.3e-03) |
| cube 1e5 | 6 | 1.3e-04 / 7.1e-05 / 6.1e-05 | 1.3e-04 / 9.8e-05 (2.0e-04 / 5.0e-04) | 1.3e-04 / 9.5e-05 (1.9e-04 / 4.2e-04) |
| cube 1e5 | 8 | 1.9e-05 / 4.1e-05 / 3.5e-05 | 4.4e-05 / 3.3e-05 (6.0e-05 / 1.5e-04) | 3.9e-05 / 2.4e-05 (5.4e-05 / 1.1e-04) |
| cube 1e5 | 12 | 7.6e-07 / 8.3e-08 / 7.8e-08 | 7.0e-07 / 9.1e-07 (8.4e-07 / 3.5e-06) | 7.0e-07 / 9.0e-07 (8.4e-07 / 3.4e-06) |
| cube 1e6 | 3 | 2.9e-03 / 2.9e-03 / 2.7e-03 | 1.9e-04 / 3.1e-04 (5.8e-03 / 9.5e-03) | 9.2e-04 / 8.7e-04 (5.7e-03 / 7.7e-03) |
| cube 1e6 | 6 | 1.3e-04 / 6.9e-05 / 6.5e-05 | 1.2e-04 / 8.0e-05 (1.9e-04 / 5.1e-04) | 1.2e-04 / 7.6e-05 (1.9e-04 / 4.4e-04) |
| cube 1e6 | 8 | 1.9e-05 / 4.1e-05 / 3.8e-05 | 4.3e-05 / 2.7e-05 (5.9e-05 / 1.6e-04) | 4.1e-05 / 2.0e-05 (5.7e-05 / 1.2e-04) |
| cube 1e6 | 12 | 6.9e-07 / 8.2e-08 / 7.8e-08 | 7.0e-07 / 7.4e-07 (7.7e-07 / 3.4e-06) | 6.9e-07 / 7.2e-07 (7.7e-07 / 3.3e-06) |
| plummer 1e5 | 3 | 3.5e-03 / 3.6e-03 / 3.3e-03 | 7.4e-04 / 7.2e-04 (7.0e-03 / 2.3e-03) | 1.1e-03 / 6.6e-04 (6.8e-03 / 1.9e-03) |
| plummer 1e5 | 6 | 1.7e-04 / 9.3e-05 / 9.0e-05 | 1.6e-04 / 7.9e-05 (2.6e-04 / 1.3e-04) | 1.6e-04 / 7.1e-05 (2.6e-04 / 1.1e-04) |
| plummer 1e5 | 8 | 2.6e-05 / 4.7e-05 / 4.7e-05 | 5.4e-05 / 2.4e-05 (7.3e-05 / 3.3e-05) | 5.2e-05 / 2.2e-05 (7.3e-05 / 3.1e-05) |
| plummer 1e5 | 12 | 9.8e-07 / 1.0e-07 / 9.5e-08 | 1.0e-06 / 6.9e-07 (1.1e-06 / 7.3e-07) | 1.0e-06 / 6.8e-07 (1.1e-06 / 7.0e-07) |
| plummer 1e6 | 3 | 4.0e-03 / 4.0e-03 / 3.9e-03 | 5.0e-04 / 7.2e-04 (8.0e-03 / 7.1e-03) | 8.3e-04 / 9.1e-04 (7.9e-03 / 6.1e-03) |
| plummer 1e6 | 6 | 1.9e-04 / 9.7e-05 / 9.6e-05 | 1.8e-04 / 1.0e-04 (2.9e-04 / 3.7e-04) | 1.8e-04 / 9.6e-05 (2.9e-04 / 3.3e-04) |
| plummer 1e6 | 8 | 2.8e-05 / 5.1e-05 / 5.1e-05 | 6.1e-05 / 3.1e-05 (7.9e-05 / 1.0e-04) | 6.0e-05 / 2.9e-05 (7.8e-05 / 9.6e-05) |
| plummer 1e6 | 12 | 1.0e-06 / 1.2e-07 / 1.2e-07 | 1.1e-06 / 9.1e-07 (1.1e-06 / 2.1e-06) | 1.1e-06 / 9.0e-07 (1.1e-06 / 2.0e-06) |

Every difference stays within its bound (at most 0.99 of it). Where FMM3D is much more
accurate than ours (p = 12, 8–10×), the difference over all N points equals our own error
at the 1,000 sampled targets: the sample represents the whole field. At p = 3 the two
codes' errors are strongly correlated: FMM3D's default build differs from ours by
9 × 10⁻⁵ – 7 × 10⁻⁴ in φ while each is 3–4 × 10⁻³ from the direct sum (same expansion
order, similar leaves), the `FAST_KER` build (larger leaves) by about 10⁻³.

## 4. Every core, one charge vector

Milliseconds per evaluation: ours the median wall time of 10 evaluations of a built FMM
(Batched, separate / fused multiply–add); FMM3D the median of 8 calls of
`lfmm3d_s_c_g`, one per charge vector, after one untimed call, with the setup it repeats in
every call in parentheses (timed apart by the driver). "best FMM3D / best ours" > 1 means
ours is faster.

### 4.1 Kathleen (node-c11a-069 for ours, job 604787; node-c11a-019 for FMM3D, job 604793)

| problem | p | ours Batched / fused | FMM3D call (setup) | FMM3D FAST_KER call (setup) | best FMM3D / best ours |
| --- | ---: | --- | --- | --- | ---: |
| cube 1e5 | 3 | 6.66 / 6.63 | 84.5 (21.3) | 39.7 (16.5) | 5.99 |
| cube 1e5 | 6 | 16.2 / 13.5 | 106 (18.0) | 42.8 (13.2) | 3.18 |
| cube 1e5 | 8 | 30.3 / 28.3 | 81.9 (19.5) | 48.2 (13.2) | 1.70 |
| cube 1e5 | 12 | 95.1 / 83.3 | 206 (15.3) | 65.4 (13.2) | 0.79 |
| cube 1e6 | 3 | 71.0 / 64.9 | 1,273 (203) | 440 (153) | 6.79 |
| cube 1e6 | 6 | 119 / 108 | 1,523 (193) | 472 (151) | 4.38 |
| cube 1e6 | 8 | 210 / 192 | 728 (183) | 505 (153) | 2.64 |
| cube 1e6 | 12 | 652 / 568 | 1,443 (183) | 628 (153) | 1.11 |
| cube 1e7 | 3 | 781 / 772 | 16,773 (3,267) | 8,371 (2,740) | 10.84 |
| cube 1e7 | 6 | 1,196 / 1,135 | 21,006 (3,617) | 8,939 (2,719) | 7.87 |
| cube 1e7 | 8 | 2,015 / 1,847 | 8,550 (3,168) | 10,130 (2,948) | 4.63 |
| cube 1e7 | 12 | 5,715 / 4,931 | 14,184 (3,143) | 7,820 (2,678) | 1.59 |
| plummer 1e5 | 3 | 32.9 / 33.4 | 237 (30.1) | 115 (21.6) | 3.48 |
| plummer 1e5 | 6 | 78.3 / 76.8 | 286 (25.0) | 132 (19.9) | 1.72 |
| plummer 1e5 | 8 | 133 / 128 | 206 (20.0) | 164 (18.9) | 1.28 |
| plummer 1e5 | 12 | 339 / 304 | 428 (19.4) | 225 (16.3) | 0.74 |
| plummer 1e6 | 3 | 215 / 210 | 1,849 (274) | 727 (211) | 3.46 |
| plummer 1e6 | 6 | 502 / 491 | 2,156 (327) | 810 (197) | 1.65 |
| plummer 1e6 | 8 | 876 / 845 | 1,495 (217) | 963 (199) | 1.14 |
| plummer 1e6 | 12 | 2,179 / 2,025 | 2,337 (210) | 1,229 (178) | 0.61 |
| plummer 1e7 | 3 | 1,717 / 1,696 | 18,344 (4,414) | 8,356 (3,681) | 4.93 |
| plummer 1e7 | 6 | 3,906 / 3,800 | 21,501 (4,745) | 8,975 (3,664) | 2.36 |
| plummer 1e7 | 8 | 6,936 / 6,644 | 15,873 (4,358) | 10,243 (3,726) | 1.54 |
| plummer 1e7 | 12 | 18,296 / 16,665 | 21,826 (4,068) | 12,189 (3,379) | 0.73 |

### 4.2 locust (72 cores; ours and FMM3D one after the other, 2026-10-18 03:00–09:00)

locust is shared and has no scheduler. The other user's work during the runs, from the
process lists of our launches, was one vLLM engine holding the H100 at about one core
(`VLLM::EngineCore`, 100–110 % CPU) and a few small Python processes; the one-minute load
before a launch was 0.8–1.0 where this sweep's previous launch had not just ended.

| problem | p | ours Batched / fused | FMM3D call (setup) | FMM3D FAST_KER call (setup) | best FMM3D / best ours |
| --- | ---: | --- | --- | --- | ---: |
| cube 1e5 | 3 | 4.23 / 3.86 | 35.6 (14.5) | 47.4 (10.1) | 9.22 |
| cube 1e5 | 6 | 8.66 / 7.08 | 38.1 (14.1) | 48.4 (11.1) | 5.38 |
| cube 1e5 | 8 | 18.2 / 14.0 | 32.8 (14.1) | 49.5 (11.2) | 2.34 |
| cube 1e5 | 12 | 78.4 / 58.4 | 83.5 (12.3) | 57.7 (12.7) | 0.99 |
| cube 1e6 | 3 | 41.9 / 41.3 | 510 (135) | 585 (111) | 12.35 |
| cube 1e6 | 6 | 67.5 / 59.4 | 577 (132) | 589 (108) | 9.72 |
| cube 1e6 | 8 | 117 / 96.2 | 311 (132) | 601 (109) | 3.23 |
| cube 1e6 | 12 | 416 / 309 | 590 (132) | 648 (110) | 1.91 |
| cube 1e7 | 3 | 528 / 522 | 6,358 (2,290) | 4,717 (1,984) | 9.04 |
| cube 1e7 | 6 | 729 / 670 | 7,126 (2,256) | 4,911 (1,962) | 7.33 |
| cube 1e7 | 8 | 1,107 / 946 | 3,802 (2,080) | 5,207 (1,960) | 4.02 |
| cube 1e7 | 12 | 3,246 / 2,453 | 6,048 (2,060) | 8,489 (1,755) | 2.47 |
| plummer 1e5 | 3 | 10.7 / 10.2 | 103 (18.9) | 65.5 (14.3) | 6.41 |
| plummer 1e5 | 6 | 27.0 / 23.6 | 122 (18.8) | 71.0 (15.1) | 3.01 |
| plummer 1e5 | 8 | 52.0 / 43.6 | 82.3 (19.5) | 83.7 (15.4) | 1.89 |
| plummer 1e5 | 12 | 208 / 163 | 187 (15.5) | 242 (13.4) | 1.15 |
| plummer 1e6 | 3 | 73.4 / 73.0 | 824 (171) | 505 (142) | 6.92 |
| plummer 1e6 | 6 | 163 / 150 | 965 (172) | 531 (143) | 3.55 |
| plummer 1e6 | 8 | 285 / 249 | 685 (159) | 590 (147) | 2.37 |
| plummer 1e6 | 12 | 928 / 757 | 1,116 (149) | 1,335 (131) | 1.47 |
| plummer 1e7 | 3 | 646 / 643 | 9,208 (3,062) | 5,975 (2,599) | 9.29 |
| plummer 1e7 | 6 | 1,335 / 1,219 | 10,538 (3,031) | 6,185 (2,621) | 5.08 |
| plummer 1e7 | 8 | 2,337 / 2,021 | 7,801 (2,819) | 6,688 (2,587) | 3.31 |
| plummer 1e7 | 12 | 7,452 / 5,866 | 10,783 (2,693) | 13,157 (2,397) | 1.84 |

### 4.3 M3 Max (12 performance cores; every launch after the harness's quiet-machine gate)

At N = 10⁷ FMM3D's time is the median of 3 calls (vectors 1–3) after one untimed call, to
save the M3 Max's time; at 10⁵ and 10⁶ of 8, as elsewhere. Three of the launches ran
beside another (a build of mine, then the two sequences overlapping for lack of a proper
wait): our Plummer sphere at 10⁷, p = 12, separate (both launches), and FMM3D's default
build at 10⁷ (cube and Plummer sphere). All were run again alone (11:25–11:55), and only the
reruns are in the tables.

| problem | p | ours Batched / fused | FMM3D call (setup) | FMM3D FAST_KER call (setup) | best FMM3D / best ours |
| --- | ---: | --- | --- | --- | ---: |
| cube 1e5 | 3 | 10.5 / 9.64 | 74.1 (32.7) | 101 (12.1) | 7.69 |
| cube 1e5 | 6 | 23.7 / 18.5 | 82.3 (19.7) | 104 (13.1) | 4.44 |
| cube 1e5 | 8 | 46.7 / 33.8 | 79.5 (16.7) | 109 (13.0) | 2.35 |
| cube 1e5 | 12 | 165 / 115 | 212 (14.9) | 127 (13.6) | 1.10 |
| cube 1e6 | 3 | 103 / 98.2 | 1,141 (164) | 1,315 (130) | 11.61 |
| cube 1e6 | 6 | 214 / 182 | 1,339 (161) | 1,347 (130) | 7.35 |
| cube 1e6 | 8 | 414 / 331 | 723 (157) | 1,393 (130) | 2.19 |
| cube 1e6 | 12 | 1,571 / 1,043 | 1,534 (195) | 1,516 (131) | 1.45 |
| cube 1e7 | 3 | 1,312 / 1,277 | 14,066 (2,298) | 8,868 (1,918) | 6.94 |
| cube 1e7 | 6 | 2,320 / 1,989 | 16,555 (2,450) | 9,389 (1,950) | 4.72 |
| cube 1e7 | 8 | 4,027 / 3,193 | 7,949 (2,072) | 10,315 (1,980) | 2.49 |
| cube 1e7 | 12 | 13,761 / 9,942 | 15,422 (2,072) | 18,919 (1,738) | 1.55 |
| plummer 1e5 | 3 | 30.2 / 28.5 | 179 (34.2) | 124 (20.8) | 4.34 |
| plummer 1e5 | 6 | 78.9 / 71.9 | 214 (28.6) | 137 (22.2) | 1.91 |
| plummer 1e5 | 8 | 136 / 116 | 171 (22.0) | 159 (20.1) | 1.37 |
| plummer 1e5 | 12 | 384 / 303 | 396 (21.1) | 302 (18.6) | 1.00 |
| plummer 1e6 | 3 | 225 / 215 | 1,246 (218) | 996 (164) | 4.62 |
| plummer 1e6 | 6 | 627 / 599 | 1,455 (209) | 1,043 (159) | 1.74 |
| plummer 1e6 | 8 | 1,141 / 904 | 1,254 (181) | 1,152 (160) | 1.27 |
| plummer 1e6 | 12 | 3,282 / 2,719 | 2,186 (166) | 2,616 (143) | 0.80 |
| plummer 1e7 | 3 | 2,060 / 1,950 | 11,720 (2,581) | 9,764 (2,128) | 5.01 |
| plummer 1e7 | 6 | 5,391 / 4,937 | 13,574 (2,612) | 10,131 (2,097) | 2.05 |
| plummer 1e7 | 8 | 9,832 / 8,210 | 11,252 (2,292) | 10,977 (2,090) | 1.34 |
| plummer 1e7 | 12 | 32,803 / 25,043 | 18,380 (2,168) | 27,356 (1,884) | 0.73 |

## 5. Eight charge vectors in one evaluation

Every core; ours `--rhs 8` (`Fmm::evaluate_many`, the median of 3 evaluations), FMM3D one
call of `lfmm3d_s_c_g_vec` with nd = 8, in milliseconds for all 8 vectors.

FMM3D's default build ran out of memory in this call at N = 10⁷ on Kathleen (188 GB; exit
137, Slurm's `oom_kill`): on the cube at `eps` = 2 × 10⁻¹, 2 × 10⁻² and 2 × 10⁻⁴, on the
Plummer sphere at 2 × 10⁻². With leaves of at most 40–200 points the cube has about
1.0 × 10⁶ boxes at 10⁷ points (`nboxes` = 991,793 at `eps` ≥ 5 × 10⁻²), and FMM3D's
plane-wave expansions are stored per box, direction and density. The single-density calls
fit, as does every `FAST_KER` call (leaves of 300–1,000 points, 37,000–195,000 boxes) and
every call on locust (480 GB).

### 5.1 Kathleen

| problem | p | ours --rhs 8: Batched / fused | FMM3D vec (8) | FMM3D FAST_KER vec (8) | best FMM3D / best ours |
| --- | ---: | --- | --- | --- | ---: |
| cube 1e5 | 3 | 34.6 / 36.2 | 202 | 92.5 | 2.68 |
| cube 1e5 | 6 | 85.5 / 71.2 | 233 | 98.7 | 1.39 |
| cube 1e5 | 8 | 160 / 143 | 193 | 109 | 0.77 |
| cube 1e5 | 12 | 523 / 436 | 536 | 161 | 0.37 |
| cube 1e6 | 3 | 340 / 339 | 3,391 | 1,001 | 2.96 |
| cube 1e6 | 6 | 643 / 595 | 4,132 | 1,047 | 1.76 |
| cube 1e6 | 8 | 1,274 / 1,095 | 1,602 | 1,111 | 1.02 |
| cube 1e6 | 12 | 4,052 / 3,360 | 3,982 | 1,468 | 0.44 |
| cube 1e7 | 3 | 3,577 / 3,474 | out of memory | 13,728 | 3.95 |
| cube 1e7 | 6 | 6,236 / 5,701 | out of memory | 14,988 | 2.63 |
| cube 1e7 | 8 | 11,561 / 10,223 | 15,565 | 16,855 | 1.52 |
| cube 1e7 | 12 | 35,869 / 30,047 | out of memory | 16,906 | 0.56 |
| plummer 1e5 | 3 | 131 / 128 | 623 | 240 | 1.87 |
| plummer 1e5 | 6 | 349 / 322 | 766 | 291 | 0.91 |
| plummer 1e5 | 8 | 616 / 581 | 529 | 377 | 0.65 |
| plummer 1e5 | 12 | 1,580 / 1,422 | 1,197 | 585 | 0.41 |
| plummer 1e6 | 3 | 899 / 895 | 4,790 | 1,401 | 1.56 |
| plummer 1e6 | 6 | 2,354 / 2,280 | 5,801 | 1,617 | 0.71 |
| plummer 1e6 | 8 | 4,381 / 4,120 | 3,630 | 1,981 | 0.48 |
| plummer 1e6 | 12 | 12,048 / 10,632 | 6,559 | 3,137 | 0.30 |
| plummer 1e7 | 3 | 7,328 / 7,199 | 47,477 | 14,145 | 1.96 |
| plummer 1e7 | 6 | 18,623 / 17,715 | out of memory | 15,737 | 0.89 |
| plummer 1e7 | 8 | 36,255 / 33,870 | 34,523 | 18,396 | 0.54 |
| plummer 1e7 | 12 | 102,812 / 91,700 | 52,612 | 27,944 | 0.30 |

### 5.2 locust

| problem | p | ours --rhs 8: Batched / fused | FMM3D vec (8) | FMM3D FAST_KER vec (8) | best FMM3D / best ours |
| --- | ---: | --- | --- | --- | ---: |
| cube 1e5 | 3 | 19.3 / 18.2 | 84.2 | 86.0 | 4.64 |
| cube 1e5 | 6 | 53.4 / 40.8 | 94.7 | 86.7 | 2.12 |
| cube 1e5 | 8 | 117 / 82.5 | 106 | 91.0 | 1.10 |
| cube 1e5 | 12 | 466 / 305 | 252 | 114 | 0.37 |
| cube 1e6 | 3 | 210 / 203 | 1,812 | 1,056 | 5.21 |
| cube 1e6 | 6 | 394 / 329 | 2,146 | 1,069 | 3.25 |
| cube 1e6 | 8 | 737 / 565 | 925 | 1,099 | 1.64 |
| cube 1e6 | 12 | 2,620 / 1,838 | 2,190 | 1,273 | 0.69 |
| cube 1e7 | 3 | 2,440 / 2,391 | 23,927 | 9,331 | 3.90 |
| cube 1e7 | 6 | 3,892 / 3,437 | 25,311 | 9,732 | 2.83 |
| cube 1e7 | 8 | 6,631 / 5,211 | 8,823 | 10,743 | 1.69 |
| cube 1e7 | 12 | 21,528 / 15,203 | 18,285 | 16,024 | 1.05 |
| plummer 1e5 | 3 | 45.6 / 42.4 | 318 | 135 | 3.19 |
| plummer 1e5 | 6 | 147 / 119 | 382 | 154 | 1.29 |
| plummer 1e5 | 8 | 307 / 230 | 246 | 192 | 0.83 |
| plummer 1e5 | 12 | 1,148 / 782 | 531 | 499 | 0.64 |
| plummer 1e6 | 3 | 329 / 318 | 2,665 | 990 | 3.11 |
| plummer 1e6 | 6 | 833 / 730 | 3,379 | 1,071 | 1.47 |
| plummer 1e6 | 8 | 1,603 / 1,321 | 2,003 | 1,266 | 0.96 |
| plummer 1e6 | 12 | 5,418 / 4,026 | 3,423 | 2,664 | 0.66 |
| plummer 1e7 | 3 | 2,899 / 2,842 | 27,023 | 10,855 | 3.82 |
| plummer 1e7 | 6 | 7,114 / 6,104 | 30,558 | 11,767 | 1.93 |
| plummer 1e7 | 8 | 13,842 / 11,129 | 19,271 | 13,196 | 1.19 |
| plummer 1e7 | 12 | 46,910 / 34,492 | 27,827 | 25,298 | 0.73 |

### 5.3 M3 Max

| problem | p | ours --rhs 8: Batched / fused | FMM3D vec (8) | FMM3D FAST_KER vec (8) | best FMM3D / best ours |
| --- | ---: | --- | --- | --- | ---: |
| cube 1e5 | 3 | 47.0 / 40.7 | 172 | 274 | 4.23 |
| cube 1e5 | 6 | 141 / 99.7 | 206 | 285 | 2.06 |
| cube 1e5 | 8 | 311 / 202 | 218 | 307 | 1.08 |
| cube 1e5 | 12 | 1,194 / 828 | 577 | 346 | 0.42 |
| cube 1e6 | 3 | 520 / 488 | 3,375 | 3,561 | 6.92 |
| cube 1e6 | 6 | 1,355 / 1,036 | 4,163 | 3,607 | 3.48 |
| cube 1e6 | 8 | 2,925 / 2,034 | 1,892 | 3,769 | 0.93 |
| cube 1e6 | 12 | 10,808 / 7,396 | 4,935 | 4,126 | 0.56 |
| cube 1e7 | 3 | 5,922 / 5,677 | not run (memory) | 29,314 | 5.16 |
| cube 1e7 | 6 | 13,135 / 10,653 | not run (memory) | 33,488 | 3.14 |
| cube 1e7 | 8 | 26,515 / 19,445 | not run (memory) | 40,530 | 2.08 |
| cube 1e7 | 12 | 99,403 / 68,784 | not run (memory) | 52,890 | 0.77 |
| plummer 1e5 | 3 | 118 / 110 | 518 | 317 | 2.88 |
| plummer 1e5 | 6 | 378 / 316 | 653 | 365 | 1.16 |
| plummer 1e5 | 8 | 735 / 580 | 510 | 461 | 0.80 |
| plummer 1e5 | 12 | 2,334 / 1,760 | 1,244 | 873 | 0.50 |
| plummer 1e6 | 3 | 1,025 / 944 | 3,758 | 2,580 | 2.73 |
| plummer 1e6 | 6 | 3,421 / 2,748 | 4,659 | 2,751 | 1.00 |
| plummer 1e6 | 8 | 6,242 / 5,032 | 3,501 | 3,154 | 0.63 |
| plummer 1e6 | 12 | 21,581 / 17,444 | 6,542 | 7,247 | 0.38 |
| plummer 1e7 | 3 | 9,057 / 8,427 | not run (memory) | 25,249 | 3.00 |
| plummer 1e7 | 6 | 29,480 / 24,362 | not run (memory) | 26,454 | 1.09 |
| plummer 1e7 | 8 | 61,154 / 47,087 | not run (memory) | 29,901 | 0.64 |
| plummer 1e7 | 12 | 233,283 / 159,805 | not run (memory) | 76,226 | 0.48 |

## 6. Precomputation

Ours: the build of the run at the node's split (tree, partition, plan, tables, evaluator;
the median of one build), with the M2L tables in parentheses (no table cache); built once,
then evaluated any number of times. FMM3D: the setup the driver times apart (subdivision
rule, tree, sort), which every call repeats; FMM3D keeps nothing between calls. FMM3D's
other precomputation (its lists and the plane-wave tables) happens inside the call and is
not separable without changing it.

### 6.1 Kathleen

| problem | p | ours build 4x10 (tables) | FMM3D setup, 40 threads: default / FAST_KER |
| --- | ---: | --- | --- |
| cube 1e5 | 3 | 60.3 (6.00) | 21.3 / 16.5 |
| cube 1e5 | 6 | 187 (135) | 18.0 / 13.2 |
| cube 1e5 | 8 | 618 (566) | 19.5 / 13.2 |
| cube 1e5 | 12 | 4,935 (4,883) | 15.3 / 13.2 |
| cube 1e6 | 3 | 468 (6.50) | 203 / 153 |
| cube 1e6 | 6 | 593 (129) | 193 / 151 |
| cube 1e6 | 8 | 1,037 (564) | 183 / 153 |
| cube 1e6 | 12 | 5,376 (4,901) | 183 / 153 |
| cube 1e7 | 3 | 5,412 (7.68) | 3,267 / 2,740 |
| cube 1e7 | 6 | 5,491 (136) | 3,617 / 2,719 |
| cube 1e7 | 8 | 5,880 (568) | 3,168 / 2,948 |
| cube 1e7 | 12 | 10,239 (4,888) | 3,143 / 2,678 |
| plummer 1e5 | 3 | 75.6 (5.83) | 30.1 / 21.6 |
| plummer 1e5 | 6 | 236 (132) | 25.0 / 19.9 |
| plummer 1e5 | 8 | 678 (574) | 20.0 / 18.9 |
| plummer 1e5 | 12 | 5,036 (4,932) | 19.4 / 16.3 |
| plummer 1e6 | 3 | 665 (6.44) | 274 / 211 |
| plummer 1e6 | 6 | 802 (135) | 327 / 197 |
| plummer 1e6 | 8 | 1,245 (575) | 217 / 199 |
| plummer 1e6 | 12 | 5,590 (4,917) | 210 / 178 |
| plummer 1e7 | 3 | 8,306 (6.58) | 4,414 / 3,681 |
| plummer 1e7 | 6 | 8,442 (134) | 4,745 / 3,664 |
| plummer 1e7 | 8 | 8,899 (575) | 4,358 / 3,726 |
| plummer 1e7 | 12 | 13,238 (4,931) | 4,068 / 3,379 |

### 6.2 locust

| problem | p | ours build 8x9 (tables) | FMM3D setup, 72 threads: default / FAST_KER |
| --- | ---: | --- | --- |
| cube 1e5 | 3 | 16.6 (2.54) | 14.5 / 10.1 |
| cube 1e5 | 6 | 66.0 (50.8) | 14.1 / 11.1 |
| cube 1e5 | 8 | 227 (213) | 14.1 / 11.2 |
| cube 1e5 | 12 | 1,855 (1,841) | 12.3 / 12.7 |
| cube 1e6 | 3 | 133 (2.53) | 135 / 111 |
| cube 1e6 | 6 | 183 (51.7) | 132 / 108 |
| cube 1e6 | 8 | 341 (211) | 132 / 109 |
| cube 1e6 | 12 | 1,961 (1,826) | 132 / 110 |
| cube 1e7 | 3 | 1,715 (3.27) | 2,290 / 1,984 |
| cube 1e7 | 6 | 1,797 (51.9) | 2,256 / 1,962 |
| cube 1e7 | 8 | 1,934 (215) | 2,080 / 1,960 |
| cube 1e7 | 12 | 3,576 (1,857) | 2,060 / 1,755 |
| plummer 1e5 | 3 | 32.5 (2.52) | 18.9 / 14.3 |
| plummer 1e5 | 6 | 80.9 (50.7) | 18.8 / 15.1 |
| plummer 1e5 | 8 | 242 (212) | 19.5 / 15.4 |
| plummer 1e5 | 12 | 1,863 (1,833) | 15.5 / 13.4 |
| plummer 1e6 | 3 | 195 (2.52) | 171 / 142 |
| plummer 1e6 | 6 | 245 (50.8) | 172 / 143 |
| plummer 1e6 | 8 | 409 (213) | 159 / 147 |
| plummer 1e6 | 12 | 2,170 (1,868) | 149 / 131 |
| plummer 1e7 | 3 | 2,620 (3.97) | 3,062 / 2,599 |
| plummer 1e7 | 6 | 2,733 (52.1) | 3,031 / 2,621 |
| plummer 1e7 | 8 | 2,867 (215) | 2,819 / 2,587 |
| plummer 1e7 | 12 | 4,574 (1,853) | 2,693 / 2,397 |

### 6.3 M3 Max

| problem | p | ours build 1x12 (tables) | FMM3D setup, 12 threads: default / FAST_KER |
| --- | ---: | --- | --- |
| cube 1e5 | 3 | 71.2 (2.55) | 32.7 / 12.1 |
| cube 1e5 | 6 | 109 (54.4) | 19.7 / 13.1 |
| cube 1e5 | 8 | 289 (232) | 16.7 / 13.0 |
| cube 1e5 | 12 | 1,925 (1,865) | 14.9 / 13.6 |
| cube 1e6 | 3 | 711 (2.76) | 164 / 130 |
| cube 1e6 | 6 | 771 (58.1) | 161 / 130 |
| cube 1e6 | 8 | 950 (233) | 157 / 130 |
| cube 1e6 | 12 | 2,552 (1,844) | 195 / 131 |
| cube 1e7 | 3 | 8,410 (2.72) | 2,298 / 1,918 |
| cube 1e7 | 6 | 8,580 (57.9) | 2,450 / 1,950 |
| cube 1e7 | 8 | 8,817 (234) | 2,072 / 1,980 |
| cube 1e7 | 12 | 10,479 (1,883) | 2,072 / 1,738 |
| plummer 1e5 | 3 | 82.3 (2.70) | 34.2 / 20.8 |
| plummer 1e5 | 6 | 139 (59.0) | 28.6 / 22.2 |
| plummer 1e5 | 8 | 312 (233) | 22.0 / 20.1 |
| plummer 1e5 | 12 | 1,932 (1,851) | 21.1 / 18.6 |
| plummer 1e6 | 3 | 970 (2.74) | 218 / 164 |
| plummer 1e6 | 6 | 1,024 (58.2) | 209 / 159 |
| plummer 1e6 | 8 | 1,191 (229) | 181 / 160 |
| plummer 1e6 | 12 | 2,814 (1,853) | 166 / 143 |
| plummer 1e7 | 3 | 12,986 (2.84) | 2,581 / 2,128 |
| plummer 1e7 | 6 | 13,049 (59.3) | 2,612 / 2,097 |
| plummer 1e7 | 8 | 13,040 (231) | 2,292 / 2,090 |
| plummer 1e7 | 12 | 14,434 (1,857) | 2,168 / 1,884 |

## 7. Not measured, and what was left out

- **One thread.** Our one-thread runs (1 × 1, every problem, Batched separate and fused)
  ran on Kathleen (jobs 604787 and 604792) and locust; FMM3D's one-thread timings were
  stopped by the user on 2026-10-18 as not necessary, on Kathleen before they ran and on
  locust after a few (the default build's), so there is no one-thread comparison. The M3
  Max ran no one-thread launch.
- **Our GPU path** (`tools/scaling/run.sh t9-device`, CUDA f64 on locust's H100): not run.
  Another user's process held the H100 (79.6 GB, 100 % utilisation) from the first check
  (02:16) to the last (09:37); docs/phase4s/README.md, "Timing", forbids timing beside it.
  Metal has no f64.
- **The M3 Max's sweep and errors at 10⁷**: to save the M3 Max's time (the user's request),
  its FMM3D runs used the four `eps` that Kathleen and locust matched, without a sweep of
  its own beyond N ≤ 10⁶, and its runs at N = 10⁷ checked no errors (`T9_ERRORS_MAX_N`);
  its FMM3D errors at those `eps` come from its own timed runs (Kathleen's for the default
  build at 10⁷, which made no 8-call output there). FMM3D's 8-density call of the default
  build at 10⁷ was not run on the M3 Max (64 GB): it ran out of 188 GB on Kathleen.

## 8. Runs

| where | job / run | what | time |
| --- | --- | --- | --- |
| Kathleen | 604785 | FMM3D build and tests (both variants), the 6 problem files | 5.5 min |
| Kathleen | 604786 | FMM3D sweep (the default build's 8-density calls at 10⁷ out of memory) | 58 min |
| Kathleen | 604787 | ours, all but 1 × 1 at 10⁷ | 2 h 28 min |
| Kathleen | 604791 | FMM3D sweep, default build at 10⁷, one density per call | 3 h 23 min |
| Kathleen | 604792 | ours, 1 × 1 at 10⁷ | 1 h 10 min |
| Kathleen | 604793 | match, FMM3D timed on 40 cores | 45 min |
| Kathleen | 604823, 604824 | FMM3D and the driver rebuilt (604823 stopped at a stale job script), the cross-check | 3 min, 9 min |
| locust | /data/ucahtbe/logs/t9 | build, problems, sweep, ours, match, time | 03:00–09:00 |
| M3 Max | bench-results (git-ignored) | sweep at N ≤ 10⁶, ours at 1 × 12, FMM3D timed at 12 threads | 02:17–02:37, 08:43–11:55 (paused in between at the user's request) |

Kathleen: 9.0 node-hours in all, `singlenode`, waits of seconds. Commands: tools/fmm3d/README.md.
