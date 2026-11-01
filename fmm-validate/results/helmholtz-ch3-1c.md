# Helmholtz CH3.1c: the acceptance runs of CH3.1

Helmholtz task CH3.1c ([docs/helmholtz/tasks/CH3.1c.md](../../docs/helmholtz/tasks/CH3.1c.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH3.1, §2 (constraints 2, 9), §9.1. Measured on 2026-11-01 on the M6 (Mac mini, Apple
M6, 24 GB; [tools/m6/machine.md](../../tools/m6/machine.md)) from branch
`tbetcke/h-ch3-1c` at ac8f23f (the FMM code of `main` at 9983d02), release build, rustc
1.99.0, outside the sandbox, BLAS threads 1, `HWLOC_COMPONENTS=-darwin`. Each run used the
6 Super and Performance cores (1 × 6, 2 × 3 or 3 × 2 ranks × threads; several ranks with
`mpirun --host localhost:6 --map-by slot --bind-to none`), or 1 × 1 for the thread
repeats of §4. The runs at ε = 10⁻⁸ on W3, W4 and W1 at kD = 500 ran afterwards, on the same
binary, after the self-review named them missing against D1. The load average was 3.2 at
the start and 8.6 at the end of the first matrix, and 4.3 to 4.7 around the second (the
machine's background processes; CH3.1b), so the times are indicative only. Nothing here is
timed for a gate.

## Summary

- **The acceptance holds.** On W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500, the
  Helmholtz FMM's relative ℓ2 errors of φ and ∇φ over 1,000 sampled targets are below ε in
  every one of the 61 cases (81 runs with the thread and rank repeats), at ε = 10⁻³, 10⁻⁶
  and 10⁻⁸ (decision D1's range), the degrees S7's closed form at ε. One case of D1's range
  was not run: W3 at kD = 100 at 10⁻⁸ (4.0 million points at p = 46, about 21 GB by the
  10⁻⁶ run's 11.7 GB, beyond the M6's 24 GB with the system). The largest ratio of error to
  ε is 3.7e-3, for ∇φ on W3 at kD = 100, ε = 10⁻³.
- **Deterministic.** The checksum of the whole output (every value of every target) is
  the same at 1 and 6 threads on 13 cases (W1 and W3 to kD = 30, W1 at kD = 100). On 3
  cases it is the same at 1 × 6, 2 × 3 and 3 × 2 ranks × threads, and on W1 at kD = 500
  (2.0 million points) at 1 × 6 and 3 × 2. The rest ran once: W3 at kD = 100 and three of
  the four W4 distributions have no repeat.
- **S7's closed form overshoots**, as CH3.1b found: the errors are about 3 × 10³ to
  6 × 10⁶ below ε on the low-frequency trees (kD ≤ 30, and W4 at kD ≤ 10), and closest to ε
  on the high-frequency ones (W1, W3 and the W4 cube at kD = 100 and 500). That is CH3.3's calibration, recorded here and not tuned.
- **Cost, for orientation** (reference P2P until CH3.4; RCR on every level; no
  vectorisation): W1 at kD = 500 evaluates in 89 s at ε = 10⁻³, 508 s at 10⁻⁶ and
  1,131 s at 10⁻⁸. W3 at kD = 100 (4.0 million points) takes 105 s and 830 s, at 11.7 GB
  resident at 10⁻⁶.

## 1. Method

The harness is `nd_fmm_validate::helmholtz_accuracy`, run by the `helmholtz_accuracy`
example:

- **Workloads** (`helmholtz_workloads`, D = 1): W1, the sphere at ten points per wavelength
  (N = ⌈100 (kD)²/(4π)⌉); W3 resolved, uniform in the cube at ten points per wavelength
  (N = ⌈1000 (kD/2π)³⌉); W4, the four Laplace distributions at N = 10⁵. Sources equal
  targets. Rank r passes a contiguous block of the points.
- **Charges** (§9.1): one complex vector with parts uniform in [−1, 1) unless stated;
  real charges and eight vectors in §3.
- **The FMM**: `HelmholtzFmmBuilder` with `Degrees::Tolerance(ε)`, gradients and the
  defaults otherwise: c_leaf = 1, so the leaf rule; at most 64 points per leaf; and the
  512 MiB M2L block budget.
- **The oracle**: `nd_fmm_ref::helmholtz::p2p::direct_sum` in f64 (compensated sums), times
  1/(4π), at 1,000 distinct sampled targets over every source.
- **Errors**: §9.1's relative ℓ2 norms over the sampled targets, of the complex potential
  and of the complex 3-vector gradient field. With m vectors, the squares are summed over
  all of them.
- **Determinism**: a checksum of the whole output. It is the wrapping sum over every value
  of a mix of the value's global point index, its output slot and its bits, summed over the
  ranks. The same output gives the same checksum on any number of ranks, and one changed
  bit changes it (`tests/helmholtz_accuracy.rs`).
- **Times**: the build and the evaluation, wall time of the slowest rank.

In the tables, "p by level" lists levels 0 to the deepest; `a×b` is degree a on b
consecutive levels. Levels 0 and 1 get degree 0, since nothing reads them.

## 2. The acceptance rows

All rows are 1 rank × 6 threads. The errors are relative ℓ2; the last column is the
larger of the two errors divided by ε.

### W1, the sphere

| kD | N | ε | leaf level | p by level | leaves | evaluate s | φ | ∇φ | max / ε |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 796 | 10⁻³ | 1 | 0×2, 16 | 64 | 0.01 | 2.89e-8 | 1.94e-7 | 1.9e-4 |
| 10 | 796 | 10⁻⁶ | 1 | 0×2, 34 | 64 | 0.04 | 1.27e-13 | 1.98e-12 | 2.0e-6 |
| 10 | 796 | 10⁻⁸ | 1 | 0×2, 46 | 64 | 0.09 | 1.06e-15 | 1.75e-15 | 1.8e-7 |
| 30 | 7,162 | 10⁻³ | 3 | 0×2, 16×2 | 456 | 0.06 | 1.08e-7 | 2.63e-7 | 2.6e-4 |
| 30 | 7,162 | 10⁻⁶ | 3 | 0×2, 34×2 | 456 | 0.46 | 4.62e-13 | 1.57e-12 | 1.6e-6 |
| 30 | 7,162 | 10⁻⁸ | 3 | 0×2, 46×2 | 456 | 1.06 | 2.06e-15 | 3.04e-15 | 3.0e-7 |
| 100 | 79,578 | 10⁻³ | 4 | 0×2, 32, 19, 16×2 | 7,568 | 1.49 | 1.17e-6 | 1.83e-6 | 1.8e-3 |
| 100 | 79,578 | 10⁻⁶ | 4 | 0×2, 38, 34×3 | 7,568 | 10.61 | 5.52e-10 | 6.60e-10 | 6.6e-4 |
| 100 | 79,578 | 10⁻⁸ | 4 | 0×2, 46×4 | 7,568 | 23.47 | 7.41e-14 | 1.34e-13 | 1.3e-5 |
| 500 | 1,989,437 | 10⁻³ | 7 | 0×2, 126, 68, 38, 23, 16×2 | 217,897 | 88.81 | 7.28e-7 | 1.41e-6 | 1.4e-3 |
| 500 | 1,989,437 | 10⁻⁶ | 7 | 0×2, 136, 76, 45, 34×3 | 217,897 | 507.74 | 3.48e-10 | 4.35e-10 | 4.4e-4 |
| 500 | 1,989,437 | 10⁻⁸ | 7 | 0×2, 142, 81, 49, 46×3 | 217,897 | 1130.93 | 3.20e-12 | 4.01e-12 | 4.0e-4 |

The tree has 3 to 8 levels; at most 64 points in a leaf (56 at kD = 500). The build took at
most 3.6 s (kD = 500); the oracle at most 5.7 s. At kD = 500 the process held 8.4 GB
resident with ε = 10⁻³ and 10⁻⁶ in one run, and 11.7 GB at 10⁻⁸.

### W3, the cube at ten points per wavelength

| kD | N | ε | leaf level | p by level | leaves | evaluate s | φ | ∇φ | max / ε |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 4,032 | 10⁻³ | 1 | 0×2, 16×2 | 225 | 0.05 | 2.02e-8 | 8.96e-8 | 9.0e-5 |
| 10 | 4,032 | 10⁻⁶ | 1 | 0×2, 34×2 | 225 | 0.28 | 3.98e-13 | 7.62e-12 | 7.6e-6 |
| 10 | 4,032 | 10⁻⁸ | 1 | 0×2, 46×2 | 225 | 0.59 | 2.09e-15 | 3.62e-14 | 3.6e-6 |
| 30 | 108,849 | 10⁻³ | 3 | 0×2, 16×3 | 4,096 | 1.38 | 1.41e-7 | 3.42e-7 | 3.4e-4 |
| 30 | 108,849 | 10⁻⁶ | 3 | 0×2, 34×3 | 4,096 | 9.17 | 7.11e-12 | 4.23e-11 | 4.2e-5 |
| 30 | 108,849 | 10⁻⁸ | 3 | 0×2, 46×3 | 4,096 | 23.93 | 2.12e-14 | 2.14e-13 | 2.1e-5 |
| 100 | 4,031,442 | 10⁻³ | 4 | 0×2, 32, 19, 16×3 | 262,144 | 104.61 | 2.69e-6 | 3.72e-6 | 3.7e-3 |
| 100 | 4,031,442 | 10⁻⁶ | 4 | 0×2, 38, 34×4 | 262,144 | 829.91 | 6.42e-10 | 6.33e-10 | 6.4e-4 |

At kD = 100 the build took 3.9 s, the oracle 11 s; maximum resident 5.9 GB (10⁻³) and
11.7 GB (10⁻⁶), at most 36 points in a leaf. 10⁻⁸ at kD = 100 was not run (Summary).

### W4, the Laplace distributions at N = 10⁵

| Distribution | kD | ε | levels | leaf level | p by level | leaves | evaluate s | φ | ∇φ | max / ε |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cube | 1 | 10⁻³ | 5 | 0 | 0×2, 16×3 | 4,096 | 1.68 | 3.46e-8 | 1.52e-7 | 1.5e-4 |
| cube | 1 | 10⁻⁶ | 5 | 0 | 0×2, 34×3 | 4,096 | 11.58 | 2.36e-12 | 2.08e-11 | 2.1e-5 |
| cube | 1 | 10⁻⁸ | 5 | 0 | 0×2, 46×3 | 4,096 | 19.13 | 1.42e-14 | 1.36e-13 | 1.4e-5 |
| cube | 10 | 10⁻³ | 5 | 1 | 0×2, 16×3 | 4,096 | 1.45 | 3.27e-8 | 1.53e-7 | 1.5e-4 |
| cube | 10 | 10⁻⁶ | 5 | 1 | 0×2, 34×3 | 4,096 | 9.25 | 2.07e-12 | 2.05e-11 | 2.1e-5 |
| cube | 10 | 10⁻⁸ | 5 | 1 | 0×2, 46×3 | 4,096 | 20.98 | 1.23e-14 | 1.34e-13 | 1.3e-5 |
| cube | 100 | 10⁻³ | 5 | 4 | 0×2, 32, 19, 16 | 4,096 | 1.49 | 1.61e-6 | 2.57e-6 | 2.6e-3 |
| cube | 100 | 10⁻⁶ | 5 | 4 | 0×2, 38, 34×2 | 4,096 | 9.51 | 5.57e-10 | 7.67e-10 | 7.7e-4 |
| cube | 100 | 10⁻⁸ | 5 | 4 | 0×2, 46×3 | 4,096 | 21.58 | 1.50e-13 | 3.09e-13 | 3.1e-5 |
| sphere | 1 | 10⁻³ | 6 | 0 | 0×2, 16×4 | 8,170 | 1.72 | 2.12e-8 | 1.16e-9 | 2.1e-5 |
| sphere | 1 | 10⁻⁶ | 6 | 0 | 0×2, 34×4 | 8,170 | 13.77 | 8.35e-13 | 2.25e-12 | 2.3e-6 |
| sphere | 1 | 10⁻⁸ | 6 | 0 | 0×2, 46×4 | 8,170 | 28.49 | 5.08e-13 | 2.25e-12 | 2.3e-4 |
| sphere | 10 | 10⁻³ | 6 | 1 | 0×2, 16×4 | 8,170 | 1.75 | 2.24e-8 | 1.19e-9 | 2.2e-5 |
| sphere | 10 | 10⁻⁶ | 6 | 1 | 0×2, 34×4 | 8,170 | 12.28 | 7.76e-13 | 2.25e-12 | 2.3e-6 |
| sphere | 10 | 10⁻⁸ | 6 | 1 | 0×2, 46×4 | 8,170 | 28.94 | 4.69e-13 | 2.25e-12 | 2.3e-4 |
| sphere | 100 | 10⁻³ | 6 | 4 | 0×2, 32, 19, 16×2 | 8,170 | 1.87 | 7.24e-7 | 2.93e-8 | 7.2e-4 |
| sphere | 100 | 10⁻⁶ | 6 | 4 | 0×2, 38, 34×3 | 8,170 | 12.42 | 4.28e-10 | 1.54e-11 | 4.3e-4 |
| sphere | 100 | 10⁻⁸ | 6 | 4 | 0×2, 46×4 | 8,170 | 29.49 | 4.62e-13 | 2.25e-12 | 2.3e-4 |
| plummer | 1 | 10⁻³ | 9 | 0 | 0×2, 16×7 | 5,573 | 2.30 | 6.04e-8 | 1.52e-7 | 1.5e-4 |
| plummer | 1 | 10⁻⁶ | 9 | 0 | 0×2, 34×7 | 5,573 | 14.25 | 7.87e-12 | 4.31e-11 | 4.3e-5 |
| plummer | 1 | 10⁻⁸ | 9 | 0 | 0×2, 46×7 | 5,573 | 37.19 | 8.04e-14 | 6.46e-13 | 6.5e-5 |
| plummer | 10 | 10⁻³ | 9 | 1 | 0×2, 16×7 | 5,573 | 2.31 | 4.50e-8 | 1.52e-7 | 1.5e-4 |
| plummer | 10 | 10⁻⁶ | 9 | 1 | 0×2, 34×7 | 5,573 | 15.71 | 5.79e-12 | 4.31e-11 | 4.3e-5 |
| plummer | 10 | 10⁻⁸ | 9 | 1 | 0×2, 46×7 | 5,573 | 33.30 | 5.90e-14 | 6.46e-13 | 6.5e-5 |
| plummer | 100 | 10⁻³ | 9 | 4 | 0×2, 32, 19, 16×5 | 7,491 | 3.03 | 4.25e-7 | 2.65e-7 | 4.3e-4 |
| plummer | 100 | 10⁻⁶ | 9 | 4 | 0×2, 38, 34×6 | 7,491 | 18.98 | 9.00e-11 | 6.30e-11 | 9.0e-5 |
| plummer | 100 | 10⁻⁸ | 9 | 4 | 0×2, 46×7 | 7,491 | 45.30 | 8.10e-14 | 6.32e-13 | 6.3e-5 |
| clusters | 1 | 10⁻³ | 9 | 0 | 0×2, 16×7 | 8,779 | 2.92 | 4.28e-8 | 2.03e-7 | 2.0e-4 |
| clusters | 1 | 10⁻⁶ | 9 | 0 | 0×2, 34×7 | 8,779 | 18.42 | 5.64e-12 | 8.71e-11 | 8.7e-5 |
| clusters | 1 | 10⁻⁸ | 9 | 0 | 0×2, 46×7 | 8,779 | 43.93 | 4.58e-14 | 9.23e-13 | 9.2e-5 |
| clusters | 10 | 10⁻³ | 9 | 0 | 0×2, 16×7 | 8,779 | 2.83 | 4.26e-8 | 2.03e-7 | 2.0e-4 |
| clusters | 10 | 10⁻⁶ | 9 | 0 | 0×2, 34×7 | 8,779 | 18.44 | 5.62e-12 | 8.71e-11 | 8.7e-5 |
| clusters | 10 | 10⁻⁸ | 9 | 0 | 0×2, 46×7 | 8,779 | 43.07 | 4.56e-14 | 9.22e-13 | 9.2e-5 |
| clusters | 100 | 10⁻³ | 9 | 4 | 0×2, 23, 16×6 | 8,779 | 2.97 | 4.18e-8 | 2.03e-7 | 2.0e-4 |
| clusters | 100 | 10⁻⁶ | 9 | 4 | 0×2, 34×7 | 8,779 | 18.49 | 5.25e-12 | 8.67e-11 | 8.7e-5 |
| clusters | 100 | 10⁻⁸ | 9 | 4 | 0×2, 46×7 | 8,779 | 46.00 | 4.27e-14 | 9.19e-13 | 9.2e-5 |

The leaf rule sets level 4 at kD = 100 on every distribution, so the clusters' and the
Plummer sphere's trees keep their adaptive depth (9 levels) below it.

## 3. Other charges and k

W1 at kD = 100, 1 × 6:

| Variant | ε | evaluate s | φ | ∇φ |
| --- | --- | ---: | ---: | ---: |
| real charges (real multipoles) | 10⁻³ | 1.39 | 1.03e-6 | 1.71e-6 |
| real charges | 10⁻⁶ | 8.47 | 4.85e-10 | 7.72e-10 |
| 8 complex vectors, over all | 10⁻⁶ | 85.96 | 5.92e-10 | 8.35e-10 |
| complex k, kD = 100 + 10i | 10⁻³ | 1.76 | 2.97e-7 | 4.61e-7 |
| complex k, kD = 100 + 10i | 10⁻⁶ | 13.42 | 1.20e-10 | 1.38e-10 |

The eight vectors' errors, φ / ∇φ: 5.52e-10 / 6.60e-10 (vector 0, the one-vector row's),
4.99e-10 / 7.31e-10, 5.16e-10 / 6.48e-10, 4.40e-10 / 5.77e-10, 9.20e-10 / 1.38e-9,
5.37e-10 / 7.87e-10, 5.54e-10 / 8.25e-10, 5.80e-10 / 8.05e-10. The largest is 2.1 times the
smallest for φ and 2.4 times for ∇φ. Vector 0's errors are the one-vector run's, consistent with CH3.1b's bit
identity of vector v and the one-vector FMM. Eight vectors took 8.1 times one vector's
time: the reference P2P and RCR run per vector.

## 4. Determinism

Checksums of the whole output (§1), equal in each row:

| Case | Runs (ranks × threads) | Checksum |
| --- | --- | --- |
| W1 kD = 10, ε = 10⁻³ / 10⁻⁶ / 10⁻⁸ | 1 × 1, 1 × 6 | `63472007828c26d8` / `cd28868ec2251dda` / `ca12f8035db47832` |
| W1 kD = 30, ε = 10⁻³ / 10⁻⁶ / 10⁻⁸ | 1 × 1, 1 × 6 | `afc24e78bbcbf09b` / `cdf7cb5ee939f351` / `37e80626bf49748f` |
| W1 kD = 100, ε = 10⁻³ / 10⁻⁸ | 1 × 1, 1 × 6 | `01003b6877ad138a` / `7c61932e7aa82fa9` |
| W1 kD = 100, ε = 10⁻⁶ | 1 × 1, 1 × 6, 2 × 3, 3 × 2 | `0062dceef767aca1` |
| W3 kD = 10, ε = 10⁻³ / 10⁻⁶ | 1 × 1, 1 × 6 | `f5965c4fd7f5834d` / `071aeaf6b7ca53e2` |
| W3 kD = 30, ε = 10⁻³ | 1 × 1, 1 × 6 | `4d28864830075fe9` |
| W3 kD = 30, ε = 10⁻⁶ | 1 × 1, 1 × 6, 2 × 3, 3 × 2 | `2bb4c75c47fc3750` |
| W4 Plummer kD = 100, ε = 10⁻³ | 1 × 6, 2 × 3, 3 × 2 | `60464264ed6f4103` |
| W1 kD = 500, ε = 10⁻³ | 1 × 6, 3 × 2 | `e01cdb7fb912e978` |

That makes 13 cases with equal checksums at 1 and 6 threads, and 4 with equal checksums
on several rank counts: 3 on 1, 2 and 3 ranks, and W1 at kD = 500 on 1 and 3. W3 at
kD = 100, three of the four W4 distributions and the 10⁻⁸ runs of the second matrix ran
once. On several ranks the errors are
the one-rank errors to the printed digits. The error sums are reduced in another order, so
their last bits may differ; the test bounds the difference at 1e-12 relative. The
evaluation times on 2 × 3 and 3 × 2 are 1–18 % above 1 × 6; W1 at kD = 500 on 3 × 2 took 91 s against 89 s, and its build 2.0 s against
3.0 s. The difference was not investigated: timing is not this task's.

## 5. Observations for later tasks

- **CH3.3 (calibration of p_l).** S7's closed form at ε gives p = 16 at ε = 10⁻³, 34 at
  10⁻⁶ and 46 at 10⁻⁸ on every low-frequency level (k a small). There the errors are
  about 3 × 10³ to 6 × 10⁶ below ε: on the W4 cube at kD = 1, ∇φ 1.5e-7 at 10⁻³, 2.1e-11
  at 10⁻⁶ and 1.4e-13 at 10⁻⁸. The
  high-frequency levels drive the errors closest to ε (max / ε 3.7e-3 on W3 at kD = 100).
  The low-frequency degrees can come down several steps.
- **Cost.** At ε = 10⁻⁶ the evaluation is 5.7–8 times its 10⁻³ time (W1 at kD = 100 and
  500, W3 at kD = 100), and 10⁻⁸ about 2.2 times 10⁻⁶ (W1 at kD = 100 and 500). Not
  measured by part; the likely cause is the translations' growth with p (RCR is O(p³)
  per pair) at p = 34 and 46 against 16, as P2P does not change with ε. P2P is the
  reference kernel; CH3.4 replaces it, and CH3.2 adds the faster M2L strategies.
- **Memory.** Maximum resident: W3 at kD = 100, ε = 10⁻⁶ 11.7 GB on one rank; W1 at
  kD = 500 8.4 GB (10⁻³ and 10⁻⁶ in one process) and 11.7 GB at 10⁻⁸. Not broken down;
  presumably the expansions, 2 (p + 1)² values per box and plane at p = 34 to 46, and the
  per-level tables within the 512 MiB block budget.

## 6. Reproducing

```sh
export HWLOC_COMPONENTS=-darwin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
X=target/release/examples/helmholtz_accuracy
$X --workload w1 --kd 10,30,100 --eps 1e-3,1e-6,1e-8 --threads 1,6
$X --workload w1 --kd 500 --eps 1e-3,1e-6 --threads 6   # then --eps 1e-8
$X --workload w3 --kd 10,30 --eps 1e-3,1e-6 --threads 1,6  # then --eps 1e-8 --threads 6
$X --workload w3 --kd 100 --eps 1e-3 --threads 6          # then --eps 1e-6
for d in cube sphere plummer clusters; do
  $X --workload w4-$d:100000 --kd 1,10,100 --eps 1e-3,1e-6,1e-8 --threads 6
done
$X --workload w1 --kd 100 --eps 1e-3,1e-6 --charges real --threads 6
$X --workload w1 --kd 100 --eps 1e-6 --vectors 8 --threads 6
$X --workload w1 --kd 100 --kim 10 --eps 1e-3,1e-6 --threads 6
M="mpirun --host localhost:6 --map-by slot --bind-to none --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0"
$M -n 2 $X --workload w1 --kd 100 --eps 1e-6 --threads 3     # and -n 3 … --threads 2
$M -n 3 $X --workload w1 --kd 500 --eps 1e-3 --threads 2
```

The first matrix took 38 minutes, the 10⁻⁸ runs of W3, W4 and W1 at kD = 500 another 26.
