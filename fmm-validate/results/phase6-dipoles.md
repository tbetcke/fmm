# Phase 6 T8: dipole sources (C6.4)

Phase 6, task T8 ([docs/phase6/T8-dipoles.md](../../docs/phase6/T8-dipoles.md)); design
[docs/design/optimisation.md](../../docs/design/optimisation.md) §7, signed off on
2026-10-15 (§7.5 records what was built). Measured on 2026-10-17 from branch
`tbetcke/phase6_t8` (main at 16c2abb plus this task's changes), release builds, rustc
1.99.0, every BLAS thread variable at 1, on the Apple M3 Max (12 performance and 4
efficiency cores) unless stated. Kathleen ran the x86 kernel tests (job 604594, one node,
node-c11a-143, `singlenode`, 2 min) and the multi-rank checks on two nodes (jobs 604595 and
604598, `test`; §6); no job used more than two nodes. locust was not used.

## Summary

1. **Conventions** (signed off by the user on 2026-10-17, with the proposal of this task):
   CONVENTIONS §3.7 "Dipole sources": a moment d at y gives φ = d · ∇_y 1/|x − y| =
   d · (x − y)/|x − y|³, with the scaled coefficients M̃ₙᵐ = Σ d̂ · ∇ conj Rₙᵐ(u) and
   L̃ₙᵐ = Σ d̂ · ∇ conj Iₙᵐ(u), d̂ = d / r; §3.13: source chunks of 6 (dipoles) and 7 (charges
   and dipoles) values per point with the moments in units of the source leaf, the
   operator rows, the f32 range. `CONVENTION_VERSION` 2; the three harmonics fixtures were
   regenerated (only their header changed) and a dipole fixture added
   (`fmm-ref/fixtures/dipoles.json`, `tools/fixtures/gen_dipoles.py`).
2. **Accuracy** (§1): the dipole FMM and the FMM of charges and dipoles pass the gate
   signed off with the brief (twice each source type's own single-translation prediction
   on the C3.2 cube, twice the cube's error on the C3.3 Plummer sphere) at f64 p = 3, 8 and
   12 and f32 p = 3 and 8, at 1, 2 and 4 ranks, with the same errors at every rank count
   (the output is bit for bit rank-count invariant). The gate is loose for dipoles: the FMM
   error is 0.05–0.26 of the prediction on the cube (charges: 1.5–1.65).
3. **Cost on the host** (§2): a dipole evaluation costs 1.04–1.26× a charge evaluation on
   the N = 10⁶ cube (f64 and f32, p = 3–12, one thread), charges and dipoles together
   1.10–1.86×; only P2M, P2L and P2P (the upward local and leaves stages) change, the
   downward pass not at all.
4. **The device** (§3): with dipoles P2M, P2L and P2P run on the host fallback (§7.4 of the
   design: first). On Metal at N = 10⁶ (f32) the fallback takes **92–97 % of a dipole
   evaluation with one host thread and 55–77 % with twelve**, far above the 20 % at which
   §7.4 calls for device kernels: a dipole evaluation is 15–44× (one thread) and 2.5–6×
   (twelve) a charge evaluation on the device. **Device kernels for the dipole kinds are
   indicated** (open question below). With every kind on the fallback the device output is
   the host's bit for bit (CPU runtime, all MPI scenarios; Metal, by hand).
5. **The SIMD dipole P2P** (§4): within its contract (32 u_T for φ, 176 u_T for ∇φ, derived
   in docs/design/simd-p2p.md §4.8) on NEON, AVX2 and AVX-512 (Kathleen); worst over 10⁶
   pairs 10.5 u (φ) and 33.8 u (∇φ).

**Open question for the user.** §7.4 builds device kernels for the dipole kinds when the
fallback exceeds 20 % of a device evaluation; it does, by a wide margin. P2P is 70 % (p = 3)
to 40 % (p = 8) of a dipole evaluation with twelve threads, P2M 11–21 %. Building them in
this task, or in a follow-up, was the decision (§5): a follow-up, T8d.

## 1. Accuracy

`nd-fmm-exec`'s ignored `tests/dipoles.rs` (its own MPI test, any rank count). The cube of
the C3.2 gate (N = 10⁵ uniform in [−1, 1)³, a uniform level-4 tree, seed `0xc32`) and the
Plummer sphere of C3.3 (N = 10⁵, scale 0.1, 64 points per leaf, seed `0xc33`), sources
equal to targets, 1,000 sampled targets; eight strength vectors per problem (charges
uniform in [−1, 1), moments uniform in [−1, 1)³). Error measure: the relative L2 error of φ
at the sampled targets against `direct_sum_dipoles` (plus `direct_sum` with charges)
divided by 4π, the root mean square over the eight vectors. Prediction: the P2M → M2L → L2P
row of `nd-fmm-validate`'s `accuracy` example (`--sources dipoles`, `--sources
charges+dipoles`; 1,000 sources, 316 offsets, the pooled targets, Phase 1 T7's seed),
where f32 agrees with f64 to the printed digits at p = 3 and 8.

| problem | sources | precision | p | error | prediction | error / prediction | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cube | dipoles | f64 | 3 | 2.810e-3 | 5.84e-2 | 0.05 | 1.17e-1 |
| cube | dipoles | f32 | 3 | 2.810e-3 | 5.84e-2 | 0.05 | 1.17e-1 |
| cube | dipoles | f64 | 8 | 3.602e-5 | 5.15e-4 | 0.07 | 1.03e-3 |
| cube | dipoles | f32 | 8 | 3.603e-5 | 5.15e-4 | 0.07 | 1.03e-3 |
| cube | dipoles | f64 | 12 | 1.780e-6 | 2.34e-5 | 0.08 | 4.68e-5 |
| cube | charges and dipoles | f64 | 3 | 2.808e-3 | 1.74e-2 | 0.16 | 3.48e-2 |
| cube | charges and dipoles | f32 | 3 | 2.808e-3 | 1.74e-2 | 0.16 | 3.48e-2 |
| cube | charges and dipoles | f64 | 8 | 3.599e-5 | 1.50e-4 | 0.24 | 3.00e-4 |
| cube | charges and dipoles | f32 | 8 | 3.599e-5 | 1.50e-4 | 0.24 | 3.00e-4 |
| cube | charges and dipoles | f64 | 12 | 1.779e-6 | 6.81e-6 | 0.26 | 1.36e-5 |
| Plummer | dipoles | f64 | 3 | 1.002e-3 | | | 5.62e-3 |
| Plummer | dipoles | f32 | 3 | 1.002e-3 | | | 5.62e-3 |
| Plummer | dipoles | f64 | 8 | 1.203e-5 | | | 7.20e-5 |
| Plummer | dipoles | f32 | 8 | 1.221e-5 | | | 7.21e-5 |
| Plummer | dipoles | f64 | 12 | 6.089e-7 | | | 3.56e-6 |
| Plummer | charges and dipoles | f64 | 3 | 1.002e-3 | | | 5.62e-3 |
| Plummer | charges and dipoles | f32 | 3 | 1.002e-3 | | | 5.62e-3 |
| Plummer | charges and dipoles | f64 | 8 | 1.203e-5 | | | 7.20e-5 |
| Plummer | charges and dipoles | f32 | 8 | 1.223e-5 | | | 7.20e-5 |
| Plummer | charges and dipoles | f64 | 12 | 6.088e-7 | | | 3.56e-6 |

The Plummer gate is twice the cube's error of the same row. Every value is the same, to the
printed digits, at 1, 2 and 4 ranks (170, 137 and 48 s), as the bitwise rank-count
invariance of the host path implies.

- **Why the gate is loose.** The prediction measures one well-separated pair of boxes, where
  the far field is all of the potential. In the FMM most of a dipole potential is near
  field: a dipole's term decays as 1/r², a charge's as 1/r, so at a target the near
  sources, summed exactly by P2P, dominate the dipole potential and the truncation error of
  the far field counts for less in the relative error (charges: 1.5–1.65 of the
  prediction, Phase 3 T9). A tighter dipole gate would be measured against the FMM rather
  than the single pair; the signed-off gate stands, with these margins recorded.
- **Charges and dipoles** together have the dipoles' error: with charges and moments of the
  same size on this point density the dipole potential is the larger part
  (Σ |d| / r² against Σ |q| / r at inter-point distances of about 0.02).
- **Small trees** (`tests/mpi_exec.rs`, **dipole sources**, N = 2,000, debug, every rank
  count of CI): on the cube 5.3e-4 (φ) and 1.7e-5 (∇φ) at p = 4, 1.7e-5 and 9.8e-7 at p = 8;
  on the Plummer sphere 1.2e-3 and 8.0e-5 at p = 4, 3.5e-5 and 5.0e-6 at p = 8, the same for
  both source types; at p = 6 four threads, overlap and a repeat bit for bit, `Reference`
  within 3.1e-16, f32 within 2.3e-6 of f64, on several ranks the one-rank `Fmm` bit for bit;
  charges and dipoles within 6.0e-16 of the sum of the two (linearity).

## 2. Cost on the host

`nd-fmm-validate`'s `dipole_cost` example: the cube (uniform in [−1, 1)³) or the Plummer
sphere (scale 0.1) at N = 10⁶, sources equal to targets, the default tree (64 points per
leaf), gradients on, one thread; medians over five evaluations after a warm-up, ms (each
stage with its ratio to the charges'). Load average 1.7–5.0 during the runs (the machine
otherwise idle).

Cube, f64:

| p | sources | evaluation | / charges | load | upward local | downward | leaves |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | charges | 1159.9 | 1.00 | 3.1 | 24.4 | 357.0 | 760.6 |
| 3 | dipoles | 1459.5 | 1.26 | 8.1 (2.62) | 50.1 (2.05) | 356.6 (1.00) | 1028.9 (1.35) |
| 3 | charges and dipoles | 2157.0 | 1.86 | 11.0 (3.57) | 72.6 (2.97) | 356.7 (1.00) | 1702.1 (2.24) |
| 8 | charges | 6479.6 | 1.00 | 3.1 | 158.2 | 5284.0 | 1018.6 |
| 8 | dipoles | 6918.0 | 1.07 | 8.2 (2.66) | 282.8 (1.79) | 5321.8 (1.01) | 1291.4 (1.27) |
| 8 | charges and dipoles | 7673.4 | 1.18 | 11.3 (3.64) | 424.7 (2.68) | 5273.8 (1.00) | 1962.5 (1.93) |
| 12 | charges | 15573.7 | 1.00 | 3.1 | 370.5 | 13805.2 | 1383.8 |
| 12 | dipoles | 16126.9 | 1.04 | 8.0 (2.60) | 611.0 (1.65) | 13838.8 (1.00) | 1659.6 (1.20) |
| 12 | charges and dipoles | 17120.2 | 1.10 | 11.2 (3.61) | 928.8 (2.51) | 13859.2 (1.00) | 2325.7 (1.68) |

Cube, f32:

| p | sources | evaluation | / charges | load | upward local | downward | leaves |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | charges | 773.7 | 1.00 | 1.8 | 24.2 | 314.8 | 421.8 |
| 3 | dipoles | 960.1 | 1.24 | 3.7 (2.05) | 49.8 (2.06) | 315.6 (1.00) | 574.8 (1.36) |
| 3 | charges and dipoles | 1311.5 | 1.70 | 8.3 (4.60) | 72.6 (3.00) | 313.0 (0.99) | 905.6 (2.15) |
| 8 | charges | 4813.7 | 1.00 | 1.8 | 156.4 | 3965.9 | 679.7 |
| 8 | dipoles | 5202.1 | 1.08 | 3.7 (2.02) | 282.0 (1.80) | 4072.1 (1.03) | 836.5 (1.23) |
| 8 | charges and dipoles | 5577.6 | 1.16 | 8.5 (4.64) | 407.3 (2.60) | 3968.9 (1.00) | 1166.0 (1.72) |

Plummer sphere, f64, p = 8:

| sources | evaluation | / charges | load | upward local | downward | leaves |
| --- | --- | --- | --- | --- | --- | --- |
| charges | 14843.5 | 1.00 | 2.9 | 185.4 | 9878.3 | 4749.2 |
| dipoles | 16738.8 | 1.13 | 7.2 (2.48) | 305.9 (1.65) | 11493.2 (1.16) | 4905.6 (1.03) |
| charges and dipoles | 18626.1 | 1.25 | 11.4 (3.88) | 440.4 (2.38) | 12695.0 (1.29) | 5458.8 (1.15) |

- **Where it goes.** The dipole P2M (the upward local stage) costs 1.65–2.06× the charge P2M:
  `regular_grad` forms the values and three gradients per point. The leaves stage (P2P,
  L2P, M2P) costs 1.20–1.36× for dipoles: the dipole P2P kernel has about 1.4× the
  operations of the charge kernel (simd-p2p.md §4.8). The downward pass (M2M is upward;
  M2L, L2L) does not change on the cube; on the Plummer sphere it holds P2L (the X list),
  whose dipole form costs more (1.16×, 1.29× with charges).
- **Charges and dipoles** run the charge and the dipole operator one after the other on each
  source leaf (two P2P kernels per pair, two harmonic evaluations per P2M and P2L), so the
  leaves stage costs the sum of the two (1.68–2.24×). A fused kernel would share the
  distance and ρ (about 8 + 3k of 23 + 3k operations per pair) and change the order of the
  additions; left for a later task.
- **Load** (the charge load, now with the moments' scaling in f64 per component) is 3–11 ms
  at N = 10⁶, under 1 % of an evaluation.

## 3. The device: the host fallback

`dipole_cost --backend metal --precision f32`, the cube at N = 10⁶, f32, gradients on; the
evaluation as in §2, then one evaluation with `KindTiming::Synchronous` (a sync after every
level call), whose kind times give the share of P2M, P2L and P2P, the kinds that run on the
host fallback with dipoles (the stage columns of a Metal evaluation without syncs say
little, since the launches queue; the totals are wall times of `evaluate`).

One host thread for the fallback kinds:

| p | sources | evaluation (ms) | / charges | P2M, P2L, P2P / evaluation | ms per kind (synchronous) |
| --- | --- | --- | --- | --- | --- |
| 3 | charges | 21.8 | 1.00 | 19.5 % | P2M 0.98, M2M 3.93, M2L 4.51, L2L 3.39, L2P 0.39, P2P 5.13 |
| 3 | dipoles | 591.3 | 27.1 | 95.1 % | P2M 50.70, M2M 2.06, M2L 8.40, L2L 1.81, L2P 0.55, P2P 520.66 |
| 3 | charges and dipoles | 964.8 | 44.3 | 96.5 % | P2M 73.43, M2M 2.06, M2L 9.72, L2L 1.93, L2P 0.58, P2P 865.03 |
| 8 | charges | 56.2 | 1.00 | 9.7 % | P2M 1.05, M2M 1.90, M2L 37.91, L2L 2.19, L2P 0.76, P2P 4.99 |
| 8 | dipoles | 839.5 | 15.0 | 91.7 % | P2M 259.22, M2M 3.31, M2L 48.17, L2L 3.06, L2P 0.55, P2P 522.91 |
| 8 | charges and dipoles | 1340.2 | 23.9 | 94.2 % | P2M 392.40, M2M 6.45, M2L 46.80, L2L 2.90, L2P 0.86, P2P 862.15 |

Twelve host threads (`FmmBuilder::threads(12)`: the pool of the host-fallback kinds):

| p | sources | evaluation (ms) | / charges | P2M, P2L, P2P / evaluation | ms per kind (synchronous) |
| --- | --- | --- | --- | --- | --- |
| 3 | charges | 17.1 | 1.00 | 23.7 % | P2M 0.41, M2M 1.76, M2L 4.38, L2L 1.55, L2P 0.58, P2P 5.17 |
| 3 | dipoles | 69.0 | 4.03 | 69.7 % | P2M 7.53, M2M 1.78, M2L 9.69, L2L 1.89, L2P 0.58, P2P 49.08 |
| 3 | charges and dipoles | 103.0 | 6.02 | 77.4 % | P2M 8.10, M2M 1.78, M2L 9.65, L2L 1.74, L2P 0.54, P2P 77.96 |
| 8 | charges | 51.2 | 1.00 | 11.0 % | P2M 1.36, M2M 1.99, M2L 37.92, L2L 2.22, L2P 0.70, P2P 4.96 |
| 8 | dipoles | 126.2 | 2.47 | 55.4 % | P2M 26.37, M2M 4.80, M2L 39.55, L2L 4.34, L2P 0.71, P2P 48.40 |
| 8 | charges and dipoles | 166.3 | 3.25 | 63.9 % | P2M 37.37, M2M 4.73, M2L 44.91, L2L 3.19, L2P 0.73, P2P 77.47 |

(For charges the three kinds run on the device; their share is given for comparison.) The
host's P2P with a dipole kernel takes about 10× the device's charge P2P with twelve
threads and 100× with one; the fallback's transfers (the target output down and up for
P2P, the multipoles up after P2M) add to it. §7.4's condition for device kernels holds at
every p and thread count measured. CUDA was not measured (locust not used).

## 4. The SIMD dipole P2P

`P2pKernel::evaluate_dipoles` (`nd-fmm-simd`, `tests/p2p/dipoles.rs`): one source and one
target over seeded separations across the kernel domain (in f32 from r² = 2⁻⁸⁰, the dipole
range of CONVENTIONS §3.13), moments in [−1, 1)³; the potential against the reference's term
relative to |d| / r², each gradient component relative to |d| / r³, in u_T. Contract 32 u_T
and 176 u_T (simd-p2p.md §4.8, a first-order bound that adds worst cases).

| ISA | machine | f32 φ | f32 ∇φ | f64 φ | f64 ∇φ |
| --- | --- | --- | --- | --- | --- |
| NEON | M3 Max | 10.49 | 32.59 | 9.60 | 33.81 |
| AVX2 | Kathleen (Xeon Gold 6248) | 9.46 | 32.87 | 8.65 | 27.13 |
| AVX-512 | Kathleen | 9.46 | 32.87 | 8.70 | 32.52 |

(10⁶ pairs each, release, `--ignored`; the scalar path equals `nd_fmm_ref::p2p::p2p_dipoles`
bit for bit.) Sums against `direct_sum_dipoles` (up to 300 sources): at most 1.6e-7 (φ) and
9.0e-7 (∇φ) in f32, 3.2e-16 and 1.5e-15 in f64, relative to the term sums. Chunk and
target-position invariance and the independence of K hold bit for bit on every ISA.
Inlining (the NEON entry points, `fmm-simd/tools/inner_loops.awk`): no call in any loop; the
source loops of a K = 2 block are 53 instructions (φ) and 104 (φ and ∇φ) for two target
vectors, against the charge kernel's (f64, K = 4) 55 for φ and ∇φ per four.

## 5. The open question: device kernels for the dipole kinds

Measured above: the fallback is 55–97 % of a dipole evaluation on Metal. The design's rule
(§7.4) then calls for device kernels:

- **P2P** (the largest part: 49 of 69 ms at p = 3, 48 of 126 ms at p = 8 with twelve
  threads): a dipole form of `nd_fmm_kernels::p2p`, the kernel of §4.8 with the device's
  arithmetic rules (CONVENTIONS §3.13, "Device kernels"; the flushing threshold of the moments
  to derive), every layout, on the CPU runtime, Metal and CUDA.
- **P2M and P2L** (8–37 ms): the gradient ladder on the device (`nd_fmm_kernels::leaf`
  evaluates the harmonics per point; a dipole form needs `regular_grad` and
  `irregular_grad`, new code).
- The source store on the device with 6 or 7 values per point and the strengths uploaded
  (the charge load's scaling of the moments on the host, as today, then one upload).

**Decided** (the user, 2026-10-17): option (a), T8 merges with the host path and the device
fallback, and the device kernels are a follow-up task, T8d
([docs/phase6/T8d-dipoles-device.md](../../docs/phase6/T8d-dipoles-device.md)): the source
store with strengths, the device dipole P2P first, then P2M and P2L.

## 6. Checks

- `cargo fmt -- --check`; `cargo clippy -- -D warnings`; `cargo clippy --examples -- -D
  warnings`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo doc --no-deps`;
  `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings`; `cargo
  check -p nd-fmm-exec --features cuda`; `cargo doc -p nd-fmm-exec --no-deps --features cpu`:
  clean.
- `cargo test -p nd-fmm-ref`, `-p nd-fmm-math`: pass (the dipole fixture tests, the
  differences, the series, the panics; the harmonics fixtures at version 2).
- `nd-fmm-simd`: clippy (and `--target x86_64-apple-darwin`), the tests and the ignored
  release tests on the M3 Max (scalar, NEON) and on Kathleen (scalar, AVX2, AVX-512; job
  604594, every step passed).
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --test mpi_exec` (one rank, debug, 157 s),
  `--features cpu --release` (every scenario on the CPU runtime, 68 s), and
  `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks (every executable passed;
  `mpi_exec` 186 and 165 s).
- `tests/dipoles.rs` (release, ignored) at 1, 2 and 4 ranks: pass (§1).
- `RUST_MIN_STACK=8388608 cargo test --workspace --no-fail-fast` (debug, one rank): every
  executable passes.
- By hand, release: `tests/device_metal.rs` (`--features metal`, outside the sandbox; the
  dipole scenario on Metal, every kind on the fallback bit for bit the host, the default
  placement within 1e-5) and `tests/device_ranks.rs` (`--features cpu`) at 2 and 4 ranks:
  pass.
- Kathleen, 2 nodes in `test` (40 ranks per node): `correctness.sbatch` (job 604595,
  node-c11a-[107,112], 31 min): the MPI list at 2, 4 and 8 ranks, rank-count invariance at
  80 against 40, 20 and 10, and the `scaling` smoke run passed; `mpi_exec` at 80 ranks
  failed in **dipole sources**, f32: the test built the f32 one-rank reference under `Auto`,
  which resolves by ranks per node (40 there, 1 for the reference), so the two ran different
  M2L strategies. A test defect, not the FMM's: the reference now names the resolved
  strategy, as every other one-rank check does. Then `mpi.sbatch 80` (job 604598,
  node-c11a-[115-116], 11 min):
  every executable of the list passed at 80 ranks, `mpi_exec` 372 s.
- Not run: the full `--features metal --release -- --ignored` list beyond
  `device_metal`, and every CUDA test (`tests/device_cuda.rs` has the dipole scenario in
  f32 and f64; locust was not used).

## 7. Commands

```sh
cargo run --release -p nd-fmm-validate --example accuracy -- --sources dipoles
cargo run --release -p nd-fmm-validate --example accuracy -- --sources charges+dipoles
RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release --test dipoles -- --ignored --nocapture
mpirun -n 4 --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 \
    target/release/deps/dipoles-<hash> --ignored --nocapture --test-threads=1
cargo run --release -p nd-fmm-validate --example dipole_cost -- --n 1000000 --precision f64
cargo run --release -p nd-fmm-validate --example dipole_cost -- --n 1000000 --precision f32
cargo run --release -p nd-fmm-validate --example dipole_cost -- --n 1000000 --dist plummer --degrees 8
cargo run --release -p nd-fmm-validate --features metal --example dipole_cost -- \
    --n 1000000 --precision f32 --backend metal [--threads 12]   # outside the sandbox
cargo test -p nd-fmm-simd --release --test p2p dipole_terms -- --ignored --show-output
tools/kathleen/sync.sh && tools/kathleen/submit.sh check.sbatch simd
```


# Phase 6 T8d: the dipole kinds on the device

Phase 6, task T8d ([docs/phase6/T8d-dipoles-device.md](../../docs/phase6/T8d-dipoles-device.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §7.4 and §7.5 ("As
built in T8d"). Measured on 2026-10-17 and 18 from branch `tbetcke/phase8_t8d` (main at
dae7e15 plus this task's changes), release builds, rustc 1.99.0, every BLAS thread variable
at 1, on the Apple M3 Max unless stated; CUDA on locust's H100 (§T8d.4). Kathleen was not
used.

## T8d summary

1. **Built**, in the order of the brief (§T8d.1): the device's source store with the
   strengths (one upload of 3 or 4 values per source per evaluation, the moments scaled on
   the host with the charge load's bits); the dipole P2P in every layout of
   `nd_fmm_kernels::p2p` (separate kernels; the charge kernels unchanged); the dipole P2M
   and P2L in both layouts of `nd_fmm_kernels::leaf` (the gradient ladders on the device);
   the placement (T8's forced host fallback removed; `host_fallback` keeps any kind on the
   host as before).
2. **Contract** (CONVENTIONS §3.13, "Device kernels", "Dipoles", added in this task; no
   version bump): the host kernel's 32 u_T (φ̂) and 176 u_T (ĝ); where subnormals flush, for
   d_t = 0 or |d_t| r ≥ 2⁻⁹⁶ (every |d̂| ≥ 2⁻⁵⁵ on the f32 range). Measured at most 9.4 u_T
   and 31.2 u_T on the CPU runtime, Metal and CUDA; the coincident-pair rule holds on every
   adversarial and random pair of `spikes/device-arith` with moments, on all three (§T8d.2).
3. **Accuracy** (§T8d.2): every test of the brief passes on the CPU runtime, Metal and CUDA:
   the kernels against `nd_fmm_ref` and the host kernel, P2M and P2L to p = 20 in f64, the
   FMM with every kind on the device within the device bounds of the host (1e-12, 1e-5) and
   every kind on the fallback the host bit for bit with the transfers of the formula, and
   the dipole gate with every kind on the device (CPU runtime f64, Metal f32, CUDA both)
   within the device-against-host bounds: the device's error 0.9995–1.0002 of the host's.
4. **Cost on Metal** (§T8d.3): a dipole evaluation now costs **1.07–1.23× a charge
   evaluation** (charges and dipoles 1.21–1.64×), at one host thread and at twelve, against
   15–28× and 2.4–4.3× with T8's fallback, which the same binary reproduces (within 5 % of
   T8's numbers). P2M, P2L and P2P are 11–25 % of a dipole evaluation (charges: 10–24 %).
5. **CUDA timings**: **not taken**. The H100 was in use by another user's job (a vLLM
   engine, 79.6 GB, 100 % utilisation) at every check from 23:07 on 2026-10-17 to 01:16 on
   2026-10-18, so by tools/gh200/README.md, "GPU etiquette", no timing ran; every CUDA
   correctness run did (§T8d.4). `dipole_cost --backend cuda` is the open measurement.
6. **For sign-off**: the §3.13 addition; the dipole leaf cells of `tests/operator/
   device_leaf.rs` report their ratio to the host operator's error instead of asserting the
   charge operators' "within twice the host" (Metal: 2.12 in one cell, §T8d.2); no fused
   charge-and-dipole P2P kernel (§T8d.1); with dipoles the P2P layout under `Auto` is the
   static rule (§T8d.1). Fixed on the way: `tests/device_tune.rs` hard-coded convention
   version 2 as "another" version, which T8's bump made the current one (the test runs only
   with a device feature, which T8 did not run it with).

## T8d.1 What was built

- **The source store with strengths** (`nd-fmm-exec`'s `device` and `fmm`). The device's
  source store keeps the chunk layout of CONVENTIONS §3.13 (6 or 7 values per point) and
  now takes the strengths. `Fmm` forwards w = 3 or 4 strengths per source, writes the host's
  chunks where a kind runs on the host (from the forwarded values, as before), scales the
  moments of the forwarded buffer into their leaf's units on its pool (`scale_moments`,
  `T::from_f64(d.to_f64() / r)`: the charge load's bits), writes the chunks of the leaves the
  source exchange sends from that buffer, and hands it to `begin_evaluation`, which uploads
  it without a copy and scatters it by the strength slots (`DeviceViews::charge_slots`,
  generalised: per source its charge, then its moment's components). One upload of
  w N_s s bytes and one launch per evaluation, as for charges with w = m;
  `tests/device_common::expected_evaluation` takes w (and 3 + w per ghost point).
- **P2P** (`nd_fmm_kernels::p2p`). `SourceKind` (charges, dipoles, both) in `P2pInputs`; two
  new kernels, `p2p_dipole_tiled_kernel` (the cube and plane layouts) and
  `p2p_dipole_vector_kernel` (the CPU layout), with the rows, blocks, tiles and order of the
  charge kernels, per source leaf first its charges (the charge formulation, operation for
  operation) and then its moments: ŷ = fma(r̂, u_s, ĉ), d_t = r̂ d̂ (exact), r² and ρ by the
  charge rules, ρ² = ρ ρ, ρ³ = ρ ρ², s = fma(d₂, w₂, fma(d₁, w₁, d₀ w₀)), φ̂ = fma(s, ρ³, φ̂),
  a = (3 s) ρ², ĝₖ = fma(ρ³, fma(−a, wₖ, dₖ), ĝₖ): never r⁻⁵. The GPU tile holds 6 values per
  source (`P2pLayout::check_sources`). The charge kernels and their launches do not change,
  so neither does the charge path (every charge test passes as before on every backend).
- **P2M and P2L** (`nd_fmm_kernels::leaf`). `SourceKind` in `SourceInputs`; comptime flags in
  the existing kernels (with charges alone the operations of before), the harmonics to
  degree p (P2L: p + 1, as M2P's gradient), `ladder`'s gradient contracted per slot as
  `nd_fmm_ref::leaf::{p2m_dipoles, p2l_dipoles}` ((d̂₀ ∂ₓX + d̂₁ ∂ᵧX) + d̂₂ ∂_zX, conjugated:
  `directional`, `stage_dipole`, `add_dipole`), P2L with d̂ · 2^(l_t − l_s); each entry's
  charges before its moments. The cube layout stages conj(d̂ · ∇X) in the shared memory of
  the charges' harmonics, so its shared memory does not grow.
- **Placement and tuning**: every kind on the device by default with dipoles. The P2P layout
  under `DeviceP2pLayout::Auto` is the static rule with dipoles, neither tuned nor taken from
  the cache: the tuning cache keys hold no source type, and their P2P decisions are the
  charge kernel's.
- **No fused charge-and-dipole kernel** (the brief: only if measured worth it). On Metal at
  p = 3 the P2P of charges and dipoles took 10.2–10.4 ms against 6.2–6.3 ms for dipoles and
  5.1–5.2 ms for charges (§T8d.3); by the operation counts of simd-p2p.md §4.2 and §4.8
  (*model*) a fused kernel would save at most the charge stage's distance and ρ, about half
  of its 5 ms, some 7 % of a charge-and-dipole evaluation. The order is that of the host:
  per source leaf the charges, then the moments.
- **Tests**: `nd-fmm-kernels`' `tests/kernels/p2p/dipoles.rs` and `tests/kernels/leaf/
  dipoles.rs` (8 runtime tests per backend), the dipole cases of `nd-fmm-exec`'s
  `tests/operator/device_p2p.rs` and `device_leaf.rs`, `device_common::check_dipoles` with
  every kind on the device (also in `tests/device_ranks.rs` on Metal and CUDA), the device
  repeat of the gate in `tests/dipoles.rs`, and the section `dipoles` of
  `spikes/device-arith` (the production kernel on the §3.13 pairs).
- **`dipole_cost --dipole-fallback`** (`nd-fmm-validate`): keeps the named dipole kinds on
  the host fallback, for the source types with dipoles only, so that one binary measures
  T8's placement, P2P alone on the device and every kind on the device.

## T8d.2 Accuracy

**The P2P kernel** (`tests/kernels/p2p/dipoles.rs`): pair terms over the kernel domain
(moments from 2⁻⁴⁰ to 1 in magnitude, every 64th zero, every 16th with a zero component;
the source in the leaf itself or in a same-level, coarser or finer neighbour, the moment
mapped by r̂ = 1, 2 or ½; in f32 from r² = 2⁻⁷⁹), against `p2p_dipoles` in the same
precision, in u_T relative to |d_t| / r² (φ̂) and |d_t| / r³ (ĝ); 32,768 pairs per layout and
source kind, every layout the same:

| backend | precision | φ̂ (bound 32) | ĝ (bound 176) |
| --- | --- | --- | --- |
| CPU runtime (M3 Max) | f32 | 9.40 | 27.57 |
| CPU runtime (M3 Max) | f64 | 9.01 | 28.96 |
| Metal | f32 | 9.40 | 21.71 |
| CUDA | f32 | 9.40 | 27.57 |
| CUDA | f64 | 9.01 | 28.96 |

Sums against `direct_sum` and `direct_sum_dipoles` (FMM-shaped rows up to 4,096 sources),
coincident pairs, and the domain ends (r² = 2⁻¹⁰⁸ in f64, the ends of the f32 range,
|d| r = 2⁻⁹⁰) pass on every layout of every backend; a row split per leaf, permuted target
leaves and repeated launches are bit for bit, and charges and dipoles in one launch equal,
leaf by leaf, a launch of the charges followed by one of the moments, bit for bit on every
layout of the CPU runtime, Metal and CUDA. At the absolute frames (`nd-fmm-exec`'s
`tests/operator/device_p2p.rs`, levels 2, 9 and 16, every near pair, relative to the term
sums): against `p2p_dipoles` within 3.5e-16 / 8.2e-16 (f64, φ / ∇φ) and 1.6e-7 / 3.7e-7
(f32), against the host kernel (`P2pKernel::evaluate_dipoles`) within 2.2e-16 / 0 (f64) and
9.6e-8 / 0 (f32) on the CPU runtime and 1.7e-7 / 5.0e-7 on Metal.

**The coincident-pair rule** (`spikes/device-arith`, section `dipoles`: the production
kernel in the backend's default layout, one row per pair, moments uniform in [−1, 1)³;
coincident pairs must add exactly zero, the others be finite and within the contract):

| backend | precision | pairs | count | coincident | coincident ≠ 0 | non-finite | φ̂ (u_T) | ĝ (u_T) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU runtime | f32 | adversarial | 209,303 | 30,403 | 0 | 0 | 7.43 | 23.47 |
| CPU runtime | f32 | random | 216,000 | 3,526 | 0 | 0 | 9.18 | 28.58 |
| CPU runtime | f64 | adversarial | 439,192 | 484 | 0 | 0 | 7.07 | 24.96 |
| CPU runtime | f64 | random | 216,000 | 3,526 | 0 | 0 | 9.43 | 31.20 |
| Metal | f32 | adversarial | 209,303 | 30,403 | 0 | 0 | 7.43 | 23.47 |
| Metal | f32 | random | 216,000 | 3,526 | 0 | 0 | 8.09 | 26.23 |
| CUDA | f32 | adversarial | 209,303 | 30,403 | 0 | 0 | 7.43 | 23.47 |
| CUDA | f32 | random | 216,000 | 3,526 | 0 | 0 | 9.18 | 28.58 |
| CUDA | f64 | adversarial | 439,192 | 484 | 0 | 0 | 7.07 | 24.96 |
| CUDA | f64 | random | 216,000 | 3,526 | 0 | 0 | 9.43 | 31.20 |

32,038 of the f32 adversarial pairs lie below the f32 dipole range (r² < 2⁻⁷⁹) and are not
held to the contract; every coincident pair among them adds zero.

**P2M and P2L** (`tests/operator/device_leaf.rs` at the absolute frames, levels 2, 9 and 16,
f64 p ∈ {0, 1, 4, 8, 12, 20}, f32 p ≤ 8, both source kinds; per degree in the §3.8 weighting
relative to the term magnitudes, the worst over the levels and kinds; the default layout,
and on the CPU runtime also the cube layout):

| backend | precision | P2M | P2L | bound |
| --- | --- | --- | --- | --- |
| CPU runtime | f64 | 7.4e-16 (6.7 u_T) | 1.1e-15 (9.7 u_T) | 1e-13 |
| CPU runtime | f32 | 1.2e-7 | 3.7e-7 | 1e-5 |
| Metal | f32 | 1.2e-7 | 4.7e-7 | 1e-5 |
| CUDA | f64 | 7.4e-16 (6.7 u_T) | 1.1e-15 (9.7 u_T) | 1e-13 |
| CUDA | f32 | 1.2e-7 | 3.7e-7 | 1e-5 |

In f64 the device is within max(2 × the host operator's error, (n + 2) u_T) at degree n,
the charge operators' rule. In f32 the host operator's error is reported beside the
device's, not asserted (`device_leaf.rs`, module documentation): on Metal one cell, P2L
dipoles at level 16, came to 2.12 times the host operator's (4.3e-7 against 2.0e-7; 1.85 on
the CPU runtime), a ratio of two maxima of rounding errors 23 times below the bound; the
other dipole cells lie at 0.46–1.69. The kernel tests (`tests/kernels/leaf/dipoles.rs`,
levels 2, 9 and 16 at the canonical frames, p ≤ 8, p = 20 in f64 in the ignored sweep) pass
on every layout of every backend, with repeated launches, rows split per entry and the order
of charges and moments bit for bit.

**The FMM** (`device_common::check_dipoles` in `tests/mpi_exec.rs`, `tests/device_metal.rs`,
`tests/device_cuda.rs` and `tests/device_ranks.rs`): for `Dipoles` and `ChargesAndDipoles`,
every kind on the host fallback is the host bit for bit, and every kind on the device is
within the device bound of the host, with two evaluations bit for bit, blocking and
overlapped bit for bit, and the transfers of the formula with w strengths per source:

| run | backend | precision, p | φ, ∇φ from the host (relative L2) | bound |
| --- | --- | --- | --- | --- |
| `mpi_exec`, cube and Plummer, N = 2,000, 1 rank | CPU runtime | f64, 6 | ≤ 1.9e-16, ≤ 3.7e-18 | 1e-12 |
| `mpi_exec`, the cube, N = 1,000 | CPU runtime | f32, 4 | 5.5e-8, ≤ 2.1e-9 | 1e-5 |
| `device_metal`, the cube, N = 2,000 | Metal | f32, 4 | within | 1e-5 |
| `device_cuda`, the cube, N = 2,000 | CUDA | f32, 4; f64, 6 | within | 1e-5; 1e-12 |
| `device_ranks`, the cube, N = 4,000, 2 and 4 ranks | CPU runtime | f64, 6 | ≤ 2.3e-16, ≤ 4.7e-18 | 1e-12 |
| `device_ranks`, the cube, N = 4,000, 2 ranks | Metal | f32, 4 | 1.6e-7, 1.5e-7 | 1e-5 |
| `device_ranks`, the cube, N = 4,000, 2 and 4 ranks | CUDA | f32, 4; f64, 6 | ≤ 1.4e-7, ≤ 3.6e-9; ≤ 2.2e-16, ≤ 5.4e-18 | 1e-5; 1e-12 |

`mpi_exec` passed with the CPU runtime at 1, 2 and 4 ranks (on several ranks the sent leaves
take their scaled moments from the upload).

**The dipole gate** (`tests/dipoles.rs`, 1 rank; the gate's problems, eight vectors): every
host case as in T8 (§1, the same errors), and every case repeated with every kind on the
device:

| backend | precision | cases | device error / host error | device φ from the host's |
| --- | --- | --- | --- | --- |
| CPU runtime | f64 | cube and Plummer, both kinds, p = 3, 8, 12 | 1.0000 | 1.9e-17 – 6.8e-17 |
| Metal | f32 | cube and Plummer, both kinds, p = 3, 8 | 0.9995–1.0000 | 7.8e-8 – 1.3e-7 |
| CUDA | f32 | cube and Plummer, both kinds, p = 3, 8 | 0.9999–1.0002 | 2.7e-8 – 7.4e-8 |
| CUDA | f64 | cube and Plummer, both kinds, p = 3, 8, 12 | 1.0000 | 2.4e-17 – 6.8e-17 |

Bounds: the error within 0.1 % (f64) and 5 % (f32) of the host's, φ within 1e-12 and 1e-5
(docs/phase4/README.md, "Accuracy measures", FMM).

## T8d.3 Cost on Metal

`dipole_cost --backend metal --precision f32 --n 1000000` (the cube, gradients, the default
tree), the three placements of `--dipole-fallback` with one and twelve host threads; the
charge rows keep every kind on the device in all of them. Median wall time of `evaluate`
over five evaluations after a warm-up, ms, and the ratio to the charge evaluation of the
same run; then the share of P2M, P2L and P2P in one evaluation with synchronous kind
timings. Load average 3–7 on the M3 Max during the runs (system daemons; no other job of
ours), the GPU otherwise idle.

| threads | p | sources | T8: P2M, P2L, P2P on the fallback | P2P on the device | every kind on the device |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | charges | 21.0 | 20.8 | 21.0 |
| 1 | 3 | dipoles | 595.0 (28.35) | 81.1 (3.89) | 25.3 (1.20) |
| 1 | 3 | charges and dipoles | 965.7 (46.01) | 116.5 (5.59) | 34.4 (1.64) |
| 1 | 8 | charges | 56.0 | 56.0 | 56.0 |
| 1 | 8 | dipoles | 845.4 (15.08) | 324.8 (5.80) | 60.1 (1.07) |
| 1 | 8 | charges and dipoles | 1326.4 (23.67) | 473.4 (8.46) | 70.1 (1.25) |
| 12 | 3 | charges | 16.8 | 17.0 | 16.2 |
| 12 | 3 | dipoles | 72.0 (4.29) | 26.9 (1.58) | 20.0 (1.23) |
| 12 | 3 | charges and dipoles | 107.1 (6.38) | 35.4 (2.08) | 26.2 (1.61) |
| 12 | 8 | charges | 51.4 | 51.2 | 51.2 |
| 12 | 8 | dipoles | 122.7 (2.39) | 79.3 (1.55) | 54.8 (1.07) |
| 12 | 8 | charges and dipoles | 168.2 (3.27) | 97.5 (1.90) | 61.8 (1.21) |

T8's numbers for the fallback (§3: 591.3, 964.8, 839.5, 1340.2 ms at one thread; 69.0, 103.0,
126.2, 166.3 at twelve) are reproduced within 0.1–4.3 %. Per kind, ms, with every kind on the
device (synchronous kind timings, one thread; twelve threads the same to ±0.6 ms):

| p | sources | P2M | M2M | M2L | L2L | L2P | P2P | P2M, P2L, P2P / evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | charges | 0.54 | 1.81 | 4.30 | 1.85 | 0.37 | 5.15 | 21.1 % |
| 3 | dipoles | 1.00 | 4.36 | 6.11 | 4.43 | 0.55 | 6.33 | 18.8 % |
| 3 | charges and dipoles | 1.05 | 3.85 | 4.45 | 2.69 | 0.55 | 10.39 | 25.9 % |
| 8 | charges | 1.43 | 2.13 | 38.05 | 2.24 | 0.73 | 4.97 | 10.1 % |
| 8 | dipoles | 1.21 | 1.91 | 38.02 | 2.41 | 0.75 | 6.15 | 11.1 % |
| 8 | charges and dipoles | 1.73 | 2.00 | 38.08 | 2.33 | 0.70 | 10.17 | 15.7 % |

(P2L has no entry on this uniform tree. M2M and L2L at p = 3 vary run to run by ±2 ms with
synchronous kind timings; see the twelve-thread run, 1.98 and 1.51 ms for dipoles.) The
dipole P2P costs 1.2× the charge P2P on Metal, close to the 1.4× of the operation counts
(simd-p2p.md §4.8), and the dipole P2M about the charge P2M. The load stage (forwarding and
scaling the moments, the upload) took 2.9–3.4 ms for dipoles and 4.6–8.2 ms for charges and
dipoles, against 1.5 ms for charges.

Step by step (the brief's order, each measured before the next): T8's placement; P2P on the
device (P2M and P2L still on the fallback): 3.9–8.5× a charge evaluation at one thread,
1.55–2.08× at twelve, P2M (51–397 ms at one thread) now the larger part; every kind on the
device: 1.07–1.64×.

## T8d.4 CUDA (locust)

Correctness only, on locust's H100 (CUDA 12.6, CubeCL 0.11.0 through LLVM NVPTX), from
the synced working tree, 2026-10-17 23:07 to 2026-10-18 00:59. The GPU was shared with
another user's vLLM engine (79.6 GB of 96, 100 % utilisation, running before the first
check and after the last); correctness runs that need little memory may share the GPU
(tools/gh200/README.md, "GPU etiquette"), timings may not, so none was taken: the cost of a
dipole evaluation on CUDA, against T8 (which had no CUDA numbers either) and against a
charge evaluation, is open. The command, for when the GPU is free:
`tools/gh200/remote.sh 'cargo run --release -p nd-fmm-validate --features cuda --example
dipole_cost -- --backend cuda --precision f32 [--dipole-fallback …]'` (and f64).

What ran and passed:

- `nd-fmm-kernels`, `--features cpu,cuda -- --include-ignored`: 159 tests (the CPU
  runtime on Grace and CUDA, f32 and f64), the dipole ones included: P2P pair terms
  9.40 / 27.57 u_T (f32) and 9.01 / 28.96 u_T (f64), sums, coincident pairs, domain ends,
  invariance and the order bit for bit on every layout; P2M and P2L on levels 2, 9, 16.
- `spikes/device-arith --sections dipoles` on CUDA: the coincident-pair rule on all pairs
  (§T8d.2).
- `nd-fmm-exec`, `--features cuda --release -- --ignored`: every executable, with
  `tests/device_cuda.rs` (the dipole scenario, f32 p = 4 and f64 p = 6), `tests/dipoles.rs`
  (the gate, CUDA in f32 and f64, §T8d.2), the operator tests (dipole P2P and P2M, P2L at the
  absolute frames, p ≤ 20 in f64), `device_fmm`, `accuracy`, `adaptive`; and without
  `--ignored` every executable (after the `device_tune` fix).
- `tests/device_ranks.rs` at 2 and 4 ranks (`mpirun`, the ranks sharing the H100) with the
  CUDA dipole scenario (§T8d.2).
- `cargo clippy --all-targets --features cpu,cuda -- -D warnings` for `nd-fmm-kernels` and
  `nd-fmm-exec`.

## T8d.5 Checks

On the M3 Max unless stated; every check passed.

- Root: `cargo fmt -- --check`; `cargo clippy -- -D warnings`; `cargo clippy --examples --
  -D warnings`; `RUST_MIN_STACK=8388608 cargo test` (`mpi_exec` 140 s in debug, T8 157 s);
  `cargo doc --no-deps`. Strict: `cargo clippy --workspace --all-targets -- -D warnings`;
  `RUST_MIN_STACK=8388608 cargo test --workspace`.
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks (loopback flags): every executable
  (`mpi_exec` 176 and 152 s).
- `nd-fmm-kernels`: clippy without features and with `cpu,metal`; `cargo check --features
  cuda`; `cargo test` (no features) and `--features cpu --release` (83 tests, 54 s) and its
  ignored p = 20 sweep (82 s); `--features metal -- --ignored` outside the sandbox (63
  tests); `cargo doc --no-deps`; on locust with `cpu,cuda` (§T8d.4).
- `nd-fmm-exec`: clippy without features and with `cpu,metal`; `cargo check --features
  cuda`; `cargo doc --no-deps --features cpu`; `--features cpu --release` and its
  `-- --ignored` (every gate, `device_fmm`, `device_ranks`, the dipole gate on the CPU
  runtime, `multi_rank`); `mpi_exec` (`--features cpu --release`) at 1, 2 and 4 ranks;
  `device_ranks` and `tests/dipoles.rs` (`--features cpu`) at 2 and 4 ranks; `--features
  metal --release -- --ignored` outside the sandbox (every executable; the operator tests
  rerun after the dipole leaf cells' change), `device_ranks` on Metal at 2 ranks; on locust
  with CUDA (§T8d.4).
- `nd-fmm-validate`: clippy with `metal`, the `dipole_cost` runs of §T8d.3; the spike:
  clippy with `cpu,metal`, `--sections dipoles` on the CPU runtime, Metal and CUDA.
- Not run: CUDA timings (§T8d.4); Kathleen (no x86 code changed: the host kernels are
  untouched).

## T8d.6 Commands

```sh
cargo test -p nd-fmm-kernels --features cpu --release -- --show-output
cargo test -p nd-fmm-kernels --release --features metal -- --ignored --show-output   # outside the sandbox
RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release [-- --ignored]
RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features metal --release -- --ignored   # outside the sandbox
cargo run --release -p nd-fmm-spike-device-arith --features cpu,metal -- --sections dipoles
cargo run --release -p nd-fmm-validate --features metal --example dipole_cost -- \
    --n 1000000 --precision f32 --backend metal --threads 1 --dipole-fallback p2m,p2l,p2p
#   … --dipole-fallback p2m,p2l, … --dipole-fallback none, and --threads 12
tools/gh200/sync.sh && tools/gh200/remote.sh '<the CUDA commands of fmm-kernels/CLAUDE.md and fmm-exec/CLAUDE.md>'
```
