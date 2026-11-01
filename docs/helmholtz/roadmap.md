# Helmholtz FMM roadmap

The tasks of `docs/design/helmholtz-fmm-proposal.md` (draft 0.8), Section 10, in the order
`/next-task` takes them. The proposal is the full record: each task's *Done when* condenses
its acceptance criterion in the §10 table, and where the two differ the proposal wins.
Progress is logged in `docs/helmholtz/progress.md`; each task's plan is
`docs/helmholtz/tasks/<ID>.md`; each milestone's gate report is
`docs/helmholtz/gates/<milestone>.md`.

Status: **not started (2026-10-26).** First: D0 and D9, then CHR.2 and CHR.4.

## How `/next-task` reads this file

- `- [ ]` is open, `- [x]` done. The tick is part of the task's own pull request, so a task
  counts as done once that PR is merged into `main`.
- Work in flight is not marked here. A task with a branch `tbetcke/h-<id>` or an open PR is
  in progress: a **draft** PR waits for machine runs, a sign-off or an open question (its
  plan file says which); a **ready** PR waits for the user's merge.
- A task is **runnable** when it is open, has no branch or open PR, and everything in its
  *Needs* is done: the tasks ticked on `main`, the decisions signed off in the table below.
  Every task also needs **D0**.
- `/next-task` takes the first runnable task of the **priority list** below, in its order.
  If the first open task of the list waits only for decisions, it asks the user for them
  first. If no task of the list is runnable (the rest wait for a merge, a machine run or a
  sign-off), it says what the list waits for and stops; it takes a task off the list only
  when the user names one. With the list empty or all ticked, it takes the first runnable
  task in the order of this file, as before.
- A task too big for one reviewable PR is split here into lettered sub-tasks (CH3.1a,
  CH3.1b), each with its own *Done when* and *Needs*, before it starts.
- Each milestone is mirrored as a GitHub issue (labels `enhancement`, `helmholtz`), ticked
  and closed from this file on every push to `main` (tools/roadmap/README.md). Change this
  file, never the issues; a task's PR ends with `Part of #<n>`, its milestone's issue.
- *Machines* names the machines its runs need besides the development Mac (the M6 since
  2026-10-31, the M3 Max before). *Sign-off* marks a task whose PR ends with questions for
  the user; the tasks that need it wait for the answer.

## Priority list: the shortest path to the FMM3D comparison

Set by the user on 2026-10-28: the shortest path to CH3.6, the first comparison of the
Helmholtz FMM with FMM3D (one Kathleen node; RCR on every level), with the vectorised
Helmholtz P2P in place for its times. The list holds CH3.6 and exactly the tasks in its
*Needs*, transitively, that are not yet done. It changes the order, plus one *Needs*: CH3.6
also needs CH3.4 (the user, 2026-10-28), which brings in CH2.4. No other task or gate
changes. Ordered so that a long task's pending run or merge leaves the next one runnable
(CH1.4's Kathleen run beside CH0.7; CH2.0's benchmarks beside CH2.4 and CH3.0; CH3.1 beside
CH3.4). CH2.0 and CH2.4 both time Laplace on the three machines, so their benchmark runs
take turns, never overlap.

1. CH1.4 FMM3D convention check and fixtures (*Machines:* Kathleen)
2. CH0.7 S7: truncation under realistic geometry
3. CH1.1 CONVENTIONS §3.15 (needs **D5**; *Sign-off:* §3.15)
4. CH1.2 `nd-fmm-math` (needs **D2**): CH1.2a, CH1.2b, CH1.2c
5. CH1.3 `nd-fmm-ref`: CH1.3a, CH1.3b, CH1.3c (CH1.3d after CH0.6, off the list)
6. CH2.0 Laplace's real host drivers made generic (*Machines:* locust, Kathleen)
7. CH2.4 Yukawa P2P in `nd-fmm-simd`, the loop skeleton: CH2.4a, CH2.4b (*Machines:*
   locust, Kathleen)
8. CH3.0 Builder: the leaf rule and complex densities (needs **D1**): CH3.0a, CH3.0b, CH3.0c
9. CH3.5 The shared drivers extended to complex expansions (*Machines:* locust, Kathleen)
10. CH3.1 Complex planes and RCR on every level: CH3.1a, CH3.1b, CH3.1c
11. CH3.4 Helmholtz P2P, real and complex k: CH3.4a, CH3.4b, CH3.4c (*Machines:* locust,
    Kathleen)
12. CH3.3 Calibration of p_l
13. CH3.6 Gate: RCR on every level against FMM3D (*Machines:* Kathleen)

Off the list until CH3.6 is done, in file order afterwards: CH0.6, CH0.8b, CH0.9, CH0.4,
CH0.5 and H0.G; CHR.3 and HR.G; CH1.3d and H1.G; CH2.1, CH2.2, CH2.3, CH2.5 and H2.G;
CH3.2 and H3.G; H6, H4, H5, H7, H8. CH3.6 times the FMM with CH3.4's vectorised kernel as
the Helmholtz operator's P2P: CH3.4 puts it there if CH3.1 is merged first (as ordered),
else CH3.1 uses it.

## Decisions

The proposal's Section 12, plus D0. A decision is recorded here when the user takes it:
`signed off <YYYY-MM-DD>: <the answer as given>`.

| ID | Decision | Proposal's recommendation | Prepared by | Needed by | Status |
| --- | --- | --- | --- | --- | --- |
| D0 | Draft 0.8 of the proposal as the plan of record | as written; later changes by sign-off | review of draft 0.7 | every task | signed off 2026-10-26: as written |
| D1 | Target kD range and tolerances | for example kD ≤ 500 at 10⁻³ to 10⁻⁸ (proposal §12) | – | CH0.8b, CH3.0 | signed off 2026-10-31: kD ≤ 500 at 10⁻³ to 10⁻⁸, as recommended |
| D2 | Placement: modules in the layer crates (A) or separate crates (B) | A (§6.1) | – | CH1.2 | signed off 2026-10-29: A, modules in the layer crates, as recommended |
| D3 | f32 scope | f64 only until H7; then f32 for low-frequency configurations that pass an accuracy contract; high-frequency levels in f32 only if S6 finds stable classes | CH0.6 | CH7.1 | open |
| D4 | Yukawa: its own operator type, or the wavenumber kind as a type parameter | one `HelmholtzOperator<T, K>` if CH2.1 shows no cost to the complex path | CH2.1 | CH2.2 | open |
| D5 | Source types of the first release | real and complex charges, potentials and gradients; dipoles once C6.4 exists for Laplace (note: Laplace dipoles were merged in Phase 6 T8 and T8d, so the condition is met) | – | CH1.1 | signed off 2026-10-29: charges now, dipoles later; §3.15 fixes real and complex charges with potentials and gradients, and dipoles get a later, separate task |
| D6 | Order of H4 and H6 | H6 first | – | CH6.1, CH4.1 | open |
| D7 | Host back ends of the φ step | the product as default and reference; FFTW behind the optional `fftw` feature; rustfft if S9 finds it deterministic per box and faster, decided per machine | CH0.9 | CHR.3, CH4.6 | open |
| D8 | Exponential M2L as a default on low-frequency levels | after CH6.2 | CH6.2 | CH7.5 | open |
| D9 | What "within noise" means for Laplace's benchmarks (constraint 1) | the run-to-run spread of the Laplace baseline, repeated on each machine before the change, per benchmark; plus the C3S.4 throughput gate | – | CHR.2, CH2.0 | signed off 2026-10-26: as recommended, the measured spread of the baseline per benchmark and machine, plus the C3S.4 gate. Clarified 2026-10-26 (CHR.2): one-sided, slower than `main` by more than the spread fails, faster passes when the outputs are bit for bit identical; from CH2.0 on five runs per side in the order A B B A A B B A A B, the spread max − min of `main`'s five run medians, never pooled with the branch's. Amended 2026-10-31 (CH2.0, the user: "as recommended"): a failing row may be repeated once and the repeat counts, with both results reported. Amended 2026-10-31 (the machine move, the user: "Amend: M6 replaces M3 Max"): the M6 (`tools/m6/machine.md`) replaces the M3 Max as a machine of record, its C3S.4 reference `spikes/p2p-simd/results-m6.md` |
| D10 | cubek-fft on the device | after S5: adopt, contribute upstream first, or keep the transform as a GEMM | CH0.5 | CH7.4 | open |
| D11 | Maximum leaf width c_leaf | from S8: the largest c_leaf with P2P and leaf costs within 10 % of their best on W1 and W3 (expected 0.5 to 1) | CH0.8a | CH3.0 | signed off 2026-10-28: as recommended (CH0.8a report §9, question 1), c_leaf = 1 by the revised rule, the largest c_leaf with no high-frequency leaf on any workload at 10⁻⁶ and 10⁻⁸, replacing the rule of this row, which is degenerate; a first-pass value that CH0.8b may revise with S6's constants and S7's p |
| D12 | The reference workloads | the set of §9.1 | CH0.8a | CH2.1 | signed off 2026-10-28: as recommended (CH0.8a report §9, question 2), the workloads of §9.1 unchanged, with W3 at 10⁶ points and kD = 500 named as the node-budget and leaf-rule-cost case |
| D13 | RLST's `fftw_mkl` back end | – | – | – | withdrawn (RLST ≥ 0.11 has none) |
| D14 | Ownership and schedule of Phase HR | – | – | – | closed (RLST 0.11.0 and 0.12.0) |

## HR: RLST's FFTW interface and its adoption

**Gate:** nd-fmm on RLST 0.12 with Laplace unchanged; the FFTW of every machine recorded;
the optional FFTW wrapper of the φ step behind its feature.

- [x] **CHR.1 RLST's FFTW rework.** Done in RLST 0.11.0 (2026-10-25) and 0.12.0
  (2026-10-26).
- [x] **CHR.2 Move to RLST 0.12.** Report: `fmm-validate/results/helmholtz-chr-2.md`. The workspace pin and `octree/Cargo.toml`'s own pin to
  0.12.0 together; green-kernels' `[patch]` left on 0.10.2 (it uses `rlst::simd`, removed in
  0.12); `CLAUDE.md`, `fmm-plan/CLAUDE.md` and `docs/design/workspace-structure.md` updated.
  *Done when:* the default members build, lint, test and document; the MPI tests and the
  weekly examples pass; `cargo check --workspace --all-targets` builds the spikes;
  `cargo tree -e normal` lists no fftw-sys; a cargo-deny licence check of the default set is
  in CI; CI's and locust's `cmake` installs checked; `cargo upgrades` no longer reports
  `rlst`; Laplace bit for bit and its benchmarks within noise (D9). Fallback: stay on 0.10.2.
  *Needs:* D9. *Machines:* locust, Kathleen.
- [x] **CHR.4 FFTW on the machines.** Report: `fmm-validate/results/helmholtz-chr-4.md`;
  the probe `tools/fftw/probe.sh`. *Done when:* `pkg-config --modversion fftw3 fftw3f`
  and `rlst::fftw::version()` recorded for the M3 Max, locust (spack's FFTW, not the RPM's
  3.3.8), Kathleen and CI, in `tools/*/machine.md`; RLST 0.12's own
  `cargo test --features fftw_system` passes on each, `fftw_determinism` included.
  Fallback: a machine without FFTW in both precisions uses the product. *Needs:* –.
  *Machines:* locust, Kathleen.
- [ ] **CHR.3 FFTW wrapper of the φ step.** In `nd-fmm-exec` behind
  `fftw = ["rlst/fftw_system"]`: one plan per (level, N_φ), interleaved per-box scratch,
  `Rigor::Estimate`, fingerprints all-gathered at build. *Done when:* against the product to
  round-off; a box's bits the same across batch positions, batch sizes and thread counts;
  ranks with differing fingerprints both use the product; the feature off by default; a CI
  job with the feature (`pkg-config`, the FFTW version report, cargo-deny's entry for
  fftw-sys). *Needs:* CHR.2, CH0.9, D7.
- [ ] **HR.G Gate.** `/milestone-gate HR`. *Needs:* every task above.

## H0: Spikes

**Gate:** S1–S9 answered, each with a `SPIKE_REPORT.md`; the decisions they prepare (D3,
D7, D10, D11, D12) put to the user.

- [x] **CH0.1 S1: rotation blocks at large degree.** Report:
  `spikes/rotation-blocks/SPIKE_REPORT.md`. The Gumerov–Duraiswami (2014) recursion
  in the orthonormal real basis. *Done when:* orthogonal and in agreement with an
  arbitrary-precision reference to p = 1000; in agreement with N-conjugated
  `nd_fmm_math::rotation::blocks` at p ≤ 30; the degree up to which conjugation stays
  accurate; the relation of the β and π − β blocks; time per block set. *Needs:* –.
- [x] **CH0.2 S2: coaxial coefficients.** Report: `spikes/coaxial/SPIKE_REPORT.md`.
  (S|R), (S|S), (R|R) by recursion. *Done when:*
  relative errors against an arbitrary-precision reference for p and k a up to 300, real,
  complex and imaginary k; the imaginary parts of (S|S) and (R|R) zero to round-off for real
  k; the index symmetry that halves the tables. *Needs:* –.
- [x] **CH0.3 S3: radial functions.** Report: `spikes/radial/SPIKE_REPORT.md`.
  j_n, h_n, i_n, k_n for real and complex arguments, with
  the scaling of §3.4 (exponential for Yukawa above κ r ≈ 1). *Done when:* accurate against
  arbitrary precision on every operator's argument and degree ranges; the scaling chosen.
  *Needs:* –.
- [x] **CH0.8a S8: workload census, first pass.** Report:
  `spikes/census/SPIKE_REPORT.md`. Trees of W1–W4 over the kD sweep from
  `nd-octree` with a prototype maximum leaf width (c_leaf 0.25 to 2), p from the formula; no
  FMM is run. *Done when:* per level the occupied boxes, U, V (by distance class), W and X
  sizes, p_l, regime and per-class choice, the memory model of §7 and the operation counts;
  c_leaf proposed (D11); no high-frequency leaf; the workloads proposed (D12); batch sizes
  for S5 and S9. *Needs:* –. *Sign-off:* D11, D12.
- [x] **CH0.7 S7: truncation under realistic geometry.** Report:
  `spikes/truncation/SPIKE_REPORT.md`, tables in `spikes/truncation/results/summary.md`.
  *Done when:* the P2M → M2M → M2L →
  L2L → L2P chain over two levels and M2M and L2L over three, on all 316 offsets, with points
  filling the boxes (uniform, corners, W1 patches), real, complex and imaginary k, in
  extended precision with the S1 and S2 prototypes; p as a function of (k a, ε, operator)
  meeting ε with a stated margin on every placement. *Needs:* CH0.1, CH0.2, CH0.3.
- [ ] **CH0.6 S6: stability of the diagonal form.** Seeded by Appendix B.1. *Done when:* the
  S → F, diagonal, F → R chain in f64 and f32 on all 34 classes against extended-precision
  RCR and the direct sum, with realistic placements, k a 3 to 400, ε 10⁻³ to 10⁻¹⁰, P from the
  formula and from S7, Im k > 0, |t| = 3w as a control; C_T and θ; the classes on RCR per
  (k a, ε); the f32 verdict (D3); whether interpolating T keeps the bound; prototype times.
  Accepted when the bound never under-predicts by more than 2×. *Needs:* CH0.3, CH0.7.
  *Sign-off:* D3.
- [ ] **CH0.8b S8: workload census, final pass.** *Done when:* the first pass repeated with
  S7's p and S6's constants; its numbers replace §7's. *Needs:* CH0.8a, CH0.6, CH0.7, D1.
- [ ] **CH0.9 S9: FFT plans, bits and host timing.** A spike crate outside the workspace on
  RLST 0.12's `rlst::fftw`, and rustfft. *Done when:* per-box plans of 2P − 1 and 2P rings,
  interleaved scratch against split plans; bits identical across batch positions, batch
  sizes, threads and fresh processes; fingerprints compared across two ranks; time per box
  for the product, FFTW and rustfft for P from 34 to 262 on smooth and power-of-two grids;
  `rlst::fftw::version()` per machine; D7 proposed per machine. *Needs:* CHR.4.
  *Machines:* locust, Kathleen. *Sign-off:* D7.
- [ ] **CH0.4 S4: plane-wave quadratures per k w_l.** *Done when:* how FMM3D obtains them;
  a generator or table for this library with its accuracy over the k w_l of low-frequency
  levels. *Needs:* –.
- [ ] **CH0.5 S5: cubek-fft for the φ step.** *Done when:* f64 accuracy against the product
  and an extended-precision DFT; time against the transform as a GEMM for N_φ = 256 to 2048
  at S8's batch sizes, on the GH200 and the CPU runtime; the cost of per-butterfly twiddles
  and power-of-two padding; D10 proposed. *Needs:* CH0.8a. *Machines:* locust.
  *Sign-off:* D10.
- [ ] **H0.G Gate.** `/milestone-gate H0`. *Needs:* every task above.

## H1: Conventions, math, reference

**Gate:** CONVENTIONS §3.15 signed off; `nd-fmm-math` and `nd-fmm-ref` carry the Helmholtz
family, checked against FMM3D's fixtures.

- [x] **CH1.4 FMM3D convention check and fixtures.** Report:
  `fmm-validate/results/helmholtz-ch1-4.md`; the draft `docs/helmholtz/conventions-3.15-draft.md`;
  the fixtures `fmm-ref/fixtures/helmholtz_fmm3d.json`. First a draft of §3.15 from proposal §4
  and S1–S3; then `hfmm3d` in `tools/fmm3d/driver.f90` and a generator in `tools/fixtures/`.
  *Done when:* on two points, FMM3D's factor, sign of k, gradient and branch identified and
  matching the draft (or the draft corrected); fixtures for real, complex and imaginary k and
  real and complex charges, committed with release, routine and `eps`; whether FMM3D is
  accurate for imaginary zk (R12). *Needs:* CH0.1, CH0.2, CH0.3. *Machines:* Kathleen.
- [x] **CH1.1 CONVENTIONS §3.15.** `docs/CONVENTIONS.md` §3.15, `CONVENTION_VERSION = 3`;
  the check `tools/fixtures/check_3_15.py`. The proposal's Section 4 as a CONVENTIONS
  section and a `CONVENTION_VERSION` bump. *Done when:* signed off by the user after
  CH1.4's two-point check. *Needs:* CH1.4, D5. *Sign-off:* §3.15.
- CH1.2 `nd-fmm-math`. Real orthonormal harmonics, radial functions (with the Yukawa
  scaling), rotation blocks at any degree. *Done when:* the identities of §3.15; the M2M and
  L2L coefficients real for real k. *Needs:* CH1.1, D2. Split on 2026-10-29 into three
  sub-tasks, one per spike it ports (S1, S3, S2), each one reviewable PR; CH1.2 is done
  when all three are:
- [x] **CH1.2a Orthonormal harmonics and rotation blocks.** `nd_fmm_math::orthonormal`,
  tests `fmm-math/tests/orthonormal/`, fixture `fmm-math/fixtures/helmholtz_angular.json`:
  the normalised Legendre functions of S1 (overflow-free, telescoped near the poles), the
  real orthonormal harmonics of §3.15 and their angular derivatives, the y-rotation blocks
  by the Gumerov–Duraiswami recursion at any degree, the z-y-z blocks and the π − β signs.
  *Done when:* §3.15's angular identities (against the Racah harmonics through N at
  p ≤ 30, the addition formula, orthonormality); the blocks against an mpmath fixture to
  p = 1000, orthogonal, against N-conjugated `rotation::blocks` at p ≤ 30, and
  Y(Qx̂) = Oⁿ Y(x̂); the π − β rule. *Needs:* CH1.1, D2.
- [x] **CH1.2b Radial functions and scaling.** `nd_fmm_math::radial` and `Complex` (num-complex),
  tests `fmm-math/tests/radial/`, fixture `fmm-math/fixtures/helmholtz_radial.json`:
  j_n, h_n for complex z and i_n, k_n for
  real x, with derivatives and the scaled forms of §3.15 (S3). *Done when:* against an
  mpmath fixture on every operator's range (S3's), the Wronskians, the relations of
  §3.15 (k₀, j_n(ix), h_n(ix)). *Needs:* CH1.1, D2.
- [x] **CH1.2c Coaxial coefficients.** `nd_fmm_math::coaxial`, tests `fmm-math/tests/coaxial/`,
  fixture `fmm-math/fixtures/helmholtz_coaxial.json`; report
  `fmm-validate/results/helmholtz-ch1-2c.md`. (R|R), (S|S) and (S|R) of §3.15 by the start
  column and the recursions (triangle, index symmetry), and by projection where
  Re(k) a > 40 (S2, S7). *Done when:* against an mpmath fixture (S2's); the M2M and L2L
  coefficients real for real k; i^{n−n′} (E|F) real for Yukawa; the symmetries; the
  rectangular truncation a block of the square table. *Needs:* CH1.2a, CH1.2b.
- CH1.3 `nd-fmm-ref`. Direct-sum oracle, reference operators, direct and RCR
  translations, the diagonal form in extended precision. *Done when:* addition theorems; the
  k → 0 limit with the tolerances of §9; Yukawa two ways; the direct sum against CH1.4's
  fixtures to round-off. *Needs:* CH1.2, CH1.4. Split on 2026-10-30 into four sub-tasks,
  each one reviewable PR. The diagonal form waits for its signatures, which §3.15 reserves
  until CH0.6 fixes them, and no acceptance item of CH1.3 tests it: CH1.3d is deferred
  after CH0.6 and off the priority list (the user, 2026-10-30, "Defer to CH1.3d"). The
  tasks that need CH1.3's operators (CH2.0, CH3.0) need CH1.3a–c; CH1.3 is done, and
  H1.G runnable, when all four are:
- [x] **CH1.3a Direct-sum oracle.** `nd_fmm_ref::helmholtz::p2p` (P2P and the direct
  sum), tests `fmm-ref/tests/helmholtz/`: the potential and gradient of e^{ikr}/r for real,
  complex and imaginary k and real and complex charges, coincident pairs excluded (§3.15,
  "Kernel and factors"), and Yukawa's real kernel e^{−κr}/r for real charges. *Done
  when:* the direct sum against CH1.4's fixtures to round-off; the k → 0 limit against
  Laplace's direct sum with the tolerances of §9 (relative to |k| r); Yukawa two ways
  (the real kernel against the complex one at k = iκ). *Needs:* CH1.4.
- [x] **CH1.3b Leaf operators.** `nd_fmm_ref::helmholtz::{leaf, Level, Workspace}`, tests
  `fmm-ref/tests/helmholtz/leaf_*.rs`. P2M, L2P, P2L and M2P in the stored form of §3.15 (the
  level's scale and damping), with gradients, for complex expansions and Yukawa's real
  ones. *Done when:* the series (P2M → M2P, P2L → L2P) against CH1.3a's direct sum within
  the truncation; the k → 0 limit against Laplace's leaf operators under the diagonal map
  of §3.4 with the tolerances of §9; Yukawa two ways; every applied exponent non-positive
  (finite at large Im(k) a). *Needs:* CH1.2, CH1.3a.
- [x] **CH1.3c Translations.** `nd_fmm_ref::helmholtz::{rcr, direct}`, tests
  `fmm-ref/tests/helmholtz/translation_*.rs`. M2M, L2L and M2L, direct (projection on a sphere by
  quadrature, from the leaf operators' fields) and RCR (`orthonormal` rotations and the
  `coaxial` tables), with rectangular degrees, for complex expansions and Yukawa's real
  ones. *Done when:* the addition theorems (P2M → M2M → M2P, P2L → L2L → L2P and
  P2M → M2L → L2P against the direct sum within the truncation; RCR against direct to
  round-off); the k → 0 limit against Laplace's translations with the tolerances of §9;
  Yukawa two ways. *Needs:* CH1.3b.
- [ ] **CH1.3d The diagonal form in extended precision.** The S → F, diagonal and F → R
  chain of §5.4 on the signatures CH0.6 fixes, its transfer functions in extended
  precision, as the reference of CH4.x. *Done when:* against CH1.3c's RCR on the same
  class within the bound E of §3.6. *Needs:* CH0.6, CH1.3c.
- [ ] **H1.G Gate.** `/milestone-gate H1`. *Needs:* every task above.

## H2: Yukawa on the host

**Gate:** CH2.5, Yukawa against FMM3D (or skipped with its reason).

- [x] **CH2.0 Laplace's real host drivers made generic over the kernel.** Report:
  `fmm-validate/results/helmholtz-ch2-0.md`; the drivers `nd-fmm-exec/src/driver.rs`. Dense, Classes,
  Rotation, Batched (§6.2). *Done when:* constraint 1 of §2: no new dispatch below the level
  call, layouts and tables unchanged, Laplace bit for bit, its benchmarks within noise on the
  three machines; the PR states how p reaches Laplace's inner loops. Else dropped, and Yukawa
  gets copies. *Needs:* CH1.3a–c, CHR.2, D9. *Machines:* locust, Kathleen.
- CH2.4 Yukawa P2P in `nd-fmm-simd`. On a loop skeleton shared with Laplace if
  constraint 1 holds. *Done when:* an accuracy contract like simd-p2p.md's; cost per pair
  against Laplace; Laplace's P2P bit for bit and at least 90 % of the spike (C3S.4).
  *Needs:* CH1.2. *Machines:* locust, Kathleen. Split on 2026-10-31 into two sub-tasks,
  as Phase 3S split the inverse square root (T4) from the kernel (T5) (the user, "Split
  a/b"); CH2.4 is done when both are:
- [x] **CH2.4a The vectorised exponential.** Report: `fmm-validate/results/helmholtz-ch2-4a.md`;
  design `docs/design/simd-p2p.md` §4.9. `nd_fmm_simd::exp` (`exp_slice`), the vector
  layer's `max` and `scale_pow2`, and `arch::exp` (Cody–Waite reduction and a polynomial), for
  x ≤ 0, f32 and f64, every ISA. *Done when:* within 4 u_T relative on [x_min, 0] (normal
  results), f32 exhaustively and f64 on 10⁷ samples against double-double, on every ISA a
  machine offers; below x_min the value at x_min; position and length invariance;
  no call in the inner loop; the existing entry points' inner loops unchanged instruction
  for instruction. *Needs:* CH1.2. *Machines:* Kathleen (AVX-512, the contract).
- [x] **CH2.4b Yukawa P2P on a shared skeleton.** Report: `fmm-validate/results/helmholtz-ch2-4b.md`;
  design `docs/design/simd-p2p.md` §4.10. `P2pKernel::evaluate_yukawa`; the P2P
  loop skeleton generic over the pair arithmetic, Laplace's the first. *Done when:*
  CH2.4's: an accuracy contract like simd-p2p.md's (§4.9 onwards) on every ISA; cost per
  pair against Laplace on the three machines, beside green-kernels' modified Helmholtz;
  Laplace's P2P bit for bit, its benchmarks within noise (D9), the plane-wave M2L steps
  (`plane_wave_parts`) on locust included (CH2.4a's question 2, signed off 2026-10-31), and
  at least 90 % of the spike (C3S.4). *Needs:* CH2.4a. *Machines:* locust, Kathleen.
- [ ] **CH2.1 The real `HelmholtzOperator` path.** On CH2.0's drivers (or copies), Rotation
  and Dense M2L. *Done when:* the FMM against the direct sum on W1, W3 and W4 for κD from
  10⁻² to 10³; deterministic. *Needs:* CH2.0, D12. *Sign-off:* D4.
- [ ] **CH2.2 Per-level tables with per-level cache keys.** *Done when:* build cost and
  memory reported; levels reused across an octave sweep. *Needs:* CH2.1, D4.
- [ ] **CH2.3 Pruning and the calibration of p_l.** *Done when:* the relative error within
  target for κD from 10⁻² to 10³; on pruned levels the V, W and X lists empty and the buffers
  zeroed; the same levels pruned on both ranks of a two-rank run; θ_D set. *Needs:* CH2.1,
  CH0.7.
- [ ] **CH2.5 Gate: Yukawa against FMM3D.** *Done when:* both codes against the same sampled
  direct sum over κD on W1 and W3, and the full-target crosscheck. CH1.4's R12 found FMM3D
  reliable for imaginary zk only in part. Signed off 2026-10-28 (CH1.4 report §5,
  question 2): the comparison with FMM3D is limited to where it met its eps, κD ≤ 30 at
  every eps and eps ≥ 10⁻⁶ (up to 2×) beyond; the sampled direct sum is the measure, and
  CH2.5 is not skipped. *Needs:* CH2.3, CH1.4. *Machines:* Kathleen.
- [ ] **H2.G Gate.** `/milestone-gate H2`. *Needs:* every task above.

## H3: Helmholtz on the host, RCR on every level

**Gate:** CH3.6, RCR on every level against FMM3D. The first Helmholtz deliverable, correct
at any kD.

- CH3.0 Builder: the leaf rule and complex densities. *Done when:* no leaf wider than
  c_leaf λ on W1–W4; if `nd-octree` gains a maximum leaf width, Laplace's trees and outputs
  bit for bit; complex charges in and split potentials and gradients out (§3.15 item 9), one
  complex expansion per complex vector. *Needs:* CH1.3a–c, D1, D11. Split on 2026-10-31
  into three sub-tasks, each one reviewable PR (the user, "Split a/b/c, proceed"); CH3.0
  is done, and CH3.1 runnable on its side, when all three are:
- [x] **CH3.0a nd-octree's maximum leaf width.** Record: `docs/helmholtz/tasks/CH3.0a.md`.
  `OctreeOptions::with_min_leaf_level`
  (the user, 2026-10-31): every box that holds a key and lies above the level is refined;
  default none. *Done when:* every leaf holding a key at or below the level (capped by
  `max_level`) on any number of ranks, the leaves the one-rank leaves and a brute-force
  refinement's; unset, every tree and Laplace's outputs bit for bit (constraint 1, by the
  byte comparison of the full outputs on the M6, without D9 timing runs: the user,
  2026-10-31, "Bytes only, no D9"). *Needs:* –.
- [x] **CH3.0b The Helmholtz builder's leaf rule.** Record: `docs/helmholtz/tasks/CH3.0b.md`
  (the sweep's table); `nd_fmm_exec::helmholtz`, `nd_fmm_validate::helmholtz_workloads`,
  tests `fmm-validate/tests/helmholtz_leaves*.rs`. k classified (§3.15), the leaf level
  from (k, D, c_leaf), the reference workloads W1–W4 in `nd-fmm-validate` (from
  `spikes/census`). *Done when:* no leaf wider than c_leaf λ on W1–W4 over D1's kD range;
  no rule for Yukawa. *Needs:* CH3.0a, D1, D11.
- [x] **CH3.0c Complex densities.** Record: `docs/helmholtz/tasks/CH3.0c.md`;
  `nd_fmm_exec::helmholtz::densities`, tests `fmm-exec/tests/helmholtz/densities.rs`.
  *Done when:* complex charges in and split potentials
  and gradients out (§3.15 "Inputs and outputs"), the leaf chunks fixed, one complex
  expansion per complex charge vector. *Needs:* CH3.0b, CH1.3a–c.
- [x] **CH3.5 The shared drivers extended to complex expansions.** Record:
  `docs/helmholtz/tasks/CH3.5.md`. Two planes, real shifts
  for real k, the epilogue. *Done when:* constraint 1 as in CH2.0; else the complex operator
  gets its own drivers. *Needs:* CH2.0. *Machines:* locust, Kathleen. Closed as a record
  on 2026-11-01 (the user, "Close as a record"). The target-by-target drivers pass whole
  box blocks sized by the plan, so two planes and real shifts need no driver change, and
  CH3.1's RCR operator runs on them unchanged. Laplace is untouched, so no machine runs.
  The batched M2L's two-plane case and the epilogue move to CH3.2, their first user.
- CH3.1 Complex planes and RCR on every level. RCR M2M and L2L (real coaxial steps
  for real k) and RCR M2L. *Done when:* the FMM against the direct sum on W1, W3 and W4 for
  kD ≤ 100 and on W1 at kD = 500; deterministic. *Needs:* CH3.0, CH3.5. Split on
  2026-11-01 into three sub-tasks, each one reviewable PR (the user, "a/b/c"); CH3.1 is
  done, and the tasks that need it runnable, when all three are:
- [x] **CH3.1a The Helmholtz operator.** Record: `docs/helmholtz/tasks/CH3.1a.md`;
  `nd_fmm_tables::helmholtz`, `nd_fmm_exec::helmholtz::{degree, operator}`, tests
  `fmm-tables/tests/helmholtz/`, `fmm-exec/tests/helmholtz/{degree, operator}.rs`:
  `HelmholtzOperator`, the plan's operator on the unchanged shared drivers (CH3.5), for
  real and complex k with real and complex charges. The per-level degree by a rule behind a
  setting, S7's closed form at a tolerance by default and degrees named per level as the
  override (the user, 2026-11-01, "as recommended"); per-level coaxial tables with the
  stored form's factors folded in; its own RCR, the real steps on both planes and the real
  multipoles of real charges for real k; P2L, M2P and P2P (the reference P2P until CH3.4).
  *Done when:* every body against `nd_fmm_ref::helmholtz` (the translations against `rcr`
  to rounding on the 8 octants and the 316 offsets, with rectangular degrees, below and
  above the projection threshold; the leaf operators and P2P against `leaf` and `p2p`);
  the chains against the direct sum within the truncation; vector v the one-vector
  operator bit for bit; real charges as complex ones with Im q = 0; Yukawa refused (its
  path is CH2.1's); Laplace untouched. *Needs:* CH3.0, CH3.5.
- [x] **CH3.1b The Helmholtz FMM.** Record: `docs/helmholtz/tasks/CH3.1b.md`;
  `nd_fmm_exec::helmholtz::fmm`, the crate-private `src/pipeline.rs`, tests
  `fmm-exec/tests/helmholtz_fmm.rs`. The operator-independent point pipeline of `fmm.rs`
  (the domain, the octree and plan, the redistributions, the leaf ranges and radii, the
  leaf-scaled coordinates, the charge forward and the output return) extracted into
  crate-private code that `Fmm` and the Helmholtz FMM share (the user, 2026-11-01, "as
  recommended"); the Helmholtz builder (k, the degrees, gradients, the charges, the
  vectors, threads, the domain) and its evaluation on any number of ranks, the tree with
  `min_leaf_level`. *Done when:* against the direct sum on small trees; bit for bit at
  every thread count and on P ranks the one-rank FMM; the level calls the operator's
  `PerPair` bit for bit; Laplace's outputs bit for bit, and minimal D9 timing runs to
  check the extraction's impact (the user, 2026-11-01: "make minimal timing runs to check
  the impact"). *Needs:* CH3.1a.
- [x] **CH3.1c The acceptance runs.** Report: `fmm-validate/results/helmholtz-ch3-1c.md`;
  record `docs/helmholtz/tasks/CH3.1c.md`; `nd_fmm_validate::helmholtz_accuracy`, the
  `helmholtz_accuracy` example, test `fmm-validate/tests/helmholtz_accuracy.rs`. *Done
  when:* CH3.1's: the FMM against the sampled
  direct sum on W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500; deterministic; a report.
  *Needs:* CH3.1b.
- [ ] **CH3.2 Dense, Classes and Batched where the node budget allows.** *Done when:*
  measured against Rotation per p, the cost of the cube-group transforms that move the z-axis
  included; Batched on the shared drivers needs the two-plane case of `driver::m2l_block`
  (`PLANES`, rectangular real-to-complex tables, the per-target epilogue of proposal §6.2;
  moved here from CH3.5 on 2026-11-01), and Dense either combines per pair or gets a
  per-target finish hook in `driver::m2l_target`, with constraint 1 as in CH2.0 (Laplace bit for
  bit, D9 on the three machines), else Helmholtz's Batched gets its own block.
  *Needs:* CH3.1. *Machines:* locust, Kathleen.
- [ ] **CH3.3 Calibration of p_l over (k a_l, ε).** *Done when:* S7's low-frequency tables
  confirmed in the full FMM. *Needs:* CH3.1, CH0.7.
- CH3.4 Helmholtz P2P, real and complex k. On CH2.4's skeleton. *Done when:* the
  accuracy contract over k r up to 22 c_leaf; cost per pair; the Helmholtz operator's P2P
  if CH3.1 is merged (priority list). *Needs:* CH2.4.
  *Machines:* locust, Kathleen. Split on 2026-11-01 into three sub-tasks, each one
  reviewable PR, as CH2.4 split the exponential from the kernel; CH3.4 is done, and CH3.6
  runnable on its side, when all three are:
- [x] **CH3.4a The vectorised sine and cosine.** Record: `docs/helmholtz/tasks/CH3.4a.md`;
  design `docs/design/simd-p2p.md` §4.11. `nd_fmm_simd::sincos` (`sincos_slice`),
  `arch::sincos`: sin θ and cos θ together, reduction by π (Cody–Waite, two parts as
  built) and Taylor polynomials, the sign of (−1)ⁿ by exact arithmetic, f32 and f64, every
  ISA. *Done when:* both within 4 u_T absolute for |θ| ≤ 2¹⁰ (above 22 c_leaf for any c_leaf
  ≤ 46), f32 exhaustively on [0, 2¹⁰] and f64 on 10⁷ samples against double-double, on
  every ISA a machine offers; odd and even bit for bit for θ ≠ 0 (sin(±0) = +0, as built);
  every ISA the scalar layer bit for
  bit; position and length invariance; no call in the inner loop; the existing entry
  points the same instruction for instruction as `main`'s. *Needs:* CH2.4a. *Machines:*
  Kathleen (AVX-512, the contract).
- [x] **CH3.4b Helmholtz P2P on the skeleton.** Report: `fmm-validate/results/helmholtz-ch3-4b.md`;
  design `docs/design/simd-p2p.md` §4.12. `P2pKernel::evaluate_helmholtz`: e^{ikr}/r
  and its gradient for real and complex k (e^{−Im(k) r} by CH2.4a's exponential) with real
  and complex charges, on the P2P skeleton widened to the planes of a complex charge and
  output, Laplace's and Yukawa's instantiations unchanged. *Done when:* CH3.4's contract
  over Re(k) r ≤ 22 c_leaf on every ISA, the scalar path bit for bit the reference; cost
  per pair against Laplace and Yukawa on the three machines, beside green-kernels'
  Helmholtz; Laplace's P2P instruction for instruction and bit for bit, and its benchmarks
  within noise (D9) and C3S.4 if the skeleton's Laplace code changes. *Needs:* CH3.4a.
  *Machines:* locust, Kathleen.
- [ ] **CH3.4c The Helmholtz operator's P2P.** `HelmholtzOperator`'s P2P through
  `evaluate_helmholtz` on the P2P choice's ISA, the reference kept as the trusted path.
  *Done when:* every ISA against the reference within CH3.4b's contract in the operator's
  and the FMM's tests; the FMM deterministic at every thread count and rank-invariant as
  before; its time on a CH3.1c case before and after. *Needs:* CH3.4b, CH3.1.
- [ ] **CH3.6 Gate: RCR on every level against FMM3D.** *Done when:* both codes against the
  same sampled direct sum on W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500 with time; the
  full-target crosscheck, timed with CH3.4's vectorised P2P. *Needs:* CH3.3, CH1.4, CH3.4
  (added by the user on 2026-10-28; not in proposal §10). *Machines:* Kathleen.
- [ ] **H3.G Gate.** `/milestone-gate H3`. *Needs:* every task above.

## H6: Exponential M2L on low-frequency levels (before H4 by default, D6)

**Gate:** the exponential M2L adopted where it wins, or kept opt-in (D8).

- [ ] **CH6.1 The §3.14 machinery generalised.** Six directions, per-level quadratures,
  merge and shift; opt-in. *Done when:* errors within about 1.1× Rotation's. *Needs:* CH0.4,
  CH3.1, D6.
- [ ] **CH6.2 Measured.** Against Rotation and the table-based strategies on the three
  machines, and against `hfmm3d`'s exponential M2L. *Done when:* per-level accuracy and time
  beside FMM3D's on the same problems; D8 proposed. *Needs:* CH6.1, CH1.4.
  *Machines:* locust, Kathleen. *Sign-off:* D8.
- [ ] **H6.G Gate.** `/milestone-gate H6`. *Needs:* every task above.

## H4: High-frequency levels

**Gate:** CH4.4 and CH4.5, the kD sweep and host performance against FMM3D.

- [ ] **CH4.1 S → F and F → R by the transform as a product.** Per-box aligned signature
  buffers; source signatures kept per level call, target accumulators chunked under the node
  budget. *Done when:* round trips to the grid's bandwidth; bits independent of the chunk
  size; scratch within the budget on W1 at kD = 1000. *Needs:* CH3.1, D6.
- [ ] **CH4.2 Diagonal M2L per (level, class).** Transfer functions per class, read through
  the class permutations. *Done when:* against RCR on the same class within E(l, c);
  deterministic with the RCR pairs added after F → R. *Needs:* CH4.1, CH0.6.
- [ ] **CH4.3 The per-(level, class) rule, calibrated.** *Done when:* S6's C_T and θ
  confirmed in the full FMM on W1 and W3; the time crossover per class; a whole level falls
  back to RCR when the scratch budget binds. *Needs:* CH4.2.
- [ ] **CH4.4 kD sweep, with the FMM3D gate.** *Done when:* the FMM against the direct sum on
  W1 to kD = 1000, W2 to 500 and W3 to 200, beside `hfmm3d`; the full-target crosscheck mapped
  to levels and classes, with no error concentrated at a class boundary; time, memory and
  build cost per k. *Needs:* CH4.3, CH1.4. *Machines:* Kathleen.
- [ ] **CH4.5 Host performance against FMM3D at matched accuracy.** As C6.8, setup apart,
  one node, on CH4.4's sweep. *Needs:* CH4.4. *Machines:* locust, Kathleen.
- [ ] **CH4.6 The optional FFTW back end of the φ step.** *Done when:* against the product to
  round-off; per-box plans with bits the same across batch positions, sizes and threads, and
  fingerprints all-gathered; off by default, a default build linking no FFTW, cargo-deny, a CI
  job with the feature; the GPL warning in the crate documentation, the README, the builder
  option and `Cargo.toml`. *Needs:* CH4.1, CHR.3, CH0.9, D7.
- [ ] **H4.G Gate.** `/milestone-gate H4`. *Needs:* every task above.

## H5: Complex wavenumber

**Gate:** CH5.3, complex k against FMM3D.

- [ ] **CH5.1 Classifier and pruning for Im k > 0.** *Done when:* regimes as §3.6 with S6's
  damped results; the error within target over a grid of (Re kD, Im kD) on W1. *Needs:*
  CH2.3, CH4.3.
- [ ] **CH5.2 Damping-dependent truncation and bound.** *Done when:* the error within target
  for weakly damped cases. *Needs:* CH5.1, CH0.6.
- [ ] **CH5.3 Gate: complex k against FMM3D.** *Done when:* both codes against the same
  sampled direct sum over the (Re kD, Im kD) grid, and the full-target crosscheck. The
  comparison with FMM3D is limited for Im kD ≳ 50 as in CH2.5 (CH1.4 report §5,
  question 2, signed off 2026-10-28). *Needs:* CH5.2, CH1.4. *Machines:* Kathleen.
- [ ] **H5.G Gate.** `/milestone-gate H5`. *Needs:* every task above.

## H7: Device path

**Gate:** CH7.6, the device path against FMM3D at matched accuracy.

- [ ] **CH7.1 Rotation and translation kernels on complex planes.** Real coaxial steps of
  M2M and L2L, the complex (S|R) step of the M2L, the stacked GEMM where Dense applies.
  *Done when:* device output within the host's error. *Needs:* CH3.1, D3.
  *Machines:* locust.
- [ ] **CH7.2 Device P2P for Helmholtz and Yukawa.** *Done when:* an accuracy contract as the
  host's; cost per pair against the host and Laplace's device P2P. *Needs:* CH3.4.
  *Machines:* locust.
- [ ] **CH7.3 Device leaf operators.** P2M, L2P, M2P, P2L with radial recursions. *Done
  when:* against the host operators within error. *Needs:* CH3.1. *Machines:* locust.
- [ ] **CH7.4 Diagonal M2L and the transforms on the device.** cubek-fft where D10 adopted it,
  else the transform as a GEMM. *Done when:* as CH4.2 on the device; device locals against
  the host's within error. *Needs:* CH4.2, CH7.1, CH0.5, D10. *Machines:* locust.
- [ ] **CH7.5 Exponential M2L on the device**, if D8 adopted it on the host. *Done when:* as
  CH6.1 on the device. *Needs:* CH6.2, CH7.1, D8. *Machines:* locust.
- [ ] **CH7.6 Device performance against FMM3D at matched accuracy.** As CH4.5, the device
  path beside the host's. *Needs:* CH7.2, CH7.3, CH7.4, CH4.5. *Machines:* locust.
- [ ] **H7.G Gate.** `/milestone-gate H7`. *Needs:* every task above.

## H8: Distributed path

**Gate:** Kathleen scaling at kD of a few hundred, rank-invariant.

- [ ] **CH8.1 Byte cap for the replicated top; exchange of the high-frequency levels.**
  *Done when:* bit for bit the one-rank output on P ranks; memory per rank and per node
  bounded. *Needs:* CH4.4. *Machines:* Kathleen.
- [ ] **CH8.2 Kathleen scaling at kD of a few hundred.** *Done when:* efficiency and memory
  reported as in Phase 5S, at most 4 nodes per job. *Needs:* CH8.1. *Machines:* Kathleen.
- [ ] **H8.G Gate.** `/milestone-gate H8`. *Needs:* every task above.
