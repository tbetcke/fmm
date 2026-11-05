# Helmholtz FMM roadmap

The tasks of `docs/design/helmholtz-fmm-proposal.md` (draft 0.8), Section 10, in the order
`/next-task` takes them. The proposal is the full record: each task's *Done when* condenses
its acceptance criterion in the §10 table, and where the two differ the proposal wins.
Progress is logged in `docs/helmholtz/progress.md`; each task's plan is
`docs/helmholtz/tasks/<ID>.md`; each milestone's gate report is
`docs/helmholtz/gates/<milestone>.md`.

Status: **H3 closed (2026-11-02, `docs/helmholtz/gates/H3.md`).** The RCR FMM on every
level meets ε against FMM3D on all 57 case × ε points of CH3.6. HR, H0, H1 and H2 are still
open. 2026-11-03: CH0.4 (S4, the plane-wave quadratures) done, its three sign-offs taken;
CH6.1 split into a/b/c, CH6.1a (the quadratures in `nd-fmm-tables`) done; CH6.1b split
again into the generator (CH6.1b, done: S4's generator in `nd-fmm-ref`) and the pair chain
(CH6.1c), merge and shift renamed CH6.1d. CH6.1c (the pair chain, CONVENTIONS §3.15
"Exponential M2L" for sign-off) done; its degree study hands CH6.1d the pairing of a
level's quadrature with its degree. CH6.1c merged
(#181); CH6.1d split into merge and shift in the FMM (CH6.1d) and the quadrature per level
by measurement (CH6.1e). 2026-11-03: CH6.1d done (`HelmholtzM2l::PlaneWave`, merge and shift
with a provisional precision rule), merged (#182). 2026-11-03: CH6.1e done, so CH6.1 is: the
rule by the FMM's errors on CH3.1c's cases, iprec 2 at ε ≥ 10⁻⁶ and Rotation below, within
1.070× Rotation's on every case; no precision of FMM3D's tables holds 1.1× below 10⁻⁶
(the follow-up CH6.1f). 2026-11-03: CH6.2 split into the bounded waves buffer (CH6.2a)
and the measurements (CH6.2b); CH6.2a done (the level call by chunks of parents within a
node budget, bit for bit at every bound; 2026-11-04 its default 8 GiB per node signed
off). 2026-11-04: CH6.2b split into our strategies timed per level (CH6.2b) and FMM3D
with D8 (CH6.2c); CH6.2b done on the M6 and locust (the exponential M2L the fastest
method on almost every level below the coarsest at 10⁻⁶, 0.19–0.45× Rotation's M2L on the
larger cases; at 10⁻³ a per-level choice; the coarse levels slow at many threads);
Kathleen down, its runs moved to CH6.2c (the user). 2026-11-05: CH7.5 split
(the user) into Laplace's plane-wave M2L on the device (CH7.5a, which builds the merge and
shift kernels) and Helmholtz's on them (CH7.5b); CH7.5a first on the priority list while
Kathleen is down (the user). 2026-11-05: CH7.5a split (the user, "a/b/c, Helmholtz → d")
into the kernels (CH7.5a), the device's `PlaneWave` behind an opt-in setting (CH7.5b) and
the measurements with the adoption (CH7.5c); Helmholtz's renamed CH7.5d. 2026-11-05:
CH7.5a done (the four steps of merge and shift as device kernels, every step bit for bit a
host `mul_add` replica on the CPU runtime, Metal and CUDA, every p ≤ 20 within the
quadrature's bound of `direct::m2l`). Next: CH7.5b (*Machines:* locust, the M6), then
CH7.5c, then CH6.2c (*Machines:* Kathleen; D8) once Kathleen is back.

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

## Priority list: the path to H6

Set by the user on 2026-11-02 at the H3 gate ("i agree with the recommendations";
`docs/helmholtz/gates/H3.md` §5 and question 6), with D6 signed off as H6 first: the
exponential M2L targets the cases where the FMM is slowest against FMM3D, the
low-frequency and intermediate trees at tight ε across most of D1's range (gate report
§3.3). It replaces the list of 2026-10-28, the shortest path to CH3.6, which finished with
CH3.6 (#170) and H3.G.

1. CH7.5a Laplace's plane-wave M2L on the device (added on 2026-11-05, the user: "Please add
   to the priority list to get it sooner", then "Please put 7.5a to the top of the list.
   Kathleen is still down"; needs only Phase 6 T13, merged; *Machines:* locust, the M6; split on
   2026-11-05 into CH7.5a, CH7.5b and CH7.5c, the user: "a/b/c, Helmholtz → d")
2. CH0.4 S4: quadratures of the Helmholtz exponential M2L per k w_l
3. CH6.1 The §3.14 machinery generalised (needs CH0.4): CH6.1a, CH6.1b, CH6.1c, CH6.1d, CH6.1e
4. CH6.2 Measured (*Machines:* locust, Kathleen; *Sign-off:* D8): CH6.2a, CH6.2b, CH6.2c
5. CH6.1f Quadratures for ε below 10⁻⁶ (needs CH6.1e; after CH6.2, before H6.G: CH6.1e's
   question 3, signed off 2026-11-03, "1–4 as recommended")

Off the list, in file order: CH0.6, CH0.8b, CH0.9, CH0.5 and H0.G; CHR.3 and HR.G; CH1.3d
and H1.G; CH2.1, CH2.2, CH2.3, CH2.5 and H2.G; H6.G; H4 (CH4.0, CH4.0b and CH4.0c first),
H5, H7 (CH7.5a, CH7.5b and CH7.5c apart), H8.

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
| D6 | Order of H4 and H6 | H6 first | H3.G | CH6.1, CH4.1 | signed off 2026-11-02 (H3 gate, question 2): "i agree with the recommendations", H6 first, as recommended; on the gate report's reason (§3.3): the calibrated degrees cut the low-frequency share of the far field at kD = 500 to about half (an estimate), and the FMM is slowest against FMM3D on low-frequency trees at tight ε, which H6 targets |
| D7 | Host back ends of the φ step | the product as default and reference; FFTW behind the optional `fftw` feature; rustfft if S9 finds it deterministic per box and faster, decided per machine | CH0.9 | CHR.3, CH4.6 | open |
| D8 | Exponential M2L as a default on low-frequency levels | after CH6.2 | CH6.2c | CH7.5d | open |
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
  the calibrated p of CH3.3 (`DegreeRule::CALIBRATED`, not S7's p, on the low-frequency
  levels; amended at H3.G, proposal §7) and S6's constants; its numbers replace §7's. *Needs:* CH0.8a, CH0.6, CH0.7, D1.
- [ ] **CH0.9 S9: FFT plans, bits and host timing.** A spike crate outside the workspace on
  RLST 0.12's `rlst::fftw`, and rustfft. *Done when:* per-box plans of 2P − 1 and 2P rings,
  interleaved scratch against split plans; bits identical across batch positions, batch
  sizes, threads and fresh processes; fingerprints compared across two ranks; time per box
  for the product, FFTW and rustfft for P from 34 to 262 on smooth and power-of-two grids;
  `rlst::fftw::version()` per machine; D7 proposed per machine. *Needs:* CHR.4.
  *Machines:* locust, Kathleen. *Sign-off:* D7.
- [x] **CH0.4 S4: plane-wave quadratures per k w_l.** Report:
  `spikes/pw-quadrature/SPIKE_REPORT.md`. *Done when:* how FMM3D obtains them;
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
- CH3.2 Dense, Classes and Batched where the node budget allows. *Done when:*
  measured against Rotation per p, the cost of the cube-group transforms that move the z-axis
  included; Batched on the shared drivers needs the two-plane case of `driver::m2l_block`
  (`PLANES`, rectangular real-to-complex tables, the per-target epilogue of proposal §6.2;
  moved here from CH3.5 on 2026-11-01), and Dense either combines per pair or gets a
  per-target finish hook in `driver::m2l_target`, with constraint 1 as in CH2.0 (Laplace bit for
  bit, D9 on the three machines), else Helmholtz's Batched gets its own block.
  *Needs:* CH3.1. *Machines:* locust, Kathleen. Split on 2026-11-02 into three sub-tasks,
  each one reviewable PR (a new table family, its symmetry form and the batched block with
  constraint 1 and the three machines do not fit one); CH3.2 is done, and H3.G runnable on
  its side, when all three are, and CH3.2d (added at CH3.2c's sign-off) too:
- [x] **CH3.2a Dense.** Report: `fmm-validate/results/helmholtz-ch3-2a.md`; record
  `docs/helmholtz/tasks/CH3.2a.md`; `nd_fmm_tables::helmholtz::dense`,
  `nd_fmm_exec::helmholtz::HelmholtzM2l`, the `helmholtz_m2l` example. Dense is Rotation to
  rounding, level with it end to end on the M6 at p = 7 and slower above (1.5–2.6×; per
  pair faster only to p ≈ 6), so it stays opt-in. Per level and V-list offset the complex M2L table as the 2n × n real
  matrix [A_r; A_i] (n = (p_l + 1)², real input to two output planes), built in f64 by the
  level's RCR (`HelmholtzTables<f64>`) and rounded to T, in `nd_fmm_tables::helmholtz`; the
  operator's M2L per level, `Rotation` (the default) or `Dense` on the levels whose tables fit
  a byte budget, from the finest level up, `Rotation` elsewhere; Dense combined per pair (L_r
  += A_r M_r − A_i M_i, L_i += A_i M_r + A_r M_i), so no driver changes (CH3.5, question 2).
  *Done when:* every offset against Rotation to rounding on several levels, real and complex
  k, one and two multipole planes; the FMM with Dense against the direct sum and against
  Rotation's output, the same at every thread count and on P ranks the one-rank FMM bit for
  bit; the tables' bytes and build time, and the M2L cost per pair against Rotation per p
  on the M6. *Needs:* CH3.1.
- [x] **CH3.2b Classes.** Report: `fmm-validate/results/helmholtz-ch3-2b.md`; record
  `docs/helmholtz/tasks/CH3.2b.md`; `nd_fmm_tables::helmholtz::classes::{ClassM2l,
  CubeTransforms}`, `HelmholtzM2l::Classes`. Classes is Dense to rounding, its tables about
  18× smaller; per pair Dense's chain plus the transforms, so faster than Rotation only to
  p = 4, and end to end 1.23–3.0× Rotation's evaluation on the M6: it stays opt-in.
  Dense over the 16 symmetry classes of CONVENTIONS §3.12 per level,
  with the cube-group coefficient transforms in the orthonormal basis of §3.15 (signed
  permutations for the elements that fix the z-axis, dense per-degree blocks for the others).
  *Done when:* against Dense to rounding at every offset; the FMM as CH3.2a's; the cost per
  pair against Rotation and Dense per p on the M6, the transforms that move the z-axis
  included. *Needs:* CH3.2a.
- [x] **CH3.2c Batched, and the measurements per p.** Report:
  `fmm-validate/results/helmholtz-ch3-2c.md`; record `docs/helmholtz/tasks/CH3.2c.md`;
  `nd_fmm_tables::helmholtz::batched::BatchedM2l`, `HelmholtzM2l::Batched`,
  `with_m2l_mul_add`. Helmholtz's own block on the column kernel (the user, 2026-11-02:
  "Own Helmholtz block"; no Laplace path changes), Dense's tables in split planes ("Split
  planes, 2n²"): Dense bit for bit with real multipoles, to rounding with complex ones;
  the fused mode opt-in. Fused Batched beats Rotation per pair to p ≈ 10 on the M6 and
  locust and p ≈ 15 on Kathleen, and end to end where the levels with tables have low p
  (0.39–0.82× on the W4 cube and W3 at 10⁻³); it stays opt-in. The batched M2L on
  CH3.2a's tables: the two-plane case of `driver::m2l_block` (`PLANES`, rectangular tables, the per-target
  epilogue of proposal §6.2) with constraint 1 as in CH2.0 (Laplace bit for bit, D9 on the
  three machines), else Helmholtz's own block. *Done when:* against Dense (bit for bit, or
  within rounding where an epilogue reorders the sums, stated); the FMM as CH3.2a's;
  Rotation, Dense, Classes and Batched per p on the M6, locust and Kathleen. *Needs:*
  CH3.2a, CH3.2b. *Machines:* locust, Kathleen.
- [x] **CH3.2d The M2L rule per level.** Report: `fmm-validate/results/helmholtz-ch3-2d.md`;
  record `docs/helmholtz/tasks/CH3.2d.md`; `HelmholtzM2l::Auto`, `auto_batched_max_degree`,
  `BatchedM2l::select_within`. The rule's choice the best on 34 of 36 case × machine rows,
  1.001 on one, 1.106 on one (Kathleen AVX-512, a sparse W1 level; accepted, CH4.0). Auto
  is `HelmholtzFmmBuilder`'s default (signed off 2026-11-02, "i agree with the
  recommendations").
  Added on 2026-11-02 at CH3.2c's sign-off (the
  user, "i agree with the recommendations": a follow-up before H3.G). An Auto setting of
  `HelmholtzM2l` choosing per level from the degrees, the budget and the machine class only
  (constraint 9): the fused Batched M2L on the levels with tables below a crossover degree
  (CH3.2c measured about 10 on the M6 and locust, about 15 on Kathleen), Rotation above;
  sparse levels (W1) counted. *Done when:* the rule's choice against the best of Rotation and
  fused Batched end to end on CH3.2a's cases on the three machines, within a stated margin;
  deterministic and rank-invariant as CH3.2c's FMM; whether it becomes the default is a
  question for sign-off. *Needs:* CH3.2c. *Machines:* locust, Kathleen.
- [x] **CH3.3 Calibration of p_l over (k a_l, ε).** Report:
  `fmm-validate/results/helmholtz-ch3-3.md`; record `docs/helmholtz/tasks/CH3.3.md`;
  `nd_fmm_exec::helmholtz::{DegreeRule, degree, s7_degree}`,
  `nd_fmm_validate::helmholtz_calibration`, the `helmholtz_calibrate` example. S7's tables
  upper bounds in the full FMM at all 176 measured points; the calibrated rule
  max(⌈0.30 L² + 0.25 L + 2.60⌉, ⌈x + 0.80 (ln 1/ε)^{2/3} x^{1/3}⌉) the default (the user,
  2026-11-01: "Calibrate, new default", "Both branches", "ε/2, verify ≤ ε"), S7's closed
  form kept as `s7_degree`. *Done when:* S7's low-frequency tables confirmed in the full
  FMM. *Needs:* CH3.1, CH0.7.
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
- [x] **CH3.4c The Helmholtz operator's P2P.** Record: `docs/helmholtz/tasks/CH3.4c.md`;
  `HelmholtzOperator::with_p2p`, `HelmholtzFmmBuilder::p2p_kernel`, tests
  `fmm-exec/tests/helmholtz/p2p.rs`. `HelmholtzOperator`'s P2P through
  `evaluate_helmholtz` on the P2P choice's ISA, the reference kept as the trusted path.
  *Done when:* every ISA against the reference within CH3.4b's contract in the operator's
  and the FMM's tests; the FMM deterministic at every thread count and rank-invariant as
  before; its time on a CH3.1c case before and after. *Needs:* CH3.4b, CH3.1.
- [x] **CH3.6 Gate: RCR on every level against FMM3D.** Report:
  `fmm-validate/results/helmholtz-ch3-6.md`; record `docs/helmholtz/tasks/CH3.6.md`;
  `nd_fmm_validate::helmholtz_fmm3d`, the `helmholtz_fmm3d` example, `tools/fmm3d/run.sh
  helmholtz`. Ours within ε on all 57 case × ε points (largest 0.63 ε), Kathleen jobs
  617649–617651. *Done when:* both codes against the
  same sampled direct sum on W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500 with time; the
  full-target crosscheck, timed with CH3.4's vectorised P2P. *Needs:* CH3.3, CH1.4, CH3.4
  (added by the user on 2026-10-28; in proposal §10 since H3.G). *Machines:* Kathleen.
- [x] **H3.G Gate.** `/milestone-gate H3`. Report: `docs/helmholtz/gates/H3.md`. Closed
  2026-11-02 (the user, "i agree with the recommendations"): D6 signed off (H6 first); the
  proposal amended (C1–C6, its Appendix A); CH4.0b and CH4.0c added; the priority list for
  H6. *Needs:* every task above.

## H6: Exponential M2L on low-frequency levels (before H4 by default, D6)

**Gate:** the exponential M2L adopted where it wins, or kept opt-in (D8).

- CH6.1 The §3.14 machinery generalised. Six directions, per-level quadratures,
  merge and shift; opt-in. The quadratures as S4 recommends, CH0.4's sign-offs 1 and 2
  taken on 2026-11-03 (`spikes/pw-quadrature/SPIKE_REPORT.md` §6): FMM3D's tables for
  iprec 1 to 3 extracted with NIST's notice, the α counts chosen by measurement (FMM3D's
  miss the stated error at the region's corner), the S4 generator as the tests' trusted
  path. *Done when:* errors
  within about 1.1× Rotation's. *Needs:* CH0.4, CH3.1, D6. Split on 2026-11-03 into three
  sub-tasks, each one reviewable PR (about 1.5 MB of NIST's tables, the factors of the
  orthonormal basis with a convention for sign-off, and merge and shift in the FMM do not
  fit one), and the second again the same day into the generator (CH6.1b) and the pair
  chain (CH6.1c), about 1,800 lines in two crates together, and the fourth again into
  merge and shift (CH6.1d) and the quadrature per level by measurement (CH6.1e); CH6.1 is
  done, and CH6.2 runnable on its side, when all five are:
- [x] **CH6.1a The quadratures.** Record: `docs/helmholtz/tasks/CH6.1a.md`;
  `nd_fmm_tables::helmholtz::plane_wave::quadrature`, tests
  `fmm-tables/tests/helmholtz/plane_wave_quadrature.rs`. The kernel within 1.50× and the
  chain within 1.18× the stated error on every domain; 1.31–1.51× FMM3D's waves where
  Im κ < π/2. `nd_fmm_tables::helmholtz::plane_wave::quadrature`:
  FMM3D's tables for iprec 1 to 3 (437 domains each), extracted with NIST's notice by a
  script in `tools/fixtures/` from the pinned v2.1.0; `hwts3e`'s domain lookup and
  precision from ε; the node maps to complex (λₖ, wₖ) at κ = k w_l; the α and Fourier
  counts per node by a Bessel bound (S4's `tail_count`, for complex tₖ), not FMM3D's
  `hnumphys`. *Done when:* every domain's rectangle and node count as FMM3D's; the
  lookup on every rectangle and outside the tables; with the counts, the kernel
  e^{iκ|u|}/|u| on the region within 2× the stated error and the point-to-point chain
  within 3× it, the kernel at nine points of every domain and precision and the chain at
  every domain's centre (release; sampled in debug); the wave count Σ Nₖ against FMM3D's recorded. *Needs:* CH0.4.
- [x] **CH6.1b The generator.** Record: `docs/helmholtz/tasks/CH6.1b.md`;
  `nd_fmm_ref::helmholtz::plane_wave::generate`, tests `fmm-ref/tests/helmholtz/plane_wave.rs`.
  Bit for bit the spike at all 35 κ × ε of its tests; ε met on the grid, 2ε off it and in
  the chain at each. S4's generator (`spikes/pw-quadrature/src/generator.rs`)
  ported to `nd_fmm_ref::helmholtz::plane_wave`, the trusted path of the quadratures for
  any κ = k w with Re κ, Im κ ≥ 0: the Sommerfeld integral on the contour of real t,
  composite Gauss–Legendre, compressed by a column-pivoted Gram–Schmidt with refitted
  weights, α and Fourier counts by S4's Bessel bound; a term-by-term evaluation and the
  kernel's error on the region. *Done when:* the spike's tests hold in the crate: ε on the
  region's grid, within 2ε at seeded points off it and in the point-to-point chain, every
  tₖ real, the panels converged, at the spike's κ and ε (debug: its five; release, ignored:
  its full set), and the generator's quadratures equal the spike's bit for bit at those κ;
  the result says when it stopped at a bound short of ε. *Needs:* CH6.1a.
- [x] **CH6.1c The pair chain.** Record: `docs/helmholtz/tasks/CH6.1c.md`;
  `nd_fmm_tables::helmholtz::plane_wave::pair`, tests
  `fmm-tables/tests/helmholtz/plane_wave_pair.rs`, CONVENTIONS §3.15 "Exponential M2L"
  (signed off 2026-11-03: as recommended, accepted as written). Within 0.76× the stated
  error of RCR (0.07 ε generated) on every offset and direction at degrees that resolve
  the multipole; below them the chain also errs on the truncated multipole's tail (the
  record's degree study). Acceptance read at those degrees signed off 2026-11-03: as
  recommended.
  `nd_fmm_tables::helmholtz::plane_wave`: the
  multipole to waves and waves to local in the orthonormal basis and stored form of §3.15,
  the six directions and their frames, the shifts, one pair translated directly along its
  direction; a CONVENTIONS §3.15 subsection for it, as §3.14 is for Laplace (unversioned),
  for sign-off. *Done when:* every V offset on several levels against RCR within the
  quadrature's error, with FMM3D's and with CH6.1b's generated quadratures, real and
  complex k, one and two multipole planes. *Needs:* CH6.1b. *Sign-off:* the convention.
- [x] **CH6.1d Merge and shift in the FMM.** Record: `docs/helmholtz/tasks/CH6.1d.md`;
  `nd_fmm_tables::helmholtz::plane_wave::merge`, `HelmholtzM2l::PlaneWave`,
  `PlaneWavePrecision`, the level call `fmm-exec/src/helmholtz/operator_plane_wave.rs`, tests
  `fmm-tables/tests/helmholtz/plane_wave_merge.rs`, `fmm-exec/tests/helmholtz/plane_wave.rs`
  and `check_plane_wave` in `fmm-exec/tests/helmholtz_fmm.rs`. Merge and shift is its pair
  path to 2.0e-16 (relative L2), the FMM's errors 1.000× and 1.013× Rotation's on the two
  test scenarios (CH6.1e measures CH3.1c's cases). `HelmholtzM2l::PlaneWave` (opt-in): the level
  call by merge and shift (CONVENTIONS §3.14's geometry, §3.15's conversions and shifts) on
  the levels that have a quadrature, Rotation on the others; the pair body CH6.1c's
  translation along the merge's direction. The precision per level a setting: a rule of
  p_l (provisional, from CH6.1c's degree study: the coarsest precision whose chain stayed
  within 1.1× RCR's error against the direct sum at every κ of the study, iprec 1 to
  p = 8, 2 to 12, 3 to 16, none above), or named per level for the measurements; a
  level whose κ lies outside the tables, or whose degree the rule leaves without a
  precision, keeps Rotation. *Done when:* merge and shift's shift factors the pair's
  to rounding on every direction, slot and child; the FMM with PlaneWave against the
  direct sum and against its pair path to rounding; deterministic at every thread count
  and on P ranks the one-rank FMM bit for bit; vector v the one-vector FMM bit for bit;
  the errors against Rotation's reported, not yet asserted (CH6.1e). *Needs:* CH6.1c.
- [x] **CH6.1e The quadrature per level, measured.** Report:
  `fmm-validate/results/helmholtz-ch6-1e.md`; `merge::precision_for_tolerance`,
  `HelmholtzOperator::with_plane_wave_tolerance`. iprec 2 at the FMM's tolerance ε ≥ 10⁻⁶
  and Rotation below (named degrees: iprec 2 to p = 15); within 1.070× Rotation's error on
  87 cases (CH3.1c's matrix and 27 at 10⁻⁴, 10⁻⁵, 10⁻⁷), asserted in the tests on small
  ones; every level at ε < 10⁻⁶ left to Rotation (iprec 3 1.12× at 10⁻⁷ and 1.76× at
  10⁻⁸ on the clustered trees' deep levels, iprec 1 1.15× at 10⁻³). The PR's four questions
  signed off 2026-11-03 (the user: "1–4 as recommended"): (1) the acceptance read as CH3.1c's matrix less
  the two large cases at 10⁻⁸ (Rotation there), asserted on the tests' small scenarios and
  the matrix in the report; (2) Rotation below 10⁻⁶, not iprec 3 with a κ guard; (3)
  CH6.1f before H6.G, CH6.2 first; (4) the waves buffer bounded and counted in the node
  budget before CH6.2 takes times (CH6.2's line). Split from CH6.1d on 2026-11-03 (the
  level call and the measurement with its report do not fit one PR, about 2,000 lines in
  three crates together). CH6.1's acceptance: the FMM's errors within about 1.1×
  Rotation's on CH3.1c's cases with each level's quadrature chosen with its degree p_l
  and ε by measurement, as §3.14's `nodes_for_degree` for Laplace, not by `hwts3e`'s
  precision from ε alone (which pairs ε = 10⁻⁸ with iprec 3, at the FMM's p ≈ 24 14–25×
  RCR's error on CH6.1c's study's charges); the rule a function of k, w_l, p_l and ε
  (constraint 9). Signed off 2026-11-03 (CH6.1c, question 3: as recommended): per
  level the coarsest precision whose chain error against the direct sum stays within about
  1.1× RCR's; a level where none does keeps Rotation, and the report names those levels
  (the study suggests 1.3–2.3× at p = 20–24 on worst-case charges, so H6's gain may shrink
  to the levels at p ≤ 16). *Done when:* the rule's constants set by the FMM's errors with
  each precision forced per level on CH3.1c's cases; the FMM within about 1.1× Rotation's
  error on every case, asserted in the tests; the report names the levels left to Rotation.
  *Needs:* CH6.1d.
- [ ] **CH6.1f Quadratures for ε below 10⁻⁶.** From CH6.1e (report §3, §7): no precision
  of FMM3D's tables keeps the FMM within 1.1× Rotation's error below ε = 10⁻⁶ (iprec 2's
  stated error is the FMM's own; iprec 3's waves of a degree-p multipole err on its
  truncated tail at p = 20–24, CH6.1c's degree study), so `HelmholtzM2l::PlaneWave` runs
  Rotation at D1's tight end. A quadrature matched to the degree, as CONVENTIONS §3.14's
  `nodes_for_degree` is for Laplace: CH6.1b's `generate` at the level's κ_l, ε and p_l
  (or a tail-aware compression of it), tabulated or built per level. *Done when:* the FMM
  within about 1.1× Rotation's error at ε = 10⁻⁷ and 10⁻⁸ on CH6.1e's cases, the clustered
  trees' deep levels included, with the rule still a function of k, w_l, p_l and ε
  (constraint 9); or the report shows no such quadrature fits the tables' budget.
  *Needs:* CH6.1e.
- CH6.2 Measured. Against Rotation and the table-based strategies on the three machines,
  and against `hfmm3d`'s exponential M2L. *Done when:* per-level accuracy and time beside
  FMM3D's on the same problems; D8 proposed. *Needs:* CH6.1 (a–e), CH1.4. Split on
  2026-11-03 into the bounded waves buffer (CH6.2a), which CH6.1e's sign-off 4 puts before
  any time is taken, and the measurements (CH6.2b), and on 2026-11-04 the measurements
  again into our strategies per level (CH6.2b) and FMM3D with D8 (CH6.2c); CH6.2 is done
  when all three are:
- [x] **CH6.2a The waves buffer bounded.** Report:
  `fmm-validate/results/helmholtz-ch6-2a.md`; `HelmholtzOperator::with_plane_wave_buffer`,
  `plane_wave_counts`, `MIN_BUFFER_BOXES`, `HelmholtzFmmBuilder::plane_wave_budget`,
  `DEFAULT_PLANE_WAVE_NODE_BUDGET`, `HelmholtzFmm::ranks_per_node`. The level call by chunks
  of whole parents, bit for bit at every bound (one checksum per case from the floor to the
  whole level, at 1 × 40 and 40 × 1 on Kathleen); 4 GiB per node 1.03–1.04× the whole
  level's time at one rank, 1.16–1.28× at 40 ranks per node, against 1.04–1.11× at 8 GiB,
  the default (signed off 2026-11-04). From CH6.1e (report §7; its question 4, signed
  off 2026-11-03): the level call's waves buffer, one frame of every box of the largest
  level with waves, takes 9.0–11.8 GB under the rule on W1 at kD = 500 and W3 at
  kD = 100 (22.9 GB for one level at iprec 3), against Rotation's few hundred MB. The
  level call by chunks of whole parents, each chunk's distinct sources' waves in a buffer
  of at most a byte bound per rank (at least one parent's 216 slots); the bound a node
  budget of `HelmholtzFmmBuilder` divided by the ranks per node (constraint 7), counted in
  `plane_wave_bytes`. *Done when:* the output bit for bit the same at every bound (the
  floor, a middle bound, the whole level), every thread count and on P ranks the one-rank
  output; `plane_wave_bytes` within the tables, the bound and the scratch; on W1 at
  kD = 500 and W3 at kD = 100 at 10⁻⁶ the bytes and the waves computed per box at the
  default and smaller bounds recorded, with the evaluation's time against the whole-level
  buffer (Kathleen). *Needs:* CH6.1e. *Machines:* Kathleen.
- [x] **CH6.2b The M2L timed per level.** Report:
  `fmm-validate/results/helmholtz-ch6-2b.md`; `HelmholtzOperator::m2l_times`,
  `helmholtz_accuracy --levels --evaluations`. Split on 2026-11-04 (the roadmap's "may split
  again") into our strategies timed per level (CH6.2b) and FMM3D with D8 (CH6.2c). The
  exponential M2L against Rotation and the table-based strategies (Batched, Auto) on the
  three machines: each level's M2L time from the Helmholtz operator's level calls (the
  slowest rank's), with its degree, V pairs and the strategy it ran, beside the
  evaluation's time and the errors against the direct sum. *Done when:* the per-level M2L
  times of Rotation, Batched (fused), Auto and PlaneWave (the rule) on CH6.1e's matrix at
  ε = 10⁻³ and 10⁻⁶ (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4 distributions
  at kD = 1 and 100), on the M6, locust and Kathleen, and on W1 at kD = 500 and W3 at
  kD = 100 on locust and Kathleen, each launch twice in reversed order; the fastest
  strategy per level and machine, and what a per-level choice would gain over each
  strategy alone. Also (CH6.2a, report §3–§4): at the default waves budget (8 GiB per
  node), whose halo recomputation cost 1.04–1.11× at 40 ranks per node on Kathleen (1.16–1.28× at
  4 GiB; a cut by spatial blocks in place of Morton runs would recompute less, if it
  matters), and the ranks × threads split on the large cases (one rank per node, CH3.6's
  split, one rank per core; W1 at kD = 500 evaluated in 39.7 s at 40 × 1 against 21.3 s
  at 1 × 40). Amended on 2026-11-04 (the user, "go ahead", Kathleen down: its login node
  refused every connection from 10:10): done on the M6 and locust, Kathleen's runs moved to
  CH6.2c, which runs on Kathleen anyway. *Needs:* CH6.2a. *Machines:* locust.
- [ ] **CH6.2c Beside FMM3D's exponential M2L; D8.** Per-level accuracy and time beside
  FMM3D's on the same problems: FMM3D's M2L per level from a timed build of `hfmm3d`
  (per-level timers in its main routine, a patch applied by `tools/fmm3d/build.sh`, outside
  the workspace), against CH6.2b's per-level times at matched ε; the per-level accuracy
  against FMM3D's (each level's waves alone, as CH6.1e's scan). *Done when:* per-level
  accuracy and time beside FMM3D's on the same problems; D8 proposed from CH6.2b's and
  this task's times (adopted as a strategy where it wins, else kept opt-in or dropped).
  Also (CH6.1a): CH6.1a's counts resolve the region's corner ρ = 4√2 at 1.31–1.51×
  FMM3D's waves where Im κ < π/2; whether a pair ever needs the corner at the FMM's ε, so
  fewer waves would do, is measured here. Also (CH6.2b, report §7): the exponential M2L's
  level call splits merge and shift over parents, so a coarse level (level 2: 8 parents)
  took 9–24× Rotation's time at 72 threads on locust (0.9–3.3× at 6 on the M6); D8's rule
  cannot read the threads (constraint 2), so before a default either the level call splits
  below the parent (by frame, direction or child) or the rule takes Rotation on levels of
  few boxes on every machine; and at ε = 10⁻³ a per-level choice among Batched, the
  exponential M2L and Rotation was up to 1.32× faster than any one strategy (CH6.2b's
  per-level tables are the input). Also (CH6.2b, moved on 2026-11-04 while Kathleen was
  down): CH6.2b's Kathleen runs, the x86 AVX-512 class and a two-socket node: fetch job
  628611's outputs (`matrix` and `large` at 1 × 40, completed;
  `/scratch/scratch/ucahtbe/fmm/logs/ch6-2b/`), run `large` at 4 × 10 and 40 × 1
  (`tools/helmholtz/ch6-2b-runs.sh`; the command in docs/helmholtz/tasks/CH6.2b.md,
  "Pending runs"), and add Kathleen's per-level tables to CH6.2b's report (§5) before D8 is
  proposed. *Needs:* CH6.2b, CH1.4. *Machines:* Kathleen. *Sign-off:* D8.
- [ ] **H6.G Gate.** `/milestone-gate H6`. *Needs:* every task above.

## H4: High-frequency levels

**Gate:** CH4.4 and CH4.5, the kD sweep and host performance against FMM3D.

- [ ] **CH4.0 The M2L rule's sparse-level term.** Added on 2026-11-02 at CH3.2d (the user,
  "Accept, follow-up"): CH3.2d's rule by degree and machine class batched a sparse W1 level
  at p = 15 on Kathleen at 1.106× the best choice (W1, kD = 30, ε = 10⁻⁶), past its 1.10
  margin, while the W4 cube at the same degree needs Batched. A per-level term from the
  global tree (the mean V-list length per box, counted once per box over the ranks, the
  replicated top included), so the rule stays rank-invariant (constraint 9). *Done when:*
  CH3.2d's sweeps within 1.10 on every case of the three machines, sparse levels included;
  deterministic and rank-invariant. Also, from the H3 gate (§5, signed off 2026-11-02): the
  Helmholtz dense budget counted per node, as Laplace's node rule does (constraint 7; it is
  1 GiB per rank since CH3.2a), before any multi-rank run with Auto as the default.
  *Needs:* CH3.2d. *Machines:* locust, Kathleen.
- [ ] **CH4.0b The worst-box concentration.** Added on 2026-11-02 at the H3 gate (question
  4a; CH3.6's sign-off 1, "a follow-up before H4"): on W1 at 10⁻⁶ and 10⁻⁸ CH3.6's worst
  box was up to 146× the all-target ℓ2 error (FMM3D 37×), pointwise up to 57 ε, in boxes
  where the sphere touches its bounding cube. Before the diagonal form adds a second source
  of error, on all-RCR trees. *Done when:* the cause located (level, list, operator or
  placement) with the box measure of proposal §9.1; whether a pointwise criterion joins
  §9.1's ℓ2 proposed. *Needs:* CH3.6. *Machines:* Kathleen. *Sign-off:* the criterion.
- [ ] **CH4.0c The degree rule beyond its measured range.** Added on 2026-11-02 at the H3
  gate (question 4b; CH3.3's sign-off 2): the high-frequency branch is measured to x = 65
  and checked at x = 108 only, and falls below S7's per-pair p* above that at ε ≤ 10⁻⁷.
  W1 at kD ≈ 1000 and 10⁻⁸ on all-RCR against the sampled direct sum; and proposal §9
  item 8's under-resolved W3 (N = 10⁶ at kD = 200 and 500), the FMM run H3 did not make.
  *Done when:* within ε, or the rule corrected or S7's degrees named per level above its
  range. *Needs:* CH3.6. *Machines:* Kathleen.
- [ ] **CH4.1 S → F and F → R by the transform as a product.** Per-box aligned signature
  buffers; source signatures kept per level call, target accumulators chunked under the node
  budget. *Done when:* round trips to the grid's bandwidth; bits independent of the chunk
  size; scratch within the budget on W1 at kD = 1000. Also, from the H3 gate (§5): the
  resident memory of the RCR FMM on W1 at kD = 500 at the calibrated degrees, the baseline
  the signature scratch is added to. *Needs:* CH3.1, D6.
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
- CH7.5 Plane-wave M2L on the device: Laplace's first. Added on 2026-11-05 at the user's
  request, splitting CH7.5: the device kernels of merge and shift (the complex products over
  a box's waves, the merge lists per parent, the shifts to the children), built first on
  Laplace, whose host strategy exists (Phase 6 T13, `M2lStrategy::PlaneWave`) and which on
  a device runs as `Dense` today ("PlaneWave, run as dense on the device",
  `fmm-exec/src/device.rs`); CH7.5d reuses them. The task T13's brief left for afterwards
  (docs/phase6/T13-plane-wave-m2l.md, "Afterwards"): T10 modelled it at 2.4–20× T7's dense
  M2L stage on the H100, larger at high p (`spikes/plane-wave/SPIKE_REPORT.md` §3), with the
  irregular merge lists the open device question. Split on 2026-11-05 (the user, "a/b/c,
  Helmholtz → d"; constraint 1 agreed: new kernels, the device's `PlaneWave` opt-in until
  CH7.5c's sign-off) into:
- [x] **CH7.5a The plane-wave kernels.** Record: `docs/helmholtz/tasks/CH7.5a.md`; tests
  `fmm-kernels/tests/kernels/translate/plane_wave.rs`. `nd-fmm-kernels` only (with read-only
  accessors of `nd_fmm_tables::PlaneWaveTables`), module `plane_wave`: the
  steps of `nd_fmm_tables::plane_wave` (CONVENTIONS §3.14) as kernels, one launch per step
  and level call (the outgoing waves per box and frame, the merge per parent and
  accumulator over a merge list built on the host, the gather with the child shifts, the
  local with the rotation back), the tables uploaded once from plain arrays, explicit `fma`
  chains, no atomics. *Done when:* each kernel bit for bit a host `mul_add` replica of its
  order and within a stated bound of the table function it restates, f32 and f64 on the
  CubeCL CPU runtime (CI), f32 on Metal (the M6, by hand), both on CUDA (locust, by hand),
  every layout; a level call over a full and an adaptive level against
  `PlaneWaveTables::m2l` per pair to rounding and against `nd_fmm_ref::direct::m2l` on
  levels 2, 9 and 16 within T13's quadrature bound (p ≤ 20 f64 in an ignored sweep); two
  launches bit for bit. No `nd-fmm-exec` change. *Needs:* Phase 6 T13 (merged).
  *Machines:* locust, the M6 (Metal).
- [ ] **CH7.5b The device's `PlaneWave`.** `nd-fmm-exec`'s device path runs M2L under
  `PlaneWave` on CH7.5a's kernels when a builder setting names it (otherwise "PlaneWave,
  run as dense on the device", as before): the merge lists from the V views at build, the
  waves buffer bounded by chunks of parents within the device's scratch budget, the
  launches in the formula of `tests/device_common`; the lists built in the order of CH7.5a's
  test (`Lists::new` in `fmm-kernels/tests/kernels/translate/plane_wave.rs`, the order of
  `src/operator_plane_wave.rs`), several vectors as columns, each chunk's buffers within
  the kernels' 32-bit index range (CH7.5a). *Done when:* `PlaneWave` on the device
  (CUDA f32 and f64, Metal f32, the CubeCL CPU runtime in CI) within 1.1× `Dense`'s errors
  against the direct sum at the C3.2 cube and the C3.3 Plummer sphere (T13's requirement 1,
  f64 p = 8, 12, 18 and f32 p = 8), device locals against the host `PlaneWave`'s within the
  stated bound, bit for bit from run to run and on P ranks against the one-rank device run;
  constraint 1: without the setting every strategy, the device's `PlaneWave` and Laplace's
  defaults bit for bit, the device `Auto` unchanged. *Needs:* CH7.5a. *Machines:* locust,
  the M6 (Metal).
- [ ] **CH7.5c Measured, and adopted or dropped.** The device M2L stage and the evaluation
  under CH7.5b's setting against T7's dense M2L at f64 p = 8, 12, 18 on locust's H100 (f32
  p = 8 on the M6's Metal), with the kernels' layouts measured (CH7.5a's `GPU_UNITS` = 128
  is a first choice; the merge's irregular lists are the spike's open question); Laplace's
  device gates and benchmarks within noise (D9). *Done when:* the report gives both; adopted as the device's `PlaneWave` where it wins at
  p ≥ 12 (T13's gate; the setting becomes the default of `PlaneWave` on the device), else
  `PlaneWave` stays `Dense` on the device and the kernels are dropped; the device `Auto`
  unchanged (a change to it is a sign-off). *Needs:* CH7.5b. *Machines:* locust, the M6.
  *Sign-off:* adoption on the device.
- [ ] **CH7.5d Exponential M2L on the device**, if D8 adopted it on the host. On CH7.5a's
  merge and shift kernels, with the Helmholtz conversions (CH6.1c's pair chain) and the
  quadrature per level (CH6.1e). *Done when:* as CH6.1 on the device. *Needs:* CH7.5c,
  CH6.2, CH7.1, D8. *Machines:* locust.
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
