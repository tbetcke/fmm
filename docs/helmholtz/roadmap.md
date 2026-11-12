# Helmholtz FMM roadmap

The tasks of `docs/design/helmholtz-fmm-proposal.md` (draft 0.8), Section 10, in the order
`/next-task` takes them. The proposal is the full record: each task's *Done when* condenses
its acceptance criterion in the §10 table, and where the two differ the proposal wins.
Progress is logged in `docs/helmholtz/progress.md`; each task's plan is
`docs/helmholtz/tasks/<ID>.md`; each milestone's gate report is
`docs/helmholtz/gates/<milestone>.md`.

Status (2026-11-12): **H3 and H6 closed** (`docs/helmholtz/gates/H3.md`, 2026-11-02;
`docs/helmholtz/gates/H6.md`, 2026-11-12). The RCR FMM on every level meets ε against FMM3D
(CH3.6). The exponential M2L is `Auto`'s default on the low-frequency levels at every
ε ≥ 10⁻⁸ (D8, signed off at CH6.2h and amended at CH6.1g and CH6.1h). It is within 1.1×
Rotation's error everywhere measured, the large cases at 10⁻⁷ and 10⁻⁸ included (H6 gate §8),
and the evaluation takes 0.05–0.91× the previous default's time at 10⁻⁶ and 0.27–0.83× at
10⁻⁷ and 10⁻⁸ (cases above 0.1 s). HR, H0, H1 and
H2 are still open; of H7, CH7.5a–c are done (Laplace's plane-wave M2L on the device, adopted
2026-11-06), and of H0, CH0.4 and CH0.6. Next: the priority list for H4 below, from CH4.0. The
history of this line until H6.G is in `progress.md` and in this file's history
(`git show 4b5b37f:docs/helmholtz/roadmap.md`).

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

## Priority list: the path to H4

Set by the user on 2026-11-12 at the H6 gate ("As proposed (Recommended)";
`docs/helmholtz/gates/H6.md` §5 and question 4), D6's order: with the low-frequency M2L done,
the far field at kD = 500 is mostly the high-frequency levels' RCR. It replaces the list of
2026-11-02, the path to H6, which finished with CH6.1h and H6.G. H4's follow-ups first, then
the signatures' convention and trusted path, then H4's tasks, then the FFTW track that H4.G
also needs.

1. CH4.0 The M2L rule's sparse-level term
2. CH4.0b The worst-box concentration
3. CH4.0c The degree rule beyond its measured range
4. CH1.3d The diagonal form in extended precision (CONVENTIONS §3.15 "Signatures", CH4.2's
   trusted path)
5. CH4.1, CH4.2, CH4.3, CH4.4, CH4.5
6. CH0.9 S9, CHR.3 the FFTW wrapper, CH4.6 the optional FFTW back end

Off the list, in file order: CH0.8b, CH0.5 and H0.G; HR.G; H1.G; CH2.1, CH2.2, CH2.3, CH2.5
and H2.G; H5, H7 (CH7.5d now has D8 and waits for CH7.1; CH7.5f before CH7.5e), H8, H9 (CH9.1,
after every other milestone).

## Decisions

The proposal's Section 12, plus D0. A decision is recorded here when the user takes it:
`signed off <YYYY-MM-DD>: <the answer as given>`.

| ID | Decision | Proposal's recommendation | Prepared by | Needed by | Status |
| --- | --- | --- | --- | --- | --- |
| D0 | Draft 0.8 of the proposal as the plan of record | as written; later changes by sign-off | review of draft 0.7 | every task | signed off 2026-10-26: as written |
| D1 | Target kD range and tolerances | for example kD ≤ 500 at 10⁻³ to 10⁻⁸ (proposal §12) | – | CH0.8b, CH3.0 | signed off 2026-10-31: kD ≤ 500 at 10⁻³ to 10⁻⁸, as recommended |
| D2 | Placement: modules in the layer crates (A) or separate crates (B) | A (§6.1) | – | CH1.2 | signed off 2026-10-29: A, modules in the layer crates, as recommended |
| D3 | f32 scope | f64 only until H7; then f32 for low-frequency configurations that pass an accuracy contract; high-frequency levels in f32 only if S6 finds stable classes | CH0.6 | CH7.1 | signed off 2026-11-06 (CH0.6, question 1; the user: "i agree with the recommendations"): as recommended, high-frequency levels may run the diagonal form in f32 at ε ≥ 10⁻⁴ for real k and weak damping, each class by the bound in u = 2⁻²⁴ with C_T = 0.18 and θ = 1/2, failing classes on RCR; f64 below 10⁻⁴ and for η ≈ 1; CH7.1 settles the f32 RCR and the low-frequency levels |
| D4 | Yukawa: its own operator type, or the wavenumber kind as a type parameter | one `HelmholtzOperator<T, K>` if CH2.1 shows no cost to the complex path | CH2.1 | CH2.2 | open |
| D5 | Source types of the first release | real and complex charges, potentials and gradients; dipoles once C6.4 exists for Laplace (note: Laplace dipoles were merged in Phase 6 T8 and T8d, so the condition is met) | – | CH1.1 | signed off 2026-10-29: charges now, dipoles later; §3.15 fixes real and complex charges with potentials and gradients, and dipoles get a later, separate task |
| D6 | Order of H4 and H6 | H6 first | H3.G | CH6.1, CH4.1 | signed off 2026-11-02 (H3 gate, question 2): "i agree with the recommendations", H6 first, as recommended; on the gate report's reason (§3.3): the calibrated degrees cut the low-frequency share of the far field at kD = 500 to about half (an estimate), and the FMM is slowest against FMM3D on low-frequency trees at tight ε, which H6 targets |
| D7 | Host back ends of the φ step | the product as default and reference; FFTW behind the optional `fftw` feature; rustfft if S9 finds it deterministic per box and faster, decided per machine | CH0.9 | CHR.3, CH4.6 | open |
| D8 | Exponential M2L as a default on low-frequency levels | after CH6.2 | CH6.2h | CH7.5d | signed off 2026-11-08 (CH6.2h, questions 1–5; the user: "1–5 as recommended"): `Auto` takes the exponential M2L by default where p_l ≥ d_pw (Batched's crossover + 1; 12 on AVX-512 with two planes), CH6.1e's rule gives a quadrature (ε ≥ 10⁻⁶) and Im κ_l < 5π, f64 only; no boxes term (Batched's few-box levels to CH4.0); proposal §5.3 and §6.4 amended. Amended 2026-11-11 (CH6.1g, question 2; the user: "Lift for 10⁻⁸–10⁻⁷ only (Recommended)"): `Auto` also takes it at 10⁻⁸ ≤ ε < 10⁻⁷, where the rule runs iprec 3 at the raised degree max(p_l, 32), and not at 10⁻⁷ ≤ ε < 10⁻⁶ until CH6.1h; proposal §5.3 amended. Amended 2026-11-11 (CH6.1h, question 2; the user: "Lift it (Recommended)"): `Auto` also takes it at 10⁻⁷ ≤ ε < 10⁻⁶, with the raised degree on the M2L alone there (`PlaneWaveRaise::Rule`); proposal §5.3 amended |
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
- [x] **CH0.6 S6: stability of the diagonal form.** Seeded by Appendix B.1. *Done when:* the
  S → F, diagonal, F → R chain in f64 and f32 on all 34 classes against extended-precision
  RCR and the direct sum, with realistic placements, k a 3 to 400, ε 10⁻³ to 10⁻¹⁰, P from the
  formula and from S7, Im k > 0, |t| = 3w as a control; C_T and θ; the classes on RCR per
  (k a, ε); the f32 verdict (D3); whether interpolating T keeps the bound; prototype times.
  Accepted when the bound never under-predicts by more than 2×. *Needs:* CH0.3, CH0.7.
  *Sign-off:* D3. Report `spikes/diagonal/SPIKE_REPORT.md` (results `spikes/diagonal/results/`;
  the raw rows archived in `~/projects/fmm-archive/locust/runs/ch0-6/`). Signed off
  2026-11-06 (questions 1–7 as recommended): D3, the proposal's §3.5 and §3.6 amended.
- [ ] **CH0.8b S8: workload census, final pass.** *Done when:* the first pass repeated with
  the calibrated p of CH3.3 (`DegreeRule::CALIBRATED`, not S7's p, on the low-frequency
  levels; amended at H3.G, proposal §7) and S6's constants, counting the exponential M2L's
  waves buffer and the raised stores of its levels below 10⁻⁶ (amended at H6.G, proposal §7);
  its numbers replace §7's. *Needs:* CH0.8a, CH0.6, CH0.7, D1.
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
  precision, as the reference of CH4.x; it writes CONVENTIONS §3.15 "Signatures" as S6's
  report proposes (its question 5). *Done when:* against CH1.3c's RCR on the same
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

Closed 2026-11-02 (`docs/helmholtz/gates/H3.md`). Its ticked lines were condensed at H6.G
(2026-11-12, the user: "Condense the roadmap"); each task's full line is in its record or
report and its progress entry (`docs/helmholtz/progress.md`), and in this file's history
(`git show ca938a5:docs/helmholtz/roadmap.md`).

- CH3.0 Builder: the leaf rule and complex densities. Split on 2026-10-31 into CH3.0a–c (the
  user, "Split a/b/c, proceed").
- [x] **CH3.0a nd-octree's maximum leaf width.** Record: `docs/helmholtz/tasks/CH3.0a.md`.
  `OctreeOptions::with_min_leaf_level`, default none; Laplace's outputs byte for byte (the
  user: "Bytes only, no D9").
- [x] **CH3.0b The Helmholtz builder's leaf rule.** Record: `docs/helmholtz/tasks/CH3.0b.md`.
  No leaf wider than c_leaf λ on W1–W4 over D1's range; W1–W4 in `nd-fmm-validate`.
- [x] **CH3.0c Complex densities.** Record: `docs/helmholtz/tasks/CH3.0c.md`. Complex charges
  in, split potentials and gradients out, one complex expansion per vector.
- [x] **CH3.5 The shared drivers extended to complex expansions.** Record:
  `docs/helmholtz/tasks/CH3.5.md`. Closed as a record on 2026-11-01 (the user, "Close as a
  record"): no driver change needed; the batched two-plane case moved to CH3.2.
- CH3.1 Complex planes and RCR on every level. Split on 2026-11-01 into CH3.1a–c (the user,
  "a/b/c").
- [x] **CH3.1a The Helmholtz operator.** Record: `docs/helmholtz/tasks/CH3.1a.md`.
  `HelmholtzOperator` on the shared drivers, real and complex k and charges, every body
  against `nd_fmm_ref::helmholtz`; Yukawa refused.
- [x] **CH3.1b The Helmholtz FMM.** Record: `docs/helmholtz/tasks/CH3.1b.md`. The point
  pipeline shared with `Fmm` (`src/pipeline.rs`); bit for bit across threads and ranks;
  Laplace bit for bit, minimal D9 on the M6 (the user).
- [x] **CH3.1c The acceptance runs.** Report: `fmm-validate/results/helmholtz-ch3-1c.md`.
  `helmholtz_accuracy`; 61 cases within ε, deterministic.
- CH3.2 Dense, Classes and Batched where the node budget allows. Split on 2026-11-02 into
  CH3.2a–c; CH3.2d added at CH3.2c's sign-off.
- [x] **CH3.2a Dense.** Report: `fmm-validate/results/helmholtz-ch3-2a.md`. Rotation to
  rounding, per pair faster only to p ≈ 6; opt-in.
- [x] **CH3.2b Classes.** Report: `fmm-validate/results/helmholtz-ch3-2b.md`. Dense to
  rounding, tables about 18× smaller, faster than Rotation per pair only to p = 4; opt-in.
- [x] **CH3.2c Batched, and the measurements per p.** Report:
  `fmm-validate/results/helmholtz-ch3-2c.md`. Its own block (the user: "Own Helmholtz block",
  "Split planes, 2n²"); fused, ahead of Rotation per pair to p ≈ 10 (M6, locust), 15 (Kathleen).
- [x] **CH3.2d The M2L rule per level.** Report: `fmm-validate/results/helmholtz-ch3-2d.md`.
  `HelmholtzM2l::Auto` by degree and machine class, the builder's default (signed off
  2026-11-02); best on 34 of 36 rows, one miss at 1.106 (CH4.0).
- [x] **CH3.3 Calibration of p_l over (k a_l, ε).** Report:
  `fmm-validate/results/helmholtz-ch3-3.md`. `DegreeRule::CALIBRATED` the default (the user:
  "Calibrate, new default", "Both branches", "ε/2, verify ≤ ε"); S7 an upper bound.
- CH3.4 Helmholtz P2P, real and complex k. Split on 2026-11-01 into CH3.4a–c.
- [x] **CH3.4a The vectorised sine and cosine.** Record: `docs/helmholtz/tasks/CH3.4a.md`.
  `nd_fmm_simd::sincos`, within 4 u_T for |θ| ≤ 2¹⁰ on every ISA.
- [x] **CH3.4b Helmholtz P2P on the skeleton.** Report:
  `fmm-validate/results/helmholtz-ch3-4b.md`. `evaluate_helmholtz`, contract (32 + 12|k|r) u_T;
  Laplace instruction for instruction, D9 on Kathleen within noise.
- [x] **CH3.4c The Helmholtz operator's P2P.** Record: `docs/helmholtz/tasks/CH3.4c.md`.
  The operator's P2P on the vectorised kernel, the reference kept as the trusted path.
- [x] **CH3.6 Gate: RCR on every level against FMM3D.** Report:
  `fmm-validate/results/helmholtz-ch3-6.md`. Ours within ε on all 57 case × ε points (largest
  0.63 ε), Kathleen jobs 617649–617651.
- [x] **H3.G Gate.** `/milestone-gate H3`. Report: `docs/helmholtz/gates/H3.md`. Closed
  2026-11-02 (the user, "i agree with the recommendations"): D6 signed off (H6 first); the
  proposal amended (C1–C6, its Appendix A); CH4.0b and CH4.0c added; the priority list for
  H6. *Needs:* every task above.

## H6: Exponential M2L on low-frequency levels (before H4 by default, D6)

**Gate:** the exponential M2L adopted where it wins, or kept opt-in (D8).

Its ticked lines were condensed at H6.G (2026-11-12, the user: "Condense the roadmap"); each
task's full line is in its record or report and its progress entry
(`docs/helmholtz/progress.md`), and in this file's history
(`git show 4b5b37f:docs/helmholtz/roadmap.md`).

- CH6.1 The §3.14 machinery generalised: six directions, per-level quadratures, merge and
  shift; errors within about 1.1× Rotation's. Split on 2026-11-03 into CH6.1a–e, and CH6.1f–h
  added (CH6.1e's question 3; CH6.1f split on 2026-11-09; CH6.1h by CH6.1g's sign-off 3).
- [x] **CH6.1a The quadratures.** Record: `docs/helmholtz/tasks/CH6.1a.md`. FMM3D's tables for
  iprec 1–3 with NIST's notice, our α counts by a Bessel bound to the corner; the kernel within
  1.50× and the chain within 1.18× the stated error on every domain.
- [x] **CH6.1b The generator.** Record: `docs/helmholtz/tasks/CH6.1b.md`.
  `nd_fmm_ref::helmholtz::plane_wave::generate`, the trusted path, bit for bit S4's.
- [x] **CH6.1c The pair chain.** Record: `docs/helmholtz/tasks/CH6.1c.md`. CONVENTIONS §3.15
  "Exponential M2L" (signed off 2026-11-03, as written); within 0.76× the stated error of RCR
  at resolving degrees.
- [x] **CH6.1d Merge and shift in the FMM.** Record: `docs/helmholtz/tasks/CH6.1d.md`.
  `HelmholtzM2l::PlaneWave` (opt-in); its pair path to 2.0e-16; deterministic, rank-invariant.
- [x] **CH6.1e The quadrature per level, measured.** Report:
  `fmm-validate/results/helmholtz-ch6-1e.md`. iprec 2 at ε ≥ 10⁻⁶, within 1.070× Rotation's
  on 87 cases; signed off 2026-11-03 ("1–4 as recommended").
- [x] **CH6.1f The degree-matched quadrature on the pair chain.** Report:
  `fmm-validate/results/helmholtz-ch6-1f.md`. The resolving degree recommended (candidate 4);
  signed off 2026-11-09 ("1–3 as recommended").
- [x] **CH6.1g The quadrature below 10⁻⁶ in the FMM.** Report:
  `fmm-validate/results/helmholtz-ch6-1g.md`. iprec 3 at max(p_l, 28) and max(p_l, 32); errors
  0.002–0.80× Rotation's on 26 cases; signed off 2026-11-11 (D8 amended; CH6.1h added).
- [x] **CH6.1h The resolving degree on the M2L alone.** Report:
  `fmm-validate/results/helmholtz-ch6-1h.md`. `PlaneWaveRaise::Rule` (`M2l` at 10⁻⁷, `Leaves`
  at 10⁻⁸); evaluation 0.43–0.83× and 0.31–0.71× Rotation's; D8 lifted at 10⁻⁷ (2026-11-11).
- CH6.2 Measured against Rotation, the table strategies and `hfmm3d`'s exponential M2L; D8
  proposed. Split from 2026-11-03 to 2026-11-08 into CH6.2a–h; CH6.2i added by CH6.2f's
  question 2.
- [x] **CH6.2a The waves buffer bounded.** Report: `fmm-validate/results/helmholtz-ch6-2a.md`.
  Chunks of whole parents under a node budget, 8 GiB by default (signed off 2026-11-04);
  bit for bit at every bound.
- [x] **CH6.2b The M2L timed per level.** Report: `fmm-validate/results/helmholtz-ch6-2b.md`.
  The M6 and locust: at 10⁻⁶ the exponential M2L the fastest on almost every level below the
  coarsest.
- [x] **CH6.2c CH6.2b's Kathleen runs.** Report: `fmm-validate/results/helmholtz-ch6-2b.md`
  (§2, §4, §5). 0.24–0.61× Rotation's M2L at 10⁻⁶; level 2 slow at 1 × 40 (CH6.2f).
- [x] **CH6.2d Beside FMM3D's exponential M2L.** Report:
  `fmm-validate/results/helmholtz-ch6-2d.md`. Per level FMM3D's over ours 0.44/0.89/1.11 at
  10⁻⁶ on Kathleen; W1 at kD = 500 22.4 s against 61.6 and 43.2 s.
- [x] **CH6.2e The counts' corner.** Report: `fmm-validate/results/helmholtz-ch6-2e.md`. The FMM
  held at reach 4.5; its adoption signed off 2026-11-07 and withdrawn by CH6.2g.
- [x] **CH6.2g The counts' reach adopted.** Report: `fmm-validate/results/helmholtz-ch6-2g.md`.
  **Not adopted**: the worst pair at reach 4.5 errs up to 2·10⁴× the stated error; the corner
  kept (the user: "Keep the corner (Recommended)").
- [x] **CH6.2f The coarse levels' merge and shift.** Report:
  `fmm-validate/results/helmholtz-ch6-2f.md`. `PlaneWaveSplit::Auto`, by child where parents
  leave threads idle; levels 2–3 0.38–0.88× Rotation's; signed off 2026-11-08.
- [x] **CH6.2h D8: the exponential M2L's level rule.** Report:
  `fmm-validate/results/helmholtz-ch6-2h.md`. D8 signed off 2026-11-08 ("1–5 as
  recommended"): `Auto`'s default; the M2L 0.03–0.70× the previous `Auto`'s at 10⁻⁶.
- [x] **CH6.2i Laplace's plane-wave M2L on the coarse levels.** Report:
  `fmm-validate/results/helmholtz-ch6-2i.md`. The split by child in Laplace's level call, bit
  for bit, D9 on three machines; signed off 2026-11-09 ("ok, 1–3 as recommended").
- [x] **H6.G Gate.** `/milestone-gate H6`. Report: `docs/helmholtz/gates/H6.md`. Closed
  2026-11-12 (the user: "Run first, then close (Recommended)", then all as recommended): the
  large cases at 10⁻⁷ and 10⁻⁸ verified (Kathleen job 633000: errors 0.45–0.84× Rotation's,
  the evaluation 0.27–0.65×); the proposal amended (C1–C8, its Appendix A); the priority list
  for H4; the roadmap condensed. *Needs:* every task above.

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
  Also (CH6.2h, report §6): Batched on levels of few boxes loses to Rotation at any degree
  (it reads each class's whole table for few pairs; level 2, 64 boxes, 3,096 pairs: at p = 7
  2.1–3.8× Rotation's time on the M6 and 8.5–9.3× on Kathleen, about 2 and 4 ms; at p = 15 on
  Kathleen 0.09–0.12 s against 3–5 ms, which D8's rule gives to the exponential M2L), so the
  term covers coarse levels as well as sparse ones, or Batched's level call splits as
  CH6.2f's did. CH6.2h's clean locust pass (report §5, 1 × 72) confirmed that at 72 threads
  Batched at p = 7–8 is behind Rotation on every level below about 3.5·10⁵ pairs, not only
  coarse ones (1.0–28×; up to 32 % of a W4 evaluation at 10⁻³), and ahead from about
  4·10⁵ pairs; the term must catch these levels too, within constraint 2 (it cannot read the
  thread count).
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
  back to RCR when the scratch budget binds. Also from S6: the damped classes measured in
  the FMM-weighted norm of CH1.3c, against the per-pair bound's factor
  e^{Im(k)(|t| + 2a)} (its question 7), and whether a shorter transfer function
  (⌈3(P − 1)/2⌉, whose rounding stays far below 2P − 2's at k a 6–45, its report §6) earns a truncation
  analysis. *Needs:* CH4.2.
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
  a device ran as `Dense` until CH7.5c ("PlaneWave, run as dense on the device",
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
- [x] **CH7.5b The device's `PlaneWave`.** Record: `docs/helmholtz/tasks/CH7.5b.md`;
  `fmm::DevicePlaneWave`, `FmmBuilder::{device_plane_wave, device_plane_wave_budget}`,
  `device::PlaneWaveReport`, `PlaneWaveScratch::covering`; tests
  `device_common::check_plane_waves`, `tests/mpi_exec.rs` (`check_plane_wave`), the gate in
  `tests/device_fmm.rs`. "The CubeCL CPU runtime in CI" met by hand for `nd-fmm-exec` (no
  CI job builds it with `cpu`; the kernels' tests run in CI): signed off 2026-11-05 as
  recommended. `nd-fmm-exec`'s device path runs M2L under
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
- [x] **CH7.5c Measured, and adopted or dropped.** Record: `docs/helmholtz/tasks/CH7.5c.md`;
  report `fmm-validate/results/helmholtz-ch7-5c.md`; `fmm::DevicePlaneWaveLayout`,
  `FmmBuilder::device_plane_wave_layout`, `device::{default_plane_wave_budget,
  CUDA_PLANE_WAVE_WIDE_UNITS, CUDA_PLANE_WAVE_WIDE_FROM, METAL_PLANE_WAVE_BUDGET}`,
  `layout_sweep --part planewave`. Adopted (signed off 2026-11-06: "sign off all three
  questions as recommended"): `DevicePlaneWave::Kernels` the default of `PlaneWave` on every
  device (`Dense` the named opt-out), cubes of 128 units on Metal and on CUDA to p = 12 and
  of 256 on CUDA from p = 13, Metal's plane-wave budget 1 GB, the kernels kept; the device
  `Auto` unchanged (CH7.5e); D9 by the device executables and gates on the CPU runtime,
  Metal and CUDA and one ABBA run of `nd-fmm-bench --backend cuda` against `main`.
  The device M2L stage and the evaluation
  under CH7.5b's setting against T7's dense M2L at f64 p = 8, 12, 18 on locust's H100 (f32
  p = 8 on the M6's Metal), with the kernels' layouts measured (CH7.5a's `GPU_UNITS` = 128
  is a first choice; the merge's irregular lists are the spike's open question) and the
  chunk budget (CH7.5b: `device_plane_wave_budget`, default the translation scratch
  budget; CH7.5b's gate printed, unmeasured, the cube at f64 p = 8 on the H100 slower than
  `Dense` and the Plummer sphere faster); Laplace's
  device gates and benchmarks within noise (D9). *Done when:* the report gives both; adopted as the device's `PlaneWave` where it wins at
  p ≥ 12 (T13's gate; the setting becomes the default of `PlaneWave` on the device), else
  `PlaneWave` stays `Dense` on the device and the kernels are dropped; the device `Auto`
  unchanged (a change to it is a sign-off). *Needs:* CH7.5b. *Machines:* locust, the M6.
  *Sign-off:* adoption on the device.
- [ ] **CH7.5d Exponential M2L on the device**, if D8 adopted it on the host. On CH7.5a's
  merge and shift kernels, with the Helmholtz conversions (CH6.1c's pair chain) and the
  quadrature per level (CH6.1e). *Done when:* as CH6.1 on the device. *Needs:* CH7.5c,
  CH6.2, CH7.1, D8. *Machines:* locust.
- [ ] **CH7.5f CubeCL 0.11.3 and cubek 0.3.1.** Added on 2026-11-06 (the user: "Please
  schedule right before CH7.5e"; off the priority list), so that CH7.5e's device baselines
  are taken on the new releases. The workspace pins `cubecl` `=0.11.0`, `cubek-matmul` and
  `cubek-std` `=0.3.0` move together to `=0.11.3` and `=0.3.1` (root `CLAUDE.md`, "Working
  rules"; `run-dependency-checks` lists them as exempt, in #193). The releases fix none of
  `docs/design/cubecl-upstream.md`'s entries; they bring LLVM for the native CPU on the CPU
  runtime (cubecl #1833, which may move the CPU runtime's results in the last bits and
  changes the `tracel-llvm` bundle, 23.1.0-3 to 23.1.0-4), a parked idle device server
  (#1806, launch latency), a CUDA PTX placeholder fix (#1691), loop conditions in their own
  scope (#1788) and the plane matmul's planes per cube floored at one (cubek #784).
  *Done when:* the pins and `Cargo.lock` moved, with the `nd-fmm-kernels` checks of root
  `CLAUDE.md` passing on the CPU runtime and Metal (the M6) and CUDA (locust); every
  device test and gate passing, any changed bit identity explained; Laplace's device
  benchmarks and CH7.5c's merge and shift times within noise of `main` (D9) on the M6's
  Metal and locust's H100; `cargo deny` clean; `cubecl-upstream.md`'s "0.11.0: still
  present" entries re-checked against 0.11.3, and the pin's sentences in root `CLAUDE.md`
  updated. *Needs:* CH7.5c. *Machines:* locust, the M6 (Metal).
- [ ] **CH7.5e The device `Auto` on `PlaneWave`.** Added on 2026-11-06 (CH7.5c's question
  2, signed off as recommended; off the priority list): the device's `M2lStrategy::Auto`
  resolves to `Dense` or `Rotation` by the static rule and never to `PlaneWave`, although
  `PlaneWave` on merge and shift took 0.19–0.21× `Dense`'s M2L time on the H100 at f64
  p = 12 and 18 (CH7.5c). At N = 10⁵ the device gate's evaluation times (printed, not
  asserted) put merge and shift behind `Dense` on the cube (11.8 against 3.7 ms at f32
  p = 8 and 13.4 against 10.2 ms at f64 p = 12 on the H100; 16.5 against 8.2 ms on Metal)
  and ahead on the Plummer sphere, so the rule needs N as well. Measure the device's
  strategies against each other by backend, precision, p and N (the static rule's crossovers and the near rule's cost models on the
  device, as Phase 6 T11 did for the host's `Auto`) and propose a rule. *Done when:* the
  report gives the device `Auto`'s rule by backend, precision and p with `PlaneWave` in it
  where it wins, adopted on sign-off; Laplace's device gates and benchmarks within noise
  (D9). *Needs:* CH7.5c, CH7.5f. *Machines:* locust, the M6 (Metal). *Sign-off:* the device
  `Auto`.
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

## H9: The final comparison with FMM3D

**Gate:** CH9.1, our FMM with every optimisation and default beside FMM3D, in depth.

- [ ] **CH9.1 The final comparison with FMM3D.** Added on 2026-11-07 (the user: "can you
  schedule an in depth comparison with fmm3d when all optimizations are applied?"), after the
  last optimisation: H4 to H8 closed and every default signed off (D7, D8, the device
  `Auto`). Our FMM with its defaults beside `hfmm3d` (the `default` and `fastker` builds,
  and CH6.2d's patched `timed` builds for the breakdown) on one node of the M6, locust and
  Kathleen, as Laplace's T9 (`fmm-validate/results/phase6-fmm3d.md`) and CH4.5, in depth:
  - accuracy: for each of our tolerances (10⁻³ to 10⁻⁸, D1) FMM3D at the same requested ε
    and at the eps of the closest error (matched accuracy); both against the same sampled
    direct sum, with the full-target crosscheck and the worst box;
  - the reference workloads W1–W4 (D12) over D1's range (kD ≤ 500), complex k and Yukawa
    where FMM3D meets its eps (CH2.5, CH5.3), real and complex charges, potentials and
    gradients, several vectors (FMM3D's vectorised interface), dipoles if D5's task has
    landed;
  - time: the build (FMM3D's setup) apart from the evaluation, and the evaluations after
    which ours is ahead; per stage and per level (FMM3D's from CH6.2d's patch: the M2L per
    level, lists 3 and 4, P2P), at one thread and on the whole node (FMM3D's OpenMP on every
    core, ours at the node's best ranks × threads split), each machine quiet for the whole
    run (no other jobs; locust's load sampled during the runs);
  - memory: the peak resident size per node;
  - the device path (CH7.6) beside the host's, and our multi-node runs (H8) labelled apart:
    FMM3D runs on one node.
  *Done when:* the report has these tables on the three machines; every case where FMM3D is
  more than 1.1× faster at matched accuracy is traced to its stage and level, with its cause
  or a follow-up task; CH4.5's and CH7.6's results cited where unchanged. *Needs:* H4.G, H5.G,
  H6.G, H7.G, H8.G. *Machines:* locust, Kathleen, the M6.
- [ ] **H9.G Gate.** `/milestone-gate H9`. *Needs:* every task above.
