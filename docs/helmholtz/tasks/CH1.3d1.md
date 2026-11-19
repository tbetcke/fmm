# CH1.3d1 The diagonal form in nd-fmm-ref

Started 2026-11-19 on `tbetcke/h-ch1-3d1`. Roadmap: "CH1.3d1 The diagonal form in nd-fmm-ref"
(H1; the first part of CH1.3d, item 8 of the priority list). Proposal: §10 CH1.3; §2, §3.6
(the bound E, its constants), §5.4 (signatures, the reflection classes); S6's report
(`spikes/diagonal/SPIKE_REPORT.md`, §1 and "The proposal for CONVENTIONS §3.15
'Signatures'", signed off at CH0.6 as its question 5).

## Split

CH1.3d is too large for one reviewable PR: a double-double scalar, the grid, the transfer
function, the classes and the operator (about 1,500 lines in `nd-fmm-ref`), and the revision
of CONVENTIONS §3.15 with a version bump, which needs the user's sign-off. Split on
2026-11-19 into

- **CH1.3d1** (this task): the diagonal form in `nd-fmm-ref`, on S6's proposed signatures,
  tested against CH1.3c's RCR; no convention changes.
- **CH1.3d2**: CONVENTIONS §3.15 "Signatures" written from CH1.3d1's code, with the rest of
  the roadmap's CH1.3d text (review M1, M3, M4, M10's §3.9; the proposal's §3.4 and §3.1,
  M2), one `CONVENTION_VERSION` bump, proposed in the PR for sign-off.

## Goal

The S → F, diagonal and F → R chain of proposal §5.4 as a reference M2L in
`nd_fmm_ref::helmholtz::diagonal`, from a stored multipole to a stored local, its transfer
functions computed in double-double and rounded once, the trusted path of CH4.1 and CH4.2.

## Acceptance (proposal §10, verbatim)

CH1.3's row: "addition theorems; the k → 0 limit with the tolerances of Section 9; Yukawa two
ways; the direct sum against the FMM3D fixtures of CH1.4 to round-off". CH1.3a–c met it; the
diagonal form has no item there (the roadmap's CH1.3 note). Roadmap, CH1.3d's *Done when*:
"against CH1.3c's RCR on the same class within the bound E of §3.6".

**Reading of "within the bound E".** §3.6's E is fitted so that the measured rounding stays
within 2E (C_T = 0.153 rounded to 0.15; CH0.6's acceptance "the bound never under-predicts by
more than 2×"), and the rule's guarantee is E ≤ θ ε ⇒ error ≤ ε with θ = 1/2. The tests
assert measured ≤ 2E = E/θ and report the largest ratio measured/E.

## Read first

- Proposal §3.6, §5.4; S6's report §1, §3, "The proposal for CONVENTIONS §3.15".
- `spikes/diagonal/src/{grid,signature,transfer,classes}.rs`, `spikes/truncation/src/{real,
  complex,radial}.rs` (`Dd`, `hankel`), `spikes/truncation/src/coaxial.rs`
  (`gauss_legendre`).
- `fmm-ref/src/helmholtz/rcr.rs` (the stored M2L: 4π ik e^{2β − Im(k)|t|}),
  `fmm-ref/src/helmholtz/leaf.rs`, `nd_fmm_math::orthonormal::legendre`.

## Changes

`nd-fmm-ref` only: a new module `helmholtz::diagonal` (a directory), no change to an existing
function, no manifest change, no convention change.

- `dd` (private): a double-double scalar and its complex numbers, ported from S7's `Dd`
  (`spikes/truncation/src/real.rs`, `complex.rs`): the arithmetic, `sqrt`, `exp`, `sin_cos`,
  π; the Hankel functions hₗ(z) by the upward recursion (S7's `radial::hankel`); the
  Gauss–Legendre rule by Newton's method (S7's `gauss_legendre`).
- `Grid<T>`: S6's grid for degree p (P = p + 1): 2P − 1 Gauss–Legendre nodes in cos θ,
  ascending, N_φ = 4P − 3 rounded up to a multiple of 4, sample (i, j) at i N_φ + j, the weights
  wᵢ 2π/N_φ, the azimuth tables and each ring's normalised Legendre values; computed in
  double-double (nodes, weights, azimuths) and rounded to T, the directions kept in
  double-double for the transfer function.
- `s2f` and `f2r` (public, for CH4.1's tests): the stored multipole to the signature
  F(ŝ) = Σₙ (−i)ⁿ sⁿ Σₘ M̂ₙᵐ Yₙᵐ(ŝ), and a signature product G back to the stored local,
  L̂ₙᵐ += sⁿ iⁿ Σᵢⱼ wᵢ (2π/N_φ) Yₙᵐ(ŝᵢⱼ) G(ŝᵢⱼ); per ring a Legendre step per order and the
  DFT along the ring as a product (S6's `s2f_ring`, `f2r_ring`).
- `Transfer<T>`: the transfer function of an offset t = c_t − c_s on the grid, in the stored
  normalisation T̂(ŝ) = e^{Im z} Σ_{l ≤ 2P − 2} (2l + 1) iˡ hₗ(z) Pₗ(ŝ · t̂), z = k|t|, as RCR's
  (S|R) table carries e^{Im z}: each sample in double-double and rounded once.
  `Transfer::permuted` reads a class representative's T̂ through an offset's sample
  permutation.
- `classes()`, `class_of(offset)`, `Symmetry` (`apply`, `sample`): S6's 34 classes of the 316
  offsets and their permutations.
- `m2l`: L̂ += 4π ik e^{2β − Im(k)|t|} F2R(T̂ · S2F(M̂)), the signature of RCR's `m2l` with the
  grid and the transfer function in place of the coaxial table, one degree p for both ends
  (the diagonal form is square), checking the frames, degrees and lengths.
- `bound`: E(l, c) of §3.6 with its constants per variant (f64: C_T 0.15, C₀ 200; f32: 0.18,
  50; interpolated T: 0.23), the Hankel term in log space (S6's `log10_term`) so that it stays
  finite outside the exponent range.
- `diagonal::Workspace<T>` for the signatures and the ring scratch, so that `m2l` allocates
  nothing (the signature of one box is (2P − 1) N_φ complex values, 2.2 MB at p = 131; the
  common `helmholtz::Workspace` stays as it is).
- `fmm-ref/CLAUDE.md`: the module's line.

Interpolated T (§5.4's O(p²) build, C_T = 0.23) is not built: CH4.2 decides whether to use
it; the bound's constant is there for it.

## Tests that prove it

Unit tests in the module (the private double-double), integration tests in
`fmm-ref/tests/helmholtz/diagonal.rs`. Error measure: the relative ℓ2 norm over the targets of
φ̃ and of ∇φ̃ after `leaf::l2p` (the larger of the two), as S6 measured.

1. **Double-double**: `two_sum`/`two_prod` exact; `sqrt`, `exp`, `sin_cos`, π against known
   values to 10⁻³⁰; hₗ(z) against `nd_fmm_math::radial::hankel_h` in f64 to 10⁻¹⁴ relative
   where f64's is accurate, and the Wronskian-free check h₀, h₁ closed forms.
2. **Grid**: N_φ a multiple of 4; the nodes symmetric bit for bit (u_{2P−2−i} = −uᵢ); the
   quadrature exact: Yₙᵐ orthonormal on the grid for all n, n′ ≤ 2P − 2 to 10⁻¹³,
   p ≤ 10 (the Gram matrix of (2p + 1)² harmonics; larger p only through the M2L tests).
3. **S → F → R**: f2r(s2f(M̂)) = M̂ (with G = F) to 10⁻¹³ relative, for s = 1 and s < 1.
4. **The signature's normalisation**: for one charge q at y in the box,
   4π F(ŝ) = q e^{−ik ŝ·(y − c)} e^{−β}-scaled to the truncation, at k a = 8, p = 30.
5. **Transfer function**: T̂ in double-double against the same sum in f64 where it is benign
   (|t| = 3w on the axis, k a = 3) to 10⁻¹⁴ relative to max|T̂|; T̂ through each of the 316
   offsets' permutations equals T̂ computed directly for the offset, to 2 ulp of max|T̂|.
6. **Acceptance: against RCR within E** — `rcr::m2l` and `diagonal::m2l` from the same P2M of
   random complex charges (uniform in the source box, and at its corners), L2P at random
   targets of the target box: measured ≤ 2E on all 34 class representatives, for real k at
   k a ≈ 2.83 (p = 7, S6's worst ratio), 8 and 16, and complex k = κ(1 + 0.1i) at k a = 8,
   p from §3.5's formula at ε = 10⁻⁶ (and 10⁻⁸ in a build without debug assertions, so not
   in CI); the largest ratio printed.
   `#[ignore]`d: k a = 32 and 54 (level 3 of W1 at kD = 500, p = 72 and 82), the release run
   by hand.
7. **All 316 offsets** through their class's permuted T̂, at k a = 5 (p = 16), against RCR
   within 2E.
8. **Against the direct sum**: the diagonal chain against `p2p::direct_sum` within RCR's
   truncation error plus 2E on the passing classes.
9. **f32**: the chain in f32 (T̂ rounded from double-double to f32) against f64 RCR within
   2E with u = 2⁻²⁴, C_T = 0.18, C₀ = 50, at k a = 16.
10. **Panics**: a grid or transfer function of another degree, wrong slice lengths, frames off
    the level, a zero shift.

## Checks

`/verify` (default and workspace Clippy, docs, the default and workspace tests at opt-level 1,
the MPI list), `cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`, `cargo test -p
nd-fmm-ref`, the ignored diagonal tests in release by hand.

## Machine runs

None (no timing; the M6 runs the release tests, which is not a timing run).

## Constraints and risks

- Constraint 1: nothing of Laplace's is touched (a new module of `nd-fmm-ref`'s `helmholtz`).
- Constraint 6: this is the trusted path of CH4.1 and CH4.2; it is itself checked against
  RCR and the direct sum.
- Risk: f64 RCR has rounding of its own on the nearest classes ((S|R) carries the same
  growing hₗ); S6's reference was RCR in double-double. If RCR's error dominates the
  difference, the test also compares both chains with the direct sum and records it.
- Risk: double-double T at p ≥ 72 is slow in a debug build; those cases are ignored and run
  in release.

## Results

Measured on the M6 (release for the large degrees; the others also in the opt-level 0
build), `cargo test -p nd-fmm-ref --release --test helmholtz diagonal -- --include-ignored
--nocapture`; `DIAGONAL_DETAIL=1` prints every class.

| k a | η | p | largest measured/E | classes with E < 10⁻⁶ |
| --- | --- | --- | --- | --- |
| 2.83 | 0 | 7 | 1.227 | 34 |
| 8 | 0 | 27 (formula, 10⁻⁶) | 0.286 | 18 |
| 8 | 0.1 | 27 | 0.065 | 15 |
| 16 | 0 | 36 (10⁻⁶) | 0.661 | 29 |
| 8 | 0 | 33 (10⁻⁸; release) | 0.204 | 5 |
| 16 | 0 | 41 (10⁻⁸; release) | 0.591 | 25 |
| 32 | 0 | 61 (10⁻⁸; ignored) | 1.187 | 29 |
| 54 | 0 | 72 (calibrated; ignored) | 1.994 | 34 |
| 54 | 0 | 82 (formula, 10⁻⁶; ignored) | 1.958 | 32 |

- The 316 offsets through the permutations (k a = 5, p = 16): measured/2E at most 0.48.
- f32 (k = 18.5, k a ≈ 16, p = 30, the formula at 10⁻⁴): measured/2E at most 0.062.
- The permuted transfer functions equal the direct ones within 2 ulp of the largest sample
  on all 316 offsets; T̂ against an f64 sum where nothing cancels to 10⁻¹³.
- Against the direct sum (k a = 8, p = 27): within RCR's truncation plus 2E on every class;
  RCR's own truncation there reaches 1.4·10⁻⁶ (the corners of both boxes, nearest class).
- **The floor.** At k a = 54 the ratios near 2 are on the farthest classes, where E is the
  floor C₀ u ≈ 2.7·10⁻¹⁴, and the reference is f64 RCR, while S6 measured against RCR in
  double-double. Against a third path, `direct::m2l` (projection), on six classes
  (`at_the_floor_against_the_projection`, ignored): the diagonal form errs 1.8–2.9·10⁻¹⁴ and
  RCR 3.4–6.9·10⁻¹⁴; the nearest class (0, 0, 2) with E = 1.0·10⁻¹¹ errs 1.4·10⁻¹². So the
  diagonal form meets E there; the margin of 2E against RCR is RCR's rounding. A note for
  CH4.2 in its roadmap line.
- Debug time (opt-level 0, M6, load 3.8–4.9): the `helmholtz` test executable 47 s with and
  without the diagonal tests (6.2 s alone, in parallel with the longer ones).

## Self-review

A subagent (the plan, §2, §10, the branch): nothing blocking. Taken: the T̂-against-f64 test
(plan test 5's first half), the plan's grid range (p ≤ 10) and degrees (from the formula,
10⁻⁸ outside debug builds), the module's panic sentence on the workspace, the documentation
that a transfer function does not record k and is not finite far outside the regime, and the
port's one change (the middle Gauss–Legendre node). Left: f32 is rounded double-double →
f64 → f32 (one f32 ulp at most, as S6); the 0.3 % margin at k a = 54 is in an ignored test.

## Pending runs

None.

## Open questions

None. The PR asked two, signed off 2026-11-19 by the user on #228 ("Agree to both questions"):

1. "Within the bound E" read as measured ≤ 2E: agreed.
2. The split into CH1.3d1 and CH1.3d2: agreed.
