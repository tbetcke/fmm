# CH0.6 S6: stability of the diagonal form

Started 2026-11-06 on `tbetcke/h-ch0-6`. Roadmap: "CH0.6 S6: stability of the diagonal
form" (H0; off the priority list, named by the user). Proposal: §10 CH0.6; §3.5, §3.6,
§5.4, §5.6, §6.5 (f32), §8 R5, R8, R9, R10, Appendix B and its seed script B.1; §12 D3.

## Goal

Measure, in full 3D with realistic sources, how far the rounding of the diagonal M2L of
§5.4 (S → F, the transfer function, F → R) carries its error at each offset class, level
(k a), tolerance and precision, and fit and test the bound E of §3.6 that decides per
(level, class) between the diagonal form and RCR.

## Acceptance (proposal §10, verbatim)

> the S → F, diagonal, F → R chain of Section 5.4 in f64 and f32, transfer functions from
> extended precision rounded to the working precision, for every one of the 34 classes,
> against extended-precision RCR and the direct sum. Realistic conditions: sources and
> targets filling the boxes (uniform, concentrated at corners, and surface patches cut from
> W1), multipoles formed by P2M from those points rather than synthetic coefficients, k a
> from 3 to 400, ε from 10⁻³ to 10⁻¹⁰, P from the formula of Section 3.5 and from S7's
> tables, Im k > 0 up to the pruning cut, and |t| = 3w as a control. Outputs: C_T and θ
> with the largest observed under-prediction; the table of classes on RCR per (k a, ε);
> the f32 verdict (D3); whether interpolating T in ŝ · t̂ keeps the bound; prototype times
> of diagonal and RCR per class. Accepted when the bound never under-predicts the measured
> error by more than 2× over the sweep, and the reading of Gumerov and Duraiswami is
> confirmed or Sections 3.5 and 5.4 corrected

(Proposal §10, row CH0.6, verbatim.)

## Read first

- Proposal §3.6 (the bound, the rule, f32, complex k), §5.4 (the scheme, the grid, the
  reflection classes), Appendix B and the seed script B.1, §3.5 (the formula; the
  calibrated rule of CH3.3).
- `spikes/truncation/` (S7): `src/real.rs` (`Real`, `Dd`), `src/complex.rs`,
  `src/radial.rs`, `src/harmonics.rs` (the real orthonormal basis, p̄ₙᵐ),
  `src/chain.rs` (P2M, L2P, the direct sum, RCR, the error norm, the kernel's
  normalisation), `src/coaxial.rs` (`gauss_legendre`), `src/geometry.rs` (offsets,
  placements, charges), `results/summary.md` (p* per kind, the closed form).
- `fmm-exec/src/helmholtz/degree.rs` (`DegreeRule::CALIBRATED`, the FMM's rule).
- CONVENTIONS §3.15 "Signatures" (reserved; what is proposed so far) and "Precision and
  range".

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- `spikes/diagonal/` (new, package `nd-fmm-spike-diagonal`), a workspace member outside
  `default-members` like the other spikes. Dependencies: S7's spike by path
  (`nd-fmm-spike-truncation`, for Dd, the basis, P2M/L2P, RCR, the direct sum and the
  placements, so that the reference is S7's double-double chain unchanged); no other
  dependency (threads by S7's `calibrate::parallel_map`; as built, `rayon` was not needed).
  - `src/grid.rs`: the signature grid of §5.4 for P = p + 1: 2P − 1 Gauss–Legendre nodes in
    cos θ, N_φ = 4P − 3 padded to a multiple of 4, φ_j = 2πj/N_φ; p̄ₙᵐ(cos θᵢ) per ring
    in double-double, rounded to the working precision.
  - `src/signature.rs`: with Yₙᵐ the basis of S7 (`harmonics`) and the kernel
    G = ik Σ jₙ hₙ Σₘ Y Y,
    - S → F: F(ŝ) = Σₙ (−i)ⁿ Σₘ Mₙᵐ Yₙᵐ(ŝ), by a Legendre step per order and a direct
      DFT per ring (the transform as a product, §6.7's default);
    - the diagonal step G(ŝ) = T(ŝ) F(ŝ) per sample;
    - F → R: Lₙᵐ = iⁿ ∫ Yₙᵐ(ŝ) G(ŝ) dŝ by the grid's quadrature (the transpose steps);
    - from G(x − y) = (ik/16π²) ∫ e^{ikŝ·(x−c_t)} T(ŝ) e^{−ikŝ·(y−c_s)} dŝ, which the
      seed uses, and e^{ikŝ·r} = 4π Σ iⁿ jₙ Σₘ Y(r̂) Y(ŝ) for complex k as well;
    - all generic over S7's `Real` (f32, f64, Dd).
  - `src/transfer.rs`: T_L(ŝ) = Σ_{l≤L} (2l + 1) iˡ hₗ(k|t|) Pₗ(ŝ · t̂) at every sample, in
    double-double (hₗ by S7's `radial::hankel`, Pₗ by the forward recurrence), rounded to
    the working precision; snapshots at several L in one pass (below). The bound
    E = C_T |k||t| u maxⱼ≤L (2j + 1)|hⱼ(k|t|)| (§3.6). Interpolation of T in ŝ · t̂: a table
    of T on a uniform grid in ψ = arccos(ŝ · t̂) (oversampled, in double-double, rounded
    to f64) and local Lagrange interpolation in f64 at the samples, the O(p²)-per-class
    build that §5.4 and R10 ask about.
  - `src/classes.rs`: the 34 reflection classes of the 316 offsets (x → −x, y → −y,
    z → −z, x ↔ y), a representative per class and the map offset → (class, symmetry);
    the symmetry's index permutation of the grid.
  - `src/run.rs`: one unit of the sweep, (kind, k a, p): sources and targets by S7's
    placements and charges (128 and 64 per box, seeded), the multipole by S7's P2M in
    double-double at the source box's centre, unscaled; per class representative t:
    the reference local by S7's RCR in double-double (rotation, (S|R), rotation back) and
    the direct sum in double-double; the diagonal chain in f64 and f32 (the multipole and
    T rounded to the working precision, S → F, T F, F → R in it, the local back to
    double-double, L2P in double-double); per placement the relative ℓ2 errors of φ and
    ∇φ over the targets (§9.1) of: RCR against the direct sum (truncation), the diagonal
    against RCR (its rounding), the diagonal against the direct sum (total).
  - `src/rules.rs`: P from §3.5's formula (at Re(k) a; S8's `degree`), from S7's p*
    (`spikes/truncation/results/summary.md`, the per-level degree per kind; the closed form
    off its grid, above k a = 256, marked as extrapolated) and from CH3.3's
    `DegreeRule::CALIBRATED` (the FMM's rule, copied with a citation; an addition to the
    acceptance's two, since it is what the FMM now uses and a smaller P widens the stable
    range, §3.5).
  - `examples/sweep.rs`: the sweep in parallel units (rayon), rows to `results/runs.csv`;
      `examples/summary.rs`: the fit of C_T and θ, the tables, to `results/summary.md`;
    `examples/times.rs`: the prototype times.
  - `tests/`, `SPIKE_REPORT.md`, `seed.py` (Appendix B.1 verbatim, the 1-D model this
    spike replaces; run once to record its numbers for a test).
- `spikes/truncation/src/real.rs`: `impl Real for f32` (additive; S7's code and results
  unchanged), so that the working-precision chain runs in f32 from the same source.
- Root `Cargo.toml` `members` gains `spikes/diagonal`; `Cargo.lock` its entry.

### The variants

- **Precision**: f64 and f32, each with T rounded from double-double at every sample.
- **T's bandwidth** L: 2P − 2 (the proposal's reading: the diagonal form is then exactly
  the truncated M2L that RCR applies, §3.6), and as alternatives ⌈3(P − 1)/2⌉ and P − 1
  (a transfer function of the signatures' bandwidth, the classical choice), all from one
  pass; the total error says which meets ε and how stable each is. This tests the reading
  of Gumerov and Duraiswami (2009): their reported accuracy at 10⁻⁴ (a prescribed 10⁻⁴
  gave about 10⁻⁶) against what each bandwidth gives at their level switch.
- **T interpolated** (f64, L = 2P − 2): against T rounded per sample. As built: a table
  oversampled σ = 3 in ψ, Gaussian-regularised sinc with 36 terms on each side, the
  position on the grid from ψ in double-double (atan2 in f64 corrected), since ψ in f64
  alone costs T about L u |T|.
- **T built in f64** (f64, L = 2P − 2; added while building): the terms rounded to f64,
  Pₗ and the sum in f64, μ from the f64 directions, as a host table build in working
  precision would (the seed's model). The seed test showed the two builds differ (the
  rounding of the terms against that of the values), and the bound of §3.6 was fitted to
  the seed's.

### The sweep

- Kinds: real k; complex k = κ(1 + iη) with η = 0.1 and η = 1 (S7's), each up to the
  pruning cut per ε: Im(k) w ≤ ln(1/ε) (§5.5's test with θ_D = 1 and a leaf far below,
  the conservative end; beyond it the level carries no V list). Yukawa has no
  high-frequency regime (§5.5) and is not run.
- k a on S7's half-octave grid 2^{j/2}, j = 3..17 (2.83 to 362), and 400.
- ε = 10⁻³ to 10⁻¹⁰ by decades; P = p + 1 from each of the three rules; the unit of work is
  each distinct (kind, k a, p), since the errors depend on p and not on ε.
- Per unit: the 34 class representatives (including (3, 0, 0), |t| = 3w, the control) ×
  3 placements × {f64, f32} × 3 bandwidths, plus f64 interpolated.
- Expected cost: T dominates, about 16 P³ double-double multiply-adds per class, about
  a minute per class at P ≈ 480; the sweep is about 20–40 core-hours, run on locust's 72
  Grace cores (no timing in it).

### Outputs (the report)

1. Measured rounding (diagonal − RCR) and total (diagonal − direct) per class against k a,
   ε, rule; the worst placement and norm.
2. **C_T and θ**: C_T fitted to the f64 rounding errors (L = 2P − 2) where they lie above
   the chain's floor; the largest under-prediction (measured / E) over the sweep; θ =
   1/(that ratio) so that E ≤ θ ε implies a measured error ≤ ε. If the data need another
   form (a floor term in u, a class-direction factor, a damping factor e^{c Im(k) a}), the
   report proposes it, a correction of §3.6 for sign-off.
3. **Classes on RCR per (k a, ε)** for each rule, by the bound with S6's constants and by
   the measurement, and the share of V pairs (interior box) on RCR; §3.6's table at
   kD = 500 redone.
4. **The f32 verdict (D3)**: any (k a, class) where f32 meets ε at 10⁻³ or 10⁻⁴.
5. **Interpolated T** against per-sample T: the bound kept or not.
6. **The reading of Gumerov and Duraiswami**: the 3D numbers at §3.6's points (kD = 500,
   10⁻⁶, levels 2–5, |t| = 2w and 3w) against the 1-D seed's, and the bandwidth variants;
   confirmed, or §3.5 / §5.4 corrections proposed for sign-off.
7. **Prototype times** per class (f64, one thread): a diagonal pair (N_l complex
   multiply-adds), S → F and F → R per box, RCR per pair (S7's f64 rotation, coaxial step,
   rotation back, the table built once per class), at several p.
8. A proposal for CONVENTIONS §3.15 "Signatures" (the grid, the normalisation of F and the
   transforms, the bandwidth, the class permutations), for sign-off; not written into
   CONVENTIONS here.

## Tests that prove it

`cargo test -p nd-fmm-spike-diagonal --release -- --include-ignored`.

1. **f32**: S7's `Real` for f32 against `std` (sqrt, exp, sin_cos to a few ulps).
2. **The grid**: Gauss–Legendre and the φ grid integrate Yₙᵐ Yₙ′ᵐ′ to δ for n + n′ ≤
   4P − 4 (orthonormality), ≤ 10⁻²⁸ in double-double; N_φ a multiple of 4.
3. **S → F → R is the identity** with T ≡ 1 (Lₙᵐ = Mₙᵐ), double-double ≤ 10⁻²⁸, f64
   ≤ 10⁻¹⁴ relative.
4. **The signature is the far field**: F of a P2M multipole against
   Σ q e^{−ikŝ·(y − c)} at the samples, within the truncation (double-double, P large).
5. **The diagonal form is RCR**: in double-double with L = 2P − 2, the diagonal chain's
   local against S7's double-double RCR local, ≤ 10⁻²⁰ relative, on several classes,
   real and complex k, k a up to 32 (the claim of §3.6 that it adds no truncation error).
6. **The diagonal form is the kernel**: the double-double diagonal field against the
   direct sum within S7's truncation error at the same p.
7. **Classes**: 34 classes, sizes summing to 316; each offset's T (double-double, rounded
   to f64) equals its representative's T read through the permutation within 1 ulp of
   max |T|, at two (k a, p).
8. **The 1-D seed reproduced**: the spike's code on the seed's geometry (sources and
   targets on the axis) gives the seed's errors (run once with `seed.py`, recorded)
   within 1.5×, and the seed's bound values to 10⁻¹².
9. **The bound's Hankel maximum** against an independent evaluation (log-space downward
   from S7's `hankel`, and at points of S3's fixture).
10. **Interpolated T** against double-double T within (Lebesgue factor) × u × max |T| at
    several (k a, p, class).
11. **Determinism**: a unit's rows bit for bit identical at 1 and several threads.

## Checks

`/verify` (the spike is a workspace member: format, Clippy workspace all-targets, `cargo
doc`, the workspace tests; constraint 1 holds trivially: no Laplace path changed). The
spike's tests in release with `--include-ignored`. `cargo deny --locked check licenses`
(the lock file changes).

## Machine runs

- locust (CPU only, 72 cores; `tools/gh200/sync.sh`, `tools/gh200/remote.sh`): the sweep
  (`cargo run -p nd-fmm-spike-diagonal --release --example sweep`), checked with `uptime`
  first, no timing; results fetched before any re-sync. Decides every output but 7.
- The prototype times (output 7): one thread, release, on locust when its load is low
  (`uptime`, `nvidia-smi`), or on the M6 if the user agrees. Decides output 7 only.
- No Kathleen job; D9 does not apply (no Laplace code or default changes).

## Constraints and risks

- Constraint 1: Laplace not touched (a spike, a member entry, the lock file, an additive
  impl in S7's spike).
- Constraint 5 / 8: the spike rounds tables from double-double as §6.2 stores them; f32
  is measured, not adopted (D3).
- Constraint 9: the bound uses k, w_l, P_l and the class only.
- Risks: the sweep's cost at k a = 362–400 (P ≈ 400–480): if it runs too long, fewer ε
  there (the report says); double-double's exponent range is f64's, so hₗ beyond 10³⁰⁸
  (P large at small k a) overflows: such units are recorded as unstable by the bound, the
  diagonal run skipped and the report says where; f32's range (T above 3·10³⁸ is
  infinite): recorded as a failure, not skipped.

## Pending runs

None. Done on locust, 2026-11-06 (all results fetched; rows archived on the M6 in
`~/projects/fmm-archive/locust/runs/ch0-6/`, 68 MB, not committed):
- the sweep, binary of ee1c88a, two shards (`sweep --threads 36 --shard i/2`), 612 units,
  09:13–12:06 UTC;
- §3.6's four levels (`sweep --kinds real --ka 13.532,27.063,54.127,108.253 --threads
  72`, binary of 01ef39d), 70 units, 17:13–17:16 UTC;
- the prototype times (`times --reps 7`, one thread, `taskset -c 8`), twice, 17:15 and
  17:17 UTC, load average 1.4 before (locust otherwise idle, GPU idle).

## Open questions

The report's §9: D3 (f32 on high-frequency levels), the bound's constants and its damping
factor e^{Im(k)(|t| + 2a)} (a correction of §3.6), T from extended precision or
interpolated, §3.5's p_high constant, CONVENTIONS §3.15 "Signatures", the PR's size, the
damped bound's conservatism. As built, test 5 runs k a up to 16, not 32 (the plan): at 32
the double-double test takes minutes, and the sweep compares the diagonal form with
double-double RCR at every k a.
