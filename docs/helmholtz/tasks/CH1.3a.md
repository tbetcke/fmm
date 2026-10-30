# CH1.3a Direct-sum oracle

Started 2026-10-30 on `tbetcke/h-ch1-3a`. Roadmap: "CH1.3a Direct-sum oracle" (H1; the
first part of CH1.3, fifth of the priority list). Proposal: §10 CH1.3; §2, §5.7, §6.1
(placement, D2: A), §9 items 1, 3 and 4; CONVENTIONS §3.15 ("Kernel and factors",
"Inputs and outputs"); CH1.4's fixture and report (`fmm-validate/results/helmholtz-ch1-4.md`).

CH1.3 is split on this branch into CH1.3a–d (roadmap). CH1.3d, the diagonal form in
extended precision, waits for CH0.6's signatures and is off the priority list (the user,
2026-10-30, "Defer to CH1.3d"); CH2.0 and CH3.0 now need CH1.3a–c.

## Goal

The f64 direct-sum oracle and the generic P2P reference of the Helmholtz family in
`nd-fmm-ref`, for real, complex and imaginary k and real and complex charges, plus
Yukawa's real kernel, tested against CH1.4's FMM3D fixture to round-off.

## Acceptance (proposal §10 CH1.3, verbatim, for the whole of CH1.3)

> addition theorems; the k → 0 limit with the tolerances of Section 9; Yukawa two ways;
> the direct sum against the FMM3D fixtures of CH1.4 to round-off

CH1.3a's part (roadmap): the direct sum against CH1.4's fixtures to round-off; the k → 0
limit against Laplace's direct sum with the tolerances of §9 (relative to |k| r); Yukawa
two ways (the real kernel against the complex one at k = iκ). The addition theorems are
CH1.3b's and CH1.3c's.

## Read first

- CONVENTIONS §3.15 "Kernel and factors", "Inputs and outputs"; §3.13 "Coincident pairs".
- `fmm-ref/src/p2p.rs` (Laplace's pair `p2p` and `direct_sum`, Neumaier, the coincidence
  rule), `fmm-ref/CLAUDE.md`, `fmm-ref/tests/p2p/` and `tests/dipoles/fixtures.rs` (the
  fixture loader).
- `fmm-ref/fixtures/helmholtz_fmm3d.json` and `tools/fixtures/gen_helmholtz_fmm3d.py`:
  200 sources in [0, 1]³, 40 targets in [−0.5, 1.5]³; values at the sources (self term
  dropped) then the targets, 1/(4π) included; k = 1, 25, 1 + 0.5i, 25 + 2.5i, i, 25i;
  real and complex charges.

## Changes

`nd-fmm-ref` only, a new module; no Laplace function changes.

- `src/helmholtz/mod.rs` (new, public `nd_fmm_ref::helmholtz`): the family's conventions
  in summary (§3.15), the home of CH1.3b's leaf operators and CH1.3c's translations.
- `src/helmholtz/p2p.rs` (new, public), with `nd_fmm_math::Complex` (no new dependency):
  - `p2p<T>(k, sources, charges: &[Complex<T>], targets, potential: &mut [Complex<T>],
    gradient: Option<&mut [[Complex<T>; 3]]>)`: adds φ̃ = Σ q e^{ikr}/r and
    ∇φ̃ = Σ q (ikr − 1) e^{ikr}/r³ (x − y), the 4πφ of §3.15 (no 1/(4π), as every
    operator), sources in input order, plain additions;
  - `direct_sum(k, …)` in f64: the same terms, Neumaier-compensated per component;
  - `p2p_yukawa<T>(kappa, sources, charges: &[T], targets, potential: &mut [T], gradient)`
    and `direct_sum_yukawa`: e^{−κr}/r and −(κr + 1) e^{−κr}/r³ (x − y), real arithmetic;
  - the term: e^{ikr} = e^{−Im(k) r} (cos(Re(k) r) + i sin(Re(k) r)), written so that at
    k = iκ every operation matches the Yukawa term's (the two-ways test is bit for bit);
  - exact coincidences skipped (§3.13); any finite k accepted (the builder enforces the
    range of §3.15), k = 0 gives Laplace's kernel; panics on length mismatches and a
    non-finite k.
  - Charges and outputs as `Complex` slices, not split planes: this is the oracle,
    clarity over speed; §3.15's split planes are the operators' storage (CH1.3b on).
- `p2p.rs`'s `Neumaier` and `check_pair` become `pub(crate)` to be shared, unchanged.
- `lib.rs` documentation, `fmm-ref/CLAUDE.md` (purpose, fixture, oracle), README if it
  lists the modules.

## Tests that prove it

`fmm-ref/tests/helmholtz/` (a new test executable), all f64 unless stated, single thread:

1. **Fixtures** (`fixtures.rs`): `direct_sum` and `p2p` against all 12 cases of
   `helmholtz_fmm3d.json` (times 4π), potential and gradient at the 200 sources and 40
   targets: relative ℓ2 per case and quantity ≤ 1e-14 (the fixture is within 1.0e-15 of
   the exact kernel; 10× margin for glibc), and per point relative to the sum of term
   magnitudes ≤ 5e-14 (the fixture's own error in this measure is up to 2.7e-15 at
   k = 25i; measured 4.6e-15, so 10×). The header's `convention_version` is checked.
2. **k → 0** (`limit.rs`), against Laplace's `p2p::direct_sum` on seeded points in the
   unit cube, at |k| = 1e-2, 1e-4, 1e-6 for real, complex (arg π/4) and imaginary k, per
   target relative to the term magnitudes Σ|q|/r (potential) and Σ|q|/r² (gradient):
   - every kind (Im k ≥ 0, so Re(ikr) ≤ 0): |φ̃ − φ_L| ≤ |k| r_max; the first-order term
     pinned, |φ̃ − φ_L − ik Σq| ≤ (|k| r_max)²/2; |∇φ̃ − ∇φ_L| ≤ (|k| r_max)², plus 1e-14;
   - real k (§9 item 3): |Re φ̃ − φ_L| ≤ (k r_max)²/2, |Im φ̃/k − Σq| ≤ (k r_max)²/6
     relative to Σ|q|, |Im ∇φ̃| ≤ (k r_max)³/3, plus round-off;
   - k = 0: the potential equal to Laplace's bit for bit, the gradient within 1e-15.
3. **Yukawa two ways** (`yukawa.rs`): `p2p_yukawa` against `p2p` at k = iκ, real charges
   directly, and complex charges as two real vectors, κ = 0.5, 25, 300: bit for bit,
   f64 and f32; `direct_sum_yukawa` against `direct_sum` likewise (complex charges).
4. **Properties** (`properties.rs`): `p2p` against `direct_sum` within 1e-14 of the term
   magnitudes; f32 `p2p` against f64 within 1e-5 at |k| r ≤ 10; accumulation (+=);
   coincident pairs skipped (targets = sources, a duplicated source); reciprocity
   (φ̃ at x from a unit charge at y equals φ̃ at y from x, bit for bit); the gradient
   against central differences of the potential (relative 1e-7); linearity in the charges;
   empty inputs.
5. **Panics** (`panics.rs`): each kind of length mismatch (the four functions share one
   check); a non-finite k or κ.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`, the
default and workspace tests), plus `cargo clippy -p nd-fmm-ref --all-targets -- -D
warnings` and `cargo test -p nd-fmm-ref` (fmm-ref/CLAUDE.md). The MPI tests do not apply
(no code an MPI crate runs changes); the licence check does not (no manifest changes).

## Machine runs

None.

## Constraints and risks

- Constraint 1: no Laplace path changes; in Laplace's `p2p.rs` only the visibility of
  `Neumaier` and `check_pair` changes (`pub(crate)`), so its outputs are bit for bit the
  same (its tests run).
- Constraint 6: this is the trusted path later fast P2P kernels (CH2.4, CH3.4) are
  tested against.
- Risk: glibc's sin, cos and exp differ from macOS's in the last bit, so tolerances keep a
  10× margin over the M3 Max (memory "glibc test bounds"); the bit-for-bit tests compare
  two paths on one machine only.
- Risk: the phase Re(k) r is rounded once, a relative error of order |k| r ε in the term;
  at the fixture's |k| r ≤ 58 (r ≤ 2.29, |k| ≤ 25.1) this is about 6e-15, within the
  fixture bounds.

## Pending runs

None.

## Open questions

None.
