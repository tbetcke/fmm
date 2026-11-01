# CH3.4c The Helmholtz operator's P2P

Started 2026-11-01 on `tbetcke/h-ch3-4c`. Roadmap: "CH3.4c The Helmholtz operator's P2P.
`HelmholtzOperator`'s P2P through `evaluate_helmholtz` on the P2P choice's ISA, the
reference kept as the trusted path." Proposal: §10 CH3.4; §5.7, §6.2, §2 (constraints 1, 2
and 6).

## Goal

`HelmholtzOperator`'s P2P runs `nd_fmm_simd::P2pKernel::evaluate_helmholtz` (CH3.4b) on the
ISA of a `P2pChoice` (default `Auto`), with `nd_fmm_ref::helmholtz::p2p::p2p` kept as
`P2pChoice::Reference`, the trusted path.

## Acceptance (proposal §10, verbatim)

CH3.4: "accuracy contract over k r up to 22 c_leaf; cost per pair" (CH3.4a and CH3.4b). The
roadmap's *Done when* of CH3.4c: every ISA against the reference within CH3.4b's contract
in the operator's and the FMM's tests; the FMM deterministic at every thread count and
rank-invariant as before; its time on a CH3.1c case before and after.

## Read first

- `fmm-exec/src/helmholtz/operator.rs` (`Kernel::p2p`, `each_target`, `Scratch`);
  `fmm-exec/src/helmholtz/fmm.rs` (the builder, `check_settings`);
  `fmm-exec/src/helmholtz/densities.rs` (`charge_index`, `potential_index`,
  `gradient_index`).
- `fmm-simd/src/kernel.rs`, `P2pKernel::evaluate_helmholtz` and `HelmholtzCharges`;
  `docs/design/simd-p2p.md` §4.12 (the contract).
- `fmm-exec/src/operator.rs`, `P2pChoice`, `LaplaceOperator::{with_p2p, p2p_kernel}`;
  `fmm-exec/src/fmm.rs`, `FmmBuilder::p2p_kernel`, `SettingsError::P2pIsaUnavailable`.
- `fmm-exec/tests/helmholtz/operator.rs`, `fmm-exec/tests/helmholtz_fmm.rs`;
  `fmm-validate/src/helmholtz_accuracy.rs` and its example.

## Changes

`nd-fmm-exec` (Helmholtz modules only) and `nd-fmm-validate`:

- `HelmholtzOperator::with_p2p(choice: P2pChoice) -> Result<Self, IsaUnavailable>` and
  `p2p_kernel() -> P2pChoice` (the resolved choice), as `LaplaceOperator`'s; the default
  `Auto` (`P2pKernel::detect()`), so the Helmholtz FMM's default output changes in the last
  bits (Laplace's does not).
- `Kernel::p2p` with a kernel: per vector v, one `evaluate_helmholtz` call on all the leaf's
  targets with the sources (mapped as now), the charges (vector v's c n values point-major:
  the chunk's own slice for m = 1, gathered into a scratch of 2 `max_leaf_points` values
  for m > 1) and the output chunk's planes as they lie (the potentials of columns 2v and
  2v + 1 are 2 n_t adjacent values, the gradients 2 n_t adjacent `[T; 3]`: the kernel's
  layout). With `Reference`, the present loop.
- `HelmholtzFmmBuilder::p2p_kernel(choice)` (an ISA the machine lacks:
  `SettingsError::P2pIsaUnavailable`, in `check_settings`, before any collective) and
  `HelmholtzFmm::p2p_kernel()`.
- `nd_fmm_validate::helmholtz_accuracy::Case::p2p` and the example's `--p2p
  auto|reference|<isa>`, the kernel printed in the header.
- Docs: the two modules' docs, `fmm-exec/CLAUDE.md`, `fmm-validate/CLAUDE.md`.

Public API: the three methods above and `Case::p2p`. Tables, layouts, conventions: none.

## Tests that prove it

- `fmm-exec/tests/helmholtz/operator.rs`, `p2p_is_the_reference_bit_for_bit`: run with
  `Reference` and with `Isa::Scalar`, each bit for bit `nd_fmm_ref::helmholtz::p2p::p2p`
  (CH3.4b: the scalar path is the reference's loop).
- `fmm-exec/tests/helmholtz/p2p.rs` (new): every ISA the machine offers against the
  `Reference` operator, f64 and f32, real k and complex k (Re k̂ r to about 53, beyond the
  leaf rule's 22 c_leaf), real and complex charges, m = 2 vectors, gradients on, a neighbour
  of the same level and of a coarser one (the frame's r̂ = 2) and the self pair: each part
  of each target's potential within Σ_j (32 + 12 |k̂| r_j + 2 n_s) u_T |q_j| e^{−Im k̂ r_j} /
  r_j and of each gradient component within Σ_j (66 + 12 |k̂| r_j + 2 n_s) u_T |q_j|
  (1 + |k̂| r_j) e^{−Im k̂ r_j} / r_j² (CH3.4b's per-term contract, plus the accumulation of
  n_s terms in input order); `p2p_kernel()` reports `Isa(isa)` and `Reference`; `Auto` is
  `Isa(Isa::detect())` bit for bit; an unavailable ISA refused.
- `fmm-exec/tests/helmholtz_fmm.rs`: the existing scenarios now run `Auto` (threads, pair
  by pair, the one-rank FMM bit for bit: determinism and rank invariance on the vectorised
  path); a new check on two debug scenarios (complex k with gradients, real k with real
  charges): `Reference` and every available ISA, `Isa::Scalar` bit for bit `Reference`,
  every ISA within a relative L2 tolerance of `Reference` over all ranks (set from the
  measurement with a factor ≥ 10 of margin, glibc-safe), and each within the direct-sum
  bound; an unavailable ISA is `SettingsError::P2pIsaUnavailable` on every rank.
- `nd-fmm-validate`'s `tests/helmholtz_accuracy.rs` unchanged (default `Auto`).

## Checks

`/verify` (format; Clippy default, examples, workspace all-targets; `cargo test`,
`--workspace`; `cargo doc`; the MPI list at 2 and 4 ranks, since `helmholtz_fmm` changes).
`tests/helmholtz_fmm.rs` in release at 1 and 3 ranks by hand.

## Machine runs

The M6 only: `helmholtz_accuracy` on CH3.1c cases with `--p2p reference` and `--p2p auto`
(W1 at kD = 100, ε = 10⁻³ and 10⁻⁶; a W4 case at kD = 10), 1 and 6 threads, release, BLAS
threads at 1; the evaluation time before and after and the errors unchanged within ε.
Recorded below under "Results" (no separate report). locust and Kathleen: none (the
kernel's ISAs were checked there in CH3.4b; their time per pair is CH3.4b's report).

## Constraints and risks

- Constraint 1: no Laplace code changes; `P2pChoice` is reused unchanged.
- Constraint 2: the kernel is deterministic per machine, ISA and build and each target sums
  its sources in input order, so threads and ranks stay bit for bit.
- Constraint 6: `Reference` stays the trusted path, tested against every ISA.
- Risk: the f32 Helmholtz operator is untested in the FMM so far; the operator test runs
  f32 P2P only.

## Results

Tests (M6, NEON): `tests/helmholtz/p2p.rs` passes; the worst difference from `Reference`
is at most 0.019 of the contract's bound (φ and ∇φ, f64 and f32, k up to 128 + 16i, so
Re k̂ r to 53). `Isa::Scalar` is `Reference` bit for bit in the operator (f64, f32) and in
the FMM. In `tests/helmholtz_fmm.rs` NEON is within 6.9e-16 (φ) and 4.8e-16 (∇φ) of
`Reference` (relative L2, `KERNEL_TOLERANCE` 1e-13), and its errors against the direct sum
are `Reference`'s to the printed digits. The scenarios on `Auto` keep their checks: 4
threads bit for bit one, pair by pair bit for bit the level calls. The debug run at one rank
took 46 s, 36 s before.

The time before and after (M6, `helmholtz_accuracy`, release, one rank, BLAS threads 1,
2026-11-01; two launches, `--p2p reference,auto`, then `auto,reference`; the evaluation's
wall time in s, the mean of the two launches; at 6 threads the range):

| Case | ε | Reference, 1 × 1 | NEON, 1 × 1 | change | Reference, 1 × 6 | NEON, 1 × 6 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| W1, kD = 100 (79,578 points) | 10⁻³ | 6.74 | 6.59 | −2.3 % | 1.43–1.49 | 1.47–1.58 |
| W1, kD = 100 | 10⁻⁶ | 42.52 | 42.31 | −0.5 % | 9.36–9.55 | 9.37–10.69 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻³ | 5.30 | 4.96 | −6.4 % | 1.19–1.50 | 1.10–1.11 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻⁶ | 35.80 | 35.54 | −0.7 % | 8.34–10.45 | 7.98–10.48 |
| W3, kD = 30 (108,849 points) | 10⁻³ | 5.51 | 5.03 | −8.7 % | 1.23 | 1.13–1.15 |

- The vectorised kernel saves 0.15–0.48 s per evaluation at one thread, about the same at
  10⁻³ and 10⁻⁶ (P2P does not depend on ε): P2P is a small share of these evaluations, which
  the RCR translations at p = 16–46 dominate (CH3.1c's follow-up). Their cost is CH3.2's.
- At 6 threads the two launches differ by up to 25 % for the same kernel (the M6's
  performance and efficiency cores, tools/m6/machine.md), so those rows show no effect.
- Errors identical to the printed digits for both kernels; the `Reference` checksums equal
  CH3.1c's (W1 kD = 100: `01003b6877ad138a`, `0062dceef767aca1`; W3 kD = 30:
  `4d28864830075fe9`), and each kernel's checksum is the same at 1 and 6 threads and in both
  launches (NEON: `4e1010c90fc21158`, `9dbab52f331a4ccc`, `43dd41e6a6fef349`,
  `82bc2b1aac617b03`, `9c3d5399b246c996`).

## Pending runs

None.

## Open questions

None.
