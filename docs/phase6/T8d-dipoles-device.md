# Phase 6 / T8d — dipole kernels on the device (C6.4, device part)

T8 built dipole sources (C6.4) through the whole host path and ran P2M, P2L and P2P on the
device's host fallback when an FMM has dipoles (docs/design/optimisation.md §7.4, "the
host fallback first, device kernels only if T8 measures the fallback above 20 % of a
device evaluation"). T8 measured it far above (`fmm-validate/results/phase6-dipoles.md`
§3): on Metal at N = 10⁶ (cube, f32, gradients) the fallback is 92–97 % of a dipole
evaluation with one host thread and 55–77 % with twelve, and a dipole evaluation costs
15–44× (one thread) and 2.5–6× (twelve) a charge evaluation on the device. This task builds
the device kernels. It was split from T8 by the user on 2026-10-17 (option (a) of T8's
open question), so that T8 merged with the host path and the device fallback.

Read first:
- root CLAUDE.md, docs/CONVENTIONS.md §3.7 "Dipole sources" and §3.13 (the dipole source
  chunks, the dipole rows of "Operators in scaled coordinates", "Range of the terms in
  f32", "Device kernels"), fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md (the device rules and
  "Dipole sources");
- docs/design/optimisation.md §7 (with §7.5, as built in T8), docs/design/simd-p2p.md §4.8
  (the host dipole kernel and its contract), docs/design/device-path.md §6 (the kernels);
- `fmm-validate/results/phase6-dipoles.md` (the measurements that call for this task);
- the code: `nd_fmm_kernels::{p2p, leaf}` (the charge kernels, their layouts and tests),
  `nd_fmm_exec::device` (`SOURCE_KINDS`, the dipole section of the module docs, the source
  store, `charge_slots`, `begin_evaluation`), `nd_fmm_ref::{leaf, p2p}` (the dipole
  reference operators), `nd_fmm_math::harmonics::{regular_grad, irregular_grad}` (the
  gradient ladders the device needs).

Do, in this order (each step measured before the next):
1. **The source store with strengths.** The device's source store holds the chunks of 6
   or 7 values per point (CONVENTIONS §3.13) with the strengths: the host's charge load
   scales the moments (as today), and one upload per evaluation carries them (the strength
   slots of `charge_slots` generalised to the source type). The transfer formula of
   `tests/device_common::expected_evaluation` with the strengths' bytes.
2. **Dipole P2P on the device** (the largest part: 49 of 69 ms at p = 3 and 48 of 126 ms
   at p = 8 on Metal with twelve threads): the kernel of simd-p2p.md §4.8 (ρ³ s and
   ρ³ (dₖ − 3 s ρ² wₖ), never r⁻⁵) in `nd_fmm_kernels::p2p`, every layout, under the device
   arithmetic rules of CONVENTIONS §3.13 "Device kernels" (the coincident-pair rule; ŷ by
   an explicit fma; the moments mapped by r̂(s|t), exact). Derive and state its contract and
   the smallest |d̂| for which it holds where subnormals flush (CONVENTIONS §3.13, a §3.13
   change in this task's PR, no version bump). Charges and dipoles: the two kernels per
   pair, charges first, as the host; a fused kernel only if measured worth it, with its
   order stated.
3. **Dipole P2M and P2L on the device** (8–37 ms of an evaluation with twelve threads):
   the gradient ladders of `harmonics::regular_grad` and `irregular_grad` on the device
   (`nd_fmm_kernels::leaf` evaluates the harmonics per point), contracted with d̂ per slot
   as `nd_fmm_ref::leaf::{p2m_dipoles, p2l_dipoles}`.
4. **Placement**: drop the forced fallback of `SOURCE_KINDS` for each kind that gains a
   kernel; `host_fallback` keeps every kind on the host as before (bit for bit the host
   path).
5. **Measurements**: `dipole_cost --backend metal` and `--backend cuda` (locust) against T8's
   numbers; the cost of a dipole evaluation against a charge evaluation on the device.
   Report in `fmm-validate/results/phase6-dipoles.md` (a T8d part) and in optimisation.md
   §7.5.

Tests that define done:
- The device dipole P2P against `nd_fmm_ref::p2p::p2p_dipoles` and the host kernel within
  its stated contract, on the CPU runtime, Metal (f32) and CUDA (f32, f64), every layout;
  the coincident-pair rule on the adversarial pairs of `spikes/device-arith` with moments.
- The device dipole P2M and P2L against `nd_fmm_ref::leaf::{p2m_dipoles, p2l_dipoles}` at
  the absolute frames, levels 2, 9 and 16, p ≤ 20 in f64 (as `tests/operator/device_leaf.rs`
  for charges).
- `device_common::check_dipoles` and `dipole_scenario` with the kinds on the device: within
  the device bounds of the host (1e-12 f64, 1e-5 f32), two evaluations bit for bit, the
  transfers of the formula; every kind on the fallback still bit for bit the host. In
  `tests/mpi_exec.rs` (CPU runtime) at 1, 2 and 4 ranks, `tests/device_metal.rs`,
  `tests/device_cuda.rs` (locust) and `tests/device_ranks.rs`.
- The dipole gate of `tests/dipoles.rs` with every kind on the device (CPU runtime f64,
  Metal f32, CUDA both), within the device-against-host bounds of docs/phase4/README.md.
- The charge path bit for bit unchanged on every backend.

Must pass: the root checks, the stricter workspace checks, the `nd-fmm-kernels` checks of
the root CLAUDE.md (CPU runtime and Metal by hand, CUDA on locust),
`.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks, and the device checks of
fmm-exec/CLAUDE.md.

Do not: change a convention of §3.1–§3.12 (§3.13 only, with the code); assert a timing; add
kernels other than Laplace dipoles.
