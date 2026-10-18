# Phase 6 / T13 — plane-wave M2L as a host strategy (C6.5)

T10's spike (`spikes/plane-wave/SPIKE_REPORT.md`) met C6.5's stop rule: on one uniform level,
plane-wave M2L with merge-and-shift was 2.2–9.8× faster than the faster of `Batched` and
`Rotation` at f64 p ≥ 8 on the M3 Max and locust, one process or every core,
and 1.5–12.5× on Kathleen against the AVX2 kernel (3.6–8.6× at p ≥ 12 against the AVX-512
kernel, but only 1.14–1.44× at p = 8); it lost at p = 3 everywhere and at p = 6 on a full
locust or Kathleen node. **Decision 6 was signed off by the user on 2026-10-18: C6.5
proceeds**, as this task, an opt-in host strategy, f64 and p ≥ 8 first, with its own error
budget (docs/phase6/README.md, "Decisions to sign off" 6). The device is not in this task
(below, "Afterwards").

Written on 2026-10-18, with T10's sign-off.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-tables/CLAUDE.md, fmm-simd/CLAUDE.md,
  fmm-validate/CLAUDE.md; docs/CONVENTIONS.md §3.3, §3.6–§3.8, §3.11, §3.12;
- docs/phase6/README.md (requirements 1–6; C6.5's criterion);
- `spikes/plane-wave/SPIKE_REPORT.md` (all of it; §2 the method in our basis, §5 the
  matched quadrature, §6 where it loses, §8 what adoption needs) and the spike's code
  (`src/planewave.rs`, `src/geometry.rs`: `Direction::for_pair`, `ParentGroups`);
- docs/design/optimisation.md §2 (the batched M2L: blocks, order, threading, scratch),
  §8.2 with "As found in T10";
- the code: `nd_fmm_exec::{tables, operator}` (`M2lStrategy`, `Tables`, `m2l_block`,
  `m2l_target`), `nd_fmm_exec::cost` (the constants per strategy), `nd_fmm_simd::matrix`
  (the column kernel's ISA dispatch, the model for a SIMD body).

Do, in this order (each step measured before the next):
1. **The convention.** The plane-wave representation in our basis (the spike's module
   documentation of `planewave.rs`: M̂ₙᵐ = 2^{−(n+1)} M̃ₙᵐ, the symbols of Oₙᵐ and Dₙᵐ, up and
   down, the frames' rotations, the half α nodes) as CONVENTIONS text proposed in the PR
   (CLAUDE.md: never in code first), saying whether it bumps `CONVENTION_VERSION` (a new
   operator family, §3.11 or a new section; no existing operator changes).
2. **The quadrature and the tables** (`nd-fmm-tables`): FMM3D's quadratures for separation 1
   with their provenance (NIST's notice; the spike's `tools/extract_quadrature.sh`), the
   choice of s per p by the measured FMM error (requirement 1, below; the spike's matched
   s = 10, 15, 18, 21, 32 at p = 3, 6, 8, 12, 18 is the starting point, chosen on a level),
   the conversion tables (λ powers, cos/sin, the rotation blocks of the frames) and the
   shift tables (source to parent, parent to child), built in f64, level-independent
   (CONVENTIONS §3.7 scaling), in the table cache if their build is worth caching.
3. **The strategy** (`nd-fmm-exec`): `M2lStrategy::PlaneWave`, M2L by plane waves per level
   call: the waves of every source box the level call reads (own and ghost: computed from
   their multipoles, so no new exchange), merge-and-shift per parent in Morton order with
   the parents derived from the boxes' keys (the plan's interface unchanged; children
   absent from an adaptive tree drop out of the masks), the incoming waves converted to
   each target's local. Threaded with rayon over blocks of parents, each local receiving
   its contributions in a fixed order (bit for bit across thread counts, requirement 2).
   `Auto` never resolves to it (until T11). On the device it runs as `Dense` ("PlaneWave,
   run as dense on the device"), as `Batched` does.
4. **Memory.** The waves take 6 × 16 H bytes per box (27 KB at p = 8, 97 KB at p = 18, against
   0.6–2.9 KB of multipole; 0.9 GB for level 5 of the N = 10⁶ cube on one rank). Bound them:
   one frame at a time (a third), or the waves of a window of parents in Morton order, with
   the scratch per thread stated as a formula and measured at the node's ranks (40 per
   Kathleen node), as optimisation.md §2.5 did for `Batched`.
5. **A SIMD body** (`nd-fmm-simd`): the shifts (complex products over H values) and the
   Fourier passes as kernels with run-time ISA dispatch (NEON, AVX2, AVX-512, scalar), under
   the crate's `unsafe` rules; each ISA against the scalar path within a stated bound (not
   bit for bit unless the order is kept). The spike's auto-vectorised loops gained 1.07–1.13×
   from an AVX-512 build on Kathleen where `Batched`'s kernel gained 1.4–2.1×.
6. **The cost model**: `PlaneWave`'s constants in `nd_fmm_exec::cost` (its work is per box
   and per merge entry, not per V pair), fitted as Phase 5S T7 did, so the work-weighted cut
   weighs it right.
7. **f32**: the strategy in f32 at p = 3, 6, 8 if its errors meet requirement 1 (the waves'
   terms reach about 3 × 10⁸ at p = 8); otherwise refused for f32 with a settings error, and
   the reason recorded.
8. **Measurements** on the three machines (Kathleen at one node; the 4-node runs are T11's):
   the evaluation, not only the level, `scaling --strategy planewave` with `--kinds`, the cube
   and the Plummer sphere at N = 10⁶ (and 10⁷ on locust and Kathleen), f64 p = 3, 6, 8, 12,
   18, one rank at one thread and the node's ranks and threads splits of T2's `t2-host` and
   `t2-node`, against `Batched` (separate and fused; AVX2 and AVX-512 on Kathleen) and
   `Rotation`; the memory per rank. Report in `fmm-validate/results/phase6-planewave.md`, and
   an "As built in T13" paragraph in optimisation.md §8.2. Propose for T11 where the strategy
   should be the default (by machine, precision and p).

Requirements (README "Requirements", as they apply to an approximate operator):
1. **Accuracy**: an error budget of its own (requirement 1, "a compressed or approximate
   operator"): errors against the direct sum within 1.1× `Dense`'s at the C3.2 cube and the
   C3.3 Plummer sphere, f64 p = 8, 12, 18 (and f32 p = 8 if offered), eight charge vectors;
   s per p chosen to meet it and stated. Not bit for bit `Dense`.
2. **Deterministic**: bit for bit from evaluation to evaluation, run to run and across
   thread counts and overlap settings.
3. **Several ranks**: on P ranks bit for bit the one-rank `PlaneWave` run over the union
   (each accumulator sums its sources in Morton order whatever rank holds them); rank-count
   invariance (`tests/invariance.rs`) kept.
4. **Trusted path**: `Dense` and `nd_fmm_ref::direct::m2l`; the level M2L against them within
   the stated quadrature error at p ≤ 20 (as the spike's `tests/accuracy.rs`).
5. Opt-in until T11 (requirement 6).

Tests that define done:
- The plane-wave tables and level M2L against `nd_fmm_ref::direct::m2l` on levels 2, 9 and 16
  at p ≤ 20 in f64, as potentials within the stated quadrature error (the spike's measure);
  merge-and-shift against direct translation to rounding.
- Every `tests/mpi_exec.rs` scenario under `PlaneWave` (f64; f32 if offered): within the
  budget of the direct sum, bit for bit across threads, overlap and on 2, 4 and 8 ranks
  against the one-rank run; rank-count invariance; the C5.1 host gate (`tests/multi_rank.rs`)
  with `PlaneWave` added.
- The SIMD bodies against the scalar path on every ISA the machine has (CI's runners and
  Kathleen by hand), skipping cleanly elsewhere.
- The C3.2 and C3.3 gates under `PlaneWave` at f64 p = 8, 12, 18 within 1.1× `Dense`'s
  errors.

Must pass: the root checks, the stricter workspace checks, the `nd-fmm-simd` checks of the
root CLAUDE.md (the other architecture's clippy), `.github/scripts/run-mpi-tests.sh` at 2 and
4 ranks, `tests/mpi_exec.rs` and the ignored `tests/multi_rank.rs` at 2, 4 and 8 ranks by
hand, and Phase 5S T2's `correctness.sbatch` on Kathleen (a change to the multi-rank path).

Do not: change an existing convention or operator; make `PlaneWave` a default (T11);
add a plane-wave kernel to `nd-fmm-kernels` (below); add a dependency to the workspace.

Afterwards (not this task): a device plane-wave M2L, modelled by T10 at 2.4–20× T7's dense M2L
stage on the H100 (SPIKE_REPORT.md §3; worth most at high p, where M2L is most of a device
evaluation: 137 of 196 ms at f64 p = 18, 10 of 50 ms at p = 8), would be its own task after
this one, gated on this task's host results and on a measured device kernel showing the gain
at p ≥ 12.
