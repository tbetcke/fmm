# Plane-wave M2L: spike report (Phase 6 T10, C6.5)

Phase 6, task T10 ([docs/phase6/T10-plane-wave-spike.md](../../docs/phase6/T10-plane-wave-spike.md));
the question and the stop rule are docs/design/optimisation.md §8.2 (signed off on
2026-10-15). Written on 2026-10-18 from branch `tbetcke/phase6_t10` (main at 40d4651 plus
this spike), release builds, rustc 1.99.0, f64, one thread per process, every BLAS thread
variable unset (no BLAS is linked). The path taken: **with Kathleen** (one node, job
605529).

## Summary

1. **The model clears the stop rule** (§3). With the conservative quadrature (β = 0.1
   of our error, judged by the worst pair) and the rates T2 measured, plane-wave M2L
   with merge-and-shift is modelled at 1.4–8.1× the faster of `Batched` and `Rotation` on
   the M2L stage of the cube at N = 10⁶, f64 p ≥ 8, on all three machines, one rank or the
   node's ranks; at p = 3 it loses (0.3–0.6×). So the spike was built.
2. **The spike** (§4) implements the six directional conversions (multipole → plane
   waves → local, derived in our basis from CONVENTIONS §3.3 and §3.11, ±x and ±y by
   rotation), the diagonal translation and FMM3D's merge-and-shift for one uniform level
   on the host in f64, with FMM3D's quadratures. Its level M2L equals
   `nd_fmm_ref::direct::m2l` within the quadrature's error at p = 3–18 (§5), direct and
   merge-and-shift translation agreeing to 4.3e-16.
3. **The quadrature FMM3D states is optimistic, but the error that matters is small**
   (§2): with FMM3D's α nodes the worst pair of the region errs 6–23× the stated error;
   on a level M2L the plane-wave error is 0.2–1.4 % of the bound that worst pair sets
   (until a rounding floor at p = 18). The *matched* quadrature (level error at most
   0.1× our own M2L truncation error) has s = 10, 15, 18, 21 and 32 λ nodes at p = 3, 6,
   8, 12 and 18.
4. **Measured on one level** (level 4, T1's: 4,096 boxes, 584,136 V pairs; §6), the
   plane waves with the matched quadrature against the faster of `Batched` and
   `Rotation`:

   | machine | processes | p = 3 | p = 6 | p = 8 | p = 12 | p = 18 |
   | --- | --- | ---: | ---: | ---: | ---: | ---: |
   | M3 Max | 1 | 0.86 | **2.04** | **3.25** | **7.49** | **7.12** |
   | M3 Max | 12 | 0.72 | **1.76** | **3.02** | **6.50** | **7.60** |
   | locust | 1 | 1.08 | **1.84** | **2.81** | **6.81** | **5.97** |
   | locust | 72 | 0.51 | 1.16 | **2.19** | **6.64** | **9.83** |
   | Kathleen, default build (`Batched` AVX2) | 1 | 0.59 | 1.05 | **1.54** | **6.78** | **9.46** |
   | Kathleen, default build (`Batched` AVX2) | 40 | 0.56 | 1.14 | **1.88** | **6.21** | **12.46** |
   | Kathleen, native build (`Batched` AVX-512) | 1 | 0.52 | 0.73 | 1.14 | **3.60** | **8.21** |
   | Kathleen, native build (`Batched` AVX-512) | 40 | 0.53 | 0.91 | **1.44** | **3.79** | **8.58** |

   **The 1.25× margin is met** at f64 p ≥ 12 everywhere (3.6–12.5×); at p = 8 everywhere
   but one Kathleen core against `Batched`'s AVX-512 kernel (1.14×; 1.44× with 40
   processes); at p = 6 on the M3 Max and one locust core. It is not met at p = 3
   anywhere. The model predicted the one-core cells at p ≥ 6 within 0.77–1.34× (§7).
   Kathleen gains least: there the auto-vectorised plane-wave loops run 2.2–2.4× slower
   than on the M3 Max and barely gain from an AVX-512 build, while `Batched`'s
   hand-written kernel gains 1.4–2.1×.
5. **Recommendation (decision 6): C6.5 proceeds past its spike.** Adopt plane-wave M2L
   as an opt-in host strategy (`M2lStrategy::PlaneWave`, f64, p ≥ 8 first) in a task of
   its own, with its own error budget (the matched quadrature per p) like a compressed
   operator (requirement 1), opt-in until T11. What that task must settle is §8: the
   memory of the waves (27–97 KB per box at p = 8–18 against 0.6–2.9 KB of multipole),
   the node's memory bandwidth at p ≤ 6, a SIMD body for x86 (Kathleen), f32, adaptive
   trees and several ranks, and the device.

## 1. The question and the stop rule

optimisation.md §8.2: at which p, precision and machine (host per core; one GPU) would
plane-wave M2L beat T2's batched dense M2L and Rotation on the uniform cube at N = 10⁶,
counting the conversions and the six directional lists with merge-and-shift? A model
first; the spike only where the model shows at least **1.25×** over the faster of
`Batched` and `Rotation` on the M2L stage at some p ≤ 18 on some machine; if built, the
spike must measure that margin on one level, else C6.5 is dropped.

laplace-fmm-plan §3.3 and §4 expected little ("within about 25 % of point-and-shoot",
"at most tens of percent on CPUs", from Gumerov and Duraiswami 2005). That comparison
was against a rotation scheme whose cost per pair is (10/3)p³ multiply-adds; here the
plane waves' cost per box is dominated by about 31 diagonal products of H complex values
(merge-and-shift) and 12 small Fourier transforms, against 165 V pairs per box at
O(p³) (Rotation) or O(p⁴) (`Batched`), so the expectation did not carry over.

## 2. The method

**The literature, checked.** Greengard and Rokhlin (1997, Acta Numerica 6, "A new
version of the fast multipole method for the Laplace equation in three dimensions") and
Cheng, Greengard and Rokhlin (1999, J. Comput. Phys. 155, "A fast adaptive multipole
algorithm in three dimensions") represent the far field in six directions by
1/|u| = (1/2π) ∫₀^∞ e^{−λz} ∫₀^{2π} e^{iλ(x cos α + y sin α)} dα dλ, discretised with
s generalised-Gaussian λ nodes and Nₖ trapezoidal α nodes per λ. The details here were
checked against FMM3D v2.1.0's implementation of that scheme (the release T9 built;
`src/Laplace/lfmm3d.f`, `lpwrouts.f`, `lwtsexp_sep1.f`), not against the papers' tables:
its quadratures for separation 1 (`vwts`, `numthetafour`, `numthetahalf`, s = 2–39)
were extracted by `tools/extract_quadrature.sh` into `src/quadrature_data.rs` (the file
is NIST's, its notice kept; FMM3D is Apache-2.0); FMM3D chooses s = 12, 20, 29 or 37 by
its `eps` (`lfmm3d.f`, `isep = 1`).

**In our basis** (`src/planewave.rs`, module documentation). In box widths
u = v/2 the scaled multipole becomes M̂ₙᵐ = 2^{−(n+1)} M̃ₙᵐ. CONVENTIONS §3.11 gives
Iₙᵐ = Oₙᵐ I₀⁰ with Oₙᵐ = (−(∂x + i∂y))ᵐ (−∂z)ⁿ⁻ᵐ, which on a plane wave is the symbol
λⁿ (−i e^{iα})ᵐ; for real charges the orders ±m pair to (−i)ᵐ 2 Re(M̂ᵐ e^{imα}). The
local side uses the ladder of §3.4: Dₙᵐ = (∂x − i∂y)ᵐ ∂zⁿ⁻ᵐ extracts the coefficient of
Rₙᵐ, giving λⁿ (−1)ⁿ⁻ᵐ (i e^{−iα})ᵐ for an upward wave. Down is the same with a sign
(−1)ⁿ⁻ᵐ, so up and down share one Fourier pass (as FMM3D). ±y and ±x are ±z after the
rotations R_x(π/2) and R_y(−π/2): multipoles rotate by K Dⁿ K and locals back by
K S Dⁿ(Qᵀ) S⁻¹ K (§3.11, "Rotation of coefficients"; `nd_fmm_math::rotation`). No
convention changes.

**Half the α nodes.** A plane-wave expansion of real charges satisfies
X(λ, α + π) = conj X(λ, α), and so do the shifts; with Nₖ made even (one node more where
FMM3D's is odd) half the nodes carry it: H = Σₖ Nₖ'/2 complex values per direction
(FMM3D stores all Nₖ).

**Directions and merge-and-shift.** A V pair goes to the axis along which its source
lies two boxes from *every* child of the target's parent (z, then y, then x; one shift
serves all eight children), and otherwise to the first axis along which it lies two
boxes from the target (`Direction::for_pair`); either way the offset along the axis is 2
or 3 boxes, the quadrature's region. Per parent and direction, each source's waves are
shifted once to the parent's centre into the accumulator of its set of children
(`ParentGroups`); each child adds its accumulators and shifts once to its own centre.
An interior parent has 208 sources: 152 reach all eight children along one axis, the
other 56 take 96 entries, so 248 shifts per parent, 31 per box. Per box on level 5:
165 V pairs against 27.0 shifts, 3.5 accumulators, 11.9 accumulator additions and 5.6
child shifts (`examples/model.rs`).

**The quadrature's error, measured** (`Quadrature::measured_error`, the largest error
for 1/|u| over z ∈ [1, 4], |x|, |y| ≤ 4, on a grid). The λ sum alone (many α nodes) meets
FMM3D's stated error; with FMM3D's Nₖ the worst pair errs **6–23× the stated error**
(largest at z = 1, ρ = 4√2; s = 12: 6.6e-4 against the stated 7.1e-5). That worst pair is
rare: on a level M2L the plane-wave error is 0.2–1.4 % of E Σ|q| (§5; more only at the
rounding floor of p = 18). The spike therefore
chooses s by the level error (§5, "matched"); the model's first pass chose by the worst
pair (β below).

## 3. The model (before the spike)

`src/model.rs`, printed by `examples/model.rs`; every input measured by T2
(fmm-validate/results/phase6-m2l.md §5, one rank at one thread; §6, one rank per core).
Per box the plane waves cost 4 rotations of 2 Σₙ (2n + 1)² flops, 3 + 3 multipole/local
↔ Fourier passes, 6 + 6 Fourier ↔ plane-wave passes and the shifts (8 H flops per merge,
2 H per accumulator addition, 6 H per child shift). The rates are T2's: `Batched`'s
achieved rate R_B = 2 (p + 1)⁴ / t_B, `Rotation`'s R_R = 2 (p + 1)(10p² + 32p + 9)/3 / t_R,
with t_R derived from T2's evaluation totals, t_R = t_B + (total_R − total_B) / pairs
(6,039,504 V pairs on the cube's levels 2–5; the derived values are within 1.5 % of the
six per-pair times §6 lists). *Optimistic*: every plane-wave flop at R_B; *central*: the
rotations and Fourier passes at R_R, the shifts at R_B; *pessimistic*: every flop at R_R.

The quadrature for the stop rule: β = 0.1 of our relative L2 error of φ at p
(2.9e-3, 1.3e-4, 1.9e-5, 6.9e-7, 1.0e-8), judged by the worst pair: s = 14, 18, 20, 25,
30 (H = 157, 284, 358, 593, 882).

**The M2L stage of the cube at N = 10⁶, the bar (faster of `Batched` and `Rotation`)
over the plane waves, central case, β = 0.1** (*model*; optimistic and pessimistic and
every ms in `examples/model.rs`):

| machine | ranks | f64 p = 3 | 6 | 8 | 12 | 18 | f32 p = 3 | 6 | 8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 Max | 1 × 1 | 0.41 | 1.45 | **2.74** | **5.59** | **7.95** | 0.31 | 0.85 | 1.51 |
| locust | 1 × 1 | 0.56 | 1.60 | **2.74** | **5.56** | **7.68** | 0.40 | 0.97 | 1.62 |
| Kathleen | 1 × 1 | 0.31 | 0.77 | **1.41** | **3.81** | **8.14** | 0.23 | 0.49 | 0.82 |
| M3 Max | 12 × 1 | | | **2.55** | **5.44** | | | | |
| locust | 72 × 1 | | | **3.36** | **5.47** | | | | |
| Kathleen | 40 × 1 | | | **1.73** | **3.81** | | | | |

The pessimistic case still clears 1.25× at f64 p ≥ 8 on the M3 Max and locust (1.8–6.9×)
and at p ≥ 12 on Kathleen. **The stop rule's condition holds; the spike is built.**

**Memory traffic.** The waves are 6 × 16 H bytes per box (f64): 7–97 KB per box with
the matched quadrature (p = 3–18), against 0.13–2.9 KB of multipole. Per box on level 5 the plane
waves move 14–202 KB at best (written once, read once) and 39–542 KB at worst (read once
per shift), against the 10–5,150 KB of tables that `Batched` streams per box (each of the
316 tables read once per block of 64 targets): more traffic than `Batched` at p = 3,
about as much at p = 6, less at p ≥ 8. The model has no bandwidth term; §6 shows where it matters.

**One GPU** (*model only*; no plane-wave kernel exists). On locust's H100 the dense M2L
stage takes 0.97–5.80 ms (f32 p = 3–8) and 1.24–139 ms (f64 p = 3–18) (phase6-device.md
§4.1, T7's best variant). The plane waves' flops at the dense kernel's measured share of
the peak (6–34 %) and their worst-case traffic at 70 % of 4 TB/s give 0.3–7 ms, so
**2.4–20× on the model** (larger at high p); the shifts are bandwidth-bound there (0.5
flop per byte). The irregular merge lists are the open device question; not measured.

## 4. The spike

`spikes/plane-wave/` (`nd-fmm-spike-plane-wave`, a workspace member outside the default
members; workspace dependencies only: nd-fmm-math, nd-fmm-ref, nd-fmm-tables,
nd-fmm-simd):

- `src/quadrature.rs`, `src/quadrature_data.rs`: FMM3D's tables, the measured error, the
  even α counts;
- `src/geometry.rs`: the uniform level in Morton order, V lists (CONVENTIONS §3.12; level
  4 has T1's 584,136 pairs), the directions, the merge groups, the counts;
- `src/planewave.rs`: `PlaneWave` (the conversions and shift factors), `LevelPlan` (the
  level's lists and shift tables, built once like a plan), `level_m2l` with
  `Translation::{Direct, MergeShift}`;
- `src/baseline.rs`: the level M2L of `Batched` exactly as `nd-fmm-exec`'s `m2l_block`
  runs it (blocks of 64 targets, offsets in order, `MatrixKernel::accumulate_columns` on
  `M2lTables` read in place), and of `Rotation` as `m2l_target` (`RotationTables::m2l`
  per pair, target by target);
- `src/model.rs`, `examples/model.rs`: §3;
- `examples/level.rs`: times one level (median of 3 calls after a warm-up), multipoles of
  4 random charges per box, and checks the plane waves' potentials against `Batched`'s;
- `tests/accuracy.rs`: §5;
- `tools/sweep.sh`, `tools/run_node.sh` (K copies at once, one per core, each with its
  own tables and data, as T1's level benchmark), `tools/tabulate.py`.

The plane-wave loops are plain Rust, auto-vectorised (complex values as real and
imaginary halves; the Fourier sums with four partial sums); no intrinsics, no `unsafe`.

**Fairness of the baselines.** The spike's level calls reproduce the FMM's own M2L per
pair: on the M3 Max `Batched` 0.494 µs (T2 in the FMM, all levels: 0.505), `Rotation`
0.871 (0.845) at p = 8; locust 0.631 (0.668) and 1.041 (1.102); at p = 18 `Rotation`
6.68 µs against T2's 6.46 (derived), locust 9.08 (9.02). On Kathleen (default build) `Batched`'s AVX2 kernel takes 0.563 µs
at p = 8 (0.659 in the FMM, all levels) and `Rotation` 2.228 (2.258).

## 5. Correctness (the test that defines done)

`tests/accuracy.rs`: per box 4 random charges in [0, 1) uniform in the box, their
multipoles M̃ = Σ q conj R(u); the level M2L by plane waves (both translations) against
`nd_fmm_ref::direct::m2l` over every V pair; error measure: the potential r φ of each
local at 4 random points of every target box, |φ_pw − φ_direct| against
**E Σ_V |q|** (E the quadrature's measured error; a unit charge contributes ½ (1/|u|),
so the quadrature alone allows ½ E; the bound doubles it for the dropped Fourier modes
and the local's truncation). Direct and merge-and-shift translation must agree within
1e-12 of the largest potential.

- `level_m2l_within_the_quadrature_error` (debug, level 2, in the workspace's
  `cargo test`): p = 3, 6, 8 at s = 11, 14, 17: error 0.006–0.013 of the bound; merge
  against direct 1.6e-16–3.5e-16. Passes.
- `level_3_every_p` (ignored; release, level 3, 512 boxes, 53,352 pairs; 246 s on the
  M3 Max): every s from 6 until the level error is at most 0.1× the M2L truncation
  error (max over the points of |φ_direct − φ_charges| / Σ|q|), each within the bound.
  Passes:

| p | M2L truncation (Σ\|q\|) | matched s | its error (Σ\|q\|) | error / bound, every s tried | the model's s (β = 0.1 / 1) |
| ---: | ---: | ---: | ---: | --- | --- |
| 3 | 1.9e-4 | 10 | 1.6e-5 | 0.005–0.013 | 14 / 11 |
| 6 | 8.1e-6 | 15 | 5.0e-7 | 0.005–0.013 | 18 / 15 |
| 8 | 1.7e-6 | 18 | 6.3e-8 | 0.004–0.013 | 20 / 17 |
| 12 | 1.6e-7 | 21 | 1.4e-8 | 0.002–0.013 | 25 / 22 |
| 18 | 8.6e-10 | 32 | 8.5e-11 | 0.002–0.92 | 30 / 27 |

At p = 18 the plane-wave error stops falling near 2–3e-10 Σ|q| from s = 27 to 31 (0.92
of the bound at s = 31), a floor of rounding in the large terms λⁿ M̂ₙ at the largest λ
(24 per box width at s = 31); s = 32 happens to fall below 0.1× the truncation. Targets near 1e-10
are where FMM3D also stops (its largest s is 37, for eps below 5e-10). In the timing runs
the plane waves' potentials agreed with `Batched`'s within 6.6e-5, 2.6e-6, 4.8e-7,
1.3e-7 and 5.1e-10 of the largest potential at p = 3, 6, 8, 12, 18 (matched s), against
our FMM errors of 2.9e-3, 1.3e-4, 1.9e-5, 6.9e-7, 1.0e-8.

## 6. Measurements

`tools/sweep.sh K 4`: level 4, the matched quadrature and the model's β = 0.1 one, one
process, then K processes at once (the node's cores; each its own data, as T1's level
benchmark and as ranks; the median over the processes). M2L per V pair, µs (lower is
better); "bar ÷ PW" the faster of `Batched` and `Rotation` (B or R) over the plane waves
with the matched s; "fused ÷ PW" against T2's opt-in fused mode.

### M3 Max (12 performance cores; outside the sandbox, load average below 2.5 before)

| p | processes | Batched | fused | Rotation | plane waves, matched s | plane waves, model's s | bar ÷ PW | fused ÷ PW |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 3 | 1 | 0.0261 | 0.0177 | 0.1578 | 0.0303 (s = 10) | 0.0576 (s = 14) | 0.86 (B) | 0.58 |
| 3 | 12 | 0.0323 | 0.0243 | 0.1662 | 0.0446 (s = 10) | 0.0996 (s = 14) | 0.72 (B) | 0.54 |
| 6 | 1 | 0.1878 | 0.1122 | 0.4591 | 0.0922 (s = 15) | 0.1285 (s = 18) | **2.04** (B) | 1.22 |
| 6 | 12 | 0.2202 | 0.1447 | 0.4902 | 0.1251 (s = 15) | 0.2014 (s = 18) | **1.76** (B) | 1.16 |
| 8 | 1 | 0.4937 | 0.2932 | 0.8709 | 0.1517 (s = 18) | 0.1950 (s = 20) | **3.25** (B) | 1.93 |
| 8 | 12 | 0.5695 | 0.3853 | 0.8940 | 0.1884 (s = 18) | 0.2547 (s = 20) | **3.02** (B) | 2.05 |
| 12 | 1 | 2.1976 | 1.3206 | 2.2245 | 0.2934 (s = 21) | 0.4289 (s = 25) | **7.49** (B) | 4.50 |
| 12 | 12 | 2.4539 | 1.7771 | 2.7347 | 0.3777 (s = 21) | 0.5723 (s = 25) | **6.50** (B) | 4.71 |
| 18 | 1 | 13.0647 | 8.8089 | 6.6812 | 0.9390 (s = 32) | 0.8222 (s = 30) | **7.12** (R) | 9.38 |
| 18 | 12 | 18.9070 | 14.2985 | 9.4384 | 1.2415 (s = 32) | 1.1009 (s = 30) | **7.60** (R) | 11.52 |

Level 5 (32,768 boxes, 5,398,920 pairs; the FMM's deepest level at N = 10⁶), one process:
p = 8 `Batched` 0.553, `Rotation` 0.881, plane waves 0.148 µs (**3.72×**; waves 893 MB);
p = 12 2.307, 2.258, 0.266 (**8.50×**; waves 1.26 GB).

### locust (GH200, 72 Neoverse-V2 cores; load average 0.0–0.3 before, GPU idle)

| p | processes | Batched | fused | Rotation | plane waves, matched s | plane waves, model's s | bar ÷ PW | fused ÷ PW |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 3 | 1 | 0.0390 | 0.0297 | 0.1646 | 0.0360 (s = 10) | 0.0827 (s = 14) | 1.08 (B) | 0.83 |
| 3 | 72 | 0.0548 | 0.0474 | 0.1704 | 0.1076 (s = 10) | 0.2196 (s = 14) | 0.51 (B) | 0.44 |
| 6 | 1 | 0.2450 | 0.1642 | 0.5525 | 0.1330 (s = 15) | 0.1927 (s = 18) | **1.84** (B) | 1.23 |
| 6 | 72 | 0.2977 | 0.2289 | 0.5609 | 0.2558 (s = 15) | 0.3555 (s = 18) | 1.16 (B) | 0.89 |
| 8 | 1 | 0.6314 | 0.4116 | 1.0413 | 0.2248 (s = 18) | 0.2948 (s = 20) | **2.81** (B) | 1.83 |
| 8 | 72 | 0.8154 | 0.6332 | 1.1047 | 0.3726 (s = 18) | 0.4618 (s = 20) | **2.19** (B) | 1.70 |
| 12 | 1 | 3.0221 | 2.0231 | 2.9970 | 0.4399 (s = 21) | 0.6571 (s = 25) | **6.81** (R) | 4.60 |
| 12 | 72 | 4.3923 | 3.5081 | 3.8834 | 0.5845 (s = 21) | 0.8779 (s = 25) | **6.64** (R) | 6.00 |
| 18 | 1 | 24.2226 | 20.5923 | 9.0804 | 1.5214 (s = 32) | 1.3013 (s = 30) | **5.97** (R) | 13.54 |
| 18 | 72 | 29.2478 | 26.8337 | 18.1500 | 1.8470 (s = 32) | 1.7932 (s = 30) | **9.83** (R) | 14.53 |

### Kathleen (one node, node-c11a-002, 2 × 20 Xeon Gold 6248; job 605529, `singlenode`)

The default build (the x86-64 baseline: the plane waves and `Rotation` compiled for
SSE2, `Batched` dispatching its AVX2 kernel at run time, as the FMM does today):

| p | processes | Batched (AVX2) | fused | Rotation | plane waves, matched s | plane waves, model's s | bar ÷ PW | fused ÷ PW |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 3 | 1 | 0.0380 | 0.0332 | 0.3414 | 0.0649 (s = 10) | 0.1377 (s = 14) | 0.59 (B) | 0.51 |
| 3 | 40 | 0.0568 | 0.0510 | 0.4642 | 0.1010 (s = 10) | 0.2046 (s = 14) | 0.56 (B) | 0.50 |
| 6 | 1 | 0.2215 | 0.1609 | 1.1599 | 0.2111 (s = 15) | 0.3079 (s = 18) | 1.05 (B) | 0.76 |
| 6 | 40 | 0.3542 | 0.2711 | 1.5814 | 0.3116 (s = 15) | 0.4568 (s = 18) | 1.14 (B) | 0.87 |
| 8 | 1 | 0.5628 | 0.4540 | 2.2284 | 0.3644 (s = 18) | 0.4481 (s = 20) | **1.54** (B) | 1.25 |
| 8 | 40 | 1.0143 | 0.8109 | 3.0288 | 0.5392 (s = 18) | 0.6416 (s = 20) | **1.88** (B) | 1.50 |
| 12 | 1 | 4.3443 | 3.0551 | 6.3641 | 0.6412 (s = 21) | 0.8865 (s = 25) | **6.78** (B) | 4.76 |
| 12 | 40 | 5.7071 | 4.1139 | 9.0151 | 0.9185 (s = 21) | 1.3380 (s = 25) | **6.21** (B) | 4.48 |
| 18 | 1 | 27.8126 | 20.1486 | 19.7813 | 2.0906 (s = 32) | 1.8632 (s = 30) | **9.46** (R) | 9.64 |
| 18 | 40 | 35.4858 | 26.1383 | 32.1413 | 2.5786 (s = 32) | 2.7044 (s = 30) | **12.46** (R) | 10.14 |

The native build (`RUSTFLAGS="-C target-cpu=native"`: the plane waves and `Rotation`
auto-vectorised for AVX-512, `Batched` with its AVX-512 body, `--isa avx512`, Phase 6
T6):

| p | processes | Batched (AVX-512) | fused | Rotation | plane waves, matched s | plane waves, model's s | bar ÷ PW | fused ÷ PW |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 3 | 1 | 0.0299 | 0.0276 | 0.3202 | 0.0577 (s = 10) | 0.1182 (s = 14) | 0.52 (B) | 0.48 |
| 3 | 40 | 0.0511 | 0.0473 | 0.4131 | 0.0966 (s = 10) | 0.1844 (s = 14) | 0.53 (B) | 0.49 |
| 6 | 1 | 0.1405 | 0.1122 | 1.0764 | 0.1931 (s = 15) | 0.2853 (s = 18) | 0.73 (B) | 0.58 |
| 6 | 40 | 0.2657 | 0.2295 | 1.5145 | 0.2931 (s = 15) | 0.4364 (s = 18) | 0.91 (B) | 0.78 |
| 8 | 1 | 0.3893 | 0.3117 | 2.0948 | 0.3412 (s = 18) | 0.3879 (s = 20) | 1.14 (B) | 0.91 |
| 8 | 40 | 0.7527 | 0.6510 | 2.8375 | 0.5219 (s = 18) | 0.6085 (s = 20) | **1.44** (B) | 1.25 |
| 12 | 1 | 2.1085 | 1.5972 | 6.0569 | 0.5860 (s = 21) | 0.7754 (s = 25) | **3.60** (B) | 2.73 |
| 12 | 40 | 3.3043 | 2.6470 | 8.5248 | 0.8725 (s = 21) | 1.2197 (s = 25) | **3.79** (B) | 3.03 |
| 18 | 1 | 15.2282 | 11.6858 | 18.7355 | 1.8554 (s = 32) | 1.6373 (s = 30) | **8.21** (B) | 6.30 |
| 18 | 40 | 21.7443 | 17.5682 | 30.7497 | 2.5352 (s = 32) | 2.4941 (s = 30) | **8.58** (B) | 6.93 |

The plane waves gain 1.07–1.13× from the AVX-512 build (p ≥ 6, one process), `Batched`
1.4–2.1×: the plane-wave loops as written do not use the wider vectors well (complex
products over split halves, short Fourier sums). Per core they run 2.2–2.4× slower than
on the M3 Max (p = 8: 0.364 against 0.152 µs), more than the clock (2.5 against
4.05 GHz). A hand-written SIMD body, as `Batched` has, is the obvious remedy and an item
for the adopting task (§8).

Notes:

- **Where the plane waves lose: low p and a busy node.** At p = 3 the waves (73 complex
  values per direction, 14 KB per box) are as costly as `Batched`'s 16² tables per pair.
  With every locust core busy their time per pair grows 3.0× at p = 3 and 1.9× at p = 6
  (`Batched` 1.4× and 1.2×): the incoming part (the shifts, reading the waves of 23
  sources per box) slows most (60 → 125 ms at p = 6), the outgoing part little; the
  waves of a level do not stay in the 1 MB L2 per core. At p ≥ 8 the work per byte
  grows and the gain holds at full load.
- **The model's s against the matched s.** The worst-pair quadrature (β = 0.1) costs
  1.3–2.3× the matched one at p ≤ 12 (more α nodes and λ nodes); at p = 18 the model's
  s = 30 is 12 % cheaper than the matched s = 32, which exists only because of the
  rounding floor (§5).
- **The parts.** Outgoing (multipole → waves, 6 directions) is a third of the plane
  waves' time on one core (the M3 Max at p = 8: 30 of 89 ms), the shifts and the
  incoming conversions the rest. Both conversions are O(H F) per box; the shifts O(31 H).

## 7. Against the model

| machine | p | model, central, matched s (stage) | measured, one level, one core | |
| --- | ---: | ---: | ---: | --- |
| M3 Max | 6 / 8 / 12 / 18 | 2.16 / 3.41 / 8.33 / 6.92 | 2.04 / 3.25 / 7.49 / 7.12 | |
| locust | 6 / 8 / 12 / 18 | 2.39 / 3.42 / 8.28 / 6.68 | 1.84 / 2.81 / 6.81 / 5.97 | |
| Kathleen (AVX2) | 6 / 8 / 12 / 18 | 1.15 / 1.76 / 5.68 / 7.08 | 1.05 / 1.54 / 6.78 / 9.46 | |

The central case (conversions at Rotation's rate, shifts at `Batched`'s) predicted the
M3 Max within 0.90–1.03×, locust within 0.77–0.89× and Kathleen (AVX2) within
0.88–1.34× (`examples/model.rs` prints the
same table with the matched quadrature). At p = 3 it predicted 0.84 (M3 Max) and 1.17
(locust) against 0.86 and 1.08. With the node's processes it overestimated locust at
p ≤ 6, where it has no bandwidth term (above).

## 8. What adoption needs (the follow-up task)

Measured here: one uniform level, the host, f64, one thread per process. Not measured,
and what a task adopting C6.5 must settle (none of it is a reason to drop it):

1. **Memory.** The waves of a level take 6 × 16 H bytes per box: at p = 8, 27 KB per box
   (893 MB for level 5 of the N = 10⁶ cube on one rank), at p = 12 38 KB, at p = 18 97 KB,
   against 648 B, 1.4 KB and 2.9 KB of multipole. Per rank with one rank per core it is
   the rank's share of the level (plus the halo of its sources). Ways down: process one
   frame at a time (a third), or produce the waves per window of parents in Morton
   order. This is laplace-fmm-plan §3.3's "about 2.5× the memory" made concrete, and
   more.
2. **The node's bandwidth at p ≤ 6** (§6 notes): a rule by machine and p, and the
   blocking of item 1 may recover p = 6.
3. **A SIMD body.** On Kathleen the auto-vectorised loops lose most of the gain against
   `Batched`'s AVX-512 kernel at p = 8 (§6); the shifts and the Fourier passes want the
   hand-written, ISA-dispatched treatment of `nd-fmm-simd` (and its `unsafe` rules).
4. **Accuracy budget and default.** An approximate operator states its budget
   (requirement 1): the matched quadrature per p keeps the level error at most 0.1× the
   M2L truncation, so the FMM's errors within about 1.1×; the gate is the C3.2/C3.3
   errors at f64 p = 8, 12, 18. Not bit for bit `Dense`; deterministic (fixed order per
   accumulator); opt-in until T11.
5. **Several ranks** (requirement 3, bit for bit the one-rank run): a parent's
   accumulators must sum their sources in the same order whatever rank holds them; every
   rank computes the waves of the ghost sources it needs from their multipoles (no new
   exchange), and the merge order is fixed by Morton order, so the bits can be kept.
6. **Adaptive trees.** The V lists of an adaptive tree are same-level pairs; parents
   whose children are partly absent use the masks as they are (FMM3D's adaptive code
   does the same). The plan's interface need not change: the operator derives the parents
   from the Morton keys of its level call.
7. **f32.** The model gives 1.0–2.0× at f32 p = 8; the waves' large terms (λⁿ M̂ₙ up to
   about 3 × 10⁸ at p = 8, s = 18) need checking in f32.
8. **The device.** Modelled at 2.4–20× the dense M2L stage on the H100 (§3), a
   bandwidth-bound kernel with irregular lists; a separate question after the host.
9. **The quadrature.** FMM3D's tables are used; their stated errors understate the
   worst pair (§2), so the adopting task chooses s by measured FMM errors (as §5 did on
   a level) and keeps the tables' provenance (NIST's notice).

## 9. Recommendation (decision 6)

The stop rule is satisfied twice over: the model shows the margin, and the spike
measures 2.2–9.8× the faster of `Batched` and `Rotation` at f64 p ≥ 8 on the M3 Max and
locust with one process and with every core and on Kathleen 1.5–12.5× against `Batched`'s AVX2 kernel; against its AVX-512
kernel 3.6–8.6× at p ≥ 12 and 1.14–1.44× at p = 8. **Recommended: C6.5 proceeds past
its spike** as its own task (an opt-in host strategy, f64, p ≥ 8 first, the matched
quadrature, §8's items as its scope), not added to `nd-fmm-exec` here (the brief's "Do
not"). If it is adopted, laplace-fmm-plan §3.3 and §4 ("at most tens of percent on
CPUs") and the strategy rule of T11 change with it; T11 would weigh it against `Batched`
and `Rotation` at every p.

**Outcome (2026-10-18).** The user signed off decision 6 as recommended: C6.5 proceeds,
as Phase 6 T13 (docs/phase6/T13-plane-wave-m2l.md); a device version after it, if its host
results justify one.

## Appendix: runs and commands

All from the spike's crate, release builds (`cargo build --release -p
nd-fmm-spike-plane-wave --example level`); raw output in `bench-results/t10/` (ignored by
git).

| what | where | command |
| --- | --- | --- |
| model | M3 Max | `cargo run --release -p nd-fmm-spike-plane-wave --example model` |
| quadratures and accuracy | M3 Max | `cargo test -p nd-fmm-spike-plane-wave --release -- --ignored --nocapture` |
| level 4 sweep | M3 Max, 2026-10-18 21:48–21:53, outside the sandbox | `spikes/plane-wave/tools/sweep.sh 12 4` |
| level 5 | M3 Max, 21:54–21:56 | `level --level 5 --p 8 --s 18` and `--p 12 --s 21`, `--variants batched,rotation,merge --check` |
| level 4 sweep | locust, 21:58–22:07, after `tools/gh200/sync.sh` | `spikes/plane-wave/tools/sweep.sh 72 4` (detached, under `timeout`) |
| level 4 sweeps | Kathleen job 605529 (`singlenode`, node-c11a-002, 22:01–22:26) | `tools/kathleen/submit.sh --time=06:00:00 --exclude=node-c11a-011 -- run.sbatch …`: the default build with `sweep.sh 40 4`, then `RUSTFLAGS="-C target-cpu=native"` with `LEVEL_ARGS="--isa avx512" sweep.sh 40 4` |
