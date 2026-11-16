# Helmholtz CH4.0j: the exponential M2L's tables within a level in parts

Task plan: `docs/helmholtz/tasks/CH4.0j.md`. Roadmap: H4, priority list item 6 (added on
2026-11-16 by CH4.0h's sign-off, #221 question 2). Baseline: CH4.0h's report
`fmm-validate/results/helmholtz-ch4-0h.md` (§4, §6) and its launches.

## 0. Summary

- **The change** (`nd-fmm-tables`, `helmholtz::plane_wave` only): a level's exponential-M2L
  tables (`MergeLevel`) were one part of the caller's map since CH4.0h. They are now built in
  three maps, every level's parts together: each quadrature node's α and Fourier counts; each
  node's values of the `PlaneWaveLevel` (cos and sin of mαⱼ, the factors Pₖ(n, m)) and each
  frame's rotation blocks; each of the 152 shift tables. The shift tables are kept one
  allocation each (their serial concatenation was half the time in parts on the M6). No
  formula, layout, value or public API changed.
- **Bit for bit**: the plane-wave tables' hashes (`print_the_hashes`, the `Debug` text of
  every value) are `main`'s and CH4.0h's on the M6, the tables on a reverse-order three-thread
  map are the serial build's, and each level in parts equals the level formed from the public
  steps one by one; every checksum of the machine runs (locust 76 launches, Kathleen 76) equals
  CH4.0h's.
- **The plane-wave tables** at one rank per node fell to 0.05–0.29 of CH4.0h's on locust
  (1 × 72; geometric mean 0.11 over the 45 case-ε pairs where CH4.0h's took at least 10 ms)
  and to 0.08–0.42 on Kathleen (1 × 40; 0.18): 2–9 ms on locust, 4–29 ms on Kathleen, from
  17–95 ms and 31–190 ms.
- **The acceptance** (the build of CH3.6's cases under 1 s at most half CH4.0h's, or why not):
  met on 8 of 51 case-ε pairs on locust and 12 of 43 on Kathleen. On the others the rest of
  the build is above half of CH4.0h's: on 39 (locust) and 28 (Kathleen) even without any
  plane-wave tables (the "floor" of §3), on 4 and 3 within 0.05–0.07 (locust) and 0.08–0.11
  (Kathleen) of the floor. The rest is the
  shared steps, serial in each rank (the plan 5–66 ms, the routes 13–24 ms, the octree
  5–19 ms: CH4.0k, off the list), and at 10⁻³ Batched's matrices (10–104 ms on W4's trees; at
  kD = 1 and 10 they have no level with waves). §3, §4.

## 1. What changed

`nd-fmm-tables` (`helmholtz::plane_wave`; Laplace's tables and every other module untouched):
- `quadrature.rs`: `QuadratureNodes` (crate-private): a quadrature's nodes λₖ, tₖ, the
  weights and each node's share of the stated error; `counts(k)` gives node k's α and Fourier
  counts (three Bessel tails), `with_counts` puts them together in node order.
  `Quadrature::with_reach` is the three in turn.
- `pair.rs`: `LevelLayout` (crate-private): a level's node blocks before their values; its
  parts are each node's values and each frame's rotation blocks (`part(i)`), `with_parts` puts
  them together. `PlaneWaveLevel::new` and `from_nodes` are the parts in turn.
- `merge.rs`: `ShiftOffsets` (the merge's shift offsets, the same on every level, formed once)
  and `build_levels`, the three maps over every level's parts on a `Parts`;
  `PlaneWaveM2l::build_with_reach_on` runs it on its levels, `MergeLevel::with_reach` on one
  level with `Serial`. A level's shift tables are a `Shifts` (one `Vec` per table, a `Debug`
  that lists the values as one `Vec`, so the text is `main`'s); the accessors are unchanged.
  The module documentation's "The build in parts".
- `nd-fmm-exec` unchanged: its operator already passes the pool's `PoolParts` to
  `build_with_reach_on` (CH4.0h).

Tests (`tests/helmholtz/parts.rs`): `the_plane_wave_tables_on_threads_are_the_serial_tables`
widened to seven cases (real and complex k̂₀; the rule at 10⁻³ and 10⁻⁶, iprec 2, and at 10⁻⁸,
iprec 3 at the raised degrees; iprec 1 named; the corner's reach and reach 4);
`every_level_in_parts_is_the_level_by_its_steps` (each level against `Quadrature::with_reach`,
`PlaneWaveLevel::new` and `shift_factors` of every slot's and child's offset, bit for bit: the
merge's staging; the steps themselves now run the node and level parts, so that they are
`main`'s bits rests on the hashes); the ignored `time_a_levels_parts` (§2). The hashes on the M6
(release, `print_the_hashes`), before and after: RCR d99e3221b99a61e3, 71223dfc2b0eba4f,
1520f647ee9b88bd; matrices e8f14c144df269a6, 9218603869fc1abd; plane waves bc43acd5206e0682,
2b62dea0a7d6d982, 38d11660252af3c9, b9419e1bdc4071ca (CH4.0h's).

## 2. A level's build by part

One thread, one level, release (`time_a_levels_parts`, the least of five; ms). "in all" is
`MergeLevel::with_reach`; shifts are in all less the quadrature and the level. The parts of
the three maps: the nodes; the nodes and the two frames; 152 shift tables.

| machine | iprec | κ | p | nodes | waves | quadrature | level | shifts | in all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locust | 1 | 0.5 | 8 | 13 | 473 | 5.0 | 0.08 | 1.6 | 6.7 |
| locust | 1 | 12 | 32 | 26 | 1894 | 13.1 | 1.6 | 6.6 | 21.4 |
| locust | 2 | 0.5 | 8 | 23 | 1372 | 10.2 | 0.20 | 3.8 | 14.1 |
| locust | 2 | 12 | 32 | 35 | 3192 | 19.1 | 1.6 | 11.2 | 31.8 |
| locust | 2 | 45 | 32 | 79 | 16624 | 73.6 | 10.5 | 55.5 | 139.6 |
| locust | 3 | 12 | 32 | 44 | 4830 | 26.2 | 2.4 | 17.1 | 45.8 |
| locust | 3 | 45 | 32 | 87 | 20012 | 83.7 | 13.4 | 67.7 | 164.8 |
| Kathleen | 1 | 0.5 | 8 | 13 | 473 | 8.8 | 0.16 | 2.6 | 11.6 |
| Kathleen | 2 | 12 | 32 | 35 | 3192 | 34.7 | 3.2 | 17.7 | 55.6 |
| Kathleen | 2 | 45 | 32 | 79 | 16624 | 139.8 | 23.5 | 94.9 | 258.2 |
| Kathleen | 3 | 45 | 32 | 87 | 20012 | 159.3 | 25.2 | 114.1 | 298.6 |

The quadrature's counts are 51–76 % of a level, the shift tables 20–45 %, the
`PlaneWaveLevel` 1–11 % (its per-node values about 90 % of it, the frames' blocks 10 %; the M6).
The nodes' counts are even (the largest 1.4× the median at κ = 6–45 on the M6), so the first
map scales with the nodes over the threads. The full grid (κ = 0.5–45, p = 8, 16, 32,
iprec 1–3) is in the raw outputs (`parts.md`).

On many threads the time is the FMM's, on its pool (the build's part "plane-wave tables" of
§3): a map in the test that starts its threads per call timed their start (7 ms at 72 threads
on locust whatever the level) and was dropped. On locust, W1 at kD = 100 at 10⁻³ (three levels
with waves, iprec 2) at 72 threads (temporary timers, not in the tree; 6.5 and 6.9 ms in two
runs): 2.0 ms the first read of FMM3D's quadrature file of the precision (parsed once per
process, serially, on `main` too), the counts 1.8 ms, the level values and their assembly
1.6 ms, the shift tables 1.1–1.6 ms.

## 3. CH3.6's matrix at one rank per node

CH4.0h's matrix (19 cases × 10⁻³, 10⁻⁶, 10⁻⁸, the library's defaults, two evaluations) at
locust 1 × 72 and Kathleen 1 × 40, each case launched in the order `main`, branch, branch,
`main` (`main` 45dcc10 from a tree of its own); times are means of the two launches; CH4.0h
is that task's launch of the same case and split. "Under 1 s": CH4.0h's build plus twice its
evaluation. The floor is CH4.0h's build less its plane-wave tables, over its build: the ratio
with no plane-wave tables at all.

| machine | ε | case-ε under 1 s | at most ½ | build ÷ CH4.0h: geometric mean | min | max | floor: geometric mean | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locust 1 × 72 | 1e-3 | 17 | 1 | 0.84 | 0.40 | 1.06 | 0.79 | 0.25 | 1.00 |
| locust 1 × 72 | 1e-6 | 17 | 2 | 0.72 | 0.32 | 1.00 | 0.66 | 0.20 | 1.00 |
| locust 1 × 72 | 1e-8 | 17 | 5 | 0.54 | 0.19 | 1.00 | 0.48 | 0.14 | 1.00 |
| Kathleen 1 × 40 | 1e-3 | 17 | 4 | 0.79 | 0.36 | 1.50 | 0.70 | 0.17 | 1.00 |
| Kathleen 1 × 40 | 1e-6 | 17 | 5 | 0.61 | 0.24 | 0.89 | 0.52 | 0.17 | 1.00 |
| Kathleen 1 × 40 | 1e-8 | 9 | 3 | 0.50 | 0.19 | 1.33 | 0.39 | 0.12 | 1.00 |

At most ½: locust W1 at kD = 30 (0.19–0.40 at every ε), W3 at kD = 10 (0.40, 0.22 at 10⁻⁶ and
10⁻⁸), W1 at kD = 100 (0.41 at 10⁻⁸), W4's cube and sphere at kD = 100 (0.35, 0.42 at 10⁻⁸);
Kathleen W1 at kD = 30 (0.19–0.36), W1 at kD = 100 (0.43–0.45), W3 at kD = 10 (0.22–0.36),
W4's cube at kD = 100 (0.36–0.45), W4's sphere at kD = 100 (0.46–0.49). The maxima above 1 are
builds of a few milliseconds (W1 at kD = 10 on Kathleen: 3–7 ms, printed to 1 ms).

The cases of CH4.0h's §4 (seconds; plane-wave tables: this branch, `main` in the same session,
CH4.0h):

| machine | case | ε | CH4.0h build | branch build | `main` build | ÷ CH4.0h | plane-wave tables | floor |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| locust | W1 kD = 100 | 1e-3 | 0.121 | 0.063 | 0.119 | 0.52 | 0.007, 0.064, 0.065 | 0.46 |
| locust | W1 kD = 100 | 1e-6 | 0.114 | 0.059 | 0.114 | 0.52 | 0.004, 0.062, 0.062 | 0.46 |
| locust | W1 kD = 100 | 1e-8 | 0.114 | 0.046 | 0.115 | 0.41 | 0.005, 0.076, 0.076 | 0.33 |
| locust | W1 kD = 30 | 1e-3 | 0.036 | 0.014 | 0.036 | 0.40 | 0.004, 0.027, 0.027 | 0.25 |
| locust | W4 Plummer kD = 10 | 1e-3 | 0.117 | 0.118 | 0.117 | 1.01 | 0, 0, 0 | 1.00 |
| Kathleen | W1 kD = 100 | 1e-3 | 0.187 | 0.084 | 0.191 | 0.45 | 0.017, 0.120, 0.119 | 0.36 |
| Kathleen | W1 kD = 100 | 1e-6 | 0.179 | 0.076 | 0.179 | 0.43 | 0.013, 0.115, 0.116 | 0.35 |
| Kathleen | W1 kD = 30 | 1e-3 | 0.058 | 0.021 | 0.061 | 0.36 | 0.009, 0.051, 0.048 | 0.17 |
| Kathleen | W4 cube kD = 100 | 1e-3 | 0.178 | 0.080 | 0.179 | 0.45 | 0.019, 0.119, 0.119 | 0.33 |

`main`'s builds in the same sessions are CH4.0h's to within the launch spread (branch ÷ `main`
and branch ÷ CH4.0h agree within 0.04 on the cases with waves), so CH4.0h's launches are a fair
baseline. The large cases change little: W1 at kD = 500 0.94–0.97 and W3 at kD = 100 0.96–1.00
of CH4.0h's (their plane-wave tables 0.06–0.19 s of 1.6–4.3 s).

## 4. Why not on the others

Of the case-ε pairs under 1 s above ½, the floor is above ½ on 39 of 43 (locust) and 28 of 31
(Kathleen): no build of the plane-wave tables can halve them. Their builds by part (this
branch; the largest first):
- **No level with waves** (W1 and W3 at kD = 10 at 10⁻³ and W1 at kD = 10 throughout, W4's
  four trees at kD = 1 and 10 at 10⁻³): the build is the shared steps and Batched's matrices
  (10–29 ms on locust, 26–76 ms on Kathleen; built eight offsets per part since CH4.0h).
- **The shared steps** everywhere else: the plan with its rules (locust 5–64 ms, Kathleen
  9–66 ms), the routes (13–19, 17–24 ms) and the octree (5–13, 11–19 ms), serial in each rank
  (CH4.0k, off the priority list since #221: Laplace's paths, constraint 1 and D9 runs); at
  10⁻³ also Batched's matrices on the levels without waves (on Kathleen 104 ms of W4's Plummer
  sphere at kD = 100).

`tools/helmholtz/ch4-0j-tables.py` prints each such case's build by part (its last table).

The other 4 (locust: W1 at kD = 100 at 10⁻³ and 10⁻⁶, 0.52 against floors 0.46; W3 at kD = 30
at 10⁻⁸, 0.55 against 0.49; W4's Plummer sphere at kD = 100 at 10⁻⁸, 0.54 against 0.47) and 3
(Kathleen: W3 at kD = 30 at 10⁻⁸, 0.55 against 0.44; W4's cube at kD = 10 at 10⁻⁸, 0.60 against
0.50; W4's Plummer sphere at kD = 100 at 10⁻⁶, 0.55 against 0.47) miss ½ by the 3–18 ms the
plane-wave tables still take (on locust's W1 at kD = 100 a third of it the first read of
FMM3D's file, §2) and the launch spread of the rest: Kathleen's W4 cube at kD = 10 at 10⁻⁸ stays
above ½ without its tables (its rest 0.058 s is 0.53 of CH4.0h's build; CH4.0h's own rest
0.054 s), and locust's W3 at kD = 30 at 10⁻⁸ reaches exactly ½ without them. The rest of those
builds is again the shared steps.

## 5. Runs

- locust (GH200, 72 Neoverse V2 cores, Open MPI 5.0.10), 2026-11-16 15:06–15:12, quiet before
  (load 0.4, GPU idle): `tools/helmholtz/ch4-0j-runs.sh <out> parts`, then `MAIN=<main's
  examples> … matrix` from `4b648aa` (the library as merged here) and `main` 45dcc10, one rank
  bound to 72 cores; the probe of §2's last paragraph after the matrix, from a separate tree
  with temporary timers (removed).
- Kathleen job 634435 (one exclusive node, `singlenode`, node-c11b-047, 2 × 20 Cascade Lake
  cores; 16 min): both release builds, `parts`, the matrix from `4b648aa` and 45dcc10; rustc
  1.99.0, BLAS threads 1, one rank bound to 40 cores.
- The M6: the hashes and tests of §1 (release), no timing reported (the observations of §2 on
  the M6 are the plan's unquiet probes, used only to choose the parts).
- Raw outputs: `~/projects/fmm-archive/{kathleen,locust}/logs/ch4-0j/`; tables from
  `tools/helmholtz/ch4-0j-tables.py matrix <these launches> <CH4.0h's launches>`. Both runs'
  `parts.md` come from the test before its last change (it then also printed the map that
  started threads per call); its serial columns are the test's as merged.

## 6. Questions for sign-off

Signed off on 2026-11-16 (the user on #223: "Yes to both questions"): both as recommended below.

1. **The acceptance as read.** *Recommended*: met. The tables are bit for bit; the plane-wave
   tables' part of the build fell to 0.05–0.42 of CH4.0h's; the build of CH3.6's cases under
   1 s is at most half CH4.0h's on 8 of 51 case-ε pairs (locust) and 12 of 43 (Kathleen), and
   §4 says why not on the others (above ½ even with no plane-wave tables on 39 and 28; on the
   rest within 0.05–0.07 of that floor on locust and 0.08–0.11 on Kathleen, the plane-wave
   tables' remaining 3–18 ms and the launch spread). *Alternative*: not met; then a follow-up for what remains
   is CH4.0k's (Laplace's paths).
2. **CH4.0k's place.** *Recommended*: unchanged, off the priority list; on CH3.6's small cases
   the build is now mostly its shared steps (§4), which several ranks per node already split,
   and H4 (CHA.1–CHA.3, CH1.3d, CH4.1–CH4.5) follows as signed off on #220. *Alternative*:
   CH4.0k on the list after CH4.0j, before CHA.1.
