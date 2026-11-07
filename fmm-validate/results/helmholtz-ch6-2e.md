# Helmholtz CH6.2e: the counts' corner

Helmholtz task CH6.2e ([docs/helmholtz/tasks/CH6.2e.md](../../docs/helmholtz/tasks/CH6.2e.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.2, §2 (constraints 1, 9), §5.3. The counts are CH6.1a's
([docs/helmholtz/tasks/CH6.1a.md](../../docs/helmholtz/tasks/CH6.1a.md)), the corner S4's
(`spikes/pw-quadrature/SPIKE_REPORT.md` §3.2), the verification and scan CH6.1e's
([helmholtz-ch6-1e.md](helmholtz-ch6-1e.md)), the times CH6.2b's harness
([helmholtz-ch6-2b.md](helmholtz-ch6-2b.md)). Measured on:

- **locust** (NVIDIA GH200, 72 Neoverse-V2 cores, 572 GB, `tools/gh200/`), one rank × 72
  threads, 2026-11-07 07:43–08:07, source 0a53256 (the reach setting; the later commits
  change tests, docs and tools only), the node idle before the runs (load 0.00, the GPU
  idle, no other session's job): `tools/helmholtz/ch6-2e-runs.sh` (§7);
- **the M6**: the tests, the per-pair counts on every domain (§1) and a smoke run of the
  ladder on W1; nothing timed.

Release builds, rustc 1.99.0, BLAS threads 1, the P2P kernel `Auto` (NEON), the default
waves budget (8 GiB per node), the rule's precision (iprec 2) on every level with a
quadrature, CH3.3's degrees.

## Summary

- **The setting.** The α counts Nₖ bound the kernel e^{iκ|u|}/|u| on lateral offsets
  ρ ≤ ρ_max = 4√2 ≈ 5.66 box widths, the region's corner (CH6.1a). `Quadrature::with_reach`
  takes a smaller reach in its place (the Fourier counts Fₖ unchanged), up through
  `HelmholtzFmmBuilder::plane_wave_reach`; the corner stays the default, bit for bit.
- **The point-to-point chain needs the corner** (§1): with the counts for ρ ≤ 4 (FMM3D's
  reach) the kernel stays within 2× the stated error on ρ ≤ 4 on every domain (at most
  1.065×, 1.188× and 1.495× at iprec 1, 2 and 3), but at the corner it errs by up to
  1.1·10⁴, 1.1·10⁷ and 1.7·10⁹ times the stated error, on the domains of large Re κ
  (14π–16π): a point pair at the corner gets a wrong value, not a small error. With the
  counts for ρ ≤ 4.5 the bound holds on ρ ≤ 4.5 (at most 1.079×, 1.188×, 1.495×), and the
  corner errs by up to 5.3·10³, 2.7·10⁵ and 1.1·10⁶ times it. This is the chain of points,
  not the FMM's chain of degree-p expansions (§4).
- **In the FMM a reach of 4.25 to 4.5 suffices** (§2): on the 60 cases of CH6.1e's
  verification at the rule's tolerances (10⁻³ to 10⁻⁶, the two large cases included) the
  FMM's error stays within 1.070× Rotation's at reach 5, 4.5 and 4.25, the corner's largest
  ratio: at 5 and 4.5 every case's ratio is the corner's to three digits, at 4.25 within
  0.003 (the W4 cube at kD = 100 and 10⁻⁶, 1.006 against 1.003). Reach 4 misses on three cases,
  all at 10⁻⁶ and kD ≥ 100 (the W4 cube at kD = 100 1.60×, W1 at kD = 500 1.62×, W3 at
  kD = 100 1.69×); 3.5 misses on 12 and 3 on 17. The threshold lies between 4 and 4.25,
  next to the largest lateral offset of two boxes' centres, 3√2 ≈ 4.243 (§4); 4.25 was
  verified on the cases, not scanned level by level.
- **Level by level** (§3): CH6.1e's scan at iprec 2 on CH3.1c's matrix at 10⁻³ and 10⁻⁶,
  166 levels, each alone: at the corner and at 4.5 every level within 1.044×; at 4 one
  level misses, level 2 of the W4 cube at kD = 100 and 10⁻⁶ (κ = 25, p = 35; 1.59×), the
  others as at the corner. 4.25 was not scanned.
- **The waves and times** (§5): the waves per direction per level are 0.84–0.86× the
  corner's at 4.5 (0.77–0.80× at 4); the M2L per level 0.75–0.96× (0.69–0.89× at 4), per
  case summed over its levels 0.80–0.92× (0.73–0.89×), and the evaluation of the large
  cases 0.89–0.95× (0.85–0.90×). The tables and buffers take 0.85–0.88× the corner's bytes
  at 4.5 on the small cases, 0.89× on W1 at kD = 500 and 0.99× on W3 at kD = 100, whose
  waves buffer is held at the node budget.
- **FMM3D's counts at iprec 2** (CH6.2d's question, §6): fewer waves than ours hold only
  down to a reach of about 4.25; the counts for ρ ≤ 4 fail at 10⁻⁶ on three cases, and
  FMM3D's own counts are fewer still where Im κ < π/2 (0.66–0.76× the corner's, CH6.1a,
  against 0.75–0.81× for ours at 4), so they would not hold either on those cases. Not run with FMM3D's
  `hnumphys` itself.
- **For sign-off** (§8): the counts' reach 4.5 in place of 4√2 (0.85× the waves, about
  0.9× the exponential M2L's time), with the per-pair bound restated on ρ ≤ 4.5; the
  adoption as its own task before D8 (CH6.2g).

## 1. Per pair: the corner

`fmm-tables/tests/helmholtz/plane_wave_quadrature.rs`, the ignored
`the_counts_for_a_smaller_reach_on_every_domain` (release, the M6): every domain of iprec
1–3, the kernel at its nine points against e^{iκ|u|}/|u| on the region's grid (z from 1 to
4, x and y from −4 to 4 by ½), and the point-to-point chain (sources at the box's centre,
corners and four seeded points, targets at the up direction's target boxes' corners, edges,
faces and centres) at its centre; the largest |Q − G| over the stated error, within the
reach and on the whole region (the corner's):

| reach | iprec | waves over the corner's (all domains) | (Im κ < π/2) | kernel, ρ ≤ reach | kernel, region | chain, ρ ≤ reach | chain, region |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.720–1.250 | 0.747–0.807 | 1.065 | 1.05·10⁴ | 0.784 | 8.81·10³ |
| 4 | 2 | 0.732–1.091 | 0.759–0.794 | 1.188 | 1.06·10⁷ | 0.716 | 1.04·10⁷ |
| 4 | 3 | 0.713–1.067 | 0.766–0.793 | 1.495 | 1.68·10⁹ | 1.175 | 1.52·10⁹ |
| 4.5 | 1 | 0.667–1.167 | 0.824–0.868 | 1.079 | 5.33·10³ | 0.921 | 2.76·10³ |
| 4.5 | 2 | 0.812–1.083 | 0.832–0.858 | 1.188 | 2.73·10⁵ | 0.716 | 1.62·10⁵ |
| 4.5 | 3 | 0.798–1.083 | 0.837–0.857 | 1.495 | 1.07·10⁶ | 1.175 | 4.95·10⁵ |

The worst region errors are on domains 45, 46 and 23 at reach 4 (Re κ from 14π to 16π, Im κ
below 0.04) and 91, 69 and 138 at 4.5 (the same Re κ, Im κ to 1). Presumably α sampled at
the count for the smaller reach aliases the orders a pair at ρ = 4√2 needs (there the counts
grow about as |tₖ| ρ_max); the errors are of the size of the kernel itself (1/|u| ≈ 0.2,
against stated errors of 10⁻⁴ to 10⁻¹⁰). Within the reach the bound holds as at the corner
(CH6.1a's acceptance: 2× for the kernel, 3× for the chain).
At reach 4 a few domains need more waves than at the corner (up to 1.25× on domain 276,
Re κ in [15π, 16π], Im κ in [4π, 5π]; 1.09× and 1.07× at iprec 2 and 3, also strongly damped
domains; at 4.5 up to 1.17× and 1.08×; and on some domains 4.5 has fewer than 4, at iprec 1
down to 0.667× the corner's against 0.720×). `tail_count` takes the maximum over ρ at 64
equal steps of [0, reach]; a plausible cause, not checked, is that on these domains the
corner's coarser steps miss a peak that a smaller reach's finer steps find, so that the
corner's own counts undersample there. The kernel test on every domain holds at the corner
(CH6.1a), so the default meets its bound on the grid either way; a finer sampling of ρ is
left to CH6.2g. Where Im κ < π/2 (the real-k levels) reach 4 has 0.75–0.81× the corner's
waves and 4.5 0.82–0.87×. The test took 707 s for both reaches on the M6 beside `cargo test`.

## 2. The verification: the FMM's error at each reach

CH6.1e's verification cases at the rule's tolerances (CH3.1c's matrix at 10⁻³ and 10⁻⁶ with
its real-charge and complex-k rows, its in-between cases at 10⁻⁴ and 10⁻⁵, and the two large
cases; ε = 10⁻⁷ and 10⁻⁸ have no waves under the rule), `--m2l rotation,planewave` at each
reach: the larger of the φ and ∇φ ratios to Rotation's error against the sampled direct
sum (1,000 targets). 4.25 ran in a third launch, after the times and scan at 4.5 (its own
Rotation rows, with the first launch's checksums); the waves per level were recorded at the
corner, 4 and 4.5 only (§5), the other reaches by their bytes.

| reach | cases | largest ratio | above 1.1 | tables and buffers over the corner's | the cases above 1.1 |
| --- | ---: | ---: | ---: | --- | --- |
| corner | 60 | 1.070 | 0 | 1 | – |
| 5 | 60 | 1.070 | 0 | 0.92–1.00 | – |
| 4.5 | 60 | 1.070 | 0 | 0.85–0.99 | – |
| 4.25 | 60 | 1.070 | 0 | 0.82–0.99 | – |
| 4 | 60 | 1.692 | 3 | 0.79–0.99 | the W4 cube at kD = 100, W1 at kD = 500, W3 at kD = 100, all at 10⁻⁶ |
| 3.5 | 60 | 522 | 12 | 0.73–0.98 | the W4 cube at kD = 100 at every ε, W1 at kD = 500 and W3 at kD = 100 at both, the W4 sphere and W1's three rows at kD = 100 at 10⁻⁶ |
| 3 | 60 | 15,700 | 17 | 0.66–0.89 | those, W1 at kD = 100 at 10⁻⁴ and 10⁻⁵, and three low-kD cases at 10⁻⁶ (Plummer at kD = 1 1.111×, the sphere at kD = 10 1.115×, W3 at kD = 30 1.102×) |

The bytes are the exponential M2L's tables and buffers (`plane_wave_bytes`). On W3 at
kD = 100 the waves buffer is held at the node budget (8 GiB; more chunks, the same buffer),
so its bytes shrink by 0.6 % at 4.5 while its waves per box shrink as on the small cases; on
W1 at kD = 500 the buffer is below the budget at every reach and the bytes fall to 0.89×.

The cases that miss at reach 4, at every reach:

| workload | kD | ε | corner | 5 | 4.5 | 4.25 | 4 | 3.5 | 3 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 cube (N = 10⁵) | 100 | 10⁻⁶ | 1.003 | 1.003 | 1.003 | 1.006 | 1.596 | 342 | 11,300 |
| W1 | 500 | 10⁻⁶ | 0.986 | 0.986 | 0.986 | 0.987 | 1.623 | 522 | 15,100 |
| W3 (N = 4.0·10⁶) | 100 | 10⁻⁶ | 1.009 | 1.009 | 1.009 | 1.010 | 1.692 | 394 | 15,700 |

At 10⁻³ the same cases hold at 4 (1.000) and miss at 3.5 (W1 at kD = 500 2.59×, W3 at
kD = 100 1.30×, the cube 2.04×). Every case and reach: the appendix, A.1.

## 3. The scan: each level alone

CH6.1e's scan at iprec 2 (`--m2l planewave-scan:2`): Rotation, then each level with a
quadrature on the exponential M2L alone, the others on Rotation; CH3.1c's matrix at 10⁻³
and 10⁻⁶, 166 levels. The ratio to Rotation's error:

| reach | levels | largest ratio | above 1.1 |
| --- | ---: | ---: | ---: |
| corner | 166 | 1.044 | 0 |
| 4.5 | 166 | 1.044 | 0 |
| 4 | 166 | 1.590 | 1 |

The one level that misses at 4 is level 2 of the W4 cube at kD = 100 and 10⁻⁶ (κ = 25,
p = 35: 1.003× at the corner, 1.590× at 4); level 2 of W1 at kD = 100 (the same κ and p)
holds (1.001×), as do every level at 10⁻³ and every level with κ ≤ 12.5. The two large
cases, which miss at 4 too (§2), were not scanned, so their failing levels are not known.
The low-kD cases miss only at 3 (§2), which suggests a level's need grows with κ. Every
level: the appendix, A.2 (the scan at 4.5 ran in the second launch).

## 4. Why about 4.25

The counts bound the point-to-point chain, whose lateral offsets reach the corner 4√2: two
points at opposite corners of two boxes whose centres are 3 boxes apart in x and y. The FMM's
chain is not point to point: the waves come from a multipole of degree p_l about the source
box's centre and end in a local of degree p_l about the target box's centre, so the α
trapezoid has to resolve the shift between the two centres, at most 3√2 ≈ 4.243 laterally,
and the orders of the two expansions, which `Quadrature` already counts apart (the count for
ρ − r_s plus Fₖ). The measured threshold, between 4 (misses at 10⁻⁶ from κ = 25) and 4.25
(holds everywhere), agrees with the centres' offset; it is an observation on these cases,
not a bound. Why a pair's error at the corner (§1) does not reach the FMM is not measured
here; presumably the degree-p expansions do not carry the orders that alias there.

## 5. The waves and times

CH6.2b's matrix and large cases at 10⁻³ and 10⁻⁶, `--m2l planewave --evaluations 2
--levels`, the corner and the other reach in one launch, then reversed (ABBA); each level's
M2L seconds (the second evaluation's, the mean of the two launches). Reach 4 ran first
(07:43–07:48), 4.5 after the accuracy stages (07:58–08:03), each with the corner beside it.

Summed over each case's levels, and the evaluation (printed to 10 ms, so its ratio only from
1 s):

| workload | kD | ε | M2L s, corner | 4.5 | ratio | 4 | ratio to its corner | evaluate s, corner | 4.5 | ratio | 4 | ratio to its corner |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 500 | 1e-3 | 3.3710 | 3.0632 | 0.909 | 2.9622 | 0.873 | 3.92 | 3.70 | 0.945 | 3.52 | 0.893 |
| W1 | 500 | 1e-6 | 4.2887 | 3.9338 | 0.917 | 3.8048 | 0.891 | 5.04 | 4.69 | 0.930 | 4.55 | 0.904 |
| W1 | 30 | 1e-3 | 0.0145 | 0.0128 | 0.878 | 0.0122 | 0.817 | 0.02 | 0.02 | – | 0.02 | – |
| W1 | 30 | 1e-6 | 0.0211 | 0.0190 | 0.897 | 0.0182 | 0.847 | 0.02 | 0.02 | – | 0.02 | – |
| W1 | 100 | 1e-3 | 0.1160 | 0.1019 | 0.878 | 0.0959 | 0.823 | 0.15 | 0.13 | – | 0.13 | – |
| W1 | 100 | 1e-6 | 0.1457 | 0.1300 | 0.892 | 0.1234 | 0.831 | 0.21 | 0.19 | – | 0.19 | – |
| W1 | 10 | 1e-3 | 0.0042 | 0.0033 | 0.798 | 0.0032 | 0.728 | 0.01 | 0.01 | – | 0.00 | – |
| W1 | 10 | 1e-6 | 0.0073 | 0.0066 | 0.905 | 0.0064 | 0.812 | 0.01 | 0.01 | – | 0.01 | – |
| W3 (N = 108849) | 30 | 1e-3 | 0.0322 | 0.0278 | 0.861 | 0.0261 | 0.810 | 0.06 | 0.05 | – | 0.05 | – |
| W3 (N = 108849) | 30 | 1e-6 | 0.0481 | 0.0427 | 0.886 | 0.0405 | 0.843 | 0.08 | 0.07 | – | 0.07 | – |
| W3 (N = 4031442) | 100 | 1e-3 | 1.3554 | 1.1210 | 0.827 | 1.0339 | 0.780 | 2.04 | 1.81 | 0.887 | 1.71 | 0.853 |
| W3 (N = 4031442) | 100 | 1e-6 | 1.7757 | 1.5744 | 0.887 | 1.4547 | 0.816 | 2.74 | 2.54 | 0.927 | 2.42 | 0.882 |
| W3 (N = 4032) | 10 | 1e-3 | 0.0075 | 0.0066 | 0.879 | 0.0063 | 0.780 | 0.01 | 0.01 | – | 0.01 | – |
| W3 (N = 4032) | 10 | 1e-6 | 0.0138 | 0.0124 | 0.901 | 0.0120 | 0.858 | 0.03 | 0.03 | – | 0.03 | – |
| W4 clusters (N = 100000) | 1 | 1e-3 | 0.0416 | 0.0363 | 0.872 | 0.0339 | 0.814 | 0.10 | 0.09 | – | 0.09 | – |
| W4 clusters (N = 100000) | 1 | 1e-6 | 0.0737 | 0.0643 | 0.872 | 0.0619 | 0.850 | 0.21 | 0.21 | – | 0.21 | – |
| W4 clusters (N = 100000) | 100 | 1e-3 | 0.0715 | 0.0620 | 0.867 | 0.0578 | 0.816 | 0.13 | 0.12 | – | 0.11 | – |
| W4 clusters (N = 100000) | 100 | 1e-6 | 0.1041 | 0.0930 | 0.894 | 0.0872 | 0.839 | 0.25 | 0.23 | – | 0.23 | – |
| W4 cube (N = 100000) | 1 | 1e-3 | 0.0229 | 0.0198 | 0.865 | 0.0185 | 0.800 | 0.04 | 0.04 | – | 0.04 | – |
| W4 cube (N = 100000) | 1 | 1e-6 | 0.0382 | 0.0336 | 0.879 | 0.0318 | 0.828 | 0.07 | 0.06 | – | 0.06 | – |
| W4 cube (N = 100000) | 100 | 1e-3 | 0.1082 | 0.0951 | 0.878 | 0.0897 | 0.830 | 0.13 | 0.12 | – | 0.11 | – |
| W4 cube (N = 100000) | 100 | 1e-6 | 0.1291 | 0.1146 | 0.888 | 0.1079 | 0.837 | 0.16 | 0.15 | – | 0.14 | – |
| W4 plummer (N = 100000) | 1 | 1e-3 | 0.0408 | 0.0361 | 0.885 | 0.0337 | 0.811 | 0.10 | 0.10 | – | 0.10 | – |
| W4 plummer (N = 100000) | 1 | 1e-6 | 0.0709 | 0.0629 | 0.888 | 0.0606 | 0.847 | 0.23 | 0.23 | – | 0.22 | – |
| W4 plummer (N = 100000) | 100 | 1e-3 | 0.1275 | 0.1116 | 0.875 | 0.1050 | 0.821 | 0.19 | 0.17 | – | 0.17 | – |
| W4 plummer (N = 100000) | 100 | 1e-6 | 0.1659 | 0.1473 | 0.888 | 0.1389 | 0.842 | 0.32 | 0.30 | – | 0.29 | – |
| W4 sphere (N = 100000) | 1 | 1e-3 | 0.0336 | 0.0294 | 0.874 | 0.0273 | 0.810 | 0.06 | 0.06 | – | 0.06 | – |
| W4 sphere (N = 100000) | 1 | 1e-6 | 0.0583 | 0.0523 | 0.898 | 0.0489 | 0.831 | 0.12 | 0.12 | – | 0.11 | – |
| W4 sphere (N = 100000) | 100 | 1e-3 | 0.1198 | 0.1049 | 0.876 | 0.0977 | 0.820 | 0.15 | 0.14 | – | 0.13 | – |
| W4 sphere (N = 100000) | 100 | 1e-6 | 0.1512 | 0.1343 | 0.888 | 0.1266 | 0.838 | 0.22 | 0.20 | – | 0.19 | – |

Per level (the appendix, A.3): the waves per direction 0.84–0.86× the corner's at 4.5
(0.77–0.80× at 4), the M2L 0.75–0.96× (0.69–0.89×). The time falls by less than the waves,
as expected where part of a level call does not scale with Nₖ (the multipoles' Fourier
modes of Fₖ orders, the locals, the chunking); the split is not measured here.

## 6. FMM3D's counts at iprec 2

CH6.2d (report §4) found FMM3D's counts too few in the FMM at iprec 1 (one level at κ ≈ 12.5
on its plane waves: 0.9–2.0·10⁻³ at ε = 10⁻³). Its counts resolve about ρ = 4 (S4 §3.2) and
are 0.66–0.76× the corner's where Im κ < π/2 (CH6.1a: the corner's are 1.31–1.51× FMM3D's).
Ours at reach 4 are 0.75–0.81× the corner's there (§1) and already miss at 10⁻⁶ on three
cases at iprec 2 (§2), so counts as few as FMM3D's would not hold at iprec 2 either on those
cases. This is an inference from our counts; FMM3D's `hnumphys` is not ported and was not
run.

## 7. Commands

On locust, after `tools/gh200/sync.sh` (source 0a53256), from the tree's root:

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch6-2e-runs.sh /data/ucahtbe/logs/ch6-2e times verify verify-large scan
REACH=4.5 SCAN_REACHES=4.5 tools/helmholtz/ch6-2e-runs.sh /data/ucahtbe/logs/ch6-2e-45 times scan
REACHES=4.25 tools/helmholtz/ch6-2e-runs.sh /data/ucahtbe/logs/ch6-2e-425 verify verify-large
```

The tables: `tools/helmholtz/ch6-2e-tables.py <out-dir>[,<out-dir>…] verify|scan|times`;
§2 and A.1 from `…/ch6-2e,…/ch6-2e-425 verify`, §3 and A.2 from `…/ch6-2e,…/ch6-2e-45
scan`, §5 and A.3 from each times directory alone. Every launch
exited 0 (38, 24 and 14 launches). The raw outputs are in
`~/projects/fmm-archive/locust/data/ucahtbe/logs/ch6-2e*` on the M6. The per-pair counts:
`cargo test --release -p nd-fmm-tables --test helmholtz -- --ignored
the_counts_for_a_smaller_reach_on_every_domain --nocapture`.

## 8. Questions for sign-off

1. **The counts' reach 4.5 in place of 4√2.** The FMM's errors hold at 4.5 as at the
   corner on all 60 cases and 166 scanned levels, and at 4.25 on the 60 cases (not
   scanned); 4.5 is a step of the ladder above the threshold (4 misses). It gives 0.84–0.86×
   the waves per level and 0.80–0.92× the exponential M2L's time per case. A point pair at
   the corner then errs by far more than the stated error (§1); whether an FMM pair at the
   largest offset does is not measured (§4), and D12's workloads do not show it.
   Recommendation: adopt 4.5 in CH6.2g, with a check this task lacks: one M2L pair at the
   largest V offset (3, 3, 2) at the level's degree against Rotation on every domain, at
   reach 4.5; CH6.1a's acceptance and the module's statement of the bound on ρ ≤ 4.5 (the
   kernel and the point-to-point chain within 2× and 3× there); and CONVENTIONS §3.15's
   sentence on the quadrature's region ("|u_x|, |u_y| ≤ 4") amended to say the counts bound
   ρ ≤ 4.5, without a version bump (its subsection "Exponential M2L" is unversioned, as
   CH6.1c's sign-off made it). Alternatives: 3√2 plus a margin as a formula; a reach by κ
   (the low-κ cases hold to 3.5, §2), which saves more where waves are few anyway; keep the
   corner.
2. **The adoption as its own task, CH6.2g, before CH6.2f (D8)** (the roadmap lists it so,
   pending this answer). Changing the default
   changes every level's waves, so CH6.1e's verification and nd-fmm-exec's tests rerun, and
   D8's times should be taken with the counts it will ship. Recommendation: CH6.2g before
   CH6.2f; alternative: fold it into CH6.2f.

## Appendix

### A.1 The verification, every case

| workload | kD | ε | ρ corner | ρ 5 | ρ 4.5 | ρ 4.25 | ρ 4 | ρ 3.5 | ρ 3 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 clusters (N = 100000) | 1 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 |
| W4 clusters (N = 100000) | 1 | 1e-5 | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 | 0.996 |
| W4 clusters (N = 100000) | 100 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-5 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 |
| W4 cube (N = 100000) | 1 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | 1e-5 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.997 |
| W4 cube (N = 100000) | 100 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8.750 | 286.980 |
| W4 cube (N = 100000) | 100 | 1e-5 | 1.001 | 1.001 | 1.001 | 1.001 | 1.007 | 34.441 | 1139.282 |
| W1 | 10 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1 | 10 | 1e-5 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 |
| W1 | 30 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1 | 30 | 1e-5 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.002 |
| W1 | 100 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.802 |
| W1 | 100 | 1e-5 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 1.077 | 9.341 |
| W3 (N = 4032) | 10 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | 1e-5 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.001 | 1.001 |
| W3 (N = 108849) | 30 | 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | 1e-5 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 0.999 | 1.007 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 0.970 | 0.970 | 0.970 | 0.970 | 0.969 | 0.967 | 0.981 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 0.951 | 0.951 | 0.951 | 0.951 | 0.950 | 0.950 | 0.951 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 | 0.968 | 0.973 |
| W4 cube (N = 100000) | 1 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | 1e-6 | 0.982 | 0.982 | 0.982 | 0.982 | 0.983 | 0.982 | 0.999 |
| W4 cube (N = 100000) | 10 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | 1e-6 | 1.004 | 1.004 | 1.004 | 1.004 | 1.004 | 1.004 | 1.068 |
| W4 cube (N = 100000) | 100 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.039 | 58.964 |
| W4 cube (N = 100000) | 100 | 1e-6 | 1.003 | 1.003 | 1.003 | 1.006 | 1.596 | 341.530 | 11306.623 |
| W1 | 500 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.589 | 69.221 |
| W1 | 500 | 1e-6 | 0.986 | 0.986 | 0.986 | 0.987 | 1.623 | 522.153 | 15065.151 |
| W3 (N = 4031442) | 100 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.296 | 32.607 |
| W3 (N = 4031442) | 100 | 1e-6 | 1.009 | 1.009 | 1.009 | 1.010 | 1.692 | 393.723 | 15722.671 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 1.070 | 1.070 | 1.070 | 1.070 | 1.070 | 1.075 | 1.111 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 1.038 | 1.038 | 1.038 | 1.038 | 1.038 | 1.038 | 1.050 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 1.015 | 1.015 | 1.015 | 1.015 | 1.016 | 1.015 | 1.033 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 0.986 | 0.986 | 0.986 | 0.986 | 0.985 | 0.989 | 0.978 |
| W4 sphere (N = 100000) | 10 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | 1e-6 | 0.994 | 0.994 | 0.994 | 0.995 | 0.995 | 0.999 | 1.115 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.032 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 3.448 | 139.898 |
| W1 | 100 + 10i | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.018 |
| W1 | 100 + 10i | 1e-6 | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.516 | 55.870 |
| W1 | 100 | 1e-3, real | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.008 |
| W1 | 100 | 1e-6, real | 0.987 | 0.987 | 0.987 | 0.987 | 0.988 | 1.317 | 57.033 |
| W1 | 10 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1 | 10 | 1e-6 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 |
| W1 | 30 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W1 | 30 | 1e-6 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.004 | 1.013 |
| W1 | 100 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.023 |
| W1 | 100 | 1e-6 | 0.995 | 0.995 | 0.995 | 0.994 | 0.999 | 2.433 | 75.866 |
| W3 (N = 4032) | 10 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | 1e-6 | 0.991 | 0.991 | 0.991 | 0.991 | 0.990 | 0.991 | 1.001 |
| W3 (N = 108849) | 30 | 1e-3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | 1e-6 | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 | 0.977 | 1.102 |

### A.2 The scan, every level

| workload | kD | ε | level | ρ corner | ρ 4.5 | ρ 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| W4 clusters (N = 100000) | 1 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 6 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 7 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 8 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 3 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 4 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 5 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 6 | 0.996 | 0.996 | 0.996 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 7 | 1.016 | 1.016 | 1.016 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 8 | 0.967 | 0.967 | 0.967 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 6 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 7 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-3 | 8 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 3 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 4 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 5 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 6 | 0.997 | 0.997 | 0.997 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 7 | 0.998 | 0.998 | 0.998 |
| W4 clusters (N = 100000) | 10 | 1e-6 | 8 | 0.968 | 0.968 | 0.968 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 6 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 7 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 8 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 3 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 4 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 5 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 6 | 0.997 | 0.997 | 0.997 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 7 | 1.005 | 1.005 | 1.005 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 8 | 0.972 | 0.972 | 0.972 |
| W4 cube (N = 100000) | 1 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | 1e-6 | 2 | 0.997 | 0.997 | 0.997 |
| W4 cube (N = 100000) | 1 | 1e-6 | 3 | 0.982 | 0.982 | 0.982 |
| W4 cube (N = 100000) | 1 | 1e-6 | 4 | 1.003 | 1.003 | 1.003 |
| W4 cube (N = 100000) | 10 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | 1e-6 | 2 | 0.994 | 0.994 | 0.994 |
| W4 cube (N = 100000) | 10 | 1e-6 | 3 | 1.003 | 1.003 | 1.003 |
| W4 cube (N = 100000) | 10 | 1e-6 | 4 | 1.011 | 1.011 | 1.011 |
| W4 cube (N = 100000) | 100 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | 1e-6 | 2 | 1.003 | 1.003 | 1.590 |
| W4 cube (N = 100000) | 100 | 1e-6 | 3 | 1.006 | 1.006 | 1.006 |
| W4 cube (N = 100000) | 100 | 1e-6 | 4 | 0.994 | 0.994 | 0.994 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 6 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 7 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 8 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 3 | 1.002 | 1.002 | 1.002 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 4 | 1.002 | 1.002 | 1.002 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 5 | 0.997 | 0.997 | 0.997 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 6 | 1.044 | 1.044 | 1.044 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 7 | 1.020 | 1.020 | 1.020 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 8 | 1.002 | 1.002 | 1.002 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 6 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 7 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-3 | 8 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 3 | 1.003 | 1.003 | 1.003 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 4 | 1.010 | 1.010 | 1.010 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 5 | 0.997 | 0.997 | 0.997 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 6 | 1.018 | 1.018 | 1.018 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 7 | 1.008 | 1.008 | 1.008 |
| W4 plummer (N = 100000) | 10 | 1e-6 | 8 | 1.002 | 1.002 | 1.002 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 6 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 7 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 8 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 3 | 1.001 | 1.001 | 1.001 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 4 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 5 | 1.008 | 1.008 | 1.008 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 6 | 1.007 | 1.007 | 1.007 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 7 | 0.998 | 0.998 | 0.998 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 8 | 1.001 | 1.001 | 1.001 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 2 | 1.001 | 1.001 | 1.001 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 3 | 0.997 | 0.997 | 0.997 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 4 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 5 | 0.990 | 0.990 | 0.990 |
| W4 sphere (N = 100000) | 10 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | 1e-6 | 2 | 0.999 | 0.999 | 0.999 |
| W4 sphere (N = 100000) | 10 | 1e-6 | 3 | 1.003 | 1.003 | 1.004 |
| W4 sphere (N = 100000) | 10 | 1e-6 | 4 | 1.007 | 1.007 | 1.007 |
| W4 sphere (N = 100000) | 10 | 1e-6 | 5 | 0.988 | 0.988 | 0.988 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 2 | 1.004 | 1.004 | 1.004 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 3 | 1.005 | 1.005 | 1.005 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 4 | 1.000 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 5 | 0.994 | 0.994 | 0.994 |
| W1 | 100 + 10i | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 100 + 10i | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W1 | 100 + 10i | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W1 | 100 + 10i | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W1 | 100 + 10i | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 100 + 10i | 1e-6 | 3 | 1.000 | 1.000 | 1.000 |
| W1 | 100 + 10i | 1e-6 | 4 | 1.005 | 1.005 | 1.005 |
| W1 | 100 + 10i | 1e-6 | 5 | 0.999 | 0.999 | 0.999 |
| W1 | 100 | 1e-3, real | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-3, real | 3 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-3, real | 4 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-3, real | 5 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-6, real | 2 | 1.001 | 1.001 | 1.001 |
| W1 | 100 | 1e-6, real | 3 | 1.002 | 1.002 | 1.003 |
| W1 | 100 | 1e-6, real | 4 | 0.984 | 0.984 | 0.984 |
| W1 | 100 | 1e-6, real | 5 | 1.001 | 1.001 | 1.001 |
| W1 | 10 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 10 | 1e-6 | 2 | 1.005 | 1.005 | 1.005 |
| W1 | 30 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 30 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W1 | 30 | 1e-6 | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 30 | 1e-6 | 3 | 1.001 | 1.001 | 1.001 |
| W1 | 100 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-3 | 5 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | 1e-6 | 2 | 1.001 | 1.001 | 1.001 |
| W1 | 100 | 1e-6 | 3 | 1.005 | 1.005 | 1.010 |
| W1 | 100 | 1e-6 | 4 | 0.988 | 0.988 | 0.988 |
| W1 | 100 | 1e-6 | 5 | 1.001 | 1.001 | 1.001 |
| W3 (N = 4032) | 10 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | 1e-6 | 2 | 0.989 | 0.989 | 0.989 |
| W3 (N = 4032) | 10 | 1e-6 | 3 | 1.002 | 1.002 | 1.002 |
| W3 (N = 108849) | 30 | 1e-3 | 2 | 1.000 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | 1e-3 | 3 | 1.000 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | 1e-3 | 4 | 1.000 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | 1e-6 | 2 | 0.986 | 0.986 | 0.986 |
| W3 (N = 108849) | 30 | 1e-6 | 3 | 1.003 | 1.003 | 1.003 |
| W3 (N = 108849) | 30 | 1e-6 | 4 | 1.001 | 1.001 | 1.001 |

### A.3 Per level, the waves and times

Reach 4.5 beside the corner (07:58–08:03):

| workload | kD | ε | level | p | V pairs | waves corner | waves ρ 4.5 | ratio | M2L s corner | M2L s ρ 4.5 | ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 500 | 1e-3 | 4 | 36 | 511200 | 9369 | 7865 | 0.839 | 0.2852 | 0.2471 | 0.866 |
| W1 | 500 | 1e-3 | 5 | 21 | 2935296 | 3943 | 3343 | 0.848 | 0.3714 | 0.3203 | 0.862 |
| W1 | 500 | 1e-3 | 6 | 13 | 10439280 | 2318 | 1984 | 0.856 | 0.5761 | 0.4869 | 0.845 |
| W1 | 500 | 1e-3 | 7 | 8 | 14492220 | 1854 | 1593 | 0.859 | 0.6205 | 0.5177 | 0.834 |
| W1 | 500 | 1e-6 | 4 | 41 | 511200 | 9369 | 7865 | 0.839 | 0.3115 | 0.2697 | 0.866 |
| W1 | 500 | 1e-6 | 5 | 25 | 2935296 | 3943 | 3343 | 0.848 | 0.4185 | 0.3659 | 0.874 |
| W1 | 500 | 1e-6 | 6 | 16 | 10439280 | 2318 | 1984 | 0.856 | 0.6508 | 0.5643 | 0.867 |
| W1 | 500 | 1e-6 | 7 | 15 | 14492220 | 1854 | 1593 | 0.859 | 0.9533 | 0.7900 | 0.829 |
| W1 | 30 | 1e-3 | 2 | 12 | 3096 | 2274 | 1951 | 0.858 | 0.0081 | 0.0072 | 0.881 |
| W1 | 30 | 1e-3 | 3 | 8 | 32256 | 1834 | 1574 | 0.858 | 0.0064 | 0.0056 | 0.873 |
| W1 | 30 | 1e-6 | 2 | 16 | 3096 | 2274 | 1951 | 0.858 | 0.0105 | 0.0094 | 0.899 |
| W1 | 30 | 1e-6 | 3 | 15 | 32256 | 1834 | 1574 | 0.858 | 0.0106 | 0.0095 | 0.897 |
| W1 | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5867 | 0.844 | 0.0523 | 0.0466 | 0.892 |
| W1 | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2784 | 0.851 | 0.0261 | 0.0231 | 0.885 |
| W1 | 100 | 1e-3 | 4 | 11 | 385344 | 2136 | 1824 | 0.854 | 0.0225 | 0.0189 | 0.842 |
| W1 | 100 | 1e-3 | 5 | 7 | 244944 | 1599 | 1370 | 0.857 | 0.0152 | 0.0132 | 0.869 |
| W1 | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5867 | 0.844 | 0.0606 | 0.0547 | 0.903 |
| W1 | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2784 | 0.851 | 0.0310 | 0.0278 | 0.897 |
| W1 | 100 | 1e-6 | 4 | 15 | 385344 | 2136 | 1824 | 0.854 | 0.0278 | 0.0239 | 0.860 |
| W1 | 100 | 1e-6 | 5 | 15 | 244944 | 1599 | 1370 | 0.857 | 0.0263 | 0.0236 | 0.896 |
| W1 | 10 | 1e-3 | 2 | 7 | 3096 | 1546 | 1321 | 0.854 | 0.0042 | 0.0033 | 0.795 |
| W1 | 10 | 1e-6 | 2 | 15 | 3096 | 1546 | 1321 | 0.854 | 0.0072 | 0.0065 | 0.901 |
| W3 (N = 108849) | 30 | 1e-3 | 2 | 12 | 3096 | 2274 | 1951 | 0.858 | 0.0083 | 0.0074 | 0.894 |
| W3 (N = 108849) | 30 | 1e-3 | 3 | 8 | 53352 | 1834 | 1574 | 0.858 | 0.0082 | 0.0072 | 0.881 |
| W3 (N = 108849) | 30 | 1e-3 | 4 | 7 | 584136 | 1494 | 1278 | 0.855 | 0.0157 | 0.0130 | 0.832 |
| W3 (N = 108849) | 30 | 1e-6 | 2 | 16 | 3096 | 2274 | 1951 | 0.858 | 0.0108 | 0.0097 | 0.904 |
| W3 (N = 108849) | 30 | 1e-6 | 3 | 15 | 53352 | 1834 | 1574 | 0.858 | 0.0132 | 0.0118 | 0.888 |
| W3 (N = 108849) | 30 | 1e-6 | 4 | 15 | 584136 | 1494 | 1278 | 0.855 | 0.0241 | 0.0212 | 0.877 |
| W3 (N = 4031442) | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5867 | 0.844 | 0.0524 | 0.0470 | 0.897 |
| W3 (N = 4031442) | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2784 | 0.851 | 0.0264 | 0.0233 | 0.885 |
| W3 (N = 4031442) | 100 | 1e-3 | 4 | 11 | 584136 | 2136 | 1824 | 0.854 | 0.0291 | 0.0248 | 0.850 |
| W3 (N = 4031442) | 100 | 1e-3 | 5 | 7 | 5398920 | 1599 | 1370 | 0.857 | 0.1494 | 0.1223 | 0.819 |
| W3 (N = 4031442) | 100 | 1e-3 | 6 | 7 | 46298376 | 1397 | 1194 | 0.855 | 1.0981 | 0.9036 | 0.823 |
| W3 (N = 4031442) | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5867 | 0.844 | 0.0619 | 0.0558 | 0.902 |
| W3 (N = 4031442) | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2784 | 0.851 | 0.0322 | 0.0282 | 0.877 |
| W3 (N = 4031442) | 100 | 1e-6 | 4 | 15 | 584136 | 2136 | 1824 | 0.854 | 0.0355 | 0.0320 | 0.900 |
| W3 (N = 4031442) | 100 | 1e-6 | 5 | 15 | 5398920 | 1599 | 1370 | 0.857 | 0.2014 | 0.1765 | 0.876 |
| W3 (N = 4031442) | 100 | 1e-6 | 6 | 15 | 46298376 | 1397 | 1194 | 0.855 | 1.4446 | 1.2819 | 0.887 |
| W3 (N = 4032) | 10 | 1e-3 | 2 | 7 | 3096 | 1546 | 1321 | 0.854 | 0.0038 | 0.0033 | 0.877 |
| W3 (N = 4032) | 10 | 1e-3 | 3 | 7 | 5544 | 1375 | 1177 | 0.856 | 0.0037 | 0.0032 | 0.881 |
| W3 (N = 4032) | 10 | 1e-6 | 2 | 15 | 3096 | 1546 | 1321 | 0.854 | 0.0069 | 0.0063 | 0.905 |
| W3 (N = 4032) | 10 | 1e-6 | 3 | 15 | 5544 | 1375 | 1177 | 0.856 | 0.0068 | 0.0061 | 0.897 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1352 | 1157 | 0.856 | 0.0035 | 0.0031 | 0.898 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 3 | 7 | 19782 | 1361 | 1165 | 0.856 | 0.0036 | 0.0032 | 0.880 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 4 | 7 | 43392 | 1260 | 1074 | 0.852 | 0.0039 | 0.0034 | 0.890 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 5 | 7 | 106500 | 1265 | 1081 | 0.855 | 0.0050 | 0.0044 | 0.872 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 6 | 7 | 214818 | 1264 | 1080 | 0.854 | 0.0085 | 0.0074 | 0.874 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 7 | 7 | 289194 | 1263 | 1080 | 0.855 | 0.0080 | 0.0068 | 0.856 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 8 | 7 | 351936 | 1263 | 1080 | 0.855 | 0.0092 | 0.0079 | 0.861 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1352 | 1157 | 0.856 | 0.0066 | 0.0059 | 0.906 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 3 | 15 | 19782 | 1361 | 1165 | 0.856 | 0.0065 | 0.0058 | 0.900 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 4 | 15 | 43392 | 1260 | 1074 | 0.852 | 0.0067 | 0.0061 | 0.903 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 5 | 15 | 106500 | 1265 | 1081 | 0.855 | 0.0087 | 0.0077 | 0.891 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 6 | 15 | 214818 | 1264 | 1080 | 0.854 | 0.0157 | 0.0130 | 0.832 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 7 | 15 | 289194 | 1263 | 1080 | 0.855 | 0.0135 | 0.0119 | 0.883 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 8 | 15 | 351936 | 1263 | 1080 | 0.855 | 0.0162 | 0.0138 | 0.853 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 2 | 21 | 3096 | 3948 | 3353 | 0.849 | 0.0219 | 0.0197 | 0.898 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 3 | 13 | 19782 | 2321 | 1985 | 0.855 | 0.0090 | 0.0080 | 0.888 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 4 | 8 | 43392 | 1857 | 1595 | 0.859 | 0.0070 | 0.0053 | 0.747 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 5 | 7 | 106500 | 1495 | 1284 | 0.859 | 0.0059 | 0.0051 | 0.854 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 6 | 7 | 214818 | 1394 | 1191 | 0.854 | 0.0093 | 0.0081 | 0.874 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 7 | 7 | 289194 | 1372 | 1171 | 0.853 | 0.0087 | 0.0075 | 0.865 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 8 | 7 | 351936 | 1324 | 1134 | 0.856 | 0.0096 | 0.0083 | 0.865 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 2 | 25 | 3096 | 3948 | 3353 | 0.849 | 0.0264 | 0.0236 | 0.897 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 3 | 16 | 19782 | 2321 | 1985 | 0.855 | 0.0110 | 0.0097 | 0.880 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 4 | 15 | 43392 | 1857 | 1595 | 0.859 | 0.0097 | 0.0087 | 0.894 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 5 | 15 | 106500 | 1495 | 1284 | 0.859 | 0.0100 | 0.0095 | 0.944 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 6 | 15 | 214818 | 1394 | 1191 | 0.854 | 0.0161 | 0.0143 | 0.889 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 7 | 15 | 289194 | 1372 | 1171 | 0.853 | 0.0146 | 0.0129 | 0.881 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 8 | 15 | 351936 | 1324 | 1134 | 0.856 | 0.0163 | 0.0144 | 0.883 |
| W4 cube (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1324 | 1134 | 0.856 | 0.0035 | 0.0031 | 0.880 |
| W4 cube (N = 100000) | 1 | 1e-3 | 3 | 7 | 53352 | 1352 | 1156 | 0.855 | 0.0056 | 0.0050 | 0.885 |
| W4 cube (N = 100000) | 1 | 1e-3 | 4 | 7 | 584136 | 1361 | 1164 | 0.855 | 0.0138 | 0.0118 | 0.854 |
| W4 cube (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1324 | 1134 | 0.856 | 0.0063 | 0.0057 | 0.899 |
| W4 cube (N = 100000) | 1 | 1e-6 | 3 | 15 | 53352 | 1352 | 1156 | 0.855 | 0.0099 | 0.0088 | 0.888 |
| W4 cube (N = 100000) | 1 | 1e-6 | 4 | 15 | 584136 | 1361 | 1164 | 0.855 | 0.0220 | 0.0192 | 0.869 |
| W4 cube (N = 100000) | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5867 | 0.844 | 0.0528 | 0.0471 | 0.891 |
| W4 cube (N = 100000) | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2784 | 0.851 | 0.0263 | 0.0233 | 0.886 |
| W4 cube (N = 100000) | 100 | 1e-3 | 4 | 11 | 584136 | 2136 | 1824 | 0.854 | 0.0291 | 0.0247 | 0.849 |
| W4 cube (N = 100000) | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5867 | 0.844 | 0.0617 | 0.0554 | 0.898 |
| W4 cube (N = 100000) | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2784 | 0.851 | 0.0316 | 0.0288 | 0.911 |
| W4 cube (N = 100000) | 100 | 1e-6 | 4 | 15 | 584136 | 2136 | 1824 | 0.854 | 0.0359 | 0.0305 | 0.849 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1324 | 1134 | 0.856 | 0.0034 | 0.0030 | 0.885 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 3 | 7 | 53352 | 1352 | 1156 | 0.855 | 0.0056 | 0.0049 | 0.875 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 4 | 7 | 53352 | 1361 | 1164 | 0.855 | 0.0056 | 0.0049 | 0.870 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 5 | 7 | 109548 | 1260 | 1074 | 0.852 | 0.0064 | 0.0057 | 0.883 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 6 | 7 | 205458 | 1265 | 1081 | 0.855 | 0.0070 | 0.0061 | 0.865 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 7 | 7 | 333330 | 1264 | 1080 | 0.854 | 0.0073 | 0.0068 | 0.935 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 8 | 7 | 35148 | 1263 | 1080 | 0.855 | 0.0053 | 0.0046 | 0.871 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1324 | 1134 | 0.856 | 0.0062 | 0.0057 | 0.914 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 3 | 15 | 53352 | 1352 | 1156 | 0.855 | 0.0098 | 0.0088 | 0.898 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 4 | 15 | 53352 | 1361 | 1164 | 0.855 | 0.0099 | 0.0088 | 0.886 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 5 | 15 | 109548 | 1260 | 1074 | 0.852 | 0.0111 | 0.0099 | 0.890 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 6 | 15 | 205458 | 1265 | 1081 | 0.855 | 0.0120 | 0.0106 | 0.887 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 7 | 15 | 333330 | 1264 | 1080 | 0.854 | 0.0121 | 0.0107 | 0.883 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 8 | 15 | 35148 | 1263 | 1080 | 0.855 | 0.0097 | 0.0084 | 0.866 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 2 | 30 | 3096 | 6869 | 5799 | 0.844 | 0.0519 | 0.0459 | 0.883 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 3 | 18 | 53352 | 3238 | 2757 | 0.851 | 0.0260 | 0.0230 | 0.885 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 4 | 11 | 380178 | 2124 | 1819 | 0.856 | 0.0201 | 0.0170 | 0.846 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 5 | 7 | 109548 | 1595 | 1364 | 0.855 | 0.0082 | 0.0070 | 0.850 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 6 | 7 | 205458 | 1397 | 1194 | 0.855 | 0.0078 | 0.0066 | 0.843 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 7 | 7 | 333330 | 1382 | 1184 | 0.857 | 0.0080 | 0.0073 | 0.914 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 8 | 7 | 35148 | 1334 | 1141 | 0.855 | 0.0055 | 0.0049 | 0.888 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 2 | 35 | 3096 | 6869 | 5799 | 0.844 | 0.0608 | 0.0542 | 0.892 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 3 | 21 | 53352 | 3238 | 2757 | 0.851 | 0.0299 | 0.0276 | 0.923 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 4 | 15 | 380178 | 2124 | 1819 | 0.856 | 0.0249 | 0.0212 | 0.851 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 5 | 15 | 109548 | 1595 | 1364 | 0.855 | 0.0140 | 0.0121 | 0.867 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 6 | 15 | 205458 | 1397 | 1194 | 0.855 | 0.0131 | 0.0115 | 0.875 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 7 | 15 | 333330 | 1382 | 1184 | 0.857 | 0.0132 | 0.0117 | 0.882 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 8 | 15 | 35148 | 1334 | 1141 | 0.855 | 0.0098 | 0.0088 | 0.899 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1324 | 1134 | 0.856 | 0.0034 | 0.0031 | 0.895 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 3 | 7 | 53352 | 1352 | 1156 | 0.855 | 0.0056 | 0.0054 | 0.961 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 4 | 7 | 385344 | 1361 | 1164 | 0.855 | 0.0104 | 0.0089 | 0.854 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 5 | 7 | 325404 | 1260 | 1074 | 0.852 | 0.0142 | 0.0120 | 0.849 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1324 | 1134 | 0.856 | 0.0063 | 0.0057 | 0.910 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 3 | 15 | 53352 | 1352 | 1156 | 0.855 | 0.0099 | 0.0094 | 0.953 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 4 | 15 | 385344 | 1361 | 1164 | 0.855 | 0.0173 | 0.0152 | 0.875 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 5 | 15 | 325404 | 1260 | 1074 | 0.852 | 0.0248 | 0.0220 | 0.889 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5867 | 0.844 | 0.0531 | 0.0468 | 0.883 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2784 | 0.851 | 0.0262 | 0.0233 | 0.888 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 4 | 11 | 385344 | 2136 | 1824 | 0.854 | 0.0222 | 0.0191 | 0.859 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 5 | 7 | 325404 | 1599 | 1370 | 0.857 | 0.0183 | 0.0157 | 0.859 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5867 | 0.844 | 0.0612 | 0.0549 | 0.896 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2784 | 0.851 | 0.0313 | 0.0280 | 0.894 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 4 | 15 | 385344 | 2136 | 1824 | 0.854 | 0.0277 | 0.0240 | 0.867 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 5 | 15 | 325404 | 1599 | 1370 | 0.857 | 0.0309 | 0.0274 | 0.886 |

Reach 4 beside the corner (07:43–07:48):

| workload | kD | ε | level | p | V pairs | waves corner | waves ρ 4 | ratio | M2L s corner | M2L s ρ 4 | ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 500 | 1e-3 | 4 | 36 | 511200 | 9369 | 7211 | 0.770 | 0.3299 | 0.2277 | 0.690 |
| W1 | 500 | 1e-3 | 5 | 21 | 2935296 | 3943 | 3080 | 0.781 | 0.3718 | 0.3030 | 0.815 |
| W1 | 500 | 1e-3 | 6 | 13 | 10439280 | 2318 | 1834 | 0.791 | 0.5763 | 0.4503 | 0.781 |
| W1 | 500 | 1e-3 | 7 | 8 | 14492220 | 1854 | 1474 | 0.795 | 0.6216 | 0.4761 | 0.766 |
| W1 | 500 | 1e-6 | 4 | 41 | 511200 | 9369 | 7211 | 0.770 | 0.3421 | 0.2525 | 0.738 |
| W1 | 500 | 1e-6 | 5 | 25 | 2935296 | 3943 | 3080 | 0.781 | 0.4170 | 0.3408 | 0.817 |
| W1 | 500 | 1e-6 | 6 | 16 | 10439280 | 2318 | 1834 | 0.791 | 0.6465 | 0.5222 | 0.808 |
| W1 | 500 | 1e-6 | 7 | 15 | 14492220 | 1854 | 1474 | 0.795 | 0.9067 | 0.7338 | 0.809 |
| W1 | 30 | 1e-3 | 2 | 12 | 3096 | 2274 | 1804 | 0.793 | 0.0084 | 0.0068 | 0.816 |
| W1 | 30 | 1e-3 | 3 | 8 | 32256 | 1834 | 1462 | 0.797 | 0.0066 | 0.0054 | 0.818 |
| W1 | 30 | 1e-6 | 2 | 16 | 3096 | 2274 | 1804 | 0.793 | 0.0106 | 0.0090 | 0.850 |
| W1 | 30 | 1e-6 | 3 | 15 | 32256 | 1834 | 1462 | 0.797 | 0.0108 | 0.0091 | 0.843 |
| W1 | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5387 | 0.775 | 0.0523 | 0.0440 | 0.841 |
| W1 | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2568 | 0.785 | 0.0264 | 0.0220 | 0.832 |
| W1 | 100 | 1e-3 | 4 | 11 | 385344 | 2136 | 1691 | 0.792 | 0.0224 | 0.0176 | 0.789 |
| W1 | 100 | 1e-3 | 5 | 7 | 244944 | 1599 | 1267 | 0.792 | 0.0153 | 0.0123 | 0.799 |
| W1 | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5387 | 0.775 | 0.0623 | 0.0521 | 0.836 |
| W1 | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2568 | 0.785 | 0.0317 | 0.0266 | 0.838 |
| W1 | 100 | 1e-6 | 4 | 15 | 385344 | 2136 | 1691 | 0.792 | 0.0281 | 0.0226 | 0.806 |
| W1 | 100 | 1e-6 | 5 | 15 | 244944 | 1599 | 1267 | 0.792 | 0.0264 | 0.0221 | 0.836 |
| W1 | 10 | 1e-3 | 2 | 7 | 3096 | 1546 | 1226 | 0.793 | 0.0044 | 0.0032 | 0.730 |
| W1 | 10 | 1e-6 | 2 | 15 | 3096 | 1546 | 1226 | 0.793 | 0.0078 | 0.0064 | 0.817 |
| W3 (N = 108849) | 30 | 1e-3 | 2 | 12 | 3096 | 2274 | 1804 | 0.793 | 0.0085 | 0.0070 | 0.828 |
| W3 (N = 108849) | 30 | 1e-3 | 3 | 8 | 53352 | 1834 | 1462 | 0.797 | 0.0083 | 0.0068 | 0.820 |
| W3 (N = 108849) | 30 | 1e-3 | 4 | 7 | 584136 | 1494 | 1185 | 0.793 | 0.0155 | 0.0123 | 0.794 |
| W3 (N = 108849) | 30 | 1e-6 | 2 | 16 | 3096 | 2274 | 1804 | 0.793 | 0.0108 | 0.0093 | 0.858 |
| W3 (N = 108849) | 30 | 1e-6 | 3 | 15 | 53352 | 1834 | 1462 | 0.797 | 0.0133 | 0.0112 | 0.839 |
| W3 (N = 108849) | 30 | 1e-6 | 4 | 15 | 584136 | 1494 | 1185 | 0.793 | 0.0240 | 0.0201 | 0.838 |
| W3 (N = 4031442) | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5387 | 0.775 | 0.0524 | 0.0444 | 0.847 |
| W3 (N = 4031442) | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2568 | 0.785 | 0.0262 | 0.0218 | 0.834 |
| W3 (N = 4031442) | 100 | 1e-3 | 4 | 11 | 584136 | 2136 | 1691 | 0.792 | 0.0290 | 0.0228 | 0.786 |
| W3 (N = 4031442) | 100 | 1e-3 | 5 | 7 | 5398920 | 1599 | 1267 | 0.792 | 0.1459 | 0.1140 | 0.781 |
| W3 (N = 4031442) | 100 | 1e-3 | 6 | 7 | 46298376 | 1397 | 1104 | 0.790 | 1.0721 | 0.8309 | 0.775 |
| W3 (N = 4031442) | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5387 | 0.775 | 0.0622 | 0.0534 | 0.858 |
| W3 (N = 4031442) | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2568 | 0.785 | 0.0318 | 0.0268 | 0.841 |
| W3 (N = 4031442) | 100 | 1e-6 | 4 | 15 | 584136 | 2136 | 1691 | 0.792 | 0.0354 | 0.0282 | 0.797 |
| W3 (N = 4031442) | 100 | 1e-6 | 5 | 15 | 5398920 | 1599 | 1267 | 0.792 | 0.2025 | 0.1632 | 0.806 |
| W3 (N = 4031442) | 100 | 1e-6 | 6 | 15 | 46298376 | 1397 | 1104 | 0.790 | 1.4498 | 1.1832 | 0.816 |
| W3 (N = 4032) | 10 | 1e-3 | 2 | 7 | 3096 | 1546 | 1226 | 0.793 | 0.0038 | 0.0032 | 0.833 |
| W3 (N = 4032) | 10 | 1e-3 | 3 | 7 | 5544 | 1375 | 1089 | 0.792 | 0.0042 | 0.0031 | 0.736 |
| W3 (N = 4032) | 10 | 1e-6 | 2 | 15 | 3096 | 1546 | 1226 | 0.793 | 0.0070 | 0.0060 | 0.855 |
| W3 (N = 4032) | 10 | 1e-6 | 3 | 15 | 5544 | 1375 | 1089 | 0.792 | 0.0069 | 0.0060 | 0.861 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1352 | 1067 | 0.789 | 0.0035 | 0.0029 | 0.850 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 3 | 7 | 19782 | 1361 | 1076 | 0.791 | 0.0036 | 0.0030 | 0.823 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 4 | 7 | 43392 | 1260 | 995 | 0.790 | 0.0039 | 0.0033 | 0.845 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 5 | 7 | 106500 | 1265 | 998 | 0.789 | 0.0051 | 0.0041 | 0.809 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 6 | 7 | 214818 | 1264 | 997 | 0.789 | 0.0084 | 0.0069 | 0.820 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 7 | 7 | 289194 | 1263 | 997 | 0.789 | 0.0079 | 0.0064 | 0.800 |
| W4 clusters (N = 100000) | 1 | 1e-3 | 8 | 7 | 351936 | 1263 | 997 | 0.789 | 0.0092 | 0.0073 | 0.794 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1352 | 1067 | 0.789 | 0.0066 | 0.0057 | 0.857 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 3 | 15 | 19782 | 1361 | 1076 | 0.791 | 0.0065 | 0.0055 | 0.844 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 4 | 15 | 43392 | 1260 | 995 | 0.790 | 0.0067 | 0.0057 | 0.859 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 5 | 15 | 106500 | 1265 | 998 | 0.789 | 0.0087 | 0.0073 | 0.840 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 6 | 15 | 214818 | 1264 | 997 | 0.789 | 0.0148 | 0.0125 | 0.844 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 7 | 15 | 289194 | 1263 | 997 | 0.789 | 0.0137 | 0.0112 | 0.820 |
| W4 clusters (N = 100000) | 1 | 1e-6 | 8 | 15 | 351936 | 1263 | 997 | 0.789 | 0.0158 | 0.0140 | 0.884 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 2 | 21 | 3096 | 3948 | 3088 | 0.782 | 0.0220 | 0.0184 | 0.836 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 3 | 13 | 19782 | 2321 | 1836 | 0.791 | 0.0092 | 0.0075 | 0.819 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 4 | 8 | 43392 | 1857 | 1478 | 0.796 | 0.0063 | 0.0049 | 0.772 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 5 | 7 | 106500 | 1495 | 1188 | 0.795 | 0.0058 | 0.0047 | 0.812 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 6 | 7 | 214818 | 1394 | 1105 | 0.793 | 0.0094 | 0.0077 | 0.819 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 7 | 7 | 289194 | 1372 | 1084 | 0.790 | 0.0086 | 0.0069 | 0.802 |
| W4 clusters (N = 100000) | 100 | 1e-3 | 8 | 7 | 351936 | 1324 | 1048 | 0.792 | 0.0096 | 0.0078 | 0.809 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 2 | 25 | 3096 | 3948 | 3088 | 0.782 | 0.0263 | 0.0223 | 0.847 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 3 | 16 | 19782 | 2321 | 1836 | 0.791 | 0.0111 | 0.0092 | 0.832 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 4 | 15 | 43392 | 1857 | 1478 | 0.796 | 0.0096 | 0.0080 | 0.832 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 5 | 15 | 106500 | 1495 | 1188 | 0.795 | 0.0100 | 0.0084 | 0.844 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 6 | 15 | 214818 | 1394 | 1105 | 0.793 | 0.0161 | 0.0135 | 0.842 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 7 | 15 | 289194 | 1372 | 1084 | 0.790 | 0.0146 | 0.0122 | 0.835 |
| W4 clusters (N = 100000) | 100 | 1e-6 | 8 | 15 | 351936 | 1324 | 1048 | 0.792 | 0.0163 | 0.0136 | 0.833 |
| W4 cube (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1324 | 1048 | 0.792 | 0.0035 | 0.0029 | 0.830 |
| W4 cube (N = 100000) | 1 | 1e-3 | 3 | 7 | 53352 | 1352 | 1067 | 0.789 | 0.0057 | 0.0048 | 0.841 |
| W4 cube (N = 100000) | 1 | 1e-3 | 4 | 7 | 584136 | 1361 | 1076 | 0.791 | 0.0140 | 0.0108 | 0.775 |
| W4 cube (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1324 | 1048 | 0.792 | 0.0063 | 0.0055 | 0.859 |
| W4 cube (N = 100000) | 1 | 1e-6 | 3 | 15 | 53352 | 1352 | 1067 | 0.789 | 0.0100 | 0.0085 | 0.849 |
| W4 cube (N = 100000) | 1 | 1e-6 | 4 | 15 | 584136 | 1361 | 1076 | 0.791 | 0.0221 | 0.0179 | 0.810 |
| W4 cube (N = 100000) | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5387 | 0.775 | 0.0527 | 0.0446 | 0.845 |
| W4 cube (N = 100000) | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2568 | 0.785 | 0.0264 | 0.0222 | 0.841 |
| W4 cube (N = 100000) | 100 | 1e-3 | 4 | 11 | 584136 | 2136 | 1691 | 0.792 | 0.0290 | 0.0230 | 0.792 |
| W4 cube (N = 100000) | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5387 | 0.775 | 0.0612 | 0.0527 | 0.860 |
| W4 cube (N = 100000) | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2568 | 0.785 | 0.0318 | 0.0267 | 0.841 |
| W4 cube (N = 100000) | 100 | 1e-6 | 4 | 15 | 584136 | 2136 | 1691 | 0.792 | 0.0359 | 0.0285 | 0.794 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1324 | 1048 | 0.792 | 0.0034 | 0.0029 | 0.839 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 3 | 7 | 53352 | 1352 | 1067 | 0.789 | 0.0056 | 0.0047 | 0.844 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 4 | 7 | 53352 | 1361 | 1076 | 0.791 | 0.0057 | 0.0047 | 0.824 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 5 | 7 | 109548 | 1260 | 995 | 0.790 | 0.0064 | 0.0054 | 0.848 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 6 | 7 | 205458 | 1265 | 998 | 0.789 | 0.0069 | 0.0057 | 0.822 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 7 | 7 | 333330 | 1264 | 997 | 0.789 | 0.0082 | 0.0058 | 0.710 |
| W4 plummer (N = 100000) | 1 | 1e-3 | 8 | 7 | 35148 | 1263 | 997 | 0.789 | 0.0053 | 0.0044 | 0.839 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1324 | 1048 | 0.792 | 0.0063 | 0.0054 | 0.859 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 3 | 15 | 53352 | 1352 | 1067 | 0.789 | 0.0099 | 0.0085 | 0.856 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 4 | 15 | 53352 | 1361 | 1076 | 0.791 | 0.0100 | 0.0084 | 0.841 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 5 | 15 | 109548 | 1260 | 995 | 0.790 | 0.0111 | 0.0095 | 0.852 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 6 | 15 | 205458 | 1265 | 998 | 0.789 | 0.0121 | 0.0100 | 0.829 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 7 | 15 | 333330 | 1264 | 997 | 0.789 | 0.0121 | 0.0107 | 0.886 |
| W4 plummer (N = 100000) | 1 | 1e-6 | 8 | 15 | 35148 | 1263 | 997 | 0.789 | 0.0099 | 0.0080 | 0.807 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 2 | 30 | 3096 | 6869 | 5327 | 0.776 | 0.0521 | 0.0436 | 0.837 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 3 | 18 | 53352 | 3238 | 2549 | 0.787 | 0.0261 | 0.0217 | 0.831 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 4 | 11 | 380178 | 2124 | 1679 | 0.790 | 0.0201 | 0.0158 | 0.785 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 5 | 7 | 109548 | 1595 | 1262 | 0.791 | 0.0081 | 0.0065 | 0.805 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 6 | 7 | 205458 | 1397 | 1104 | 0.790 | 0.0078 | 0.0062 | 0.799 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 7 | 7 | 333330 | 1382 | 1097 | 0.794 | 0.0080 | 0.0064 | 0.802 |
| W4 plummer (N = 100000) | 100 | 1e-3 | 8 | 7 | 35148 | 1334 | 1056 | 0.792 | 0.0055 | 0.0046 | 0.840 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 2 | 35 | 3096 | 6869 | 5327 | 0.776 | 0.0606 | 0.0518 | 0.854 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 3 | 21 | 53352 | 3238 | 2549 | 0.787 | 0.0299 | 0.0254 | 0.849 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 4 | 15 | 380178 | 2124 | 1679 | 0.790 | 0.0246 | 0.0198 | 0.805 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 5 | 15 | 109548 | 1595 | 1262 | 0.791 | 0.0139 | 0.0116 | 0.835 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 6 | 15 | 205458 | 1397 | 1104 | 0.790 | 0.0131 | 0.0109 | 0.837 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 7 | 15 | 333330 | 1382 | 1097 | 0.794 | 0.0131 | 0.0109 | 0.834 |
| W4 plummer (N = 100000) | 100 | 1e-6 | 8 | 15 | 35148 | 1334 | 1056 | 0.792 | 0.0098 | 0.0084 | 0.859 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 2 | 7 | 3096 | 1324 | 1048 | 0.792 | 0.0035 | 0.0029 | 0.842 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 3 | 7 | 53352 | 1352 | 1067 | 0.789 | 0.0057 | 0.0047 | 0.837 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 4 | 7 | 385344 | 1361 | 1076 | 0.791 | 0.0105 | 0.0082 | 0.784 |
| W4 sphere (N = 100000) | 1 | 1e-3 | 5 | 7 | 325404 | 1260 | 995 | 0.790 | 0.0141 | 0.0114 | 0.810 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 2 | 15 | 3096 | 1324 | 1048 | 0.792 | 0.0069 | 0.0054 | 0.781 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 3 | 15 | 53352 | 1352 | 1067 | 0.789 | 0.0099 | 0.0085 | 0.852 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 4 | 15 | 385344 | 1361 | 1076 | 0.791 | 0.0174 | 0.0143 | 0.820 |
| W4 sphere (N = 100000) | 1 | 1e-6 | 5 | 15 | 325404 | 1260 | 995 | 0.790 | 0.0244 | 0.0207 | 0.845 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 2 | 30 | 3096 | 6954 | 5387 | 0.775 | 0.0522 | 0.0436 | 0.836 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 3 | 18 | 53352 | 3270 | 2568 | 0.785 | 0.0263 | 0.0218 | 0.830 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 4 | 11 | 385344 | 2136 | 1691 | 0.792 | 0.0223 | 0.0178 | 0.795 |
| W4 sphere (N = 100000) | 100 | 1e-3 | 5 | 7 | 325404 | 1599 | 1267 | 0.792 | 0.0183 | 0.0145 | 0.791 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 2 | 35 | 3096 | 6954 | 5387 | 0.775 | 0.0611 | 0.0521 | 0.854 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 3 | 22 | 53352 | 3270 | 2568 | 0.785 | 0.0314 | 0.0264 | 0.840 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 4 | 15 | 385344 | 2136 | 1691 | 0.792 | 0.0277 | 0.0224 | 0.810 |
| W4 sphere (N = 100000) | 100 | 1e-6 | 5 | 15 | 325404 | 1599 | 1267 | 0.792 | 0.0308 | 0.0257 | 0.832 |
