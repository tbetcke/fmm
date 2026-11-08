# Helmholtz CH6.2g: the counts' reach, kept at the corner

Helmholtz task CH6.2g ([docs/helmholtz/tasks/CH6.2g.md](../../docs/helmholtz/tasks/CH6.2g.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.1 and CH6.2, §2 (constraints 6, 9). It was to adopt the α counts for lateral offsets
ρ ≤ 4.5 in place of the region's corner 4√2 as the default (CH6.2e's question 1, signed off
2026-11-07, [helmholtz-ch6-2e.md](helmholtz-ch6-2e.md) §8), with a check CH6.2e lacked: one
M2L pair at the largest V offset at the level's degree against Rotation on every domain. That
check fails at high κ, so the default stays the corner (the user, 2026-11-08: "Keep the
corner"). Measured on **the M6** (Apple M6, 12 cores, `tools/m6/machine.md`), 2026-11-08, in
release builds, one thread, accuracy only (nothing timed): the tests of nd-fmm-tables and
nd-fmm-exec (§2, §3) and a throwaway example for the sampling (§1).

## Summary

- **Reach 4.5 does not hold the FMM's chain at high κ** (§2). At the 24 largest V offsets,
  (±3, ±3, ±2), along every direction merge and shift gives them, a multipole and a local of
  the level's calibrated degree, iprec 2 (the rule's precision): against the error of the
  same nodes with converged α counts, the counts for reach 4.5 add more than 10× the stated
  error from Re κ = 7π on the domains of real κ's edge (16× at 7π–8π, 47× at 8π–9π, 2.1·10⁴×
  at 15π–16π, at ε = 10⁻⁶; from 8π at 10⁻³) and up to 16× on some damped domains from 1.5π.
  The corner's counts add at most 3.2× on every domain below Im κ = 5π. Against the direct
  sum, at ε = 10⁻⁶ reach 4.5 errs 3.6× Rotation's error on these pairs at Re κ ≈ 8.5π and
  about 2,000× at 15.5π (1.8·10⁻³ against 9·10⁻⁷ in box widths); the corner matches
  Rotation's wherever the counts matter.
- **Why CH6.2e did not see it** (§4): its largest κ with waves was 31.25 (W1 at kD = 500,
  level 4, Re κ in [9π, 10π]), where the FMM's error at reach 4.5 held (0.986× Rotation's at
  10⁻⁶) while the worst-case pair there errs about 10× Rotation's against the direct sum: the
  FMM's error over sampled targets averages the worst case away. Above κ ≈ 31, up to about
  16π (kD ≈ 200 on level 2, kD ≈ 400 on level 3, inside D1's range), no FMM run exists.
- **The decision**: the question went to the user with this evidence; answer 2026-11-08,
  "Keep the corner (Recommended)", confirmed the same day after a correction (the first
  question said wrongly that CH6.2e had no level above κ ≈ 25; §4). The default counts are unchanged, so the FMM's outputs are
  unchanged bit for bit; no machine run was needed. CONVENTIONS §3.15 is unchanged.
  `with_reach` stays for measurements.
- **`tail_count`'s sampling** (§1): with 64 steps of ρ the counts at 4.5 exceed the corner's on
  11–16 of 437 domains per precision, all strongly damped (Im κ ≥ 17), by one or two waves in
  all; at 1,024 steps (the same at 4,096) never. Finer sampling would add 32–95 waves at the
  corner over all 437 domains (one to a few per domain on 38–55 of them); the bounds hold at
  64 steps, so it is kept.
- **Beyond the counts** (§3, follow-ups for CH6.2f): at the level's degree the exponential
  M2L's largest-offset pairs err 3·10⁻⁶ against the direct sum at ε = 10⁻⁶ on low-κ domains,
  where Rotation errs 10⁻¹²–10⁻⁹ (CH6.1c's truncated tail; the converged counts the same), and
  from Im κ = 5π both counts' chains err 10⁻⁹–10⁻⁵ absolute where the field across a V offset
  is below 2·10⁻¹⁴ of the near field. CH6.1e's FMM errors on D12's workloads show neither.

## 1. `tail_count`'s sampling

`Quadrature`'s Nₖ takes max_{m ≥ n} max_{ρ ≤ reach} |J_m(tₖ ρ)| at 64 equal steps of
[0, reach] (`tail_count`). CH6.2e's §1 found reach 4 and 4.5 with more waves than the corner on
some strongly damped domains and suspected the sampling. The same bound at 64, 256, 1,024 and
4,096 steps (a throwaway example restating `tail_count` with the number of steps as a
parameter, on `Quadrature`'s public nodes and `bessel_j_norms`), at every domain's centre:

| iprec | domains whose counts change at 1,024 steps | waves added at 1,024, corner | at 4.5 | domains with 4.5 above the corner at 64 | at 1,024 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41 | 32 | 26 | 16 | 0 |
| 2 | 38 | 35 | 21 | 15 | 0 |
| 3 | 55 | 95 | 41 | 11 | 0 |

At 4,096 steps every number equals 1,024's; at 256 one domain (344) still has 4.5 above the
corner (1.071×). Every inversion at 64 steps is on a domain with Im κ ≥ 17 (Re κ from 0.3 to
49), by one or two waves in all (e.g. iprec 2, domain 294: corner 26, 4.5 28 at 64 steps, both
28 at 1,024). So the cause is the sampling, as CH6.2e supposed: on these domains the corner's
coarser steps miss the peak that a smaller reach's finer steps find. The kernel bound holds on
every domain at 64 steps (CH6.1a's ignored `the_counts_on_every_domain`), so the sampling is
kept; a finer one would add the waves above and change every affected level's output.

## 2. The largest V offsets at the level's degree

nd-fmm-exec's `tests/helmholtz/plane_wave_reach.rs`: on level 2 of a tree with κ = k w the
domain's centre (and, on the domains of real κ's edge, its real part too, with one and two
multipole planes), the degree `degree(k, √3/2, ε)` of CH3.3's rule, the RCR tables
(`HelmholtzTables`) and three exponential M2Ls on iprec 2's nodes: the default counts (the
corner), converged counts (Nₖ = ⌈|tₖ| 4√2⌉ + 2Fₖ + 40, far beyond aliasing) and the counts for
reach 4.5. A multipole of charges at the source box's corners, centre and four seeded points
goes to each of the 24 offsets whose components are a permutation of (±3, ±3, ±2) along every
direction merge and shift gives it, and both locals are evaluated at the target box's 27
points {−½, 0, ½}³, corners included (CH6.1c's measure: the largest |Δ| in box widths). The
table gives the largest addition of each counts' error over the converged counts' error, in
units of the stated error, over the Im κ of each Re κ column below 5π ("real": the domains of
real κ's edge, Im κ < π/2):

| Re κ | corner, 10⁻³ | 4.5, 10⁻³ | 4.5 real, 10⁻³ | corner, 10⁻⁶ | 4.5, 10⁻⁶ | 4.5 real, 10⁻⁶ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–0.4π | 0.00 | 0.90 | 0.00 | 3.21 | 3.77 | 0.50 |
| 0.75π | 0.01 | 0.31 | 0.00 | 2.33 | 0.16 | 0.06 |
| 1.5π | 0.01 | 0.14 | 0.01 | 1.08 | 10.6 | 0.39 |
| 2.5π | 0.08 | 2.22 | 0.09 | 1.23 | 15.7 | 0.21 |
| 3.5π | 0.00 | 2.93 | 0.18 | 0.00 | 1.96 | 0.00 |
| 4.5π | 0.00 | 10.7 | 0.00 | 0.00 | 1.43 | 0.00 |
| 5.5π | 0.00 | 10.0 | 1.31 | 0.00 | 2.91 | 2.13 |
| 6.5π | 0.00 | 2.67 | 2.67 | 0.00 | 6.98 | 6.98 |
| 7.5π | 2.45 | 14.2 | 5.32 | 0.24 | 16.9 | 15.7 |
| 8.5π | 0.00 | 26.0 | 26.0 | 0.00 | 46.7 | 46.7 |
| 9.5π | 0.00 | 110 | 110 | 0.00 | 114 | 114 |
| 10.5π | 0.00 | 273 | 273 | 0.00 | 276 | 276 |
| 11.5π | 0.00 | 473 | 473 | 0.00 | 475 | 475 |
| 12.5π | 0.00 | 1,818 | 1,818 | 0.00 | 1,827 | 1,827 |
| 13.5π | 0.00 | 4,220 | 4,220 | 0.00 | 4,234 | 4,234 |
| 14.5π | 0.00 | 11,083 | 11,083 | 0.00 | 11,058 | 11,058 |
| 15.5π | 0.00 | 20,898 | 20,898 | 0.00 | 20,840 | 20,840 |

The degrees run from 7 (10⁻³) and 15 (10⁻⁶) on the low-κ plateau to 53 and 59 at Re κ ≈ 15.5π.
The corner's largest additions, 3.2× (domain 188, κ ≈ 0.15 + 4.7i, 10⁻⁶) and 2.4× (domain 199,
κ ≈ 23.6 + 4.7i, 10⁻³), are on damped domains; the test asserts at most 10×
(`COUNTS_ADDED`) on every domain below Im κ = 5π, and prints reach 4.5's.

Against the direct sum (an instrumented copy of the same test, not committed, which also
evaluated Σ q e^{ik̂r}/r at the 27 targets), on the domains of real κ's edge, the largest error
of each chain in box widths (the field is about 0.4 there):

| Re κ | p (10⁻⁶) | Rotation | reach 4.5 | corner |
| --- | ---: | ---: | ---: | ---: |
| 0.01–2.4 | 15 | 2.4–9.0·10⁻¹² | 2.9–3.4·10⁻⁶ | 2.9–3.4·10⁻⁶ |
| 4.71 | 15 | 1.9·10⁻⁹ | 3.2·10⁻⁶ | 3.2·10⁻⁶ |
| 7.85–11.0 | 16–20 | 3.9–8.7·10⁻⁷ | 2.7–2.9·10⁻⁶ | 2.7–2.9·10⁻⁶ |
| 14.1–23.6 | 23–33 | 0.6–1.6·10⁻⁶ | 0.7–1.6·10⁻⁶ | 0.6–1.6·10⁻⁶ |
| 26.7 | 37 | 7.0·10⁻⁷ | 2.5·10⁻⁶ | 7.2·10⁻⁷ |
| 29.8 | 40 | 9.2·10⁻⁷ | 1.0·10⁻⁵ | 9.2·10⁻⁷ |
| 33.0 | 43 | 1.6·10⁻⁶ | 2.3·10⁻⁵ | 1.6·10⁻⁶ |
| 39.3 | 49 | 1.9·10⁻⁶ | 1.7·10⁻⁴ | 1.9·10⁻⁶ |
| 45.6 | 56 | 7.8·10⁻⁷ | 8.9·10⁻⁴ | 7.8·10⁻⁷ |
| 48.7 | 59 | 9.2·10⁻⁷ | 1.8·10⁻³ | 9.1·10⁻⁷ |

At 10⁻³ reach 4.5 errs 1.45× Rotation's error at Re κ ≈ 39 (1.6·10⁻⁴ against 1.1·10⁻⁴), 8×
at 45.6 and 12× at 48.7, and matches it below.

Why: the corner 4√2 is the centres' largest lateral offset 3√2 plus the reach √2/2 of a
source and of a target from their box centres. The counts take the source's orders apart
(Nₖ at least the count for ρ_max − r_s plus Fₖ), not the target's. CH6.2e's §4 supposed the
local of degree p_l does not carry the target's orders that alias. That holds while p_l is
small against |tₖ| r_s, which is the low-frequency plateau. At high κ the degree grows with κ
(p ≈ 0.87 Re κ + …), the local resolves its box, and the counts need the whole corner.

## 3. Beyond the counts

Two observations from the same runs that the counts do not govern (the converged counts give
the same errors):

- **The truncated tail at ε = 10⁻⁶.** On the low-κ plateau (p = 15) the exponential M2L's
  largest-offset pairs err about 3·10⁻⁶ against the direct sum, against Rotation's 10⁻¹²–10⁻⁹:
  relative to the field about 10⁻⁵, ten times ε, at these worst-case points (charges at the
  source box's corners, targets at the target box's). This is CH6.1c's degree study (the
  waves of a truncated multipole are a truncated series in λ, and the quadrature errs on its
  tail): CH6.1e's rule meets 1.1× Rotation's error in the FMM because the FMM's error is set
  by the nearest offsets, where Rotation's own error is of the size of ε. A distribution
  that concentrates points at box corners could see it.
- **Strongly damped domains.** From Im κ = 5π (a V offset's field below e^{−10π} ≈ 2·10⁻¹⁴ of
  the near field) the stated errors fall from 6·10⁻⁸ to 4·10⁻¹⁶ while both counts' chains err
  10⁻⁹–10⁻⁵ absolute (e.g. domain 303, κ ≈ 0.15 + 20.4i, p = 15: 1.8·10⁻⁵ with the corner's
  counts, 5.6·10⁻⁶ converged), with Rotation accurate. The test leaves these domains out.
  Under the rule such levels get waves only for strongly damped complex k (Yukawa's real path
  has none); D1's workloads have none.

Both go to CH6.2f (D8's rule) as roadmap sentences: the rule may keep Rotation on levels of
the low-κ plateau at ε near 10⁻⁶ if points at box corners matter, and on levels with
Im κ ≥ 5π.

## 4. Why CH6.2e's runs did not show it

CH6.2e's verification (CH6.1e's cases at 10⁻³ to 10⁻⁶ and the two large ones) and scan (166
levels) had waves on levels with κ up to 31.25: W1 at kD = 500 on level 4 (Re κ in [9π, 10π],
p = 36 at 10⁻³ and 41 at 10⁻⁶), the other cases at most 25 (W1 and the W4 cases at kD = 100,
level 2). On W1 at kD = 500 the FMM's error at reach 4.5 held, 0.986× Rotation's at 10⁻⁶
(CH6.2e §2), while the pair test at that κ has the worst-case pair at about 10× Rotation's
error against the direct sum (1.0·10⁻⁵ against 9.2·10⁻⁷ at Re κ ≈ 29.8, the table of §2). The
FMM's relative L2 error over sampled targets averages the worst case away: the largest V
offsets are 24 of the 316, and their worst points (charges and targets near box corners) are a
small share of uniform points. Its larger additions at lower κ are on damped domains, which no
case had. The pair test is the worst case the FMM's errors cannot exclude, and above κ ≈ 31,
to about 16π, no FMM run exists at all.

The first question to the user on 2026-11-08 said that CH6.2e had no level with waves above
κ ≈ 25; that was wrong (kD = 500's level 4), and the corrected question was put the same day.
The answer stayed "Keep the corner".

## 5. Commands

From the repository root on the M6:

```sh
# §2, the guard and reach 4.5's additions (prints the largest per tolerance)
cargo test --release -p nd-fmm-exec --test helmholtz plane_wave_reach -- --include-ignored --nocapture
```

The per-column tables of §2 came from the same test with a line printed per case (κ, domain,
ε, p, planes, stated error, the three errors), and the direct-sum table from an instrumented
copy that also evaluated the direct sum; §1 from a throwaway example. None is committed; the
raw lines are in `~/projects/fmm-archive/m6/ch6-2g/` on the M6.
