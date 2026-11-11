# Helmholtz CH6.1g: the quadrature below 10⁻⁶ in the FMM

Helmholtz task CH6.1g ([docs/helmholtz/tasks/CH6.1g.md](../../docs/helmholtz/tasks/CH6.1g.md)),
the FMM half of the former CH6.1f; proposal
[docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md) §10 CH6.1
("errors within about 1.1× Rotation's"), §5.3, §2 (constraints 6, 7 and 9), §12 D8; CH6.1f's
report ([helmholtz-ch6-1f.md](helmholtz-ch6-1f.md) §3, §5) and CH6.1e's
([helmholtz-ch6-1e.md](helmholtz-ch6-1e.md) §2, §3). Errors and times measured on 2026-11-11 on
Kathleen, one node (2 × 20 Cascade Lake cores, exclusive), jobs 632472 (ε = 10⁻⁷) and 632473
(10⁻⁸), one rank × 40 threads, release build, f64, AVX-512 P2P, BLAS threads 1, from branch
`tbetcke/h-ch6-1g` at the rule as signed off (§2), before Auto's change (§4), which these runs
do not exercise (they name `rotation` and `planewave`). The test trees of §1 on the M6
(release, one thread).

## Summary

- **The rule as built:** below ε = 10⁻⁶ a level with a quadrature runs the exponential M2L with
  **iprec 3** at the raised degree p_w = max(p_l, 28) for 10⁻⁷ ≤ ε < 10⁻⁶ and max(p_l, 32) for
  10⁻⁸ ≤ ε < 10⁻⁷, Rotation below 10⁻⁸. The degree is raised on the whole level (CH6.1f's
  sign-off 2(a)): the operator builds its RCR tables at p_w, so P2M, M2M, L2L, L2P and the
  stores of the level run at p_w too. A function of ε and p_l, on the levels whose κ_l = k w_l
  lies in the tables (constraint 9).
- **CH6.1f's iprec 2 at 10⁻⁷ did not hold in the FMM (§1).** iprec 2's own error (its stated
  error reaches 10⁻⁷) lies above Rotation's on the FMM's trees at 10⁻⁷ whatever the degree: on
  the tests' trees 1.25–6.6× Rotation's at degrees 24 and 32 alike. The pair chain compared
  against RCR's worst-case error (5.8·10⁻⁸ at p = 20), which is above that floor, so it did not
  show. Asked and signed off 2026-11-11 ("iprec 3 at max(p,28)").
- **Acceptance met on CH6.1e's 26 cases at 10⁻⁷ and 10⁻⁸ (§3):** the errors over Rotation's at
  the degrees given are **0.004–0.80 (φ) and 0.004–0.54 (∇φ) at 10⁻⁷, 0.003–0.48 and
  0.002–0.32 at 10⁻⁸**, the clustered trees' deep levels included (the clusters and the Plummer
  sphere at kD = 1, levels with waves to κ₈ = 2⁻⁸ ≈ 0.004). The raised degree over-delivers:
  on the cases whose levels with waves are all raised, φ 0.003–0.076× Rotation's error
  (13–330× below it).
- **Times beside Rotation (§3):** the M2L takes **0.26–0.84×** Rotation's at 10⁻⁸ and
  0.42–1.46× at 10⁻⁷ (above 1 only on two ms-scale cases); the evaluation 0.37–0.81× at 10⁻⁸ on
  every case above 0.1 s, and 0.54–1.05× at 10⁻⁷ there (1.00× on the clusters, 1.05× on W1 at
  kD = 100; 1.10× on W1 at kD = 30, 0.06 s). The rest of the evaluation (leaves, M2M, L2L at p_w) costs 1.29–1.88×, which at
  10⁻⁷ eats the M2L's gain on the deep trees. The build takes 2.1–10× Rotation's (the RCR and
  plane-wave tables at p_w; at most 0.97 s on these cases, Rotation's at most 0.31 s).
- **D8's bound revisited (§4), signed off 2026-11-11:** `Auto` takes the exponential M2L also at
  10⁻⁸ ≤ ε < 10⁻⁷ (at these degrees Auto runs Rotation, so the times above are Auto's change),
  and keeps Rotation at 10⁻⁷ ≤ ε < 10⁻⁶ until the degree raised on the M2L alone (CH6.1f's
  option (b)) is measured: **CH6.1h**, next on the priority list.

## 1. iprec 2 at 10⁻⁷: the quadrature's own floor

`check_resolving_degree`'s first debug run (`tests/helmholtz_fmm.rs`) failed with the
signed-off rule: on the clusters at kD = 1 (N = 1,000) iprec 2 at degree 24 erred 3.25·10⁻⁹
against Rotation's 7.95·10⁻¹⁰ (4.1×). An exploration on the tests' four trees (release, the
M6; a temporary test, not committed), each level with a quadrature forced to a precision and a
degree, φ over Rotation's at the degrees given:

| tree | ε | Rotation φ | iprec 2 at p_l | at max(p_l, 24) | at 32 | iprec 3 at p_l | at 24 | at 32 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clusters, kD = 1, N = 1,000 | 10⁻⁷ | 7.95e-10 | 4.18 | 4.09 | 4.09 | 0.99 | 0.044 | 0.000 |
| clusters, kD = 1, N = 4,000 | 10⁻⁷ | 2.08e-9 | 6.61 | 6.56 | 6.56 | 0.97 | 0.071 | 0.001 |
| cube, kD = 20 + 2i, N = 1,000 | 10⁻⁷ | 1.86e-9 | 1.15 | 0.42 | 0.41 | 0.99 | 0.046 | 0.001 |
| cube, kD = 20 + 2i, N = 4,000 | 10⁻⁷ | 5.70e-9 | 1.42 | 1.25 | 1.25 | 0.94 | 0.102 | 0.002 |
| clusters, kD = 1, N = 1,000 | 10⁻⁸ | 6.17e-11 | 52.6 | 52.6 | 52.6 | 0.56 | 0.56 | 0.005 |
| clusters, kD = 1, N = 4,000 | 10⁻⁸ | 1.88e-10 | 72.5 | 72.5 | 72.5 | 0.78 | 0.78 | 0.007 |
| cube, kD = 20 + 2i, N = 1,000 | 10⁻⁸ | 8.05e-11 | 9.64 | 9.64 | 9.56 | 1.07 | 1.07 | 0.012 |
| cube, kD = 20 + 2i, N = 4,000 | 10⁻⁸ | 8.42e-10 | 8.48 | 8.48 | 8.48 | 0.69 | 0.69 | 0.014 |

iprec 2 does not move with the degree: its error is the quadrature's own (stated up to 10⁻⁷),
2–14·10⁻⁹ relative here, above Rotation's on trees where the calibrated p_l over-delivers.
CH6.1e saw the same at p_l (up to 5.4× at 10⁻⁷) and read it as the tail; the tail is iprec 3's
problem (0.56–1.07× at p_l on these trees, up to 53× on the pair chain), cured by the degree.
So the 10⁻⁷ band takes iprec 3; its degree, 28, is the smallest at which iprec 3 held on every
κ of CH6.1f's pair-chain study at 10⁻⁷ (p + 8: 0.02–0.93× RCR's worst-case error; at p + 4 = 24
up to 8.3×), although the FMM's trees here hold already at 24 (0.04–0.10×).

## 2. The rule in the code

- `nd_fmm_tables::helmholtz::plane_wave::merge`: `precision_for_tolerance` gives iprec 2 for
  ε ≥ `PLANE_WAVE_MIN_TOLERANCE` (10⁻⁶) and iprec 3 for `RESOLVING_MIN_TOLERANCE` (10⁻⁸) ≤ ε <
  10⁻⁶; `resolving_degree` 0 (the level's own) for ε ≥ 10⁻⁶, `RESOLVING_DEGREE` (28) to
  `RESOLVING_STEP_TOLERANCE` (10⁻⁷), `RESOLVING_DEGREE_TIGHT` (32) below; `plane_wave_degree(p,
  ε)` = max(p, d). Named degrees keep CH6.1e's rule (iprec 2 to p = 15), never raised.
- `nd_fmm_exec::helmholtz::HelmholtzOperator`: keeps the degrees given (`base_degree`); under
  `PlaneWave`'s rule with a tolerance, and under `Auto` (§4), raises each level with a quadrature
  to `plane_wave_degree` and builds `HelmholtzTables` again where the degrees change (back at
  the degrees given when a later setting takes the waves away); `degree` and
  `HelmholtzFmm::degrees` report the degrees run, and the evaluator's stores take them. Named
  precisions raise nothing.

## 3. CH6.1e's cases at 10⁻⁷ and 10⁻⁸

CH6.1e's cases at these tolerances (its `tools/helmholtz/ch6-1e-runs.sh`: the matrix at 10⁻⁸,
the `between` list at 10⁻⁷; the two large cases at 10⁻⁸ were left to Rotation by CH6.1e's
sign-off 1 and are not run), each case `--m2l rotation,planewave` and the order reversed,
`--evaluations 2 --levels` (`tools/helmholtz/ch6-1g-runs.sh`; the tables by
`tools/helmholtz/ch6-1g-tables.py`). Times: the second evaluation of each launch, the smaller
of the two launches; "rest" is the evaluation less the M2L summed over the levels. The
checksums of the two launches agree in every case. "p by level" in the harness's
notation (`0×2, 28×3`: degree 0 on two levels, 28 on three).

| ε | workload | kD | p by level, Rotation → PlaneWave | φ Rot. | φ PW | ÷ | ∇φ Rot. | ∇φ PW | ÷ | M2L s Rot. | PW | ÷ | evaluate s Rot. | PW | ÷ | rest s Rot. | PW | ÷ | build ÷ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-7 | W1 | 10 | 0×2, 20 → 0×2, 28 | 1.42e-09 | 5.59e-12 | 0.004 | 1.22e-08 | 6.78e-11 | 0.006 | 0.006 | 0.007 | 1.22 | 0.008 | 0.010 | 1.27 | 0.003 | 0.003 | 1.29 | 10.00 |
| 1e-7 | W1 | 30 | 0×2, 20×2 → 0×2, 28×2 | 4.33e-09 | 1.83e-11 | 0.004 | 1.21e-08 | 6.43e-11 | 0.005 | 0.050 | 0.049 | 0.97 | 0.060 | 0.066 | 1.10 | 0.010 | 0.017 | 1.68 | 6.33 |
| 1e-7 | W1 | 100 | 0×2, 36, 23, 20×2 → 0×2, 36, 28×3 | 9.12e-09 | 7.32e-09 | 0.803 | 1.78e-08 | 9.68e-09 | 0.544 | 0.970 | 0.785 | 0.81 | 1.293 | 1.357 | 1.05 | 0.311 | 0.572 | 1.84 | 2.95 |
| 1e-7 | W3 (N = 108849) | 30 | 0×2, 20×3 → 0×2, 28×3 | 5.84e-09 | 2.79e-11 | 0.005 | 2.18e-08 | 1.05e-10 | 0.005 | 0.853 | 0.393 | 0.46 | 0.950 | 0.544 | 0.57 | 0.097 | 0.151 | 1.56 | 2.38 |
| 1e-7 | W3 (N = 4032) | 10 | 0×2, 20×2 → 0×2, 28×2 | 1.19e-09 | 8.73e-12 | 0.007 | 7.68e-09 | 2.75e-11 | 0.004 | 0.014 | 0.021 | 1.45 | 0.065 | 0.109 | 1.69 | 0.050 | 0.088 | 1.75 | 8.50 |
| 1e-7 | W4 clusters (N = 100000) | 1 | 0×2, 20×7 → 0×2, 28×7 | 4.22e-09 | 7.97e-11 | 0.019 | 2.87e-08 | 5.54e-10 | 0.019 | 1.385 | 0.797 | 0.58 | 2.072 | 2.077 | 1.00 | 0.681 | 1.280 | 1.88 | 2.81 |
| 1e-7 | W4 clusters (N = 100000) | 100 | 0×2, 26, 20×6 → 0×2, 28×7 | 3.99e-09 | 2.40e-11 | 0.006 | 2.87e-08 | 1.83e-10 | 0.006 | 1.391 | 0.792 | 0.57 | 2.068 | 2.046 | 0.99 | 0.677 | 1.254 | 1.85 | 2.93 |
| 1e-7 | W4 cube (N = 100000) | 1 | 0×2, 20×3 → 0×2, 28×3 | 2.21e-09 | 1.01e-11 | 0.005 | 1.37e-08 | 7.67e-11 | 0.006 | 0.856 | 0.358 | 0.42 | 0.945 | 0.509 | 0.54 | 0.086 | 0.149 | 1.73 | 2.12 |
| 1e-7 | W4 cube (N = 100000) | 100 | 0×2, 36, 23, 20 → 0×2, 36, 28×2 | 9.37e-09 | 6.03e-09 | 0.644 | 1.71e-08 | 8.00e-09 | 0.468 | 0.919 | 0.534 | 0.58 | 1.016 | 0.687 | 0.68 | 0.095 | 0.153 | 1.61 | 2.95 |
| 1e-8 | W1 | 10 | 0×2, 24 → 0×2, 32 | 7.81e-11 | 3.54e-12 | 0.045 | 9.22e-10 | 6.82e-12 | 0.007 | 0.009 | 0.006 | 0.64 | 0.012 | 0.010 | 0.80 | 0.003 | 0.004 | 1.29 | 5.50 |
| 1e-8 | W1 | 30 | 0×2, 24×2 → 0×2, 32×2 | 2.62e-10 | 5.70e-12 | 0.022 | 8.32e-10 | 6.73e-12 | 0.008 | 0.085 | 0.051 | 0.59 | 0.100 | 0.072 | 0.72 | 0.014 | 0.021 | 1.53 | 7.00 |
| 1e-8 | W1 | 100 | 0×2, 38, 24×3 → 0×2, 38, 32×3 | 1.39e-09 | 5.51e-10 | 0.396 | 3.40e-09 | 6.59e-10 | 0.194 | 1.577 | 0.857 | 0.54 | 2.003 | 1.629 | 0.81 | 0.427 | 0.772 | 1.81 | 3.00 |
| 1e-8 | W3 (N = 108849) | 30 | 0×2, 24×3 → 0×2, 32×3 | 7.34e-10 | 6.01e-12 | 0.008 | 2.96e-09 | 7.85e-12 | 0.003 | 1.450 | 0.425 | 0.29 | 1.575 | 0.626 | 0.40 | 0.125 | 0.200 | 1.60 | 2.53 |
| 1e-8 | W3 (N = 4032) | 10 | 0×2, 24×2 → 0×2, 32×2 | 9.83e-11 | 7.44e-12 | 0.076 | 8.92e-10 | 3.73e-12 | 0.004 | 0.025 | 0.021 | 0.84 | 0.092 | 0.133 | 1.45 | 0.067 | 0.110 | 1.65 | 6.33 |
| 1e-8 | W4 clusters (N = 100000) | 1 | 0×2, 24×7 → 0×2, 32×7 | 5.37e-10 | 2.42e-12 | 0.005 | 5.00e-09 | 1.96e-11 | 0.004 | 2.357 | 0.888 | 0.38 | 3.319 | 2.497 | 0.75 | 0.958 | 1.609 | 1.68 | 2.67 |
| 1e-8 | W4 clusters (N = 100000) | 10 | 0×2, 24×7 → 0×2, 32×7 | 5.35e-10 | 2.74e-12 | 0.005 | 5.00e-09 | 1.50e-11 | 0.003 | 2.356 | 0.830 | 0.35 | 3.292 | 2.457 | 0.75 | 0.936 | 1.627 | 1.74 | 2.60 |
| 1e-8 | W4 clusters (N = 100000) | 100 | 0×2, 27, 24×6 → 0×2, 32×7 | 5.01e-10 | 4.41e-12 | 0.009 | 4.98e-09 | 9.40e-12 | 0.002 | 2.360 | 0.890 | 0.38 | 3.303 | 2.526 | 0.76 | 0.943 | 1.635 | 1.73 | 2.87 |
| 1e-8 | W4 cube (N = 100000) | 1 | 0×2, 24×3 → 0×2, 32×3 | 2.57e-10 | 1.98e-12 | 0.008 | 1.73e-09 | 9.41e-12 | 0.005 | 1.453 | 0.389 | 0.27 | 1.573 | 0.577 | 0.37 | 0.119 | 0.187 | 1.57 | 2.29 |
| 1e-8 | W4 cube (N = 100000) | 10 | 0×2, 24×3 → 0×2, 32×3 | 2.29e-10 | 1.62e-11 | 0.071 | 1.71e-09 | 9.06e-12 | 0.005 | 1.449 | 0.377 | 0.26 | 1.562 | 0.574 | 0.37 | 0.113 | 0.197 | 1.74 | 2.29 |
| 1e-8 | W4 cube (N = 100000) | 100 | 0×2, 38, 24×2 → 0×2, 38, 32×2 | 2.11e-09 | 5.54e-10 | 0.263 | 4.52e-09 | 7.63e-10 | 0.169 | 1.480 | 0.573 | 0.39 | 1.611 | 0.784 | 0.49 | 0.122 | 0.209 | 1.72 | 3.00 |
| 1e-8 | W4 plummer (N = 100000) | 1 | 0×2, 24×7 → 0×2, 32×7 | 5.90e-10 | 2.01e-12 | 0.003 | 2.41e-09 | 1.20e-11 | 0.005 | 1.832 | 0.608 | 0.33 | 2.758 | 2.175 | 0.79 | 0.926 | 1.567 | 1.69 | 2.96 |
| 1e-8 | W4 plummer (N = 100000) | 10 | 0×2, 24×7 → 0×2, 32×7 | 4.34e-10 | 2.54e-12 | 0.006 | 2.41e-09 | 1.40e-11 | 0.006 | 1.831 | 0.616 | 0.34 | 2.755 | 2.192 | 0.80 | 0.922 | 1.576 | 1.71 | 3.09 |
| 1e-8 | W4 plummer (N = 100000) | 100 | 0×2, 37, 24×6 → 0×2, 37, 32×6 | 1.27e-09 | 2.89e-10 | 0.228 | 2.59e-09 | 1.28e-10 | 0.049 | 2.583 | 0.931 | 0.36 | 3.474 | 2.497 | 0.72 | 0.886 | 1.559 | 1.76 | 3.10 |
| 1e-8 | W4 sphere (N = 100000) | 1 | 0×2, 24×4 → 0×2, 32×4 | 1.31e-10 | 1.05e-12 | 0.008 | 1.20e-11 | 2.25e-12 | 0.187 | 1.742 | 0.719 | 0.41 | 2.167 | 1.442 | 0.67 | 0.423 | 0.714 | 1.69 | 2.32 |
| 1e-8 | W4 sphere (N = 100000) | 10 | 0×2, 24×4 → 0×2, 32×4 | 1.27e-10 | 6.81e-12 | 0.054 | 1.21e-11 | 2.25e-12 | 0.186 | 1.736 | 0.707 | 0.41 | 2.172 | 1.447 | 0.67 | 0.435 | 0.704 | 1.62 | 2.41 |
| 1e-8 | W4 sphere (N = 100000) | 100 | 0×2, 38, 24×3 → 0×2, 38, 32×3 | 8.86e-10 | 4.28e-10 | 0.483 | 4.86e-11 | 1.54e-11 | 0.317 | 1.760 | 0.919 | 0.52 | 2.201 | 1.635 | 0.74 | 0.441 | 0.707 | 1.60 | 2.84 |

Ranges, PlaneWave ÷ Rotation:

| ε | measure | PlaneWave ÷ Rotation, smallest | largest |
| --- | --- | ---: | ---: |
| 1e-7 | M2L | 0.418 | 1.455 |
| 1e-7 | evaluate | 0.539 | 1.686 |
| 1e-7 | rest | 1.288 | 1.879 |
| 1e-7 | φ | 0.004 | 0.803 |
| 1e-7 | ∇φ | 0.004 | 0.544 |
| 1e-8 | M2L | 0.260 | 0.840 |
| 1e-8 | evaluate | 0.367 | 1.451 |
| 1e-8 | rest | 1.289 | 1.810 |
| 1e-8 | φ | 0.003 | 0.483 |
| 1e-8 | ∇φ | 0.002 | 0.317 |

Every case meets the acceptance, with the largest ratio 0.80 (φ, W1 at kD = 100, 10⁻⁷; its
level 2 at p = 36 is above the resolving degree and keeps its own degree). Where the M2L or the
evaluation is slower, the case is small (W1 at kD = 10, 796 points, 8–12 ms; W3 at kD = 10,
4,032 points, 65–92 ms), or the raised rest outweighs the M2L at 10⁻⁷ on the deep trees (the
clusters, 1.00×; W1 at kD = 30 and 100, 1.10× and 1.05×). The per-level tables (the degree, the
M2L that ran, pairs, seconds, the waves) are in the raw outputs:
`~/projects/fmm-archive/kathleen/logs/ch6-1g/` (20 launches, `runs.log`). Peak resident
memory at most 985 MB.

## 4. D8's bound, revisited

D8 (signed off 2026-11-08) let `Auto` take the exponential M2L where the rule gives a
quadrature at ε ≥ 10⁻⁶. Below 10⁻⁶ the calibrated degrees (p ≥ 20) lie above Batched's crossover,
so Auto runs Rotation there, and §3's PlaneWave ÷ Rotation is what Auto would gain. The question
and its answer, signed off 2026-11-11 (the user: "Lift for 10⁻⁸–10⁻⁷ only (Recommended)"):

- **10⁻⁸ ≤ ε < 10⁻⁷: Auto takes it** (the evaluation 0.37–0.81× on every case above 0.1 s; one
  ms case slower, W3 at N = 4,032, 0.09 → 0.13 s);
- **10⁻⁷ ≤ ε < 10⁻⁶: Auto keeps Rotation** (0.54–1.05× on the cases above 0.1 s, 0.99–1.05× on
  the clusters and W1 at kD = 100, 1.10× on W1 at kD = 30 at 0.06 s) until CH6.1h measures the degree raised on the M2L alone.

Auto's other conditions are unchanged (degree at least `auto_plane_wave_min_degree`,
Im κ_l < 5π, f64). `tests/helmholtz_fmm.rs` checks the default (Auto) bit for bit PlaneWave's
output at 10⁻⁸ and Rotation's at 10⁻⁷ on its four trees.

## 5. Follow-up: CH6.1h (signed off 2026-11-11, next on the priority list)

The whole-level raise costs the leaves, M2M and L2L of a raised level 1.3–1.9× and buys accuracy
nobody asked for (φ 13–330× below Rotation's error where every level with waves is raised). CH6.1f's option (b): a second multipole at p_w
on a level with waves, formed by M2M from the children at p_l, and its local at p_w, taken to the
children by L2L at p_l, so only the M2L runs at p_w. CH6.1h builds it and times it against
CH6.1g's on these cases; if it wins at 10⁻⁷, D8's bound there is asked again.

## 6. Tests

- nd-fmm-tables `tests/helmholtz/plane_wave_merge.rs`: the rule's tolerances and precisions at
  and around every bound, the resolving degree and `plane_wave_degree`, `select` at 10⁻⁷, 10⁻⁸ and
  10⁻⁹.
- nd-fmm-exec `tests/helmholtz/plane_wave.rs`: `the_rule_raises_the_degrees_below_the_level_degrees_tolerance`
  (the degrees, store sizes and tables per level against an operator named at the raised degrees,
  the order of the calls, Rotation afterwards back at the degrees given, named precisions, Auto
  in each band); `auto_takes_the_waves_by_its_rule` (Auto at 10⁻⁸ raised, at 10⁻⁷ and 10⁻⁹ as
  before).
- nd-fmm-exec `tests/helmholtz_fmm.rs`, `check_resolving_degree`: the clusters at kD = 1 (levels
  with waves to |κ_l| = 0.062) and the cube at kD = 20 + 2i, each at 10⁻⁷ and 10⁻⁸: the errors
  0.003–0.011× Rotation's (the M6, debug), the degrees and precisions per level, 2 threads and on
  P ranks the one-rank FMM bit for bit, Auto as in §4.
- nd-fmm-validate `tests/helmholtz_accuracy.rs`: W1 at kD = 10, ε = 10⁻⁸ under the rule within
  1.1× Rotation's, its degrees raised on the levels with waves.

## 7. Reproducing

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch6-1g-runs.sh <out-dir> e7 e8     # on Kathleen through run.sbatch
python3 tools/helmholtz/ch6-1g-tables.py <out-dir>
```
