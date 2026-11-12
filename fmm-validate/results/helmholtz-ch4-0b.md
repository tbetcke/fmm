# Helmholtz CH4.0b: the worst-box concentration

Task: `docs/helmholtz/tasks/CH4.0b.md`. Roadmap: H4, CH4.0b, with the review's amendment
(`docs/reviews/2026-11-12-H6.md`, M5 and A3: a smooth density). Proposal §9.1 (the box
measure), §10 CH4.0b.

## 0. Summary

- **The cause is placement, on the level whose degree has the least slack; not a list, the
  boundary or a defect.** The error at a target grows steeply with the target's distance from
  the centre of its box, on every level. On W1 at kD = 100, the dominant level's part of the
  error at the targets beyond 1.5 half-widths (towards a corner, √3 at the corner; about 1 %
  of the targets) is 19×, 34× and 45× its RMS over the targets within one half-width, at
  10⁻³, 10⁻⁶ and 10⁻⁸. By an argument from the operators' form (not measured) the same holds
  for sources near a corner. The worst boxes are of two kinds:
  - **W1's six tangent points**, where the sphere touches its bounding cube: CH3.6's worst
    boxes, and the worst here under Rotation, under the default and for the constant density.
    Each tangent point is a vertex of a box on every level ((0.5, 0.5, 1) lies on the planes
    x = 0.5 and y = 0.5, which every level's grid holds, and on the domain's face), so the
    worst targets there lie near a corner on levels 2–5 at once (|v|/r 1.52–1.72) and every
    level's part is near its largest.
  - **Boxes the surface clips at one corner**, whose few points all lie near it: the worst
    boxes once a margin and an offset move the grid off the tangent points. There one level
    alone gives as much (30.6 ε at 10⁻⁸ on W1 at kD = 100, level 5 the whole error), so the
    tangent points are not worse than a clipped box, only where the worst boxes happened to
    be.
- **Which level.** The levels' parts (the error minus that of a run with one level's degree
  raised) add up to the whole error (the rest at most 0.2 % of Σ|e|²), so it is the levels'
  truncation and nothing else. Mostly the coarsest level on the degree rule's plateau, where
  the plateau's degree has the least slack (x = Re(k) a_l largest): on W1 at kD = 100, level
  5 (p = 7) carries 96 % of Σ|Δ∇φ|² at 10⁻³, level 4 (p = 15) 70 % at 10⁻⁶, level 3 (p = 24)
  82 % at 10⁻⁸. Which operator of the level (its M2L, or the translations and leaf operators
  at the same degree) was not separated: raising p_l raises them all.
- **Not a list**: a tree of one leaf level (no W or X list, leaves one level up) has the
  errors of the default tree with its finest level raised, to every printed digit. **Not the
  boundary**: with a margin the worst boxes move inside and stay (17.1 ε, 30.6 ε). **Not a
  defect**: the decomposition by degree accounts for all of it.
- **Why it grows as ε falls** (8–25× the ℓ2 error at 10⁻³, 13–146× at 10⁻⁸ in CH3.6): the
  truncation error falls like a power p of a distance ratio, which is largest at a corner, so
  the higher p, the more the corner stands out. ε is a relative ℓ2 tolerance calibrated on
  all targets, of which about 1 % lie beyond 1.5 half-widths.
- **The smooth density (review M5, A3).** On W1 at kD = 10, 30, 100 and 500, at 10⁻³, 10⁻⁶
  and 10⁻⁸, under Rotation and the default, the plane wave's trace and the constant density
  (kD ≤ 100) meet ε in §9.1's ℓ2 norm on every case. The plane wave's trace is no harder
  than the random charges (∇φ at 10⁻⁸ 0.02–0.42 ε against 0.08–0.63 ε under Rotation); the
  constant density is the hardest, 0.66 ε (∇φ, kD = 100, 10⁻⁸, Rotation). No degree changes
  for them in ℓ2, the exponential M2L's p_w at 10⁻⁶ included. Per box they do not meet ε, as
  the random charges do not: the constant density's worst box is 42 ε at 10⁻⁸, at the same
  tangent points (§3); the degrees it needs pointwise are in §4.
- **Below D1's range** the constant density finds what the random charges did not: under the
  rule its ℓ2 error stops near 3.4·10⁻¹⁰ (∇φ, W1 at kD = 100), 1.5× ε at 10⁻⁹ and 3.4× at
  10⁻¹⁰, all of it from level 2 on the high-frequency branch (review M7; §4). It belongs to
  CH4.0c, whose roadmap line now includes the constant density.
- **A pointwise criterion** would cost the degrees of a tolerance 10× tighter at 10⁻³ and
  10⁻⁶ (1.1–2.3× the evaluation) and 30–100× tighter at 10⁻⁸ (2.9–5.2×), on W1 at kD = 100
  and 500 (§4). The recommendation (question 1): ε stays §9.1's ℓ2 tolerance, and the box
  measure joins §9.1 as a bound the gates check, 100 ε.

## 1. Method

- **The tool.** `nd_fmm_validate::helmholtz_boxes` and the `helmholtz_boxes` example (one
  rank, release), `tools/helmholtz/ch4-0b-runs.sh`. The FMM is CH3.1c's harness
  (`helmholtz_accuracy`) on the workloads of §9.1 at D = 1, one complex charge vector,
  gradients on, at the default degree rule (`Degrees::Tolerance(ε)`).
- **The reference.** The direct sum in f64 (`nd_fmm_ref::helmholtz::p2p::direct_sum`, with
  1/(4π)) at every point up to N = 2·10⁵, so on every case but W1 at kD = 500; there at §9.1's
  1,000 samples plus two targets drawn in each box of the FMM's deepest level that holds
  points (143,695 targets; the box maximum is then a lower bound of the box's, and the ℓ2
  over these targets weighs sparse boxes more than the points do).
- **The box measure** (§9.1, as CH3.6 built it): per box of the FMM's deepest level, the
  largest error at one of its reference targets over the reference's RMS over its targets,
  of φ and of ∇φ, over ε ("worst box / ε") and over the relative ℓ2 error over the same
  targets ("worst / ℓ2"). With the reference at every point it is CH3.6's `crosscheck` bit
  for bit (`tests/helmholtz_accuracy.rs`, `check_box_study`). CH3.6 measured against FMM3D
  at 10⁻¹²; the cases re-run here reproduce its numbers to every printed digit (W1 at kD = 30
  and 10⁻⁶: 3.99 ε and 16.69 ε; W1 at kD = 100 and 10⁻⁸: 9.59 ε and 21.89 ε; W3 at kD = 30
  and 10⁻⁸: 3.25 ε and 13.27 ε; the Plummer sphere at kD = 10 and 10⁻⁸: 1.73 ε and 10.33 ε).
- **The levels.** For each level l with M2L, a run with p_l raised to the rule's degree at
  ε/1000 (at least p_l + 1), the other levels as the rule gives them, and one with every
  level raised. Level l's part of the error at a target is the first run's error minus the
  raised run's (complex, per component); its share is Σ|part|² over Σ|error|²; "rest" is
  what the parts leave of the error. Named degrees change the exponential M2L's precision
  rule, so the decomposition runs on Rotation (all-RCR, as CH3.6).
- **The place.** For a target x and a level l, |v|/r: x's distance from the centre of its
  level-l box over the box's half-width r (0 at the centre, 1 at a face's centre, √3 at a
  corner). The level parts are binned by it.
- **Lists and boundary.** The same case on a tree of one leaf level (`--leaf-size uniform`:
  no W or X list), and in a domain with a margin of 0.03, and of 0.1 with the offset
  (0.013, 0.029, 0.041), so that no level's grid meets the sphere as the bounding cube's does.
- **Machines.** The M6 (one rank × 6 threads; accuracy only, its times not read) for every
  case to kD = 100 (`~/projects/fmm-archive/m6/logs/ch4-0b/`: `cause/`, `densities/`,
  `extra/`); D1's accuracy does not depend on the machine beyond the P2P ISA's rounding; Kathleen job 633154 (one
  node, node-c11b-041, one rank × 40 threads) for W1 at kD = 500 and the times of §4
  (`~/projects/fmm-archive/kathleen/logs/ch4-0b/`).

## 2. The cause

### 2.1 The levels

Each level's share of Σ|Δφ|² and Σ|Δ∇φ|² over every target, Rotation, at the rule's degrees:

| Case | ε | p by level (2…) | share φ by level | share ∇φ by level | rest φ, ∇φ |
| --- | --- | --- | --- | --- | --- |
| W1, kD = 30 | 10⁻⁶ | 16, 15 | 0.84, 0.12 | 0.57, 0.33 | 0.000, 0.000 |
| W1, kD = 100 | 10⁻³ | 30, 18, 11, 7 | 0.01, 0.01, 0.07, **0.90** | 0.00, 0.00, 0.03, **0.96** | 0.002, 0.002 |
| W1, kD = 100 | 10⁻⁶ | 35, 22, 15, 15 | 0.05, 0.02, **0.83**, 0.08 | 0.01, 0.01, **0.70**, 0.29 | 0.000, 0.000 |
| W1, kD = 100 | 10⁻⁸ | 38, 24, 24, 24 | 0.15, **0.89**, 0.01, 0.01 | 0.07, **0.82**, 0.05, 0.11 | 0.000, 0.000 |
| W1, kD = 100, margin 0.1 + offset | 10⁻⁸ | 43, 27, 24, 24 | 0.09, **0.80**, 0.08, 0.04 | 0.03, 0.36, 0.16, **0.45** | 0.000, 0.000 |
| W3, kD = 30 (N = 108,849) | 10⁻⁸ | 24, 24, 24 | 0.67, 0.23, 0.12 | 0.19, 0.24, **0.57** | 0.000, 0.000 |
| W4 Plummer, kD = 10 (N = 10⁵) | 10⁻⁸ | 24 × 7 | 0.03, 0.20, 0.22, 0.26, 0.20, 0.06, 0.00 | 0.00, 0.01, 0.03, 0.12, 0.37, 0.39, 0.06 | 0.000, 0.000 |

On W1 the level that dominates is the coarsest one on the plateau of the degree rule (the
plateau's degree, 7, 15 or 24, with the largest x = Re(k) a_l): at kD = 100 and 10⁻⁸ level 3
has x = 10.8, where the high-frequency branch gives 23.2 and the plateau 24, the edge of
both. At the low frequencies of W4 (every level on the plateau, small x) the shares spread
over the levels, ∇φ's towards the fine ones. The decomposition is exact to the digits
printed (the parts add up to the error), so the error is the levels' truncation and nothing
else; raising every level to the degrees of ε/1000 removes it (W1 at kD = 100 and 10⁻⁸: worst
box 0.06 ε).

### 2.2 The place within a box

Level l's part of the ∇φ error by the target's distance from the centre of its level-l box,
RMS over the reference's RMS, over ε (W1 at kD = 100, Rotation; 79,578 targets, about 1 % in
the last column):

| ε | level | < 0.5 | 0.5–0.75 | 0.75–1.0 | 1.0–1.25 | 1.25–1.5 | ≥ 1.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10⁻³ | 5 (dominant) | 0.022 | 0.026 | 0.039 | 0.090 | 0.266 | 0.649 |
| 10⁻⁶ | 4 (dominant) | 0.018 | 0.032 | 0.075 | 0.127 | 0.392 | 2.018 |
| 10⁻⁶ | 5 | 0.007 | 0.016 | 0.049 | 0.081 | 0.220 | 1.393 |
| 10⁻⁸ | 2 | 0.039 | 0.037 | 0.035 | 0.036 | 0.037 | 0.912 |
| 10⁻⁸ | 3 (dominant) | 0.040 | 0.050 | 0.104 | 0.160 | 0.275 | 3.906 |
| 10⁻⁸ | 4 | 0.000 | 0.001 | 0.008 | 0.017 | 0.048 | 0.805 |
| 10⁻⁸ | 5 | 0.000 | 0.001 | 0.010 | 0.019 | 0.051 | 1.167 |

The same profile on every level and case of §2.1 (the outputs' last table): W3 at kD = 30 and
10⁻⁸ 0.42–0.75 beyond 1.5 against 0.024–0.032 at 1.25–1.5; the Plummer sphere at kD = 10
0.27–0.68 against 0.016–0.048 on levels 5–8. Level 2 at 10⁻⁸ (p = 38, x = 21.7, the
high-frequency branch) is flat inside 1.5 and steps up beyond it.

The table bins targets. A source near a corner counts as much: on one level the FMM's chain
for a pair of boxes is P2M, M2L, L2P with L2P the transpose of P2M (the same ĵ_n Y_n^m) and
M2L the transpose of the M2L the other way, so its error for a source at y and a target at x
is its error for a source at x and a target at y. This is an argument from the operators'
form, not a measurement here.

### 2.3 The worst boxes

W1 at kD = 100, 10⁻⁸, Rotation, the bounding cube (CH3.6's domain): the eight worst boxes of
∇φ (level 5) all touch the six tangent points, with their worst target's |v|/r on levels 2–5
and the levels' parts of its error (each part's magnitude over the error's; the parts are
complex and partly cancel, so they need not add to 1):

| # | cell | error / ε | targets | |v|/r, levels 2–5 | level parts |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | (15, 31, 15) | 21.89 | 25 | 1.72 1.70 1.68 1.62 | 2: 0.26, 3: 1.22, 4: 0.46 |
| 2 | (31, 15, 16) | 21.49 | 25 | 1.72 1.71 1.69 1.64 | 2: 0.17, 3: 0.94, 4: 0.12, 5: 0.22 |
| 3 | (15, 0, 15) | 20.13 | 25 | 1.71 1.69 1.65 1.56 | 2: 0.32, 3: 1.05, 4: 0.16 |
| 4 | (16, 15, 31) | 19.69 | 26 | 1.72 1.71 1.69 1.64 | 2: 0.25, 3: 1.02 |
| 5 | (0, 15, 16) | 19.02 | 25 | 1.60 1.49 1.38 1.63 | 5: 1.00 |
| 6 | (15, 15, 0) | 16.75 | 27 | 1.72 1.70 1.68 1.62 | 2: 0.19, 3: 0.86, 4: 0.17 |
| 7 | (0, 15, 14) | 16.09 | 25 | 1.45 1.20 0.94 1.62 | 5: 1.00 |
| 8 | (0, 16, 15) | 15.99 | 24 | 1.72 1.71 1.68 1.63 | 2: 0.22, 3: 0.84, 4: 0.46 |

The tangent point (0.5, 0.5, 1) and its five images are vertices of a box on every level,
so the sheet of points there lies near a box corner on levels 2–5 at once (six of the
eight), where level 3's part dominates; in #5 and #7 the worst target is near a corner of
its level-5 box only, and level 5's part is its whole error. Under the default (the
exponential M2L on levels 3–5, p = 32 there) and for the constant density the worst boxes
are the same tangent boxes, their worst targets at |v|/r 1.52–1.72 on levels 2–5
(`extra/w1-100-e8-default.md`, `extra/w1-100-e8-constant.md`: 8.67 ε and 42.16 ε). The
errors by the number of levels on which a target lies within 0.1 of a face say the same:
targets near a face on all four levels have an RMS error 15× and a largest error 19× those
of targets near none (∇φ, 1.49 ε against 0.10 ε RMS, 21.9 ε against 1.18 ε).

With the grid moved off the tangent points the worst boxes are the clipped ones:

| Domain | worst φ, ∇φ / ε | worst / ℓ2 | worst box | targets | |v|/r, levels 2–5 | level parts |
| --- | --- | --- | --- | ---: | --- | --- |
| bounding cube (CH3.6) | 9.59, 21.89 | 58, 59 | (15, 31, 15), boundary | 25 | 1.72 1.70 1.68 1.62 | 3: 1.22 |
| margin 0.03 | 8.31, 17.05 | 48, 43 | (24, 3, 11), inside | 2 | 1.00 1.69 1.64 1.56 | – |
| margin 0.1, offset | 3.85, 30.59 | 40, 110 | (2, 14, 19), inside | 3 | 0.83 0.86 0.04 1.66 | 5: 1.00 |
| one leaf level (no W, X) | 9.55, 22.21 | 58, 64 | (7, 15, 7), boundary (level 4) | 100 | 1.72 1.70 1.68 | – |

(The table's "worst / ℓ2" is over the reference's targets, every point.) A clipped box at one
level gives more than the tangent boxes at all of them (30.6 ε against 21.9 ε). On W1 at
kD = 500 the per-box sample's worst boxes at 10⁻³ (Rotation) are boxes with one or two sampled
targets (one target: a box of one point) whose worst target lies at |v|/r 1.66–1.71 on
level 7.

### 2.4 What it is not

- **Not a list.** A tree of one leaf level has no W or X list and its leaves one level up
  (level 4 at kD = 100). Its errors equal those of the default tree with level 5 raised to
  every printed digit (`cause/w1-100-e8-uniform.md` and the row "level 5 at 42" of
  `cause/w1-100-e8-levels.md`: 1.64e-9 and 3.49e-9 over every point, worst box 9.55 ε and
  22.21 ε), so the W and X lists add nothing visible.
- **Not the boundary.** The margin moves the worst boxes inside the domain and keeps them, as
  CH3.6's throwaway experiment saw; they move to the clipped boxes.
- **Not a defect.** Raising the degrees removes the errors level by level, and the levels'
  parts add to the whole error. Rotation and the default concentrate alike (§3: W1 at
  kD = 500 and 10⁻⁶, worst box 14.7 ε and 14.5 ε; §2.3). Which operator of a level carries
  its part (M2L against the translations and leaf operators at the same degree) is not
  separated here: raising p_l raises all of them, and the per-pair chain that would separate
  them (the plan's stage B) was not built.
- **Not W1 alone.** W3's and the Plummer sphere's errors have the same profile in |v|/r
  (§2.2); W1 adds the tangent points, where every level's corner coincides.

### 2.5 FMM3D

CH3.6's FMM3D worst boxes are 6–37× its ℓ2 error at 10⁻⁸ against our 13–146×, and below ε
pointwise, since its ℓ2 error is 10–100× below the requested ε (review B3). That fits a
degree chosen for the worst place in a box rather than the average, but FMM3D's code was not
read for it here; an inference, not a finding.

## 3. The smooth density

W1, §9.1's relative ℓ2 errors (φ, ∇φ) at the 1,000 samples, and the worst box (∇φ) over ε,
under Rotation and the default (`Auto`, the exponential M2L by D8):

| kD | density | ε | Rotation φ, ∇φ | default φ, ∇φ | worst ∇φ box / ε, Rotation, default |
| ---: | --- | --- | --- | --- | --- |
| 10 | random | 10⁻³ | 4.40e-5, 1.53e-4 | 4.40e-5, 1.53e-4 | 1.41, 1.41 |
|  |  | 10⁻⁶ | 5.62e-8, 4.11e-7 | 5.65e-8, 4.09e-7 | 5.98, 5.90 |
|  |  | 10⁻⁸ | 7.81e-11, 9.22e-10 | 3.54e-12, 6.82e-12 | 1.26, 0.01 |
|  | plane wave | 10⁻³ | 1.21e-5, 6.85e-5 | 1.21e-5, 6.85e-5 | 0.42, 0.42 |
|  |  | 10⁻⁶ | 1.14e-8, 1.07e-7 | 1.29e-8, 1.07e-7 | 1.37, 1.38 |
|  |  | 10⁻⁸ | 1.87e-11, 3.12e-10 | 3.46e-12, 4.63e-12 | 0.36, 0.00 |
|  | constant | 10⁻³ | 1.09e-5, 9.72e-5 | 1.09e-5, 9.72e-5 | 0.63, 0.63 |
|  |  | 10⁻⁶ | 1.13e-8, 1.85e-7 | 1.32e-8, 1.86e-7 | 2.54, 2.52 |
|  |  | 10⁻⁸ | 1.79e-11, 5.51e-10 | 7.93e-12, 1.49e-11 | 0.67, 0.01 |
| 30 | random | 10⁻³ | 2.66e-5, 7.15e-5 | 2.66e-5, 7.15e-5 | 0.99, 0.99 |
|  |  | 10⁻⁶ | 1.12e-7, 2.97e-7 | 1.12e-7, 2.97e-7 | 16.69, 16.54 |
|  |  | 10⁻⁸ | 2.62e-10, 8.32e-10 | 5.70e-12, 6.73e-12 | 6.82, 0.04 |
|  | plane wave | 10⁻³ | 1.31e-5, 6.04e-5 | 1.31e-5, 6.04e-5 | 1.26, 1.26 |
|  |  | 10⁻⁶ | 3.47e-8, 1.49e-7 | 3.53e-8, 1.49e-7 | 3.13, 3.12 |
|  |  | 10⁻⁸ | 4.98e-11, 1.86e-10 | 3.32e-12, 3.74e-12 | 0.74, 0.00 |
|  | constant | 10⁻³ | 1.99e-5, 1.49e-4 | 1.99e-5, 1.49e-4 | 1.48, 1.48 |
|  |  | 10⁻⁶ | 5.12e-8, 2.31e-7 | 5.23e-8, 2.32e-7 | 5.34, 5.28 |
|  |  | 10⁻⁸ | 8.96e-11, 4.51e-10 | 9.75e-12, 1.45e-11 | 2.08, 0.01 |
| 100 | random | 10⁻³ | 4.15e-5, 1.36e-4 | 4.15e-5, 1.36e-4 | 1.63, 1.63 |
|  |  | 10⁻⁶ | 9.61e-8, 2.94e-7 | 9.56e-8, 2.88e-7 | 11.41, 10.36 |
|  |  | 10⁻⁸ | 1.39e-9, 3.40e-9 | 6.28e-10, 8.74e-10 | 21.89, 8.67 |
|  | plane wave | 10⁻³ | 1.99e-5, 1.01e-4 | 1.99e-5, 1.01e-4 | 1.62, 1.62 |
|  |  | 10⁻⁶ | 3.33e-8, 1.31e-7 | 3.42e-8, 1.30e-7 | 4.19, 4.20 |
|  |  | 10⁻⁸ | 6.71e-10, 2.30e-9 | 2.55e-10, 1.09e-9 | 21.90, 18.21 |
|  | constant | 10⁻³ | 4.63e-5, 1.86e-4 | 4.63e-5, 1.86e-4 | 3.09, 3.09 |
|  |  | 10⁻⁶ | 1.13e-7, 4.20e-7 | 1.16e-7, 4.16e-7 | 8.19, 8.24 |
|  |  | 10⁻⁸ | 3.70e-9, 6.61e-9 | 3.22e-9, 5.48e-9 | 42.16, 34.85 |
| 500 | random | 10⁻³ | 3.24e-5, 8.71e-5 | 3.24e-5, 8.71e-5 | 1.52, 1.52 |
|  |  | 10⁻⁶ | 1.49e-7, 3.79e-7 | 1.45e-7, 3.73e-7 | 14.71, 14.52 |
|  |  | 10⁻⁸ | 2.72e-9, 6.33e-9 | 1.62e-9, 3.21e-9 | 33.77, 24.73 |
|  | plane wave | 10⁻³ | 1.90e-5, 8.19e-5 | 1.90e-5, 8.19e-5 | 2.32, 2.32 |
|  |  | 10⁻⁶ | 5.98e-8, 2.51e-7 | 6.00e-8, 2.51e-7 | 11.64, 11.64 |
|  |  | 10⁻⁸ | 1.34e-9, 4.20e-9 | 9.32e-10, 2.50e-9 | 36.76, 25.87 |

kD = 500's worst boxes are the per-box sample's (§1), lower bounds; CH3.6's full crosscheck
found 56.6 ε for the random charges at 10⁻⁸ there. The constant density was run to kD = 100
(the roadmap's "a constant density at low kD").

- Every run meets ε in ℓ2; the largest is the constant density at kD = 100 and 10⁻⁸ under
  Rotation, ∇φ 0.66 ε (the default 0.55 ε).
- The plane wave's trace is not harder than the random charges: under Rotation ∇φ is
  0.02–0.42 ε against 0.08–0.63 ε at 10⁻⁸, and 0.11–0.25 ε against 0.29–0.41 ε at 10⁻⁶. M5's
  dilution is real for the constant density (1.1–2.7× the random charges' errors at
  kD = 100) but stays inside ε; the random charges were not the easy case for the plane
  wave's trace.
- At 10⁻⁶ the default runs the exponential M2L with p_w = p_l (iprec 2): its errors on the
  smooth densities are Rotation's at kD = 100 and 500 (the levels with waves
  there; within 3 % at kD = 100 and 500, 13–17 % at kD = 10, all far inside ε), so A3's question (iprec 2 at p = 15 on a smooth density) is answered:
  no p_w change is needed.
- The worst boxes are of the random charges' size or up to twice it (42 ε at kD = 100). The
  constant density's worst at kD = 100 and 10⁻⁸ are the tangent boxes, their worst targets
  at |v|/r 1.58–1.72 on levels 2–5 (§2.3); the other runs' worst boxes were not placed (the
  example places its first run's only).

## 4. A pointwise criterion

What it would take for the worst box to meet the requested ε, under the default: W1 at each
kD, the rule run at tighter tolerances, the worst box (∇φ) over the *requested* ε. Kathleen,
one rank × 40 threads, job 633154, one evaluation each (single launches; kD = 100's times
are a few hundred ms and only indicative).

| kD | requested ε | rule run at | p by level (2…) | §9.1 ∇φ | worst ∇φ box / requested ε | evaluation s (× the rule at ε) |
| ---: | --- | --- | --- | --- | ---: | --- |
| 100 | 10⁻³ | 10⁻³ | 30, 18, 11, 7 | 1.36e-4 | 1.63 | 0.24 |
|  |  | 10⁻⁴ | 32, 19, 12, 9 | 1.70e-5 | **0.31** | 0.31 (1.3×) |
|  | 10⁻⁶ | 10⁻⁶ | 35, 22, 15, 15 | 2.88e-7 | 10.36 | 0.53 |
|  |  | 10⁻⁷ | 36, 28, 28, 28 | 1.15e-8 | **0.88** | 1.20 (2.3×) |
|  | 10⁻⁸ | 10⁻⁸ | 38, 32, 32, 32 | 8.74e-10 | 8.67 | 1.65 |
|  |  | 10⁻⁹ | 39, 30, 30, 30 | 2.24e-10 | 2.19 | 3.74 (2.3×) |
|  |  | 3·10⁻¹⁰ | 40, 33, 33, 33 | 6.13e-11 | **0.63** | 4.82 (2.9×) |
| 500 | 10⁻³ | 10⁻³ | 123, 66, 36, 21, 13, 8 | 8.71e-5 | 1.52 | 14.27 |
|  |  | 10⁻⁴ | 126, 68, 38, 22, 14, 9 | 2.34e-5 | **0.48** | 15.99 (1.1×) |
|  | 10⁻⁶ | 10⁻⁶ | 131, 72, 41, 25, 16, 15 | 3.73e-7 | 14.52 | 21.34 |
|  |  | 10⁻⁷ | 133, 74, 43, 28, 28, 28 | 1.39e-8 | **0.93** | 40.26 (1.9×) |
|  | 10⁻⁸ | 10⁻⁸ | 135, 76, 44, 32, 32, 32 | 3.21e-9 | 24.73 | 45.48 |
|  |  | 10⁻⁹ | 138, 77, 46, 30, 30, 30 | 4.78e-10 | 3.17 | 146.85 (3.2×) |
|  |  | 3·10⁻¹⁰ | 139, 78, 46, 33, 33, 33 | 1.73e-10 | 1.33 | 186.54 (4.1×) |
|  |  | 10⁻¹⁰ | 140, 79, 47, 36, 36, 36 | 5.90e-11 | **0.55** | 234.80 (5.2×) |

The constant density on W1 at kD = 100 on the M6 (accuracy only; `extra/`):

| requested ε | rule run at | §9.1 φ, ∇φ | worst ∇φ box / requested ε |
| --- | --- | --- | ---: |
| 10⁻³ | 10⁻³ | 4.63e-5, 1.86e-4 | 3.09 |
|  | 10⁻⁴ | 6.32e-6, 2.36e-5 | **0.45** |
| 10⁻⁶ | 10⁻⁶ | 1.16e-7, 4.16e-7 | 8.24 |
|  | 10⁻⁷ | 2.60e-8, 3.48e-8 | 2.11 |
|  | 3·10⁻⁸ | 5.35e-9, 2.26e-8 | 1.38 |
|  | 10⁻⁸ | 3.22e-9, 5.48e-9 | **0.35** |
| 10⁻⁸ | 10⁻⁸ | 3.22e-9, 5.48e-9 | 34.85 |
|  | 10⁻⁹ | 4.33e-10, 1.51e-9 | 9.95 |
|  | 3·10⁻¹⁰ | 1.87e-10, 3.37e-10 | 2.14 |
|  | 10⁻¹⁰ | 1.88e-10, 3.39e-10 | 2.17 |

- A tolerance 10× tighter puts the worst box within ε at 10⁻³ and 10⁻⁶, at 1.1–1.3× the
  evaluation at 10⁻³ and 1.9–2.3× at 10⁻⁶ (from 10⁻⁷ the rule raises the levels with waves
  to p_w = 28). At 10⁻⁸ it takes 30× (kD = 100) to 100× (kD = 500), 2.9–5.2× the
  evaluation: below 10⁻⁸ the default has no exponential M2L (D8) and the degree rule is
  extrapolated (calibrated on 10⁻⁸ ≤ ε ≤ 10⁻³).
- kD = 500's worst boxes are the per-box sample's, lower bounds (§1), so its factors may
  be short; kD = 100's are over every point.
- The constant density needs more: ε/100 at 10⁻⁶, and at 10⁻⁸ no tolerance reaches it. Below
  10⁻⁸ its error stops falling at about 3.4·10⁻¹⁰ (∇φ), so the rule misses ε in ℓ2 there:
  1.5× at 10⁻⁹, 1.1× at 3·10⁻¹⁰, 3.4× at 10⁻¹⁰, outside D1's range (10⁻³–10⁻⁸) and the rule's
  calibration. The levels' parts at 10⁻¹⁰ (`extra/w1-100-e10-constant-levels.md`): level 2,
  p = 40 at x = 21.7 on the high-frequency branch, carries the whole error (share 1.00);
  raising it to 57 gives 3.1·10⁻¹² (∇φ). This is review M7's short high-frequency branch below
  10⁻⁸, CH4.0c's subject; the random charges did not show it (5.9·10⁻¹¹ at 10⁻¹⁰).
- The build takes 10.7–15.0 s at kD = 500 in every run, with no trend in the tolerance.
- At 10⁻⁸ the ε/10 run's ℓ2 error (∇φ) is 0.02–0.05 ε, near FMM3D's at its requested 10⁻⁸
  (CH3.6: 2.1·10⁻¹⁰ and 6.3·10⁻¹¹ on W1 at kD = 100 and 500): matching FMM3D's accuracy and
  matching its pointwise behaviour cost about the same.

## 5. Questions for sign-off

1. **The criterion.** Recommended: ε stays §9.1's relative ℓ2 tolerance, on which the rule is
   calibrated and every gate measured; the box measure joins §9.1 as a bound the gates
   check: the worst box of the deepest level at most 100 ε, reported per case as now. The
   largest measured in D1's range is 57 ε (CH3.6's W1 at kD = 500 and 10⁻⁸; 42 ε here);
   below it, at 10⁻¹⁰, outside the rule's calibration, 56.3 ε (kD = 100) and 54.9 ε
   (kD = 500, a lower bound) (§4). A gate that
   exceeds it is a finding, as an ℓ2 miss is. The proposal's §9.1 and the degree rule's
   documentation say that pointwise errors reach 10–60 ε near box corners. Alternatives:
   (b) a pointwise ε, the rule recalibrated to put the worst box within ε, at the cost of §4;
   (c) (a) now, and an opt-in pointwise tolerance later (a `Degrees` form mapping ε to the
   rule's tolerance at ε/10 to ε/100), a follow-up task off the priority list, which CH4.5's
   comparison with FMM3D at matched accuracy (review B3) could use.
2. **The charge model.** Recommended: the gates keep §9.1's random charges (D12 unchanged);
   the smooth densities met ε everywhere in D1's range, at most about twice the random
   charges' error. CH4.0c, CH4.3 and CH4.4 add the constant density (kD ≤ 100) and the plane
   wave's trace (kD = 500) to their verification (`Density` in the harness): the constant
   density is the case nearest ε, and below 10⁻⁸ it shows the high-frequency branch short
   where the random charges do not (§4). Alternative: a smooth density as a standing
   workload of §9.1, amending D12.

## 6. Reproduce

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_boxes
tools/helmholtz/ch4-0b-runs.sh cause <out>                  # the M6, about 12 min
tools/helmholtz/ch4-0b-runs.sh densities <out>              # the M6, kD = 10, 30, 100
tools/kathleen/sync.sh
tools/kathleen/submit.sh --time=04:00:00 -- run.sbatch \
    'cargo build --release -p nd-fmm-validate --example helmholtz_boxes' \
    "KDS=500 DENSITIES='random planewave' tools/helmholtz/ch4-0b-runs.sh densities <out>/densities" \
    "KDS='100 500' tools/helmholtz/ch4-0b-runs.sh pointwise <out>/pointwise"
```

The runs added after the self-review (`extra/` on the M6), each `helmholtz_boxes --kd 100
--threads 6` on one rank with:

```text
--eps 1e-8 --m2l default                                  # the default's worst boxes
--eps 1e-8 --density constant [--m2l default]             # the constant density's
--density constant --m2l default --eps 1e-3,1e-4,3e-5,1e-5     # and e-6 to e-8, e-8 to e-10
--density constant --eps 1e-10 --study levels              # the floor below 10⁻⁸
```
