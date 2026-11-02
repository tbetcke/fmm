# Plane-wave quadratures per k w_l: spike report (Helmholtz CH0.4, S4)

Helmholtz task CH0.4 ([docs/helmholtz/tasks/CH0.4.md](../../docs/helmholtz/tasks/CH0.4.md)).
The question is proposal §10 CH0.4 ("how FMM3D obtains them, and a generator or table for
this library with its accuracy over the k w_l of low-frequency levels"), with §5.3, §5.5,
§5.6 and risk R7. Written on 2026-11-02 from branch `tbetcke/h-ch0-4` (main at 4f7b055
plus this spike), on the M6, release builds, rustc 1.99.0, f64, threads from rayon (every
figure is a maximum, independent of the thread count). Accuracy only; nothing is timed.

## Summary

1. **FMM3D tabulates its quadratures; it does not generate them** (§2). `hwts3e.f`
   reads one quadrature per **domain**, a rectangle of the complex wavenumber in box
   widths κ = k w (23 × 19 rectangles up to Re κ < 16π, Im κ < 12π), and **precision**
   (`iprec` 0 to 4 from eps: 2, 3, 6, 9 and 12 digits), each valid uniformly over its
   rectangle: 437 × 5 tables of 1 to 96 real nodes with weights, their Fourier and α
   counts per node, and a stated error. Below Im κ = π/2 the nodes lie on a contour in λ
   from −iκ to ∞ (type 2), above it on the real t axis (type 1). The files do not say
   how they were made (NIST's notice; a generalised Gaussian quadrature is the evident
   method, not stated). FMM3D uses plane waves on a level when Re κ ≤ 16π, Im κ < 12π
   and the level is finer than the ones its decay cut drops: boxes of at most **8
   wavelengths**, not the 32 its documentation and our proposal state (§2.4).
2. **The tables are what they state** (§3). Over all 437 × 5 tables at 9 points of each
   rectangle, the λ sums with α resolved meet the stated error to 1.5× (5.5× at iprec 4
   next to Re κ = 16π, at 5·10⁻¹³, round-off), in FMM3D's norm: the absolute error of
   e^{iκ|u|}/|u| in box widths over the region 1 ≤ z ≤ 4, |x|, |y| ≤ 4 (the norm of
   CONVENTIONS §3.14's Laplace tables).
3. **FMM3D's α counts do not keep the stated error at the region's corner** (§3.2). With
   FMM3D's own Nₖ the kernel's error reaches 69–119× the stated error (3.8·10⁻⁶ at
   iprec 2, the precision of 5·10⁻⁷ ≤ ε < 5·10⁻⁴; 3.0·10⁻⁹ at iprec 3), always at
   ρ = 4√2, z = 1; within ρ ≤ 4 it is the stated error, and Nₖ × 1.25 to × 2 restores
   it. The Laplace tables behave alike (6–23×, CONVENTIONS §3.14's spike), and there the
   counts were chosen by measurement.
4. **An independent generator works for any κ** (§4): the Sommerfeld integral on the
   contour of real t (no wave grows), composite Gauss–Legendre, compressed by a
   column-pivoted Gram–Schmidt with refitted weights, α and Fourier counts from Bessel
   bounds. It meets ε on the region, off its samples and in the point-to-point chain,
   for real, imaginary and complex κ, inside and outside FMM3D's domains. At FMM3D's own
   measured error it needs 1.5–1.9× FMM3D's wave values Σ Nₖ for real and weakly damped
   κ, and 2.3–9.6× for Yukawa and strongly damped κ, where FMM3D's tables get by with a few
   nodes (§4.1). So the tables are the cheaper source; the generator is the fallback.
5. **The tables cover every level that would use them** (§5). On the reference trees
   (root side 1, κ_l = kD/2^l) every unpruned level with Re κ_l < 16π falls in a domain;
   at kD = 500 levels 2 and 3 (κ = 125, 62.5, boxes 20 and 10 λ) fall outside (high
   frequency, H4's), and Yukawa levels outside Im κ < 12π are pruned (§5.5) before they
   reach the edge.
6. **Recommendation for CH6.1** (§6): FMM3D's tables, extracted with NIST's notice as
   §3.14's Laplace tables were (iprec 1 to 3 serve D1's ε from 10⁻³ to 10⁻⁸), with the α
   counts chosen by measurement on the region, not taken from `hnumphys`; the generator
   as the trusted path of the tests and for any level outside the tables.

## 1. The representation

For a target above a source by u = (x, y, z) in box widths and κ = k w,

```math
\frac{e^{i\kappa|u|}}{|u|} = \int_\Gamma e^{-\lambda z} J_0(t\rho)\, d\lambda,
\qquad t = \sqrt{\lambda^2 + \kappa^2},\ \rho = \sqrt{x^2 + y^2},
```

on a contour Γ from λ = −iκ (t = 0) to +∞; the integrand is entire in λ, so Γ is free
up to growth. With J₀(tρ) = (1/2π) ∫ e^{it(x cos α + y sin α)} dα and the trapezoidal
rule in α, a quadrature is (λₖ, wₖ, Nₖ, Fₖ):

```math
\frac{e^{i\kappa|u|}}{|u|} \approx \sum_k \frac{w_k}{N_k} \sum_{j<N_k}
e^{-\lambda_k z + i t_k (x\cos\alpha_j + y\sin\alpha_j)}, \qquad \alpha_j = 2\pi j / N_k,
```

and a source's waves at λₖ keep the orders |m| ≤ Fₖ. The spike evaluates three things on
the region (`src/region.rs`): the λ sum with α resolved to round-off (Nₖ ≥ |tₖ| ρ_max + 60),
the **kernel** with the quadrature's Nₖ, on the grid z ∈ {1, 1.25, …, 4},
x, y ∈ {−4, −3.5, …, 4}, and the **chain**: sources at the corners, the centre and seeded
points of the box [−½, ½]³, their waves truncated to |m| ≤ Fₖ (coefficients by a DFT
exact to round-off) and sampled at Nₖ nodes, evaluated at the 15 × 15 × 5 tensor of
corners, edge and face midpoints and centres of the up direction's target boxes. Errors
are absolute in box widths (FMM3D's norm), and also relative to the sample's largest |G|
(`results/domains.md`).

## 2. How FMM3D obtains them

FMM3D v2.1.0, `src/Helmholtz/` (the release `tools/fmm3d/build.sh` pins).

### 2.1 The lookup

`hwts3e(ier, eps, rk, cxs, cws, n)` takes rk = zk · boxsize(ilev), κ in our notation
(`boxsize` is the box width: `pts_tree3d.f` halves `boxsize(0) = bs0` per level), and:

- `hwts3dgetd` finds the domain: Re rk on the breakpoints 0, 0.02, 0.04, 0.1, 0.2, 0.4,
  1, π/2, π, 2π, …, 16π (23 intervals), Im rk on the first 20 of them (19 intervals, to
  12π); idomain = ix + (iy − 1)·23. Outside, `ier = 4` and no nodes.
- `iprec` from eps: 0 for eps ≥ 5·10⁻³, then 1, 2, 3, 4 below 5·10⁻³, 5·10⁻⁴, 5·10⁻⁷,
  5·10⁻¹⁰ (comments: 2, 3, 6, 9, 12 digits; `errmax` 10⁻³ … 10⁻¹³ is set and unused).
- `hwts3p<iprec>` includes `hwts_3d_iprec<iprec>.txt`: per domain its rectangle (rk0,
  rk1), `iquadtype`, n, `err` and n real nodes xₖ and weights wₖ (6,443 to 18,983 nodes
  per precision).
- The map to (λₖ, wₖ): type 1, λ = √(x² − rk²), w x / λ (t = x real; 276 domains, all
  with Im rk ≥ π/2); type 2, u = 100 x, λ = u + Im rk − i Re rk / (1 + u),
  w·100·(1 + i Re rk / (1 + u)²) (161 domains, Im rk < π/2); type 5 is coded and unused.
- `hnumfour`, `hnumphys` (`hnumfour.f`, `hnumphys.f`, 67,000 lines each) give per
  precision and domain Fₖ and Nₖ; `hmkexps` builds tₖ = √(λₖ² + rk²) and αⱼ = 2πj/Nₖ.

The node counts grow with Re κ (iprec 3: 33 at κ ≈ 0, 44 at 3π, 92 at 15π) and fall with
Im κ: at iprec 3 from Im κ = 8π one node meets the precision, since |G| ≤ e^{−Im κ} on the region and
the error is absolute (`results/domains.md`, the node maps).

### 2.2 Which levels

`hfmm3d.f` (lines 750–770, 1050–1130): a level uses plane waves when Re(zk w) ≤ 16π,
Im(zk w) < 12π and it is finer than `ilevcutoff`, the finest level with
e^{−Im zk · w}/w < eps, where FMM3D drops the far field. Other levels use its rotation and
point-and-shoot M2L.

### 2.3 The multipole-to-wave factors

`hrlscini` forms the factors of the multipole's orders at each λₖ from scaled associated
Legendre functions of i λₖ / κ (`zylgndrsc`), the Helmholtz counterpart of the λⁿ of
CONVENTIONS §3.14; CH6.1 derives ours in the orthonormal basis of §3.15 and needs their
scaling at small κ.

### 2.4 Eight wavelengths, not 32

FMM3D's documentation (`docs/index.rst`: "in the low frequency regime (for box sizes up to
32 wavelengths)") and the proposal after it (§5.3, the §9.2 table, §11, §13) give 32
wavelengths. The code stops at Re(k w) ≤ 16π, w ≤ 8 λ, and the tables end there.

## 3. The tables' accuracy

`results/domains.md` (`examples/sweep.rs domains`, 54 s): every domain and precision at
9 points of its rectangle (corners, with the upper edges pulled in by 10⁻⁹ relative, edge
midpoints, centre).

| iprec | largest stated | λ sum / stated, worst | kernel / stated, worst | kernel, absolute | chain at the centre |
| --- | --- | --- | --- | --- | --- |
| 0 | 9.99e-4 | 1.24 | 69 | 3.9e-2 | 8.8e-2 |
| 1 | 9.94e-5 | 1.08 | 83 | 3.4e-3 | 9.5e-3 |
| 2 | 9.97e-8 | 1.19 | 94 | 3.8e-6 | 6.8e-6 |
| 3 | 1.00e-10 | 1.50 | 119 | 3.0e-9 | 6.3e-9 |
| 4 | 1.00e-13 | 5.51 | 93 | 3.6e-12 | 8.2e-12 |

### 3.1 The λ sums

They meet the stated error within 1.5× at iprec 0–3; the worst points are rectangle
corners. iprec 4's 5.5× is at Re κ ≈ 16π, 5·10⁻¹³ absolute, where the type-2 contour's
waves grow by e^{|Im t| ρ} and round-off sets the floor.

### 3.2 The α counts

With FMM3D's Nₖ the error peaks at u = (±4, ±4, 1), the corner of the region
(`examples/alpha.rs`):

| iprec, domain, κ | stated | Nₖ × 1 | × 1.25 | × 1.5 | × 2 | ρ ≤ 4 only |
| --- | --- | --- | --- | --- | --- | --- |
| 2, 88, 37.70 + 0.10i | 5.8e-8 | 3.84e-6 | 5.40e-8 | 5.40e-8 | 5.40e-8 | 5.40e-8 |
| 3, 85, 28.27 + 0.10i | 8.3e-11 | 3.00e-9 | 2.15e-10 | 1.02e-10 | 8.45e-11 | 1.19e-10 |
| 2, 34, 12.57 + 0.02i | 6.9e-8 | 1.39e-6 | 5.03e-7 | 4.13e-7 | 3.84e-7 | 3.76e-7 |
| 2, 11, 12.57 | 7.2e-8 | 1.12e-6 | 2.75e-7 | 1.14e-7 | 8.46e-8 | 1.99e-7 |

So FMM3D's counts resolve J₀(tρ) to about ρ = 4, not to the corner's 4√2; a V pair
reaches the corner only from its extreme lateral offsets (3 boxes across in both
directions and the nearest along). Whether FMM3D's chain, with its multipole and local
truncated to degree p on both sides, sees this error is CH6.2's comparison; the
point-to-point chain here does (the last column of the table above).

## 4. The generator

`src/generator.rs`, for any κ with Re κ, Im κ ≥ 0:

1. **Contour of real t**: λ = −iq, q = √(κ² − t²) with Im q ≥ 0 (outgoing), dλ = t dt/λ,
   t ∈ [0, t_max] with Re λ(t_max) = ln(1/ε) + 4. J₀ has a real argument and |e^{−λz}| ≤ 1,
   so no term grows (FMM3D's type 1; its type 2 contour lets |Im t| > 0). For Re κ > 0 the
   variables θ (t = Re κ sin θ) and φ (t = Re κ cosh φ) remove the 1/√ singularity at
   t = κ for real κ; for 0 < Im κ ≪ Re κ the panels are graded towards it.
2. **Panels**: 20-point Gauss–Legendre, the count doubled until the λ sum meets ε/16.
3. **Compression**: a column-pivoted Gram–Schmidt (twice orthogonalised) of the
   integrand at 25 × 65 (z, ρ) samples selects nodes; least squares refits their weights
   to the full rule's values; the check runs also at the 24 × 64 midpoints.
4. **Counts**: Nₖ the fewest with 2|wₖ| e^{−Re λₖ} max_{m ≥ N, x ≤ tₖ ρ_max} |J_m(x)| ≤
   ε/(4s), Fₖ the fewest with the same bound on the J_m, m > Fₖ, at x ≤ tₖ √2/2, and Nₖ
   at least the count for ρ_max − √2/2 plus Fₖ (so the chain's truncated orders do not
   alias).

Tests (`tests/generator.rs`): ε met on the grid, at 200 seeded points off it (within 2ε)
and in the chain (2ε), at κ = 0, 5, 20, 16π, 5i, 30i, 10 + 3i, 40 + 20i, 10 + 10⁻³i and
ε = 10⁻³, 10⁻⁶, 10⁻⁹ (the release-only test `meets_eps_everywhere`); every tₖ real; the
panels converged.

### 4.1 Against FMM3D on rays of κ

`results/rays.md` (`examples/sweep.rs rays`, 32 s): real κ ∈ [0.25, 49] (32 points),
imaginary κ ∈ i[0.25, 36] (24), κ = x(1 + 0.05i) (32) and x(1 + 0.5i) (24), at iprec 1–3,
the generator run at FMM3D's measured kernel error (at least 1 % of the precision's
`errmax` and 10⁻¹³). The generator met its target on every point, in the kernel and in
the chain. Σ Nₖ of the generator
over FMM3D's, mean (range), over the points where FMM3D has at least 8 nodes:

| ray | iprec 1 | iprec 2 | iprec 3 | points left out (FMM3D < 8 nodes) |
| --- | --- | --- | --- | --- |
| real κ | 1.86 (1.57–2.98) | 1.79 (1.59–2.44) | 1.82 (1.67–2.34) | none |
| κ (1 + 0.05i) | 1.50 (1.22–3.00) | 1.52 (1.24–2.58) | 1.55 (1.25–2.35) | none |
| κ (1 + 0.5i) | 3.78 (1.59–7.32) | 3.24 (1.65–6.61) | 2.45 (1.57–3.57) | 12, 4, 0 |
| imaginary κ | 3.88 (2.81–5.52) | 4.41 (2.36–9.61) | 3.90 (2.26–7.73) | 20, 14, 10 |

On the points left out FMM3D uses 1 to 7 nodes where |G| ≤ e^{−Im κ} is near or below
the precision, and the generator, which has no such shortcut, is up to 2,000× larger;
those rows are in the file. The chain's error is the generator's own ε or below at
every point, and below FMM3D's on every ray and precision: at iprec 2 the largest is
8.9·10⁻⁸ against 5.7·10⁻⁶ on the real ray, 6.9·10⁻⁸ against 4.0·10⁻⁶ on κ(1 + 0.5i).
(A first version stopped Fₖ at the first order with a small |J_{F+1}|, which below the
turning point can be a zero of J_{F+1}; its chain missed ε by up to 120×. The bound now
takes the maximum over all larger orders, and the four worst κ of that version are
regression cases in `tests/generator.rs`.) At iprec 4 the generator was not run: near 10⁻¹³ it took
minutes per κ, and D1's ε needs iprec ≤ 3.

## 5. The levels

`results/levels.md` (`examples/sweep.rs levels`, 17 s): root side D = 1, so
κ_l = kD/2^l, for real kD = 10, 100, 500, Yukawa κD = 10, 100, 500 and damped
kD = 100 + 5i, 500 + 25i, levels 2 to 8, ε = 10⁻³, 10⁻⁶, 10⁻⁸ (iprec 1, 2, 3).

- **Coverage.** Every level that is neither pruned nor at Re κ_l ≥ 16π falls in a
  domain: 138 of the 168 (case, level, ε) rows. The 11 outside are levels 2 and 3 at
  kD = 500 (κ = 125 and 62.5, boxes of 20 and 10 λ), real and damped; they are high
  frequency (proposal §3.6, H4), and the generator serves them if ever needed, at
  21,000 to 158,000 wave values per direction. 20 rows are pruned by §5.5's test
  (θ_D = 1): the coarse Yukawa levels at κD = 100 and 500. The unpruned Yukawa levels
  reach Im κ_l = 15.6 (κD = 500, level 5, ε = 10⁻⁸), well inside Im κ < 12π, so Yukawa
  is pruned before it reaches the tables' upper edge; FMM3D's own cut drops only levels
  that §5.5 prunes too.
- **Accuracy.** With FMM3D's α counts the kernel's error exceeds ε on 11 of the 138
  rows, by at most 1.8× (1.8·10⁻³ at ε = 10⁻³ on real kD = 100, level 5): all at ε =
  10⁻³ or 10⁻⁶, all §3.2's corner. At ε = 10⁻⁸ iprec 3 gives 10⁻¹⁰ to 10⁻⁹, 10 to 100
  times below ε: FMM3D's precisions are three digits apart.
- **The generator at ε** needs 1.92× FMM3D's Σ Nₖ on the in-table rows (1.28–3.61),
  less than §4.1's ratios at ε = 10⁻⁸, where it may use the slack FMM3D's precision step
  leaves.

## 6. Recommendation for CH6.1

1. **FMM3D's tables**, iprec 1 to 3 (D1's ε from 10⁻³ to 10⁻⁸; about 32,600 nodes and
   their counts), extracted into `nd-fmm-tables` with NIST's notice kept and a script
   from FMM3D's pinned release, as CONVENTIONS §3.14's Laplace tables were. They are the
   cheapest source by the measurements here, and they cover every level CH6.1 targets.
   The choice per level is `hwts3e`'s: the domain of κ_l and a precision from ε, a
   function of k, w_l and ε only (constraint 9).
2. **The α counts by measurement**, not from `hnumphys` as they are: FMM3D's misses its
   stated error at the region's corner by up to 119×, and up to 1.8× above ε on the
   reference levels. Either Nₖ scaled by 1.25 to 2 until the region meets the precision
   (as §3.14 chose its Laplace counts), or this spike's Bessel bound applied to the
   table's nodes. CH6.1's criterion (errors within about 1.1× Rotation's, end to end) is
   what decides; the precision may then sit one step tighter than `hwts3e`'s at
   ε = 10⁻³ and 10⁻⁶.
3. **The generator** (`src/generator.rs`) as the trusted path of CH6.1's tests (it is
   independent of FMM3D's tables and meets ε by construction), and as the source for any
   level outside the tables, should one ever want the exponential M2L there.
4. **Correct the proposal**: FMM3D's plane waves stop at boxes of 8 wavelengths
   (§2.4).

## 7. Reproducing

```sh
export FMM3D_ROOT=…                      # as for tools/fmm3d/build.sh (fetch first)
export FMM3D_HELMHOLTZ=$(spikes/pw-quadrature/tools/fetch.sh)
cargo test -p nd-fmm-spike-pw-quadrature                         # fixture, generator
cargo test --release -p nd-fmm-spike-pw-quadrature -- --ignored # full tables, all κ
cargo run --release -p nd-fmm-spike-pw-quadrature --example sweep -- "$FMM3D_HELMHOLTZ" all
cargo run --release -p nd-fmm-spike-pw-quadrature --example alpha -- "$FMM3D_HELMHOLTZ"
```

The fixture (`fixtures/`, domains 1, 11, 139, 172 and 236 at iprec 2 and 3) is cut from
FMM3D's files by `tools/extract_fixture.sh`, with NIST's notice kept.
