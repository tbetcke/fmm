# CH0.2 S2: coaxial coefficients

Started 2026-10-27 on `tbetcke/h-ch0-2`. Roadmap: "CH0.2 S2: coaxial coefficients" (H0).
Proposal: §10 CH0.2; §3.1 ("Which quantities are real"), §3.4 (scaling), §3.5 (p), §5.2,
§5.3 (Rotation), §4 item 6, §8 R2.

## Goal

Show that the coaxial translation coefficients (S|R), (S|S) and (R|R) in the real
orthonormal basis can be computed in f64 by the Gumerov–Duraiswami recursions to p and
k a of 300 and beyond, for real, complex and imaginary k, how accurately, in which
direction and normalisation, and which symmetries halve the tables.

## Acceptance (proposal §10, verbatim)

> relative error against an arbitrary-precision reference for p up to 300 and k a up to
> 300, real, complex and imaginary k; the imaginary parts of (S|S) and (R|R) zero to
> round-off for real k; the index symmetry used to halve the tables

## Read first

- Proposal §3.1 (the expansion with ik and real orthonormal Yₙᵐ; (S|S) = (R|R); which
  operators are real), §3.4 (FMM3D's scale s_l = min(|k| a_l, 1); the exponential factor
  for Yukawa), §3.5 (p_l from Gumerov–Duraiswami 2009 eq. 18), §5.2, §5.3, §8 R2.
- CONVENTIONS §3.3 (Racah harmonics, no Condon–Shortley phase), §3.6 (real storage: cos
  in slot +m, sin in −m).
- `spikes/rotation-blocks/` (CH0.1): its basis (the real storage of N Rₙ, orthonormal up
  to a factor per degree) and its reference tool, which this spike follows.
- Gumerov and Duraiswami, SIAM J. Sci. Comput. 25(4) (2003) and their 2004 book, §3.2:
  the coaxial recursions and (S|S) = (R|R).

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- The basis: real orthonormal Yₙᵐ = √((2n + 1)/(4π)) times CH0.1's slot normalisation
  (the degree factor matters for translations, not for rotations; it is the basis of
  proposal §3.1, so that Σₘ Yₙᵐ(x̂) Yₙᵐ(ŷ) = (2n + 1)/(4π) Pₙ(x̂ · ŷ)). Fₙᵐ(r) = fₙ(k r)
  Yₙᵐ(r̂) with f = j (regular, R) or h = h⁽¹⁾ (singular, S). A coaxial translation by
  t = t ẑ: Fₙᵐ(r + t) = Σ_{n′} (E|F)ᵐ_{n′n}(t) G_{n′}ᵐ(r), the same coefficient for the
  slots +m and −m. §3.15 is not written (CH1.1); the report states the convention used.
- `spikes/coaxial/` (new, package `nd-fmm-spike-coaxial`), a workspace member outside
  `default-members` like CH0.1's spike; dev-dependency `serde_json` (in
  `[workspace.dependencies]` already); no new dependency.
  - `src/complex.rs`: a minimal complex f64 type (the workspace has no complex crate the
    spike may use without a new dependency).
  - `src/radial.rs`: spike-grade jₗ and hₗ for complex z to l of about 750, normalised by
    the common exponential factor (jₗ e^{−|Im z|}, hₗ e^{Im z}) and with an optional power
    scale gˡ: hₗ upward, jₗ by Miller's downward recursion with periodic rescaling,
    normalised against j₀ and j₁. (S3, CH0.3, does the radial functions properly; the
    spike also runs the recursions from the reference's start values to separate the two
    sources of error.)
  - `src/coaxial.rs`: the coefficients for orders 0 ≤ m ≤ min(p_src, p_tgt) and the
    rectangular truncation p_src → p_tgt, from the start column
    (E|F)⁰_{n′0} = (−1)^{n′} √(2n′ + 1) f_{n′}(k t) by the recursion in m on the sectorial
    column and the recursion in n per order (derived from ∂_z and ∂_x + i∂_y commuting
    with the translation), in two variants: the full rectangle, and the lower triangle
    n′ ≥ n filled by the index symmetry (E|F)ᵐ_{nn′} = (−1)^{n+n′} (E|F)ᵐ_{n′n}. (S|R) in
    the symmetric scaled form s^{n+n′} (E|F) with s = min(|k| a, 1), the FMM3D scale at
    both ends of an M2L; (R|R) = (S|S) unscaled. Complex arithmetic throughout.
  - `tools/reference.py` (PEP 723, `mpmath==1.4.1` as in CH0.1): the coefficients from the
    Gaunt sum (E|F)ᵐ_{n′n} = Σₗ i^{n′−n+l} (2l + 1) √((2n + 1)(2n′ + 1))
    (n n′ l; 0 0 0)(n n′ l; m −m 0) (−1)ᵐ fₗ(k t), with the 3j symbols by their three-term
    recursion in l from the closed form at l = n + n′, and fₗ from mpmath's Bessel
    functions; each value at two precisions 50 digits apart, required to agree to 10⁻²⁵.
    First, at n ≤ 6, the formula against the definition: the addition theorems evaluated
    pointwise in mpmath, (R|R) at any r, (S|R) for |r| < t, (S|S) = (R|R) for |r| > t,
    for every slot −n..n. Writes `fixtures/reference.json` (committed, under 1 MB): sampled
    entries (rows and columns at both ends, the diagonal, random entries) for orders
    m ∈ {0, 1, mid, max} of each configuration below. (As built: 21,809 entries in 120
    configurations, 1.2 MB, over the 1 MB first planned.)
  - `tests/accuracy.rs`, `examples/{sweep,time}.rs`, `SPIKE_REPORT.md`.
  - Added during implementation, once the recursion proved unstable above k a ≈ 50
    (SPIKE_REPORT.md §4.3): `src/projection.rs`, the coefficients by projection on a
    sphere (FMM3D's stabilised point and shoot; Gauss–Legendre in the polar angle, least
    squares from value and radial derivative for (S|R), (S|S) = (R|R) on an outer
    sphere); `src/metric.rs`, the FMM-weighted error (§4.1 of the report);
    `tools/full_table.py`, the whole recursion table in f64 against the same recursion
    in arbitrary precision, because the fixture's samples miss the unstable entries.
- Root `Cargo.toml` `members` gains `spikes/coaxial`; `Cargo.lock` gains its entry.

Configurations (a the circumradius of a box, a = √3 w/2, as in proposal §3.5's k a_l):
k a ∈ {0.01, 0.1, 1, 3, 10, 30, 100, 300}; k real, complex (k = κ(1 + iη), η ∈ {0.1, 1},
k a meaning κ a) and imaginary (k = iκ); p = p(k a, ε) of §3.5 for ε ∈ {10⁻³, 10⁻¹⁰}
(p from 12 to about 370); M2M and L2L at the child shift k t = k a/2 with the rectangular
p_child = p(k a/2) → p_parent = p(k a); M2L at the 15 V-list distances d w, d from 2 to
√27 (the reference samples d = 2, √12 and √27; the f64 sweep all 15).

## Tests that prove it

All in f64 against the reference, one thread,
`cargo test -p nd-fmm-spike-coaxial --release -- --include-ignored`.

1. **Reference, p and k a to 300**: every sampled entry, by the recursion (from the
   spike's own start values, checked against the reference's to 10⁻¹³, test 1a) and by
   the projection. As built, the metric is the FMM-weighted error of SPIKE_REPORT.md
   §4.1, bound 10⁻¹¹: the row-relative error first planned is meaningless where a row
   spans tens of orders of magnitude, and the per-entry error is unreachable at large p
   by any method (report §4.2); it is reported. The projection is asserted on every
   configuration, the recursion where the full-table sweep finds it stable. The
   full-table claims rest on `examples/sweep.rs` and `tools/full_table.py`.
2. **Real k**: the f64 (R|R) has imaginary parts exactly zero (the recursion closes over
   the reals); the reference's (R|R) and (S|S) imaginary parts below 10^{−(digits − 10)};
   Re (S|R) = (R|R) in f64 (h = j + i y, and the Gaunt phases are ±1 for real k), a
   non-trivial check of the claim; as built in the M2L weights to 10⁻¹¹, by the
   recursion to k a = 30 and by the projection at k a = 100 and 300.
3. **Imaginary k**: i^{n−n′} (E|F)ᵐ_{n′n} real to round-off (the powers of i absorbed per
   degree, proposal §3.1), in the reference and in f64.
4. **Index symmetry**: in the reference (E|F)ᵐ_{nn′} = (−1)^{n+n′} (E|F)ᵐ_{n′n} to
   10⁻³⁰; the slots ±m share one coefficient (the definition check); in f64 the triangle
   variant against the full rectangle where both are accurate; the reflection
   (E|F)(−t) = (−1)^{n+n′} (E|F)(t), which serves the β and π − β shifts of M2M.
5. **Rectangular truncation**: the table p_src → p_tgt equals the corresponding block of
   the square table at max(p_src, p_tgt) bit for bit.
6. **Definition** (in the reference tool, before it writes the fixture): the addition
   theorems at n ≤ 6 to 10⁻⁴⁰.

As built (after the self-review): the weights of the FMM-weighted error are the envelope
max_{r ≤ a} |jₙ(k r)|, not |jₙ(k a)|, which is no bound for n < |k| a. Added: test 6, the
whole recursion table against the projection at the proposed threshold k a = 40, every
kind of k, M2M and all 15 distances, bound 10⁻¹¹ (the fixture has nothing between k a = 30
and 100, and its samples miss the unstable entries); in test 2 the outer-sphere
projection's imaginary parts, the f64 reality check that can fail (the recursion's zeros
are structural, and Re (S|R) = (R|R) only checks the start values); in test 1 a check
that the normalising maximum agrees between projection and recursion; in test 4 the
weighted error of the full variant (negligible) and assertions. Dropped as separate
checks, covered otherwise: the reference's explicit symmetry check to 10⁻³⁰ and its
imaginary parts of (R|R) and (S|S) (real by the formula; the definition check covers
both at n ≤ 6, and test 1 reads 8,375 upper-triangle samples through the symmetry).
`examples/nodes.rs` records the quadrature convergence of the projection.

## Checks

`/verify` (no default member changes: format, Clippy of the workspace with all targets,
the workspace tests, the spike's tests in release with `--include-ignored`);
`uv run spikes/coaxial/tools/reference.py --check` regenerates the fixture byte for byte.

## Machine runs

None required. Times per coaxial table set on the M3 Max (release, one thread) for the
configurations above: one (S|S) table per level, one (S|R) table per (level, distance);
these feed §7's table-build cost and CH2.2.

## Constraints and risks

- Constraint 1 (Laplace): untouched; only the root `Cargo.toml` (`members`) and
  `Cargo.lock` (the spike's entry) are shared files.
- §3.15 is not written (CH1.4, CH1.1); the spike fixes the basis and normalisation as
  above, and the scale as FMM3D's, and says so. The scaling for Yukawa above κa ≈ 1 is
  S3's decision; here a common exponential factor per table is removed, which a linear
  recursion carries unchanged.
- Risk: the reference's cost (3j recursions and Bessel functions of order to about 750 at
  hundreds of digits); sampled entries only, cached per argument.
- Risk: where the unscaled (R|R) start values underflow (very low frequency, large p),
  the entries they feed are negligible; the report quantifies it.

## Pending runs

None.

## Open questions

For sign-off in the PR (SPIKE_REPORT.md §8):

1. Build the coaxial tables by the recursion (triangle, index symmetry) where
   Re(k) a ≤ 40 and by projection above; or projection everywhere.
2. Record CH0.2's acceptance in the FMM-weighted error (bound 10⁻¹¹) instead of the
   per-entry relative error, which no f64 method reaches at large p.
3. Replace R2's fallback "through signatures" by the projection; carry the M2L coaxial
   table memory (132 MiB per table at p = 371, 15 per level) and the table-free
   projection into §7 and CH0.8.
4. What §3.15 item 6 states (symmetries, reality, the Yukawa powers of i).
