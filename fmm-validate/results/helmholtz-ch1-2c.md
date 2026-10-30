# Helmholtz CH1.2c: coaxial coefficients

Helmholtz task CH1.2c ([docs/helmholtz/tasks/CH1.2c.md](../../docs/helmholtz/tasks/CH1.2c.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH1.2, §5.2, §5.3 and R2; CONVENTIONS §3.15, "Coaxial translations". Measured on
2026-10-30 from branch `tbetcke/h-ch1-2c` on the M3 Max, release builds, rustc 1.99.0,
f64, one thread. The machine was shared: the load average was about 40 during the
timings, so the times of §4 are an upper bound.

## Summary

- **`nd_fmm_math::coaxial`** holds the coaxial coefficients (R|R) = (S|S) and (S|R) of
  §3.15 in their stored normalisation. A `CoaxialTable` stores, per order, the triangle
  n′ ≥ n of degrees max(p_src, p_tgt) by min(p_src, p_tgt), and reads the rest through the
  index symmetry. One stored table therefore serves both orientations (`reversed`). The
  generators are S2's recursion (start column from `radial`, triangle closed under both
  recursions) and S2's three projections. `m2l`, `m2m` and `l2l` choose between them by
  §3.15's rule: the recursion where Re(k) a ≤ 40 (a the circumradius of the M2L's level
  or of the shift's parent), else the projection on S2's spheres. Above the threshold `l2l` reads
  M2M's outer-projection table reversed (question 1 of §5, signed off 2026-10-30); the
  inner projection is its independent check.
- **Against S2's mpmath fixture** (120 sets, k a to 300, p to 371, every kind of k), the
  rule's tables meet the FMM-weighted bound 10⁻¹¹ with **6.8·10⁻¹³** at worst
  (Im k = Re k, k a = 300, M2M). The recursion meets it with 8.0·10⁻¹⁵ wherever S2 found it
  stable, and per entry within 8.1·10⁻¹⁴ at k a ≤ 3 (1.4·10⁻¹³ on CI's Linux, whose
  glibc differs in the last bit; the test's bound is 10⁻¹², signed off 2026-10-30).
- **The identities of §3.15.** The addition theorems hold pointwise to 5.9·10⁻¹⁵ for
  (R|R), (S|R) and (S|S), every slot of n ≤ 6, real, complex and imaginary k, with the
  reflection t → −t and the entries read through the index symmetry. For real k the M2M
  and L2L coefficients are real: exactly by the recursion, to 4.1·10⁻¹⁴ in the shift
  weights by the outer projection, and measured exactly by the inner one. For Yukawa,
  i^{n−n′} (E|F) is exactly real at κ a = 1, 30 and 300. A rectangular table is the block
  of the square one bit for bit, for every generator.
- **The L2L question of S7 is answered** (§2). S7 found that S2's outer projection fails
  for L2L with damped k. That failure came from computing the L2L orientation by
  projection: target degrees low, source degrees high, which projects high-degree waves
  on the outer sphere. Computed in M2M's orientation and read through the index symmetry,
  the same outer projection agrees with the independent inner projection within
  **6.8·10⁻¹³** in L2L's weights. This holds at every k a from 40 to 300 and for every kind
  of k, weakly damped k included. Both projections therefore serve L2L, and question 1
  asks whether one table may serve M2M and L2L.

## 1. Method

- **The metric** is S2's FMM-weighted error (SPIKE_REPORT §4.1): max |ΔÊ| w / max |Ê| w
  over a table. The M2L weights are uₙ u_{n′} with uₙ = Jₙ(a) √(2n + 1)/sⁿ. For the shifts,
  a child degree n weighs Jₙ(t) √(2n + 1) and a parent degree n weighs
  |hₙ(k 8t/√3)| √(2n + 1), with t = a_child. The weights go with the degrees, so M2M and L2L
  read the same weights in opposite orientations. Per-entry relative error is no measure
  at large p (S2 §4.2).
- **The fixture** `fmm-math/fixtures/helmholtz_coaxial.json`, from
  `tools/fixtures/gen_helmholtz_coaxial.py`, holds S2's 120 sets and 21,809 entries value
  for value (checked against `spikes/coaxial/fixtures/reference.json`). It adds the
  fmm-math header and each set's k, t, a and kind. `--check` regenerates it byte for byte.
- **Bit-exact rectangular tables.** Miller's start order for jₙ grows with the requested
  length (CH1.2b). Every regular radial column is therefore evaluated to the square
  table's length: the recursion's start column to 2 max(p_src, p_tgt), and the inner
  projections' target functions to max(p_src, p_tgt). hₙ does not depend on the length.
- **Projections** fill the triangle only from the half they compute: the outer one from
  target degree ≥ source degree, the inner ones from target ≤ source. Each point value
  carries a factor that takes the exponential of the scaled radial functions to a
  constant, chosen so that no applied exponent is positive. Nodes:
  max(p_src, p_tgt) + 40, plus ⌈|k| t⌉ for the inner (R|R) on the child's sphere ρ = t.

## 2. L2L above the threshold (test 8)

Whole tables, L2L orientation (p_parent → p_child), L2L weights, t = a_child = 0.5,
ε = 10⁻¹⁰ (`tests/coaxial/high_frequency.rs`, `l2l_tables_above_the_threshold`). The
columns are the recursion against the outer projection (computed in M2M's orientation,
read reversed) and against the inner projection, the two projections against each other,
and the inner projection against itself with 60 more nodes:

| kind | k a | p_P → p_c | rec − outer | rec − inner | outer − inner | inner, +60 nodes |
| --- | --- | --- | --- | --- | --- | --- |
| real | 40 | 77 → 51 | 3.3e-14 | 3.3e-14 | 8.6e-15 | 4.1e-15 |
| Im k = 0.1 Re k | 40 | 77 → 51 | 6.2e-14 | 6.0e-14 | 3.1e-14 | 3.7e-14 |
| Im k = Re k | 40 | 77 → 51 | 3.6e-14 | 1.6e-14 | 3.6e-14 | 3.6e-14 |
| imaginary | 40 | 77 → 51 | 2.9e-14 | 1.5e-14 | 3.4e-14 | 1.4e-14 |
| real | 64 | 106 → 67 | 1.7e-12 | 1.7e-12 | 1.6e-14 | 2.4e-15 |
| Im k = 0.1 Re k | 64 | 106 → 67 | 1.3e-12 | 1.3e-12 | 8.4e-14 | 1.5e-14 |
| Im k = Re k | 64 | 106 → 67 | 1.1e-13 | 1.9e-14 | 1.1e-13 | 3.7e-14 |
| imaginary | 64 | 106 → 67 | 9.9e-14 | 2.3e-15 | 9.9e-14 | 3.4e-15 |
| real | 100 | 149 → 89 | 2.3e-9 | 2.3e-9 | 4.1e-14 | 1.2e-14 |
| Im k = 0.1 Re k | 100 | 149 → 89 | 9.1e-11 | 9.1e-11 | 3.4e-13 | 1.3e-14 |
| Im k = Re k | 100 | 149 → 89 | 4.7e-13 | 5.4e-14 | 4.7e-13 | 7.8e-14 |
| imaginary | 100 | 149 → 89 | 4.0e-13 | 9.0e-15 | 4.0e-13 | 1.4e-14 |
| real | 128 | 181 → 106 | 8.1e-7 | 8.1e-7 | 1.8e-14 | 4.6e-14 |
| Im k = 0.1 Re k | 128 | 181 → 106 | 3.5e-9 | 3.5e-9 | 1.4e-13 | 1.2e-13 |
| Im k = Re k | 128 | 181 → 106 | 1.5e-13 | 7.4e-15 | 1.5e-13 | 4.9e-14 |
| imaginary | 128 | 181 → 106 | 1.4e-13 | 4.9e-14 | 1.4e-13 | 1.1e-13 |
| real | 200 | 262 → 149 | 0.91 | 0.91 | 4.1e-14 | 1.9e-14 |
| Im k = 0.1 Re k | 200 | 262 → 149 | 3.1e-5 | 3.1e-5 | 2.6e-13 | 6.1e-14 |
| Im k = Re k | 200 | 262 → 149 | 4.5e-13 | 8.5e-14 | 4.5e-13 | 8.0e-14 |
| imaginary | 200 | 262 → 149 | 4.1e-13 | 3.7e-14 | 4.1e-13 | 1.4e-14 |
| real | 300 | 371 → 206 | 1.0 | 1.0 | 6.3e-14 | 7.4e-14 |
| Im k = 0.1 Re k | 300 | 371 → 206 | 1.0 | 1.0 | 6.1e-13 | 1.6e-13 |
| Im k = Re k | 300 | 371 → 206 | 6.8e-13 | 3.4e-13 | 6.8e-13 | 1.1e-13 |
| imaginary | 300 | 371 → 206 | 5.2e-13 | 7.1e-14 | 5.2e-13 | 1.3e-13 |

- The recursion fails for real and weakly damped k above the threshold, as S2 found for
  M2M: 1.7·10⁻¹² at k a = 64, 2.3·10⁻⁹ at 100 and O(1) from 200. It stays at the
  projections' floor for Im k = Re k and Yukawa. §3.15's threshold of 40 on the parent's
  k a holds for L2L too: at k a = 40 every method agrees within 6.2·10⁻¹⁴.
- The two projections are independent methods: different spheres, different radial
  functions (h onto h against j onto j), and opposite halves of the table. They agree
  within 6.8·10⁻¹³ everywhere. The inner projection is converged in its nodes (≤ 1.6·10⁻¹³).
- Test 8 asserts, at 10⁻¹¹: outer against inner everywhere, the inner projection's node
  convergence, and the recursion against the inner projection where S2 found it stable.

## 3. The other whole-table tests

- **Test 7, the threshold** (S2's test 6): at Re(k) a = 40 (p 51 → 77), every kind of k,
  M2M (outer), L2L (inner) and M2L at all 15 distances, the recursion against the
  projection: 6.5·10⁻¹⁴ at worst (real k, M2L d² = 4).
- **Test 3, the projections for real k**: the imaginary parts of the outer (S|S) are
  3.9·10⁻¹⁴ (k a = 100) and 4.1·10⁻¹⁴ (300) in the shift weights, read as M2M and as L2L.
  The inner (R|R)'s are measured exactly zero (the test asserts 10⁻¹¹): jₙ of a real argument
  stays in real arithmetic.
- **Test 1 at full size**: the rule's tables against the fixture, at most 6.8·10⁻¹³
  (Im k = Re k, k a = 300, M2M, by the projection); 5.9·10⁻¹³ for Im k = 0.1 Re k at
  k a = 300, where the recursion reaches 2.3·10⁻⁶, as S2 found.

## 4. Time and size per table

One thread, real k, ε = 10⁻¹⁰, best of three
(`cargo run -p nd-fmm-math --release --example coaxial_time`), load average about 40:

| k a | p_child → p | M2M recursion | M2M outer | L2L inner | M2L recursion | M2L projection | M2L table |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 35 → 35 | 0.06 ms | 0.8 ms | 1.4 ms | 0.06 ms | 1.3 ms | 0.1 MiB |
| 10 | 37 → 40 | 0.08 ms | 1.1 ms | 1.9 ms | 0.08 ms | 1.9 ms | 0.2 MiB |
| 30 | 46 → 64 | 0.19 ms | 3.4 ms | 6.2 ms | 0.30 ms | 7.0 ms | 0.7 MiB |
| 50 | 58 → 89 | 0.41 ms | 8.1 ms | 17.0 ms | 0.76 ms | 20.5 ms | 1.9 MiB |
| 100 | 89 → 149 | 1.57 ms | 39.5 ms | 103.6 ms | 3.43 ms | 130.1 ms | 8.8 MiB |
| 200 | 149 → 262 | 7.87 ms | 303.0 ms | 798.0 ms | 18.60 ms | 1014.2 ms | 46.8 MiB |
| 300 | 206 → 371 | 25.43 ms | 1124.2 ms | 2891.1 ms | 61.59 ms | 3838.1 ms | 132.0 MiB |

The times are within S2's §7 (85 ms and 5.1 s for the p = 371 M2L by recursion and
projection), the projections now computing half a table. Above the threshold the L2L
inner projection costs about 2.6 times the M2M outer projection, because it has more
nodes and its least squares double the accumulation. With one shared table (question
1, signed off) a level above k a = 40 builds only the outer one. The (S|R) table of degree p holds about p³/6 complex values.

## 5. Questions for sign-off

Both signed off on 2026-10-30, "1 and 2 as recommended": `CoaxialTable::l2l` reads
`m2m`'s table reversed, and §3.15's generator text is amended (no version bump).

1. **One shift table for M2M and L2L above the threshold.** §3.15 (S7's question 3)
   builds L2L's (R|R) by the recursion where it is stable and tests an inner-sphere
   projection for the rest. This task does that, and the inner projection meets the bound.
   §2 also shows that M2M's outer-projection table, read reversed, is accurate in L2L's
   weights. One table per level pair would then serve both translations: one build
   instead of two, at about 40 % of the inner projection's cost, and half the shift-table
   memory. *Recommended*: allow it, with §3.15's generator text amended to "(R|R) = (S|S)
   by the outer projection, computed in M2M's orientation and read through the index
   symmetry, for both M2M and L2L; the inner projection as the independent check". The
   code changes only `CoaxialTable::l2l`. *Alternative*: keep the rule signed off for S7
   (the inner projection for L2L).
2. **§3.15's generator text says "the projection on an outer sphere above that, for (S|R)
   and (S|S)"**, but S2's (S|R) projection is on the target box's sphere ρ = a < t, an
   inner sphere. This task implements S2's (S|R) projection. *Recommended*: correct the
   wording together with question 1. No stored quantity changes, so
   `CONVENTION_VERSION` stays.

## 6. Files

- `fmm-math/src/coaxial/` (`mod.rs`: the table, the rule; `recursion.rs`;
  `projection.rs`), `fmm-math/src/lib.rs`, `fmm-math/src/radial.rs` (Smith's division made
  crate-visible), `fmm-math/CLAUDE.md`.
- `fmm-math/tests/coaxial/` (`fixtures.rs`: test 1; `identities.rs`: tests 2–6, 9, 10;
  `high_frequency.rs`: tests 3, 7, 8, ignored). Run all of them with
  `cargo test -p nd-fmm-math --release --test coaxial -- --include-ignored` (about 85 s).
- `fmm-math/examples/coaxial_time.rs` (§4).
- `tools/fixtures/gen_helmholtz_coaxial.py`, `fmm-math/fixtures/helmholtz_coaxial.json`
  (1.2 MB), `tools/fixtures/README.md`.
