# CH1.1 CONVENTIONS §3.15

Started 2026-10-29 on `tbetcke/h-ch1-1`. Roadmap: "CH1.1 CONVENTIONS §3.15" (H1; third of
the priority list). Proposal: §10 CH1.1; §4 (the nine items); §1.1, §3.1, §3.3, §3.4 and
§5.5 (the kernel, expansions, storage, scaling, Yukawa); §2 constraint 1.

## Goal

Turn the draft `docs/helmholtz/conventions-3.15-draft.md` (CH1.4) into
`docs/CONVENTIONS.md` §3.15 with a `CONVENTION_VERSION` bump, the choices the draft left to
CH1.1 made, for the user's sign-off.

## Acceptance (proposal §10, verbatim)

> signed off by the user, after the two-point check of CH1.4

Decisions: D5 signed off 2026-10-29 (charges now, dipoles later: §3.15 fixes real and
complex charges with potentials and gradients only). D2 signed off 2026-10-29 (A), for
CH1.2.

## Read first

- The draft §3.15 and the CH1.4 report §5 (sign-off 1: the draft is the starting point; the
  factor ik left to CH1.1).
- `docs/CONVENTIONS.md` §3.1, §3.3, §3.6–§3.10, §3.13 (its style; the version rule).
- The S1–S3 and S7 sign-offs (progress log): S3's scale (circumradius, exponential
  always); S7's question 3 (L2L by recursion; S2's projection for M2L and M2M only).
- `fmm-math/src/lib.rs` (`CONVENTION_VERSION`), `fmm-math/tests/convention_version.rs`;
  the fixture generators that write the version into their headers
  (`tools/fixtures/gen_harmonics.py`, `gen_dipoles.py`, `gen_helmholtz_fmm3d.py`).

## Changes

- `docs/CONVENTIONS.md`: a new §3.15 after §3.14, from the draft, items 1–9 plus
  versioning; the header and §3.10 name §3.15 among the versioned sections; version 3
  with its history line. The draft file is reduced to a pointer to §3.15.
- Choices CH1.1 makes (questions 1–3 below, asked before writing):
  1. where the factor 4π ik (Yukawa 4πκ) goes;
  2. whether the stored coefficients carry a per-box exponential factor for Im k > 0;
  3. the version bump.
- Scope by D5: charges only; dipoles named as excluded until a later task.
- Items still open (signatures, item 7; f32, item 8) are written as reserved, each with
  the task that fixes it and the rule that fixing it bumps the version.
- `fmm-math/src/lib.rs`: `CONVENTION_VERSION = 3`.
- Fixtures regenerated for the header only: `fmm-math/fixtures/harmonics_{A,B,C}.json`,
  `fmm-ref/fixtures/dipoles.json`, `fmm-ref/fixtures/helmholtz_fmm3d.json`.
- A numerical check of §3.15's identities, `tools/conventions/check_3_15.py` (mpmath, run
  by hand): the addition theorem with the chosen factor placement, the k → 0 limit,
  Yukawa's real form and the relations M^Y = i^{−n} M and L^Y = −iⁿ L, the scaled
  coefficients and the per-box factor, the real orthonormal harmonics against Racah's
  (§3.3, through N), the β and π − β blocks at low degree.
- No public API, no table, no Laplace code path. Conventions: this task is the proposal.

## Tests that prove it

- `fmm-math/tests/convention_version.rs`: the constant against the file (both 3).
- The fixture tests that read the headers (`fmm-math` harmonics, `fmm-ref` dipoles) at
  version 3; each generator's `--check` (byte for byte for harmonics and dipoles; the
  FMM3D one to 10⁻¹³ and byte for byte on the M3 Max, which wrote it).
- `tools/conventions/check_3_15.py`: every identity to 10⁻²⁵ or better at 40 digits, on
  random points, n ≤ 20, k real, complex and imaginary.
- Laplace bit for bit: no Laplace code changes; only the cache key's version changes
  (cached tables and tune files are rebuilt once), checked by the default and workspace
  tests.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`, the
default and workspace tests, the licence check); the three generators' `--check`; the
check script.

## Machine runs

None.

## Constraints and risks

- Constraint 1: no Laplace path changes. The version bump changes the cache file names and
  rejects old cache and tune files (format unchanged, contents rebuilt identically): a
  one-off rebuild on each machine, as for version 2 (Phase 6 T8).
- The section is a contract for CH1.2 onward: a wrong factor or scaling costs a later
  version bump. The check script is the guard.

## Pending runs

None.

## Open questions

Asked at planning, 2026-10-29:

1. The factor 4π ik: on the crossing operators M2L (in its (S|R) table), M2P and P2L
   (recommended), or P2P computing the series' normalisation and `nd-fmm-exec` applying
   ik once.
2. The stored coefficients for Im k > 0: a per-box factor, M̂ = e^{−Im(k) a_l} M / s_lⁿ
   and L̂ = e^{Im(k) a_l} L s_lⁿ (recommended), or none (M̂ = M / s_lⁿ, L̂ = L s_lⁿ), or
   left to CH1.2 with a second bump.
3. The version: `CONVENTION_VERSION` 3 for §3.15 (the proposal), or a version of §3.15
   of its own that leaves Laplace's caches alone.
