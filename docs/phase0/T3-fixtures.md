# Phase 0 / T3 — high-precision fixtures for solid harmonics

Read first: docs/CONVENTIONS.md §3.3, §3.6, §3.9; tools/fixtures/crosscheck_scipy.py.

Do:
- tools/fixtures/gen_harmonics.py using mpmath at 40 digits. Compute P_n^m from the
  definition in §3.3 WITHOUT the Condon–Shortley phase. If you use mpmath.legenp,
  assert its sign convention first (P_1^1(t) must equal +sqrt(1 - t^2)) and correct it.
- Output, in real storage order (§3.6), R and I and their gradients (gradients by
  high-precision differentiation) at seeded random points:
  regular at |u| in [0, sqrt(3)], irregular at |v| in [2, 8];
  set A: 10 points, p = 30; set B: 50 points, p = 8.
- Write JSON to fmm-math/fixtures/, values as decimal strings with 17 significant
  digits; include the seed, p, CONVENTION_VERSION and generator version in the header.
- tools/fixtures/README.md: how to regenerate; pinned mpmath version.
- Keep crosscheck_scipy.py working as an independent double-precision second opinion;
  it must agree with the mpmath fixtures to about 1e-13.

Must pass: regeneration is byte-identical; total fixture size under 3 MB (raised from
2 MB: the point counts above at 17 significant digits need about 2.8 MB).

Do not: write any Rust in this task.
