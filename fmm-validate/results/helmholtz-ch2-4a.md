# Helmholtz CH2.4a: the vectorised exponential

Helmholtz task CH2.4a ([docs/helmholtz/tasks/CH2.4a.md](../../docs/helmholtz/tasks/CH2.4a.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH2.4, §2 (constraint 1), §5.7; design
[docs/design/simd-p2p.md](../../docs/design/simd-p2p.md) §4.9. Measured on 2026-10-31 from
branch `tbetcke/h-ch2-4a`, release builds, rustc 1.99.0, on the M3 Max (NEON, scalar) and on
Kathleen (jobs 615552, node-c11a-033, and 615691, node-c11a-135, both Intel Xeon Gold 6248:
scalar, AVX2, AVX-512).

## Summary

- **`arch::exp::exp`** computes e^x for x ≤ 0 on the vector layer of every ISA, and
  **`nd_fmm_simd::exp::exp_slice`** exposes it for tests and reports. It is the e^{−κr} of
  CH2.4b's Yukawa P2P. The formulation: clamp to x_min, n = round(x / ln 2) through the
  constant 1.5 · 2^m, a Cody–Waite reduction, Taylor of degree 13 (f64) or 7 (f32) by
  Horner, and the exponent raised by an integer add on the bits (design §4.9). That is 20
  operations per lane in f64 and 14 in f32. The vector layer gains two methods, `max` and
  `scale_pow2`.
- **The contract holds: within 4 u_T relative on [x_min, 0]**, with x_min = −125 ln 2
  (f32) and −1021 ln 2 (f64), where every result is normal. Derived bounds 3.3 u₃₂ and 3.2 u₆₄. Measured:
  **1.324 u₃₂** over every float of the f32 domain (1.12 · 10⁹ values) and **1.228 u₆₄**
  over 10⁷ f64 samples against a double-double reference. The figures are the same on
  every ISA of both machines (§1).
- **Bit for bit the same on every ISA**: NEON, AVX2 and AVX-512 equal the scalar ISA on
  every value tested. The steps are IEEE-rounded operations only, so the exponential has
  the same bits on the M3 Max and on Kathleen.
- **Laplace: bits unchanged; code unchanged except in one kernel's set-up.** Bits cannot
  change: no existing method or body was edited, and Rust neither contracts nor reorders
  floating-point operations. Of the existing functions of `nd-fmm-simd`, all 124 on
  x86_64 and 49 of 54 on aarch64 are identical to `main`'s, instruction for instruction
  (§3). The five that differ are the plane-wave M2L's `sums` step on aarch64. They differ
  because rustc partitions the crate into codegen units differently once the new code is
  in it. With one codegen unit every function is identical to `main`'s. The innermost
  loops are identical. In NEON's f32 `sums` one closure of `sums_body` is now called out of
  line, once per outer iteration.
- **That costs NEON's f32 `sums` step 4–6 %** (§4, D9 on the M3 Max). On the first run
  the step at p = 8 failed D9 (+6.1 % against a spread of 3.6 %). On the one repeat D9
  allows it passed (+3.6 % against 4.2 %). Every other row passes, f64 included, and so
  does the plane-wave M2L's whole time per box. This is question 1 of §5.

## 1. The contract on every ISA

| machine | ISA | f32, every float of [x_min, 0] | f64, 10⁷ samples | special values | bits against scalar |
| --- | --- | ---: | ---: | --- | --- |
| M3 Max | scalar | 1.324 u | 1.228 u | pass | – |
| M3 Max | NEON | 1.324 u | 1.228 u | pass | identical |
| Kathleen 615691 | scalar | 1.324 u | 1.228 u | pass | – |
| Kathleen 615691 | AVX2 | 1.324 u | 1.228 u | pass | identical |
| Kathleen 615691 | AVX-512 | 1.324 u | 1.228 u | pass | identical |

The worst f32 error is at x = −5.8911…, the worst f64 sample at x = −690.752…. The
special values are: the reduction's boundaries (n + ½) ln 2 for every n of the domain,
with three neighbours on each side (worst 1.212 u₃₂, 1.174 u₆₄); the domain end and its
neighbours; e^{±0} = 1 exactly; and below x_min (x_min minus 1 ulp and minus 1, −10³⁰,
the most negative finite value, −∞), which gives the value at x_min. That value is
positive and at most 2^(e_min + 2), and the first float above x_min differs from it.
Lane and length independence and the panics are tested as well (`tests/exp`).

The f64 reference is `exp_dd`: the reduction by a triple-double ln 2 and a Taylor series
in double-double. It is itself checked against e^x to 80 digits (Python's `decimal`) at
three values, within 2⁻¹⁰⁰, and against `f64::exp` within one ulp on 10⁴ samples. It
returns e^r and n, and the error is taken on y 2⁻ⁿ, which is exact. The first version
scaled its low part by 2ⁿ, which goes subnormal near x_min. It reported 1.519 u₆₄ at
x = −707.43 (Kathleen job 615552 and the M3 Max), half a unit of which was the oracle's
own error (found by the self-review). Job 615691 and the M3 Max ran the fixed oracle.

## 2. Inner loops

From the release rlibs (`fmm-simd/tools/inner_loops.awk`; x86_64 from the
`x86_64-apple-darwin` build). None of the loops contains a call.

| entry point | f32 | f64 |
| --- | ---: | ---: |
| `neon::exp_slice_*` | 27 (14 operations, 13 moves) | 39 (20 operations, 19 moves) |
| `avx2::exp_slice_*` | 23 | 33 |
| `avx512::exp_slice_*` | 23 | 29 |

On x86_64 LLVM folds some negations into the constants (`vfmsub` with −c, `vfmadd` with
−ln2_hi). These round identically, as the bit identity on Kathleen shows. The moves on NEON
are register copies before the destructive `fmla`.

## 3. Laplace's code, `main` against the branch

Every function of the release rlib of `nd-fmm-simd` was compared, `main` (16d7950, built in
a detached worktree with its own target directory) against the branch. The comparison
ignores operands that only locate data or code in an unlinked object: aarch64 `adrp`
pages, x86_64 RIP displacements, objdump's comments and call offsets.

| build | functions on `main` | identical | differing |
| --- | ---: | ---: | --- |
| aarch64, default (16 codegen units) | 54 | 49 | NEON `sums_f32`, `sums_f64`; scalar `sums_f32`, `sums_f64`; one closure of `sums_body` |
| aarch64, `-C codegen-units=1` | 69 | 69 | none |
| x86_64, default | 124 | 124 | none |

The branch adds only new functions. In the default aarch64 build the four `sums` entry
points have the same innermost loops (16, 61, 72 and 32 instructions, …) as on `main`. The
differences are register allocation, the stack frame, bounds-check paths, and one more
call per outer iteration in NEON's `sums_f32`: `<[&[f32]; 2]>::try_map` with
`sums_body`'s first closure, inlined on `main`. The Linux targets of locust and Kathleen
were not compared; their partitions may differ (§5).

## 4. D9: the plane-wave steps on the M3 Max

`Auto` runs the plane-wave M2L for Laplace at f64 p ≥ 7 and f32 p ≥ 8, so the `sums`
change is on Laplace's default path at high degree. `plane_wave_parts` was run (`--ps 8,12
--isa neon --repeats 15`, one thread, BLAS threads at 1; 4,096 boxes, µs per box) from
both builds in the order A B B A A B B A A B, with A = `main` and B = the branch. Per D9,
the spread is max − min of `main`'s five run medians, and a row fails if the branch's
median exceeds `main`'s by more than that spread. The machine's load was not recorded
(the run was sandboxed).

| precision | p | step | `main` min–max | `main` | branch | branch − `main` | spread | D9 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| f32 | 8 | sums | 1.913–1.983 | 1.937 | 2.055 | +0.118 (+6.1 %) | 0.070 | **fail** |
| f32 | 8 | waves | 1.678–1.870 | 1.698 | 1.677 | −0.021 | 0.192 | pass |
| f32 | 8 | merge | 1.843–2.758 | 1.975 | 2.007 | +0.032 | 0.915 | pass |
| f32 | 8 | gather | 0.751–0.797 | 0.782 | 0.777 | −0.005 | 0.046 | pass |
| f32 | 8 | per box | 10.219–11.431 | 10.403 | 10.537 | +0.134 | 1.212 | pass |
| f32 | 12 | sums | 3.127–3.426 | 3.139 | 3.298 | +0.159 (+5.1 %) | 0.299 | pass |
| f32 | 12 | waves | 2.842–2.960 | 2.915 | 2.905 | −0.010 | 0.118 | pass |
| f32 | 12 | merge | 3.608–4.985 | 3.674 | 3.782 | +0.108 | 1.377 | pass |
| f32 | 12 | gather | 1.253–1.311 | 1.262 | 1.284 | +0.022 | 0.058 | pass |
| f32 | 12 | per box | 19.812–22.159 | 20.037 | 20.560 | +0.523 | 2.347 | pass |
| f64 | 8 | sums | 2.016–2.140 | 2.066 | 2.030 | −0.036 | 0.124 | pass |
| f64 | 8 | waves | 2.167–2.265 | 2.206 | 2.240 | +0.034 | 0.098 | pass |
| f64 | 8 | merge | 4.159–5.544 | 4.281 | 4.312 | +0.031 | 1.385 | pass |
| f64 | 8 | gather | 1.495–1.559 | 1.534 | 1.503 | −0.031 | 0.064 | pass |
| f64 | 8 | per box | 14.311–16.506 | 14.737 | 14.671 | −0.066 | 2.195 | pass |
| f64 | 12 | sums | 4.501–5.093 | 4.688 | 4.637 | −0.051 | 0.592 | pass |
| f64 | 12 | waves | 4.073–4.889 | 4.122 | 3.979 | −0.143 | 0.816 | pass |
| f64 | 12 | merge | 7.942–14.433 | 8.472 | 8.661 | +0.189 | 6.491 | pass |
| f64 | 12 | gather | 2.350–2.522 | 2.431 | 2.423 | −0.008 | 0.172 | pass |
| f64 | 12 | per box | 29.559–38.183 | 30.232 | 30.322 | +0.090 | 8.624 | pass |

"per box" is the example's sum over every step. The steps "rotate in", "modes", "local"
and "rotate out" do not run `nd-fmm-simd` code and are left out.

**Repeat of the failing row** (f32, p = 8, `sums`; the same order, D9's one repeat):
`main` 1.920, 1.937, 1.946, 1.964, 2.002; branch 2.007, 2.012, 2.016, 2.026, 2.062.
Medians 1.946 and 2.016: +0.070 (+3.6 %) against a spread of 0.082, so it **passes**.

In both runs every branch median is above `main`'s median, and in the repeat above every
`main` run. The out-of-line call costs this step about 4–6 % on NEON in f32. The step is
about a fifth of the plane-wave M2L's time per box at f32 p = 8, so about 1 % of that
level call. f64 `sums`, whose differences are bounds-check paths only, shows no cost.

## 5. Questions for sign-off

1. **NEON f32 `sums`, 4–6 % slower from a codegen-unit shift.** It failed D9 once and
   passed on the repeat, both reported. The cause is the out-of-line closure call of §3,
   from rustc's partitioning, not a change of code; CH2.4b's additions will move the
   partition again. Options:
   - (a) **accept**, as CH2.0's Kathleen rows were accepted as code placement
     (recommended for this PR: it passes D9 as amended, and the effect is about 1 % of a
     level call);
   - (b) make the slicing in `waves::sums_body` independent of inlining decisions
     (explicit slices instead of `[0, h].map(…)`, bit for bit the same): a source change
     to a Laplace kernel, with constraint 1's checks, in CH2.4b or a task of its own;
   - (c) `codegen-units = 1` for the release profile: deterministic code for every crate,
     at a cost in build time across the workspace.
2. **locust and Kathleen were not timed** for the plane-wave steps. Kathleen's x86_64
   code is unchanged on macOS's x86_64 build. Neither Linux target's partition was
   compared. The plan (signed off, "Proceed as planned") left Laplace's benchmarks to
   CH2.4b; this report proposes keeping that, and running the plane-wave steps on locust
   in CH2.4b's D9 runs.
