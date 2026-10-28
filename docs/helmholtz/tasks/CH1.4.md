# CH1.4 FMM3D convention check and fixtures

Started 2026-10-28 on `tbetcke/h-ch1-4`. Roadmap: "CH1.4 FMM3D convention check and
fixtures" (H1; first of the priority list). Proposal: §10 CH1.4; §9.2 (FMM3D as an
independent reference); §4 (the items of §3.15); §1.1 and §3.1 (the kernel and its
expansion); §8 R12.

## Goal

Identify FMM3D 2.1.0's Helmholtz conventions on two points against a written draft of
CONVENTIONS §3.15, commit FMM3D-made fixtures for `nd-fmm-ref` (CH1.3), and answer R12
(whether `hfmm3d` is accurate for purely imaginary zk).

## Acceptance (proposal §10, verbatim)

> on two points, FMM3D's factor, sign of k, gradient and branch identified and matching the
> draft §3.15 (or the draft corrected); fixtures for real, complex and purely imaginary k,
> and real and complex charges, committed with release, routine and `eps`; whether FMM3D is
> accurate for purely imaginary zk (R12)

Needs (§10): a draft of §3.15. Roadmap: "First a draft of §3.15 from proposal §4 and
S1–S3".

## Read first

- Proposal §1.1, §3.1, §3.3, §3.4, §4, §9.1, §9.2, §8 R12.
- The S1–S3 reports and their sign-offs (progress log): `spikes/rotation-blocks/`,
  `spikes/coaxial/`, `spikes/radial/` `SPIKE_REPORT.md` §1 and §8.
- `tools/fmm3d/` (T9's driver, build and README); `tools/kathleen/jobs/fmm3d.sbatch`.
- FMM3D 2.1.0 (`$FMM3D_ROOT/build/default/FMM3D-2.1.0/src/Helmholtz/`): `hfmm3dwrap.f`
  (`hfmm3d_s_c_g`, `hfmm3d_st_c_g`: u(x) = Σ c_j e^{ik|x−x_j|}/(4π|x−x_j|), complex zk and
  charges, self term dropped), `helmkernels.f` (`h3ddirectcg`: the direct sum and its
  gradient, (ik d − 1) e^{ikd}/(4π d³) (x − y), skipping |x − y| < thresh), `hfmm3d.f`
  (the level cut-off exp(−Im k w)/w < eps; plane waves while Re(k w) ≤ 16π and
  Im(k w) < 12π, point and shoot beyond).

Found at planning: the pinned release has a Helmholtz direct routine (`h3ddirectcg`), so
by §9.2 the fixtures are its values, not `hfmm3d` at a tight `eps`. The M3 Max has FMM3D
2.1.0 built in `../fmm3d` (T9, Homebrew GCC 16.2.0); Kathleen in
`$FMM_KATHLEEN_ROOT/fmm3d` (T9), with Python 3.9 on the login node.

## Changes

No crate changes; nothing on Laplace's paths.

- `docs/helmholtz/conventions-3.15-draft.md` (new): the draft §3.15, items 1–9 of
  proposal §4, from the proposal and the S1–S3 sign-offs; what is left to later tasks
  (signatures, precision ranges) named as open. CH1.1 turns it into CONVENTIONS §3.15
  (with its sign-off); `docs/CONVENTIONS.md` is not touched here.
- `tools/fmm3d/driver.f90`: two Helmholtz modes, selected by the first argument; the
  Laplace usage (6 or 7 arguments) unchanged, its output byte for byte (checked):
  - `fmm3d_driver helmholtz-check`: the two-point check. Unit charges at (0,0,0) and
    (1,0,0); `hfmm3d_s_c_g` (eps 1e-12) and `h3ddirectcg` for k = 1, 2.5, 1 + 0.5i,
    2.5 + 1i, 0.5i, 3i; prints the raw values; stops unless they are the draft's
    e^{ikr}/(4πr) and ∇ₓ, to 1e-12. Also, for the record only, k = 1 − 0.5i and −1
    (outside the draft's branch).
  - `fmm3d_driver helmholtz PROBLEM PREFIX ZK_RE ZK_IM TOKEN[,TOKEN...]`: runs the check
    first, then for every token (`direct`, or an `eps`) the potential and gradient at the
    problem's sampled sources (self excluded) and its extra targets: `h3ddirectcg` or
    `hfmm3d_s_c_g`/`hfmm3d_st_c_g`; writes `PREFIX-<token>.out`, prints one line of
    times per token. Problem and output formats (`NDFMMHP1`, `NDFMMHO1`, little-endian,
    complex as (re, im) pairs) documented in the driver.
- `tools/fmm3d/build.sh`: a step `driver` that relinks only the driver against an
  existing build (Kathleen and the M3 Max have FMM3D from T9).
- `tools/fmm3d/helmholtz.py` (new, Python 3.9 standard library only): writes problem
  files, runs the driver, reads outputs, the error norms of §9.1; and the R12 sweep
  (`r12`), which prints a Markdown table.
- `tools/fixtures/gen_helmholtz_fmm3d.py` (new, PEP 723, `mpmath==1.4.1` as the other
  generators): runs the check, identifies FMM3D's conventions on the two points among
  the candidates (factor 1 or 1/(4π); e^{+ikr} or e^{−ikr}; gradient in x or in y), builds
  the fixture problems (seed 20261028: 200 sources in [0, 1]³, 40 extra targets in
  [−0.5, 1.5]³; charges real and complex, uniform in [−1, 1] per part; k = 1, 25,
  1 + 0.5i, 25 + 2.5i, i, 25i), runs `h3ddirectcg`, checks every value against the draft's
  kernel in mpmath (30 digits) and writes `fmm-ref/fixtures/helmholtz_fmm3d.json`: header
  with FMM3D's release, commit, routine, `thresh`, `hfmm3d`'s `eps` of the check, the
  build (compilers, machine), the draft's identification, the mpmath agreement per case,
  the generator version; then the points, charges and values. `--check` regenerates in
  memory and compares with the committed file to 1e-13 relative per case (the bits of
  FMM3D's direct sum depend on the compiler), `--no-reference` skips mpmath (for
  Kathleen's Python).
- `tools/kathleen/jobs/fmm3d.sbatch`: steps `driver` (relink), `helmholtz-check`,
  `helmholtz-fixtures` (the generator's `--check --no-reference`) and `r12`.
- `tools/fmm3d/README.md`, `tools/fixtures/README.md`: the new modes and the generator.
- Report `fmm-validate/results/helmholtz-ch1-4.md`: the identification, the fixtures,
  R12.

## Tests that prove it

- Two-point identification: the driver's hard check against the draft (both routines,
  six k, 1e-12 relative), and the generator's candidate table with exactly one candidate
  matching per case. On the M3 Max and on Kathleen.
- Fixtures: every `h3ddirectcg` value against the draft's kernel e^{ikr}/(4πr) and its
  gradient summed in mpmath at 30 digits: relative ℓ2 per case and quantity ≤ 1e-13
  (and the maximum pointwise relative error recorded); `--check` reproduces the committed
  file on the M3 Max (byte for byte expected there) and on Kathleen to 1e-13.
- Laplace mode unchanged: the old and new driver binaries on `cube-n100000.bin` at
  eps = 1e-6: the `.out` files byte for byte identical.
- R12: `hfmm3d_s_c_g` against `h3ddirectcg` on 1,000 sampled sources, relative ℓ2 of φ
  and ∇φ (§9.1), uniform cube (side D = 1) and sphere surface (diameter 1), imaginary
  κD ∈ {0.01, 0.1, 1, 10, 30, 100, 300, 1000}, real kD ∈ {1, 10, 30, 100} and complex
  kD = 10 + 1i, 100 + 10i as controls, eps ∈ {1e-3, 1e-6, 1e-9, 1e-12}; time per call
  beside real k of the same |k|. "Accurate" = error ≤ eps (FMM3D's own target, as T9
  read it for Laplace) on every case; anything else reported per case.
  N = 10⁵ on the M3 Max (12 threads), N = 10⁶ on one Kathleen node (40 threads).

## Checks

`/verify`: no Rust changes, so format, Clippy and tests only confirm nothing moved;
`cargo deny` not needed (no manifest change). This task's: `gen_helmholtz_fmm3d.py
--check`, the driver's Laplace byte comparison, `bash -n` and `shellcheck` (if present)
on the changed scripts, `python3 -m py_compile` on the new scripts.

## Machine runs

- M3 Max: the check, the fixtures, R12 at N = 10⁵ (unsandboxed for timing).
- Kathleen, one node in `singlenode`: `fmm3d.sbatch driver helmholtz-check
  helmholtz-fixtures r12` (N = 10⁶): the gate machine's build matches the draft and
  reproduces the fixtures, and R12 at the larger N.

## Constraints and risks

- Constraint 6 (trusted paths): FMM3D is the independent reference of §9.2; the fixtures
  are checked against the draft's kernel in extended precision, so a fixture that
  disagrees with the draft cannot be committed silently.
- FMM3D stays outside the workspace, never linked (T9's rule); only its values are
  committed, with its release and routine.
- Laplace: not touched. The Laplace mode of the driver (a tool, not a crate) is shown
  byte for byte unchanged.
- Risk: FMM3D's direct sum differs between compilers in the last bits, so fixtures are
  reproducible to round-off, not byte for byte, across machines; CH1.3's tolerance takes
  that into account.

## Pending runs

None. The runs that were pending:

- Kathleen job 614099 (`singlenode`, node-c11b-009, 40 threads) completed on 2026-10-28:
  `fmm3d.sbatch driver helmholtz-check helmholtz-fixtures r12`. Every step passed; R12 at
  N = 10⁶ is in the report.
- Job 614095 (node-c11a-089) passed the check and the fixtures. Its R12 ran on one thread
  (env.sh sets `OMP_NUM_THREADS=1`) and was cancelled; the step now sets `R12_THREADS`.

## Open questions

For sign-off (report §5); signed off 2026-10-28, "1–3 as recommended": each as recommended below.

1. The draft §3.15 as CH1.1's starting point. It needs no correction from the check.
   Where the factor 4π ik is applied is left to CH1.1.
2. R12 is partial: `hfmm3d` is reliable for imaginary κD ≤ 30 at every eps, and at
   eps ≥ 10⁻⁶ everywhere (to 1.7×). It misses at eps ≤ 10⁻⁹ for κD 50–300, and for
   complex k with Im kD ≈ 100. The proposal's fallback skips CH2.5. Recommended
   instead: CH2.5 compares against FMM3D only where it met its eps, and CH5.3 likewise.
3. The proposal's next draft: R12's row, §9.2's sentence on imaginary zk, and FMM3D's
   thresh rule beside §3.13's.
