# CHR.4 FFTW on the machines

Started 2026-10-27 on `tbetcke/h-chr-4`. Roadmap: "CHR.4 FFTW on the machines" (HR).
Proposal: §10 CHR.4; §6.7 ("FFTW stays optional", "RLST's FFTW interface"); §8 R15, R22.

## Goal

Know, and record per machine of record, which FFTW the φ step's optional back end would
link (version, both precisions, SIMD), and show that RLST 0.12's own FFTW tests, its
determinism test included, pass against it.

## Acceptance (proposal §10, verbatim)

> compiled and run: `pkg-config --modversion fftw3 fftw3f` and `rlst::fftw::version()` (its
> SIMD suffixes) on the M3 Max (Homebrew), locust (spack's FFTW through
> `tools/gh200/env.sh`, not the RPM's 3.3.8), Kathleen (`fftw/3.3.10`) and CI; RLST 0.12's
> own `cargo test --features fftw_system` on each, `fftw_determinism` included; the FFTW
> version, precisions and SIMD of each machine recorded in `tools/*/machine.md`. Fallback: a
> machine without FFTW in both precisions runs the product, or rustfft (D7)

## Read first

- RLST 0.12.0 from the cargo registry: `build.rs` (`discover_fftw`: pkg-config for `fftw3`
  and `fftw3f`, Homebrew's prefix as the macOS fallback, the minimum version),
  `src/fftw/backend.rs` (`version()`, read from the linked library at run time; `has_simd`;
  `check_version` for both precisions), `tests/fftw_determinism.rs` (fresh processes,
  `harness = false`), `tests/fftw_check_allocations.rs`. Its Linux dev-dependencies link
  system OpenBLAS (`openblas-src` with `system`).
- `tools/gh200/env.sh` (the spack view on `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH`),
  `tools/gh200/machine.md` (RPM `fftw-devel` 3.3.8 beside spack's fftw 3.3.11),
  `tools/kathleen/env.sh` and `machine.md` (`fftw/3.3.10/gcc-12.3.0`), the Kathleen job
  `run.sbatch`, `.github/workflows/run-tests.yml` (`libfftw3-dev` already installed).
- Found at planning, on the M3 Max: Homebrew's fftw 3.3.11 in both precisions;
  its version strings carry **no SIMD suffix** (`fftw-3.3.11` from both libraries), the
  case R22 names; RLST 0.12's `cargo test --locked --features fftw_system` passes in a
  scratch copy (486 unit tests, doctests, `fftw_determinism` 10 scenarios).

## Changes

- `tools/fftw/probe.sh` (new): run on any machine inside its environment, from the root
  of the checkout. Prints the host, OS, CPU and its SIMD flags; `pkg-config --modversion
  fftw3 fftw3f` and their `libdir`; the version strings of both libraries from a small C
  program built against pkg-config's flags (both precisions; `version()` reads only the
  double one); copies the registry's `rlst-<the workspace pin>` into
  `target/fftw-probe/`, adds an example that prints `rlst::fftw::version()`, runs RLST's
  own `cargo test --locked --features fftw_system` and that example, and lists the FFTW
  libraries the C program and the example load (`ldd` or `otool -L`), so that a build against spack's
  FFTW that loads the RPM's 3.3.8 at run time shows. Nothing enters the workspace or its
  lock file; RLST's packaged `Cargo.lock` is used with `--locked`.
- `tools/fftw/README.md` (new, short): what the probe does and how to run it on each
  machine.
- `.github/workflows/fftw-probe.yml` (new): the probe on `ubuntu-latest`, on pull
  requests to `main` that change `tools/fftw/` or the workflow, and by hand
  (`workflow_dispatch`); `libfftw3-dev libopenblas-dev pkg-config`. Not on every pull
  request: it tests RLST, not nd-fmm. CHR.3's job with the `fftw` feature can reuse the
  probe for its FFTW version report.
- Records: a section "FFTW (Helmholtz CHR.4)" in `tools/gh200/machine.md` and
  `tools/kathleen/machine.md`; new `tools/m3max/machine.md` and `tools/ci/machine.md` (no
  machine file existed for either), with the FFTW facts and the few machine facts they
  rest on.
- Root `CLAUDE.md`: the probe workflow under "Checks", one sentence.
- Report `fmm-validate/results/helmholtz-chr-4.md`: per machine the probe's output,
  condensed, and what it means for S9 and D7 (R22).
- Public API: none. Conventions: none. Dependencies: none (the probe builds RLST on its own,
  outside the workspace). Laplace: not touched.

## Tests that prove it

No Rust code of the workspace changes, so no new test in it. The evidence is the probe's
run on each machine:
- `pkg-config --modversion fftw3 fftw3f` prints a version for both, at least 3.3.4 (RLST's
  minimum, DF8);
- the C program's `fftw_version` and `fftwf_version`, and `rlst::fftw::version()` equal to
  the double one (the library RLST loads is the one pkg-config names; on locust spack's,
  not `/usr/lib64`'s 3.3.8, also shown by `ldd`);
- RLST's `cargo test --locked --features fftw_system`: every test executable passes,
  `fftw_determinism`'s "N scenarios, 0 failed" and `fftw_check_allocations` included.

## Checks

`/verify`: only scripts, a workflow and docs change, so its "Always" checks
(`cargo fmt --all -- --check`) and the docs checks; `shellcheck` on the probe if
installed; the CI run of the PR (`run-tests` and the new `fftw-probe`).

## Machine runs

- **M3 Max**: `tools/fftw/probe.sh` (in the sandbox; outside it only if `sysctl` is
  denied, for the CPU line).
- **locust**: `tools/gh200/sync.sh`, then `tools/gh200/remote.sh 'tools/fftw/probe.sh'`,
  after `uptime` (no timing, so load only matters for run time).
- **Kathleen**: `tools/kathleen/sync.sh`, then
  `tools/kathleen/submit.sh run.sbatch 'tools/fftw/probe.sh'` (one node, `singlenode`).
- **CI**: the `fftw-probe` workflow on this PR.

What they decide: per machine whether FFTW exists in both precisions with SIMD; the
fallback (the product, or rustfft by D7) for any machine without; input to S9 (CH0.9)
and D7.

## Constraints and risks

- Constraint 1: nothing of Laplace changes (no crate, manifest or lock file).
- §6.7 "FFTW stays optional": the default build is unchanged; the probe builds RLST
  outside the workspace, so `deny.toml`'s check is unaffected.
- R15 (one FFTW build per machine) and R22 (missing or SIMD-less FFTW): this task records
  the facts; the remedies are S9's and D7's.
- Risk: a machine's FFTW lacks single precision, or pkg-config cannot find it (Kathleen's
  module, CI's package): the probe says so and the fallback applies.
- Risk: on locust the RPM's 3.3.8 is loaded at run time behind spack's 3.3.11 at link
  time; the probe's `ldd` and version line show it.

## Pending runs

None yet.

## Open questions

None yet.
