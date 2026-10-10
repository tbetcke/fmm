#!/bin/bash -l
# tools/kathleen/jobs/check.sbatch: every CPU-side check of root CLAUDE.md, "Checks"
# (Phase 5N T1). One node, `singlenode`.
#
#     tools/kathleen/submit.sh check.sbatch                 # every group
#     tools/kathleen/submit.sh check.sbatch root simd       # some groups
#
# Groups, in this order (arguments select a subset):
#   root      the run-tests job: fmt, clippy, clippy --examples, cargo test, cargo doc
#   strict    clippy --workspace --all-targets, cargo test --workspace (with
#             --no-fail-fast, so that one failing executable does not hide the others)
#   simd      the run-tests-simd job: clippy, the tests, the ignored release tests (the
#             4 u_T contract, exhaustive in f32), with the CPU and the ISAs printed
#   kernels   the run-tests-kernels job: clippy with cpu, check with cuda (nd-fmm-kernels
#             and nd-fmm-exec; type-checked only, Kathleen has no GPU), the CPU-runtime
#             tests in release
# Every test that initialises MPI runs as a singleton here; the multi-rank tests are
# mpi.sbatch. Each command is a step: a failure is reported and the job carries on.
#SBATCH --job-name=fmm-check
#SBATCH --partition=kathleen
#SBATCH --qos=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

groups=${*:-root strict simd kernels}
for group in $groups; do
    case "$group" in
    root)
        step fmt 'cargo fmt -- --check'
        step clippy 'cargo clippy -- -D warnings'
        step clippy-examples 'cargo clippy --examples -- -D warnings'
        step test 'cargo test'
        step doc 'cargo doc --no-deps'
        ;;
    strict)
        step clippy-workspace 'cargo clippy --workspace --all-targets -- -D warnings'
        step test-workspace 'cargo test --workspace --no-fail-fast'
        ;;
    simd)
        step cpu 'lscpu | grep -E "Model name|Flags" | sed "s/Flags:.*\(avx2\).*\(avx512f\).*\(fma\).*/Flags: ... \1 ... \2 ... \3 .../"'
        step simd-clippy 'cargo clippy -p nd-fmm-simd --all-targets -- -D warnings'
        step simd-test 'cargo test -p nd-fmm-simd -- --show-output'
        step simd-ignored 'cargo test -p nd-fmm-simd --release -- --ignored --show-output'
        ;;
    kernels)
        step kernels-clippy 'cargo clippy -p nd-fmm-kernels --all-targets --features cpu -- -D warnings'
        step kernels-check-cuda 'cargo check -p nd-fmm-kernels --features cuda'
        step exec-check-cuda 'cargo check -p nd-fmm-exec --features cuda'
        step kernels-test 'cargo test -p nd-fmm-kernels --features cpu --release -- --show-output'
        step bundle 'du -sh "$HOME/.cache/tracel" "$XDG_DATA_HOME/tracel" 2>&1; ls "$XDG_DATA_HOME/tracel" "$HOME/.cache/tracel" 2>&1'
        ;;
    *) step "unknown-group-$group" false ;;
    esac
done
