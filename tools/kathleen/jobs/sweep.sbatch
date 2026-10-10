#!/bin/bash -l
# tools/kathleen/jobs/sweep.sbatch: sweeps of tools/scaling/run.sh inside one allocation,
# and the x86 host's P2P examples (Phase 5N T2, docs/phase5n/T2-kathleen-baseline.md).
# One node, `singlenode`, 6 hours, unless the submission says otherwise.
#
#     tools/kathleen/submit.sh sweep.sbatch strong threads strategy check host p2p
#     tools/kathleen/submit.sh --qos=test --nodes=2 -- sweep.sbatch nodes-strong
#     tools/kathleen/submit.sh --qos=small --nodes=2 --time=03:00:00 -- sweep.sbatch nodes-weak nodes-check
#
# Each argument is one step: a sweep of tools/scaling/run.sh (its header lists them; under
# Slurm it maps the ranks over the job's nodes and sockets with `mpirun --map-by ppr:…
# --bind-to core`, decision 4, and runs every launch under `timeout`), or `p2p`, which
# runs `p2p_kernels` (every ISA of the machine, f32 and f64, φ and φ with ∇φ; `--ghz 2.5`,
# the Xeon Gold 6248's base clock, for the model's cycles) and `p2p_fmm --part kernels`
# (the kernels inside the FMM at p = 3 and 8), one rank on core 0 with one thread.
# One file per launch goes to $OUT (default the root's logs/scaling-<job id>/, never a
# source tree: raw output stays out of the repository); the job's own output lists the
# files and every launch that failed. The variables of run.sh (RANKS, SPLITS, EVALS,
# TIMEOUT) go in as sbatch options, since submit.sh submits from a fresh login shell:
#
#     tools/kathleen/submit.sh --qos=test --nodes=2 '--export=ALL,SPLITS=2x20' -- sweep.sbatch nodes-strong
#SBATCH --job-name=fmm-sweep
#SBATCH --partition=kathleen
#SBATCH --qos=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=06:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

export OUT=${OUT:-$FMM_KATHLEEN_ROOT/logs/scaling-$SLURM_JOB_ID}
mkdir -p "$OUT"
echo "=== output of the launches: $OUT"

step build 'cargo build --release -p nd-fmm-validate --example scaling --example p2p_kernels --example p2p_fmm'
for sweep in "$@"; do
    case "$sweep" in
    p2p)
        step p2p-kernels "RAYON_NUM_THREADS=1 taskset -c 0 target/release/examples/p2p_kernels --ghz 2.5 >'$OUT/p2p_kernels.md' 2>'$OUT/p2p_kernels.err'"
        step p2p-fmm "taskset -c 0 target/release/examples/p2p_fmm --part kernels --threads 1 >'$OUT/p2p_fmm.md' 2>'$OUT/p2p_fmm.err'"
        ;;
    *)
        step "$sweep" "tools/scaling/run.sh $sweep"
        ;;
    esac
done
step files "ls '$OUT' | wc -l; echo 'failed or skipped launches:'; grep -l -e '^FAILED' '$OUT'/*.txt; cat '$OUT/skipped.txt' 2>/dev/null; true"
