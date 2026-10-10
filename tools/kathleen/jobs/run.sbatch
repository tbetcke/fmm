#!/bin/bash -l
# tools/kathleen/jobs/run.sbatch: runs each argument as one step, in the environment
# (Phase 5N T1). One node, `singlenode`, unless the submission says otherwise.
#
#     tools/kathleen/submit.sh run.sbatch 'cargo test -p nd-fmm-exec' 'du -sh target'
#
# For the commands the other job scripts do not cover. Never on the login node: a build
# or a test is a job (README.md, "The login-node rule"). The allocation has 40 tasks per
# node, so that an `mpirun` inside a step has a slot per core.
#SBATCH --job-name=fmm-run
#SBATCH --partition=kathleen
#SBATCH --qos=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

i=1
for command in "$@"; do
    step "$i" "$command"
    i=$((i + 1))
done
