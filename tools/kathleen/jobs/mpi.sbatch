#!/bin/bash -l
# tools/kathleen/jobs/mpi.sbatch: the MPI test list of .github/scripts/run-mpi-tests.sh
# at the given rank counts (Phase 5N T1). One node, `singlenode`, unless the submission
# says otherwise.
#
#     tools/kathleen/submit.sh mpi.sbatch 1 2 4 8                           # one node
#     tools/kathleen/submit.sh --qos=test --nodes=2 -- mpi.sbatch 80        # 2 x 40
#
# Builds the list (debug, as in CI), then runs every executable at each rank count under
# `timeout` (RUN_TIMEOUT, here 900 s unless the submission exports another; CI's 300 s
# is too short for mpi_exec at 80 ranks in debug: 242 s and over 300 s in jobs 238841
# and 238835) with `mpirun` inside the allocation, one rank per core
# (`--bind-to core`; Open MPI takes the slots, 40 per node, from Slurm). The launcher is
# mpirun because run-mpi-tests.sh launches with it; decision 4 (launcher.sbatch) compares
# srun.
#SBATCH --job-name=fmm-mpi
#SBATCH --partition=kathleen
#SBATCH --qos=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

ranks=${*:-1 2 4 8}
export RUN_TIMEOUT=${RUN_TIMEOUT:-900}
list=target/mpi-executables.txt
step build ".github/scripts/run-mpi-tests.sh build $list"
step "run-$(echo "$ranks" | tr ' ' '-')" "MPIRUN_FLAGS='--bind-to core' .github/scripts/run-mpi-tests.sh run $list $ranks"
