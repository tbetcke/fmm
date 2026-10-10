#!/bin/bash -l
# tools/kathleen/jobs/scaling.sbatch: one nd-fmm-validate `scaling` launch across two
# nodes, 2 ranks per node x 20 threads (one rank per socket), the N = 10^6 cube, f64,
# p = 3, with the one-rank reference and 8 error vectors (Phase 5N T1). Two nodes, `test`.
#
#     tools/kathleen/submit.sh scaling.sbatch
#
# It shows threads, binding and the errors across nodes; its timings are not results
# (T2 times). The release example is built by build.sbatch (or here).
#SBATCH --job-name=fmm-scaling
#SBATCH --partition=kathleen
#SBATCH --qos=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=00:45:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

step build 'cargo build --release -p nd-fmm-validate --example scaling'
step scaling "timeout 1800 mpirun -n 4 --map-by ppr:1:socket:PE=20 --bind-to core --report-bindings target/release/examples/scaling --dist cube --n 1000000 --precision f64 --p 3 --threads 20 --overlap both --errors 8 --reference --evals 2"
