#!/bin/bash -l
# tools/kathleen/jobs/correctness.sbatch: the correctness job of Phase 5S T2
# (docs/phase5s/T2-validation-at-scale.md; docs/design/scale-out.md §5, §9.2). Two nodes
# in `test` unless the submission says otherwise; at 4 nodes in `small` (never more,
# Phase 5S):
#
#     tools/kathleen/submit.sh correctness.sbatch                  # 2 nodes: mpi invariance smoke
#     tools/kathleen/submit.sh --qos=small --nodes=4 --time=02:00:00 \
#         '--export=ALL,INV_N=10000000' -- correctness.sbatch invariance large oversubscribed
#
# Each argument is a part (default: mpi invariance smoke), run in this order:
#   mpi             .github/scripts/run-mpi-tests.sh, built in debug as in CI, at 2, 4 and 8
#                   ranks and at one rank per core of the job (80 on 2 nodes), with
#                   `mpirun --bind-to core` (RUN_TIMEOUT per launch, 900 s here)
#   invariance      rank-count invariance (nd-fmm-validate's `scaling --invariance`, release)
#                   at one rank per core of the job against P/k for k of INV_FACTORS (on 2
#                   nodes 80 against 40, 20 and 10; on 4 nodes 160 against 80, 40 and 20):
#                   the cube and the Plummer sphere in f64 at p = 8 with Rotation and with
#                   PlaneWave (Phase 6 T13), the cube in f32 at p = 3 with Dense, N = INV_N,
#                   the random share, overlap both
#   smoke           the `scaling` smoke run of jobs/scaling.sbatch: 2 ranks x 20 threads per
#                   node, the N = 10^6 cube, f64 p = 3, the one-rank reference and 8 error
#                   vectors
#   large           the harness at N = LARGE_N (10^8) without the one-rank reference: the
#                   cube, f64 at p = 8 and f32 at p = 3 (where truncation, not f32 rounding,
#                   sets the error) with Rotation, the errors of one charge vector at 1,000
#                   targets (the distributed direct sum), at one rank per core of the job
#                   and at a quarter of those ranks, spread over every node
#   oversubscribed  rank-count invariance at 160 ranks per node (640 on 4 nodes) against
#                   P/8 (`--oversubscribe`, PSM2 with shared contexts; Kathleen's ceiling,
#                   docs/design/scale-out.md §10.1), the N = INV_N cube, f64 p = 8 with
#                   Rotation; correctness and counts only, never a timing. Put it last: a
#                   failed oversubscribed launch can leave the nodes' HFI contexts busy for
#                   every later launch of the job
#
# Every launch runs under `timeout` (LAUNCH_TIMEOUT, 1,800 s; RUN_TIMEOUT for the MPI list),
# with every BLAS thread variable at 1 (env.sh) and `--report-bindings`. The job's output
# ends with the table of its steps, one per launch: result and wall time. A step fails if
# its launch fails or, for the invariance launches, if any value differs (the example exits
# with status 1 on every rank). Raw output stays in the root's logs/, never in a tree.
#SBATCH --job-name=fmm-correctness
#SBATCH --partition=kathleen
#SBATCH --qos=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

parts=${*:-mpi invariance smoke}
nodes=$SLURM_JOB_NUM_NODES
cores=$((40 * nodes))
export RUN_TIMEOUT=${RUN_TIMEOUT:-900}
LAUNCH_TIMEOUT=${LAUNCH_TIMEOUT:-1800}
INV_N=${INV_N:-1000000}
INV_FACTORS=${INV_FACTORS:-2,4,8}
LARGE_N=${LARGE_N:-100000000}
scaling=target/release/examples/scaling
echo "=== parts: $parts; $nodes node(s), $cores cores; INV_N $INV_N, INV_FACTORS $INV_FACTORS, LARGE_N $LARGE_N"

# One rank per core, 20 per socket (decision 4 of docs/phase5n/README.md); ranks/socket
# for fewer.
per_core="--map-by ppr:20:socket:PE=1 --bind-to core --report-bindings"

for part in $parts; do
    case "$part" in
    mpi)
        list=target/mpi-executables.txt
        step mpi-build ".github/scripts/run-mpi-tests.sh build $list"
        step "mpi-2-4-8-$cores" "MPIRUN_FLAGS='--bind-to core' .github/scripts/run-mpi-tests.sh run $list 2 4 8 $cores"
        ;;
    invariance)
        step scaling-build "cargo build --release -p nd-fmm-validate --example scaling"
        for run in cube:f64:8:rotation plummer:f64:8:rotation cube:f32:3:dense \
            cube:f64:8:planewave plummer:f64:8:planewave; do
            IFS=: read -r dist precision p strategy <<<"$run"
            step "invariance-$dist-$precision-p$p-$cores-k$INV_FACTORS" \
                "timeout $LAUNCH_TIMEOUT mpirun -n $cores $per_core $scaling --dist $dist --n $INV_N --precision $precision --p $p --strategy $strategy --overlap both --invariance $INV_FACTORS"
        done
        ;;
    smoke)
        step scaling-build "cargo build --release -p nd-fmm-validate --example scaling"
        step "smoke-$((2 * nodes))x20" \
            "timeout $LAUNCH_TIMEOUT mpirun -n $((2 * nodes)) --map-by ppr:1:socket:PE=20 --bind-to core --report-bindings $scaling --dist cube --n 1000000 --precision f64 --p 3 --threads 20 --overlap both --errors 8 --reference --evals 2"
        ;;
    large)
        step scaling-build "cargo build --release -p nd-fmm-validate --example scaling"
        quarter=$((cores / 4))
        for run in f64:8 f32:3; do
            precision=${run%%:*} p=${run##*:}
            for ranks in $cores $quarter; do
                per_socket=$((ranks / nodes / 2))
                step "large-cube-n$LARGE_N-$precision-p$p-$ranks" \
                    "timeout $LAUNCH_TIMEOUT mpirun -n $ranks --map-by ppr:$per_socket:socket:PE=1 --bind-to core --report-bindings $scaling --dist cube --n $LARGE_N --precision $precision --p $p --strategy rotation --overlap off --evals 2 --errors 1"
            done
        done
        ;;
    oversubscribed)
        step scaling-build "cargo build --release -p nd-fmm-validate --example scaling"
        ranks=$((160 * nodes))
        step "oversubscribed-invariance-cube-f64-p8-$ranks-k8" \
            "timeout $LAUNCH_TIMEOUT mpirun -n $ranks --oversubscribe --map-by ppr:160:node --bind-to none -x PSM2_SHAREDCONTEXTS=1 $scaling --dist cube --n $INV_N --precision f64 --p 8 --strategy rotation --overlap both --invariance 8"
        ;;
    *)
        echo "unknown part $part (mpi, invariance, smoke, large, oversubscribed)"
        exit 2
        ;;
    esac
done
