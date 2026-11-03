#!/bin/bash
# tools/helmholtz/ch6-2a-runs.sh: the machine runs of Helmholtz CH6.2a
# (docs/helmholtz/tasks/CH6.2a.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch6-2a-runs.sh <out-dir> <stage…>
#
# Stages: `budgets` (W1 at kD = 500 and W3 at kD = 100 at ε = 1e-6 under `--m2l planewave`
# at one rank × THREADS threads: Rotation once, then the waves budget whole, the default,
# 1024 MB, 256 MB and the floor (0), then whole and the default again, for the run-to-run
# spread) and `ranks` (the same cases at RANKS ranks × 1 thread, whole and the default
# budget, whose share per rank is the budget over the ranks per node). Each launch prints
# the bytes, the chunks and the waves computed, the evaluation's seconds, the checksum and
# rank 0's peak resident size (VmHWM on Linux). Meant for a Kathleen node (an exclusive
# node, the job named in the report); BLAS threads 1; each launch's output in
# <out-dir>/<stage>-<name>.md, a line per launch in <out-dir>/runs.log.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
    M="mpirun --host localhost:6 --map-by slot --bind-to none \
        --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0"
else
    M="mpirun"
fi
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
R=${RANKS:-40}

# `launch <name> <command…>`: one launch, under a one-hour timeout.
launch() {
    local name=$1
    shift
    local start=$SECONDS
    timeout 3600 "$@" > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

cases=(
    "w1-500|--workload w1 --kd 500 --eps 1e-6"
    "w3-100|--workload w3 --kd 100 --eps 1e-6"
)

for stage in "$@"; do
    case $stage in
        budgets)
            for entry in "${cases[@]}"; do
                name=${entry%%|*}
                # Word splitting of the case's arguments is intended.
                # shellcheck disable=SC2086
                launch "budgets-$name-rotation" "$X" ${entry#*|} --threads "$T" \
                    --m2l rotation
                for budget in whole default 1024 256 0 whole default; do
                    flag=()
                    [ "$budget" != default ] && flag=(--plane-wave-budget "$budget")
                    run=1
                    while [ -e "$out/budgets-$name-$budget-$run.md" ]; do run=$((run + 1)); done
                    # shellcheck disable=SC2086
                    launch "budgets-$name-$budget-$run" "$X" ${entry#*|} --threads "$T" \
                        --m2l planewave ${flag[@]+"${flag[@]}"}
                done
            done
            ;;
        ranks)
            for entry in "${cases[@]}"; do
                name=${entry%%|*}
                for budget in whole default; do
                    flag=()
                    [ "$budget" != default ] && flag=(--plane-wave-budget "$budget")
                    # shellcheck disable=SC2086
                    launch "ranks-$name-$budget" $M -n "$R" "$X" ${entry#*|} --threads 1 \
                        --m2l planewave ${flag[@]+"${flag[@]}"}
                done
            done
            ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
