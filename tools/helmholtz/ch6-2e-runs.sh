#!/bin/bash
# tools/helmholtz/ch6-2e-runs.sh: the machine runs of Helmholtz CH6.2e
# (docs/helmholtz/tasks/CH6.2e.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch6-2e-runs.sh <out-dir> <stage…>
#
# Stages (one rank × THREADS threads, default 72; BLAS threads 1; the default waves budget):
# - `verify`: CH6.1e's verification cases at the rule's tolerances (CH3.1c's matrix at
#   ε = 1e-3 and 1e-6 with its real-charge and complex-k rows, `between`'s at 1e-4 and
#   1e-5), `--m2l rotation,planewave` with the α counts at every reach of REACHES (default
#   "corner,5,4.5,4,3.5,3");
# - `verify-large`: the same on W3 at kD = 100 and W1 at kD = 500;
# - `scan`: CH3.1c's matrix at 1e-3 and 1e-6, `--m2l planewave-scan:2` (Rotation, then each
#   level with a quadrature alone at iprec 2) at each reach of SCAN_REACHES (default
#   "corner,4");
# - `times`: CH6.2b's matrix and large cases at 1e-3 and 1e-6, `--m2l planewave
#   --evaluations 2 --levels` at the corner and at REACH (default 4), each case once in each
#   order (ABBA). Timed: run it on a quiet machine (locust: `nvidia-smi`, `uptime`, no other
#   session's tests).
# Accuracy stages are not timed for a gate. TIMEOUT (default 3600) bounds a launch; ONLY (a
# regular expression on the launch names) runs a subset. Each launch's output in
# <out-dir>/<stage>-<name>.md, a line per launch in <out-dir>/runs.log.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
fi
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-72}
REACHES=${REACHES:-corner,5,4.5,4,3.5,3}
SCAN_REACHES=${SCAN_REACHES:-corner,4}
REACH=${REACH:-4}
TIMEOUT=${TIMEOUT:-3600}

# `launch <name> <command…>`: one launch under the timeout.
launch() {
    local name=$1
    shift
    [[ "$name" =~ ${ONLY:-.} ]] || return 0
    local start=$SECONDS
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

# CH6.1e's cases at the rule's tolerances (its `small` less ε = 1e-8, its `between` less
# 1e-7: no level has waves below 1e-6), and its large cases.
matrix=(
    "w1|--workload w1 --kd 10,30,100 --eps 1e-3,1e-6"
    "w3|--workload w3 --kd 10,30 --eps 1e-3,1e-6"
    "cube|--workload w4-cube:100000 --kd 1,10,100 --eps 1e-3,1e-6"
    "sphere|--workload w4-sphere:100000 --kd 1,10,100 --eps 1e-3,1e-6"
    "plummer|--workload w4-plummer:100000 --kd 1,10,100 --eps 1e-3,1e-6"
    "clusters|--workload w4-clusters:100000 --kd 1,10,100 --eps 1e-3,1e-6"
    "w1-real|--workload w1 --kd 100 --eps 1e-3,1e-6 --charges real"
    "w1-complex-k|--workload w1 --kd 100 --kim 10 --eps 1e-3,1e-6"
)
between=(
    "b-w1|--workload w1 --kd 10,30,100 --eps 1e-4,1e-5"
    "b-w3|--workload w3 --kd 10,30 --eps 1e-4,1e-5"
    "b-cube|--workload w4-cube:100000 --kd 1,100 --eps 1e-4,1e-5"
    "b-clusters|--workload w4-clusters:100000 --kd 1,100 --eps 1e-4,1e-5"
)
large=(
    "w3-100|--workload w3 --kd 100 --eps 1e-3,1e-6"
    "w1-500|--workload w1 --kd 500 --eps 1e-3,1e-6"
)
# CH6.2b's matrix (its `matrix` stage).
timed=(
    "w1|--workload w1 --kd 10,30,100"
    "w3|--workload w3 --kd 10,30"
    "w4-cube|--workload w4-cube:100000 --kd 1,100"
    "w4-sphere|--workload w4-sphere:100000 --kd 1,100"
    "w4-plummer|--workload w4-plummer:100000 --kd 1,100"
    "w4-clusters|--workload w4-clusters:100000 --kd 1,100"
    "w1-500|--workload w1 --kd 500"
    "w3-100|--workload w3 --kd 100"
)

# `each <stage> <case…> -- <arguments…>`: every case with the arguments.
each() {
    local stage=$1
    shift
    local cases=()
    while [ "$1" != -- ]; do
        cases+=("$1")
        shift
    done
    shift
    for entry in "${cases[@]}"; do
        # Word splitting of the case's arguments is intended.
        # shellcheck disable=SC2086
        launch "$stage-${entry%%|*}" "$X" ${entry#*|} --threads "$T" "$@"
    done
}

for stage in "$@"; do
    case $stage in
        verify)
            each verify "${matrix[@]}" "${between[@]}" -- --m2l rotation,planewave \
                --reach "$REACHES"
            ;;
        verify-large)
            each verify-large "${large[@]}" -- --m2l rotation,planewave --reach "$REACHES"
            ;;
        scan)
            each scan "${matrix[@]}" -- --m2l planewave-scan:2 --reach "$SCAN_REACHES"
            ;;
        times)
            for entry in "${timed[@]}"; do
                for order in forward reverse; do
                    reaches="corner,$REACH"
                    [ "$order" = reverse ] && reaches="$REACH,corner"
                    # shellcheck disable=SC2086
                    launch "times-${entry%%|*}-$order" "$X" ${entry#*|} --eps 1e-3,1e-6 \
                        --threads "$T" --m2l planewave --reach "$reaches" --evaluations 2 \
                        --levels
                done
            done
            ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
