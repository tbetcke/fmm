#!/bin/bash
# tools/helmholtz/ch6-1e-runs.sh: the machine runs of Helmholtz CH6.1e
# (docs/helmholtz/tasks/CH6.1e.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch6-1e-runs.sh <out-dir> <stage…>
#
# Stages: `scan` (CH3.1c's matrix, `--m2l planewave-scan`: Rotation, then each level with a
# quadrature at iprec 1, 2 and 3 alone), `scan-large` (the same on W3 at kD = 100 and W1 at
# kD = 500), `between` (the scan at ε = 1e-4, 1e-5, 1e-7 on the smaller trees), `verify`
# (CH3.1c's matrix and `between`'s cases with `--m2l rotation,planewave`, the rule),
# `verify-large` (the same on the large cases) and `ranks` (the rule's checksum at 1 × 6,
# 2 × 3 and 3 × 2 ranks × threads, under a 30-minute timeout). Accuracy only, nothing
# timed for a gate: no quiet gate. `scan-large` needs more than 24 GB (the report's §7).
# One rank × THREADS threads (default 6), BLAS threads 1; each launch's output in
# <out-dir>/<stage>-<name>.md, a line per launch in <out-dir>/runs.log.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
    loopback="--mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0"
else
    loopback=""
fi
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-6}

# `launch <name> <command…>`: one launch.
launch() {
    local name=$1
    shift
    local start=$SECONDS
    "$@" > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

# CH3.1c's matrix (its §6), the 8-vector row left to `verify`.
small=(
    "w1|--workload w1 --kd 10,30,100 --eps 1e-3,1e-6,1e-8"
    "w3|--workload w3 --kd 10,30 --eps 1e-3,1e-6,1e-8"
    "cube|--workload w4-cube:100000 --kd 1,10,100 --eps 1e-3,1e-6,1e-8"
    "sphere|--workload w4-sphere:100000 --kd 1,10,100 --eps 1e-3,1e-6,1e-8"
    "plummer|--workload w4-plummer:100000 --kd 1,10,100 --eps 1e-3,1e-6,1e-8"
    "clusters|--workload w4-clusters:100000 --kd 1,10,100 --eps 1e-3,1e-6,1e-8"
    "w1-real|--workload w1 --kd 100 --eps 1e-3,1e-6 --charges real"
    "w1-complex-k|--workload w1 --kd 100 --kim 10 --eps 1e-3,1e-6"
)
large=(
    "w3-100|--workload w3 --kd 100 --eps 1e-3,1e-6"
    "w1-500|--workload w1 --kd 500 --eps 1e-3,1e-6"
)
between=(
    "w1|--workload w1 --kd 10,30,100 --eps 1e-4,1e-5,1e-7"
    "w3|--workload w3 --kd 10,30 --eps 1e-4,1e-5,1e-7"
    "cube|--workload w4-cube:100000 --kd 1,100 --eps 1e-4,1e-5,1e-7"
    "clusters|--workload w4-clusters:100000 --kd 1,100 --eps 1e-4,1e-5,1e-7"
)

# `each <stage> <m2l> <case…>`: every case of the list with `--m2l <m2l>`.
each() {
    local stage=$1 m2l=$2
    shift 2
    for entry in "$@"; do
        # Word splitting of the case's arguments is intended.
        # shellcheck disable=SC2086
        launch "$stage-${entry%%|*}" "$X" ${entry#*|} --threads "$T" --m2l "$m2l"
    done
}

for stage in "$@"; do
    case $stage in
        scan) each scan planewave-scan "${small[@]}" ;;
        scan-large) each scan-large planewave-scan "${large[@]}" ;;
        between) each between planewave-scan "${between[@]}" ;;
        verify)
            each verify rotation,planewave "${small[@]}" "${between[@]/#/b-}"
            launch verify-w1-vectors "$X" --workload w1 --kd 100 --eps 1e-6 --vectors 8 \
                --threads "$T" --m2l rotation,planewave
            ;;
        verify-large) each verify-large rotation,planewave "${large[@]}" ;;
        ranks)
            M="mpirun --host localhost:6 --map-by slot --bind-to none $loopback"
            for split in "1 6" "2 3" "3 2"; do
                read -r n t <<< "$split"
                # shellcheck disable=SC2086
                launch "ranks-$n" timeout 1800 $M -n "$n" "$X" --workload w1 --kd 30,100 \
                    --eps 1e-3,1e-6 --threads "$t" --m2l planewave
            done
            ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
