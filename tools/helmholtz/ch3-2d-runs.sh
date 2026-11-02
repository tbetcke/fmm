#!/bin/bash
# tools/helmholtz/ch3-2d-runs.sh: the machine runs of Helmholtz CH3.2d
# (docs/helmholtz/tasks/CH3.2d.md, "Machine runs"), from the repository root after both
# examples were built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_m2l --example helmholtz_accuracy
#     tools/helmholtz/ch3-2d-runs.sh <out-dir> [p2p,…] [m2l-isa …]
#
# For each P2P choice of the second argument (default `auto`; `avx2` on Kathleen runs the
# AVX2 column kernel too) the end-to-end cases with `--m2l rotation,auto,sweep` and then in
# the reverse order; for each ISA after it, `helmholtz_m2l` per pair twice. One rank, one
# thread, BLAS threads 1. On macOS every launch waits until the other processes use at
# most QUIET % CPU (default 150), samples them every 5 s during the launch, and repeats a
# launch above DURING % (default 200) up to RETRIES times (default 2), as CH3.2a's gate.
set -u
out=$1
p2ps=${2:-auto}
shift $(( $# < 2 ? $# : 2 ))
m2l_isas=("$@")
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
bin=target/release/examples
QUIET=${QUIET:-150}
DURING=${DURING:-200}
RETRIES=${RETRIES:-2}
macos=false
if [ "$(uname)" = Darwin ]; then
    macos=true
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
fi

# The CPU percentage of every process but this script's launch and ps itself.
others() {
    ps -A -o %cpu=,pid=,comm= | awk -v me="$1" '$2 != me && $3 !~ /(^| |\/)ps$/ { s += $1 } END { printf "%d", s }'
}

# Runs one launch: `launch <name> <command…>`, its output in <out>/<name>.md.
launch() {
    local name=$1
    shift
    local attempt=0
    while :; do
        if $macos; then
            while [ "$(others 0)" -gt "$QUIET" ]; do sleep 10; done
        fi
        local before
        before=$($macos && others 0 || echo -)
        "$@" > "$out/$name.md" 2> "$out/$name.err" &
        local pid=$! peak=0
        if $macos; then
            while kill -0 "$pid" 2> /dev/null; do
                sleep 5
                local now
                now=$(others "$pid")
                [ "$now" -gt "$peak" ] && peak=$now
            done
        fi
        wait "$pid"
        local status=$?
        echo "$name: exit $status, others ${before}% before, ${peak}% peak during, $(date +%T)" | tee -a "$out/gate.log"
        if $macos && [ "$peak" -gt "$DURING" ] && [ "$attempt" -lt "$RETRIES" ]; then
            attempt=$((attempt + 1))
            echo "$name: void, repeated" | tee -a "$out/gate.log"
            continue
        fi
        break
    done
}

cases=(
    "w1|--workload w1 --kd 30,100 --eps 1e-3,1e-6"
    "cube|--workload w4-cube:100000 --kd 10 --eps 1e-3,1e-6"
    "w3|--workload w3 --kd 30 --eps 1e-3"
    "cube-real|--workload w4-cube:100000 --kd 10 --eps 1e-6 --charges real"
    "w1-real|--workload w1 --kd 100 --eps 1e-3 --charges real"
)
for p2p in ${p2ps//,/ }; do
    for order in forward reverse; do
        m2l=rotation,auto,sweep
        [ "$order" = reverse ] && m2l=sweep,auto,rotation
        for entry in "${cases[@]}"; do
            name=${entry%%|*}
            # shellcheck disable=SC2086
            launch "accuracy-$p2p-$name-$order" "$bin/helmholtz_accuracy" ${entry#*|} \
                --threads 1 --p2p "$p2p" --m2l "$m2l"
        done
    done
done
for isa in "${m2l_isas[@]}"; do
    for i in 1 2; do
        launch "m2l-$isa-$i" "$bin/helmholtz_m2l" --ps 3,4,5,6,7,8,9,10,12,15,18,20 --isa "$isa"
    done
done
