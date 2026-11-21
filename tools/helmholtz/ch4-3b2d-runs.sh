#!/bin/bash
# tools/helmholtz/ch4-3b2d-runs.sh: the machine runs of Helmholtz CH4.3b2d, the V near rule's
# price per box before the plan on the levels whose waves are raised (docs/helmholtz/tasks/
# CH4.3b2d.md, "Machine runs"), from the repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-3b2d-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` on one rank with every core (Kathleen 1 × 40), with
# `--levels`, §9.1's complex charges and samples, two evaluations (the rows the second's), the
# interpolated build (the library's default), the W and X rules of the defaults, and the
# defaults' M2L with the cost model's V rule (`--m2l auto-pw --near-v cost`). Each launch runs,
# under `--diagonal off,auto`, `--v-raised pair,box` × `--v-after-plan off,raised` (the per-pair
# price, the trusted measure of the second plan on the raised levels, the per-box price, and
# the per-box price with the second plan, which shows what it leaves). Stages:
#   w1        W1 at kD = 100, 200, 500 at ε 10⁻⁷ and 10⁻⁸
#   w3        W3 at N = 10⁶, kD = 200 and 500, the same ε
#   reverse   the w1 and w3 cases with every list reversed (`--diagonal auto,off --v-raised
#             box,pair --v-after-plan raised,off`), to separate a run's place in the launch
#             from the setting's effect
#   counts    the w1 and w3 cases under the defaults alone with `--v-raised pair,box` (the count's
#             build time after it ran on the pool, deec3e0 onwards; the plans are unchanged)
#   smoke     W1 at kD = 30, 10⁻⁷, the runs of w1 (a check of the script)
# TIMEOUT per launch (default 3600 s), ONLY (a regular expression on the launch names),
# DRY_RUN (print the launches only).
# Each launch writes <out-dir>/<name>.md and .err and a line in <out-dir>/runs.log.
set -u
out=$1
shift
mkdir -p "$out"
target=${CARGO_TARGET_DIR:-target}/release/examples
A=$target/helmholtz_accuracy
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS--darwin}
    cores=$(sysctl -n hw.perflevel0.physicalcpu 2> /dev/null || echo 6)
    MPI=(mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 --bind-to none
        --host "localhost:$cores" --map-by slot -n 1)
else
    # GNU nproc honours OMP_NUM_THREADS: count the cores without it.
    cores=$(env -u OMP_NUM_THREADS nproc)
    MPI=(mpirun --report-bindings --map-by "slot:PE=$cores" --bind-to core -n 1)
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TIMEOUT=${TIMEOUT:-3600}
COMMON=(--near-wx cost --levels --threads "$cores" --evaluations 2
    --diagonal-build interpolated)
RUNS=(--m2l auto-pw --near-v cost --diagonal off,auto --v-raised pair,box
    --v-after-plan off,raised)
COUNTS=(--m2l auto-pw --near-v cost --v-raised pair,box)
REVERSE=(--m2l auto-pw --near-v cost --diagonal auto,off --v-raised box,pair
    --v-after-plan raised,off)

# `launch <name> <arguments…>`: one launch of the example under the timeout.
launch() {
    local n=$1
    shift
    [[ "$n" =~ ${ONLY:-.} ]] || return 0
    if [ -n "${DRY_RUN:-}" ]; then
        echo "$n: $*"
        return 0
    fi
    local start=$SECONDS
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S') $(hostname) job ${SLURM_JOB_ID:-none}"
        echo "# ${MPI[*]} $A ${COMMON[*]} $*"
    } > "$out/$n.err"
    timeout "$TIMEOUT" "${MPI[@]}" "$A" "${COMMON[@]}" "$@" < /dev/null > "$out/$n.md" \
        2>> "$out/$n.err"
    local status=$?
    echo "$n: exit $status, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

EPS="1e-7 1e-8"
w1() {
    local kd eps
    for kd in 100 200 500; do
        for eps in $EPS; do
            launch "w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps" "${RUNS[@]}"
        done
    done
}
w3() {
    local kd eps
    for kd in 200 500; do
        for eps in $EPS; do
            launch "w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps" "${RUNS[@]}"
        done
    done
}
counts() {
    local kd eps
    for kd in 100 200 500; do
        for eps in $EPS; do
            launch "counts-w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps" "${COUNTS[@]}"
        done
    done
    for kd in 200 500; do
        for eps in $EPS; do
            launch "counts-w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps" \
                "${COUNTS[@]}"
        done
    done
}
reverse() {
    local kd eps
    for kd in 100 200 500; do
        for eps in $EPS; do
            launch "reverse-w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps" "${REVERSE[@]}"
        done
    done
    for kd in 200 500; do
        for eps in $EPS; do
            launch "reverse-w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps" \
                "${REVERSE[@]}"
        done
    done
}

for stage in "$@"; do
    case $stage in
    smoke) launch smoke --workload w1 --kd 30 --eps 1e-7 "${RUNS[@]}" ;;
    w1) w1 ;;
    w3) w3 ;;
    reverse) reverse ;;
    counts) counts ;;
    *)
        echo "ch4-3b2d-runs.sh: unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
