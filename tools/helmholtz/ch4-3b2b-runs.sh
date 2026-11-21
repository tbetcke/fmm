#!/bin/bash
# tools/helmholtz/ch4-3b2b-runs.sh: the machine runs of Helmholtz CH4.3b2b, the V near rule on
# the levels the time rule decides after the plan, and the defaults' errors at 10⁻⁶
# (docs/helmholtz/tasks/CH4.3b2b.md, "Machine runs"), from the repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-3b2b-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` on one rank with every core (Kathleen 1 × 40), with
# `--levels`, §9.1's complex charges and samples, two evaluations (the rows the second's), the
# interpolated build (the library's default), the W and X rules of the defaults. Stages:
#   errors    W1 at kD = 100, 200, 500, 1000 and W3 at N = 10⁶, kD = 200 and 500, at 10⁻³ and
#             10⁻⁶: Rotation and the defaults (`--m2l rotation,auto-pw`), each with the V rule
#             off and of the cost model (`--near-v off,cost`): the explanation of the defaults'
#             errors against Rotation's
#   w1        W1 at kD = 100, 200, 500 at ε 10⁻³, 10⁻⁶, 10⁻⁷, 10⁻⁸, and kD = 1000 at 10⁻⁶: Rotation,
#             then the defaults with `--diagonal off,auto` × `--v-after-plan off,taken,all`
#   w3        W3 at N = 10⁶, kD = 200 and 500, the same ε and runs
#   reverse   the defaults' runs of w1 and of W3 at kD = 200, 10⁻⁶ and 10⁻⁷, in the reverse order
#             (`--diagonal auto,off --v-after-plan all,taken,off`), to separate a run's place in
#             the launch from the setting's effect
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
# The V rule's runs: Rotation (its plan) first, then the defaults with the time rule and the
# V rule after the plan.
ROTATION=(--m2l rotation --near-v cost)
DEFAULTS=(--m2l auto-pw --near-v cost --diagonal off,auto --v-after-plan off,taken,all)
REVERSE=(--m2l auto-pw --near-v cost --diagonal auto,off --v-after-plan all,taken,off)

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

# `pair <name> <case arguments…>`: Rotation's plan, then the defaults' runs, in one launch each.
pair() {
    local n=$1
    shift
    launch "$n-rotation" "$@" "${ROTATION[@]}"
    launch "$n" "$@" "${DEFAULTS[@]}"
}

EPS="1e-3 1e-6 1e-7 1e-8"
errors() {
    local kd eps
    for eps in 1e-6 1e-3; do
        for kd in 100 200 500 1000; do
            launch "errors-w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps" \
                --m2l rotation,auto-pw --near-v off,cost
        done
        for kd in 200 500; do
            launch "errors-w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps" \
                --m2l rotation,auto-pw --near-v off,cost
        done
    done
}
w1() {
    local kd eps
    for kd in 100 200 500; do
        for eps in $EPS; do
            pair "w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps"
        done
    done
    pair "w1-1000-1e-6" --workload w1 --kd 1000 --eps 1e-6
}
w3() {
    local kd eps
    for kd in 200 500; do
        for eps in $EPS; do
            pair "w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps"
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
    launch "reverse-w1-1000-1e-6" --workload w1 --kd 1000 --eps 1e-6 "${REVERSE[@]}"
    for eps in 1e-6 1e-7; do
        launch "reverse-w3-200-$eps" --workload w3:1000000 --kd 200 --eps "$eps" "${REVERSE[@]}"
    done
}

for stage in "$@"; do
    case $stage in
    smoke) pair smoke --workload w1 --kd 30 --eps 1e-7 ;;
    errors) errors ;;
    w1) w1 ;;
    w3) w3 ;;
    reverse) reverse ;;
    *)
        echo "ch4-3b2b-runs.sh: unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
