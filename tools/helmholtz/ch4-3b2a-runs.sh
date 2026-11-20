#!/bin/bash
# tools/helmholtz/ch4-3b2a-runs.sh: the machine runs of Helmholtz CH4.3b2a, the diagonal form's
# time rule at matched accuracy (docs/helmholtz/tasks/CH4.3b2a.md, "Machine runs"), from the
# repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-3b2a-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` on one rank with every core (Kathleen 1 × 40, locust
# 1 × 72), with `--levels`, the near rules of the defaults, §9.1's complex charges and samples,
# two evaluations (the rows the second's), the interpolated build (the library's default), one
# case (kD, ε) under the library's defaults (`--m2l auto-pw`) with `--diagonal off,auto` and
# `--diagonal-costs machine,form,rcr`: the defaults (the waves), the time rule at the machine's
# costs, and the time rule forcing the form (where a class passes, else RCR) and RCR on every
# level it reaches, all at the levels' resolving degrees. `tools/helmholtz/ch4-3b2a-tables.py`
# makes the report's tables. Stages:
#   w1        W1 at kD = 100, 200, 500 at ε 10⁻³, 10⁻⁶, 10⁻⁷, 10⁻⁸, and kD = 1000 at 10⁻⁶
#   w3        W3 at N = 10⁶, kD = 200 and 500, at ε 10⁻³, 10⁻⁶, 10⁻⁷, 10⁻⁸
#   smoke     W1 at kD = 30, 10⁻⁷ (a check of the script)
# TIMEOUT per launch (default 3600 s), ONLY (a regular expression on the launch names),
# DRY_RUN (print the launches only), COSTS (the `--diagonal-costs` list, default
# machine,form,rcr).
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
COMMON=(--near-v cost --near-wx cost --levels --m2l auto-pw --threads "$cores"
    --evaluations 2 --diagonal-build interpolated --diagonal off,auto
    --diagonal-costs "${COSTS:-machine,form,rcr}")

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

EPS="1e-3 1e-6 1e-7 1e-8"
w1() {
    local kd eps
    for kd in 100 200 500; do
        for eps in $EPS; do
            launch "w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps"
        done
    done
    launch "w1-1000-1e-6" --workload w1 --kd 1000 --eps 1e-6
}
w3() {
    local kd eps
    for kd in 200 500; do
        for eps in $EPS; do
            launch "w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps"
        done
    done
}

for stage in "$@"; do
    case $stage in
    smoke) launch smoke --workload w1 --kd 30 --eps 1e-7 ;;
    w1) w1 ;;
    w3) w3 ;;
    *)
        echo "ch4-3b2a-runs.sh: unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
