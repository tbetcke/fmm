#!/bin/bash
# tools/helmholtz/ch4-3a-runs.sh: the machine runs of Helmholtz CH4.3a, the diagonal form's
# bound constants in the full FMM (docs/helmholtz/tasks/CH4.3a.md, "Machine runs"), from the
# repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-3a-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` on one rank with every core (Kathleen 1 × 40), with
# `--levels` (the tables "M2L per level" and "Diagonal M2L"), the near rules of the defaults,
# §9.1's complex charges and samples, one case (kD, ε) under `--m2l rotation` and
# `--diagonal off,rule,<s ε>…`: Rotation without the diagonal form (the reference of
# "÷ Rotation"), the rule at the FMM's tolerance (s = 1), then the rule at s ε, that is at
# θ′ = s θ (the margin). The transfer functions are built as the library's default builds
# them, interpolated (`--diagonal-build interpolated`: the example's own default is the direct
# build), unless a stage names both. Stages:
#   w1        W1 at kD = 100, 200, 500 (500 with both builds), ε 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁸
#   w1-large  W1 at kD = 1000, ε 10⁻⁶ and 10⁻⁸, s = 1, 16, 256
#   w3        W3 at N = 10⁶, kD = 200 and 500, the four ε
#   densities the constant density on W1 at kD = 100, the plane wave's trace at kD = 500, the
#             four ε, s = 1, 16, 256
#   damped    W1 at kD = 100 + 10i, 500 + 50i (the four ε) and 100 + 100i (10⁻³, 10⁻⁶)
#   smoke     W1 at kD = 30, 10⁻⁴ (a check of the script)
# SCALES (default "4 16 64 256 1024"), TIMEOUT per launch (default 3600 s), ONLY (a regular
# expression on the launch names), DRY_RUN (print the launches only). Each launch writes
# <out-dir>/<name>.md and .err and a line in <out-dir>/runs.log.
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
SCALES=${SCALES:-4 16 64 256 1024}
COMMON=(--near-v cost --near-wx cost --levels --m2l rotation --threads "$cores")
INTERP=(--diagonal-build interpolated)

# `diagonal <ε> <scale…>`: off, the rule, and the rule at s ε for each scale s.
diagonal() {
    local eps=$1 list=off,rule s
    shift
    for s in "$@"; do
        list+=,$(python3 -c "print(repr($s * $eps))")
    done
    echo "$list"
}

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

EPS="1e-3 1e-4 1e-6 1e-8"
# shellcheck disable=SC2086
for stage in "$@"; do
    case $stage in
    smoke)
        launch smoke --workload w1 --kd 30 --eps 1e-4 "${INTERP[@]}" --diagonal "$(diagonal 1e-4 16)"
        ;;
    w1)
        for kd in 100 200 500; do
            build=("${INTERP[@]}")
            [ "$kd" = 500 ] && build=(--diagonal-build direct,interpolated)
            for eps in $EPS; do
                launch "w1-$kd-$eps" --workload w1 --kd "$kd" --eps "$eps" \
                    --diagonal "$(diagonal "$eps" $SCALES)" "${build[@]}"
            done
        done
        ;;
    w1-large)
        for eps in 1e-6 1e-8; do
            launch "w1-1000-$eps" --workload w1 --kd 1000 --eps "$eps" \
                "${INTERP[@]}" --diagonal "$(diagonal "$eps" 16 256)"
        done
        ;;
    w3)
        for kd in 200 500; do
            for eps in $EPS; do
                launch "w3-$kd-$eps" --workload w3:1000000 --kd "$kd" --eps "$eps" \
                    "${INTERP[@]}" --diagonal "$(diagonal "$eps" $SCALES)"
            done
        done
        ;;
    densities)
        for eps in $EPS; do
            launch "constant-100-$eps" --workload w1 --kd 100 --eps "$eps" \
                --density constant "${INTERP[@]}" --diagonal "$(diagonal "$eps" 16 256)"
            launch "planewave-500-$eps" --workload w1 --kd 500 --eps "$eps" \
                --density planewave "${INTERP[@]}" --diagonal "$(diagonal "$eps" 16 256)"
        done
        ;;
    damped)
        for c in 100:10 500:50; do
            for eps in $EPS; do
                launch "damped-${c%:*}+${c#*:}i-$eps" --workload w1 --kd "${c%:*}" \
                    --kim "${c#*:}" --eps "$eps" "${INTERP[@]}" --diagonal "$(diagonal "$eps" $SCALES)"
            done
        done
        for eps in 1e-3 1e-6; do
            launch "damped-100+100i-$eps" --workload w1 --kd 100 --kim 100 --eps "$eps" \
                "${INTERP[@]}" --diagonal "$(diagonal "$eps" $SCALES)"
        done
        ;;
    *)
        echo "ch4-3a-runs.sh: unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
