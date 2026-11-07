#!/bin/bash
# tools/helmholtz/ch6-2d-runs.sh: the machine runs of Helmholtz CH6.2d
# (docs/helmholtz/tasks/CH6.2d.md, "Machine runs"): FMM3D's exponential M2L per level, timed
# and level by level in accuracy, on CH6.2b's problems. From the repository root, after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_fmm3d
#     VARIANTS="default fastker timed fastker-timed" tools/fmm3d/build.sh build
#     tools/helmholtz/ch6-2d-runs.sh <out-dir> <stage…>
#
# Stages:
#   check   the patch changes no value: on W1 at kD = 30 and the W4 cube at kD = 100, both
#           eps, `timed` against `default` and `fastker-timed` against `fastker`, FULL = 1,
#           the full outputs compared byte for byte (`cmp`); a line per pair in
#           <out-dir>/check.log, exit 1 if any differs
#   times   per case of CASES (default CH6.2b's matrix and large cases) and variant of
#           VARIANTS (default "timed fastker-timed"), the driver at EPS with REPEAT 2 on
#           THREADS OpenMP threads, twice (launches a and b); the `hfmm3d-level:` lines in
#           <out-dir>/times-<case>-<variant>-<a|b>.log, launch a's errors at the samples
#           (`helmholtz_fmm3d errors`) in <out-dir>/errors-<case>.md
#   scan    per case of SCAN_CASES (default the matrix), the variant `timed`: FMM3D as it
#           is, FMM3D_PW_LEVEL=-1 (every level on point-and-shoot), then each level the
#           first run put on plane waves (at any eps) alone; then `helmholtz_fmm3d errors`
#           over them into <out-dir>/scan-<case>.md (each run's errors over the
#           point-and-shoot run's)
#
# A case is `<workload>@<kD>` with the example's workload names. The problem files are
# written first into <out-dir>/problems/ (`helmholtz_fmm3d problem`). FMM3D_ROOT as for
# tools/fmm3d/build.sh (on Kathleen $FMM_KATHLEEN_ROOT/fmm3d, on locust /data/ucahtbe/fmm3d);
# THREADS (default every core), EPS (default 1e-3,1e-6), TIMEOUT per launch (default 3600 s),
# ONLY (a regular expression on the launch names). BLAS threads 1. No quiet-machine gate:
# on the M6 run `check` and `scan` only (accuracy), or wait for the user's go-ahead.
set -u
out=$1
shift
mkdir -p "$out/problems" "$out/fmm3d"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=${CARGO_TARGET_DIR:-target}/release/examples/helmholtz_fmm3d
if [ -z "${FMM3D_ROOT:-}" ]; then
    if [ -n "${FMM_KATHLEEN_ROOT:-}" ]; then
        FMM3D_ROOT=$FMM_KATHLEEN_ROOT/fmm3d
    elif [ -d /data/ucahtbe ]; then
        FMM3D_ROOT=/data/ucahtbe/fmm3d
    else
        echo "ch6-2d-runs.sh: set FMM3D_ROOT as for tools/fmm3d/build.sh" >&2
        exit 2
    fi
fi
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
    cores=$(sysctl -n hw.perflevel0.physicalcpu 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 6)
else
    cores=$(nproc)
fi
T=${THREADS:-$cores}
EPS=${EPS:-1e-3,1e-6}
TIMEOUT=${TIMEOUT:-3600}
VARIANTS=${VARIANTS:-"timed fastker-timed"}
MATRIX="w1@10 w1@30 w1@100 w3@10 w3@30"
for d in cube sphere plummer clusters; do
    MATRIX="$MATRIX w4-$d:100000@1 w4-$d:100000@100"
done
CASES=${CASES:-"$MATRIX w1@500 w3@100"}
SCAN_CASES=${SCAN_CASES:-$MATRIX}

# `name <case>`: the case's file name, `w4-cube:100000@1` -> `w4-cube-1`.
name() {
    local workload=${1%@*}
    echo "${workload%%:*}-${1#*@}"
}

# `problem <case>`: writes the case's problem file once, prints its path; fails (in its
# command substitution, so the caller tests it) without one.
problem() {
    local workload=${1%@*} kd=${1#*@} file
    file=$out/problems/$(name "$1").bin
    if [ ! -f "$file" ]; then
        "$X" problem --workload "$workload" --kd "$kd" --out "$file" >> "$out/problems.log" ||
            { echo "ch6-2d-runs.sh: no problem file for $1" >&2; exit 1; }
    fi
    echo "$file"
}

# `driver <log> <variant> <problem> <prefix> <kD> <repeat> <full>`: one driver launch on T
# threads under the timeout, its output appended to <log>.
driver() {
    local log=$1 variant=$2 prob=$3 pre=$4 kd=$5 repeat=$6 full=$7 start=$SECONDS
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S') $(hostname) FMM3D_PW_LEVEL=${FMM3D_PW_LEVEL:-unset}"
        echo "# OMP_NUM_THREADS=$T $FMM3D_ROOT/$variant/bin/fmm3d_driver helmholtz $prob $pre $kd 0 $EPS $repeat $full"
        [ "$(uname)" = Darwin ] || echo "# load before: $(uptime)"
    } >> "$log"
    OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$FMM3D_ROOT/$variant/bin/fmm3d_driver" helmholtz \
        "$prob" "$pre" "$kd" 0 "$EPS" "$repeat" "$full" < /dev/null >> "$log" 2>&1
    local status=$?
    echo "$(basename "$log") ${FMM3D_PW_LEVEL:-}: exit $status, $((SECONDS - start)) s, $(date +%T)" |
        tee -a "$out/runs.log"
    return $status
}

check() {
    local status=0 c v w prob pre t
    for c in w1@30 w4-cube:100000@100; do
        prob=$(problem "$c") || return 1
        for v in default timed fastker fastker-timed; do
            pre=$out/fmm3d/check-$(name "$c")-$v
            driver "$pre.log" "$v" "$prob" "$pre" "${c#*@}" 1 1 || status=1
        done
        for pair in default:timed fastker:fastker-timed; do
            v=${pair%:*} w=${pair#*:}
            for t in ${EPS//,/ }; do
                if cmp -s "$out/fmm3d/check-$(name "$c")-$v-$t.full.out" \
                    "$out/fmm3d/check-$(name "$c")-$w-$t.full.out"; then
                    echo "$c eps $t: $w the same as $v, byte for byte" | tee -a "$out/check.log"
                else
                    echo "$c eps $t: $w DIFFERS from $v" | tee -a "$out/check.log"
                    status=1
                fi
            done
        done
        rm -f "$out/fmm3d/check-$(name "$c")"-*.full.out
    done
    return $status
}

times() {
    local c v run prob pre labels
    for c in $CASES; do
        prob=$(problem "$c") || continue
        labels=
        for run in a b; do
            for v in $VARIANTS; do
                pre=$out/times-$(name "$c")-$v-$run
                [[ "times-$(name "$c")-$v-$run" =~ ${ONLY:-.} ]] || continue
                driver "$pre.log" "$v" "$prob" "$pre" "${c#*@}" 2 0
                [ "$run" = a ] && labels="$labels,$v=$pre"
            done
        done
        [ -n "$labels" ] || continue
        timeout "$TIMEOUT" "$X" errors --workload "${c%@*}" --kd "${c#*@}" --eps "$EPS" \
            --problem "$prob" --threads "$T" --fmm3d "${labels#,}" < /dev/null \
            > "$out/errors-$(name "$c").md" 2>> "$out/errors-$(name "$c").err" ||
            echo "errors-$(name "$c"): errors failed" | tee -a "$out/runs.log"
        rm -f "$out/times-$(name "$c")"-*.out
    done
}

scan() {
    local c n prob pre levels l labels
    for c in $SCAN_CASES; do
        n=$(name "$c")
        [[ "scan-$n" =~ ${ONLY:-.} ]] || continue
        prob=$(problem "$c") || continue
        pre=$out/fmm3d/scan-$n
        driver "$pre-fmm3d.log" timed "$prob" "$pre-fmm3d" "${c#*@}" 1 0
        FMM3D_PW_LEVEL=-1 driver "$pre-ps.log" timed "$prob" "$pre-ps" "${c#*@}" 1 0
        # The levels FMM3D put on plane waves at any eps (mode 1), after the driver's
        # two-point check.
        levels=$(awk '/^helmholtz: problem/ { on = 1 }
            on && /hfmm3d-level:/ {
                for (i = 1; i < NF; i++) {
                    if ($i == "ilev=") l = $(i + 1)
                    if ($i == "mode=") m = $(i + 1)
                }
                if (m == 1) print l
            }' "$pre-fmm3d.log" | sort -n -u | tr '\n' ' ')
        labels="ps=$pre-ps,fmm3d=$pre-fmm3d"
        for l in $levels; do
            FMM3D_PW_LEVEL=$l driver "$pre-pw$l.log" timed "$prob" "$pre-pw$l" "${c#*@}" 1 0
            labels="$labels,pw$l=$pre-pw$l"
        done
        timeout "$TIMEOUT" "$X" errors --workload "${c%@*}" --kd "${c#*@}" --eps "$EPS" \
            --problem "$prob" --threads "$T" --fmm3d "$labels" < /dev/null \
            > "$out/scan-$n.md" 2>> "$out/scan-$n.err" ||
            echo "scan-$n: errors failed" | tee -a "$out/runs.log"
    done
}

for stage in "$@"; do
    case $stage in
        check) check || exit 1 ;;
        times) times ;;
        scan) scan ;;
        *) echo "ch6-2d-runs.sh: unknown stage $stage (check, times or scan)" >&2; exit 2 ;;
    esac
done
