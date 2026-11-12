#!/bin/bash
# tools/helmholtz/ch4-0-runs.sh: the machine runs of Helmholtz CH4.0
# (docs/helmholtz/tasks/CH4.0.md, "Machine runs"), from the repository root after the
# examples were built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_m2l --example helmholtz_accuracy
#     tools/helmholtz/ch4-0-runs.sh pairs <out-dir> <isa…>
#     tools/helmholtz/ch4-0-runs.sh fmm <out-dir> <stage…>
#
# `pairs`: for each ISA, `helmholtz_m2l` per pair at every degree of PS (default 3–15) and
# every column count of COLUMNS (default 1 2 4 8 16 32 64), real k (one and two planes);
# the column counts ascending, then a second launch of each descending. One thread.
#
# `fmm`, `helmholtz_accuracy` with `--evaluations 2 --levels`, each case once and once with
# the entries' order reversed (ABBA), at P2P (default `auto`; `avx2` on Kathleen for the AVX2
# column kernel). Stages:
# - `ch3-2d`: CH3.2d's nine cases at one rank × one thread (W1 at kD = 30 and 100, the W4
#   cube of 10⁵ points at kD = 10, each at 10⁻³ and 10⁻⁶; W3 at kD = 30 at 10⁻³; the cube
#   at 10⁻⁶ and W1 at kD = 100 at 10⁻³ with real charges), `--m2l rotation,batched-fused,auto`
#   with `--term on,off` (`auto` is CH3.2d's rule, the exponential M2L off): the rule with
#   and without the term against the per-level best of Rotation and Batched;
# - `matrix`: CH6.2h's matrix at one rank × THREADS (W1 at kD = 10, 30, 100, W3 at kD = 10,
#   30, the four W4 at N = 10⁵ and kD = 1, 100, each at 10⁻³ and 10⁻⁶) and `large` (W1 at
#   kD = 500, W3 at kD = 100), under the default (`auto-pw`, D8) with `--term on,off` and
#   `--block auto,64`, beside Rotation and (`matrix` only) Batched at both widths: the new
#   default against CH6.2h's (term off, blocks of 64) and the per-level best.
# Each launch's output in <out-dir>/<stage>-<case>-<order>.md, a line per launch in
# <out-dir>/runs.log. BLAS threads 1. On macOS (and with GATE=1 elsewhere) every launch
# waits until the other processes use at most QUIET % CPU (default 150), samples them every
# 5 s during the launch, and repeats a launch above DURING % (default 200) up to RETRIES
# times (default 2), as CH6.2h's gate; on Linux `ps`'s %CPU is each process's average over
# its lifetime, so the gate there sees a long-running process's new load late. TIMEOUT (default 3600) bounds a launch; ONLY (a regular expression on the launch
# names) runs a subset.
set -u
mode=$1
out=$2
shift 2
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
bin=target/release/examples
PS=${PS:-3,4,5,6,7,8,9,10,11,12,13,14,15}
COLUMNS=${COLUMNS:-1 2 4 8 16 32 64}
T=${THREADS:-40}
P2P=${P2P:-auto}
QUIET=${QUIET:-150}
DURING=${DURING:-200}
RETRIES=${RETRIES:-2}
TIMEOUT=${TIMEOUT:-3600}
macos=false
if [ "$(uname)" = Darwin ]; then
    macos=true
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
fi
gate=$macos
[ "${GATE:-0}" = 1 ] && gate=true

# The CPU percentage of every process but ps itself outside this script's process group
# and the launch's (`others <pid>`: `timeout` runs the command in a group of its own).
group=$(ps -o pgid= -p $$ | tr -d ' ')
others() {
    ps -A -o %cpu=,pgid=,comm= | awk -v me="$group" -v launch="${1:-0}" \
        '$2 != me && $2 != launch && $3 !~ /(^| |\/)ps$/ { s += $1 } END { printf "%d", s }'
}

# `launch <name> <command…>`: one launch under the timeout (and the gate on macOS).
launch() {
    local name=$1
    shift
    [[ "$name" =~ ${ONLY:-.} ]] || return
    local attempt=0
    while :; do
        if $gate; then
            while [ "$(others)" -gt "$QUIET" ]; do sleep 10; done
        fi
        local before start=$SECONDS
        before=$($gate && others || echo -)
        timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err" &
        local pid=$! peak=0
        if $gate; then
            while kill -0 "$pid" 2> /dev/null; do
                sleep 5
                local now
                now=$(others "$pid")
                [ "$now" -gt "$peak" ] && peak=$now
            done
        fi
        wait "$pid"
        local status=$?
        echo "$name: exit $status, $((SECONDS - start)) s, others ${before}% before," \
            "${peak}% peak during, $(date +%T)" | tee -a "$out/runs.log"
        if $gate && [ "$peak" -gt "$DURING" ] && [ "$attempt" -lt "$RETRIES" ]; then
            attempt=$((attempt + 1))
            echo "$name: contended, repeat $attempt" | tee -a "$out/runs.log"
            continue
        fi
        return
    done
}

# `cases <stage>`: the stage's cases, "name|arguments".
cases() {
    case $1 in
        ch3-2d)
            echo "w1|--workload w1 --kd 30,100 --eps 1e-3,1e-6"
            echo "cube|--workload w4-cube:100000 --kd 10 --eps 1e-3,1e-6"
            echo "w3|--workload w3 --kd 30 --eps 1e-3"
            echo "cube-real|--workload w4-cube:100000 --kd 10 --eps 1e-6 --charges real"
            echo "w1-real|--workload w1 --kd 100 --eps 1e-3 --charges real"
            ;;
        matrix)
            echo "w1|--workload w1 --kd 10,30,100 --eps 1e-3,1e-6"
            echo "w3|--workload w3 --kd 10,30 --eps 1e-3,1e-6"
            for d in cube sphere plummer clusters; do
                echo "w4-$d|--workload w4-$d:100000 --kd 1,100 --eps 1e-3,1e-6"
            done
            ;;
        large)
            echo "w1-500|--workload w1 --kd 500 --eps 1e-3,1e-6"
            echo "w3-100|--workload w3 --kd 100 --eps 1e-3,1e-6"
            ;;
    esac
}

case $mode in
pairs)
    descending=$(echo "$COLUMNS" | tr ' ' '\n' | sort -rn | tr '\n' ' ')
    for isa in "$@"; do
        for c in $COLUMNS; do
            launch "pairs-$isa-c$c-a" "$bin/helmholtz_m2l" --ps "$PS" --k 1 --columns "$c" --isa "$isa"
        done
        for c in $descending; do
            launch "pairs-$isa-c$c-b" "$bin/helmholtz_m2l" --ps "$PS" --k 1 --columns "$c" --isa "$isa"
        done
    done
    ;;
fmm)
    for stage in "$@"; do
        case $stage in
            ch3-2d)
                threads=1
                forward=(--m2l rotation,batched-fused,auto --term on,off)
                reverse=(--m2l auto,batched-fused,rotation --term off,on)
                ;;
            matrix)
                threads=$T
                forward=(--m2l rotation,batched-fused,auto-pw --term on,off --block auto,64)
                reverse=(--m2l auto-pw,batched-fused,rotation --term off,on --block 64,auto)
                ;;
            large)
                threads=$T
                forward=(--m2l rotation,auto-pw --term on,off --block auto,64)
                reverse=(--m2l auto-pw,rotation --term off,on --block 64,auto)
                ;;
            *) echo "unknown stage $stage" >&2; exit 2 ;;
        esac
        while IFS= read -r entry; do
            name=${entry%%|*}
            for order in forward reverse; do
                if [ "$order" = forward ]; then entries=("${forward[@]}"); else entries=("${reverse[@]}"); fi
                # shellcheck disable=SC2086
                launch "$stage-$name-$order" "$bin/helmholtz_accuracy" ${entry#*|} \
                    --threads "$threads" --p2p "$P2P" --evaluations 2 --levels "${entries[@]}"
            done
        done < <(cases "$stage")
    done
    ;;
*)
    echo "usage: ch4-0-runs.sh pairs <out-dir> <isa…> | fmm <out-dir> <stage…>" >&2
    exit 2
    ;;
esac
