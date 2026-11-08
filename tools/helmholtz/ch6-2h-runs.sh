#!/bin/bash
# tools/helmholtz/ch6-2h-runs.sh: the machine runs of Helmholtz CH6.2h (D8)
# (docs/helmholtz/tasks/CH6.2h.md, "Machine runs"): CH6.2b's stages, from the repository
# root after the example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch6-2h-runs.sh <out-dir> <stage…>
#
# Stages: `matrix` (CH6.1e's matrix at ε = 1e-3 and 1e-6: W1 at kD = 10, 30, 100, W3 at
# kD = 10, 30, the four W4 distributions at N = 10⁵ and kD = 1, 100; one rank × THREADS
# threads) and `large` (W1 at kD = 500 and W3 at kD = 100 at the same ε, at each split of
# SPLITS, ranks x threads, default "1x$THREADS"). Each case is one launch with
# `--m2l rotation,auto,auto-pw,planewave` (EXTRA appends entries, such as `auto-pw:16` on
# Kathleen), then one with the order reversed (ABBA),
# `--evaluations 2 --levels`: per run and level the M2L's seconds of the second evaluation.
# The default waves budget (8 GiB per node); BLAS threads 1. Each launch's output in
# <out-dir>/<stage>-<case>-<split>-<order>.md, a line per launch in <out-dir>/runs.log. A
# split with one rank runs without mpirun; several ranks run `mpirun -n R --map-by
# slot:PE=T --bind-to core` (MPIRUN_FLAGS adds flags). On macOS every launch waits until
# the other processes use at most QUIET % CPU (default 150), samples them every 5 s during
# the launch, and repeats a launch above DURING % (default 200) up to RETRIES times
# (default 2), as CH3.2d's gate; there the launcher is `mpirun --host localhost:6 --map-by
# slot --bind-to none` with the loopback flags. TIMEOUT (default 3600) bounds a launch;
# ONLY (a regular expression on the launch names) runs a subset.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
SPLITS=${SPLITS:-1x$T}
QUIET=${QUIET:-150}
DURING=${DURING:-200}
RETRIES=${RETRIES:-2}
TIMEOUT=${TIMEOUT:-3600}
macos=false
if [ "$(uname)" = Darwin ]; then
    macos=true
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
fi
forward="rotation,auto,auto-pw,planewave${EXTRA:+,$EXTRA}"
reverse=$(echo "$forward" | awk -F, '{ for (i = NF; i > 1; i--) printf "%s,", $i; print $1 }')
common=(--eps 1e-3,1e-6 --evaluations 2 --levels)

# The CPU percentage of every process but ps itself outside this script's process group
# and the launch's (`others <pid>`: `timeout` runs the command in a group of its own,
# whose id is its pid, so the example and anything it starts are left out).
group=$(ps -o pgid= -p $$ | tr -d ' ')
others() {
    ps -A -o %cpu=,pgid=,comm= | awk -v me="$group" -v launch="${1:-0}" \
        '$2 != me && $2 != launch && $3 !~ /(^| |\/)ps$/ { s += $1 } END { printf "%d", s }'
}

# The launcher of a split `RxT` (an array; empty for one rank).
launcher() {
    local ranks=${1%x*} threads=${1#*x}
    if [ "$ranks" -eq 1 ]; then
        return
    elif $macos; then
        echo mpirun -n "$ranks" --host localhost:6 --map-by slot --bind-to none \
            --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 ${MPIRUN_FLAGS:-}
    else
        echo mpirun -n "$ranks" --map-by "slot:PE=$threads" --bind-to core ${MPIRUN_FLAGS:-}
    fi
}

# `launch <name> <command…>`: one launch under the timeout (and the gate on macOS).
launch() {
    local name=$1
    shift
    local attempt=0
    while :; do
        if $macos; then
            while [ "$(others)" -gt "$QUIET" ]; do sleep 10; done
        fi
        local before start=$SECONDS
        before=$($macos && others || echo -)
        timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err" &
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
        echo "$name: exit $status, $((SECONDS - start)) s, others ${before}% before," \
            "${peak}% peak during, $(date +%T)" | tee -a "$out/runs.log"
        if $macos && [ "$peak" -gt "$DURING" ] && [ "$attempt" -lt "$RETRIES" ]; then
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
        matrix)
            echo "w1|--workload w1 --kd 10,30,100"
            echo "w3|--workload w3 --kd 10,30"
            for d in cube sphere plummer clusters; do
                echo "w4-$d|--workload w4-$d:100000 --kd 1,100"
            done
            ;;
        large)
            echo "w1-500|--workload w1 --kd 500"
            echo "w3-100|--workload w3 --kd 100"
            ;;
    esac
}

for stage in "$@"; do
    case $stage in
        matrix) splits="1x$T" ;;
        large) splits=$SPLITS ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
    for split in $splits; do
        threads=${split#*x}
        # Word splitting of the launcher is intended.
        # shellcheck disable=SC2207
        run=($(launcher "$split"))
        while IFS= read -r entry; do
            name=${entry%%|*}
            for order in forward reverse; do
                [[ "$stage-$name-$split-$order" =~ ${ONLY:-.} ]] || continue
                m2l=$forward
                [ "$order" = reverse ] && m2l=$reverse
                # shellcheck disable=SC2086
                launch "$stage-$name-$split-$order" ${run[@]+"${run[@]}"} "$X" ${entry#*|} \
                    "${common[@]}" --threads "$threads" --m2l "$m2l"
            done
        done < <(cases "$stage")
    done
done
