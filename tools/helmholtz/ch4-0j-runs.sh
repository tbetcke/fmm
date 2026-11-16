#!/bin/bash
# tools/helmholtz/ch4-0j-runs.sh: the machine runs of Helmholtz CH4.0j, the exponential M2L's
# tables within a level in parts (docs/helmholtz/tasks/CH4.0j.md, "Machine runs"), from the
# repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-0j-runs.sh <out-dir> <stage…>
#
# Stages:
#   parts   `nd-fmm-tables`' ignored `time_a_levels_parts` (release): a level's build by part
#           on one thread, into <out-dir>/parts.md (on many threads: the matrix's build part
#           "plane-wave tables", on the FMM's pool)
#   matrix  CH3.6's matrix as CH4.0h's (tools/helmholtz/ch4-0h-runs.sh: the library's
#           defaults, two evaluations, the table "Build") at one rank per node (1x40 under
#           Slurm, 1x72 elsewhere; SPLIT names another), each case four launches in the
#           order main, branch, branch, main (matrix-<case>-<split>-<main|branch>-<1|2>):
#           `main`'s binary from MAIN (a directory holding `main`'s `helmholtz_accuracy`, for
#           example another tree's target/release/examples); without MAIN the branch twice
# CASES, ONLY, TIMEOUT and DRY_RUN as in ch4-0h-runs.sh. Every launch's checksum must equal
# CH4.0h's of the same case (tools/helmholtz/ch4-0j-tables.py). Times are read on Kathleen's
# exclusive nodes and a quiet locust only, never asserted; no run on the M6 without the
# user's go-ahead.
set -u
out=$1
shift
mkdir -p "$out"
out=$(cd "$out" && pwd)
target=${CARGO_TARGET_DIR:-target}/release/examples
B=$target/helmholtz_accuracy
A=${MAIN:+$MAIN/helmholtz_accuracy}
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS--darwin}
    cores=$(sysctl -n hw.perflevel0.physicalcpu 2> /dev/null || echo 6)
else
    # GNU nproc honours OMP_NUM_THREADS: count the cores without it.
    cores=$(env -u OMP_NUM_THREADS nproc)
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ -n "${SLURM_JOB_ID:-}" ]; then
    slurm=1
    SPLIT=${SPLIT:-1x40}
else
    slurm=0
    SPLIT=${SPLIT:-1x72}
fi
TIMEOUT=${TIMEOUT:-3600}
EPS=1e-3,1e-6,1e-8
matrix="w1@10 w1@30 w1@100 w1@500 w3@10 w3@30 w3@100"
for d in cube sphere plummer clusters; do
    for kd in 1 10 100; do matrix="$matrix w4-$d:100000@$kd"; done
done
CASES=${CASES:-$matrix}
DEFAULTS=(--m2l auto-pw --near-v cost --near-wx cost)

# `name <case>`: the case's file name, `w4-cube:100000@1` -> `w4-cube-1`.
name() {
    local workload=${1%@*}
    echo "${workload%%:*}-${1#*@}"
}

# `launch <name> <command…>`: one launch under the timeout, its output in <name>.md and
# <name>.err.
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
        echo "# $*"
        [ "$slurm" = 1 ] || echo "# load before: $(uptime)"
    } > "$out/$n.err"
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$n.md" 2>> "$out/$n.err"
    local status=$?
    [ "$slurm" = 1 ] || echo "# load after: $(uptime)" >> "$out/$n.err"
    echo "$n: exit $status, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

# `mpi <ranks> <threads>`: the launcher of a split (ch4-0h-runs.sh's).
mpi() {
    local r=$1 t=$2
    if [ "$(uname)" = Darwin ]; then
        echo "mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 --bind-to none --host localhost:$cores --map-by slot -n $r"
    elif [ "$slurm" = 1 ] && [ $((r % 2)) -eq 0 ]; then
        echo "mpirun --report-bindings --map-by ppr:$((r / 2)):socket:PE=$t --bind-to core -n $r"
    else
        echo "mpirun --report-bindings --map-by slot:PE=$t --bind-to core -n $r"
    fi
}

for stage in "$@"; do
    case $stage in
    parts)
        launch parts cargo test -p nd-fmm-tables --release --test helmholtz -- --ignored \
            --nocapture --exact parts::time_a_levels_parts
        ;;
    matrix)
        r=${SPLIT%x*} t=${SPLIT#*x}
        if [ -n "$A" ]; then
            order="main:$A:1 branch:$B:1 branch:$B:2 main:$A:2"
        else
            order="branch:$B:1 branch:$B:2"
        fi
        # Word splitting of CASES and order is intended.
        # shellcheck disable=SC2086
        for c in $CASES; do
            for o in $order; do
                IFS=: read -r who bin i <<< "$o"
                # shellcheck disable=SC2046
                launch "matrix-$(name "$c")-$SPLIT-$who-$i" $(mpi "$r" "$t") "$bin" \
                    --workload "${c%@*}" --kd "${c#*@}" --eps "$EPS" "${DEFAULTS[@]}" \
                    --levels --evaluations 2 --threads "$t"
            done
        done
        ;;
    *)
        echo "ch4-0j-runs.sh: unknown stage $stage (parts or matrix)" >&2
        exit 2
        ;;
    esac
done
