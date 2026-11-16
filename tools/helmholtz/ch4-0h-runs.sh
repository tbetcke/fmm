#!/bin/bash
# tools/helmholtz/ch4-0h-runs.sh: the machine runs of Helmholtz CH4.0h, the build by part and
# in parallel (docs/helmholtz/tasks/CH4.0h.md, "Machine runs"), from the repository root
# after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-0h-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` at the library's defaults as CH4.0e's did
# (`--m2l auto-pw --near-v cost --near-wx cost`, the leaf size and c_leaf the library's), with
# its table "Build" (the build by part, each the slowest rank's) and the checksums, which
# must equal CH4.0e's launches of the same case. Stages:
#   build   W1 at kD = 500 at 10⁻³, 10⁻⁶, 10⁻⁸ on one core (1 x 1) and on every split of
#           SPLITS, in that order and again reversed (launch names build-<split>-a, -b); one
#           evaluation (the build is what is measured; the one-core evaluation is minutes)
#   matrix  CH3.6's matrix (CASES) at EPS on MATRIX_SPLITS (default the first and third of
#           SPLITS), one launch each (matrix-<case>-<split>), two evaluations as CH4.0e
# SPLITS defaults to "1x40 2x20 4x10 8x5" under Slurm and "1x72 2x36 4x18 8x9" elsewhere
# (CH4.0e's); under Slurm an even rank count is bound per socket (`ppr:<r/2>:socket`),
# elsewhere `slot:PE=<t>`. TIMEOUT per launch (default 3600 s), ONLY (a regular expression on
# the launch names), DRY_RUN (print the launches only). Each launch writes <out-dir>/<name>.md
# and .err and a line in <out-dir>/runs.log. Times are read on Kathleen's exclusive nodes and
# a quiet locust only, never asserted; no run on the M6 without the user's go-ahead.
set -u
out=$1
shift
mkdir -p "$out"
target=${CARGO_TARGET_DIR:-target}/release/examples
A=$target/helmholtz_accuracy
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
    SPLITS=${SPLITS:-"1x40 2x20 4x10 8x5"}
else
    slurm=0
    SPLITS=${SPLITS:-"1x72 2x36 4x18 8x9"}
fi
read -r -a split_list <<< "$SPLITS"
MATRIX_SPLITS=${MATRIX_SPLITS:-"${split_list[0]} ${split_list[2]}"}
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

# `mpi <ranks> <threads>`: the launcher of a split.
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

# `reversed <words…>`: the words in reverse order.
reversed() { printf '%s\n' "$@" | awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) printf "%s%s", a[i], (i > 1 ? " " : "\n") }'; }

for stage in "$@"; do
    case $stage in
    build)
        # Word splitting of SPLITS is intended.
        # shellcheck disable=SC2086
        for order in a b; do
            splits="1x1 $SPLITS"
            [ "$order" = b ] && splits=$(reversed 1x1 $SPLITS)
            for s in $splits; do
                r=${s%x*} t=${s#*x}
                # Every launch's oracle runs on its own bound cores (one in the one-core
                # launch; untimed).
                # shellcheck disable=SC2046
                launch "build-$s-$order" $(mpi "$r" "$t") "$A" --workload w1 --kd 500 \
                    --eps "$EPS" "${DEFAULTS[@]}" --threads "$t"
            done
        done
        ;;
    matrix)
        # Word splitting of CASES and MATRIX_SPLITS is intended.
        # shellcheck disable=SC2086
        for c in $CASES; do
            for s in $MATRIX_SPLITS; do
                r=${s%x*} t=${s#*x}
                # shellcheck disable=SC2046
                launch "matrix-$(name "$c")-$s" $(mpi "$r" "$t") "$A" --workload "${c%@*}" \
                    --kd "${c#*@}" --eps "$EPS" "${DEFAULTS[@]}" --levels --evaluations 2 \
                    --threads "$t"
            done
        done
        ;;
    *)
        echo "ch4-0h-runs.sh: unknown stage $stage (build or matrix)" >&2
        exit 2
        ;;
    esac
done
