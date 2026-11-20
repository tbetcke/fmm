#!/bin/bash
# tools/helmholtz/ch4-2b2-runs.sh: the machine runs of Helmholtz CH4.2b2, the diagonal M2L on
# W1 (docs/helmholtz/tasks/CH4.2b2.md, "Machine runs"), from the repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-2b2-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` with `--levels` (the tables "M2L per level" and
# "Diagonal M2L"), two evaluations of each build (the rows the second's), the near rules of
# the defaults, and ends with rank 0's peak resident size. Stages:
#   w1     W1 at kD = 500 at 10⁻⁶ and 10⁻⁸ and at kD = 1000 at 10⁻⁶, one launch each at one
#          rank with every core: `--m2l rotation,auto-pw` (every level RCR, and the library's
#          defaults) each `--diagonal off,rule`, so that "÷ Rotation" of the diagonal rows is
#          against RCR (the acceptance: within 1.1×), and each level's M2L seconds with and
#          without the diagonal form
#   ranks  W1 at kD = 500 at 10⁻⁶, `--m2l rotation --diagonal rule`, at 2 ranks × half the
#          cores and at one rank per core: the checksum the one-rank run's where the same
#          levels take the diagonal form (constraint 2), and the levels a node budget sends to
#          RCR at many ranks per node
# TIMEOUT per launch (default 5400 s), ONLY (a regular expression on the launch names),
# DRY_RUN (print the launches only). Each launch writes <out-dir>/<name>.md and .err and a
# line in <out-dir>/runs.log.
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
slurm=0
[ -n "${SLURM_JOB_ID:-}" ] && slurm=1
TIMEOUT=${TIMEOUT:-5400}
COMMON=(--workload w1 --near-v cost --near-wx cost --levels --evaluations 2)

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
    } > "$out/$n.err"
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$n.md" 2>> "$out/$n.err"
    local status=$?
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

for stage in "$@"; do
    case $stage in
    w1)
        for c in 500:1e-6 500:1e-8 1000:1e-6; do
            kd=${c%:*} eps=${c#*:}
            # shellcheck disable=SC2046
            launch "w1-$kd-$eps-1x$cores" $(mpi 1 "$cores") "$A" "${COMMON[@]}" --kd "$kd" \
                --eps "$eps" --m2l rotation,auto-pw --diagonal off,rule --threads "$cores"
        done
        ;;
    ranks)
        half=$((cores / 2))
        for s in "2x$half" "${cores}x1"; do
            r=${s%x*} t=${s#*x}
            # shellcheck disable=SC2046
            launch "ranks-500-1e-6-$s" $(mpi "$r" "$t") "$A" "${COMMON[@]}" --kd 500 \
                --eps 1e-6 --m2l rotation --diagonal rule --threads "$t"
        done
        ;;
    *)
        echo "unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
