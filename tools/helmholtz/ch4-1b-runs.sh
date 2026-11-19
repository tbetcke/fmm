#!/bin/bash
# tools/helmholtz/ch4-1b-runs.sh: the machine runs of Helmholtz CH4.1b, the signature buffers
# and the level's scratch (docs/helmholtz/tasks/CH4.1b.md, "Machine runs"), from the
# repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-1b-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` with `--levels` (the table "Diagonal-form scratch":
# per level the targets and V sources of the most loaded rank and the diagonal form's scratch
# under 8 GiB per node over the ranks per node, as if every V pair ran it) and ends with rank
# 0's peak resident size. Stages:
#   memory  W1 at kD = 500 at 10⁻⁶ and at 10⁻⁸, one process each (the peak is the process's),
#           `--m2l rotation` (every level RCR) and the library's defaults (`--m2l auto-pw`),
#           both with the near rules of the defaults, at one rank with every core: the RCR
#           FMM's resident memory, the baseline the signature scratch is added to (H3 gate §5)
#   census  W1 at kD = 1000 at 10⁻⁶ and at 10⁻⁸, the defaults, at one rank with every core
#           and at one rank per core: the scratch per level within the budget (CH4.1's
#           acceptance) and the levels a node budget sends to RCR at many ranks per node
# TIMEOUT per launch (default 3600 s), ONLY (a regular expression on the launch names),
# DRY_RUN (print the launches only). Each launch writes <out-dir>/<name>.md and .err and a
# line in <out-dir>/runs.log. Memory and counts only, no timing; on Kathleen's exclusive
# nodes.
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
TIMEOUT=${TIMEOUT:-3600}
NEAR=(--near-v cost --near-wx cost)

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
    memory)
        for eps in 1e-6 1e-8; do
            for m2l in rotation auto-pw; do
                # shellcheck disable=SC2046
                launch "memory-$m2l-$eps" $(mpi 1 "$cores") "$A" --workload w1 --kd 500 \
                    --eps "$eps" --m2l "$m2l" "${NEAR[@]}" --levels --threads "$cores"
            done
        done
        ;;
    census)
        for eps in 1e-6 1e-8; do
            for s in "1x$cores" "${cores}x1"; do
                r=${s%x*} t=${s#*x}
                # shellcheck disable=SC2046
                launch "census-$eps-$s" $(mpi "$r" "$t") "$A" --workload w1 --kd 1000 \
                    --eps "$eps" --m2l auto-pw "${NEAR[@]}" --levels --threads "$t"
            done
        done
        ;;
    *)
        echo "unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
