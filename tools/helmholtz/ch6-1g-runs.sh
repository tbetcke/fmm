#!/bin/bash
# tools/helmholtz/ch6-1g-runs.sh: the machine runs of Helmholtz CH6.1g
# (docs/helmholtz/tasks/CH6.1g.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch6-1g-runs.sh <out-dir> <stage…>
#
# Stages: `e7` (CH6.1e's cases at ε = 1e-7, its `between` list: W1 at kD = 10, 30, 100, W3
# at kD = 10, 30, the cube and the clusters at N = 10⁵ and kD = 1, 100) and `e8` (its
# matrix at ε = 1e-8: W1 at kD = 10, 30, 100, W3 at kD = 10, 30, the four W4 distributions
# at N = 10⁵ and kD = 1, 10, 100). Each case is one launch with `--m2l rotation,planewave`
# (PlaneWave by the rule: the resolving degree on the levels with waves), then one with the
# order reversed (A B B A), `--evaluations 2 --levels`: the errors, the evaluation and per
# run and level the degree, the M2L that ran and its seconds of the second evaluation. One
# rank × THREADS threads (default 40), BLAS threads 1, no quiet gate (for Kathleen's
# exclusive nodes; on a shared machine time nothing from it). TIMEOUT (default 3600)
# bounds a launch; ONLY (a regular expression on the launch names) runs a subset. Each
# launch's output in <out-dir>/<stage>-<case>-<order>.md, a line per launch in
# <out-dir>/runs.log.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
fi
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
TIMEOUT=${TIMEOUT:-3600}
common=(--evaluations 2 --levels --threads "$T")

# `launch <name> <command…>`: one launch under the timeout.
launch() {
    local name=$1
    shift
    local start=$SECONDS
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

# `cases <stage>`: the stage's cases, "name|arguments".
cases() {
    case $1 in
        e7)
            echo "w1|--workload w1 --kd 10,30,100 --eps 1e-7"
            echo "w3|--workload w3 --kd 10,30 --eps 1e-7"
            for d in cube clusters; do
                echo "w4-$d|--workload w4-$d:100000 --kd 1,100 --eps 1e-7"
            done
            ;;
        e8)
            echo "w1|--workload w1 --kd 10,30,100 --eps 1e-8"
            echo "w3|--workload w3 --kd 10,30 --eps 1e-8"
            for d in cube sphere plummer clusters; do
                echo "w4-$d|--workload w4-$d:100000 --kd 1,10,100 --eps 1e-8"
            done
            ;;
        *) echo "unknown stage $1" >&2; exit 2 ;;
    esac
}

for stage in "$@"; do
    while IFS= read -r entry; do
        name=${entry%%|*}
        for order in forward reverse; do
            [[ "$stage-$name-$order" =~ ${ONLY:-.} ]] || continue
            m2l=rotation,planewave
            [ "$order" = reverse ] && m2l=planewave,rotation
            # Word splitting of the case's arguments is intended.
            # shellcheck disable=SC2086
            launch "$stage-$name-$order" "$X" ${entry#*|} "${common[@]}" --m2l "$m2l"
        done
    done < <(cases "$stage")
done
