#!/bin/bash
# tools/helmholtz/h6-gate-runs.sh: the machine runs of the H6 gate's large-case verification
# (docs/helmholtz/gates/H6.md, §2 and §8), from the repository root after the example was
# built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/h6-gate-runs.sh <out-dir>
#
# D1's two large cases, W1 at kD = 500 and W3 at kD = 100, at ε = 1e-7 and 1e-8, which
# CH6.1e's sign-off 1 left to Rotation and CH6.1g and CH6.1h did not run. Each case is one
# launch with `--m2l rotation,auto-pw --raise rule` (Rotation, then the library's default:
# `Auto` with the exponential M2L by D8's rule, in the raise form of `PlaneWaveRaise::Rule`),
# then one with the order reversed (A B B A), `--evaluations 2 --levels`: the errors, the
# evaluation, rank 0's peak resident size and per run and level the degree, the M2L that ran
# and its seconds of the second evaluation. One rank × THREADS threads (default 40), BLAS
# threads 1, no quiet gate (for Kathleen's exclusive nodes; on a shared machine time nothing
# from it). TIMEOUT (default 3600) bounds a launch; ONLY (a regular expression on the launch
# names) runs a subset. Each launch's output in <out-dir>/<case>-<order>.md, a line per
# launch in <out-dir>/runs.log.
set -u
out=$1
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
fi
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
TIMEOUT=${TIMEOUT:-3600}
common=(--evaluations 2 --levels --threads "$T" --raise rule)

# `launch <name> <command…>`: one launch under the timeout.
launch() {
    local name=$1
    shift
    local start=$SECONDS
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

cases=(
    "w1-500-e7|--workload w1 --kd 500 --eps 1e-7"
    "w1-500-e8|--workload w1 --kd 500 --eps 1e-8"
    "w3-100-e7|--workload w3 --kd 100 --eps 1e-7"
    "w3-100-e8|--workload w3 --kd 100 --eps 1e-8"
)

for entry in "${cases[@]}"; do
    name=${entry%%|*}
    for order in forward reverse; do
        [[ "$name-$order" =~ ${ONLY:-.} ]] || continue
        m2l=rotation,auto-pw
        [ "$order" = reverse ] && m2l=auto-pw,rotation
        # Word splitting of the case's arguments is intended.
        # shellcheck disable=SC2086
        launch "$name-$order" "$X" ${entry#*|} "${common[@]}" --m2l "$m2l"
    done
done
