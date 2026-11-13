#!/bin/bash
# tools/helmholtz/ch4-0c-runs.sh: the machine runs of Helmholtz CH4.0c
# (docs/helmholtz/tasks/CH4.0c.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_boxes
#     tools/helmholtz/ch4-0c-runs.sh <stage> <out-dir>
#
# Stages, each launch one `helmholtz_boxes` on one rank × THREADS threads (default: the
# machine's cores, 6 on the M6, 40 on a Kathleen node), against the direct sum, under
# Rotation (all-RCR) and the library's default:
# - `w1-random`: W1 at kD = 1000, §9.1's random charges, at 10⁻⁸ (first: it fixes the box
#   grid), 10⁻⁷, 10⁻⁶ and 10⁻³; the reference at the samples and one target per box of the
#   deepest level;
# - `w1-smooth`: the same with the plane wave's trace and the constant density, the reference
#   at the samples;
# - `w3`: W3 at N = 10⁶ (under-resolved) and kD = 200 and 500, the three densities, at 10⁻⁸,
#   10⁻⁶ and 10⁻³; at kD = 200 two targets per box of the deepest level (level 5), at
#   kD = 500 the samples alone (level 7 has about as many boxes as points);
# - `w1-levels`: W1 at kD = 1000, random charges, 10⁻⁸, Rotation, the levels' parts of the
#   error (`--study levels`), the reference at the samples.
# Each launch's output in <out-dir>/<name>.md, a line per launch in <out-dir>/runs.log. BLAS
# threads 1. Accuracy runs; times are read on Kathleen only (exclusive nodes), never
# asserted. TIMEOUT (default 18000) bounds a launch; ONLY (a regular expression on the launch
# names) runs a subset.
set -u
stage=$1
out=$2
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=target/release/examples/helmholtz_boxes
TIMEOUT=${TIMEOUT:-18000}
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
    T=${THREADS:-6}
    run=(mpirun -n 1 --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0
        --host localhost:6 --bind-to none "$X")
else
    T=${THREADS:-40}
    run=("$X")
fi

# `launch <name> <arguments…>`: one launch under the timeout.
launch() {
    local name=$1
    shift
    [[ "$name" =~ ${ONLY:-.} ]] || return 0
    local start=$SECONDS
    timeout "$TIMEOUT" "${run[@]}" "$@" --threads "$T" < /dev/null \
        > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

W1=(--workload w1 --kd 1000 --eps 1e-8,1e-7,1e-6,1e-3 --m2l rotation,default)
W3=(--eps 1e-8,1e-6,1e-3 --m2l rotation,default)
case $stage in
w1-random)
    launch w1-1000-random "${W1[@]}" --density random --per-box 1
    ;;
w1-smooth)
    for density in planewave constant; do
        launch "w1-1000-$density" "${W1[@]}" --density "$density" --per-box 0
    done
    ;;
w3)
    for density in random planewave constant; do
        launch "w3-200-$density" --workload w3:1000000 --kd 200 "${W3[@]}" \
            --density "$density" --per-box 2
        launch "w3-500-$density" --workload w3:1000000 --kd 500 "${W3[@]}" \
            --density "$density" --per-box 0
    done
    ;;
w1-levels)
    launch w1-1000-levels --workload w1 --kd 1000 --eps 1e-8 --m2l rotation \
        --density random --per-box 0 --study levels
    ;;
*)
    echo "usage: $0 w1-random|w1-smooth|w3|w1-levels <out-dir>" >&2
    exit 2
    ;;
esac
