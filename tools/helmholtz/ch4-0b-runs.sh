#!/bin/bash
# tools/helmholtz/ch4-0b-runs.sh: the machine runs of Helmholtz CH4.0b
# (docs/helmholtz/tasks/CH4.0b.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_boxes
#     tools/helmholtz/ch4-0b-runs.sh <stage> <out-dir>
#
# Stages, each launch one `helmholtz_boxes` on one rank × THREADS threads (default: the
# machine's cores, 6 on the M6, 40 on a Kathleen node), against the direct sum:
# - `cause`: where the worst boxes come from, all-RCR (Rotation, as CH3.6): W1 at kD = 30 and
#   100 with the error by level (`--study levels`) at 10⁻⁶ (kD = 30) and 10⁻³, 10⁻⁶, 10⁻⁸
#   (kD = 100); W1 at kD = 100 and 10⁻⁸ with a margin of 0.03, with a margin of 0.1 and an
#   offset (by level), and on a tree of one leaf level (no W or X list); W3 at kD = 30 and the
#   W4 Plummer sphere (N = 10⁵) at kD = 10, at 10⁻⁸, by level;
# - `densities`: W1 at each kD of KDS (default "10 30 100") with the densities of DENSITIES
#   (default "random planewave constant"), each at 10⁻³, 10⁻⁶ and 10⁻⁸ under Rotation and the
#   library's default; above 2·10⁵ points the reference is the per-box sample;
# - `pointwise`: W1 at each kD of KDS under the default at ε and tighter tolerances (10⁻³ to
#   10⁻⁵, 10⁻⁶ to 10⁻⁸, 10⁻⁸ to 10⁻¹⁰), for the degrees and times that bring the worst box
#   to the requested ε.
# Each launch's output in <out-dir>/<name>.md, a line per launch in <out-dir>/runs.log. BLAS
# threads 1. Accuracy runs; the times of `pointwise` are read on Kathleen only (exclusive
# nodes), never from the M6. TIMEOUT (default 7200) bounds a launch; ONLY (a regular
# expression on the launch names) runs a subset.
set -u
stage=$1
out=$2
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=target/release/examples/helmholtz_boxes
TIMEOUT=${TIMEOUT:-7200}
if [ "$(uname)" = Darwin ]; then
    export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--darwin}
    T=${THREADS:-6}
    run=(mpirun -n 1 --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0
        --host localhost:6 --bind-to none "$X")
else
    T=${THREADS:-40}
    run=("$X")
fi
KDS=${KDS:-10 30 100}
DENSITIES=${DENSITIES:-random planewave constant}

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

case $stage in
cause)
    launch w1-30-e6-levels --kd 30 --eps 1e-6 --study levels
    for e in 3 6 8; do
        launch "w1-100-e$e-levels" --kd 100 --eps "1e-$e" --study levels
    done
    launch w1-100-e8-margin3 --kd 100 --eps 1e-8 --margin 0.03
    launch w1-100-e8-margin10-levels --kd 100 --eps 1e-8 --margin 0.1 \
        --offset 0.013,0.029,0.041 --study levels
    launch w1-100-e8-uniform --kd 100 --eps 1e-8 --leaf-size uniform
    launch w3-30-e8-levels --workload w3 --kd 30 --eps 1e-8 --study levels
    launch w4-plummer-10-e8-levels --workload w4-plummer:100000 --kd 10 --eps 1e-8 \
        --study levels
    ;;
densities)
    for kd in $KDS; do
        for density in $DENSITIES; do
            launch "w1-$kd-$density" --kd "$kd" --density "$density" \
                --eps 1e-3,1e-6,1e-8 --m2l rotation,default
        done
    done
    ;;
pointwise)
    for kd in $KDS; do
        launch "w1-$kd-pointwise-e3" --kd "$kd" --m2l default --eps 1e-3,1e-4,3e-5,1e-5
        launch "w1-$kd-pointwise-e6" --kd "$kd" --m2l default --eps 1e-6,1e-7,3e-8,1e-8
        launch "w1-$kd-pointwise-e8" --kd "$kd" --m2l default --eps 1e-8,1e-9,3e-10,1e-10
    done
    ;;
*)
    echo "usage: $0 cause|densities|pointwise <out-dir>" >&2
    exit 2
    ;;
esac
