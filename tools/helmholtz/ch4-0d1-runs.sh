#!/bin/bash
# tools/helmholtz/ch4-0d1-runs.sh: the machine runs of Helmholtz CH4.0d1, the empty-box rule
# (docs/helmholtz/tasks/CH4.0d1.md, "Machine runs"), from the repository root after the
# example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-0d1-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` with `--m2l rotation,auto-pw` (Rotation, all-RCR,
# and the library's default, Auto with the exponential M2L), `--empty-boxes on,off` or the
# reverse, `--levels`, BLAS threads 1:
# - `matrix`: CH3.6's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4
#   distributions at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶ and 10⁻⁸, two evaluations,
#   each case twice (`on,off`, then `off,on`: ABBA);
# - `large`: W1 at kD = 500 and W3 at kD = 100 at 10⁻⁶ and 10⁻⁸, two evaluations, once in
#   each order;
# - `w3`: the under-resolved W3 (N = 10⁶) at kD = 200 and 500 at 10⁻⁶ and 10⁻⁸, one
#   evaluation (level 7's M2L of the rule off takes about 1,000 s under Rotation at 10⁻⁸),
#   one order;
# - `traffic`: W3 (N = 10⁶) at kD = 500, 10⁻⁶, the default alone, on 4 ranks × T/4 threads
#   (`mpirun -n 4 --map-by slot:PE=T/4 --bind-to core`), for the multipoles read from other
#   ranks.
# One rank × THREADS threads (default 40, a Kathleen node) unless named. Each launch's output
# in <out-dir>/<stage>-<case>-<order>.md, a line per launch in <out-dir>/runs.log. Times are
# read on Kathleen's exclusive nodes only, never asserted. TIMEOUT (default 18000) bounds a
# launch; ONLY (a regular expression on the launch names) runs a subset.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
TIMEOUT=${TIMEOUT:-18000}

# `launch <name> <launcher and command…>`: one launch under the timeout.
launch() {
    local name=$1
    shift
    [[ "$name" =~ ${ONLY:-.} ]] || return 0
    local start=$SECONDS
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

M2L=(--m2l rotation,auto-pw --levels)
for stage in "$@"; do
    case $stage in
    matrix)
        cases=("w1|--workload w1 --kd 10,30,100"
            "w3|--workload w3 --kd 10,30")
        for d in cube sphere plummer clusters; do
            cases+=("w4-$d|--workload w4-$d:100000 --kd 1,10,100")
        done
        for entry in "${cases[@]}"; do
            name=${entry%%|*}
            # Word splitting of the arguments is intended.
            # shellcheck disable=SC2086
            for order in on,off off,on; do
                launch "matrix-$name-${order/,/-}" "$X" ${entry#*|} --eps 1e-3,1e-6,1e-8 \
                    "${M2L[@]}" --evaluations 2 --empty-boxes "$order" --threads "$T"
            done
        done
        ;;
    large)
        for entry in "w1-500|--workload w1 --kd 500" "w3-100|--workload w3 --kd 100"; do
            name=${entry%%|*}
            # shellcheck disable=SC2086
            for order in on,off off,on; do
                launch "large-$name-${order/,/-}" "$X" ${entry#*|} --eps 1e-6,1e-8 \
                    "${M2L[@]}" --evaluations 2 --empty-boxes "$order" --threads "$T"
            done
        done
        ;;
    w3)
        for kd in 200 500; do
            launch "w3-$kd-on-off" "$X" --workload w3:1000000 --kd "$kd" --eps 1e-6,1e-8 \
                "${M2L[@]}" --empty-boxes on,off --threads "$T"
        done
        ;;
    traffic)
        launch "traffic-w3-500-on-off" mpirun -n 4 --map-by "slot:PE=$((T / 4))" \
            --bind-to core "$X" --workload w3:1000000 --kd 500 --eps 1e-6 --m2l auto-pw \
            --levels --empty-boxes on,off --threads "$((T / 4))"
        ;;
    *)
        echo "unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
