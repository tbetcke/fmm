#!/bin/bash
# tools/helmholtz/ch4-0d2-runs.sh: the machine runs of Helmholtz CH4.0d2, the V pairs
# between leaves as near pairs (docs/helmholtz/tasks/CH4.0d2.md, "Machine runs"), from the
# repository root after the example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-0d2-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` with `--m2l rotation,auto-pw` (Rotation, all-RCR,
# and the library's default, Auto with the exponential M2L), a list of `--near-v`
# crossovers of n_s n_t (`off` the rule off, `inf` every V pair between leaves), `--levels`
# (the M2L and the P2P per level, for the cost model's constants), BLAS threads 1:
# - `matrix`: CH3.6's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4
#   distributions at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶ and 10⁻⁸, crossovers
#   off, 300, 1000, 3000, 10⁴, inf and the reverse (ABBA), two evaluations;
# - `large`: W1 at kD = 500 and W3 at kD = 100 at 10⁻⁶ and 10⁻⁸, crossovers off, 1000, 3000,
#   10⁴, 3·10⁴, inf, two evaluations;
# - `w3`: the under-resolved W3 (N = 10⁶) at kD = 500 at 10⁻⁶ and 10⁻⁸ (leaves of about one
#   point), crossovers off, 3, 30, inf, one evaluation;
# - `traffic`: on 4 ranks × T/4 threads (`mpirun -n 4 --map-by slot:PE=T/4 --bind-to
#   core`) and again on one rank × T threads, the default alone: the under-resolved W3 at
#   kD = 500 at 10⁻⁶ (crossovers off, inf) and W1 at kD = 500 at 10⁻⁶ (off, 3000): the source
#   points and multipoles moved, and the checksums of 4 ranks against one (rank-count
#   invariance with the rule on);
# - `custom`: CROSSOVERS (a `--near-v` list) on CASES (`name|arguments` entries separated by
#   `;`), for the cost model's check once its constants are set.
# CROSSOVERS (a `--near-v` list, e.g. `off,cost,inf`) replaces every stage's list (the
# matrix's second order is its reverse): the second round, the cost model against the rule
# off and every pair converted.
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
# `crossovers <default>`: CROSSOVERS if set, else the stage's own list.
crossovers() { echo "${CROSSOVERS:-$1}"; }
# `reversed <list>`: the comma-separated list in reverse order.
reversed() { echo "$1" | tr ',' '\n' | awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) printf "%s%s", a[i], (i > 1 ? "," : "\n") }'; }
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
            list=$(crossovers off,300,1000,3000,10000,inf)
            for order in "$list" "$(reversed "$list")"; do
                # Word splitting of the arguments is intended.
                # shellcheck disable=SC2086
                launch "matrix-$name-${order%%,*}" "$X" ${entry#*|} --eps 1e-3,1e-6,1e-8 \
                    "${M2L[@]}" --evaluations 2 --near-v "$order" --threads "$T"
            done
        done
        ;;
    large)
        for entry in "w1-500|--workload w1 --kd 500" "w3-100|--workload w3 --kd 100"; do
            name=${entry%%|*}
            # shellcheck disable=SC2086
            launch "large-$name" "$X" ${entry#*|} --eps 1e-6,1e-8 "${M2L[@]}" \
                --evaluations 2 --near-v "$(crossovers off,1000,3000,10000,30000,inf)" --threads "$T"
        done
        ;;
    w3)
        launch "w3-500" "$X" --workload w3:1000000 --kd 500 --eps 1e-6,1e-8 "${M2L[@]}" \
            --near-v "$(crossovers off,3,30,inf)" --threads "$T"
        ;;
    traffic)
        for entry in "w3-500|--workload w3:1000000 --kd 500|off,inf" \
            "w1-500|--workload w1 --kd 500|off,3000"; do
            name=${entry%%|*}
            rest=${entry#*|}
            list=$(crossovers "${rest#*|}")
            # shellcheck disable=SC2086
            launch "traffic-$name-4x" mpirun -n 4 --map-by "slot:PE=$((T / 4))" \
                --bind-to core "$X" ${rest%|*} --eps 1e-6 --m2l auto-pw --levels \
                --near-v "$list" --threads "$((T / 4))"
            # shellcheck disable=SC2086
            launch "traffic-$name-1x" "$X" ${rest%|*} --eps 1e-6 --m2l auto-pw --levels \
                --near-v "$list" --threads "$T"
        done
        ;;
    custom)
        IFS=';' read -r -a cases <<< "${CASES:?CASES names the cases of the custom stage}"
        for entry in "${cases[@]}"; do
            name=${entry%%|*}
            # shellcheck disable=SC2086
            launch "custom-$name" "$X" ${entry#*|} "${M2L[@]}" \
                --near-v "${CROSSOVERS:?CROSSOVERS names the --near-v list}" --threads "$T"
        done
        ;;
    *)
        echo "unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
