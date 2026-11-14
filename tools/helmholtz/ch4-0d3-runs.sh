#!/bin/bash
# tools/helmholtz/ch4-0d3-runs.sh: the machine runs of Helmholtz CH4.0d3, the leaf size and
# c_leaf (docs/helmholtz/tasks/CH4.0d3.md, "Machine runs"), from the repository root after
# the example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-0d3-runs.sh <out-dir> <stage…>
#
# Every launch runs `helmholtz_accuracy` under the library's default M2L (`--m2l auto-pw`,
# Auto with the exponential M2L), the V rule off and by its cost model (`--near-v
# off,cost`), `--levels`, BLAS threads 1, and a list of leaf sizes (`--leaf-size`, the
# outermost loop):
# - `matrix`: CH3.6's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4
#   distributions at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶ and 10⁻⁸, sizes SIZES
#   (default 32,64,128,256,512,1024) and their reverse (ABBA), two evaluations;
# - `large`: W1 at kD = 500 (sizes SIZES) and W3 at kD = 100 (N = 4·10⁶; sizes to 512) at
#   10⁻³, 10⁻⁶ and 10⁻⁸, two evaluations;
# - `cleaf`: c_leaf 1, 1.5, 2 × sizes 64, 128, 256, 512 on W1 at kD = 100 and 500 and W4's
#   cube and Plummer sphere at N = 10⁵ and kD = 100, at 10⁻³, 10⁻⁶, 10⁻⁸, two evaluations;
# - `planes`: real charges (one multipole plane) on W1 at kD = 30, 100 and W4's cube at
#   kD = 10, 100, sizes SIZES and their reverse, two evaluations;
# - `traffic`: on 4 ranks × T/4 threads (`mpirun -n 4 --map-by slot:PE=T/4 --bind-to
#   core`) and again on one rank × T threads, at the sizes TRAFFIC_SIZES (default
#   auto,256): W1 at kD = 500 and W4's Plummer sphere at kD = 10, 10⁻⁶ (the checksums of 4
#   ranks against one: rank-count invariance at each size);
# - `calibrate`: the W and X rule's constants (`--near-wx off`, `--levels`, the table "Leaf
#   operators": thread-ns per P2L and M2P interaction by degree): W4's Plummer sphere, clusters
#   and sphere at N = 10⁵ and kD = 1, 10, 100, and W1 at kD = 100, at 10⁻³, 10⁻⁶, 10⁻⁸, leaf
#   sizes 64 and 256, one evaluation;
# - `custom`: CASES (`name|arguments` entries separated by `;`) at SIZES and their
#   reverse, for round 2 (the proposed rule, `auto`, against 64).
# NEAR (a `--near-v` list, default off,cost) replaces every stage's; NEAR_WX (a `--near-wx`
# list, default off: round 1 ran without the W and X rule) and C_LEAF (a `--c-leaf` list,
# default the library's, `DEFAULT_C_LEAF` = 2 since this task's proposed defaults, 1 when
# round 1 ran: repeat round 1 with C_LEAF=1; not with `cleaf`, which names its own) add to
# every stage but `calibrate` (round 2: NEAR_WX=off,cost C_LEAF=1,2; round 3: report §7). One rank × THREADS
# threads (default 40, a Kathleen node) unless named. Each launch's output in
# <out-dir>/<stage>-<case>-<order>.md, a line per launch in <out-dir>/runs.log. Times are
# read on Kathleen's exclusive nodes and a quiet locust only, never asserted. TIMEOUT
# (default 18000) bounds a launch; ONLY (a regular expression on the launch names) runs a
# subset.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
TIMEOUT=${TIMEOUT:-18000}
SIZES=${SIZES:-32,64,128,256,512,1024}
NEAR=${NEAR:-off,cost}
NEAR_WX=${NEAR_WX:-off}
EPS=1e-3,1e-6,1e-8

# `launch <name> <launcher and command…>`: one launch under the timeout.
launch() {
    local name=$1
    shift
    [[ "$name" =~ ${ONLY:-.} ]] || return 0
    local start=$SECONDS
    timeout "$TIMEOUT" "$@" < /dev/null > "$out/$name.md" 2> "$out/$name.err"
    echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
}

RUN=(--m2l auto-pw --levels --near-v "$NEAR" --near-wx "$NEAR_WX")
[[ -n "${C_LEAF:-}" ]] && RUN+=(--c-leaf "$C_LEAF")
# `reversed <list>`: the comma-separated list in reverse order.
reversed() { echo "$1" | tr ',' '\n' | awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) printf "%s%s", a[i], (i > 1 ? "," : "\n") }'; }
# `abba <stage> <entries…>`: each `name|arguments` entry at SIZES and their reverse.
abba() {
    local stage=$1
    shift
    for entry in "$@"; do
        name=${entry%%|*}
        for order in "$SIZES" "$(reversed "$SIZES")"; do
            # Word splitting of the arguments is intended.
            # shellcheck disable=SC2086
            launch "$stage-$name-${order%%,*}" "$X" ${entry#*|} "${RUN[@]}" \
                --evaluations 2 --leaf-size "$order" --threads "$T"
        done
    done
}
for stage in "$@"; do
    case $stage in
    matrix)
        cases=("w1|--workload w1 --kd 10,30,100 --eps $EPS"
            "w3|--workload w3 --kd 10,30 --eps $EPS")
        for d in cube sphere plummer clusters; do
            cases+=("w4-$d|--workload w4-$d:100000 --kd 1,10,100 --eps $EPS")
        done
        abba matrix "${cases[@]}"
        ;;
    large)
        launch "large-w1-500" "$X" --workload w1 --kd 500 --eps "$EPS" "${RUN[@]}" \
            --evaluations 2 --leaf-size "$SIZES" --threads "$T"
        launch "large-w3-100" "$X" --workload w3 --kd 100 --eps "$EPS" "${RUN[@]}" \
            --evaluations 2 --leaf-size "${LARGE_W3_SIZES:-32,64,128,256,512}" --threads "$T"
        ;;
    cleaf)
        for entry in "w1|--workload w1 --kd 100,500" \
            "w4-cube|--workload w4-cube:100000 --kd 100" \
            "w4-plummer|--workload w4-plummer:100000 --kd 100"; do
            name=${entry%%|*}
            # shellcheck disable=SC2086
            launch "cleaf-$name" "$X" ${entry#*|} --eps "$EPS" "${RUN[@]}" --evaluations 2 \
                --leaf-size 64,128,256,512 --c-leaf 1,1.5,2 --threads "$T"
        done
        ;;
    planes)
        abba planes "w1|--workload w1 --kd 30,100 --eps $EPS --charges real" \
            "w4-cube|--workload w4-cube:100000 --kd 10,100 --eps $EPS --charges real"
        ;;
    traffic)
        for entry in "w1-500|--workload w1 --kd 500" \
            "w4-plummer|--workload w4-plummer:100000 --kd 10"; do
            name=${entry%%|*}
            sizes=${TRAFFIC_SIZES:-auto,256}
            # shellcheck disable=SC2086
            launch "traffic-$name-4x" mpirun -n 4 --map-by "slot:PE=$((T / 4))" \
                --bind-to core "$X" ${entry#*|} --eps 1e-6 "${RUN[@]}" --leaf-size "$sizes" \
                --threads "$((T / 4))"
            # shellcheck disable=SC2086
            launch "traffic-$name-1x" "$X" ${entry#*|} --eps 1e-6 "${RUN[@]}" \
                --leaf-size "$sizes" --threads "$T"
        done
        ;;
    calibrate)
        for entry in "w4-plummer|--workload w4-plummer:100000 --kd 1,10,100" \
            "w4-clusters|--workload w4-clusters:100000 --kd 1,10,100" \
            "w4-sphere|--workload w4-sphere:100000 --kd 1,10,100" \
            "w1|--workload w1 --kd 100"; do
            name=${entry%%|*}
            # shellcheck disable=SC2086
            launch "calibrate-$name" "$X" ${entry#*|} --eps "$EPS" --m2l auto-pw --levels \
                --near-wx off --leaf-size 64,256 --threads "$T"
        done
        ;;
    custom)
        IFS=';' read -r -a cases <<< "${CASES:?CASES names the cases of the custom stage}"
        abba custom "${cases[@]}"
        ;;
    *)
        echo "unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
