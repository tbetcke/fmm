#!/bin/bash
# tools/helmholtz/ch4-0e-runs.sh: the machine runs of Helmholtz CH4.0e, the comparison with
# FMM3D under the defaults (docs/helmholtz/tasks/CH4.0e.md, "Machine runs"), from the
# repository root after
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy --example helmholtz_fmm3d
#     tools/helmholtz/ch4-0e-runs.sh <out-dir> <stage…>
#
# Ours runs `helmholtz_accuracy` at the library's defaults as they stand after CH4.0d3
# (`--m2l auto-pw --near-v cost --near-wx cost`, the leaf size and c_leaf the library's),
# `--levels`, two evaluations of one build; its case is `helmholtz_fmm3d problem`'s, so
# FMM3D's outputs are measured against the same oracle by `helmholtz_fmm3d errors`.
# Stages:
#   ours   CH3.6's matrix (CASES) at EPS on every split of SPLITS (ranks x threads; one
#          launch per case and split under mpirun), in the order of SPLITS and again
#          reversed (launch names ...-a and ...-b)
#   fmm3d  per case of CASES and variant of VARIANTS (default "default fastker"), the
#          driver on every core over the eps grid GRID (default 1e-3, 3e-4, ..., 1e-9; looser
#          eps cannot match our errors at 10⁻³, and FMM3D's default variant cannot allocate
#          its plane-wave workspace on W1 at kD = 500 at 10⁻² on a 188 GB node) with REPEAT
#          2 (the best of two calls, FMM3D's setup included), launches a and b (the
#          variants in order, then reversed);
#          then `helmholtz_fmm3d errors` on launch a's outputs into fmm3d-<case>.md
#   blocks W1 at kD = 500 at EPS with the M2L's rotation blocks stored on every level
#          (`--block-budget BLOCK_MB`, default 2048 MiB per rank; the library's 512 MiB leaves
#          level 2's, p = 123–135, to be formed per V pair), at every split, ABBA, and on one
#          core at 10⁻⁶ (launch names blocks-...); the same bits as `ours`
#   t1     one core: ours (1 x 1) on T1_CASES at EPS, both FMM3D variants on T1_CASES over
#          GRID (REPEAT 1), and ours on W1 at kD = 500 at 10⁻⁶ (the per-core part of the
#          Kathleen-against-locust comparison)
# A case is `<workload>@<kD>` with the examples' workload names. SPLITS defaults to
# "1x40 2x20 4x10 8x5" under Slurm and "1x72 2x36 4x18 8x9" elsewhere; under Slurm an even
# rank count is bound per socket (`ppr:<r/2>:socket`), elsewhere `slot:PE=<t>`. FMM3D_ROOT as
# for tools/fmm3d/build.sh; THREADS (FMM3D's, default every core), TIMEOUT per launch
# (default 7200 s), ONLY (a regular expression on the launch names), DRY_RUN (print the
# launches only). Each launch writes <out-dir>/<name>.md and .err (ours) or <name>.log
# (FMM3D), and a line in <out-dir>/runs.log. Times are read on Kathleen's exclusive nodes and
# a quiet locust only, never asserted; no run on the M6 without the user's go-ahead.
set -u
out=$1
shift
mkdir -p "$out/problems" "$out/fmm3d"
target=${CARGO_TARGET_DIR:-target}/release/examples
A=$target/helmholtz_accuracy
F=$target/helmholtz_fmm3d
if [ -z "${FMM3D_ROOT:-}" ]; then
    if [ -n "${FMM_KATHLEEN_ROOT:-}" ]; then
        FMM3D_ROOT=$FMM_KATHLEEN_ROOT/fmm3d
    elif [ -d /data/ucahtbe ]; then
        FMM3D_ROOT=/data/ucahtbe/fmm3d
    else
        FMM3D_ROOT=
    fi
fi
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
T=${THREADS:-$cores}
TIMEOUT=${TIMEOUT:-7200}
EPS=1e-3,1e-6,1e-8
GRID=${GRID:-1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8,3e-9,1e-9}
VARIANTS=${VARIANTS:-"default fastker"}
matrix="w1@10 w1@30 w1@100 w1@500 w3@10 w3@30 w3@100"
for d in cube sphere plummer clusters; do
    for kd in 1 10 100; do matrix="$matrix w4-$d:100000@$kd"; done
done
CASES=${CASES:-$matrix}
T1_CASES=${T1_CASES:-w1@100}
DEFAULTS=(--m2l auto-pw --near-v cost --near-wx cost --levels --evaluations 2)

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

# `ours_case <case> <split> <order>`: one launch of ours at the defaults.
ours_case() {
    local c=$1 split=$2 order=$3 r t
    r=${split%x*} t=${split#*x}
    # Word splitting of the launcher is intended.
    # shellcheck disable=SC2046
    launch "ours-$(name "$c")-$split-$order" $(mpi "$r" "$t") "$A" --workload "${c%@*}" \
        --kd "${c#*@}" --eps "$EPS" "${DEFAULTS[@]}" --threads "$t"
}

# `reversed <words…>`: the words in reverse order.
reversed() { printf '%s\n' "$@" | awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) printf "%s%s", a[i], (i > 1 ? " " : "\n") }'; }

# `problem <case>`: writes the case's problem file once, prints its path.
problem() {
    local file
    file=$out/problems/$(name "$1").bin
    if [ ! -f "$file" ] && [ -z "${DRY_RUN:-}" ]; then
        "$F" problem --workload "${1%@*}" --kd "${1#*@}" --out "$file" >> "$out/problems.log" ||
            { echo "ch4-0e-runs.sh: no problem file for $1" >&2; return 1; }
    fi
    echo "$file"
}

# `driver <name> <variant> <threads> <problem> <kD> <repeat>`: one launch of FMM3D's driver
# over GRID, its log in <out-dir>/fmm3d/<name>.log, its outputs beside it.
driver() {
    local n=$1 variant=$2 threads=$3 prob=$4 kd=$5 repeat=$6 log start
    [[ "$n" =~ ${ONLY:-.} ]] || return 1
    log=$out/fmm3d/$n.log
    local command="OMP_NUM_THREADS=$threads $FMM3D_ROOT/$variant/bin/fmm3d_driver helmholtz $prob $out/fmm3d/$n $kd 0 $GRID $repeat 0"
    if [ -n "${DRY_RUN:-}" ]; then
        echo "$n: $command"
        return 0
    fi
    start=$SECONDS
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S') $(hostname) job ${SLURM_JOB_ID:-none}"
        echo "# $command"
        cat "$FMM3D_ROOT/$variant/build-info.txt" 2> /dev/null | sed 's/^/# /'
        [ "$slurm" = 1 ] || echo "# load before: $(uptime)"
    } > "$log"
    OMP_NUM_THREADS=$threads timeout "$TIMEOUT" "$FMM3D_ROOT/$variant/bin/fmm3d_driver" \
        helmholtz "$prob" "$out/fmm3d/$n" "$kd" 0 "$GRID" "$repeat" 0 < /dev/null >> "$log" 2>&1
    local status=$?
    [ "$slurm" = 1 ] || echo "# load after: $(uptime)" >> "$log"
    echo "$n: exit $status, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
    return $status
}

# `errors <name> <case> <labels>`: FMM3D's outputs at the samples against the case's oracle.
errors() {
    local n=$1 c=$2 labels=$3 prob
    prob=$(problem "$c") || return 0
    if [ -n "${DRY_RUN:-}" ]; then
        echo "$n: $F errors --workload ${c%@*} --kd ${c#*@} --eps $GRID --fmm3d $labels"
        return 0
    fi
    timeout "$TIMEOUT" "$F" errors --workload "${c%@*}" --kd "${c#*@}" --eps "$GRID" \
        --problem "$prob" --threads "$T" --fmm3d "$labels" < /dev/null \
        > "$out/$n.md" 2>> "$out/$n.err" || echo "$n: errors failed" | tee -a "$out/runs.log"
}

# `fmm3d_case <case> <threads> <repeat> <prefix> <runs…>`: FMM3D's variants on one case, a
# launch per run (a: the variants in order, b: reversed), then the errors of launch a.
fmm3d_case() {
    local c=$1 threads=$2 repeat=$3 prefix=$4 run v prob labels= order
    shift 4
    if [ -z "$FMM3D_ROOT" ]; then
        echo "ch4-0e-runs.sh: set FMM3D_ROOT as for tools/fmm3d/build.sh" >&2
        exit 2
    fi
    prob=$(problem "$c") || return 0
    for run in "$@"; do
        order=$VARIANTS
        # Word splitting of the variants is intended.
        # shellcheck disable=SC2086
        [ "$run" = b ] && order=$(reversed $VARIANTS)
        for v in $order; do
            if driver "$prefix-$(name "$c")-$v-$run" "$v" "$threads" "$prob" "${c#*@}" "$repeat" &&
                [ "$run" = a ]; then
                labels="$labels,$v=$out/fmm3d/$prefix-$(name "$c")-$v-$run"
            fi
        done
    done
    [ -n "$labels" ] && errors "$prefix-$(name "$c")" "$c" "${labels#,}"
    [ -n "${DRY_RUN:-}" ] || rm -f "$out/fmm3d/$prefix-$(name "$c")"-*.out
}

for stage in "$@"; do
    case $stage in
    ours)
        # Word splitting of SPLITS is intended.
        # shellcheck disable=SC2086
        for order in a b; do
            splits=$SPLITS
            [ "$order" = b ] && splits=$(reversed $SPLITS)
            for c in $CASES; do
                for s in $splits; do ours_case "$c" "$s" "$order"; done
            done
        done
        ;;
    fmm3d)
        for c in $CASES; do fmm3d_case "$c" "$T" 2 fmm3d a b; done
        ;;
    blocks)
        # Word splitting of SPLITS is intended.
        # shellcheck disable=SC2086
        for order in a b; do
            splits=$SPLITS
            [ "$order" = b ] && splits=$(reversed $SPLITS)
            for s in $splits; do
                r=${s%x*} t=${s#*x}
                # shellcheck disable=SC2046
                launch "blocks-w1-500-$s-$order" $(mpi "$r" "$t") "$A" --workload w1 --kd 500 \
                    --eps "$EPS" "${DEFAULTS[@]}" --block-budget "${BLOCK_MB:-2048}" --threads "$t"
            done
        done
        launch "blocks-t1-w1-500" "$A" --workload w1 --kd 500 --eps 1e-6 "${DEFAULTS[@]}" \
            --block-budget "${BLOCK_MB:-2048}" --threads 1 --oracle-threads "$T"
        ;;
    t1)
        for c in $T1_CASES; do
            launch "t1-ours-$(name "$c")" "$A" --workload "${c%@*}" --kd "${c#*@}" \
                --eps "$EPS" "${DEFAULTS[@]}" --threads 1
            fmm3d_case "$c" 1 1 t1 a
        done
        launch "t1-ours-w1-500" "$A" --workload w1 --kd 500 --eps 1e-6 "${DEFAULTS[@]}" \
            --threads 1 --oracle-threads "$T"
        ;;
    *)
        echo "ch4-0e-runs.sh: unknown stage $stage (ours, fmm3d, blocks or t1)" >&2
        exit 2
        ;;
    esac
done
