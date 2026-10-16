#!/usr/bin/env bash
# tools/scaling/run.sh: the scaling sweeps of Phase 5 T10 (C5.3), one launch of
# nd-fmm-validate's `scaling` example per configuration, each under `mpirun` and an
# external timeout.
#
#     tools/scaling/run.sh <sweep> [<sweep> ...]
#     tools/scaling/run.sh strong weak threads redistribution balance check
#
# Sweeps (docs/phase5/T10-scaling.md, docs/design/distributed-fmm.md §11):
#   strong          the cube and the Plummer sphere at N = 10^6; f64 p = 3 and 8, f32 p = 8;
#                   every rank count of RANKS, one thread each; overlap off and on
#   strong-small    the same at N = 10^5 (where the work runs out)
#   weak            N = 10^5 per rank, the cube and the Plummer sphere, the same p
#   threads         ranks x threads at the machine's cores (12 on the M3 Max, 72 on
#                   locust), the N = 10^6 cube and Plummer, f64 p = 8
#   redistribution  every point on rank 0, a random share, the owners; at 8 ranks (and on
#                   locust also at the largest count), three builds, the N = 10^6 cube and
#                   Plummer, f64 p = 8
#   balance         the cube, the Plummer sphere and the Gaussian clusters at 8 ranks (and
#                   on locust at the largest count), N = 10^6, f64 p = 8
#   sixteen         16 ranks, efficiency cores included (M3 Max only), the N = 10^6 cube
#   strategy        the M2L strategies with small tables, Rotation and Classes, and since
#                   Phase 6 T2 the batched M2L (Batched), against the default (Dense at
#                   p = 8 until Phase 5N T4, Rotation since), at every rank count of RANKS:
#                   the N = 10^6 cube and Plummer, f64 p = 8, overlap off (does the dense
#                   tables' size per rank limit the downward pass?)
#   node-rule       Phase 5N T4, the strategy rule on the node (docs/design/node-m2l.md §4,
#                   §8): the N = 10^6 cube and Plummer, overlap off, at every ranks x
#                   threads of NODE_GRID: f64 p = 8, f32 p = 8 and f64 p = 3 with `auto`
#                   (the new default) and the strategy auto does not resolve to (Dense or
#                   Rotation; with Dense, the default before T4 at p <= 8), f64 p = 12
#                   with `auto` alone (Rotation, before and after T4); f64 p = 7 and f32
#                   p = 9 at one rank with Dense and Rotation (the 9 MB cap); and the
#                   errors (8 vectors, 1,000 targets, the one-rank reference) of `auto`
#                   and Dense at f64 and f32 p = 8 at one rank and the largest count
#   crossover       Dense against Rotation where the rule's budget decides (f32 p = 8,
#                   8.7 MB per rank, and f64 p = 6, 6.4 MB), at the ranks per node of
#                   CROSSOVER, one thread each: the N = 10^6 cube and Plummer
#   check           the errors against the direct sum (8 charge vectors, 1,000 targets)
#                   and the one-rank reference: N = 10^5 at every rank count, N = 10^6 at
#                   one rank and the largest count
#   oversubscribed  locust only: 256 and 512 ranks on 72 cores, the N = 10^6 cube, f64
#                   p = 8, correctness and memory only (never a timing); the soft limit
#                   of open files is raised to the hard limit for these launches
#
# Sweeps of the batched host M2L (Phase 6 T2, docs/phase6/T2-host-batched-m2l.md;
# docs/design/optimisation.md §2), every launch with --kinds, overlap off, against Dense,
# Classes and Rotation; `batched-fused` is Batched with `--mul-add fused`:
#   t2-host         one rank at 1 thread and at every core of the machine (CORES): the
#                   cube and the Plummer sphere at N = 10^6, f64 p = 3, 6, 8, 12, 18 and
#                   f32 p = 3, 6, 8, the five variants; three timed evaluations (two at
#                   p = 12, one at p = 18)
#   t2-node         many ranks per node: every ranks x threads of T2_SPLITS, the N = 10^6
#                   cube and Plummer at f64 p = 8 and the cube at f64 p = 12, the five
#                   variants; three timed evaluations
#
# Sweeps of several charge vectors (Phase 6 T5, docs/phase6/T5-multiple-rhs.md;
# docs/design/optimisation.md §4), every host launch with --kinds, overlap off, three timed
# evaluations, at `--rhs m` for every m of T5_RHS (default "1 2 4 8 16 32"); the time per
# vector is the summary's wall_ms over rhs:
#   t5-host         one rank per core (CORES ranks x 1 thread; under Slurm CORES_PER_NODE
#                   per node on the job's nodes): the N = 10^6 cube, f64 p = 3 and 8 and
#                   f32 p = 3 and 8, with `auto` (the rule's strategy) and `batched`
#   t5-threads      one rank at every core (1 x CORES), the same problems and strategies
#   t5-device       one rank on the device of DEVICE (metal or cuda; the example built with
#                   that feature, FEATURES=DEVICE by default), every core for the host's
#                   parts: the N = 10^6 cube, f32 p = 3 and 8, on CUDA also f64 p = 3 and
#                   8, the device path's own strategy rule (`auto`), without --kinds
#   t5-nodes        the exchanges at every rank of the job (under Slurm, CORES_PER_NODE per
#                   node; else LARGEST ranks): the cube at N = 10^6 and 10^7, f64 p = 8
#                   with `auto`: the messages and bytes per exchange, and the replicated
#                   top's all-gather, against m
#
# Sweeps of the work-weighted cut (Phase 5S T7, docs/phase5s/T7-work-weighted-cut.md;
# docs/design/scale-out.md §6), every launch with --kinds unless counted only:
#   calibrate       the constants of the cost model: one rank per core of the machine (or
#                   of the job's nodes), the cube, the Plummer sphere and the clusters at
#                   N = CAL_N (default 10^7), f64 p = 3 Dense, f64 p = 8 Rotation, f32
#                   p = 8 Rotation and Dense, the first cut (by points), overlap off, three
#                   timed evaluations; the summary's alpha_ns are the fitted constants
#   cut             the first cut against the second: the three workloads at every N of
#                   CUT_NS (default "1000000 10000000"; under Slurm "10000000"), f64 p = 8
#                   with the strategy of the rule, overlap off, three timed evaluations,
#                   `--cut points` and `--cut work`; at the rank counts of CUT_RANKS
#                   (default "8 72" on Linux, "8" on macOS), under Slurm at every split
#                   of SPLITS on the job's nodes
#   cut-counts      counted, never timed: the three workloads at N = 10^6 and 10^7, f64
#                   p = 8 Rotation, both cuts, one warm-up evaluation; oversubscribed at
#                   COUNT_RANKS (default 512 on Linux; under Slurm 80 and 160 ranks per
#                   node, PSM2 with shared contexts, `--bind-to none`)
#
# Sweeps of the scaling report (Phase 5S T9, docs/phase5s/T9-scaling-at-scale.md), every
# launch under `auto` (the rule's strategy), the library's defaults otherwise, and the
# errors of one charge vector at 1,000 targets (the distributed direct sum) unless counted:
#   t9-strong       Slurm only: the cube and the Plummer sphere at every N of T9_NS (default
#                   "10000000 100000000"), f64 p = 3 and 8 and f32 p = 8, every split of
#                   SPLITS (2x20 first), overlap both; EVALS timed evaluations at N <= 10^7,
#                   T9_LARGE_EVALS (default 2) above; at N <= 10^7 also f64 p = 8 with the
#                   first cut only (`--cut points`, the launch name ends in -cutpoints)
#   t9-weak         Slurm only: 2.5 x 10^4 points per rank at 40x1 and 10^6 per socket at
#                   2x20 (`--n-per-rank`), the cube, the Plummer sphere and the clusters,
#                   f64 p = 3 and 8, f32 p = 8 and 3, overlap both
#   t9-model        the model's inputs (scale-out.md §10.2): one rank per core of the
#                   machine (or of the job's nodes) at 10^4 and 10^6 points per rank, the
#                   three workloads, f64 p = 3 and 8, f32 p = 8 and 3, overlap off, five
#                   timed evaluations, no errors
#   t9-counts       counted, never timed: the three workloads at N = 10^6, 10^7 and at
#                   2.5 x 10^4 points per rank, f64 p = 8 Rotation, overlap off, the warm-up
#                   evaluation only; at COUNT_RANKS (default "128 256 512" on Linux, "80 160"
#                   ranks per node under Slurm, PSM2 with shared contexts, put it last)
# The Phase 5 tree (docs/phase5s/T9, "Phase 5 against Phase 5S") runs the same launches
# with this script copied into it; its harness has no `--cut`, so leave out -cutpoints
# there (ONLY or by name).
#
# Sweeps under Slurm only (Phase 5N T2, docs/phase5n/T2-kathleen-baseline.md), at the
# job's node count, with SPLITS ranks x threads per node (default "40x1 2x20"):
#   host            one rank on one node, at 1 thread and at every core: the N = 10^6
#                   cube and Plummer, f64 p = 3 and 8 with Dense and with Rotation, f32
#                   p = 8 (the default); overlap off. The runs that strong, strategy and
#                   threads already make (one rank, one thread, the default; Rotation at
#                   p = 8 on one thread; Dense at p = 8 on 1 x 40) are left out
#   nodes-strong    the cube and the Plummer sphere at N = 10^6 and 10^7, f64 p = 3 and 8
#                   (the default) and Rotation at p = 8, every split of SPLITS, overlap
#                   both
#   nodes-weak      the same at N = 10^6 x nodes (on one node, nodes-strong's N = 10^6)
#   nodes-check     the errors (8 vectors, 1,000 targets) at N = 10^6 with the one-rank
#                   reference, f64 p = 3 and 8 and f32 p = 8, and at N = 10^7 without it,
#                   f64 p = 3 and 8; one rank per core, two timed evaluations
#   nodes-large     Phase 5S T0: the cube and the Plummer sphere at N = 10^8, f64 p = 3
#                   and 8 with the strategy of the rule (`auto`), every split of SPLITS
#                   (2x20 first, then 40x1, where the memory runs out first), overlap off,
#                   two timed evaluations, the errors of one charge vector (1,000 targets,
#                   the distributed direct sum; no one-rank reference)
#
# Environment:
#   OUT      output directory (default bench-results/scaling-<host>-<date>); one file per
#            launch, with the load before and after on Linux
#   RANKS    the rank counts of strong and weak (default "1 2 4 8 12" on macOS,
#            "1 2 4 8 16 32 64 72" on Linux)
#   BUDGET   the node budget of the strategy rule in MB, passed to every launch as
#            --budget (Phase 5N T4, per machine class: default 56 on macOS, the M3 Max,
#            and 96 on Linux, locust, and under Slurm, Kathleen; signed off as 64 for the
#            M3 Max, set to 56 by the user on 2026-10-11 after T4's f64 p = 6 runs at 8
#            and 10 ranks per node, fmm-validate/results/phase5n-kathleen.md, "Phase 5N,
#            T4")
#   NODE_GRID  ranks x threads per node of node-rule (default T3's grid: "1x1 2x1 4x1 8x1
#            12x1 1x12 2x6" on macOS, "1x1 8x1 16x1 32x1 64x1 72x1 1x72 4x18 8x9" on
#            Linux, "1x1 2x1 10x1 20x1 40x1 1x40 2x20 4x10" under Slurm)
#   CROSSOVER  ranks per node of crossover (default "6 8 10" on macOS, "8 12 16 24" on
#            Linux, "8 10 12 16 20" under Slurm)
#   T2_SPLITS  ranks x threads per node of t2-node (default "1x1 2x1 4x1 8x1 12x1 1x12" on
#            macOS, "1x1 8x1 16x1 32x1 64x1 72x1 1x72 8x9" on Linux, "1x1 10x1 20x1 40x1
#            2x20 1x40" under Slurm)
#   EVALS    timed evaluations per path (default 10)
#   T5_RHS   the charge vectors of the t5 sweeps (default "1 2 4 8 16 32")
#   DEVICE   the device of t5-device: metal or cuda (default metal on macOS, cuda on
#            Linux)
#   FEATURES cargo features of nd-fmm-validate for the example's build (default: none;
#            t5-device builds with DEVICE unless FEATURES is set)
#   TIMEOUT  seconds per launch (default 3600)
#   QUIET    on macOS, the CPU percentage the other processes may use before a launch
#            starts (default 200, two cores): each launch waits, up to an hour, until
#            they use less (a laptop's background jobs, such as a virus scanner, would
#            otherwise share the cores), and the file records the wait
#   ALPHA    if set, the eight constants of the cost model (`--alpha`, ns, comma-separated:
#            P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P) for every launch: a machine's own,
#            from its `calibrate` sweep (Phase 5S T7); the library's otherwise
#   ONLY     if set, an extended regular expression: only the launches whose names match
#            it run (for example the repeats of Phase 5S T0); the others are left out
#            without a file
#   DRY_RUN  if set, print the commands only
#   CARGO_TARGET_DIR, RUSTFLAGS  as for cargo: the example is built and run from
#            $CARGO_TARGET_DIR (default target/), so a labelled build with other flags
#            (for example RUSTFLAGS="-C target-cpu=cascadelake", Phase 5N T2) keeps its own
#            directory; under Slurm each file records RUSTFLAGS
#
# Every BLAS thread variable is set to 1. On macOS `mpirun` gets the loopback flags (root
# CLAUDE.md, "MPI"); on Linux it reports the binding and binds each rank to its cores
# (`--map-by slot:PE=<threads> --bind-to core`). On locust it first sources
# tools/gh200/env.sh if that environment is not active. Timings go into the report only
# with the load stated (docs/phase5/README.md, "Ranks on locust").
#
# Under Slurm (SLURM_JOB_ID set; Kathleen, Phase 5N T2) the sweep runs inside the job's
# allocation: `mpirun` (decision 4 of docs/phase5n/README.md) with the ranks spread
# evenly over the job's nodes and over the SOCKETS sockets of each (default 2):
# `--map-by ppr:<ranks per socket>:socket:PE=<threads> --bind-to core` when the ranks per
# node divide by the sockets, else `--map-by ppr:<ranks per node>:node:PE=<threads>`; so
# rank r of a node with k ranks per socket sits on socket r / k, and 2 ranks on one node
# take one core of each socket. CORES_PER_NODE (default 40) bounds ranks x threads per
# node, and RANKS defaults to "1 2 4 8 10 16 20 40", the thread splits to 1x40 ... 40x1.
# Kathleen's nodes are exclusive, so there is no load check: each file records instead
# the job id, the QoS, the node list, the mapping, and (from mpirun --report-bindings,
# and the report's placement per rank) the binding. tools/kathleen/jobs/sweep.sbatch
# submits a sweep; tools/kathleen/README.md, "Scaling sweeps", has the commands.
#
# The inter-node run (docs/design/distributed-fmm.md §11), documented and not run in
# Phase 5, on a cluster with Open MPI (8 ranks per node on 8 nodes):
#
#     mpirun -n 64 --map-by ppr:8:node --bind-to core target/release/examples/scaling \
#         --dist cube --n 1000000 --precision f64 --p 8 --overlap both
#
# and under Slurm `srun -n 64 --ntasks-per-node=8 --cpu-bind=cores` with the same
# arguments; with MPICH `mpiexec -n 64 -ppn 8 -bind-to core`.

set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"

if [ -z "${FMM_GH200_ENV:-}" ] && [ -d /data/ucahtbe/envs/fmm-gh200 ]; then
    set +u
    . "$root/tools/gh200/env.sh"
    set -u
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

host=$(hostname -s 2>/dev/null || hostname)
OUT=${OUT:-bench-results/scaling-$host-$(date +%Y%m%d-%H%M)}
EVALS=${EVALS:-10}
TIMEOUT=${TIMEOUT:-3600}
QUIET=${QUIET:-200}
SOCKETS=${SOCKETS:-2}
NODES=1
case "$(uname -s)" in
Darwin)
    os=macos
    RANKS=${RANKS:-"1 2 4 8 12"}
    CORES=12
    THREAD_SPLITS="1x12 2x6 4x3 6x2 12x1"
    LARGEST=12
    BUDGET=${BUDGET:-56}
    NODE_GRID=${NODE_GRID:-"1x1 2x1 4x1 8x1 12x1 1x12 2x6"}
    CROSSOVER=${CROSSOVER:-"6 8 10"}
    T2_SPLITS=${T2_SPLITS:-"1x1 2x1 4x1 8x1 12x1 1x12"}
    ;;
*)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        os=slurm
        NODES=${SLURM_JOB_NUM_NODES:-1}
        CORES_PER_NODE=${CORES_PER_NODE:-40}
        RANKS=${RANKS:-"1 2 4 8 10 16 20 40"}
        CORES=$CORES_PER_NODE
        THREAD_SPLITS="1x40 2x20 4x10 8x5 10x4 20x2 40x1"
        LARGEST=$((CORES_PER_NODE * NODES))
        SPLITS=${SPLITS:-"40x1 2x20"}
        BUDGET=${BUDGET:-96}
        NODE_GRID=${NODE_GRID:-"1x1 2x1 10x1 20x1 40x1 1x40 2x20 4x10"}
        CROSSOVER=${CROSSOVER:-"8 10 12 16 20"}
        T2_SPLITS=${T2_SPLITS:-"1x1 10x1 20x1 40x1 2x20 1x40"}
    else
        os=linux
        RANKS=${RANKS:-"1 2 4 8 16 32 64 72"}
        CORES=72
        THREAD_SPLITS="1x72 4x18 8x9 18x4 72x1"
        LARGEST=72
        BUDGET=${BUDGET:-96}
        NODE_GRID=${NODE_GRID:-"1x1 8x1 16x1 32x1 64x1 72x1 1x72 4x18 8x9"}
        CROSSOVER=${CROSSOVER:-"8 12 16 24"}
        T2_SPLITS=${T2_SPLITS:-"1x1 8x1 16x1 32x1 64x1 72x1 1x72 8x9"}
    fi
    ;;
esac

T5_RHS=${T5_RHS:-"1 2 4 8 16 32"}
if [ "$os" = macos ]; then DEVICE=${DEVICE:-metal}; else DEVICE=${DEVICE:-cuda}; fi
for sweep in "$@"; do
    if [ "$sweep" = t5-device ] && [ -z "${FEATURES:-}" ]; then FEATURES=$DEVICE; fi
done
exe=${CARGO_TARGET_DIR:-target}/release/examples/scaling
if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p "$OUT"
    if [ -n "${FEATURES:-}" ]; then
        cargo build --release -p nd-fmm-validate --example scaling --features "$FEATURES"
    else
        cargo build --release -p nd-fmm-validate --example scaling
    fi
fi

# Under Slurm: the job and the mapping of a launch, in place of the load.
job() {
    echo "### job"
    echo
    echo "job $SLURM_JOB_ID (${SLURM_JOB_NAME:-}), QoS ${SLURM_JOB_QOS:-?}, partition \
${SLURM_JOB_PARTITION:-?}, $NODES node(s): ${SLURM_JOB_NODELIST:-?}; launched \
$(date '+%Y-%m-%d %H:%M:%S'); nodes exclusive, so no load check"
    echo
    echo "build: release, RUSTFLAGS ${RUSTFLAGS:-unset} (the default target unless set)"
    echo
    echo "mapping: $1 rank(s) per node x $2 thread(s), \`$3\`; the binding of every rank: \
mpirun's --report-bindings lines (stderr, below) and the report's \"Placement per rank\""
}

# The load of the machine: before and after every launch on Linux.
load() {
    echo "### load $1: $(date '+%Y-%m-%d %H:%M:%S')"
    echo '```text'
    uptime
    if [ "$os" = macos ]; then
        ps -Ao user,pcpu,pmem,etime,comm -r | head -6
    else
        ps -eo user,pcpu,pmem,etime,cmd --sort=-pcpu | head -8
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
        fi
    fi
    echo '```'
}

# On macOS: wait until the processes other than this sweep's use at most QUIET% CPU.
quiet() {
    if [ "$os" != macos ]; then return 0; fi
    waited=0
    while :; do
        busy=$(ps -A -o pcpu=,comm= | awk '$2 !~ /scaling|prterun|mpirun|prted/ { s += $1 }
            END { printf "%d", s }')
        if [ "$busy" -le "$QUIET" ]; then break; fi
        if [ "$waited" -ge 3600 ]; then
            echo "still busy after an hour"
            break
        fi
        sleep 15
        waited=$((waited + 15))
    done
    echo "other processes at ${busy}% CPU when launched, after waiting ${waited} s"
}

# launch <name> <ranks> <threads> <scaling arguments...>
launch() {
    name=$1 ranks=$2 threads=$3
    shift 3
    if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$name" | grep -Eq -- "$ONLY"; then
        return
    fi
    if [ "$os" = macos ]; then
        flags="--mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0"
        if [ "$ranks" -gt 16 ]; then flags="$flags --oversubscribe"; fi
    elif [ "$os" = slurm ] && [ -n "${OVERSUBSCRIBED:-}" ]; then
        # Counts only (Phase 5S T7): Kathleen's oversubscribed launch of
        # docs/design/scale-out.md §10.1, at most 160 ranks per node.
        per_node=$((ranks / NODES))
        flags="--report-bindings --oversubscribe --map-by ppr:$per_node:node --bind-to none \
-x PSM2_SHAREDCONTEXTS=1"
    elif [ "$os" = slurm ]; then
        per_node=$((ranks / NODES))
        if [ $((per_node * NODES)) -ne "$ranks" ] || [ $((per_node * threads)) -gt "$CORES" ]; then
            echo "SKIPPED: $name: $ranks ranks x $threads threads do not fit $NODES node(s) \
of $CORES cores evenly" | tee -a "${OUT}/skipped.txt"
            return
        fi
        if [ $((per_node % SOCKETS)) -eq 0 ]; then
            map="ppr:$((per_node / SOCKETS)):socket:PE=$threads"
        else
            map="ppr:$per_node:node:PE=$threads"
        fi
        flags="--report-bindings --map-by $map --bind-to core"
    else
        if [ $((ranks * threads)) -gt "$CORES" ]; then
            flags="--report-bindings --map-by :OVERSUBSCRIBE --bind-to none"
            # Hundreds of ranks need more pipes and sockets than the default soft limit
            # of 1,024 open files (Open MPI fails at 512 ranks on locust without this).
            ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
        else
            flags="--report-bindings --map-by slot:PE=$threads --bind-to core"
        fi
    fi
    alpha=""
    if [ -n "${ALPHA:-}" ]; then alpha="--alpha $ALPHA"; fi
    command="timeout $TIMEOUT mpirun $flags -n $ranks $exe --threads $threads --evals $EVALS \
--budget $BUDGET $alpha $*"
    if [ -n "${DRY_RUN:-}" ]; then
        echo "$command"
        return
    fi
    file="$OUT/$name.txt"
    echo "[$(date '+%H:%M:%S')] $name: $command"
    gate=$(quiet)
    {
        echo "# $name"
        echo
        echo "command: \`$command\`"
        echo
        if [ -n "$gate" ]; then echo "$gate"; echo; fi
        if [ "$os" = slurm ]; then job "$per_node" "$threads" "$flags"; else load before; fi
        echo
    } >"$file"
    # mpirun's binding report goes to stderr; keep it in the file.
    status=0
    $command >>"$file" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then
        echo "FAILED (exit $status): $name" | tee -a "$file"
    fi
    if [ "$os" != slurm ]; then { echo; load after; } >>"$file"; fi
}

degrees="f64:3 f64:8 f32:8"

# auto_resolves <precision> <p> <ranks per node>: the strategy `auto` resolves to under the
# rule of Phase 5N T4 with BUDGET (M2lStrategy::resolve_on_node): dense iff the dense
# tables of a rank, 332 (p + 1)^4 values, are at most 9 MB and those of the node at most
# the budget, else rotation.
auto_resolves() {
    bytes=8
    if [ "$1" = f32 ]; then bytes=4; fi
    rank=$((332 * ($2 + 1) * ($2 + 1) * ($2 + 1) * ($2 + 1) * bytes))
    if [ "$rank" -le 9000000 ] && [ $(($3 * rank)) -le $((BUDGET * 1000000)) ]; then
        echo dense
    else
        echo rotation
    fi
}

# t2_launch <name prefix> <ranks> <threads> <variant> <scaling arguments...>: one launch of
# T2's sweeps with the strategy of <variant> (dense, classes, rotation, batched, or
# batched-fused: Batched with --mul-add fused), --kinds and overlap off.
t2_launch() {
    prefix=$1 r=$2 t=$3 variant=$4
    shift 4
    strategy=${variant%-fused} mul_add=separate
    if [ "$variant" = batched-fused ]; then mul_add=fused; fi
    launch "$prefix-$variant-r${r}x$t" "$r" "$t" --strategy "$strategy" --mul-add "$mul_add" \
        --kinds --overlap off "$@"
}

t2_variants="dense classes rotation batched batched-fused"

# t5_launch <name prefix> <ranks> <threads> <scaling arguments...>: one launch per m of
# T5_RHS, with --kinds (on the host; t5_kinds empty on a device, where the synchronous kind
# timings would add a sync per level call), overlap off and three timed evaluations.
t5_kinds=--kinds
t5_launch() {
    prefix=$1 r=$2 t=$3
    shift 3
    for m in $T5_RHS; do
        launch "$prefix-rhs$m-r${r}x$t" "$r" "$t" --rhs "$m" $t5_kinds --overlap off --evals 3 "$@"
    done
}

for sweep in "$@"; do
    case "$sweep" in
    t5-host | t5-threads)
        if [ "$sweep" = t5-host ]; then
            r=$((CORES * NODES)) t=1
        else
            r=$NODES t=$CORES
        fi
        for dp in f64:3 f64:8 f32:3 f32:8; do
            prec=${dp%%:*} p=${dp##*:}
            for strategy in auto batched; do
                t5_launch "$sweep-cube-$prec-p$p-$strategy" "$r" "$t" --dist cube \
                    --n 1000000 --precision "$prec" --p "$p" --strategy "$strategy"
            done
        done
        ;;
    t5-device)
        t5_kinds=""
        degrees="f32:3 f32:8"
        if [ "$DEVICE" = cuda ]; then degrees="f32:3 f32:8 f64:3 f64:8"; fi
        for dp in $degrees; do
            prec=${dp%%:*} p=${dp##*:}
            t5_launch "t5-device-$DEVICE-cube-$prec-p$p" 1 "$CORES" --dist cube --n 1000000 \
                --precision "$prec" --p "$p" --backend "$DEVICE"
        done
        ;;
    t5-nodes)
        if [ "$os" = slurm ]; then r=$((CORES_PER_NODE * NODES)); else r=$LARGEST; fi
        for n in 1000000 10000000; do
            t5_launch "t5-nodes-cube-n$n-f64-p8" "$r" 1 --dist cube --n "$n" --precision f64 \
                --p 8
        done
        ;;
    t2-host)
        for dist in cube plummer; do
            for dp in f64:3 f64:6 f64:8 f64:12 f64:18 f32:3 f32:6 f32:8; do
                prec=${dp%%:*} p=${dp##*:} evals=3
                if [ "$p" -ge 12 ]; then evals=2; fi
                if [ "$p" -ge 18 ]; then evals=1; fi
                for t in 1 "$CORES"; do
                    for variant in $t2_variants; do
                        t2_launch "t2-host-$dist-$prec-p$p" 1 "$t" "$variant" --dist $dist \
                            --n 1000000 --precision "$prec" --p "$p" --evals "$evals"
                    done
                done
            done
        done
        ;;
    t2-node)
        for run in cube:8 plummer:8 cube:12; do
            dist=${run%%:*} p=${run##*:}
            for split in $T2_SPLITS; do
                r=${split%x*} t=${split#*x}
                for variant in $t2_variants; do
                    t2_launch "t2-node-$dist-f64-p$p" "$r" "$t" "$variant" --dist "$dist" \
                        --n 1000000 --precision f64 --p "$p" --evals 3
                done
            done
        done
        ;;
    strong | strong-small)
        n=1000000
        if [ "$sweep" = strong-small ]; then n=100000; fi
        for dist in cube plummer; do
            for dp in $degrees; do
                for r in $RANKS; do
                    launch "$sweep-$dist-n$n-${dp%%:*}-p${dp##*:}-r${r}x1" "$r" 1 \
                        --dist $dist --n $n --precision "${dp%%:*}" --p "${dp##*:}" \
                        --overlap both
                done
            done
        done
        ;;
    weak)
        for dist in cube plummer; do
            for dp in $degrees; do
                for r in $RANKS; do
                    launch "weak-$dist-${dp%%:*}-p${dp##*:}-r${r}x1" "$r" 1 \
                        --dist $dist --n-per-rank 100000 --precision "${dp%%:*}" \
                        --p "${dp##*:}" --overlap both
                done
            done
        done
        ;;
    threads)
        for dist in cube plummer; do
            for split in $THREAD_SPLITS; do
                r=${split%x*} t=${split#*x}
                launch "threads-$dist-r${r}x$t" "$r" "$t" \
                    --dist $dist --n 1000000 --precision f64 --p 8 --overlap both
            done
        done
        ;;
    redistribution)
        counts=8
        if [ "$os" = linux ]; then counts="8 $LARGEST"; fi
        for r in $counts; do
            for dist in cube plummer; do
                for input in rank0 share owners; do
                    launch "redistribution-$dist-$input-r${r}x1" "$r" 1 \
                        --dist $dist --n 1000000 --precision f64 --p 8 --overlap off \
                        --input $input --builds 3
                done
            done
        done
        ;;
    balance)
        counts=8
        if [ "$os" = linux ]; then counts="8 $LARGEST"; fi
        for r in $counts; do
            for dist in cube plummer clusters; do
                launch "balance-$dist-r${r}x1" "$r" 1 \
                    --dist $dist --n 1000000 --precision f64 --p 8 --overlap off
            done
        done
        ;;
    sixteen)
        if [ "$os" != macos ]; then
            echo "sixteen: the M3 Max only" >&2
            continue
        fi
        launch "sixteen-cube-r16x1" 16 1 \
            --dist cube --n 1000000 --precision f64 --p 8 --overlap both
        ;;
    node-rule)
        # T3's grid, one launch per cell and strategy; one rank evaluates 3 times.
        for split in $NODE_GRID; do
            r=${split%x*} t=${split#*x}
            evals=$EVALS
            if [ "$r" -eq 1 ] && [ "$t" -eq 1 ]; then evals=3; fi
            for dist in cube plummer; do
                for dp in f64:8 f32:8 f64:3 f64:12; do
                    prec=${dp%%:*} p=${dp##*:}
                    strategies=auto
                    if [ "$p" -ne 12 ]; then
                        case $(auto_resolves "$prec" "$p" "$r") in
                        dense) strategies="auto rotation" ;;
                        *) strategies="auto dense" ;;
                        esac
                    fi
                    for strategy in $strategies; do
                        launch "node-rule-$dist-$prec-p$p-$strategy-r${r}x$t" "$r" "$t" \
                            --dist $dist --n 1000000 --precision "$prec" --p "$p" \
                            --strategy $strategy --overlap off --evals $evals
                    done
                done
            done
        done
        # The cap: f64 p = 7 (10.9 MB) and f32 p = 9 (13.3 MB) on one rank.
        for dist in cube plummer; do
            for dp in f64:7 f32:9; do
                for strategy in dense rotation; do
                    launch "node-rule-$dist-${dp%%:*}-p${dp##*:}-$strategy-r1x1" 1 1 \
                        --dist $dist --n 1000000 --precision "${dp%%:*}" --p "${dp##*:}" \
                        --strategy $strategy --overlap off --evals 3
                done
            done
        done
        # The errors of the new default against Dense (the default before T4).
        for r in 1 $LARGEST; do
            for dp in f64:8 f32:8; do
                for strategy in auto dense; do
                    launch "node-rule-check-cube-${dp%%:*}-p${dp##*:}-$strategy-r${r}x1" "$r" 1 \
                        --dist cube --n 1000000 --precision "${dp%%:*}" --p "${dp##*:}" \
                        --strategy $strategy --overlap off --errors 8 --reference --evals 1
                done
            done
        done
        ;;
    crossover)
        for r in $CROSSOVER; do
            for dist in cube plummer; do
                for dp in f32:8 f64:6; do
                    for strategy in dense rotation; do
                        launch "crossover-$dist-${dp%%:*}-p${dp##*:}-$strategy-r${r}x1" "$r" 1 \
                            --dist $dist --n 1000000 --precision "${dp%%:*}" \
                            --p "${dp##*:}" --strategy $strategy --overlap off
                    done
                done
            done
        done
        ;;
    strategy)
        for dist in cube plummer; do
            for strategy in rotation classes batched; do
                for r in $RANKS; do
                    launch "strategy-$dist-$strategy-r${r}x1" "$r" 1 \
                        --dist $dist --n 1000000 --precision f64 --p 8 --overlap off \
                        --strategy $strategy
                done
            done
        done
        ;;
    check)
        first=${RANKS%% *}
        for dist in cube plummer; do
            for dp in $degrees; do
                for r in $RANKS; do
                    launch "check-$dist-n100000-${dp%%:*}-p${dp##*:}-r${r}x1" "$r" 1 \
                        --dist $dist --n 100000 --precision "${dp%%:*}" --p "${dp##*:}" \
                        --overlap off --errors 8 --reference --evals 2
                done
                for r in $first $LARGEST; do
                    launch "check-$dist-n1000000-${dp%%:*}-p${dp##*:}-r${r}x1" "$r" 1 \
                        --dist $dist --n 1000000 --precision "${dp%%:*}" --p "${dp##*:}" \
                        --overlap off --errors 8 --reference --evals 2
                done
            done
        done
        ;;
    oversubscribed)
        if [ "$os" != linux ]; then
            echo "oversubscribed: locust only" >&2
            continue
        fi
        for r in 256 512; do
            launch "oversubscribed-cube-r${r}x1" "$r" 1 \
                --dist cube --n 1000000 --precision f64 --p 8 --overlap both \
                --errors 8 --reference --evals 1
        done
        ;;
    calibrate)
        n=${CAL_N:-10000000}
        for dist in cube plummer clusters; do
            for run in f64:3:dense f64:8:rotation f32:8:rotation f32:8:dense; do
                prec=${run%%:*} rest=${run#*:}
                p=${rest%%:*} strategy=${rest#*:}
                launch "calibrate-$dist-n$n-$prec-p$p-$strategy-r${LARGEST}x1" "$LARGEST" 1 \
                    --dist $dist --n "$n" --precision "$prec" --p "$p" --strategy "$strategy" \
                    --overlap off --kinds --cut points --evals 3
            done
        done
        ;;
    cut)
        if [ "$os" = slurm ]; then
            ns=${CUT_NS:-10000000}
            runs=""
            for split in $SPLITS; do runs="$runs $((${split%x*} * NODES))x${split#*x}"; done
        else
            ns=${CUT_NS:-"1000000 10000000"}
            runs=""
            if [ "$os" = macos ]; then default=8; else default="8 72"; fi
            for r in ${CUT_RANKS:-$default}; do runs="$runs ${r}x1"; done
        fi
        for n in $ns; do
            for run in $runs; do
                r=${run%x*} t=${run#*x}
                for dist in cube plummer clusters; do
                    for cut in points work; do
                        launch "cut-$dist-n$n-f64-p8-$cut-N$NODES-r${r}x$t" "$r" "$t" \
                            --dist $dist --n "$n" --precision f64 --p 8 --overlap off \
                            --kinds --cut $cut --evals 3
                    done
                done
            done
        done
        ;;
    cut-counts)
        if [ "$os" = macos ]; then
            echo "cut-counts: locust and Kathleen only" >&2
            continue
        fi
        if [ "$os" = slurm ]; then
            counts=${COUNT_RANKS:-"$((80 * NODES)) $((160 * NODES))"}
            OVERSUBSCRIBED=1
        else
            counts=${COUNT_RANKS:-512}
        fi
        for r in $counts; do
            for n in 1000000 10000000; do
                for dist in cube plummer clusters; do
                    for cut in points work; do
                        launch "cut-counts-$dist-n$n-f64-p8-$cut-N$NODES-r${r}x1" "$r" 1 \
                            --dist $dist --n "$n" --precision f64 --p 8 --strategy rotation \
                            --overlap off --cut $cut --evals 0
                    done
                done
            done
        done
        OVERSUBSCRIBED=
        ;;
    t9-strong)
        if [ "$os" != slurm ]; then
            echo "$sweep: under Slurm only" >&2
            continue
        fi
        for n in ${T9_NS:-10000000 100000000}; do
            evals=$EVALS
            if [ "$n" -gt 10000000 ]; then evals=${T9_LARGE_EVALS:-2}; fi
            for split in $(printf '%s\n' $SPLITS | sort -t x -k 1,1n); do
                per=${split%x*} t=${split#*x}
                for dist in cube plummer; do
                    for dp in $degrees; do
                        prec=${dp%%:*} p=${dp##*:}
                        launch "t9-strong-$dist-n$n-$prec-p$p-auto-N$NODES-${per}x$t" \
                            $((per * NODES)) "$t" --dist $dist --n "$n" --precision "$prec" \
                            --p "$p" --strategy auto --overlap both --evals "$evals" --errors 1
                        if [ "$n" -le 10000000 ] && [ "$dp" = f64:8 ]; then
                            launch "t9-strong-$dist-n$n-$prec-p$p-auto-N$NODES-${per}x$t-cutpoints" \
                                $((per * NODES)) "$t" --dist $dist --n "$n" --precision "$prec" \
                                --p "$p" --strategy auto --overlap both --evals "$evals" \
                                --errors 1 --cut points
                        fi
                    done
                done
            done
        done
        ;;
    t9-weak)
        if [ "$os" != slurm ]; then
            echo "$sweep: under Slurm only" >&2
            continue
        fi
        for split in $SPLITS; do
            per=${split%x*} t=${split#*x}
            # 2.5 x 10^4 points per rank at one thread, 10^6 per socket at 2 x 20.
            if [ "$t" -eq 1 ]; then nper=25000; else nper=$((1000000 * SOCKETS / per)); fi
            for dist in cube plummer clusters; do
                for dp in $degrees f32:3; do
                    prec=${dp%%:*} p=${dp##*:}
                    launch "t9-weak-$dist-$prec-p$p-auto-N$NODES-${per}x$t" \
                        $((per * NODES)) "$t" --dist $dist --n-per-rank "$nper" \
                        --precision "$prec" --p "$p" --strategy auto --overlap both --errors 1
                done
            done
        done
        ;;
    t9-model)
        for nper in 10000 1000000; do
            for dist in cube plummer clusters; do
                for dp in $degrees f32:3; do
                    prec=${dp%%:*} p=${dp##*:}
                    launch "t9-model-$dist-per$nper-$prec-p$p-auto-r${LARGEST}x1" "$LARGEST" 1 \
                        --dist $dist --n-per-rank "$nper" --precision "$prec" --p "$p" \
                        --strategy auto --overlap off --evals 5
                done
            done
        done
        ;;
    t9-counts)
        if [ "$os" = macos ]; then
            echo "$sweep: locust and Kathleen only" >&2
            continue
        fi
        if [ "$os" = slurm ]; then
            counts=${COUNT_RANKS:-"$((80 * NODES)) $((160 * NODES))"}
            OVERSUBSCRIBED=1
        else
            counts=${COUNT_RANKS:-"128 256 512"}
        fi
        for r in $counts; do
            for size in n1000000 n10000000 per25000; do
                case $size in
                n*) arg="--n ${size#n}" ;;
                *) arg="--n-per-rank ${size#per}" ;;
                esac
                for dist in cube plummer clusters; do
                    # shellcheck disable=SC2086
                    launch "t9-counts-$dist-$size-f64-p8-rotation-r${r}x1" "$r" 1 \
                        --dist $dist $arg --precision f64 --p 8 --strategy rotation \
                        --overlap off --evals 0
                done
            done
        done
        OVERSUBSCRIBED=
        ;;
    host | nodes-strong | nodes-weak | nodes-check | nodes-large)
        if [ "$os" != slurm ]; then
            echo "$sweep: under Slurm only" >&2
            continue
        fi
        case "$sweep" in
        host)
            # Dense was the default at p <= 8 until Phase 5N T4: named as such; "auto" (f32
            # p = 8) is Dense on one rank under the rule of T4 too.
            for t in 1 "$CORES"; do
                for dist in cube plummer; do
                    for run in f64:3:dense f64:3:rotation f64:8:dense f64:8:rotation f32:8:auto; do
                        prec=${run%%:*} rest=${run#*:}
                        p=${rest%%:*} strategy=${rest#*:}
                        case "$t:$run" in
                        1:f64:3:dense | 1:f64:8:dense | 1:f32:8:auto | 1:f64:8:rotation | "$CORES":f64:8:dense)
                            continue
                            ;;
                        esac
                        launch "host-$dist-n1000000-$prec-p$p-$strategy-r1x$t" 1 "$t" \
                            --dist $dist --n 1000000 --precision "$prec" --p "$p" \
                            --strategy "$strategy" --overlap off
                    done
                done
            done
            ;;
        nodes-strong | nodes-weak)
            ns="1000000 10000000"
            if [ "$sweep" = nodes-weak ]; then ns=$((1000000 * NODES)); fi
            for n in $ns; do
                for split in $SPLITS; do
                    per=${split%x*} t=${split#*x}
                    for dist in cube plummer; do
                        # 8:dense is Phase 5N T2's 8:auto (the default before T4).
                        for run in 3:auto 8:dense 8:rotation; do
                            p=${run%%:*} strategy=${run#*:}
                            launch "$sweep-$dist-n$n-f64-p$p-$strategy-N$NODES-${per}x$t" \
                                $((per * NODES)) "$t" --dist $dist --n "$n" --precision f64 \
                                --p "$p" --strategy "$strategy" --overlap both
                        done
                    done
                done
            done
            ;;
        nodes-check)
            r=$((CORES * NODES))
            for dist in cube plummer; do
                for dp in $degrees; do
                    launch "nodes-check-$dist-n1000000-${dp%%:*}-p${dp##*:}-N$NODES-${CORES}x1" \
                        "$r" 1 --dist $dist --n 1000000 --precision "${dp%%:*}" \
                        --p "${dp##*:}" --overlap off --errors 8 --reference --evals 2
                done
                for dp in f64:3 f64:8; do
                    launch "nodes-check-$dist-n10000000-${dp%%:*}-p${dp##*:}-N$NODES-${CORES}x1" \
                        "$r" 1 --dist $dist --n 10000000 --precision "${dp%%:*}" \
                        --p "${dp##*:}" --overlap off --errors 8 --evals 2
                done
            done
            ;;
        nodes-large)
            n=100000000
            for split in $(printf '%s\n' $SPLITS | sort -t x -k 1,1n); do
                per=${split%x*} t=${split#*x}
                for dist in cube plummer; do
                    for p in 3 8; do
                        launch "$sweep-$dist-n$n-f64-p$p-auto-N$NODES-${per}x$t" \
                            $((per * NODES)) "$t" --dist $dist --n "$n" --precision f64 \
                            --p "$p" --strategy auto --overlap off --evals 2 --errors 1
                    done
                done
            done
            ;;
        esac
        ;;
    *)
        echo "unknown sweep $sweep" >&2
        exit 2
        ;;
    esac
done
