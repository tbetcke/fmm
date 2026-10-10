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
#   strategy        the M2L strategies with small tables, Rotation and Classes, against the
#                   default (Dense at p = 8), at every rank count of RANKS: the N = 10^6
#                   cube and Plummer, f64 p = 8, overlap off (does the dense tables' size
#                   per rank limit the downward pass?)
#   check           the errors against the direct sum (8 charge vectors, 1,000 targets)
#                   and the one-rank reference: N = 10^5 at every rank count, N = 10^6 at
#                   one rank and the largest count
#   oversubscribed  locust only: 256 and 512 ranks on 72 cores, the N = 10^6 cube, f64
#                   p = 8, correctness and memory only (never a timing); the soft limit
#                   of open files is raised to the hard limit for these launches
#
# Environment:
#   OUT      output directory (default bench-results/scaling-<host>-<date>); one file per
#            launch, with the load before and after on Linux
#   RANKS    the rank counts of strong and weak (default "1 2 4 8 12" on macOS,
#            "1 2 4 8 16 32 64 72" on Linux)
#   EVALS    timed evaluations per path (default 10)
#   TIMEOUT  seconds per launch (default 3600)
#   QUIET    on macOS, the CPU percentage the other processes may use before a launch
#            starts (default 200, two cores): each launch waits, up to an hour, until
#            they use less (a laptop's background jobs, such as a virus scanner, would
#            otherwise share the cores), and the file records the wait
#   DRY_RUN  if set, print the commands only
#
# Every BLAS thread variable is set to 1. On macOS `mpirun` gets the loopback flags (root
# CLAUDE.md, "MPI"); on Linux it reports the binding and binds each rank to its cores
# (`--map-by slot:PE=<threads> --bind-to core`). On locust it first sources
# tools/gh200/env.sh if that environment is not active. Timings go into the report only
# with the load stated (docs/phase5/README.md, "Ranks on locust").
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
case "$(uname -s)" in
Darwin)
    os=macos
    RANKS=${RANKS:-"1 2 4 8 12"}
    CORES=12
    THREAD_SPLITS="1x12 2x6 4x3 6x2 12x1"
    LARGEST=12
    ;;
*)
    os=linux
    RANKS=${RANKS:-"1 2 4 8 16 32 64 72"}
    CORES=72
    THREAD_SPLITS="1x72 4x18 8x9 18x4 72x1"
    LARGEST=72
    ;;
esac

exe=target/release/examples/scaling
if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p "$OUT"
    cargo build --release -p nd-fmm-validate --example scaling
fi

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
    if [ "$os" = macos ]; then
        flags="--mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0"
        if [ "$ranks" -gt 16 ]; then flags="$flags --oversubscribe"; fi
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
    command="timeout $TIMEOUT mpirun $flags -n $ranks $exe --threads $threads --evals $EVALS $*"
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
        load before
        echo
    } >"$file"
    # mpirun's binding report goes to stderr; keep it in the file.
    if ! $command >>"$file" 2>&1; then
        echo "FAILED (exit $?): $name" | tee -a "$file"
    fi
    { echo; load after; } >>"$file"
}

degrees="f64:3 f64:8 f32:8"

for sweep in "$@"; do
    case "$sweep" in
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
    strategy)
        for dist in cube plummer; do
            for strategy in rotation classes; do
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
    *)
        echo "unknown sweep $sweep" >&2
        exit 2
        ;;
    esac
done
