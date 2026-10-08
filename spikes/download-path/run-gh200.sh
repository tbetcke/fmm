#!/bin/bash
# spikes/download-path/run-gh200.sh: the spike's runs on locust (Phase 4S T11, REPORT.md).
#
#     tools/gh200/remote.sh 'setsid nohup spikes/download-path/run-gh200.sh > /dev/null 2>&1 &'
#
# Run from the checkout tools/gh200/sync.sh made, inside tools/gh200/env.sh (remote.sh does
# both). Writes every output to $1 (default /data/ucahtbe/logs/t11), outside the synced
# tree, so the next sync does not delete it. The load checks of docs/phase4s/README.md,
# "Timing", go to load-before.txt and load-after.txt; the clocks to clocks-*.txt.
set -u

out=${1:-/data/ucahtbe/logs/t11}
mkdir -p "$out"
bin=target/release/nd-fmm-spike-download-path
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

load() {
    date -u
    uptime
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    ps -eo user,pcpu,pmem,etime,comm --sort=-pcpu | head -8
}

load > "$out/load-before.txt"
nvidia-smi -q -d CLOCK > "$out/clocks-before.txt"
cargo build --release -p nd-fmm-spike-download-path --features cuda > "$out/build.txt" 2>&1

# The download, upload, zeroing and Device sections, f32 and f64.
timeout 1800 "$bin" --backends cuda --precisions f32,f64 \
    > "$out/cuda-main.md" 2> "$out/cuda-main.err"

# The host pool, each precision in a process of its own.
for p in f32 f64; do
    timeout 600 "$bin" --backends cuda --precisions "$p" --sections pool \
        > "$out/cuda-pool-$p.md" 2>> "$out/cuda-main.err"
done

# cuMemAllocHost per download: one sequence per process under nsys.
for run in steady:1 steady:10 evaluation:1 evaluation:10 mixed:10 persistent:10; do
    sequence=${run%:*}
    iterations=${run#*:}
    name="nsys-$sequence-$iterations"
    timeout 900 nsys profile --trace=cuda,osrt --force-overwrite=true -o "$out/$name" \
        "$bin" --backends cuda --precisions f32 --sections pool \
        --pool-sequences "$sequence" --pool-iterations "$iterations" \
        > "$out/$name.md" 2>&1
    nsys stats --report cuda_api_sum --format csv --force-export=true \
        "$out/$name.nsys-rep" > "$out/$name-api.csv" 2>&1
    nsys stats --report cuda_api_trace --format csv \
        "$out/$name.nsys-rep" 2>&1 | grep -E 'Start|cuMemAllocHost|cuMemFreeHost' \
        > "$out/$name-alloc.csv"
done

load > "$out/load-after.txt"
nvidia-smi -q -d CLOCK > "$out/clocks-after.txt"
echo done > "$out/done"
