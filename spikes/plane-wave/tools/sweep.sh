#!/bin/sh
# The spike's timing sweep (SPIKE_REPORT.md, "Measurements"): level 4 (T1's level, 4,096
# boxes, 584,136 V pairs), f64, p = 3, 6, 8, 12, 18, one process and K processes at once.
# Each line of `level` or `run_node.sh` output goes to stdout.
#
#   tools/sweep.sh K [LEVEL]
#
# The plane waves run with the matched quadrature (the smallest s whose level error is at
# most 0.1× the M2L truncation error: tests/accuracy.rs, level_3_every_p) and with the
# model's β = 0.1 choice; the baselines once per p. Build first, from the workspace root:
#   cargo build --release -p nd-fmm-spike-plane-wave --example level
set -eu
k=${1:?usage: sweep.sh K [LEVEL]}
level=${2:-4}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
bin=${LEVEL_BIN:-$root/target/release/examples/level}
reps=${REPS:-3}
extra=${LEVEL_ARGS:-}
# p, matched s, the model's β = 0.1 s
for spec in "3 10 14" "6 15 18" "8 18 20" "12 21 25" "18 32 30"; do
    set -- $spec
    p=$1 matched=$2 model=$3
    echo "# p=$p level=$level copies=1 $(date +%H:%M:%S)"
    "$bin" --level "$level" --p "$p" --s "$matched" --reps "$reps" --variants batched,fused,rotation,merge --check $extra
    "$bin" --level "$level" --p "$p" --s "$model" --reps "$reps" --variants merge $extra
    if [ "$k" -gt 1 ]; then
        echo "# p=$p level=$level copies=$k $(date +%H:%M:%S)"
        LEVEL_BIN=$bin "$here/run_node.sh" "$k" --level "$level" --p "$p" --s "$matched" --reps "$reps" --variants batched,fused,rotation,merge $extra
        LEVEL_BIN=$bin "$here/run_node.sh" "$k" --level "$level" --p "$p" --s "$model" --reps "$reps" --variants merge $extra
    fi
done
