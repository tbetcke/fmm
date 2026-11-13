#!/bin/bash
# tools/helmholtz/ch4-0d1b-runs.sh: the machine runs of Helmholtz CH4.0d1b, the empty-box
# rule's plan built once (docs/helmholtz/tasks/CH4.0d1b.md, "Machine runs"), from the
# repository root after the example was built in release:
#
#     cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
#     tools/helmholtz/ch4-0d1b-runs.sh <out-dir> <stage…>
#
# The cases of CH4.0d1's runs (tools/helmholtz/ch4-0d1-runs.sh), so that the builds and the
# checksums compare with its jobs 633596 (`large`) and 633566 (`w3`):
# - `large`: tools/helmholtz/ch4-0d1-runs.sh's `large` (W1 at kD = 500 and W3 at
#   N = 4·10⁶, kD = 100; 10⁻⁶ and 10⁻⁸; Rotation and the default; two evaluations; both
#   orders of `--empty-boxes`);
# - `w3`: the under-resolved W3 (N = 10⁶) at kD = 500, 10⁻⁶ only (Rotation's evaluation of
#   the rule off at 10⁻⁸ takes about 1,000 s and adds nothing to the build), both orders;
# - `traffic`: tools/helmholtz/ch4-0d1-runs.sh's `traffic` (4 ranks × T/4 threads).
# One rank × THREADS threads (default 40, a Kathleen node) unless named. Each launch's output
# in <out-dir>/<stage>-<case>-<order>.md, a line per launch in <out-dir>/runs.log. Times are
# read on Kathleen's exclusive nodes only, never asserted. TIMEOUT (default 18000) bounds a
# launch.
set -u
out=$1
shift
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
X=target/release/examples/helmholtz_accuracy
T=${THREADS:-40}
TIMEOUT=${TIMEOUT:-18000}

for stage in "$@"; do
    case $stage in
    large | traffic)
        tools/helmholtz/ch4-0d1-runs.sh "$out" "$stage" || exit
        ;;
    w3)
        for order in on,off off,on; do
            name="w3-500-${order/,/-}"
            start=$SECONDS
            timeout "$TIMEOUT" "$X" --workload w3:1000000 --kd 500 --eps 1e-6 \
                --m2l rotation,auto-pw --levels --empty-boxes "$order" --threads "$T" \
                < /dev/null > "$out/$name.md" 2> "$out/$name.err"
            echo "$name: exit $?, $((SECONDS - start)) s, $(date +%T)" | tee -a "$out/runs.log"
        done
        ;;
    *)
        echo "unknown stage $stage" >&2
        exit 2
        ;;
    esac
done
