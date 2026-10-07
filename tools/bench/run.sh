#!/bin/sh
# tools/bench/run.sh: the one-command FMM benchmark (Phase 4S T6). POSIX sh.
#
#     tools/bench/run.sh [options of nd-fmm-bench]
#     tools/bench/run.sh --backend cuda --precision f32,f64 --degree 3,6,8
#     tools/bench/run.sh --help
#
# It picks the cargo features from --backend (the union over the listed backends:
# `host,cuda` builds with `--features cuda`, the host alone with none), sets every BLAS
# thread variable to 1 and runs
#
#     cargo run --release -p nd-fmm-bench --features ... -- "$@"
#
# from any directory: the report goes to bench-results/ under the current directory
# unless --output says otherwise. On locust it first sources tools/gh200/env.sh if that
# environment is not active yet. Metal needs a process with GPU access (outside the
# macOS sandbox).

set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)

# On locust (the environment of tools/gh200/ exists), activate it unless it is active.
if [ -z "${FMM_GH200_ENV:-}" ] && [ -d /data/ucahtbe/envs/fmm-gh200 ]; then
    # env.sh and spack's activation are not written for `set -u`.
    set +u
    . "$root/tools/gh200/env.sh"
    set -u
fi

# The features: the union over every --backend list.
cpu='' metal='' cuda=''
next=''
for arg in "$@"; do
    case "$next$arg" in
    backend:*) list=${arg} ;;
    --backend=*) list=${arg#--backend=} ;;
    --backend) next=backend:; continue ;;
    *) list='' ;;
    esac
    next=''
    for backend in $(printf '%s' "$list" | tr ',' ' '); do
        case "$backend" in
        cpu) cpu=cpu ;;
        metal) metal=metal ;;
        cuda) cuda=cuda ;;
        esac
    done
done
features=$(printf '%s\n' $cpu $metal $cuda | paste -s -d, -)

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ -n "$features" ]; then
    exec cargo run --release --manifest-path "$root/Cargo.toml" -p nd-fmm-bench \
        --features "$features" -- "$@"
else
    exec cargo run --release --manifest-path "$root/Cargo.toml" -p nd-fmm-bench -- "$@"
fi
