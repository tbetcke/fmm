#!/usr/bin/env bash
# Builds and runs the MPI-owning test executables and the nd-octree MPI examples at
# several rank counts, for the run-tests-mpi job (Phase 5 T3). Correctness only: the
# times it prints are for the job's measurement, never asserted.
#
# Usage:
#   run-mpi-tests.sh build <list>              build in debug, write the executables to <list>
#   run-mpi-tests.sh run <list> <ranks>...     run every executable of <list> at each count
#
# `build` builds the targets of TARGETS below without running them (`cargo test --no-run`
# for the tests, `cargo build` for the examples) and takes each executable's path from
# cargo's JSON output (`--message-format=json`), never from a glob over target/debug/deps.
# It writes one line per target to <list>: kind, name, path, separated by tabs, and fails
# if a target produced no executable.
#
# `run` launches each executable as `timeout mpirun -n <ranks>`, tests with
# `--test-threads=1 --nocapture`, examples with no arguments, both with stdin from
# /dev/null (mpirun would otherwise read it). `--nocapture` puts a panic's message into
# the log at once: a rank whose test panics then blocks in MPI_Finalize while unwinding,
# before libtest would print its captured output, so without it a hang shows no cause.
# It also logs the per-scenario times of mpi_exec. It runs every launch even after a
# failure, prints a table of the results and their wall times (also to
# $GITHUB_STEP_SUMMARY when set), and exits non-zero if any launch failed. A failed assertion on one rank can leave the other ranks
# blocked in a collective: the `timeout` (RUN_TIMEOUT seconds, default 300, then KILL
# after 30 more) ends such a run, and the workflow bounds the step as well.
#
# The environment passes through mpirun to the ranks: the workflow sets RUST_MIN_STACK,
# the BLAS thread variables and Open MPI's oversubscription there. MPIRUN_FLAGS adds
# flags to every mpirun, for example the macOS loopback flags of the root CLAUDE.md for a
# run by hand (never on the runner).
#
# Plain bash 3.2 (macOS's /bin/bash), so the same script runs on the M3 Max.
set -euo pipefail

# kind, package, target: the MPI-owning test executables (one MPI-initialising test
# each; fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md) and the nd-octree MPI examples
# (octree/CLAUDE.md, "MPI examples"). mpi_threading runs on several ranks too: its
# MpiThreading error rides on the agreement all-reduce of `build`'s step 1, which the
# several-rank run exercises.
TARGETS="test nd-fmm-plan mpi_regressions
test nd-fmm-exec mpi_exec
test nd-fmm-exec mpi_threading
example nd-octree test_mpi_complete_tree
example nd-octree test_mpi_global_bounding_box
example nd-octree test_mpi_leaf_lookup
example nd-octree test_mpi_vtk
example nd-octree test_mpi_construction_edge_cases
example nd-octree test_mpi_weighted_partition"

now() {
    python3 -c 'import time; print(f"{time.time():.3f}")'
}

# Reads cargo's JSON messages on stdin; prints the rendered compiler messages to stderr
# and "kind<TAB>name<TAB>executable" for every test or example executable to stdout.
executables() {
    python3 -c '
import json, sys
for line in sys.stdin:
    if not line.startswith("{"):
        continue
    message = json.loads(line)
    if message.get("reason") == "compiler-message":
        sys.stderr.write(message["message"].get("rendered") or "")
    elif message.get("reason") == "compiler-artifact" and message.get("executable"):
        kinds = message["target"]["kind"]
        kind = "test" if "test" in kinds else "example" if "example" in kinds else None
        if kind:
            print(kind, message["target"]["name"], message["executable"], sep="\t")
'
}

build() {
    local list=$1 found kind package target args path
    mkdir -p "$(dirname "$list")"
    found=$list.cargo
    : >"$found"
    while read -r kind package target; do
        if [ "$kind" = test ]; then
            args=(test --no-run -p "$package" --test "$target")
        else
            args=(build -p "$package" --example "$target")
        fi
        echo "=== cargo ${args[*]}" >&2
        cargo "${args[@]}" --message-format=json </dev/null | executables >>"$found"
    done <<<"$TARGETS"
    : >"$list"
    while read -r kind package target; do
        path=$(awk -F '\t' -v k="$kind" -v t="$target" '$1 == k && $2 == t { p = $3 } END { print p }' "$found")
        if [ -z "$path" ]; then
            echo "::error::$package: no executable for $kind $target in cargo's output" >&2
            exit 1
        fi
        printf '%s\t%s\t%s\n' "$kind" "$target" "$path" >>"$list"
    done <<<"$TARGETS"
    rm -f "$found"
    echo "Executables:" >&2
    cat "$list" >&2
}

run() {
    local list=$1 kind name path ranks start end status seconds result failed=0 table
    local command=()
    shift
    local flags=()
    read -r -a flags <<<"${MPIRUN_FLAGS:-}"
    table=$'| executable | ranks | result | wall time (s) |\n| --- | ---: | --- | ---: |'
    while IFS=$'\t' read -r kind name path; do
        for ranks in "$@"; do
            command=(timeout -k 30 "${RUN_TIMEOUT:-300}" mpirun ${flags[@]+"${flags[@]}"} -n "$ranks" "$path")
            if [ "$kind" = test ]; then
                command+=(--test-threads=1 --nocapture)
            fi
            echo
            echo "=== $kind $name at $ranks ranks: ${command[*]}"
            start=$(now)
            status=0
            "${command[@]}" </dev/null || status=$?
            end=$(now)
            seconds=$(python3 -c "print(f'{$end - $start:.1f}')")
            if [ "$status" -eq 0 ]; then
                result=passed
            elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
                result="timed out (exit $status)"
                failed=$((failed + 1))
            else
                result="failed (exit $status)"
                failed=$((failed + 1))
            fi
            echo "=== $kind $name at $ranks ranks: $result in $seconds s"
            table+=$'\n'"| $name | $ranks | $result | $seconds |"
        done
    done <"$list"
    echo
    echo "$table"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        printf '%s\n\n' "$table" >>"$GITHUB_STEP_SUMMARY"
    fi
    if [ "$failed" -gt 0 ]; then
        echo "::error::$failed MPI launch(es) failed or timed out"
        exit 1
    fi
}

case "${1:-}" in
build) [ "$#" -eq 2 ] || { echo "usage: $0 build <list>" >&2; exit 2; }; build "$2" ;;
run) [ "$#" -ge 3 ] || { echo "usage: $0 run <list> <ranks>..." >&2; exit 2; }; shift; run "$@" ;;
*) echo "usage: $0 build <list> | run <list> <ranks>..." >&2; exit 2 ;;
esac
