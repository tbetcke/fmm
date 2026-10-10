#!/bin/sh
# tools/kathleen/submit.sh: submits a job script to Kathleen and waits for it (Phase 5N
# T1). POSIX sh.
#
#     tools/kathleen/submit.sh <job> [job arguments...]
#     tools/kathleen/submit.sh [sbatch options...] -- <job> [job arguments...]
#
#     tools/kathleen/submit.sh build.sbatch
#     tools/kathleen/submit.sh check.sbatch simd
#     tools/kathleen/submit.sh --qos=test --nodes=2 -- mpi.sbatch 80
#
# Run it on the M3 Max after tools/kathleen/sync.sh. <job> is a file in
# tools/kathleen/jobs/. It submits the job from the branch's tree on Kathleen (so the job
# runs in that tree, SLURM_SUBMIT_DIR) with its output in the root's logs/
# (<job name>-<job id>.out), then polls `sacct` every FMM_KATHLEEN_POLL seconds (default
# 30) until the job has ended, printing a line when its state changes. At the end it
# prints the job's submit, start and end times, its wait and run time, its state and
# exit code, the output file's path, and the output's last FMM_KATHLEEN_TAIL lines
# (default 40). It exits 0 if the job completed with exit code 0.
#
# Interrupting it does not cancel the job: `ssh kathleen scancel <id>` does.
#
# FMM_KATHLEEN_HOST, FMM_KATHLEEN_DIR and FMM_KATHLEEN_ROOT override the host, the tree
# and the root. ssh reads ~/.ssh, which the Claude Code sandbox denies: run it outside
# the sandbox.

set -eu

host=${FMM_KATHLEEN_HOST:-kathleen}
root=${FMM_KATHLEEN_ROOT:-/scratch/scratch/ucahtbe/fmm}
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" != HEAD ] || branch=detached-$(git rev-parse --short HEAD)
dir=${FMM_KATHLEEN_DIR:-$root/src/$(printf '%s' "$branch" | tr '/' '_')}
poll=${FMM_KATHLEEN_POLL:-30}
tail_lines=${FMM_KATHLEEN_TAIL:-40}

# sbatch options, when the first argument is one, up to `--` (a job argument may hold a
# `--` of its own).
options=
case "${1:-}" in
-*)
    while [ "$#" -gt 0 ] && [ "$1" != -- ]; do
        options="$options '$1'"
        shift
    done
    [ "$#" -gt 0 ] || { echo "submit.sh: sbatch options need a -- before the job" >&2; exit 2; }
    shift
    ;;
esac
[ "$#" -gt 0 ] || { echo "usage: submit.sh [sbatch options... --] <job> [arguments...]" >&2; exit 2; }
job=$1
shift
[ -f "tools/kathleen/jobs/$job" ] || { echo "submit.sh: no tools/kathleen/jobs/$job" >&2; exit 2; }
arguments=
for a in "$@"; do
    arguments="$arguments '$(printf '%s' "$a" | sed "s/'/'\\\\''/g")'"
done

remote() { ssh -o BatchMode=yes "$host" "$1" 2>/dev/null; }

id=$(remote "cd '$dir' && sbatch --parsable --output='$root/logs/%x-%j.out' $options tools/kathleen/jobs/$job $arguments")
case "$id" in
'' | *[!0-9]*) echo "submit.sh: sbatch failed: $id" >&2; exit 1 ;;
esac
echo "submitted $job as job $id at $(date '+%F %T'); output in $root/logs/"

last=
while :; do
    state=$(remote "sacct -X -n -P -j $id --format=State" | head -n 1 | cut -d ' ' -f 1)
    if [ "$state" != "$last" ]; then
        echo "$(date '+%F %T') job $id: ${state:-unknown}"
        last=$state
    fi
    case "$state" in
    PENDING | RUNNING | REQUEUED | RESIZING | SUSPENDED | CONFIGURING | COMPLETING | "") sleep "$poll" ;;
    *) break ;;
    esac
done

remote "sacct -X -n -P -j $id --format=JobName,QOS,NNodes,Submit,Start,End,Elapsed,State,ExitCode,NodeList" |
    head -n 1 | {
    IFS='|' read -r name qos nodes submit start end elapsed state code nodelist
    # Seconds since the epoch of a sacct time (BSD date on macOS, GNU date elsewhere).
    epoch() { date -j -f '%Y-%m-%dT%H:%M:%S' "$1" +%s 2>/dev/null || date -d "$1" +%s 2>/dev/null; }
    if s=$(epoch "$start") && b=$(epoch "$submit"); then wait=$((s - b)); else wait=-; fi
    echo "job $id ($name, $qos, $nodes node(s): $nodelist): submitted $submit, started $start, ended $end"
    echo "job $id: waited ${wait} s, ran $elapsed, $state, exit code $code"
    out=$root/logs/$name-$id.out
    echo "output: $host:$out"
    echo "----- last $tail_lines lines"
    remote "tail -n $tail_lines '$out'"
    [ "$state" = COMPLETED ] && [ "$code" = 0:0 ]
}
