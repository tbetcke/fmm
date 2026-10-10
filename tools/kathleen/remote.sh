#!/bin/sh
# tools/kathleen/remote.sh: runs a short command on a Kathleen login node inside the
# environment (Phase 5N T1). POSIX sh.
#
#     tools/kathleen/remote.sh cargo fmt -- --check
#     tools/kathleen/remote.sh 'lquota; squeue -u $USER'
#
# Run it on the M3 Max, inside the checkout that tools/kathleen/sync.sh copied. It runs
# the command in the branch's directory on Kathleen
# (/scratch/scratch/ucahtbe/fmm/src/<branch>) in a bash login shell, after
# `. tools/kathleen/env.sh`, and returns its exit status. As with ssh, the arguments are
# joined by spaces and read by the remote shell, so quote what that shell should see as
# one word, a pipe or a `;`.
#
# The login node takes only short (under 15 minutes), light commands: sbatch, squeue,
# sacct, reading output, cargo fmt, cargo metadata (README.md, "The login-node rule").
# Builds, tests and runs are jobs: tools/kathleen/submit.sh.
#
# FMM_KATHLEEN_HOST (default kathleen) and FMM_KATHLEEN_DIR (default the branch's
# directory) override the target. ssh reads ~/.ssh, which the Claude Code sandbox
# denies: run it outside the sandbox.

set -eu

[ "$#" -gt 0 ] || { echo "usage: remote.sh <command> [args...]" >&2; exit 2; }
host=${FMM_KATHLEEN_HOST:-kathleen}
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" != HEAD ] || branch=detached-$(git rev-parse --short HEAD)
dir=${FMM_KATHLEEN_DIR:-/scratch/scratch/ucahtbe/fmm/src/$(printf '%s' "$branch" | tr '/' '_')}

# The command travels as one argument of `bash -lc`, quoted for the remote shell.
command="cd '$dir' && . tools/kathleen/env.sh && $*"
quoted=$(printf '%s' "$command" | sed "s/'/'\\\\''/g")
exec ssh "$host" "bash -lc '$quoted'"
