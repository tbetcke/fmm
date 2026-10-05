#!/bin/sh
# tools/gh200/remote.sh: runs a command on locust inside the environment (Phase 4S T1).
# POSIX sh.
#
#     tools/gh200/remote.sh cargo test -p nd-fmm-math
#     tools/gh200/remote.sh 'nvidia-smi; uptime'
#
# Run it on the M3 Max, inside the checkout that tools/gh200/sync.sh copied. It runs
# the command in the branch's directory on locust (/data/ucahtbe/fmm/<branch>) after
# `. tools/gh200/env.sh`, and returns its exit status. As with ssh, the arguments are
# joined by spaces and read by the remote shell (bash), so quote what that shell should
# see as one word, a pipe or a `;`.
#
# FMM_GH200_HOST (default locust) and FMM_GH200_DIR (default the branch's directory)
# override the target. ssh reads ~/.ssh, which the Claude Code sandbox denies: run it
# outside the sandbox.

set -eu

[ "$#" -gt 0 ] || { echo "usage: remote.sh <command> [args...]" >&2; exit 2; }
host=${FMM_GH200_HOST:-locust}
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" != HEAD ] || branch=detached-$(git rev-parse --short HEAD)
dir=${FMM_GH200_DIR:-/data/ucahtbe/fmm/$branch}

exec ssh "$host" "cd '$dir' && . tools/gh200/env.sh && $*"
