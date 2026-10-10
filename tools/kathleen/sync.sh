#!/bin/sh
# tools/kathleen/sync.sh: copies this working tree to Kathleen (Phase 5N T1). POSIX sh.
#
#     tools/kathleen/sync.sh [host [directory]]
#
# Run it on the M3 Max, anywhere inside a checkout or worktree. It copies the working
# tree, uncommitted changes included, to <host>:<directory>, by default
# kathleen:/scratch/scratch/ucahtbe/fmm/src/<branch>, so that work is built on Kathleen
# without a commit (root CLAUDE.md, "Working agreement"). It leaves out target/ and .git
# (and the session-local .claude/), deletes remote files that are gone locally, and keeps
# the remote target/ so builds stay incremental. It first writes .source-revision
# (`git describe --always --dirty` and the branch; ignored by git), which travels with
# the tree. The model is tools/gh200/sync.sh; this copy differs in the host, the
# directory and the root it refuses to leave.
#
# rsync --delete removes what exists only on Kathleen in that directory: job output goes
# to the root's logs/, never into the tree.
#
# ssh reads ~/.ssh, which the Claude Code sandbox denies: run it outside the sandbox.

set -eu

host=${1:-kathleen}
top=$(git rev-parse --show-toplevel)
branch=$(git -C "$top" rev-parse --abbrev-ref HEAD)
[ "$branch" != HEAD ] || branch=detached-$(git -C "$top" rev-parse --short HEAD)
# Branch names may hold slashes; one directory per branch.
dir_name=$(printf '%s' "$branch" | tr '/' '_')
dest=${2:-/scratch/scratch/ucahtbe/fmm/src/$dir_name}
case "$dest" in
/scratch/scratch/ucahtbe/*/src/?*) ;;
*) echo "sync.sh: $dest is not a src/<name> directory below /scratch/scratch/ucahtbe" >&2; exit 1 ;;
esac

printf '%s\n%s\n' "$(git -C "$top" describe --always --dirty)" "$branch" \
    >"$top/.source-revision"

rsync -a --delete --compress \
    --exclude=/target/ --exclude=/.git --exclude=/.claude/ \
    --rsync-path="mkdir -p '$dest' && rsync" \
    "$top/" "$host:$dest/"
echo "synced $(head -n 1 "$top/.source-revision") ($branch) to $host:$dest"
