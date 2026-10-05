#!/bin/sh
# tools/gh200/sync.sh: copies this working tree to locust (Phase 4S T1). POSIX sh.
#
#     tools/gh200/sync.sh [host [directory]]
#
# Run it on the M3 Max, anywhere inside a checkout or worktree. It copies the working
# tree, uncommitted changes included, to <host>:<directory>, by default
# locust:/data/ucahtbe/fmm/<branch>, so that work is built on locust without a commit
# (root CLAUDE.md, "Working agreement"). It leaves out target/ and .git (and the
# session-local .claude/), deletes remote files that are gone locally, and keeps the
# remote target/ so builds stay incremental. It first writes .source-revision
# (`git describe --always --dirty` and the branch; ignored by git), which travels with
# the tree.
#
# ssh reads ~/.ssh, which the Claude Code sandbox denies: run it outside the sandbox.

set -eu

host=${1:-locust}
top=$(git rev-parse --show-toplevel)
branch=$(git -C "$top" rev-parse --abbrev-ref HEAD)
[ "$branch" != HEAD ] || branch=detached-$(git -C "$top" rev-parse --short HEAD)
case "$branch" in
repo) echo "sync.sh: branch 'repo' would overwrite the clone /data/ucahtbe/fmm/repo" >&2; exit 1 ;;
esac
dest=${2:-/data/ucahtbe/fmm/$branch}
case "$dest" in
/data/ucahtbe/*) ;;
*) echo "sync.sh: $dest is not under /data/ucahtbe" >&2; exit 1 ;;
esac

printf '%s\n%s\n' "$(git -C "$top" describe --always --dirty)" "$branch" \
    >"$top/.source-revision"

rsync -a --delete --compress \
    --exclude=/target/ --exclude=/.git --exclude=/.claude/ \
    --rsync-path="mkdir -p '$dest' && rsync" \
    "$top/" "$host:$dest/"
echo "synced $(head -n 1 "$top/.source-revision") ($branch) to $host:$dest"
