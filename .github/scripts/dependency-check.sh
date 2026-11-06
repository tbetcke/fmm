#!/usr/bin/env bash
# The weekly dependency check: `cargo upgrades` over the workspace, with the exact pins
# exempt.
#
# Usage: dependency-check.sh (from the workspace root)
#
# Fails when a dependency has a newer release than its requirement allows, or when
# `cargo upgrades` cannot look a dependency up (it then prints an error and exits 7, the
# same code as for an upgrade). A dependency whose requirement is exact (`=x.y.z`, as
# cubecl and cubek are: root CLAUDE.md, "Working rules") is a decision of its own; its
# upgrades are listed as a notice and do not fail the check.
set -uo pipefail

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cargo metadata --no-deps --format-version 1 \
    | jq -r '.packages[].dependencies[] | select(.req | startswith("=")) | .name' \
    | sort -u > "$work/pinned"

cargo upgrades --plumbing > "$work/out" 2> "$work/err"
status=$?

if [ -s "$work/err" ]; then
    cat "$work/err" >&2
    echo "error: cargo upgrades could not check every dependency (exit $status)" >&2
    exit 1
fi
if [ "$status" -ne 0 ] && [ "$status" -ne 7 ]; then
    cat "$work/out"
    echo "error: cargo upgrades exited $status" >&2
    exit 1
fi

# Lines are `<manifest>\t<name>@<current>\t<latest>`; a line starting with `#` is the
# "all up to date" message.
grep -v '^#' "$work/out" | awk -F'\t' -v pinned="$work/pinned" \
    -v exempt="$work/exempt" -v upgrades="$work/upgrades" '
    BEGIN { while ((getline name < pinned) > 0) exact[name] = 1 }
    { split($2, dep, "@"); print > (dep[1] in exact ? exempt : upgrades) }
'

if [ -s "$work/exempt" ]; then
    echo "Exact pins with newer releases (exempt):"
    sed 's/^/  /' "$work/exempt"
fi
if [ -s "$work/upgrades" ]; then
    echo "Dependencies with newer releases:"
    sed 's/^/  /' "$work/upgrades"
    exit 1
fi
echo "All dependencies match up-to-date releases, exact pins aside."
