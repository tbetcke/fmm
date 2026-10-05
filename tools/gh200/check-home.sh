#!/bin/sh
# tools/gh200/check-home.sh: shows what was written outside /data/ucahtbe (Phase 4S T1).
# POSIX sh.
#
#     tools/gh200/check-home.sh
#
# Run it on locust, inside the environment or not. It lists every file and directory
# under the real home directory (from the password database, not $HOME, which env.sh
# moves) that is newer than the marker setup.sh wrote, $FMM_GH200_ROOT/.home-marker.
# It also lists what this user owns, newer than the marker, in /tmp, /var/tmp and
# /dev/shm, where tools write when TMPDIR is not honoured. Expected: nothing, apart
# from what an interactive login itself writes (the shell history, .ssh).
#
# Exits 0 when nothing is listed, 1 otherwise, 2 without a marker.

set -u

root=${FMM_GH200_ROOT:-/data/ucahtbe}
marker=$root/.home-marker
user=$(id -un)
home=$(getent passwd "$user" | cut -d: -f6)

[ -e "$marker" ] || { echo "check-home.sh: no marker $marker; run setup.sh" >&2; exit 2; }
echo "check-home.sh: newer than $marker ($(cat "$marker")):"

found=0
list() {
    # $1: label; the rest: find arguments
    label=$1
    shift
    out=$(find "$@" -newer "$marker" 2>/dev/null)
    if [ -n "$out" ]; then
        found=1
        printf '%s\n' "$out" | sed "s|^|  $label: |"
    fi
}

list home "$home" -mindepth 1
for d in /tmp /var/tmp /dev/shm; do
    list "$d" "$d" -mindepth 1 -maxdepth 2 -user "$user"
done

[ "$found" = 0 ] && echo "  nothing"
exit "$found"
