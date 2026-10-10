#!/bin/sh
# tools/kathleen/check-root.sh: shows what was written outside the root on Kathleen
# (Phase 5N T1). POSIX sh.
#
#     tools/kathleen/check-root.sh
#
# Run it on a Kathleen login node, inside the environment or not. It lists every file
# and directory under the real home directory (from the password database, not $HOME,
# which env.sh moves) and under the real Scratch (/scratch/scratch/$USER) outside the
# root that is newer than the marker setup.sh wrote, $FMM_KATHLEEN_ROOT/.root-marker,
# and what this user owns, newer than it, in the login node's /tmp, /var/tmp and
# /dev/shm. Compute nodes' /dev/shm and /tmp are RAM and per job; the job scripts report
# and remove their own directory there (jobs/lib.sh). Expected: nothing, apart from what
# an interactive login itself writes (the shell history, .ssh), and the queue probe's
# ~/Scratch/fmm-probe/ (step 0, before the root existed).
#
# Exits 0 when nothing is listed, 1 otherwise, 2 without a marker.

set -u

root=${FMM_KATHLEEN_ROOT:-/scratch/scratch/ucahtbe/fmm}
marker=$root/.root-marker
user=$(id -un)
home=$(getent passwd "$user" | cut -d: -f6)
scratch=/scratch/scratch/$user

[ -e "$marker" ] || { echo "check-root.sh: no marker $marker; run setup.sh" >&2; exit 2; }
echo "check-root.sh: newer than $marker ($(cat "$marker")), outside $root:"

found=0
# list <label> <directory> [<path to skip>...]: what under the directory (or, for the
# temporary directories, owned by this user two levels deep) is newer than the marker.
list() {
    label=$1 dir=$2
    shift 2
    if [ "$#" -gt 0 ]; then
        skip=
        for p in "$@"; do
            skip="$skip${skip:+ -o} -path $p"
        done
        # shellcheck disable=SC2086 # the skip list is split on purpose
        out=$(find "$dir" -mindepth 1 \( $skip \) -prune -o -newer "$marker" -print 2>/dev/null)
    else
        out=$(find "$dir" -mindepth 1 -maxdepth 2 -user "$user" -newer "$marker" -print 2>/dev/null)
    fi
    if [ -n "$out" ]; then
        found=1
        printf '%s\n' "$out" | sed "s|^|  $label: |"
    fi
}

list home "$home" "$home/Scratch"
list scratch "$scratch" "$root" "$scratch/fmm-probe"
for d in /tmp /var/tmp /dev/shm; do
    list "$d" "$d"
done

[ "$found" = 0 ] && echo "  nothing"
exit "$found"
