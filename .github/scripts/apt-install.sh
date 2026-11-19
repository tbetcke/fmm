#!/usr/bin/env bash
# Installs Ubuntu packages for a CI job without the mirror when actions/cache has
# restored their .deb files (Phase 4S T10).
#
# Usage: apt-install.sh <archive directory> <package>...
#
# Cache hit (the directory holds .deb files): installs exactly those files, offline
# and without `apt-get update`, then checks that every named package is installed. The files are the ones a miss downloaded for the same runner image and
# package list (the cache key), so the versions are the same as on that miss. If the
# offline install fails, the script falls through to the miss path.
#
# Cache miss: `apt-get update`, then the packages downloaded (`--download-only`) in up to
# three attempts of at most ATTEMPT_SECONDS each, then `apt-get install` from the
# downloaded files; it copies them into the directory for actions/cache to save.
#
# Both paths retry a failed download and give up on a stalled connection after 30 s. A
# connection that trickles never stalls: on 2026-11-19 two jobs' downloads ran at about
# 85 kB/s from one backend of the runner's mirror and hit the step's 5-minute limit, while
# another job of the same run fetched the same 86 MB in 18 s. So a download attempt that
# takes longer than ATTEMPT_SECONDS (120 s; 18–50 s is the normal miss) is stopped and
# started again on new connections; apt resumes the partial files. Only the download is
# stopped, never dpkg. The workflow bounds the whole step with `timeout-minutes` (10).
# Both paths print the installed versions of the named packages.
set -euo pipefail
shopt -s nullglob

archives=$1
shift
apt=(sudo apt-get -y --no-install-recommends
    -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
attempts=3
attempt_seconds=${ATTEMPT_SECONDS:-120}

debs=("$archives"/*.deb)
if ((${#debs[@]})); then
    echo "Cache hit: installing ${#debs[@]} cached packages offline"
    # An empty package-list directory: apt knows no repository package, so it can only
    # install the given files and what is installed already, and never downloads.
    # (`--no-download` refuses the given files too.)
    lists=$(mktemp -d)
    if "${apt[@]}" -o Dir::State::Lists="$lists" install "${debs[@]}" &&
        dpkg -s "$@" > /dev/null; then
        dpkg-query -W "$@"
        exit 0
    fi
    echo "::warning::The cached packages did not install offline; downloading them"
fi

echo "Cache miss: downloading from the mirror"
sudo apt-get clean
"${apt[@]}" update
downloaded_all=
for ((attempt = 1; attempt <= attempts; attempt++)); do
    if timeout --kill-after=10 "$attempt_seconds" "${apt[@]}" install --download-only "$@"; then
        downloaded_all=yes
        break
    fi
    echo "::warning::Download attempt $attempt of $attempts failed or took over" \
        "$attempt_seconds s; starting again on new connections"
done
if [ -z "$downloaded_all" ]; then
    echo "The packages did not download in $attempts attempts" >&2
    exit 1
fi
"${apt[@]}" install "$@"
mkdir -p "$archives"
downloaded=(/var/cache/apt/archives/*.deb)
if ((${#downloaded[@]})); then
    cp "${downloaded[@]}" "$archives"/
fi
echo "Kept ${#downloaded[@]} packages for the cache"
dpkg-query -W "$@"
