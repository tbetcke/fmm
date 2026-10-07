#!/usr/bin/env bash
# Installs Ubuntu packages for a CI job without the mirror when actions/cache has
# restored their .deb files (Phase 4S T10).
#
# Usage: apt-install.sh <archive directory> <package>...
#
# Cache hit (the directory holds .deb files): installs exactly those files, offline
# (`--no-download`) and without `apt-get update`, then checks that every named package
# is installed. The files are the ones a miss downloaded for the same runner image and
# package list (the cache key), so the versions are the same as on that miss. If the
# offline install fails, the script falls through to the miss path.
#
# Cache miss: `apt-get update` and `apt-get install` from the mirror, as before, and
# copies the downloaded .deb files into the directory for actions/cache to save.
#
# Both paths retry a failed download and give up on a stalled connection after 30 s;
# the workflow bounds the whole step with `timeout-minutes`. Both print the installed
# versions of the named packages.
set -euo pipefail
shopt -s nullglob

archives=$1
shift
apt=(sudo apt-get -y --no-install-recommends
    -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

debs=("$archives"/*.deb)
if ((${#debs[@]})); then
    echo "Cache hit: installing ${#debs[@]} cached packages offline"
    if "${apt[@]}" --no-download install "${debs[@]}" && dpkg -s "$@" > /dev/null; then
        dpkg-query -W "$@"
        exit 0
    fi
    echo "::warning::The cached packages did not install offline; downloading them"
fi

echo "Cache miss: downloading from the mirror"
sudo apt-get clean
"${apt[@]}" update
"${apt[@]}" install "$@"
mkdir -p "$archives"
downloaded=(/var/cache/apt/archives/*.deb)
if ((${#downloaded[@]})); then
    cp "${downloaded[@]}" "$archives"/
fi
echo "Kept ${#downloaded[@]} packages for the cache"
dpkg-query -W "$@"
