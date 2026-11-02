#!/bin/sh
# Unpacks FMM3D's Helmholtz sources for S4's sweeps and ignored tests from the release
# tarball that tools/fmm3d/build.sh fetches and pins (v2.1.0, SHA-256 checked here too),
# into $FMM3D_ROOT/build/s4, outside the repository like the rest of FMM3D, and prints
# the directory of hwts3e.f and its tables:
#
#   export FMM3D_HELMHOLTZ=$(spikes/pw-quadrature/tools/fetch.sh)
#
# Run tools/fmm3d/build.sh fetch first if the tarball is missing.
set -eu
FMM3D_VERSION=2.1.0
FMM3D_SHA256=16f18c47e8c559adfb71b83a974faaeb6ee6de0bb153c183dc8dfe69f7449778
: "${FMM3D_ROOT:?set FMM3D_ROOT as for tools/fmm3d/build.sh}"
tarball=$FMM3D_ROOT/src/FMM3D-$FMM3D_VERSION.tar.gz
[ -f "$tarball" ] || { echo "fetch.sh: no $tarball; run tools/fmm3d/build.sh fetch" >&2; exit 2; }
if command -v sha256sum >/dev/null 2>&1; then
    sum=$(sha256sum "$tarball" | cut -d ' ' -f 1)
else
    sum=$(shasum -a 256 "$tarball" | cut -d ' ' -f 1)
fi
[ "$sum" = "$FMM3D_SHA256" ] || { echo "fetch.sh: $tarball has SHA-256 $sum" >&2; exit 2; }
dest=$FMM3D_ROOT/build/s4
dir=$dest/FMM3D-$FMM3D_VERSION/src/Helmholtz
if [ ! -f "$dir/hwts3e.f" ]; then
    mkdir -p "$dest"
    tar xzf "$tarball" -C "$dest"
fi
echo "$dir"
