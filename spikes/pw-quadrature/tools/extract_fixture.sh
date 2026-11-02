#!/bin/sh
# Writes fixtures/: a few domains of FMM3D's Helmholtz plane-wave tables (FMM3D v2.1.0,
# src/Helmholtz/hwts_3d_iprec{2,3}.txt, hnumfour.f, hnumphys.f), in FMM3D's own formats,
# so that the parser and the evaluator are tested without FMM3D's sources.
#
# The files are NIST's: each extract keeps their notice (the first comment block) and
# says that it was cut from them. FMM3D as a whole is Apache-2.0. Run from the spike
# directory, with the directory tools/fetch.sh unpacked:
#
#   tools/extract_fixture.sh ../../target/fmm3d-src/FMM3D-2.1.0/src/Helmholtz
set -eu
src=${1:?usage: extract_fixture.sh path/to/FMM3D/src/Helmholtz}
# Domains: 1 (κ ≈ 0, type 2), 11 (Re κ in [3π, 4π], type 2), 139 (Im κ in [1, π/2],
# type 2), 172 (κ near 10 + 3i, type 1), 236 (κ near 0.5 + 10i, type 1).
domains="1 11 139 172 236"
out=fixtures
mkdir -p "$out"
notice() {
    awk '/^c/ { print; next } { exit }' "$1"
    echo "c"
    echo "c Cut by spikes/pw-quadrature/tools/extract_fixture.sh from FMM3D v2.1.0,"
    echo "c src/Helmholtz/$(basename "$1"): domains $domains only, otherwise unchanged."
    echo "c"
}
for p in 2 3; do
    f="$src/hwts_3d_iprec$p.txt"
    {
        notice "$f"
        awk -v keep="$domains" '
            BEGIN { n = split(keep, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 }
            /idomain \.eq\./ { d = $4 + 0; on = (d in want) }
            on { print }
            on && /^ *endif/ { on = 0 }
        ' "$f"
    } >"$out/hwts_3d_iprec$p.txt"
done
for name in hnumfour hnumphys; do
    f="$src/$name.f"
    {
        notice "$f"
        awk -v keep="$domains" '
            BEGIN { n = split(keep, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 }
            /^ *subroutine/ { sub_ = ($0 ~ /_iprec[23]\(/); if (sub_) print; next }
            sub_ && /idomain\.eq\./ { s = $0; gsub(/[^0-9]/, " ", s); split(s, a, " "); on = (a[1] in want) }
            sub_ && on { print }
            sub_ && on && /^ *endif/ { on = 0 }
        ' "$f"
    } >"$out/$name.f"
done
wc -l "$out"/*
