#!/usr/bin/env bash
# tools/fmm3d/build.sh: fetches FMM3D at a pinned release and builds it with OpenMP, and
# the driver of the comparison against it, outside the repository (Phase 6 T9, C6.8;
# docs/design/optimisation.md §8.1, decision 5). README.md beside it has the commands.
#
#     tools/fmm3d/build.sh [fetch] [build] [test]
#
# Steps (default: fetch build):
#   fetch   downloads the release tarball into $FMM3D_ROOT/src and checks its SHA-256
#           (light: run it on Kathleen's login node, whose compute nodes may not reach
#           GitHub)
#   build   for every variant of VARIANTS, unpacks a fresh copy into
#           $FMM3D_ROOT/build/<variant>, runs FMM3D's `make lib` with OpenMP, copies its
#           static library to $FMM3D_ROOT/<variant>/lib and links the driver
#           (driver.f90) to $FMM3D_ROOT/<variant>/bin/fmm3d_driver; records the compilers,
#           the flags and the tarball in $FMM3D_ROOT/<variant>/build-info.txt
#   test    FMM3D's own Laplace tests (`make test` builds and runs them all; a couple of
#           minutes) in each variant's build directory
#
# Variants (VARIANTS, default "default fastker"):
#   default   FMM3D as `make lib` builds it, with OpenMP: its "easy-to-install version"
#   fastker   `make lib FAST_KER=ON`: its "high-performance mode", the vectorised direct
#             kernels (C++, gcc 9 or later); FMM3D's install guide calls it that
#
# The compilers are the machine's own, with FMM3D's flags for it:
#   macOS (the M3 Max)  Homebrew's GCC (gfortran, gcc-<v>, g++-<v>; GCC 16 in use) with
#                       FMM3D's make.inc.macos_arm64.gnu, its compilers replaced by those
#   Kathleen            the gcc module of tools/kathleen/env.sh (source it first; the build
#                       is a job, tools/kathleen/README.md, "FMM3D"), FMM3D's Linux defaults
#                       (-O3 -march=native -funroll-loops)
#   locust              the spack environment of tools/gh200/env.sh, if it has gfortran;
#                       FMM3D's Linux defaults
# Override with FC, CC and CXX.
#
# FMM3D_ROOT, where everything goes (never inside a checkout): on Kathleen
# $FMM_KATHLEEN_ROOT/fmm3d, on locust /data/ucahtbe/fmm3d, elsewhere it must be set.
# Nothing of FMM3D is committed: its sources, builds and outputs stay there.

set -eu

FMM3D_VERSION=2.1.0
FMM3D_TAG=v$FMM3D_VERSION
# The commit of the tag (GitHub, 2026-01-19, the latest release on 2026-10-18) and the
# SHA-256 of GitHub's tarball of it.
FMM3D_COMMIT=70455e020be1ff64445757cf71ae361f741ea282
FMM3D_SHA256=16f18c47e8c559adfb71b83a974faaeb6ee6de0bb153c183dc8dfe69f7449778
FMM3D_URL=https://github.com/flatironinstitute/FMM3D/archive/refs/tags/$FMM3D_TAG.tar.gz

here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH='' cd -- "$here/../.." && pwd)

if [ -z "${FMM3D_ROOT:-}" ]; then
    if [ -n "${FMM_KATHLEEN_ROOT:-}" ]; then
        FMM3D_ROOT=$FMM_KATHLEEN_ROOT/fmm3d
    elif [ -d /data/ucahtbe ]; then
        FMM3D_ROOT=/data/ucahtbe/fmm3d
    else
        echo "build.sh: set FMM3D_ROOT to a directory outside the repository" >&2
        exit 2
    fi
fi
case "$FMM3D_ROOT/" in
"$repo"/*)
    echo "build.sh: FMM3D_ROOT $FMM3D_ROOT is inside the repository; FMM3D stays outside" >&2
    exit 2
    ;;
esac
VARIANTS=${VARIANTS:-"default fastker"}
tarball=$FMM3D_ROOT/src/FMM3D-$FMM3D_VERSION.tar.gz

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

fetch() {
    mkdir -p "$FMM3D_ROOT/src"
    if [ ! -f "$tarball" ]; then
        curl -fsSL -o "$tarball.part" "$FMM3D_URL"
        mv "$tarball.part" "$tarball"
    fi
    sum=$(sha256 "$tarball")
    if [ "$sum" != "$FMM3D_SHA256" ]; then
        echo "build.sh: $tarball has SHA-256 $sum, expected $FMM3D_SHA256" >&2
        exit 1
    fi
    echo "fetched FMM3D $FMM3D_TAG ($FMM3D_COMMIT): $tarball, SHA-256 $sum"
}

# The compilers and FMM3D's make.inc for this machine.
compilers() {
    case "$(uname -s)" in
    Darwin)
        brew=${HOMEBREW_PREFIX:-/opt/homebrew}
        FC=${FC:-$brew/bin/gfortran}
        gcc_version=$("$FC" -dumpversion | cut -d . -f 1)
        CC=${CC:-$brew/bin/gcc-$gcc_version}
        CXX=${CXX:-$brew/bin/g++-$gcc_version}
        make_inc=make.inc.macos_arm64.gnu
        ;;
    *)
        FC=${FC:-gfortran}
        CC=${CC:-gcc}
        CXX=${CXX:-g++}
        make_inc=
        ;;
    esac
    for tool in "$FC" "$CC" "$CXX"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "build.sh: no $tool on this machine (README.md, \"Building\")" >&2
            exit 1
        fi
    done
}

build_variant() {
    variant=$1
    case "$variant" in
    default) fast=OFF ;;
    fastker) fast=ON ;;
    *)
        echo "build.sh: unknown variant $variant (default or fastker)" >&2
        exit 2
        ;;
    esac
    dir=$FMM3D_ROOT/build/$variant
    prefix=$FMM3D_ROOT/$variant
    rm -rf "$dir"
    mkdir -p "$dir" "$prefix/lib" "$prefix/bin"
    tar -xzf "$tarball" -C "$dir"
    src=$dir/FMM3D-$FMM3D_VERSION
    if [ -n "$make_inc" ]; then
        # FMM3D's own make.inc for the machine, with this machine's compilers.
        {
            cat "$src/$make_inc"
            echo
            echo "# tools/fmm3d/build.sh: this machine's compilers"
            echo "CC=$CC"
            echo "CXX=$CXX"
            echo "FC=$FC"
        } >"$src/make.inc"
    fi
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
    echo "building FMM3D $FMM3D_TAG, variant $variant (FAST_KER=$fast), in $src"
    (cd "$src" && make -j "$jobs" lib FAST_KER="$fast" CC="$CC" CXX="$CXX" FC="$FC") \
        >"$prefix/build.log" 2>&1 || {
        tail -n 40 "$prefix/build.log" >&2
        echo "build.sh: FMM3D's build failed; the log is $prefix/build.log" >&2
        exit 1
    }
    cp "$src/lib-static/libfmm3d.a" "$prefix/lib/"
    libs="-lm"
    if [ "$fast" = ON ]; then libs="-lstdc++ $libs"; fi
    # The driver with FMM3D's Fortran flags but -O2 (it only reads, copies and times).
    # shellcheck disable=SC2086 # the list is split on purpose
    "$FC" -O2 -fopenmp -o "$prefix/bin/fmm3d_driver" "$here/driver.f90" \
        "$prefix/lib/libfmm3d.a" $libs
    {
        echo "FMM3D $FMM3D_TAG, commit $FMM3D_COMMIT, tarball SHA-256 $FMM3D_SHA256"
        echo "variant $variant: make lib FAST_KER=$fast (OpenMP on: FMM3D's default)"
        echo "built $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname) ($(uname -sm))"
        echo "FC  $FC: $("$FC" --version | head -n 1)"
        echo "CC  $CC: $("$CC" --version | head -n 1)"
        echo "CXX $CXX: $("$CXX" --version | head -n 1)"
        if [ -n "$make_inc" ]; then
            echo "make.inc: FMM3D's $make_inc with these compilers"
        else
            echo "make.inc: none, FMM3D's Linux defaults"
        fi
        echo "the first Fortran and C++ compile lines of the build:"
        grep -m 1 -e "$FC .*-c .*\.f" "$prefix/build.log" || true
        grep -m 1 -e "$CXX .*-c" "$prefix/build.log" || true
        echo "driver: $FC -O2 -fopenmp driver.f90 libfmm3d.a $libs"
    } >"$prefix/build-info.txt"
    cat "$prefix/build-info.txt"
}

test_variant() {
    src=$FMM3D_ROOT/build/$1/FMM3D-$FMM3D_VERSION
    fast=OFF
    if [ "$1" = fastker ]; then fast=ON; fi
    (cd "$src" && make test FAST_KER="$fast" CC="$CC" CXX="$CXX" FC="$FC") \
        >"$FMM3D_ROOT/$1/test.log" 2>&1 || true
    echo "FMM3D's tests, variant $1 ($FMM3D_ROOT/$1/test.log):"
    grep -e 'Successfully completed' -e 'tests in' "$FMM3D_ROOT/$1/test.log" || {
        tail -n 20 "$FMM3D_ROOT/$1/test.log"
        exit 1
    }
}

steps=${*:-fetch build}
for step in $steps; do
    case "$step" in
    fetch) fetch ;;
    build)
        [ -f "$tarball" ] || fetch
        compilers
        for variant in $VARIANTS; do build_variant "$variant"; done
        ;;
    test)
        compilers
        for variant in $VARIANTS; do test_variant "$variant"; done
        ;;
    *)
        echo "build.sh: unknown step $step (fetch, build or test)" >&2
        exit 2
        ;;
    esac
done
