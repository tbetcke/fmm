#!/bin/sh
# tools/fftw/probe.sh: the FFTW of this machine, as RLST sees it (Helmholtz CHR.4). POSIX sh.
#
#     tools/fftw/probe.sh
#
# Run it from anywhere in the checkout, inside the machine's environment (on locust it
# sources tools/gh200/env.sh if that is not active; on Kathleen run it in a job,
# `tools/kathleen/submit.sh run.sbatch 'tools/fftw/probe.sh'`). It prints, in order:
#
#   1. the host, OS, CPU and the CPU's SIMD flags;
#   2. `pkg-config --modversion fftw3 fftw3f` and each library's `libdir`;
#   3. the version strings of both precisions (`fftw_version`, `fftwf_version`, with the
#      SIMD extensions FFTW was built with as suffixes), from a small C program built with
#      pkg-config's flags, and the FFTW libraries it loads;
#   4. RLST's own tests with FFTW: the registry's copy of the workspace's RLST (the version
#      of `[workspace.dependencies]`) is copied to target/fftw-probe/ and tested there with
#      `cargo test --locked --features fftw_system`, `fftw_determinism` included;
#   5. `rlst::fftw::version()` from an example added to that copy, and the FFTW libraries
#      the example loads (`ldd`, `otool -L`): a build against one FFTW that loads another
#      at run time (locust's RPM 3.3.8 behind spack's) shows here.
#
# Nothing enters the workspace or its lock file: the copy is a package of its own, built
# with RLST's packaged Cargo.lock. The exit status is non-zero if a step failed; the steps
# after a failed one still run.

set -u

root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)

# On locust (the environment of tools/gh200/ exists), activate it unless it is active.
if [ -z "${FMM_GH200_ENV:-}" ] && [ -d /data/ucahtbe/envs/fmm-gh200 ]; then
    # env.sh and spack's activation are not written for `set -u`.
    set +u
    . "$root/tools/gh200/env.sh"
    set -u
fi

probe=$root/target/fftw-probe
mkdir -p "$probe"
failed=

section() {
    printf '\n## %s\n\n' "$1"
}

# `run <name> <command...>` runs a command, prints its output, keeps it in
# target/fftw-probe/<name>.log for the summary, and records a failure without stopping.
run() {
    name=$1
    shift
    printf '$ %s\n' "$*"
    if "$@" >"$probe/$name.log" 2>&1; then
        cat "$probe/$name.log"
    else
        status=$?
        cat "$probe/$name.log"
        echo "(failed: exit $status)"
        failed="$failed $name"
    fi
}

# The FFTW libraries a binary loads.
fftw_libraries() {
    case $(uname -s) in
    Darwin) otool -L "$1" | grep -i fftw ;;
    *) ldd "$1" | grep -i fftw ;;
    esac
}

section "Machine"
echo "- host: $(hostname)"
echo "- OS: $(uname -srm)"
echo "- date: $(date -u '+%Y-%m-%d %H:%M UTC')"
case $(uname -s) in
Darwin)
    # sysctl is denied inside Claude Code's sandbox; the lines then say "unknown".
    echo "- CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    case $(sysctl -n hw.optional.neon 2>/dev/null) in
    1) echo "- SIMD: neon" ;;
    0) echo "- SIMD: none" ;;
    *) echo "- SIMD: unknown" ;;
    esac
    ;;
*)
    echo "- CPU: $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
    echo "- SIMD: $(grep -m1 -E '^(flags|Features)' /proc/cpuinfo | tr ' ' '\n' |
        grep -x -E 'sse2|avx|avx2|fma|avx512f|asimd|sve|sve2' | sort -u | paste -sd' ' -)"
    ;;
esac

section "pkg-config"
run pkg-config pkg-config --modversion fftw3 fftw3f
for library in fftw3 fftw3f; do
    echo "- $library libdir: $(pkg-config --variable=libdir "$library" 2>/dev/null)"
done

section "The libraries' version strings (C)"
cat >"$probe/version.c" <<'EOF'
/* Written by tools/fftw/probe.sh. */
#include <stdio.h>
#include <fftw3.h>

int main(void)
{
    printf("fftw_version:  %s\nfftwf_version: %s\n", fftw_version, fftwf_version);
    return 0;
}
EOF
# shellcheck disable=SC2046 # pkg-config's flags are separate words
if ${CC:-cc} $(pkg-config --cflags fftw3 fftw3f) "$probe/version.c" -o "$probe/version" \
    $(pkg-config --libs fftw3 fftw3f); then
    run c-version "$probe/version"
    fftw_libraries "$probe/version"
else
    echo "(the C program did not build)"
    failed="$failed cc"
fi

section "RLST's tests with FFTW"
rlst_version=$(sed -n 's/^rlst = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -1)
registry=${CARGO_HOME:-$HOME/.cargo}/registry/src
# The registry's unpacked copy of this RLST, if any.
registry_copy() {
    for directory in "$registry"/*/"rlst-$rlst_version"; do
        [ -d "$directory" ] && { echo "$directory"; return; }
    done
}
source_dir=$(registry_copy)
if [ -z "$source_dir" ]; then
    # The workspace's lock file names this RLST; fetching it fills the registry.
    (cd "$root" && cargo fetch --locked >/dev/null)
    source_dir=$(registry_copy)
fi
echo "- RLST $rlst_version from $source_dir"
copy=$probe/rlst-$rlst_version
rm -rf "$copy"
cp -R "$source_dir" "$copy"
chmod -R u+w "$copy"
# A package of its own: without `[workspace]`, cargo would take it for an undeclared
# member of nd-fmm's workspace.
cat >>"$copy/Cargo.toml" <<'EOF'

[[example]]
name = "fftw_version"
path = "examples/fftw_version.rs"
required-features = ["fftw"]

[workspace]
EOF
cat >"$copy/examples/fftw_version.rs" <<'EOF'
//! Written by nd-fmm's tools/fftw/probe.sh: FFTW's version string, as RLST reads it from the
//! library it loaded.

extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

fn main() {
    println!("rlst::fftw::version(): {}", rlst::fftw::version());
}
EOF
export CARGO_TARGET_DIR="$probe/target"
run rlst-tests cargo test --manifest-path "$copy/Cargo.toml" --locked --features fftw_system

section "rlst::fftw::version()"
run rlst-version cargo run --manifest-path "$copy/Cargo.toml" --locked --quiet --features fftw_system \
    --example fftw_version
fftw_libraries "$CARGO_TARGET_DIR/debug/examples/fftw_version"

section "Summary"
echo "- pkg-config fftw3 fftw3f: $(paste -sd' ' "$probe/pkg-config.log")"
sed 's/^/- /' "$probe/c-version.log" 2>/dev/null
echo "- $(cat "$probe/rlst-version.log" 2>/dev/null)"
awk '/^test result:/ { executables++; passed += $4; failed += $6 }
     /allocations per execution .* ok$/ { allocations++ }
     /^fftw_determinism:/ { determinism = $0 }
     END {
         printf "- RLST'"'"'s tests: %d passed, %d failed in %d harness runs; ", passed, failed, executables
         printf "fftw_check_allocations %d checks ok; %s\n", allocations, determinism
     }' "$probe/rlst-tests.log"
if [ -n "$failed" ]; then
    echo "FAILED:$failed"
    exit 1
fi
echo "all steps passed"
