#!/bin/bash
# tools/kathleen/setup.sh: builds Kathleen's environment from nothing (Phase 5N T1). Bash.
#
#     tools/kathleen/setup.sh
#
# Run it on a Kathleen login node, from a copy of the repository (tools/kathleen/sync.sh
# puts one at /scratch/scratch/ucahtbe/fmm/src/<branch>). It only downloads and unpacks
# (about two minutes; no build), so it fits the login-node rule (README.md). In order,
# it:
#   1. creates the layout under FMM_KATHLEEN_ROOT (default /scratch/scratch/ucahtbe/fmm;
#      README.md, "Layout");
#   2. writes the marker that check-root.sh compares against (first run only);
#   3. checks that every pinned module of env.sh loads;
#   4. installs rustup and the toolchain pinned in env.sh, with rustfmt and clippy.
# Every step is skipped when already done, so a second run only checks.
#
# It refuses to run unless FMM_KATHLEEN_ROOT is below /scratch/scratch/ucahtbe, and it
# writes nowhere else: HOME, TMPDIR and every cache point there (env.sh). Cargo's crates
# and the tracel-llvm bundle are downloaded by the first build job, not here (compute
# nodes reach crates.io and GitHub; machine.md).

set -euo pipefail

die() {
    printf 'setup.sh: %s\n' "$*" >&2
    exit 1
}
step() { printf '\n=== %s: %s\n' "$(date '+%F %T')" "$*"; }

[ "$#" -eq 0 ] || die "usage: setup.sh (no arguments; FMM_KATHLEEN_ROOT moves the root)"
[ -z "${SLURM_JOB_ID:-}" ] || die "run it on a login node, not in a job"

here=$(cd "$(dirname "$0")" && pwd)
root=${FMM_KATHLEEN_ROOT:-/scratch/scratch/ucahtbe/fmm}
case "$root" in
/scratch/scratch/ucahtbe/*) ;;
*) die "FMM_KATHLEEN_ROOT=$root is not below /scratch/scratch/ucahtbe" ;;
esac

# The variables only: HOME, TMPDIR, CARGO_HOME, RUSTUP_HOME and the rest point under $root.
export FMM_KATHLEEN_ROOT=$root
# shellcheck source=tools/kathleen/env.sh
FMM_KATHLEEN_SETUP=1 . "$here/env.sh"

step "layout under $root"
mkdir -p "$root" "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" \
    "$XDG_STATE_HOME" "$TMPDIR" "$CARGO_HOME" "$RUSTUP_HOME" "$root/src" "$root/logs"

# check-root.sh lists what outside the root is newer than this file.
if [ ! -e "$root/.root-marker" ]; then
    date '+%F %T %z' >"$root/.root-marker"
    echo "wrote the marker $root/.root-marker"
fi

step "modules"
module purge
# shellcheck disable=SC2086 # the list is split on purpose
module load $FMM_KATHLEEN_MODULES
module -t list 2>&1
llvm-config --libdir
mpirun --version | head -n 1

step "Rust $FMM_KATHLEEN_RUST"
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
    triple=$(uname -m)-unknown-linux-gnu
    url=https://static.rust-lang.org/rustup/dist/$triple/rustup-init
    curl --proto '=https' --tlsv1.2 -sSf -o "$TMPDIR/rustup-init" "$url"
    expected=$(curl --proto '=https' --tlsv1.2 -sSf "$url.sha256" | cut -d' ' -f1)
    actual=$(sha256sum "$TMPDIR/rustup-init" | cut -d' ' -f1)
    [ "$expected" = "$actual" ] || die "rustup-init checksum mismatch"
    chmod +x "$TMPDIR/rustup-init"
    "$TMPDIR/rustup-init" -y --quiet --no-modify-path --profile minimal \
        --default-toolchain none
    rm -f "$TMPDIR/rustup-init"
fi
"$CARGO_HOME/bin/rustup" toolchain install "$FMM_KATHLEEN_RUST" --profile minimal \
    --component rustfmt --component clippy
"$CARGO_HOME/bin/rustc" --version
"$CARGO_HOME/bin/cargo" --version

step "done"
du -sh "$CARGO_HOME" "$RUSTUP_HOME" 2>/dev/null || true
echo "next: . tools/kathleen/env.sh"
