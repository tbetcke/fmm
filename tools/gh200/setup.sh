#!/usr/bin/env bash
# tools/gh200/setup.sh: builds locust's environment from nothing (Phase 4S T1). Bash.
#
#     tools/gh200/setup.sh                 # install from spack.yaml and spack.lock
#     tools/gh200/setup.sh --update-lock   # concretise afresh and write a new lock
#
# Run it on locust from a copy of the repository (tools/gh200/sync.sh puts one at
# /data/ucahtbe/fmm/<branch>). In order, it:
#   1. creates the layout under FMM_GH200_ROOT (default /data/ucahtbe; README.md);
#   2. writes the home marker that check-home.sh compares against (first run only);
#   3. clones spack at the pinned tag and writes its user configuration;
#   4. creates the environment envs/fmm-gh200 from spack.yaml and spack.lock and
#      installs it (with --update-lock: concretises afresh, writes the new lock to
#      envs/fmm-gh200/spack.lock, to be copied into the repository by hand);
#   5. installs rustup and the toolchain pinned in env.sh, with rustfmt and clippy;
#   6. clones the repository (public, https, read-only) into fmm/repo.
# Every step is skipped when already done, so a second run only checks.
#
# It refuses to run unless FMM_GH200_ROOT is /data/ucahtbe or below it, and it writes
# nowhere else: HOME, TMPDIR and every cache point there (env.sh). The exception is
# outside its control: NVIDIA's CUDA installer writes /tmp/cuda-installer.log,
# /var/tmp/dnf-$USER-* and /tmp/hsperfdata_$USER while it runs; spack's cuda package
# deletes the first, this script the others (check-home.sh looks for all three).
#
# FMM_GH200_SOURCE_CACHE (default $FMM_GH200_ROOT/spack-sources) lets a second prefix
# reuse the first one's downloaded sources.

set -euo pipefail

# --- Pins -----------------------------------------------------------------------
# Spack: the newest 1.x release on 2026-10-05.
SPACK_TAG=v1.2.2
SPACK_COMMIT=3e19345b6e12f5ff1b874f4059622fc6a1fd804a
# spack/spack-packages at the head of releases/v2026.06 (tag v2026.06.0), the branch
# spack v1.2.2 uses by default. Also in spack.yaml (`repos:`).
SPACK_PACKAGES_COMMIT=d4f7c711a6a42f1c4d551c8fd10fce9a11340a81
REPOSITORY=https://github.com/tbetcke/fmm.git

die() {
    printf 'setup.sh: %s\n' "$*" >&2
    exit 1
}
step() { printf '\n=== %s: %s\n' "$(date '+%F %T')" "$*"; }

update_lock=0
case "${1:-}" in
"") ;;
--update-lock) update_lock=1 ;;
*) die "usage: setup.sh [--update-lock]" ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
[ -d /data/ucahtbe ] || die "/data/ucahtbe does not exist"
root=${FMM_GH200_ROOT:-/data/ucahtbe}
case "$root" in
/data/ucahtbe | /data/ucahtbe/*) ;;
*) die "FMM_GH200_ROOT=$root is not under /data/ucahtbe" ;;
esac
grep -q "commit: $SPACK_PACKAGES_COMMIT" "$here/spack.yaml" ||
    die "spack.yaml does not pin spack-packages at $SPACK_PACKAGES_COMMIT"

# The variables only: HOME, TMPDIR, CARGO_HOME, SPACK_* and the rest point under $root.
export FMM_GH200_ROOT=$root
FMM_GH200_SETUP=1 . "$here/env.sh"
source_cache=${FMM_GH200_SOURCE_CACHE:-$root/spack-sources}
spack=$SPACK_ROOT/bin/spack

step "layout under $root"
mkdir -p "$root" "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" \
    "$XDG_STATE_HOME" "$TMPDIR" "$CUDA_CACHE_PATH" "$CARGO_HOME" "$RUSTUP_HOME" \
    "$SPACK_USER_CONFIG_PATH" "$SPACK_USER_CACHE_PATH" "$root/spack-install" \
    "$root/spack-stage" "$source_cache" "$root/envs" "$root/fmm"

# The start of this run, for the clean-up after the CUDA install.
run_stamp=$TMPDIR/setup-run-stamp
touch "$run_stamp"

# check-home.sh lists what in the real home is newer than this file.
if [ ! -e "$root/.home-marker" ]; then
    date '+%F %T %z' >"$root/.home-marker"
    echo "wrote the home marker $root/.home-marker"
fi

step "spack $SPACK_TAG"
if [ ! -d "$SPACK_ROOT/.git" ]; then
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$SPACK_TAG" \
        https://github.com/spack/spack.git "$SPACK_ROOT"
fi
[ "$(git -C "$SPACK_ROOT" rev-parse HEAD)" = "$SPACK_COMMIT" ] ||
    die "$SPACK_ROOT is not at $SPACK_TAG ($SPACK_COMMIT)"

# Spack's user scope: where it installs, stages and caches, and the package repository
# pin (repeated from spack.yaml, so that commands outside the environment and an
# environment created from spack.lock alone use the same packages).
cat >"$SPACK_USER_CONFIG_PATH/config.yaml" <<EOF
# Written by tools/gh200/setup.sh; edits are overwritten.
config:
  install_tree:
    root: $root/spack-install
  build_stage:
  - $root/spack-stage
  source_cache: $source_cache
  build_jobs: 36
EOF
cat >"$SPACK_USER_CONFIG_PATH/repos.yaml" <<EOF
# Written by tools/gh200/setup.sh; edits are overwritten.
repos:
  builtin:
    git: https://github.com/spack/spack-packages.git
    commit: $SPACK_PACKAGES_COMMIT
EOF
"$spack" --version

step "spack environment $FMM_GH200_ENV"
if [ ! -f "$FMM_GH200_ENV/spack.yaml" ]; then
    "$spack" env create -d "$FMM_GH200_ENV" "$here/spack.yaml"
fi
cp "$here/spack.yaml" "$FMM_GH200_ENV/spack.yaml"
if [ "$update_lock" = 1 ]; then
    rm -f "$FMM_GH200_ENV/spack.lock"
    time "$spack" -e "$FMM_GH200_ENV" concretize --fresh
else
    [ -f "$here/spack.lock" ] || die "no spack.lock beside spack.yaml; run with --update-lock"
    cp "$here/spack.lock" "$FMM_GH200_ENV/spack.lock"
    # With the lock in place this concretises nothing; if it changed the lock, the
    # manifest and the lock disagree.
    "$spack" -e "$FMM_GH200_ENV" concretize
    cmp -s "$here/spack.lock" "$FMM_GH200_ENV/spack.lock" ||
        die "spack.yaml and spack.lock disagree; regenerate the lock with --update-lock"
fi
time "$spack" -e "$FMM_GH200_ENV" install --fail-fast
"$spack" -e "$FMM_GH200_ENV" find --long

# NVIDIA's CUDA runfile installer, which spack's cuda package runs, also writes outside
# $root: its `dnf list --installed nvidia-driver` leaves /var/tmp/dnf-$USER-*, and an
# empty /tmp/hsperfdata_$USER (a JVM's) appears during it. Remove what appeared during
# this run. (Spack's package removes the installer's /tmp/cuda-installer.log itself.)
find /var/tmp -maxdepth 1 -user "$(id -un)" -name 'dnf-*' -newer "$run_stamp" \
    -exec rm -rf {} +
if [ "/tmp/hsperfdata_$(id -un)" -nt "$run_stamp" ]; then
    rmdir "/tmp/hsperfdata_$(id -un)" 2>/dev/null || true
fi

step "Rust $FMM_GH200_RUST"
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
"$CARGO_HOME/bin/rustup" toolchain install "$FMM_GH200_RUST" --profile minimal \
    --component rustfmt --component clippy
"$CARGO_HOME/bin/rustc" --version

step "repository $REPOSITORY"
if [ ! -d "$root/fmm/repo/.git" ]; then
    git clone --quiet "$REPOSITORY" "$root/fmm/repo"
fi
git -C "$root/fmm/repo" log -1 --format='fmm/repo at %h (%cs) %s'

step "done"
if [ "$update_lock" = 1 ]; then
    echo "new lock: $FMM_GH200_ENV/spack.lock; copy it to tools/gh200/spack.lock"
fi
echo "next: . tools/gh200/env.sh"
