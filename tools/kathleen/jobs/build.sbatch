#!/bin/bash -l
# tools/kathleen/jobs/build.sbatch: the build job (Phase 5N T1). One node, `singlenode`.
#
#     tools/kathleen/submit.sh build.sbatch
#
# Fetches the crates, builds the default members in release, runs `cargo test -p
# nd-fmm-math`, builds every workspace test executable (debug) and the release examples
# that the MPI and scaling jobs launch, and reports the sizes of target/, the cargo
# registry, the toolchain and the tracel-llvm bundle (`lquota` runs on the login node
# only). A cold run is the "cold build" of the README; a second run on an unchanged tree
# is the "warm build".
#SBATCH --job-name=fmm-build
#SBATCH --partition=kathleen
#SBATCH --qos=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

step toolchain 'rustc -vV && cargo -V && gcc --version | head -n 1 && echo "CC=$CC CXX=$CXX"'
step fetch 'cargo fetch'
step build-release 'cargo build --release'
step test-math 'cargo test -p nd-fmm-math'
step tests-no-run 'cargo test --workspace --no-run'
step examples-release 'cargo build --release -p nd-octree --examples && cargo build --release -p nd-fmm-validate --example scaling'
# du counts a directory once, so each separately; lquota exists on the login nodes only.
step sizes 'for d in target target/debug target/release "$CARGO_HOME/registry" "$RUSTUP_HOME" "$XDG_DATA_HOME/tracel" "$HOME/.cache/tracel"; do du -sh "$d"; done'
