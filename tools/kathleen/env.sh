# tools/kathleen/env.sh: the build and run environment on Kathleen (Phase 5N T1). Bash.
#
# Source it from bash (a login shell or a job script started with `#!/bin/bash -l`, so
# that the `module` function exists), in the repository root on Kathleen:
#
#     . tools/kathleen/env.sh
#
# It purges the modules and loads the pinned list below, points every tool that would
# write into the home directory at the root (decision 2), HOME included, puts the pinned
# Rust toolchain on PATH, sets LIBCLANG_PATH from the llvm module, CC and CXX to the gcc
# module's compilers and its runtime libraries first on LD_LIBRARY_PATH, every BLAS
# thread variable to 1, RUST_MIN_STACK, TMPDIR and Open MPI's session directory, unsets
# the batch step's SLURM_STEP_ID (MPI singletons, below), and prints a one-line summary
# on stderr. Sourcing it twice is harmless.
#
# Temporary files: on the login node TMPDIR is $FMM_KATHLEEN_ROOT/tmp (Lustre). Inside a
# Slurm job (SLURM_JOB_ID set) it is /dev/shm/fmm-$USER-$SLURM_JOB_ID, in the node's RAM
# (the nodes are diskless; README.md, "Temporary files"), and Open MPI's session
# directories go to /dev/shm as well. tools/kathleen/jobs/lib.sh creates that directory on
# every node of the job and removes it at the end.
#
# FMM_KATHLEEN_ROOT moves everything to another directory under /scratch/scratch/ucahtbe
# (setup.sh uses that to rebuild into a second root). setup.sh sources this file with
# FMM_KATHLEEN_SETUP=1, which sets the variables only: no modules, no summary.

# --- Pins -----------------------------------------------------------------------
# The Rust toolchain, through rustup. Keep it at the M3 Max's `rustc --version` and
# CI's stable, as tools/gh200/env.sh does.
FMM_KATHLEEN_RUST=1.99.0
# The modules, with full versions, from the ucl-stack/2026-03 stack (machine.md).
# Open MPI's own requirements (numactl) and cmake's and llvm's (curl, xz, pcre2) load
# with them; userscripts is UCL's, for `lquota`.
FMM_KATHLEEN_MODULES="ucl-stack/2026-03 compilers/gcc/12.3.0/gcc-12.3.0
mpi/openmpi/4.1.6/gcc-12.3.0 openblas/0.3.28/gcc-12.3.0 fftw/3.3.10/gcc-12.3.0
cmake/3.30.5/gcc-12.3.0 llvm/17.0.6/gcc-12.3.0-zzkohqr userscripts/2026-03"

# --- Layout ---------------------------------------------------------------------
FMM_KATHLEEN_ROOT=${FMM_KATHLEEN_ROOT:-/scratch/scratch/ucahtbe/fmm}
export FMM_KATHLEEN_ROOT FMM_KATHLEEN_RUST FMM_KATHLEEN_MODULES

# A stand-in HOME, so that what resolves its paths from HOME alone (the tracel-llvm
# bundle's download cache, $HOME/.cache/tracel; git; the shells) stays under the root.
# Kathleen's home and Scratch share one quota: the reason is tidiness, not space.
export HOME="$FMM_KATHLEEN_ROOT/home"
export XDG_CACHE_HOME="$FMM_KATHLEEN_ROOT/cache"
# The tracel-llvm bundle installs into $XDG_DATA_HOME/tracel (dirs::data_local_dir).
export XDG_DATA_HOME="$FMM_KATHLEEN_ROOT/share"
export XDG_CONFIG_HOME="$FMM_KATHLEEN_ROOT/config"
export XDG_STATE_HOME="$FMM_KATHLEEN_ROOT/state"
export CARGO_HOME="$FMM_KATHLEEN_ROOT/rust/cargo"
export RUSTUP_HOME="$FMM_KATHLEEN_ROOT/rust/rustup"
export RUSTUP_TOOLCHAIN="$FMM_KATHLEEN_RUST"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Slurm sets TMPDIR=/tmp, which on these diskless nodes is the RAM root file system.
    # A directory per job in /dev/shm instead, created and removed by jobs/lib.sh.
    TMPDIR="/dev/shm/fmm-$(id -un)-$SLURM_JOB_ID"
    export TMPDIR
    # Open MPI 4.1's session directories (and with them vader's shared-memory backing
    # files): node-local, never on Lustre. Open MPI creates and removes its own
    # subdirectory below this.
    export OMPI_MCA_orte_tmpdir_base=/dev/shm
    # In the batch step Slurm 26.05 sets SLURM_STEP_ID=-5. Open MPI 4.1 reads any step id
    # as "direct-launched by srun", so every MPI singleton (each test executable that
    # initialises MPI, run by `cargo test`) aborts in MPI_Init ("OMPI was not built with
    # SLURM's PMI support"; measured in job 238824). Without it a singleton starts as
    # one. Only the batch shell sources this file: srun steps, and mpirun's daemons,
    # which srun starts, get their own step id from Slurm.
    case "${SLURM_STEP_ID:-}" in
    -5 | 4294967291) unset SLURM_STEP_ID SLURM_STEPID ;;
    esac
else
    export TMPDIR="$FMM_KATHLEEN_ROOT/tmp"
    export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
fi

# Every BLAS (and OpenMP) thread variable to 1 (docs/phase5n/README.md, "Threads and
# BLAS").
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Some tests of the MPI crates need it (root CLAUDE.md, "Checks").
export RUST_MIN_STACK=8388608

# Prepends a directory to a colon-separated variable, once.
_fmm_kathleen_prepend() {
    _fmm_old=
    eval "_fmm_old=\${$1:-}"
    case ":$_fmm_old:" in
    *":$2:"*) ;;
    *) eval "export $1=\"$2\${_fmm_old:+:\$_fmm_old}\"" ;;
    esac
    unset _fmm_old
}

_fmm_kathleen_prepend PATH "$CARGO_HOME/bin"

if [ "${FMM_KATHLEEN_SETUP:-0}" != 1 ]; then
    if command -v module >/dev/null 2>&1; then
        module purge >/dev/null 2>&1
        # shellcheck disable=SC2086 # the list is split on purpose
        module load $FMM_KATHLEEN_MODULES >/dev/null 2>&1 ||
            printf 'fmm-kathleen: module load failed: %s\n' "$FMM_KATHLEEN_MODULES" >&2
        # bindgen's libclang (rsmpi's mpi-sys): the llvm module's, not on the default
        # search path (RHEL 9 has no libclang in /usr/lib64).
        LIBCLANG_PATH=$(llvm-config --libdir 2>/dev/null) && export LIBCLANG_PATH
        # The llvm module also sets CC and CXX to its clang; C and C++ from build
        # scripts (the cc crate) stay with the gcc module's compilers.
        CC=$(command -v gcc) && CXX=$(command -v g++) && export CC CXX
        # The gcc module adds no library path, so binaries with C++ built by its g++
        # (CubeCL's CPU runtime) would load RHEL 9's older /usr/lib64/libstdc++.so.6 and
        # fail ("GLIBCXX_3.4.30 not found"). Its own runtime libraries come first.
        _fmm_kathleen_prepend LD_LIBRARY_PATH "$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
    else
        echo 'fmm-kathleen: no module command; source this from bash -l' >&2
    fi
    mkdir -p "$TMPDIR" 2>/dev/null || true

    printf 'fmm-kathleen: root %s; %s, %s; modules %s; %s; LIBCLANG_PATH %s; TMPDIR %s\n' \
        "$FMM_KATHLEEN_ROOT" \
        "$(rustc --version 2>/dev/null || echo 'rustc missing')" \
        "$(cargo --version 2>/dev/null | cut -d ' ' -f 1-2 || echo 'cargo missing')" \
        "$(module -t list 2>&1 | grep -v ':$' | tr '\n' ' ' | sed 's/ $//')" \
        "$(mpirun --version 2>/dev/null | head -n 1 || echo 'mpirun missing')" \
        "${LIBCLANG_PATH:-unset}" "$TMPDIR" >&2
fi
