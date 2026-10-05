# tools/gh200/env.sh: the build and run environment on locust (Phase 4S T1).
#
# Source it, from bash or a POSIX sh, in the repository root on locust:
#
#     . tools/gh200/env.sh
#
# It points every tool that would write into the home directory at
# /data/ucahtbe (Phase 4S decision 2), HOME included, activates the spack environment
# (tools/gh200/spack.yaml), puts the pinned Rust toolchain, the view and the CUDA
# toolkit on the paths, sets every BLAS thread variable to 1 and RUST_MIN_STACK, and
# prints a one-line summary on stderr. Sourcing it twice is harmless.
#
# FMM_GH200_ROOT moves everything to another directory under /data/ucahtbe (setup.sh
# uses that to rebuild into a second prefix). setup.sh sources this file with
# FMM_GH200_SETUP=1, which sets the variables only: no activation, no summary.

# --- Pins -----------------------------------------------------------------------
# The Rust toolchain (decision 1: rustup, because spack's newest `rust` was older than
# the current stable). Keep it at the M3 Max's `rustc --version` and CI's stable.
FMM_GH200_RUST=1.99.0

# --- Layout ---------------------------------------------------------------------
FMM_GH200_ROOT=${FMM_GH200_ROOT:-/data/ucahtbe}
FMM_GH200_ENV=$FMM_GH200_ROOT/envs/fmm-gh200
FMM_GH200_VIEW=$FMM_GH200_ENV/.spack-env/view
export FMM_GH200_ROOT FMM_GH200_ENV FMM_GH200_VIEW FMM_GH200_RUST

# A stand-in HOME, so that what resolves its paths from HOME alone (the tracel-llvm
# bundle's download cache, $HOME/.cache/tracel; git; pip; the shells) stays here too.
export HOME="$FMM_GH200_ROOT/home"
export XDG_CACHE_HOME="$FMM_GH200_ROOT/cache"
# The tracel-llvm bundle installs into $XDG_DATA_HOME/tracel (dirs::data_local_dir).
export XDG_DATA_HOME="$FMM_GH200_ROOT/share"
export XDG_CONFIG_HOME="$FMM_GH200_ROOT/config"
export XDG_STATE_HOME="$FMM_GH200_ROOT/state"
# Temporary files: cargo, rustc, nvcc, spack, Open MPI's session directories.
export TMPDIR="$FMM_GH200_ROOT/tmp"
# The CUDA driver's PTX JIT cache (default ~/.nv/ComputeCache).
export CUDA_CACHE_PATH="$FMM_GH200_ROOT/nv"
export CARGO_HOME="$FMM_GH200_ROOT/rust/cargo"
export RUSTUP_HOME="$FMM_GH200_ROOT/rust/rustup"
export RUSTUP_TOOLCHAIN="$FMM_GH200_RUST"
export SPACK_ROOT="$FMM_GH200_ROOT/spack"
export SPACK_USER_CONFIG_PATH="$FMM_GH200_ROOT/spack-config"
export SPACK_USER_CACHE_PATH="$FMM_GH200_ROOT/spack-cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"

# Every BLAS (and OpenMP) thread variable to 1 (Phase 4S, "Timing").
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Some tests of the MPI crates need it (root CLAUDE.md, "Checks").
export RUST_MIN_STACK=8388608

# Open MPI 5's shared-memory backing files, /dev/shm by default, go to TMPDIR (a local
# disk). Every MPI-initialising test executable runs as a singleton and leaves its
# 16 MiB `sm_segment.*` file behind (measured on 2026-10-05: 18 files after two test
# runs). Phase 5 may want /dev/shm back for multi-rank timings.
export OMPI_MCA_btl_sm_backing_directory="$TMPDIR"
export OMPI_MCA_osc_sm_backing_directory="$TMPDIR"
export OMPI_MCA_osc_rdma_backing_directory="$TMPDIR"
export OMPI_MCA_shmem_mmap_backing_file_base_dir="$TMPDIR"

# Link with the toolchain's own rust-lld, as rustc does by default on x86_64 Linux
# (since 1.90), instead of the system's GNU ld. GNU ld resolves static archives in
# command-line order, and CubeCL's LLVM link line (tracel-llvm-bundler) puts the LLVM
# archives before pliron-llvm, which needs them: every binary that links CubeCL's CPU
# runtime or CUDA fails with undefined LLVMOrc* and LLVM_InitializeNative* symbols.
# lld does not depend on the order. The gcc-ld directory holds rustup's `ld.lld`
# wrapper around rust-lld. Measured on 2026-10-05 (Phase 4S T1).
_fmm_gh200_triple=$(uname -m)-unknown-linux-gnu
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-Clink-arg=-fuse-ld=lld -Clink-arg=-B$RUSTUP_HOME/toolchains/$FMM_GH200_RUST-$_fmm_gh200_triple/lib/rustlib/$_fmm_gh200_triple/bin/gcc-ld"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS
unset _fmm_gh200_triple

# Prepends a directory to a colon-separated variable, once.
_fmm_gh200_prepend() {
    eval "_fmm_old=\${$1:-}"
    case ":$_fmm_old:" in
    *":$2:"*) ;;
    *) eval "export $1=\"$2\${_fmm_old:+:\$_fmm_old}\"" ;;
    esac
    unset _fmm_old
}

_fmm_gh200_prepend PATH "$SPACK_ROOT/bin"
_fmm_gh200_prepend PATH "$CARGO_HOME/bin"

if [ "${FMM_GH200_SETUP:-0}" != 1 ]; then
    if [ -f "$FMM_GH200_ENV/spack.lock" ] && [ -x "$SPACK_ROOT/bin/spack" ]; then
        # The activation puts the view's bin on PATH and sets CUDA_HOME (the cuda
        # package's run environment).
        eval "$("$SPACK_ROOT/bin/spack" env activate --sh -d "$FMM_GH200_ENV")"
        # Spack's activation does not add library and pkg-config directories; do it here.
        _fmm_gh200_prepend PATH "$FMM_GH200_VIEW/bin"
        _fmm_gh200_prepend LD_LIBRARY_PATH "$FMM_GH200_VIEW/lib64"
        _fmm_gh200_prepend LD_LIBRARY_PATH "$FMM_GH200_VIEW/lib"
        _fmm_gh200_prepend PKG_CONFIG_PATH "$FMM_GH200_VIEW/share/pkgconfig"
        _fmm_gh200_prepend PKG_CONFIG_PATH "$FMM_GH200_VIEW/lib64/pkgconfig"
        _fmm_gh200_prepend PKG_CONFIG_PATH "$FMM_GH200_VIEW/lib/pkgconfig"
        _fmm_gh200_prepend CMAKE_PREFIX_PATH "$FMM_GH200_VIEW"
        # cudarc's build script reads the toolkit version from `nvcc --version` and,
        # without nvcc, silently assumes CUDA 13.4. nvcc is on PATH through the view;
        # CUDARC_CUDA_VERSION states the version as well (it takes precedence).
        export CUDA_HOME="${CUDA_HOME:-$FMM_GH200_VIEW}"
        export CUDA_PATH="$CUDA_HOME"
        export CUDARC_CUDA_VERSION=12060
        # bindgen's libclang: the system clang 17.0.6, the `llvm` external of spack.yaml.
        export LIBCLANG_PATH=/usr/lib64

        printf 'fmm-gh200: root %s; HOME %s; env %s; %s, linked by rust-lld; %s; nvcc %s (CUDARC_CUDA_VERSION %s); LIBCLANG_PATH %s\n' \
            "$FMM_GH200_ROOT" "$HOME" "$FMM_GH200_ENV" \
            "$(rustc --version 2>/dev/null || echo 'rustc missing')" \
            "$(mpirun --version 2>/dev/null | head -n 1 || echo 'mpirun missing')" \
            "$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' || true)" \
            "$CUDARC_CUDA_VERSION" "$LIBCLANG_PATH" >&2
    else
        printf 'fmm-gh200: no environment at %s; run tools/gh200/setup.sh first\n' \
            "$FMM_GH200_ENV" >&2
    fi
fi
