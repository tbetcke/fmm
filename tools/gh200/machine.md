# locust: machine facts

Measured on 2026-10-05 (Phase 4S T1), read-only, with the commands in the last column.
The Phase 4S README ("Machines") carries the earlier probe of the same day; nothing
differed.

## Hardware and system

| Item | Value | Command |
| --- | --- | --- |
| Host | `locust.rc.ucl.ac.uk`, shared and interactive, no batch scheduler | `hostname` |
| OS | Red Hat Enterprise Linux 9.3 (Plow), kernel `5.14.0-362.18.1.el9_3.aarch64+64k` | `cat /etc/redhat-release`, `uname -r` |
| Page size | **65536 bytes (64 KiB)** | `getconf PAGESIZE` |
| CPU | 72 × Arm Neoverse-V2 (Grace), 1 thread per core, one socket, max 3474 MHz; flags include `asimd sve sve2 i8mm bf16` | `lscpu` |
| Caches | L1d 64 KiB, L2 1 MiB per core; L3 114 MiB shared | `lscpu` |
| NUMA | 9 nodes; node 0 holds all 72 CPUs and 489 GB, the others no CPUs | `lscpu`, `numactl -H` |
| Host memory | 572 GB (LPDDR5X) | `free -g` |
| GPU | NVIDIA GH200 480GB, Hopper, compute capability 9.0, 97871 MiB (96 GB HBM3), max SM clock 1980 MHz, max memory clock 2619 MHz, power limit 900 W, ECC on, MIG off, addressing mode ATS | `nvidia-smi -q`, `nvidia-smi --query-gpu=compute_cap,memory.total,clocks.max.sm,clocks.max.mem,power.limit,ecc.mode.current --format=csv` |
| Driver | 565.57.01 (CUDA 12.7) | `nvidia-smi -q` |
| CUDA toolkits (system, unused) | `/usr/local/cuda-11.7`, `12.2`, `12.3`, `12.4`, `12.6` (and `cuda-12`); `nvcc` not on the `PATH` | `ls -d /usr/local/cuda*` |
| System packages (RPM) | gcc and gcc-gfortran 11.4.1-2.1, clang and clang-devel 17.0.6-5, openmpi 4.1.7a1 (not on the `PATH`), openblas-devel 0.3.21, fftw-devel 3.3.8, cmake 3.26.5, git 2.39.3, python3 3.9.18, rsync 3.2.3 | `rpm -q …` |
| Tools | environment modules without site modules; no Rust, no spack before T1 | `module avail` |
| Storage | `/data` 1.8 TB, 459 GB free after T1's setup (472 GB before); `/home` 819 GB, 78% used, not to be used | `df -h /data /home` |
| Network | github.com, static.rust-lang.org, index.crates.io, mirror.spack.io reachable; the CUDA runfile downloaded from developer.download.nvidia.com through spack | `curl -sI` |

The GPU was in use by another user at every check on 2026-10-05: one `python` process,
74 GB, 100% utilisation, 680 W. T1 opened the device only to print `DeviceInfo`.

## What the environment resolved to

From `spack.lock` (lockfile v6, 55 specs) and `rustup`, all under `/data/ucahtbe`:

| Component | Version | Note |
| --- | --- | --- |
| spack | 1.2.2 (`3e19345b`) | packages at spack-packages `d4f7c711` |
| target | `linux-rhel9-neoverse_v2` | |
| compiler | gcc 11.4.1 (external, `/usr`) | C, C++, Fortran |
| libclang | llvm 17.0.6 (external, `/usr`, `+clang`) | `LIBCLANG_PATH=/usr/lib64` |
| MPI | openmpi 5.0.10 `~fortran`, with pmix 6.1.0, prrte 4.1.0, hwloc 2.13.0, libevent 2.1.12 | openssh 10.3p1 built for the default `+rsh` |
| BLAS/LAPACK | openblas 0.3.33 `threads=none` | |
| FFTW | fftw 3.3.11 | |
| build tools | cmake 3.31.11, pkgconf 2.5.1 | |
| CUDA | cuda 12.6.3 | `CUDARC_CUDA_VERSION=12060` |
| Rust | 1.99.0 (`b940084d7 2026-09-28`), rustfmt, clippy; rustup 1.29.1 | the M3 Max's and CI's stable on 2026-10-05 |

Disk use after the first setup: `spack-install` 6.5 GB, `spack-sources` 3.8 GB (CUDA's
runfile most of it), `rust` 1.0 GB, `spack-cache` 129 MB.

Times on 2026-10-05, the node otherwise lightly loaded (load average about 1):

| Step | Wall time |
| --- | --- |
| `spack concretize --fresh` (clingo bootstrap included) | 20 s |
| `spack install`, 55 specs (52 built from source; gcc, glibc and llvm external) | 14 min 50 s (68 min CPU); CUDA 12.6.3 11 min 50 s of it, 10 min 30 s in its runfile installer |
| rustup and the 1.99.0 toolchain | 5 s |
| `setup.sh` again, with the lock (nothing to do) | 1.2 s |

## Observed while building and testing (T1)

- Cold build of the workspace (`cargo build --workspace --all-targets`, debug, empty
  `target/`, registry and `tracel-llvm` bundle already downloaded): 80 s on the 72
  cores. The bundle (`tracel-llvm-23.1.0-3-linux-AArch64.tar.xz`, 43 MB) landed in
  `/data/ucahtbe/home/.cache/tracel/` and was installed into
  `/data/ucahtbe/share/tracel/tracel-llvm-23.1.0-3/`.
- 64 KiB pages: nothing failed or warned. The CubeCL CPU runtime's LLVM JIT ran all 91
  kernel tests, Open MPI 5 ran a two-rank example on the node, and the 715 workspace
  tests passed. Not checked: which Open MPI transport carried the two-rank run.
- GNU ld 2.35.2 cannot link CubeCL's LLVM (archive order); `env.sh` links with
  rust-lld (README.md, "The linker").
- cudarc resolved CUDA 12.6 (`feature="cuda-12060"`, `CUDA_MAJOR_VERSION=12`,
  `CUDA_MINOR_VERSION=6` in its build-script output), with `nvcc` 12.6 on the `PATH` and
  `CUDARC_CUDA_VERSION=12060`.
- `Device::open(BackendKind::Cuda)` on the H100:
  `cuda (cuda), NVIDIA GH200 480GB, CubeCL 0.11.0-pre.4, f32 f64`; plane size 32,
  shared memory 232448 bytes per cube, 1024 units per cube, cube count
  (2147483647, 65535, 65535), max memory 102005473280 bytes.
- The CubeCL CPU runtime: `cpu (cpu), Neoverse-V2, CubeCL 0.11.0-pre.4, f32 f64`.
