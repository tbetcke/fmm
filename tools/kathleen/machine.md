# Kathleen: machine facts

Measured on 2026-10-10 (Phase 5N T1) from the M3 Max over `ssh kathleen`, on the login
node `login21.kathleen.ucl.ac.uk` and, where marked, on a compute node inside a job. The
login nodes are of the same kind as the compute nodes.

## The machine

| Item | Value | Command |
| --- | --- | --- |
| Nodes | 190 compute nodes in partition `kathleen` (`node-c11a-[002-144]`, `node-c11b-[002-048]`), whole-node jobs; at 14:59 on a Saturday 188 allocated, 1 idle | `sinfo -s` |
| CPU | 2 × Intel Xeon Gold 6248 @ 2.50 GHz (Cascade Lake), 20 cores per socket, 2 threads per core (80 hardware threads; jobs default to `--hint=nomultithread`) | `lscpu` |
| ISA | `avx512f` and `avx2`, `fma` | `lscpu` (flags) |
| Caches per core | L1d 32 KiB (8-way), L1i 32 KiB, L2 1 MiB (16-way) | `lscpu -C` |
| L3 | 27.5 MiB per socket (11-way), 55 MiB in 2 instances | `lscpu -C` |
| NUMA | 2 nodes: node 0 CPUs 0–19 and 40–59, node 1 CPUs 20–39 and 60–79; distance 10/21; 96 GB each | `numactl -H` |
| Memory | 188 GB visible, no swap | `free -g` |
| Disks | none: `/` (rootfs, `/tmp`) and `/dev/shm` are RAM, 95 GB each | `df -h` (compute node) |
| OS | Red Hat Enterprise Linux 9.6 (Plow), kernel 5.14.0-570.140.1.el9_6.x86_64 | `/etc/os-release`, `uname -r` |
| Slurm | 26.05.4; MPI plugins `none`, `pmix` (`pmix_v3`), `pmi2`, `cray_shasta` | `sinfo --version`, `srun --mpi=list` |
| QoS | `test` 2 nodes exactly, 1 h, 2 jobs per user, priority 1000; `singlenode` 1 node, 6 h, 2 jobs, priority 0; `small` 2–6 nodes, 48 h, priority 100; `medium` 7–12, 24 h; `large` 13+, 12 h; account `allusers` | `sacctmgr show qos`, `sacctmgr show assoc user=$USER` |
| Storage | HOME `/home/ucahtbe` (Lustre); `~/Scratch -> /scratch/scratch/ucahtbe` (Lustre, `/lustre/scratch/ucahtbe`); one hard quota of 250 GiB for both | `ls -l ~/Scratch`, `lquota` |
| Network from compute nodes | static.rust-lang.org, index.crates.io, github.com: HTTP 200 | `curl` in job 238821 |
| Job environment | `TMPDIR=/tmp` set by Slurm; `/dev/shm` empty at job start; the batch step may use CPUs 0–79 | job 238821 |

## Modules (ucl-stack/2026-03)

Loaded by default at login: `ucl-stack/2026-03`, `compilers/gcc/12.3.0/gcc-12.3.0`,
`mpi/openmpi/4.1.6/gcc-12.3.0` (with `numactl/2.0.18`), `cmake/3.30.5/gcc-12.3.0`,
`git/2.46.2/gcc-12.3.0` and their dependencies, `default-modules/2026-03`
(`module -t list`). Available and relevant (`module -t avail --all`):
`mpi/openmpi/4.1.2`, `4.1.6`, `4.1.8` (gcc 12.3.0), Intel oneAPI MPI 2021.14,
`openblas/0.3.28/gcc-12.3.0` (and `-omp`), `fftw/3.3.10/gcc-12.3.0`,
`llvm/17.0.6/gcc-12.3.0-zzkohqr` (hidden), `pkgconf/2.2.0`, `python/3.11.6`,
`intel-oneapi-mkl/2023.2.0`, `opa-psm2/12.0.1`, `libfabric/1.22.0`, `pmix/3.2.3`,
`ucx/1.17.0`, `rust/1.81.0` (too old; unused). The system has `/usr/bin/pkg-config`
and `python3` 3.9.

## What the environment resolved to

`. tools/kathleen/env.sh` on the login node and in jobs:

| Item | Value |
| --- | --- |
| Modules | `ucl-stack/2026-03`, `compilers/gcc/12.3.0/gcc-12.3.0`, `numactl/2.0.18/gcc-12.3.0`, `mpi/openmpi/4.1.6/gcc-12.3.0`, `openblas/0.3.28/gcc-12.3.0`, `fftw/3.3.10/gcc-12.3.0`, `curl/8.10.1/gcc-12.3.0`, `cmake/3.30.5/gcc-12.3.0`, `xz/5.4.6/gcc-12.3.0`, `pcre2/10.44/gcc-12.3.0`, and the hidden `llvm/17.0.6/gcc-12.3.0-zzkohqr` (`module -t list` omits hidden modules) |
| Rust | rustc 1.99.0 (b940084d7 2026-09-28), LLVM 23.1.1; cargo 1.99.0 (5f94df478 2026-08-27); rustup 1.29.1 |
| C and C++ | gcc and g++ 12.3.0 (Spack) as `CC` and `CXX` |
| libclang | 17.0.6, `LIBCLANG_PATH=…/llvm-17.0.6-zzkohqryuv33zorajn5yyjg3pvu5swcn/lib` |
| Open MPI | 4.1.6, configured `--with-slurm --without-pmi --with-psm2=/usr --with-ofi=/usr --without-ucx --without-verbs`; components: PML `cm`, `ob1`, `v`, `monitoring`; MTL `psm2`, `ofi`; BTL `self`, `vader`, `tcp`, `ofi`, `usnic`; PMIx `ext3x` (external PMIx 3.2.3), `isolated`, `flux`; thread support `MPI_THREAD_MULTIPLE: yes` (`ompi_info`) |
| OpenBLAS, FFTW | 0.3.28 and 3.3.10 (`pkg-config --modversion openblas fftw3`); not linked by the workspace |
| Transport | PML `cm` + MTL `psm2` (Omni-Path PSM2, priority 40; `ofi` 25) between nodes and within a node, under `mpirun` and `srun --mpi=pmix`; `ob1`/`vader` initialised, not selected; no TCP. The module sets `OMPI_MCA_btl=self,vader` (jobs 238826, 238840) |
| Launch at 2 × 40 ranks | `mpirun` inside the allocation 0.89–0.91 s, `srun --mpi=pmix` 1.28–1.33 s, for a small nd-octree example (job 238840) |
| Slurm settings that matter | `MpiDefault=pmix`; `TmpFS=/tmp` with `NamespaceType=namespace/tmpfs` (a private `/tmp` per job); `TaskPlugin=task/cgroup,task/affinity`; the batch step has `SLURM_STEP_ID=-5` and `OMPI_MCA_plm_slurm_args=--external-launcher` (`scontrol show config`, job 238830) |
