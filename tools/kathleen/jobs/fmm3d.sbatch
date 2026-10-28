#!/bin/bash -l
# tools/kathleen/jobs/fmm3d.sbatch: the FMM3D side of the comparison of Phase 6 T9 (C6.8;
# tools/fmm3d/README.md, "Kathleen"). One node, `singlenode`, 6 hours (the QoS's limit),
# unless the submission says otherwise. Each argument is one step:
#   build     tools/fmm3d/build.sh build test: FMM3D (both variants) and the driver, built
#             on the compute node (FMM3D's -march=native) with the gcc module, then
#             FMM3D's own tests; the tarball must be fetched first, on the login node
#             (tools/kathleen/remote.sh 'tools/fmm3d/build.sh fetch')
#   problems, sweep, match, time, crosscheck   the steps of tools/fmm3d/run.sh
#   driver    tools/fmm3d/build.sh driver: relinks only the driver against the existing
#             build (after a change to driver.f90)
#   helmholtz-check      the two-point check of FMM3D's Helmholtz conventions against the
#                        draft §3.15, with the identification (Helmholtz CH1.4)
#   helmholtz-fixtures   fmm-ref/fixtures/helmholtz_fmm3d.json reproduced from this
#                        machine's FMM3D, to round-off (`--check --no-reference`)
#   r12       the R12 sweep of tools/fmm3d/helmholtz.py at N = R12_N (default 10⁶) on
#             R12_THREADS OpenMP threads (default 40; env.sh sets OMP_NUM_THREADS=1 for the
#             BLAS, so the step sets it), its table written to OUT/r12-n<N>.md
#
#     tools/kathleen/remote.sh 'tools/fmm3d/build.sh fetch'
#     tools/kathleen/submit.sh --time=01:00:00 -- fmm3d.sbatch build problems
#     tools/kathleen/submit.sh fmm3d.sbatch sweep
#     tools/kathleen/submit.sh '--export=ALL,OURS=/scratch/scratch/ucahtbe/fmm/logs/scaling-<id>' -- fmm3d.sbatch match time
#     tools/kathleen/submit.sh fmm3d.sbatch crosscheck     # after match: the outputs compared
#
# FMM3D lives in $FMM_KATHLEEN_ROOT/fmm3d (FMM3D_ROOT), its problem files in
# $FMM3D_ROOT/problems; the launches' files go to OUT, by default the root's logs/fmm3d/,
# one directory for every job of the comparison, since match reads what sweep wrote
# (never a source tree: raw output stays out of the repository). The variables of run.sh
# go in as sbatch options (--export), as for sweep.sbatch.
#SBATCH --job-name=fmm-fmm3d
#SBATCH --partition=kathleen
#SBATCH --qos=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=06:00:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

export FMM3D_ROOT=${FMM3D_ROOT:-$FMM_KATHLEEN_ROOT/fmm3d}
export OUT=${OUT:-$FMM_KATHLEEN_ROOT/logs/fmm3d}
mkdir -p "$OUT"
echo "=== FMM3D in $FMM3D_ROOT; output of the launches: $OUT"

for name in "$@"; do
    case "$name" in
    build) step build 'tools/fmm3d/build.sh build test' ;;
    problems | sweep | match | time | crosscheck) step "$name" "tools/fmm3d/run.sh $name" ;;
    driver) step driver 'tools/fmm3d/build.sh driver' ;;
    helmholtz-check) step helmholtz-check 'python3 tools/fmm3d/helmholtz.py check' ;;
    helmholtz-fixtures)
        step helmholtz-fixtures \
            'python3 tools/fixtures/gen_helmholtz_fmm3d.py --check --no-reference'
        ;;
    r12)
        n=${R12_N:-1000000}
        step r12 "set -o pipefail; OMP_NUM_THREADS=${R12_THREADS:-40} python3 tools/fmm3d/helmholtz.py r12 \
            --n $n | tee '$OUT/r12-n$n.md'"
        ;;
    *)
        echo "fmm3d.sbatch: unknown step $name (build, problems, sweep, match, time, crosscheck,"
        echo "  driver, helmholtz-check, helmholtz-fixtures or r12)"
        exit 2
        ;;
    esac
done
step files "ls '$OUT' | wc -l; echo 'failed launches:'; grep -l -e '^FAILED' '$OUT'/*.txt; true"
