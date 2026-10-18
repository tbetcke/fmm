#!/bin/bash -l
# tools/kathleen/jobs/fmm3d.sbatch: the FMM3D side of the comparison of Phase 6 T9 (C6.8;
# tools/fmm3d/README.md, "Kathleen"). One node, `singlenode`, 6 hours (the QoS's limit),
# unless the submission says otherwise. Each argument is one step:
#   build     tools/fmm3d/build.sh build test: FMM3D (both variants) and the driver, built
#             on the compute node (FMM3D's -march=native) with the gcc module, then
#             FMM3D's own tests; the tarball must be fetched first, on the login node
#             (tools/kathleen/remote.sh 'tools/fmm3d/build.sh fetch')
#   problems, sweep, match, time, crosscheck   the steps of tools/fmm3d/run.sh
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
    *)
        echo "fmm3d.sbatch: unknown step $name (build, problems, sweep, match, time or crosscheck)"
        exit 2
        ;;
    esac
done
step files "ls '$OUT' | wc -l; echo 'failed launches:'; grep -l -e '^FAILED' '$OUT'/*.txt; true"
