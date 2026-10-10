#!/bin/bash -l
# tools/kathleen/jobs/launcher.sbatch: decision 4, the launcher and the transport on two
# nodes (Phase 5N T1). Two nodes, `test`.
#
#     tools/kathleen/submit.sh launcher.sbatch
#
# On an nd-octree MPI example (built in debug by mpi.sbatch, or here):
#   - the start-up wall time at 2 x 40 ranks, three launches each, of `mpirun` inside
#     the allocation and of `srun --mpi=pmix`, with the bindings printed once each;
#   - the transport Open MPI 4.1.6 selects between the nodes and within one, from the
#     PML/MTL/BTL selection messages (verbosity 10) of a two-rank run each way.
#SBATCH --job-name=fmm-launcher
#SBATCH --partition=kathleen
#SBATCH --qos=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00

cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh

exe=target/debug/examples/test_mpi_global_bounding_box
step build 'cargo build -p nd-octree --example test_mpi_global_bounding_box'
step srun-mpi-list 'srun --mpi=list'
step bindings-mpirun "timeout 300 mpirun -n 80 --bind-to core --map-by core --report-bindings $exe 2>&1 | grep -E 'MCW rank (0|19|20|39|40|79) ' | sort -u"
step bindings-srun "timeout 300 srun --mpi=pmix -n 80 --ntasks-per-node=40 --cpu-bind=verbose,cores $exe 2>&1 | grep -iE 'cpu.bind|mask' | sort -u | head -n 12"
for i in 1 2 3; do
    step "mpirun-80-$i" "timeout 300 mpirun -n 80 --bind-to core --map-by core $exe >/dev/null"
    step "srun-80-$i" "timeout 300 srun --mpi=pmix -n 80 --ntasks-per-node=40 --cpu-bind=cores $exe >/dev/null"
done
step transport-between "OMPI_MCA_pml_base_verbose=10 OMPI_MCA_mtl_base_verbose=10 timeout 120 mpirun -n 2 --map-by ppr:1:node $exe 2>&1 | grep -vE 'components_|close:|Querying' | grep -iE 'select|psm2|ofi|priority' | sort -u | head -n 40"
step transport-within "OMPI_MCA_pml_base_verbose=10 OMPI_MCA_mtl_base_verbose=10 OMPI_MCA_btl_base_verbose=10 timeout 120 mpirun -n 2 --map-by ppr:2:node -H \$(hostname -s):2 $exe 2>&1 | grep -vE 'components_|close:|Querying' | grep -iE 'select|vader|psm2|ofi|priority' | sort -u | head -n 40"
step transport-srun "OMPI_MCA_pml_base_verbose=10 OMPI_MCA_mtl_base_verbose=10 timeout 120 srun --mpi=pmix -n 2 --ntasks-per-node=1 $exe 2>&1 | grep -vE 'components_|close:|Querying' | grep -iE 'select|psm2|ofi|priority' | sort -u | head -n 40"
