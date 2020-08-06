#!/bin/bash
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J xy
#SBATCH -t 1:30:00

#OpenMP settings:
export OMP_NUM_THREADS=64
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread


#run the application:
#srun -n 1 -c 64 --cpu_bind=cores scripts/make_batches.sh -i $SCRATCH/larsim/reco_1GeV_parsed/ -o reco_1GeV_xy -n 64
srun -n 2 -c 64 scripts/make_batches.sh -i $SCRATCH/larsim/reco_1GeV_parsed/ -o reco_1GeV_xy_hw -n $OMP_NUM_THREADS
