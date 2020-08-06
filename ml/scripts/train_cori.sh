#!/bin/bash
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -J train-hw-xyghost3d
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/v1.5.0
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

# Run the training
srun -l -u -c 64 python train.py -d mpi $@
