#!/bin/bash 
#SBATCH -C haswell -J mcprod -L SCRATCH,project
#SBATCH --volume=/global/project/projectdirs/dune/users/cjslin/pnfs:/pnfs
#SBATCH --module=cvmfs
#SBATCH -N 1     #Number of nodes (Haswell node have 32 cores)
#SBATCH --tasks-per-node=15
#SBATCH --image=ethlu/sl7_dune:a
#SBATCH -q regular -t 2:30:00 #Max hours: regular,premium=48 hrs; debug=0.5 hrs

export my_output=mcProduction_beamcosmic.log

echo 'Production starting at  '`date` > ${my_output}

srun --kill-on-bad-exit=0 --no-kill --label shifter -- mc_scripts/gen_fullchain_BeamCosmic.sh >> ${my_output}

echo 'Production ending at  '`date` >> ${my_output}
