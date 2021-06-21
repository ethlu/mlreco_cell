#!/bin/bash 
#SBATCH -C haswell -J mcprod -L SCRATCH,project
#SBATCH --volume=/global/project/projectdirs/dune/users/cjslin/pnfs:/pnfs
#SBATCH --module=cvmfs
#SBATCH -N 12     #Number of nodes (Haswell node have 32 cores)
#SBATCH -n 300    #Total number of tasks (all nodes combined)
#SBATCH --tasks-per-node=25
#SBATCH --image=ethlu/sl7_dune:a
#SBATCH -q regular -t 4:15:00 #Max hours: regular,premium=48 hrs; debug=0.5 hrs
#SBATCH -A dune

cd /global/project/projectdirs/dune/users/cjslin/protodune/mc_scripts/
export my_output=mcProduction_pion.log

echo 'Production starting at  '`date` > ${my_output}

srun --kill-on-bad-exit=0 --no-kill --label shifter -- ./gen_fullchain_singlePion.sh >> ${my_output}

echo 'Production ending at  '`date` >> ${my_output}
