#!/bin/bash
#SBATCH --image=ethlu/sl7_dune:a
#SBATCH --module=cvmfs
#SBATCH -N 6
#SBATCH --ntasks-per-node=15
#SBATCH -C haswell
#SBATCH -q debug 
#SBATCH -J parse_art
#SBATCH -t 30:00

#run the application:
srun shifter scripts/parse_art.sh -i /project/projectdirs/dune/users/cjslin/protodune/mc_data/singleElectron/reco_1GeV -o reco_1GeV_Electron_parsed 
