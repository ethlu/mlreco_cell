#!/bin/bash
#shifter --module=cvmfs --image=ethlu/sl7_dune:a scripts/parse_art.sh

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunetpc v08_60_00 -q e19:prof
export FW_SEARCH_PATH=/project/projectdirs/dune/users/cjslin/pnfs/dune/persistent/stash:$FW_SEARCH_PATH

BASE_DIR=${SCRATCH}/larsim	
IN_DIR=${SCRATCH}/larsim/single_muon_n1
OUT_DIR=single_muon_n1_parsed
while getopts i:o: flag
	do
	    case "${flag}" in
		i) IN_DIR=${OPTARG};;
	        o) OUT_DIR=${OPTARG};;
	    esac
	done
OUT_DIR=${BASE_DIR}/${OUT_DIR}/
mkdir $OUT_DIR
export PYTHONPATH=$PYTHONPATH:../
python io/parse_art.py $IN_DIR $OUT_DIR
