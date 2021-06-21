#!/bin/bash
#shifter --module=cvmfs --image=ethlu/sl7_dune:a scripts/wirecell_img.sh

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunetpc v08_60_00 -q e19:prof
export FW_SEARCH_PATH=/project/projectdirs/dune/users/cjslin/pnfs/dune/persistent/stash:$FW_SEARCH_PATH

BASE_DIR=${SCRATCH}/larsim	
IN_DIR=${SCRATCH}/larsim/single_muon_n1
OUT_DIR=wirecell_img
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
export WIRECELL_PATH=~/wire-cell-toolkit/cfg:$WIRECELL_PATH

N_EVENTS=100
I_FIRST=301
I_LAST=301
for i in $(seq $I_FIRST $I_LAST); do
	#INDEX=`printf '%04d' $i`
	INDEX=`printf '%03d' $i`
	IN_FILE=`find $IN_DIR -name *_$INDEX*`
	echo "FILE ${IN_FILE}"
	BATCH_OUT=${OUT_DIR}/${INDEX}
	mkdir $BATCH_OUT
	for event in $(seq 0 $(($N_EVENTS-1))); do
		EVENT_OUT=$BATCH_OUT/$(($event))
		mkdir $EVENT_OUT
		cd $EVENT_OUT
		lar -c ~/wire-cell-toolkit/cfg/pgrapher/experiment/pdsp/wcls-sig-to-img.fcl -n 1 --nskip $event $IN_FILE
		rm wcls-sig-to-img.npz
	done
done

