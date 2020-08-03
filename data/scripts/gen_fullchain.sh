#!/bin/bash
#shifter --module=cvmfs --image=ethlu/sl7_dune:a scripts/gen_fullchain.sh

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunetpc v08_57_00 -q e19:prof
export FW_SEARCH_PATH=/project/projectdirs/dune/users/cjslin/pnfs/dune/persistent/stash:$FW_SEARCH_PATH

BASE_DIR=${SCRATCH}/larsim

JOB=larjob/single_muon.fcl
BATCH_SIZE=1
NUM_BATCH=1
OUT_DIR=single_muon_n1
while getopts j:b:n:o: flag
	do
	    case "${flag}" in
	        j) JOB=${OPTARG};;
	        b) BATCH_SIZE=${OPTARG};;
	        n) NUM_BATCH=${OPTARG};;
	        o) OUT_DIR=${OPTARG};;
	    esac
	done

OUT_DIR=${BASE_DIR}/${OUT_DIR}
mkdir $OUT_DIR
mkdir ${OUT_DIR}/0

cd logs
JOB=../${JOB}
I0=$(($(ls -1 ${OUT_DIR} | sort --numeric | tail -1)+1))
for i in $(seq $I0 $((I0+NUM_BATCH-1))); do
	echo "BATCH NUMBER ${i}"
	BATCH_OUT=${OUT_DIR}/${i}
	mkdir $BATCH_OUT
	lar -n $BATCH_SIZE -c $JOB -o ${BATCH_OUT}/gen.root
	lar -n $BATCH_SIZE -c protoDUNE_refactored_g4_sce_datadriven.fcl ${BATCH_OUT}/gen.root -o ${BATCH_OUT}/g4.root
	lar -n $BATCH_SIZE -c protoDUNE_refactored_detsim.fcl ${BATCH_OUT}/g4.root -o ${BATCH_OUT}/detsim.root
	lar -n $BATCH_SIZE -c protoDUNE_refactored_reco_35ms_sce_datadriven.fcl ${BATCH_OUT}/detsim.root -o ${BATCH_OUT}/reco.root
done

rm -r ${OUT_DIR}/0

