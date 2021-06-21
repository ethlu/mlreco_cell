#!/bin/bash

module load python
source activate myenv

BASE_DIR=${SCRATCH}/larsim
IN_DIR=${SCRATCH}/larsim/single_muon_parsed
OUT_DIR=single_muon_xy
NUM_THREADS=32
BATCH_SIZE=""
TPC=1
while getopts i:o:n:b:t: flag
	do
	    case "${flag}" in
		i) IN_DIR=${OPTARG};;
	        o) OUT_DIR=${OPTARG};;
	        t) TPC=${OPTARG};;
	        n) NUM_THREADS=${OPTARG};;
	        b) BATCH_SIZE=${OPTARG};;
	    esac
	done
OUT_DIR=${BASE_DIR}/${OUT_DIR}/
mkdir $OUT_DIR
export PYTHONPATH=$PYTHONPATH:../
NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=$NUM_THREADS python -u make_batches.py $IN_DIR $OUT_DIR $TPC $BATCH_SIZE
