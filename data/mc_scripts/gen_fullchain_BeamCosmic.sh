#!/bin/bash

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunetpc v08_60_00 -q e19:prof
export FW_SEARCH_PATH=/project/projectdirs/dune/users/cjslin/pnfs/dune/persistent/stash:$FW_SEARCH_PATH

OffSet=0
JOBID=`expr ${SLURM_PROCID} + ${OffSet} + 1 `
JOBID=`printf %03d%s ${JOBID}`

BASEDIR=/project/projectdirs/dune/users/cjslin/protodune/mc_data
EVTYPE=BeamCosmic
TAG=reco_1GeV
#TAG=reco_1GeV-pdsp_chnoiseAug2018

cd ${BASEDIR}/${EVTYPE}/${TAG}

#Create temporary data output directory
mkdir ${JOBID}
cd ${JOBID}
cp /global/homes/e/ethanlu/mlreco_cell/data/mc_scripts/larjob/* .

DATAOUT=./
DATAOUT2=../
LOGOUT=${BASEDIR}/logs/${EVTYPE}-${TAG}_${JOBID}.log

GENJOB=mcc12_gen_protoDune_beam_cosmics_p1GeV.fcl
DETSIMJOB=protoDUNE_refactored_detsim.fcl
NUM_EVENTS=5

GENFILE=gen_${JOBID}.root
G4FILE=g4_${EVTYPE}_${JOBID}.root
DETSIMFILE=detsim_${JOBID}.root
RECOFILE=reco_${EVTYPE}_${JOBID}.root

echo 'MC Gen starting at  '`date` > ${LOGOUT}
lar -n $NUM_EVENTS -c ${GENJOB} -o ${DATAOUT}/${GENFILE} >> ${LOGOUT}

echo 'G4 starting at  '`date` >> ${LOGOUT}
lar -n $NUM_EVENTS -c protoDUNE_refactored_g4_sce_datadriven.fcl ${DATAOUT}/${GENFILE} -o ${DATAOUT}/${G4FILE} >> ${LOGOUT}
#lar -n $NUM_EVENTS -c protoDUNE_refactored_g4.fcl ${DATAOUT}/${GENFILE} -o ${DATAOUT}/${G4FILE} >> ${LOGOUT}

echo 'DETSIM starting at  '`date` >> ${LOGOUT}
lar -n $NUM_EVENTS -c ${DETSIMJOB} ${DATAOUT}/${G4FILE} -o ${DATAOUT}/${DETSIMFILE} >> ${LOGOUT}

echo 'Reconstruction starting at  '`date` >> ${LOGOUT}
lar -n $NUM_EVENTS -c protoDUNE_refactored_reco_35ms_sce_datadriven.fcl ${DATAOUT}/${DETSIMFILE} -o ${DATAOUT}/${RECOFILE} >> ${LOGOUT}
#lar -n $NUM_EVENTS -c protoDUNE_refactored_reco.fcl ${DATAOUT}/${DETSIMFILE} -o ${DATAOUT}/${RECOFILE} >> ${LOGOUT}

#mv ${DATAOUT}/${G4FILE} ${DATAOUT2}
mv ${DATAOUT}/${RECOFILE} ${DATAOUT2}

echo 'Job ending at  '`date` >> ${LOGOUT}

cd ../
rm -rf ${JOBID}
