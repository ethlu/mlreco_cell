#!/bin/bash

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunetpc v08_60_00 -q e19:prof
export FW_SEARCH_PATH=/project/projectdirs/dune/users/cjslin/pnfs/dune/persistent/stash:$FW_SEARCH_PATH

OffSet=0
JOBID=`expr ${SLURM_PROCID} + ${OffSet} + 1 `
JOBID=`printf %03d%s ${JOBID}`

cd /global/homes/c/cjslin/DUNE/protodune/mc_data/singlePion

#Create temporary data output directory
mkdir ${JOBID}
cd ${JOBID}

#BASE_DIR=${SCRATCH}/larsim
DATAOUT=/project/projectdirs/dune/users/cjslin/protodune/mc_data/singlePion/${JOBID}
DATAOUT2=/project/projectdirs/dune/users/cjslin/protodune/mc_data/singlePion/
LOGOUT=/project/projectdirs/dune/users/cjslin/protodune/mc_data/logs/singlePion_${JOBID}.log

GENJOB=/project/projectdirs/dune/users/cjslin/protodune/mc_scripts/single_pion.fcl
DETSIMJOB=/project/projectdirs/dune/users/cjslin/protodune/mc_scripts/protoDUNE_refactored_detsim.fcl
NUM_EVENTS=100

GENFILE=gen_${JOBID}.root
G4FILE=g4_singlePion_${JOBID}.root
DETSIMFILE=detsim_${JOBID}.root
RECOFILE=reco_singlePion_${JOBID}.root

#Copy kerberos ticket to batch node (need to make sure it's renewable)
hostname
cp -f ~/.kerberos/* /tmp/
kinit -R;klist

kx509;voms-proxy-init -noregen -rfc -voms dune:/dune/Role=Analysis --pwstdin < /global/homes/c/cjslin/.globus/gridpw.txt


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
rm -rf ${DATAOUT}

echo 'Job ending at  '`date` >> ${LOGOUT}
