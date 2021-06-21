#!/bin/bash

for i in `seq 11 110`;
do
    j=" `expr ${i} - 10` "
    mv g4_singleMu_${i}.root  g4_singleMu_`printf %03d%s ${j}`.root 
    mv reco_singleMu_${i}.root reco_singleMu_`printf %03d%s ${j}`.root 
done    
    
