#!/bin/bash

i=1
while read p; do
    ./data_prepare_one_ibs_ensemble.sh $p --ensemble
    i=$((i+1))
done < lists/pmp_train_test.txt
