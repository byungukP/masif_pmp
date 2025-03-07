#!/bin/bash
cat lists/pmp_train_test.txt | while read line 
do
    ./data_prepare_one_ibs_ensemble.sh $line\_ --ensemble
done

./predict_site.sh -l lists/pmp_train_test.txt
./color_site.sh -l lists/pmp_train_test.txt
