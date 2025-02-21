#!/bin/bash
cat lists/pmp_5cv_trainset_low_auc_100_noOutlier.txt | while read line 
do
    ./data_prepare_one_ibs_ensemble.sh $line\_ --ensemble
done

./predict_site.sh -l lists/pmp_5cv_trainset_low_auc_100_noOutlier.txt
./color_site.sh -l lists/pmp_5cv_trainset_low_auc_100_noOutlier.txt
