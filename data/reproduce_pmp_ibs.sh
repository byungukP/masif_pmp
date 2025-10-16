#!/bin/bash
cat lists/pmp_test.txt | while read line 
do
    ./data_prepare_one.sh $line\_
done

./predict_ibs.sh -l lists/pmp_test.txt
./color_ibs.sh -l lists/pmp_test.txt
