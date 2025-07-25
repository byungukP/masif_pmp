#!/bin/bash
cat lists/pmp_test.txt | while read line 
do
    ./data_prepare_one.sh $line\_
done

./predict_site.sh -l lists/pmp_test.txt
./color_site.sh -l lists/pmp_test.txt
