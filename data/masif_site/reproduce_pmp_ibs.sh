#!/bin/bash
cat lists/pmp_full_list.txt | while read line 
do
    ./data_prepare_one_ibs.sh $line\_
done

./predict_site.sh -l lists/pmp_full_list.txt
./color_site.sh -l lists/pmp_full_list.txt
