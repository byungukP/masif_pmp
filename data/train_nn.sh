#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data
python3 $masif_source/masif_pmp/masif_pmp_train.py $1
