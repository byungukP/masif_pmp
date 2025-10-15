#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data
python -W ignore $masif_source/masif_pmp/masif_pmp_predict.py nn_models.all_feat_3l.custom_params $1 $2
