#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
python -W ignore $masif_source/masif_site/masif_site_generate_vec.py custom_params $1 $2
# e.g.) $ ./gen_fingerprint_vec.sh -l lists/masif_site_only.txt
