#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
export PYTHONPATH=$PYTHONPATH:$masif_source

if [ "$1" == "--file" ]
then
	echo "Running masif pmp on $2"
	PDB_CHAIN_ID=$3
	PDB_ID=$(echo $PDB_CHAIN_ID| cut -d"_" -f1)
	CHAIN=$(echo $PDB_CHAIN_ID| cut -d"_" -f2)
	FILENAME=$2
	mkdir -p data_preparation/00-raw_pdbs/
	cp $FILENAME data_preparation/00-raw_pdbs/$PDB_ID\.pdb
else
	PDB_CHAIN_ID=$1
	PDB_ID=$(echo $PDB_CHAIN_ID| cut -d"_" -f1)
	CHAIN=$(echo $PDB_CHAIN_ID| cut -d"_" -f2)
	python -W ignore $masif_source/data_preparation/00-pdb_download.py $PDB_CHAIN_ID
fi

python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN
python $masif_source/data_preparation/02-masif_precompute.py masif_pmp $PDB_CHAIN_ID
