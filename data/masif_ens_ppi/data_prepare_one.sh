#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab
if [ "$1" == "--file" ]
then
	echo "Running masif site on $2"
	PPI_PAIR_ID=$3
	PDB_ID=$(echo $PPI_PAIR_ID| cut -d"_" -f1)
	CHAIN1=$(echo $PPI_PAIR_ID| cut -d"_" -f2)
	CHAIN2=$(echo $PPI_PAIR_ID| cut -d"_" -f3)
	FILENAME=$2
	mkdir -p data_preparation/00-raw_pdbs/
	cp $FILENAME data_preparation/00-raw_pdbs/$PDB_ID\.pdb
else
	PPI_PAIR_ID=$1
	PDB_ID=$(echo $PPI_PAIR_ID| cut -d"_" -f1)
	CHAIN1=$(echo $PPI_PAIR_ID| cut -d"_" -f2)
	CHAIN2=$(echo $PPI_PAIR_ID| cut -d"_" -f3)
	python -W ignore $masif_source/data_preparation/00-pdb_download.py $PPI_PAIR_ID
fi

# representative cluster filter
python -W ignore $masif_source/conf_ensemble/filter_clusteredPDB.py $PDB_ID

ensemble_pdb_dir=$(python $masif_source/default_config/masif_opts.py "ensemble_pdb_dir")
if [ ! -d "${ensemble_pdb_dir}/${PDB_ID}" ]
then
	echo -e "\n${PDB_ID}_${CHAIN1} RCSB structure detected."
	if [ -z $CHAIN2 ]
	then
	    echo "Empty"
	    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
	else
	    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
	    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
	fi
	python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID
else
	echo -e "\n${PDB_ID}_${CHAIN1} conformational ensemble detected."
	python -W ignore $masif_source/data_preparation/01f-conf_ensemble_pdb_extract_and_triangulate_center.py $PDB_ID\_$CHAIN1
	python $masif_source/data_preparation/04d-conf_ensemble_masif_precompute_center.py masif_ensemble $PDB_ID\_$CHAIN1

	# loop for centers of PDB_CHAIN_ID w/ 01, 04.py
	# for # OR bash script that do the loop for input PDB_CHAIN_ID
		# python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
		# python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID
fi
