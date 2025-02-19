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
	# python -W ignore $masif_source/data_preparation/00-pdb_download.py $PPI_PAIR_ID
fi

# # $2 for taking csv_path as argument from cmd
# if [ -z $CHAIN2 ]
# then
#     echo "Empty"
#     python -W ignore $masif_source/data_preparation/01c-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
# else
#     python -W ignore $masif_source/data_preparation/01c-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
#     python -W ignore $masif_source/data_preparation/01c-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
# fi

if [ "$2" == "--ensemble" ]
then
	echo "Running HTMD & CLoNe for conformational ensemble sampling."
	### python scripts for HTMD & CLoNe runs on the benchmark pdb file (extracted single chain pdb file, already protonated)
	### then save the result files at data_preparation/01-benchmark_pdbs_htmd/, data_preparation/01-benchmark_pdbs_clone/
	# python -W ignore $masif_source/conf_ensemble/runHTMD.py $PDB_ID\_$CHAIN1
	# python -W ignore $masif_source/conf_ensemble/runCLoNe.py $PDB_ID\_$CHAIN1
	python -W ignore $masif_source/conf_ensemble/filter_clusteredPDB.py $PDB_ID\_$CHAIN1

	# no need for PDB protonation of clustered pdb files
	# only need removing water & ions from the PDBs (extraction), precomputation of surface features & triangulation into mesh
	python -W ignore $masif_source/data_preparation/01d-conf_ensemble_pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
	python $masif_source/data_preparation/04c-conf_ensemble_masif_precompute.py masif_site $PPI_PAIR_ID
else
	echo "No conformational ensemble provided. Using default."
	python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID
fi
