# MaSIF-PMP

MaSIF-PMP is a method to predict membrane-binding interfaces of proteins based on spatially varying geometric and chemical features computed across protein surfaces. MaSIF-PMP is built upon a prior approach [MaSIF](https://github.com/LPDI-EPFL/masif) that uses protein molecular surface fingerprints to predict protein-protein interactions. This project includes codes adapted from the prior work that are licensed under the Apache 2.0.

## Quick try
The easiest way of try out MaSIF-PMP is through a docker container.
```sh
docker run -it byungukp/masif_pmp:latest
```

## Software prerequisites
> Note: will be updated

## Installation
> Note: will be updated
After preinstalling dependencies, add the following environment variables to your path, changing the appropriate directories:
```sh
export APBS_BIN=/path/to/apbs/APBS-1.5-linux64/bin/apbs
export MULTIVALUE_BIN=/path/to/apbs/APBS-1.5-linux64/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=/path/to/apbs/apbs/pdb2pqr-linux-bin64-2.1.1/pdb2pqr
export PATH=$PATH:/path/to/reduce/
export REDUCE_HET_DICT=/path/to/reduce/reduce_wwPDB_het_dict.txt
export PYMESH_PATH=/path/to/PyMesh
export MSMS_BIN=/path/to/msms/msms
export PDB2XYZRN=/path/to/msms/pdb_to_xyzrn
```
Then clone the code to a local directory:
```sh
git clone https://github.com/byungukP/masif_pmp.git
cd masif_pmp
```

## Tutorial
> Note: will be updated

## Reference
> Note: will be updated
If you use this code, please use the bibtex entry in [citation.bib](LINK_TO_BIB)