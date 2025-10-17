# MaSIF-PMP

MaSIF-PMP is a method for predicting membrane-binding interfaces — referred to as interfacial binding sites (IBSs) — of proteins, based on spatially varying geometric and chemical features computed across protein surfaces. It is built upon the prior framework [MaSIF](https://github.com/LPDI-EPFL/masif), which uses molecular surface fingerprints to predict protein–protein interactions. This project includes code adapted from the original MaSIF implementation, licensed under the Apache 2.0.

## Quick try
The easiest way of try out MaSIF-PMP is through a docker container.
```sh
docker run -it byungukp/masif_pmp:main
```
To fully utilize the GPU for model retraining and mount a local directory into the container:
```sh
docker run -it --gpus all -v ${LOCAL_DIR_PATH}:${MOUNT_PATH} -w /masif_pmp --rm byungukp/masif_pmp:main
```

## System Information & Software Prerequisites
MaSIF-PMP has been tested on Linux/amd64, Ubuntu 20.04, CUDA 12.1.1, and cuDNN 8.9.0. The following software and libraries are required to perform data preprocessing, training, and inferences. Versions in parentheses indicate those used during testing.

* [Python](https://www.python.org/) (3.10)
* [reduce](https://github.com/rlabduke/reduce) (4.14) — Adds protons to protein structures.
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1) — Computes molecular surfaces of proteins. 
* [BioPython](https://github.com/biopython/biopython) (1.85) — Parses PDB files.
* [PyMesh](https://github.com/PyMesh/PyMesh) (0.3.1) — Handles `.ply` surface files, attributes, and mesh regularization.
* PDB2PQR (2.1.1), multivalue, and [APBS](http://www.poissonboltzmann.org/) (1.5) — Compute electrostatic charges.
* [PyTorch](https://pytorch.org/) (2.1) — Framework used for modeling, training, and evaluating neural networks.
Models were trained and evaluated on an NVIDIA L40 GPU.
* [Dask](https://dask.org/) (2.2.0) — Enables multi-threaded execution of function calls.
* [Pymol](https://pymol.org/) — Optional; plugin allows visualization of surface files.

A complete list of dependencies with their versions is provided in [requirements.txt](requirements.txt).

Alternatively, you can use the prebuilt Docker container (see [Quick try](#Quick-try)) or build your own image using the provided [Dockerfile](Dockerfile), which offers the simplest setup.


## Installation
After installing the required dependencies, add the following environment variables to your path (updating the directories as appropriate):
```sh
export PYMESH_PATH=/path/to/PyMesh
export MSMS_BIN=/path/to/msms
export APBS_BIN=/path/to/apbs
export MULTIVALUE_BIN=/path/to/multivalue
export PDB2PQR_BIN=/path/to/pdb2pqr.py
```
Then, clone the repository and navigate to the project directory:
```sh
git clone https://github.com/byungukP/masif_pmp.git
cd data
```

## Usage
The `/masif-pmp/data/` directory contains scripts for training and prediction using MaSIF-PMP, while `/masif-pmp/source/` includes the model source code.
The `/masif-pmp/analysis/` and `/masif-pmp/benchmark/` directories contain scripts and files for reproducing the results presented in:

B. Park and R. C. Van Lehn*. (2025). Decoding protein-membrane binding interfaces from surface-fingerprint-based geometric deep learning and molecular dynamics simulations. *bioRxiv*. https://doi.org/10.1101/2025.10.14.682447. 

### Model Parameters
[masif_opts.py](source/default_config/masif_opts.py) defines the parameters used for data preprocessing, training, and prediction.
Users can adjust these parameters as needed for their specific purposes.
A detailed description of each parameter is provided in the subsections below.

### Prediction
The input can be provided as either a single PDB chain ID (*e.g.*, the perforin C2 domain, PDB ID 4Y1T, chain A → 4Y1T_A) or as a list of IDs in a .txt file.
After data preprocessing, the model takes the processed input and generates predictions for the corresponding protein surface.


```sh
cd data

# For a single PDB_CHAIN ID
./data_prepare_one.sh PDB_CHAIN
./predict_ibs.sh PDB_CHAIN
./color_ibs.sh PDB_CHAIN

# For multiple PDB_CHAIN IDs
# Edit data_prepare_all.sh to use your id_list.txt file as input
./data_prepare_all.sh
./predict_ibs.sh -l id_list.txt
./color_ibs.sh -l id_list.txt
```

By default, as specified in [masif_opts.py](source/default_config/masif_opts.py), all preprocessed and precomputed files are saved under `data/data_preparation/`:
- Raw PDB files downloaded from the RCSB PDB: `data/data_preparation/00-raw_pdbs/`
- Extracted chain PDB files: `data/data_preparation/01-benchmark_pdbs/`
- Mesh files (optionally colored by ground-truth IBS labels if `masif_opts["compute_ibs"] = True`): `data/data_preparation/01-benchmark_surfaces/`
- Precomputed input features: `data/data_preparation/04a-precomputation_9A/precomputation/`

Prediction scores and mesh files colored by the predicted scores are saved in:
- `data/output/all_feat_3l/pred_data/` (prediction scores)
- `data/output/all_feat_3l/pred_surfaces/` (colored mesh files)


### Reproducing Test Set Predictions
To reproduce the predictions on the test set, run the following command:
```sh
./reproduce_pmp_ibs.sh
```

### Retraining
The dataset path for IBS labels is defined in [masif_opts["pmp_dataset"]](source/default_config/masif_opts.py).
Users may replace this path with a custom dataset, but the file must include the following columns:
cathpdb, pdb, residue_name, IBS, chain_id, and residue_number.
Refer to the provided [CSV file](data/lists/pmp_dataset.csv) for the required format.

To train the model, modify [custom_params.py](data/nn_models/all_feat_3l/custom_params.py)
 (*e.g.*, update `custom_params["training_list"]`), move the file to the working directory, and then run:

```sh
./train_nn.sh custom_params.py
```

The best-performing model from the training process will be saved in the directory specified by `custom_params["model_dir"]`.


### Surface Visualization
To visualize the computed surface features and predicted interfaces using the provided PyMOL plugin, please refer to the tutorial from the original MaSIF project:

https://github.com/LPDI-EPFL/masif/blob/master/pymol_plugin_installation.md

## Reference
If you use this code, please cite the work using the BibTeX entry provided in [citation.bib](citation.bib)
