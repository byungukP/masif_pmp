# Modified from files in MaSIF (Pablo Gainza - LPDI STI EPFL 2019)
# Released under an Apache License 2.0

import tempfile

masif_opts = {}
# Default directories
masif_opts["raw_pdb_dir"] = "data_preparation/00-raw_pdbs/"
masif_opts["pdb_chain_dir"] = "data_preparation/01-benchmark_pdbs/"
masif_opts["ply_chain_dir"] = "data_preparation/01-benchmark_surfaces/"
masif_opts["tmp_dir"] = tempfile.gettempdir()
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_{}.ply"

# IBS labels
masif_opts["compute_ibs"] = True
# Path to PMP dataset csv file
masif_opts["pmp_dataset"] = "/masif_pmp/data/lists/pmp_dataset.csv"
masif_opts["annotation_type"] = "boolean"          # "boolean" or "score"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True

# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True

# Coords params
masif_opts["radius"] = 12.0

# Neural network patch application specific parameters.
masif_opts["pmp"] = {}
masif_opts["pmp"]["training_list"] = "lists/pmp_train.txt"
masif_opts["pmp"]["testing_list"] = "lists/pmp_test.txt"
masif_opts["pmp"]["max_shape_size"] = 100
masif_opts["pmp"]["n_conv_layers"] = 3
masif_opts["pmp"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["pmp"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_9A/precomputation/"
masif_opts["pmp"]["range_val_samples"] = 0.1  # ratio for validation, 0.1 to 0.0
masif_opts["pmp"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
masif_opts["pmp"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
masif_opts["pmp"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
masif_opts["pmp"]["feat_mask"] = [1.0] * 5
## k-fold CV test
masif_opts["pmp"]["cv_test"] = False
masif_opts["pmp"]["k_fold"] = 0
## transfer learning
masif_opts["pmp"]["transferLR"] = False
## thresholding for binary classification
masif_opts["pmp"]["threshold"] = None        # default: None (for no thresholding)


if __name__ == "__main__":
    import sys
    key = sys.argv[1]  # Get the argument from Bash
    print(masif_opts.get(key, "Key not found"))  # Print the value associated with the key
