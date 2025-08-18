import tempfile

masif_opts = {}
# Default directories
masif_opts["raw_pdb_dir"] = "data_preparation/00-raw_pdbs/"
masif_opts["pdb_chain_dir"] = "data_preparation/01-benchmark_pdbs/"
masif_opts['htmd_dir'] = "data_preparation/01-pdbs_htmd/"
masif_opts['clone_dir'] = "data_preparation/01-pdbs_clone/"
masif_opts['ensemble_pdb_dir'] = "data_preparation/01-benchmark_pdbs_ensemble/"
masif_opts["ply_chain_dir"] = "data_preparation/01-benchmark_surfaces/"
masif_opts["tmp_dir"] = tempfile.gettempdir()
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_{}.ply"
# Path to PMP dataset csv file
masif_opts["pmp_dataset"] = "/masif_pmp/data/masif_pmp/lists/pmp_dataset.csv"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True
masif_opts["compute_ibs"] = True
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
masif_opts["pmp"]["model_dir"] = "nn_models/all_feat_5l/model_data/"
masif_opts["pmp"]["out_pred_dir"] = "output/all_feat_5l/pred_data/"
masif_opts["pmp"]["out_surf_dir"] = "output/all_feat_5l/pred_surfaces/"
masif_opts["pmp"]["out_FPVec_dir"] = "output/all_feat_5l/FPVec/"
masif_opts["pmp"]["feat_mask"] = [1.0] * 5
# k-fold CV test
masif_opts["pmp"]["cv_test"] = False
masif_opts["pmp"]["k_fold"] = 0
# transfer learning
masif_opts["pmp"]["transferLR"] = False

# Neural network patch application specific parameters.
masif_opts["ensemble"] = {}
masif_opts["ensemble"]["training_list"] = "lists/pmp_train.txt"
masif_opts["ensemble"]["testing_list"] = "lists/pmp_test.txt"
masif_opts["ensemble"]["max_shape_size"] = 100
masif_opts["ensemble"]["n_conv_layers"] = 3
masif_opts["ensemble"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["ensemble"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_9A/precomputation/"
masif_opts["ensemble"]["range_val_samples"] = 0.1  # ratio for validation, 0.1 to 0.0
masif_opts["ensemble"]["data_augmentation"] = "naive"  # "naive" or "group"
masif_opts["ensemble"]["model_dir"] = "nn_models/all_feat_5l/model_data/"
masif_opts["ensemble"]["out_pred_dir"] = "output/all_feat_5l/pred_data/"
masif_opts["ensemble"]["out_surf_dir"] = "output/all_feat_5l/pred_surfaces/"
masif_opts["ensemble"]["out_FPVec_dir"] = "output/all_feat_5l/FPVec/"
masif_opts["ensemble"]["feat_mask"] = [1.0] * 5
# k-fold CV test
masif_opts["ensemble"]["cv_test"] = False
masif_opts["ensemble"]["k_fold"] = 0
# transfer learning
masif_opts["ensemble"]["transferLR"] = False


if __name__ == "__main__":
    import sys
    key = sys.argv[1]  # Get the argument from Bash
    print(masif_opts.get(key, "Key not found"))  # Print the value associated with the key
