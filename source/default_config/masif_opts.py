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
masif_opts["pmp_dataset"] = "/masif_pmp/data/masif_site/lists/S2File_pmp_dataset.csv"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True
masif_opts["compute_ibs"] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True

# k-fold CV test
masif_opts["cv_test"] = False
masif_opts["k_fold"] = 0

# transfer learning
masif_opts["transferLR"] = False

# Coords params
masif_opts["radius"] = 12.0

# Neural network patch application specific parameters.
masif_opts["ppi_search"] = {}
masif_opts["ppi_search"]["training_list"] = "lists/training.txt"
masif_opts["ppi_search"]["testing_list"] = "lists/testing.txt"
masif_opts["ppi_search"]["max_shape_size"] = 200
masif_opts["ppi_search"]["max_distance"] = 12.0  # Radius for the neural network.
masif_opts["ppi_search"][
    "masif_precomputation_dir"
] = "data_preparation/04b-precomputation_12A/precomputation/"
masif_opts["ppi_search"]["feat_mask"] = [1.0] * 5
masif_opts["ppi_search"]["max_sc_filt"] = 1.0
masif_opts["ppi_search"]["min_sc_filt"] = 0.5
masif_opts["ppi_search"]["pos_surf_accept_probability"] = 1.0
masif_opts["ppi_search"]["pos_interface_cutoff"] = 1.0
masif_opts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["ppi_search"]["cache_dir"] = "nn_models/sc05/cache/"
masif_opts["ppi_search"]["model_dir"] = "nn_models/sc05/all_feat/model_data/"
masif_opts["ppi_search"]["desc_dir"] = "descriptors/sc05/all_feat/"
masif_opts["ppi_search"]["gif_descriptors_out"] = "gif_descriptors/"
# Parameters for shape complementarity calculations.
masif_opts["ppi_search"]["sc_radius"] = 12.0
masif_opts["ppi_search"]["sc_interaction_cutoff"] = 1.5
masif_opts["ppi_search"]["sc_w"] = 0.25

# Neural network patch application specific parameters.
masif_opts["site"] = {}
masif_opts["site"]["training_list"] = "lists/pmp_train.txt"
masif_opts["site"]["testing_list"] = "lists/pmp_test.txt"
masif_opts["site"]["max_shape_size"] = 100
masif_opts["site"]["n_conv_layers"] = 3
masif_opts["site"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["site"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_9A/precomputation/"
masif_opts["site"]["range_val_samples"] = 0.1  # ratio for validation, 0.1 to 0.0
masif_opts["site"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
masif_opts["site"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
masif_opts["site"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
masif_opts["site"]["out_FPVec_dir"] = "output/all_feat_3l/FPVec/"
masif_opts["site"]["feat_mask"] = [1.0] * 5

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
masif_opts["ensemble"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
masif_opts["ensemble"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
masif_opts["ensemble"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
masif_opts["ensemble"]["out_FPVec_dir"] = "output/all_feat_3l/FPVec/"
masif_opts["ensemble"]["feat_mask"] = [1.0] * 5

# Neural network ligand application specific parameters.
masif_opts["ligand"] = {}
masif_opts["ligand"]["assembly_dir"] = "data_preparation/00b-pdbs_assembly"
masif_opts["ligand"]["ligand_coords_dir"] = "data_preparation/00c-ligand_coords"
masif_opts["ligand"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_12A/precomputation/"
masif_opts["ligand"]["max_shape_size"] = 200
masif_opts["ligand"]["feat_mask"] = [1.0] * 5
masif_opts["ligand"]["train_fract"] = 0.9 * 0.8
masif_opts["ligand"]["val_fract"] = 0.1 * 0.8
masif_opts["ligand"]["test_fract"] = 0.2
masif_opts["ligand"]["tfrecords_dir"] = "data_preparation/tfrecords"
masif_opts["ligand"]["max_distance"] = 12.0
masif_opts["ligand"]["n_classes"] = 7
masif_opts["ligand"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
masif_opts["ligand"]["costfun"] = "dprime"
masif_opts["ligand"]["model_dir"] = "nn_models/all_feat/"
masif_opts["ligand"]["test_set_out_dir"] = "test_set_predictions/"

if __name__ == "__main__":
    import sys
    key = sys.argv[1]  # Get the argument from Bash
    print(masif_opts.get(key, "Key not found"))  # Print the value associated with the key
