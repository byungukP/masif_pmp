# Header variables and parameters.
import time
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from IPython.core.debugger import set_trace
import sys
import importlib
# from masif_modules.train_masif_site_tf2 import run_masif_site, compute_roc_auc
from default_config.masif_opts import masif_opts

"""
masif_site_predict.py: Evaluate one or multiple proteins on MaSIF-site. 
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


params = masif_opts["site"]
custom_params_file = sys.argv[1]
spec=importlib.util.spec_from_file_location("custom_params",custom_params_file)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
custom_params = foo.custom_params

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]


# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

all_roc_auc_scores = []

if len(sys.argv) == 3:
    ppi_pair_ids = [sys.argv[2]]
# Read a list of pdb_chain entries to evaluate.
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    listfile = open(sys.argv[3])
    ppi_pair_ids = []
    for line in listfile:
        eval_list.append(line.rstrip())
    for mydir in os.listdir(parent_in_dir):
        ppi_pair_ids.append(mydir)
else:
    sys.exit(1)

# Build the neural network model
from masif_modules.MaSIF_site import MaSIF_site

model = MaSIF_site(
    max_rho=params["max_distance"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    idx_gpu="/gpu:0",
    feat_mask=params["feat_mask"],
    n_conv_layers=params["n_conv_layers"],
)
print("Restoring model from: " + params["model_dir"] + "model\n")
model.load_weights(params["model_dir"] + "model.weights.h5")

# Model Summary
model.count_number_parameters()

if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])

idx_count = 0
for ppi_pair_id in ppi_pair_ids:
    print(ppi_pair_id)
    in_dir = parent_in_dir + ppi_pair_id + "/"

    fields = ppi_pair_id.split('_')
    if len(fields) < 2:
        continue
    pdbid = ppi_pair_id.split("_")[0]
    chain1 = ppi_pair_id.split("_")[1]
    pids = ["p1"]
    chains = [chain1]
    if len(fields) == 3 and fields[2] != "":
        chain2 = fields[2]
        pids = ["p1", "p2"]
        chains = [chain1, chain2]

    for ix, pid in enumerate(pids):
        pdb_chain_id = pdbid + "_" + chains[ix]
        if (
            len(eval_list) > 0
            and pdb_chain_id not in eval_list
            and pdb_chain_id + "_" not in eval_list
        ):
            continue

        print("Evaluating {}".format(pdb_chain_id))

        try:
            rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
        except:
            print("File not found: {}".format(in_dir + pid + "_rho_wrt_center.npy"))
            continue
        theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
        input_feat = np.load(in_dir + pid + "_input_feat.npy")
        input_feat = mask_input_feat(input_feat, params["feat_mask"])
        mask = np.load(in_dir + pid + "_mask.npy")
        indices = np.load(in_dir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
        iface_labels = np.load(in_dir + pid + "_iface_labels.npy")  # (batch_size,)

        input_dict = {
            "rho_coords": rho_wrt_center,
            "theta_coords": theta_wrt_center,
            "input_feat": input_feat,
            "mask": mask,
            "labels": iface_labels,
            "indices_tensor": indices,
        }

        print("Total number of patches: {}".format(len(mask)))

        tic = time.time()
        scores = tf.nn.sigmoid(model(input_dict))   # scores = (batch_size,n_labels)
        # scores = run_masif_site(
        #     params,
        #     model,
        #     rho_wrt_center,
        #     theta_wrt_center,
        #     input_feat,
        #     mask,
        #     indices,
        # )
        toc = time.time()
        print(
            "Total number of patches for which scores were computed: {}".format(
                len(scores[:,0])
            )
        )

        # AUC computation.
        try:
            roc_auc = roc_auc_score(iface_labels, scores[:,0])    # ground_truth(=iface_labels), scores = (batch_size,)
            all_roc_auc_scores.append(roc_auc)
            print("ROC AUC score for protein {} : {:.2f} ".format(pdbid+'_'+chains[ix], roc_auc))
        except: 
            print("No ROC AUC computed for protein (possibly, no ground truth defined in input)") 

        print("GPU time (real time, not actual GPU time): {:.3f}s\n".format(toc-tic))
        np.save(
            params["out_pred_dir"] + "/pred_" + pdbid + "_" + chains[ix] + ".npy",
            scores,
        )   # scores = (batch_size,)

med_roc = np.median(all_roc_auc_scores)

if len(all_roc_auc_scores) > 0:
    print("Computed the ROC AUC for {} proteins".format(len(all_roc_auc_scores)))
    print("Median ROC AUC score: {}".format(med_roc))
