# Header variables and parameters.
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import sys
import importlib
from masif_modules.train_masif_site import pad_indices
from default_config.masif_opts import masif_opts
import torch
from torchinfo import summary

"""
masif_site_generate_vec.py
: Generate fingerprint vectors for one or multiple proteins with pretrained MaSIF-site. 
ByungUk Park - UW-Madison 2024

This file is part of MaSIF.
Released under an Apache License 2.0
"""

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


params = masif_opts["site"]
custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params
for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]


# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

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

# Confirm that PyTorch is using the GPU
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu"
)
print(f"Using device: {device}")

# Build the neural network model
from masif_modules.MaSIF_site_wLayers import MaSIF_site

if "n_theta" in params:
    model = MaSIF_site(
        max_rho=params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
else:
    model = MaSIF_site(
        max_rho=params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
model.to(device)

print("Restoring model from: " + params["model_dir"] + "model.pt")
model.load_state_dict(torch.load(params["model_dir"]+'model.pt'))

print("\nModel Summary: trainable variables & structure\n")
model.count_number_parameters()
summary(model)


if not os.path.exists(params["out_FPVec_dir"]):
    os.makedirs(params["out_FPVec_dir"])

idx_count = 0
for ppi_pair_id in ppi_pair_ids:
    print(ppi_pair_id)
    in_dir = parent_in_dir + ppi_pair_id + "/"

    fields = ppi_pair_id.split("_")
    if len(fields) < 2:
        continue
    pdbid = fields[0]
    chain1 = fields[1]
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
            print("Skipping {}: not included in the eval_list".format(pdb_chain_id))
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
        mask = np.expand_dims(mask, 2)
        indices = np.load(in_dir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
        indices = pad_indices(indices, mask.shape[1])
        labels = np.zeros((len(mask))) 

        print("Total number of patches: {}".format(len(mask)))

        # generate subdir
        if not os.path.exists(params["out_FPVec_dir"] + "/" + pdb_chain_id):
            os.makedirs(params["out_FPVec_dir"] + "/" + pdb_chain_id)

        tic = time.time()

        input_dict = {
            "rho_coords": torch.tensor(rho_wrt_center),
            "theta_coords": torch.tensor(theta_wrt_center),
            "input_feat": torch.tensor(input_feat),
            "mask": torch.tensor(mask),
            "indices_tensor": torch.tensor(indices),
            "labels": torch.tensor(labels), # dummy labels, pos_idx, neg_idx are not used in the forward pass, placeholder for input_dict
            "pos_idx": torch.tensor(labels),
            "neg_idx": torch.tensor(labels),
        }
        # move input tensors to the same device w/ parameter tensors of the model
        input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

        with torch.no_grad():   # reduce memory consumption for gradient computations
            fpvec_list_, feat_vec_list_ = model.gen_FPVec(input_dict)
            # fpvec_list: n_conv_layers * (batch_size, n_gauss*n_feat),
            # feat_vec_list: n_conv_layers * (batch_size, n_feat)

        fpvec_list = [tensor.detach().cpu().numpy() for tensor in fpvec_list_]
        feat_vec_list = [tensor.detach().cpu().numpy() for tensor in feat_vec_list_]

        toc = time.time()
        print(
            "Total number of patches for which fingerprint vectors were computed: {}".format(
                len(fpvec_list[0])
            )
        )
        print("Calculation time (real time, not actual GPU time): {:.3f}s \n".format(toc-tic))

        # save each fingerprint vector to a separate npy file
        try:
            len(fpvec_list) == len(feat_vec_list)
        except:
            raise ValueError("The number of fingerprint vectors and feature vectors must be the same.")

        for i in range(len(fpvec_list)):
            np.save(
                params["out_FPVec_dir"] + "/" + pdb_chain_id + "/fpvec_l" + str(i+1) + ".npy",
                fpvec_list[i],
            )
            np.save(
                params["out_FPVec_dir"] + "/" + pdb_chain_id + "/featvec_l" + str(i+1) + ".npy",
                feat_vec_list[i],
            )

        # Clear GPU memory after processing each protein
        torch.cuda.empty_cache()
