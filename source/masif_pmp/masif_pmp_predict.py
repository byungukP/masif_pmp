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
masif_pmp_predict.py: Output prediction scores using trained MaSIF-PMP.
ByungUk Park - UW-Madison 2025
Modified from files in MaSIF (Pablo Gainza - LPDI STI EPFL 2019)
"""

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


params = masif_opts["pmp"]
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
from masif_modules.MaSIF_PMP import MaSIF_PMP

if "n_theta" in params:
    model = MaSIF_PMP(
        max_rho=params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
else:
    model = MaSIF_PMP(
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

# print("\nModel Summary: trainable variables & structure\n")
# model.count_number_parameters()
summary(model)


if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])

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

        tic = time.time()

        input_dict = {
            "rho_coords": torch.tensor(rho_wrt_center, dtype=torch.float32),
            "theta_coords": torch.tensor(theta_wrt_center, dtype=torch.float32),
            "input_feat": torch.tensor(input_feat, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "indices_tensor": torch.tensor(indices, dtype=torch.int32),
            "labels": torch.tensor(labels, dtype=torch.int32), # dummy labels, pos_idx, neg_idx are not used in the forward pass, placeholder for input_dict
            "pos_idx": torch.tensor(labels, dtype=torch.int32),
            "neg_idx": torch.tensor(labels, dtype=torch.int32),
        }
        # move input tensors to the same device w/ parameter tensors of the model
        input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

        with torch.no_grad():   # reduce memory consumption for gradient computations
            logits = model(input_dict)
        full_logits = torch.sigmoid(logits)
        full_score_ = torch.squeeze(full_logits)[:, 0]
        full_score = full_score_.detach().cpu().numpy() # (batch_size,)

        toc = time.time()
        print(
            "Total number of patches for which scores were computed: {}".format(
                len(full_score)
            )
        )
        print("Inference time (real time, not actual GPU time): {:.3f}s \n".format(toc-tic))
        np.save(
            params["out_pred_dir"] + "/pred_" + pdbid + "_" + chains[ix] + ".npy",
            full_score,
        )

        # Clear GPU memory after processing each protein
        torch.cuda.empty_cache()
