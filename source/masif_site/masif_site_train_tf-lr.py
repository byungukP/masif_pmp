# Header variables and parameters.
import os
import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
import torch
import torch.nn as nn

"""
masif_site_train.py: Entry function to train MaSIF-site.
ByungUk Park - UW-Madison 2024
Updated from MaSIF by Pablo Gainza - LPDI STI EPFL 2019
#Released under an Apache License 2.0

"""

params = masif_opts["site"]

# Load custom parameters if provided.
if len(sys.argv) > 1:
    custom_params_file = sys.argv[1]
    spec=importlib.util.spec_from_file_location("custom_params",custom_params_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    custom_params = foo.custom_params
    for key in custom_params:
        print("Setting {} to {} ".format(key, custom_params[key]))
        params[key] = custom_params[key]

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


if "pids" not in params:
    params["pids"] = ["p1", "p2"]

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

# train
print(params["feat_mask"])
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])
elif os.path.exists(params["model_dir"] + "model.pt"):
    # Load existing network.
    print('Reading pre-trained network')
    model.load_state_dict(torch.load(params["model_dir"]+'model.pt'))




### Transfer Learning
# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with a new one for our specific task
num_classes = 2
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)




# os.makedirs(dirs) for out_pred_dir, out_surf_dir
if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])
if not os.path.exists(params["out_surf_dir"]):
    os.makedirs(params["out_surf_dir"])

if not params["cv_test"]:
    from masif_modules.train_masif_site_transferLR import train_masif_site_transferLR
    train_masif_site_transferLR(model, params, device, num_epochs=params['epoch_num'])
else:
    ### Update later for Transfer Learning + K-Fold Cross Validation
    from masif_modules.train_masif_site_kfold import train_masif_site_kfold
    train_masif_site_kfold(model, params, device, num_epochs=params['epoch_num'])
