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
masif_pmp_train.py: Entry function to train MaSIF-PMP.
ByungUk Park - UW-Madison 2025
Modified from files in MaSIF (Pablo Gainza - LPDI STI EPFL 2019)
"""

torch.cuda.empty_cache()

params = masif_opts["pmp"]

# Load custom parameters if provided.
if len(sys.argv) > 1:
    custom_params_file = sys.argv[1]
    # custom parse
    spec=importlib.util.spec_from_file_location("custom_params",custom_params_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    custom_params = foo.custom_params
    # # fixed parse
    # custom_params = importlib.import_module(custom_params_file, package=None)
    # custom_params = custom_params.custom_params

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

# train
print(params["feat_mask"])
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])
elif os.path.exists(params["model_dir"] + "model.pt"):
    # Load existing network.
    print('Reading pre-trained network')
    model.load_state_dict(torch.load(params["model_dir"]+'model.pt'))

### Transfer Learning
if params["transferLR"]:
    print("Applying transfer learning")

    # =============================================================    
    ## TL opt. 1

    # # Freeze all layers in the model except the final FC block
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Replace the final fully connected layer with a new one for our specific task
    # # FC128 FC64 FC4 FC2
    # from masif_modules.masif_layers import Final_MLPBlock_transferLR
    # model.final_MLPBlock = Final_MLPBlock_transferLR(model.n_thetas, model.n_feat, model.n_labels)

    # =============================================================
    ## TL opt. 2

    # additional soft grid conv layers
    for name, param in model.named_parameters():
        if not name.startswith("tflr_soft_grid"):
            param.requires_grad = False

    from masif_modules.masif_layers import SoftGrid
    initial_coords = model.compute_initial_coordinates()
    mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype("float32")
    mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype("float32")

    model.tflr_soft_grid = nn.ModuleList([
        SoftGrid(
            model.n_thetas,
            model.n_rhos,
            mu_rho_initial,
            model.sigma_rho_init,
            mu_theta_initial,
            model.sigma_theta_init,
            model.n_feat,
            name=f"transfer_l{i+1}",
        )
        for i in range(3)  # 3 new convolutional layers
    ])

    # initialize final_MLPBlock
    from masif_modules.masif_layers import Final_MLPBlock
    model.final_MLPBlock = Final_MLPBlock(model.n_thetas, model.n_feat, model.n_labels)

    # =============================================================

    model.to(device)


# os.makedirs(dirs) for out_pred_dir, out_surf_dir
if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])
if not os.path.exists(params["out_surf_dir"]):
    os.makedirs(params["out_surf_dir"])

if not params["cv_test"]:
    from masif_modules.train_masif_pmp import train_masif_pmp
    train_masif_pmp(model, params, device, num_epochs=params['epoch_num'])
else:
    from masif_modules.train_masif_pmp_kfold import train_masif_pmp_kfold
    train_masif_pmp_kfold(model, params, device, num_epochs=params['epoch_num'])