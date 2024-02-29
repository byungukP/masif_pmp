# Header variables and parameters.
import os
import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
import tensorflow as tf

"""
masif_site_train.py: Entry function to train MaSIF-site.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

params = masif_opts["site"]

### if IndexError: commented out the following if statements ###
### edited the following to parse HTC argument as custom_parameters for training
### sys.argv[1] = custom_params.py absolute path
if len(sys.argv) > 1:
    custom_params_file = sys.argv[1]
    spec=importlib.util.spec_from_file_location("custom_params",custom_params_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    #custom_params = importlib.import_module(custom_params_file, package=None)
    custom_params = foo.custom_params

    for key in custom_params:
        print("Setting {} to {} ".format(key, custom_params[key]))
        params[key] = custom_params[key]

## original code ##
# if len(sys.argv) > 0:
#     custom_params_file = sys.argv[1]
#     custom_params = importlib.import_module(custom_params_file, package=None)
#     custom_params = custom_params.custom_params

#     for key in custom_params:
#         print("Setting {} to {} ".format(key, custom_params[key]))
#         params[key] = custom_params[key]


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


if "pids" not in params:
    params["pids"] = ["p1", "p2"]

# Confirm that TensorFlow is using the GPU
## for tf v.2.1 or above
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# print("GPUs Available: ", physical_devices)

## for tf v.1.9
from tensorflow.python.client import device_lib
print("All devices available:\n",device_lib.list_local_devices())
print("All GPUs available:\n",tf.test.is_gpu_available())
print("All GPUs name:\n",tf.test.gpu_device_name())

# Build the neural network model
from masif_modules.MaSIF_site import MaSIF_site

if "n_theta" in params:
    learning_obj = MaSIF_site(
        params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        idx_gpu="/device:GPU:0",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
else:
    learning_obj = MaSIF_site(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_gpu="/device:GPU:0",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )

# train
from masif_modules.train_masif_site import train_masif_site

print(params["feat_mask"])
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])
else:
    # Load existing network.
    print ('Reading pre-trained network')
    learning_obj.saver.restore(learning_obj.session, params['model_dir']+'model')

# os.makedirs(dirs) for out_pred_dir, out_surf_dir
if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])
if not os.path.exists(params["out_surf_dir"]):
    os.makedirs(params["out_surf_dir"])

# edit num_iterations ARG for epoch number
train_masif_site(learning_obj, params, num_iterations=params['epoch_num'])

