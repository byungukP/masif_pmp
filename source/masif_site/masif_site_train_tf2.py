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
ByungUk Park - UW-Madison 2024
Updated from MaSIF by Pablo Gainza - LPDI STI EPFL 2019
#Released under an Apache License 2.0

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

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


if "pids" not in params:
    params["pids"] = ["p1", "p2"]

# Confirm that TensorFlow is using the GPU
## for tf v.2.1 or above
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
print("GPUs Available: ", physical_devices)

# limiting memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Build the neural network model
# from masif_modules.MaSIF_site import MaSIF_site
from masif_modules.MaSIF_site_wLayers import MaSIF_site

if "n_theta" in params:
    model = MaSIF_site(
        max_rho=params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        idx_gpu="/GPU:0",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
else:
    model = MaSIF_site(
        max_rho=params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_gpu="/GPU:0",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )

# Compile the model ---> no need when using custom training loop (train_step())
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
    # model.fit() ---> no need when using custom training loop (train_step())
# Assuming you have your data loaded as numpy arrays or TensorFlow datasets
# feed_dict as input for the model.fit()

# may try to use tf.data.Dataset.from_tensor_slices() to create dataset (if not already done so)
# provide input_dict in the format shown below
# feed_dict = {
    # "rho_coords": rho_wrt_center,
    # "theta_coords": theta_wrt_center,
    # "input_feat": input_feat,
    # "mask": mask,
    # "labels": iface_labels_dc,
    # "pos_idx": pos_labels,
    # "neg_idx": neg_labels,
    # "indices_tensor": indices,
# }


print(params["feat_mask"])
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])
else:
    # Load existing model.
    print ('Reading pre-trained model')
    model = tf.keras.model.load_weights(params["model_dir"]+'model.weights.h5')

# os.makedirs(dirs) for out_pred_dir, out_surf_dir
if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])
if not os.path.exists(params["out_surf_dir"]):
    os.makedirs(params["out_surf_dir"])

if not params["cv_test"]:
    from masif_modules.train_masif_site_tf2 import train_masif_site
    train_masif_site(model, params, num_epochs=params['epoch_num'])
else:
    from masif_modules.train_masif_site_kfold import train_masif_site_kfold
    train_masif_site_kfold(params, num_epochs=params['epoch_num'])

