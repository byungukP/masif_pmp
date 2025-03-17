import time
import os
import torch
from torchinfo import summary
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, roc_auc_score

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))]
        )
    return padded_ix

def check_ensemble(precomp_pdb_dir):
    # check if the precomputed data is from ensemble or not
    # return True if ensemble, False if not or empty
    if not os.listdir(precomp_pdb_dir):
        return False
    for suddir in os.listdir(precomp_pdb_dir):
        if suddir.startswith("Center_"):
            return True
    return False

def naive_data_augmentation(train_dirs, precomp_dir):
    # each representative center as completely independent data (i.e., random shuffle all the inputs and split into train & validation sets)
    aug_train_dirs = []
    for pdb_chain_id in train_dirs:
        target_dir = f"{precomp_dir}/{pdb_chain_id}"
        if check_ensemble(target_dir):
            for suddir in os.listdir(target_dir):
                aug_train_dirs.append(f"{pdb_chain_id}/{suddir}")
        else:
            aug_train_dirs.append(pdb_chain_id)
    return aug_train_dirs

# def group_data_augmentation(input_dict):
#     # treat as a group the centers from same PDB_CHAIN_ID (i.e., random shuffle & split train/validation sets in terms of PDB_CHAIN_IDs)

def updateID(aug_ppi_pair_id):
    # update the pdb_chain_id to include the center_id if from ensemble
    if '/' in aug_ppi_pair_id:
        ppi_pair_id = aug_ppi_pair_id.split('/')[0]
        center_id = aug_ppi_pair_id.split('/')[1].split('_')[1]
        ensemble_id = f"{ppi_pair_id} c{center_id}"
    else:
        ppi_pair_id = aug_ppi_pair_id
        center_id = None
        ensemble_id = ppi_pair_id
    return ppi_pair_id, center_id, ensemble_id


# Run masif site on a protein, on a previously trained network.
# def run_masif_site(
#     params, model, rho_wrt_center, theta_wrt_center, input_feat, mask, indices
# ):
#     indices = pad_indices(indices, mask.shape[1])
#     mask = np.expand_dims(mask, 2)
#     feed_dict = {
#         model.rho_coords: rho_wrt_center,
#         model.theta_coords: theta_wrt_center,
#         model.input_feat: input_feat,
#         model.mask: mask,
#         model.indices_tensor: indices,
#     }

#     logits = model(input_dict)
#     score = model.session.run([model.full_score], feed_dict=feed_dict)
#     return score


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)

def reset_weights(model):
    for layer in model.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters()

from masif_modules.MaSIF_site_wLayers import MaSIF_site

def build_model(params):
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
    return model

def train_masif_ensemble_kfold(
    model,
    params,
    device,
    batch_size=100,
    num_epochs=100
):

    """
    k-fold CV test

    model: loaded model passed from masif_ensemble_train.py
    params: dictionary of hyperparameters

    """

    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    # Open training list + data augmentation
    training_list = open(params["training_list"]).readlines()
    training_list = [x.rstrip() for x in training_list]
    ## data augmentation setup
    if params["data_augmentation"] == "naive":
        training_list = naive_data_augmentation(
            training_list, params["masif_precomputation_dir"]
        )       # ['1CX1_A/Center_1', '1CX1_A/Center_6', '1CX1_A/Center_7', '1AOD_A', ...]
    elif params["data_augmentation"] == "group":
        training_list = training_list
    else:
        raise ValueError("data_augmentation for MaSIF-ensemble should be either 'naive' or 'group'")
    
    training_list = np.array([x.rstrip() for x in training_list])

    # k-fold splits
    kfold = KFold(
        n_splits=params["k_fold"],
        shuffle=True,
        random_state=1
    )    # random_state=1 --> for reproducibility (same splits each time)
    split_count = 0

    # enumerate splits
    for train_idx, test_idx in kfold.split(training_list):  # train, test: array of indices for split data samples
        split_count += 1
        
        logfile.write("\nStarting split {}\n".format(split_count))
        print("\nStarting split {}\n".format(split_count))

        print('train: {}\ntest: {}\ntrain_num: {}, test_num: {}\n'.format(
            training_list[train_idx], training_list[test_idx], len(train_idx), len(test_idx)
            )
        )
        train_dirs = training_list[train_idx]
        val_dirs = training_list[test_idx]

        # re-instantiate model & optimizer for each split
        # reset_weights(model)    # not working
        model = build_model(params)
        print("new model w/ new optimizer built for split {}\n".format(split_count))
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Custom training loop
        for epoch in range(num_epochs):
            # Start training epoch:
            count_proteins = 0
            skipped_pdb_list = []

            # list_training_loss = []     # loss shape = [batch_size, 2] --> list_loss cannot be averaged directly
            list_training_auc = []
            list_val_auc = []

            all_val_labels = []
            all_val_scores = []

            logfile.write("\nStarting epoch {}\n".format(epoch +1))
            print("\nStarting epoch {}\n".format(epoch + 1))
            tic = time.time()

            """
            shuffling the training set before each epoch is a common practice to ensure
            that the model generalizes well to unseen data and is not biased by the
            sequence of training examples.
            For k-fold CV, however, samples within each split will not be shuffled.
            """
            # np.random.shuffle(train_dirs)

            # Training loop: since each protein as batch
            for ppi_pair_id in train_dirs:
                # load all the preprocessed_data (e.g. input feat, labels, label_indices, mask, indices, etc.)
                mydir = params["masif_precomputation_dir"] + "/" + ppi_pair_id + "/"
                pdbid = ppi_pair_id.split("_")[0]
                chains1 = ppi_pair_id.split("_")[1]
                if len(ppi_pair_id.split("_")) > 2:
                    chains2 = ppi_pair_id.split("_")[2]
                else: 
                    chains2 = ''
                pids = []   # might need to use pids for handling representative confs from dynamic ensemble from unbiased htmd
                if pdbid + "_" + chains1 in training_list:
                    pids.append("p1")
                if pdbid + "_" + chains2 in training_list:
                    pids.append("p2")
                for pid in pids:    # each pid representing different conf among ensemble --> thus, multiple trainings (var updates) w/ diff conf from same pdb_chain_id
                    try:
                        iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                    except:
                        continue
                    if (
                        len(iface_labels) > 8000
                        or np.sum(iface_labels) > 0.75 * len(iface_labels)
                        or np.sum(iface_labels) < 30
                    ):
                        skipped_pdb_list.append(f"{ppi_pair_id} {pid}")
                        continue
                    count_proteins += 1

                    rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                    theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                    input_feat = np.load(mydir + pid + "_input_feat.npy")
                    if np.sum(params["feat_mask"]) < 5:
                        input_feat = mask_input_feat(input_feat, params["feat_mask"])
                    mask = np.load(mydir + pid + "_mask.npy")
                    mask = np.expand_dims(mask, 2)
                    indices = np.load(mydir + pid + "_list_indices.npy", allow_pickle=True, encoding="latin1")
                    indices = pad_indices(indices, mask.shape[1])
                    tmp = np.zeros((len(iface_labels), 2))
                    for i in range(len(iface_labels)):
                        if iface_labels[i] == 1:
                            tmp[i, 0] = 1
                        else:
                            tmp[i, 1] = 1
                    iface_labels_dc = tmp
                    logfile.flush()
                    pos_labels = np.where(iface_labels == 1)[0]
                    neg_labels = np.where(iface_labels == 0)[0]
                    np.random.shuffle(neg_labels)
                    np.random.shuffle(pos_labels)
                    # Scramble neg idx, and only get as many as pos_labels to balance the training.
                    # if params["n_conv_layers"] == 1:
                    if params["n_conv_layers"] > 1:
                        n = min(len(pos_labels), len(neg_labels))
                        neg_labels = neg_labels[:n]
                        pos_labels = pos_labels[:n]

                    # then, save as input_dict
                    input_dict = {
                        "rho_coords": torch.tensor(rho_wrt_center),
                        "theta_coords": torch.tensor(theta_wrt_center),
                        "input_feat": torch.tensor(input_feat),
                        "mask": torch.tensor(mask),
                        "labels": torch.tensor(iface_labels_dc),
                        "pos_idx": torch.tensor(pos_labels),
                        "neg_idx": torch.tensor(neg_labels),
                        "indices_tensor": torch.tensor(indices),
                    }
                    # move input tensors to the same device w/ parameter tensors of the model
                    input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

                    # Perform training step
                    logfile.write("Training on {} {}\n".format(ppi_pair_id, pid))
                    # input_dict["keep_prob"] = 1.0

                    print("\nTraining on {} {}\n".format(ppi_pair_id, pid))
                    logs = model.training_step(input_dict, optimizer)
                    list_training_auc.append(logs["auc"])
                    logfile.flush()

            # Validation loop
            for ppi_pair_id in val_dirs:
                # load all the preprocessed_data (e.g. input feat, labels, label_indices, mask, indices, etc.)
                mydir = params["masif_precomputation_dir"] + "/" + ppi_pair_id + "/"
                pdbid = ppi_pair_id.split("_")[0]
                chains1 = ppi_pair_id.split("_")[1]
                if len(ppi_pair_id.split("_")) > 2:
                    chains2 = ppi_pair_id.split("_")[2]
                else: 
                    chains2 = ''
                pids = []   # might need to use pids for handling representative confs from dynamic ensemble from unbiased htmd
                if pdbid + "_" + chains1 in training_list:
                    pids.append("p1")
                if pdbid + "_" + chains2 in training_list:
                    pids.append("p2")
                for pid in pids:    # each pid representing different conf among ensemble --> thus, multiple trainings (var updates) w/ diff conf from same pdb_chain_id
                    try:
                        iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                    except:
                        continue
                    if (
                        len(iface_labels) > 8000
                        or np.sum(iface_labels) > 0.75 * len(iface_labels)
                        or np.sum(iface_labels) < 30
                    ):
                        skipped_pdb_list.append(f"{ppi_pair_id} {pid}")
                        continue
                    count_proteins += 1

                    rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                    theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                    input_feat = np.load(mydir + pid + "_input_feat.npy")
                    if np.sum(params["feat_mask"]) < 5:
                        input_feat = mask_input_feat(input_feat, params["feat_mask"])
                    mask = np.load(mydir + pid + "_mask.npy")
                    mask = np.expand_dims(mask, 2)
                    indices = np.load(mydir + pid + "_list_indices.npy", allow_pickle=True, encoding="latin1")
                    indices = pad_indices(indices, mask.shape[1])
                    tmp = np.zeros((len(iface_labels), 2))
                    for i in range(len(iface_labels)):
                        if iface_labels[i] == 1:
                            tmp[i, 0] = 1
                        else:
                            tmp[i, 1] = 1
                    iface_labels_dc = tmp
                    logfile.flush()
                    pos_labels = np.where(iface_labels == 1)[0]
                    neg_labels = np.where(iface_labels == 0)[0]
                    np.random.shuffle(neg_labels)
                    np.random.shuffle(pos_labels)
                    # Scramble neg idx, and only get as many as pos_labels to balance the training.
                    # if params["n_conv_layers"] == 1:
                    if params["n_conv_layers"] > 1:
                        n = min(len(pos_labels), len(neg_labels))
                        neg_labels = neg_labels[:n]
                        pos_labels = pos_labels[:n]

                    # then, save as input_dict
                    input_dict = {
                        "rho_coords": torch.tensor(rho_wrt_center),
                        "theta_coords": torch.tensor(theta_wrt_center),
                        "input_feat": torch.tensor(input_feat),
                        "mask": torch.tensor(mask),
                        "labels": torch.tensor(iface_labels_dc),
                        "pos_idx": torch.tensor(pos_labels),
                        "neg_idx": torch.tensor(neg_labels),
                        "indices_tensor": torch.tensor(indices),
                    }
                    # move input tensors to the same device w/ parameter tensors of the model
                    input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

                    logfile.write("Validating on {} {} ==> ".format(ppi_pair_id, pid))
                    # input_dict["keep_prob"] = 1.0   # not sure of the purpose of keep_prob, remove later if unnecessary

                    logs = model.validation_step(input_dict)
                    logfile.write("Per protein AUC: {:.4f}\n".format(logs["auc"]))
                    list_val_auc.append(logs["auc"])
                    all_val_labels.append(iface_labels)
                    all_val_scores.append(logs["full_score"])

                    logfile.flush()

            # Run testing cycle. --> not in this code, but might be added later
            # focus on running 5-CV with this code, prediction test w/ predict_site.sh later

            # Summary of epoch
            outstr = "Epoch ran on {} proteins\n".format(count_proteins)
            outstr += "{} proteins skipped to prevent biased fitting: {}\n\n".format(
                len(skipped_pdb_list), skipped_pdb_list
            )
            ## per protein metrics (training)
            outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for epoch {}\n".format(
                np.mean(list_training_auc), np.median(list_training_auc), epoch +1
            )
            ## per protein metrics (validation)
            outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for epoch {}\n".format(
                np.mean(list_val_auc), np.median(list_val_auc), epoch +1
            )
            ## all points metrics (validation)
            flat_all_val_labels = np.concatenate(all_val_labels, axis=0)
            flat_all_val_scores = np.concatenate(all_val_scores, axis=0)
            outstr += "Validation auc (all points): {:.2f}\n".format(
                metrics.roc_auc_score(flat_all_val_labels, flat_all_val_scores)
            )

            # outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for epoch {}\n".format(
            #     np.mean(list_test_auc), np.median(list_test_auc), epoch +1
            # )
            # flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
            # flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
            # outstr += "Testing auc (all points): {:.2f}\n".format(
            #     metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
            # )
            outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
            logfile.write(outstr + "\n")
            print(outstr)

            if epoch + 1 == num_epochs:
                outstr = ">>> Split {:d} CV test done\n".format(split_count)
                outstr += ">>> Per protein AUC mean (training): {:.4f}; median: {:.4f}\n".format(
                    np.mean(list_training_auc), np.median(list_training_auc)
                )
                outstr += ">>> Per protein AUC mean (validation): {:.4f}; median: {:.4f}\n".format(
                    np.mean(list_val_auc), np.median(list_val_auc)
                )
                outstr += ">>> Validation auc (all points): {:.2f}\n".format(
                    metrics.roc_auc_score(flat_all_val_labels, flat_all_val_scores)
                )
                logfile.write(outstr + "\n")
                print(outstr)

    # Display the model's architecture: built-in model.summary() for functional API models
    print("\nModel Summary: trainable variables & structure\n")
    model.count_number_parameters()
    summary(model)

    logfile.close()
