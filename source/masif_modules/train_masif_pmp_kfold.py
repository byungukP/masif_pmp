import time
import os
import torch
from torchinfo import summary
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold


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

def load_data(mydir, pid, params, for_training=True):
    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
    rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
    theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
    input_feat = np.load(mydir + pid + "_input_feat.npy")
    if np.sum(params["feat_mask"]) < 5:
        input_feat = mask_input_feat(input_feat, params["feat_mask"])
    mask = np.load(mydir + pid + "_mask.npy")
    mask = np.expand_dims(mask, 2)
    indices = np.load(mydir + pid + "_list_indices.npy", allow_pickle=True, encoding="latin1")
    # since some patches may have less than max shape size (100) for patch representation but all the matrix are generated based on max shape size for computational reason,
    # need padding for matrix of indice list
    indices = pad_indices(indices, mask.shape[1])
    tmp = np.zeros((len(iface_labels), 2))
    for i in range(len(iface_labels)):
        if iface_labels[i] == 1:
            tmp[i, 0] = 1
        else:
            tmp[i, 1] = 1
    iface_labels_dc = tmp
    pos_labels = np.where(iface_labels == 1)[0]
    neg_labels = np.where(iface_labels == 0)[0]

    if for_training:
        # Scramble neg idx, and only get as many as pos_labels to balance the training.
        np.random.shuffle(neg_labels)
        np.random.shuffle(pos_labels)
        n = min(len(pos_labels), len(neg_labels))
        neg_labels = neg_labels[:n]
        pos_labels = pos_labels[:n]

    input_dict = {
        "rho_coords": torch.tensor(rho_wrt_center, dtype=torch.float32),
        "theta_coords": torch.tensor(theta_wrt_center, dtype=torch.float32),
        "input_feat": torch.tensor(input_feat, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.float32),
        "labels": torch.tensor(iface_labels_dc, dtype=torch.int32),
        "pos_idx": torch.tensor(pos_labels, dtype=torch.int32),
        "neg_idx": torch.tensor(neg_labels, dtype=torch.int32),
        "indices_tensor": torch.tensor(indices, dtype=torch.int32),
    }
    return input_dict


from masif_modules.MaSIF_PMP import MaSIF_PMP

def build_model(params):
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
    return model

def train_masif_pmp_kfold(
    model,
    params,
    device,
    batch_size=100,
    num_epochs=100
):

    """
    k-fold CV test

    model: loaded model passed from masif_pmp_train.py
    params: dictionary of hyperparameters

    """

    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    # Open training list.
    training_list = open(params["training_list"]).readlines()
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

            # Training loop
            for pdb_chain_id in train_dirs:
                # load all the preprocessed_data (e.g. input feat, labels, label_indices, mask, indices, etc.)
                mydir = params["masif_precomputation_dir"] + "/" + pdb_chain_id + "/"
                pdbid = pdb_chain_id.split("_")[0]
                chain = pdb_chain_id.split("_")[1]
                pid = 'p1'
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if (
                    len(iface_labels) > 8000
                    or np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    skipped_pdb_list.append(f"{pdb_chain_id} {pid}")
                    continue
                count_proteins += 1

                input_dict = load_data(mydir, pid, params, for_training=True)
                # move input tensors to the same device w/ parameter tensors of the model
                input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

                # Perform training step
                logfile.write("Training on {} {}\n".format(pdb_chain_id, pid))
                print("\nTraining on {} {}\n".format(pdb_chain_id, pid))
                logs = model.training_step(input_dict, optimizer)
                list_training_auc.append(logs["auc"])
                logfile.flush()

            # Validation loop
            for pdb_chain_id in val_dirs:
                # load all the preprocessed_data (e.g. input feat, labels, label_indices, mask, indices, etc.)
                mydir = params["masif_precomputation_dir"] + "/" + pdb_chain_id + "/"
                pdbid = pdb_chain_id.split("_")[0]
                chain = pdb_chain_id.split("_")[1]
                pid = 'p1'
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if (
                    len(iface_labels) > 8000
                    or np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    skipped_pdb_list.append(f"{pdb_chain_id} {pid}")
                    continue
                count_proteins += 1

                input_dict = load_data(mydir, pid, params, for_training=True)
                input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

                logfile.write("Validating on {} {} ==> ".format(pdb_chain_id, pid))
                logs = model.validation_step(input_dict)
                logfile.write("Per protein AUC: {:.4f}\n".format(logs["auc"]))
                list_val_auc.append(logs["auc"])
                all_val_labels.append(iface_labels)
                all_val_scores.append(logs["full_score"])
                logfile.flush()

            # Summary of epoch
            outstr = "Epoch ran on {} proteins\n".format(count_proteins)
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
    # print("\nModel Summary: trainable variables & structure\n")
    # model.count_number_parameters()
    summary(model)

    logfile.close()
