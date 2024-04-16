import time
import os
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score
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


def train_masif_site_kfold(
    params,
    batch_size=100,
    num_epochs=100
):

    """
    k-fold CV test

    model: loaded model passed from masif_site_train_tf2.py
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
    kfold = KFold(params["k_fold"], True, 1)    # random_state=1 --> for reproducibility (same splits each time)
    split_count = 0

    # enumerate splits
    for train_idx, test_idx in kfold.split(training_list):  # train, test: array of indices for split data samples
        split_count += 1
        
        logfile.write("\nStarting split {}\n".format(split_count))
        print("\nStarting split {}\n".format(split_count))
        
        # for debug
        # print(f"train: {train_idx}\ntest: {test_idx}\ntrain_num: {len(train_idx)}\ntest_num: {len(test_idx)}")
        # logfile.write('train: {}, test: {}\ntrain_num: {}, test_num: {}\n'.format(
        #     training_list[train_idx], training_list[test_idx], len(train_idx), len(test_idx)
        #     )
        # )
        print('train: {}\ntest: {}\ntrain_num: {}, test_num: {}\n'.format(
            training_list[train_idx], training_list[test_idx], len(train_idx), len(test_idx)
            )
        )
        train_dirs = training_list[train_idx]
        val_dirs = training_list[test_idx]

        # Build new neural network model for each split
        from masif_modules.MaSIF_site import MaSIF_site

        if "n_theta" in params:
            model = MaSIF_site(
                max_rho=params["max_distance"],
                n_thetas=params["n_theta"],
                n_rhos=params["n_rho"],
                n_rotations=params["n_rotations"],
                idx_gpu="/device:GPU:0",
                feat_mask=params["feat_mask"],
                n_conv_layers=params["n_conv_layers"],
            )
        else:
            model = MaSIF_site(
                max_rho=params["max_distance"],
                n_thetas=4,
                n_rhos=3,
                n_rotations=4,
                idx_gpu="/device:GPU:0",
                feat_mask=params["feat_mask"],
                n_conv_layers=params["n_conv_layers"],
            )

        # Custom training loop
        for epoch in range(num_epochs):
            # Start training epoch:
            count_proteins = 0
            skipped_pdb_list = []

            # list_training_loss = []     # loss shape = [batch_size, 2] --> list_loss cannot be averaged directly
            list_training_auc = []
            list_val_auc = []

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
                        "rho_coords": rho_wrt_center,
                        "theta_coords": theta_wrt_center,
                        "input_feat": input_feat,
                        "mask": mask,
                        "labels": iface_labels_dc,
                        "pos_idx": pos_labels,
                        "neg_idx": neg_labels,
                        "indices_tensor": indices,
                    }

                    # Perform training step
                    logfile.write("Training on {} {}\n".format(ppi_pair_id, pid))
                    input_dict["keep_prob"] = 1.0

                    logs = model.train_step(
                        input_dict,
                        optimizer_method="Adam",
                        learning_rate=1e-3
                    )

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
                        "rho_coords": rho_wrt_center,
                        "theta_coords": theta_wrt_center,
                        "input_feat": input_feat,
                        "mask": mask,
                        "labels": iface_labels_dc,
                        "pos_idx": pos_labels,
                        "neg_idx": neg_labels,
                        "indices_tensor": indices,
                    }

                    logfile.write("Validating on {} {}\n".format(ppi_pair_id, pid))
                    input_dict["keep_prob"] = 1.0   # not sure of the purpose of keep_prob, remove later if unnecessary

                    logs = model.test_step(input_dict)
                    
                    list_val_auc.append(logs["auc"])
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
                logfile.write(">>> Split {:d} CV done:\n>>> Per protein AUC mean (training): {:.4f}; median: {:.4f}\n>>> Per protein AUC mean (validation): {:.4f}; median: {:.4f}\n",format(
                    split_count, np.mean(list_training_auc), np.median(list_training_auc), np.mean(list_val_auc), np.median(list_val_auc)
                    )
                )
                print(">>> Split {:d} CV done:\n>>> Per protein AUC mean (training): {:.4f}; median: {:.4f}\n>>> Per protein AUC mean (validation): {:.4f}; median: {:.4f}\n",format(
                    split_count, np.mean(list_training_auc), np.median(list_training_auc), np.mean(list_val_auc), np.median(list_val_auc)
                    )
                )

    # Display the model's architecture: built-in model.summary() for functional API models
    print("\nModel Summary: trainable variables & structure\n")
    model.count_number_parameters()

    logfile.close()
