import time
import os
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score

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


# # Run masif site on a protein, on a previously trained network.
# def run_masif_site(
#     params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices
# ):
#     indices = pad_indices(indices, mask.shape[1])
#     mask = np.expand_dims(mask, 2)
#     feed_dict = {
#         learning_obj.rho_coords: rho_wrt_center,
#         learning_obj.theta_coords: theta_wrt_center,
#         learning_obj.input_feat: input_feat,
#         learning_obj.mask: mask,
#         learning_obj.indices_tensor: indices,
#     }

#     score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
#     return score


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


def train_masif_site(
    model,
    params,
    batch_size=100,
    num_epochs=100
):

    """
    tf v2 training loop

    model: loaded model passed from masif_site_train_tf2.py
    params: dictionary of hyperparameters

    """

    best_val_auc = 0
    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    # Open training list.
    training_list = open(params["training_list"]).readlines()
    training_list = [x.rstrip() for x in training_list]

    testing_list = open(params["testing_list"]).readlines()
    testing_list = [x.rstrip() for x in testing_list]

    """
    update this part for CV test later
    below: validation split among train&test set
    """   
    data_dirs = os.listdir(params["masif_precomputation_dir"])
    # train, test split among all the precomputed data
    train_dirs = []
    test_dirs = []
    for pdb_id in data_dirs:
        if pdb_id in training_list:
            train_dirs.append(pdb_id)
        else:
            test_dirs.append(pdb_id)

    # train, valid split among train set
    np.random.shuffle(train_dirs)
    n_val = int(len(train_dirs) * params["range_val_samples"])
    val_dirs = set(train_dirs[(len(train_dirs) - n_val) :])
    # Sets use hash lookups and hash functions, which makes searching for an item significantly faster compared to lists
    
    # for check
    # print(f"train_dirs: {len(train_dirs)}, {train_dirs}")
    # print(f"val_dirs: {len(val_dirs)}, {val_dirs}")

    # Custom training loop
    for epoch in range(num_epochs):
        # Start training epoch:
        
        """
        train loss, auc, acc --> defined in __init__() & train_set()
        in MaSIF_site.py model

        validation loss, auc, acc --> defined in __init__() & test_set()
        in MaSIF_site.py model

        test loss, auc, acc --> same with valid test_set()
        
        ==> still need to save each metric of proteins (batches) as list
            to track the epoch-series evolution of each metric

        """

        count_proteins = 0
        skipped_pdb_list = []

        # list_training_loss = []     # loss shape = [batch_size, 2] --> list_loss cannot be averaged directly
        list_training_acc = []
        list_training_precision = []
        list_training_recall = []
        list_training_auc = []

        list_val_acc = []
        list_val_precision = []
        list_val_recall = []
        list_val_auc = []

        logfile.write("\nStarting epoch {}\n".format(epoch +1))
        print("\nStarting epoch {}\n".format(epoch + 1))
        tic = time.time()
        all_training_labels = []
        all_training_scores = []
        all_val_labels = []
        all_val_scores = []
        all_test_labels = []
        all_test_scores = []

        list_test_auc = []
        list_test_names = []
        list_test_acc = []
        all_test_labels = []
        all_test_scores = []
        
        """
        shuffling the training set before each epoch is a common practice to ensure
        that the model generalizes well to unseen data and is not biased by the
        sequence of training examples.
        however, in this case, the training set is shuffled at the beginning of the training
        and the order of the training examples is not changed during the training
        to put validation set at the last section in order.
        """
        # np.random.shuffle(train_dirs)
        
        # train/valid loop: since each protein as batch
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
                # indice_lists: array of lists, each list representing each patch with indices of vertices that are encompassed in the patch
                # each indice list of patches starts with index of the center, then vertices inside the patch
                # max number of indices for patch <= custom_params["max_shape_size"] (default=100)
                # since some patches may have less than max shape size (100) for patch representation but all the matrix are generated based on max shape size for computational reason,
                # need padding for matrix of indice list --> pad_indices()
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
                """
                skipping if for n_conv_layers == 1
                check original source code if have to test w/ n_conv_layers == 1
                """
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

                # Validation checkpoint
                # search for val_dirs 1st since it's much faster with small search space of val_dirs
                if ppi_pair_id in val_dirs:
                    logfile.write("Validating on {} {}\n".format(ppi_pair_id, pid))
                    input_dict["keep_prob"] = 1.0   # not sure of the purpose of keep_prob, remove later if unnecessary
                    
                    """
                    validation step
                    For 5-CV --> maybe write new script using KFold() from sklearn.model_selection
                    """
                    logs = model.test_step(input_dict)
                    acc, precision, recall, auc = logs["binary_accuracy"], logs["precision"], logs["recall"], logs["auc"]
                    list_val_acc.append(acc)
                    list_val_precision.append(precision)
                    list_val_recall.append(recall)
                    list_val_auc.append(auc)

                # Perform training step
                else:
                    logfile.write("Training on {} {}\n".format(ppi_pair_id, pid))
                    input_dict["keep_prob"] = 1.0
                    
                    logs = model.train_step(
                        input_dict,
                        optimizer_method="Adam",
                        learning_rate=1e-3
                    )

                    # auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    acc, precision, recall, auc = logs["binary_accuracy"], logs["precision"], logs["recall"], logs["auc"]
                    list_training_acc.append(acc)
                    list_training_precision.append(precision)
                    list_training_recall.append(recall)
                    list_training_auc.append(auc)
                logfile.flush()

        # Run testing cycle. --> not in this code, but might be added later
        # focus on running 5-CV with this code, prediction test w/ predict_site.sh later

        # Summary of epoch
        outstr = "Epoch ran on {} proteins\n".format(count_proteins)
        outstr += "{} proteins skipped to prevent biased fitting: {}\n\n".format(
            len(skipped_pdb_list), skipped_pdb_list
        )
        ## per protein metrics (training)
        outstr += "Per protein Accuracy mean (training): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_training_acc), np.median(list_training_acc), epoch +1
        )
        outstr += "Per protein Precision mean (training): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_training_precision), np.median(list_training_precision), epoch +1
        )
        outstr += "Per protein Recall mean (training): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_training_recall), np.median(list_training_recall), epoch +1
        )
        outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_training_auc), np.median(list_training_auc), epoch +1
        )
        ## per protein metrics (validation)
        outstr += "Per protein Accuracy mean (validation): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_val_acc), np.median(list_val_acc), epoch +1
        )
        outstr += "Per protein Precision mean (validation): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_val_precision), np.median(list_val_precision), epoch +1
        )
        outstr += "Per protein Recall mean (validation): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_val_recall), np.median(list_val_recall), epoch +1
        )
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

        # save model: save_weights() more efficient & flexible
        if np.mean(list_val_auc) > best_val_auc:
            logfile.write(">>> Saving model.\n")
            print(">>> Saving model.\n")
            best_val_auc = np.mean(list_val_auc)
            output_model = out_dir + "model.weights.h5"   # model.keras format: for functional API models
            model.save_weights(output_model, overwrite=True)
            # # Save the scores for test.
            # np.save(out_dir + "test_labels.npy", all_test_labels)
            # np.save(out_dir + "test_scores.npy", all_test_scores)
            # np.save(out_dir + "test_names.npy", list_test_names)

    # Display the model's architecture: built-in model.summary() for functional API models
    print("\nModel Summary: trainable variables & structure\n")
    model.count_number_parameters()

    logfile.close()
