import time
import os
import torch
from torchinfo import summary
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
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



def train_masif_ensemble(
    model,
    params,
    device,
    batch_size=100,
    num_epochs=100,
    optim="adam",
    learning_rate=1e-3,
):

    """
    PyTorch v2 training loop

    model: loaded model passed from masif_site_train.py
    params: dictionary of hyperparameters

    """

    best_test_auc = 0
    # best_val_auc = 0
    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    # Open training list.
    training_list = open(params["training_list"]).readlines()
    training_list = [x.rstrip() for x in training_list]

    testing_list = open(params["testing_list"]).readlines()
    testing_list = [x.rstrip() for x in testing_list]

    data_dirs = os.listdir(params["masif_precomputation_dir"])
    # train, test split among all the precomputed data
    train_dirs = []
    test_dirs = []
    for pdb_id in data_dirs:
        if pdb_id in training_list:
            train_dirs.append(pdb_id)
        elif pdb_id in testing_list:
            test_dirs.append(pdb_id)
        else:
            print("Warning: {} from precomputation dir not in either training or testing list".format(pdb_id))

    # data augmentation setup
    if params["data_augmentation"] == "naive":
        train_dirs = naive_data_augmentation(
            train_dirs, params["masif_precomputation_dir"]
        )       # ['1CX1_A/Center_1', '1CX1_A/Center_6', '1CX1_A/Center_7', '1AOD_A', ...]
    elif params["data_augmentation"] == "group":
        train_dirs = train_dirs
    else:
        raise ValueError("data_augmentation for MaSIF-ensemble should be either 'naive' or 'group'")

    # train, valid split among train set
    np.random.shuffle(train_dirs)
    n_val = int(len(train_dirs) * params["range_val_samples"])
    val_dirs = set(train_dirs[(len(train_dirs) - n_val) :])
    # Sets use hash lookups and hash functions, which makes searching for an item significantly faster compared to lists
    
    # define optimizer outside of the loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Custom training loop
    for epoch in range(num_epochs):
        # Start training epoch:
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
        for aug_ppi_pair_id in train_dirs:
            # load all the preprocessed_data (e.g. input feat, labels, label_indices, mask, indices, etc.)
            mydir = params["masif_precomputation_dir"] + "/" + aug_ppi_pair_id + "/"
            
            # check whether the pdb_chain_id is from ensemble or not and update the pdb_chain_id accordingly
            ppi_pair_id, center_id, ensemble_id = updateID(aug_ppi_pair_id)
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
                    skipped_pdb_list.append(f"{ensemble_id} {pid}")
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
                    "rho_coords": torch.tensor(rho_wrt_center, dtype=torch.float32),
                    "theta_coords": torch.tensor(theta_wrt_center, dtype=torch.float32),
                    "input_feat": torch.tensor(input_feat, dtype=torch.float32),
                    "mask": torch.tensor(mask, dtype=torch.float32),
                    "labels": torch.tensor(iface_labels_dc, dtype=torch.int32),
                    "pos_idx": torch.tensor(pos_labels, dtype=torch.int32),
                    "neg_idx": torch.tensor(neg_labels, dtype=torch.int32),
                    "indices_tensor": torch.tensor(indices, dtype=torch.int32),
                }
                # move input tensors to the same device w/ parameter tensors of the model
                input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}

                # Validation checkpoint
                # search for val_dirs 1st since it's much faster with small search space of val_dirs
                if aug_ppi_pair_id in val_dirs:
                    logfile.write("Validating on {} {}\n".format(ensemble_id, pid))
                    # input_dict["keep_prob"] = 1.0   # not sure of the purpose of keep_prob, remove later if unnecessary

                    print("\nValidating on {} {}\n".format(ensemble_id, pid))
                    logs = model.validation_step(input_dict)
                    list_val_auc.append(logs["auc"])

                # Perform training step
                else:
                    logfile.write("Training on {} {}\n".format(ensemble_id, pid))
                    # input_dict["keep_prob"] = 1.0
                    
                    # Adam optimizer as default, look into Masif_site_wLayers.py later if want to test different opt
                    # learning rate: 1e-3 as default, look into Masif_site_wLayers.py later if want to test different lr
                    print("\nTraining on {} {}\n".format(ensemble_id, pid))
                    logs = model.training_step(input_dict, optimizer)
                    list_training_auc.append(logs["auc"])
                logfile.flush()

        # Run testing cycle
        for ppi_pair_id in test_dirs:
            mydir = params["masif_precomputation_dir"] + "/" + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else: 
                chains2 = ''
            pids = []   # might need to use pids for handling representative confs from dynamic ensemble from unbiased htmd
            if pdbid + "_" + chains1 in testing_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in testing_list:
                pids.append("p2")
            for pid in pids:
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if (
                    len(iface_labels) > 20000
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
                # move input tensors to the same device w/ parameter tensors of the model
                input_dict = {key: tensor.to(device) for key, tensor in input_dict.items()}
                
                logfile.write("Testing on {} {}\n".format(ppi_pair_id, pid))
                print("\nTesting on {} {}\n".format(ppi_pair_id, pid))
                logs = model.test_step(input_dict)
                list_test_auc.append(logs["auc"])
                list_test_names.append((ppi_pair_id, pid))
                all_test_labels.append(iface_labels)
                all_test_scores.append(logs["full_score"])
                logfile.flush()

        # Summary of epoch
        outstr = "Epoch ran on {} proteins\n".format(count_proteins)
        outstr += "{} proteins skipped to prevent biased fitting: {}\n\n".format(
            len(skipped_pdb_list), skipped_pdb_list
        )
        ## per protein metrics
        outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_training_auc), np.median(list_training_auc), epoch +1
        )
        outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_val_auc), np.median(list_val_auc), epoch +1
        )
        outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for epoch {}\n".format(
            np.mean(list_test_auc), np.median(list_test_auc), epoch +1
        )
        ## all points metrics (test)
        flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
        flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
        outstr += "Testing auc (all points): {:.2f}\n".format(
            metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
        )
        outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
        logfile.write(outstr + "\n")
        print(outstr)

        # save model (check whether to save model based on val_auc or test_auc, default is val_auc)
        if np.mean(list_test_auc) > best_test_auc:
        # if np.mean(list_val_auc) > best_val_auc:
            logfile.write(">>> Saving model.\n")
            print(">>> Saving model.\n")
            best_test_auc = np.mean(list_test_auc)
            # best_val_auc = np.mean(list_val_auc)
            output_model = out_dir + "model.pt"
            torch.save(model.state_dict(), output_model)
            # Save the scores for test.
            all_test_labels = np.array(all_test_labels, dtype=object)  # Explicitly change the dtype to object
            all_test_scores = np.array(all_test_scores, dtype=object)
            np.save(out_dir + "test_labels.npy", all_test_labels)
            np.save(out_dir + "test_scores.npy", all_test_scores)
            np.save(out_dir + "test_names.npy", list_test_names)

    # Display the model's architecture
    print("\nModel Summary: trainable variables & structure\n")
    model.count_number_parameters()
    summary(model)

    logfile.close()
