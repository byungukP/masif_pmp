import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional import auroc
import numpy as np
# from sklearn import metrics
from masif_modules.masif_layers import SoftGrid, Init_MLPBlock, Final_MLPBlock


class MaSIF_PMP(L.LightningModule):

    """
    The neural network model. PyTorch v2.1.2
    ByungUk Park, UW-Madison, 2025

    """

    def count_number_parameters(self):
        total_parameters = 0
        for variable in self.parameters():
            variable_parameters = torch.numel(variable)
            print(f"<Parameters w/ shape={variable.shape}, {variable.dtype}>")
            total_parameters += variable_parameters
        print("Total number parameters: %d" % total_parameters)

    def frobenius_norm(self, tensor):
        square_tensor = torch.square(tensor)
        tensor_sum = torch.sum(square_tensor)
        frobenius_norm = torch.sqrt(tensor_sum)
        return frobenius_norm

    def compute_initial_coordinates(self):
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)
        grid_rho = grid_rho[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1]

        # Return a list of coordinate matrices from coordinate vectors, shape = (n_theta, n_rho), ex) (16, 5)
        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        # the traspose here is needed to have the same behaviour as Matlab code --> shape = (n_rho, n_theta)
        grid_rho_ = grid_rho_.T
        grid_theta_ = grid_theta_.T
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection
        print("initial polar coords shape: {}".format(coords.shape))
        return coords

    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_gamma=1.0,
        learning_rate=1e-3,
        n_rotations=16,
        feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
        n_conv_layers=1,
        optimizer_method="Adam",
        name="MaSIF_PMP",
        **kwargs
    ):
        """
        Initialize the MaSIF_PMP object.

        Parameters:
        - max_rho (float): The maximum value of the radial coordinate.
        - n_thetas (int): The number of theta values in the polar grid.
        - n_rhos (int): The number of rho values in the polar grid.
        - n_gamma (float): The value of gamma.
        - learning_rate (float): The learning rate for optimization.
        - n_rotations (int): The number of rotations for data augmentation.
        - idx_gpu (str): The index of the GPU to use.
        - feat_mask (list): The mask for selecting surface features.
        - n_conv_layers (int): The number of geometric convolutional layers.
        - optimizer_method (str): The optimization method to use.

        """
        super(MaSIF_PMP, self).__init__()

        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.learning_rate = learning_rate

        self.sigma_rho_init = max_rho / 8  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_labels = 2
        self.n_conv_layers = n_conv_layers

        # enabling mixed precision (AMP)
        self.scaler = GradScaler()

        torch.manual_seed(0)
        
        initial_coords = self.compute_initial_coordinates()
        mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype("float32")
        mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype("float32")

        # Gaussian Kernels for layer 1
        self.soft_grid_feat_l1 = nn.ModuleList()

        for i in range(self.n_feat):
            self.soft_grid_feat_l1.append(
                SoftGrid(
                    n_thetas,
                    n_rhos,
                    mu_rho_initial,
                    self.sigma_rho_init,
                    mu_theta_initial,
                    self.sigma_theta_init,
                    n_feat=1,
                    name="l1_feat{}".format(i),     # name='l1_0' or 'l2'
                )
            )

        # init_MLP = FC12, FC5
        self.init_MLPBlock = Init_MLPBlock(self.n_thetas, self.n_rhos, self.n_feat)

        # Gaussian Kernels for additional layers
        # single soft grid for all the surf feat in additional convolutional layers
        if n_conv_layers > 1:
            self.soft_grid_l2 = SoftGrid(
                n_thetas,
                n_rhos,
                mu_rho_initial,
                self.sigma_rho_init,
                mu_theta_initial,
                self.sigma_theta_init,
                self.n_feat,
                name="l2",
            )
        if n_conv_layers > 2:
            self.soft_grid_l3 = SoftGrid(
                n_thetas,
                n_rhos,
                mu_rho_initial,
                self.sigma_rho_init,
                mu_theta_initial,
                self.sigma_theta_init,
                self.n_feat,
                name="l3",
            )
        if n_conv_layers > 3:
            self.soft_grid_l4 = SoftGrid(
                n_thetas,
                n_rhos,
                mu_rho_initial,
                self.sigma_rho_init,
                mu_theta_initial,
                self.sigma_theta_init,
                self.n_feat,
                name="l4",
            )

        if n_conv_layers > 4:
            self.soft_grid_l5 = SoftGrid(
                n_thetas,
                n_rhos,
                mu_rho_initial,
                self.sigma_rho_init,
                mu_theta_initial,
                self.sigma_theta_init,
                self.n_feat,
                name="l5",
            )

        if n_conv_layers > 5:
            self.soft_grid_l6 = SoftGrid(
                n_thetas,
                n_rhos,
                mu_rho_initial,
                self.sigma_rho_init,
                mu_theta_initial,
                self.sigma_theta_init,
                self.n_feat,
                name="l6",
            )

        if n_conv_layers > 6:
            self.soft_grid_l7 = SoftGrid(
                n_thetas,
                n_rhos,
                mu_rho_initial,
                self.sigma_rho_init,
                mu_theta_initial,
                self.sigma_theta_init,
                self.n_feat,
                name="l7",
            )

        # final_MLP = FC4, FC2
        self.final_MLPBlock = Final_MLPBlock(self.n_thetas, self.n_feat, self.n_labels)


    def forward(self, input_dict):
        # Define the forward pass

        self.rho_coords = input_dict["rho_coords"]          # batch_size, n_vertices, 1
        self.theta_coords = input_dict["theta_coords"]      # batch_size, n_vertices, 1
        self.input_feat = input_dict["input_feat"]          # batch_size, n_vertices, n_feat
        self.mask = input_dict["mask"]                      # batch_size, n_vertices, 1
        self.pos_idx = input_dict["pos_idx"]                # batch_size/2
        self.neg_idx = input_dict["neg_idx"]                # batch_size/2
        self.labels = input_dict["labels"]                  # batch_size, n_labels
        self.indices_tensor = input_dict["indices_tensor"]  # batch_size, max_verts (< 30)

        global_desc = []

        # Use Geometric deep learning

        # 1st GDL layer: surf feat-wise convolution
        for i in range(self.n_feat):
            my_input_feat = self.input_feat[:, :, i].unsqueeze(2)

            global_desc.append(
                self.soft_grid_feat_l1[i](
                    my_input_feat,
                    self.rho_coords,
                    self.theta_coords,
                    self.mask
                )   # batch_size, n_gauss*1
            )   # n_feat, batch_size, n_gauss*1

        # global_desc should be batch_size, n_gauss*n_feat (12 x 5)
        global_desc = torch.stack(global_desc, axis=1)  # batch_size, n_feat, n_gauss*1
        global_desc = torch.reshape(
            global_desc, [-1, self.n_thetas * self.n_rhos * self.n_feat]
        )

        # init_MLP = FC12 (n_thease * n_rhos), FC5 (n_feat)
        global_desc = self.init_MLPBlock(global_desc)

        # additional GDL layers: simple convolutions
        # second convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 1:           
            # Rebuild a patch based on the output of the first layer
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat

            global_desc = self.soft_grid_l2(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat

            batch_size = global_desc.shape[0]
            # Reduce the dimensionality by averaging over the last dimension
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)

        # third convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 2:
            # Rebuild a patch based on the output of the second layer
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat

            global_desc = self.soft_grid_l3(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            batch_size = global_desc.shape[0]
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)

        # fourth convolutional layer. input: batch_size, n_gauss, output: batch_size, n_gauss
        if self.n_conv_layers > 3:
            # Rebuild a patch based on the output of the third layer
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat

            global_desc = self.soft_grid_l4(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            batch_size = global_desc.shape[0]
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)

        # conv_l5
        if self.n_conv_layers > 4:
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat
            global_desc = self.soft_grid_l5(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            batch_size = global_desc.shape[0]
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)        
        # conv_l6
        if self.n_conv_layers > 5:
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat
            global_desc = self.soft_grid_l6(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            batch_size = global_desc.shape[0]
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)
        # conv_l7
        if self.n_conv_layers > 6:
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat
            global_desc = self.soft_grid_l7(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            batch_size = global_desc.shape[0]
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2) 

        # Additional SoftGrid layers for transfer learning
        if hasattr(self, "tflr_soft_grid"):
            for i, softgrid in enumerate(self.tflr_soft_grid):
                global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat

                global_desc = softgrid(
                    global_desc,
                    self.rho_coords,
                    self.theta_coords,
                    self.mask
                )   # batch_size, n_gauss*n_feat

                batch_size = global_desc.shape[0]
                global_desc = torch.reshape(
                    global_desc,
                    [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
                )
                global_desc = torch.mean(global_desc, dim=2)

        # refine global desc with MLP
        # final_MLP = FC4, FC2
        logits = self.final_MLPBlock(global_desc)
        return logits

    # definition of the optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


    def training_step(
        self,
        input_dict,
        optimizer
    ):
        optimizer.zero_grad()

        # Forward + loss inside autocast context
        # Forward pass (self() ~ self.forward())
        with autocast():
            logits = self(input_dict)
            eval_labels = torch.cat(
                [
                    self.labels[self.pos_idx],
                    self.labels[self.neg_idx],
                ],
                dim=0,
            )   # 2*pos_idx(=neg_idx), n_labels

            eval_logits = torch.cat(
                [
                    logits[self.pos_idx],
                    logits[self.neg_idx],
                ],
                dim=0,
            )   # 2*pos_idx(=neg_idx), n_labels

            # Compute the loss
            loss = F.binary_cross_entropy_with_logits(
                eval_logits, eval_labels.float()
            )
        # Backward and optimizer step using GradScaler
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        # eval_logits and eval_scores are reordered according to pos and neg_idx.
        eval_logits = torch.sigmoid(eval_logits)
        eval_score = torch.squeeze(eval_logits)[:, 0]   # 2*pos_idx(=neg_idx),

        full_logits = torch.sigmoid(logits)
        full_score = torch.squeeze(full_logits)[:, 0]   # batch_size(=total mesh vertices num),
        
        # Update metrics
        auc = auroc(
            eval_score,
            eval_labels[:, 0].long(),
            task="binary"
        )

        return {
                    "loss": loss.item(),
                    "eval_score": eval_score.detach().cpu().numpy(),
                    "full_score": full_score.detach().cpu().numpy(),
                    "auc": auc.item()
                }

    def validation_step(self, input_dict):
        return self._shared_eval(input_dict)

    def test_step(self, input_dict):
        return self._shared_eval(input_dict)

    def _shared_eval(self, input_dict):
        logits = self(input_dict)
        eval_labels = torch.cat(
            [
                self.labels[self.pos_idx],
                self.labels[self.neg_idx],
            ],
            dim=0,
        )
        eval_logits = torch.cat(
            [
                logits[self.pos_idx],
                logits[self.neg_idx],
            ],
            dim=0,
        )
        # Compute the loss
        loss = F.binary_cross_entropy_with_logits(
            eval_logits, eval_labels.float()
        )
        # eval_logits and eval_scores are reordered according to pos and neg_idx.
        eval_logits = torch.sigmoid(eval_logits)
        eval_score = torch.squeeze(eval_logits)[:, 0]                            

        full_logits = torch.sigmoid(logits)
        full_score = torch.squeeze(full_logits)[:, 0]
        
        # Update metrics
        auc = auroc(
            eval_score,
            eval_labels[:, 0].long(),
            task="binary"
        )

        return {
                    "loss": loss.item(),
                    "eval_score": eval_score.detach().cpu().numpy(),
                    "full_score": full_score.detach().cpu().numpy(),
                    "auc": auc.item()
                }

    def gen_FPVec(self, input_dict):
        # save surface fingerprint vectors generated during the forward pass

        self.rho_coords = input_dict["rho_coords"]          # batch_size, n_vertices, 1
        self.theta_coords = input_dict["theta_coords"]      # batch_size, n_vertices, 1
        self.input_feat = input_dict["input_feat"]          # batch_size, n_vertices, n_feat
        self.mask = input_dict["mask"]                      # batch_size, n_vertices, 1
        self.pos_idx = input_dict["pos_idx"]                # batch_size/2
        self.neg_idx = input_dict["neg_idx"]                # batch_size/2
        self.labels = input_dict["labels"]                  # batch_size, n_labels
        self.indices_tensor = input_dict["indices_tensor"]  # batch_size, max_verts (< 30)

        global_desc = []
        fpvec_list = []         # list of fingerprint vectors (n_gauss*n_feat-dim)
        feat_vec_list = []      # list of n_feature vectors (n_feat-dim)

        # 1st GDL layer: surf feat-wise convolution
        for i in range(self.n_feat):
            my_input_feat = self.input_feat[:, :, i].unsqueeze(2)

            global_desc.append(
                self.soft_grid_feat_l1[i](
                    my_input_feat,
                    self.rho_coords,
                    self.theta_coords,
                    self.mask
                )   # batch_size, n_gauss*1
            )   # n_feat, batch_size, n_gauss*1

        global_desc = torch.stack(global_desc, axis=1)  # batch_size, n_feat, n_gauss*1
        global_desc = torch.reshape(
            global_desc, [-1, self.n_thetas * self.n_rhos * self.n_feat]
        )   # 1st 60-D fingerprint vec
        fpvec_list.append(global_desc)

        # init_MLP = FC12 (n_thease * n_rhos), FC5 (n_feat)
        global_desc = self.init_MLPBlock(global_desc)   # 1st 5-D feat vec
        feat_vec_list.append(global_desc)

        # additional GDL layers: simple convolutions
        # second convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 1:
            # Rebuild a patch based on the output of the first layer
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat

            # 2nd 60-D fingerprint vec
            global_desc = self.soft_grid_l2(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            fpvec_list.append(global_desc)

            batch_size = global_desc.shape[0]
            # Reduce the dimensionality by averaging over the last dimension
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)    # 2nd 5-D fingerprint vec
            feat_vec_list.append(global_desc)

        # third convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 2:
            # Rebuild a patch based on the output of the first layer
            global_desc = global_desc[self.indices_tensor]  # batch_size, max_verts, n_feat

            # 3rd 60-D fingerprint vec
            global_desc = self.soft_grid_l3(
                global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            fpvec_list.append(global_desc)

            batch_size = global_desc.shape[0]
            global_desc = torch.reshape(
                global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            global_desc = torch.mean(global_desc, dim=2)   # 3rd 5-D fingerprint vec
            feat_vec_list.append(global_desc)

        return fpvec_list, feat_vec_list

