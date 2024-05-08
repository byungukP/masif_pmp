import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
# import keras
import numpy as np

"""
Layers used for the MaSIF neural network model. PyTorch version
ByungUk Park UW-Madison, 2024
"""


class SoftGrid(L.LightningModule):

    """
    Gaussian Kernel layer (GDL layer)

    - define learnable soft grid w/ n_rho * n_theta gaussians kernels defined in a local geodesic polar system
    - the parameters of the Gaussians are learnable on their own
    - learnable Gaussian kernels locally average the vertex-wise patch features (acting as soft pixels)
    - map input features to soft grid by multiplying Gaussian-like functions to feature values
    - thus, input features --> gaussian descriptors via local averaging of vertex-wise patch features

    """

    def __init__(
            self,
            n_thetas,
            n_rhos,
            mu_rho_initial,
            sigma_rho_init,
            mu_theta_initial,
            sigma_theta_init,
            n_feat,
            name,           # name='l1_0' or 'l2'
            **kwargs
    ):
        super(SoftGrid, self).__init__()
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_gauss = n_thetas * n_rhos
        self.n_feat = n_feat

        # mu, sigma for rho, theta
        self.mu_rho = nn.Parameter(
            torch.tensor(mu_rho_initial, dtype=torch.float32),
            requires_grad=True,
        )
        #     name="mu_rho_{}".format(name)  # var_name='mu_rho_l1_0' or 'mu_rho_l2'
        # )   # 1, n_gauss
        self.sigma_rho = nn.Parameter(
            torch.ones_like(self.mu_rho) * sigma_rho_init,
            requires_grad=True,
        )
        #     name="sigma_rho_{}".format(name)
        # )   # 1, n_gauss
        self.mu_theta = nn.Parameter(
            torch.tensor(mu_theta_initial, dtype=torch.float32),
            requires_grad=True,
        )
        #     name="mu_theta_{}".format(name)
        # )   # 1, n_gauss
        self.sigma_theta = nn.Parameter(
            torch.ones_like(self.mu_theta) * sigma_theta_init,
            requires_grad=True,
        )
        #     name="sigma_theta_{}".format(name)
        # )   # 1, n_gauss

        # weights for convolution
        self.W = nn.Parameter(
            torch.empty(
                (self.n_thetas * self.n_rhos * self.n_feat,
                 self.n_thetas * self.n_rhos * self.n_feat)
            ),
            requires_grad=True,
        )
        #     name="W_conv_{}".format(name)
        # )
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(
            torch.zeros(
                self.n_thetas * self.n_rhos * self.n_feat
            ),
            requires_grad=True,
        )
        #     name="b_conv_{}".format(name)
        # )

    def forward(
            self,
            input_feat,
            rho_coords,
            theta_coords,
            mask,
            eps=1e-5,
            mean_gauss_fxns=True
    ):
        # define the forward pass in the layer
        batch_size = rho_coords.shape[0]
        n_vertices = rho_coords.shape[1]
        n_feat = input_feat.shape[2]

        # soft grid mapping w/ rotation_replicas
        all_conv_desc = []
        n_rotations = self.n_rhos
        for k in range(n_rotations):
            print("rotation replica: ", k)
            rho_coords_ = torch.reshape(rho_coords, (-1, 1))  # batch_size*n_vertices_in_patch
            thetas_coords_ = torch.reshape(theta_coords, (-1, 1))  # batch_size*n_vertices_in_patch

            thetas_coords_ += k * 2 * np.pi / n_rotations
            thetas_coords_ = torch.fmod(thetas_coords_, 2 * np.pi)

            # turning rho_, theta_coords into corresponding probabilities in gaussian distribution function
            rho_coords_ = torch.exp(
                -torch.square(rho_coords_ - self.mu_rho) / (torch.square(self.sigma_rho) + eps)
            )
            thetas_coords_ = torch.exp(
                -torch.square(thetas_coords_ - self.mu_theta) / (torch.square(self.sigma_theta) + eps)
            )

            """
            if NaN first appears after gaussian exp prob calculation of rho coords,
            we can put either 1) simply ignore the input pdb_id and continue with the next pdb_id
            OR 2) put the upper & lower bounds on the input to prevent the NaN values
            """
            if torch.isnan(rho_coords_).any():
                print("nan value appears after gaussian exp prob calculation of rho coords")
            else:
                print("no nan value after gaussian exp prob calculation of rho coords")
            if torch.isnan(thetas_coords_).any():
                print("nan value appears after gaussian exp prob calculation of theta coords")
            else:
                print("no nan value after gaussian exp prob calculation of theta coords")

            # map to soft grid (gaussian descriptors)
            gauss_fxns = torch.mul(
                rho_coords_, thetas_coords_
            ) # batch_size*n_vertices, n_gauss (n_gauss = n_thetas*n_rhos)

            if torch.isnan(gauss_fxns).any():
                print("nan value appears after gaussian exp prob multiplication")
            else:
                print("no nan value after gaussian exp prob multiplication")

            gauss_fxns = torch.reshape(
                gauss_fxns, (batch_size, n_vertices, -1)
            )  # batch_size, n_vertices, n_gauss
            gauss_fxns = torch.mul(
                gauss_fxns, mask
            )

            if torch.isnan(gauss_fxns).any():
                print("nan value appears after gaussian fxn * mask multiplication")
            else:
                print("no nan value after gaussian fxn * mask multiplication")

            if mean_gauss_fxns:
            # computes mean weights for the different gaussians = gaussian-wise(OR local) averaging
            # --> sum( probability fxns of all vertices in a patch wrt any Gaussian) = 1
                gauss_fxns /= (
                    torch.sum(gauss_fxns, 1, keepdim=True) + eps
                )  # batch_size, n_vertices, n_gauss

            if torch.isnan(gauss_fxns).any():
                print("nan value appears after mean gauss fxns")
            else:
                print("no nan value after mean gauss fxns")

            gauss_fxns = gauss_fxns.unsqueeze(2)  # batch_size, n_vertices, 1, n_gauss,
            print("gauss_fxns shape: ", gauss_fxns.shape)
            if torch.isnan(gauss_fxns).any():
                print("nan value appears after gauss_fxns.unsqueeze(2)")
            else:
                print("no nan value after gauss_fxns.unsqueeze(2)")

            input_feat_ = input_feat.unsqueeze(3)  # batch_size, n_vertices, n_feat, 1
            print("input_feat_ shape: ", input_feat_.shape)
            if torch.isnan(input_feat_).any():
                print("nan value appears after input_feat_.unsqueeze(3)")
            else:
                print("no nan value after input_feat_.unsqueeze(3)")

            # gaussian descriptors: gaussian kernels w/ probability weights locally (=gaussian-wise) average the vertex-wise patch features (by torch.multiply(gauss_fxns, input_feat_), thus acting as soft pixels)
            gauss_desc = torch.mul(
                gauss_fxns, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            print("gauss_desc shape: ", gauss_desc.shape)

            if torch.isnan(gauss_desc).any():
                print("nan value appears after gauss fxn * input_feat_ multiplication")
            else:
                print("no nan value after gauss fxn * input_feat_ multiplication")

            gauss_desc = torch.sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss, (=abstract out vertices factor)
            gauss_desc = torch.reshape(
                gauss_desc, (batch_size, self.n_gauss * n_feat)
            )  # batch_size, self.n_thetas*self.n_rhos*n_feat

            # convolution
            conv_desc = torch.matmul(gauss_desc, self.W) + self.b  # batch_size, self.n_thetas*self.n_rhos*n_feat

            if torch.isnan(conv_desc).any():
                print("nan value appears after convolution")
            else:
                print("no nan value after convolution")

            all_conv_desc.append(conv_desc)

        # (gaussian-wise) angular max pooling
        all_conv_desc = torch.stack(all_conv_desc)  # n_rotations, batch_size, self.n_thetas*self.n_rhos*n_feat
        conv_desc = torch.max(all_conv_desc, 0)[0]  # batch_size, self.n_thetas*self.n_rhos*n_feat
        conv_desc = nn.ReLU()(conv_desc)

        if torch.isnan(conv_desc).any():
            print("nan value appears after ReLU activation")
        else:
            print("no nan value after ReLU activation")
        return conv_desc


class Init_MLPBlock(L.LightningModule):
    """
    inputs: global_desc, (batch_size, n_gauss*n_feat)
    output: global_desc, (batch_size, n_feat)
    """
    def __init__(self, n_thetas, n_rhos, n_feat):
        super().__init__()
        self.FC12 = nn.Linear(n_thetas * n_rhos * n_feat, n_thetas * n_rhos)
        self.FC5 = nn.Linear(n_thetas * n_rhos, n_feat)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        desc = self.relu(self.FC12(inputs))
        return self.relu(self.FC5(desc))

class Final_MLPBlock(L.LightningModule):
    """
    inputs: global_desc, (batch_size, n_gauss*n_feat)
    output: logits, (batch_size, n_labels)
    """
    def __init__(self, n_thetas, n_feat, n_labels):
        super().__init__()
        self.FC4 = nn.Linear(n_feat, n_thetas)
        self.FC2 = nn.Linear(n_thetas, n_labels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        desc = self.relu(self.FC4(inputs))
        return self.FC2(desc)

