import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np

"""
Layers used for the MaSIF neural network model. PyTorch version
ByungUk Park UW-Madison, 2025
"""


class SoftGrid(L.LightningModule):

    """
    Soft Grid layer (system of learnable Gaussian kernels)

    - define learnable soft grid w/ n_rho * n_theta gaussians kernels defined in a local geodesic polar system
    - learnable Gaussian kernels locally average the vertex-wise patch features (acting as soft pixels)
    - map input features to soft grid by multiplying Gaussian-like functions to feature values

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
        self.sigma_rho = nn.Parameter(
            torch.ones_like(self.mu_rho) * sigma_rho_init,
            requires_grad=True,
        )
        self.mu_theta = nn.Parameter(
            torch.tensor(mu_theta_initial, dtype=torch.float32),
            requires_grad=True,
        )
        self.sigma_theta = nn.Parameter(
            torch.ones_like(self.mu_theta) * sigma_theta_init,
            requires_grad=True,
        )

        # weights for convolution
        self.W = nn.Parameter(
            torch.empty(
                (self.n_thetas * self.n_rhos * self.n_feat,
                 self.n_thetas * self.n_rhos * self.n_feat),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(
            torch.zeros(
                self.n_thetas * self.n_rhos * self.n_feat,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        # self.ln = nn.LayerNorm(self.n_thetas * self.n_rhos * self.n_feat)


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
        n_rotations = self.n_thetas
        for k in range(n_rotations):
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
            # map to soft grid (gaussian descriptors)
            gauss_fxns = torch.mul(
                rho_coords_, thetas_coords_
            ) # batch_size*n_vertices, n_gauss (n_gauss = n_thetas*n_rhos)
            gauss_fxns = torch.reshape(
                gauss_fxns, (batch_size, n_vertices, -1)
            )  # batch_size, n_vertices, n_gauss
            gauss_fxns = torch.mul(
                gauss_fxns, mask
            )

            if mean_gauss_fxns:
            # computes mean weights for the different gaussians = gaussian-wise(OR local) averaging
                gauss_fxns /= (
                    torch.sum(gauss_fxns, 1, keepdim=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_fxns = gauss_fxns.unsqueeze(2)  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = input_feat.unsqueeze(3)  # batch_size, n_vertices, n_feat, 1
            gauss_desc = torch.mul(
                gauss_fxns, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,

            gauss_desc = torch.sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss, (=abstract out vertices factor)
            gauss_desc = torch.reshape(
                gauss_desc, (batch_size, self.n_gauss * n_feat)
            )  # batch_size, self.n_thetas*self.n_rhos*n_feat
            
            # Ensure input tensor has the same dtype as weights
            gauss_desc = gauss_desc.to(dtype=torch.float32)

            # convolution
            conv_desc = torch.matmul(gauss_desc, self.W) + self.b  # batch_size, self.n_thetas*self.n_rhos*n_feat
            all_conv_desc.append(conv_desc)

        # angular max pooling
        all_conv_desc = torch.stack(all_conv_desc)  # n_rotations, batch_size, self.n_thetas*self.n_rhos*n_feat
        conv_desc = torch.max(all_conv_desc, 0)[0]  # batch_size, self.n_thetas*self.n_rhos*n_feat
        # conv_desc = self.ln(conv_desc)              # nn.LayerNorm(n_thetas * n_rhos * n_feat)
        conv_desc = nn.ReLU()(conv_desc)
        return conv_desc


class Init_MLPBlock(L.LightningModule):
    """
    inputs: global_desc, (batch_size, n_gauss*n_feat)
    output: global_desc, (batch_size, n_feat)
    """
    def __init__(self, n_thetas, n_rhos, n_feat):
        super().__init__()
        self.FC12 = nn.Linear(n_thetas * n_rhos * n_feat, n_thetas * n_rhos)
        # self.norm1 = nn.LayerNorm(n_thetas * n_rhos)
        self.FC5 = nn.Linear(n_thetas * n_rhos, n_feat)
        # self.norm2 = nn.LayerNorm(n_feat)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        desc = self.relu(self.FC12(inputs))
        return self.relu(self.FC5(desc))

class Final_MLPBlock(L.LightningModule):
    """
    inputs: global_desc, (batch_size, n_feat)
    output: logits, (batch_size, n_labels)
    """
    def __init__(self, n_thetas, n_feat, n_labels):
        super().__init__()
        self.FC4 = nn.Linear(n_feat, n_thetas)
        # self.norm1 = nn.LayerNorm(n_thetas)
        self.FC2 = nn.Linear(n_thetas, n_labels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        desc = self.relu(self.FC4(inputs))
        return self.FC2(desc)

class Final_MLPBlock_transferLR(L.LightningModule):
    """
    new FC layers at the end of the model for transfer learning
    basic architecture: FC128 FC64 FC4 FC2 (for now)
    inputs: global_desc, (batch_size, n_feat)
    output: logits, (batch_size, n_labels)
    """
    def __init__(self, n_thetas, n_feat, n_labels):
        super().__init__()
        self.FC128  = nn.Linear(n_feat, 128)
        self.FC64   = nn.Linear(128, 64)
        self.FC4    = nn.Linear(64, n_thetas)
        self.FC2    = nn.Linear(n_thetas, n_labels)
        self.relu   = nn.ReLU()

    def forward(self, inputs):
        desc = self.relu(self.FC128(inputs))
        desc = self.relu(self.FC64(desc))
        desc = self.relu(self.FC4(desc))
        return self.FC2(desc)

