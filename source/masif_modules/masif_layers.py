import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import metrics
from default_config.masif_opts import masif_opts

"""
Layers used for the MaSIF neural network model. TF v2
ByungUk Park UW-Madison, 2024

"""


class SoftGrid(keras.layers.Layer):

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
        super().__init__(name=name, **kwargs)
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_gauss = n_thetas * n_rhos
        self.n_feat = n_feat
        # mu, sigma for rho, theta
        self.mu_rho = tf.Variable(
            mu_rho_initial, name="mu_rho_{}".format(name)   # var_name='mu_rho_l1_0' or 'mu_rho_l2'
        )   # 1, n_gauss
        self.sigma_rho = tf.Variable(
            np.ones_like(mu_rho_initial) * sigma_rho_init,
            name="sigma_rho_{}".format(name),
        )   # 1, n_gauss
        self.mu_theta = tf.Variable(
            mu_theta_initial, name="mu_theta_{}".format(name)
        )   # 1, n_gauss
        self.sigma_theta = tf.Variable(
            (np.ones_like(mu_theta_initial) * sigma_theta_init),
            name="sigma_theta_{}".format(name),
        )   # 1, n_gauss
        # weights for convolution
        self.W = self.add_weight(
            shape=(
                self.n_thetas * self.n_rhos * self.n_feat,
                self.n_thetas * self.n_rhos * self.n_feat,
            ),
            initializer="glorot_uniform",
            trainable=True,
            name="W_conv_{}".format(name),
        )
        self.b = self.add_weight(
            shape=(self.n_thetas * self.n_rhos * self.n_feat),
            initializer="zeros",
            trainable=True,
            name="b_conv_{}".format(name),
        )

    def call(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        eps=1e-5,
        mean_gauss_fxns=True,
    ):
        # define the forward pass in the layer
        batch_size = tf.shape(rho_coords)[0]
        n_vertices = tf.shape(rho_coords)[1]
        n_feat = tf.shape(input_feat)[2]

        # soft grid mapping w/ rotation_replicas
        all_conv_desc = []
        n_rotations = self.n_rhos
        for k in range(n_rotations):
            rho_coords_ = tf.reshape(rho_coords, [-1, 1])  # batch_size*n_vertices_in_patch
            thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices_in_patch

            thetas_coords_ += k * 2 * np.pi / n_rotations
            thetas_coords_ = tf.math.floormod(thetas_coords_, 2 * np.pi)
            # turning rho_, theta_coords into corresponding probabilities in gaussian distribution function
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - self.mu_rho) / (tf.square(self.sigma_rho) + eps)
            )
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - self.mu_theta) / (tf.square(self.sigma_theta) + eps)
            )
            self.myshape = tf.shape(rho_coords_)    # for debugging
            self.rho_coords_debug = rho_coords_
            self.thetas_coords_debug = thetas_coords_

            # map to soft grid (gaussian descriptors)
            gauss_fxns = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss (n_gauss = n_thetas*n_rhos)
            gauss_fxns = tf.reshape(
                gauss_fxns, [batch_size, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss
            gauss_fxns = tf.multiply(gauss_fxns, mask)
            if (
                mean_gauss_fxns
            ):  # computes mean weights for the different gaussians = gaussian-wise(OR local) averaging
                # --> sum( probability fxns of all vertices in a patch wrt any Gaussian) = 1
                gauss_fxns /= (
                    tf.reduce_sum(gauss_fxns, 1, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_fxns = tf.expand_dims(
                gauss_fxns, 2
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 3
            )  # batch_size, n_vertices, n_feat, 1

            # gaussian descriptors: gaussian kernels w/ probability weights locally (=gaussian-wise) average the vertex-wise patch features (by tf.multiply(gauss_fxns, input_feat_), thus acting as soft pixels)
            gauss_desc = tf.multiply(
                gauss_fxns, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss, (=abstract out vertices factor)
            gauss_desc = tf.reshape(
                gauss_desc, [batch_size, self.n_gauss * n_feat]
            )  # batch_size, self.n_thetas*self.n_rhos*n_feat
            
            # convolution
            conv_desc = tf.matmul(gauss_desc, self.W) + self.b  # batch_size, self.n_thetas*self.n_rhos*n_feat
            all_conv_desc.append(conv_desc)

        # (gaussian-wise) angular max pooling
        all_conv_desc = tf.stack(all_conv_desc)  # n_rotations, batch_size, self.n_thetas*self.n_rhos*n_feat
        conv_desc = tf.reduce_max(all_conv_desc, 0)  # batch_size, self.n_thetas*self.n_rhos*n_feat
        conv_desc = tf.nn.relu(conv_desc)       # OR keras.activations.relu(conv_desc)
        return conv_desc


class Init_MLPBlock(keras.layers.Layer):
    """
    inputs: global_desc, (batch_size, n_gauss*n_feat)
    output: global_desc, (batch_size, n_feat)
    """
    def __init__(self, n_thetas, n_rhos, n_feat):
        super().__init__()
        self.FC12 = keras.layers.Dense(n_thetas * n_rhos, activation=tf.nn.relu)
        self.FC5 = keras.layers.Dense(n_feat, activation=tf.nn.relu)

    def call(self, inputs):
        desc = self.FC12(inputs)
        return self.FC5(desc)

class Final_MLPBlock(keras.layers.Layer):
    """
    inputs: global_desc, (batch_size, n_gauss*n_feat)
    output: logits, (batch_size, n_feat)
    """
    def __init__(self, n_thetas, n_labels):
        super().__init__()
        self.FC4 = keras.layers.Dense(n_thetas, activation=tf.nn.relu)
        self.FC2 = keras.layers.Dense(n_labels, activation=None)

    def call(self, inputs):
        desc = self.FC4(inputs)
        return self.FC2(desc)

