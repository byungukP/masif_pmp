import tensorflow as tf
import numpy as np
from sklearn import metrics
from masif_modules.masif_layers import SoftGrid, Init_MLPBlock, Final_MLPBlock


class MaSIF_site(tf.keras.Model):

    """
    The neural network model. TF v2
    ByungUk Park UW-Madison, 2024
    
    """

    def count_number_parameters(self):
        total_parameters = 0
        for variable in self.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.shape
            print(
                "<tf.Variable '{}' shape={} {}>".format(
                    variable.name, variable.shape, variable.dtype
                )
            )
            variable_parameters = tf.reduce_prod(shape)
            total_parameters += variable_parameters
        print("Total number parameters: %d" % total_parameters.numpy())

    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
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
        idx_gpu="/GPU:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
        n_conv_layers=1,
        optimizer_method="Adam",
        name="MaSIF_site",
        **kwargs
    ):
        """
        Initialize the MaSIF_site object.

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
        super().__init__(name=name, **kwargs)

        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.learning_rate = learning_rate

        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_labels = 2
        self.n_conv_layers = n_conv_layers

        tf.random.set_seed(0)
        
        initial_coords = self.compute_initial_coordinates()
        mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype("float32")
        mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype("float32")

        # Gaussian Kernels for layer 1
        # constructing learnable soft grids w/ n_rho * n_theta gaussians kernels defined in a local geodesic polar system
        # separate soft grids for each surface feature (e.g. 5 learnable gaussians with separate parameters for 5 surface features)
        # the parameters of the Gaussians are learnable on their own
        self.mu_rho = []
        self.sigma_rho = []
        self.mu_theta = []
        self.sigma_theta = []

        self.soft_grid_feat_l1 = []

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
                    name="l1_feat{}".format(i),           # name='l1_0' or 'l2'
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

        # final_MLP = FC4, FC2
        self.final_MLPBlock = Final_MLPBlock(self.n_thetas, self.n_labels)

        # metrics
        self.metrics_auc = tf.keras.metrics.AUC(name="AUC")

    def call(self, input_dict):
        # Define the forward pass
        # simplify the inference & GDL layers by writing py for custom_layers then importing them (for cleaner & more modularized code)

        self.rho_coords = tf.cast(input_dict["rho_coords"], dtype=tf.float32)  # batch_size, n_vertices, 1
        self.theta_coords = tf.cast(input_dict["theta_coords"], dtype=tf.float32)  # batch_size, n_vertices, 1
        self.input_feat = tf.cast(input_dict["input_feat"], dtype=tf.float32)  # batch_size, n_vertices, n_feat
        self.mask = tf.cast(input_dict["mask"], dtype=tf.float32)  # batch_size, n_vertices, 1
        self.pos_idx = tf.cast(input_dict["pos_idx"], dtype=tf.int32)  # batch_size/2
        self.neg_idx = tf.cast(input_dict["neg_idx"], dtype=tf.int32)  # batch_size/2
        self.labels = tf.cast(input_dict["labels"], dtype=tf.int32)  # batch_size, n_labels
        self.indices_tensor = tf.cast(input_dict["indices_tensor"], dtype=tf.int32)  # batch_size, max_verts (< 30)
        # self.keep_prob = tf.cast(input_dict["keep_prob"], dtype=tf.float32)  # scalar

        self.global_desc = []

        # Use Geometric deep learning

        # 1st GDL layer: surf feat-wise convolution
        for i in range(self.n_feat):
            my_input_feat = tf.expand_dims(self.input_feat[:, :, i], 2)

            self.global_desc.append(
                self.soft_grid_feat_l1[i](
                    my_input_feat,
                    self.rho_coords,
                    self.theta_coords,
                    self.mask
                )   # batch_size, n_gauss*1
            )   # n_feat, batch_size, n_gauss*1

        # global_desc should be batch_size, n_gauss*n_feat (12 x 5)
        self.global_desc = tf.stack(self.global_desc, axis=1)  # batch_size, n_feat, n_gauss*1
        self.global_desc = tf.reshape(
            self.global_desc, [-1, self.n_thetas * self.n_rhos * self.n_feat]
        )

        # init_MLP = FC12 (n_thease * n_rhos), FC5 (n_feat)
        self.global_desc = self.init_MLPBlock(self.global_desc)

        # additional GDL layers: simple convolutions
        # second convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 1:
            # Rebuild a patch based on the output of the first layer
            self.global_desc = tf.gather(
                self.global_desc, self.indices_tensor
            )  # batch_size, max_verts, n_feat

            self.global_desc = self.soft_grid_l2(
                self.global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat

            batch_size = tf.shape(self.global_desc)[0]
            # Reduce the dimensionality by averaging over the last dimension
            self.global_desc = tf.reshape(
                self.global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            self.global_desc = tf.reduce_mean(self.global_desc, axis=2)
            # self.global_desc_shape = tf.shape(self.global_desc)       # for debugging

        # third convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 2:
            # Rebuild a patch based on the output of the first layer
            self.global_desc = tf.gather(
                self.global_desc, self.indices_tensor
            )  # batch_size, max_verts, n_feat

            self.global_desc = self.soft_grid_l3(
                self.global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            batch_size = tf.shape(self.global_desc)[0]
            self.global_desc = tf.reshape(
                self.global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            self.global_desc = tf.reduce_mean(self.global_desc, axis=2)

        # fourth convolutional layer. input: batch_size, n_gauss, output: batch_size, n_gauss
        # W_conv_l4, b_conv_l4 shape looks werid, different from prev W_conv & b_conv
            # never used l4 though since n_conv_layers = 3
        if self.n_conv_layers > 3:
            # Rebuild a patch based on the output of the first layer
            self.global_desc = tf.gather(
                self.global_desc, self.indices_tensor
            )  # batch_size, max_verts, n_gauss (nope, n_feat)
            print("ConvL4 input global_desc shape: {}".format(self.global_desc.get_shape()))

            self.global_desc = self.soft_grid_l4(
                self.global_desc,
                self.rho_coords,
                self.theta_coords,
                self.mask
            )   # batch_size, n_gauss*n_feat
            print("ConvL4 output global_desc shape: {}".format(self.global_desc.get_shape()))
            batch_size = tf.shape(self.global_desc)[0]
            self.global_desc = tf.reshape(
                self.global_desc,
                [
                    batch_size,
                    self.n_thetas * self.n_rhos,
                    self.n_thetas * self.n_rhos,
                ],
            )
            self.global_desc = tf.reduce_max(self.global_desc, axis=2)
            self.global_desc_shape = tf.shape(self.global_desc)

        # refine global desc with MLP
        # final_MLP = FC4, FC2
        logits = self.final_MLPBlock(self.global_desc)
        return logits


    """
    train_step(): not used when using fit(), but used when using custom training loop
    where you have more control over the training process and want to define the exact operations performed during each training step.
    
    may need to modify this function to fit the custom training loop for transfer learning
    so let's use train_step() for now
    """

    # signature_dict ={"rho_coords": tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    #                  "theta_coords": tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    #                  "input_feat": tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    #                  "mask": tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
    #                  "labels": tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
    #                  "pos_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    #                  "neg_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    #                  "indices_tensor": tf.TensorSpec(shape=[None, None], dtype=tf.int32),}

    # @tf.function(input_signature=[signature_dict])
    # @tf.function(
    #         input_signature=[
    #             tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="rho_coords"),
    #             tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="theta_coords"),
    #             tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="input_feat"),
    #             tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32, name="mask"),
    #             tf.TensorSpec(shape=[None, 2], dtype=tf.int32, name="labels"),
    #             tf.TensorSpec(shape=[None], dtype=tf.int32, name="pos_idx"),
    #             tf.TensorSpec(shape=[None], dtype=tf.int32, name="neg_idx"),
    #             tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="indices_tensor"),
    #         ],
    # )
    # @tf.function(reduce_retracing=True)

    def train_step(
        self,
        input_dict,
        optimizer_method,
        learning_rate
    ):
        # input = input_dict
        # self.labels = tf.cast(input_dict["labels"], dtype=tf.int32)  # batch_size, n_labels
        # print('Tracing with', input_dict)    # for debugging, check whether trace only one tf.Graph for train_step()
        self.metrics_auc.reset_states()
        with tf.GradientTape() as tape:
            # Forward pass (self() ~ model.call())
            logits = self(input_dict, training=True)
            eval_labels = tf.concat(
                [
                    tf.gather(self.labels, self.pos_idx),
                    tf.gather(self.labels, self.neg_idx),
                ],
                axis=0,
            )
            eval_logits = tf.concat(
                [
                    tf.gather(logits, self.pos_idx),
                    tf.gather(logits, self.neg_idx),
                ],
                axis=0,
            )
            # Compute the loss
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(eval_labels, tf.float32),
                logits=eval_logits,
            )
            # eval_logits and eval_scores are reordered according to pos and neg_idx.
            eval_logits = tf.nn.sigmoid(eval_logits)
            eval_score = tf.squeeze(eval_logits)[:, 0]                            

            full_logits = tf.nn.sigmoid(logits)
            full_score = tf.squeeze(full_logits)[:, 0]

        # definition of the solver
        if optimizer_method == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compute gradients wrt trainable_variables (or weights)
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # # Log for gradients & norm of gradients
        # for k in range(len(gradients)):
        #     if gradients[k] is None:
        #         print(self.trainable_variables[k])
        # self.norm_grad = self.frobenius_norm(
        #     tf.concat([tf.reshape(g, [-1]) for g in gradients], 0)
        # )   # a shape of [-1] flattens into 1-D.
        
        # Update metrics
        # print("true:",tf.cast(eval_labels[:, 0], tf.int32))
        # print("pred:",eval_score)

        true = tf.cast(eval_labels[:, 0], tf.int32)
        pred = eval_score
        self.metrics_auc.update_state(true, pred)
        # true, pred = true.numpy(), pred.numpy()
        return {
                    "loss": loss,
                    "eval_score": eval_score,
                    "full_score": full_score,
                    "auc": self.metrics_auc.result() # metrics.roc_auc_score(true, pred)
                }

    # for manually iterating over the validation dataset using a custom validation loop
    def test_step(self, input_dict):
        # self.labels = tf.cast(input_dict["labels"], dtype=tf.int32)  # batch_size, n_labels
        self.metrics_auc.reset_states()
        # Forward pass
        logits = self(input_dict, training=False)
        eval_labels = tf.concat(
            [
                tf.gather(self.labels, self.pos_idx),
                tf.gather(self.labels, self.neg_idx),
            ],
            axis=0,
        )
        eval_logits = tf.concat(
            [
                tf.gather(logits, self.pos_idx),
                tf.gather(logits, self.neg_idx),
            ],
            axis=0,
        )
        # Compute the loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(eval_labels, tf.float32),
            logits=eval_logits,
        )
        # eval_logits and eval_scores are reordered according to pos and neg_idx.
        eval_logits = tf.nn.sigmoid(eval_logits)
        eval_score = tf.squeeze(eval_logits)[:, 0]

        full_logits = tf.nn.sigmoid(logits)
        full_score = tf.squeeze(full_logits)[:, 0]

        # Update metrics
        true=tf.cast(eval_labels[:, 0],tf.int32)
        pred=eval_score
        # true, pred = true.numpy(), pred.numpy()
        self.metrics_auc.update_state(true, pred)
        return {
                    "loss": loss,
                    "eval_score": eval_score,
                    "full_score": full_score,
                    "auc": self.metrics_auc.result() # metrics.roc_auc_score(true, pred)
                }

