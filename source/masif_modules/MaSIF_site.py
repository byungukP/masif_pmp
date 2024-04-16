import tensorflow as tf
import numpy as np
from sklearn import metrics

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

    def build_sparse_matrix_softmax(self, idx_non_zero_values, X, dense_shape_A):
        A = tf.sparse.SparseTensor(idx_non_zero_values, tf.squeeze(X), dense_shape_A) # SparseTensorValue (TFv1) --> SparseTensor (TFv2)
        A = tf.sparse.reorder(A)  # n_edges x n_edges
        A = tf.sparse.softmax(A)
        return A

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

    def inference(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        n_samples = tf.shape(rho_coords)[0]
        n_vertices = tf.shape(rho_coords)[1]
        n_feat = tf.shape(input_feat)[2]

        all_conv_feat = []
        for k in range(self.n_rotations):
            rho_coords_ = tf.reshape(rho_coords, [-1, 1])  # batch_size*n_vertices_in_patch (batch_size = total num of patches = total num of vertices on ptn mesh)
            thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices_in_patch

            thetas_coords_ += k * 2 * np.pi / self.n_rotations
            thetas_coords_ = tf.math.floormod(thetas_coords_, 2 * np.pi)
            # turning rho_, theta_coords into corresponding probabilities in gaussian distribution function
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - mu_rho) / (tf.square(sigma_rho) + eps)
            )
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - mu_theta) / (tf.square(sigma_theta) + eps)
            )
            self.myshape = tf.shape(rho_coords_)
            self.rho_coords_debug = rho_coords_
            self.thetas_coords_debug = thetas_coords_

            gauss_activations = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss (n_gauss = n_thetas*n_rhos)
            gauss_activations = tf.reshape(
                gauss_activations, [n_samples, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss
            gauss_activations = tf.multiply(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians (gaussian-wise averaging)
                gauss_activations /= (
                    tf.reduce_sum(gauss_activations, 1, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 3
            )  # batch_size, n_vertices, n_feat, 1

            # gaussian descriptors: gaussian activation kernels w/ probability weights locally (=gaussian-wise) average the vertex-wise patch features (by tf.multiply(gauss_activations, input_feat_), thus acting as soft pixels)
            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss, (=abstract out vertices factor)
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos * n_feat]
            )  # batch_size, self.n_thetas*self.n_rhos*n_feat

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, self.n_thetas*self.n_rhos*n_feat
            all_conv_feat.append(conv_feat)
        all_conv_feat = tf.stack(all_conv_feat)  # n_rotations, batch_size, n_gauss
        conv_feat = tf.reduce_max(all_conv_feat, 0)  # (gaussian-wise) angular max pooling locally averaged surface features, (batch_size, n_gauss)
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_gamma=1.0,
        learning_rate=1e-3,
        n_rotations=16,
        idx_gpu="/device:GPU:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
        n_conv_layers=1,
        optimizer_method="Adam",
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
        - n_conv_layers (int): The number of convolutional layers.
        - optimizer_method (str): The optimization method to use.

        """
        super().__init__()

        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos

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

        for i in range(self.n_feat):
            self.mu_rho.append(
                tf.Variable(mu_rho_initial, name="mu_rho_{}".format(i))
            )  # 1, n_gauss
            self.sigma_rho.append(
                tf.Variable(
                    np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                    name="sigma_rho_{}".format(i),
                )
            )  # 1, n_gauss
            self.mu_theta.append(
                tf.Variable(mu_theta_initial, name="mu_theta_{}".format(i))
            )  # 1, n_gauss
            self.sigma_theta.append(
                tf.Variable(
                    (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                    name="sigma_theta_{}".format(i),
                )
            )  # 1, n_gauss

        # Gaussian Kernels for additional layers
        # single soft grid for all the surf feat in additional convolutional layers
        if n_conv_layers > 1:
            self.mu_rho_l2 = tf.Variable(
                mu_rho_initial, name="mu_rho_{}".format("l2")
            )
            self.sigma_rho_l2 = tf.Variable(
                np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                name="sigma_rho_{}".format("l2"),
            )
            self.mu_theta_l2 = tf.Variable(
                mu_theta_initial, name="mu_theta_{}".format("l2")
            )
            self.sigma_theta_l2 = tf.Variable(
                (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                name="sigma_theta_{}".format("l2"),
            )
        if n_conv_layers > 2:
            self.mu_rho_l3 = tf.Variable(
                mu_rho_initial, name="mu_rho_{}".format("l3")
            )
            self.sigma_rho_l3 = tf.Variable(
                np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                name="sigma_rho_{}".format("l3"),
            )
            self.mu_theta_l3 = tf.Variable(
                mu_theta_initial, name="mu_theta_{}".format("l3")
            )
            self.sigma_theta_l3 = tf.Variable(
                (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                name="sigma_theta_{}".format("l3"),
            )
        if n_conv_layers > 3:
            self.mu_rho_l4 = tf.Variable(
                mu_rho_initial, name="mu_rho_{}".format("l4")
            )
            self.sigma_rho_l4 = tf.Variable(
                np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                name="sigma_rho_{}".format("l4"),
            )
            self.mu_theta_l4 = tf.Variable(
                mu_theta_initial, name="mu_theta_{}".format("l4")
            )
            self.sigma_theta_l4 = tf.Variable(
                (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                name="sigma_theta_{}".format("l4"),
            )

        """
        Define GDL layers (below) --> as separate custom layers for cleaner code, variables + inference() (import)
        """
        # 1st GDL layer: surf feat-wise convolution
        self.W_conv = []
        self.b_conv = []
        for i in range(self.n_feat):
            self.W_conv.append(tf.Variable(
                    tf.keras.initializers.GlorotUniform()(
                        shape=(self.n_thetas * self.n_rhos,
                               self.n_thetas * self.n_rhos)
                    ),
                    name=f"W_conv_{i}",
            ))
            self.b_conv.append(tf.Variable(
                tf.zeros([self.n_thetas * self.n_rhos]),
                name=f"b_conv_{i}",
            ))

        # init_MLP = FC12, FC5
        self.init_MLP = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n_thetas * self.n_rhos, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.n_feat, activation=tf.nn.relu)
        ])

        # additional GDL layers: simple convolutions
        if self.n_conv_layers > 1:
            self.W_conv_l2 = tf.Variable(
                tf.keras.initializers.GlorotUniform()(
                    shape=(
                        self.n_thetas * self.n_rhos * self.n_feat,
                        self.n_thetas * self.n_rhos * self.n_feat,
                    )
                ),
                name="W_conv_l2",
            )
            self.b_conv_l2 = tf.Variable(
                tf.zeros([self.n_thetas * self.n_rhos * self.n_feat]),
                name="b_conv_l2",
            )

        if self.n_conv_layers > 2:
            self.W_conv_l3 = tf.Variable(
                tf.keras.initializers.GlorotUniform()(
                    shape=(
                        self.n_thetas * self.n_rhos * self.n_feat,
                        self.n_thetas * self.n_rhos * self.n_feat,
                    )
                ),
                name="W_conv_l3",
            )
            self.b_conv_l3 = tf.Variable(
                tf.zeros([self.n_thetas * self.n_rhos * self.n_feat]),
                name="b_conv_l3",
            )

        if self.n_conv_layers > 3:
            self.W_conv_l4 = tf.Variable(
                tf.keras.initializers.GlorotUniform()(
                    shape=(
                        self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos,
                        self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos,
                    )
                ),
                name="W_conv_l4",
            )
            self.b_conv_l4 = tf.Variable(
                tf.zeros([self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos]),
                name="b_conv_l4",
            )

        # final_MLP = FC4, FC2
        self.final_MLP = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n_thetas, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.n_labels, activation=None)
        ])

        # # metrics definition: arg name='binary_accuracy', 'precision', 'recall', 'auc'
        # self.metrics_list = [
        #     tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        #     tf.keras.metrics.Precision(name='precision'),
        #     tf.keras.metrics.Recall(name='recall'),
        #     tf.keras.metrics.AUC(name='auc')
        # ]

        # Stateful metrics from tf.keras accumulate information over time and require manual resetting
        # if you want to start fresh (e.g., at the beginning of a new epoch or evaluation phase)
        # in our case, have to reset states of the metrics per batch since metrics for a batch (per protein) is all we need        
        # also, memory issues may arise if you don't reset the states of the metrics
        # thus, explicitly using simple sklearn.metrics.methods may be more efficient


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
        self.keep_prob = tf.cast(input_dict["keep_prob"], dtype=tf.float32)  # scalar

        self.global_desc = []

        # Use Geometric deep learning
        for i in range(self.n_feat):
            my_input_feat = tf.expand_dims(self.input_feat[:, :, i], 2)
            rho_coords = self.rho_coords
            theta_coords = self.theta_coords
            mask = self.mask

            # inference()
            # a function that computes the output of a convolutional layer (Geometric deep learning)
            # performing local averaging of vertex-wise patch features via Gaussian kernels,
            # then max pooling w/ rotation_replicas, then ReLU activation
            self.global_desc.append(
                self.inference(
                    my_input_feat,
                    rho_coords,
                    theta_coords,
                    mask,
                    self.W_conv[i],
                    self.b_conv[i],
                    self.mu_rho[i],
                    self.sigma_rho[i],
                    self.mu_theta[i],
                    self.sigma_theta[i],
                )
            )  # batch_size, n_gauss*1
        # global_desc is n_feat, batch_size, n_gauss*1
        # They should be batch_size, n_gauss*n_feat (12 x 5)
        self.global_desc = tf.stack(self.global_desc, axis=1)
        self.global_desc = tf.reshape(
            self.global_desc, [-1, self.n_thetas * self.n_rhos * self.n_feat]
        )

        # init_MLP = FC12 (n_thease * n_rhos), FC5 (n_feat)
        self.global_desc = self.init_MLP(self.global_desc)

        # Do a second convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 1:
            # Rebuild a patch based on the output of the first layer
            self.global_desc = tf.gather(
                self.global_desc, self.indices_tensor
            )  # batch_size, max_verts, n_feat

            self.global_desc = self.inference(
                self.global_desc,
                rho_coords,
                theta_coords,
                mask,
                self.W_conv_l2,
                self.b_conv_l2,
                self.mu_rho_l2,
                self.sigma_rho_l2,
                self.mu_theta_l2,
                self.sigma_theta_l2,
            )  # batch_size, n_gauss*n_feat
            batch_size = tf.shape(self.global_desc)[0]
            # Reduce the dimensionality by averaging over the last dimension
            self.global_desc = tf.reshape(
                self.global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            self.global_desc = tf.reduce_mean(self.global_desc, axis=2)
            # self.global_desc_shape = tf.shape(self.global_desc)       # for debugging

        # Do a third convolutional layer. input: batch_size, n_feat, output: batch_size, n_feat
        if self.n_conv_layers > 2:
            # Rebuild a patch based on the output of the first layer
            self.global_desc = tf.gather(
                self.global_desc, self.indices_tensor
            )  # batch_size, max_verts, n_feat

            self.global_desc = self.inference(
                self.global_desc,
                rho_coords,
                theta_coords,
                mask,
                self.W_conv_l3,
                self.b_conv_l3,
                self.mu_rho_l3,
                self.sigma_rho_l3,
                self.mu_theta_l3,
                self.sigma_theta_l3,
            )  # batch_size, n_gauss*n_feat
            batch_size = tf.shape(self.global_desc)[0]
            self.global_desc = tf.reshape(
                self.global_desc,
                [batch_size, self.n_feat, self.n_thetas * self.n_rhos],
            )
            self.global_desc = tf.reduce_mean(self.global_desc, axis=2)

        # Do a fourth convolutional layer. input: batch_size, n_gauss, output: batch_size, n_gauss
        # W_conv_l4, b_conv_l4 shape looks werid, different from prev W_conv & b_conv
            # never used l4 though since n_conv_layers = 3
        if self.n_conv_layers > 3:
            # Rebuild a patch based on the output of the first layer
            self.global_desc = tf.gather(
                self.global_desc, self.indices_tensor
            )  # batch_size, max_verts, n_gauss (nope, n_feat)
            print("ConvL4 input global_desc shape: {}".format(self.global_desc.get_shape()))

            self.global_desc = self.inference(
                self.global_desc,
                rho_coords,
                theta_coords,
                mask,
                self.W_conv_l4,
                self.b_conv_l4,
                self.mu_rho_l4,
                self.sigma_rho_l4,
                self.mu_theta_l4,
                self.sigma_theta_l4,
            )  # batch_size, n_gauss, n_gauss*n_theta
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
        self.logits = self.final_MLP(self.global_desc)
        # self.count_number_parameters()

        return self.logits



    """
    train_step(): not used when using fit(), but used when using custom training loop
    where you have more control over the training process and want to define the exact operations performed during each training step.
    
    may need to modify this function to fit the custom training loop for transfer learning
    so let's use train_step() for now
    """
    # @tf.function
    def train_step(
        self,
        input_dict,
        optimizer_method,
        learning_rate=1e-3
    ):
        # input = input_dict
        # self.labels = tf.cast(input_dict["labels"], dtype=tf.int32)  # batch_size, n_labels
        with tf.GradientTape() as tape:
            # Forward pass (self() ~ model.call())
            self.eval_logits = self(input_dict, training=True)
            
            self.eval_labels = tf.concat(
                [
                    tf.gather(self.labels, self.pos_idx),
                    tf.gather(self.labels, self.neg_idx),
                ],
                axis=0,
            )
            self.eval_logits = tf.concat(
                [
                    tf.gather(self.logits, self.pos_idx),
                    tf.gather(self.logits, self.neg_idx),
                ],
                axis=0,
            )
            # Compute the loss
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(self.eval_labels, tf.float32),
                logits=self.eval_logits,
            )
            # eval_logits and eval_scores are reordered according to pos and neg_idx.
            self.eval_logits = tf.nn.sigmoid(self.eval_logits)
            self.eval_score = tf.squeeze(self.eval_logits)[:, 0]                            

            self.full_logits = tf.nn.sigmoid(self.logits)
            self.full_score = tf.squeeze(self.full_logits)[:, 0]

        # definition of the solver
        if optimizer_method == "AMSGrad":
            from monet_modules import AMSGrad

            print("Using AMSGrad as the optimizer")
            self.optimizer = AMSGrad.AMSGrad(
                learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8
            )
        else:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            )
        # Compute gradients
        gradients = tape.gradient(self.loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Log for gradients & norm of gradients
        for k in range(len(gradients)):
            if gradients[k] is None:
                print(self.trainable_variables[k])
        self.norm_grad = self.frobenius_norm(
            tf.concat([tf.reshape(g, [-1]) for g in gradients], 0)
        )   # a shape of [-1] flattens into 1-D.
        
        # Update metrics
        true=tf.cast(self.eval_labels[:, 0],tf.int32)
        pred=self.eval_score
        true, pred = true.numpy(), pred.numpy()
        return {
                    "loss": self.loss,
                    "eval_score": self.eval_score,
                    "full_score": self.full_score,
                    "auc": metrics.roc_auc_score(true, pred)
                }

    # for manually iterating over the validation dataset using a custom validation loop
    def test_step(
        self,
        input_dict
    ):
        # input = input_dict
        # self.labels = tf.cast(input_dict["labels"], dtype=tf.int32)  # batch_size, n_labels
        # Forward pass
        self.eval_logits = self(input_dict, training=True)
            
        self.eval_labels = tf.concat(
            [
                tf.gather(self.labels, self.pos_idx),
                tf.gather(self.labels, self.neg_idx),
            ],
            axis=0,
        )
        self.eval_logits = tf.concat(
            [
                tf.gather(self.logits, self.pos_idx),
                tf.gather(self.logits, self.neg_idx),
            ],
            axis=0,
        )
        # Compute the loss
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(self.eval_labels, tf.float32),
            logits=self.eval_logits,
        )
        # eval_logits and eval_scores are reordered according to pos and neg_idx.
        self.eval_logits = tf.nn.sigmoid(self.eval_logits)
        self.eval_score = tf.squeeze(self.eval_logits)[:, 0]

        self.full_logits = tf.nn.sigmoid(self.logits)
        self.full_score = tf.squeeze(self.full_logits)[:, 0]

        # Update metrics
        true=tf.cast(self.eval_labels[:, 0],tf.int32)
        pred=self.eval_score
        true, pred = true.numpy(), pred.numpy()
        return {
                    "loss": self.loss,
                    "eval_score": self.eval_score,
                    "full_score": self.full_score,
                    "auc": metrics.roc_auc_score(true, pred)
                }

        # legacy code
        # Update metrics
        # for metric in self.metrics_list:
        #     metric.update_state(
        #         tf.cast(self.eval_labels[:, 0],tf.float32),
        #         self.eval_score
        #     )
