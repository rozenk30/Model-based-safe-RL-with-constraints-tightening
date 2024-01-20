# ===========================
#   Critic DNN
# ==========================
import tensorflow as tf
import numpy as np

class VNetwork(object):
    def __init__(self, sess, s_dim, u_dim, V_dim, EPS_, layer_dims, activations, Q1, Q2, R, mode, lb, ub, delt,\
                 BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, mismatch,\
                 initializer = tf.contrib.layers.variance_scaling_initializer(factor = 1, mode='FAN_IN')):
        self.sess = sess
        self.s_dim = s_dim
        self.s_ext_dim = s_dim + u_dim
        self.V_dim = V_dim
        self.u_dim = u_dim
        self.EPS_ = EPS_
        self.input_dim = s_dim + u_dim  #neural network input dim
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.initializer = initializer
        self.Q1 = Q1
        self.Q2 = Q2
        self.R = R
        self.mode = mode
        self.lb = lb
        self.ub = ub
        self.delt = delt

        self.mismatch = mismatch
        # Create the critic network
        if layer_dims[0] < self.input_dim:
            raise ValueError('The first layer dimension must be at least the input dimension!')
        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase the dimension of its input!')
        self.hidden_dims = np.zeros(self.num_layers, dtype=int)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)

        self.s1_, self.s2_, self.s3_, self.s4_, self.u1_, self.u2_, self.a1_, self.a2_, \
        self.vout, self.vout2, self.reg_, self.state_ext, \
        self.z1_ss, self.z2_ss, self.z3_ss, self.z4_ss, self.u1_ss, self.u2_ss, self.mu_, \
        self.BO_LB_z, self.BO_UB_z, self.BO_LB_u, self.BO_UB_u, self.gamma_ = self.create_v_network()

        self.weight = tf.trainable_variables()

        self.f_dyn, self.g1_dyn, self.g2_dyn = self.dyn()
        self.f_dyn  = tf.reshape(self.f_dyn,  [self.s_ext_dim, -1])  # [4, BATCHSIZE]
        self.g1_dyn = tf.reshape(self.g1_dyn, [self.s_ext_dim, -1])
        self.g2_dyn = tf.reshape(self.g2_dyn, [self.s_ext_dim, -1])

        #self.state_ext_grads = tf.reshape(tf.gradients(self.vout, self.state_ext), [self.s_ext_dim, -1])  # [4,BATCHSIZE]
        self.state_ext_grads = tf.transpose(tf.gradients(self.vout, self.state_ext)[0])  # [6,BATCHSIZE]
        self.LfV  = tf.reduce_sum(tf.multiply(self.state_ext_grads, self.f_dyn), 0)  # tf.multiply: element-wise multiply
        self.Lg1V = tf.reduce_sum(tf.multiply(self.state_ext_grads, self.g1_dyn), 0)  # (batchsize,)
        self.Lg2V = tf.reduce_sum(tf.multiply(self.state_ext_grads, self.g2_dyn), 0)

        #self.state_ext_grads2 = tf.reshape(tf.gradients(self.vout2, self.state_ext), [self.s_ext_dim, -1]) # [4,BATCHSIZE]
        self.state_ext_grads2 = tf.transpose(tf.gradients(self.vout2, self.state_ext)[0]) # [6,BATCHSIZE]
        self.LfV2 = tf.reduce_sum(tf.multiply(self.state_ext_grads2, self.f_dyn), 0)  # tf.multiply는 element-wise multiply
        self.Lg1V2 = tf.reduce_sum(tf.multiply(self.state_ext_grads2, self.g1_dyn), 0)
        self.Lg2V2 = tf.reduce_sum(tf.multiply(self.state_ext_grads2, self.g2_dyn), 0)
        # [4, BATCHSIZE]  [4, BATCHSIZE] => [4, BATCHSIZE] => (batchsize,) 4개씩 더해서 총 BATCHSIZE인 결과

        self.error, self.Compare_, self.q = self.create_e()

        with tf.name_scope("loss") as scope:
         self.loss1 = tf.reduce_mean(0.5*tf.pow(self.error, 2))
         self.learning_rate = tf.placeholder(tf.float32, [])
         self.loss_tot = self.loss1 # + 100*tf.reduce_mean(tf.pow(self.Compare_ - (-0.5), 2))
         self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_tot)

         #self.train_op = tf.contrib.estimator.clip_gradients_by_norm(self.optimize, clip_norm = 5.0).minimize(self.loss_tot)

    def create_v_network(self):
        s1_ = tf.placeholder(tf.float32, [None, 1])
        s2_ = tf.placeholder(tf.float32, [None, 1])
        s3_ = tf.placeholder(tf.float32, [None, 1])
        s4_ = tf.placeholder(tf.float32, [None, 1])
        u1_ = tf.placeholder(tf.float32, [None, 1])
        u2_ = tf.placeholder(tf.float32, [None, 1])
        z1_ss = tf.placeholder(tf.float32, [None, 1])
        z2_ss = tf.placeholder(tf.float32, [None, 1])
        z3_ss = tf.placeholder(tf.float32, [None, 1])
        z4_ss = tf.placeholder(tf.float32, [None, 1])
        u1_ss = tf.placeholder(tf.float32, [None, 1])
        u2_ss = tf.placeholder(tf.float32, [None, 1])
        a1_ = tf.placeholder(tf.float32, [None, 1])
        a2_ = tf.placeholder(tf.float32, [None, 1])
        mu_ = tf.placeholder(tf.float32, [2,])

        BO_LB_z = tf.placeholder(tf.float32, [self.s_dim,])
        BO_UB_z = tf.placeholder(tf.float32, [self.s_dim,])
        BO_LB_u = tf.placeholder(tf.float32, [self.u_dim,])
        BO_UB_u = tf.placeholder(tf.float32, [self.u_dim,])
        gamma_  = tf.placeholder(tf.float32, [1, 1])

        with tf.name_scope("V_now") as scope:
            z_min, z_max = [1.,1.,1.,1.], [28, 28, 28, 28]
            u_min, u_max = [0.,0.], [60., 60.]
            self.BO_LB_z = BO_LB_z
            self.BO_UB_z = BO_UB_z
            self.BO_LB_u = BO_LB_u
            self.BO_UB_u = BO_UB_u
            self.gamma_  = gamma_
            z_min_n = [z_min[0] + BO_LB_z[0],
                       z_min[1] + BO_LB_z[1],
                       z_min[2] + BO_LB_z[2],
                       z_min[3] + BO_LB_z[3]]
            z_max_n = [z_max[0] - BO_UB_z[0],
                       z_max[1] - BO_UB_z[1],
                       z_max[2] - BO_UB_z[2],
                       z_max[3] - BO_UB_z[3]]
            u_min_n = [u_min[0] + BO_LB_u[0],
                       u_min[1] + BO_LB_u[1]]
            u_max_n = [u_max[0] - BO_UB_u[0],
                       u_max[1] - BO_UB_u[1]]

            self.z1_dev = (s1_ - z1_ss)/(z_max_n[0]-z_min_n[0])
            self.z2_dev = (s2_ - z2_ss)/(z_max_n[1]-z_min_n[1])
            self.z3_dev = (s3_ - z3_ss)/(z_max_n[2]-z_min_n[2])
            self.z4_dev = (s4_ - z4_ss)/(z_max_n[3]-z_min_n[3])
            self.u1_dev = (u1_ - u1_ss)/(u_max_n[0]-u_min_n[0])
            self.u2_dev = (u2_ - u2_ss)/(u_max_n[1]-u_min_n[1])

            state_ext = tf.concat([self.z1_dev, self.z2_dev, self.z3_dev, self.z4_dev, self.u1_dev, self.u2_dev],1)  # state_ext # N, 4
            net = state_ext
            if isinstance(net, np.ndarray):
                net = tf.constant(net)
            for i in range(self.num_layers):
                if i == 0:
                    layer_input_dim = self.input_dim
                else:
                    layer_input_dim = self.output_dims[i - 1]
                W = tf.get_variable('weights_posdef_{}'.format(i), [self.hidden_dims[i], layer_input_dim], tf.float32,
                                    initializer=self.initializer)
                kernel = tf.matmul(W, W, transpose_a=True) + self.EPS_ * tf.eye(layer_input_dim, dtype=tf.float32)
                dim_diff = self.output_dims[i] - layer_input_dim
                if dim_diff > 0:
                    W = tf.get_variable('weights_{}'.format(i), [dim_diff, layer_input_dim], tf.float32,
                                        initializer=self.initializer)
                    kernel = tf.concat([kernel, W], axis=0)
                layer_output = tf.matmul(net, kernel, transpose_b=True)
                net = self.activations[i](layer_output, name='layer_output_{}'.format(i))

                if i == self.num_layers-1:
                   # W = tf.get_variable('output', [1, 1], tf.float32, initializer=tf.constant_initializer(value=0.5))
                    net = layer_output * 0.5

            values = tf.reduce_sum(tf.square(net), axis=1, keepdims=True, name='quadratic_form')


            if self.mode == 1:


                h1 = (z1_ss - z_min_n[0]) / (z_max_n[0] - z_min_n[0]) + self.z1_dev
                h2 = (z_max_n[0] - z1_ss) / (z_max_n[0] - z_min_n[0]) - self.z1_dev
                h3 = (z2_ss - z_min_n[1]) / (z_max_n[1] - z_min_n[1]) + self.z2_dev
                h4 = (z_max_n[1] - z2_ss) / (z_max_n[1] - z_min_n[1]) - self.z2_dev

                h5 = (z3_ss - z_min_n[2]) / (z_max_n[2] - z_min_n[2]) + self.z3_dev
                h6 = (z_max_n[2] - z3_ss) / (z_max_n[2] - z_min_n[2]) - self.z3_dev
                h7 = (z4_ss - z_min_n[3]) / (z_max_n[3] - z_min_n[3]) + self.z4_dev
                h8 = (z_max_n[3] - z4_ss) / (z_max_n[3] - z_min_n[3]) - self.z4_dev

                h9 = (u1_ss - u_min_n[0]) / (u_max_n[0] - u_min_n[0]) + self.u1_dev
                h10 = (u_max_n[0] - u1_ss) / (u_max_n[0] - u_min_n[0]) - self.u1_dev
                h11 = (u2_ss - u_min_n[1]) / (u_max_n[1] - u_min_n[1]) + self.u2_dev
                h12 = (u_max_n[1] - u2_ss) / (u_max_n[1] - u_min_n[1]) - self.u2_dev

                s1 = 1 / ((z1_ss - z_min_n[0]) * (1 + z1_ss - z_min_n[0]) / (z_max_n[0] - z1_ss) / (1 + z_max_n[0] - z1_ss) + 1)
                s2 = 1 / ((z2_ss - z_min_n[1]) * (1 + z2_ss - z_min_n[1]) / (z_max_n[1] - z2_ss) / (1 + z_max_n[1] - z2_ss) + 1)
                s3 = 1 / ((z3_ss - z_min_n[2]) * (1 + z3_ss - z_min_n[2]) / (z_max_n[2] - z3_ss) / (1 + z_max_n[2] - z3_ss) + 1)
                s4 = 1 / ((z4_ss - z_min_n[3]) * (1 + z4_ss - z_min_n[3]) / (z_max_n[3] - z4_ss) / (1 + z_max_n[3] - z4_ss) + 1)
                s5 = 1 / ((u1_ss - u_min_n[0]) * (1 + u1_ss - u_min_n[0]) / (u_max_n[0] - u1_ss) / (1 + u_max_n[0] - u1_ss) + 1)
                s6 = 1 / ((u2_ss - u_min_n[1]) * (1 + u2_ss - u_min_n[1]) / (u_max_n[1] - u2_ss) / (1 + u_max_n[1] - u2_ss) + 1)

                h1_ss, h2_ss = (z1_ss - z_min_n[0]) / (z_max_n[0] - z_min_n[0]), \
                               (z_max_n[0] - z1_ss) / (z_max_n[0] - z_min_n[0])
                lambda1 = ((1 - s1) * tf.math.log((h1_ss) / (h1_ss + mu_[1])) \
                           + s1 * tf.math.log((h2_ss) / (h2_ss + mu_[1])))

                h3_ss, h4_ss = (z2_ss - z_min_n[1]) / (z_max_n[1] - z_min_n[1]), \
                               (z_max_n[1] - z2_ss) / (z_max_n[1] - z_min_n[1])
                lambda2 = ((1 - s2) * tf.math.log((h3_ss) / (h3_ss + mu_[1])) \
                           + s2 * tf.math.log((h4_ss) / (h4_ss + mu_[1])))

                h5_ss, h6_ss = (z3_ss - z_min_n[2]) / (z_max_n[2] - z_min_n[2]), \
                               (z_max_n[2] - z3_ss) / (z_max_n[2] - z_min_n[2])
                lambda3 = ((1 - s3) * tf.math.log((h5_ss) / (h5_ss + mu_[1])) \
                           + s3 * tf.math.log((h6_ss) / (h6_ss + mu_[1])))

                h7_ss, h8_ss = (z4_ss - z_min_n[3]) / (z_max_n[3] - z_min_n[3]), \
                               (z_max_n[3] - z4_ss) / (z_max_n[3] - z_min_n[3])
                lambda4 = ((1 - s4) * tf.math.log((h7_ss) / (h7_ss + mu_[1])) \
                           + s4 * tf.math.log((h8_ss) / (h8_ss + mu_[1])))

                h9_ss, h10_ss = (u1_ss - u_min_n[0]) / (u_max_n[0] - u_min_n[0]), \
                                (u_max_n[0] - u1_ss) / (u_max_n[0] - u_min_n[0])
                lambda5 = ((1 - s5) * tf.math.log((h9_ss) / (h9_ss + mu_[1])) \
                           + s5 * tf.math.log((h10_ss) / (h10_ss + mu_[1])))

                h11_ss, h12_ss = (u2_ss - u_min_n[1]) / (u_max_n[1] - u_min_n[1]), \
                                 (u_max_n[1] - u2_ss) / (u_max_n[1] - u_min_n[1])
                lambda6 = ((1 - s6) * tf.math.log((h11_ss) / (h11_ss + mu_[1])) \
                           + s6 * tf.math.log((h12_ss) / (h12_ss + mu_[1])))

                reg_ = - mu_[0] * ((1 - s1) * tf.math.log((h1) / (h1 + mu_[1])) \
                                       + s1 * tf.math.log((h2) / (h2 + mu_[1])) - lambda1) \
                       - mu_[0] * ((1 - s2) * tf.math.log((h3) / (h3 + mu_[1])) \
                                       + s2 * tf.math.log((h4) / (h4 + mu_[1])) - lambda2) \
                       - mu_[0] * ((1 - s3) * tf.math.log((h5) / (h5 + mu_[1])) \
                                       + s3 * tf.math.log((h6) / (h6 + mu_[1])) - lambda3) \
                       - mu_[0] * ((1 - s4) * tf.math.log((h7) / (h7 + mu_[1])) \
                                       + s4 * tf.math.log((h8) / (h8 + mu_[1])) - lambda4) \
                       - mu_[0] * ((1 - s5) * tf.math.log((h9) / (h9 + mu_[1])) \
                                       + s5 * tf.math.log((h10) / (h10 + mu_[1])) - lambda5) \
                       - mu_[0] * ((1 - s6) * tf.math.log((h11) / (h11 + mu_[1])) \
                                       + s6 * tf.math.log((h12) / (h12 + mu_[1])) - lambda6)
                values_2 = values + reg_

            return s1_, s2_, s3_, s4_, u1_, u2_, a1_, a2_, values, values_2, reg_, \
                   state_ext, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, \
                   BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_

    def create_e(self):  #
        with tf.name_scope("e_now") as scope:
            z_min, z_max = [1.,1.,1.,1.], [28, 28, 28, 28]
            u_min, u_max = [self.lb[1], self.lb[2]], [self.ub[1], self.ub[2]]
            Q1, Q2, R_ = self.Q1, self.Q2, self.R
            r1_, r2_ = tf.expand_dims([R_], 0), tf.expand_dims([R_], 0)
            mu_ = self.mu_
            z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss = self.z1_ss, self.z2_ss, self.z3_ss, self.z4_ss, self.u1_ss, self.u2_ss
            state_ext = self.state_ext
            BO_LB_z = self.BO_LB_z
            BO_UB_z = self.BO_UB_z
            BO_LB_u = self.BO_LB_u
            BO_UB_u = self.BO_UB_u
            gamma_  = self.gamma_

            if self.mode == 1:
                z_min_n = [z_min[0] + BO_LB_z[0],
                           z_min[1] + BO_LB_z[1],
                           z_min[2] + BO_LB_z[2],
                           z_min[3] + BO_LB_z[3]]
                z_max_n = [z_max[0] - BO_UB_z[0],
                           z_max[1] - BO_UB_z[1],
                           z_max[2] - BO_UB_z[2],
                           z_max[3] - BO_UB_z[3]]
                u_min_n = [u_min[0] + BO_LB_u[0],
                           u_min[1] + BO_LB_u[1]]
                u_max_n = [u_max[0] - BO_UB_u[0],
                           u_max[1] - BO_UB_u[1]]

                h1 = (z1_ss - z_min_n[0]) / (z_max_n[0] - z_min_n[0]) + self.z1_dev
                h2 = (z_max_n[0] - z1_ss) / (z_max_n[0] - z_min_n[0]) - self.z1_dev
                h3 = (z2_ss - z_min_n[1]) / (z_max_n[1] - z_min_n[1]) + self.z2_dev
                h4 = (z_max_n[1] - z2_ss) / (z_max_n[1] - z_min_n[1]) - self.z2_dev

                h5 = (z3_ss - z_min_n[2]) / (z_max_n[2] - z_min_n[2]) + self.z3_dev
                h6 = (z_max_n[2] - z3_ss) / (z_max_n[2] - z_min_n[2]) - self.z3_dev
                h7 = (z4_ss - z_min_n[3]) / (z_max_n[3] - z_min_n[3]) + self.z4_dev
                h8 = (z_max_n[3] - z4_ss) / (z_max_n[3] - z_min_n[3]) - self.z4_dev

                h9  = (u1_ss - u_min_n[0]) / (u_max_n[0] - u_min_n[0]) + self.u1_dev
                h10 = (u_max_n[0] - u1_ss) / (u_max_n[0] - u_min_n[0]) - self.u1_dev
                h11 = (u2_ss - u_min_n[1]) / (u_max_n[1] - u_min_n[1]) + self.u2_dev
                h12 = (u_max_n[1] - u2_ss) / (u_max_n[1] - u_min_n[1]) - self.u2_dev

                s1 = 1 / ((z1_ss - z_min_n[0]) * (1 + z1_ss - z_min_n[0]) / (z_max_n[0] - z1_ss) / (
                            1 + z_max_n[0] - z1_ss) + 1)
                s2 = 1 / ((z2_ss - z_min_n[1]) * (1 + z2_ss - z_min_n[1]) / (z_max_n[1] - z2_ss) / (
                            1 + z_max_n[1] - z2_ss) + 1)
                s3 = 1 / ((z3_ss - z_min_n[2]) * (1 + z3_ss - z_min_n[2]) / (z_max_n[2] - z3_ss) / (
                            1 + z_max_n[2] - z3_ss) + 1)
                s4 = 1 / ((z4_ss - z_min_n[3]) * (1 + z4_ss - z_min_n[3]) / (z_max_n[3] - z4_ss) / (
                            1 + z_max_n[3] - z4_ss) + 1)
                s5 = 1 / ((u1_ss - u_min_n[0]) * (1 + u1_ss - u_min_n[0]) / (u_max_n[0] - u1_ss) / (
                            1 + u_max_n[0] - u1_ss) + 1)
                s6 = 1 / ((u2_ss - u_min_n[1]) * (1 + u2_ss - u_min_n[1]) / (u_max_n[1] - u2_ss) / (
                            1 + u_max_n[1] - u2_ss) + 1)

                h1_ss, h2_ss = (z1_ss - z_min_n[0]) / (z_max_n[0] - z_min_n[0]), \
                               (z_max_n[0] - z1_ss) / (z_max_n[0] - z_min_n[0])

                lambda1 = ((1 - s1) * tf.math.log((h1_ss) / (h1_ss + mu_[1])) \
                           + s1 * tf.math.log((h2_ss) / (h2_ss + mu_[1])))

                h3_ss, h4_ss = (z2_ss - z_min_n[1]) / (z_max_n[1] - z_min_n[1]), \
                               (z_max_n[1] - z2_ss) / (z_max_n[1] - z_min_n[1])

                lambda2 = ((1 - s2) * tf.math.log((h3_ss) / (h3_ss + mu_[1])) \
                           + s2 * tf.math.log((h4_ss) / (h4_ss + mu_[1])))

                h5_ss, h6_ss = (z3_ss - z_min_n[2]) / (z_max_n[2] - z_min_n[2]), \
                               (z_max_n[2] - z3_ss) / (z_max_n[2] - z_min_n[2])

                lambda3 = ((1 - s3) * tf.math.log((h5_ss) / (h5_ss + mu_[1])) \
                           + s3 * tf.math.log((h6_ss) / (h6_ss + mu_[1])))

                h7_ss, h8_ss = (z4_ss - z_min_n[3]) / (z_max_n[3] - z_min_n[3]), \
                               (z_max_n[3] - z4_ss) / (z_max_n[3] - z_min_n[3])

                lambda4 = ((1 - s4) * tf.math.log((h7_ss) / (h7_ss + mu_[1])) \
                           + s4 * tf.math.log((h8_ss) / (h8_ss + mu_[1])))

                h9_ss, h10_ss = (u1_ss - u_min_n[0]) / (u_max_n[0] - u_min_n[0]), \
                                (u_max_n[0] - u1_ss) / (u_max_n[0] - u_min_n[0])
                lambda5 = ((1 - s5) * tf.math.log((h9_ss) / (h9_ss + mu_[1])) \
                           + s5 * tf.math.log((h10_ss) / (h10_ss + mu_[1])))

                h11_ss, h12_ss = (u2_ss - u_min_n[1]) / (u_max_n[1] - u_min_n[1]), \
                                 (u_max_n[1] - u2_ss) / (u_max_n[1] - u_min_n[1])

                lambda6 = ((1 - s6) * tf.math.log((h11_ss) / (h11_ss + mu_[1])) \
                           + s6 * tf.math.log((h12_ss) / (h12_ss + mu_[1])))

                reg_ = - mu_[0] * ((1 - s1) * tf.math.log((h1) / (h1 + mu_[1])) \
                                       + s1 * tf.math.log((h2) / (h2 + mu_[1])) - lambda1) \
                       - mu_[0] * ((1 - s2) * tf.math.log((h3) / (h3 + mu_[1])) \
                                       + s2 * tf.math.log((h4) / (h4 + mu_[1])) - lambda2) \
                       - mu_[0] * ((1 - s3) * tf.math.log((h5) / (h5 + mu_[1])) \
                                       + s3 * tf.math.log((h6) / (h6 + mu_[1])) - lambda3) \
                       - mu_[0] * ((1 - s4) * tf.math.log((h7) / (h7 + mu_[1])) \
                                       + s4 * tf.math.log((h8) / (h8 + mu_[1])) - lambda4) \
                       - mu_[0] * ((1 - s5) * tf.math.log((h9) / (h9 + mu_[1])) \
                                       + s5 * tf.math.log((h10) / (h10 + mu_[1])) - lambda5) \
                       - mu_[0] * ((1 - s6) * tf.math.log((h11) / (h11 + mu_[1])) \
                                       + s6 * tf.math.log((h12) / (h12 + mu_[1])) - lambda6)

            q_ = Q1 * tf.pow(self.z1_dev, 2)  + Q1 * tf.pow(self.z2_dev, 2) \
               + Q1 * tf.pow(self.z3_dev, 2)  + Q1 * tf.pow(self.z4_dev, 2) \
               + Q1 * tf.pow(self.u1_dev, 2)  + Q1 * tf.pow(self.u2_dev, 2)+ reg_
            uRu = tf.pow(self.a1_, 2) * r1_ + tf.pow(self.a2_, 2) * r2_
            error = q_ + uRu + tf.reshape(self.LfV2, [-1, 1]) + tf.reshape(self.Lg1V2, [-1, 1]) * self.a1_ \
                                                              + tf.reshape(self.Lg2V2, [-1, 1]) * self.a2_ \
                                                              - 1/gamma_ * self.vout2

            den = tf.pow(self.Lg1V2, 2) / r1_ + tf.pow(self.Lg2V2, 2) / r2_ + self.EPS_  # (batchsize,)
            sqrt_ = tf.sqrt(tf.pow(self.LfV2, 2) + (q_ - 1/gamma_ * self.vout2) * (tf.pow(self.Lg1V2, 2) / r1_ + tf.pow(self.Lg2V2, 2) / r2_))
            Compare_ = tf.divide(-(self.LfV2 + sqrt_), den)  # have to converge to -1/2
            return error, Compare_, q_

    def dyn(self):
        z1 = self.s1_  # (N, 1)
        z2 = self.s2_  # (N, 1)
        z3 = self.s3_  # (N, 1)
        z4 = self.s4_  # (N, 1)
        u1 = self.u1_
        u2 = self.u2_

        z_min, z_max = [1.,1.,1.,1.], [28, 28, 28, 28]
        u_min, u_max = [self.lb[1], self.lb[2]], [self.ub[1], self.ub[2]]

        BO_LB_z = self.BO_LB_z
        BO_UB_z = self.BO_UB_z
        BO_LB_u = self.BO_LB_u
        BO_UB_u = self.BO_UB_u

        mismatch = self.mismatch

        z_min_n = [z_min[0] + BO_LB_z[0],
                   z_min[1] + BO_LB_z[1],
                   z_min[2] + BO_LB_z[2],
                   z_min[3] + BO_LB_z[3]]
        z_max_n = [z_max[0] - BO_UB_z[0],
                   z_max[1] - BO_UB_z[1],
                   z_max[2] - BO_UB_z[2],
                   z_max[3] - BO_UB_z[3]]
        u_min_n = [u_min[0] + BO_LB_u[0],
                   u_min[1] + BO_LB_u[1]]
        u_max_n = [u_max[0] - BO_UB_u[0],
                   u_max[1] - BO_UB_u[1]]

        # Constant
        A1, A2, A3, A4 = 50.27 * mismatch[0], 50.27 * mismatch[1], 28.27 * mismatch[2], 28.27 * mismatch[3]
        gamma1, gamma2 = 0.4, 0.4
        a1, a2, a3, a4 = 0.233, 0.242, 0.127, 0.127
        g = 980  # cm/s2

        f1 = - a1 / A1 * tf.math.sqrt(2 * g * z1) + a3 / A1 * tf.math.sqrt(2 * g * z3) + gamma1 / A1 * u1
        f2 = - a2 / A2 * tf.math.sqrt(2 * g * z2) + a4 / A2 * tf.math.sqrt(2 * g * z4) + gamma2 / A2 * u2
        f3 = - a3 / A3 * tf.math.sqrt(2 * g * z3) + (1 - gamma2) / A3 * u2
        f4 = - a4 / A4 * tf.math.sqrt(2 * g * z4) + (1 - gamma1) / A4 * u1

        f_dyn = [f1 / (z_max_n[0] - z_min_n[0]), f2 / (z_max_n[1] - z_min_n[1]),\
                 f3 / (z_max_n[2] - z_min_n[2]), f4 / (z_max_n[3] - z_min_n[3]), tf.zeros_like(z1), tf.zeros_like(z1)]
        g1_dyn = [tf.zeros_like(z1), tf.zeros_like(z1), tf.zeros_like(z1), tf.zeros_like(z1), \
                  tf.ones_like(z1)/ (u_max_n[0] - u_min_n[0]), tf.zeros_like(z1)]
        g2_dyn = [tf.zeros_like(z1), tf.zeros_like(z1), tf.zeros_like(z1), tf.zeros_like(z1), \
                  tf.zeros_like(z1), tf.ones_like(z1)/ (u_max_n[1] - u_min_n[1])]
        return f_dyn, g1_dyn, g2_dyn

    def cal_Lie(self, z1_, z2_, z3_, z4_, u1_, u2_, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, \
                BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_):
        return self.sess.run([self.LfV, self.Lg1V, self.Lg2V], feed_dict={
            self.s1_: z1_,
            self.s2_: z2_,
            self.s3_: z3_,
            self.s4_: z4_,
            self.u1_: u1_,
            self.u2_: u2_,
            self.z1_ss: z1_ss,
            self.z2_ss: z2_ss,
            self.z3_ss: z3_ss,
            self.z4_ss: z4_ss,
            self.u1_ss: u1_ss,
            self.u2_ss: u2_ss,
            self.mu_: mu_,
            self.BO_LB_z: BO_LB_z,
            self.BO_UB_z: BO_UB_z,
            self.BO_LB_u: BO_LB_u,
            self.BO_UB_u: BO_UB_u,
            self.gamma_: gamma_
        })

    def cal_Lie2(self, z1_, z2_, z3_, z4_, u1_, u2_, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, \
                 BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_):
        return self.sess.run([self.LfV2, self.Lg1V2, self.Lg2V2, self.state_ext_grads2], feed_dict={
            self.s1_: z1_,
            self.s2_: z2_,
            self.s3_: z3_,
            self.s4_: z4_,
            self.u1_: u1_,
            self.u2_: u2_,
            self.z1_ss: z1_ss,
            self.z2_ss: z2_ss,
            self.z3_ss: z3_ss,
            self.z4_ss: z4_ss,
            self.u1_ss: u1_ss,
            self.u2_ss: u2_ss,
            self.mu_: mu_,
            self.BO_LB_z: BO_LB_z,
            self.BO_UB_z: BO_UB_z,
            self.BO_LB_u: BO_LB_u,
            self.BO_UB_u: BO_UB_u,
            self.gamma_: gamma_
        })


    def cal_W(self, z1_, z2_, z3_, z4_, u1_, u2_, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, \
              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_):
        return self.sess.run(self.weight, feed_dict={
            self.s1_: z1_,
            self.s2_: z2_,
            self.s3_: z3_,
            self.s4_: z4_,
            self.u1_: u1_,
            self.u2_: u2_,
            self.z1_ss: z1_ss,
            self.z2_ss: z2_ss,
            self.z3_ss: z3_ss,
            self.z4_ss: z4_ss,
            self.u1_ss: u1_ss,
            self.u2_ss: u2_ss,
            self.mu_: mu_,
            self.BO_LB_z: BO_LB_z,
            self.BO_UB_z: BO_UB_z,
            self.BO_LB_u: BO_LB_u,
            self.BO_UB_u: BO_UB_u,
            self.gamma_:gamma_
        })

    def cal_V(self, z1_, z2_, z3_, z4_, u1_, u2_, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, \
              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_):
        return self.sess.run([self.vout, self.vout2, self.reg_, self.q], feed_dict={
            self.s1_: z1_,
            self.s2_: z2_,
            self.s3_: z3_,
            self.s4_: z4_,
            self.u1_: u1_,
            self.u2_: u2_,
            self.z1_ss: z1_ss,
            self.z2_ss: z2_ss,
            self.z3_ss: z3_ss,
            self.z4_ss: z4_ss,
            self.u1_ss: u1_ss,
            self.u2_ss: u2_ss,
            self.mu_: mu_,
            self.BO_LB_z: BO_LB_z,
            self.BO_UB_z: BO_UB_z,
            self.BO_LB_u: BO_LB_u,
            self.BO_UB_u: BO_UB_u,
            self.gamma_: gamma_
        })

    def train(self, z1_, z2_, z3_, z4_, u1_, u2_, a1_, a2_, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, lr, \
              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_):
        return self.sess.run([self.loss_tot, self.optimize, self.weight], feed_dict={
            self.s1_: z1_,
            self.s2_: z2_,
            self.s3_: z3_,
            self.s4_: z4_,
            self.u1_: u1_,
            self.u2_: u2_,
            self.a1_: a1_,
            self.a2_: a2_,
            self.z1_ss: z1_ss,
            self.z2_ss: z2_ss,
            self.z3_ss: z3_ss,
            self.z4_ss: z4_ss,
            self.u1_ss: u1_ss,
            self.u2_ss: u2_ss,
            self.mu_: mu_,
            self.learning_rate: lr,
            self.BO_LB_z: BO_LB_z,
            self.BO_UB_z: BO_UB_z,
            self.BO_LB_u: BO_LB_u,
            self.BO_UB_u: BO_UB_u,
            self.gamma_: gamma_
        })

    def cal_info(self, z1_, z2_, z3_, z4_, u1_, u2_, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, mu_, \
                 BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_):
        return self.sess.run([self.z1_dev, self.z2_dev, self.z3_dev, self.z4_dev, self.u1_dev, self.u2_dev,
                              self.state_ext_grads2],
                             feed_dict={
             self.s1_: z1_,
             self.s2_: z2_,
             self.s3_: z3_,
             self.s4_: z4_,
             self.u1_: u1_,
             self.u2_: u2_,
             self.z1_ss: z1_ss,
             self.z2_ss: z2_ss,
             self.z3_ss: z3_ss,
             self.z4_ss: z4_ss,
             self.u1_ss: u1_ss,
             self.u2_ss: u2_ss,
             self.mu_: mu_,
             self.BO_LB_z: BO_LB_z,
             self.BO_UB_z: BO_UB_z,
             self.BO_LB_u: BO_LB_u,
             self.BO_UB_u: BO_UB_u,
             self.gamma_: gamma_
                             })

    def re_init0(self, w_pre0):
        op = [self.weight[0].assign(w_pre0)]
        return self.sess.run(op)

    def re_init1(self, w_pre1):
        op = [self.weight[1].assign(w_pre1)]
        return self.sess.run(op)

    def re_init2(self, w_pre2):
        op = [self.weight[2].assign(w_pre2)]
        return self.sess.run(op)

    def re_init3(self, w_pre3):
        op = [self.weight[3].assign(w_pre3)]
        return self.sess.run(op)

    def re_init4(self, w_pre4):
        op = [self.weight[4].assign(w_pre4)]
        return self.sess.run(op)
