# ===========================
#   Actor DNN
# ==========================
import tensorflow as tf
class Actor(object):
    def __init__(self, sess, s_dim, u_dim, a_dim, V_dim, EPS_, alpha_, Q1, Q2, R, mode, lb, ub, delt,\
                 BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_, alpha2_):
        self.sess = sess
        self.s_dim = s_dim
        self.s_ext_dim = s_dim + u_dim
        self.V_dim = V_dim
        self.u_dim = u_dim
        self.a_dim = a_dim

        self.mode = mode
        self.EPS_ = EPS_

        self.Q1 = Q1
        self.Q2 = Q2
        self.R = R
        self.lb = lb
        self.ub = ub

        self.LfVss = tf.placeholder(tf.float32, [1, self.V_dim])
        self.Lg1Vss = tf.placeholder(tf.float32, [1, self.V_dim])
        self.Lg2Vss = tf.placeholder(tf.float32, [1, self.V_dim])

        self.s1_ = tf.placeholder(tf.float32, [1, 1])
        self.s2_ = tf.placeholder(tf.float32, [1, 1])
        self.s3_ = tf.placeholder(tf.float32, [1, 1])
        self.s4_ = tf.placeholder(tf.float32, [1, 1])
        self.u1_ = tf.placeholder(tf.float32, [1, 1])
        self.u2_ = tf.placeholder(tf.float32, [1, 1])
        self.delt = delt
        self.z1_ss = tf.placeholder(tf.float32, [1, 1])
        self.z2_ss = tf.placeholder(tf.float32, [1, 1])
        self.z3_ss = tf.placeholder(tf.float32, [1, 1])
        self.z4_ss = tf.placeholder(tf.float32, [1, 1])
        self.u1_ss = tf.placeholder(tf.float32, [1, 1])
        self.u2_ss = tf.placeholder(tf.float32, [1, 1])
        self.alpha_ = tf.placeholder(tf.float32, [1, 1])

        self.BO_LB_z = tf.placeholder(tf.float32, [self.s_dim,])
        self.BO_UB_z = tf.placeholder(tf.float32, [self.s_dim,])
        self.BO_LB_u = tf.placeholder(tf.float32, [self.u_dim,])
        self.BO_UB_u = tf.placeholder(tf.float32, [self.u_dim,])
        self.gamma_ = tf.placeholder(tf.float32, [1, 1])
        self.alpha2_= tf.placeholder(tf.float32, [1, 1])
        self.V2 = tf.placeholder(tf.float32, [1, 1])
        self.a1_out, self.a2_out, self.sqrt_, self.mu_, self.reg_, self.Compare_, self.Cri, self.Check \
            = self.create_action()

    def create_action(self):
       with tf.name_scope("action") as scope:
        z1 = self.s1_  # (N, 1)
        z2 = self.s2_  # (N, 1)
        z3 = self.s3_  # (N, 1)
        z4 = self.s4_  # (N, 1)
        u1 = self.u1_
        u2 = self.u2_
        z1_ss = self.z1_ss
        z2_ss = self.z2_ss
        z3_ss = self.z3_ss
        z4_ss = self.z4_ss
        u1_ss = self.u1_ss
        u2_ss = self.u2_ss

        LfVss = self.LfVss
        Lg1Vss = self.Lg1Vss
        Lg2Vss = self.Lg2Vss

        Q1 = self.Q1
        Q2 = self.Q2
        R_ = self.R

        z_min, z_max = [1.,1.,1.,1.], [28, 28, 28, 28]
        u_min, u_max = [self.lb[1], self.lb[2]], [self.ub[1], self.ub[2]]

        mu_ = tf.placeholder(tf.float32, [2,])

        BO_LB_z = self.BO_LB_z
        BO_UB_z = self.BO_UB_z
        BO_LB_u = self.BO_LB_u
        BO_UB_u = self.BO_UB_u

        V2 = self.V2

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

            self.z1_dev = (z1 - z1_ss) / (z_max_n[0] - z_min_n[0])
            self.z2_dev = (z2 - z2_ss) / (z_max_n[1] - z_min_n[1])
            self.z3_dev = (z3 - z3_ss) / (z_max_n[2] - z_min_n[2])
            self.z4_dev = (z4 - z4_ss) / (z_max_n[3] - z_min_n[3])
            self.u1_dev = (u1 - u1_ss) / (u_max_n[0] - u_min_n[0])
            self.u2_dev = (u2 - u2_ss) / (u_max_n[1] - u_min_n[1])

            h1 = (z1_ss - z_min_n[0])/ (z_max_n[0] - z_min_n[0]) + self.z1_dev
            h2 = (z_max_n[0] - z1_ss)/ (z_max_n[0] - z_min_n[0]) - self.z1_dev
            h3 = (z2_ss - z_min_n[1])/ (z_max_n[1] - z_min_n[1]) + self.z2_dev
            h4 = (z_max_n[1] - z2_ss)/ (z_max_n[1] - z_min_n[1]) - self.z2_dev

            h5 = (z3_ss - z_min_n[2])/ (z_max_n[2] - z_min_n[2]) + self.z3_dev
            h6 = (z_max_n[2] - z3_ss)/ (z_max_n[2] - z_min_n[2]) - self.z3_dev
            h7 = (z4_ss - z_min_n[3])/ (z_max_n[3] - z_min_n[3]) + self.z4_dev
            h8 = (z_max_n[3] - z4_ss)/ (z_max_n[3] - z_min_n[3]) - self.z4_dev

            h9  = (u1_ss - u_min_n[0]) / (u_max_n[0] - u_min_n[0]) + self.u1_dev
            h10 = (u_max_n[0] - u1_ss) / (u_max_n[0] - u_min_n[0]) - self.u1_dev
            h11 = (u2_ss - u_min_n[1]) / (u_max_n[1] - u_min_n[1]) + self.u2_dev
            h12 = (u_max_n[1] - u2_ss) / (u_max_n[1] - u_min_n[1]) - self.u2_dev

            s1 = 1 / ((z1_ss-z_min_n[0]) * (1+z1_ss-z_min_n[0]) / (z_max_n[0]-z1_ss) / (1+z_max_n[0]-z1_ss) + 1)
            s2 = 1 / ((z2_ss-z_min_n[1]) * (1+z2_ss-z_min_n[1]) / (z_max_n[1]-z2_ss) / (1+z_max_n[1]-z2_ss) + 1)
            s3 = 1 / ((z3_ss-z_min_n[2]) * (1+z3_ss-z_min_n[2]) / (z_max_n[2]-z3_ss) / (1+z_max_n[2]-z3_ss) + 1)
            s4 = 1 / ((z4_ss-z_min_n[3]) * (1+z4_ss-z_min_n[3]) / (z_max_n[3]-z4_ss) / (1+z_max_n[3]-z4_ss) + 1)
            s5 = 1 / ((u1_ss-u_min_n[0]) * (1+u1_ss-u_min_n[0]) / (u_max_n[0]-u1_ss) / (1+u_max_n[0]-u1_ss) + 1)
            s6 = 1 / ((u2_ss-u_min_n[1]) * (1+u2_ss-u_min_n[1]) / (u_max_n[1]-u2_ss) / (1+u_max_n[1]-u2_ss) + 1)

            h1_ss, h2_ss = (z1_ss - z_min_n[0]) / (z_max_n[0] - z_min_n[0]), (z_max_n[0] - z1_ss) / (z_max_n[0] - z_min_n[0])
            lambda1 = ((1 - s1) * tf.math.log((h1_ss) / (h1_ss + mu_[1])) \
                           + s1 * tf.math.log((h2_ss) / (h2_ss + mu_[1])))

            h3_ss, h4_ss = (z2_ss - z_min_n[1]) / (z_max_n[1] - z_min_n[1]), (z_max_n[1] - z2_ss) / (z_max_n[1] - z_min_n[1])
            lambda2 = ((1 - s2) * tf.math.log((h3_ss) / (h3_ss + mu_[1])) \
                           + s2 * tf.math.log((h4_ss) / (h4_ss + mu_[1])))

            h5_ss, h6_ss = (z3_ss - z_min_n[2]) / (z_max_n[2] - z_min_n[2]), (z_max_n[2] - z3_ss) / (z_max_n[2] - z_min_n[2])
            lambda3 = ((1 - s3) * tf.math.log((h5_ss) / (h5_ss + mu_[1])) \
                           + s3 * tf.math.log((h6_ss) / (h6_ss + mu_[1])))

            h7_ss, h8_ss = (z4_ss - z_min_n[3]) / (z_max_n[3] - z_min_n[3]), (z_max_n[3] - z4_ss) / (z_max_n[3] - z_min_n[3])
            lambda4 = ((1 - s4) * tf.math.log((h7_ss) / (h7_ss + mu_[1])) \
                           + s4 * tf.math.log((h8_ss) / (h8_ss + mu_[1])))

            h9_ss, h10_ss = (u1_ss - u_min_n[0]) / (u_max_n[0] - u_min_n[0]), (u_max_n[0] - u1_ss) / (u_max_n[0] - u_min_n[0])
            lambda5 = ((1 - s5) * tf.math.log((h9_ss) / (h9_ss + mu_[1])) \
                           + s5 * tf.math.log((h10_ss) / (h10_ss + mu_[1])))

            h11_ss, h12_ss = (u2_ss - u_min_n[1]) / (u_max_n[1] - u_min_n[1]), (u_max_n[1] - u2_ss) / (u_max_n[1] - u_min_n[1])
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

        q_ = Q1 * tf.pow(self.z1_dev, 2) + Q1 * tf.pow(self.z2_dev, 2) \
           + Q1 * tf.pow(self.z3_dev, 2) + Q1 * tf.pow(self.z4_dev, 2) \
           + Q1 * tf.pow(self.u1_dev, 2) + Q1 * tf.pow(self.u2_dev, 2) + reg_

        r1_ = tf.expand_dims([R_],0)
        r2_ = tf.expand_dims([R_],0)
        den = tf.pow(Lg1Vss, 2) / r1_ + tf.pow(Lg2Vss, 2) / r2_ + self.EPS_

        sqrt_ = self.alpha_ * tf.sqrt(tf.pow(LfVss, 2) + (q_ - self.alpha2_/self.gamma_ * V2) * (tf.pow(Lg1Vss, 2) / r1_ + tf.pow(Lg2Vss, 2) / r2_))
        Cri1 = tf.reduce_sum(tf.sqrt(tf.pow(Lg1Vss,2)) - self.EPS_)
        Cri2 = tf.reduce_sum(tf.sqrt(tf.pow(Lg2Vss,2)) - self.EPS_)
        Cri = tf.reduce_sum(tf.sqrt(tf.pow(Lg1Vss, 2)+tf.pow(Lg2Vss, 2)) - self.EPS_)

        a1_out = tf.cond(Cri < 0, lambda: tf.constant(0.),
                        lambda:  tf.reduce_sum(tf.divide(-(LfVss + sqrt_), den) / r1_ * Lg1Vss))
        a2_out = tf.cond(Cri < 0, lambda: tf.constant(0.),
                         lambda: tf.reduce_sum(tf.divide(-(LfVss + sqrt_), den) / r2_ * Lg2Vss))
        Compare_ = tf.reduce_sum(tf.divide(-(LfVss + sqrt_), den))  # have to converge to 1/2
        Check = -0.5 / r1_ * Lg1Vss
        # a = tf.reduce_sum(tf.divide(-(LfVss + sqrt_), den)) # scalar
        # b = tf.divide(-(LfVss + sqrt_), den)                # [[]]
        return a1_out, a2_out, sqrt_, mu_, reg_, [Compare_,den], [Cri1, Cri2], Check

    def action(self, z1_, z2_, z3_, z4_, u1_, u2_, LfVss, Lg1Vss, Lg2Vss, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss, alpha_, mu_, \
               BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_, alpha2_, V2):
        return self.sess.run([self.a1_out, self.a2_out, self.sqrt_, self.reg_, self.Compare_, self.Cri, self.Check], feed_dict={
            self.s1_: z1_,
            self.s2_: z2_,
            self.s3_: z3_,
            self.s4_: z4_,
            self.u1_: u1_,
            self.u2_: u2_,
            self.LfVss: LfVss,
            self.Lg1Vss: Lg1Vss,
            self.Lg2Vss: Lg2Vss,
            self.z1_ss: z1_ss,
            self.z2_ss: z2_ss,
            self.z3_ss: z3_ss,
            self.z4_ss: z4_ss,
            self.u1_ss: u1_ss,
            self.u2_ss: u2_ss,
            self.alpha_: alpha_,
            self.mu_: mu_,
            self.BO_LB_z: BO_LB_z,
            self.BO_UB_z: BO_UB_z,
            self.BO_LB_u: BO_LB_u,
            self.BO_UB_u: BO_UB_u,
            self.gamma_:gamma_,
            self.alpha2_:alpha2_,
            self.V2:V2
        })