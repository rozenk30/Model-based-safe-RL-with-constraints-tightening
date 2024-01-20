from __future__ import division
from __future__ import print_function
from pyomo.environ import *
import tensorflow as tf
import numpy as np
from model_FHOCP3 import model_FHOCP3
from model_FHOCP3_allknown import model_FHOCP3_allknown
from UTIL import *
import scipy.io as sio
from plant3_GP import plant, model
from replay_buffer import ReplayBuffer
from Final_ACTOR_GP import Actor
from Final_CRITIC_GP import VNetwork
import pickle as pickle
import random
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from AL_util import *
from AL_plot_util import *

# FHOCP Solver
MAX_EP_STEPS = 75
delt = 0.5
t_duration = int(MAX_EP_STEPS/delt)
ipopt = SolverFactory('ipopt')
ipopt.options['linear_solver'] = 'ma57'

mismatch = [1.00, 1.00, 1.00, 1.00]

def remove_(sess, z_ss, u_ss, Q1, Q2, R, Hor_n, MAX_EPISODES, lb, ub, delt):
    s1_pool = np.random.uniform(0.5, 1.5, size=(MAX_EPISODES, 1))  # random initial states
    s2_pool = np.random.uniform(0.5, 1.5, size=(MAX_EPISODES, 1))  # random initial states
    s3_pool = np.random.uniform(0.5, 1.5, size=(MAX_EPISODES, 1))  # random initial states
    s4_pool = np.random.uniform(0.5, 1.5, size=(MAX_EPISODES, 1))  # random initial states
    s_pool = np.concatenate([s1_pool * z_ss[0], s2_pool * z_ss[1], s3_pool * z_ss[2], s4_pool * z_ss[3]], axis=1)  # (Max_Epi, 4)
    u1_pool = np.random.uniform(0.5, 1.5, size=(MAX_EPISODES, 1))  # random initial states
    u2_pool = np.random.uniform(0.5, 1.5, size=(MAX_EPISODES, 1))  # random initial states
    u_pool = np.concatenate([u1_pool * u_ss[0], u2_pool * u_ss[1]], axis=1)  # (Max_Epi, 2)
    sio.savemat('total_s.mat', {'state_pool': s_pool, 'u_pool': u_pool})
    z_min, z_max = [1., 1., 1., 1.], [28, 28, 28, 28]
    u_min, u_max = [lb[1], lb[2]], [ub[1], ub[2]]
    ns, nu, delt = 4, 2, delt

    dist_s1_pool = np.random.normal(0., 0.01 * z_ss[0], size=(MAX_EPISODES, int(MAX_EP_STEPS/delt)))  # random disturbance
    dist_s2_pool = np.random.normal(0., 0.01 * z_ss[1], size=(MAX_EPISODES, int(MAX_EP_STEPS/delt)))   # random disturbance
    dist_s3_pool = np.random.normal(0., 0.01 * z_ss[2], size=(MAX_EPISODES, int(MAX_EP_STEPS/delt)))   # random disturbance
    dist_s4_pool = np.random.normal(0., 0.01 * z_ss[3], size=(MAX_EPISODES, int(MAX_EP_STEPS/delt)))   # random disturbance

    # dist_u1_pool = np.random.normal(0., 0.01 * u_ss[0], size=(MAX_EPISODES, 1))  # random disturbance
    # dist_u2_pool = np.random.normal(0., 0.01 * u_ss[1], size=(MAX_EPISODES, 1))  # random disturbance
    # dist_u_pool  = np.concatenate([dist_u1_pool, dist_u2_pool], axis=1)  # (Max_Epi, 2)
    sio.savemat('total_dist.mat', {'dist_s1_pool': dist_s1_pool,
                                   'dist_s2_pool': dist_s2_pool,
                                   'dist_s3_pool': dist_s3_pool,
                                   'dist_s4_pool': dist_s4_pool}) #, 'dist_u_pool': dist_u_pool})

    GAT_i, GAT_i_del = [], []
    GAT_cost = []
    count_GAT = 0

    for i in range(MAX_EPISODES):  # 0, 1, ... 1999
        s0, u0 = [s_pool[i, 0], s_pool[i, 1], s_pool[i, 2], s_pool[i, 3]], [u_pool[i, 0], u_pool[i, 1]] # initial state (4,) (2,)
        pa0 = [0, 0]
        OCP3 = model_FHOCP3(s0, delt, Hor_n, Q1, Q2, R, ns, nu,\
                          np.ones(Hor_n) * z_ss[0], np.ones(Hor_n) * z_ss[1],
                          np.ones(Hor_n) * z_ss[2], np.ones(Hor_n) * z_ss[3],
                          np.ones(Hor_n) * u_ss[0], np.ones(Hor_n) * u_ss[1], u0, lb, ub, pa0, mismatch)
        _, stat, _, e_time, _ = ipopt_solve_with_stats(OCP3, ipopt)
        if stat == True:
            gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2 = [], [], [], [], [], [], [], []
            gat_i, gat_j, gat_stat = [], [], []
            gat_z1_m, gat_z2_m, gat_z3_m, gat_z4_m, gat_u1_m, gat_u2_m     = [], [], [], [], [], []
            for j in range(int(MAX_EP_STEPS/delt)):
                if j == 0:  # initial state of each episode
                    s, u = s0, u0
                    s_m, u_m = s0, u0
                    pu0 = u
                    OCP = model_FHOCP3(s, delt, Hor_n, Q1, Q2, R, ns, nu, \
                                       np.ones(Hor_n) * z_ss[0], np.ones(Hor_n) * z_ss[1],
                                       np.ones(Hor_n) * z_ss[2], np.ones(Hor_n) * z_ss[3],
                                       np.ones(Hor_n) * u_ss[0], np.ones(Hor_n) * u_ss[1], pu0, lb, ub, pa0, mismatch)
                    _, stat, _, e_time, _ = ipopt_solve_with_stats(OCP, ipopt)
                print([i,j], s, u)
                # check if it is possible by NMPC
                try:
                    a1_inject, a2_inject = value(OCP.a[delt * 1, 1]), value(OCP.a[delt * 1, 2])
                    # think about ncp = 1 for input
                    print('ipopt True', stat, a1_inject, a2_inject)
                    if stat == True:
                        gat_z1.append(s[0]), gat_z2.append(s[1]), gat_z3.append(s[2]), gat_z4.append(s[3])
                        gat_u1.append(u[0]), gat_u2.append(u[1])
                        gat_a1.append(a1_inject), gat_a2.append(a2_inject)
                        gat_i.append(i), gat_j.append(j)

                        gat_z1_m.append(s_m[0]), gat_z2_m.append(s_m[1]), gat_z3_m.append(s_m[2]), gat_z4_m.append(s_m[3])
                        gat_u1_m.append(u_m[0]), gat_u2_m.append(u_m[1])
                        # NMPC update
                        dist_s_pool = np.array([dist_s1_pool[i,j], dist_s2_pool[i,j], \
                                                dist_s3_pool[i,j], dist_s4_pool[i,j]])  # (4,)

                        Plant = plant(s, u, delt, ns, nu, [a1_inject, a2_inject], lb, ub, dist_s_pool, [0., 0.]) #dist_u_pool[j,:])
                        Model = model(s, u, delt, ns, nu, [a1_inject, a2_inject], lb, ub, [0.,0.,0.,0.], [0., 0.], mismatch)
                        _, stat_p, _, e_time, _   = ipopt_solve_with_stats(Plant, ipopt)
                        _, stat_m, _, e_time_m, _ = ipopt_solve_with_stats(Model, ipopt)
                        if stat_p == True:
                            s = [value(Plant.z[delt, 1]), value(Plant.z[delt, 2]), value(Plant.z[delt, 3]), value(Plant.z[delt, 4])]
                            u = [value(Plant.u[delt, 1]), value(Plant.u[delt, 2])]

                            s_m = [value(Model.z[delt, 1]), value(Model.z[delt, 2]), value(Model.z[delt, 3]), value(Model.z[delt, 4])]
                            u_m = [value(Model.u[delt, 1]), value(Model.u[delt, 2])]

                            pu0 = u
                            OCP = model_FHOCP3(s, delt, Hor_n, Q1, Q2, R, ns, nu, \
                                               np.ones(Hor_n) * z_ss[0], np.ones(Hor_n) * z_ss[1],
                                               np.ones(Hor_n) * z_ss[2], np.ones(Hor_n) * z_ss[3],
                                               np.ones(Hor_n) * u_ss[0], np.ones(Hor_n) * u_ss[1], pu0, lb, ub, pa0, mismatch)
                            _, stat, _, e_time, _ = ipopt_solve_with_stats(OCP, ipopt)
                        else:
                            print('plant fail')
                            GAT_i_del.append(i)
                            break
                    else:
                        print('ipopt fail')
                        GAT_i_del.append(i)
                        break

                    if j == int(MAX_EP_STEPS/delt)-1:
                        count_GAT = count_GAT + 1
                        if count_GAT == 1:
                            GAT_z1, GAT_z2, GAT_z3, GAT_z4 = [gat_z1], [gat_z2], [gat_z3], [gat_z4]
                            GAT_u1, GAT_u2 = [gat_u1], [gat_u2]
                            GAT_z1_m, GAT_z2_m, GAT_z3_m, GAT_z4_m = [gat_z1_m], [gat_z2_m], [gat_z3_m], [gat_z4_m]
                            GAT_u1_m, GAT_u2_m = [gat_u1_m], [gat_u2_m]
                            GAT_a1, GAT_a2 = [gat_a1], [gat_a2]
                            cost  = delt * (Q1 * (np.array(gat_z1) - z_ss[0]) ** 2 / ((z_max[0] - z_min[0]) ** 2)
                                   + Q1 * (np.array(gat_z2) - z_ss[1]) ** 2 / ((z_max[1] - z_min[1]) ** 2) \
                                   + Q1 * (np.array(gat_z3) - z_ss[2]) ** 2 / ((z_max[2] - z_min[2]) ** 2) \
                                   + Q1 * (np.array(gat_z4) - z_ss[3]) ** 2 / ((z_max[3] - z_min[3]) ** 2) \
                                   + Q2 * (np.array(gat_u1) - u_ss[0]) ** 2 / ((u_max[0] - u_min[0]) ** 2) \
                                   + Q2 * (np.array(gat_u2) - u_ss[1]) ** 2 / ((u_max[1] - u_min[1]) ** 2) \
                                   + R * (np.array(gat_a1)) ** 2 + R * (np.array(gat_a2)) ** 2)
                            cost_i = np.sum(cost)
                            GAT_cost.append(cost_i), GAT_i.append(i)
                        else:
                            GAT_z1, GAT_z2 = np.concatenate((GAT_z1, [gat_z1])), np.concatenate((GAT_z2, [gat_z2]))
                            GAT_z3, GAT_z4 = np.concatenate((GAT_z3, [gat_z3])), np.concatenate((GAT_z4, [gat_z4]))
                            GAT_u1, GAT_u2 = np.concatenate((GAT_u1, [gat_u1])), np.concatenate((GAT_u2, [gat_u2]))
                            GAT_z1_m, GAT_z2_m = np.concatenate((GAT_z1_m, [gat_z1_m])), np.concatenate((GAT_z2_m, [gat_z2_m]))
                            GAT_z3_m, GAT_z4_m = np.concatenate((GAT_z3_m, [gat_z3_m])), np.concatenate((GAT_z4_m, [gat_z4_m]))
                            GAT_u1_m, GAT_u2_m = np.concatenate((GAT_u1_m, [gat_u1_m])), np.concatenate((GAT_u2_m, [gat_u2_m]))
                            GAT_a1, GAT_a2 = np.concatenate((GAT_a1, [gat_a1])), np.concatenate((GAT_a2, [gat_a2]))
                            cost  = delt * (Q1 * (np.array(gat_z1) - z_ss[0]) ** 2 / ((z_max[0] - z_min[0]) ** 2)
                                   + Q1 * (np.array(gat_z2) - z_ss[1]) ** 2 / ((z_max[1] - z_min[1]) ** 2) \
                                   + Q1 * (np.array(gat_z3) - z_ss[2]) ** 2 / ((z_max[2] - z_min[2]) ** 2) \
                                   + Q1 * (np.array(gat_z4) - z_ss[3]) ** 2 / ((z_max[3] - z_min[3]) ** 2) \
                                   + Q2 * (np.array(gat_u1) - u_ss[0]) ** 2 / ((u_max[0] - u_min[0]) ** 2) \
                                   + Q2 * (np.array(gat_u2) - u_ss[1]) ** 2 / ((u_max[1] - u_min[1]) ** 2) \
                                   + R * (np.array(gat_a1)) ** 2 + R * (np.array(gat_a2)) ** 2)
                            cost_i = np.sum(cost)
                            GAT_cost.append(cost_i), GAT_i.append(i)
                            # plot_(gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2)

                except ValueError:
                    GAT_i_del.append(i)
                    print('ValueError')
                    break
        else:
            GAT_i_del.append(i)
            print('ipopt fail')

        print(np.shape(GAT_i_del))
        # one episode ends
        if i > 0:
            if i % 50 == 0:
                sio.savemat('RL{0}_del_GP.mat'.format(i),
                            {'Gat_z1': GAT_z1, 'Gat_z2': GAT_z2, 'Gat_z3': GAT_z3, 'Gat_z4': GAT_z4,
                             'Gat_u1': GAT_u1, 'Gat_u2': GAT_u2,
                             'Gat_z1_m': GAT_z1_m, 'Gat_z2_m': GAT_z2_m, 'Gat_z3_m': GAT_z3_m, 'Gat_z4_m': GAT_z4_m,
                             'Gat_u1_m': GAT_u1_m, 'Gat_u2_m': GAT_u2_m,
                             'Gat_a1': GAT_a1, 'Gat_a2': GAT_a2,
                             'Gat_i' : GAT_i,  'Gat_i_del': GAT_i_del, 'Gat_cost': GAT_cost
                             })
    #all episodes end
    sio.savemat('RL{0}_del_GP.mat'.format(i),
                {'Gat_z1': GAT_z1, 'Gat_z2': GAT_z2, 'Gat_z3': GAT_z3, 'Gat_z4': GAT_z4,
                 'Gat_u1': GAT_u1, 'Gat_u2': GAT_u2,
                 'Gat_z1_m': GAT_z1_m, 'Gat_z2_m': GAT_z2_m, 'Gat_z3_m': GAT_z3_m, 'Gat_z4_m': GAT_z4_m,
                 'Gat_u1_m': GAT_u1_m, 'Gat_u2_m': GAT_u2_m,
                 'Gat_a1': GAT_a1, 'Gat_a2': GAT_a2,
                 'Gat_i': GAT_i,   'Gat_i_del': GAT_i_del, 'Gat_cost': GAT_cost,
                 })


def train(sess, actor, critic, z_ss, u_ss, alpha_, Q1, Q2, R, mode, EPS_, EPS_CLF, mu_, MAX_EPISODES, lb, ub, delt,\
              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, EPS_C, gamma_, alpha2_):
    A = sio.loadmat('total_s.mat')['state_pool']
    B = sio.loadmat('total_s.mat')['u_pool']

    C = sio.loadmat('total_dist.mat')['dist_s1_pool']
    D = sio.loadmat('total_dist.mat')['dist_s2_pool']
    E = sio.loadmat('total_dist.mat')['dist_s3_pool']
    F = sio.loadmat('total_dist.mat')['dist_s4_pool']
    # D = sio.loadmat('total_dist.mat')['dist_u_pool']
    MPC_ = sio.loadmat('RL{0}_del_GP.mat'.format(MAX_EPISODES-1))
    del_i = MPC_['Gat_i_del']
    if np.size(del_i) > 0:
        AA = np.delete(A, np.squeeze(del_i), axis=0)
        BB = np.delete(B, np.squeeze(del_i), axis=0)
        s_pool, u_pool = AA, BB

        CC = np.delete(C, np.squeeze(del_i), axis=0)
        DD = np.delete(D, np.squeeze(del_i), axis=0)
        EE = np.delete(E, np.squeeze(del_i), axis=0)
        FF = np.delete(F, np.squeeze(del_i), axis=0)
        # DD = np.delete(D, np.squeeze(del_i), axis=0)
        dist_s1_pool = CC #, dist_u_pool = CC, DD
        dist_s2_pool = DD
        dist_s3_pool = EE
        dist_s4_pool = FF
    else:
        s_pool = A
        u_pool = B

    MAX_EPISODES = len(s_pool)
    sio.savemat('train_s.mat', {'state_pool': s_pool, 'u_pool': u_pool})
    MPC_z1, MPC_z2, MPC_z3, MPC_z4, MPC_u1, MPC_u2, MPC_a1, MPC_a2, MPC_cost \
                   = MPC_['Gat_z1'], MPC_['Gat_z2'], MPC_['Gat_z3'], MPC_['Gat_z4'], MPC_['Gat_u1'], MPC_['Gat_u2'], \
                     MPC_['Gat_a1'], MPC_['Gat_a2'], MPC_['Gat_cost']
    V_LEARNING_RATE = 0.003


    GAT_HJB_cost, RL_cost = [], []
    bad_hist = [0, 0, 0]
    count, count_GAT, count_GP = 0, 0, 0
    ns, nu, delt = 4, 2, delt
    train_duration = 75
    MINIBATCH_SIZE       = int(MAX_EP_STEPS/delt)*3 #150
    MINIBATCH_SIZE_GP    = 10
    REPLAY_START_SIZE    = int(MAX_EP_STEPS/delt)*3
    REPLAY_START_SIZE_GP = 10
    BUFFER_SIZE    = int(MAX_EP_STEPS/delt)*3*5
    BUFFER_SIZE_GP = int(MAX_EP_STEPS/delt)*10
    print(BUFFER_SIZE_GP)
    REPLAY_MAX_SIZE_GP = BUFFER_SIZE_GP
    z_min, z_max = [1., 1., 1., 1.], [28, 28, 28, 28]
    u_min, u_max = [lb[1], lb[2]], [ub[1], ub[2]]

    # defining the kernel for the Gaussian process
    kernel_Y1 = Matern(length_scale=1.0)
    regressor_Y1 = GaussianProcessRegressor(kernel=kernel_Y1)
    kernel_Y2 = Matern(length_scale=1.0)
    regressor_Y2 = GaussianProcessRegressor(kernel=kernel_Y2)
    kernel_Y3 = Matern(length_scale=1.0)
    regressor_Y3 = GaussianProcessRegressor(kernel=kernel_Y3)
    kernel_Y4 = Matern(length_scale=1.0)
    regressor_Y4 = GaussianProcessRegressor(kernel=kernel_Y4)
    # kernel_Y5 = Matern(length_scale=1.0)
    # regressor_Y5 = GaussianProcessRegressor(kernel=kernel_Y5)
    # kernel_Y6 = Matern(length_scale=1.0)
    # regressor_Y6 = GaussianProcessRegressor(kernel=kernel_Y6)

    # initialization of V
    for ini in range(1000):
        sess.run(tf.global_variables_initializer())
        W_pre = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        W_pre0 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'weights_posdef_0'))
        W_pre1 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'weights_posdef_1'))
        W_pre2 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'weights_1'))
        print(np.shape(W_pre))
        print(np.shape(W_pre0))
        print(np.shape(W_pre1))
        print(np.shape(W_pre2))
        #W_pre2 = np.clip(W_pre2, 0, np.infty) # clip
        #critic.re_init2(W_pre2[0])

        z_min_n = [z_min[0] + BO_LB_z[0],
                   z_min[1] + BO_LB_z[1],
                   z_min[2] + BO_LB_z[2],
                   z_min[3] + BO_LB_z[3]]
        z_max_n = [z_max[0] - BO_UB_z[0],
                   z_max[1] - BO_UB_z[1],
                   z_max[2] - BO_UB_z[2],
                   z_max[3] - BO_UB_z[3]]
        u_min_n = [u_min[0] + 0,
                   u_min[1] + 0]
        u_max_n = [u_max[0] - 0,
                   u_max[1] - 0]

        r_EPS = 0.01
        z1_range = np.linspace(z_min_n[0]+r_EPS, z_max_n[0]-r_EPS, 10)
        z2_range = np.linspace(z_min_n[1]+r_EPS, z_max_n[1]-r_EPS, 10)
        z3_range = np.linspace(z_min_n[2]+r_EPS, z_max_n[2]-r_EPS, 10)
        z4_range = np.linspace(z_min_n[3]+r_EPS, z_max_n[3]-r_EPS, 10)
        u1_range = np.linspace(u_min_n[0]+r_EPS, u_max_n[0]-r_EPS, 10)
        u2_range = np.linspace(u_min_n[1]+r_EPS, u_max_n[1]-r_EPS, 10)
        Z1, Z2, Z3, Z4, U1, U2 = np.meshgrid(z1_range, z2_range, z3_range, z4_range, u1_range, u2_range, indexing='ij')

        mesh_batchS1, mesh_batchS2, mesh_batchS3, mesh_batchS4 \
            = np.reshape(Z1, [-1, 1]), np.reshape(Z2, [-1, 1]), np.reshape(Z3, [-1, 1]), np.reshape(Z4, [-1, 1])  # Z1, Z2
        mesh_batchU1, mesh_batchU2 = np.reshape(U1, [-1, 1]), np.reshape(U2, [-1, 1])  # U1, U2
        LfV_mesh, Lg1V_mesh, Lg2V_mesh, gradV2 = critic.cal_Lie2(mesh_batchS1, mesh_batchS2, mesh_batchS3, mesh_batchS4, \
                                         mesh_batchU1, mesh_batchU2, \
                                         np.ones_like(mesh_batchS1)*z_ss[0], np.ones_like(mesh_batchS1)*z_ss[1],\
                                         np.ones_like(mesh_batchS1)*z_ss[2], np.ones_like(mesh_batchS1)*z_ss[3],\
                                         np.ones_like(mesh_batchU1)*u_ss[0], np.ones_like(mesh_batchU1)*u_ss[1], mu_,\
                                         BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)
        bad = check_CLF(LfV_mesh, Lg1V_mesh, Lg2V_mesh, z1_range, z2_range, z3_range, z4_range, \
                        u1_range, u2_range, EPS_CLF, R)
        if bad.size > 0:
            sess.run(tf.global_variables_initializer())
            #W_pre2 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='weights_1'))
            #W_pre2 = np.clip(W_pre2, 0, np.infty)  # clip
            #critic.re_init2(W_pre2[0])
        else:
            print('initial set')
            break

    replay_buffer    = ReplayBuffer(BUFFER_SIZE)
    replay_buffer_GP = ReplayBuffer(BUFFER_SIZE_GP)
    for i in range(MAX_EPISODES):
        # initial condition setting for u
        s0, u0 = [MPC_z1[i,0], MPC_z2[i,0], MPC_z3[i,0], MPC_z4[i,0]], [MPC_u1[i,0], MPC_u2[i,0]]
        gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2 = [], [], [], [], [], [], [], []
        gat_i, gat_j, gat_stat_p = [], [], []
        #gat_Compare_ = []

        gat_e1, gat_e2, gat_e3, gat_e4 = [], [], [], []
        gat_e1_pred, gat_e2_pred, gat_e3_pred, gat_e4_pred = [], [], [], []
        gat_e1_std, gat_e2_std, gat_e3_std, gat_e4_std = [], [], [], []
        gat_z1_m, gat_z2_m, gat_z3_m, gat_z4_m, gat_u1_m, gat_u2_m = [], [], [], [], [], []
        gat_stat_m = []

        gat_z1_BO_U, gat_z2_BO_U, gat_z3_BO_U, gat_z4_BO_U, gat_z1_BO_L, gat_z2_BO_L, gat_z3_BO_L, gat_z4_BO_L \
             = [], [], [], [], [], [], [], []

        gat_a1_son_org, gat_a2_son_org, gat_a1_opt_org, gat_a2_opt_org = [], [], [], []
        gat_a1_son, gat_a2_son, gat_a1_opt, gat_a2_opt = [], [], [], []
        gat_V_dot_son, gat_V_dot_opt = [], []
        gat_LfV, gat_Lg1V, gat_Lg2V = [], [], []
        gat_q, gat_V1, gat_V2 = [], [], []

        for j in range(int(MAX_EP_STEPS/delt)):
            if j == 0:  # initial state of each episode
                s, u = s0, u0
                s_m, u_m = s0, u0

                e_s = [s[0] - s_m[0], s[1] - s_m[1], s[2] - s_m[2], s[3] - s_m[3]]  # (s_prev, u_prev, a_prev, e_s)
                pred_Y1, pred_Y2, pred_Y3, pred_Y4 = 0 , 0 , 0 , 0

                e_pred = [pred_Y1, pred_Y2, pred_Y3, pred_Y4]

                std_Y1, std_Y2, std_Y3, std_Y4 = 0 , 0 , 0 , 0

                e_std = [std_Y1, std_Y2, std_Y3, std_Y4]

                W = critic.cal_W([[s[0]]], [[s[1]]], [[s[2]]], [[s[3]]], [[u[0]]], [[u[1]]], [[z_ss[0]]], [[z_ss[1]]],
                                 [[z_ss[2]]], [[z_ss[3]]],[[u_ss[0]]], [[u_ss[1]]], mu_,\
                                 BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)

                gat_e1.append(e_s[0]), gat_e2.append(e_s[1]), gat_e3.append(e_s[2]), gat_e4.append(e_s[3])
                gat_e1_pred.append(e_pred[0]), gat_e2_pred.append(e_pred[1])
                gat_e3_pred.append(e_pred[2]), gat_e4_pred.append(e_pred[3])
                gat_e1_std.append(e_std[0]), gat_e2_std.append(e_std[1])
                gat_e3_std.append(e_std[2]), gat_e4_std.append(e_std[3])


            #### calculate BO ####
            if j > 0:  # at the first time, there is no e_s
                replay_buffer_GP.add([s_prev[0]], [s_prev[1]], [s_prev[2]], [s_prev[3]], \
                                     [u_prev[0]], [u_prev[1]], [a_prev[0]], [a_prev[1]], \
                                     [e_s[0]], [e_s[1]], [e_s[2]], [e_s[3]], [e_u[0]], [e_u[1]])

            print(replay_buffer_GP.count())
            if replay_buffer_GP.count() > REPLAY_START_SIZE_GP:
                count_GP = count_GP + 1
                if count_GP == 1:
                    s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_e_batch, z2_e_batch, z3_e_batch, z4_e_batch, u1_e_batch, u2_e_batch \
                        = data_get(replay_buffer_GP, MINIBATCH_SIZE_GP)

                    X_init = np.hstack((s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch))
                    # , a1_batch,a2_batch)
                    Y1_init, Y2_init, Y3_init, Y4_init = \
                        z1_e_batch, z2_e_batch, z3_e_batch, z4_e_batch  # , u1_e_batch, u2_e_batch

                    print('count_GP', count_GP, np.shape(X_init), np.shape(Y1_init))  # (10, 6)
                    print('Y1_init', Y1_init)  # (10, 6)
                    optimizer_Y1 = BayesianOptimizer(
                        estimator=regressor_Y1,
                        X_training=X_init, y_training=Y1_init  # ,
                        # query_strategy=max_EI
                    )
                    optimizer_Y2 = BayesianOptimizer(
                        estimator=regressor_Y2,
                        X_training=X_init, y_training=Y2_init  # ,
                        # query_strategy=max_EI
                    )
                    optimizer_Y3 = BayesianOptimizer(
                        estimator=regressor_Y3,
                        X_training=X_init, y_training=Y3_init  # ,
                        # query_strategy=max_EI
                    )
                    optimizer_Y4 = BayesianOptimizer(
                        estimator=regressor_Y4,
                        X_training=X_init, y_training=Y4_init  # ,
                        # query_strategy=max_EI
                    )
                    # optimizer_Y5 = BayesianOptimizer(
                    #     estimator=regressor_Y5,
                    #     X_training=X_init, y_training=Y5_init #,
                    #    # query_strategy=max_EI
                    # )
                    # optimizer_Y6 = BayesianOptimizer(
                    #     estimator=regressor_Y6,
                    #     X_training=X_init, y_training=Y6_init #,
                    #    # query_strategy=max_EI
                    # )
                elif replay_buffer_GP.count() > REPLAY_MAX_SIZE_GP-1:
                    s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_e_batch, z2_e_batch, z3_e_batch, z4_e_batch, u1_e_batch, u2_e_batch \
                        = data_get(replay_buffer_GP, REPLAY_MAX_SIZE_GP)

                    X_init = np.hstack((s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch))
                    # , a1_batch,a2_batch)
                    Y1_init, Y2_init, Y3_init, Y4_init = \
                        z1_e_batch, z2_e_batch, z3_e_batch, z4_e_batch  # , u1_e_batch, u2_e_batch
                    print('here')
                    optimizer_Y1 = BayesianOptimizer(
                        estimator=regressor_Y1,
                        X_training=X_init, y_training=Y1_init  # ,
                        # query_strategy=max_EI
                    )
                    optimizer_Y2 = BayesianOptimizer(
                        estimator=regressor_Y2,
                        X_training=X_init, y_training=Y2_init  # ,
                        # query_strategy=max_EI
                    )
                    optimizer_Y3 = BayesianOptimizer(
                        estimator=regressor_Y3,
                        X_training=X_init, y_training=Y3_init  # ,
                        # query_strategy=max_EI
                    )
                    optimizer_Y4 = BayesianOptimizer(
                        estimator=regressor_Y4,
                        X_training=X_init, y_training=Y4_init  # ,
                        # query_strategy=max_EI
                    )
                elif replay_buffer_GP.count() < REPLAY_MAX_SIZE_GP - 1:
                    X_train = np.array([s_prev[0], s_prev[1], s_prev[2], s_prev[3], u_prev[0], u_prev[1]]).reshape(
                        1, -1)  # , [a1_inject], [a2_inject]]
                    Y1_train = np.array([e_s[0]]).reshape(1, -1)
                    Y2_train = np.array([e_s[1]]).reshape(1, -1)
                    Y3_train = np.array([e_s[2]]).reshape(1, -1)
                    Y4_train = np.array([e_s[3]]).reshape(1, -1)
                    # Y5_train = np.array([e_u[0]]).reshape(1, -1)
                    # Y6_train = np.array([e_u[1]]).reshape(1, -1)

                    print(np.shape(X_train), np.shape(Y1_train))

                    optimizer_Y1.teach(X_train.reshape(1, -1), Y1_train)
                    optimizer_Y2.teach(X_train.reshape(1, -1), Y2_train)
                    optimizer_Y3.teach(X_train.reshape(1, -1), Y3_train)
                    optimizer_Y4.teach(X_train.reshape(1, -1), Y4_train)
                    # optimizer_Y5.teach(X_train.reshape(1, -1), Y5_train)
                    # optimizer_Y6.teach(X_train.reshape(1, -1), Y6_train)

            if replay_buffer_GP.count() > REPLAY_START_SIZE_GP:
                X_test = np.array([s[0], s[1], s[2], s[3], u[0], u[1]]).reshape(1, -1)
                pred_Y1, std_Y1 = optimizer_Y1.predict(X_test, return_std=True)
                pred_Y2, std_Y2 = optimizer_Y2.predict(X_test, return_std=True)
                pred_Y3, std_Y3 = optimizer_Y3.predict(X_test, return_std=True)
                pred_Y4, std_Y4 = optimizer_Y4.predict(X_test, return_std=True)
                # pred_Y5, std_Y5 = optimizer_Y5.predict(X_test, return_std=True)
                # pred_Y6, std_Y6 = optimizer_Y6.predict(X_test, return_std=True)

                print(pred_Y1, std_Y1)

                BO_UB_z1, BO_LB_z1 = np.max(pred_Y1 + 2.58 * std_Y1, 0), np.max(2.58 * std_Y1 - pred_Y1, 0)
                BO_UB_z2, BO_LB_z2 = np.max(pred_Y2 + 2.58 * std_Y2, 0), np.max(2.58 * std_Y2 - pred_Y2, 0)
                BO_UB_z3, BO_LB_z3 = np.max(pred_Y3 + 2.58 * std_Y3, 0), np.max(2.58 * std_Y3 - pred_Y3, 0)
                BO_UB_z4, BO_LB_z4 = np.max(pred_Y4 + 2.58 * std_Y4, 0), np.max(2.58 * std_Y4 - pred_Y4, 0)
                # BO_UB_u1, BO_LB_u1 = np.max(pred_Y5 + 2.58*std_Y5, 0), np.max(2.58*std_Y5 - pred_Y5, 0)
                # BO_UB_u2, BO_LB_u2 = np.max(pred_Y6 + 2.58*std_Y6, 0), np.max(2.58*std_Y6 - pred_Y6, 0)

                BO_UB_z, BO_UB_u, BO_LB_z, BO_LB_u = [BO_UB_z1[0], BO_UB_z2[0], BO_UB_z3[0], BO_UB_z4[0]], \
                                                     [0, 0], \
                                                     [BO_LB_z1[0], BO_LB_z2[0], BO_LB_z3[0], BO_LB_z4[0]], \
                                                     [0, 0]
            # [BO_UB_u1[0], BO_UB_u2[0]], \
            #     [BO_LB_u1[0], BO_LB_u2[0]]
            print('BO_UB_z', BO_UB_z)
            print('BO_LB_z', BO_LB_z)

            print('z1', z_min_n[0], s[0], z_max_n[0])
            print('z2', z_min_n[1], s[1], z_max_n[1])
            print('z3', z_min_n[2], s[2], z_max_n[2])
            print('z4', z_min_n[3], s[3], z_max_n[3])
            print('u1', u_min_n[0], u[0], u_max_n[0])
            print('u2', u_min_n[1], u[1], u_max_n[1])

            z_min_n = [z_min[0] + BO_LB_z[0],
                       z_min[1] + BO_LB_z[1],
                       z_min[2] + BO_LB_z[2],
                       z_min[3] + BO_LB_z[3]]
            z_max_n = [z_max[0] - BO_UB_z[0],
                       z_max[1] - BO_UB_z[1],
                       z_max[2] - BO_UB_z[2],
                       z_max[3] - BO_UB_z[3]]
            u_min_n = [u_min[0] + 0.,
                       u_min[1] + 0.]
            u_max_n = [u_max[0] - 0.,
                       u_max[1] - 0.]

            r_EPS = 0.01
            z1_range = np.linspace(z_min_n[0] + r_EPS, z_max_n[0] - r_EPS, 10)
            z2_range = np.linspace(z_min_n[1] + r_EPS, z_max_n[1] - r_EPS, 10)
            z3_range = np.linspace(z_min_n[2] + r_EPS, z_max_n[2] - r_EPS, 10)
            z4_range = np.linspace(z_min_n[3] + r_EPS, z_max_n[3] - r_EPS, 10)
            u1_range = np.linspace(u_min_n[0] + r_EPS, u_max_n[0] - r_EPS, 10)
            u2_range = np.linspace(u_min_n[1] + r_EPS, u_max_n[1] - r_EPS, 10)
            Z1, Z2, Z3, Z4, U1, U2 = np.meshgrid(z1_range, z2_range, z3_range, z4_range, u1_range, u2_range,
                                                 indexing='ij')

            mesh_batchS1, mesh_batchS2, mesh_batchS3, mesh_batchS4 \
                = np.reshape(Z1, [-1, 1]), np.reshape(Z2, [-1, 1]), np.reshape(Z3, [-1, 1]), np.reshape(Z4, [-1,
                                                                                                             1])  # Z1, Z2
            mesh_batchU1, mesh_batchU2 = np.reshape(U1, [-1, 1]), np.reshape(U2, [-1, 1])  # U1, U2
            LfV_mesh, Lg1V_mesh, Lg2V_mesh, gradV2 = critic.cal_Lie2(mesh_batchS1, mesh_batchS2, mesh_batchS3,
                                                                     mesh_batchS4, \
                                                                     mesh_batchU1, mesh_batchU2, \
                                                                     np.ones_like(mesh_batchS1) * z_ss[0],
                                                                     np.ones_like(mesh_batchS1) * z_ss[1], \
                                                                     np.ones_like(mesh_batchS1) * z_ss[2],
                                                                     np.ones_like(mesh_batchS1) * z_ss[3], \
                                                                     np.ones_like(mesh_batchU1) * u_ss[0],
                                                                     np.ones_like(mesh_batchU1) * u_ss[1], mu_, \
                                                                     BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)

            ## calculate BO

            # fix BO
            BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u \
                = FIX_BO(s, u, z_min, z_max, u_min, u_max, BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, EPS_C)

            gat_z1_BO_U.append(BO_UB_z[0])
            gat_z2_BO_U.append(BO_UB_z[1])
            gat_z3_BO_U.append(BO_UB_z[2])
            gat_z4_BO_U.append(BO_UB_z[3])
            gat_z1_BO_L.append(BO_LB_z[0])
            gat_z2_BO_L.append(BO_LB_z[1])
            gat_z3_BO_L.append(BO_LB_z[2])
            gat_z4_BO_L.append(BO_LB_z[3])


            z_min_n = [z_min[0] + BO_LB_z[0],
                       z_min[1] + BO_LB_z[1],
                       z_min[2] + BO_LB_z[2],
                       z_min[3] + BO_LB_z[3]]
            z_max_n = [z_max[0] - BO_UB_z[0],
                       z_max[1] - BO_UB_z[1],
                       z_max[2] - BO_UB_z[2],
                       z_max[3] - BO_UB_z[3]]
            u_min_n = [u_min[0] + 0.,
                       u_min[1] + 0.]
            u_max_n = [u_max[0] - 0.,
                       u_max[1] - 0.]


            LfV2, Lg1V2, Lg2V2, gradV2 = critic.cal_Lie2([[s[0]]], [[s[1]]], [[s[2]]], [[s[3]]], [[u[0]]], [[u[1]]],\
                                     [[z_ss[0]]], [[z_ss[1]]], [[z_ss[2]]], [[z_ss[3]]], [[u_ss[0]]], [[u_ss[1]]], mu_,\
                                     BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)

            V1, V2, _, q = critic.cal_V([[s[0]]], [[s[1]]], [[s[2]]], [[s[3]]], [[u[0]]], [[u[1]]],\
                                     [[z_ss[0]]], [[z_ss[1]]], [[z_ss[2]]], [[z_ss[3]]], [[u_ss[0]]], [[u_ss[1]]], mu_,\
                                     BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)
            print('q', q, 'V2', V2)
            # here
            action1, action2, sqrt_, add_, Compare_, Cri, Check \
                   = actor.action([[s[0]]], [[s[1]]], [[s[2]]], [[s[3]]], [[u[0]]], [[u[1]]],\
                       np.reshape(LfV2,  (1, actor.V_dim)),  \
                       np.reshape(Lg1V2, (1, actor.V_dim)), np.reshape(Lg2V2, (1, actor.V_dim)), \
                       [[z_ss[0]]],[[z_ss[1]]],[[z_ss[2]]],[[z_ss[3]]],[[u_ss[0]]],[[u_ss[1]]], [[alpha_]], mu_,\
                       BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, \
                       gamma_, [[alpha2_]], np.reshape(V2, (1,1)))
            print(sqrt_)
            # sqrt issue : (q-1/gamma V)
            alpha2_re = alpha2_
            for cheche in range(10 ** 3):
                if is_nan(sqrt_):

                    action1, action2, sqrt_, add_, Compare_, Cri, Check \
                        = actor.action([[s[0]]], [[s[1]]], [[s[2]]], [[s[3]]], [[u[0]]], [[u[1]]], \
                                       np.reshape(LfV2, (1, actor.V_dim)), \
                                       np.reshape(Lg1V2, (1, actor.V_dim)), np.reshape(Lg2V2, (1, actor.V_dim)), \
                                       [[z_ss[0]]], [[z_ss[1]]], [[z_ss[2]]], [[z_ss[3]]], [[u_ss[0]]], [[u_ss[1]]],
                                       [[alpha_]], mu_, \
                                       BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, \
                                       gamma_, [[alpha2_re]], np.reshape(V2, (1, 1)))
                    alpha2_re = alpha2_re / 2
                    print('alpha reduce for sqrt issue')

                else:
                    print('no clip for sqrt issue')
                    break

            action1_opt, action2_opt = -0.5 * Lg1V2 / R, -0.5 * Lg2V2 / R
            gat_a1_son_org.append(action1), gat_a2_son_org.append(action2)
            gat_a1_opt_org.append(action1_opt), gat_a2_opt_org.append(action2_opt)
            print([i,j], s[0]-z_ss[0], s[1]-z_ss[1], s[2]-z_ss[2], s[3]-z_ss[3], \
                  u[0]-u_ss[0], u[1]-u_ss[1])
            print('compare -1/2', Compare_, 'Cri for 0', Cri)
            print('LFV2:', LfV2, 'LG1V2:', Lg1V2, 'LG2V2:',Lg2V2, 'sqrt', sqrt_, 'add', add_, 'gradV2', gradV2)
            set_cri = 0.01
            if np.abs(s[0]-z_ss[0]) < (z_max[0]-z_min[0]) * set_cri and np.abs(s[1]-z_ss[1]) < (z_max[1]-z_min[1]) * set_cri:
                if np.abs(s[2] - z_ss[2]) < (z_max[2]-z_min[2]) * set_cri and np.abs(s[3] - z_ss[3]) < (z_max[3]-z_min[3]) * set_cri:
                    if np.abs(u[0] - u_ss[0]) < (ub[1]-lb[1]) * set_cri and np.abs(u[1] - u_ss[1]) < (ub[2]-lb[2]) * set_cri:
                        print('settle')
                        action1, action2 = (EPS_ * action1), (EPS_ * action2)
                        action1_opt, action2_opt = -0.5 * EPS_ * Lg1V2 / R, -0.5 * EPS_ * Lg2V2 / R

            print('before clip', action1, action2, Lg1V2, Lg2V2)
            alpha_re = alpha_
            for cheche in range(10 ** 3):
                if u[0] + action1 * delt < u_min_n[0] or u[0] + action1 * delt > u_max_n[0] or \
                   u[1] + action2 * delt < u_min_n[1] or u[1] + action2 * delt > u_max_n[1]:

                    action1, action2, sqrt_, add_, Compare_, Cri, Check \
                        = actor.action([[s[0]]], [[s[1]]], [[s[2]]], [[s[3]]], [[u[0]]], [[u[1]]], \
                           np.reshape(LfV2, (1, actor.V_dim)), \
                           np.reshape(Lg1V2, (1, actor.V_dim)), np.reshape(Lg2V2, (1, actor.V_dim)), \
                           [[z_ss[0]]], [[z_ss[1]]],[[z_ss[2]]],[[z_ss[3]]],[[u_ss[0]]],[[u_ss[1]]],[[alpha_re]], mu_,\
                           BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, \
                           gamma_, [[alpha2_]], np.reshape(V2, (1,1)))
                    alpha_re = alpha_re / 10
                    action1_opt, action2_opt = -0.5*alpha_re*Lg1V2/R, -0.5*alpha_re*Lg2V2/R
                    print('alpha reduce')
                else:
                    print('no clip')
                    break

            alpha_re = alpha_
            for cheche in range(10 ** 3):
                if u[0] + action1_opt * delt < u_min_n[0] or u[0] + action1_opt * delt > u_max_n[0] or \
                   u[1] + action2_opt * delt < u_min_n[1] or u[1] + action2_opt * delt > u_max_n[1]:

                    action1_opt, action2_opt = -0.5 * alpha_re * Lg1V2 / R, -0.5 * alpha_re * Lg2V2 / R
                    alpha_re = alpha_re / 10
                    print('alpha reduce for LgV form')
                else:
                    print('no clip for LgV form')
                    break

            a1_inject, a2_inject = action1, action2
            print('after clip', a1_inject, a2_inject)
            gat_a1_son.append(action1), gat_a2_son.append(action2)
            gat_a1_opt.append(action1_opt), gat_a2_opt.append(action2_opt)
            V_dot_son, V_dot_opt = LfV2 + Lg1V2 * action1 + Lg2V2 * action2, \
                                   LfV2 + Lg1V2 * action1_opt + Lg2V2 * action2_opt
            gat_V_dot_son.append(V_dot_son), gat_V_dot_opt.append(V_dot_opt)
            gat_LfV.append(LfV2), gat_Lg1V.append(Lg1V2), gat_Lg2V.append(Lg2V2)

            gat_q.append(q), gat_V1.append(V1), gat_V2.append(V2)
            print(V_dot_son, V_dot_opt)
            print(action1, action1_opt, action2, action2_opt, 'here', action1-action1_opt, action2-action2_opt)

            gat_z1.append(s[0]), gat_z2.append(s[1]), gat_z3.append(s[2]), gat_z4.append(s[3]), \
            gat_u1.append(u[0]), gat_u2.append(u[1])
            gat_z1_m.append(s_m[0]), gat_z2_m.append(s_m[1]), gat_z3_m.append(s_m[2]), gat_z4_m.append(s_m[3])
            gat_u1_m.append(u_m[0]), gat_u2_m.append(u_m[1])
            gat_a1.append(a1_inject), gat_a2.append(a2_inject), gat_i.append(i), gat_j.append(j)
            #gat_Compare_.append(Compare_)
            HJB_cost = 1.
            replay_buffer.add([s[0]], [s[1]], [s[2]], [s[3]], [u[0]], [u[1]], [a1_inject], [a2_inject], \
                              [z_ss[0]], [z_ss[1]], [z_ss[2]], [z_ss[3]], [u_ss[0]], [u_ss[1]])



            if replay_buffer.count() > REPLAY_START_SIZE and i < train_duration:
                print('train')
                s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch \
                    = data_get(replay_buffer, MINIBATCH_SIZE)

                W_pre = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                V_LEARNING_RATE_med = V_LEARNING_RATE
                for cheche in range(10):
                    warning_b(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, z_min, z_max, lb, ub)
                    warning_n(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, z_min_n, z_max_n, u_min_n, u_max_n)

                    s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch,\
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch = \
                    warning_n_del(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                                  z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch, \
                                  z_min_n, z_max_n, u_min_n, u_max_n)
                    HJB_cost, _, W = critic.train(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                                   z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch, mu_, V_LEARNING_RATE_med,\
                                   BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)
                    # check CLF
                    LfV_mesh, Lg1V_mesh, Lg2V_mesh, gradV2 = critic.cal_Lie2(mesh_batchS1, mesh_batchS2, mesh_batchS3, mesh_batchS4,
                                                                             mesh_batchU1, mesh_batchU2, \
                                         np.ones_like(mesh_batchS1)*z_ss[0], np.ones_like(mesh_batchS1)*z_ss[1],
                                         np.ones_like(mesh_batchS1)*z_ss[2], np.ones_like(mesh_batchS1)*z_ss[3],
                                         np.ones_like(mesh_batchU1)*u_ss[0], np.ones_like(mesh_batchU1)*u_ss[1], mu_,\
                                         BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_)

                    bad = check_CLF(LfV_mesh, Lg1V_mesh, Lg2V_mesh, z1_range, z2_range, z3_range, z4_range, u1_range, u2_range, EPS_CLF, R)
                    print(cheche)
                    if bad.size > 0:
                        print('Warning, No CLF')
                        temp_bad = np.append([i,j], bad.size)
                        bad_hist = np.concatenate([bad_hist, temp_bad])
                        critic.re_init0(W_pre[0])
                        critic.re_init1(W_pre[1])
                        critic.re_init2(W_pre[2])
                        #critic.re_init3(W_pre[3])
                        #critic.re_init4(W_pre[4])
                        V_LEARNING_RATE_med = V_LEARNING_RATE_med / 10
                    elif cheche == int(19):
                        critic.re_init0(W_pre[0])
                        critic.re_init1(W_pre[1])
                        critic.re_init2(W_pre[2])
                        print('previous CLF')
                        break
                    else:
                        print('Find CLF')
                        break

            GAT_HJB_cost.append([HJB_cost])

            count = count+1
            if count == 1:
                W_ = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                GAT_Wi = [i]
                GAT_W0, GAT_W1, GAT_W2  = [W_[0]], [W_[1]], [W_[2]]
            else:
                W_ = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                GAT_Wi = np.concatenate((GAT_Wi, [i]))
                GAT_W0, GAT_W1, GAT_W2  = np.concatenate((GAT_W0, [W_[0]])), \
                                          np.concatenate((GAT_W1, [W_[1]])),\
                                          np.concatenate((GAT_W2, [W_[2]]))

            # NMPC update
            dist_s_pool = np.array([dist_s1_pool[i, j], dist_s2_pool[i, j], \
                                    dist_s3_pool[i, j], dist_s4_pool[i, j]])  # (4,)
            Plant = plant(s, u, delt, ns, nu, [a1_inject, a2_inject], lb, ub, dist_s_pool, [0., 0.]) # dist_u_pool[j, :])
            Model = model(s, u, delt, ns, nu, [a1_inject, a2_inject], lb, ub, [0., 0., 0., 0.], [0., 0.], mismatch)
            _, stat_p, _, e_time, _ = ipopt_solve_with_stats(Plant, ipopt)
            _, stat_m, _, e_time_m, _ = ipopt_solve_with_stats(Model, ipopt)
            gat_stat_p.append(stat_p)
            gat_stat_m.append(stat_m)

            s_prev = s
            u_prev = u
            a_prev = [a1_inject, a2_inject]

            s = [value(Plant.z[delt, 1]), value(Plant.z[delt, 2]), value(Plant.z[delt, 3]), value(Plant.z[delt, 4])]
            u = [value(Plant.u[delt, 1]), value(Plant.u[delt, 2])]
            s_m = [value(Model.z[delt, 1]), value(Model.z[delt, 2]), value(Model.z[delt, 3]), value(Model.z[delt, 4])]
            u_m = [value(Model.u[delt, 1]), value(Model.u[delt, 2])]

            e_s = [s[0] - s_m[0], s[1] - s_m[1], s[2] - s_m[2], s[3] - s_m[3]]  # (s_prev, u_prev, a_prev, e_s)
            e_u = [0, 0]
            e_pred = [pred_Y1, pred_Y2, pred_Y3, pred_Y4]

            gat_e1.append(e_s[0]), gat_e2.append(e_s[1]), gat_e3.append(e_s[2]), gat_e4.append(e_s[3])
            gat_e1_pred.append(e_pred[0]), gat_e2_pred.append(e_pred[1])
            gat_e3_pred.append(e_pred[2]), gat_e4_pred.append(e_pred[3])

            e_std  = [ std_Y1,  std_Y2,  std_Y3,  std_Y4]
            gat_e1_std.append(e_std[0]), gat_e2_std.append(e_std[1])
            gat_e3_std.append(e_std[2]), gat_e4_std.append(e_std[3])

            if j == int(MAX_EP_STEPS/delt)-1:
                count_GAT = count_GAT + 1
                if count_GAT == 1:
                    GAT_z1, GAT_z2 = [gat_z1], [gat_z2]
                    GAT_z3, GAT_z4 = [gat_z3], [gat_z4]
                    GAT_u1, GAT_u2 = [gat_u1], [gat_u2]
                    GAT_a1, GAT_a2 = [gat_a1], [gat_a2]
                    GAT_stat_p = [gat_stat_p] #GAT_Compare_ = [gat_Compare_]

                    GAT_e1, GAT_e2, GAT_e3, GAT_e4 = [gat_e1], [gat_e2], [gat_e3], [gat_e4]
                    GAT_e1_pred, GAT_e2_pred, \
                    GAT_e3_pred, GAT_e4_pred = [gat_e1_pred], [gat_e2_pred], [gat_e3_pred], [gat_e4_pred]

                    GAT_e1_std, GAT_e2_std, \
                    GAT_e3_std, GAT_e4_std = [gat_e1_std], [gat_e2_std], [gat_e3_std], [gat_e4_std]

                    GAT_z1_m, GAT_z2_m, GAT_z3_m, GAT_z4_m = [gat_z1_m], [gat_z2_m], [gat_z3_m], [gat_z4_m]
                    GAT_u1_m, GAT_u2_m = [gat_u1_m], [gat_u2_m]
                    GAT_stat_m = [gat_stat_m]

                    GAT_z1_BO_U, GAT_z2_BO_U = [gat_z1_BO_U], [gat_z2_BO_U]
                    GAT_z3_BO_U, GAT_z4_BO_U = [gat_z3_BO_U], [gat_z4_BO_U]
                    GAT_z1_BO_L, GAT_z2_BO_L = [gat_z1_BO_L], [gat_z2_BO_L]
                    GAT_z3_BO_L, GAT_z4_BO_L = [gat_z3_BO_L], [gat_z4_BO_L]

                    GAT_a1_son_org, GAT_a2_son_org = [gat_a1_son_org], [gat_a2_son_org]
                    GAT_a1_opt_org, GAT_a2_opt_org = [gat_a1_opt_org], [gat_a2_opt_org]
                    GAT_a1_son, GAT_a2_son = [gat_a1_son], [gat_a2_son]
                    GAT_a1_opt, GAT_a2_opt = [gat_a1_opt], [gat_a2_opt]
                    GAT_LfV, GAT_Lg1V, GAT_Lg2V = [gat_LfV], [gat_Lg1V], [gat_Lg2V]
                    GAT_q = [gat_q]
                    GAT_V1, GAT_V2 = [gat_V1], [gat_V2]
                    GAT_V_dot_son, GAT_V_dot_opt = [gat_V_dot_son], [gat_V_dot_opt]
                    cost  = delt * (Q1 * (np.array(gat_z1) - z_ss[0]) ** 2 / ((z_max[0] - z_min[0]) ** 2)
                                   + Q1 * (np.array(gat_z2) - z_ss[1]) ** 2 / ((z_max[1] - z_min[1]) ** 2) \
                                   + Q1 * (np.array(gat_z3) - z_ss[2]) ** 2 / ((z_max[2] - z_min[2]) ** 2) \
                                   + Q1 * (np.array(gat_z4) - z_ss[3]) ** 2 / ((z_max[3] - z_min[3]) ** 2) \
                                   + Q2 * (np.array(gat_u1) - u_ss[0]) ** 2 / ((u_max[0] - u_min[0]) ** 2) \
                                   + Q2 * (np.array(gat_u2) - u_ss[1]) ** 2 / ((u_max[1] - u_min[1]) ** 2) \
                                   + R * (np.array(gat_a1)) ** 2 + R * (np.array(gat_a2)) ** 2)
                    cost_i = np.sum(cost)
                    RL_cost.append(cost_i)
                else:
                    GAT_z1, GAT_z2 = np.concatenate((GAT_z1, [gat_z1])), np.concatenate((GAT_z2, [gat_z2]))
                    GAT_z3, GAT_z4 = np.concatenate((GAT_z3, [gat_z3])), np.concatenate((GAT_z4, [gat_z4]))
                    GAT_u1, GAT_u2 = np.concatenate((GAT_u1, [gat_u1])), np.concatenate((GAT_u2, [gat_u2]))
                    GAT_a1, GAT_a2 = np.concatenate((GAT_a1, [gat_a1])), np.concatenate((GAT_a2, [gat_a2]))
                    GAT_stat_p   = np.concatenate((GAT_stat_p, [gat_stat_p]))
                    #GAT_Compare_ = np.concatenate((GAT_Compare_, [gat_Compare_]))

                    GAT_e1, GAT_e2 = np.concatenate((GAT_e1, [gat_e1])), np.concatenate((GAT_e2, [gat_e2]))
                    GAT_e3, GAT_e4 = np.concatenate((GAT_e3, [gat_e3])), np.concatenate((GAT_e4, [gat_e4]))

                    GAT_e1_pred, GAT_e2_pred = np.concatenate((GAT_e1_pred, [gat_e1_pred])), \
                                               np.concatenate((GAT_e2_pred, [gat_e2_pred]))
                    GAT_e3_pred, GAT_e4_pred = np.concatenate((GAT_e3_pred, [gat_e3_pred])), \
                                               np.concatenate((GAT_e4_pred, [gat_e4_pred]))

                    GAT_e1_std, GAT_e2_std = np.concatenate((GAT_e1_std, [gat_e1_std])), \
                                             np.concatenate((GAT_e2_std, [gat_e2_std]))
                    GAT_e3_std, GAT_e4_std = np.concatenate((GAT_e3_std, [gat_e3_std])), \
                                             np.concatenate((GAT_e4_std, [gat_e4_std]))

                    GAT_z1_m, GAT_z2_m = np.concatenate((GAT_z1_m, [gat_z1_m])), np.concatenate((GAT_z2_m, [gat_z2_m]))
                    GAT_z3_m, GAT_z4_m = np.concatenate((GAT_z3_m, [gat_z3_m])), np.concatenate((GAT_z4_m, [gat_z4_m]))
                    GAT_u1_m, GAT_u2_m = np.concatenate((GAT_u1_m, [gat_u1_m])), np.concatenate((GAT_u2_m, [gat_u2_m]))
                    GAT_stat_m   = np.concatenate((GAT_stat_m, [gat_stat_m]))

                    GAT_z1_BO_U, GAT_z2_BO_U = np.concatenate((GAT_z1_BO_U, [gat_z1_BO_U])), \
                                               np.concatenate((GAT_z2_BO_U, [gat_z2_BO_U]))
                    GAT_z3_BO_U, GAT_z4_BO_U = np.concatenate((GAT_z3_BO_U, [gat_z3_BO_U])), \
                                               np.concatenate((GAT_z4_BO_U, [gat_z4_BO_U]))
                    GAT_z1_BO_L, GAT_z2_BO_L = np.concatenate((GAT_z1_BO_L, [gat_z1_BO_L])), \
                                               np.concatenate((GAT_z2_BO_L, [gat_z2_BO_L]))
                    GAT_z3_BO_L, GAT_z4_BO_L = np.concatenate((GAT_z3_BO_L, [gat_z3_BO_L])), \
                                               np.concatenate((GAT_z4_BO_L, [gat_z4_BO_L]))

                    GAT_a1_son_org = np.concatenate((GAT_a1_son_org, [gat_a1_son_org]))
                    GAT_a2_son_org = np.concatenate((GAT_a2_son_org, [gat_a2_son_org]))
                    GAT_a1_opt_org = np.concatenate((GAT_a1_opt_org, [gat_a1_opt_org]))
                    GAT_a2_opt_org = np.concatenate((GAT_a2_opt_org, [gat_a2_opt_org]))
                    GAT_a1_son = np.concatenate((GAT_a1_son, [gat_a1_son]))
                    GAT_a2_son = np.concatenate((GAT_a2_son, [gat_a2_son]))
                    GAT_a1_opt = np.concatenate((GAT_a1_opt, [gat_a1_opt]))
                    GAT_a2_opt = np.concatenate((GAT_a2_opt, [gat_a2_opt]))
                    GAT_LfV = np.concatenate((GAT_LfV, [gat_LfV]))
                    GAT_Lg1V = np.concatenate((GAT_Lg1V, [gat_Lg1V]))
                    GAT_Lg2V = np.concatenate((GAT_Lg2V, [gat_Lg2V]))
                    GAT_q = np.concatenate((GAT_q, [gat_q]))
                    GAT_V1 = np.concatenate((GAT_V1, [gat_V1]))
                    GAT_V2 = np.concatenate((GAT_V2, [gat_V2]))
                    GAT_V_dot_son = np.concatenate((GAT_V_dot_son, [gat_V_dot_son]))
                    GAT_V_dot_opt = np.concatenate((GAT_V_dot_opt, [gat_V_dot_opt]))
                    cost  = delt * (Q1 * (np.array(gat_z1) - z_ss[0]) ** 2 / ((z_max[0] - z_min[0]) ** 2)
                                   + Q1 * (np.array(gat_z2) - z_ss[1]) ** 2 / ((z_max[1] - z_min[1]) ** 2) \
                                   + Q1 * (np.array(gat_z3) - z_ss[2]) ** 2 / ((z_max[2] - z_min[2]) ** 2) \
                                   + Q1 * (np.array(gat_z4) - z_ss[3]) ** 2 / ((z_max[3] - z_min[3]) ** 2) \
                                   + Q2 * (np.array(gat_u1) - u_ss[0]) ** 2 / ((u_max[0] - u_min[0]) ** 2) \
                                   + Q2 * (np.array(gat_u2) - u_ss[1]) ** 2 / ((u_max[1] - u_min[1]) ** 2) \
                                   + R * (np.array(gat_a1)) ** 2 + R * (np.array(gat_a2)) ** 2)

                    cost_i = np.sum(cost)
                    print('cost_del', cost_i - MPC_cost[0, i])
                    RL_cost.append(cost_i)
                    time_ = np.linspace(0, int(MAX_EP_STEPS/delt)-1, int(MAX_EP_STEPS/delt))
                    #plot_t(gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2,
                    #       MPC_z1, MPC_z2, MPC_z3, MPC_z4, MPC_u1, MPC_u2, MPC_a1, MPC_a2,
                    #       GAT_HJB_cost, GAT_Compare_, RL_cost, MPC_cost, i, mode, time_,
                    #       GAT_a1_son, GAT_a2_son, GAT_a1_opt, GAT_a2_opt)

        # one episode ends
        if i >= 0:
            if i % 10 == 0:
                sio.savemat('RL{0}_mode{1}.mat'.format(i, mode),
                            {'Gat_z1' : GAT_z1, 'Gat_z2' : GAT_z2,
                             'Gat_z3': GAT_z3, 'Gat_z4': GAT_z4,
                             'Gat_u1': GAT_u1, 'Gat_u2': GAT_u2,
                             'Gat_a1': GAT_a1, 'Gat_a2': GAT_a2,
                             'Gat_HJB_cost': GAT_HJB_cost, #'GAT_Compare_':GAT_Compare_,
                             'Gat_RL_cost': RL_cost, 'Gat_stat_p': GAT_stat_p,

                             'Gat_e1': GAT_e1, 'Gat_e2': GAT_e2, 'Gat_e3': GAT_e3, 'Gat_e4': GAT_e4,
                             'Gat_e1_pred': GAT_e1_pred, 'Gat_e2_pred': GAT_e2_pred,
                             'Gat_e3_pred': GAT_e3_pred, 'Gat_e4_pred': GAT_e4_pred,

                             'Gat_e1_std': GAT_e1_std, 'Gat_e2_std': GAT_e2_std,
                             'Gat_e3_std': GAT_e3_std, 'Gat_e4_std': GAT_e4_std,

                             'Gat_z1_m': GAT_z1_m, 'Gat_z2_m': GAT_z2_m, 'Gat_z3_m': GAT_z3_m, 'Gat_z4_m': GAT_z4_m,
                             'Gat_u1_m': GAT_u1_m, 'Gat_u2_m': GAT_u2_m,
                             'Gat_stat_m': GAT_stat_m,

                             'Gat_z1_BO_U': GAT_z1_BO_U, 'Gat_z2_BO_U': GAT_z2_BO_U,
                             'Gat_z3_BO_U': GAT_z3_BO_U, 'Gat_z4_BO_U': GAT_z4_BO_U,
                             'Gat_z1_BO_L': GAT_z1_BO_L, 'Gat_z2_BO_L': GAT_z2_BO_L,
                             'Gat_z3_BO_L': GAT_z3_BO_L, 'Gat_z4_BO_L': GAT_z4_BO_L,

                             'GAT_a1_son_org' : GAT_a1_son_org, 'GAT_a2_son_org' : GAT_a2_son_org,
                             'GAT_a1_opt_org' : GAT_a1_opt_org, 'GAT_a2_opt_org': GAT_a2_opt_org,
                             'GAT_a1_son': GAT_a1_son, 'GAT_a2_son': GAT_a2_son,
                             'GAT_a1_opt': GAT_a1_opt, 'GAT_a2_opt': GAT_a2_opt,
                             'GAT_LfV': GAT_LfV, 'GAT_Lg1V': GAT_Lg1V, 'GAT_Lg2V': GAT_Lg2V,
                             'GAT_q': GAT_q, 'GAT_V1': GAT_V1, 'GAT_V2': GAT_V2,
                             'GAT_V_dot_son': GAT_V_dot_son,'GAT_V_dot_opt': GAT_V_dot_opt,
                             'GAT_W0': GAT_W0, 'GAT_W1': GAT_W1, 'GAT_W2': GAT_W2, 'GAT_Wi': GAT_Wi, #'GAT_Wout': GAT_Wout,
                             'MPC_z1' :MPC_z1, 'MPC_z2': MPC_z2, 'MPC_z3' :MPC_z3, 'MPC_z4': MPC_z4,
                             'MPC_u1': MPC_u1, 'MPC_u2': MPC_u2,
                             'MPC_a1' :MPC_a1, 'MPC_a2': MPC_a2, 'MPC_cost' :MPC_cost
                             })
                # Save
                pickle.dump(optimizer_Y1, open("GP_Y1_{0}_mode{1}.pkl".format(i, mode), "wb"))
                pickle.dump(optimizer_Y2, open("GP_Y2_{0}_mode{1}.pkl".format(i, mode), "wb"))
                pickle.dump(optimizer_Y3, open("GP_Y3_{0}_mode{1}.pkl".format(i, mode), "wb"))
                pickle.dump(optimizer_Y4, open("GP_Y4_{0}_mode{1}.pkl".format(i, mode), "wb"))

    #all episodes end
    sio.savemat('RL{0}_mode{1}.mat'.format(i, mode),
                {'Gat_z1': GAT_z1, 'Gat_z2': GAT_z2,
                 'Gat_z3': GAT_z3, 'Gat_z4': GAT_z4,
                 'Gat_u1': GAT_u1, 'Gat_u2': GAT_u2,
                 'Gat_a1': GAT_a1, 'Gat_a2': GAT_a2,
                 'Gat_HJB_cost': GAT_HJB_cost, #'GAT_Compare_':GAT_Compare_,
                 'Gat_RL_cost': RL_cost, 'Gat_stat_p': GAT_stat_p,

                 'Gat_e1': GAT_e1, 'Gat_e2': GAT_e2, 'Gat_e3': GAT_e3, 'Gat_e4': GAT_e4,
                 'Gat_e1_pred': GAT_e1_pred, 'Gat_e2_pred': GAT_e2_pred,
                 'Gat_e3_pred': GAT_e3_pred, 'Gat_e4_pred': GAT_e4_pred,

                 'Gat_e1_std': GAT_e1_std, 'Gat_e2_std': GAT_e2_std,
                 'Gat_e3_std': GAT_e3_std, 'Gat_e4_std': GAT_e4_std,

                 'Gat_z1_m': GAT_z1_m, 'Gat_z2_m': GAT_z2_m, 'Gat_z3_m': GAT_z3_m, 'Gat_z4_m': GAT_z4_m,
                 'Gat_u1_m': GAT_u1_m, 'Gat_u2_m': GAT_u2_m,
                 'Gat_stat_m': GAT_stat_m,

                 'Gat_z1_BO_U': GAT_z1_BO_U, 'Gat_z2_BO_U': GAT_z2_BO_U,
                 'Gat_z3_BO_U': GAT_z3_BO_U, 'Gat_z4_BO_U': GAT_z4_BO_U,
                 'Gat_z1_BO_L': GAT_z1_BO_L, 'Gat_z2_BO_L': GAT_z2_BO_L,
                 'Gat_z3_BO_L': GAT_z3_BO_L, 'Gat_z4_BO_L': GAT_z4_BO_L,

                 'GAT_a1_son_org': GAT_a1_son_org, 'GAT_a2_son_org': GAT_a2_son_org,
                 'GAT_a1_opt_org': GAT_a1_opt_org, 'GAT_a2_opt_org': GAT_a2_opt_org,
                 'GAT_a1_son': GAT_a1_son, 'GAT_a2_son': GAT_a2_son,
                 'GAT_a1_opt': GAT_a1_opt, 'GAT_a2_opt': GAT_a2_opt,
                 'GAT_LfV': GAT_LfV, 'GAT_Lg1V': GAT_Lg1V, 'GAT_Lg2V': GAT_Lg2V,
                 'GAT_q': GAT_q, 'GAT_V1': GAT_V1, 'GAT_V2': GAT_V2,
                 'GAT_V_dot_son': GAT_V_dot_son, 'GAT_V_dot_opt': GAT_V_dot_opt,
                 'GAT_W0': GAT_W0, 'GAT_W1': GAT_W1, 'GAT_W2': GAT_W2, 'GAT_Wi': GAT_Wi, #'GAT_Wout': GAT_Wout,
                 'MPC_z1': MPC_z1, 'MPC_z2': MPC_z2, 'MPC_z3' :MPC_z3, 'MPC_z4': MPC_z4,
                 'MPC_u1': MPC_u1, 'MPC_u2': MPC_u2,
                 'MPC_a1': MPC_a1, 'MPC_a2': MPC_a2, 'MPC_cost': MPC_cost
                 })
    pickle.dump(optimizer_Y1, open("GP_Y1_{0}_mode{1}.pkl".format(i, mode), "wb"))
    pickle.dump(optimizer_Y2, open("GP_Y2_{0}_mode{1}.pkl".format(i, mode), "wb"))
    pickle.dump(optimizer_Y3, open("GP_Y3_{0}_mode{1}.pkl".format(i, mode), "wb"))
    pickle.dump(optimizer_Y4, open("GP_Y4_{0}_mode{1}.pkl".format(i, mode), "wb"))


def MPC_all_known(sess, z_ss, u_ss, Q1, Q2, R, Hor_n, MAX_EPISODES, lb, ub, delt):
    # FHOCP Solve
    A = sio.loadmat('total_s.mat')['state_pool']
    B = sio.loadmat('total_s.mat')['u_pool']

    C = sio.loadmat('total_dist.mat')['dist_s1_pool']
    D = sio.loadmat('total_dist.mat')['dist_s2_pool']
    E = sio.loadmat('total_dist.mat')['dist_s3_pool']
    F = sio.loadmat('total_dist.mat')['dist_s4_pool']
    # D = sio.loadmat('total_dist.mat')['dist_u_pool']
    MPC_ = sio.loadmat('RL{0}_del_GP.mat'.format(MAX_EPISODES - 1))
    del_i = MPC_['Gat_i_del']
    if np.size(del_i) > 0:
        AA = np.delete(A, np.squeeze(del_i), axis=0)
        BB = np.delete(B, np.squeeze(del_i), axis=0)
        s_pool, u_pool = AA, BB

        CC = np.delete(C, np.squeeze(del_i), axis=0)
        DD = np.delete(D, np.squeeze(del_i), axis=0)
        EE = np.delete(E, np.squeeze(del_i), axis=0)
        FF = np.delete(F, np.squeeze(del_i), axis=0)
        # DD = np.delete(D, np.squeeze(del_i), axis=0)
        dist_s1_pool = CC  # , dist_u_pool = CC, DD
        dist_s2_pool = DD
        dist_s3_pool = EE
        dist_s4_pool = FF
    else:
        s_pool = A
        u_pool = B

    MAX_EPISODES = len(s_pool)
    MPC_z1, MPC_z2, MPC_z3, MPC_z4, MPC_u1, MPC_u2, MPC_a1, MPC_a2, MPC_cost \
                   = MPC_['Gat_z1'], MPC_['Gat_z2'], MPC_['Gat_z3'], MPC_['Gat_z4'], MPC_['Gat_u1'], MPC_['Gat_u2'], \
                     MPC_['Gat_a1'], MPC_['Gat_a2'], MPC_['Gat_cost']

    z_min, z_max = [1., 1., 1., 1.], [28, 28, 28, 28]
    u_min, u_max = [lb[1], lb[2]], [ub[1], ub[2]]
    ns, nu, delt = 4, 2, delt
    GAT_i, GAT_i_del = [], []
    GAT_cost = []
    count_GAT = 0

    for i in range(MAX_EPISODES):  # 0, 1, ... 1999
        # initial condition setting for u
        s0, u0 = [MPC_z1[i, 0], MPC_z2[i, 0], MPC_z3[i, 0], MPC_z4[i, 0]], [MPC_u1[i, 0], MPC_u2[i, 0]]
        pa0 = [0, 0]

        gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2 = [], [], [], [], [], [], [], []
        gat_i, gat_j, gat_stat = [], [], []
        gat_z1_m, gat_z2_m, gat_z3_m, gat_z4_m, gat_u1_m, gat_u2_m = [], [], [], [], [], []

        for j in range(t_duration):
            if j == 0:  # initial state of each episode
                s, u = s0, u0
                s_m, u_m = s0, u0
                pu0 = u
                OCP = model_FHOCP3_allknown(s, delt, Hor_n, Q1, Q2, R, ns, nu, \
                                   np.ones(Hor_n) * z_ss[0], np.ones(Hor_n) * z_ss[1],
                                   np.ones(Hor_n) * z_ss[2], np.ones(Hor_n) * z_ss[3],
                                   np.ones(Hor_n) * u_ss[0], np.ones(Hor_n) * u_ss[1], pu0, lb, ub, pa0, mismatch,
                                   dist_s1_pool[i, j:j+Hor_n+1], dist_s2_pool[i, j:j+Hor_n+1],
                                   dist_s3_pool[i, j:j+Hor_n+1], dist_s4_pool[i, j:j+Hor_n+1])

                _, stat, _, e_time, _ = ipopt_solve_with_stats(OCP, ipopt)
            print([i, j], s, u)
            a1_inject, a2_inject = value(OCP.a[delt * 1, 1]), value(OCP.a[delt * 1, 2])
            # think about ncp = 1 for input
            print('ipopt True', stat, a1_inject, a2_inject)

            gat_z1.append(s[0]), gat_z2.append(s[1]), gat_z3.append(s[2]), gat_z4.append(s[3])
            gat_u1.append(u[0]), gat_u2.append(u[1])
            gat_a1.append(a1_inject), gat_a2.append(a2_inject)
            gat_i.append(i), gat_j.append(j)

            gat_z1_m.append(s_m[0]), gat_z2_m.append(s_m[1]), gat_z3_m.append(s_m[2]), gat_z4_m.append(s_m[3])
            gat_u1_m.append(u_m[0]), gat_u2_m.append(u_m[1])
            # NMPC update
            dist_s_pool = np.array([dist_s1_pool[i, j], dist_s2_pool[i, j], \
                                    dist_s3_pool[i, j], dist_s4_pool[i, j]])
            Plant = plant(s, u, delt, ns, nu, [a1_inject, a2_inject], lb, ub, dist_s_pool,[0., 0.])  # dist_u_pool[j,:])
            Model = model(s, u, delt, ns, nu, [a1_inject, a2_inject], lb, ub, [0., 0., 0., 0.], [0., 0.],mismatch)
            _, stat_p, _, e_time, _ = ipopt_solve_with_stats(Plant, ipopt)
            _, stat_m, _, e_time_m, _ = ipopt_solve_with_stats(Model, ipopt)

            s = [value(Plant.z[delt, 1]), value(Plant.z[delt, 2]), value(Plant.z[delt, 3]),value(Plant.z[delt, 4])]
            u = [value(Plant.u[delt, 1]), value(Plant.u[delt, 2])]

            s_m = [value(Model.z[delt, 1]), value(Model.z[delt, 2]), value(Model.z[delt, 3]),value(Model.z[delt, 4])]
            u_m = [value(Model.u[delt, 1]), value(Model.u[delt, 2])]

            pu0 = u

            if j <= t_duration - Hor_n:
                OCP = model_FHOCP3_allknown(s, delt, Hor_n, Q1, Q2, R, ns, nu, \
                                   np.ones(Hor_n) * z_ss[0], np.ones(Hor_n) * z_ss[1],
                                   np.ones(Hor_n) * z_ss[2], np.ones(Hor_n) * z_ss[3],
                                   np.ones(Hor_n) * u_ss[0], np.ones(Hor_n) * u_ss[1], pu0, lb, ub, pa0,
                                   mismatch,
                                   dist_s1_pool[i, j:j+Hor_n+1], dist_s2_pool[i, j:j+Hor_n+1],
                                   dist_s3_pool[i, j:j+Hor_n+1], dist_s4_pool[i, j:j+Hor_n+1])

            else:
                OCP = model_FHOCP3_allknown(s, delt, Hor_n+Hor_n-j-1, Q1, Q2, R, ns, nu, \
                                            np.ones(Hor_n+Hor_n-j-1) * z_ss[0], np.ones(Hor_n+Hor_n-j-1) * z_ss[1],
                                            np.ones(Hor_n+Hor_n-j-1) * z_ss[2], np.ones(Hor_n+Hor_n-j-1) * z_ss[3],
                                            np.ones(Hor_n+Hor_n-j-1) * u_ss[0], np.ones(Hor_n+Hor_n-j-1) * u_ss[1], pu0, lb, ub, pa0,
                                            mismatch,
                                            dist_s1_pool[i, j:], dist_s2_pool[i, j:],
                                            dist_s3_pool[i, j:], dist_s4_pool[i, j:])

            _, stat, _, e_time, _ = ipopt_solve_with_stats(OCP, ipopt)

            if j == t_duration-1:
                count_GAT = count_GAT + 1
                if count_GAT == 1:
                    GAT_z1, GAT_z2, GAT_z3, GAT_z4 = [gat_z1], [gat_z2], [gat_z3], [gat_z4]
                    GAT_u1, GAT_u2 = [gat_u1], [gat_u2]
                    GAT_z1_m, GAT_z2_m, GAT_z3_m, GAT_z4_m = [gat_z1_m], [gat_z2_m], [gat_z3_m], [gat_z4_m]
                    GAT_u1_m, GAT_u2_m = [gat_u1_m], [gat_u2_m]
                    GAT_a1, GAT_a2 = [gat_a1], [gat_a2]
                    cost = delt * (Q1 * (np.array(gat_z1) - z_ss[0]) ** 2 / ((z_max[0] - z_min[0]) ** 2)
                                   + Q1 * (np.array(gat_z2) - z_ss[1]) ** 2 / ((z_max[1] - z_min[1]) ** 2) \
                                   + Q1 * (np.array(gat_z3) - z_ss[2]) ** 2 / ((z_max[2] - z_min[2]) ** 2) \
                                   + Q1 * (np.array(gat_z4) - z_ss[3]) ** 2 / ((z_max[3] - z_min[3]) ** 2) \
                                   + Q2 * (np.array(gat_u1) - u_ss[0]) ** 2 / ((u_max[0] - u_min[0]) ** 2) \
                                   + Q2 * (np.array(gat_u2) - u_ss[1]) ** 2 / ((u_max[1] - u_min[1]) ** 2) \
                                   + R * (np.array(gat_a1)) ** 2 + R * (np.array(gat_a2)) ** 2)
                    cost_i = np.sum(cost)
                    GAT_cost.append(cost_i), GAT_i.append(i)
                else:
                    GAT_z1, GAT_z2 = np.concatenate((GAT_z1, [gat_z1])), np.concatenate((GAT_z2, [gat_z2]))
                    GAT_z3, GAT_z4 = np.concatenate((GAT_z3, [gat_z3])), np.concatenate((GAT_z4, [gat_z4]))
                    GAT_u1, GAT_u2 = np.concatenate((GAT_u1, [gat_u1])), np.concatenate((GAT_u2, [gat_u2]))
                    GAT_z1_m, GAT_z2_m = np.concatenate((GAT_z1_m, [gat_z1_m])), np.concatenate((GAT_z2_m, [gat_z2_m]))
                    GAT_z3_m, GAT_z4_m = np.concatenate((GAT_z3_m, [gat_z3_m])), np.concatenate((GAT_z4_m, [gat_z4_m]))
                    GAT_u1_m, GAT_u2_m = np.concatenate((GAT_u1_m, [gat_u1_m])), np.concatenate((GAT_u2_m, [gat_u2_m]))
                    GAT_a1, GAT_a2 = np.concatenate((GAT_a1, [gat_a1])), np.concatenate((GAT_a2, [gat_a2]))
                    cost = delt * (Q1 * (np.array(gat_z1) - z_ss[0]) ** 2 / ((z_max[0] - z_min[0]) ** 2)
                                   + Q1 * (np.array(gat_z2) - z_ss[1]) ** 2 / ((z_max[1] - z_min[1]) ** 2) \
                                   + Q1 * (np.array(gat_z3) - z_ss[2]) ** 2 / ((z_max[2] - z_min[2]) ** 2) \
                                   + Q1 * (np.array(gat_z4) - z_ss[3]) ** 2 / ((z_max[3] - z_min[3]) ** 2) \
                                   + Q2 * (np.array(gat_u1) - u_ss[0]) ** 2 / ((u_max[0] - u_min[0]) ** 2) \
                                   + Q2 * (np.array(gat_u2) - u_ss[1]) ** 2 / ((u_max[1] - u_min[1]) ** 2) \
                                   + R * (np.array(gat_a1)) ** 2 + R * (np.array(gat_a2)) ** 2)
                    cost_i = np.sum(cost)
                    GAT_cost.append(cost_i), GAT_i.append(i)
                    # plot_(gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2)


        print(np.shape(GAT_i_del))
        # one episode ends
        if i > 0:
            if i % 10 == 0:
                sio.savemat('RL{0}_MPC_all_known.mat'.format(i),
                            {'Gat_z1': GAT_z1, 'Gat_z2': GAT_z2, 'Gat_z3': GAT_z3, 'Gat_z4': GAT_z4,
                             'Gat_u1': GAT_u1, 'Gat_u2': GAT_u2,
                             'Gat_z1_m': GAT_z1_m, 'Gat_z2_m': GAT_z2_m, 'Gat_z3_m': GAT_z3_m, 'Gat_z4_m': GAT_z4_m,
                             'Gat_u1_m': GAT_u1_m, 'Gat_u2_m': GAT_u2_m,
                             'Gat_a1': GAT_a1, 'Gat_a2': GAT_a2,
                             'Gat_i': GAT_i, 'Gat_i_del': GAT_i_del, 'Gat_cost': GAT_cost
                             })
    # all episodes end
    sio.savemat('RL{0}_MPC_all_known.mat'.format(i),
                {'Gat_z1': GAT_z1, 'Gat_z2': GAT_z2, 'Gat_z3': GAT_z3, 'Gat_z4': GAT_z4,
                 'Gat_u1': GAT_u1, 'Gat_u2': GAT_u2,
                 'Gat_z1_m': GAT_z1_m, 'Gat_z2_m': GAT_z2_m, 'Gat_z3_m': GAT_z3_m, 'Gat_z4_m': GAT_z4_m,
                 'Gat_u1_m': GAT_u1_m, 'Gat_u2_m': GAT_u2_m,
                 'Gat_a1': GAT_a1, 'Gat_a2': GAT_a2,
                 'Gat_i': GAT_i, 'Gat_i_del': GAT_i_del, 'Gat_cost': GAT_cost,
                 })



def main():
    n_cpus = 4
    with tf.Session(config=tf.ConfigProto(
            device_count={"CPU": n_cpus},
            inter_op_parallelism_threads=n_cpus,
            intra_op_parallelism_threads=1)) as sess:
        z_ss = [14., 14., 14.4444, 21.1237]
        u_ss = [43.0691, 35.6148]

        RANDOM_SEED = 10
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        s_dim, u_dim, a_dim, V_dim = 4, 2, 2, 1

        Q1, Q2,   R = 1., 1., 0.1
        Hor_n = 100

        layer_dims = [6, 7]
        activations = [tf.nn.elu, tf.nn.elu]
        #activations = [tf.nn.leaky_relu, tf.nn.leaky_relu]
        #activations = [tf.nn.tanh, tf.nn.tanh]
        #activations = [tf.nn.relu, tf.nn.relu]

        lb = {1: 0., 2: 0.}
        ub = {1: 60., 2: 60.}

        EPS_, EPS_CLF, EPS_a, EPS_C = 1e-4, 1e-4, 1e-4, 1e-4 # NN & settle,  in CLF,  action condition
        alpha_, alpha2_ = 1., 1.
        mu_ = [1e-4, 1.]  # 1.] # = [1e-6, 1]
        MAX_EPISODES = 200


        remove_(sess, z_ss, u_ss, Q1, Q2, R, Hor_n, MAX_EPISODES, lb, ub, delt)
        gamma_ = [[0.99]]

        BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u = [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0], [0, 0]
        mode = 1 # mode = 1 new BF  # mode = 0: mu_ =10. / mode = 3: mu_ = 1.
        critic0 = VNetwork(sess, s_dim, u_dim, V_dim, EPS_, layer_dims, activations, Q1, Q2, R, mode, lb, ub, delt,\
                           BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, mismatch)
        actor0 = Actor(sess, s_dim, u_dim, a_dim, V_dim, EPS_a, alpha_, Q1, Q2, R, mode, lb, ub, delt,\
                       BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, gamma_, alpha2_)
        train(sess, actor0, critic0, z_ss, u_ss, alpha_, Q1, Q2, R, mode, EPS_, EPS_CLF, mu_, MAX_EPISODES, lb, ub, delt,\
              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, EPS_C, gamma_, alpha2_)

        MPC_all_known(sess, z_ss, u_ss, Q1, Q2, R, Hor_n, MAX_EPISODES, lb, ub, delt)


if __name__ == '__main__':
    main()