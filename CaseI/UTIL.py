import pyutilib.services
from pyomo.opt import TerminationCondition
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting

def ipopt_solve_with_stats(model, solver):
    """
    Run the solver (must be ipopt) and return the convergence statistics
    Parameters
    ----------
    model : Pyomo model
       The pyomo model to be solved
    solver : Pyomo solver
       The pyomo solver to use - it must be ipopt, but with whichever options are preferred
    max_iter : int
       The maximum number of iterations to allow for ipopt
    max_cpu_time : int
       The maximum cpu time to allow for ipopt (in seconds)
    Returns
    -------
       Returns a tuple with (solve status object, bool (solve successful or not), number of iters, solve time, regularization value at solution)
    """
    # ToDo: Check that the "solver" is, in fact, IPOPT

    pyutilib.services.TempfileManager.push()
    tempfile = pyutilib.services.TempfileManager.create_tempfile(suffix='ipopt_out', text=True)
    opts = {'output_file': tempfile}

    status_obj = solver.solve(model, options=opts, tee=False)
    solved = True
    if status_obj.solver.termination_condition != TerminationCondition.optimal:
        solved = False

    iters = 0
    time = 0
    line_m_2 = None
    line_m_1 = None
    # parse the output file to get the iteration count, solver times, etc.
    with open(tempfile, 'r') as f:
        for line in f:
            if line.startswith('Number of Iterations....:'):
                tokens = line.split()
                iters = int(tokens[3])
                tokens_m_2 = line_m_2.split()
                regu = str(tokens_m_2[3])
            elif line.startswith('Total CPU secs in IPOPT (w/o function evaluations)   ='):
                tokens = line.split()
                time += float(tokens[9])
            elif line.startswith('Total CPU secs in NLP function evaluations           ='):
                tokens = line.split()
                time += float(tokens[8])
            line_m_2 = line_m_1
            line_m_1 = line

    pyutilib.services.TempfileManager.pop(remove=True)
    return status_obj, solved, iters, time, regu

def check_CLF(LfV_mesh, Lg1V_mesh, Lg2V_mesh, z1_range, z2_range,  z3_range, z4_range, u1_range, u2_range, EPS_, R):
    r1_ = R
    r2_ = R
    #print(Lg1V_mesh, Lg2V_mesh, LfV_mesh)
    LfV_mesh  = np.reshape(LfV_mesh, [len(z1_range) * len(z2_range) * len(z3_range) * len(z4_range) * len(u1_range) * len(u2_range), -1])
    Lg1V_mesh = np.reshape(Lg1V_mesh, [len(z1_range) * len(z2_range) * len(z3_range) * len(z4_range) * len(u1_range) * len(u2_range), -1])
    Lg2V_mesh = np.reshape(Lg2V_mesh, [len(z1_range) * len(z2_range) * len(z3_range) * len(z4_range) * len(u1_range) * len(u2_range), -1])
    #print(np.sqrt(Lg1V_mesh ** 2/r1_ + Lg2V_mesh ** 2/r2_))
    nan_detect = is_nan(np.sqrt(Lg1V_mesh ** 2 + Lg2V_mesh ** 2))
    print('nan', len(np.where(nan_detect==True)[0]))
    if len(np.where(nan_detect==True)[0]) > 0:
        bad = np.array([1.])
    else:
        should_neg = LfV_mesh[np.sqrt(Lg1V_mesh ** 2 + Lg2V_mesh ** 2) < EPS_]
        bad = should_neg[should_neg >= 0]
    return bad

def data_get(replay_buffer, MINIBATCH_SIZE):
    minibatch = replay_buffer.get_batch(MINIBATCH_SIZE)
    s1_batch = np.asarray([data[0] for data in minibatch])  # 0:2
    s2_batch = np.asarray([data[1] for data in minibatch])  # 0:2
    s3_batch = np.asarray([data[2] for data in minibatch])  # 0:2
    s4_batch = np.asarray([data[3] for data in minibatch])  # 0:2
    u1_batch = np.asarray([data[4] for data in minibatch])  # 0:2
    u2_batch = np.asarray([data[5] for data in minibatch])  # 0:2
    a1_batch = np.asarray([data[6] for data in minibatch])  # 0:2
    a2_batch = np.asarray([data[7] for data in minibatch])  # 0:2)
    z1_ss_batch = np.asarray([data[8] for data in minibatch])  # 0:2
    z2_ss_batch = np.asarray([data[9] for data in minibatch])  # 0:2
    z3_ss_batch = np.asarray([data[10] for data in minibatch])  # 0:2
    z4_ss_batch = np.asarray([data[11] for data in minibatch])  # 0:2
    u1_ss_batch = np.asarray([data[12] for data in minibatch])  # 0:2)
    u2_ss_batch = np.asarray([data[13] for data in minibatch])  # 0:2)
    return s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
           z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch

def plot_(gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2):
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(gat_z1)
    plt.plot(gat_z2)
    plt.plot(gat_z3)
    plt.plot(gat_z4)
    plt.subplot(2, 2, 2)
    plt.plot(gat_u1)
    plt.plot(gat_u2)
    plt.subplot(2, 2, 3)
    plt.plot(gat_a1)
    plt.plot(gat_a2)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

def plot_t(gat_z1, gat_z2, gat_z3, gat_z4, gat_u1, gat_u2, gat_a1, gat_a2, MPC_z1, MPC_z2, MPC_z3, MPC_z4,
           MPC_u1, MPC_u2, MPC_a1, MPC_a2,
           GAT_HJB_cost, GAT_Compare_, RL_cost, MPC_cost, i, mode, time, GAT_a1_son, GAT_a2_son, GAT_a1_opt, GAT_a2_opt):
    plt.figure(1)
    plt.subplot(2, 4, 1)
    plt.plot(time,gat_z1, 'b-')
    plt.plot(time,gat_z2, 'b-')
    plt.plot(MPC_z1[i, :], 'r-')
    plt.plot(MPC_z2[i, :], 'r-')
    plt.title('x')
    plt.subplot(2, 4, 2)
    plt.plot(time,gat_z3, 'b-')
    plt.plot(time,gat_z4, 'b-')
    plt.plot(MPC_z3[i, :], 'r-')
    plt.plot(MPC_z4[i, :], 'r-')
    plt.subplot(2, 4, 3)
    plt.plot(time,gat_u1, 'b-')
    plt.plot(time,gat_u2, 'b-')
    plt.plot(MPC_u1[i, :], 'r-')
    plt.plot(MPC_u2[i, :], 'r-')
    plt.title('u')
    plt.subplot(2, 4, 4)
    plt.plot(time, gat_a1, 'b-')
    plt.plot(time, gat_a2, 'b-')
    plt.plot(MPC_a1[i, :], 'r-')
    plt.plot(MPC_a2[i, :], 'r-')
    plt.title('a')
    plt.subplot(2, 4, 5)
    plt.plot(np.array(GAT_a1_son.flat)-np.array(GAT_a1_opt.flat), 'b-')
    plt.plot(np.array(GAT_a2_son.flat)-np.array(GAT_a2_opt.flat), 'r-')
    plt.title('son - opt')
    plt.subplot(2, 4, 6)
    plt.plot(RL_cost, 'b-')
    plt.plot(MPC_cost[0, :i], 'r-')
    plt.title('RL cost vs MPC cost')
    plt.subplot(2, 4, 7)
    plt.plot(np.log(GAT_HJB_cost))
    plt.title('log(HJB_Error)')
    plt.savefig('fig{0}_mode{1}'.format(i, mode))
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

def plot_V(critic, z_ss, u_ss, mu_, i, mode, z_min, z_max,\
                      BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u):
    z1_range = np.linspace(z_min[0] + 0.1, z_max[0] - 0.1, 100)
    z2_range = np.linspace(z_min[0] + 0.1, z_max[0] - 0.1, 100)
    z3_range = z_ss[2]
    z4_range = z_ss[3]
    u1_range = u_ss[0]
    u2_range = u_ss[1]
    Z1, Z2 = np.meshgrid(z1_range, z2_range, indexing='ij')
    a, b = np.shape(Z1)
    mesh_batchS1, mesh_batchS2 = np.reshape(Z1, [-1, 1]), np.reshape(Z2, [-1, 1])  # Z1, Z2

    V, V2, reg = critic.cal_V(mesh_batchS1, mesh_batchS2, \
                              np.ones_like(mesh_batchS1) * z_ss[2], np.ones_like(mesh_batchS1) * z_ss[3], \
                              np.ones_like(mesh_batchS1) * u_ss[0], np.ones_like(mesh_batchS1) * u_ss[1], \
                              np.ones_like(mesh_batchS1) * z_ss[0], np.ones_like(mesh_batchS1) * z_ss[1],\
                              np.ones_like(mesh_batchS1) * z_ss[2], np.ones_like(mesh_batchS1) * z_ss[3],\
                              np.ones_like(mesh_batchS1) * u_ss[0], np.ones_like(mesh_batchS1) * u_ss[1], mu_,\
                              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u)

    X, Y, Z = np.reshape(mesh_batchS1, [a,b]), np.reshape(mesh_batchS2, [a,b]), np.reshape(V, [a,b])
    X, Y, Z2 = np.reshape(mesh_batchS1, [a, b]), np.reshape(mesh_batchS2, [a, b]), np.reshape(V2, [a, b])
    X, Y, Reg = np.reshape(mesh_batchS1, [a, b]), np.reshape(mesh_batchS2, [a, b]), np.reshape(reg, [a, b])
    # fig = plt.figure(1)
    # ax = fig.add_subplot(311, projection='3d')
    # ax.plot_surface(X, Y, Z)
    # ax.set_xlabel('x1'), ax.set_ylabel('x2'), ax.set_zlabel('V')
    # ax2 = fig.add_subplot(312, projection='3d')
    # ax2.plot_surface(X, Y, Z2)
    # ax2.set_xlabel('x1'), ax2.set_ylabel('x2'), ax2.set_zlabel('V2')
    # ax3 = fig.add_subplot(313, projection='3d')
    # ax3.plot_surface(X, Y, Reg)
    # ax3.set_xlabel('x1'), ax3.set_ylabel('x2'), ax3.set_zlabel('reg')
    # plt.savefig('figV{0}_mode{1}.pdf'.format(i, mode))
    # plt.pause(1)
    # plt.close('all')


def plot_info(critic, z_ss, u_ss, mu_, i, mode, z_min, z_max, \
                            BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u):
    z1_range = np.linspace(z_min[0] + 0.1, z_max[0] - 0.1, 100)
    z2_range = np.linspace(z_min[0] + 0.1, z_max[0] - 0.1, 100)
    z3_range = z_ss[2]
    z4_range = z_ss[3]
    u1_range = u_ss[0]
    u2_range = u_ss[1]
    Z1, Z2 = np.meshgrid(z1_range, z2_range, indexing='ij')
    a, b = np.shape(Z1)
    mesh_batchS1, mesh_batchS2 \
        = np.reshape(Z1, [-1, 1]), np.reshape(Z2, [-1, 1])  # Z1, Z2

    _, _, _,_,_,_, dV2dx = critic.cal_info(mesh_batchS1, mesh_batchS2, \
                              np.ones_like(mesh_batchS1) * z_ss[2], np.ones_like(mesh_batchS1) * z_ss[3], \
                              np.ones_like(mesh_batchS1) * u_ss[0], np.ones_like(mesh_batchS1) * u_ss[1], \
                              np.ones_like(mesh_batchS1) * z_ss[0], np.ones_like(mesh_batchS1) * z_ss[1],\
                              np.ones_like(mesh_batchS1) * z_ss[2], np.ones_like(mesh_batchS1) * z_ss[3],\
                              np.ones_like(mesh_batchS1) * u_ss[0], np.ones_like(mesh_batchS1) * u_ss[1], mu_,\
                              BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u)

    X, Y, Z  = np.reshape(mesh_batchS1, [a, b]), np.reshape(mesh_batchS2, [a, b]), np.reshape(dV2dx[0], [a, b])
    X, Y, Z2 = np.reshape(mesh_batchS1, [a, b]), np.reshape(mesh_batchS2, [a, b]), np.reshape(dV2dx[1], [a, b])
    X, Y, Z3 = np.reshape(mesh_batchS1, [a, b]), np.reshape(mesh_batchS2, [a, b]), np.reshape(dV2dx[2], [a, b])
    X, Y, Z4 = np.reshape(mesh_batchS1, [a, b]), np.reshape(mesh_batchS2, [a, b]), np.reshape(dV2dx[3], [a, b])

    # fig2 = plt.figure(2)
    # ax = fig2.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z)
    # ax.plot_surface(X, Y, np.zeros_like(Z))
    # ax.set_xlabel('x1'), ax.set_ylabel('x2'), ax.set_zlabel('dV/dx1')
    # plt.savefig('figdVdx1_{0}_mode{1}.pdf'.format(i, mode))
    # plt.pause(1)
    # plt.close('all')
    # fig3 = plt.figure(3)
    # ax2 = fig3.add_subplot(111, projection='3d')
    # ax2.plot_surface(X, Y, Z2)
    # ax2.plot_surface(X, Y, np.zeros_like(Z2))
    # ax2.set_xlabel('x1'), ax2.set_ylabel('x2'), ax2.set_zlabel('dV/dx2')
    # plt.savefig('figdVdx2_{0}_mode{1}.pdf'.format(i, mode))
    # plt.pause(1)
    # plt.close('all')

def warning_b(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, z_min, z_max, lb, ub):
    ind1 = np.where(u1_batch <= lb[1])
    if len(ind1[0]) > 0:
        print('warning!', u1_batch[ind1])
    ind2 = np.where(ub[1] <= u1_batch)
    if len(ind2[0]) > 0:
        print('warning!', u1_batch[ind2])
    ind3 = np.where(u2_batch <= lb[2])
    if len(ind3[0]) > 0:
        print('warning!', u2_batch[ind3])
    ind4 = np.where(ub[2] <= u2_batch)
    if len(ind4[0]) > 0:
        print('warning!', u2_batch[ind4])
    ind5 = np.where(s1_batch <= z_min[0])
    if len(ind5[0]) > 0:
        print('warning!', s1_batch[ind5])
    ind6 = np.where(z_max[0] <= s1_batch)
    if len(ind6[0]) > 0:
        print('warning!', s1_batch[ind6])
    ind7 = np.where(s2_batch <= z_min[1])
    if len(ind7[0]) > 0:
        print('warning!', s2_batch[ind7])
    ind8 = np.where(z_max[1] <= s2_batch)
    if len(ind8[0]) > 0:
        print('warning!', s2_batch[ind8])
    ind9 = np.where(s3_batch <= z_min[2])
    if len(ind9[0]) > 0:
        print('warning!', s3_batch[ind9])
    ind10 = np.where(z_max[2] <= s3_batch)
    if len(ind10[0]) > 0:
        print('warning!', s3_batch[ind10])
    ind11 = np.where(s4_batch <= z_min[3])
    if len(ind11[0]) > 0:
        print('warning!', s4_batch[ind11])
    ind12 = np.where(z_max[3] <= s4_batch)
    if len(ind12[0]) > 0:
        print('warning!', s4_batch[ind12])

def warning_n(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, z_min, z_max, lb, ub):
    ind1 = np.where(u1_batch <= lb[0])
    if len(ind1[0]) > 0:
        print('warning with new constraint!', u1_batch[ind1])
    ind2 = np.where(ub[0] <= u1_batch)
    if len(ind2[0]) > 0:
        print('warning with new constraint!', u1_batch[ind2])
    ind3 = np.where(u2_batch <= lb[1])
    if len(ind3[0]) > 0:
        print('warning with new constraint!', u2_batch[ind3])
    ind4 = np.where(ub[1] <= u2_batch)
    if len(ind4[0]) > 0:
        print('warning with new constraint!', u2_batch[ind4])
    ind5 = np.where(s1_batch <= z_min[0])
    if len(ind5[0]) > 0:
        print('warning with new constraint!', s1_batch[ind5])
    ind6 = np.where(z_max[0] <= s1_batch)
    if len(ind6[0]) > 0:
        print('warning with new constraint!', s1_batch[ind6])
    ind7 = np.where(s2_batch <= z_min[1])
    if len(ind7[0]) > 0:
        print('warning with new constraint!', s2_batch[ind7])
    ind8 = np.where(z_max[1] <= s2_batch)
    if len(ind8[0]) > 0:
        print('warning with new constraint!', s2_batch[ind8])
    ind9 = np.where(s3_batch <= z_min[2])
    if len(ind9[0]) > 0:
        print('warning with new constraint!', s3_batch[ind9])
    ind10 = np.where(z_max[2] <= s3_batch)
    if len(ind10[0]) > 0:
        print('warning with new constraint!', s3_batch[ind10])
    ind11 = np.where(s4_batch <= z_min[3])
    if len(ind11[0]) > 0:
        print('warning with new constraint!', s4_batch[ind11])
    ind12 = np.where(z_max[3] <= s4_batch)
    if len(ind12[0]) > 0:
        print('warning with new constraint!', s4_batch[ind12])

def warning_n_del(s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                  z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch, z_min, z_max, lb, ub):
    ind1 = np.where(u1_batch <= lb[0])
    if len(ind1[0]) > 0:
        print('warning with new constraint!', u1_batch[ind1])
        ind_ = ind1
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind2 = np.where(ub[0] <= u1_batch)
    if len(ind2[0]) > 0:
        print('warning with new constraint!', u1_batch[ind2])
        ind_ = ind2
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind3 = np.where(u2_batch <= lb[1])
    if len(ind3[0]) > 0:
        print('warning with new constraint!', u2_batch[ind3])
        ind_ = ind3
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind4 = np.where(ub[1] <= u2_batch)
    if len(ind4[0]) > 0:
        print('warning with new constraint!', u2_batch[ind4])
        ind_ = ind4
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind5 = np.where(s1_batch <= z_min[0])
    if len(ind5[0]) > 0:
        print('warning with new constraint!', s1_batch[ind5])
        ind_ = ind5
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind6 = np.where(z_max[0] <= s1_batch)
    if len(ind6[0]) > 0:
        print('warning with new constraint!', s1_batch[ind6])
        ind_ = ind6
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind7 = np.where(s2_batch <= z_min[1])
    if len(ind7[0]) > 0:
        print('warning with new constraint!', s2_batch[ind7])
        ind_ = ind7
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind8 = np.where(z_max[1] <= s2_batch)
    if len(ind8[0]) > 0:
        print('warning with new constraint!', s2_batch[ind8])
        ind_ = ind8
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind9 = np.where(s3_batch <= z_min[2])
    if len(ind9[0]) > 0:
        print('warning with new constraint!', s3_batch[ind9])
        ind_ = ind9
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind10 = np.where(z_max[2] <= s3_batch)
    if len(ind10[0]) > 0:
        print('warning with new constraint!', s3_batch[ind10])
        ind_ = ind10
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind11 = np.where(s4_batch <= z_min[3])
    if len(ind11[0]) > 0:
        print('warning with new constraint!', s4_batch[ind11])
        ind_ = ind11
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    ind12 = np.where(z_max[3] <= s4_batch)
    if len(ind12[0]) > 0:
        print('warning with new constraint!', s4_batch[ind12])
        ind_ = ind12
        s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
        z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch= \
            delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
                    z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch)

    return s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
           z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch


def delete_(ind_, s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
            z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch):
    print(ind_)
    print(np.shape(s1_batch))
    s1_batch = np.delete(s1_batch, np.squeeze(ind_), axis=0)
    s2_batch = np.delete(s2_batch, np.squeeze(ind_), axis=0)
    s3_batch = np.delete(s3_batch, np.squeeze(ind_), axis=0)
    s4_batch = np.delete(s4_batch, np.squeeze(ind_), axis=0)
    u1_batch = np.delete(u1_batch, np.squeeze(ind_), axis=0)
    u2_batch = np.delete(u2_batch, np.squeeze(ind_), axis=0)
    a1_batch = np.delete(a1_batch, np.squeeze(ind_), axis=0)
    a2_batch = np.delete(a2_batch, np.squeeze(ind_), axis=0)

    z1_ss_batch = np.delete(z1_ss_batch, np.squeeze(ind_), axis=0)
    z2_ss_batch = np.delete(z2_ss_batch, np.squeeze(ind_), axis=0)
    z3_ss_batch = np.delete(z3_ss_batch, np.squeeze(ind_), axis=0)
    z4_ss_batch = np.delete(z4_ss_batch, np.squeeze(ind_), axis=0)
    u1_ss_batch = np.delete(u1_ss_batch, np.squeeze(ind_), axis=0)
    u2_ss_batch = np.delete(u2_ss_batch, np.squeeze(ind_), axis=0)

    print(np.shape(s1_batch))
    return s1_batch, s2_batch, s3_batch, s4_batch, u1_batch, u2_batch, a1_batch, a2_batch, \
           z1_ss_batch, z2_ss_batch, z3_ss_batch, z4_ss_batch, u1_ss_batch, u2_ss_batch


def FIX_BO(s, u, z_min, z_max, u_min, u_max, BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u, EPS_C):

    if u[0] - u_min[0] < BO_LB_u[0]:
        print('fix BO_LB_u[0]')
        BO_LB_u[0] = u[0] - u_min[0] - EPS_C
    if u[1] - u_min[1] < BO_LB_u[1]:
        print('fix BO_LB_u[1]')
        BO_LB_u[1] = u[1] - u_min[1] - EPS_C
    if u_max[0] - u[0] < BO_UB_u[0]:
        print('fix BO_UB_u[0]')
        BO_UB_u[0] = u_max[0] - u[0] - EPS_C
    if u_max[1] - u[1] < BO_UB_u[1]:
        print('fix BO_UB_u[1]')
        BO_UB_u[1] = u_max[1] - u[1] - EPS_C

    if s[0] - z_min[0] < BO_LB_z[0]:
        print('fix BO_LB_z[0]')
        BO_LB_z[0] = s[0] - z_min[0] - EPS_C
    if s[1] - z_min[1] < BO_LB_z[1]:
        print('fix BO_LB_z[1]')
        BO_LB_z[1] = s[1] - z_min[1] - EPS_C
    if s[2] - z_min[2] < BO_LB_z[2]:
        print('fix BO_LB_z[2]')
        BO_LB_z[2] = s[2] - z_min[2] - EPS_C
    if s[3] - z_min[3] < BO_LB_z[3]:
        print('fix BO_LB_z[3]')
        BO_LB_z[3] = s[3] - z_min[3] - EPS_C
    if z_max[0] - s[0] < BO_UB_z[0]:
        print('fix BO_UB_z[0]')
        BO_UB_z[0] = z_max[0] - s[0] - EPS_C
    if z_max[1] - s[1] < BO_UB_z[1]:
        print('fix BO_UB_z[1]')
        BO_UB_z[1] = z_max[1] - s[1] - EPS_C
    if z_max[2] - s[2] < BO_UB_z[2]:
        print('fix BO_UB_z[2]')
        BO_UB_z[2] = z_max[2] - s[2] - EPS_C
    if z_max[3] - s[3] < BO_UB_z[3]:
        print('fix BO_UB_z[3]')
        BO_UB_z[3] = z_max[3] - s[3] - EPS_C

    return BO_LB_z, BO_UB_z, BO_LB_u, BO_UB_u


def is_nan(x):
    return (x != x)