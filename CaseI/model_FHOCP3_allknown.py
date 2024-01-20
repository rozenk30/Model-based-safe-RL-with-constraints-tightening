#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *

def model_FHOCP3_allknown(z, delt, Hor_n, Q1_, Q2_, R_, nz, nu, z1des, z2des, z3des, z4des, u1des, u2des, pu0, lb, ub, pa0, mismatch,
                          dist_s1_pool, dist_s2_pool, dist_s3_pool, dist_s4_pool):
    m = ConcreteModel()
    Hp = delt * Hor_n
    m.t = ContinuousSet(bounds=(0,Hp))
    # State index
    state_idx = [i for i in range(1, nz+1)]  #range(1, 3) => [1, 2]
    m.z_idx = Set(initialize=state_idx) # [1,2]
    # States
    # z_warm = {1: z[0], 2: z[1]}
    z_warm = {1: z1des[0], 2: z2des[0], 3: z3des[0], 4: z4des[0]}
    def z_init(m, ti, zi):
        return z_warm[zi]

    z_min, z_max = [1.,1.,1.,1.], [28, 28, 28, 28]
    z_lb, z_ub = {1: z_min[0], 2: z_min[1], 3: z_min[2], 4: z_min[3]}, {1: z_max[0], 2: z_max[1], 3: z_max[2], 4: z_max[3]}
    def z_const(m, ti, zi):
        return (z_lb[zi], z_ub[zi])
    m.z = Var(m.t, m.z_idx, domain=NonNegativeReals, bounds=z_const, initialize=z_init)
    # Input index
    input_idx = [i for i in range(1, nu+1)]  #rangeset(1, 2) => [1, 2]
    m.u_idx = Set(initialize=input_idx) # [1,2]
    #u_warm = {1: u1des[0], 2: u2des[0]}#
    u_warm = {1: pu0[0], 2: pu0[1]}
    def u_init(m, ti, ui):
        return u_warm[ui]
    def u_const(m, ti, ui):
        return (lb[ui], ub[ui])
    # Inputs
    m.u = Var(m.t, m.u_idx, bounds=u_const, initialize=u_init)
    # Parameters
    est_idx = [i for i in range(0, Hor_n+1)]  # 0, 1, 2, 3, 4, Hor_n-1
    m.est_idx = Set(initialize=est_idx)
    m.dist_s1 = Param(m.est_idx, initialize=dist_s1_pool)
    m.dist_s2 = Param(m.est_idx, initialize=dist_s2_pool)
    m.dist_s3 = Param(m.est_idx, initialize=dist_s3_pool)
    m.dist_s4 = Param(m.est_idx, initialize=dist_s4_pool)
    # Model
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.dudt = DerivativeVar(m.u, wrt=m.t)
    # Constant
    A1, A2, A3, A4 = 50.27 * mismatch[0], 50.27 * mismatch[1], 28.27 * mismatch[2], 28.27 * mismatch[3]
    gamma1, gamma2 = 0.4, 0.4
    a1, a2, a3, a4 = 0.233, 0.242, 0.127, 0.127
    g = 980 # cm/s2
    def _z1dot(m, i):   # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 1] == - a1 / A1 * sqrt(2 * g * m.z[i,1]) + a3 / A1 * sqrt(2 * g * m.z[i,3]) \
                                   + gamma1 / A1 * m.u[i,1] + dist_s1_pool[int(i/delt)]

    def _z2dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 2] == - a2 / A2 * sqrt(2 * g * m.z[i, 2]) + a4 / A2 * sqrt(2 * g * m.z[i, 4]) \
                   + gamma2 / A2 * m.u[i, 2] + dist_s2_pool[int(i/delt)]

    def _z3dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 3] == - a3 / A3 * sqrt(2 * g * m.z[i, 3]) \
                   + (1 - gamma2) / A3 * m.u[i, 2] + dist_s3_pool[int(i/delt)]

    def _z4dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 4] == - a4 / A4 * sqrt(2 * g * m.z[i, 4]) \
                   + (1 - gamma1) / A4 * m.u[i, 1] + dist_s4_pool[int(i/delt)]

    m.z1_inicon = Constraint(expr=m.z[0, 1] - z[0] == 0.0)
    m.z2_inicon = Constraint(expr=m.z[0, 2] - z[1] == 0.0)
    m.z3_inicon = Constraint(expr=m.z[0, 3] - z[2] == 0.0)
    m.z4_inicon = Constraint(expr=m.z[0, 4] - z[3] == 0.0)
    m.z1_dotcon = Constraint(m.t, rule=_z1dot)
    m.z2_dotcon = Constraint(m.t, rule=_z2dot)
    m.z3_dotcon = Constraint(m.t, rule=_z3dot)
    m.z4_dotcon = Constraint(m.t, rule=_z4dot)

    # Input bounds
    a_warm = {1: pa0[0], 2: pa0[1]}
    def a_init(m, ti, ui):
        return a_warm[ui]
    m.a = Var(m.t, m.u_idx, initialize=a_init) #, bounds=delu_const

    def _u1dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dudt[i, 1] - m.a[i, 1] == 0

    def _u2dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dudt[i, 2] - m.a[i, 2] == 0

    m.u1_inicon = Constraint(expr=m.u[0, 1] - pu0[0] == 0.0)
    m.u2_inicon = Constraint(expr=m.u[0, 2] - pu0[1] == 0.0)
    m.u1_dotcon = Constraint(m.t, rule=_u1dot)
    m.u2_dotcon = Constraint(m.t, rule=_u2dot)

    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=Hor_n, ncp=3, wrt=m.t)
    discretizer.reduce_collocation_points(m, var=m.a, ncp=1, contset=m.t)
    m.nfe = m.t.get_finite_elements()[1:]  # [0., 1., ..., Hp] =>[1., ..., Hp]
    # nfe = [delt * i for i in range(1, Hp+1)] # range(1, Hp+1) => [1, Hp]
    m.ncp = m.t.get_discretization_info()['tau_points'][1:]  # [0.1550510257216822, 0.6449489742783179, 1.0]
    # m.ncp = m.t.get_discretization_info()['tau_points'] # [0.0, 0.1550510257216822, 0.6449489742783179, 1.0]
    def _obj(m):
        return sum(delt*Q1_*(m.z[i, 1]-z1des[int((i-1)/delt)])**2/ ((z_max[0] - z_min[0])** 2) \
                 + delt*Q1_*(m.z[i, 2]-z2des[int((i-1)/delt)])**2/ ((z_max[1] - z_min[1])** 2) \
                 + delt*Q1_*(m.z[i, 3]-z3des[int((i-1)/delt)])**2/ ((z_max[2] - z_min[2])** 2) \
                 + delt*Q1_*(m.z[i, 4]-z4des[int((i-1)/delt)])**2/ ((z_max[3] - z_min[3])** 2) \
                 + delt * Q2_ * (m.u[i, 1] - u1des[int(i - 1)]) ** 2 / ((ub[1] - lb[1]) ** 2) \
                 + delt * Q2_ * (m.u[i, 2] - u2des[int(i - 1)]) ** 2 / ((ub[2] - lb[2]) ** 2) \
                 + delt*R_*(m.a[i,1]**2) + delt*R_*(m.a[i,2]**2) for i in m.nfe)

    m.obj = Objective(rule=_obj)
    return m