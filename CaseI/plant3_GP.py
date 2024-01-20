#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *
import warnings
def plant(z, u, delt, nz, nu, a, lb, ub, dist_z, dist_u):
  #  warnings.filterwarnings("ignore")
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, delt))
    # state index
    state_idx = [i for i in range(1, nz+1)]  # range(1, 3) => [1, 2]
    m.z_idx = Set(initialize=state_idx)      # [1,2]
    # States
    z_warm = {1: z[0], 2: z[1], 3: z[2], 4: z[3]}
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
    u_warm = {1: u[0], 2: u[1]}
    def u_init(m, ti, ui):
        return u_warm[ui]
    def u_const(m, ti, ui):
        return (lb[ui], ub[ui])
    # Inputs
    m.u = Var(m.t, m.u_idx, bounds=u_const, initialize=u_init)
    # Model
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.dudt = DerivativeVar(m.u, wrt=m.t)
    # Constant
    A1, A2, A3, A4 = 50.27, 50.27, 28.27, 28.27
    gamma1, gamma2 = 0.4, 0.4
    a1, a2, a3, a4 = 0.233, 0.242, 0.127, 0.127
    g = 980 # cm/s2

    def _z1dot(m, i):   # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 1] == - a1 / A1 * sqrt(2 * g * m.z[i,1]) + a3 / A1 * sqrt(2 * g * m.z[i,3]) \
                                   + gamma1 / A1 * m.u[i,1] + dist_z[0]

    def _z2dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 2] == - a2 / A2 * sqrt(2 * g * m.z[i, 2]) + a4 / A2 * sqrt(2 * g * m.z[i, 4]) \
                   + gamma2 / A2 * m.u[i, 2] + dist_z[1]

    def _z3dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 3] == - a3 / A3 * sqrt(2 * g * m.z[i, 3]) \
                   + (1 - gamma2) / A3 * m.u[i, 2] + dist_z[2]

    def _z4dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 4] == - a4 / A4 * sqrt(2 * g * m.z[i, 4]) \
                   + (1 - gamma1) / A4 * m.u[i, 1] + dist_z[3]

    def _u1dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dudt[i, 1] - a[0] == dist_u[0]

    def _u2dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dudt[i, 2] - a[1] == dist_u[1]

    m.u1_inicon = Constraint(expr=m.u[0, 1] - u[0] == 0.0)
    m.u2_inicon = Constraint(expr=m.u[0, 2] - u[1] == 0.0)
    m.u1_dotcon = Constraint(m.t, rule=_u1dot)
    m.u2_dotcon = Constraint(m.t, rule=_u2dot)
    m.z1ini = Constraint(expr=m.z[0,1] - z[0] == 0.0)
    m.z2ini = Constraint(expr=m.z[0,2] - z[1] == 0.0)
    m.z3ini = Constraint(expr=m.z[0,3] - z[2] == 0.0)
    m.z4ini = Constraint(expr=m.z[0,4] - z[3] == 0.0)

    m.z1_dotcon = Constraint(m.t, rule=_z1dot)
    m.z2_dotcon = Constraint(m.t, rule=_z2dot)
    m.z3_dotcon = Constraint(m.t, rule=_z3dot)
    m.z4_dotcon = Constraint(m.t, rule=_z4dot)
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=20, ncp=3, wrt=m.t)
    m.obj = 1.
    return m

def model(z, u, delt, nz, nu, a, lb, ub, dist_z, dist_u, mismatch):
  #  warnings.filterwarnings("ignore")
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, delt))
    # state index
    state_idx = [i for i in range(1, nz+1)]  # range(1, 3) => [1, 2]
    m.z_idx = Set(initialize=state_idx)      # [1,2]
    # States
    z_warm = {1: z[0], 2: z[1], 3: z[2], 4: z[3]}
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
    u_warm = {1: u[0], 2: u[1]}
    def u_init(m, ti, ui):
        return u_warm[ui]
    def u_const(m, ti, ui):
        return (lb[ui], ub[ui])
    # Inputs
    m.u = Var(m.t, m.u_idx, bounds=u_const, initialize=u_init)
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
                                   + gamma1 / A1 * m.u[i,1] + dist_z[0]

    def _z2dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 2] == - a2 / A2 * sqrt(2 * g * m.z[i, 2]) + a4 / A2 * sqrt(2 * g * m.z[i, 4]) \
                   + gamma2 / A2 * m.u[i, 2] + dist_z[1]

    def _z3dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 3] == - a3 / A3 * sqrt(2 * g * m.z[i, 3]) \
                   + (1 - gamma2) / A3 * m.u[i, 2] + dist_z[2]

    def _z4dot(m, i):  # u1 = u[0], u2 = u[1]  BE CAREFUL !!
        if i == 0:
            return Constraint.Skip
        else:
            return m.dzdt[i, 4] == - a4 / A4 * sqrt(2 * g * m.z[i, 4]) \
                   + (1 - gamma1) / A4 * m.u[i, 1] + dist_z[3]

    def _u1dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dudt[i, 1] - a[0] == 0#dist_u[0]

    def _u2dot(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dudt[i, 2] - a[1] == 0#dist_u[1]

    m.u1_inicon = Constraint(expr=m.u[0, 1] - u[0] == 0.0)
    m.u2_inicon = Constraint(expr=m.u[0, 2] - u[1] == 0.0)
    m.u1_dotcon = Constraint(m.t, rule=_u1dot)
    m.u2_dotcon = Constraint(m.t, rule=_u2dot)
    m.z1ini = Constraint(expr=m.z[0,1] - z[0] == 0.0)
    m.z2ini = Constraint(expr=m.z[0,2] - z[1] == 0.0)
    m.z3ini = Constraint(expr=m.z[0,3] - z[2] == 0.0)
    m.z4ini = Constraint(expr=m.z[0,4] - z[3] == 0.0)

    m.z1_dotcon = Constraint(m.t, rule=_z1dot)
    m.z2_dotcon = Constraint(m.t, rule=_z2dot)
    m.z3_dotcon = Constraint(m.t, rule=_z3dot)
    m.z4_dotcon = Constraint(m.t, rule=_z4dot)
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=20, ncp=3, wrt=m.t)
    m.obj = 1.
    return m