import gym
import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
from gym.utils import seeding

class ToyEnv(gym.Env):
    def __init__(self, z_ss, u_ss):  #initializatio
        self.zmin = [7.5, 7.5, 14.2,  4.5]
        self.zmax = [28., 28.,  28., 21.3]
        self.umin = [ 0.,  0.]
        self.umax = [60., 60.]
        self.t0 = 0.
        self.dt = 3
        self.reset(z_ss, u_ss)
        self.state_dim = 4
        self.V_dim = 1
        self.u_dim = 2
        self.action_dim = 2


    def reset(self, s0, u0):
        self.state = [s0[0], s0[1], s0[2], s0[3]]  #random initial states
        self.u = [u0[0], u0[1]]
        self.state_ext= np.concatenate((self.state, self.u), axis=None) # (4,)
        # self.state = [1., 1.]  #check code
        self.action = [0., 0.]
        self.time = self.t0 #initialization
        print("reset")
        return np.array(self.state), np.array(self.u)