import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt

import do_mpc
from casadi import *

class three_tank_system(object):

    def __init__(self):
        model_type = 'continuous'
        self.model = do_mpc.model.Model(model_type=model_type)

        ######################################################
        self.xs = np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877])
        self.us = 1.12 * np.array([2.9e9, 1.0e9, 2.9e9])
        ######################################################

        self.t = 0
        self.action_sample_period = 20
        self.sampling_period = 0.005
        self.h = 0.001
        self.sampling_steps = int(self.sampling_period/self.h)
        self.delay = 5

        self.s2hr = 3600
        self.MW = 250e-3
        self.sum_c = 2E3
        self.T10 = 300
        self.T20 = 300
        self.F10 = 5.04
        self.F20 = 5.04
        self.Fr = 50.4
        self.Fp = 0.504
        self.V1 = 1
        self.V2 = 0.5
        self.V3 = 1
        self.E1 = 5e4
        self.E2 = 6e4
        self.k1 = 2.77e3 * self.s2hr
        self.k2 = 2.6e3 * self.s2hr
        self.dH1 = -6e4 / self.MW
        self.dH2 = -7e4 / self.MW
        self.aA = 3.5
        self.aB = 1
        self.aC = 0.5
        self.Cp = 4.2e3
        self.R = 8.314
        self.rho = 1000
        self.xA10 = 1
        self.xB10 = 0
        self.xA20 = 1
        self.xB20 = 0
        self.Hvap1 = -35.3E3 * self.sum_c
        self.Hvap2 = -15.7E3 * self.sum_c
        self.Hvap3 = -40.68E3 * self.sum_c

        self.kw = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # noise deviation
        self.bw = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5]) # noise bound

        self.action_low = 0.2 * self.us
        self.action_high = 1.5 * self.us
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.a_dim = self.us.shape[0]

        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.x_dim = self.xs.shape[0]

        self.seed()
        self.state = None

        self.init = False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action, impulse = 0):
        action = np.clip(action, self.action_low, self.action_high)
        
        x0 = self.state
        for i in range(self.sampling_steps):
            process_noise = np.random.normal(np.zeros_like(self.kw),self.kw)
            process_noise = np.clip(process_noise, -self.bw, self.bw)
            x0 = x0 + self.derivate(x0, action)[1]*self.h + process_noise*self.h
            
        self.state = x0
        self.t += 1
        
        cost = np.linalg.norm(self.state - self.xs)
        done = False
        data_collection_done = False
        
        return x0, cost, done, dict(reference=self.xs,data_collection_done=data_collection_done)
    

    def onestep(self,x,u,shift_):
        [shift, scale, shift_u, scale_u] = shift_

        x[0] = x[0]*scale[0]+shift[0]
        x[1] = x[1]*scale[1]+shift[1]
        x[2] = x[2]*scale[2]+shift[2]

        x[3] = x[3]*scale[3]+shift[3]
        x[4] = x[4]*scale[4]+shift[4]
        x[5] = x[5]*scale[5]+shift[5]

        x[6] = x[6]*scale[6]+shift[6]
        x[7] = x[7]*scale[7]+shift[7]
        x[8] = x[8]*scale[8]+shift[8]

        u[0] = u[0]*scale_u[0]+shift_u[0]
        u[1] = u[1]*scale_u[1]+shift_u[1]
        u[2] = u[2]*scale_u[2]+shift_u[2]

        for i in range(self.sampling_steps):
            x = x+self.derivate(x,u)[0]
        x = (x-shift)/scale

        return x

    def _build_controller(self,shift_, N, Q, R):
        [self.shift, self.scale, self.shift_u, self.scale_u] = shift_        
        self.xs_norm = (self.xs - self.shift) / self.scale
        
        self.a_bound_high = (self.action_high - self.shift_u) / self.scale_u
        self.a_bound_low =  (self.action_low - self.shift_u) / self.scale_u

        self.control_horizon = N

        x =  MX.sym('x',self.x_dim,self.control_horizon+1)
        u =  MX.sym('u',self.a_dim,self.control_horizon)
        self.x0    = MX.sym('x0', self.x_dim)
        self.J   = 0

        constraints = []
        constraints.append(x[:, 0] - self.x0)
        for k in range(self.control_horizon-1):
            constraints.append(x[:,k+1]-self.onestep(x[:,k],u[:,k],shift_))
            error_x = x[:,k]-self.xs_norm
            delta_u = u[:,k+1]-u[:,k]
            self.J += mtimes([error_x.T, Q, error_x]) + mtimes([delta_u.T, R, delta_u])
        
        k+=1
        error_x = x[:,k]-self.xs_norm
        self.J +=  mtimes([error_x.T, Q, error_x])


        nlp = {'x': vertcat(reshape(u, -1, 1), reshape(x,-1, 1)),  #first U and then X
               'f': self.J, 
               'g': vertcat(*constraints),
               'p': self.x0}
        
        opts = {'ipopt': {'print_level': 0}}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)


        self.lbx = np.concatenate([np.tile(self.a_bound_low,self.control_horizon),  np.full(self.x_dim*(self.control_horizon+1), -np.inf)])  # No constraints on x
        self.ubx = np.concatenate([np.tile(self.a_bound_high,self.control_horizon), np.full(self.x_dim*(self.control_horizon+1), np.inf)]) 

        self.lbg = 0
        self.ubg = 0
        
    def choose_action(self,x_0):
        x0_norm = (x_0-self.shift)/self.scale
        if not self.init:
            self.init = True
            self.x0_guess = np.concatenate([np.zeros((self.a_dim*self.control_horizon, 1)), np.tile(x0_norm, (self.control_horizon+1, 1)).T.reshape(-1, 1)])
        sol = self.solver(x0=self.x0_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x0_norm)
        
        U_opt = np.array(vertsplit(sol['x'])[:self.a_dim*(self.control_horizon)]).flatten()
        X_opt = np.array(vertsplit(sol['x'])[self.a_dim*(self.control_horizon):]).flatten()
        Pred_U = U_opt.reshape([self.a_dim,-1])
        Pred_X = X_opt.reshape([self.x_dim,-1])

        self.x0_guess = np.array(vertsplit(sol['x'])).flatten()
        u_out  = Pred_U[:,0] * self.scale_u + self.shift_u
        
        return u_out

    def reset(self):
        self.a_holder = self.action_space.sample()
        self.state = np.random.uniform(0.8, 1.2) * self.xs + np.random.normal(np.zeros_like(self.xs), self.xs*0.01)
        # self.state = self.xs
        self.t = 0
        self.time = 0
        return self.state

    def run_MPC(self, Q, R, N, shift_, episode_length):
        [shift, scale, _, _] = shift_
        self._build_controller(shift_, N, Q, R)
        
        path = []
        a_path = []
        cost_path = []
        self.reset()
        for i in range(episode_length):
            a = self.choose_action(self.state)
            x_ = self.step(a)[0]
            x_norm = (x_ - shift) / scale
            error_norm = np.linalg.norm(x_norm - self.xs_norm)
            path.append(x_)
            a_path.append(a)
            cost_path.append(error_norm)

        path = np.array(path)
        a_path = np.array(a_path)
        cost_path = np.array(cost_path)
        
        fig, ax = plt.subplots(self.x_dim, sharex=True, figsize=(15, 15))
        t = range(episode_length)
        for i in range(self.x_dim):
            #state实线，黄色，xs虚线，红色
            ax[i].plot(t, path[:, i], color='blue', label='state')
            ax[i].plot(t, [self.xs[i]]*episode_length, color='red', linestyle='--', label='xs')

        # 画出cost_norm轨迹
        fig2, ax2 = plt.subplots(1, figsize=(15, 5))
        ax2.plot(t, cost_path, color='blue', label='cost_norm')
        ax2.set_xlabel('time')
        ax2.set_ylabel('cost_norm')
        ax2.set_title('cost_norm')

        # 返回数值和图片
        return path, a_path, cost_path, fig, ax, fig2, ax2

    
    def derivate(self, x, us):
        xA1 = x[0]
        xB1 = x[1]
        T1 = x[2]

        xA2 = x[3]
        xB2 = x[4]
        T2 = x[5]

        xA3 = x[6]
        xB3 = x[7]
        T3 = x[8]

        Q1 = us[0]
        Q2 = us[1]
        Q3 = us[2]

        xC3 = 1 - xA3 - xB3
        x3a = self.aA * xA3 + self.aB * xB3 + self.aC * xC3

        xAr = self.aA * xA3 / x3a
        xBr = self.aB * xB3 / x3a
        xCr = self.aC * xC3 / x3a

        F1 = self.F10 + self.Fr
        F2 = F1 + self.F20
        F3 = F2 - self.Fr - self.Fp

        f1 = self.F10 * (self.xA10 - xA1) / self.V1 + self.Fr * (xAr - xA1) / self.V1 - self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
        f2 = self.F10 * (self.xB10 - xB1) / self.V1 + self.Fr * (xBr - xB1) / self.V1 + self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1 - self.k2 * np.exp(
            -self.E2 / (self.R * T1)) * xB1
        f3 = self.F10 * (self.T10 - T1) / self.V1 + self.Fr * (T3 - T1) / self.V1 - self.dH1 * self.k1 * np.exp(
            -self.E1 / (self.R * T1)) * xA1 / self.Cp - self.dH2 * self.k2 * np.exp(
            -self.E2 / (self.R * T1)) * xB1 / self.Cp + Q1 / (self.rho * self.Cp * self.V1)

        f4 = F1 * (xA1 - xA2) / self.V2 + self.F20 * (self.xA20 - xA2) / self.V2 - self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2
        f5 = F1 * (xB1 - xB2) / self.V2 + self.F20 * (self.xB20 - xB2) / self.V2 + self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2 - self.k2 * np.exp(
            -self.E2 / (self.R * T2)) * xB2
        f6 = F1 * (T1 - T2) / self.V2 + self.F20 * (self.T20 - T2) / self.V2 - self.dH1 * self.k1 * np.exp(
            -self.E1 / (self.R * T2)) * xA2 / self.Cp - self.dH2 * self.k2 * np.exp(
            -self.E2 / (self.R * T2)) * xB2 / self.Cp + Q2 / (self.rho * self.Cp * self.V2)

        f7 = F2 * (xA2 - xA3) / self.V3 - (self.Fr + self.Fp) * (xAr - xA3) / self.V3
        f8 = F2 * (xB2 - xB3) / self.V3 - (self.Fr + self.Fp) * (xBr - xB3) / self.V3
        f9 = F2 * (T2 - T3) /self.V3 + Q3 / (self.rho * self.Cp * self.V3) + (self.Fr + self.Fp) * (xAr * self.Hvap1 + xBr * self.Hvap2 + xCr * self.Hvap3) / (
                self.rho * self.Cp * self.V3)

        F = [f1*self.h, f2*self.h, f3*self.h, f4*self.h, f5*self.h, f6*self.h, f7*self.h, f8*self.h, f9*self.h]
        
        return vertcat(*F), np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])
    

    def get_action(self):

        if self.t % self.action_sample_period == 0:
            self.a_holder = self.action_space.sample()
        a = self.a_holder + np.random.normal(np.zeros_like(self.us), self.us*0.01)
        a = np.clip(a, self.action_low, self.action_high)

        return a

    def get_noise(self):
        scale = 0.1 * self.xs
        return np.random.normal(np.zeros_like(self.xs), scale)


if __name__=='__main__':

    env = three_tank_system()
    Q = np.diag([0.5,0.5,1.5,0.5,0.5,1.5,0.5,0.5,1.5])
    R = np.diag([0.005,0.005,0.005])
    N = 15
    shift_u = np.array([2.75185422e+09, 9.44883669e+08, 2.75257156e+09])
    scale_u = np.array([1.20379914e+09, 4.24970517e+08, 1.25602098e+09])
    shift = np.array([5.17164371e-01, 4.17246447e-01, 4.35001943e+02, 5.24451997e-01, 4.10199461e-01, 4.29703381e+02, 3.44656391e-01, 5.39367112e-01, 4.29313955e+02])
    scale = np.array([0.28416151,  0.21946247, 41.2900136,   0.27521402,  0.21124655, 39.26672551,  0.27504289,  0.19883226, 41.11384814])
    shift_ = [shift, scale, shift_u, scale_u]
    episode_length = 200
    path, a_path, cost_path, fig, ax, fig2, ax2 = env.run_MPC(Q, R, N, shift_, episode_length)
    plt.show()








