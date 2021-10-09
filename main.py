"""
*************************************************************************************************************
Version
    Last revision: October 2021
    Author: Spilios Evmorfos

Purpose
    The purpose of this code is to support the paper:
    S. Evmorfos, A. Petropulu, Deep Actor-Critic for Continuous 3D Motion Control in Mobile Relay Networks,
    International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

   Any part of this code used in your work should cite the above publication.

This code is provided "as is" to support the ideals of reproducible research. Any issues with this
code should be reported by email to se386@scarletmail.rutgers.edu

The code is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
available at https://creativecommons.org/licenses/by-nc-sa/4.0/

*************************************************************************************************************
"""




import numpy as np
import scipy
from scipy import spatial, linalg, io 
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from helper_functions import helper_function, create_data_for_shadowing_multipath_phase
from Environment import ContinuousEnvironment
from Replay_Memory import ReplayMemory
from CriticNetwork import CriticNetwork
from ValueNetwork import ValueNetwork
from Actor import ActorNetwork
from Agent import Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # initialize a device

if __name__ ==  '__main__':
    # parameters for the environment
    grid_min = 0
    grid_max = 25
    spacing = ((grid_max-grid_min)/ grid_max)
    max_dim = 20
    rMap_RowCells = max_dim
    rMap_ColCells = max_dim
    rMap_zCells = max_dim
    ell = 2.3
    rho = 3
    P_S = 10**((45-30)/10)
    P_R = 10**((55-30)/10)
    sigmaSQ = 1
    sigma_xiSQ = 1.5 
    sigma_DSQ = 1
    etaSQ = 6
    c1 = 1.2
    c2 = 0.6
    c3 = 0.5
    m_step = 1
    numSlots = 400
    numRelays = 3
    
    
    
    
    
    
    
    
    T, C_SD, C_SD_chol, kappa, grid_X, grid_Y, S_cord, D_cord  = helper_function(grid_min, spacing, rMap_RowCells, rMap_zCells, rMap_ColCells, ell, rho, sigmaSQ, sigma_xiSQ, sigma_DSQ, etaSQ, c1, c2, c3, numSlots)

    f_Smaps = np.load(r"C:\Users\Spilios\OneDrive\Desktop\SoftActorCriticSiren3D\F_S_3D.npy")
    f_Dmaps = np.load(r"C:\Users\Spilios\OneDrive\Desktop\SoftActorCriticSiren3D\F_D_3D.npy")
    f_Sphase = np.load(r"C:\Users\Spilios\OneDrive\Desktop\SoftActorCriticSiren3D\F_Sph_3D.npy")
    f_Dphase = np.load(r"C:\Users\Spilios\OneDrive\Desktop\SoftActorCriticSiren3D\F_Dph_3D.npy")

    drone_dim = 0.2
    env = ContinuousEnvironment(max_dim, drone_dim, f_Smaps, f_Dmaps, f_Sphase, f_Dphase, S_cord, D_cord, ell, rho, P_R, P_S, sigma_DSQ, sigmaSQ)
    
    
    agent = Agent()
    epochs = 50
    rewards_per_epoch = []
    import matplotlib.pyplot as plt
    #memory fill
    print("filling memory")
    for j in range(10):
        state_relays = np.array([[10., 8., 0.],[10.,10., 0.],[10.,12., 0.]], np.float32)
        state_relays = state_relays
        for i in range(numSlots):
            new_state_relays = np.zeros_like(state_relays)
            for k in range(3):
                state = state_relays[k]
                state = np.array(state)
                action = np.random.normal(0, 1/np.sqrt(2), size= state.size)
                action = np.array(action)
                new_state = env.env_step(state, action)
                new_state = np.clip(new_state, drone_dim/2, max_dim - 0.5*drone_dim)
                new_state_relays[k] = new_state
                VI = env.calculate_reward(new_state, i) / 3
                agent.remember(state, action, VI, new_state)
            state_relays = new_state_relays

    print("memory is filled... beginning training")


    for i in range(epochs):
        state_relays = np.array([[10.,8., 0.],[10.,10., 0.],[10.,12, 0.]], np.float32)
        state_relays = state_relays
        accumulated_reward = 0
        for j in range(numSlots):
            new_state_relays = np.zeros_like(state_relays)
            for k in range(3):
                state = state_relays[k]
                state = np.array(state)
                action = agent.choose_action(state)
                new_state = env.env_step(state, action)
                
                new_state_relays[k] = new_state
                VI = env.calculate_reward(new_state, j) / 3
                if k==1:
                    agent.remember(state, action, VI, state)
                accumulated_reward += VI
            print(new_state_relays)
            state_relays = new_state_relays
            agent.learn()
        accumulated_reward = accumulated_reward / numSlots
        rewards_per_epoch.append(accumulated_reward)



    rewards_per_epoch = np.array(rewards_per_epoch)
    k = rewards_per_epoch.size



    a = [i for i in range(k)]

    results = np.zeros(k)
    for i in range(k):
        results[i] = rewards_per_epoch[i]
        results[i] = 10*np.log10(results[i])

    

    plt.plot(a,results)
    plt.show()