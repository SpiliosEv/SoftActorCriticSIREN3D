import numpy as np

class ContinuousEnvironment(object):
  def __init__(self, max_dim, drone_dim, f_Smaps, f_Dmaps, f_Sphase, f_Dphase, S_cord, D_cord, ell, rho, P_R, P_S, sigma_DSQ, sigmaSQ):
    self.max_dim = max_dim
    self.drone_dim = drone_dim
    self.f_Smaps = f_Smaps
    self.f_Dmaps = f_Dmaps
    self.f_Sphase = f_Sphase 
    self.f_Dphase = f_Dphase
    self.S_cord = S_cord
    self.D_cord = D_cord
    self.ell = ell
    self.rho = rho
    self.P_R = P_R
    self.P_S = P_S
    self.sigma_DSQ = sigma_DSQ
    self.sigmaSQ = sigmaSQ




  def env_step(self, state, action):
    # state is a numpy array (1x2) np.float32
    # action is a numpy array (1x2) np.float32
    new_state = state + action 
    new_state = np.clip(new_state, (0 + self.drone_dim/2), self.max_dim - (self.drone_dim/2)) # clip the new state parameters to make sure that the drone is not out of bounds
    

    return new_state

  def calculate_reward(self, state, time_step):
 

    pathlossF = -self.ell*10*np.log10(np.linalg.norm(state - self.S_cord)) + self.rho
    pathlossG = -self.ell*10*np.log10(np.linalg.norm(state - self.D_cord)) + self.rho

    Temp_F_Smaps = pathlossF + self.f_Smaps[np.int(state[0]), np.int(state[1]), np.int(state[2]), time_step]
    Temp_F_Dmaps = pathlossF + self.f_Dmaps[np.int(state[0]), np.int(state[1]), np.int(state[2]), time_step]

    f_S = (10**(Temp_F_Smaps/20)) * self.f_Sphase[np.int(state[0]), np.int(state[1]), np.int(state[2]), time_step]
    f_D = (10**(Temp_F_Dmaps/20)) * self.f_Dphase[np.int(state[0]), np.int(state[1]), np.int(state[2]), time_step]

    VI_numerator= self.P_R* self.P_S* (np.abs(f_S)**2) * (np.abs(f_D)**2)
    VI_denominator= (self.P_S*self.sigma_DSQ*(np.abs(f_S)**2)) + (self.P_R*self.sigmaSQ*(np.abs(f_D)**2)) + (self.sigmaSQ*self.sigma_DSQ)
    VI_opt= VI_numerator / VI_denominator 

    return VI_opt


