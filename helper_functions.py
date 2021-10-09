import numpy as np
import scipy
from scipy import spatial, linalg, io 






def helper_function(grid_min, spacing, rMap_RowCells, rMap_ColCells, rMap_zCells, ell, rho, sigmaSQ, sigma_xiSQ, sigma_DSQ, etaSQ, c1, c2,c3, numSlots):

  totalGridcells = rMap_ColCells*rMap_ColCells*rMap_zCells
  grid_bound = np.linspace(grid_min, rMap_RowCells, rMap_RowCells+1)
  grid_X_temp, grid_Y_temp = np.meshgrid(grid_bound,grid_bound)
  grid_X=np.flipud(grid_Y_temp)
  grid_Y=grid_X_temp

  relay_pos_X = np.array(grid_X-(spacing/2))
  relay_pos_X = np.delete(relay_pos_X, -1, axis=0)
  relay_pos_X = np.delete(relay_pos_X, -1, axis=1)       

  relay_pos_Y = np.array(grid_Y-(spacing/2))
  relay_pos_Y = np.delete(relay_pos_Y, -1, axis=0)
  relay_pos_Y = np.delete(relay_pos_Y, 0, axis=1)

  relay_pos_Z = np.array(grid_X-(spacing/2))
  relay_pos_Z = np.delete(relay_pos_X, -1, axis=0)
  relay_pos_Z = np.delete(relay_pos_X, -1, axis=1) 

  All_gridcords = np.zeros([totalGridcells,3]) 
  for grid_cell in range(totalGridcells):
    All_gridcords[grid_cell,:] = np.unravel_index(grid_cell,(rMap_RowCells,rMap_ColCells,rMap_zCells)) + np.array([spacing/2, spacing/2, spacing/2])

  T = np.zeros([numSlots,numSlots])
  for k in range(numSlots): 
    for l in range(numSlots):
      T[k,l] = np.exp(-np.abs(k-l)/c2)

  dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(All_gridcords, 'euclidean'))

  C = np.zeros([totalGridcells, totalGridcells])
  for i in range(totalGridcells):
    for j in range(totalGridcells):
      C[i,j]= etaSQ*np.exp(-dist[i,j]/c1)

  del dist

  S_pos = np.zeros([1, 3]) 
  D_pos = np.zeros([1, 3]) 

  S_pos[0,:] = np.array([24, 10, 0])         
  D_pos[0,:] = np.array([-3, 10, 0])

  S_cord = S_pos + np.array([spacing/2, spacing/2, spacing/2]) 
  D_cord = D_pos + np.array([spacing/2, spacing/2, spacing/2]) 


  C_SD = np.zeros([2*totalGridcells, 2*totalGridcells])
  C_SD = np.vstack((np.hstack((C,C*np.exp(-np.linalg.norm(S_cord-D_cord)/c3))), np.hstack((C*np.exp(-np.linalg.norm(S_cord-D_cord)/c3),C)) ))          
  kappa = np.exp(-1/c2)
  C_SD_chol = scipy.linalg.cholesky( (1-(kappa**2))*C_SD)

  return T, C_SD, C_SD_chol, kappa, grid_X, grid_Y, S_cord, D_cord



def create_data_for_shadowing_multipath_phase(rMap_RowCells, rMap_ColCells, rMap_zCells, C_SD, C_SD_chol, numSlots, sigma_xiSQ, kappa):


  f_Sphase = np.exp(1j*2*np.pi*np.random.uniform(0,1,(rMap_RowCells, rMap_ColCells, rMap_zCells, numSlots+1)))
  f_Dphase = np.exp(1j*2*np.pi*np.random.uniform(0,1,(rMap_RowCells, rMap_ColCells, rMap_zCells, numSlots+1)))

  f_Smaps = np.zeros([rMap_RowCells, rMap_ColCells, rMap_zCells, numSlots+1], dtype=complex)
  f_Dmaps = np.zeros([rMap_RowCells, rMap_ColCells, rMap_zCells, numSlots+1], dtype=complex)

  beta_t = (np.random.randn(C_SD[0,:].shape[0]) @ scipy.linalg.cholesky(C_SD) ).T 

  for i in range(numSlots+1):
    W_t = (np.random.randn(C_SD[0,:].shape[0]) @ C_SD_chol ).T
    beta_t = kappa * beta_t + W_t
    temp = np.array(beta_t + np.sqrt(sigma_xiSQ)*np.random.randn(C_SD[0,:].shape[0]).T)
    temp_S = temp[0:int(temp.shape[0]/2)]
    temp_D = temp[int(temp.shape[0]/2):] 

    Temp_F_Smaps = np.reshape(temp_S, (rMap_RowCells, rMap_ColCells, rMap_zCells) )
    Temp_F_Dmaps = np.reshape(temp_D, (rMap_RowCells, rMap_ColCells, rMap_zCells) )

    f_Smaps[:,:,:,i] = Temp_F_Smaps
    f_Dmaps[:,:,:,i] = Temp_F_Dmaps
  
  return f_Smaps, f_Dmaps, f_Sphase, f_Dphase