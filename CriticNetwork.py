import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

''' 
The class of the Critic Network (inherited from nn.Module)
'''
class CriticNetwork(nn.Module):
  def __init__(self, beta, input_dims, n_actions, fc1_dims=200, fc2_dims=200 ):
    ''' 
    beta: learning rate
    input_dims : the dimensionality of the features representing the state (coordinates in our case)
    n_actions :  the dimensionality of the action space
    fc1_dims: for the first dense layer
    fc2_dims : for the second dense layer
    '''
    super(CriticNetwork, self).__init__()
    
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions

    self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.q = nn.Linear(self.fc2_dims, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=beta)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.to(self.device)

  def forward(self, state, action):
    action_value = self.fc1(torch.cat([state, action], dim=1))
    action_value = F.relu(action_value)
    action_value = self.fc2(action_value)
    action_value = F.relu(action_value)
    action_value = self.q(action_value)



    return action_value



