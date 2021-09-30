import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

''' 
The class of the actor Network (inherited from nn.Module)
'''
class ActorNetwork(nn.Module):
  def __init__(self, alpha, input_dims, max_action, fc1_dims=200, fc2_dims=200, n_actions=2):
    ''' 
    alpha : learning rate
    input_dims : dimensionality of the state 
    max_action : the maximum range for every component of the action
    fc1_dims : neurons for the first dense layer
    fc2_dims : neurons for the second dense layer
    n_actions : dimensionality of the action space
    '''
    super(ActorNetwork, self).__init__()
    self.input_dims = input_dims 
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions
    self.max_action = max_action
    self.n_actions = n_actions
    self.reparam_noise = 1e-6

    self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.mu = nn.Linear(self.fc2_dims, self.n_actions)
    self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)

  def forward(self, state):
    prob = self.fc1(state)
    prob = F.relu(prob)
    prob = self.fc2(prob)
    prob = F.relu(prob)

    mu = self.mu(prob)
    sigma = self.sigma(prob)

    sigma = torch.clamp(sigma, min=self.reparam_noise, max=1.)

    return mu, sigma


  def sample_normal(self, state, reparameterize=True):
    mu, sigma = self.forward(state)
    probabilities = Normal(mu, sigma)

    if reparameterize:
      actions = probabilities.rsample()
    else:
      actions = probabilities.sample()
    
    action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
    log_probs = probabilities.log_prob(actions)
    log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
    log_probs = log_probs.sum(1, keepdim=True)

    return action, log_probs



