import torch
from siren_pytorch import SirenNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np



''' 
The class of the Value Network (inherited from nn.Module) -> we use Sinusoidal
Representation Networks (SIRENs)
'''
class ValueNetwork(nn.Module):
  def __init__(self, beta, input_dims, fc1_dims=200, fc2_dims=200):
    ''' 
    beta: learning rate
    input_dims : dimensionality of the state
    fc1_dims : neurons for the first input layer
    fc2_dims : neurons for the second input layer
    '''
    super(ValueNetwork, self).__init__()

    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims

    self.siren = SirenNet(dim_in = self.input_dims, dim_hidden = self.fc1_dims, dim_out = 1 , num_layers = 3, final_activation = nn.Identity(),w0_initial = 5.)


    self.optimizer = optim.Adam(self.parameters(), lr=beta)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.to(self.device)


  def forward(self, state):
    v = self.siren(state)

    return v
    