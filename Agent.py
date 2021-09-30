import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from Replay_Memory import ReplayMemory
from CriticNetwork import CriticNetwork
from ValueNetwork import ValueNetwork
from Actor import ActorNetwork

''' 
class of the Agent 
'''

class Agent(object):
  def __init__(self, alpha=0.0002, beta=0.0002, input_dims=3, gamma=0.99, n_actions=3, max_size=1000000, tau=0.05, batch_size=256, reward_scale=1 ):
    ''' 
    alpha : learning rate for the actor network
    beta : learning rate for the critic and value networks
    input_dims : dimensions of the state
    gamma : discount factor
    n_actions : dimensionality of the actions
    max_size : size of the Replay Memory
    tau : parameter for the update of the target value network
    batch_size : size of the batch
    reward_scale : temperature coefficient (how importance to the entropy maximization)
    '''
    self.gamma = gamma
    self.tau = tau
    self.memory = ReplayMemory(max_size, input_dims, n_actions)
    self.batch_size = batch_size

    self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, max_action=1/np.sqrt(2))
    
    self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions)
    self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions)

    self.value = ValueNetwork(beta, input_dims)
    self.target_value = ValueNetwork(beta,input_dims)

    self.scale = reward_scale
    self.update_network_parameters(tau=1)

  '''
  method for choosing the next action
  '''
  def choose_action(self, observation):
    state = torch.tensor([observation]).to(self.actor.device)

    actions, _ = self.actor.sample_normal(state, reparameterize=False)

    return actions.cpu().detach().numpy()[0]

  ''' 
  method for storing a tuple in the Replay Memory
  '''
  def remember(self, state, action, reward, new_state):
    self.memory.store_transition(state, action, reward, new_state)
  ''' 
  method for updating the value network parameters
  '''
  def update_network_parameters(self, tau=None):
    if tau is None:
      tau = self.tau 

    target_value_params = self.target_value.named_parameters()
    value_params = self.value.named_parameters()

    target_value_state_dict = dict(target_value_params)
    value_state_dict = dict(value_params)

    for name in value_state_dict:
      value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

    self.target_value.load_state_dict(value_state_dict)

  '''
  method for sampling a mini batch of experiences and updating the weights of the neural networks
  '''
  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return 

    ''' 
    sample a mini batch of experiences
    '''
    state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)

    ''' 
    transform them all to torch tensors
    '''
    reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
    state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
    state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
    action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
    
    ''' 
    train the value network
    '''

    value = self.value(state).view(-1)
    value_ = self.target_value(state_).view(-1)

    actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
    log_probs = log_probs.view(-1)
    q1_new_policy = self.critic_1.forward(state, actions)
    q2_new_policy = self.critic_2.forward(state, actions)

    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)

    self.value.optimizer.zero_grad()
    value_target = critic_value - log_probs
    value_loss = 0.5 * F.mse_loss(value, value_target)
    value_loss.backward(retain_graph=True)
    self.value.optimizer.step()
    

    ''' 
    train the actor network
    '''
    actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
    log_probs = log_probs.view(-1)

    q1_new_policy = self.critic_1.forward(state, actions)
    q2_new_policy = self.critic_2.forward(state, actions)

    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)

    actor_loss = log_probs - critic_value
    actor_loss = torch.mean(actor_loss)
    self.actor.optimizer.zero_grad()
    actor_loss.backward(retain_graph = True)
    self.actor.optimizer.step()
    

    ''' 
    train the critics
    '''
    self.critic_1.optimizer.zero_grad()
    self.critic_2.optimizer.zero_grad()
    q_hat = self.scale*reward + self.gamma*value_
    q1_old_policy = self.critic_1.forward(state, action).view(-1)
    q2_old_policy = self.critic_2.forward(state, action).view(-1)
    critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
    critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

    critic_loss = critic_1_loss + critic_2_loss
    critic_loss.backward()
    self.critic_1.optimizer.step()
    self.critic_2.optimizer.step()


    ''' 
    updating the weights of the target value network
    '''
    self.update_network_parameters()






