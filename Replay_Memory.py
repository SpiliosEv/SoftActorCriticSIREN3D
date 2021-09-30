import numpy as np

''' 
The Replay Memory class

'''
class ReplayMemory(object):
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size # the size of the Buffer
    self.mem_cntr = 0 # a counter pointing at the next position to be filled in the Buffer
    self.state_memory = np.zeros((self.mem_size, input_shape)) # current state memory
    self.new_state_memory = np.zeros((self.mem_size, input_shape)) # next state memory
    self.action_memory = np.zeros((self.mem_size, n_actions)) # memory for the actions
    self.reward_memory = np.zeros(self.mem_size) # memory for the rewards

    ''' 
    method for storing a tuple from the agent's trajectory to the Replay Memory
    '''
  def store_transition(self, state, action, reward, state_):
    index = self.mem_cntr % self.mem_size # find the index to place the next tuple
    self.state_memory[index] = state # insert the current state
    self.new_state_memory[index] = state_ # insert the next state
    self.action_memory[index] = action # insert the action
    self.reward_memory[index] = reward # insert the reward
    
    self.mem_cntr += 1 # increment the counter
    ''' 
    method to sample a mini batch of tuples from the Replay Memory for updating the weights
    ''' 
  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size) # fing the current size

    batch = np.random.choice(max_mem, batch_size) # find random indices
    ''' 
    sample a mini batch from all the sub memories
    '''
    states = self.state_memory[batch]
    states_ = self.new_state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]

    return states, actions, rewards, states_


