import random
import torch

import numpy as np
from collections import deque, namedtuple
from itertools import chain


from config import dilemma_state_uses_identity, episodes


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, action_size, buffer_size, seed): #batch_size, 
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum capacity of memory buffer
            batch_size (int): size of each training batch (no of experiences to sample when training)
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = None #batch_size will be set flexibly from the number of experiences stored in memory from the last episode 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)
        #TO DO figure out of 'random' is a good package to use here for sampling 
        self.device = device
    
    def add(self, state, action, reward, next_state):
        """Add a new experience to memory - add at the end."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)
    
    def sample(self):
        """In order, sample a batch of experiences from memory."""
        if episodes: 
            experiences = []
            self.batch_size = len(self.memory)
            for k in range(self.batch_size): 
                item = self.memory.popleft()
                experiences.append(item)
        else: #if sequential (not episodes), sample randomly = NOTE TO DO check that this uses the RNG streams
            experiences = random.sample(self.memory, k=self.batch_size)

        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        
        if dilemma_state_uses_identity:
            states_list = [e.state for e in experiences if e is not None]
            states_chains_list = [list(chain(*(i if isinstance(i, tuple) else (i,) for i in l))) for l in states_list]
            #TO DO understand if the above line is redundant 

            states = torch.from_numpy(np.vstack(states_chains_list)).squeeze(0).long().to(self.device)
            #NOTE THIS WIlL ONLY WORK FOR BATCH SIZE 1 !!! 
        else:  
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).squeeze(0).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
  
        return states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def refresh(self, buffer_size):
        self.memory = deque(maxlen=buffer_size) 

class SelectionReplayBuffer: 
    """Fixed-size buffer to store selection + dilemma game experience tuples."""
     
    def __init__(self, device, selection_size, buffer_size, seed): #action_size
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.selection_size = selection_size #NOTE check why we need this parameter at all 
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size_selection = 1 #only one selection experience can be available per episode 
#        self.experience = namedtuple("Experience", field_names=["selection_state", "selection", "selection_reward", "selection_next_state", "state", "action", "reward", "next_state"])
        self.experience = namedtuple("Experience", field_names=["selection_state", "selection", "selection_reward", "selection_next_state"])
        #NOTE selection_state and selection are using a custom, player-specific (NOT global) index
        self.seed = random.seed(seed)
        #TO DO figure out of 'random' is a good package to use here for sampling 
        self.device = device
    
    def add(self, selection_state, selection, selection_reward, selection_next_state):
        """Add a new experience to memory.
        NOTE selection_state is stored as the tuple of tuples - i.e. the state itself"""
        e = self.experience(selection_state, selection, selection_reward, selection_next_state)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if episodes: 
            experiences = []
            for k in range(self.batch_size_selection): 
                item = self.memory.popleft()
                experiences.append(item)
        else: #if sequential (not episodes), sample randomly 
            experiences = random.sample(self.memory, k=self.batch_size_selection)

        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        
        #selection_states = torch.from_numpy(np.vstack([e.selection_state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        #selection_states = torch.from_numpy(np.array([e.selection_state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        selection_states = torch.tensor([e.selection_state for e in experiences if e is not None], dtype=torch.int).squeeze(0).long().float().to(self.device)
        selections = torch.from_numpy(np.vstack([e.selection for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        selection_rewards = torch.from_numpy(np.vstack([e.selection_reward for e in experiences if e is not None])).squeeze(0).float().to(self.device)
        #selection_next_states = torch.from_numpy(np.vstack([e.selection_next_state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        #selection_next_states = torch.from_numpy(np.array([e.selection_next_state for e in experiences if e is not None])).squeeze(0).long().to(self.device)
        selection_next_states = torch.tensor([e.selection_next_state for e in experiences if e is not None], dtype=torch.int).squeeze(0).long().float().to(self.device)


#NOTE buffer.sample() returns unprocessed state 
  
        return selection_states, selections, selection_rewards, selection_next_states#, states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def refresh(self, buffer_size):
        self.memory = deque(maxlen=buffer_size) 