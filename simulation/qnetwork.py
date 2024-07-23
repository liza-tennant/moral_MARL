import torch
import torch.nn as nn
import torch.nn.functional as F

from config import network_size

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """ 
        super(QNetwork, self).__init__()
        #self.seed = 
        torch.manual_seed(seed)
        self.input = nn.Linear(state_size, network_size)
        #self.hidden = nn.Linear(network_size, network_size)
        self.output = nn.Linear(network_size, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input(state)
        x = F.relu(x)
        #x = self.hidden(x)
        #x = F.relu(x)
        return self.output(x)