import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(self.bn0(state)))
        x = F.relu(self.fc2(self.bn1(x)))
        return F.tanh(self.fc3(x))
    
class PairedActor(nn.Module):
    """Paired Actor (Policy) Model"""
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(PairedActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor1 = Actor(state_size, action_size, seed, fc1_units, fc2_units)
        self.actor2 = Actor(state_size, action_size, seed, fc1_units, fc2_units)
        
    def forward(self, state_1, state_2):
        """ Pass in states independently """
        act1 = torch.unsqueeze(self.actor1(state_1), 0)
        act2 = torch.unsqueeze(self.actor2(state_2), 0)
        
        # Output shape pre-transpose is [actor_num, batch_size, action_states]
        # Output shape post-transpose is [batch_size, actor_num, action_states]
        # Need this to match the shape of the actions in the memory buffer
        return torch.cat( (act1, act2), 0).transpose(0,1)
        
        

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
                
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        xs = F.elu(self.fcs1(state))
        xs = self.bn(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    
class PairedCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        super(PairedCritic, self).__init__()
        self.critic = Critic(state_size, action_size, seed, fcs1_units, fc2_units)
        self.fc_out = nn.Linear(2, 1)
        self.reset_parameters()
        
    def reset_parameters(self):   
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        # x1/x2.shape = (batch_size, 1)
        x1 = self.critic(state[:,0], action[:,0]) 
        x2 = self.critic(state[:,1], action[:,1])
        # The two critics should be returned individually
        return torch.cat( (x1, x2), 1 )
        