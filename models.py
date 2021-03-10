import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, state_dimension, action_dimension, action_max):
        super(Actor, self).__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.action_max = action_max

        # 3 shirheg dawharga uusgene

        self.fc1 = nn.Linear(state_dimension, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dimension)
        
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        output = self.fc1(state)
        output = F.relu(output)

        output = self.fc2(output)
        output = F.relu(output)

        output = self.fc3(output)
        output = torch.tanh(output)

        action = output * self.action_max

        # action butsaana

        return action

class Critic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        
        # 4 layer uusgene

        self.fcs1 = nn.Linear(state_dimension, 400)
        self.fcs2 = nn.Linear(400, 300)
        self.fca1 = nn.Linear(action_dimension, 300)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, 1)

        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        s1 = self.fcs1(state)
        s1 = F.relu(s1)
        s2 = self.fcs2(s1)
        s2 = F.relu(s2)
        a1 = self.fca1(action)
        a1 = F.relu(a1)

        output = torch.cat((s2, a1), dim=1)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)

        # Q value-g butsaana

        return output



