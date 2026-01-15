"""
model.py
Classic DDPG networks (Actor + Critic) for continuous control.
Works with Reacher: state_size=33, action_size=4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer: nn.Linear):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Deterministic policy network: state -> action in [-1, 1]."""

    def __init__(self, state_size: int, action_size: int, seed: int, fc1: int = 400, fc2: int = 300):
        super().__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Q-network: (state, action) -> Q-value."""

    def __init__(self, state_size: int, action_size: int, seed: int, fcs1: int = 400, fc2: int = 300):
        super().__init__()
        torch.manual_seed(seed)

        # state pathway
        self.fcs1 = nn.Linear(state_size, fcs1)
        # combine with action
        self.fc2 = nn.Linear(fcs1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
