import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous=False):
        super().__init__()
        self.is_continuous = is_continuous

        # TODO: customize this (network depth, width, activation functions)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # TODO: customize this (head depending on action space type)
        if is_continuous:
            self.mean_head = nn.Linear(64, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log_std
        else:
            self.logits = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.net(state)
        if self.is_continuous:
            return Normal(self.mean_head(x), self.log_std.exp())  # Continuous action distribution
        else:
            return Categorical(logits=self.logits(x))  # Discrete action distribution
