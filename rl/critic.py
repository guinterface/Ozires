import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        # TODO: customize this (architecture, activation functions)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)  # Output a scalar value per state
