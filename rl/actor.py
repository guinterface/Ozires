# rl/actor.py

import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, image_shape, state_dim, action_dim):
        """
        Actor network for PPO with image + state input and continuous action output.

        Args:
            image_shape (tuple): Shape of the input image (C, H, W)
            state_dim (int): Dimensionality of scalar input (e.g., goal vector, velocity)
            action_dim (int): Number of continuous actions (e.g., [vx, vz, yaw_rate])
        """
        super().__init__()

        c, h, w = image_shape

        # TODO: Customize CNN architecture if higher image quality or depth input is used
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Determine CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            cnn_output_dim = self.cnn(dummy).shape[1]

        # TODO: Modify MLP depth, width, or activation if needed
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Action output (mean of Gaussian) + log_std (shared)
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # TODO: Make trainable per-dim if needed

    def forward(self, image, state_vec):
        """
        Forward pass through the actor network.

        Args:
            image (Tensor): Tensor of shape [B, C, H, W]
            state_vec (Tensor): Tensor of shape [B, state_dim]

        Returns:
            dist (Normal): PyTorch Normal distribution over actions
        """
        x1 = self.cnn(image)                    # image → CNN → features
        x2 = state_vec                          # state vector input
        x = torch.cat([x1, x2], dim=1)          # concatenate features
        x = self.mlp(x)                         # MLP for fusion
        mean = self.mean_head(x)                # Action mean
        std = self.log_std.exp()                # Fixed std

        # TODO: Consider clamping or bounding std if too unstable
        return Normal(mean, std)
