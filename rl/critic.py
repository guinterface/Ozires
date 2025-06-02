# rl/critic.py

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, image_shape, state_dim):
        """
        Critic network: takes image and scalar state, outputs value estimate V(s)

        Args:
            image_shape (tuple): (C, H, W)
            state_dim (int): Dimension of scalar state vector
        """
        super().__init__()

        c, h, w = image_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Dynamically compute output size of CNN
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            cnn_out_dim = self.cnn(dummy).shape[1]
            print(f"[Critic] CNN output dim: {cnn_out_dim} (from input shape {image_shape})")

        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, state_vec):
        """
        Args:
            image: Tensor [B, C, H, W]
            state_vec: Tensor [B, state_dim]

        Returns:
            value: Tensor [B] (scalar value per observation)
        """
        x1 = self.cnn(image)
        x2 = state_vec
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x).squeeze(-1)
