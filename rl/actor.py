import torch
import torch.nn as nn
from torch.distributions import Normal
from torchvision.models import resnet50

class DroneActor(nn.Module):
    def __init__(self, image_shape, state_dim, action_dim, use_pretrained=True, freeze_backbone=False):
        """
        Flexible Actor for drone self-driving using PPO.
        
        Args:
            image_shape (tuple): (C, H, W) input image shape, usually RGB (3, 224, 224)
            state_dim (int): Dimensionality of additional state info (velocity, IMU, GPS, etc.)
            action_dim (int): Number of continuous actions (e.g., [vx, vy, vz, yaw_rate])
            use_pretrained (bool): Whether to load pretrained ResNet weights
            freeze_backbone (bool): Whether to freeze ResNet layers
        """
        super().__init__()
        c, h, w = image_shape
        assert c == 3, "Currently expects RGB images; extend here for RGB-D."

        # CNN Backbone (ResNet-50)
        backbone = resnet50(pretrained=use_pretrained)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # [B, 2048, 1, 1]
        self.cnn_output_dim = 2048

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Fusion MLP for image features + state
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.cnn_output_dim + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Action head
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learned, shared log_std

    def forward(self, image, state_vec):
        """
        Forward pass for action distribution.

        Args:
            image (Tensor): [B, 3, H, W] RGB input
            state_vec (Tensor): [B, state_dim] auxiliary drone state input

        Returns:
            dist (Normal): Gaussian distribution over actions
        """
        img_feat = self.cnn(image)                  # [B, 2048, 1, 1]
        img_feat = img_feat.view(img_feat.size(0), -1)  # Flatten to [B, 2048]
        fused = torch.cat([img_feat, state_vec], dim=1)
        fused_feat = self.fusion_mlp(fused)
        mean = self.mean_head(fused_feat)
        std = self.log_std.exp()
        return Normal(mean, std)
