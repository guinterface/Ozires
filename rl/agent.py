# rl/agent.py

import torch
from rl.actor import Actor
from rl.critic import Critic  # You'll need to implement this too


class PPOAgent:
    def __init__(self, image_shape, state_dim, action_dim, device='cpu'):
        """
        Initializes the PPO agent with separate actor and critic networks.
        Args:
            image_shape (tuple): (C, H, W) of image input
            state_dim (int): Dimensionality of state vector
            action_dim (int): Number of continuous action outputs
            device (str): 'cpu' or 'cuda'
        """
        self.device = device

        self.actor = Actor(image_shape, state_dim, action_dim).to(device)
        self.critic = Critic(image_shape, state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        print(f"[agent.py] Loaded agent from {path}")
