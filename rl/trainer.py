# rl/trainer.py

import torch
import torch.nn.functional as F

def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               images, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, entropy_coef=0.01):
    """
    Performs a PPO update step on actor and critic networks.

    Args:
        actor (nn.Module): The actor network
        critic (nn.Module): The critic network
        optimizer_actor (torch.optim.Optimizer)
        optimizer_critic (torch.optim.Optimizer)
        images (Tensor): [B, C, H, W]
        states (Tensor): [B, state_dim]
        actions (Tensor): [B, action_dim]
        old_log_probs (Tensor): [B]
        returns (Tensor): [B]
        advantages (Tensor): [B]
    """

    dist = actor(images, states)
    new_log_probs = dist.log_prob(actions).sum(dim=1)  # Continuous actions → sum over action dims
    entropy = dist.entropy().sum(dim=1).mean()         # Encourage exploration

    # PPO ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Critic loss
    value_preds = critic(images, states)
    value_loss = F.mse_loss(value_preds, returns)

    # Actor update
    optimizer_actor.zero_grad()
    (policy_loss - entropy_coef * entropy).backward()
    optimizer_actor.step()

    # Critic update
    optimizer_critic.zero_grad()
    value_loss.backward()
    optimizer_critic.step()

    return policy_loss.item(), value_loss.item(), entropy.item()