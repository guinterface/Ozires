import torch
import torch.nn.functional as F

def ppo_update(actor, critic, optimizer_actor, optimizer_critic, batch, clip_eps=0.2, entropy_coef=0.01):
    states, actions, old_log_probs, returns, advantages = batch

    dist = actor(states)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # PPO clipped objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # TODO: customize this (critic loss type: MSE, Huber, or clipped value)
    value_pred = critic(states)
    value_loss = F.mse_loss(value_pred, returns)

    # Actor update
    optimizer_actor.zero_grad()
    (policy_loss - entropy_coef * entropy).backward()  # TODO: customize this (entropy regularization)
    optimizer_actor.step()

    # Critic update
    optimizer_critic.zero_grad()
    value_loss.backward()
    optimizer_critic.step()
