import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from rl.agent import PPOAgent
from rl.buffer import RolloutBuffer
from rl.trainer import ppo_update
from envs.drone_env import DroneSimEnv
from rl.utils import buffer_to_tensors


def train(agent, env, buffer, num_episodes, writer, device='cpu'):
    epochs = 10             # Number of training epochs per episode
    batch_size = 64         # PPO mini-batch size

    os.makedirs("gifs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    success_count = 0
    episode_rewards = []
    step_counts = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # === Preprocess image ===
            image_np = obs["image"] / 255.0  # normalize
            print(f"[DEBUG] Raw image shape from env: {image_np.shape}")  # (H, W, C)

            # Resize and convert to tensor
            image_resized = TF.resize(Image.fromarray((image_np * 255).astype(np.uint8)), size=(64, 64))
            image_tensor = T.ToTensor()(image_resized).unsqueeze(0).to(device)

            state_np = obs["state"]
            state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)

            # === Save one debug image ===
            if not hasattr(train, "_image_saved"):
                print(f"[DEBUG] image_tensor shape: {image_tensor.shape}")  # [1, C, H, W]
                sample_image = image_tensor[0].cpu()  # [C, H, W]
                to_pil = T.ToPILImage()
                pil_img = to_pil(sample_image)
                os.makedirs("debug_images", exist_ok=True)
                pil_img.save("debug_images/sample_input_image.png")
                print("[DEBUG] Saved sample image to debug_images/sample_input_image.png")
                train._image_saved = True
            # === End debug ===

            # Select action
            dist = agent.actor(image_tensor, state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            value = agent.critic(image_tensor, state_tensor)

            # Interact with env
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)

            buffer.add(
                image_tensor.squeeze(0).cpu().numpy(),  # (C, H, W)
                state_tensor.squeeze(0).cpu().numpy(),  # (state_dim,)
                action_np,
                reward,
                done,
                value.item(),
                log_prob.item()
            )

            obs = next_obs
            ep_reward += reward

        # Track metrics
        episode_rewards.append(ep_reward)
        step_counts.append(env.current_step)
        
        if env.current_step < 200:
            success_count += 1

        # Bootstrap value for final state
        last_image_np = obs["image"] / 255.0
        last_image_resized = TF.resize(Image.fromarray((last_image_np * 255).astype(np.uint8)), size=(64, 64))
        last_image = T.ToTensor()(last_image_resized).unsqueeze(0).to(device)
        last_state = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0).to(device)
        last_value = agent.critic(last_image, last_state).item()
        buffer.compute_advantages(last_value)

        # Convert buffer to PyTorch tensors
        images, states, actions, log_probs, returns, advantages = buffer_to_tensors(buffer, device)
        num_samples = images.shape[0]
        indices = np.arange(num_samples)

        # PPO training loop
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch = (
                    images[batch_idx],
                    states[batch_idx],
                    actions[batch_idx],
                    log_probs[batch_idx],
                    returns[batch_idx],
                    advantages[batch_idx]
                )
                ploss, vloss, entropy = ppo_update(
                    agent.actor,
                    agent.critic,
                    agent.actor_optimizer,
                    agent.critic_optimizer,
                    *batch
                )

                writer.add_scalar("Loss/Policy", ploss, episode)
                writer.add_scalar("Loss/Value", vloss, episode)
                writer.add_scalar("Loss/Entropy", entropy, episode)

        buffer.reset()

        # Episode summary
        print(f"[Episode {episode}] Reward: {ep_reward:.2f}, Steps: {env.current_step}")

        writer.add_scalar("Reward/Episode", ep_reward, episode)
        writer.add_scalar("Steps/Episode", env.current_step, episode)
        writer.add_scalar("SuccessRate", success_count / (episode + 1), episode)

        # Save model periodically
        if episode % 50 == 0:
            save_path = f"checkpoints/agent_ep{episode}.pth"
            agent.save(save_path)
            print(f"[Checkpoint] Saved model to {save_path}")

        # Save GIF every 10 episodes
        if episode % 10 == 0 and env.render_eval:
            gif_path = f"gifs/episode_{episode}.gif"
            env.save_episode_gif(gif_path)
            print(f"[GIF] Saved rollout to {gif_path}")


# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # === Dynamically infer image shape and state_dim ===
    env = DroneSimEnv(render_eval=True)
    sample_obs = env.reset()
    image_np = sample_obs["image"] / 255.0
    image_resized = TF.resize(Image.fromarray((image_np * 255).astype(np.uint8)), size=(64, 64))
    image_tensor = T.ToTensor()(image_resized)
    image_shape = tuple(image_tensor.shape)  # (C, H, W)
    state_dim = len(sample_obs["state"])
    action_dim = 3
    # === ===

    agent = PPOAgent(image_shape, state_dim, action_dim, device=args.device)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)

    buffer = RolloutBuffer(size=2048, image_shape=image_shape, state_dim=state_dim, action_dim=action_dim)
    writer = SummaryWriter(log_dir="runs/ppo_run")

    train(agent, env, buffer, num_episodes=args.episodes, writer=writer, device=args.device)
