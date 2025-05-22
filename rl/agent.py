def train(agent, env, buffer, trainer, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            dist = agent.actor(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = agent.critic(state_tensor)

            next_state, reward, done, _ = env.step(action.numpy())
            buffer.add(state, action.numpy(), reward, done, value.item(), log_prob.item())

            state = next_state
            ep_reward += reward

        last_value = agent.critic(torch.tensor(state, dtype=torch.float32)).item()
        buffer.compute_advantages(last_value)

        # TODO: customize this (number of epochs, batch size, shuffle strategy)
        for _ in range(10):
            batch = buffer_to_tensors(buffer)
            trainer.ppo_update(*batch)

        buffer.reset()
