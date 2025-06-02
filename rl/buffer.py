import numpy as np

class RolloutBuffer:
    def __init__(self, size, image_shape, state_dim, action_dim):
        self.images = np.zeros((size, *image_shape), dtype=np.uint8)
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, image, state, action, reward, done, value, log_prob):
        self.images[self.ptr] = (image * 255).astype(np.uint8)  # Ensure it's uint8 (for storage)
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_advantages(self, last_value, gamma=0.99, lam=0.95):
        gae = 0
        for t in reversed(range(self.ptr)):
            delta = self.rewards[t] + gamma * (1 - self.dones[t]) * last_value - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]
            last_value = self.values[t]

    def reset(self):
        self.ptr = 0
