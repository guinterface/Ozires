import numpy as np

class RolloutBuffer:
    def __init__(self, size, state_dim, action_dim):
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, s, a, r, d, v, logp):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d
        self.values[self.ptr] = v
        self.log_probs[self.ptr] = logp
        self.ptr += 1

    def compute_advantages(self, last_value, gamma=0.99, lam=0.95):
        # TODO: customize this (gamma, lambda for GAE)
        gae = 0
        for t in reversed(range(self.ptr)):
            delta = self.rewards[t] + gamma * (1 - self.dones[t]) * last_value - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]
            last_value = self.values[t]

    def reset(self):
        self.ptr = 0
