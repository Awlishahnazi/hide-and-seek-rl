import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """
        Computes Generalized Advantage Estimation (GAE).
        """
        rewards = self.rewards
        values = self.values + [next_value]
        dones = self.dones

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def get_batches(self, batch_size):
        """
        Yields mini-batches for PPO updates.
        (To be implemented in the trainer step)
        """
        pass