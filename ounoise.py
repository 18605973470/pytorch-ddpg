#!/usr/bin/python3

import numpy as np


class OUNoise:
    def __init__(self, action_space, max_epsilon, min_epsilon, mu=0.0, theta=0.15, sigma=0.2, decay_period=30000):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high

        self.decay_period = decay_period
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.delta_epsilon = (self.max_epsilon - self.min_epsilon) / self.decay_period

    def add_noise(self, action, step=0):
        noise = self.theta * (self.mu - action) + self.sigma * np.random.randn(self.action_dim)
        epsilon = max(self.max_epsilon - step * self.delta_epsilon, 0)
        noise = noise * epsilon
        action = np.clip(action + noise, self.low, self.high)
        return action

# class OUNoise:
#     def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.01, decay_period=30000):
#         self.mu = mu
#         self.theta = theta
#         self.sigma = max_sigma
#         self.max_sigma = max_sigma
#         self.min_sigma = min_sigma
#         self.decay_period = decay_period
#         self.action_dim = action_space.shape[0]
#         self.low = action_space.low
#         self.high = action_space.high
#         self.state = np.zeros(self.action_dim)
#
#         self.reset()
#
#     def reset(self):
#         self.state = np.ones(self.action_dim) * self.mu
#
#     def evolve_state(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
#         self.state = x + dx
#         return self.state
#
#     def get_action(self, action, t=0):
#         ou_state = self.evolve_state()
#         self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
#         return np.clip(action + ou_state, self.low, self.high)
