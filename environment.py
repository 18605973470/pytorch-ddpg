#!/usr/bin/python3

import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action

# env = NormalizedActions(gym.make("Pendulum-v0"))
# print(env.action_space)
# print(env.observation_space)