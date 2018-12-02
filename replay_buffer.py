#!/usr/bin/python3

import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, feature_vector, next_feature_vector):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, feature_vector, next_feature_vector)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, feature_vector, next_feature_vector = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones, feature_vector, next_feature_vector

    def __len__(self):
        return len(self.buffer)
