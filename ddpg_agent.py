#!/usr/bin/python3

import numpy as np
import torch
import os

from torch import optim
from torch import nn
from models import ValueNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, device, args, env, ou_noise):

        self.noise = ou_noise

        # Hyper parameters
        self.value_lr = args.value_lr
        self.policy_lr = args.policy_lr
        self.gamma = args.gamma
        self.soft_tau = args.soft_tau
        self.batch_size = args.batch_size
        self.replay_buffer_size = args.replay_buffer_size

        # Other parameters
        self.heatup = args.num_heatup
        self.device = device
        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dim = 512

        self.value_net = ValueNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy_net(state)
        return action.detach().cpu().numpy()[0]

    def noise_action(self, state, step):
        return self.noise.add_noise(self.action(state), step)

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self, optimize_step=1):
        if len(self.replay_buffer) >= self.heatup:
            # critic_loss = 0
            # actor_loss = 0
            for i in range(optimize_step):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                states = torch.FloatTensor(states).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)

                policy_loss = self.value_net(states, self.policy_net(states))
                policy_loss = -policy_loss.mean()

                next_actions = self.target_policy_net(next_states)
                target_values = self.target_value_net(next_states, next_actions).detach()
                expected_values = rewards + (1.0 - dones) * self.gamma * target_values
                # expected_value = torch.clamp(expected_value, min_value, max_value)

                value = self.value_net(states, actions)
                value_loss = self.value_criterion(value, expected_values)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                    )

                for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                    )
                # actor_loss += policy_loss.data
                # critic_loss += value_loss.data

    def save_model(self, path, notes):
        policy_state_to_save = self.target_policy_net.state_dict()
        value_state_to_save = self.target_value_net.state_dict()
        torch.save(policy_state_to_save, os.path.join(path, 'policy-{}.pkl'.format(notes)))
        torch.save(value_state_to_save, os.path.join(path, 'value-{}.pkl'.format(notes)))

    def load_model(self, path, notes):
        print(os.path.join(path, 'policy-{}.pkl'.format(notes)))
        policy_state_to_load = torch.load(os.path.join(path, 'policy-{}.pkl'.format(notes)))
        self.target_policy_net.load_state_dict(policy_state_to_load)
        self.policy_net.load_state_dict(policy_state_to_load)

        value_state_to_load = torch.load(os.path.join(path, 'value-{}.pkl'.format(notes)))
        self.target_value_net.load_state_dict(value_state_to_load)
        self.value_net.load_state_dict(value_state_to_load)