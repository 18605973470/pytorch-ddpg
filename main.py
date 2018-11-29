#!/usr/bin/python3

import math
import random

import gym
import os
import matplotlib
matplotlib.use('Agg')

import numpy as np

import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent
from arguments import ParseArguments
from environment import NormalizedActions


def main():
    args = ParseArguments()
    time_string = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join("experiment", args.experiment_name, time_string)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    cuda_available = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    torch.manual_seed(args.seed)

    print("Init environment ...")
    env = gym.make("BipedalWalker-v2")
    # env = NormalizedActions(gym.make("MountainCarContinuous-v0"))
    print("Init environment successfully ...")

    print("Init agent ...")
    agent = DDPGAgent(device, args, env)
    print("Init agent successfully ...")

    if args.load:
        print("Try to load model ...")
        try:
            agent.load_model(experiment_dir, 10000)
            print("Load model successfully ...")
        except:
            print("Fail to load model ...")

    # Statistics
    total_step_list = []
    total_step_reward_list = []
    episode_reward_list = []
    episode_step_list = []
    episode_policy_loss_list = []
    episode_value_loss_list = []

    total_step = 0
    episode = 0
    start_time = time.time()
    while True:
        episode_reward = 0
        episode_step = 0
        state = env.reset()

        while True:
            env.render()
            action = agent.action_with_exploration(state, total_step)
            next_state, reward, done, info = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.optimize()

            # Limitation at length of an episode
            episode_step += 1
            if episode_step >= args.max_episode_step:
                done = True

            # Limitation at length of all episodes
            total_step += 1
            if total_step >= args.max_total_step:
                done = True

            if total_step % args.save_interval_step == 0:
                agent.save_model(experiment_dir, total_step)

            if done:
                end_time = time.time()
                delta_time = end_time - start_time
                fps = total_step / delta_time
                episode += 1
                print("Training time {}, episode {}, total step {}, episode step {}, episode reward {}, fps {}".format(
                    time.strftime('%H:%M:%S', time.gmtime(delta_time)), episode, total_step,
                    episode_step, episode_reward, fps)
                )
                break

        episode_reward_list.append(episode_reward)
        total_step_list.append(total_step)
        total_step_reward_list.append([total_step, episode_reward])
        episode_step_list.append(episode_step)

        # Logging
        fig1, ax1 = plt.subplots(figsize=(11, 8))
        ax1.plot(range(episode), episode_reward_list)
        ax1.set_title("Reward vs Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        plt.savefig('%s/reward_vs_episode.png' % (experiment_dir))
        plt.clf()
        np.save("%s/episode_reward.npy" % experiment_dir, episode_reward_list)

        fig2, ax2 = plt.subplots(figsize=(11, 8))
        ax2.plot(total_step_list, episode_reward_list)
        ax2.set_title("Reward vs Step")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Reward")
        plt.savefig('%s/reward_vs_step.png' % (experiment_dir))
        plt.clf()
        np.save("%s/step_reward.npy" % experiment_dir, total_step_reward_list)

        fig3, ax3 = plt.subplots(figsize=(11, 8))
        ax3.plot(range(episode), episode_step_list)
        ax3.set_title("Episode length vs Episode")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Episode length")
        plt.savefig('%s/episode_length_vs_episode.png' % (experiment_dir))
        np.save("%s/episode_length.npy" % experiment_dir, episode_step_list)

        if total_step >= args.max_total_step:
            break

    env.close()

    print("Training ends ........ ")

if __name__ == "__main__":
    main()
