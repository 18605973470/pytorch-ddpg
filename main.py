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
from ounoise import OUNoise
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent
from arguments import parse_arguments
from environment import create_env 
from train import train
from evaluate import evaluate


def main():
    args = parse_arguments()
    # time_string = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join("experiment", args.experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    cuda_available = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    torch.manual_seed(args.seed)

    print("Init environment ...")
    from torcs_wrapper import TorcsWrapper
    env = TorcsWrapper(port=12345, noisy=True, throttle=0.80, control_dim = 1, k = 2.0)
    print("\t Observation space {}".format(env.observation_space.shape))
    print("\t Action space {}".format(env.action_space.shape))

    # env = create_env("BipedalWalker-v2")
    # env = NormalizedActions(gym.make("Pendulum-v0"))
    print("Init environment successfully ...")

    print("Init agent & noise ...")
    ou_noise = OUNoise(env.action_space, min_epsilon=args.min_epsilon, max_epsilon=args.max_epsilon,
                       decay_period=int(args.max_total_step * 0.6))
    agent = DDPGAgent(device, args, env, ou_noise)

    print("Init agent & noise successfully ...")
    if args.load:
        print("Try to load model ...")
        try:
            agent.load_model(os.path.join("experiment", args.load_dir), "last")
            print("Load model successfully ...")
        except:
            print("Fail to load model ...")

    if args.mode == "train":
        train(agent, env, args, experiment_dir)
    else:
        for track_id in [5]:
            evaluate(agent, env, args, experiment_dir, track_id)

    env.close()

if __name__ == "__main__":
    main()
