#!/usr/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate(agent, env, args, experiment_dir, track_id):
    average_episode_reward = 0
    episode_reward_list = []
    dir = os.path.join(experiment_dir, str(track_id))
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    action_list = []
    total_step = 0
    for i in range(args.eval_episode):
        state = env.reset()
        episode_reward = 0
        for j in range(args.max_episode_step):
            # env.render()

            action = agent.action(state)
            action_list.append(action)

            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward

            total_step = total_step + 1
            if done:
                break

        episode_reward_list.append(episode_reward)

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.plot(range(i+1), episode_reward_list)
        ax.set_title("Evaluation reward vs Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Evaluation reward")
        plt.savefig('%s/evaluation_reward_vs_step.png' % (dir))
        plt.clf()

        print("Evaluation {}, track id {}, reward {}".format(i, track_id, episode_reward))
        average_episode_reward += episode_reward

    average_episode_reward /= args.eval_episode
    print("Evaluation {} times, track id {} : average reward {}".format(args.eval_episode, track_id, average_episode_reward))

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(total_step), action_list)
    ax1.set_title("Evaluation action vs Step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Evaluation action")
    plt.savefig('%s/evaluation_action_vs_step.png' % (dir))
    plt.clf()

    np.save("%s/episode_reward_list.npy" % dir, episode_reward_list)
    np.save("%s/action_list.npy" % dir, action_list)

    # np.save(dir, episode_reward_list)

