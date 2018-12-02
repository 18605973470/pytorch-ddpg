#!/usr/bin/python
import time
import numpy as np


def train(agent, env, args, experiment_dir):

    # Training Statistics
    total_training_time = 0
    total_step_list = []
    total_step_reward_list = []
    episode_reward_list = []
    episode_step_list = []
    episode_policy_loss_list = []
    episode_value_loss_list = []

    # Evaluation Statistics
    evaluate_total_step_list = []
    evaluate_episode_reward_list = []

    total_step = 0
    episode = 0
    print("Training start ...")

    wall_time_start = time.time()
    total_wall_time = 0
    while True:
        episode_reward = 0
        episode_step = 0
        state = env.reset()

        start_time = time.time()
        while True:
            # env.render()
            action = agent.noise_action(state, total_step)
            # action = ou_noise.add_noise(action, total_step)
            next_state, reward, done, info = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.optimize(optimize_step=args.optimize_step)

            # Limitation at length of an episode
            episode_step += 1
            if episode_step >= args.max_episode_step:
                done = True

            # Limitation at length of all episodes
            total_step += 1
            if total_step >= args.max_total_step:
                done = True

            agent.save_model(experiment_dir, "last")
            if total_step % args.save_interval_step == 0:
                agent.save_model(experiment_dir, str(total_step))

            if done:
                end_time = time.time()
                delta_time = end_time - start_time
                total_training_time += delta_time
                fps = total_step / total_training_time
                total_wall_time = (end_time - wall_time_start)
                episode += 1
                print("Training time {}, episode {}, total step {}, episode step {}, episode reward {}, fps {}".format(
                    time.strftime('%H:%M:%S', time.gmtime(total_wall_time)), episode, total_step,
                    episode_step, episode_reward, fps)
                )
                break

        # Logging
        episode_reward_list.append(episode_reward)
        # fig1, ax1 = plt.subplots(figsize=(11, 8))
        # ax1.plot(range(episode), episode_reward_list)
        # ax1.set_title("Reward vs Episode")
        # ax1.set_xlabel("Episode")
        # ax1.set_ylabel("Reward")
        # plt.savefig('%s/reward_vs_episode.png' % (experiment_dir))
        # plt.clf()
        np.save("%s/episode_reward.npy" % experiment_dir, episode_reward_list)

        total_step_list.append(total_step)
        total_step_reward_list.append([total_step, episode_reward])
        # fig2, ax2 = plt.subplots(figsize=(11, 8))
        # ax2.plot(total_step_list, episode_reward_list)
        # ax2.set_title("Reward vs Step")
        # ax2.set_xlabel("Step")
        # ax2.set_ylabel("Reward")
        # plt.savefig('%s/reward_vs_step.png' % (experiment_dir))
        # plt.clf()
        np.save("%s/step_reward.npy" % experiment_dir, total_step_reward_list)

        episode_step_list.append(episode_step)
        # fig3, ax3 = plt.subplots(figsize=(11, 8))
        # ax3.plot(range(episode), episode_step_list)
        # ax3.set_title("Episode length vs Episode")
        # ax3.set_xlabel("Episode")
        # ax3.set_ylabel("Episode length")
        # plt.savefig('%s/episode_length_vs_episode.png' % (experiment_dir))
        # plt.clf()
        np.save("%s/episode_length.npy" % experiment_dir, episode_step_list)

        # Begin evaluation
        # if args.eval_episode > 0 and total_step - last_eval_step >= args.eval_interval_step:
        #     last_eval_step = total_step
        #     evaluate_total_step_list.append(total_step)
        #     evaluate_episode_reward_list.append(evaluate(agent, env, args))

            # fig, ax = plt.subplots(figsize=(11, 8))
            # ax.plot(evaluate_total_step_list, evaluate_episode_reward_list)
            # ax.set_title("Evaluation reward vs Step")
            # ax.set_xlabel("Step")
            # ax.set_ylabel("Evaluation reward")
            # plt.savefig('%s/evaluation_reward_vs_step.png' % (experiment_dir))
            # plt.clf()
        # End evaluation

        if total_step >= args.max_total_step:
            break

    # env.close()

    print("Training ends ...")
