#!/usr/bin/python3

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="pytorch-ddpg")

    # Other parameter
    parser.add_argument('--seed', type=int, default=123456789, metavar='S',
                        help="Random seed (default: 123456789)")
    parser.add_argument('--experiment-name', type=str, default="ddpg_gym", metavar="EN",
                        help="Experiment name, which also stands for directory to save and load models")
    parser.add_argument('--load', default=False, metavar='L',
                        help='Whether or not to load a trained model')
    parser.add_argument('--gpu', type=bool, default=True, metavar="G",
                        help='Whether to use gpu or not, meaningless if gpu is not available')
    parser.add_argument('--save-interval-step', type=int, default=50000, metavar='SIS',
                        help='Steps between saving operation')

    # Hyper parameter
    parser.add_argument('--value-lr', type=float, default=0.001, metavar='VLR',
                        help='Learning rate for value network (default: 0.001)')
    parser.add_argument('--policy-lr', type=float, default=0.0001, metavar='PLR',
                        help='Learning rate for policy network (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='GM',
                        help='Discount factor for rewards (default: 0.99)')
    parser.add_argument('--soft-tau', type=float, default=0.001, metavar='ST',
                        help='Updating rate for target network (default: 0.01)')

    # Replay buffer
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                        help='Training batch size')
    parser.add_argument('--replay-buffer-size', type=int, default=1000000, metavar='RBS',
                        help='Length of replay buffer')
    parser.add_argument('--num-heatup', type=int, default=2000, metavar='NH',
                        help='Number of heats up')

    # Training
    parser.add_argument('--max-episode-step', type=int, default=2000, metavar='MES',
                        help='Maximum length of an episode')
    parser.add_argument('--max-total-step', type=int, default=2000000, metavar='MTS',
                        help='Maximum training iteration')
    parser.add_argument('--optimize-step', type=int, default=1, metavar='TS',
                        help='Optimization times per step, used in agent.optimize()')

    # Evaluation
    parser.add_argument('--eval-interval-step', type=int, default=10000, metavar="EIS",
                        help='Steps between two evaluation operation')
    parser.add_argument('--eval-episode', type=int, default=1, metavar="EE",
                        help='Episodes per evaluation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
