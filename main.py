import gym
import numpy as np
from agent import Agent
from collections import deque
import argparse

def in_green(text): return "\033[92m" + text + "\033[0m"


def in_red(text): return "\033[91m" + text + "\033[0m"


def train(n_trials, batch_size, memory_size, lr, gamma, epsilon, epsilon_decay):
    env = gym.make("MountainCar-v0")
    agent = Agent(env, memory_size, lr, gamma, epsilon, epsilon_decay)
    successful_trials = 0
    for i in range(n_trials):
        try:
            rewards = agent.run_episode(batch_size=batch_size)
            if rewards > -200:
                successful_trials += 1
                print(in_green(
                    f'Trial: {i} Return: {rewards} Success %: {round(successful_trials*100/(i+1),3)}'))
                agent.run_episode(batch_size=32, learn=False)
            else:
                win_streak = 0.
                print(in_red(
                    f'Trial: {i} Return: {rewards} Success %: {round(successful_trials*100/(i+1),3)}'))
        except KeyboardInterrupt:
            print(f"Training interrupted at {i+1}th trial. Saving model..")
            agent.model.save(f"{i} trials.model")
            exit()


def main():
    print("Booo")
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--n_trials', type=int, default=4000, help='Number of episodes to train on')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of training batches')
    parser.add_argument('--memory', type=int, default=3000, help='Size of agent\'s replay memory')
    parser.add_argument('--lr', type=float, default=0.0009, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps', type=float, default=0.9, help='Epsilon')
    parser.add_argument('--eps_decay', type=float, default=0.99, help='Epsilon decay')
    args = parser.parse_args()

    train(args.n_trials, args.batch_size, args.memory, args.lr, args.gamma, args.eps, args.eps_decay)

if __name__ == "__main__":
    main()
