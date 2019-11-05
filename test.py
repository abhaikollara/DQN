import gym
from agent import Agent
import argparse

def test(model):
    env = gym.make("MountainCar-v0")
    agent = Agent(env, None, None, None, None, None, model=model) #Refactor to avoid this
    current_state = env.reset()
    done = False
    while not done:
        action = agent.pick_action(current_state)
        next_state, _, _, _ = env.step(action)
        current_state = next_state
        env.render()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model', type=str, help='Path to saved model')

    args = parser.parse_args()

    test(args.model)