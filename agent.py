import numpy as np
import random
from collections import deque
from model import QModel
from functools import wraps
import time


def batchify(func):
    '''
    Modifies function call depending on whether
    a single prediction is to be made or batch of predictions
    '''
    @wraps(func)
    def wrapped_func(self, state, *args, **kwargs):
        if not isinstance(state, (list, tuple)):
            batch = [state]
            batch = np.vstack(batch)
            return func(self, batch, *args, **kwargs)[0]
        else:
            batch = np.vstack(state)
            return func(self, batch, *args, **kwargs)

    return wrapped_func


class Agent:

    def __init__(self, env, memory_size, lr, gamma, epsilon, epsilon_decay, model=None):
        self.env = env
        self.memory = deque(maxlen=memory_size)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if model is None:
            self.build_model()
        else:
            import keras
            self.model = keras.models.load_model(model)

    def build_model(self):
        self.model = QModel(
            self.env.observation_space.shape[0], self.env.action_space.n, self.lr)

    @batchify
    def q(self, state):
        return self.model.predict_on_batch(state)

    @batchify
    def q_a(self, state, action):
        return self.model.predict_on_batch(state)[action]

    @batchify
    def max_q(self, state):
        return np.max(self.model.predict_on_batch(state), axis=-1)

    @batchify
    def argmax_q(self, state):
        return np.argmax(self.model.predict_on_batch(state), axis=-1)

    def pick_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return self.env.action_space.sample()

        return self.argmax_q(state)

    def batch_update(self, bs):
        loss = 0.0
        if len(self.memory) < bs:
            return loss
        batch = random.sample(self.memory, bs)
        cstates, actions, new_states, rewards, dones = list(zip(*batch))
        old_qs = self.q(cstates)

        max_qs = self.max_q(new_states)
        new_qs = old_qs[:]

        for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
            new_qs[i][action] = reward + (not done) * self.gamma * max_qs[i]

        loss = self.model.train_on_batch(np.vstack(cstates), new_qs)
        return loss

    def run_episode(self, batch_size, learn=True):
        current_state = self.env.reset()

        done = False
        total_reward = 0.0
        self.epsilon *= self.epsilon_decay
        while not done:
            if learn:
                action = self.pick_action(current_state, self.epsilon)
                new_state, reward, done, _ = self.env.step(action)
                self.memory.append(
                    (current_state, action, new_state, reward, done))
                loss = self.batch_update(batch_size)
            else:
                action = self.pick_action(current_state)
                new_state, reward, done, _ = self.env.step(action)

            total_reward += reward
            current_state = new_state
        return total_reward
