# coding=utf-8
import random
import numpy as np
import pandas as pd
from collections import deque
import keras
from keras import regularizers
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# ai_stock_4:強化学習(DQN)による株取引最適化


class Agent(object):

    def __init__(self):
        self.input_shape = (8, 9)
        self.num_actions = 2  # 「0:売り 1:買い」の二つ
        he_normal = keras.initializers.he_normal()
        # build layer
        model = Sequential()
        model.add(LSTM(32, dropout=0.3, batch_input_shape=(None, self.input_shape[0], self.input_shape[1]), return_sequences=False))
        model.add(Dense(32, activation='relu', kernel_initializer=he_normal))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_actions, activation='relu', kernel_initializer=he_normal))
        self.model = model

    def evaluate(self, state, model=None):
        _model = model if model else self.model
        _state = np.expand_dims(state, axis=0)  # add batch size dimension
        return _model.predict(_state)[0]

    def act(self, state, epsilon=0):
        if np.random.rand() <= epsilon:
            a = random.choice([0, 1])
        else:
            q = self.evaluate(state)
            a = np.argmax(q)
        return a


class Observer(object):

    def __init__(self, input_shape):
        self.time_length = input_shape[0]
        self._states = []

    def observe(self, state):
        if len(self._states) == 0:
            # full fill the frame cache
            self._states = [state] * self.time_length
        else:
            self._states.append(state)
            self._states.pop(0)  # remove most old state

        input_state = np.array(self._states)
        return input_state


class Environment(object):

    def __init__(self, limit_days):
        self.limit_days = limit_days
        self.total_days = 253
        csv_data = pd.read_csv('data/test_data.csv')
        x = csv_data.drop(['Date'], axis=1).values
        y = csv_data['^GSPC'].values.reshape(-1, 1)
        # Normalize the numerical values
        xScaler = StandardScaler()
        yScaler = StandardScaler()
        self.x = xScaler.fit_transform(x)
        self.y = yScaler.fit_transform(y)

    def begin(self):
        state_idx = random.randint(0, self.total_days - self.limit_days)
        initial_state = self.x[state_idx]
        return initial_state, state_idx

    def step(self, state_idx, action):  # action =「0:売り 1:買い」
        next_state = self.x[state_idx + 1]
        if action == 0:
            reward = self.y[state_idx + 1] - self.y[state_idx]
        else:
            reward = self.y[state_idx] - self.y[state_idx + 1]

        return next_state, reward


class Trainer(object):

    def __init__(self, agent, optimizer, limit_days):
        self.agent = agent
        self.observer = Observer(agent.input_shape)
        self.env = Environment(limit_days)
        self.experience = []
        self._target_model = clone_model(self.agent.model)
        self.agent.model.compile(optimizer=optimizer, loss="mse")
        self.limit_days = limit_days

    def get_batch(self, batch_size, gamma):
        batch_indices = np.random.randint(
            low=0, high=len(self.experience), size=batch_size)
        X = np.zeros((batch_size,) + self.agent.input_shape)
        y = np.zeros((batch_size, self.agent.num_actions))
        for i, b_i in enumerate(batch_indices):
            s, a, r, next_s = self.experience[b_i]
            X[i] = s
            y[i] = self.agent.evaluate(s)
            # future reward
            Q_sa = np.max(self.agent.evaluate(next_s, model=self._target_model))
            y[i, a] = r + gamma * Q_sa

        return X, y

    def train(self, gamma=0.99, initial_epsilon=0.4, final_epsilon=0.001,
              memory_size=500, observation_epochs=8, training_epochs=20, batch_size=4):

        fmt = "Epoch {:d}/{:d} | Score: {} | epsilon={:.4f}"
        self.experience = deque(maxlen=memory_size)
        epochs = observation_epochs + training_epochs
        epsilon = initial_epsilon

        for e in range(epochs):
            # initialize
            rewards = []
            self.observer._states = []
            initial_state, state_idx = self.env.begin()
            state = self.observer.observe(initial_state)
            game_days = 1
            is_training = True if e > observation_epochs else False
            # let's play the game
            while True:
                if not is_training:
                    action = self.agent.act(state, epsilon=1)
                else:
                    action = self.agent.act(state, epsilon)

                next_state, reward = self.env.step(state_idx, action)
                next_state = self.observer.observe(next_state)
                self.experience.append((state, action, reward, next_state))
                rewards.append(reward)
                state = next_state
                state_idx += 1
                game_days += 1
                if is_training:
                    X, y = self.get_batch(batch_size, gamma)
                    self.agent.model.train_on_batch(X, y)
                if game_days >= self.limit_days:
                    break

            score = sum(rewards)
            if is_training:
                self._target_model.set_weights(self.agent.model.get_weights())

            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / epochs

            if e % 1 == 0:
                print (fmt.format(e + 1, epochs, score, epsilon))


if __name__ == "__main__":
    agent = Agent()
    trainer = Trainer(agent, Adam(), limit_days=30)
    trainer.train()
