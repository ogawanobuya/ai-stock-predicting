# coding=utf-8
import random
import numpy as np
import pandas as pd
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import keras as K
tf.compat.v1.disable_eager_execution()  # tf.compat.v1.placeholder()を使うために要る
# ai_stock_5:強化学習(A2C)による株取引最適化


class Agent(object):

    def __init__(self):
        self._updater = None
        self.input_shape = (12, 9)  # (steps, channels)の形式
        self.num_actions = 2  # 「0:売り 1:買い」の二つ
        he_normal = keras.initializers.he_normal()
        # build layer
        model = Sequential()
        model.add(Conv1D(16, kernel_size=4, strides=1, padding="same",
                         input_shape=self.input_shape, kernel_initializer=he_normal, activation="relu"))
        model.add(Conv1D(16, kernel_size=4, strides=1, padding="same", kernel_initializer=he_normal, activation="relu"))
        model.add(Flatten())
        model.add(Dense(32, kernel_initializer=he_normal, activation="relu"))

        actor_layer = K.layers.Dense(self.num_actions, kernel_initializer=he_normal)
        action_evals = actor_layer(model.output)
        actions = K.layers.Lambda(self.choice_action, output_shape=(1,))(action_evals)  # policyベース->ノイズをかけてactionを選択しているためある程度のランダム性が保証される
        # actions = tf.argmax(action_evals, axis=1)  # valueベース

        critic_layer = K.layers.Dense(1, kernel_initializer=he_normal)
        critic_values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input, outputs=[actions, action_evals, critic_values])

    def set_updater(self, optimizer, value_loss_weight=1.0, entropy_weight=0.1):
        actions = tf.compat.v1.placeholder(shape=(None,), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None,), dtype="float32")

        _, action_evals, critic_values = self.model.output
        # -log(π(a|s))を出す
        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_evals, labels=actions)
        # tf.stop_gradient: Prevent policy_loss influences critic_layer.
        advantages = rewards - tf.stop_gradient(critic_values)
        # policy_lossにより大きいadvantagesに対応するneg_logsを小さくする(つまりπ(a|s)を大きくする)インセンティブが働く
        policy_loss = tf.reduce_mean(neg_logs * advantages)  # actor's loss->期待値算出するためreduce_meanしている
        value_loss = tf.keras.losses.MeanSquaredError()(rewards, critic_values)  # critic's loss->より正確に状態価値を算出することを目指す
        action_entropy = tf.reduce_mean(self.categorical_entropy(action_evals))  # action_evalsが極端な形になることを防ぐ

        loss = policy_loss + value_loss_weight * value_loss
        loss -= entropy_weight * action_entropy

        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)

        self._updater = K.backend.function(inputs=[self.model.input, actions, rewards],
                                           outputs=[loss], updates=updates)

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])

    @tf.function
    def choice_action(self, action_evals):  # 注意:model作成とは器を作るイメージのため、action_evals[0]など器であるオブジェクトからこのように値を取ろうとすると不具合が起こりやすい
        noise = tf.random.uniform(tf.shape(action_evals))  # tf.shape()を挟むことでデータ型を統一して進めれる
        return tf.argmax(action_evals - tf.math.log(-tf.math.log(noise)), axis=1)

    @tf.function
    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def evaluate(self, state):
        _state = np.expand_dims(state, axis=0)  # add batch size dimension
        return self.model.predict(_state)[0]

    def act(self, state, epsilon=0):
        if np.random.rand() <= epsilon:
            action = random.choice([0, 1])
        else:
            action = self.evaluate(state)
        return action


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
        self.actual_x = csv_data.drop(['Date'], axis=1).values
        self.actual_y = csv_data['^GSPC'].values.reshape(-1, 1)
        # Normalize the numerical values
        xScaler = StandardScaler()
        yScaler = StandardScaler()
        self.x = xScaler.fit_transform(self.actual_x)
        self.y = yScaler.fit_transform(self.actual_y)

    def begin(self):
        state_idx = random.randint(0, self.total_days - self.limit_days)
        initial_state = self.x[state_idx]
        return initial_state, state_idx

    def step(self, state_idx, action):  # action =「0:売り 1:買い」
        next_state = self.x[state_idx + 1]
        if action == 0:
            reward = self.y[state_idx + 1] - self.y[state_idx]
            actual_reward = self.actual_y[state_idx + 1] - self.actual_y[state_idx]
        else:
            reward = self.y[state_idx] - self.y[state_idx + 1]
            actual_reward = self.actual_y[state_idx] - self.actual_y[state_idx + 1]

        return next_state, reward, actual_reward


class Trainer(object):

    def __init__(self, agent, optimizer, limit_days):
        self.agent = agent
        self.observer = Observer(agent.input_shape)
        self.env = Environment(limit_days)
        self.d_experiences = deque(maxlen=500)
        self.agent.set_updater(optimizer=optimizer)  # A2Cモデルを有効にする
        self.limit_days = limit_days

    def get_batch(self, batch_size):
        batch = random.sample(self.d_experiences, batch_size)
        states = np.vstack([e[0] for e in batch])  # state = e[0]
        states = np.reshape(states, (batch_size,) + self.agent.input_shape)
        actions = [e[1] for e in batch]  # action = e[1]
        actions = np.reshape(actions, (batch_size,))  # sparse_softmax_cross_entropy_with_logitsのlabelsは1次元にする
        rewards = [e[2] for e in batch]  # reward = e[2]

        return states, actions, rewards

    def train(self, gamma=0.99, initial_epsilon=0.4, final_epsilon=0.001,
              observation_epochs=20, training_epochs=100, batch_size=16):

        fmt = "Epoch {:d}/{:d} | Loss {:.4f} | Score: {} | ActualScore: {} | epsilon={:.4f}"
        epochs = observation_epochs + training_epochs
        epsilon = initial_epsilon

        for e in range(epochs):
            # initialize
            loss = 0.0
            experiences = []
            actual_rewards = []
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

                next_state, reward, actual_reward = self.env.step(state_idx, action)
                next_state = self.observer.observe(next_state)
                experiences.append((state, action, reward, next_state))
                actual_rewards.append(actual_reward)
                state = next_state
                state_idx += 1
                game_days += 1
                if is_training:
                    states, actions, rewards = self.get_batch(batch_size)
                    loss += self.agent.update(states, actions, rewards)[0]
                if game_days >= self.limit_days:
                    break

            score = self.episode_end(experiences, gamma)
            actual_score = sum(actual_rewards)

            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / epochs

            if e % 1 == 0:
                print (fmt.format(e + 1, epochs, loss, score, actual_score, epsilon))

    def episode_end(self, experiences, gamma):
        rewards = [e[2] for e in experiences]  # reward = e[2]
        score = sum(rewards)
        d_r_list = []
        for t, r in enumerate(rewards):
            # 15日後までの報酬から割引現在価値(=Q(s,a))の算出
            future_r = [_r * (gamma ** i) for i, _r in enumerate(rewards[t:t+15])]
            d_r = sum(future_r)
            d_r_list.append(d_r)
            # データは30日分なので15日後のデータを取得できる最大数が[t = 15]
            if t == 15:
                break

        for i, e in enumerate(experiences):
            s, a, r, n_s = e
            d_r = d_r_list[i]
            self.d_experiences.append((s, a, d_r, n_s))
            if i == 15:
                break

        return score


if __name__ == "__main__":
    agent = Agent()
    trainer = Trainer(agent, Adam(), limit_days=30)
    trainer.train()
