from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        try:
            self.model = self.network("weights.hdf5")
        except:
            pass
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def set_reward(self, score, ships_left):
        self.reward = 0
        if ships_left  ==  2 :
            self.reward = -1 * score
            return self.reward
        if ships_left == 1:
            self.reward = -1.5 * score
            return self.reward
        self.reward = score
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=3536))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=4, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 3536))))
            target_f = self.model.predict(state.reshape((1, 3536)))
            target_f[0][np.argmax(action)] = target
            self.model.fit(state.reshape((1, 3536)), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 3536)))[0])
        target_f = self.model.predict(state.reshape((1, 3536)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 3536)), target_f, epochs=1, verbose=0)
