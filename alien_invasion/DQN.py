from pathlib import Path
import random

import numpy as np
import pandas as pd

try:
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError as error:  # pragma: no cover - depends on optional ML stack.
    raise ImportError(
        "DQNAgent requires TensorFlow/Keras. Install the ML environment with "
        "`python3.12 -m pip install -r requirements-ml.txt`."
    ) from error


DEFAULT_WEIGHTS = Path(__file__).with_name("weights.weights.h5")
LEGACY_WEIGHTS = Path(__file__).with_name("weights.hdf5")


class DQNAgent(object):

    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        self.weights_path = Path(weights_path)
        if self.weights_path.exists():
            self.model.load_weights(str(self.weights_path))
        elif LEGACY_WEIGHTS.exists():
            print(
                f"Legacy weights found at {LEGACY_WEIGHTS}; pass "
                f"`--weights {LEGACY_WEIGHTS}` if you need to inspect/load them."
            )
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def set_reward(self, score, beforescore, ships_left):
        self.reward = 0
        if ships_left == 2 and beforescore == score:
            self.reward = -10
            return self.reward
        if ships_left == 1 and beforescore == score:
            self.reward = -20
            return self.reward
        if beforescore > score:
            self.reward = score
        if ships_left == 3 and beforescore == score:
            self.reward = -2
            return self.reward
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(120, activation='relu', input_shape=(3536,)))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(4, activation='softmax'))
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(str(weights))
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
