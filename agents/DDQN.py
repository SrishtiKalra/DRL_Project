import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


from utils1 import Portfolio





class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DDQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=1000)
        # self.buffer_size = 60
        self.buffer_size = 128

        self.tau = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval


        if is_eval:
            # Load model with custom_objects to resolve 'mse' loss function
            print(f"in iseval")
            self.model = load_model(
                f'saved_models/{model_name}.h5',
                custom_objects={'mse': MeanSquaredError()}
            )
        else:
            self.model = self.model()

        if is_eval:
            # Load model with custom_objects to resolve 'mse' loss function
            print(f"in iseval")
            self.model_target = load_model(
                f'saved_models/{model_name}_target.h5',
                custom_objects={'mse': MeanSquaredError()}
            )
        else:
            self.model_target = self.model

        # self.model = load_model(f'saved_models/{model_name}.h5') if is_eval else self.model()
        # self.model_target = load_model(f'saved_models/{model_name}_target.h5') if is_eval else self.model
        self.model_target.set_weights(self.model.get_weights())  # hard copy model parameters to target model parameters

        self.tensorboard = TensorBoard(log_dir='./logs/DDQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def update_model_target(self):
        model_weights = self.model.get_weights()
        model_target_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            model_target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * model_target_weights[i]
        self.model_target.set_weights(model_target_weights)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        # sample random buffer_size long memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        for state, actions, reward, next_state, done in mini_batch:
            Q_expected = reward + (1 - done) * self.gamma * np.amax(self.model_target.predict(next_state)[0])

            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_expected

            history = self.model.fit(state, next_actions, epochs=1, verbose=0)
            self.update_model_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]
