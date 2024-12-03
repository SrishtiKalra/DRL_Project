import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input


from utils1 import Portfolio


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Stores actual experiences
        self.write = 0  # Current write position for new data

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # Leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]  # Total priority

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # Overwrite when capacity is full

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = SumTree(capacity=1000)  # Replay memory using SumTree
        self.buffer_size = 32  # Mini-batch size for experience replay

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4  # Importance sampling weight exponent
        self.beta_increment_per_step = 0.001  # Anneal beta over time
        self.is_eval = is_eval

        if is_eval:
            self.model = load_model(f'saved_models/{model_name}.h5', custom_objects={'mse': MeanSquaredError()})
        else:
            self.model = self.build_model()

        self.tensorboard = TensorBoard(log_dir=f'./logs/{model_name}_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_dim,)))  # Define input shape
        model.add(Dense(units=64, activation='relu'))  # First hidden layer
        model.add(Dense(units=32, activation='relu'))  # Second hidden layer
        model.add(Dense(units=8, activation='relu'))  # Third hidden layer
        model.add(Dense(self.action_dim))  # Output layer with no activation
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))  # Compile with MSE loss
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0

    def remember(self, state, action, reward, next_state, done, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.memory.add(priority, (state, action, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        mini_batch = []
        indices = []
        priorities = []

        for _ in range(self.buffer_size):
            s = random.uniform(0, self.memory.total())
            idx, priority, data = self.memory.get(s)
            mini_batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Normalize priorities for importance sampling
        sampling_probabilities = priorities / self.memory.total()
        is_weights = np.power(len(self.memory.data) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_step)

        losses = []

        for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
            Q_expected = reward
            if not done:
                Q_expected += self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            td_error = Q_expected - target[0][action]  # Use action as an index directly
            target[0][action] = Q_expected

            # Update priority in SumTree
            self.memory.update(indices[i], abs(td_error) + 1e-5)

            # Train the model and capture the loss
            history = self.model.fit(
                state[np.newaxis, :] if len(state.shape) == 1 else state,  # Ensure shape is (batch_size, state_dim)
                target,
                epochs=1,
                verbose=0,
                sample_weight=is_weights[i:i + 1]
            )

            losses.append(history.history['loss'][0])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return the average loss for this replay batch
        return np.mean(losses)

# import random
# from collections import deque
#
# import numpy as np
# from tensorflow.keras import Sequential
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import MeanSquaredError
# import logging
#
# from utils1 import Portfolio
#
#
# import numpy as np
#
#
# class SumTree:
#     def __init__(self, capacity):
#         """
#         Create a SumTree with a fixed capacity.
#         :param capacity: The maximum number of elements the tree can hold.
#         """
#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity - 1)  # Binary tree structure
#         self.data = np.zeros(capacity, dtype=object)  # Experience buffer
#         self.write = 0  # Pointer to the next write location
#         self.total_elements = 0  # Number of elements added so far
#
#     def add(self, priority, data):
#         """
#         Add a new experience with a given priority to the tree.
#         :param priority: Priority of the experience.
#         :param data: The experience data to store.
#         """
#         tree_idx = self.write + self.capacity - 1  # Map data index to tree index
#         self.data[self.write] = data  # Store experience in the buffer
#         self.update(tree_idx, priority)  # Update the tree with the new priority
#
#         # Increment write pointer and handle circular buffer logic
#         self.write += 1
#         if self.write >= self.capacity:
#             self.write = 0
#
#         # Track the number of elements in the buffer
#         if self.total_elements < self.capacity:
#             self.total_elements += 1
#
#     def update(self, tree_idx, priority):
#         """
#         Update the priority of a specific leaf node.
#         :param tree_idx: The index of the leaf node in the tree.
#         :param priority: The new priority value.
#         """
#         change = priority - self.tree[tree_idx]  # Calculate the change in priority
#         self.tree[tree_idx] = priority  # Update the leaf node
#         # Propagate the change up the tree
#         while tree_idx != 0:
#             tree_idx = (tree_idx - 1) // 2
#             self.tree[tree_idx] += change
#
#     def get(self, value):
#         """
#         Retrieve an experience and its priority based on a cumulative value.
#         :param value: A cumulative sum value (sampled from [0, total_priority]).
#         :return: (tree_idx, priority, data) of the sampled experience.
#         """
#         tree_idx = 0  # Start at the root
#         while tree_idx < self.capacity - 1:  # While not at a leaf node
#             left = 2 * tree_idx + 1  # Left child index
#             right = left + 1  # Right child index
#             if value <= self.tree[left]:
#                 tree_idx = left
#             else:
#                 value -= self.tree[left]
#                 tree_idx = right
#         data_idx = tree_idx - (self.capacity - 1)  # Map tree index to data index
#         return tree_idx, self.tree[tree_idx], self.data[data_idx]
#
#     def total_priority(self):
#         """
#         Get the total priority (sum of all priorities in the tree).
#         :return: The total priority value.
#         """
#         return self.tree[0]
#
#     def __len__(self):
#         """
#         Get the current size of the buffer (number of stored experiences).
#         :return: The number of elements in the buffer.
#         """
#         return self.total_elements
#
#
# class Agent(Portfolio):
#     def __init__(self, state_dim, balance, is_eval=False, model_name=""):
#         super().__init__(balance=balance)
#         self.model_type = 'DQN'
#         self.state_dim = state_dim
#         self.action_dim = 3  # hold, buy, sell
#         self.memory = SumTree(capacity=1000)  # Replace deque with SumTree
#
#         self.gamma = 0.95
#         self.epsilon = 1.0  # Initial exploration rate
#         self.epsilon_min = 0.01  # Minimum exploration rate
#         self.epsilon_decay = 0.995  # Decay rate for exploration
#         self.is_eval = is_eval
#
#         self.alpha = 0.6  # Degree of prioritization
#         self.beta = 0.4  # Importance-sampling exponent (initial value)
#         self.beta_increment_per_sampling = 0.001  # Increment per sampling step
#         self.epsilon_priority = 0.01  # Small constant to avoid zero priority
#
#         if is_eval:
#             # Load model with custom_objects to resolve 'mse' loss function
#             print(f"in iseval")
#             self.model = load_model(
#                 f'saved_models/{model_name}.h5',
#                 custom_objects={'mse': MeanSquaredError()}
#             )
#         else:
#             self.model = self.build_model()
#
#         # self.model = load_model(f'saved_models/{model_name}.h5') if is_eval else self.build_model()
#
#         self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
#         self.tensorboard.set_model(self.model)
#
#     def build_model(self):
#         print(f"in build model")
#         model = Sequential()
#         model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
#         model.add(Dense(units=32, activation='relu'))
#         model.add(Dense(units=8, activation='relu'))
#         model.add(Dense(self.action_dim, activation='softmax'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
#         return model
#
#     def reset(self):
#         self.reset_portfolio()
#         self.epsilon = 1.0 # reset exploration rate
#
#     def remember(self, state, actions, reward, next_state, done):
#         priority = max(self.memory.tree[-self.memory.capacity:]) if self.memory.total_elements > 0 else 1.0
#         self.memory.add(priority, (state, actions, reward, next_state, done))
#
#     # def act(self, state):
#     #     if not self.is_eval and np.random.rand() <= self.epsilon:
#     #         return random.randrange(self.action_dim)
#     #     options = self.model.predict(state)
#     #     # Log the predicted Q-values in evaluate.py
#     #     if self.is_eval:  # Log only during evaluation
#     #         logging.info(f"Predicted Q-values: {options[0]}")
#     #         logging.info("Action chosen: %d", np.argmax(options[0]))
#     #     return np.argmax(options[0])
#
#     def act(self, state):
#         # Ensure state has the correct shape (add batch dimension if needed)
#         if len(state.shape) == 1:  # If state is 1D, add a batch dimension
#             state = np.expand_dims(state, axis=0)
#
#         # Exploration: Choose a random action
#         if not self.is_eval and np.random.rand() <= self.epsilon:
#             action = random.randrange(self.action_dim)
#             logging.info(f"Exploration: Random action {action}")
#             return action
#
#         # Exploitation: Predict Q-values
#         options = self.model.predict(state)
#
#         # Log predicted Q-values during evaluation
#         if self.is_eval:
#             logging.info(f"Predicted Q-values: {options[0]}")
#             logging.info("Action chosen: %d", np.argmax(options[0]))
#
#         # Return the action with the highest Q-value
#         return np.argmax(options[0])
#
#     def sample(self, batch_size):
#         mini_batch = []
#         indices = []
#         priorities = []
#         segment = self.memory.total_priority() / batch_size
#         for i in range(batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             s = np.random.uniform(a, b)  # Sample within the segment
#             idx, priority, data = self.memory.get(s)
#             mini_batch.append(data)
#             indices.append(idx)
#             priorities.append(priority)
#         sampling_probabilities = np.array(priorities) / self.memory.total_priority()
#         is_weights = (1 / (len(self.memory) * sampling_probabilities)) ** self.beta
#         is_weights /= is_weights.max()
#         return mini_batch, indices, is_weights
#
#     def update_priorities(self, indices, td_errors):
#         for idx, error in zip(indices, td_errors):
#             priority = abs(error) + self.epsilon
#             self.memory.update(idx, priority)
#
#     def experience_replay(self, batch_size=32):
#         """
#         Perform experience replay with prioritized sampling.
#         """
#         if len(self.memory) < batch_size:
#             return 0.0  # No replay performed if memory size is insufficient
#
#         # Use the SumTree's sampling method
#         mini_batch, indices, is_weights = self.sample(batch_size)
#
#         # Unpack the sampled mini-batch
#         td_errors = []
#         losses = []
#         for i in range(batch_size):
#             state, actions, reward, next_state, done = mini_batch[i]
#
#             # Calculate the target Q-value
#             target = reward
#             if not done:
#                 target += self.gamma * np.amax(self.model.predict(next_state)[0])
#
#             # Update Q-value for the chosen action
#             current_q_values = self.model.predict(state)
#             td_error = target - current_q_values[0][np.argmax(actions)]
#             td_errors.append(td_error)
#             current_q_values[0][np.argmax(actions)] = target
#
#             # Perform gradient descent with importance-sampling weight
#             history = self.model.fit(
#                 state,
#                 current_q_values,
#                 epochs=1,
#                 verbose=0,
#                 sample_weight=np.array([is_weights[i]])
#             )
#             # Collect the loss for this update
#             if history and 'loss' in history.history:
#                 losses.append(history.history['loss'][0])
#             else:
#                 losses.append(0.0)
#
#         # Update priorities in SumTree
#         self.update_priorities(indices, td_errors)
#
#         # Decay epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#         # Return the average loss for this replay batch
#         return np.mean(losses) if losses else 0.0
#
#
#
