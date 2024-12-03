
# Deep Reinforcement Learning for Stock Trading

This project implements Deep Reinforcement Learning (DRL) algorithms for stock trading using historical stock data. The models included are:

- **DQN** (Deep Q-Network)
- **DQN with Prioritized Experience Replay**
- **DDQN** (Double Deep Q-Network)

## Folder Structure

- `agents/`: Contains the model implementations for DQN, DDQN, and DQN with Prioritized Replay.
- `logs/`: Stores logs for each training and evaluation session.
- `visualizations/`: Stores the portfolio return graphs and other visual results.
- `saved_models/`: Stores the trained model weights.

---

## Running the Project

First, run the utility script (utils1.py) to set up the required functions:
python utils1.py

2. Training

Depending on the model you want to train, follow the steps below:
i. DQN with Prioritized Replay
python train.py

ii. DDQN
python train2.py

ii. DQN
python train3.py

Logs for each training session will be stored in the logs/ folder.

3. Evaluation

After training, evaluate the model's performance:

i. DQN with Prioritized Replay
python evaluate2.py

ii. DDQN
python evaluate.py

iii. DQN
python evaluate2.py

Evaluation logs will also be saved in the logs/ folder.


### Results

Visualizations: Portfolio return plots and other results will be stored in the visualizations/ folder.

Logs: Detailed logs for each session (training and evaluation) can be found in the logs/ folder.

