import argparse
import importlib
import logging
import time
import numpy as np
from utils1 import stock_close_prices, generate_combined_state, evaluate_portfolio_performance, plot_portfolio_returns_across_episodes

# Parse command-line arguments
parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int,
                    help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=5, type=int, help='number of episodes')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
                    help='initial balance')
inputs = parser.parse_args()

# Unpack inputs
model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

# Load stock prices and initialize variables
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# Select learning model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)


def hold(actions):
    """Encourage selling for profit and liquidity"""
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[next_probable_action] = 1  # Reset this action's value to the highest
            return 'Hold', actions


def buy(t):
    """Execute a buy action"""
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])


def sell(t):
    """Execute a sell action"""
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)


# Configure logging
logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

# Start training
start_time = time.time()

for e in range(1, num_episode + 1):
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset()  # Reset to initial balance and hyperparameters
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        # Select action
        # actions = agent.model.predict(np.expand_dims(state, axis=0))[0]  # Add batch dimension

        # Ensure the state has the correct shape
        state = np.squeeze(state) if len(state.shape) > 2 else state  # Remove extra dimensions if needed
        state = state[np.newaxis, :] if len(state.shape) == 1 else state  # Add batch dimension (1, state_dim)

        # Predict actions
        actions = agent.model.predict(state)[0]  # No further reshaping needed

        action = agent.act(state)

        # Execute position
        logging.info(
            'Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1],
                                                                                           actions[2]))
        if action != np.argmax(actions):
            logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0:  # Hold
            execution_result = hold(actions)
        elif action == 1:  # Buy
            execution_result = buy(t)
        elif action == 2:  # Sell
            execution_result = sell(t)

        # Check execution result
        if execution_result is None:
            reward -= 0.01  # Small penalty for inaction
        else:
            if isinstance(execution_result, tuple):  # If execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]
            logging.info(execution_result)

        # Reward logic
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value  # Base reward
        reward *= 100  # Scale reward

        # Calculate TD error for prioritized replay
        Q_expected = reward
        if t < trading_period:  # Ensure next_state is valid
            # Fix next_state shape
            print(f"Next state shape before reshaping: {next_state.shape}")
            next_state = np.squeeze(next_state) if len(next_state.shape) > 2 else next_state
            next_state = next_state[np.newaxis, :] if len(next_state.shape) == 1 else next_state

            print(f"Next state shape after reshaping: {next_state.shape}")

            # Calculate Q_expected
            Q_expected += agent.gamma * np.amax(agent.model.predict(next_state)[0])

        current_q_value = agent.model.predict(state)[0][action]
        td_error = Q_expected - current_q_value

        # Append experience to replay memory
        done = True if t == trading_period else False
        agent.remember(state, action, reward, next_state, done, td_error)

        # Update state
        state = next_state

        # Experience replay
        if len(agent.memory.data) > agent.buffer_size:
            num_experience_replay += 1
            loss = agent.experience_replay()
            logging.info(
                'Episode: {}\tLoss: {:.4f}\tAction: {}\tReward: {:.4f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(
                    e, loss, action_dict[action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 5 == 0:
        if model_name == 'DQN':
            agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
        elif model_name == 'DDPG':
            agent.actor.model.save_weights('saved_models/DDPG_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/DDPG_ep{}_critic.h5'.format(str(e)))
        elif model_name == 'DDQN':  # Add this for DDQN
            agent.model.save(f'saved_models/DDQN_ep{e}.h5')
            agent.model_target.save(f'saved_models/DDQN_ep{e}_target.h5')  # Save target network

        logging.info('model saved')

logging.info('total training time: {0:.2f} min'.format((time.time() - start_time) / 60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)

# import argparse
# import importlib
# import logging
# import sys
# import time
#
# from utils1 import *
#
# parser = argparse.ArgumentParser(description='command line options')
# parser.add_argument('--model_name', action="store", dest="model_name", default='DDQN', help="model name")
# parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
# parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int,
#                     help="span (days) of observation")
# parser.add_argument('--num_episode', action="store", dest="num_episode", default=1, type=int, help='episode number')
# parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
#                     help='initial balance')
# inputs = parser.parse_args()
#
# model_name = inputs.model_name
# stock_name = inputs.stock_name
# window_size = inputs.window_size
# num_episode = inputs.num_episode
# initial_balance = inputs.initial_balance
#
# stock_prices = stock_close_prices(stock_name)
# trading_period = len(stock_prices) - 1
# returns_across_episodes = []
# num_experience_replay = 0
# action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
#
# # select learning model
# model = importlib.import_module(f'agents.{model_name}')
# agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)
#
#
# def hold(actions):
#     # encourage selling for profit and liquidity
#     next_probable_action = np.argsort(actions)[1]
#     if next_probable_action == 2 and len(agent.inventory) > 0:
#         max_profit = stock_prices[t] - min(agent.inventory)
#         if max_profit > 0:
#             sell(t)
#             actions[next_probable_action] = 1  # reset this action's value to the highest
#             return 'Hold', actions
#
#
# def buy(t):
#     if agent.balance > stock_prices[t]:
#         agent.balance -= stock_prices[t]
#         agent.inventory.append(stock_prices[t])
#         return 'Buy: ${:.2f}'.format(stock_prices[t])
#
#
# def sell(t):
#     if len(agent.inventory) > 0:
#         agent.balance += stock_prices[t]
#         bought_price = agent.inventory.pop(0)
#         profit = stock_prices[t] - bought_price
#         global reward
#         reward = profit
#         return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)
#
#
# # configure logging
# logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
#                     format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
#
# logging.info(f'Trading Object:           {stock_name}')
# logging.info(f'Trading Period:           {trading_period} days')
# logging.info(f'Window Size:              {window_size} days')
# logging.info(f'Training Episode:         {num_episode}')
# logging.info(f'Model Name:               {model_name}')
# logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))
#
# start_time = time.time()
# for e in range(1, num_episode + 1):
#     logging.info(f'\nEpisode: {e}/{num_episode}')
#
#     agent.reset()  # reset to initial balance and hyperparameters
#     state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
#
#     for t in range(1, trading_period + 1):
#         if t % 100 == 0:
#             logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')
#
#         reward = 0
#         next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
#         previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
#
#         if model_name == 'DDPG':
#             actions = agent.act(state, t)
#             action = np.argmax(actions)
#         elif model_name == 'DDQN':
#             action = agent.act(state)  # Handle exploration and exploitation directly
#             actions = agent.model.predict(state)[0]  # Optional: for logging only
#         else:
#             actions = agent.model.predict(state)[0]
#             action = agent.act(state)
#
#         # execute position
#         logging.info(
#             'Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1],
#                                                                                            actions[2]))
#         if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
#         if action == 0:  # hold
#             execution_result = hold(actions)
#         if action == 1:  # buy
#             execution_result = buy(t)
#         if action == 2:  # sell
#             execution_result = sell(t)
#
#             # check execution result
#         if execution_result is None:
#             reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
#         else:
#             if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
#                 actions = execution_result[1]
#                 execution_result = execution_result[0]
#             logging.info(execution_result)
#
#             # calculate reward
#         current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
#         unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
#         reward += unrealized_profit
#
#         agent.portfolio_values.append(current_portfolio_value)
#         agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
#
#         done = True if t == trading_period else False
#         agent.remember(state, actions, reward, next_state, done)
#
#         # update state
#         state = next_state
#
#         # experience replay
#         if len(agent.memory) > agent.buffer_size:
#             num_experience_replay += 1
#             loss = agent.experience_replay()
#             logging.info(
#                 'Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e,
#                                                                                                                       loss,
#                                                                                                                       action_dict[
#                                                                                                                           action],
#                                                                                                                       reward,
#                                                                                                                       agent.balance,
#                                                                                                                       len(agent.inventory)))
#             agent.tensorboard.on_batch_end(num_experience_replay,
#                                            {'loss': loss, 'portfolio value': current_portfolio_value})
#
#         if done:
#             portfolio_return = evaluate_portfolio_performance(agent, logging)
#             returns_across_episodes.append(portfolio_return)
#
#     # save models periodically
#     if e % 5 == 0:
#         if model_name == 'DQN':
#             agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
#         elif model_name == 'DDQN':
#             # Save both the primary model and target model for DDQN
#             agent.model.save('saved_models/DDQN_ep' + str(e) + '_model.h5')
#             agent.model_target.save('saved_models/DDQN_ep' + str(e) + '_target_model.h5')
#         elif model_name == 'DDPG':
#             agent.actor.model.save_weights('saved_models/DDPG_ep{}_actor.h5'.format(str(e)))
#             agent.critic.model.save_weights('saved_models/DDPG_ep{}_critic.h5'.format(str(e)))
#         logging.info('model saved')
#
#     if model_name == 'DDQN':
#         agent.update_model_target()  # Perform soft or hard updates
#
# logging.info('total training time: {0:.2f} min'.format((time.time() - start_time) / 60))
# plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)














# import argparse
# import importlib
# import logging
# import sys
# import time
#
# from utils1 import *
#
# parser = argparse.ArgumentParser(description='command line options')
# parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
# parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
# parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int,
#                     help="span (days) of observation")
# parser.add_argument('--num_episode', action="store", dest="num_episode", default=5, type=int, help='episode number')
# parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
#                     help='initial balance')
# parser.add_argument('--batch_size', action="store", dest="batch_size", default=32, type=int,
#                     help="mini-batch size for experience replay")
# inputs = parser.parse_args()
#
# model_name = inputs.model_name
# stock_name = inputs.stock_name
# window_size = inputs.window_size
# num_episode = inputs.num_episode
# initial_balance = inputs.initial_balance
# batch_size = inputs.batch_size
#
# stock_prices = stock_close_prices(stock_name)
# trading_period = len(stock_prices) - 1
# returns_across_episodes = []
# num_experience_replay = 0
# action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
#
# # select learning model
# model = importlib.import_module(f'agents.{model_name}')
# agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)
#
#
# def hold(actions):
#     # encourage selling for profit and liquidity
#     next_probable_action = np.argsort(actions)[1]
#     if next_probable_action == 2 and len(agent.inventory) > 0:
#         max_profit = stock_prices[t] - min(agent.inventory)
#         if max_profit > 0:
#             sell(t)
#             actions[next_probable_action] = 1  # reset this action's value to the highest
#             return 'Hold', actions
#
#
# # def buy(t):
#     # if agent.balance > stock_prices[t]:
#     #     agent.balance -= stock_prices[t]
#     #     agent.inventory.append(stock_prices[t])
#     #     return 'Buy: ${:.2f}'.format(stock_prices[t])
#
# def buy(t):
#     if agent.balance > stock_prices[t]:
#         agent.balance -= stock_prices[t]
#         agent.inventory.append(stock_prices[t])
#         return 'Buy: ${:.2f}'.format(stock_prices[t])
#     else:
#         return None  # Avoid invalid trades
#
#
#
#
# # def sell(t):
# #     if len(agent.inventory) > 0:
# #         agent.balance += stock_prices[t]
# #         bought_price = agent.inventory.pop(0)
# #         profit = stock_prices[t] - bought_price
# #         global reward
# #         reward = profit
# #         return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)
#
# def sell(t):
#     if len(agent.inventory) > 0:
#         agent.balance += stock_prices[t]
#         bought_price = agent.inventory.pop(0)
#         profit = stock_prices[t] - bought_price
#         global reward  # Update the global reward
#         reward = profit
#         return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)
#     else:
#         return None  # Avoid invalid trades
#
#
#
# # configure logging
# logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
#                     format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
#
# logging.info(f'Trading Object:           {stock_name}')
# logging.info(f'Trading Period:           {trading_period} days')
# logging.info(f'Window Size:              {window_size} days')
# logging.info(f'Training Episode:         {num_episode}')
# logging.info(f'Model Name:               {model_name}')
# logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))
#
# start_time = time.time()
# for e in range(1, num_episode + 1):
#     logging.info(f'\nEpisode: {e}/{num_episode}')
#
#     agent.reset()  # Reset to initial balance and hyperparameters
#     state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
#
#     for t in range(1, trading_period + 1):
#         if t % 100 == 0:
#             logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')
#
#         reward = 0
#         next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
#         previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
#
#         if model_name == 'DDPG':
#             actions = agent.act(state, t)
#             action = np.argmax(actions)
#         else:
#             actions = agent.model.predict(state)[0]
#             action = agent.act(state)
#
#         # Execute position
#         logging.info(
#             'Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1],
#                                                                                            actions[2]))
#         if action != np.argmax(actions):
#             logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
#         if action == 0:  # Hold
#             execution_result = hold(actions)
#         elif action == 1:  # Buy
#             execution_result = buy(t)
#         elif action == 2:  # Sell
#             execution_result = sell(t)
#
#         # Check execution result
#         if execution_result is None:
#             reward -= treasury_bond_daily_return_rate() * agent.balance  # Missing opportunity cost
#         else:
#             if isinstance(execution_result, tuple):  # If execution_result is 'Hold'
#                 actions = execution_result[1]
#                 execution_result = execution_result[0]
#             logging.info(execution_result)
#
#         # Calculate portfolio value
#         current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
#
#         # Normalize portfolio value
#         max_possible_value = max(stock_prices) * (len(agent.inventory) + 1) + agent.balance
#         portfolio_value_normalized = current_portfolio_value / max_possible_value
#
#         # Initialize reward components
#         realized_profit = 0
#         unrealized_profit = 0
#
#         # Handle realized profit for Sell actions
#         if action == 2:  # Sell
#             realized_profit = reward  # Profit from the sell action
#             if realized_profit < 0:  # Penalize unprofitable sells
#                 reward -= abs(realized_profit)
#
#         # Calculate unrealized profit (difference in normalized portfolio value)
#         unrealized_profit = current_portfolio_value - previous_portfolio_value
#
#
#         # Penalize unprofitable Buy actions
#         if action == 1 and agent.balance < stock_prices[t]:
#             reward -= 0.01  # Small penalty for trying to buy without enough balance
#
#         # Combine realized and unrealized profit into reward
#         reward += realized_profit + unrealized_profit
#
#         # Normalize reward magnitude (scaling)
#         reward /= max_possible_value
#
#         # Apply a discount factor (gamma) for long-term rewards
#         gamma = 0.99  # Discount factor for long-term rewards
#         discounted_reward = reward * (gamma ** t)
#
#         # Assign the discounted reward back to the final reward
#         reward = discounted_reward
#
#
#         agent.portfolio_values.append(current_portfolio_value)
#         agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
#
#         done = True if t == trading_period else False
#         agent.remember(state, actions, reward, next_state, done)
#
#         # Update state
#         state = next_state
#
#         # Experience replay
#         if len(agent.memory) > batch_size:
#             num_experience_replay += 1
#             loss = agent.experience_replay(batch_size=batch_size)
#             logging.info(
#                 'Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e,
#                                                                                                                       loss,
#                                                                                                                       action_dict[
#                                                                                                                           action],
#                                                                                                                       reward,
#                                                                                                                       agent.balance,
#                                                                                                                       len(agent.inventory)))
#             agent.tensorboard.on_batch_end(num_experience_replay,
#                                            {'loss': loss, 'portfolio value': current_portfolio_value})
#
#         if done:
#             portfolio_return = evaluate_portfolio_performance(agent, logging)
#             returns_across_episodes.append(portfolio_return)
#
#     # Save models periodically
#     if e % 5 == 0:
#         if model_name == 'DQN':
#             agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
#         elif model_name == 'DDPG':
#             agent.actor.model.save_weights('saved_models/DDPG_ep{}_actor.h5'.format(str(e)))
#             agent.critic.model.save_weights('saved_models/DDPG_ep{}_critic.h5'.format(str(e)))
#         logging.info('model saved')
#
# logging.info('total training time: {0:.2f} min'.format((time.time() - start_time) / 60))
# plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
