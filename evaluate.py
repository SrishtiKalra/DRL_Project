import argparse
import importlib
import logging
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

from utils1 import *

parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DDQN_ep5', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2000-2017', help="stock name")
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
stock_name = inputs.stock_name
initial_balance = inputs.initial_balance
display = True
window_size = 10
action_dict = {0: 'Hold', 1: 'Hold', 2: 'Sell'}

# Load agent model
model = importlib.import_module(f'agents.{model_name}')


def hold(t, current_portfolio_value, previous_portfolio_value):
    # Calculate unrealized profit/loss for holding
    unrealized_profit = current_portfolio_value - previous_portfolio_value

    # Deduct opportunity cost for holding
    opportunity_cost = treasury_bond_daily_return_rate() * agent.balance

    # Adjust reward for holding
    global reward
    reward += unrealized_profit - opportunity_cost

    # Log the action and impacts
    logging.info(f"Hold: Unrealized Profit: ${unrealized_profit:.2f}, Opportunity Cost: ${opportunity_cost:.2f}")


def buy(t):
    agent.balance -= stock_prices[t]
    agent.inventory.append(stock_prices[t])
    agent.buy_dates.append(t)
    logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))


def sell(t):
    agent.balance += stock_prices[t]
    bought_price = agent.inventory.pop(0)
    profit = stock_prices[t] - bought_price
    global reward
    reward = profit
    agent.sell_dates.append(t)
    logging.info('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))


# Configure logging
logging.basicConfig(
    filename=f'logs/{model_name}_evaluation_{stock_name}.log',
    filemode='w',
    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

portfolio_return = 0
while portfolio_return == 0:  # A hack to avoid stationary case
    # Initialize the agent
    agent = model.Agent(state_dim=13, balance=initial_balance, is_eval=True, model_name=model_to_load)

    # Load the model with a custom object for the loss function
    agent.model = load_model(
        f'saved_models/{model_to_load}.h5',
        custom_objects={'mse': MeanSquaredError()},
    )

    # Load stock prices
    stock_prices = stock_close_prices(stock_name)
    trading_period = len(stock_prices) - 1
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
    action_counts = {0: 0, 1: 0, 2: 0}  # Initialize counters for Hold, Buy, Sell

    for t in range(1, trading_period + 1):
        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)

        action_counts[action] += 1
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        # Execute position
        logging.info(f'Step: {t}')
        if action != np.argmax(actions):
            logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0:  # Hold
            current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
            hold(t, current_portfolio_value, previous_portfolio_value)  # Call the modified hold function
        if action == 1 and agent.balance > stock_prices[t]:
            buy(t)  # Buy
        if action == 2 and len(agent.inventory) > 0:
            sell(t)  # Sell

        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        agent.portfolio_values.append(current_portfolio_value)
        state = next_state

        done = True if t == trading_period else False
        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            logging.info("Action Distribution:")
            logging.info(f"Hold: {action_counts[0]} times")
            logging.info(f"Buy: {action_counts[1]} times")
            logging.info(f"Sell: {action_counts[2]} times")

if display:
    plot_all(stock_name, agent)


# import argparse
# import importlib
# import logging
# import sys
#
# import numpy as np
# # np.random.seed(3)  # for reproducible Keras operations
#
# from utils1 import *
#
#
# parser = argparse.ArgumentParser(description='command line options')
# parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DQN_ep10', help="model name")
# parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2018', help="stock name")
# parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
# inputs = parser.parse_args()
#
# model_to_load = inputs.model_to_load
# model_name = model_to_load.split('_')[0]
# stock_name = inputs.stock_name
# initial_balance = inputs.initial_balance
# display = True
# window_size = 10
# action_dict = {0: 'Hold', 1: 'Hold', 2: 'Sell'}
#
# # select evaluation model
# model = importlib.import_module(f'agents.{model_name}')
#
# def hold():
#     logging.info('Hold')
#
# def buy(t):
#     agent.balance -= stock_prices[t]
#     agent.inventory.append(stock_prices[t])
#     agent.buy_dates.append(t)
#     logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))
#
# def sell(t):
#     agent.balance += stock_prices[t]
#     bought_price = agent.inventory.pop(0)
#     profit = stock_prices[t] - bought_price
#     global reward
#     reward = profit
#     agent.sell_dates.append(t)
#     logging.info('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))
#
# # configure logging
# logging.basicConfig(filename=f'logs/{model_name}_evaluation_{stock_name}.log', filemode='w',
#                     format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
#
# portfolio_return = 0
# while portfolio_return == 0: # a hack to avoid stationary case
#     agent = model.Agent(state_dim=13, balance=initial_balance, is_eval=True, model_name=model_to_load)
#
#     stock_prices = stock_close_prices(stock_name)
#     trading_period = len(stock_prices) - 1
#     state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
#
#     for t in range(1, trading_period + 1):
#         if model_name == 'DDPG':
#             actions = agent.act(state, t)
#             action = np.argmax(actions)
#         else:
#             actions = agent.model.predict(state)[0]
#             action = agent.act(state)
#
#         # print('actions:', actions)
#         # print('chosen action:', action)
#
#         next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
#         previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
#
#         # execute position
#         logging.info(f'Step: {t}')
#         if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
#         if action == 0: hold() # hold
#         if action == 1 and agent.balance > stock_prices[t]: buy(t) # buy
#         if action == 2 and len(agent.inventory) > 0: sell(t) # sell
#
#         current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
#         agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
#         agent.portfolio_values.append(current_portfolio_value)
#         state = next_state
#
#         done = True if t == trading_period else False
#         if done:
#             portfolio_return = evaluate_portfolio_performance(agent, logging)
#
# if display:
#     # plot_portfolio_transaction_history(stock_name, agent)
#     # plot_portfolio_performance_comparison(stock_name, agent)
#     plot_all(stock_name, agent)
