import argparse
import importlib
import logging
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from utils1 import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained DQN model')
parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DQN_ep10', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2018', help="stock name")
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
                    help='initial balance')
inputs = parser.parse_args()

# Parameters
model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
stock_name = inputs.stock_name
initial_balance = inputs.initial_balance
display = True
window_size = 10
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# Load the DQN agent
model = importlib.import_module(f'agents.{model_name}')

def hold(t, current_portfolio_value, previous_portfolio_value):
    """
    Reward adjustment for the 'Hold' action.
    """
    unrealized_profit = current_portfolio_value - previous_portfolio_value
    opportunity_cost = treasury_bond_daily_return_rate() * agent.balance
    global reward
    reward += unrealized_profit - opportunity_cost
    logging.info(f"Hold: Unrealized Profit: ${unrealized_profit:.2f}, Opportunity Cost: ${opportunity_cost:.2f}")

def buy(t):
    """
    Execute a 'Buy' action.
    """
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        agent.buy_dates.append(t)
        logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))

def sell(t):
    """
    Execute a 'Sell' action.
    """
    if len(agent.inventory) > 0:
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

# Portfolio return initialization
portfolio_return = 0
while portfolio_return == 0:  # Avoid stationary case
    agent = model.Agent(state_dim=13, balance=initial_balance, is_eval=True, model_name=model_to_load)

    # Load the model with the custom loss function
    agent.model = load_model(
        f'saved_models/{model_to_load}.h5',
        custom_objects={'mse': MeanSquaredError()},
    )

    # Load stock prices
    stock_prices = stock_close_prices(stock_name)
    trading_period = len(stock_prices) - 1
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
    action_counts = {0: 0, 1: 0, 2: 0}  # Count actions: Hold, Buy, Sell

    for t in range(1, trading_period + 1):
        actions = agent.model.predict(state)[0]
        action = agent.act(state)

        action_counts[action] += 1
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        # Execute the action
        logging.info(f'Step: {t}')
        if action != np.argmax(actions):
            logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0:  # Hold
            current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
            hold(t, current_portfolio_value, previous_portfolio_value)
        elif action == 1 and agent.balance > stock_prices[t]:  # Buy
            buy(t)
        elif action == 2 and len(agent.inventory) > 0:  # Sell
            sell(t)

        # Update portfolio value and returns
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        agent.portfolio_values.append(current_portfolio_value)
        state = next_state

        # End of trading period
        done = True if t == trading_period else False
        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            logging.info("Action Distribution:")
            logging.info(f"Hold: {action_counts[0]} times")
            logging.info(f"Buy: {action_counts[1]} times")
            logging.info(f"Sell: {action_counts[2]} times")

# Visualization
if display:
    plot_all(stock_name, agent)
