import argparse
import importlib
import logging
import sys
import time

from utils1 import *

parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DDQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int,
                    help="span (days) of observation") 
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
                    help='initial balance')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select learning model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)


def hold(actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[next_probable_action] = 1  # reset this action's value to the highest
            return 'Hold', actions


def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])


def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)


# configure logging
logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

start_time = time.time()
for e in range(1, num_episode + 1):
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset()  # reset to initial balance and hyperparameters
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)

        # execute position
        logging.info(
            'Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1],
                                                                                           actions[2]))
        if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0:  # hold
            execution_result = hold(actions)
        if action == 1:  # buy
            execution_result = buy(t)
        if action == 2:  # sell
            execution_result = sell(t)

            # check execution result
        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
        else:
            if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]
            logging.info(execution_result)

            # calculate reward

        # Reward logic integration
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value  # Base reward
        reward *= 100  # Scale reward for gradient sensitivity

        # Action-specific rewards/penalties
        if action == 0:  # Hold
            if stock_prices[t] > stock_prices[t - 1]:  # Missed profit opportunity
                reward -= 0.1 * (stock_prices[t] - stock_prices[t - 1])
            elif len(agent.inventory) > 0 and stock_prices[t] < stock_prices[t - 1]:  # Penalize holding in downtrend
                reward -= 0.1 * len(agent.inventory) * (stock_prices[t - 1] - stock_prices[t])

        elif action == 1:  # Buy
            if agent.balance > stock_prices[t]:  # Reward for buying within balance
                if stock_prices[t] < np.mean(stock_prices[max(0, t - 10):t]):  # Buying below recent average
                    reward += 0.2 * (np.mean(stock_prices[max(0, t - 10):t]) - stock_prices[t])
            else:  # Penalize buying with insufficient balance
                reward -= 0.5

        elif action == 2:  # Sell
            if len(agent.inventory) > 0:  # Reward for profit from selling
                profit = stock_prices[t] - min(agent.inventory)  # Profit per share
                reward += 0.2 * profit
                if profit > 0:
                    reward += 0.1 * len(agent.inventory)  # Encourage selling when profitable
                else:  # Penalize selling at a loss
                    reward -= 0.1 * abs(profit)
            else:  # Penalize selling with no inventory
                reward -= 0.2

        # Risk Management: Penalize large inventory during downtrends
        if len(agent.inventory) > 0:
            recent_trend = np.mean(stock_prices[max(0, t - 5):t]) - stock_prices[t]
            if recent_trend > 0:  # Downtrend
                reward -= 0.1 * len(agent.inventory) * recent_trend

        # Encourage maintaining a balanced portfolio
        portfolio_ratio = agent.balance / (agent.balance + sum(agent.inventory) * stock_prices[t])
        if 0.4 <= portfolio_ratio <= 0.6:  # Balanced portfolio
            reward += 0.1
        else:  # Over-exposed or under-utilized balance
            reward -= 0.1

        # Append reward to replay memory
        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)

        # current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        # unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        # reward += unrealized_profit
        #
        # agent.portfolio_values.append(current_portfolio_value)
        # agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        #
        # done = True if t == trading_period else False
        # agent.remember(state, actions, reward, next_state, done)
        #
        # update state
        state = next_state

        # # experience replay
        # if len(agent.memory) > agent.buffer_size:
        #     num_experience_replay += 1
        #     loss = agent.experience_replay()
        #     logging.info(
        #         'Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e,
        #                                                                                                               loss,
        #                                                                                                               action_dict[
        #                                                                                                                   action],
        #                                                                                                               reward,
        #                                                                                                               agent.balance,
        #                                                                                                               len(agent.inventory)))
        #     agent.tensorboard.on_batch_end(num_experience_replay,
        #                                    {'loss': loss, 'portfolio value': current_portfolio_value})
        #
        # if done:
        #     portfolio_return = evaluate_portfolio_performance(agent, logging)
        #     returns_across_episodes.append(portfolio_return)

        # Experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            loss = agent.experience_replay()
            logging.info(
                'Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e,
                                                                                                                      loss,
                                                                                                                      action_dict[
                                                                                                                          action],
                                                                                                                      reward,
                                                                                                                      agent.balance,
                                                                                                                      len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay,
                                           {'loss': loss, 'portfolio value': current_portfolio_value})

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
