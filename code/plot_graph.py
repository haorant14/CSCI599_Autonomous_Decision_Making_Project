
import numpy as np
import pickle
import sys
import agent as a
import matplotlib.pyplot as plot
import rooms
def loadData(agentmame, rooms_instance):
    # for reading also binary mode is important
    file = open(f"qtable/{agentmame}_{rooms_instance}.pkl", 'rb')
    q_table = pickle.load(file)
    # for keys in q_table:
    #     print(keys, '=>', q_table[keys])
    # print(f"loaded {agentmame}_{rooms_instance}.pkl from qtable folder")
    file.close()
    return q_table


# start the agent with the learned q table
def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.997
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return

# agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
# agent = a.TemporalDifferenceLearningAgent(params)
# agent = a.QLearner(params)

rooms_instance = "hard_1"
"""
plot graph for medium_0
"""
params = {}
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.997
# params["epsilon_decay"] = 0.001
# params["alpha"] = 0.1
params["env"] = env
# params['lambda'] = 0.5
# params['planning_steps'] = 50

dynaq_greedy_agent = a.GreedyEvaluateAgent(params)
sarsa_lambda_greedy_agent = a.GreedyEvaluateAgent(params)
q_learning_greedy_agent = a.GreedyEvaluateAgent(params)
sarsa_greedy_agent = a.GreedyEvaluateAgent(params)
# montecarlo_greedy_agent = a.GreedyEvaluateAgent(params)
# offpolicy_montecarlo_greedy_agent = a.GreedyEvaluateAgent(params)

dyna_q_qtable = loadData("Dyna-Q", rooms_instance)
sarsa_lambda_qtable = loadData("SARSAlambda", rooms_instance)
q_learning_qtable = loadData("Q-Learning", rooms_instance)
sarsa_qtable = loadData("SARSA", rooms_instance)
# montecarlo_qtable = loadData("onpolicy_MonteCarlo", rooms_instance)
# offpolicy_montecarlo_qtable = loadData("offpolicy_MonteCarlo", rooms_instance)


dynaq_greedy_agent.Q_values = dyna_q_qtable
sarsa_lambda_greedy_agent.Q_values = sarsa_lambda_qtable
q_learning_greedy_agent.Q_values = q_learning_qtable
sarsa_greedy_agent.Q_values = sarsa_qtable
# montecarlo_greedy_agent.Q_values = montecarlo_qtable
# offpolicy_montecarlo_greedy_agent.Q_values = offpolicy_montecarlo_qtable

# agent = a.OffpolicyMonteCarloAgent(params)
testing_episodes = 100
dynaq_returns = [episode(env, dynaq_greedy_agent, i) for i in range(testing_episodes)]
sarsa_lambda_returns = [episode(env, sarsa_lambda_greedy_agent, i) for i in range(testing_episodes)]
q_learning_returns = [episode(env, q_learning_greedy_agent, i) for i in range(testing_episodes)]
sarsa_returns = [episode(env, sarsa_greedy_agent, i) for i in range(testing_episodes)]
# montecarlo_returns = [episode(env, montecarlo_greedy_agent, i) for i in range(testing_episodes)]
# offpolicy_montecarlo_returns = [episode(env, offpolicy_montecarlo_greedy_agent, i) for i in range(testing_episodes)]

x = range(testing_episodes)
plot.plot(x, dynaq_returns, label="Dyna-Q")
plot.plot(x, sarsa_lambda_returns, label="SARSA Lambda")
plot.plot(x, q_learning_returns, label="Q Learning")
plot.plot(x, sarsa_returns, label="SARSA")
# plot.plot(x, montecarlo_returns, label="On-policy Monte Carlo")
# plot.plot(x, offpolicy_montecarlo_returns, label="Off-policy Monte Carlo")

plot.title(f"validating discounted Return on {rooms_instance}")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.legend()
plot.show()


