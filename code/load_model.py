# load q table model from folder qtable and start the agent with the learned q table

import numpy as np
import os
import pickle
import sys
import agent as a
import matplotlib.pyplot as plot
import rooms
"""
    load q table model from folder qtable and start the agent with the learned q table
"""
def loadData(agentmame, rooms_instance):
    # for reading also binary mode is important
    file = open(f"qtable/{agentmame}_{rooms_instance}.pkl", 'rb')
    q_table = pickle.load(file)
    # for keys in q_table:
    #     print(keys, '=>', q_table[keys])
    # print(f"loaded {agentmame}_{rooms_instance}.pkl from qtable folder")
    file.close()
    return q_table

"""
    Run an episode with the greedy agent
"""
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

params = {}
rooms_instance = sys.argv[1]
agentmame = sys.argv[2]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.997
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["env"] = env
params['lambda'] = 0.5
params['planning_steps'] = 50
# agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
# agent = a.TemporalDifferenceLearningAgent(params)
# agent = a.QLearner(params)
agent = a.GreedyEvaluateAgent(params)
rooms_instance = sys.argv[1]
q_table = loadData(agentmame, rooms_instance)
agent.Q_values = q_table
# agent = a.OffpolicyMonteCarloAgent(params)
testing_episodes = 100

returns = [episode(env, agent, i) for i in range(testing_episodes)]
"""
    Plot the discounted return over episodes after testing a greedy agent
"""
x = range(testing_episodes)
plot.plot(x, returns, label="after training")

plot.title("greedy agent discounted Return over Episodes after training")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.legend()
plot.show()
env.save_video()
print("Average discounted return after training: ", np.mean(returns))


