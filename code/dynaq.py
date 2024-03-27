import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import numpy as np
import pickle
def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return

# every visit
def monte_carlo_episode(env, agent, nr_episode=0):
    state = env.reset()
    done = False
    time_step = 0
    #generate episode
    episode = []
    agent.g = 0

    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
        time_step += 1
    #update value function
    t = time_step - 1
    for t in range(time_step-1, -1, -1):
        state, action, reward = episode[t]
        agent.update(state, action, reward, next_state, terminated, truncated)
    # 3. Integrate new experience into agent
    print(nr_episode, ":", agent.g)
    return agent.g

def off_policy_monte_carlo_episode(env, agent, nr_episode=0):
    state = env.reset()
    done = False
    time_step = 0
    #generate episode
    episode = []
    while not done:
        # 1. Select action according to policy
        action = agent.behavior_policy(state)
        
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
        time_step += 1
    #update value function
    t = time_step - 1
    agent.g = 0
    agent.w = 1 
    for t in range(time_step-1, -1, -1):
        state, action, reward = episode[t]
        agent.update(state, action, reward, next_state, terminated, truncated)
        if agent.policy(state) != agent.behavior_policy(state):
            break
        agent.w *= 1/0.25
    # 3. Integrate new experience into agent
    print(nr_episode, ":", agent.g)
    return agent.g

def store_qtable(qtable, params):     
    # Its important to use binary mode
    # store at qtable folder

    dbfile = open(f"qtable/{agent.name}_{sys.argv[1]}.pkl", 'ab')     
    # source, destination
    pickle.dump(qtable, dbfile)                    
    dbfile.close()

params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
print("Number of actions:", params["nr_actions"])
params["gamma"] = 0.99
params["epsilon_decay"] = 0.0001
params["alpha"] = 0.1
params["env"] = env
params['lambda'] = 0.5
params['planning_steps'] = 50
# agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
# agent = a.TemporalDifferenceLearningAgent(params)
# agent = a.QLearner(params)
agent = a.DynaQLearner(params)

# agent = a.OffpolicyMonteCarloAgent(params)
training_episodes = 500
returns = [episode(env, agent, i) for i in range(training_episodes)]
x = range(training_episodes)
plot.plot(x, returns, label="Dyna Q")

plot.title(f"Discounted Return for agent {agent.name} on {sys.argv[1]}")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.legend()
plot.show()
env.save_video()


## log the state action value function table and optimal action for each state
epsilon_decay = params["epsilon_decay"]
store_qtable(agent.Q_values, params)