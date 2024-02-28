import random
import numpy as np
from multi_armed_bandits import *

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, terminated, truncated):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent base for learning Q-values.
"""
class TemporalDifferenceLearningAgent(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon = 1.0
        self.action_counts = np.zeros(self.nr_actions)
    def Q(self, state):
        state = np.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        Q_values = self.Q(state)
        return epsilon_greedy(Q_values, None, epsilon=self.epsilon)
    
    def decay_exploration(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_decay)

"""
 Autonomous agent using on-policy SARSA with epsillon decay.
"""
class SARSALearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            next_action = self.policy(next_state)
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error

"""
 Autonomous agent using on-policy SARSA lambda.
"""
class SARSALambdaLearner(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.lambda_ = params['lambda']
        self.E = {}  # Use a dictionary for eligibility traces, similar to Q-values
        
    def update_eligibility_traces(self, state, action):
        state_key = np.array2string(state)
        if state_key not in self.E:
            self.E[state_key] = np.zeros(self.nr_actions)
        self.E[state_key][action] += 1  # Increment eligibility for the current state-action pair
    
    def decay_eligibility_traces(self):
        # Decay eligibility traces for all states
        for state_key in self.E:
            self.E[state_key] *= self.gamma * self.lambda_

    
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]

        # Update eligibility trace for the current state-action pair
        self.update_eligibility_traces(state, action)
        
        TD_target = reward
        if not terminated:
            next_action = self.policy(next_state)
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma * Q_next
        
        TD_error = TD_target - Q_old
        
        # Update Q-values for all states based on their eligibility traces
        for state_key in self.E:
            for action_index in range(self.nr_actions):  # Use range for action indices
                self.Q_values[state_key][action_index] += self.alpha * TD_error * self.E[state_key][action_index]
        
        # Decay eligibility traces for all states
        self.decay_eligibility_traces()
                    
        if terminated:
            # Reset eligibility traces after each episode
            self.E = {}


"""
 Autonomous agent using on-policy SARSA with UCB exploration.
"""
class SARSALambda_UCB_learner(SARSALambdaLearner):
    def policy(self, state):
        Q_values = self.Q(state)
        action = UCB1(Q_values, self.action_counts, exploration_constant=1)
        self.action_counts[action] += 1
        return action

"""
    Autonomous agent using on-policy SARSA with Boltzmann exploration.
"""
class SARSALambda_Boltzmann_learner(SARSALambdaLearner):
    def policy(self, state):
        Q_values = self.Q(state)
        return boltzmann(Q_values, None, temperature=1.0)

"""
 Autonomous agent using off-policy Q-Learning.
"""
class QLearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error