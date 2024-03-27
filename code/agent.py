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
        self.name = "Temporal Difference Learning"
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
on-policy monte carlo agent
"""
class MonteCarloAgent(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.discount_factor = 0.99
        self.action_counts = np.zeros(self.nr_actions)
        self.return_values = {}
        self.g = 0
        self.name = "Monte Carlo"
    def update(self, state, action, reward, next_state, terminated, truncated):
        # append discounted return to Returns(s,a)
        # update Q(s,a) = average(Returns(s,a))
        self.decay_exploration()
        self.g = self.discount_factor * self.g + reward

        state_key = np.array2string(state)
        if state_key not in self.return_values:
            self.return_values[state_key] = {action:[] for action in range(self.nr_actions)}
        self.return_values[state_key][action].append(self.g)
        # print("return values: ", self.return_values[state_key][action])
        # print(np.mean(self.return_values[state_key][action]))
        self.Q(state)[action] = np.mean(self.return_values[state_key][action])
        self.action_counts[action] += 1

class OffpolicyMonteCarloAgent(MonteCarloAgent):
    def __init__(self, params):
        super().__init__(params)
        self.behavior_policy = self.behavior_policy
        self.w = 1
        self.g = 0
        self.C_values = {}
        self.epsilon = 0.1
        self.name = "Off-policy Monte Carlo"
    def behavior_policy(self, state):
        #epsilon greedy policy
        return epsilon_greedy(self.Q(state), self.action_counts,epsilon=self.epsilon)
        #random policy
        #return random_bandit(self.Q(state), self.action_counts)
    def C(self,state):
        state_key = np.array2string(state)
        if state_key not in self.C_values:
            self.C_values[state_key] = np.zeros(self.nr_actions)
        return self.C_values[state_key]
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.g = self.discount_factor * self.g + reward
        self.C(state)[action] += self.w
        self.Q(state)[action] += (self.w/self.C(state)[action])*(self.g-self.Q(state)[action])
        self.action_counts[action] += 1
    def policy(self, state):
        #greedy policy
        return epsilon_greedy(self.Q(state), self.action_counts, epsilon=0.0)
    
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
        self.name = "SARSA(lambda)"
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
        action = UCB1(Q_values, self.action_counts, exploration_constant=1.2)
        self.action_counts[action] += 1
        return action

"""
    Autonomous agent using on-policy SARSA with Boltzmann exploration.
"""
class SARSALambda_Boltzmann_learner(SARSALambdaLearner):

    def policy(self, state):
        Q_values = self.Q(state)
        return boltzmann(Q_values, None, temperature=0.5)

"""
 Autonomous agent using off-policy Q-Learning.
"""
class QLearner(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.name = "Q-Learning"
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error

"""
    Autonomous agent using Dyna-Q.
"""

class DynaQLearner(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.model = {}  # Dictionary to store the transition model
        self.planning_steps = params["planning_steps"]
        self.name = "Dyna-Q"
    def update_model(self, state, action, reward, next_state, terminated, truncated):
        state_key = np.array2string(state)
        next_state_key = np.array2string(next_state)
        if state_key not in self.model:
            self.model[state_key] = {}
        if action not in self.model[state_key]:
            self.model[state_key][action] = []
        self.model[state_key][action].append((reward, next_state_key, terminated, truncated))

    def planning_step(self):
        # Sample a state and action from the model
        state_key = random.choice(list(self.model.keys()))
        state = np.fromstring(state_key, sep='\n')
        action = random.choice(list(self.model[state_key].keys()))

        # Sample a transition from the model
        reward, next_state_key, terminated, truncated = random.choice(self.model[state_key][action])
        next_state = np.fromstring(next_state_key, sep='\n')

        # Update Q-values based on the sampled transition
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma * Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha * TD_error

    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        self.update_model(state, action, reward, next_state, terminated, truncated)

        # Perform Q-learning update
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma * Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha * TD_error

        # Perform planning steps
        for _ in range(self.planning_steps):
            self.planning_step()

class DynaQLearnerWithEligibilityTraces(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.model = {}  # Dictionary to store the transition model
        self.planning_steps = params["planning_steps"]
        self.lambda_ = params['lambda']
        self.E = {}  # Use a dictionary for eligibility traces, similar to Q-values

    def update_model(self, state, action, reward, next_state, terminated, truncated):
        state_key = np.array2string(state)
        next_state_key = np.array2string(next_state)
        if state_key not in self.model:
            self.model[state_key] = {}
        if action not in self.model[state_key]:
            self.model[state_key][action] = []
        self.model[state_key][action].append((reward, next_state_key, terminated, truncated))

    def update_eligibility_traces(self, state, action):
        state_key = np.array2string(state)
        if state_key not in self.E:
            self.E[state_key] = np.zeros(self.nr_actions)
        self.E[state_key][action] += 1  # Increment eligibility for the current state-action pair

    def decay_eligibility_traces(self):
        # Decay eligibility traces for all states
        for state_key in self.E:
            self.E[state_key] *= self.gamma * self.lambda_

    def planning_step(self):
        # Sample a state and action from the model
        state_key = random.choice(list(self.model.keys()))
        state = np.fromstring(state_key, sep='\n')
        action = random.choice(list(self.model[state_key].keys()))

        # Sample a transition from the model
        reward, next_state_key, terminated, truncated = random.choice(self.model[state_key][action])
        next_state = np.fromstring(next_state_key, sep='\n')

        # Update eligibility trace for the current state-action pair
        self.update_eligibility_traces(state, action)

        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma * Q_next

        TD_error = TD_target - Q_old

        # Update Q-values for all states based on their eligibility traces
        for state_key in self.E:
            for action_index in range(self.nr_actions):  # Use range for action indices
                self.Q_values[state_key][action_index] += self.alpha * TD_error * self.E[state_key][action_index]

        # Decay eligibility traces for all states
        self.decay_eligibility_traces()

    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        self.update_model(state, action, reward, next_state, terminated, truncated)

        # Update eligibility trace for the current state-action pair
        self.update_eligibility_traces(state, action)

        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma * Q_next

        TD_error = TD_target - Q_old

        # Update Q-values for all states based on their eligibility traces
        for state_key in self.E:
            for action_index in range(self.nr_actions):  # Use range for action indices
                self.Q_values[state_key][action_index] += self.alpha * TD_error * self.E[state_key][action_index]

        # Decay eligibility traces for all states
        self.decay_eligibility_traces()

        # Perform planning steps
        for _ in range(self.planning_steps):
            self.planning_step()

        if terminated:
            # Reset eligibility traces after each episode
            self.E = {}