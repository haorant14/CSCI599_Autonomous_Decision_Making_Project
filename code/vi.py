import numpy as np

def value_iteration(env, max_iterations=500, delta=0.001, gamma=0.9):
    """
    Value Iteration algorithm.
    :param env: grid world environment
    :param max_iterations: maximum iterations to run value iteration for
    :param delta: threshold for change in values to check for convergence
    :param gamma: discount factor

    Store the values of each state in the dict vf
    Store the optimal actions for each state in the dict op_actions

    env.available_actions(state): List of actions that can be executed from that state
    env.possible_transitions(state, available_action): List of resulting state tuples after executing available action.
                                                  Each tuple = (next state, probability of transitioning to that state)
    env.reward(state): Reward received in the state.
    """
    valid_states = [s for s in env.int_states if env.int_to_feature(s).tolist() not in env.obstacles.tolist()]
    vf = {s: 0 for s in valid_states}  # values
    op_actions = {s: 0 for s in valid_states}  # optimal actions
     # ------------------------------------------- FILL YOUR CODE HERE ------------------------------------------------ #
    for s in vf.keys():
        print("available actions for state", s,  env.available_actions(s) ) 
 
    iter = 0
    while iter < max_iterations:
        de = 0
        for state in vf.keys():
            state_value = vf[state]
            
            # available actions: 
            actions = env.available_actions(state)
            # transition probability: 
            max_action_return  = 0
            for action in actions:  
                transitions = env.possible_transitions(state,action)
                action_total_return = 0
                for next_state, transition_prob in transitions: 

                    reward = env.reward(next_state)
                    reward_with_next_state_value = reward + gamma * vf[next_state]
                    action_return = transition_prob * reward_with_next_state_value
                    action_total_return += action_return
                if action_total_return > max_action_return:
                    op_actions[state] = action
                    max_action_return =  action_total_return 
            vf[state] = max_action_return
            de = max(de,abs(vf[state]-state_value))
        if de < delta:  
            break
        iter+=1 

    print("state value function table: ", vf)
    print("optimal action for each state: ", op_actions)
    # ---------------------------------------------------------------------------------------------------------------- #

    return vf, op_actions
