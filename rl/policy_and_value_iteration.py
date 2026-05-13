import numpy as np


def policy_evaluation(pi, P, gamma, theta):
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):
            # action from policy pi is pi(s)
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            
        if np.max(np.abs(prev_V - V)) < theta:
            break
    
        prev_V = V.copy()
    return V
                

def policy_improvement(V, P, gamma):
    """
    len(P) - is number if of state
    len(P[0]) - is number of actions (in each state the same)
    """
    
    Q = np.zeros(len(P), len(P[0]), dtype=np.float64)

    for s in range(len(P)): # iterate over state
        for a in range(len(P[s])): # iterate over action
            for prob, next_state, reward, done in P[s][a]:
                # receive probablity of transition into next_state, reward and flag if it terminal
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    
    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    # new policy is a function which return argmax Q function on actions for any state

    return new_pi


def policy_iteration(P, gamma, theta):
    """
    P - environment, for each state (s) we have actions A: {a_1, a_2, .. a_n}
    gamma - discount factor
    theta - threshold for convergence
    """

    action_names = tuple(P[0].keys())
    random_actions = np.random.choice(action_names, len(action_names))

    # create policy, func that return action on state
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]

    while True:
        old_pi = {s: pi(s) for s in range(len(P))}

        V = policy_evaluation(pi, P, gamma, theta)

        pi = policy_improvement(V, P, gamma)

        if old_pi == {s:pi(s) for s in range(len(P))}:
            # no change in policy -> exit
            break
    return V, pi

def value_iteration(P, gamma, theta):
    V = np.zeros(len(P), len(P[0]), dtype=np.float64)

    while True:
        Q = np.zeros(len(P), len(P[0]), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        V = np.max(Q, axis=1) # V^* = max Q^* by action

    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi

# TODO write global intution on difference between policy iteration and value iteration