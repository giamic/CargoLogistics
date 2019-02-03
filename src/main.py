from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from datetime import datetime
from utils import create_goal_buffer, find_available_containers, find_needed_containers, find_full_stacks, \
    assign_reward, find_new_state, softmax, initialize_random_state, visualize_state, initialize_even_state

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.8
T = 0.1
EPISODES = 1000
ROWS = 4  # the number of rows in each bay and in the buffer
COLUMNS = 4  # the number of columns in each bay and in the buffer
BAYS = 2  # I assume containers con move from one bay to the other during removal
N_CONTAINERS = 16
TOTAL_STACKS = BAYS * COLUMNS + 1
assert N_CONTAINERS <= ROWS * COLUMNS
B_GOAL = create_goal_buffer(ROWS, COLUMNS)

q_table = dict()
X = initialize_even_state(N_CONTAINERS)


def q(state, state_string, container=None, target_stack=None):
    """
    This function creates all the needed q_table values and returns their current status.

    :param state_string:
    :param container:
    :param target_stack: -1 if you want to put it in the buffer
    :return: the q-value for the current state and action "move container c to target_stack s";
        if the container is not specified, return sup_t Q(x, c, s) for all c
        if the target_stack is not specified, return Q(x, c, s) for all s

    """
    if state_string not in q_table:  # create all the needed dictionaries
        q_table[state_string] = np.zeros((N_CONTAINERS, TOTAL_STACKS))
        ac = find_available_containers(state, TOTAL_STACKS)
        nc = find_needed_containers(state, ROWS, COLUMNS, B_GOAL)
        fs = find_full_stacks(state, ROWS, COLUMNS, BAYS)
        for c in range(N_CONTAINERS):
            if c not in ac:  # if the container is not available, prevent every move
                q_table[state_string][c] = np.ones(TOTAL_STACKS) * (-np.inf)
            else:
                os = state[c] % (TOTAL_STACKS - 1)  # origin stack
                q_table[state_string][c, os] = -np.inf  # prevent the choice: target = origin
                for ts in fs:  # if the target stack is full, prevent that move
                    q_table[state_string][c, ts] = -np.inf
                if c not in nc:  # if the container is not currently needed in the buffer, prevent that move
                    q_table[state_string][c, -1] = -np.inf

    if container is None:
        return np.max(q_table[state_string], axis=1)

    if target_stack is None:
        return q_table[state_string][container]

    return q_table[state_string][container, target_stack]


def update_q(x0, xs0, c0, s0, x1, xs1, c1, reward):
    q_table[xs0][c0, s0] = q(x0, xs0, c0, s0) + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(q(x1, xs1, c1)) - q(x0, xs0, c0, s0))
    return


steps = []
for e in range(EPISODES):
    if e % 100 == 0:
        print("Episode {} of {}".format(e + 1, EPISODES))
    n = 0
    x = X.copy()
    xs = str(x)
    # visualize_state(x, ROWS, COLUMNS, BAYS)

    c = np.random.choice(N_CONTAINERS, p=(softmax(q(x, xs), T)))
    s = np.random.choice(TOTAL_STACKS, p=(softmax(q(x, xs, c), T)))

    while not np.all(x, -1):
        if s != TOTAL_STACKS - 1:
            n += 1
        x1 = find_new_state(x, c, s, TOTAL_STACKS)
        xs1 = str(x1)
        reward = assign_reward(x1)
        c1 = np.random.choice(N_CONTAINERS, p=(softmax(q(x1, xs1), T)))
        s1 = np.random.choice(TOTAL_STACKS, p=(softmax(q(x1, xs1, c1), T)))
        update_q(x, xs, c, s, x1, xs1, c1, reward)
        x, xs, c, s = x1, xs1, c1, s1
    steps.append(n)

plt.plot(steps)
plt.show()
plt.plot(np.average(np.array(steps).reshape((-1, 100)), axis=1))
plt.show()

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# output_file = os.path.join('..', 'data', 'qtable_' + datetime.now().strftime('%F_%R') + '.pkl')
# with open(output_file, 'wb') as fp:
#     pickle.dump(q_table, fp)
