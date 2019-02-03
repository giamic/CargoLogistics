import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from datetime import datetime
from utils import create_goal_buffer, find_available_containers, find_needed_containers, find_full_stacks, \
    assign_reward, find_new_state, softmax, initialize_random_state

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.1
EPISODES = 10000
ROWS = 4  # the number of rows in each bay and in the buffer
COLUMNS = 4  # the number of columns in each bay and in the buffer
BAYS = 2  # I assume containers con move from one bay to the other during removal
N_CONTAINERS = 16
TOTAL_STACKS = BAYS * COLUMNS + 1
assert N_CONTAINERS <= ROWS * COLUMNS
B_GOAL = create_goal_buffer(ROWS, COLUMNS)

q_table = dict()


def q(state, container=None, target_stack=None):
    """
    This function creates all the needed q_table values and returns their current status.

    :param state:
    :param container:
    :param target_stack: -1 if you want to put it in the buffer
    :return: the q-value for the current state and action "move container c to target_stack s";
        if the container is not specified, return sup_t Q(x, c, s) for all c
        if the target_stack is not specified, return Q(x, c, s) for all s

    """
    xs = str(state)
    if xs not in q_table:  # create all the needed dictionaries
        q_table[xs] = dict()
        ac = find_available_containers(state, ROWS, COLUMNS, BAYS)
        nc = find_needed_containers(state, ROWS, COLUMNS, B_GOAL)
        fs = find_full_stacks(state, ROWS, COLUMNS, BAYS)
        for c in range(N_CONTAINERS):
            cs = str(c)
            if c not in ac:  # if the container is not available, prevent every move
                q_table[xs][cs] = np.ones(TOTAL_STACKS) * (-np.inf)
            else:
                q_table[xs][cs] = np.zeros(TOTAL_STACKS)
                os = state[c] % (TOTAL_STACKS - 1)  # origin stack
                q_table[xs][cs][os] = -np.inf  # prevent the choice: target = origin
                for ts in fs:  # if the target stack is full, prevent that move
                    q_table[xs][cs][ts] = -np.inf
                if c not in nc:  # if the container is not currently needed in the buffer, prevent that move
                    q_table[xs][cs][-1] = -np.inf

    if container is None:
        return np.array([np.max(q_table[xs][str(c)]) for c in range(N_CONTAINERS)])

    if target_stack is None:
        return q_table[xs][str(container)]

    return q_table[xs][str(container)][target_stack]


def update_q(x0, c0, s0, x1, c1, s1, reward):
    q_table[str(x0)][str(c0)][s0] = q_table[str(x0)][str(c0)][s0] + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * q_table[str(x1)][str(c1)][s1] - q_table[str(x0)][str(c0)][s0])
    return


steps = []
for e in range(EPISODES):
    if e % 100 == 0:
        print("Episode {} of {}".format(e+1, EPISODES))
    n = 0
    x = initialize_random_state(N_CONTAINERS, ROWS, COLUMNS, BAYS)
    # visualize_state(x, ROWS, COLUMNS, BAYS)

    c = np.random.choice(N_CONTAINERS, p=(softmax(q(x))))
    s = np.random.choice(TOTAL_STACKS, p=(softmax(q(x, c))))

    while not np.all(x, -1):
        if s != TOTAL_STACKS - 1:
            n += 1
        x1 = find_new_state(x, c, s, TOTAL_STACKS)
        reward = assign_reward(x1)
        c1 = np.random.choice(N_CONTAINERS, p=(softmax(q(x1))))
        s1 = np.random.choice(TOTAL_STACKS, p=(softmax(q(x1, c1))))
        update_q(x, c, s, x1, c1, s1, reward)
        x, c, s = x1, c1, s1
    steps.append(n)

plt.plot(steps)
plt.show()

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# output_file = os.path.join('..', 'data', 'qtable_' + datetime.now().strftime('%F_%R') + '.pkl')
# with open(output_file, 'wb') as fp:
#     pickle.dump(q_table, fp)
