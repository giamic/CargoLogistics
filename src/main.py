import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from utils import create_goal_buffer, find_available_containers, find_needed_containers, find_full_stacks, \
    assign_reward, find_new_state, softmax, initialize_random_state, visualize_state, initialize_even_state

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.8
T = 1
EPISODES = 5_000
REPETITIONS = 1
ROWS = 6  # the number of rows in each bay and in the buffer
COLUMNS = 6  # the number of columns in each bay and in the buffer
BAYS = 2  # I assume containers con move from one bay to the other during removal
N_CONTAINERS = 36
TOTAL_STACKS = BAYS * COLUMNS + 1
assert N_CONTAINERS <= ROWS * COLUMNS
B_GOAL = create_goal_buffer(ROWS, COLUMNS)

# This is the state as taken from the paper, once the containers are renamed
X = np.array([13, 18, 4, 17, 8, 10,
              21, 3, 32, 12, 26, 24,
              5, 28, 22, 33, 20, 30,
              35, 31, 27, 19, 2, 7,
              0, 34, 11, 1, 15, 9,
              29, 23, 31, 6, 16, 14])


def q(state, state_string, container=None, target_stack=None):
    """
    This function creates all the needed q_table values and returns their current status.

    :param state_string:
    :param container:
    :param target_stack: -1 if you want to put it in the buffer
    :return: the q-value for the current state and action "move container c to target_stack s";
        if the container is not specified, return sup_s Q(x, c, s) for all c
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
                else:  # otherwise set the value to 1 (this is most probably a good move!)
                    q_table[state_string][c, -1] = 1

    if container is None:
        return np.max(q_table[state_string], axis=1)

    if target_stack is None:
        return q_table[state_string][container]

    return q_table[state_string][container, target_stack]


steps_to_average = []
for rep in range(REPETITIONS):
    print("Repetition {} of {}".format(rep + 1, REPETITIONS))
    steps = []
    q_table = dict()
    # initialize the value of the terminal state to a high value
    q_table[(np.ones(N_CONTAINERS, dtype=int) * (-1)).tostring()] = np.ones((N_CONTAINERS, TOTAL_STACKS)) * 10
    for e in range(EPISODES):
        if e % 100 == 0:
            print("Episode {} of {}".format(e + 1, EPISODES))
        n = 0
        x0 = initialize_random_state(N_CONTAINERS, ROWS, COLUMNS, BAYS)
        # x0 = X.copy()
        xs0 = x0.tostring()
        x1 = x0.copy()
        # visualize_state(x, ROWS, COLUMNS, BAYS)
        while not np.all(x1 == -1):
            c0 = np.random.choice(N_CONTAINERS, p=(softmax(q(x0, xs0), T)))
            s0 = np.random.choice(TOTAL_STACKS, p=(softmax(q(x0, xs0, c0), T)))
            if s0 != TOTAL_STACKS - 1:
                n += 1
            x1 = find_new_state(x0, c0, s0, TOTAL_STACKS)
            xs1 = x1.tostring()
            reward = assign_reward(s0, TOTAL_STACKS, ROWS)
            delta = reward + DISCOUNT_FACTOR * np.max(q(x1, xs1)) - q_table[xs0][c0, s0]
            if delta == -np.inf:
                raise ValueError('infinite change in delta, something wrong')
            q_table[xs0][c0, s0] += LEARNING_RATE * delta
            x0, xs0 = x1, xs1
            if e == 3999:
                print(x0, c0, s0)

        steps.append(n)
    print("=========\n")
    steps_to_average.append(steps)

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
plt.plot(np.average(np.array(steps_to_average), axis=0) / N_CONTAINERS, label='our_method')
# plt.axhline(27, color='k', linestyle='--', label='baseline')  # simulated elsewhere
plt.xlabel('number of trials')
plt.ylabel('average number of removals per container')
plt.title('average results over {} repetitions'.format(REPETITIONS))
plt.savefig(os.path.join('..', 'data', 'plot_' + datetime.now().strftime('%F_%R') + '.pdf'))
plt.show()
