import matplotlib.pyplot as plt
import numpy as np


def create_goal_buffer(rows, columns):
    return [set(range(r * columns, (r + 1) * columns)) for r in range(rows)]


def initialize_even_state(n):
    return np.random.permutation(n)


def initialize_random_state(n, rows, columns, bays):
    slots = rows * columns * bays
    x = np.random.choice(slots, size=n, replace=False)

    def _gravity_fall(x, total_columns):
        for col in range(total_columns):  # column by column, bring all containers to the bottom
            mask, = np.where(np.logical_and(x % total_columns == col, x >= 0))  # select the containers in that column
            if len(mask) == 0:
                continue
            order = mask[np.argsort(x[mask])]  # order them from bottom to top
            for end_row, i in enumerate(order):  # let them fall to the ground
                x[i] = col + end_row * total_columns
        return x

    return _gravity_fall(x, columns * bays)


def visualize_state(state, rows, columns, bays):
    total_columns = columns * bays
    grid = np.zeros((rows, total_columns))
    for c in state:
        if c >= 0:
            i, j = c // total_columns, c % total_columns
            grid[i, j] = 1
    plt.pcolormesh(grid, cmap='gray_r')
    plt.show()
    return


def find_available_containers(state, total_stacks):
    available_containers = set()
    for s in range(total_stacks - 1):  # -1 to remove the ship stack
        # select the containers in stack s
        mask, = np.where(np.logical_and(state % (total_stacks - 1) == s, state >= 0))
        if len(mask) == 0:  # if there are none, go to the next stack
            continue
        order = mask[np.argsort(state[mask])]  # order them from bottom to top
        available_containers.add(order[-1])  # add the one on the top of the stack to the available containers
    return available_containers


def find_needed_containers(state, rows, columns, b_goal):
    buffered_containers, = np.where(state < 0)
    needed_containers = set()
    floor = columns
    for r in range(rows):
        mask, = np.where(np.logical_and(r * columns <= buffered_containers, buffered_containers < (r + 1) * columns))
        if len(mask) < floor:  # check if there is enough space in this level for a new container
            temp = set(buffered_containers[mask])  # containers in this group
            needed_containers = needed_containers | (b_goal[r] - temp)  # set difference
        elif len(mask) > floor:
            raise AssertionError('The algorithm has tried to conquer gravity by putting a floating container!')
        floor = len(mask)
    return needed_containers


def find_full_stacks(state, rows, columns, bays):
    full_stacks = set()
    total_columns = columns * bays
    for c in range(total_columns):
        mask, = np.where(np.logical_and(state % total_columns == c, state >= 0))  # select the containers in that column
        if len(mask) == rows:
            full_stacks.add(c)
    return full_stacks


def find_destination(state, target_stack, total_stacks):
    if target_stack == total_stacks - 1:
        return -1
    # select the containers in the target_stack
    mask, = np.where(np.logical_and(state % total_stacks == target_stack, state >= 0))
    height = len(mask)  # current height of the target_stack, before putting the new container
    return height * (total_stacks - 1) + target_stack  # -1 because the ship stack needs to be removed


def find_new_state(previous_state, container, target_stack, total_stacks):
    next_state = previous_state.copy()
    next_state[container] = find_destination(previous_state, target_stack, total_stacks)
    return next_state


def softmax(x, temperature):
    y = (x - np.max(x)) / temperature
    return np.exp(y) / np.sum(np.exp(y), axis=0)


def assign_reward(state):
    if np.all(state) == -1:
        return 1
    return 0
