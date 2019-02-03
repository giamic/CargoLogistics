import matplotlib.pyplot as plt
import numpy as np


def create_goal_buffer(rows, columns):
    return [set(range(r * columns, (r + 1) * columns)) for r in range(rows)]


def initialize_empty_buffer(rows):
    return [set() for _ in range(rows)]


def initialize_random_state(n, rows, columns, bays):
    slots = rows * columns * bays
    x = np.random.choice(range(slots), size=n, replace=False)

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


def find_open_groups(state, rows, columns):
    open_groups = []
    buffered_containers, = np.where(state < 0)
    if len(buffered_containers) == 0:
        return [0]
    filled = [len(np.where(i * columns <= buffered_containers < (i + 1) * columns)[0]) for i in range(rows)]
    if filled[0] < columns:
        open_groups.append(0)
    for level in range(1, rows):
        if filled[level] < filled[level - 1]:
            open_groups.append(level)
        elif filled[level] > filled[level - 1]:
            raise AssertionError('The algorithm has tried to conquer gravity by putting a floating container!')
    return open_groups


def find_available_containers(state, rows, columns, bays):
    available_containers = set()
    total_columns = columns * bays
    for c in range(total_columns):
        mask, = np.where(np.logical_and(state % total_columns == c, state >= 0))  # select the containers in that column
        if len(mask) == 0:
            continue
        order = mask[np.argsort(state[mask])]  # order them from bottom to top
        available_containers.add(order[-1])
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


def find_forbidden_moves(state, container, rows, columns, bays, b_goal):
    if state[container] < 0:
        return set(np.arange(-1, bays * columns))
    forbidden_moves = find_full_stacks(state, rows, columns, bays)
    if container not in find_needed_containers(state, rows, columns, b_goal):
        forbidden_moves.add(-1)
    return forbidden_moves


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)
