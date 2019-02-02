import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_goal_buffer(rows, columns):
    y = list(np.reshape(np.random.permutation(rows * columns), (rows, columns)))
    for i in range(rows):
        y[i] = set(y[i])
    return y


def initialize_empty_buffer(rows):
    return [set() for _ in range(rows)]


def initialize_random_state(n, rows, columns, bays):
    slots = rows * columns * bays
    x = np.random.choice(range(slots), size=n, replace=False)

    def _gravity_fall(x, total_columns):
        for col in range(total_columns):  # column by column, bring all containers to the bottom
            mask, = np.where(np.logical_and(x % total_columns == c, x >= 0))  # select the containers in that column
            if len(mask) == 0:
                continue
            order = mask[np.argsort(x[mask])]  # order them from bottom to top
            for end_row, i in enumerate(order):  # let them fall to the ground
                x[i] = col + end_row * total_columns
        return x

    return _gravity_fall(x, columns * bays)


def visualize_state(x, rows, columns, bays):
    total_columns = columns * bays
    grid = np.zeros((rows, total_columns))
    for c in x:
        if c >= 0:
            i, j = c // total_columns, c % total_columns
            grid[i, j] = 1
    plt.pcolormesh(grid, cmap=cm.gray)
    plt.show()
    return


def find_open_groups(buffer, rows, columns):
    open_groups = []
    if len(buffer[0]) < columns:
        open_groups.append(0)
    for level in range(1, rows):
        if len(buffer[level]) < len(buffer[level - 1]):
            open_groups.append(level)
    return open_groups


def find_available_containers(x, rows, columns, bays):
    available_containers = set()
    total_columns = columns * bays
    for c in range(total_columns):
        mask, = np.where(np.logical_and(x % total_columns == c, x >= 0))  # select the containers in that column
        if len(mask) == 0:
            continue
        order = mask[np.argsort(x[mask])]  # order them from bottom to top
        available_containers.add(order[-1])
    return available_containers
