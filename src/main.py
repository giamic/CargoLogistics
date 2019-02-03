from utils import *

DISCOUNT_FACTOR = 0.9
ROWS = 6  # the number of rows in each bay and in the buffer
COLUMNS = 6  # the number of columns in each bay and in the buffer
BAYS = 2  # I assume containers con move from one bay to the other during removal
N_CONTAINERS = 36
TOTAL_STACKS = BAYS * COLUMNS + 1
assert N_CONTAINERS <= ROWS * COLUMNS
B_GOAL = create_goal_buffer(ROWS, COLUMNS)

x = initialize_random_state(N_CONTAINERS, ROWS, COLUMNS, BAYS)

visualize_state(x, ROWS, COLUMNS, BAYS)
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
    if str(state) not in q_table:  # create all the needed dictionaries
        q_table[str(state)] = dict()
        ac = find_available_containers(state, ROWS, COLUMNS, BAYS)
        nc = find_needed_containers(state, ROWS, COLUMNS, B_GOAL)
        fs = find_full_stacks(state, ROWS, COLUMNS, BAYS)
        for c in range(len(state)):
            if c not in ac:  # if the container is not available, prevent every move
                q_table[str(state)][str(c)] = np.ones(TOTAL_STACKS) * (-np.inf)
            else:
                q_table[str(state)][str(c)] = np.zeros(TOTAL_STACKS)
                for ts in fs:  # if the target stack is full, prevent that move
                    q_table[str(state)][str(c)][ts] = -np.inf
                if c not in nc:  # if the container is not currently needed in the buffer, prevent that move
                    q_table[str(state)][str(c)][-1] = -np.inf

    if container is None:
        return np.array([np.max(q_table[str(state)][str(c)]) for c in range(len(state))])

    if target_stack is None:
        return q_table[str(state)][str(container)]

    return q_table[str(state)][str(container)][target_stack]


p_container = softmax(q(x))
c = np.random.choice(N_CONTAINERS, size=1, p=p_container)
p_stack = softmax(q(x, c))
s = np.random.choice(TOTAL_STACKS, size=1, p=p_stack)

