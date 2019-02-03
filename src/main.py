from utils import *

DISCOUNT_FACTOR = 0.9
ROWS = 6  # the number of rows in each bay and in the buffer
COLUMNS = 6  # the number of columns in each bay and in the buffer
BAYS = 2  # I assume containers con move from one bay to the other during removal
N_CONTAINERS = 36
assert N_CONTAINERS <= ROWS * COLUMNS
B_GOAL = create_goal_buffer(ROWS, COLUMNS)

x = initialize_random_state(N_CONTAINERS, ROWS, COLUMNS, BAYS)

visualize_state(x, ROWS, COLUMNS, BAYS)
q_table = dict()


def q(state, container=None, target_column=None):
    """
    This function creates all the needed q_table values and returns their current status.

    :param state:
    :param container:
    :param target_column: -1 if you want to put it in the buffer
    :return: the q-value for the current state and action "move container c to target_column t";
        if the container is not specified, return sup_t Q(x, c, t) for all c
        if the target_column is not specified, return Q(x, c, t) for all t

    """
    if str(state) not in q_table:  # create all the needed dictionaries
        q_table[str(state)] = dict()
        for c in range(len(state)):
            q_table[str(state)][str(c)] = np.zeros(COLUMNS * BAYS + 1)
            mask = find_forbidden_moves(state, c, ROWS, COLUMNS, BAYS, B_GOAL)
            for i in mask:
                q_table[str(state)][str(c)][i] = -np.inf

    if container is None:
        return np.array([np.max(q_table[str(state)][str(c)]) for c in range(len(state))])

    if target_column is None:
        return q_table[str(state)][str(container)]

    return q_table[str(state)][str(container)][target_column]


p_container = softmax(q(x))
c = np.random.choice(range(N_CONTAINERS), size=1, p=p_container)

