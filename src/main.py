from collections import defaultdict

from utils import *

DISCOUNT_FACTOR = 0.9
ROWS = 6  # the number of rows in each bay and in the buffer
COLUMNS = 6  # the number of columns in each bay and in the buffer
BAYS = 2  # I assume containers con move from one bay to the other during removal
N_CONTAINERS = 36
assert N_CONTAINERS <= ROWS * COLUMNS

b_goal = create_goal_buffer(ROWS, COLUMNS)
b_curr = initialize_empty_buffer(ROWS)
x = initialize_random_state(N_CONTAINERS, ROWS, COLUMNS, BAYS)
# visualize_state(x, ROWS, COLUMNS, BAYS)
# Q = defaultdict(lambda: np.zeros())
