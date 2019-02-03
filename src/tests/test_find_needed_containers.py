from unittest import TestCase
import numpy as np

from utils import find_needed_containers, create_goal_buffer


class TestFind_needed_containers(TestCase):
    rows, columns, bays = 6, 6, 2
    b_goal = create_goal_buffer(rows, columns)

    def test_1(self):
        x = np.random.choice(self.rows * self.columns * self.bays, self.rows * self.columns, replace=False)
        buffered_containers = [0, 1, 2, 7, 8, 14]
        needed_containers = {3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23}
        for c in buffered_containers:
            x[c] = -1
        self.assertEqual(find_needed_containers(x, self.rows, self.columns, self.b_goal), needed_containers)

    def test_floating(self):
        x = np.random.choice(self.rows * self.columns * self.bays, self.rows * self.columns, replace=False)
        buffered_containers = [7]
        for c in buffered_containers:
            x[c] = -1
        f = lambda: find_needed_containers(x, self.rows, self.columns, self.b_goal)
        self.assertRaises(AssertionError, f)
