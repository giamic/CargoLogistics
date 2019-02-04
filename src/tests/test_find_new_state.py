from unittest import TestCase
import numpy as np
from utils import find_new_state
from numpy.testing import assert_array_equal


class TestFind_new_state(TestCase):
    rows, columns, bays = 6, 6, 2
    total_stacks = columns * bays + 1
    x = np.arange(rows * columns)

    def test_1(self):
        y = self.x.copy()
        y[35] = 36
        assert_array_equal(find_new_state(self.x, 35, 0, self.total_stacks), y)

    def test_2(self):
        y = self.x.copy()
        y[24] = 24
        assert_array_equal(find_new_state(self.x, 24, 0, self.total_stacks), y)
