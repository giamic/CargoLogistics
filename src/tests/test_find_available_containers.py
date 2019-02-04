from unittest import TestCase

from utils import find_available_containers
import numpy as np


class TestFind_available_containers(TestCase):
    rows, columns, bays = 6, 6, 2
    total_stacks = columns * bays + 1

    def test_1(self):
        x = np.array([0, 12, 13, 1, 5])
        self.assertEqual(find_available_containers(x, self.total_stacks), {1, 2, 4})

    def test_negative(self):
        x = np.array([0, 12, 13, 1, 5, -1, -1, 4])
        self.assertEqual(find_available_containers(x, self.total_stacks), {1, 2, 4, 7})

    def test_simple(self):
        x = np.arange(self.rows * self.columns)
        self.assertEqual(find_available_containers(x, self.total_stacks), set(np.arange(24, 36)))
