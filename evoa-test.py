import unittest
import numpy as np
from EVOA import EVOA

# Test Case
item_weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
item_values = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
knapsack_capacity = 165


class TestEVOA(unittest.TestCase):
    def test_evoa(self):
        evoa = EVOA(item_weights=item_weights,
                    item_values=item_values,
                    capacity=knapsack_capacity,
                    seed=0)
        evoa.init_param(population_size=500,
                        max_iterations=50,
                        replace_iteration=10,
                        replace_rate=0.2)
        solution, fitness, weight = evoa()
        self.assertEqual(solution, [1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        self.assertEqual(fitness, 309)
        self.assertEqual(weight, 165)


if __name__ == '__main__':
    unittest.main()
