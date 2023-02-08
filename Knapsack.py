import time
import numpy as np
import abc


class Knapsack:
    """
    Knapsack problem solver class (abstract)
    :param item_weights: item weights
    :param item_values: item values
    :param capacity: knapsack capacity
    :param seed: random seed
    """
    def __init__(self, item_weights: np.array, item_values: np.array, capacity: int, seed: int = 0):
        self.item_weights = item_weights
        self.item_values = item_values
        self.capacity = capacity
        self.seed = seed
        np.random.seed(seed)
        self.solution = np.random.randint(0, 2, size=len(item_weights))
        self.params = {}

    def knapsack_value(self) -> float:
        total_weight = np.sum(self.solution * self.item_weights)
        total_value = np.sum(self.solution * self.item_values)
        return total_value if total_weight <= self.capacity else -1

    def knapsack_weight(self) -> float:
        return float(np.sum(self.solution * self.item_weights))

    def set_solution(self, solution: np.array) -> None:
        self.solution = solution

    def __call__(self, *args, **kwargs) -> (np.array, float, float):
        start_time = time.time()
        self.search()
        end_time = time.time()
        print("Time: ", end_time - start_time)
        return list(self.solution), self.knapsack_value(), self.knapsack_weight()

    def __str__(self) -> str:
        return f"Solutions: {list(self.solution)} \n" \
               f"Fitness: {self.knapsack_value()} \n" \
               f"Weight: {self.knapsack_weight()} \n" \
               f"Capacity: {self.capacity}"

    def __repr__(self) -> str:
        return self.__str__()

    @abc.abstractmethod
    def search(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def init_param(self, **kwargs):
        raise NotImplementedError
