from abc import ABC
import numpy as np
import Knapsack
from typing import Tuple
np.seterr(invalid='ignore')


class SFLA(Knapsack.Knapsack, ABC):
    def __init__(self, item_weights: np.array, item_values: np.array, capacity: int, seed: int = 0):
        super().__init__(item_weights, item_values, capacity, seed)
        self.global_best_solution = None
        self.global_best_fitness = -1

    def init_param(self, population_size: int,
                   max_iterations: int,
                   memeplex_size: int,
                   mutation: float = 0.05,
                   method: int = 1,
                   alpha: float = 0.5,
                   ):
        """
        Initialize parameters for SLFA (Shuffled Leaping Frog Algorithm)

        Parameters
        ----------
        :param population_size: int
            Number of frogs in the population
        :param max_iterations: int
            Maximum number of iterations for local search
        :param memeplex_size: int
            Number of memeplexes
        :param mutation: float (default: 0.05)
            Probability of mutation
        :param method: int (default: 1)
            select method for updating worst frog(xw)
        :param alpha: float (default: 0.5)
            static probability for updating worst frog(xw) with method 3
        :return: None

        Notes
        -----
        The method for updating worst frog(xw) can be one of the following:
            First, We define `D_i = Rand() \times (X_b - X_w)`

            `Rand() ~ U(0, 1)` , and `X_b, X_w` are the best and worst frogs in the memeplex, respectively.

            Then, we have 3 methods for updating worst frog(X_w):
             - Method 1:
                t = X_w + D_i \\
                X_w(new) = \begin{cases}
                    0 & \text{if } t \leq 0 \\
                    round(t) & \text{if } 0 < t < 1 \\
                    1 & \text{if } t \geq 1
                \end{cases}

            - Method 2:
                t = \frac{1}{1 + e^{-D_i}} \\
                u ~ U(0, 1) \\
                X_w(new) = \begin{cases}
                    0 & \text{if } t \leq u \\
                    1 & \text{if } t \geq u
                \end{cases}

            - Method 3:
                t = \frac{1}{1 + e^{-D_i}} \\
                set \alpha (called static probability) \\
                X_w(new) = \begin{cases}
                    0 & \text{if } t \leq \alpha \\
                    X_w & \text{if } \alpha < t < 0.5 \times (1 + \alpha) \\
                    1 & \text{if } t \geq 0.5 \times (1 + \alpha)
                \end{cases}

        """
        if not (method == 1 or method == 2 or method == 3):
            raise ValueError("Method must be 1, 2 or 3")
        rho = np.max(self.item_values / self.item_weights)
        self.params = {
            "population_size": population_size,
            "max_iterations": max_iterations,
            "memeplex_size": memeplex_size,
            "mutation": mutation,
            "method": method,
            "alpha": alpha,
            "rho": rho,
        }
        np.random.seed(self.seed)

    @staticmethod
    def _check_params(self):
        """
        Check parameters

        Parameters
        ----------
        :return: None
        """
        params_list = ["population_size", "memeplex_size", "max_iterations", "rho", "alpha", "method"]
        for param in params_list:
            if param not in self.params:
                raise ValueError("Parameter {} is required".format(param))
        if self.params["population_size"] % self.params["memeplex_size"] != 0:
            raise ValueError("Population size must be a multiple of memeplex size")

    @staticmethod
    def set_global_best_solution(self, global_best_solution: np.array, global_best_fitness: float) -> None:
        self.global_best_solution = global_best_solution
        self.global_best_fitness = global_best_fitness

    @staticmethod
    def _calc_fitness(self, solution: np.array) -> Tuple[float, np.array]:
        """
        Calculate fitness of a solution. This function is not penalty function.
        If the total weight of the solution is greater than the capacity of the knapsack,
        the solution is repaired by greedy method.

        Parameters
        ----------
        :param self:
        :param solution: np.array
        :return: float
            fitness of the solution
        """
        knapsack_weight = solution * self.item_weights
        knapsack_value = solution * self.item_values
        total_weight = knapsack_weight.sum()
        total_value = knapsack_value.sum()
        if total_weight <= self.capacity:
            return float(total_value), solution
        # greedy repair:All items "in the knapsack" are sorted in the decreasing order of their profit to weight ratios.
        # The selection procedure always chooses the last item for deletion.
        # knapsack profit to weight ratios, when knapsack weight = 0, knapsack_ratio = 0
        knapsack_ratio = np.where(knapsack_weight == 0, 0, knapsack_value / knapsack_weight)
        # sort knapsack_ratio in decreasing order
        sorted_index = np.argsort(knapsack_ratio)
        # repair
        for i in sorted_index:
            if total_weight <= self.capacity:
                break
            if solution[i] == 1:
                total_weight -= self.item_weights[i]
                total_value -= self.item_values[i]
                solution[i] = 0
        return float(total_value), solution

    @staticmethod
    def _generate_random_solution(self) -> np.array:
        """
        Generate a random solution

        Parameters
        ----------
        :param self:
        :return: np.array
            random solution
        """
        return np.random.randint(0, 2, size=len(self.item_weights))

    @staticmethod
    def _mutation(self, population: np.array) -> np.array:
        """
        Mutation operator

        Parameters
        ----------
        :param self:
        :param population: np.array
        :return: np.array
            population after mutation
        """
        for i in range(len(population)):
            if np.random.uniform(0, 1) < self.params["mutation"]:
                population[i] = np.random.randint(0, 2, size=len(self.item_weights))
        return population

    @staticmethod
    def _local_search(self, memeplex) -> np.array:
        for iteration in range(self.params["max_iterations"]):
            index_xb, index_xw = -1, -1
            fitness_xb, fitness_xw = -np.inf, np.inf
            for i in range(len(memeplex)):
                fitness, memeplex[i] = self._calc_fitness(self, memeplex[i])
                if fitness > fitness_xb:
                    fitness_xb, index_xb = fitness, i
                elif fitness < fitness_xw:
                    fitness_xw, fitness_xw = fitness, i
            # if local best solution is better than global best solution, update global best solution
            if fitness_xb > self.global_best_fitness:
                self.set_global_best_solution(self, memeplex[index_xb], fitness_xb)

            # replace new xw
            # di = np.random.uniform(0, 1, size=len(self.item_weights)) * (memeplex[index_xb] - memeplex[index_xw])
            di = np.random.uniform(0, 1, 1) * (memeplex[index_xb] - memeplex[index_xw])
            xw_new = None
            # method 1
            if self.params["method"] == 1:
                t = memeplex[index_xw] + di
                # if t <= 0, set t = 0, elif 0 < t < 1, set t = round(t), else t >= 1, set t = 1
                xw_new = np.vectorize(lambda x: 0 if x <= 0 else 1 if x >= 1 else np.round(x))(t)
            elif self.params["method"] == 2:
                t = 1 / (1 + np.exp(-di))
                u = np.random.uniform(0, 1, 1)
                xw_new = np.vectorize(lambda x, y: 1 if x <= y else 0)(t, u)
            elif self.params["method"] == 3:
                t = 1 / (1 + np.exp(-di))
                alpha = self.params["alpha"]
                # if t <= alpha set xw_new = 0, elif alpha < t <= 0.5*(1 + alpha), set xw_new = xw,
                # else t > 0.5*(1 + alpha), set xw_new = 1
                xw_new = np.vectorize(
                    lambda x, y: 0 if x <= alpha else 1 if x > 0.5 * (1 + alpha) else y)(t, memeplex[index_xw])

            fitness, xw_new = self._calc_fitness(self, xw_new)
            if fitness > fitness_xw:
                memeplex[index_xw] = xw_new
                # if xw_new is better than global best solution (rare case), update global best solution
                if fitness > self.global_best_fitness:
                    self.set_global_best_solution(self, xw_new, fitness)
                continue

            # if xw_new is not better than xw, then change xb -> xg
            di = np.random.uniform(0, 1, 1) * (memeplex[index_xb] - self.global_best_solution)
            xw_new = None
            # method 1
            if self.params["method"] == 1:
                t = memeplex[index_xw] + di
                # if t <= 0, set t = 0, elif 0 < t < 1, set t = round(t), else t >= 1, set t = 1
                xw_new = np.vectorize(lambda x: 0 if x <= 0 else 1 if x >= 1 else np.round(x))(t)
            elif self.params["method"] == 2:
                t = 1 / (1 + np.exp(-di))
                u = np.random.uniform(0, 1, 1)
                xw_new = np.vectorize(lambda x, y: 1 if x <= y else 0)(t, u)
            elif self.params["method"] == 3:
                t = 1 / (1 + np.exp(-di))
                alpha = self.params["alpha"]
                # if t <= alpha set xw_new = 0, elif alpha < t <= 0.5*(1 + alpha), set xw_new = xw,
                # else t > 0.5*(1 + alpha), set xw_new = 1
                xw_new = np.vectorize(
                    lambda x, y: 0 if x <= alpha else 1 if x > 0.5 * (1 + alpha) else y)(t, memeplex[index_xw])
            fitness, xw_new = self._calc_fitness(self, xw_new)
            if fitness > fitness_xw:
                memeplex[index_xw] = xw_new
                # if xw_new is better than global best solution (rare case), update global best solution
                if fitness > self.global_best_fitness:
                    self.set_global_best_solution(self, xw_new, fitness)
                continue

            # if xw_new is not better than xw, replace xw with a random solution
            xw_new = self._generate_random_solution(self)
            fitness, xw_new = self._calc_fitness(self, xw_new)
            memeplex[index_xw] = xw_new
            # if xw_new is better than global best solution (rare case), update global best solution
            if fitness > self.global_best_fitness:
                self.set_global_best_solution(self, xw_new, fitness)
        return memeplex

    def search(self, **kwargs):
        """
        Search for the best solution

        Parameters
        ----------
        :param kwargs:
        :return: np.array
            best solution
        """
        # parameters check
        self._check_params(self)
        # generate initial population
        population = [self._generate_random_solution(self) for _ in range(self.params["population_size"])]
        # if the best solution is not changed for delta, stop searching
        # delta is recommended to [max_iterations / 20, max_iterations / 10]
        # then delta = int(max_iterations / 20)
        delta = int(self.params["max_iterations"] / 20)
        best_time = 0
        best_fitness = -np.inf
        while True:
            # calculate fitness for each solution
            population_fitness, population = map(np.array,
                                                 zip(*[self._calc_fitness(self, solution) for solution in population]))
            # sort population by population_fitness reverse
            sorted_index = np.argsort(population_fitness)[::-1]
            population = population[sorted_index]
            # get global best solution
            global_fitness, global_solution = population_fitness[sorted_index[0]], population[sorted_index[0]]
            self.set_global_best_solution(self, global_solution, global_fitness)
            # generate memeplexes
            memeplexes = np.array(
                [population[i::self.params["memeplex_size"]] for i in range(self.params["memeplex_size"])])
            result_memplexes = []
            # local search
            for memeplex in memeplexes:
                memeplex = self._local_search(self, memeplex)
                result_memplexes.append(memeplex)
            # sort memeplexes by fitness
            population = np.concatenate(result_memplexes)
            # shuffled population
            np.random.shuffle(population)
            # apply mutation on the population
            population = self._mutation(self, population)
            # check the termination condition
            # calculate fitness
            population_fitness, population = map(np.array,
                                                 zip(*[self._calc_fitness(self, solution) for solution in population]))
            # sort population by population_fitness reverse
            sorted_index = np.argsort(population_fitness)[::-1]
            population = population[sorted_index]
            # get the best solution fitness
            fitness, solution = self._calc_fitness(self, population[0])
            if fitness > best_fitness:
                best_fitness = fitness
                best_time = 0
                self.set_solution(solution)
            else:
                best_time += 1
            if best_time >= delta:
                break
        return self.solution, fitness
