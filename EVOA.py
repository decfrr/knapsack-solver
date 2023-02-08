"""
Cite From : https://link.springer.com/chapter/10.1007/978-3-642-37371-8_26
Egyptian Vulture Optimization Algorithm – A New Nature Inspired Meta-heuristics for Knapsack Problem
Sur, C., Sharma, S., Shukla, A. (2013). Egyptian Vulture Optimization Algorithm – A New Nature Inspired Meta-heuristics
for Knapsack Problem. In: Meesad, P., Unger, H., Boonkrong, S. (eds)
The 9th International Conference on Computing and InformationTechnology (IC2IT2013).
Advances in Intelligent Systems and Computing, vol 209. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-37371-8_26

Tossing of Pebbles:
If PS > 0 Then “Get In” Else “No Get In”. Also, If FT > 0 Then “Removal” Else “No Removal”
Overall there are four combinations of operations are possible and are :
    Case 1 :GetIn & NoRemoval,
    Case 2: NoGetIn & Removal,
    Case 3:GetIn & Removal,
    Case 4: NoGetIn & NoRemoval.

Rolling of Twigs:
DS = Degree of Roll where DS ≥ 0 denoting number of rolls.
DR as Direction of Rolling where probabilistically we have:
DR = 0 for Right Rolling/Shift = 1 for Left Rolling/Shift
where 0 and 1 is generated randomly and deterministically the equation can be framed as :
DR = Left Rolling/Shift for RightHalf > LeftHalf = Right Rolling/Shift for RightHalf < LeftHalf
where RightHalf is the secondary fitness for the right half of the solution solution and LeftHalf is for left half.
The reason behind this is if the RightHalf is better, then this will be a provision to extent the source
with the connected node portion and same is for LeftHalf, which can be connected with the destination.

Change of Angle:
Now the change of the angle is represented as a mutation step where the unconnected linked node sequence are reversed
for the expectation of being connected and thus complete the sequence of nodes.


Step 1: Consider a dataset of n items and xi is the ith bag, for m number of bags
where each bag is has maximum capacity Wm.

Step 2: Generate N solutions for each type of bag xi where solution for xi for i = 1,2,...n consisting of n items
and each is represented is by a number from 1 to n without repetition of the numbers in the solution.

Step 3: Now generate a threshold between 1 and n in integer values,
so that the solution of integer values can be converted to solution of 0s and 1s and the random value of threshold will decide
how many of them should be 0s and 1s and also the positional values will create combinations
and due to the EVOA the integer values for each position will change.

Step 4: For xi > threshold Make it 1, else make it 0. So solution of 0 & 1 represent xi set.

Step 5: Evaluate the fitness of each solution, Store the value of profit if constraint satisfied else make it zero.
Update the Global best if required.

Step 6: Perform Tossing of Pebbles operation at selected or random points
depending upon implementation on deterministic approach or probability.

Step 7: Perform Rolling of Twigs operation on selected or the whole solution
depending on the pseudorandom generation of the two operation parameters.

Step 8: Perform Change of Angle operation through selective reversal of solution subset.
This is some kind of extra effort introduced by the bird for efficient result.

Step 9: Evaluate the fitness of each solution. Store the value of profit if constraint satisfied else make it zero.
Update the Global best if required.
Replace if fitness is better (here the best is the maximization of the profit subjected to the satisfaction of the capacity of the bags).
The global best consists of the fitness value along with the solution consisting of 1s and 0s.

Step 10: After a certain iterations, replace a certain percentage of worst fitted solutions
with the best solution of the iteration or global best with probability.

Step 11: Stop with stopping criteria fulfilled.
"""

from abc import ABC
import numpy as np
import Knapsack
from typing import Tuple

np.seterr(invalid='ignore')


class EVOA(Knapsack.Knapsack, ABC):
    def __init__(self, item_weights: np.array, item_values: np.array, capacity: int, seed: int = 0):
        super().__init__(item_weights, item_values, capacity, seed)
        self.params = {}

    def init_param(self, population_size: int, max_iterations: int, replace_iteration: int, replace_rate: float):
        if max_iterations < replace_iteration or replace_iteration < 0:
            raise ValueError("replace_iteration must be between 0 and max_iterations")
        if replace_rate < 0 or replace_rate >= 1:
            raise ValueError("replace_rate must be between 0 and 1")
        self.params['population_size'] = population_size
        self.params['max_iterations'] = max_iterations
        self.params['replace_iteration'] = replace_iteration
        self.params['replace_rate'] = replace_rate
        self.params['threshold'] = np.random.randint(1, len(self.item_weights))

    @staticmethod
    def _check_params(self):
        """
        Check parameters

        Parameters
        ----------
        :return: None
        """
        params_list = ["population_size", "max_iterations", "replace_iteration", "replace_rate", "threshold"]
        for param in params_list:
            if param not in self.params:
                raise ValueError("Parameter {} is required".format(param))

    @staticmethod
    def _calc_fitness(self, string: np.array) -> Tuple[float, float]:
        """
        Calculate fitness of a string
        :param self:
        :param string: np.array
        :return:
        """
        solution = self._convert_to_solution(string, self.params['threshold'])
        profit = self.item_values * solution
        weight = self.item_weights * solution
        if weight.sum() > self.capacity:
            return 0, string
        return profit.sum(), string

    @staticmethod
    def _convert_to_solution(string: np.array, threshold: int) -> np.array:
        """
        convert string to solution.
        if string is greater than the threshold, make it 1, otherwise make it 0
        :param string: np.array
        :param threshold: int
        :return:
        """
        return np.array([1 if string[i] > threshold else 0 for i in range(len(string))])

    @staticmethod
    def _remove_duplicates(self, string: np.array) -> np.array:
        """
        remove duplicates and compensate randomly continue until there is no duplicate
        if string length is not equal to item length, remove the extra items or add the missing items
        :param string: np.array
        :return: np.array
        """
        while len(np.unique(string)) != len(self.item_weights) or len(string) != len(self.item_weights):
            # remove duplicates
            string = np.unique(string)
            # compensate randomly
            # if string length is less than item length, add the missing items
            if len(string) < len(self.item_weights):
                string = np.append(string, np.random.randint(1, len(self.item_weights) + 1,
                                                             len(self.item_weights) - len(string)))
            # if string length is greater than item length, remove the extra items
            elif len(string) > len(self.item_weights):
                string = np.delete(string, np.random.randint(0, len(string), len(string) - len(self.item_weights)))
        return string

    @staticmethod
    def _generate_strings(n: int) -> np.array:
        """
        generate strings randomly
        :param n:
        :return:
        """
        return np.random.permutation(n) + 1

    @staticmethod
    def _tossing_of_pebbles(self, population: list) -> list:
        """
        for each string, operate tossing of pebbles
        First, determine the pebble size randomly. (0 <= pebble_size <= int(len(population[i]) / 5))
        and generate pebble randomly. ( 1 <= pebble <= (pebble_size + 1) * len(population[i]) )
        Second, determine the hitting point randomly. (0 <= hitting_point <= len(population[i]) - 1)
        Third, determine the force of tossing randomly. (0 <= force < len(population[i] - hitting_point))
        Note: hitting_point + force <= len(population[i]) - 1 therefore force < len(population[i] - hitting_point)
        :param population: list of np.array
        :return: list of np.array
        """
        for i in range(len(population)):
            n = len(population[i])
            # determine the pebble size
            pebble_size = np.random.randint(0, int(n / 5))
            # determine the hitting point
            hitting_point = np.random.randint(0, n - 1)
            # determine the force of tossing
            force = np.random.randint(0, n - hitting_point - 1)

            if pebble_size == 0 and force == 0:
                continue
            # perform tossing of pebbles
            if force != 0:
                # if force is not zero, pop the force number of elements from the hitting point
                population[i] = np.delete(population[i], np.arange(hitting_point, hitting_point + force))
            if pebble_size != 0:
                # pebble contains some number of pebble size
                pebble = np.random.randint(1, (pebble_size + 1) * n, pebble_size)
                # if pebble size is not zero, insert the pebble to the hitting point
                s = np.insert(population[i], hitting_point, pebble)
                population[i] = np.insert(population[i], hitting_point, pebble)
            # remove duplicates and compensate randomly
            population[i] = self._remove_duplicates(self, population[i])

        return population

    @staticmethod
    def _rolling_of_twigs(self, population: list) -> list:
        """
        for each string, operate rolling of twigs
        First, determine the degree of rolling.
        Second, determine the direction of rolling.
        Third, perform rolling of twigs.
        End, call remove duplicates function.
        :param population: list of np.array
        :return: list of np.array
        """
        for i in range(len(population)):
            # determine the degree of rolling
            degree = np.random.randint(1, len(population[i]))
            # determine the direction of rolling
            direction = np.random.randint(0, 2)
            # perform rolling of twigs
            if direction == 0:
                population[i] = np.roll(population[i], degree)
            else:
                population[i] = np.roll(population[i], -degree)
            # remove duplicates
            population[i] = EVOA._remove_duplicates(self, population[i])
        return population

    @staticmethod
    def _change_of_angle(self, population: list) -> list:
        """
        For each solution, operate change of angle.
        First, determine the number of items to be reversed.
        Second, determine the starting point of reversal.
        Third, perform reversal, and call remove duplicates function.

        ex. [1, 2, 3, 4, 7, 6, 5, 8, 9, 10] -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        a link (1, 2, 3, 4), b link (7, 6, 5), c link (8, 9, 10)
        b link is reversed expected result is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Reverse each link to connect each link start and end point.

        :param population: list of np.array
        :return: list of np.array
        """
        for i in range(len(population)):
            s = population[i].tolist()
            # search the links in string
            links = []
            link = []
            for j in range(len(s)):
                link.append(s[j])
                if j == len(s) - 1:
                    links.append(link)
                    break
                if s[j] + 1 != s[j + 1] and s[j] - 1 != s[j + 1]:
                    links.append(link)
                    link = []
            # check whether each link can be reversed
            # If, when you reverse a link, the beginning of that link is contiguous with the end of the previous link
            # and the end of that link is contiguous with the beginning of the next link, reverse that link.
            if len(links) == 1:
                continue
            for j in range(len(links)):
                if j == 0:
                    if links[j][0] + 1 == links[j + 1][0] or links[j][0] - 1 == links[j + 1][0]:
                        links[j] = links[j][::-1]
                elif j == len(links) - 1:
                    if links[j][-1] + 1 == links[j - 1][-1] or links[j][-1] - 1 == links[j - 1][-1]:
                        links[j] = links[j][::-1]
                else:
                    if links[j][0] + 1 == links[j + 1][0] and links[j][-1] - 1 == links[j - 1][-1]:
                        links[j] = links[j][::-1]
            # extract the result and call remove duplicates function and change population[i]
            population[i] = np.array([item for link in links for item in link])
            population[i] = EVOA._remove_duplicates(self, population[i])

        return population

    """
    string means the array of 1 ~ n (n = item number) without repetition
        ex: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 6, 7, 8, 9, 10, 1, 5]...
    solution means the array of 0s and 1s, 1 means the item is selected, 0 means the item is not selected
        ex: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]...
    To convert string to solution, use the following process.
        1. Determine a probabilistic integer from the numbers 1 to n, and use it as the threshold value.
        2. If the number of the string is greater than the threshold value, make it 1, otherwise make it 0.
        ex: threshold = 5
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            [2, 3, 4, 6, 7, 8, 9, 10, 1, 5] -> [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    """

    def search(self):
        # parameters check
        self._check_params(self)
        n = len(self.item_weights)

        # Step 1 ~ 4: Initialize the population of strings (solution, fitness)
        # population has always strings
        population = [self._generate_strings(n) for _ in range(self.params['population_size'])]
        global_best_fitness = -1
        # global_best_string = None
        global_best_time = 0
        delta = int(self.params["max_iterations"] / 20)
        # Step 5: (??)Evaluate the fitness of each string
        population_fitness, population = map(list,
                                             zip(*[self._calc_fitness(self, solution) for solution in population]))
        for iteration in range(self.params['max_iterations']):

            # Step 6: Perform Tossing of Pebbles operation
            population = self._tossing_of_pebbles(self, population)

            # Step 7: Perform Rolling of Twigs operation
            population = self._rolling_of_twigs(self, population)

            # Step 9-1: Evaluate the fitness of each solution again
            population_fitness, population = map(list,
                                                 zip(*[self._calc_fitness(self, solution) for solution in population]))

            # Step 9-2: Update the global best if necessary
            # sort population by population_fitness reverse
            sorted_index = np.argsort(population_fitness)[::-1]
            best_fitness, best_string = population_fitness[sorted_index[0]], population[sorted_index[0]]
            if best_fitness > global_best_fitness and best_fitness != 0:
                global_best_fitness = best_fitness
                # global_best_string = best_string
                self.set_solution(self._convert_to_solution(best_string, self.params['threshold']))
                global_best_time = 0
                # Step 8?: If fitness is improved, then perform Change of Angle operation
                population = self._change_of_angle(self, population)
            else:
                global_best_time += 1
            population = np.array(population)[sorted_index].tolist()

            # Step 10: Replace a certain percentage of the worst-fitted strings with the best string of the iteration
            # not always, but some iterations
            if iteration % self.params['replace_iteration'] == 0:
                population = np.array(population)
                population[int(self.params['population_size'] * self.params['replace_rate']):] = best_string
                population = population.tolist()

            # Step 11: Check stopping criteria
            if global_best_time > delta:
                break
        return self.solution, global_best_fitness
