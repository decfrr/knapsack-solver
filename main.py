import numpy as np
from numpy import ndarray
from EVOA import EVOA
from SFLA import SFLA


def read_input_data(problem_number: int) -> tuple[int, ndarray, ndarray, ndarray]:
    """
    Read input data from files
    :param problem_number: problem number
    :return: max_weight, weights, values, solution
    """
    w = []
    v = []
    s = []

    # read capacity from file
    with open(f"datasets/p{problem_number:02d}_c.txt", "r") as file:
        c = int(file.readline().strip())

    # read weights from file
    with open(f"datasets/p{problem_number:02d}_w.txt", "r") as file:
        for line in file:
            w.append(int(line.strip()))
        w = np.array(w)

    # read values from file
    with open(f"datasets/p{problem_number:02d}_p.txt", "r") as file:
        for line in file:
            v.append(int(line.strip()))
        v = np.array(v)

    # read solution from file
    with open(f"datasets/p{problem_number:02d}_s.txt", "r") as file:
        for line in file:
            s.append(int(line.strip()))
        s = np.array(s)

    return int(c), w, v, s


capacity, weights, values, solution = read_input_data(7)
evoa = EVOA(weights, values, capacity, seed=0)
evoa.init_param(population_size=500, max_iterations=500, replace_iteration=20, replace_rate=0.2)

sfla = SFLA(weights, values, capacity, seed=0)
sfla.init_param(population_size=500, max_iterations=500, memeplex_size=20, method=1)

print("Optimized Solution: ", solution)
print("EVOA Solution: ", evoa())
print("SFLA Solution: ", sfla())
