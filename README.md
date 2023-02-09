## knapsack-solver

### Description
This repository contains two solvers for the knapsack problem. 
Both solvers are implemented in Python.

Note: All datasets are from [here](https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html).

### Setup
The solvers are implemented in Python 3.9, and the following packages are required:
- numpy (~=1.24.1)

```shell
# clone the repository
$ git clone https://github.com/decfrr/knapsack-solver.git
$ cd knapsack-solver
# install the required packages
$ pip install -r requirements.txt
```

### Usage
The following code is an example of how to use the solvers.
This repository contains unit tests for the solvers. (`evoa-test.py` and `sfla-test.py`)
Please refer to the unit tests as well.

```python
# import the Egyptian Vulture Optimization Algorithm (EVOA) and the Shuffled Frogs Leaping Algorithm (SFLA)
from EVOA import EVOA
from SFLA import SFLA
import numpy as np

# define the knapsack problem
# capacity of the knapsack: int
capacity = 165
# weights of the items: np.array
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
# profits of the items: np.array
profits = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
# set seed
seed = 0

# initialize the EVOA solver
evoa = EVOA(weights, profits, capacity, seed)
# initialize the SFLA solver
sfla = SFLA(weights, profits, capacity, seed)

# initialize some parameters
# For EVOA
# population size: int
pop_size = 500
# maximum number of iterations: int
max_iter = 50
# iteration of replacement: int (0 <= iter_replace <= max_iter)
iter_replace = 10
# replacement rate: float (0.0 <= rate < 1.0)
replace_rate = 0.2

evoa.init_param(pop_size, max_iter, iter_replace, replace_rate)

# For SFLA
# population size: int
pop_size = 500
# maximum number of iterations: int
max_iter = 50
# number of memeplexes: int ( pop_size % memeplexes == 0 )
memeplexes = 10
# method of update the worst solution: int (1 or 2 or 3, default: 1)
method = 1

sfla.init_param(pop_size, max_iter, memeplexes, method)

# solve the knapsack problem
# For EVOA
solution_evoa, fitness_evoa, weight_evoa = evoa()
# For SFLA
solution_sfla, fitness_sfla, weight_sfla = sfla()

# print the results
print("EVOA")
print("solution: ", solution_evoa)
print("fitness: ", fitness_evoa)
print("weight: ", weight_evoa)
print("SFLA")
print("solution: ", solution_sfla)
print("fitness: ", fitness_sfla)
print("weight: ", weight_sfla)
```


