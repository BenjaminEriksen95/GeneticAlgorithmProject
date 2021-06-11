"""
Mutation operators

These are accessed as an argument through their name defined in return_mutation_operator.
"""

# Built-in/Generic Imports
import random
from copy import copy

# Libs
import numpy as np

# Own modules
from .solution import Solution
from . import ICrossoverOperator

__author__ = "Benjamin Eriksen"


def return_crossover_operator(operator):
    '''
        Translates an operator name to an operator.
        This removes the need to import specific operators and thus makes adding additional operators simpler
    '''
    if operator == "singlepoint":
        return SinglePointCrossover
    elif operator == "c-cross":
        return BiasedUniformCrossover
    elif operator == "ordered":
        return OrderedCrossover
    elif operator == "position":
        raise NotImplementedError(operator + " not implemented")
    elif operator == "cycle":
        raise NotImplementedError(operator + " not implemented")
    else:
        raise NotImplementedError(operator + " not found")


"""
    Binary crossovers
"""


class SinglePointCrossover(ICrossoverOperator):
    def crossover(parent1, parent2, n, option=None):
        # set to make sure that one chromosome is not entirely overwritten
        point = random.randint(1, n - 1)  # add bias by biased sample of point
        return Solution(np.append(parent1.chromosome[:point], parent2.chromosome[point:]))


class BiasedUniformCrossover(ICrossoverOperator):
    # takes element from parent1 with probability 1-c and from parent1 with probability c
    # If option=0.5 then crossover is uniform.
    def crossover(parent1, parent2, n, option: float = 0.5):
        offspring = copy(parent1.chromosome)

        k = np.random.binomial(n, option)
        flip = np.random.choice(range(n), k, replace=False)

        for i in flip:
            offspring[i] = parent2.chromosome[i]

        return Solution(offspring)

    def get_name():
        return "biased-uniform crossover"


"""
    Permutation crossovers
"""


class OrderedCrossover(ICrossoverOperator):
    def crossover(parent1, parent2, n, option=None):
        """ Ordered crossover (OX)^Cmake: *** [Makefile:2: hpc] Interrupt


        [Goldberg1989] Goldberg. Genetic algorithms in search,
           optimization and machine learning. Addison Wesley, 1989
        """
        parent1 = parent1.chromosome
        parent2 = parent2.chromosome
        # Select substring to take from parent1
        a, b = random.sample(range(n), 2)
        if a > b:
            a, b = b, a

        holes = [True] * n
        for i in range(n):
            if i < a or i > b:
                holes[parent2[i]] = False

        # Fill offspring with genes from parent1
        offspring = copy(parent1)
        k = b + 1
        for i in range(n):
            if not holes[parent1[(i + b + 1) % n]]:
                offspring[k % n] = parent1[(i + b + 1) % n]
                k += 1
        # Fill remaning from parent2
        for i in range(a, b + 1):
            offspring[i] = parent2[i]

        return Solution(offspring)

    def get_name():
        return "ordered crossover(OX)"


class PositionCrossover(ICrossoverOperator):
    def crossover(parent1, parent2, n, option=None):
        raise NotImplementedError

    def get_name():
        raise NotImplementedError


class CycleCrossover(ICrossoverOperator):
    def crossover(parent1, parent2, n, option=None):
        raise NotImplementedError

    def get_name():
        raise NotImplementedError
