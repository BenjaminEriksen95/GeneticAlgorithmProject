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
from . import IMutationOperator

__author__ = "Benjamin Eriksen"


def return_mutation_operator(operator):
    '''
        Translates an operator name to an operator.
        This removes the need to import specific operators and thus makes adding additional operators simpler
    '''
    if operator == "k-flip":
        return KFlipMutation
    elif operator == "p-flip":
        return PFlipMutation
    elif operator == "k-exchange":
        return KExchangeMutation
    elif operator == "k-reverse":
        return ReverseMutation
    elif operator == "k-swap":
        return KSwapMutation
    else:
        raise NotImplementedError(operator + " not found")


"""
    Binary mutation
"""


class KFlipMutation(IMutationOperator):
    """
    Picks k(option) random bits and flips them
    """
    def mutate(proto, n, option: int = 1):
        k = int(option)
        flips = np.random.randint(low=0, high=n, size=k)
        offspring = copy(proto.chromosome)
        for i in flips:
            offspring[i] ^= 1

        return Solution(offspring)

    def get_name():
        return "k-bit flip"


class PFlipMutation(IMutationOperator):
    """
    Flips each bit with a probability option/len(s)
    If p is not set 1/len(s) is used
    """
    def mutate(proto, n, option: int = 1):
        offspring = copy(proto.chromosome)
        p = int(option) / n if option else 1 / n

        for gene in range(n):
            if random.random() < p:
                offspring[gene] ^= 1
        return Solution(offspring)

    def get_name():
        return "p-bit flip"

# For TSN

# The Analysis of Evolutionary Algorithms on Sorting and Shortest Paths Problems
# The most simple local operation is swap(i) which exchanges the elements at the positions i and i + 1.
# exchange(i, j ) exchanges the elements at the positions i and j ,
# – jump(i, j ) causes the element at position i to jump to position j while the elements at positions i + 1,...,j (if j>i) or j,... , i − 1 (if j<i) are shifted in the appropriate direction,
# – reverse(i, j ), where i<j , reverses the ordering of the elements at the positions i, . . . , j
# def swap(chromosome, length):
#     return None


"""
    Permutation mutation
"""


class KExchangeMutation(IMutationOperator):
    def mutate(proto, n, option: int = 1):
        """
        Picks 2 random items of the list and swaps them option(k) times
        """
        chromosome = proto.chromosome
        for i in range(option):
            offspring = copy(chromosome)
            [gene1, gene2] = np.random.randint(low=0, high=n, size=2)
            offspring[gene1] = chromosome[gene2]
            offspring[gene2] = chromosome[gene1]
            chromosome = offspring
        return Solution(chromosome)

    def get_name():
        return "k-exchange"


class KSwapMutation(IMutationOperator):
    def mutate(proto, n, option: int = 1):
        """
        Picks moves a random item of the list to a new location, shifting the rest of the list toward the position the item came from
        """
        chromosome = proto.chromosome
        for i in range(option):
            offspring = copy(chromosome)
            gene = offspring[random.randint(0, n - 1)]
            offspring.remove(gene)
            offspring.insert(random.randint(0, n - 1), gene)
            chromosome = offspring
        return Solution(chromosome)

    def get_name():
        return "k-swap"


class ReverseMutation(IMutationOperator):
    def mutate(proto, n, option: int = 1):
        """
        Also known as k-2-OPT mutation. Picks 2 entries and reverses the genes between them. And repeats option(k) times
        """
        chromosome = proto.chromosome
        for l in range(option):
            [i, j] = np.random.randint(low=0, high=n, size=2)
            if i > j:
                i, j = j, i
            chromosome = chromosome[:i] + chromosome[i:j + 1][::-1] + chromosome[j + 1:]  # much slower: return np.append(proto.chromosome[:i - 1], np.append(np.flip(proto.chromosome[i - 1:k]), proto.chromosome[k:]))

        return Solution(chromosome)

    def get_name():
        return "k-reverse"
