"""
Optimization Problems
"""

# Libs
import random
import numpy as np

# Own modules
from common.solution import Solution


__author__ = "Benjamin Eriksen"


# Problem Interface
class IProblem:
    def __init__(self, problem_instance):
        raise NotImplementedError

    def calculate_fitness(self, s: Solution):
        raise NotImplementedError

    def generate_rnd_sol(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError



class OneMax(IProblem):
    ''''
    OneMax
    Maximize the number of ones in a bit string
    h = argmax_{x \in {0,1}^n}(f(x)=\sum_{i=1}^n x_i)
    '''

    def __init__(self, problem_instance):
        self.encoding = "Binary"
        self.problem_size = problem_instance

    def calculate_fitness(self, s):
        return np.sum(s.chromosome)

    def generate_rnd_sol(self):
        return Solution(np.random.randint(low=0, high=2, size=self.problem_size))

    def __repr__(self):
        return "OneMax(" + str(self.problem_instance) + ")"


class LeadingOnes(IProblem):
    '''
    LeadingOnes
    Maximize the number of leading ones in a bit string
    h = argmax_{x \in {0,1}^n}(f(x)=\sum_{i=1}^n \prod_{j=1}^i x_j)
    '''

    def __init__(self, problem_instance):
        self.encoding = "Binary"
        self.problem_size = problem_instance

    def calculate_fitness(self, s):
        h = 0
        for i in range(1, len(s.chromosome) + 1):
            h += np.prod(s.chromosome[:i])
        return h

    def generate_rnd_sol(self):
        return Solution(np.random.randint(low=0, high=2, size=self.problem_size))

    def __repr__(self):
        return "LeadingOnes(" + str(self.problem_instance) + ")"


class JumpM(IProblem):
    ''''
    Jump-m
    Maximize the number of ones in a bit string, where an m sized gap in fitness exists. Maximum value for a problem is n+m
    |x|_1: number of ones in bit string
    if |x|_1 <= n-m or |x|_1=n -> h = m + |x|_1
    else -> h = n - |x|_1
    '''

    def __init__(self, problem_instance):
        self.encoding = "Binary"
        self.problem_size = problem_instance[0]
        self.m = problem_instance[1]

    def calculate_fitness(self, s):
        no_1 = np.sum(s.chromosome)
        if no_1 <= self.problem_size - self.m or no_1 == self.problem_size:
            return self.m + no_1
        else:
            return self.problem_size - no_1

    def generate_rnd_sol(self):
        return Solution(np.random.randint(low=0, high=2, size=self.problem_size))

    def __repr__(self):
        return "Jump" + str(self.m) + "(" + str(self.problem_size) + ")"


class ThreeSAT(IProblem):
    '''

    '''

    def __init__(self, problem_instance):
        self.encoding = "Binary"
        self.problem_size = problem_instance[0]
        self.problem_instance = problem_instance[1]

    def calculate_fitness(self, s):
        h = 0
        for clause in self.problem_instance:
            truth = False
            for x in clause:
                if s.chromosome[abs(x) - 1] == (x > 0):
                    truth = True
            if truth:
                h += 1
        return h

    def generate_rnd_sol(self):
        return Solution(np.random.randint(low=0, high=2, size=self.problem_size))

    def __repr__(self):
        return "3-Sat"


class TSP(IProblem):
    '''
    TravelingSalesman Problem
    Find the shortest route that reaches all cities
    h = ..
    '''

    def __init__(self, problem_instance):
        # https://www.hindawi.com/journals/cin/2017/7430125/
        self.encoding = "Permutation"
        self.problem_instance = problem_instance
        self.problem_size = len(self.problem_instance)

    def calculate_fitness(self, s):
        h = 0
        size = len(s.chromosome)
        for i in range(size - 1):
            h += np.linalg.norm([self.problem_instance[s.chromosome[i]], self.problem_instance[s.chromosome[(i + 1) % (size)]]])
        return -1 * round(h, 4)

    def generate_rnd_sol(self):
        return Solution(random.sample(range(0, self.problem_size), self.problem_size))

    def __repr__(self):
        return str(self.problem_instance)


class Sorting(IProblem):
    '''
    Sorting Problem
    Sort the list of n items from smallest to largest
    h =
    '''

    def __init__(self, problem_instance):
        self.encoding = "Permutation"
        self.problem_size = problem_instance

    def calculate_fitness(self, s):
        # Used INV heuristic: count number of item pairs that are in order.
        h = 0
        n = len(s.chromosome)
        for i in range(n - 1):
            h += 0 if s.chromosome[i] <= s.chromosome[i + 1] else 1
        return n - h

    def generate_rnd_sol(self):
        return Solution(random.sample(range(0, self.problem_size), self.problem_size))

    def __repr__(self):
        return str(self.problem_instance)
