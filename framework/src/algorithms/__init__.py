"""
Interface for Genetic Algorithms
"""

# Own modules
from common import IResult, ILog
from problems import IProblem
from common.mutation_operators import return_mutation_operator
from common.crossover_operators import return_crossover_operator

__author__ = "Benjamin Eriksen"


class IGeneticAlgorithm:
    def __init__(self, problem: IProblem, parameters: dict):
        raise NotImplementedError

    def run(self, log: ILog) -> IResult:
        raise NotImplementedError

    def set_operators(self, mutation_operator=None, crossover_operator=None):
        if mutation_operator is not None:
            self.mutation_operator = return_mutation_operator(mutation_operator)
        if crossover_operator is not None:
            self.crossover_operator = return_crossover_operator(crossover_operator)
        return

    def __repr__(self):
        raise NotImplementedError
