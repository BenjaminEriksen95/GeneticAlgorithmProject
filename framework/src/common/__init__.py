"""
    Interfaces
"""

__author__ = "Benjamin Eriksen"


class IMutationOperator:
    def mutate(proto, n, option=None):
        raise NotImplementedError

    def get_name():
        raise NotImplementedError


class ICrossoverOperator:
    def crossover(parent1, parent2, n, option=None):
        raise NotImplementedError

    def get_name():
        raise NotImplementedError


class ILog:
    def __init__(self):
        raise NotImplementedError

    def add_listener(self, id):
        raise NotImplementedError

    def add_entry(self, id):
        raise NotImplementedError


class IResult:
    def __init__(self, algorithm, goal, solved, explored, best_fitness, parameters, chromosome, iterations, time):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class IPersistence:
    def __init__(self, filename: str):
        raise NotImplementedError

    def add_result(self, result: IResult):
        raise NotADirectoryError
