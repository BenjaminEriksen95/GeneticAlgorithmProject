"""
Result objects
"""

# Own modules
from problems import *
from . import IResult

__author__ = "Benjamin Eriksen"


class Result(IResult):
    def __init__(self, algorithm, problem, goal, solved, explored, best_fitness, parameters, chromosome, iterations, time):
        self.algorithm = algorithm
        self.solved = solved
        self.explored = explored
        self.best_fitness = best_fitness
        self.iterations = iterations
        self.parameters = parameters
        self.chromosome = chromosome
        self.optimum = goal
        self.n = problem.problem_size
        self.m = None
        if type(problem) is JumpM:
            self.m = problem.m
        if type(problem) is ThreeSAT:
            self.m = len(problem.problem_instance)
        self.time = time

    def __repr__(self):
        return "{algorithm}: Goal: {optimum}, Solved: {solved}, Best: {best}, Time(s): {time}, O(n)={explored}, chromosome={chromosome}"\
            .format(algorithm=self.algorithm,
                    solved=self.solved,
                    best=self.best_fitness,
                    optimum=self.optimum,
                    time=self.time,
                    explored=self.explored,
                    chromosome=self.chromosome)
