"""
Non Self-adjusting Genetic Algorithms
"""

# Built-in/Generic Imports
import random
from functools import partial
from copy import copy
from timeit import default_timer as timer

# Libs
import numpy as np

# Own modules
from . import IGeneticAlgorithm
from problems import IProblem
from common.results import Result
from common.solution import Solution
from common.mutation_operators import PFlipMutation, KFlipMutation, ReverseMutation
from common.crossover_operators import BiasedUniformCrossover, OrderedCrossover

__author__ = "Benjamin Eriksen"


class GAStatic(IGeneticAlgorithm):
    """
    Genetic Algorithm
    Static (1+(lamba,lambda))
    Crossover implementation 3?
    https://dl.acm.org/doi/10.1145/3321707.3321725
    inspired by: https://github.com/ndangtt/1LLGA/blob/master/source-code/algorithms/algorithms.py
    parameters:
    lambda1: number of mutations
    lamdba2: number of crossovers
    k: used for probability calculation p = k/n
    c: crossover bias
    """

    # Todo: upgrade crossover phase to ?
    def __init__(self, problem: IProblem, parameters):
        self.problem = problem
        self.parameters = parameters

        if self.problem.encoding == "Binary":
            self.mutation_operator = KFlipMutation
            self.crossover_operator = BiasedUniformCrossover
        elif self.problem.encoding == "Permutation":
            self.mutation_operator = ReverseMutation
            self.crossover_operator = OrderedCrossover

    def __repr__(self):
        return "Static GA"  # with " str(self.parameters)

    def run(self, log=None):
        def LL_crossover_phase(problem: IProblem, s0: Solution, proto: Solution, crossover, chromosome_length: int, lambda2: int, c: float):
            T = 0
            s1 = copy(proto)

            for i in range(lambda2):
                offspring = crossover(s0, proto, option=c)
                offspring.fitness = problem.calculate_fitness(offspring)
                T += 1
                if offspring > s1:
                    s1 = offspring
            return s1, T
        # set hyperparameters
        time_limit = self.parameters["time_limit"]
        t0 = timer()
        goal = self.parameters["goal"]
        lambda1 = self.parameters["GA_static_lambda1"]
        lambda2 = self.parameters["GA_static_lambda2"]
        n = self.problem.problem_size
        p = min(self.parameters["GA_static_k"] / float(n), 0.99)
        c = self.parameters["GA_static_c"]
        mutate = partial(self.mutation_operator.mutate, n=n)
        crossover = partial(self.crossover_operator.crossover, n=n)

        parameters = {"lambda1": lambda1, "lambda2": lambda2, "p": p, "c": c, "mutation_operator": self.mutation_operator.get_name(), "crossover_operators": self.crossover_operator.get_name()}

        # Internal log
        id = "GAStatic"
        if log:
            log.add_listener(id)
            settings = (lambda1, lambda2, p, c)
            log.add_entry(
                id,
                "lambda1;lambda2;p;c\n" +
                ";".join([str(val) for val in settings]) + "\n" +
                "iteration;explored;best_fitness\n"
            )

        # Generate random solution s, that will be evolved
        s = self.problem.generate_rnd_sol()
        s.fitness = self.problem.calculate_fitness(s)
        T = 1

        # run iterations
        it = 0
        while True:
            # Mutation Phase
            # Sample mutation rate k from Bin_{>0}(n,p)
            z = 0
            while z == 0:
                z = np.random.binomial(n, p)

            proto1 = mutate(s, option=z)
            proto1.fitness = self.problem.calculate_fitness(proto1)
            T += 1
            for i in range(lambda1 - 1):
                proto2 = mutate(s, option=z)
                proto2.fitness = self.problem.calculate_fitness(proto2)
                T += 1
                if proto2 > proto1:
                    proto1 = proto2

            # Crossover Phase
            proto1, Ty = LL_crossover_phase(self.problem, s, proto1, crossover, n, lambda2, c)
            T += Ty

            # Selection step
            if proto1 > s:
                s = proto1

            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    s.fitness,
                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            # Stopping Criterion
            if (timer() - t0 > time_limit) or (goal is not None and goal <= s.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= s.fitness, T, s.fitness, parameters, s.chromosome, it, min(timer() - t0, time_limit))
            it += 1


## TODO: run specified tournament size where pool fills when empty so that population size is maintained?
# input: t: number of tournaments, ts: tournament_size
# if t*ts <= |population|
#       pick t*ts samples without repetition
# else
#       pick |population| samples without repetition, and t*ts-|population| samples with repetition.


class GAStandard(IGeneticAlgorithm):
    """
    Standard GA Multi Population (µ+(1,1))
    with static mutation and crossover rates
    https://link.springer.com/article/10.1007/BF02823145#citeas
    parameters:
    lambda: population size
    pm: factor of aggression of the mutation
    pc: population that is not copied directly to the next generation
    """

    def __init__(self, problem: IProblem, parameters):
        self.problem = problem
        self.parameters = parameters

        if self.problem.encoding == "Binary":
            self.mutation_operator = PFlipMutation
            self.crossover_operator = BiasedUniformCrossover
        elif self.problem.encoding == "Permutation":
            self.mutation_operator = ReverseMutation
            self.crossover_operator = OrderedCrossover

    def __repr__(self):
        return "Standard GA"

    def run(self, log=None):
        # set hyperparameters
        time_limit = self.parameters["time_limit"]
        t0 = timer()
        goal = self.parameters["goal"]
        population_size = self.parameters["GA_standard_lambda"]
        non_elite_size = int(population_size * self.parameters["GA_standard_pc"])
        pm = self.parameters["GA_standard_pm"]
        elite_size = population_size - non_elite_size
        population_t1 = list()  # Population 0
        n = self.problem.problem_size
        mutate = partial(self.mutation_operator.mutate, n=n)
        crossover = partial(self.crossover_operator.crossover, n=n)
        parameters = {"population_size": population_size, "tournament_size": 2, "pc": self.parameters["GA_standard_pc"], "pm": self.parameters["GA_standard_pm"], "mutation_operator": self.mutation_operator.get_name(), "crossover_operators": self.crossover_operator.get_name()}
        best = None
        T = population_size * 2

        # Internal log
        id = "GAStandard"
        if log:
            log.add_listener(id)
            settings = parameters.values()
            log.add_entry(
                id,
                "population_size;pc;pm;mutation_operator;crossover_operator\n" +
                ";".join([str(val) for val in settings]) + "\n" +
                "iteration;explored;best_fitness\n"
            )

        # generate initial population
        for _ in range(population_size * 2):
            s = self.problem.generate_rnd_sol()
            s.fitness = self.problem.calculate_fitness(s)
            population_t1.append(s)

        it = 0
        while True:
            # compute fitness
            T += len(population_t1) - elite_size
            best = max(population_t1)

            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    best.fitness,

                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            # Stopping Criterion
            if (timer() - t0 > time_limit) or (goal is not None and goal <= best.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= best.fitness, T, best.fitness, parameters, best.chromosome, it, min(timer() - t0, time_limit))
            it += 1

            # selection
            survivors = list()

            ## run tournaments
            shuffled_population = random.sample(population_t1, population_size)
            for i in range(population_size):
                survivors.append(max(population_t1[i], shuffled_population[i]))

            s = sorted(survivors)
            elite = s[non_elite_size:]
            # next generation
            population_t1 = copy(elite)

            # crossover
            protos = list()
            shuffled_pop1 = random.sample(survivors, non_elite_size)
            shuffled_pop2 = random.sample(survivors, non_elite_size)
            ## merge
            for i in range(non_elite_size):
                proto1 = crossover(shuffled_pop1[i], shuffled_pop2[i])
                proto2 = crossover(shuffled_pop2[i], shuffled_pop1[i])
                proto1.fitness = self.problem.calculate_fitness(proto1)
                proto2.fitness = self.problem.calculate_fitness(proto2)
                protos.append(proto1)
                protos.append(proto2)

            # mutate and add offsprings to next generation

            for proto in elite:
                s = mutate(proto, option=pm)
                s.fitness = self.problem.calculate_fitness(s)
                population_t1.append(s)

            for proto in protos:
                s = mutate(proto, option=pm)
                s.fitness = self.problem.calculate_fitness(s)
                population_t1.append(s)

            assert len(population_t1) == 2 * population_size
