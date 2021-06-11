"""
Self-adjusting Genetic Algorithms
"""

# Built-in/Generic Imports
from functools import partial
from copy import copy
import math

# Libs
import numpy as np
from numpy import log as ln
import random
from timeit import default_timer as timer

# Own modules
from . import IGeneticAlgorithm
from problems import IProblem
from common.solution import Solution
from common.results import Result
from common.mutation_operators import KFlipMutation, PFlipMutation, ReverseMutation, KExchangeMutation
from common.crossover_operators import BiasedUniformCrossover, OrderedCrossover


__author__ = "Benjamin Eriksen"


class GABE1(IGeneticAlgorithm):
    """
    Genetic Algorithm
    Self-adaptive (1,(lda1+lda2))
    parameters:
    lda1: number of mutations
    lda2: number of randomly explored solutions
    F: adaptation factor
    r_0: initial mutation rate
    """

    def __init__(self, problem: IProblem, parameters):
        self.problem = problem
        self.parameters = parameters
        if self.problem.encoding == "Binary":
            self.mutation_operator = KFlipMutation
        elif self.problem.encoding == "Permutation":
            self.mutation_operator = ReverseMutation

    def __repr__(self):
        return "GABE 1"

    def run(self, log=None):
        # set hyperparameters
        time_limit = self.parameters["time_limit"]
        t0 = timer()
        goal = self.parameters["goal"]
        mutate = partial(self.mutation_operator.mutate, n=self.problem.problem_size)
        n = self.problem.problem_size
        lda = self.parameters["GABE1_lambda"]
        lda2 = self.parameters["GABE1_lambda2"]
        F = self.parameters["GABE1_F"]
        r_t1 = self.parameters["GABE1_r0"]

        parameters = {"lambda": lda, "lambda2": lda2, "F": F, "r_t1": r_t1, "mutation_operator": self.mutation_operator.get_name()}

        # Internal log
        id = "GABE1"
        if log:
            log.add_listener(id)
            settings = parameters.values()
            log.add_entry(
                id,
                "lambda,lambda2;F;r_0;mutation_operator\n" +
                ";".join([str(val) for val in settings]) + "\n" +
                "iteration;explored;best_fitness;r_{t-1}\n"
            )

        # Generate random solution x0 that will be evolved
        x = self.problem.generate_rnd_sol()
        x.fitness = self.problem.calculate_fitness(x)
        T = 1

        # run iterations
        it = 0
        while True:
            best = copy(x)
            best_r = r_t1
            T += (lda)
            for i in range(0, lda):
                r_ti = random.choice([int(round(r_t1 / F)), int(round(F * r_t1))])  # !! Choose r t,i ∈ {r t −1 /F , Fr t −1 } uniformly at random.
                if not r_ti:
                    r_ti = 1
                x_ti = mutate(x, option=r_ti)
                x_ti.fitness = self.problem.calculate_fitness(x_ti)
                if x_ti >= best or (x_ti == best and r_ti < best_r):
                    best = x_ti
                    best_r = r_ti

            T += (lda2)
            for i in range(0, lda2):
                x_ti = self.problem.generate_rnd_sol()
                x_ti.fitness = self.problem.calculate_fitness(x_ti)

                if x_ti >= best:
                    best = x_ti

            x = best
            r_t1 = int(min([max([F, best_r]), F**math.floor(math.log(n / (2 * F), F))]))  # n / (2 * F)]))
            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    best.fitness,
                    r_t1

                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            if (timer() - t0 > time_limit) or (goal is not None and goal <= x.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= x.fitness, T, x.fitness, parameters, x.chromosome, it, min(timer() - t0, time_limit))
            it += 1


class GABE2(IGeneticAlgorithm):
    """
    Genetic Algorithm
    Self-adaptive (1,(lda1+lda2))
    parameters:
    lda1: number of local mutations
    lda2: number of pseudo global mutations
    F: adaptation factor
    r_0: initial mutation rate
    p: success probability for global mutation step size sample
    """

    def __init__(self, problem: IProblem, parameters):
        self.problem = problem
        self.parameters = parameters
        if self.problem.encoding == "Binary":
            self.mutation_operator = KFlipMutation
            self.global_mutation_operator = KFlipMutation
        elif self.problem.encoding == "Permutation":
            self.mutation_operator = KExchangeMutation
            self.global_mutation_operator = ReverseMutation

    def __repr__(self):
        return "GABE 2"

    def run(self, log=None):
        # set hyperparameters
        time_limit = self.parameters["time_limit"]
        t0 = timer()
        goal = self.parameters["goal"]
        local_mutate = partial(self.mutation_operator.mutate, n=self.problem.problem_size)
        global_mutate = partial(self.global_mutation_operator.mutate, n=self.problem.problem_size)

        n = self.problem.problem_size
        lda = self.parameters["GABE2_lambda"]
        lda2 = self.parameters["GABE2_lambda2"]
        F = self.parameters["GABE2_F"]
        r_t1 = self.parameters["GABE2_r0"]
        p = self.parameters["GABE2_p"]
        parameters = {"lambda": lda, "lambda2": lda2, "F": F, "r_t1": r_t1, "p": p, "local_mutation_operator": self.mutation_operator.get_name(), "global_mutation_operator": self.global_mutation_operator.get_name()}

        # Internal log
        id = "GABE2"
        if log:
            log.add_listener(id)
            settings = parameters.values()
            log.add_entry(
                id,
                "lambda,lambda2;F;r_0;,p,mutation_operator\n" +
                ";".join([str(val) for val in settings]) + "\n" +
                "iteration;explored;best_fitness;r_{t-1}\n"
            )

        # Generate random solution x0 that will be evolved
        x = self.problem.generate_rnd_sol()
        x.fitness = self.problem.calculate_fitness(x)
        T = 1

        # run iterations
        it = 0
        while True:
            best = copy(x)
            best_r = r_t1
            T += (lda)
            for i in range(0, lda):
                r_ti = random.choice([int(round(r_t1 / F)), int(round(F * r_t1))])  # !! Choose r t,i ∈ {r t −1 /F , Fr t −1 } uniformly at random.
                if not r_ti:
                    r_ti = 1
                x_ti = local_mutate(x, option=r_ti)
                x_ti.fitness = self.problem.calculate_fitness(x_ti)
                if x_ti >= best or (x_ti == best and r_ti < best_r):
                    best = x_ti
                    best_r = r_ti
            T += (lda)
            for i in range(0, lda2):
                z = np.random.binomial(n, p)
                x_ti = global_mutate(x, option=z)
                x_ti.fitness = self.problem.calculate_fitness(x_ti)

                if x_ti >= best:
                    best = copy(x_ti)

            r_t1 = int(min([max([F, best_r]), F**math.floor(math.log(n / (2 * F), F))]))  # n / (2 * F)]))
            x = best

            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    best.fitness,
                    r_t1

                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            if (timer() - t0 > time_limit) or (goal is not None and goal <= x.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= x.fitness, T, x.fitness, parameters, x.chromosome, it, min(timer() - t0, time_limit))
            it += 1


class GAAdaptiveMut(IGeneticAlgorithm):
    """
    Genetic Algorithm
    Self-adaptive (1,lambda)
    https://dl.acm.org/doi/10.1145/3205455.3205569
    parameters:
    lda: number of mutations
    F: adaptation factor
    r_0: initial mutation rate
    """

    def __init__(self, problem: IProblem, parameters):
        self.problem = problem
        self.parameters = parameters
        if self.problem.encoding == "Binary":
            self.mutation_operator = PFlipMutation
        elif self.problem.encoding == "Permutation":
            self.mutation_operator = ReverseMutation

    def __repr__(self):
        return "Adaptive Mutation 2021 GA"

    def run(self, log=None):
        # set hyperparameters
        time_limit = self.parameters["time_limit"]
        t0 = timer()
        goal = self.parameters["goal"]
        mutate = partial(self.mutation_operator.mutate, n=self.problem.problem_size)
        n = self.problem.problem_size
        lda = self.parameters["GA_Adaptive_Mutation_lambda"]
        F = self.parameters["GA_Adaptive_Mutation_F"]
        r_t1 = self.parameters["GA_Adaptive_Mutation_r0"]  # r_t1(t_{t-1}) is initially = t_0
        parameters = {"lambda": lda, "F": F, "r_t1": r_t1, "mutation_operator": self.mutation_operator.get_name()}

        # Internal log
        id = "GAAdaptiveMutation"
        if log:
            log.add_listener(id)
            settings = parameters.values()
            log.add_entry(
                id,
                "lambda;F;r_0;mutation_operator\n" +
                ";".join([str(val) for val in settings]) + "\n" +
                "iteration;explored;best_fitness;r_t-1\n"
            )

        # Generate random solution x0 that will be evolved
        x = self.problem.generate_rnd_sol()
        x.fitness = self.problem.calculate_fitness(x)
        T = 1

        # run iterations
        it = 0
        while True:
            best = copy(x)
            best_r = r_t1
            T += (lda)
            for i in range(0, lda):
                r_ti = random.choice([int(round(r_t1 / F)), int(round(F * r_t1))])  # !! Choose r t,i ∈ {r t −1 /F , Fr t −1 } uniformly at random.
                if not r_ti:
                    r_ti = 1
                x_ti = mutate(x, option=r_ti)
                x_ti.fitness = self.problem.calculate_fitness(x_ti)
                if x_ti >= best or (x_ti == best and r_ti < best_r):
                    best = x_ti
                    best_r = r_ti

            x = best
            r_t1 = int(min([max([F, best_r]), F**math.floor(math.log(n / (2 * F), F))]))

            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    best.fitness,
                    r_t1

                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            if (timer() - t0 > time_limit) or (goal is not None and goal <= x.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= x.fitness, T, x.fitness, parameters, x.chromosome, it, min(timer() - t0, time_limit))
            it += 1


class GADynamic(IGeneticAlgorithm):
    """
    Genetic Algorithm
    Dynamic (1+(lamba,lambda))
    dyn(α, β, γ, A, b)
    https://dl.acm.org/doi/10.1145/3321707.3321725
    inspired by: https://github.com/ndangtt/1LLGA/blob/master/source-code/algorithms/algorithms.py
    parameters:
    alpha: the rate of impact of lambda0 on mutation step size (through the selection of z)
    beta: the rate of diffence between lambda0 and lambda2
    gamma: crossover bias factor
    a: change in lambda0 on success
    b: change in lambda0 on failure
    """

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
        return "Dynamic GA"

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
        alpha = self.parameters["GA_dynamic_alpha"]
        beta = self.parameters["GA_dynamic_beta"]
        gamma = self.parameters["GA_dynamic_gamma"]
        a = self.parameters["GA_dynamic_a"]
        b = self.parameters["GA_dynamic_b"]
        mutate = partial(self.mutation_operator.mutate, n=self.problem.problem_size)
        crossover = partial(self.crossover_operator.crossover, n=self.problem.problem_size)
        n = self.problem.problem_size
        parameters = {"alpha": alpha, "beta": beta, "gamma": gamma, "a": a, "b": b, "mutation_operator": self.mutation_operator.get_name(), "crossover_operator": self.crossover_operator.get_name()}

        # Internal log
        id = "GADynamic"
        if log:
            log.add_listener(id)
            settings = parameters.values()
            log.add_entry(
                id,
                "alpha,beta,gamma,a,b,mutation_operator,crossover_operator\n" +
                ",".join([str(val) for val in settings]) + "\n" +
                "iteration,explored,best_fitness,p,c,lambda0,lambda1,lambda2\n"
            )

        # Generate random solution s, that will be evolved
        s = self.problem.generate_rnd_sol()
        s.fitness = self.problem.calculate_fitness(s)
        T = 1

        # probability limits for p and c.
        lambda0 = float(1)
        min_prob = 1.0 / n
        max_prob = 0.99

        # run iterations
        it = 0
        while True:
            # Update interal parameters
            lambda1 = int(round(lambda0))
            lambda2 = int(round(beta * lambda0))
            p = max([min([alpha * lambda0 / float(n), max_prob]), min_prob])
            c = max([min([gamma / lambda0, max_prob]), min_prob])
            # Mutation Phase
            z = np.random.binomial(n, p)
            while z == 0:
                z = np.random.binomial(n, p)
            proto1 = mutate(s, option=z)
            proto1.fitness = self.problem.calculate_fitness(proto1)
            T += 1
            for i in range(int(round(lambda1) - 1)):
                proto2 = mutate(s, option=z)
                proto2.fitness = self.problem.calculate_fitness(proto2)
                T += 1
                if proto2.fitness > proto1.fitness:
                    proto1 = proto2

            # Crossover Phase implementation 1 - performs better than 2 for OneMax, but might not for leadingones
            proto1, Ty = LL_crossover_phase(self.problem, s, proto1, crossover, n, lambda2, c)
            T += Ty
            # Selection step
            if proto1.fitness > s.fitness:
                s = proto1
                lambda0 = max([lambda0 * b, 1])
            if proto1.fitness < s.fitness:
                lambda0 = min([lambda0 * a, n - 1])

            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    s.fitness,
                    p,
                    c,
                    lambda0,
                    lambda1,
                    lambda2,
                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            # Stopping Criterion
            if (timer() - t0 > time_limit) or (goal is not None and goal <= s.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= s.fitness, T, s.fitness, parameters, s.chromosome, it, min(timer() - t0, time_limit))
            it += 1


# R in SD: n < R  (R=n+1)
class SD_RLS(IGeneticAlgorithm):
    """
    RLS with robust stagnation detection (SD-RLS∗)
    Mutation only Genetic Algorithm that features robust stagnation detection
    Algorithm 3 from https://arxiv.org/abs/2101.12054
    parameters:
    R: stagnation detection factor
    """

    def __init__(self, problem: IProblem, parameters):
        self.problem = problem
        self.parameters = parameters

        if self.problem.encoding == "Binary":
            self.mutation_operator = KFlipMutation
        elif self.problem.encoding == "Permutation":
            self.mutation_operator = ReverseMutation

    def __repr__(self):
        return "SD-RLS* GA"  # with " str(self.parameters)

    def run(self, log=None):

        # set hyperparameters
        time_limit = self.parameters["time_limit"]
        t0 = timer()
        goal = self.parameters["goal"]
        R = self.parameters["SD_RLS_R"]
        mutate = partial(self.mutation_operator.mutate, n=self.problem.problem_size)
        n = self.problem.problem_size
        parameters = {"R": R, "mutation_operator": self.mutation_operator.get_name()}
        # Internal log
        id = "GASD"
        if log:
            log.add_listener(id)
            settings = parameters.values()
            log.add_entry(
                id,
                "R;mutation_operator\n" +
                ";".join([str(val) for val in settings]) + "\n" +
                "iteration;explored;best_fitness;r_t-1;r_t;u\n"
            )

        # set internal parameters
        r_t = 1
        s_t = 1
        u = 0

        r_t1 = -1

        # Generate random solution s, that will be evolved
        s = self.problem.generate_rnd_sol()
        s.fitness = self.problem.calculate_fitness(s)
        T = 1

        # run iterations
        it = 0
        while True:
            s1 = mutate(s, option=s_t)
            s1.fitness = self.problem.calculate_fitness(s1)
            T += 1
            u += 1

            # Update based on fitness
            if s1 > s:
                s = s1
                s_t1 = 1
                r_t1 = 1
                u = 0
            elif s1 == s and r_t == 1:
                s = s1

            # Internal log
            if log is not None:
                entry = (
                    it,
                    T,
                    s.fitness,
                    r_t1,
                    r_t,
                    u
                )
                log.add_entry(id, ",".join([str(val) for val in entry]) + "\n")

            # Stopping Criterion
            if (timer() - t0 > time_limit) or (goal is not None and goal <= s.fitness):
                if log is not None:
                    log.remove_listener(id)
                return Result(self.__repr__(), self.problem, goal, goal <= s.fitness, T, s.fitness, parameters, s.chromosome, it, min(timer() - t0, time_limit))
            it += 1

            # Guided search
            if u > math.comb(n, s_t) * ln(R):
                if s_t == 1:
                    if r_t < n / 2:
                        r_t1 = r_t + 1
                    else:
                        r_t1 = n
                    s_t1 = r_t1
                else:
                    r_t1 = r_t
                    s_t1 = s_t - 1
                u = 0
            else:
                s_t1 = s_t
                r_t1 = r_t
            s_t = s_t1
            r_t = r_t1
