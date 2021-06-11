"""
Framework for Evaluation of Genetic Algorithms by Benjamin Eriksen

Command-line User Interface

Example command:
    "python3 --problem OneMax --size 500 --goal 500 --time_limit 10 --algorithm "[GADynamic, GAStatic, GAStandard]" --sample_size 5"

Several test scripts are available through the Makefile.

Inspirations:
    - Argument parsing inspired by https://github.com/ndangtt/1LLGA/blob/master/source-code/algorithms/main.py
"""

# Built-in/Generic Imports
import sys
import argparse
from copy import copy

# Own modules
import problems
from common import utils
from algorithms.self_adjusting_GAs import *
from algorithms.GAs import *
from common.log import Log
from common.persistence import Persistence
from common.solution import Solution

__author__ = "Benjamin Eriksen"


# Problem
parser = argparse.ArgumentParser(
    description="Framework for Evaluation of Genetic Algorithms by Benjamin Eriksen"
)
parser.add_argument(
    "--problem", help="Available problems: OneMax, LeadingOnes, TSP, JumpM, 3-Sat."
)
parser.add_argument("--size", help="Size of problem.", default=100)
parser.add_argument("--m", help="Used for JumpM and 3Sat problem generation", default=10)

parser.add_argument("--problem_file", help="file to load problem", default=None)

parser.add_argument(
    "--goal", help="Stop when this is reached, if within time limit."
)

parser.add_argument(
    "--algorithm",
    help='Format: ["GAStandard, GAStatic"]; Available: GABE1, GABE2, GAstandard, SD_RLS, GAAdaptiveMut, GAStatic, GADynamic',
)
parser.add_argument(
    "--sample_size", help="Number of runs for each algorithm", default=1
)
parser.add_argument(
    "--time_limit", help="Number of seconds before the algorithm will stop.", default=10
)

parser.add_argument(
    "--internal_log",
    help='True/False', default="False"
)

parser.add_argument("--random_seed")

parser.add_argument(
    "--mutation_operator",
    help="Specify mutation_operator, otherwise problem default is used.",
)

parser.add_argument(
    "--crossover_operator",
    help="Specify crossover_operator, otherwise problem default is used.",
)


##### GA Standard
parser.add_argument("--GA_standard_lambda", help="population size", default=100)
parser.add_argument("--GA_standard_pc", help="population that is not copied directly to the next generation", default=0.20)
parser.add_argument("--GA_standard_pm", help="factor of aggression of the mutation", default=1)

##### GA Static (LL_static)
parser.add_argument("--GA_static_lambda1", help="number of mutations", default=15)
parser.add_argument("--GA_static_lambda2", help="number of crossovers", default=4)
parser.add_argument("--GA_static_k", help="used for probability calculation p = k/n", default=3)
parser.add_argument("--GA_static_c", help="crossover bias", default=0.3)

##### GA Dynamic (LL_dynamic_02)
parser.add_argument("--GA_dynamic_alpha", help="the rate of impact of lambda0 on mutation step size", default=0.7)#0.36)#0.7)
parser.add_argument("--GA_dynamic_beta", help="the rate of diffence between lambda0 and lambda2", default=1.4)#3)
parser.add_argument("--GA_dynamic_gamma", help="crossover bias factor", default=1.24)
parser.add_argument("--GA_dynamic_a", help="change in lambda0 on success", default=1.67)
parser.add_argument("--GA_dynamic_b", help="change in lambda0 on failure", default=0.69)

##### GA SD
parser.add_argument("--SD_RLS_R", help="stagnation detection factor", default=10000)

##### GA Adaptive Mutation
parser.add_argument("--GA_Adaptive_Mutation_lambda", help="number of mutations", default=12)#5)
parser.add_argument("--GA_Adaptive_Mutation_F", help="adaptation factor", default=2.5)#1.2)#2)
parser.add_argument("--GA_Adaptive_Mutation_r0", help="initial mutation rate", default=10)

##### GABE1
parser.add_argument("--GABE1_lambda", help="number of mutations", default=12)
parser.add_argument("--GABE1_lambda2", help="number of pseudo global mutations", default=2)
parser.add_argument("--GABE1_F", help="adaptation factor", default=2)
parser.add_argument("--GABE1_r0", help="initial mutation rate", default=10)

##### GABE2
parser.add_argument("--GABE2_lambda", help="number of mutations", default=12)
parser.add_argument("--GABE2_lambda2", help="", default=2)
parser.add_argument("--GABE2_F", help="adaptation factor", default=2)
parser.add_argument("--GABE2_p", help="success probability for global mutation step size sample", default=0.3)
parser.add_argument("--GABE2_r0", help="initial mutation rate", default=10)


def read_problem(args):
    if args.problem is None or args.size is None:
        raise Exception("Problem or problem size: is unspecified")
    if args.problem == "ThreeSAT":
        if args.problem_file is not None:
            clauses, n = utils.load_3sat("problems/3sat/" + args.problem_file)
            args.goal = len(clauses)
            problem = getattr(problems, args.problem)((n, clauses))
        else:
            problem = getattr(problems, args.problem)(utils.generate_3sat(int(args.size), int(args.m)))
            args.goal = int(args.m)
    elif args.problem == "TSP":
        if args.problem_file is not None:
            name, problem_instance = utils.load_tsp("problems/tsp/" + args.problem_file)
            problem = getattr(problems, args.problem)(problem_instance)
            #
            args.goal = problem.calculate_fitness(Solution(utils.load_tsp_sol("problems/tsp_solutions/" + args.problem_file)))
        else:
            problem = getattr(problems, args.problem)(utils.generate_TSP(int(args.size)))
            args.goal = 0  # if we don't know the goal it is set to 0
    elif args.problem == "JumpM":
        problem = getattr(problems, args.problem)((int(args.size), int(args.m)))
    else:
        problem = getattr(problems, args.problem)(int(args.size))
    if args.algorithm is None:
        raise Exception("No algorithm specified")
    return (problem, args)


def main():

    # Read arguments
    args = parser.parse_args()
    problem, args = read_problem(args)

    log = None
    if eval(args.internal_log):
        log = Log()

    # eval for list
    algorithms = eval(args.algorithm)
    parameters = dict()

    ## GAStandard parameters
    if "GAStandard" in args.algorithm:
        print("Population size:", args.GA_standard_lambda)
        parameters["GA_standard_lambda"] = int(args.GA_standard_lambda)
        print("GA_standard_pc:", args.GA_standard_pc)
        parameters["GA_standard_pc"] = float(args.GA_standard_pc)
        print("GA_standard_pm:", args.GA_standard_pm)
        parameters["GA_standard_pm"] = int(args.GA_standard_pm)

    ## GAStatic parameters
    if "GAStatic" in args.algorithm:
        print("GA_static_lambda1:", args.GA_static_lambda1)
        parameters["GA_static_lambda1"] = int(args.GA_static_lambda1)
        print("GA_static_lambda2:", args.GA_static_lambda2)
        parameters["GA_static_lambda2"] = int(args.GA_static_lambda2)
        print("GA_static_k:", args.GA_static_k)
        parameters["GA_static_k"] = float(args.GA_static_k)
        print("GA_static_c:", args.GA_static_c)
        parameters["GA_static_c"] = float(args.GA_static_c)

    ## GA dynamic parameters
    if "GADynamic" in args.algorithm:
        print("GA_dynamic_alpha:", args.GA_dynamic_alpha)
        parameters["GA_dynamic_alpha"] = float(args.GA_dynamic_alpha)
        print("GA_dynamic_beta:", args.GA_dynamic_beta)
        parameters["GA_dynamic_beta"] = float(args.GA_dynamic_beta)
        print("GA_dynamic_gamma:", args.GA_dynamic_gamma)
        parameters["GA_dynamic_gamma"] = float(args.GA_dynamic_gamma)
        print("GA_dynamic_a:", args.GA_dynamic_a)
        parameters["GA_dynamic_a"] = float(args.GA_dynamic_a)
        print("GA_dynamic_b:", args.GA_dynamic_b)
        parameters["GA_dynamic_b"] = float(args.GA_dynamic_b)

    if "SD_RLS" in args.algorithm:
        print("SD_RLS_R:", args.SD_RLS_R)
        parameters["SD_RLS_R"] = int(args.SD_RLS_R)

    if "GAAdaptiveMut" in args.algorithm:
        print("GA_Adaptive_Mutation_lambda:", args.GA_Adaptive_Mutation_lambda)
        parameters["GA_Adaptive_Mutation_lambda"] = int(args.GA_Adaptive_Mutation_lambda)
        print("GA_Adaptive_Mutation_F:", args.GA_Adaptive_Mutation_F)
        parameters["GA_Adaptive_Mutation_F"] = float(args.GA_Adaptive_Mutation_F)
        print("GA_Adaptive_Mutation_r0:", args.GA_Adaptive_Mutation_r0)
        parameters["GA_Adaptive_Mutation_r0"] = int(args.GA_Adaptive_Mutation_r0)

    if "GABE1" in args.algorithm:
        print("GABE1_lambda:", args.GABE1_lambda)
        parameters["GABE1_lambda"] = int(args.GABE1_lambda)
        print("GABE1_lambda2:", args.GABE1_lambda2)
        parameters["GABE1_lambda2"] = int(args.GABE1_lambda2)
        print("GABE1_F:", args.GABE1_F)
        parameters["GABE1_F"] = float(args.GABE1_F)
        print("GABE1_r0:", args.GABE1_r0)
        parameters["GABE1_r0"] = int(args.GABE1_r0)

    if "GABE2" in args.algorithm:
        print("GABE2_lambda:", args.GABE2_lambda)
        parameters["GABE2_lambda"] = int(args.GABE2_lambda)
        print("GABE2_lambda2:", args.GABE2_lambda2)
        parameters["GABE2_lambda2"] = int(args.GABE2_lambda2)
        print("GABE2_F:", args.GABE2_F)
        parameters["GABE2_F"] = float(args.GABE2_F)
        print("GABE2_p:", args.GABE2_p)
        parameters["GABE2_p"] = float(args.GABE2_p)
        print("GABE2_r0:", args.GABE2_r0)
        parameters["GABE2_r0"] = int(args.GABE2_r0)

    parameters["time_limit"] = int(args.time_limit)
    print("time_limit: ", args.time_limit, "sec")

    assert args.goal is not None
    parameters["goal"] = int(args.goal)
    print("goal: ", args.goal)

    output = Persistence(args.problem + ".csv")

    for alg in algorithms:
        for i in range(int(args.sample_size)):
            # copy problem because static and dynamic overwrites crossover operator
            ea = alg(copy(problem), parameters)

            # Set mutation or crossover operators
            ea.set_operators(args.mutation_operator, args.crossover_operator)

            if args.problem_file is not None:
                print(f"Computing {ea} sample {i+1} on", args.problem_file, "...")
            else:
                print(f"Computing {ea} sample {i+1} ...")
            result = ea.run(log)
            print(result)

            output.add_result(result)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
