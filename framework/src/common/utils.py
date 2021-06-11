"""
Result objects
"""

# Built-in/Generic Imports
import csv
import random

# Libs
import numpy as np


__author__ = "Benjamin Eriksen"


def load_3sat(filename: str):
    """
        Loads 3-Sat instances of the DIMACS cnf format, as those found on:
        https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html
        Inspired by https://stackoverflow.com/questions/28890268/parse-dimacs-cnf-file-python
    """
    filename = filename
    in_data = open(filename, "r")

    cnf = list()
    cnf.append(list())
    maxvar = 0

    for line in in_data:
        tokens = line.split()
        if len(tokens) != 0 and tokens[0] not in ("p", "c"):
            for tok in tokens:
                if tok == '%':
                    break

                lit = int(tok)
                maxvar = max(maxvar, abs(lit))
                if lit == 0:
                    cnf.append(list())
                else:
                    cnf[-1].append(lit)

    while len(cnf[-1]) == 0:
        cnf.pop()

    return (cnf, maxvar)


def load_tsp(filename: str):
    """
        Loads TSP instances of the ??? format, as those found on:
        http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/
        Inspired by: https://github.com/ecoslacker/yaaco/blob/master/yaaco.py
    """
    name = ""
    filename = filename
    coordinates = []
    read = False  # Token that indicates when to read the coordinates

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if row[0] == 'NAME:':
                name = row[1]
            if row[0] == 'NODE_COORD_SECTION':
                read = True
            elif row[0] == 'EOF':
                break
            elif read is True:
                coordinates.append((float(row[1]), float(row[2])))
    return name, coordinates


def load_tsp_sol(filename: str):
    filename = filename.split('.')[0] + ".opt.tour"
    locations = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        read = False
        for row in reader:
            if row[0] == 'TOUR_SECTION':
                read = True
            elif row[0] == '-1':
                break
            elif read is True:
                locations.append(int(row[0]) - 1)
    return locations


def generate_3sat(n: int, m: int):
    # n = variables
    # m = clauses
    s0 = np.random.randint(low=0, high=2, size=n)
    clauses = []
    for i in range(1, m + 1):
        i1, i2, i3 = random.sample(range(1, n + 1), 3)
        x1 = i1 if s0[i1 - 1] else -1 * i1
        x2 = i2
        x3 = i3
        clauses.append((x1, x2, x3))

    return (n, clauses)


def generate_TSP(n):
    # Generates a random problem with no known optimal solution
    locations = dict()
    for i in range(n):
        locations[i] = np.random.randint(low=0, high=100, size=2)
    return locations
