class Solution:
    def __init__(self, chromosome, fitness: float = None):
        self.chromosome = chromosome
        self.fitness = fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def _eq__(self, other):
        return self.fitness == other.fitness

    def __repr__(self):
        return str(self.chromosome) + ":" + str(self.fitness)
