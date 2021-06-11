from . import IResult, IPersistence

seperator = ";"

class Persistence(IPersistence):
    def __init__(self, filename: str):
        self.output_file = open("results/" + filename, "at")
        self.output_file.seek(0, 2)
        if self.output_file.tell() == 0:
            entry = ("algorithm", "goal", "solved", "best_fitness", "explored", "iterations", "time", "n", "m", "parameters")
            self.output_file.write(seperator.join([str(val) for val in entry]) + "\n")

    def add_result(self, result: IResult):
        entry = (result.algorithm, result.optimum, result.solved, result.best_fitness, result.explored, result.iterations, result.time, result.n, result.m, result.parameters)
        self.output_file.write(seperator.join([str(val) for val in entry]) + "\n")
