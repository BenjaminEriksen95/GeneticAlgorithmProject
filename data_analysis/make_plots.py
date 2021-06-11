import pandas as pd
from plots import plot_problem

# TSP LIB missing!

problems =  ["ThreeSAT","Sorting","SwapSorting" ,"OneMax","LeadingOnes", "JumpM","ThreeSAT_SATLIB","TSP","TSP_LIB"]  #["Sorting"]# #
path = "../results/final/"
for problem in problems:
    df = pd.read_csv(path + problem + ".csv", sep=";")
    plot_problem(df, problem, True) # filename
