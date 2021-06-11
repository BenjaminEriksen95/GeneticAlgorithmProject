import pandas as pd


def pd_drop_row(name, row):
    df = pd.read_csv(name + ".csv", sep=";")
    df.drop(df.index[(df['algorithm'] == row)], axis=0, inplace=True)
    df.to_csv(name + ".csv", sep=";", index=False)


def pd_sort(name):
    df = pd.read_csv(name + ".csv", sep=";")
    df.sort_values(['algorithm', 'n'], axis=0, inplace=True)
    df.to_csv(name + ".csv", sep=";", index=False)

#  for file in ["ThreeSAT","Sorting","SwapSorting" ,"OneMax","LeadingOnes", "JumpM","ThreeSAT_SATLIB","TSP","TSP_LIB"]:
#       pd_sort(file)
