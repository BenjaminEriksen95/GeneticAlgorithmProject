import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = "../results/plots/"

def col_vs_problem_size(dataset, col, problem, export):
    g = None
    if problem == "JumpM":
        g = sns.FacetGrid(dataset.loc[(dataset['solved'] == True)], hue="algorithm", col="m", height=10, col_wrap=2)
    else:
        g = sns.FacetGrid(dataset.loc[(dataset['solved'] == True)], hue="algorithm", height=10)
    g.map(sns.scatterplot, "n", col)
    g.map(sns.lineplot, "n", col)
    if col == "explored":
        g.set_axis_labels('problem size', 'solutions explored')
    elif col == "time":
        g.set_axis_labels('problem size', 'time [sec]')
    else:
        raise NotImplementedError

    g.add_legend()

    if export:
        g.savefig(path + problem + "_" + col + ".svg", format='svg', dpi=1200)
        plt.close()
    else:
        plt.show()


def mean_score(dataset, problem, export):
    df = None
    if problem == "JumpM":
        df = dataset.groupby(["algorithm", 'n', 'goal'])[['best_fitness', 'm']].agg(['mean'])
        df.columns = ["_".join(x) for x in df.columns]
        df = df.reset_index()
        df['m'] = df['m_mean']
    else:
        df = dataset.groupby(["algorithm", 'n', 'goal'])[['best_fitness']].agg(['mean'])
        df.columns = ["_".join(x) for x in df.columns]
        df = df.reset_index()

    df['abs_diff'] = df['goal'] - df['best_fitness_mean']
    g = None
    if problem == "JumpM":
        g = sns.FacetGrid(df, hue="algorithm", col="m", height=5, col_wrap=2, legend_out=True, hue_kws={"ls" : ["-","--","-.",":","-","--","-."]})
        g.map(sns.lineplot, "n", 'abs_diff')
    else:
        g = sns.FacetGrid(df, hue='algorithm', height=10)
        g.map(sns.lineplot, "n", 'abs_diff')

    g.set_axis_labels('problem size', 'average diffence from optimum')
    g.add_legend()

    if export:
        g.savefig(path + problem + "_average" + ".svg", format='svg', dpi=1200)
        plt.close()
    else:
        plt.show()



def mean_score_tsp(dataset, problem, export):
    # Calculates difference from the best score for each n, regardless of which algorithm found it!
    df = dataset.groupby(["algorithm", 'n', 'goal'])[['best_fitness']].agg(['mean'])
    best = dataset.groupby(['n'])[['best_fitness']].agg(['max'])
    best.columns = ["_".join(x) for x in best.columns]
    best = best.reset_index()
    df.columns = ["_".join(x) for x in df.columns]
    df = df.reset_index()
    df = pd.merge(left=df, right=best, left_on='n', right_on='n')

    df['delta'] = df['best_fitness_max'] - df['best_fitness_mean']

    g = sns.FacetGrid(df, hue='algorithm', height=10)
    g.map(sns.lineplot, "n", 'delta')

    g.set_axis_labels('problem size', 'average diffence from optimum')
    g.add_legend()
    if export:
        g.savefig(path + problem + "_average" + ".svg", format='svg', dpi=1200)
        plt.close()
    else:
        plt.show()

def plot_problem(df, problem, export=False):
    if problem == "TSP":
        mean_score_tsp(df, problem, export)
    else:
        col_vs_problem_size(df, "time", problem, export)
        col_vs_problem_size(df, "explored", problem, export)
        mean_score(df, problem, export)
