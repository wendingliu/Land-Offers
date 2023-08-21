"""Show the fit of the model by comparing empirical cdf 
   and descriptive statistics of observed data and simulated data.
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Modules.solver import Params, solver_creator, Supervisor
from Modules.simulator import closure_draw_short_lists, closure_sim_prices

# load data
from estimation import firms, cities, distance_matrix, all_obs_prices


def f_sim_sample_prices(params, S=30):
    """a function simulating land price for each firm for given parameters"""
    # solver function of nash equilibrium
    f_ne = solver_creator(margin=2000, tol=1e-10, max_iter=10_000)

    f_draw_short_lists = closure_draw_short_lists(
        distance_matrix,
        pool_size=10,
        min_size=3,
        max_size=5,
        seed=42,
    )

    f_sim_prices, all_short_lists, all_wages = closure_sim_prices(
        firms=firms,
        cities=cities,
        Sup=Supervisor,
        ne=f_ne,
        draw_short_lists=f_draw_short_lists,
        S=S,
        fix_obs=False,
    )

    all_sim_prices, all_sim_probs = f_sim_prices(params, prob=True)
    N = len(all_sim_prices) // S
    sim_sample_prices = np.zeros(N)

    for n in range(N):
        for s in range(S):
            sim_sample_prices[n] += (1 / S) * np.sum(
                all_sim_prices[n + s * N] * all_sim_probs[n + s * N]
            )

    return sim_sample_prices


def em_cdf(data, fname=None, xlabel=None, label_list=None):
    """draw the empirical cdf of an array x
    data: a list of data arrays.
    fname: the file name of the figure
    xlabel: the label of x axis
    ax_titles: the list of titles for each axis of the plot
    """
    default_cycle = itertools.cycle(["r-", "b--", "g:", "y-."])

    for x, label in zip(data, label_list):  # draw cdf for each array in data
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        # change x to the array of step wise cdf values
        x2 = np.empty(2 * len(x))
        y2 = np.empty(2 * len(y))
        for i in range(len(x)):
            x2[2 * i + 1] = x[i]
            x2[2 * i] = x[i]
            if i == 0:
                y2[2 * i] = 0
                y2[2 * i + 1] = y[i]
            else:
                y2[2 * i] = y2[2 * i - 1]
                y2[2 * i + 1] = y[i]
        sns.set_theme()
        plt.plot(x2[x2 > 0 + 0.01], y2[x2 > 0 + 0.01], next(default_cycle), label=label)
        plt.xlabel(xlabel)
        plt.ylabel("cdf")
        plt.legend()
        plt.tight_layout()
    if fname:  # draw single cdf curve
        plt.savefig("Graphs/{}.pdf".format(fname))
    plt.close()


def main():
    sim_sample_prices = f_sim_sample_prices(Params(0.5, 0.45, 178), S=30)
    # draw empirical cdf
    em_cdf(
        [all_obs_prices, sim_sample_prices],
        xlabel="land price (1,000 yuan/hectare)",
        label_list=["data", "simulation"],
        fname="cdf_data_sim",
    )
    # descriptive statistics
    df = pd.DataFrame(
        {"observed price": all_obs_prices, "simulated price": sim_sample_prices}
    )
    description = df.describe()
    description.drop("count", inplace=True)
    s = description.style
    s.format({"observed price": "{:.3f}", "simulated price": "{:.3f}"})
    s.to_latex(
        buf="Tables/fit_of_model.tex",
        column_format="lcc",
        position="H",
        position_float="centering",
        hrules=True,
        label="table: fit_of_model",
        caption="Descriptive Statistics of Observed Prices and Simulated Prices",
        multirow_align="t",
        multicol_align="r",
    )


if __name__ == "__main__":
    main()
