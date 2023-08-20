"""Counterfactual experiments"""

import ray
import numpy as np
import pandas as pd
from numba.typed import List
import Modules.solver as solver
from Modules.solver import City, Firm, Params, solver_creator, Supervisor
from Modules.simulator import closure_draw_short_lists, closure_sim_prices
from model_fitness import em_cdf
from estimation import firms, cities, distance_matrix, all_obs_prices  # load data


# a generator of land price simulator
def creator_f_sim_prices(cities, S=30):
    """Generator of f_sim_prices, all_short_lists, all_wages.
    cities: the data class of all cities, we need this argument for the counterfactual
            of rising wage level.
    S: number of simulations. (30 by default)
    """
    # NE solver
    f_ne = solver_creator(margin=2000, tol=1e-10, max_iter=10_000)
    # draw short list function
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
    return f_sim_prices, all_short_lists, all_wages


# a simulator of all sample prices and land selling revenue given params and cities
def f_sim_sample_prices(params, cities, S=30):
    """a function simulating land price and land selling revenue for each firm for given parameters"""
    try:
        ray.shutdown()
    except:
        pass
    f_sim_prices, *_ = creator_f_sim_prices(cities, S=30)

    all_sim_prices, all_sim_probs = f_sim_prices(params, prob=True)
    N = len(all_sim_prices) // S
    sim_sample_prices = np.zeros(N)
    sim_sample_land_revenue = np.zeros(N)

    for n in range(N):
        for s in range(S):
            sim_sample_prices[n] += (1 / S) * np.sum(
                all_sim_prices[n + s * N] * all_sim_probs[n + s * N]
            )
        sim_sample_land_revenue[n] = sim_sample_prices[n] * firms[n].T

    return sim_sample_prices, sim_sample_land_revenue


# 1. allocation efficiency
def allocation(params, S=30):
    """compare the distribution of highest prob
    under fixed land price case and Bertrand game by 30 simulations.

    params: the estimates of parameters (dataclass).
    """
    f_sim_prices, all_short_lists, all_wages = creator_f_sim_prices(cities)
    # simulated land prices and probs for the whole sample under Bertrand Game
    all_sim_prices1, all_sim_probs1 = f_sim_prices(Params(0.5, 0.45, 178), prob=True)

    # simulated probs for the whole sample under fixed land price
    all_sim_probs2 = []
    for firm, wages in zip(S * list(firms), all_wages):
        all_sim_probs2.append(
            solver.prob(
                firm, cities[firm.c], params, wages, prices=np.zeros(len(wages))
            )
        )
    # cities with highest prob to win in the cases of banning all fiscal competitions
    winners = np.array([np.argmax(probs) for probs in all_sim_probs2])
    winners_probs = np.array([np.max(probs) for probs in all_sim_probs2])
    # original winners' probs in Bertrand games
    bertrand_winner_probs = np.array(
        [probs[winner] for (probs, winner) in zip(all_sim_probs1, winners)]
    )

    # compare empirical cdf
    em_cdf(
        [bertrand_winner_probs, winners_probs],
        label_list=["Bertrand game", "Fixed land price"],
        xlabel="probability",
        fname="allocation_efficiency",
    )

    # compare descriptive statistics
    df1 = pd.DataFrame(
        {"fixed land price": winners_probs, "Bertrand game": bertrand_winner_probs}
    )
    des1 = df1.describe()
    des1["difference"] = des1["fixed land price"] - des1["Bertrand game"]
    des1.drop("count", inplace=True)
    des1.index.name = "probability"
    des1.index = [
        "mean",
        "std",
        "min",
        "25\% percentile",
        "median",
        "75\% percentile",
        "max",
    ]
    s1 = des1.style
    s1.format(
        {
            "fixed land price": "{:.3f}",
            "Bertrand game": "{:.3f}",
            "difference": "{:.3f}",
        }
    )
    s1.to_latex(
        buf="Tables/allocation_efficiency.tex",
        column_format="lccc",
        position="H",
        position_float="centering",
        hrules=True,
        label="table: allocation_efficiency",
        caption="Empirical distribution of probability of getting the firms for advantaged cities",
        multirow_align="t",
        multicol_align="r",
    )


# 2. impacts of fiscal centralization
def decrease_beta(alpha=0.5, beta=0.45, sigma=178, S=30):
    beta_list = [
        beta,
        beta * 0.9,
        beta * 0.75,
        beta * 0.5,
    ]
    total_output = np.sum([firm.Y for firm in firms])
    average_land_price = np.empty(4)
    total_land_revenue = np.empty(4)
    total_profit_share = np.empty(4)
    total_fiscal_revenue = np.empty(4)
    all_sim_sample_prices = []

    for i, new_beta in enumerate(beta_list):
        params = Params(alpha, new_beta, sigma)
        sim_sample_prices, sim_sample_land_revenue = f_sim_sample_prices(
            params, cities, S
        )
        all_sim_sample_prices.append(sim_sample_prices)
        average_land_price[i] = np.mean(sim_sample_prices)
        total_land_revenue[i] = np.sum(sim_sample_land_revenue)
        total_profit_share[i] = new_beta * total_output
        total_fiscal_revenue[i] = total_profit_share[i] + total_land_revenue[i]

    # Calculate the change ratio of elements of x[1:] w.r.t x[0].
    change_ratio = lambda x: (x - x[0])[1:] / x[0]

    # compare empirical cdf
    em_cdf(
        all_sim_sample_prices,
        label_list=[r"$\beta$={:.3f}".format(beta) for beta in beta_list],
        xlabel="land price (1,000 yuan/hectare)",
        fname="cdf_beta_decrease",
    )
    # make table
    df = pd.DataFrame(
        {
            r"$\beta$": beta_list[1:],
            r"change of $\beta$": ["-10%", "-25%", "-50%"],
            r"average land price": change_ratio(average_land_price),
            r"total land selling revenue": change_ratio(total_land_revenue),
            r"total fiscal revenue": change_ratio(total_fiscal_revenue),
        }
    )
    s = df.style
    s.hide(axis="index")
    s.format(
        {
            r"$\beta$": "{:.3f}",
            "average land price": "{:.2%}",
            "total land selling revenue": "{:.2%}",
            "total fiscal revenue": "{:.2%}",
        },
    )
    s.to_latex(
        buf="Tables/decrease_beta.tex",
        column_format="ccccc",
        position="H",
        position_float="centering",
        hrules=True,
        label="table: decrease_beta",
        caption=r"The impacts of decrease in $\beta$",
        multirow_align="t",
        multicol_align="r",
    )
    with open("Tables/decrease_beta.tex", "r") as f:
        f_text = f.read()

    f_text = f_text.replace("%", "\%")
    with open("Tables/decrease_beta.tex", "w") as f:
        f_text = f.write(f_text)

    return None


# 3. impacts of rising wage level
def update_city(cities, ratio):
    """create a numba List of new city dataclass,
    each city in the new list has wage level equivalent to the original wage level * (1 + ratio).
    """
    new_cities = []
    for city in cities:
        k, wage, gdp = city
        new_cities.append(City(k, wage * ratio, gdp))
    return List(new_cities)


def increase_wage(alpha=0.5, beta=0.45, sigma=178, S=30):
    increase_ratio = [1, 1.1, 1.25, 1.5]
    cities_list = [update_city(cities, ratio) for ratio in increase_ratio]
    total_output = np.sum([firm.Y for firm in firms])
    average_land_price = np.empty(4)
    total_land_revenue = np.empty(4)
    total_profit_share = np.empty(4)
    total_fiscal_revenue = np.empty(4)
    all_sim_sample_prices = []
    params = Params(alpha, beta, sigma)

    for i, new_cities in enumerate(cities_list):
        sim_sample_prices, sim_sample_land_revenue = f_sim_sample_prices(
            params, new_cities, S=S
        )
        all_sim_sample_prices.append(sim_sample_prices)
        average_land_price[i] = np.mean(sim_sample_prices)
        total_land_revenue[i] = np.sum(sim_sample_land_revenue)
        total_profit_share[i] = beta * total_output
        total_fiscal_revenue[i] = total_profit_share[i] + total_land_revenue[i]

    # Calculate the change ratio of elements of x[1:] w.r.t x[0].
    change_ratio = lambda x: (x - x[0])[1:] / x[0]

    # compare empirical cdf
    em_cdf(
        all_sim_sample_prices,
        label_list=["original wage level"]
        + ["wage increases {:.0%}".format(i) for i in [0.1, 0.25, 0.5]],
        xlabel="land price (1,000 yuan/hectare)",
        fname="cdf_wage_increase",
    )

    # make table
    df = pd.DataFrame(
        {
            "wage increase": [0.1, 0.25, 0.5],
            "average land price": change_ratio(average_land_price),
            "total land selling revenue": change_ratio(total_land_revenue),
            "total fiscal revenue": change_ratio(total_fiscal_revenue),
        }
    )
    s = df.style
    s.hide(axis="index")
    s.format(
        {
            "wage increase": "{:.2%}",
            "average land price": "{:.2%}",
            "total land selling revenue": "{:.2%}",
            "total fiscal revenue": "{:.2%}",
        },
    )
    s.to_latex(
        buf="Tables/increase_wage.tex",
        column_format="ccccc",
        position="H",
        position_float="centering",
        hrules=True,
        label="table: increase_wage",
        caption=r"The impacts of rising wage level",
        multirow_align="t",
        multicol_align="r",
    )
    with open("Tables/increase_wage.tex", "r") as f:
        f_text = f.read()

    f_text = f_text.replace("%", "\%")
    with open("Tables/increase_wage.tex", "w") as f:
        f_text = f.write(f_text)

    return None


def main():
    allocation(Params(0.5, 0.45, 178), S=30)
    decrease_beta(alpha=0.5, beta=0.45, sigma=178, S=30)
    increase_wage(alpha=0.5, beta=0.45, sigma=178, S=30)


if __name__ == "__main__":
    main()
