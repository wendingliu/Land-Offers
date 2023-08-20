"""Program generating all the tables and graphs of estimation results"""
import ray
import numpy as np
import pandas as pd
from numba.typed import List
from Modules.solver import City, Firm, solver_creator
from Modules.estimator import msm_2stage

try:
    ray.shutdown()
except:
    pass

# 1.loading data
firm_data = pd.read_csv("./Data/2012_firm_land.csv", index_col=0)
firms = List(
    [
        Firm(
            i=i,
            c=firm_data.loc[i, "city_id"],
            Y=firm_data.loc[i, "Y"],
            L=firm_data.loc[i, "L"],
            T=firm_data.loc[i, "T"],
            p=firm_data.loc[i, "price"],
            m=50.0,
        )
        for i in firm_data.index
    ]
)

all_obs_prices = np.array(firm_data["price"])
pmin, pmax = all_obs_prices.min(), all_obs_prices.max()
assert len(all_obs_prices) == len(firms), "We should have observed price for each firm!"

city_data = pd.read_csv("./Data/2012_city_statistics.csv", index_col=0)
cities = List(
    [
        City(k=k, wage=city_data.loc[k, "wage"], gdp=city_data.loc[k, "gdp"])
        for k in city_data.index
    ]
)

distance_matrix = np.array(pd.read_csv("./Data/2012_city_distance.csv", index_col=0))


# 2.estimation and robustness checks
def main(margin, size_list=[(3, 5)]):
    """estimator of the model.
    margin: the participation constraints,
    action space for firm i = [p_i - margin, p_i + margin].
    size_list: a list of different choice set settings.
    """
    # present data set
    print("number of firms: ", len(firms))
    print("number of cities: ", len(cities))
    print("shape of distance matrix: ", distance_matrix.shape)
    print(f"pmax = {pmax:.2f}, pmin = {pmin:.2f}")

    # solver function of nash equilibrium
    f_ne = solver_creator(margin=margin, tol=1e-10, max_iter=10_000)

    def ests_to_latex(min_size, max_size, caption, label, file_name):
        """a function to create latex table of estimation results
        under particular candidate set specification (min_size, max_size)"""
        ests_dict = {}
        se_dict = {}

        for alpha in [0.33, 0.5, 0.67]:
            ests, se = msm_2stage(
                firms,
                cities,
                all_obs_prices,
                distance_matrix,
                f_ne,
                alpha=alpha,
                min_size=min_size,
                max_size=max_size,
                S1=10,
                S2=30,
            )
            ests_dict[alpha] = ests
            se_dict[alpha] = se

        print(ests_dict)

        df = pd.DataFrame(
            {
                r"Calibrated $\alpha$": ["0.33", "", "0.5", "", "0.67", ""],
                "Parameters": [r"$\beta$", r"$\sigma$"] * 3,
                "Estimates": np.ravel(list(ests_dict.values())),
                "Standard Error": np.ravel(list(se_dict.values())),
            }
        )
        s = df.style
        s.format({"Estimates": "{:.3f}", "Standard Error": "{:.3f}"})
        s.hide(axis="index")
        s.to_latex(
            buf="./Tables/" + file_name + ".tex",
            column_format="cccc",
            position="H",
            position_float="centering",
            hrules=True,
            label=label,
            caption=caption,
            multirow_align="t",
            multicol_align="r",
        )
        print(
            "min_size = {}, max_size = {}, margin = {}\n".format(
                min_size, max_size, margin
            ),
            df,
            end="\n\n",
        )
        return None

    caption_list = [
        "Estimates of Parameters ($%i \leq |\mathbf{C}_i| \leq %i$)" % sizes
        for sizes in size_list
    ]
    label_list = [
        "table: estimates (min_size={} max_size={} margin={})".format(*sizes, margin)
        for sizes in size_list
    ]
    file_name_list = [
        "estimates_min_size_{}_max_size_{}_margin_{}".format(*sizes, margin)
        for sizes in size_list
    ]

    for sizes, caption, label, file_name in zip(
        size_list, caption_list, label_list, file_name_list
    ):
        ests_to_latex(*sizes, caption, label, file_name)


# test
if __name__ == "__main__":
    main(
        margin=2000,
        size_list=[
            (3, 5),
        ],
    )
