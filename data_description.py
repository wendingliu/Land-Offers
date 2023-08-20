"""
draw the firms each city lands in data set,
need to install pyecharts version 0.5.11, and install all the map package.
https://05x-docs.pyecharts.org/#/zh-cn/customize_map
the default map is in html, need to manually save png file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyecharts import Geo


def draw_map():
    # load geographical data of cities
    cities = pd.read_csv("Data/city_longitude_latitude.csv")
    # update the city name for direct-administered municipalities and SAR
    cities.loc[cities.city.isnull(), "city"] = cities.province[cities.city.isnull()]
    # new column: number of firms in each city
    cities["nfirms"] = 0
    # load firm data we use in the paper (1039 obs)
    firms = pd.read_csv("Data/2012_firm_land.csv")

    city_list = list(cities.city)
    # loop for each firm to get the number of firms landing in each city
    for f in firms.city:
        i = city_list.index(f)  # the index of the city where firm f lands
        cities.loc[i, "nfirms"] += 1

    # only draw the plot for cities have at least one firm
    cities = cities[cities.nfirms > 0]
    cities.index = range(len(cities))  # reindex
    cities.nfirms.max()

    # create China map
    geo = Geo(
        "number of new firms in data",
        title_color="black",
        title_pos="center",
        width=800,
        height=600,
        background_color="#D3D3D3",  # shallow grey background
    )

    # value shown in the map is the number of firms each city lands in data
    attr = list(cities["city"])
    value = list(cities["nfirms"])
    geo_cities_coords = {
        cities.loc[i, "city"]: [cities.loc[i, "longitude"], cities.loc[i, "latitude"]]
        for i in range(len(cities))
    }

    geo.add(
        "",  # we have added title when we initialize the class
        attr,  # city names
        value,  # number of firms in each city
        visual_range=[
            1,
            cities["nfirms"].max(),
        ],  # visualize firm numbers between 1 and max_nfirms
        visual_text_color="black",  # legend color
        is_piecewise=False,  # no lables by group
        visual_split_number=cities["nfirms"].max(),  # 32 groups
        symbol_size=7.5,
        is_visualmap=True,
        geo_cities_coords=geo_cities_coords,
        geo_normal_color="white",
        geo_emphasis_color="white",
    )

    # save the html file
    geo.render(path="Graphs/nfirms.html")
    print(
        "there are {} cities landing at least one firm in data set.".format(len(cities))
    )


def describe_data():
    """make descriptive statistics of industrial land price"""
    firm_land = pd.read_csv("Data/2012_firm_land.csv", index_col=0)
    city_statistics = pd.read_csv("Data/2012_city_statistics.csv", index_col=0)
    p = np.array(firm_land["price"])  # land prices
    y = np.array(firm_land["Y"])  # output
    l = np.array(firm_land["L"])  # number of labors
    t = np.array(firm_land["T"])  # area of land
    w = np.array(city_statistics["wage"])  # wages of all cities
    df1 = pd.DataFrame(
        {
            "land price (1,000 yuan/hectare)": p,
            "output (1,000 yuan)": y,
            "number of workers (person)": l,
            "area of land (hectare)": t,
        }
    )
    df2 = pd.Series(w)
    stats = df1.describe()
    stats["city wage (1,000 yuan/year)"] = df2.describe()
    stats = stats.round(decimals=2)
    stats.loc["N"] = stats.loc["count"]
    stats.drop("count", inplace=True)
    # print the latex table of descriptive statistics
    print(stats)
    # print(stats.to_latex(index=True))
    s = stats.style
    s.format({cl: "{:.2f}" for cl in stats.columns})
    s.to_latex(
        buf="Tables/descriptive_statistics.tex",
        column_format="cccccc",
        position="H",
        position_float="centering",
        hrules=True,
        label="table: descriptive_statistics",
        caption="Descriptive statistics of firm level variables and city wage",
        multirow_align="t",
        multicol_align="r",
    )

    # draw the histograms of output, land price distributions
    sns.set_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    sns.histplot(
        data=(y / 1000)[(y / 1000) < 2000],
        kde=True,
        binwidth=100,
        stat="percent",
        ax=ax1,
    ).set(xlabel="output levels of firms (1 million yuan)")
    sns.histplot(
        data=p,
        kde=True,
        binwidth=500,
        stat="percent",
        ax=ax2,
    ).set(xlabel="land price (1,000 yuan/hectare)")
    ax1.set_title("output level")
    ax2.set_title("land price")
    plt.savefig("Graphs/data_distribution.pdf")


# test
if __name__ == "__main__":
    draw_map()
    describe_data()
