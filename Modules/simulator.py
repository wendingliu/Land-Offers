"""
Simulator for the model:
Simulate the NE land prices for each firm in each simulation.
"""
import ray
import numpy as np
from numba import njit
import Modules.solver as solver

# 1. draw choice sets, prepare inputs for simulation


@njit
def short_lists_sizes(firms, min_size, max_size):
    """Calculate the size of short list for each firm in the list (firms).
    There should be 100% / (max_size - min_size + 1) firms having short list of each size.
    Return the list of each firm's candidates size.

    firms: a numba List of Firm data class.
    min_size (int): the minimum size of any firm's choice set.
    max_size (int): the maximum size of any firm's choice set.
    """
    # list comprehension doesn't work for numba here, let's use loop!
    nfirms = len(firms)
    Ys = np.empty(nfirms, dtype=np.float64)  # array of all firms' outputs
    for i in range(nfirms):
        Ys[i] = firms[i].Y

    # firm which has output level in brackets[i] should have choice set of size = min_size + i
    # i = 0, 1, ..., max_size - min_size
    n_bracket = max_size - min_size + 1
    brackets = np.array_split(np.sort(Ys), n_bracket)
    sizes = np.empty(nfirms, dtype=np.int16)

    for i, Y in enumerate(Ys):
        for j in range(n_bracket):
            if brackets[j][0] <= Y <= brackets[j][-1]:
                sizes[i] = min_size + j
                break

    return sizes


def closure_draw_short_lists(
    distance_matrix, pool_size=10, min_size=3, max_size=3, seed=42
):
    """Create draw_short_lists function.

    distance matrix: a (288 * 288) square matrix of distances between Chinese cities.
    pool_size (int): the number of cities to be drawn from (10 candidates by default).
    min_size (int): the minimum size of any firm's choice set (3 cities by default).
    max_size (int): the maximum size of any firm's choice set (3 cities by default).
    seed: the seed of random number generator (42 by default).
    """

    @njit
    def draw_short_lists(firms, S=10, fix_obs=False):
        """Draw choice set for each firm in each simulation.
        Return a nested list of each firm's choice set in
        all the simulations
        (the nested list is constituted by S*N short lists, each short list
        is a firm's choice set in one simulation).

        firms: a numba List of Firm data class.
        S: number of simulations.
        fix_obs: the city which firm chooses in data is always in the choice set if True
                 (Default False).
        """
        np.random.seed(seed)
        nfirms = len(firms)
        # the list of each firm's candidates size.
        sizes = short_lists_sizes(firms, min_size, max_size)
        # a nested list constituted by S*N short lists
        # (first N elements are the N firms' short lists in the 1st simulation,
        # N+1~2N elements are the N firms' short lists in the 2nd simulation...)
        all_short_lists = []

        for s in range(S):
            # short_lists is the nested array for each firm's choice set in simulation s
            # this initialization is required by numba
            short_lists = [np.empty(1, dtype=np.int64) for i in range(nfirms)]

            for i, f, size in zip(range(nfirms), firms, sizes):
                if not fix_obs:
                    # for firm i, draw sizes[i] cities from n=pool_size cities
                    # nearest to the city in data (including f.c)
                    pool = np.argsort(distance_matrix[f.c])[0:pool_size]
                    short_lists[i] = np.random.choice(pool, size, replace=False)
                else:
                    # 1. for firm i, draw sizes[i] cities from n = pool_size cities - 1
                    # nearest to the city in data (excluding f.c)
                    # 2. f.c is always in the choice set of f
                    pool = np.argsort(distance_matrix[f.c])[1:pool_size]
                    short_lists[i] = np.empty(size, dtype=np.int64)
                    short_lists[i][0] = f.c
                    short_lists[i][1:] = np.random.choice(pool, size - 1, replace=False)

            # each element in the list returned is choice set of a game
            for choice_set in short_lists:
                all_short_lists.append(choice_set)

        return all_short_lists

    return draw_short_lists


# 2.simulate NE prices


def closure_sim_prices(firms, cities, Sup, ne, draw_short_lists, S=10, fix_obs=False):
    """A creator of simulator.

    Returning:
    1.a simulator (function) which returns the nested list of N*S NE *prices* vectors
    and (optional) the nested list of N*S *probs* vectors for each firm in all simulations,
    (The simulator should only accept an instance of Params data class as argument)
    (The first N vectors belong to the first simulation,
    and the second N vectors belong to the second simulations...)

    2.all short candidate lists for every firm in each simulation.

    3.N*S nested list of all wage arrays for every choice set in each simulation.
    (The first N wage arrays belong to the first simulation,
    and the second N wage arrays belong to the second simulations...)


    firms: a numba List of all Firm data classes. (draw_short_lists needs numba list)
    cities: a list (or numba List) of all City data classes.
    Sup: a Ray supervisor class (used to solve the games).
    ne: **solver** of the equilibrium price vector in the Bertrand game.
    draw_short_lists: a **function** drawing choice set for each firm in each simulation.
    S: number of simulations.
    fix_obs: if True, the city which firm chooses in data is always in the choice set (Default False).
    """
    # dict: city id -- city wage
    wage_dict = {c.k: c.wage for c in cities}
    all_short_lists = draw_short_lists(firms, S, fix_obs)
    # nested list (len = S*N) for the wage array in each game
    all_wages = [
        np.array([wage_dict[k] for k in short_list]) for short_list in all_short_lists
    ]  # the elements in all_wages must be np.array!!

    # create a ray supervisor (change numba List to normal list here)
    sup = Sup.remote(S * list(firms), list(cities), ne)
    # update the supervisor's attributes and solve all the games
    sup.update_all_wages.remote(all_wages)

    def sim_prices(params, prob=False):
        """Simulate prices offered for each firm by every city in its choice set
        in all simulations.

        The result is a nest list (len = N*S), each element of the list
        is an array of NE price vectors. (if prob == False)
        (The first N elements are ne price vectors in the first simulation,
         The second N elements are ne price vectors in the second simulation...)

        If prob == True, The function will also return a nested list (len = N*S) of
        NE probs vectors.

        params: an instance of Params data class.
        """

        sup.update_params.remote(params)
        # nested list of N*S NE prices vectors
        all_prices = ray.get(sup.solve_games.remote())
        if prob == False:
            return all_prices
        else:
            all_probs = []  # N*S nested list of NE probs
            for (firm, wages, prices) in zip(S * list(firms), all_wages, all_prices):
                all_probs.append(
                    solver.prob(firm, cities[firm.c], params, wages, prices)
                )
            return all_prices, all_probs

    return sim_prices, all_short_lists, all_wages

