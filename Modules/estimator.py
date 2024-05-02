"""Estimator of the model"""

import ray
import itertools
import numpy as np
from numba import njit
from scipy import optimize
from Modules.solver import Params, Supervisor
from Modules.simulator import closure_draw_short_lists, closure_sim_prices


# 1. empirical moments


@njit
def func_obs(price):
    """Function of one observation of price used for constructing empirical moments.
    We use price, price^2 (scaled by 1e4 to [0, 1]) for empirical moments.

    price: one observed land price scalar.
    """
    return np.array([[price / 1e4], [(price / 1e4) ** 2]])  # 2*1 vector


# 2. IV function


@njit
def iv(firm):
    """Technical IV vector for an observation (a firm).
    We use constant, scaled output level, scaled land usage as IVs.

    firm: an instance of firm data class.
    """
    # make sure IVs are scaled into [0, 1]
    y = firm.Y / 1e7  # max(Y) = 7e6 in data
    t = firm.T / 1e2  # max(T) = 65 in data
    return np.array([[1.0], [y], [t], [y * t]])  # 4*1 vector


# 3. moment conditions


@njit
def mom(firm, obs_price, sim_prices, sim_probs):
    """Moment conditions for one observation (a firm).

    firm: an instance of firm data class.
    obs_price:  observed land price of the firm.
    sim_prices: S*c matrix of simulated prices for the firm
                (c is the size of the firm's choice set),
                each row of the matrix is the price vector in one simulation.
    sim_probs:  S*c matrix of simulated probs for the firm to land in the cities
                of its choice set,
                each row of the matrix is one simulated prob vector for the firm to land
                in each city of its choice set.
    """
    k = len(func_obs(0.0))  # number of raw moments
    S = sim_prices.shape[0]  # number of simulations

    # simulated raw moments
    sim_moms = np.zeros((k, 1))

    for price, prob in zip(np.ravel(sim_prices), np.ravel(sim_probs)):
        sim_moms += (1 / S) * prob * func_obs(price)

    # moment conditions is kronecker product of IVs and structural residuals
    return np.kron(iv(firm), func_obs(obs_price) - sim_moms)


# 4. criteria functions


def closure_criteria(all_firms, all_wages, all_obs_prices, f_sim_prices, f_mom, W=None):
    """Creator of MSM criteria function.
    Returning a MSM criteria function with only parameters of the model as arguments.

    all_firms: a numba List (len=N) of all Firm data classes in the data set.
    all_wages: a nested list (len=N*S) of all wage arrays for every firm's choice set in each simulation.
    all_obs_prices: an array (len=N) of observed prices for all firms.
    f_sim_prices: a *function* simulating prices offered for each firm by every city in its choice set
                  in all simulations, which should return
                  a nested list (len = N*S) of NE *price* vectors,
                  (The first N elements are ne price vectors in first simulation, ...)
                  and another nested list (len = N*S) of NE *prob* vectors.
                  (See simulator Module for details.)
    f_mom: moment conditions *function* for one observation (a firm).
    W: the weighting matrix (identity matrix by default).
    """
    N = len(all_firms)  # number of observations
    S = len(all_wages) // N  # number of simulations
    # number of unconditional moments
    k = f_mom(all_firms[0], 0.0, np.array([0.0]), np.array([1.0])).shape[0]
    if W is None:
        W = np.eye(k)  # use identity weighting matrix by default

    # MSM criteria function w.r.t. parameters
    def criteria(params):
        g = np.zeros((k, 1))  # sum of moment conditions of all firms
        # two (len=N*S) nested list of price vector and prob vector in all simulations
        # first N elements are the results of N firms in first simulation ...
        all_prices, all_probs = f_sim_prices(params, prob=True)

        for i in range(N):
            firm = all_firms[i]
            obs_price = all_obs_prices[i]
            # construct S*c price matrix and S*c prob matrix for firm i
            sim_prices = np.array([all_prices[i + N * s] for s in range(S)])
            sim_probs = np.array([all_probs[i + N * s] for s in range(S)])
            g += f_mom(
                firm, obs_price, sim_prices, sim_probs
            )  # update sum of moment conditions

        return float(g.T @ W @ g) / (N**2)  # scale the criteria here by N**2

    return criteria


# 5. minimizers of criteria function


def grid_search(params_grids, f_criteria, details=False):
    """Finding the smallest criteria value on the grids.
    It also returns the dict of params-criteria if details==True.

    params_grids: a list of parameters vectors.
    f_criteria: the criteria function.
    """
    dict_criteria = {params: f_criteria(params) for params in params_grids}
    minimizer = min(dict_criteria, key=dict_criteria.get)
    if details:
        return minimizer, dict_criteria
    else:
        return minimizer


def finer_search(
    x0,
    f_criteria,
    alpha=None,
    alpha_bnd=(0.0, 1.0),
    beta_bnd=(0.0, 5.0),
    sigma_bnd=(1.0, 1000.0),
    method="L-BFGS-B",
    details=False,
):
    """Finding the minimizer of criteria function by using finer search method.

    x0: the initial vector (type: float) of search. If alpha is fixed, x0 = (beta0, sigma0).
        If alpha is not fixed, x0 = (alpha0, beta0, sigma0).
    f_criteria: the criteria *function*.
    alpha: the fixed value of alpha. If alpha == None, we estimate alpha, beta, sigma,
           otherwise we only estimate beta and sigma.
    alpha_bnd: the bounds of alpha. (alpha is between 0.0 and 1.0 by default)
    beta_bnd: the bounds of beta. (beta is between 0 and 5 by default)
    sigma_bnd: the bounds of sigma. (sigma is between 1.0 and 1000 by default)
    method: the minimization method used by scipy. (L-BFGS by default)
    details: whether to print the detailed results of scipy minimizer.
    """
    sigma_bnd = (sigma_bnd[0] / 100, sigma_bnd[1] / 100)  # scale sigma
    x0[-1] /= 100.0

    # estimate alpha, beta, sigma
    if alpha is None:
        bnds = (alpha_bnd, beta_bnd, sigma_bnd)
        f = lambda x: f_criteria(Params(alpha=x[0], beta=x[1], sigma=x[2] * 100))

    # fix alpha, estimate beta and sigma
    else:
        bnds = (beta_bnd, sigma_bnd)
        f = lambda x: f_criteria(Params(alpha=alpha, beta=x[0], sigma=x[1] * 100))

    res = optimize.minimize(f, x0=x0, method=method, bounds=bnds, tol=1e-6)
    if details:
        print(res)
    assert res.success == True, "minimizer cannot be found!"

    x = res.x
    x[-1] *= 100.0  # recover sigma
    return x


# 6. variance of estimated beta (calibrate alpha, estimate beta and sigma)
def closure_msm_cov(
    all_firms, all_wages_2, all_obs_prices, f2_sim_prices, f_obs, f_iv, S1=10
):
    """Creator of covariance matrix estimator.
    The closure should creates:
    1. a function takes calibrated or estimated alpha, estimates beta0, sigma0,
    weighting matrix W as arguments, and returns the standard error of the estimates.
    2. a function returns the optimal weighting matrix.

    all_firms: a numba List (len = N) of all Firm data classes in the data set.
    all_wages2: a nested list (len = N*S2) of all wage arrays for every firm's choice set
                in each simulation. (S2 should be larger relative to S1)
    all_obs_prices: an array (len = N) of observed prices for all firms.
    f2_sim_prices: a *function* does a **large number** of simulations (e.g. S2 = 20) of prices
                   offered for each firm by every city in its choice set in all simulations,
                   which should return
                   1.a nested list (len = N*S2) of NE *price* vectors,
                   (The first N elements are ne price vectors in first simulation, ...)
                   2.another nested list (len = N*S2) of NE *prob* vectors.
                   (See simulator Module for details.)
    f_obs: a function (one observation of price as its argument)
           used for constructing empirical moments.
    f_iv: a function returning IV vector for an observation (a firm).
    S1: the number of simulations in parameters estimation. (10 by default)
    """
    N = len(all_firms)  # number of observations
    S2 = len(all_wages_2) // N  # number of simulations (S2 should be at least 20)
    k1 = f_obs(0.0).shape[0]  # number of raw moments
    k2 = f_iv(all_firms[0]).shape[0]  # number of IVs
    k = k1 * k2  # number of unconditional moments
    # the list (len=N) of iv arrays for all observation
    all_ivs = [f_iv(firm) for firm in all_firms]

    def sim_mom(params):
        """A function
        1. approximates the theoretical raw moments
        for each observation based on large number of simulations; (output is N*k1 matrix)
        2. returns the raw moments calculated in one simulation for each observation.
        (used for measuring the simulation noise). (output is also N*k1 matrix)

        params: a dataclass of parameters.
        """
        # each row is approximated raw moments for each observation based on many simulations
        raw_moments = np.zeros((N, k1))
        # each row is one randomly drawn raw moments of ne price for each observation
        example_moments = np.zeros((N, k1))
        # two (len=N*S) nested list of price vector and prob vector in all simulations
        # first N elements are the results of N firms in first simulation ...
        all_prices, all_probs = f2_sim_prices(params, prob=True)

        for i in range(N):
            # f_obs(price) is column vector, thus, we need to flatten it
            # all_probs[i] is 1*c vector
            # np.array([f_obs(price).ravel() for price in all_prices[i]]) is c*k1 matrix
            example_moments[i, :] = all_probs[i] @ np.array(
                [f_obs(price).ravel() for price in all_prices[i]]
            )

            for s in range(S2):
                raw_moments[i, :] += (
                    (1 / S2)
                    * all_probs[i + N * s]
                    @ np.array(
                        [f_obs(price).ravel() for price in all_prices[i + N * s]]
                    )
                )

        return raw_moments, example_moments

    def bread(ests, W=None, fixed_alpha=None):
        """a function return
        1. the gradient matrix of moment conditions at the estimated beta0.
        2. the bread, i.e. inv(D'WD)D'W, in the sandwich form covariance matrix.

        ests: the list [alpha0, beta0, sigma0] or [beta0, sigma0].
        W: the weighting matrix. (identity matrix by default)
        fixed_alpha: the calibrated value of alpha. alpha is estimated if fixed_alpha == None.
        """
        if W is None:
            W = np.eye(k)
        # construct the gradient matrix D
        D = np.zeros((k, len(ests)))
        # define the function which convert estimates tuple to params dataclass
        if fixed_alpha is None:  # estimated alpha
            convert_pars = lambda ests: Params(*ests)
        else:  # calibrated alpha
            convert_pars = lambda ests: Params(fixed_alpha, *ests)

        h = 1e-6  # the delta in numerical differentiation
        for i in range(len(ests)):
            ests0 = ests.copy()
            ests0[i] -= 0.5 * h
            ests1 = ests.copy()
            ests1[i] += 0.5 * h
            params0, params1 = convert_pars(ests0), convert_pars(ests1)
            raw_moments0, *_ = sim_mom(params0)
            raw_moments1, *_ = sim_mom(params1)

            D[:, i] = (
                np.mean(
                    [
                        np.kron(all_ivs[j], raw_moments1[j][:, np.newaxis])
                        for j in range(N)
                    ],
                    axis=0,
                )
                - np.mean(
                    [
                        np.kron(all_ivs[j], raw_moments0[j][:, np.newaxis])
                        for j in range(N)
                    ],
                    axis=0,
                )
            ).ravel() / h
        b = np.linalg.inv(D.T @ W @ D) @ D.T @ W
        return D, b

    def meat(alpha0, beta0, sigma0):
        """the meat in the sandwich form covariance matrix.

        alpha0: the estimated or calibrated alpha.
        beta0: the estimated beta.
        sigma0: the estimated sigma.

        """
        gmm_var = np.zeros((k, k))  # gmm covariance matrix of moment conditions
        sim_noise = np.zeros((k, k))  # covariance matrix of simulation noise
        params = Params(alpha0, beta0, sigma0)
        raw_moments, example_moments = sim_mom(params)

        # mean of moment conditions
        mom_mean = np.mean(
            [
                np.kron(
                    all_ivs[i],
                    (f_obs(all_obs_prices[i]) - raw_moments[i][:, np.newaxis]),
                )
                for i in range(N)
            ],
            axis=0,
        )
        # mean of simulation noise
        noise_mean = np.mean(
            [
                np.kron(
                    all_ivs[i],
                    (example_moments[i][:, np.newaxis] - raw_moments[i][:, np.newaxis]),
                )
                for i in range(N)
            ],
            axis=0,
        )

        # variance of moment conditions and simulation noise
        for i in range(N):
            iv = all_ivs[i]  # k2*1 column vector
            raw_mom = raw_moments[i][:, np.newaxis]  # k1*1 column vector
            obs_mom = f_obs(all_obs_prices[i])
            ex_mom = example_moments[i][:, np.newaxis]
            gmm_var += (
                (1 / N)
                * (np.kron(iv, (obs_mom - raw_mom)) - mom_mean)
                @ (np.kron(iv, (obs_mom - raw_mom)) - mom_mean).T
            )
            sim_noise += (
                (1 / (N * S1))
                * (np.kron(iv, (ex_mom - raw_mom)) - noise_mean)
                @ (np.kron(iv, (ex_mom - raw_mom)) - noise_mean).T
            )

        return gmm_var + sim_noise

    def update_W(alpha0, beta0, sigma0):
        """update weighting matrix"""
        return np.linalg.inv(meat(alpha0, beta0, sigma0))

    def se(alpha0, beta0, sigma0, W=None, optimal_W=False, calibrate_alpha=False):
        """The standard errors of estimates.
        If optimal_W == True, we use the simplified covariance matrix formula.
        alpha0 is calibrated value if calibrate_alpha==True else estimated value.
        """
        if calibrate_alpha:
            D, b = bread([beta0, sigma0], W, fixed_alpha=alpha0)
        else:
            D, b = bread([alpha0, beta0, sigma0], W, fixed_alpha=None)

        if optimal_W:
            return np.sqrt(np.diag((1 / N) * np.linalg.inv(D.T @ W @ D)))
        else:
            m = meat(alpha0, beta0, sigma0)
            return np.sqrt(np.diag((1 / N) * b @ m @ b.T))

    return se, update_W


# 7. two stage estimation
def msm_2stage(
    firms,
    cities,
    all_obs_prices,
    distance_matrix,
    f_ne,
    alpha,
    min_size=3,
    max_size=5,
    S1=10,
    S2=30,
):
    """Two stage MSM estimation.
    Return the estimates beta, sigma and their standard errors. (alpha is calibrated.)

    firms: the numba list of all observed firm dataclasses.
    cities: the numba list of all observed city dataclasses.
    all_obs_prices: the array of all observed housing prices.
    distance_matrix: the numpy array of distances matrix.
    f_ne: solver function of nash equilibrium price vector.
    draw_short_lists: the function used to create short lists of candidates across simulations.
    alpha: the calibrated alpha.
    min_size: the minimum size of candidate size.
    max_size: the maximum size of candidate size.
    S1: the number of simulations used in estimation.
    S2: the (large) number of simulations used in calculating the standard errors.
    """
    # draw short lists function for estimating parameters
    draw_short_lists1 = closure_draw_short_lists(
        distance_matrix,
        pool_size=10,
        min_size=min_size,
        max_size=max_size,
        seed=42,
    )
    # draw short lists function for estimating covariance matrix (change the seed)
    draw_short_lists2 = closure_draw_short_lists(
        distance_matrix,
        pool_size=10,
        min_size=min_size,
        max_size=max_size,
        seed=52,
    )

    # a function used to create price simulator and nested wage list
    # for parameters or covariance matrix estimation
    def create_sim_prices(n_sim, f_draw_short_lists):
        f_sim_prices, all_short_lists, all_wages = closure_sim_prices(
            firms=firms,
            cities=cities,
            Sup=Supervisor,
            ne=f_ne,
            draw_short_lists=f_draw_short_lists,
            S=n_sim,
            fix_obs=False,
        )
        return f_sim_prices, all_wages

    # price simulator and nested wage list for parameters estimation
    f1_sim_prices, all_wages1 = create_sim_prices(S1, draw_short_lists1)

    # criteria function for first stage estimation (identity weighting matrix)
    criteria1 = closure_criteria(
        firms, all_wages1, all_obs_prices, f1_sim_prices, f_mom=mom, W=None
    )

    # grid search
    beta_grids = np.arange(0.0, 2.1, 0.1)
    sigma_grids = np.arange(100.0, 1100.0, 100.0)
    params_grids = [
        Params(alpha, beta, sigma)
        for (beta, sigma) in itertools.product(beta_grids, sigma_grids)
    ]
    minimizer = grid_search(params_grids, f_criteria=criteria1, details=False)

    # finer search
    beta1, sigma1 = finer_search(
        x0=np.array([minimizer[1], minimizer[2]]),
        f_criteria=criteria1,
        method="L-BFGS-B",
        details=False,
        alpha=alpha,
        beta_bnd=(0, 10.0),
        sigma_bnd=(1.0, 1000.0),
    )

    # second-stage estimation
    # price simulator and nested wage list for covariance matrix estimation
    ray.shutdown()
    f2_sim_prices, all_wages2 = create_sim_prices(S2, draw_short_lists2)
    # update weighting matrix
    *_, update_W = closure_msm_cov(
        all_firms=firms,
        all_wages_2=all_wages2,
        all_obs_prices=all_obs_prices,
        f2_sim_prices=f2_sim_prices,
        f_obs=func_obs,
        f_iv=iv,
        S1=S1,
    )
    W2 = update_W(alpha, beta1, sigma1)

    # second-stage criteria function with the optimal weighting matrix
    # price simulator and nested wage list for parameters estimation
    ray.shutdown()
    f1_sim_prices, all_wages1 = create_sim_prices(S1, draw_short_lists1)
    # criteria function for second stage estimation (optimal weighting matrix)
    criteria2 = closure_criteria(
        firms, all_wages1, all_obs_prices, f1_sim_prices, f_mom=mom, W=W2
    )
    # second stage estimates
    beta2, sigma2 = finer_search(
        x0=np.array([beta1, sigma1]),
        f_criteria=criteria2,
        method="L-BFGS-B",
        details=False,
        alpha=alpha,
        beta_bnd=(0, 10.0),
        sigma_bnd=(1.0, 1000.0),
    )

    # price simulator and nested wage list for covariance matrix estimation
    ray.shutdown()
    f2_sim_prices, all_wages2 = create_sim_prices(S2, draw_short_lists2)
    # standard error function for second stage estimates
    se, *_ = closure_msm_cov(
        all_firms=firms,
        all_wages_2=all_wages2,
        all_obs_prices=all_obs_prices,
        f2_sim_prices=f2_sim_prices,
        f_obs=func_obs,
        f_iv=iv,
        S1=S1,
    )
    se_beta, se_sigma = se(
        alpha, beta2, sigma2, calibrate_alpha=True, optimal_W=True, W=W2
    )

    return (beta2, sigma2), (se_beta, se_sigma)
