'''Visualize iterates of the stochastic Frank-Wolfe algo in a
low-dimensional setting'''
import functools as ft
from typing import Callable
import numpy as np
import numpy.typing as npt
import scipy.stats as ss  # type: ignore

import matplotlib.pyplot as plt
import sys
sys.path.append('C:/stochastic_programming')
from ProjectedSGD import ProjectedSGD  # noqa: E402
from logger import get_logger  # noqa: E402
from helper_functions import timing  # noqa: E402


def true_function(x: npt.NDArray[np.float64],
                  sampling_dist: ss.rv_continuous
                  ) -> npt.NDArray[np.float64]:
    '''f(x) = E[(x - W)^T(x - W)], as described below\n
    I can evaluate this for an arbitrary distribution using built-in
    scipy methods

    :param x: .shape = (num inputs, iterate dim)
    :param sampling_dist: sampling distribution for W
    :return: one function output for each row of the vectorized input'''
    x = np.atleast_2d(x)
    iterate_dim = x.shape[-1]
    inner_products = np.sum(x**2, axis=-1)
    E_W = np.full(iterate_dim, sampling_dist.mean())
    E_WTW = sampling_dist.moment(2) * 2
    return inner_products - 2 * x @ E_W + E_WTW


def gradient_func(iterate: npt.NDArray[np.float64],
                  stochastic_samples: npt.NDArray[np.float64]
                  ) -> npt.NDArray[np.float64]:
    '''Objective: min f(x) = E[(x - W)^T(x - W)], where x is our optimization
    variable and W is our vector random variable\n
    del_f(x) = E[2*(x - W)] (see Wikipedia matrix calculus identities)

    :return: .shape = (len(stochastic_samples), len(iterate))
        The above gradient for each sample'''
    return 2 * (iterate - stochastic_samples)


def hessian_func(iterate: npt.NDArray[np.float64],
                 stochastic_samples: npt.NDArray[np.float64]
                 ) -> npt.NDArray[np.float64]:
    '''Given that same min f(x) = E[(x - W)^T(x - W)], compute
    the Hessian f''(x)\n
    Not much "computation"; it's just 2*I. Just make sure to
    broadcast this appropriately:

    :return: .shape = (len(stochastic_samples), len(iterate), len(iterate))'''
    single_eye = np.eye(len(iterate))
    return np.tile(single_eye, (len(stochastic_samples), 1, 1))


def stochastic_sampler(num_samples: int,
                       sampling_dist: ss.rv_continuous,
                       ndim: int
                       ) -> npt.NDArray[np.float64]:
    '''Sample n-dimensional independent random varaiables

    :param num_samples: number of 2d vectors to sample
    :param sampling_dist: distribution of each component
    :param ndim: number of dimensions of each iterate
    :return: the (num_samples, ndim) array of samples'''
    return sampling_dist.rvs(size=(num_samples, ndim))  # type: ignore


def satisfies_constraints(
        grid: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64], w: float,
        a: float, b: float,
        ) -> npt.NDArray[np.bool_]:
    '''Given a grid of points, let return[i, j] = True iff the point
    at grid[i, j, :] falls within the given constraints

    :param grid: .shape = (x discretization, y discretization, 2)
    :param d: vector of the set {x | d @ x <= w, a <= x <= b}
    :param w: see above
    :param a: see above
    :param b: see above
    :return: return[i, j] = True iff the point
        at grid[i, j, :] falls within the given constraints
    '''
    d_grid = np.squeeze(d @ grid[..., None], axis=-1)
    logger.debug(f'{d_grid.shape=}')
    constraints_satisfied: npt.NDArray[np.bool_] = np.all(
        (a <= grid) & (grid <= b), axis=-1) & (d_grid <= w)
    return constraints_satisfied


def plot_iterates_on_contour(
        iterate_histories: dict[str, npt.NDArray[np.float64]],
        d: npt.NDArray[np.float64], w: float, a: float, b: float,
        sampling_dist: ss.rv_continuous,
        x_lim: tuple[float, float] = (0, 1),
        y_lim: tuple[float, float] = (0, 1)):
    '''Make a contour plot of the true function along with the
    constraint set {x | d @ x <= w, a <= x <= b}\n
    Then, plot iterate histories, where each is keyed by a string
        that references its hyperparam combo

    :param sampling_dist: needed to get moments for the true_function
    :param x_lim: limits of the x coordinate domain
    :param y_lim: limits of the y coordinate domain'''
    iterate_dim = len(d)
    if iterate_dim != 2:
        print('plot_iterates_on_contour only meant for 2D iterates...'
              f'{iterate_dim=}')
        return
    _, ax = plt.subplots()

    x_space = np.linspace(*x_lim, num=100)
    y_space = np.linspace(*y_lim, num=100)
    X, Y = np.meshgrid(x_space, y_space)
    grid = np.dstack((X, Y))

    constraints_satisfied = satisfies_constraints(grid, d, w, a, b)
    ax.contourf(X, Y, constraints_satisfied, levels=[1-1e-7, 1+1e-7],
                colors=['lightblue'])

    for history_name, iterate_history in iterate_histories.items():
        ax.scatter(iterate_history[:, 0], iterate_history[:, 1],
                   label=history_name)
        ax.annotate('x_0', iterate_history[0])

    true_func_values = true_function(grid, sampling_dist)
    plt.contour(X, Y, true_func_values, levels=30)

    ax.set_aspect('equal')
    ax.set_title('Stochastic Frank Wolfe iterate histories')
    ax.legend()
    plt.colorbar(label='True func values')


def plot_convergence_to_optimal(
        iterate_histories: dict[str, npt.NDArray[np.float64]],
        sampling_dist: ss.rv_continuous):
    '''I can get the optimal x* from the sampling_dist moments
    Plot the l2-norm differences between each iterate history
        and the optimal (labeled by the dict key)\n
    Also plot the convergence of the objective function value to
        the optimal'''
    _, axes = plt.subplots(1, 2)
    ax1: plt.Axes = axes[0]
    ax2: plt.Axes = axes[1]
    ax1.set_xlabel('Iterate number')
    ax1.set_ylabel('l2-norm difference from x*')
    ax1.set_title('l2-norm differences between iterates & the optimal')
    ax2.set_xlabel('Iterate number')
    ax2.set_ylabel('log(E[f(x_k)] - f(x)) (objective excesses)')
    ax2.set_title('Objective func excesses between iterates & the optimal')

    for history_name, iterate_history in iterate_histories.items():
        x_star = np.full(iterate_history.shape[1], sampling_dist.mean())
        norm_diffs = np.linalg.norm(iterate_history - x_star, axis=1)
        ax1.plot(norm_diffs, label=history_name)

        true_objectives = true_function(iterate_history, sampling_dist)
        optimal_objective = true_function(x_star, sampling_dist)
        objective_excesses = true_objectives - optimal_objective
        ax2.plot(objective_excesses, label=history_name)

    ax1.legend()
    ax2.legend()
    ax2.set_yscale('log')


def get_pg_stepsizes_iterate_histories(
        d: npt.NDArray[np.float64], w: float,
        a: float, b: float,
        x_0: npt.NDArray[np.float64],
        stochastic_sampling_func: Callable[[int], npt.NDArray[np.float64]]
        ) -> dict[str, npt.NDArray[np.float64]]:
    '''Return iterate histories to compare convergence of different stepsize
    rules for Projected Stochastic Gradient Descent

    :return: dict of (plot label, iterate history)
    '''
    stepsize_class_names = [
        'OneOverTStepsize',
        'MultivariateKestenStepsize'
    ]
    iterate_histories: dict[str, npt.NDArray[np.float64]] = {}
    for stepsize_class_name in stepsize_class_names:
        pg_instance = ProjectedSGD(
            d, w, a, b,
            gradient_func, stochastic_sampling_func)
        cur_hist = timing(pg_instance.iterate_from)(
            x_0, batch_size=100, num_iters=5000)
        iterate_histories[stepsize_class_name] = cur_hist
    return iterate_histories


def main():
    ITERATE_DIM = 2

    d = np.ones(ITERATE_DIM)
    w = 1
    lower_bound = 0
    upper_bound = 0.75

    true_opt = 0.5

    sampling_dist: ss.rv_continuous = \
        ss.t(df=4, loc=true_opt, scale=1)  # type: ignore
    stochastic_sampling_func = ft.partial(
        stochastic_sampler, sampling_dist=sampling_dist,
        ndim=ITERATE_DIM)

    x_0 = np.insert(np.zeros(ITERATE_DIM-1), 0, upper_bound)
    # x_0 = np.full(ITERATE_DIM, 0.5)

    iterate_histories = get_pg_stepsizes_iterate_histories(
        d, w, lower_bound, upper_bound,
        x_0, stochastic_sampling_func)

    plot_iterates_on_contour(
        iterate_histories, d, w, lower_bound, upper_bound, sampling_dist)
    plot_convergence_to_optimal(iterate_histories, sampling_dist)
    plt.show()


if __name__ == '__main__':
    logger = get_logger(log_out_file='./debug.log')
    main()
