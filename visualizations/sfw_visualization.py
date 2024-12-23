'''Visualize iterates of the stochastic Frank-Wolfe algo in a
low-dimensional setting'''
import functools as ft
import numpy as np
import numpy.typing as npt
import scipy.stats as ss  # type: ignore

import matplotlib.pyplot as plt
import sys
sys.path.append('C:/stochastic_programming')
from StochasticFrankWolfe import StochasticFrankWolfe  # noqa: E402
from logger import get_logger  # noqa: E402


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
    del_f(x) = E[x - W] (see Wikipedia matrix calculus identities)

    :return: .shape = (len(stochastic_samples), len(iterate))
        The above gradient for each sample'''
    return 2 * (iterate - stochastic_samples)


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
        A: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
        box_constraints: tuple[float | None, float | None],
        ) -> npt.NDArray[np.bool_]:
    '''Given a grid of points, let return[i, j] = True iff the point
    at grid[i, j, :] falls within the given constraints

    :param grid: .shape = (x discretization, y discretization, 2)
    :param A: polyhedron matrix of the set {x | Ax <= b}
    :param b: see above
    :param box_constraints: specifies the (optional) endpoints of the
        box constraints {x | l <= x <= h}
    :return: return[i, j] = True iff the point
        at grid[i, j, :] falls within the given constraints
    '''
    iterate_dim = A.shape[1]
    A_grid = np.squeeze(A @ grid[..., None], axis=-1)
    in_polyhedron: npt.NDArray[np.bool_] = np.all(A_grid <= b, axis=-1)
    logger.debug(f'{A_grid.shape=}, {in_polyhedron.shape=}')
    lower_box, upper_box = box_constraints
    lower_box = float('-inf') if lower_box is None else lower_box
    upper_box = float('-inf') if upper_box is None else upper_box
    in_box = np.all(
        (np.full(iterate_dim, lower_box) <= grid)
        & (grid <= np.full(iterate_dim, upper_box)),
        axis=-1
    )
    return in_polyhedron & in_box


def plot_iterates_on_contour(
        iterate_histories: dict[str, npt.NDArray[np.float64]],
        A: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
        box_constraints: tuple[float | None, float | None],
        sampling_dist: ss.rv_continuous,
        x_lim: tuple[float, float] = (-2, 2),
        y_lim: tuple[float, float] = (-2, 2)):
    '''Make a contour plot of the true function along with the
    constraint set {x | Ax <= b}\n
    Then, plot iterate histories, where each is keyed by a string
        that references its hyperparam combo

    :param sampling_dist: needed to get moments for the true_function
    :param x_lim: limits of the x coordinate domain
    :param y_lim: limits of the y coordinate domain'''
    iterate_dim = A.shape[1]
    if iterate_dim != 2:
        print('plot_iterates_on_contour only meant for 2D iterates...'
              f'{iterate_dim=}')
        return
    _, ax = plt.subplots()

    x_space = np.linspace(*x_lim, num=100)
    y_space = np.linspace(*y_lim, num=100)
    X, Y = np.meshgrid(x_space, y_space)
    grid = np.dstack((X, Y))

    constraints_satisfied = satisfies_constraints(grid, A, b, box_constraints)
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


def main():
    ITERATE_DIM = 100

    A = np.ones(ITERATE_DIM)[None, :]
    b = np.array([1])
    logger.debug(f'A={A},\n{b=}')
    box_constraints = (-1, 1)
    sampling_dist: ss.rv_continuous = ss.norm()  # type: ignore
    stochastic_sampling_func = ft.partial(
        stochastic_sampler, sampling_dist=sampling_dist,
        ndim=ITERATE_DIM)

    sfw_instance = StochasticFrankWolfe(
        A, b, box_constraints, gradient_func, stochastic_sampling_func)
    x_0 = np.full(ITERATE_DIM, -1)
    iterate_history = sfw_instance.iterate_from(
        x_0, batch_size=10, num_iters=(n_iters := 1000))
    logger.debug(f'iterate_history=\n{iterate_history}')
    large_batch_iterate_history = sfw_instance.iterate_from(
        x_0, batch_size=1_000, num_iters=n_iters)

    iterate_histories = {
        'vanilla': iterate_history,
        'large_batch': large_batch_iterate_history,
    }
    plot_iterates_on_contour(
        iterate_histories, A, b, box_constraints, sampling_dist)
    plot_convergence_to_optimal(iterate_histories, sampling_dist)
    plt.show()


if __name__ == '__main__':
    logger = get_logger(log_out_file='./debug.log')
    main()
