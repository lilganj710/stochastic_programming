'''Solve the PROD_MIX problem. Continuous distributions of the stochastic
components invite usage of Stochastic Frank-Wolfe'''
import functools as ft
import numpy as np
import numpy.typing as npt

from logger import get_logger
import sys
sys.path.append('C:/stochastic_programming')
from StochasticFrankWolfe import StochasticFrankWolfe  # noqa: E402


def stochastic_sampler_for_test(num_samples: int) -> npt.NDArray[np.float64]:
    '''Concatenate the given A matrix and h vector so that
    each row has 10 entries\n
    Final return .shape = (num_samples, 10)'''
    uniform_lbs = [3.5, 8.0, 6.0, 9.0, 0.8, 0.8, 2.5, 36.0]
    uniform_ubs = [4.5, 10.0, 8.0, 11.0, 1.2, 1.2, 3.5, 44.0]
    flattened_A_matrix_samples = rng.uniform(
        low=uniform_lbs, high=uniform_ubs,
        size=(num_samples, len(uniform_lbs)))
    normal_locs = [6000., 4000.]
    normal_scales = [100., 50.]
    h_vector_samples = rng.normal(
        normal_locs, normal_scales,
        size=(num_samples, len(normal_locs)))
    all_samples = np.concatenate(
        (flattened_A_matrix_samples, h_vector_samples), axis=1)
    logger.debug(f'{all_samples.shape=}')
    return all_samples


def gradient_func_for_test(
        iterate: npt.NDArray[np.float64],
        stochastic_samples: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
    '''Computes gradients wrt iterate given stochastic_samples
    For this problem, more details on pg 845

    :param c: profits per unit of each product class
    :param q: cost per man-hour of extra labor for each workstation'''
    flattened_A_matrix_samples = stochastic_samples[:, :8]
    A_matrix_samples = np.reshape(flattened_A_matrix_samples,
                                  (flattened_A_matrix_samples.shape[0], 2, 4))
    h_vector_samples = stochastic_samples[:, 8:]
    logger.debug(f'{np.transpose(A_matrix_samples, (0, 2, 1)).shape=}')
    man_hours_per_workstation = A_matrix_samples @ iterate
    extra_hours_needed = man_hours_per_workstation - h_vector_samples
    masked_qs = np.where(
        extra_hours_needed > 0,
        np.tile(q, (len(stochastic_samples), 1)),
        0
    )
    logger.debug(f'masked_qs=\n{masked_qs}')
    conditional_gradients = np.transpose(
        A_matrix_samples, (0, 2, 1)) @ masked_qs[..., None]
    conditional_gradients = np.squeeze(conditional_gradients, axis=-1)
    logger.debug(f'{conditional_gradients.shape=}')
    return conditional_gradients - c


def main():
    c = np.array([12., 20., 18., 40.])
    q = np.array([5., 10.])
    logger.debug(f'{c=}, {q=}')
    gradient_func = ft.partial(gradient_func_for_test, c=c, q=q)
    sfw_instance = StochasticFrankWolfe(
        A=None, b=None, box_constraints=(0, None),
        gradient=gradient_func,
        stochastic_sampler=stochastic_sampler_for_test,
        iterate_dims=4
    )
    x_0 = np.zeros(4)
    iterate_history = sfw_instance.iterate_from(x_0, batch_size=10)
    logger.debug(f'iterate_history=\n{iterate_history}')


if __name__ == '__main__':
    logger = get_logger(log_out_file='./debug.log')
    rng = np.random.default_rng()
    main()
