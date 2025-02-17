import logging
from typing import Callable
import numpy as np
import numpy.typing as npt
import scipy.linalg as sl  # type: ignore


class StochasticNewton:
    '''Try implementing a stochastic version of Newton's method
    Given the stochastic gradient & stochastic Hessian,
    minimize the 2nd Taylor expansion of f() (see Boyd pg 498)\n
    Then, project this onto the constraint set using previous code'''
    def __init__(self, d: npt.NDArray[np.float64], w: float,
                 gradient: Callable[[npt.NDArray[np.float64],
                                     npt.NDArray[np.float64]],
                                    npt.NDArray[np.float64]],
                 hessian: Callable[[npt.NDArray[np.float64],
                                    npt.NDArray[np.float64]],
                                   npt.NDArray[np.float64]],
                 stochastic_sampler: Callable[[int], npt.NDArray[np.float64]]):
        '''Still, our domain is {x | d @ x <= w, x >= 0}

        :param d: param vector for the above constraint
        :param w: float for the above constraint
        :param gradient: gradient as function of current iterate and
            stochastic samples
        :param hessian: hessian as function of current iterate and
            stochastic samples
        :param stochastic_sampler: stochastic samples as function of
            num samples
        '''
        self.logger = logging.getLogger('__main__')
        self.d = d
        '''Weighted inner product vector in constraints'''
        self.w = w
        '''Weighted inner product total in constraints'''
        self.gradient_func = gradient
        self.hessian_func = hessian
        self.stochastic_sampler = stochastic_sampler

    def project_onto_halfspace(
            self, y: npt.NDArray[np.float64],
            a: npt.NDArray[np.float64], b: float) -> npt.NDArray[np.float64]:
        '''Project y onto {x | a @ x <= b}'''
        if a @ y <= b:
            return y
        else:
            return y - a * ((a @ y - b) / (a @ a))

    def run_alternating_projections_on(
            self, x_i: npt.NDArray[np.float64],
            d: npt.NDArray[np.float64], w: float,
            max_iters: int = 10
            ) -> npt.NDArray[np.float64]:
        '''Project x_k onto {x | d @ x <= w} ∩ {x | x >= 0} using
        an alternating projections algo

        :param x_i: the original vector, possibly not in the constraint set
        :param d: vector in {x | d @ x <= w}
        :param w: scalar in {x | d @ x <= w}
        :param max_iters: only do at most this many alternating projections'''
        if np.all(x_i >= 0) and d @ x_i <= w:
            self.logger.debug(f'No projections needed on {x_i=}')
            return x_i
        for _ in range(max_iters):
            x_intermediate = self.project_onto_halfspace(x_i, d, w)
            x_i = np.maximum(x_intermediate, 0)
            if d @ x_i <= w:
                break
        return x_i

    def get_newton_step(self, x_k: npt.NDArray[np.float64],
                        stochastic_samples: npt.NDArray[np.float64]
                        ) -> npt.NDArray[np.float64]:
        '''Get the "(inv hessian) @ gradient" Newton step (based on
        stochastic approximations for each)

        :param x_k: current iterate for the Newton step
        :param stochastic_samples: used to get stochastic gradient &
            stochastic Hessian
        :return: the Newton step
        '''
        gradient_components = self.gradient_func(x_k, stochastic_samples)
        avg_gradient = np.mean(gradient_components, axis=0)
        hessian_components = self.hessian_func(x_k, stochastic_samples)
        self.logger.debug(f'{gradient_components.shape=}, '
                          f'{hessian_components.shape=}')
        avg_hessian = np.mean(hessian_components, axis=0)
        return sl.solve(avg_hessian, avg_gradient)

    def iterate_from(self, x_0: npt.NDArray[np.float64],
                     batch_size: int = 10, num_iters: int = 50
                     ) -> npt.NDArray[np.float64]:
        '''Perform num_iters of this Stochastic Frank Wolfe instance starting
        from x_0

        :param batch_size: num samples to draw from the stochastic_sampler
            in computing the stochastic_gradient
        :return: the iterate history'''
        x_k = x_0
        iterate_hist_list: list[npt.NDArray[np.float64]] = [x_0]
        for cur_iter in range(1, num_iters+1):
            stochastic_samples = self.stochastic_sampler(batch_size)
            newton_step = self.get_newton_step(x_k, stochastic_samples)
            cur_stepsize = 1 / (cur_iter + 1)
            x_k = x_k - cur_stepsize * newton_step
            x_k = self.run_alternating_projections_on(x_k, self.d, self.w)
            iterate_hist_list.append(x_k)
        return np.array(iterate_hist_list)
