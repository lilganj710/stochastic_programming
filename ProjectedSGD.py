import logging
from typing import Callable
import numpy as np
import numpy.typing as npt


class ProjectedSGD:
    '''Instead of solving the Stochastic Frank-Wolfe linear program at every
    iteration, try instead doing gradient updates, then projecting onto
    the constraint set\n
    pg 867: For constraints = {x | d @ x <= w} ∩ {x | x >= 0}, the projection
    should be quickly achievable'''
    def __init__(self, d: npt.NDArray[np.float64], w: float,
                 gradient: Callable[[npt.NDArray[np.float64],
                                     npt.NDArray[np.float64]],
                                    npt.NDArray[np.float64]],
                 stochastic_sampler: Callable[[int],
                                              npt.NDArray[np.float64]]):
        '''Given that our domain is {x | d @ x <= w, x >= 0}

        :param d: param vector for the above constraint
        :param w: float for the above constraint
        :param gradient: function that takes in the current iterate x_k and
            a batch of samples from the stochastic_sampler. Returns gradient
            of the objective wrt x for each sample
        :param stochastic_sampler: function that takes in the number of
            samples, returns samples of the stochastic component
        '''
        self.logger = logging.getLogger('__main__')
        self.d = d
        '''Weighted inner product vector in constraints'''
        self.w = w
        '''Weighted inner product total in constraints'''
        self.gradient_func = gradient
        self.stochastic_sampler = stochastic_sampler
        self.first_order_suboptimality_bound = 1e-5
        '''early termination condition based on the first-order condition
        for convexity (Ding & Udell, p4)'''

    def project_onto_halfspace(
            self, y: npt.NDArray[np.float64],
            a: npt.NDArray[np.float64], b: float) -> npt.NDArray[np.float64]:
        '''Project y onto {x | a @ x <= b}'''
        if a @ y <= b:
            return y
        else:
            return y - a * ((a @ y - b) / (a @ a))

    def project_iterate(self, x_i: npt.NDArray[np.float64],
                        max_iters: int = 10
                        ) -> npt.NDArray[np.float64]:
        '''Project x_k onto {x | d @ x <= w} ∩ {x | x >= 0} using
        an alternating projections algo

        :param x_i: the original iterate, possibly not in the constraint set
        :param max_iters: only do at most this many alternating projections'''
        if np.all(x_i >= 0) and self.d @ x_i <= self.w:
            self.logger.debug(f'No projections needed on {x_i=}')
            return x_i
        for _ in range(max_iters):
            x_intermediate = self.project_onto_halfspace(x_i, self.d, self.w)
            x_i = np.maximum(x_intermediate, 0)
            if self.d @ x_i <= self.w:
                break
        return x_i

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
            gradient_components = self.gradient_func(x_k, stochastic_samples)
            avg_gradient = np.mean(gradient_components, axis=0)
            self.logger.debug(f'{avg_gradient=}')
            cur_stepsize = 2 / (cur_iter + 2)
            x_k = x_k - cur_stepsize * avg_gradient
            x_k = self.project_iterate(x_k)
            iterate_hist_list.append(x_k)
        return np.array(iterate_hist_list)