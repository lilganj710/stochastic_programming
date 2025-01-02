import logging
from typing import Callable
import numpy as np
import numpy.typing as npt


class StochasticFrankWolfe:
    '''Runs the Stochastic Frank-Wolfe algorithm given a function for
    computing the stochastic gradient & a polyhedral constraint set'''
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

    def knapsack_lp_solve(self, g: npt.NDArray[np.float64],
                          d: npt.NDArray[np.float64], w: float
                          ) -> npt.NDArray[np.float64]:
        '''max s @ g s.t. s >= 0, d @ s <= w\n
        pg 846: we don't need cvxpy to solve this. Instead, we can
            use a "knapsack mindset". Go "all in" on the s_i with
            highest "return on investment"
        Edge case: g < 0, in which case we choose s = 0'''
        s_opt = np.zeros_like(g)
        if np.all(g < 0):
            return s_opt
        s_returns_on_investment = g / d
        best_s_idx = np.argmax(s_returns_on_investment)
        s_opt[best_s_idx] = w / d[best_s_idx]
        return s_opt

    def binary_search_stepsize(self, cur_iterate: npt.NDArray[np.float64],
                               sfw_vector: npt.NDArray[np.float64],
                               stochastic_samples: npt.NDArray[np.float64],
                               precision: float = 1e-6
                               ) -> float:
        '''Convex function restricted to a line is still convex\n
        pg 861: use this and binary search to find the optimal stepsize
        in [0, 1], given:

        :param cur_iterate: current iterate in the algo to step from
        :param sfw_vector: result of the SFW linear oracle minimization
        :param stochastic_samples: self.gradient_func() needs these; we're
            working with stochastic gradients after all
        :param precision: end the binary search when the range width gets
            below this
        :return: optimal stepsize in [0, 1] (up to the given precision)
        '''
        low_stepsize = 0
        high_stepsize = 1
        while high_stepsize - low_stepsize >= precision:
            mid_stepsize = (low_stepsize + high_stepsize) / 2
            cur_gradient_vector = self.gradient_func(
                mid_stepsize*sfw_vector + (1-mid_stepsize)*cur_iterate,
                stochastic_samples
            ).mean(axis=0)
            scalar_derivative = cur_gradient_vector @ (sfw_vector-cur_iterate)
            if scalar_derivative < 0:
                low_stepsize = mid_stepsize
            elif scalar_derivative > 0:
                high_stepsize = mid_stepsize
            else:
                return mid_stepsize
        return (low_stepsize + high_stepsize) / 2

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
            # self.logger.debug(f'{avg_gradient=}')
            sfw_vector = self.knapsack_lp_solve(-avg_gradient, self.d, self.w)
            # cur_stepsize = self.binary_search_stepsize(
            #     x_k, sfw_vector, stochastic_samples)
            # self.logger.debug(f'{cur_iter=} -> {cur_stepsize=}')
            cur_stepsize = 2 / (cur_iter + 2)
            first_order_check = (
                avg_gradient @ (x_k - sfw_vector))
            if first_order_check <= self.first_order_suboptimality_bound:
                self.logger.debug(f'{first_order_check=} <= '
                                  f'{self.first_order_suboptimality_bound} '
                                  f'-> break on {cur_iter=}')
                break
            x_k = x_k + cur_stepsize * (sfw_vector - x_k)
            iterate_hist_list.append(x_k)
        return np.array(iterate_hist_list)
