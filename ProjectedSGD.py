import logging
from typing import Callable
import numpy as np
import numpy.typing as npt

from stepsize_rules.projected_SGD_stepsizes import ProjectedSGDStepsize
from subclass_instantiation import get_class_instance


class ProjectedSGD:
    '''Instead of solving the Stochastic Frank-Wolfe linear program at every
    iteration, try instead doing gradient updates, then projecting onto
    the constraint set\n
    pg 867: For constraints = {x | d @ x <= w} ∩ {x | a <= x <= b}, the
    projection should be quickly achievable'''
    def __init__(self, d: npt.NDArray[np.float64], w: float,
                 a: float, b: float,
                 gradient: Callable[[npt.NDArray[np.float64],
                                     npt.NDArray[np.float64]],
                                    npt.NDArray[np.float64]],
                 stochastic_sampler: Callable[[int], npt.NDArray[np.float64]],
                 stepsize_class_name: str = 'OneOverTStepsize'):
        '''Given that our domain is {x | d @ x <= w, a <= x <= b}

        :param d: param vector for the above constraint
        :param w: float for the above constraint
        :param a: coordinate-wise lower bound on the constraint
        :param b: coordinate-wise upper bound on the constraint
        :param gradient: function that takes in the current iterate x_k and
            a batch of samples from the stochastic_sampler. Returns gradient
            of the objective wrt x for each sample
        :param stochastic_sampler: function that takes in the number of
            samples, returns samples of the stochastic component
        :param stepsize_class_name: of the form "class_name(kwargs)"
            Gets parsed by a helper function to yield a subclass that
            implements a stepsize rule
        '''
        self.logger = logging.getLogger('__main__')
        self.d = d
        '''Weighted inner product vector in constraints'''
        self.w = w
        '''Weighted inner product total in constraints'''
        self.lower_bound = a
        '''For feasible x, we must have self.lower_bound <= x (all coords)'''
        self.upper_bound = b
        '''For feasible x, we must have x <= self.upper_bound (all coords)'''
        self.gradient_func = gradient
        self.stochastic_sampler = stochastic_sampler
        self.first_order_suboptimality_bound = 1e-5
        '''early termination condition based on the first-order condition
        for convexity (Ding & Udell, p4)'''
        self.stepsize_class = get_class_instance(
            stepsize_class_name, ProjectedSGDStepsize)

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
            a: float, b: float,
            max_iters: int = 10
            ) -> npt.NDArray[np.float64]:
        '''Project x_k onto {x | d @ x <= w} ∩ {x | a <= x <= b} using
        an alternating projections algo

        :param x_i: the original vector, possibly not in the constraint set
        :param d: vector in {x | d @ x <= w}
        :param w: scalar in {x | d @ x <= w}
        :param max_iters: only do at most this many alternating projections'''
        if np.all((a <= x_i) & (x_i <= b)) and d @ x_i <= w:
            return x_i
        for _ in range(max_iters):
            x_intermediate = self.project_onto_halfspace(x_i, d, w)
            x_i = np.clip(x_intermediate, a, b)
            if d @ x_i <= w:
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
            cur_stepsize = self.stepsize_class.get_stepsize(
                cur_iter, avg_gradient)
            x_k = x_k - cur_stepsize * avg_gradient
            x_k = self.run_alternating_projections_on(
                x_k, self.d, self.w, self.lower_bound, self.upper_bound)
            iterate_hist_list.append(x_k)
        return np.array(iterate_hist_list)
