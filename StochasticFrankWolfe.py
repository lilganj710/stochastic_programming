import logging
from typing import Callable
import numpy as np
import numpy.typing as npt
import cvxpy as cp


class StochasticFrankWolfe:
    '''Runs the Stochastic Frank-Wolfe algorithm given a function for
    computing the stochastic gradient & a polyhedral constraint set'''
    def __init__(self, A: npt.NDArray[np.float64] | None,
                 b: npt.NDArray[np.float64] | None,
                 box_constraints: tuple[float | None, float | None],
                 gradient: Callable[[npt.NDArray[np.float64],
                                     npt.NDArray[np.float64]],
                                    npt.NDArray[np.float64]],
                 stochastic_sampler: Callable[[int],
                                              npt.NDArray[np.float64]],
                 iterate_dims: int | None = None):
        '''Given that our domain is {x | Ax <= b & x in box_constraints}

        :param A: (optional) matrix defining the polyhedral constraints
        :param b: (optional) vector defining the polyhedral constraints
        :param iterate_dims: if A isn't passed, I can explicitly specify
            the number of dimensions in the iterate here
        :param box_constraints: to avoid large identity matrices in A, define
            the box constraints separately
        :param gradient: function that takes in the current iterate x_k and
            a batch of samples from the stochastic_sampler. Returns gradient
            of the objective wrt x for each sample
        :param stochastic_sampler: function that takes in the number of
            samples, returns samples of the stochastic component
        '''
        self.logger = logging.getLogger('__main__')
        if A is not None and A.ndim != 2:
            raise ValueError(f'Expected polyhedral A.ndim=2, not {A.ndim=}')
        if A is None and iterate_dims is None:
            raise ValueError('If no polyhedral constraints, we need to '
                             'explicitly pass in the iterate_dims')
        self.A = A
        self.b = b
        self.box_constraints = box_constraints
        self.gradient_func = gradient
        self.stochastic_sampler = stochastic_sampler
        self.LMO, self.gradient_param, self.LMO_var = self.get_LMO(
            A, b, iterate_dims)
        self.logger.debug(f'{self.LMO.is_dpp()=}')

    def get_LMO(self, A: npt.NDArray[np.float64] | None,
                b: npt.NDArray[np.float64] | None,
                iterate_dims: int | None
                ) -> tuple[cp.Problem, cp.Parameter, cp.Variable]:
        '''After initializing constraints, get
        the Linear Minimization Oracle as a cvxpy linear program

        :return: references to the LMO problem, also references to the
            gradient param and optimization variable
        Also return a reference to the parametrizable gradient (the
        LP will follow DPP rules)'''
        all_constraints = []
        if iterate_dims is None:
            if A is not None:
                s = cp.Variable(A.shape[1])
                g = cp.Parameter(A.shape[1])
                all_constraints.append(A @ s <= b)
            else:
                raise ValueError('iterate_dims is None and A is None')
        else:
            s = cp.Variable(iterate_dims)
            g = cp.Parameter(iterate_dims)

        lower_box, upper_box = self.box_constraints
        if lower_box is not None:
            all_constraints.append(s >= lower_box)
            self.logger.debug(f'added {lower_box=} constraint')
        if upper_box is not None:
            all_constraints.append(s <= upper_box)
            self.logger.debug(f'added {upper_box=} constraint')
        prob = cp.Problem(cp.Minimize(s @ g), all_constraints)  # type: ignore
        return prob, g, s

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
            self.gradient_param.value = avg_gradient
            self.LMO.solve(solver=cp.CLARABEL)
            self.logger.debug(f'after solving {self.LMO.status=}')
            cur_stepsize = 2 / (cur_iter + 2)
            x_k = x_k + cur_stepsize * (self.LMO_var.value - x_k)
            iterate_hist_list.append(x_k)
        return np.array(iterate_hist_list)
