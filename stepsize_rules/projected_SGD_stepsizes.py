'''Stepsize rules I can try out in Projected Stochastic Gradient Descent'''
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class ProjectedSGDStepsize(ABC):
    '''Get a stepsize to use in Projected SGD. I chose to make this a class
    because some stepsize rules may rely on an internal state'''

    @abstractmethod
    def get_stepsize(
            self, cur_iter: int,
            cur_grad: npt.NDArray[np.float64]) -> float:
        '''Return the next stepsize according to this stepsize rule

        :param cur_iter: cur iteration number (0-indexed)
        :param cur_grad: gradient that's about to be smoothed into the iterate
        :return: stepsize α. Will set cur_iterate -= α * cur_grad
        '''


class OneOverTStepsize(ProjectedSGDStepsize):
    '''A common, simple stepsize rule: 1 / (cur_iter + 1)'''
    def get_stepsize(
            self, cur_iter: int,
            cur_grad: npt.NDArray[np.float64]) -> float:
        return 1 / (cur_iter + 1)


class MultivariateKestenStepsize(ProjectedSGDStepsize):
    '''Inspired by the univariate example on pg 456, only decrease
    the stepsize when the last_grad @ cur_grad < 0\n
    This corresponds to the "change in direction" we'd see after
    overshooting the min'''
    def __init__(self):
        self.last_grad = np.array([])
        self.sign_changes_so_far = 0
        '''Records the number of times last_grad @ cur_grad < 0'''

    def get_stepsize(
            self, cur_iter: int,
            cur_grad: npt.NDArray[np.float64]) -> float:
        if len(self.last_grad) > 0 and self.last_grad @ cur_grad < 0:
            self.sign_changes_so_far += 1
        self.last_grad = cur_grad
        return 1 / (1 + self.sign_changes_so_far - 1)
