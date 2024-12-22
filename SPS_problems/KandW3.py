'''Solves the raw material blending problem (adapted from Kall and Wallace)'''
import itertools as it
from dataclasses import dataclass, astuple
import numpy as np
import numpy.typing as npt
import cvxpy as cp

from logger import get_logger
from helper_functions import timing


@dataclass
class ProblemParams:
    '''Holds the params for a problem instance'''
    c: npt.NDArray[np.float64]
    '''c_i is the cost of material i'''
    a: npt.NDArray[np.float64]
    '''SPS claims a_(i, j) is the amount of material i required per unit of
    prod j\n
    pg 842: this interpretation doesn't actually make sense. That's fine
    though; I'm just doing this to get acquainted with SP methods'''
    f: npt.NDArray[np.float64]
    '''f_(j, t) is the cost of outsourcing prod j at time t'''
    b: float
    '''Material inventory capacity'''
    scenario_demands: npt.NDArray[np.float64]
    '''.shape = (num_scenarios, M, T). The realized demand matrices in
    each scenario'''
    scenario_probs: npt.NDArray[np.float64]
    '''.shape = (num_scenarios,)...probabilities of each scenario'''


def solve_w_cvxpy(c: npt.NDArray[np.float64], a: npt.NDArray[np.float64],
                  f: npt.NDArray[np.float64], b: float,
                  scenario_demands: npt.NDArray[np.float64],
                  scenario_probs: npt.NDArray[np.float64]
                  ) -> tuple[float, npt.NDArray[np.float64]]:
    '''pg 841: eliminate the "y" variables, recast this as a cvxpy problem.
    Should work well for the test data, but will suffer from the curse
    of dimensionality. Below:\n

    N = number of materials
    M = number of products
    T = number of time periods

    :param c: c_i is the cost of material i
    :param a: a_(i, j) is the amount of material i required per unit of prod j
    :param f: f_(j, t) is the cost of outsourcing prod j at time t
    :param b: inventory capacity
    :param scenario_demands: .shape = (num_scenarios, M, T). The realized
        demand matrices in each scenario
    :param scenario_probs: .shape = (num_scenarios,)
    :return: (objective value, x* as a (N, T) matrix)
    '''
    num_materials = len(c)
    _, num_time_periods = f.shape

    x = cp.Variable((num_materials, num_time_periods))
    products_made_from_storage = a.T @ x
    scenario_outsourcing_costs: list[cp.Expression] = []

    for _, scenario_demand in enumerate(scenario_demands):
        scenario_demand_gaps = cp.pos(
            scenario_demand - products_made_from_storage)
        scenario_outsourcing_cost = cp.sum(
            cp.multiply(f, scenario_demand_gaps))
        scenario_outsourcing_costs.append(
            scenario_outsourcing_cost)  # type: ignore

    deterministic_cost = cp.sum(c @ x)
    expected_outsourcing_cost: cp.Expression = (
        scenario_probs @ scenario_outsourcing_costs)  # type: ignore
    expected_total_cost = deterministic_cost + expected_outsourcing_cost
    objective = cp.Minimize(expected_total_cost)
    constraints: list[cp.Constraint] = [cp.sum(x) <= b, x >= 0]  # type: ignore
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    opt_value, x_star = prob.value, x.value
    return opt_value, x_star  # type: ignore


def get_test_data() -> ProblemParams:
    '''Get the (c, a, f, b, scenario_demands, scenario_probs)
    from the listed test data

    pg 842: the given "a" matrix needs to be transposed to match
        the SPS solution'''
    c = np.array([2., 3.], dtype=float)
    a = np.array([[2., 6.], [3., 3.4]], dtype=float).T
    f = np.array([[7., 10.], [12., 15.]], dtype=float)
    logger.debug(f'a=\n{a}')
    logger.debug(f'f=\n{f}')
    b = 50

    demand_rows = [[200., 180.], [180., 160.], [160., 140.]]
    scenario_demands = np.array([
        np.array(period_demands, dtype=float).T
        for period_demands in it.product(demand_rows, repeat=2)
    ])
    logger.debug(f'{scenario_demands.shape=}')
    logger.debug(f'{scenario_demands}')
    scenario_probs = np.array([
        0.06, 0.15, 0.09, 0.12, 0.16, 0.12, 0.12, 0.12, 0.06
    ])
    logger.debug(f'{sum(scenario_probs)=}')
    return ProblemParams(c, a, f, b, scenario_demands, scenario_probs)


def simulate_data(rng: np.random.Generator,
                  num_materials: int, num_products: int, num_time_periods: int,
                  num_demand_scenarios: int) -> ProblemParams:
    '''Simulate data for use in the above problem

    :param rng: rng used for random number generation
    :param num_materials: number of raw materials
    :param num_products: number of final products
    :param num_time_periods: number of time periods of potential demand
    :param num_demand_scenarios: total number of demand scenarios over
        all time periods
    :return: a ProblemParams instance with the randomly generated params
    '''
    c = rng.uniform(0, 5, size=num_materials)
    a = rng.uniform(0, 10, size=(num_materials, num_products))
    f = rng.uniform(0, 20, size=(num_products, num_time_periods))
    b = rng.uniform(0, 100)
    scenario_demands = rng.uniform(
        0, 300,
        size=(num_demand_scenarios, num_products, num_time_periods)
    )
    scenario_probs = (
        unnormalized := rng.uniform(size=num_demand_scenarios)
    ) / sum(unnormalized)
    return ProblemParams(c, a, f, b, scenario_demands, scenario_probs)


def get_empirical_objective(
        x: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64], a: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64], b: float,
        scenario_demands: npt.NDArray[np.float64],
        scenario_probs: npt.NDArray[np.float64],
        num_sims: int = int(1e5)) -> float:
    '''Given an initial material purchase matrix x, simulate realizations
    of demand and the resulting total costs
    Return the empirical average total cost'''
    rng = np.random.default_rng()
    realized_demand_idxs = rng.choice(
        len(scenario_probs), size=num_sims, p=scenario_probs)
    realized_demands = scenario_demands[realized_demand_idxs]
    products_made_from_storage = a.T @ x
    demand_gaps = np.maximum(
        realized_demands - products_made_from_storage, 0)
    logger.debug(f'{demand_gaps.shape=}')
    outsourcing_costs = np.sum(f * demand_gaps, axis=(1, 2))
    deterministic_cost = np.sum(c @ x)
    total_costs = deterministic_cost + outsourcing_costs
    return np.mean(total_costs)


def main():
    # problem_params = get_test_data()

    rng = np.random.default_rng()
    problem_params = simulate_data(
        rng, num_materials=5, num_products=6, num_time_periods=7,
        num_demand_scenarios=100)
    val_and_x_star, exec_time = timing(
        solve_w_cvxpy)(*astuple(problem_params))
    objective_value, x_star = val_and_x_star
    np.set_printoptions(precision=4, suppress=True)
    print(f'{objective_value=}')
    print(f'x_star=\n{x_star}')
    print(f'solve_w_cvxpy {exec_time=}s')
    empirical_objective = get_empirical_objective(
        x_star, *astuple(problem_params))
    print(f'{empirical_objective=}')


if __name__ == '__main__':
    logger = get_logger(log_out_file='./debug.log')
    main()
