import logging
import numpy as np
import cvxpy as cp
import pytest

from mcm.utils import InfeasibleOptimization
from mcm.network_optimization import time_sharing, optimize_network_app_phy, timesharing_network, proportional_fair, \
    weighted_sum_rate, time_sharing_dual, Network, optimize_network_app_network

LOGGER = logging.getLogger(__name__)

@pytest.fixture(scope="function", autouse=True)
def seed():
    np.random.seed(42)


def test_timesharing_wsr():
    A = np.array([[4, 1], [1, 2]])
    q_min = np.array([0, 0])
    q_max = np.array([10, 10])

    weights = [1, 0]
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([4, 1])
    weights = [0, 1]
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([1, 2])
    weights = [1, 1]
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([4,1])
    q_min = np.array([0, 2])
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([1,2])
    q_min = np.array([0, 0])
    q_max = np.array([2, 2])
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([2,1 + 2/3])
    q_min = np.array([0, 0])
    q_max = np.array([0.5, 0])
    with pytest.raises(InfeasibleOptimization):
        time_sharing(weighted_sum_rate(weights), A, q_min, q_max)


# These tow functions are my very strange way to build a wsr from the time_sharing use an object
def time_sharing_no_duals(costs, A, q_min=None, q_max=None):
    value, rates, _, _ = time_sharing(costs, A, q_min=q_min, q_max=q_max)
    return value, rates

def I_C(A):
    return lambda weights: time_sharing_no_duals(weighted_sum_rate(weights), A)

def I_C_Q(A, q_min, q_max):
    return lambda weights: time_sharing_no_duals(weighted_sum_rate(weights), A, q_min, q_max)


@pytest.mark.parametrize("A", [np.array([[0.4, 0.1], [0.1, 0.2]]).T, np.random.rand(10,200)*3])
def test_timesharing_fair(A):

    n_users = A.shape[0]
    q_min = np.array([0.1]*n_users)
    q_max = np.array([10.0]*n_users)
    q_max[0] = 0.2

    value, rates, alpha, [lambda_opt, w_min, w_max, mu] = time_sharing_dual(proportional_fair, A, q_min, q_max)
    assert 1/rates + w_min - w_max - lambda_opt == pytest.approx(np.zeros(len(rates)), rel=1e-2, abs=1e-1)

    # verifiy KKT
    q_app = np.minimum(q_max, np.maximum(q_min, 1 / lambda_opt))
    q_app[lambda_opt <= 0] = q_max[lambda_opt <= 0]
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    dual_value_app, q_app = dual_problem_app(lambda_opt, q_max, q_min)
    assert dual_value_app == pytest.approx(sum(np.log(q_app)) - lambda_opt@q_app)
    assert rates == pytest.approx(q_app)

    rates_phy = dual_problem_phy(A, lambda_opt)
    dual_value_phy = lambda_opt @ rates
    assert lambda_opt @ rates_phy == pytest.approx(dual_value_phy)
    assert value == pytest.approx(dual_value_app + dual_value_phy)

    # now by iterative optimization
    wsr = I_C(A)
    opt_value, opt_q = optimize_network_app_phy(q_min, q_max, wsr)
    assert value == pytest.approx(opt_value)
    assert rates == pytest.approx(opt_q)

    wsr_C_Q = I_C_Q(A, q_min, q_max)
    opt_value_C_Q, opt_C_Q = optimize_network_app_phy(q_min = np.array([0.001] * n_users), q_max = np.array([10] * n_users), wsr_phy=wsr_C_Q)
    assert value == pytest.approx(opt_value_C_Q)
    assert rates == pytest.approx(opt_C_Q)



    # TODO check more KKT
    #assert np.isclose(dual_value + max((lambda @ A).tolist()[0]), prob.value)
    #dual_value = prob.value - max((lambda_p @ A).tolist()[0])
    #assert dual_value2.value == pytest.approx(dual_value)
    #mu = max(lambda_phy @ A)
    #print(
    #    f"The optimimum of the dual approximation is {prob_dual.value} + {mu} = {prob_dual.value + mu}"
    #)
    #print(f"The optimal rates of the dual approximation are {q.value}")


def dual_problem_phy(A, weights):
    opt_rates = cp.Variable(len(weights))
    cost_function = weighted_sum_rate(weights)
    value, rates, alpha, dual_variables = time_sharing(cost_function, A)
    max_i = np.argmax(weights@A)
    rates2 = A[:,max_i]
    assert pytest.approx(rates) == rates
    return rates2


def dual_problem_app(weights, q_max, q_min):
    q = cp.Variable(len(q_max))
    cost_dual = (
            cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(q)) - weights @ q
    )
    constraints_dual = [q >= q_min, q <= q_max]
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve(solver=cp.SCS, eps=1e-8)
    return prob_dual.value, q.value



def test_network(n_rate_points_per_mode_and_transmitter = 20, sample_function=np.ones):
    users_per_mode_and_transmitter = {
        'reuse1': {0: list(range(0, 10)), 1: list(range(10, 20)), 2: list(range(20, 30))},
        'reuse3-0': {0: list(range(0, 10))},
        'reuse3-1': {1: list(range(10, 20))},
        'reuse3-2': {2: list(range(20, 30))},
    }

    As = {}
    for mode, transmitters_and_users in users_per_mode_and_transmitter.items():
        Am = As[mode] = {}
        for t, users in transmitters_and_users.items():
            Am[t] = sample_function((len(users), n_rate_points_per_mode_and_transmitter))
            if mode == 'reuse1':
                Am[t] = Am[t] * 1 / 3
    return Network(users_per_mode_and_transmitter, As)


@pytest.mark.parametrize("network", [test_network(), test_network(20, np.random.random)])
def test_global_network(network):
    q_min = np.array([0.1] * 30)
    q_min[0] = 0.5
    q_max = np.array([10.0] * 30)
    q_max[29] = 0.15
    value, rates, alphas, dual_values = timesharing_network(proportional_fair, network.users_per_mode_and_transmitter, network.As, q_min, q_max)
    # TODO recalculate the rates
    assert all(rates >= q_min*0.99)
    assert all(rates*0.99 <= q_max)
    verfiy_fractional_schedule(alphas)

    # Calculate by approximation algorithm

    opt_value, opt_q = optimize_network_app_phy(q_min, q_max, network.wsr)
    assert opt_value == pytest.approx(value, 1e-3)
    assert opt_q == pytest.approx(rates, rel=1e-1, abs=1e-1)

    opt_value_per_mode, opt_q_per_mode = optimize_network_app_phy(q_min, q_max, network.wsr_per_mode)
    assert opt_value_per_mode == pytest.approx(value, 1e-3)
    assert opt_q_per_mode == pytest.approx(rates, rel=1e-1, abs=1e-1)

    opt_value_network, opt_q_network = optimize_network_app_network(q_min, q_max, network)
    assert opt_value_network == pytest.approx(value, 1e-3)
    assert opt_q_network == pytest.approx(rates, rel=1e-1, abs=1e-1)



def verfiy_fractional_schedule(alphas):
    total_time = 0
    for mode, alphas_per_mode in alphas.items():
        sum_per_transmitter_and_mode = []
        for alpha in alphas_per_mode.values():
            sum_per_transmitter_and_mode.append(sum(alpha))
        mean_sum = np.mean(sum_per_transmitter_and_mode)
        LOGGER.info(f"Mode {mode} - Fraction {mean_sum}")
        assert sum_per_transmitter_and_mode == pytest.approx(np.ones(len(sum_per_transmitter_and_mode)) * mean_sum, rel=1e-2, abs=1e-2)
        total_time += mean_sum
    assert total_time == pytest.approx(1, 1e-2)