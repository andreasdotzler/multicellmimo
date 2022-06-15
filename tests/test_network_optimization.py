import logging

import cvxpy as cp
import numpy as np
import pytest
from mcm.algorithms import (optimize_dual_decomp_subgradient,
                            optimize_dual_cuttingplane,
                            optimize_primal_sub)

from mcm.network_optimization import (I_C, I_C_Q, Network, dual_problem_app,
                                      optimize_app_phy, Network,
                                      optimize_network_app_network,
                                      optimize_network_app_phy,
                                      optimize_network_explict,
                                      proportional_fair, time_sharing,
                                      time_sharing_cvx, time_sharing_no_duals,
                                      timesharing_network, weighted_sum_rate)
from mcm.utils import InfeasibleOptimization

LOGGER = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def seed():
    np.random.seed(42)


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


def test_fixed_f():
    network = test_network(20, np.random.random)
    q_min = np.array([0.1] * 30)
    q_max = np.array([10.0] * 30)
    value, rates, alphas_network, [d_rates, w_min, w_max, d_f_network, d_sum_f, d_c_m_t] = timesharing_network(proportional_fair, network.users_per_mode_and_transmitter, network.As, q_min, q_max)
    # verfiy the dual variables
    assert 1/rates == pytest.approx(d_rates, 1e-3)
    assert w_min == pytest.approx(np.zeros(len(w_min)), rel=1e-3, abs=1e-3)
    assert w_min == pytest.approx(np.zeros(len(w_max)), rel=1e-3, abs=1e-3)

    la_m = {}
    c_m = {}
    f_cal = {}
    for mode, Am  in network.As.items():
        la_m[mode] = 0
        c_m[mode] = np.zeros(len(rates))        
        for t, A in Am.items():
            c_m_t = A@alphas_network[mode][t]
            f_cal[mode] = sum(alphas_network[mode][t])
            users = network.users_per_mode_and_transmitter[mode][t]
            # c_m_t rescaled to 100% resources
            if sum(alphas_network[mode][t]) >= 1e-3:
                c_m[mode][users] += 1/sum(alphas_network[mode][t]) * c_m_t
 
            la_m[mode] += 1/sum(alphas_network[mode][t]) * d_c_m_t[mode][t]@c_m_t
    for mode in la_m:
        assert la_m[mode] == pytest.approx(sum(d_f_network[mode].values()), 1e-3)
        assert la_m[mode] == pytest.approx(d_sum_f, 1e-3)
    # max_r U(r) : r in R <-> min_la max_r U(r) : la(r - sum_m f_m c_m)
    # la = 1/r -> la r = n_user -> sum_m fm la c_m = n_user
    # d_f = [la c_1, ..., la c_M]
    # eq:ver1_dual
    # sum d_f = xi
    # xi = max_m df
    # d_f_m = la c_m is the same for all active modes!

    for mode, c in c_m.items():
        assert d_rates@c == pytest.approx(30, 1e-3)

    me = sum(la_m.values()) / len (la_m.values())
    for la in la_m.values():
        # does this imply we already know the optimal dual parameters? n_users? -> optimal solution in one mode?
        assert la == pytest.approx(me, 1e-3)   

    fractions = {}    
    for mode, alphas_per_transmitter in alphas_network.items():
        for alphas in alphas_per_transmitter.values():
            fractions[mode] = sum(alphas)
            break

    v_n, r_n, alphas_n, d_f_n, F_t = network.util_fixed_fractions(fractions, proportional_fair, q_min, q_max)
    assert v_n == pytest.approx(value, 1e-3)
    assert r_n == pytest.approx(rates, 1e-3)
    for t, d_f_t_n in d_f_n.items():
        for mode, d in d_f_t_n.items():
            assert d == pytest.approx(d_f_network[mode][t])

@pytest.fixture(scope="function")
def seed():
    np.random.seed(41)

@pytest.mark.parametrize("network", [test_network(), test_network(20, np.random.random)])
def test_global_network(network, seed):
    q_min = np.array([0.1] * 30)
    #q_min[0] = 0.5
    q_max = np.array([10.0] * 30)
    #q_max[29] = 0.15
    value, rates, alphas_network, [user_rates, w_min, w_max, d_f_network, d_sum_f, d_c_m_t] = timesharing_network(proportional_fair, network.users_per_mode_and_transmitter, network.As, q_min, q_max)
 
    assert all(rates >= q_min*0.97)
    assert all(rates*0.97 <= q_max)
    verfiy_fractional_schedule(alphas_network)

    # Calculate by approximation algorithm
    opt_value, opt_q, _, _ = optimize_network_app_phy(proportional_fair, q_min, q_max, network)
    assert opt_value == pytest.approx(value, 1e-2)
    assert opt_q == pytest.approx(rates, rel=1e-1, abs=1e-1)


    opt_value_network, opt_q_network, alphas = optimize_network_app_network(proportional_fair, q_min, q_max, network)
    assert opt_value_network == pytest.approx(value, 1e-3)
    assert opt_q_network == pytest.approx(rates, rel=1e-1, abs=1e-1)

    opt_value_explicit, opt_q_explicit = optimize_network_explict(proportional_fair, q_min, q_max, network)
    assert opt_value_explicit == pytest.approx(value, 1e-3)
    assert opt_q_explicit == pytest.approx(rates, rel=1e-1, abs=1e-1)
    

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
