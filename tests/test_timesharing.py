import random

import numpy as np
import pytest
from mcm.network_optimization import (
    I_C,
    U_Q,
    U_Q_conj,
    V,
    V_conj,
    dual_problem_app_f,
    optimize_app_phy,
    proportional_fair,
    weighted_sum_rate,
    wsr_for_A,
)
from mcm.no_utils import InfeasibleOptimization
from mcm.timesharing import (
    F_t_R_approx,
    time_sharing_cvx,
    F_t_R_approx_conj,
    timesharing_network,
    timesharing_network_dual,
)
from mcm.regions import Q_vector, R_m_t_approx

from .utils import gen_test_network


@pytest.mark.parametrize("util", (proportional_fair, weighted_sum_rate(np.ones(10))))
def test_U_Q_conj(util) -> None:

    Q = Q_vector(q_min=np.zeros(10), q_max=np.ones(10))
    la = np.random.random(10)
    val_conj, q = U_Q_conj(util, la, Q)
    val = U_Q(util, q, Q)
    assert la @ q == pytest.approx(val - val_conj)


def test_timesharing_wsr():
    A = np.array([[4, 1], [1, 2]])
    q_min = np.array([0, 0])
    q_max = np.array([10, 10])
    Q = Q_vector(q_min=q_min, q_max=q_max)
    n_users, n = A.shape
    R = R_m_t_approx(list(range(0, n_users)), A)

    weights = [1, 0]
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([4, 1], 1e-3)

    weights = [0, 1]
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([1, 2], 1e-3)

    weights = [1, 1]
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([4, 1], 1e-3)

    Q.q_min = np.array([0, 2])
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([1, 2], 1e-3)

    Q.q_min = np.array([0, 0])
    Q.q_max = np.array([2, 2])
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([2, 1 + 2 / 3])

    Q.q_min = np.array([0, 0])
    Q.q_max = np.array([0.5, 0])
    with pytest.raises(InfeasibleOptimization):
        _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)


def test_timesharing_fair(A):

    n_users = A.shape[0]
    q_min = np.array([0.1] * n_users)
    q_max = np.array([10.0] * n_users)
    # todo this currently breaks the test
    q_max[0] = 0.2

    Q = Q_vector(q_min=q_min, q_max=q_max)
    n_users, n = A.shape
    R = R_m_t_approx(list(range(0, n_users)), A)
    value, rates = time_sharing_cvx(proportional_fair, R, Q)
    (lambda_opt, mu) = R.dual_values()
    (w_min, w_max) = Q.dual_values()

    # verifiy KKT
    assert 1 / rates + w_min - w_max - lambda_opt == pytest.approx(
        np.zeros(len(rates)), rel=1e-2, abs=1e-1
    )
    assert all(lambda_opt @ (A - rates.reshape(len(rates), 1)) <= 0.001)
    q_app = np.minimum(q_max, np.maximum(q_min, 1 / lambda_opt))
    q_app[lambda_opt <= 0] = q_max[lambda_opt <= 0]

    # todo assert mu = max((weights @ A).tolist()[0])
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    dual_value_app, q_app = U_Q_conj(
        proportional_fair, lambda_opt, Q_vector(q_min, q_max)
    )
    assert dual_value_app == pytest.approx(sum(np.log(q_app)) - lambda_opt @ q_app)
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    rates_phy = dual_problem_phy(A, lambda_opt)
    dual_value_phy = lambda_opt @ rates
    assert lambda_opt @ rates_phy == pytest.approx(dual_value_phy)
    assert value == pytest.approx(dual_value_app + dual_value_phy)

    # now by iterative optimization
    wsr = I_C(A)
    opt_value, opt_q, _, _ = optimize_app_phy(proportional_fair, Q, wsr)
    assert value == pytest.approx(opt_value, rel=1e-3, abs=1e-1)
    assert rates == pytest.approx(opt_q, rel=1e-3, abs=1e-1)


def test_timesharing_fixed_fractions():
    As, network = gen_test_network()
    As, network = gen_test_network(20, np.random.random)
    q_min = np.array([0.05] * 30)
    q_max = np.array([20.0] * 30)
    Q = Q_vector(q_min=q_min, q_max=q_max)
    network.initialize_approximation(As)

    fractions = {m: 1 / 6 for m in network.modes}
    fractions["reuse1"] = 1 / 2
    for t_id, t in network.transmitters.items():
        R_m = {m: R.approx for m, R in t.R_m_t_s.items()}

        (
            value_2,
            rates_2,
        ) = F_t_R_approx(proportional_fair, fractions, t.users, R_m, Q[t.users])

        schedules_2 = {mode: R.alphas.value for mode, R in R_m.items()}
        c_m_2 = {mode: R.c.value for mode, R in R_m.items()}
        d_sum_f_2 = {mode: R.sum_alpha.dual_value for mode, R in R_m.items()}
        d_c_m_2 = {mode: R.r_in_A_x_alpha.dual_value for mode, R in R_m.items()}

        q_2 = np.zeros(len(t.users))
        for m in t.modes:
            assert t.users_per_mode[m] == t.users
            q_2 += fractions[m] * c_m_2[m]
        for user, q in zip(t.users, q_2):
            assert q == pytest.approx(rates_2[user], 1e-3)
        for m in t.modes:
            assert fractions[m] / q_2 == pytest.approx(d_c_m_2[m], abs=1e-3)
        la_2 = 1 / q_2

        la = {m: la_2 @ c for m, c in c_m_2.items()}
        (value_d, rates_d, f_d) = F_t_R_approx_conj(
            proportional_fair, la, t.users, R_m, Q[t.users]
        )
        for m, f in f_d.items():
            assert fractions[m] == pytest.approx(f, 1e-2)

        # concave functions!
        sum(fractions[m] * l for m, l in la.items()) == pytest.approx(value_2 - value_d)


def test_fixed_f():
    As, network = gen_test_network(20, np.random.random)
    # As, network = gen_test_network()
    q_min = np.array([0.05] * 30)
    q_max = np.array([20.0] * 30)
    Q = Q_vector(q_min=q_min, q_max=q_max)
    network.initialize_approximation(As)
    (value, rates, d_rates) = timesharing_network(proportional_fair, network, Q)
    # verfiy the dual variables
    d_c_m_t = network.d_c_m_t
    alphas_network = network.alphas_m_t
    (w_min, w_max) = Q.dual_values()
    assert 1 / rates + w_min - w_max - d_rates == pytest.approx(
        np.zeros(len(rates)), rel=1e-2, abs=1e-1
    )
    [dual_value, dual_rates, dual_c_m_t_s, _] = timesharing_network_dual(
        proportional_fair, d_c_m_t, network, q_min, q_max
    )

    mm = 0
    for m, c_t in dual_c_m_t_s.items():
        for t, c in c_t.items():
            mm += d_c_m_t[m][t] @ c

    la_m = {}
    c_m = {}
    f_cal = {}
    c_m_t_s = {m: {} for m in network.modes}
    for mode, Am in As.items():
        la_m[mode] = 0
        c_m[mode] = np.zeros(len(rates))
        for t, A in Am.items():
            c_m_t_s[mode][t] = c_m_t = (
                1 / sum(alphas_network[mode][t]) * A @ alphas_network[mode][t]
            )
            f_cal[mode] = sum(alphas_network[mode][t])
            users = network.transmitters[t].users_per_mode[mode]
            # c_m_t rescaled to 100% resources
            if sum(alphas_network[mode][t]) >= 1e-3:
                c_m[mode][users] += c_m_t

            la_m[mode] += 1 / sum(alphas_network[mode][t]) * d_c_m_t[mode][t] @ c_m_t

    Q = Q_vector(q_min=q_min, q_max=q_max)
    val_conj, q_conj = V_conj(network, proportional_fair, d_c_m_t, Q)
    val, q, f = V(network, proportional_fair, c_m_t_s, Q)
    mm = 0
    for mode, d_c_t in d_c_m_t.items():
        for t, d in d_c_t.items():
            mm = d @ c_m_t_s[mode][t]
    assert val_conj == pytest.approx(dual_value, 1e-3)
    assert val - val_conj == pytest.approx(mm, 1e-3)
    # for mode in la_m:
    #    assert la_m[mode] == pytest.approx(sum(d_f_network[mode].values()), 1e-3)
    #    assert la_m[mode] == pytest.approx(d_sum_f, 1e-3)
    # max_r U(r) : r in R <-> min_la max_r U(r) : la(r - sum_m f_m c_m)
    # la = 1/r -> la r = n_user -> sum_m fm la c_m = n_user
    # d_f = [la c_1, ..., la c_M]
    # eq:ver1_dual
    # sum d_f = xi
    # xi = max_m df
    # d_f_m = la c_m is the same for all active modes!

    # for mode, c in c_m.items():
    #    assert d_rates @ c == pytest.approx(30, 1e-3)

    # me = sum(la_m.values()) / len(la_m.values())
    # for la in la_m.values():
    # does this imply we already know the optimal dual parameters? n_users? -> optimal solution in one mode?
    # assert la == pytest.approx(me, 1e-3)

    fractions = {}
    for mode, alphas_per_transmitter in alphas_network.items():
        for alphas in alphas_per_transmitter.values():
            fractions[mode] = sum(alphas)
            break

    # v_n, r_n, alphas_n, d_f_n, F_t = network.scheduling(
    #    fractions, proportional_fair, q_min, q_max
    # )
    # assert v_n == pytest.approx(value, 1e-3)
    # assert r_n == pytest.approx(rates, 1e-1)
    # for t, d_f_t_n in d_f_n.items():
    #    for mode, d in d_f_t_n.items():
    #        assert d == pytest.approx(d_f_network[mode][t], 1e-2)
    for t in network.transmitters.values():
        (
            F_t,
            r_t,
            alpha_t,
            c_m,
            [d_f_t_m, d_c_m, la],
        ) = t.scheduling(fractions, proportional_fair, Q[t.users])
        v_a, q, c = dual_problem_app_f(proportional_fair, d_c_m, fractions, Q[t.users])

        v_p = 0
        for mode, As in t.As_per_mode.items():
            w, r = wsr_for_A(d_c_m[mode], As)
            v_p += w
        assert F_t == pytest.approx(v_a + v_p, 1e-2)


def dual_problem_phy(A, weights):
    max_i = np.argmax(weights @ A)
    rates = A[:, max_i]
    return rates
