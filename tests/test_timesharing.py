import numpy as np
import pytest

from mcm.network_optimization import (
    I_C,
    I_C_Q,
    dual_problem_app,
    optimize_app_phy,
    proportional_fair,
    time_sharing,
    time_sharing_cvx,
    weighted_sum_rate,
    timesharing_network,
    wsr_for_A, 
    dual_problem_app_f
)


from mcm.no_utils import InfeasibleOptimization
from .utils import gen_test_network

def test_timesharing_wsr():
    A = np.array([[4, 1], [1, 2]])
    q_min = np.array([0, 0])
    q_max = np.array([10, 10])

    weights = [1, 0]
    value, rates, alpha, lambda_phy = time_sharing(
        weighted_sum_rate(weights), A, q_min, q_max
    )
    assert rates.tolist() == pytest.approx([4, 1], 1e-3)
    weights = [0, 1]
    value, rates, alpha, lambda_phy = time_sharing(
        weighted_sum_rate(weights), A, q_min, q_max
    )
    assert rates.tolist() == pytest.approx([1, 2], 1e-3)
    weights = [1, 1]
    value, rates, alpha, lambda_phy = time_sharing(
        weighted_sum_rate(weights), A, q_min, q_max
    )
    assert rates.tolist() == pytest.approx([4, 1], 1e-3)
    q_min = np.array([0, 2])
    value, rates, alpha, lambda_phy = time_sharing(
        weighted_sum_rate(weights), A, q_min, q_max
    )
    assert rates.tolist() == pytest.approx([1, 2], 1e-3)
    q_min = np.array([0, 0])
    q_max = np.array([2, 2])
    value, rates, alpha, lambda_phy = time_sharing(
        weighted_sum_rate(weights), A, q_min, q_max
    )
    assert rates.tolist() == pytest.approx([2, 1 + 2 / 3])
    q_min = np.array([0, 0])
    q_max = np.array([0.5, 0])
    with pytest.raises(InfeasibleOptimization):
        time_sharing(weighted_sum_rate(weights), A, q_min, q_max)


def test_timesharing_fair(A):

    n_users = A.shape[0]
    q_min = np.array([0.1] * n_users)
    q_max = np.array([10.0] * n_users)
    # todo this currently breaks the test
    q_max[0] = 0.2

    value, rates, alpha, [lambda_opt, w_min, w_max, mu] = time_sharing_cvx(
        proportional_fair, A, q_min, q_max
    )

    # verifiy KKT
    assert 1 / rates + w_min - w_max - lambda_opt == pytest.approx(
        np.zeros(len(rates)), rel=1e-2, abs=1e-1
    )
    assert all(lambda_opt @ (A - rates.reshape(len(rates), 1)) <= 0.001)
    q_app = np.minimum(q_max, np.maximum(q_min, 1 / lambda_opt))
    q_app[lambda_opt <= 0] = q_max[lambda_opt <= 0]

    # todo assert mu = max((weights @ A).tolist()[0])
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    dual_value_app, q_app = dual_problem_app(
        proportional_fair, lambda_opt, q_max, q_min
    )
    assert dual_value_app == pytest.approx(sum(np.log(q_app)) - lambda_opt @ q_app)
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    rates_phy = dual_problem_phy(A, lambda_opt)
    dual_value_phy = lambda_opt @ rates
    assert lambda_opt @ rates_phy == pytest.approx(dual_value_phy)
    assert value == pytest.approx(dual_value_app + dual_value_phy)

    # now by iterative optimization
    wsr = I_C(A)
    opt_value, opt_q, _, _ = optimize_app_phy(proportional_fair, q_min, q_max, wsr)
    assert value == pytest.approx(opt_value, rel=1e-3, abs=1e-1)
    assert rates == pytest.approx(opt_q, rel=1e-3, abs=1e-1)

def test_fixed_f():
    As, network = gen_test_network(20, np.random.random)
    As, network = gen_test_network()
    q_min = np.array([0.05] * 30)
    q_max = np.array([20.0] * 30)
    network.initialize_approximation(As)    
    (
        value,
        rates,
        alphas_network,
        [d_rates, w_min, w_max, d_f_network, d_sum_f, d_c_m_t],
    ) = timesharing_network(proportional_fair, network, q_min, q_max)
    # verfiy the dual variables
    assert 1 / rates + w_min - w_max - d_rates == pytest.approx(
        np.zeros(len(rates)), rel=1e-2, abs=1e-1
    )


    la_m = {}
    c_m = {}
    f_cal = {}
    for mode, Am in As.items():
        la_m[mode] = 0
        c_m[mode] = np.zeros(len(rates))
        for t, A in Am.items():
            c_m_t = A @ alphas_network[mode][t]
            f_cal[mode] = sum(alphas_network[mode][t])
            users = network.transmitters[t].users_per_mode[mode]
            # c_m_t rescaled to 100% resources
            if sum(alphas_network[mode][t]) >= 1e-3:
                c_m[mode][users] += 1 / sum(alphas_network[mode][t]) * c_m_t

            la_m[mode] += 1 / sum(alphas_network[mode][t]) * d_c_m_t[mode][t] @ c_m_t
    #for mode in la_m:
    #    assert la_m[mode] == pytest.approx(sum(d_f_network[mode].values()), 1e-3)
    #    assert la_m[mode] == pytest.approx(d_sum_f, 1e-3)
    # max_r U(r) : r in R <-> min_la max_r U(r) : la(r - sum_m f_m c_m)
    # la = 1/r -> la r = n_user -> sum_m fm la c_m = n_user
    # d_f = [la c_1, ..., la c_M]
    # eq:ver1_dual
    # sum d_f = xi
    # xi = max_m df
    # d_f_m = la c_m is the same for all active modes!

    #for mode, c in c_m.items():
    #    assert d_rates @ c == pytest.approx(30, 1e-3)

    #me = sum(la_m.values()) / len(la_m.values())
    #for la in la_m.values():
        # does this imply we already know the optimal dual parameters? n_users? -> optimal solution in one mode?
        #assert la == pytest.approx(me, 1e-3)

    fractions = {}
    for mode, alphas_per_transmitter in alphas_network.items():
        for alphas in alphas_per_transmitter.values():
            fractions[mode] = sum(alphas)
            break
    fractions = {m: 1/6 for m in fractions}
    fractions['reuse1'] = 1/2
    #v_n, r_n, alphas_n, d_f_n, F_t = network.scheduling(
    #    fractions, proportional_fair, q_min, q_max
    #)
    #assert v_n == pytest.approx(value, 1e-3)
    #assert r_n == pytest.approx(rates, 1e-1)
    #for t, d_f_t_n in d_f_n.items():
    #    for mode, d in d_f_t_n.items():
    #        assert d == pytest.approx(d_f_network[mode][t], 1e-2)
    for t in network.transmitters.values():
        (
            F_t,
            r_t,
            alpha_t,
            [lambdas, w_min, w_max, d_f_t_m, d_c_m],
        ) = t.scheduling(fractions, proportional_fair, q_min[t.users], q_max[t.users])
        v_a, q, c = dual_problem_app_f(proportional_fair, d_c_m, fractions, q_max[t.users], q_min[t.users])

        v_p = 0
        for mode, As in t.As_per_mode.items():
            w, r = wsr_for_A(d_c_m[mode], As)
            v_p += w
        assert F_t == pytest.approx(v_a + v_p, 1e-2)




def dual_problem_phy(A, weights):
    max_i = np.argmax(weights @ A)
    rates = A[:, max_i]
    return rates
