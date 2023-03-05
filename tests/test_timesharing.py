import numpy as np
import pytest
from mcm.network_optimization import (
    I_C,
    U_Q,
    U_Q_conj,
    V_conj,
    V_new,
    optimize_app_phy,
    proportional_fair,
    weighted_sum_rate,
    wsr_for_A,
)
from mcm.no_utils import (
    InfeasibleOptimization,
    d_c_m_t_X_c_m_t,
    fractions_from_schedule,
)
from mcm.timesharing import (
    F_t_R_approx,
    time_sharing_cvx,
    F_t_R_approx_conj,
    timesharing_network,
)
from mcm.regions import Q_vector, R_m_t_approx
from mcm.network import Network
from .utils import gen_test_network
from mcm.my_typing import A_m_t, Util_cvx
from typing import Tuple


@pytest.mark.parametrize("util", (proportional_fair, weighted_sum_rate(np.ones(10))))
def test_U_Q(util: Util_cvx) -> None:

    Q = Q_vector(q_min=np.zeros(10), q_max=np.ones(10))
    la = np.random.random(10)
    val_conj, q = U_Q_conj(util, la, Q)
    val = U_Q(util, q, Q)
    assert la @ q == pytest.approx(val - val_conj)


def test_timesharing_wsr() -> None:
    A = np.array([[4, 1], [1, 2]])
    q_min = np.array([0, 0])
    q_max = np.array([10, 10])
    Q = Q_vector(q_min=q_min, q_max=q_max)
    n_users, n = A.shape
    R = R_m_t_approx(list(range(0, n_users)), A)

    weights = np.array([1, 0])
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([4, 1], 1e-3)

    weights = np.array([0, 1])
    _, rates = time_sharing_cvx(weighted_sum_rate(weights), R, Q)
    assert rates.tolist() == pytest.approx([1, 2], 1e-3)

    weights = np.array([1, 1])
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


def test_timesharing_fair(A: Matrix) -> None:

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

    _, rates_phy = wsr_for_A(lambda_opt, A)
    dual_value_phy = lambda_opt @ rates
    assert lambda_opt @ rates_phy == pytest.approx(dual_value_phy)
    assert value == pytest.approx(dual_value_app + dual_value_phy)

    # now by iterative optimization
    wsr = I_C(A)
    opt_value, opt_q, _, _ = optimize_app_phy(proportional_fair, Q, wsr)
    assert value == pytest.approx(opt_value, rel=1e-3, abs=1e-1)
    assert rates == pytest.approx(opt_q, rel=1e-3, abs=1e-1)


def test_V() -> None:
    network, Q, d_c_m_t = network_and_random_duals()
    # todo verifiy utilities
    val_conj, c_m_t_s = V_conj(network, proportional_fair, d_c_m_t, Q)
    val, q_m_t, f, d_c_m_t_2 = V_new(network, proportional_fair, c_m_t_s, Q)
    mm = d_c_m_t_X_c_m_t(d_c_m_t, c_m_t_s)
    assert mm == pytest.approx(val - val_conj, 1e-3)
    val_conj_2, c_m_t_s_2 = V_conj(network, proportional_fair, d_c_m_t_2, Q)
    assert val_conj == pytest.approx(val_conj_2)


def test_F_t_R_approx():
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

        # schedules_2 = {mode: R.alphas.value for mode, R in R_m.items()}
        c_m_2 = {mode: R.c.value for mode, R in R_m.items()}
        # d_sum_f_2 = {mode: R.sum_alpha.dual_value for mode, R in R_m.items()}
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


def network_and_random_duals() -> Tuple[Network, Q_vector, A_m_t]:
    As, network = gen_test_network(20, np.random.random)
    q_min = np.array([0.05] * 30)
    q_max = np.array([20.0] * 30)
    Q = Q_vector(q_min=q_min, q_max=q_max)
    d_c_m_t = {
        m: {t.id: np.random.random(len(t.users)) for t in ts_in_m}
        for m, ts_in_m in network.t_m.items()
    }

    return network, Q, d_c_m_t


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
    c_m_t = network.c_m_t
    (w_min, w_max) = Q.dual_values()
    assert 1 / rates + w_min - w_max - d_rates == pytest.approx(
        np.zeros(len(rates)), rel=1e-2, abs=1e-1
    )

    # r, la is primal dual solution - V(c_m_t) - I(c_m_t) == V*(la_c_m_t) + I*(la_c_m_t)
    f = fractions_from_schedule(network.alphas_m_t)
    c_m_t = {m: {t: 1 / f[m] * c for t, c in c_m.items()} for m, c_m in c_m_t.items()}
    d_c_m_t = {m: {t: d_c for t, d_c in d_c_m.items()} for m, d_c_m in d_c_m_t.items()}
    xi = {
        t_id: {m: d_c_m_t[m][t_id] @ c_m_t[m][t_id] for m in t.modes}
        for t_id, t in network.transmitters.items()
    }
    # V(c_m_t)
    V_c_m_t, V_q_m_t, V_f, V_d_c_m_t = V_new(network, proportional_fair, c_m_t, Q)
    # I(c_m_t)
    # we are using the approximation because we know it is complete
    I_c_m_t = network.I_C_m_t_approx(c_m_t)
    assert V_c_m_t - I_c_m_t == pytest.approx(value, 1e-3)

    # V*(la_c_m_t)
    V_conj_c_m_t, _ = V_conj(network, proportional_fair, d_c_m_t, Q)
    # I*(la_c_m_t)
    I_conj_c_m_t = 0
    for t_id, t in network.transmitters.items():
        for m in t.modes:
            v, r = t.wsr(d_c_m_t[m][t_id], m)
            v == pytest.approx(d_c_m_t[m][t_id] @ c_m_t[m][t_id], 1e-3)
            I_conj_c_m_t += v
    assert V_conj_c_m_t + I_conj_c_m_t == pytest.approx(value, 1e-3)

    # f, x_i is primal dual solution - sum_t F_t(f, R_approx) - I_F(f) == sum_t F_t*(xi) + I_F*(xi)
    # sum_t F_t(f, R_approx)
    F, r, alphas, d_f, F_t_s = network.F_t_R_appprox(f, proportional_fair, Q)
    # I_f
    assert sum(f.values()) == pytest.approx(
        1, 1e-3
    ), f"fracrions do not sum to 1, found {f.values()} with sum {sum(f.values())}"
    assert value == pytest.approx(F, 1e-3)

    # sum_t F*(xi)
    val_d = 0
    for t_id, t in network.transmitters.items():
        R_m = {m: R.approx for m, R in t.R_m_t_s.items()}
        (value_d, rates_d, f_d) = F_t_R_approx_conj(
            proportional_fair, xi[t_id], t.users, R_m, Q[t.users]
        )
        val_d += value_d
    # I_F*(xi)
    xi_sum = {m: 0 for m in network.modes}
    for t, d_m in xi.items():
        for m, d in d_m.items():
            xi_sum[m] += d
    assert value == pytest.approx(val_d + max(xi_sum.values()), 1e-3)
