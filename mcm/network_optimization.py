import logging
from typing import Optional, Tuple


import cvxpy as cp
import numpy as np

from mcm.timesharing import time_sharing_cvx
from mcm.regions import Q_vector, R_m_t_approx
from mcm.no_utils import solve_problem

LOGGER = logging.getLogger(__name__)


def wsr_for_A(weights, A):
    max_i = np.argmax(weights @ A)
    rates = A[:, max_i]
    return weights @ rates, rates


def I_C(A):
    return lambda weights: wsr_for_A(weights, A)


def I_C_Q(A, Q: Q_vector):
    R = R_m_t_approx(A=A)
    return lambda weights: time_sharing_cvx(weighted_sum_rate(weights), R, Q)


def weighted_sum_rate(weights):
    def weighted_sum_rate(r):
        return weights @ r
    return weighted_sum_rate


def proportional_fair(r):
    # return cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r))
    return cp.sum(cp.log(r))


def app_layer(weights):
    def app_layer(r):
        return cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r)) - weights @ r

    return app_layer


def U_Q(util, q, Q):
    # U(q)
    if q not in Q:
        return -np.Inf
    else:
        return util(cp.Variable(len(q), value=q)).value


def U_Q_conj(util, weights, Q):
    # max_q U(q) - la@q : q in Q
    q = cp.Variable(len(Q))
    cost_dual = util(q) - weights @ q
    constraints_dual = Q.constraints(q)
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve()
    assert "optimal" in prob_dual.status
    return prob_dual.value, q.value


def dual_problem_app_f(util, weights_per_mode, f, Q):
    #use K_conj(proportional_fair, Q[t.users], d_c_m, f = fractions)
    q = cp.Variable(len(Q))
    c_s = {m: cp.Variable(len(w), nonneg=True) for m, w in weights_per_mode.items()}

    cost_dual = util(q)
    for m, weights in weights_per_mode.items():
        assert min(weights) >= 0
        cost_dual -= weights @ c_s[m]
    const_q = 0
    for m, c in c_s.items():
        const_q += f[m] * c
    constraints_dual = [q == const_q]
    # TODO convert to Q.constraints
    if Q.q_max is not None:
        constraints_dual.append(q <= Q.q_max)
    if Q.q_min is not None:
        constraints_dual.append(q >= Q.q_min)
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve()
    if "optimal" not in prob_dual.status:
        a = 1
    return prob_dual.value, q.value, {m: c.value for m, c in c_s.items()}

def K_conj(util, Q, la_m, f = None) -> Tuple[float, dict[str, np.ndarray]]:
    
    q_m = {m: cp.Variable(len(la), nonneg=True) for m, la in la_m.items()}
    if f:
        q_sum = cp.sum([q * f[m] for m, q in q_m.items()], axis=1)
    else:
        q_sum = cp.sum(list(q_m.values()), axis=1)

    cost = util(q_sum) - cp.sum([la @ q_m[m] for m, la in la_m.items()], axis=1)
    prob = solve_problem(cp.Maximize(cost), Q.constraints(q_sum))

    return prob.value, {m: q.value for m, q in q_m.items()}

def V(network, util, c_m_t, Q):

    c_m = {mode: np.zeros(len(network.users)) for mode in network.modes}
    f = {mode: cp.Variable(1, nonneg=True) for mode in network.modes}
    for mode, c_t in c_m_t.items():
        for t, c in c_t.items():
            c_m[mode][network.transmitters[t].users] += c
    q_sum = cp.sum([f[mode] * c for mode, c in c_m.items()], axis=1)

    constraints = [cp.sum(list(f.values())) == 1] + Q.constraints(q_sum)
    prob = cp.Problem(cp.Maximize(util(q_sum)), constraints)
    prob.solve()
    assert "optimal" in prob.status, f"Optimization failed: {prob.status}"
    return (
        prob.value,
        sum([f[mode].value * c for mode, c in c_m.items()]),
        {mode: f_m.value for mode, f_m in f.items()},
    )


def V_new(network, util, c_m_t, Q):
    
    f = {mode: cp.Variable(1, nonneg=True) for mode in network.modes}
    q_m_t = {m: {} for m in network.modes}
    con_m_t = {m: {} for m in network.modes}
    constraints = []
    #c_m_t = {m: {} for m in network.modes}
    q_t_sum = {}
    for t_id, t in network.transmitters.items():
        for m in t.modes:
            q_m_t[m][t_id] = cp.Variable(len(t.users), nonneg=True)
            con_m_t[m][t_id] = q_m_t[m][t_id] == f[m] * c_m_t[m][t_id]
            constraints.append(con_m_t[m][t_id])
        q_t_sum[t_id] = cp.sum([q_m_t[mode][t_id] for mode in t.modes], axis=1)

    constraints.append(cp.sum(list(f.values())) == 1)
    for t_id, t in network.transmitters.items():
        constraints+= Q[t.users].constraints(q_t_sum[t_id])
    util = cp.sum([util(q_t) for q_t in q_t_sum.values()])
    prob = cp.Problem(cp.Maximize(util), constraints)
    prob.solve()
    q = np.zeros(len(network.users))
    for t_id, t in network.transmitters.items():
        q[t.users] += q_t_sum[t_id].value

    assert "optimal" in prob.status, f"Optimization failed: {prob.status}"
    return (
        prob.value,
        q,
        {mode: f_m.value for mode, f_m in f.items()},
        {m: {t_id: f[m].value * con.dual_value for t_id, con in con_t.items()} for m, con_t in con_m_t.items()}
    )





def V_conj(network, util, la_m_t, Q) -> Tuple[float, dict[str, dict[str, np.ndarray]]]:

    #q = np.zeros(len(Q))
    q_m_t = {m: {} for m in network.modes}
    v_opt = 0
    q_t = {}
    for t_id, t in network.transmitters.items():
        val, q_t = K_conj(util, Q[t.users], {m: la_m_t[m][t_id] for m in t.modes})
        for m, q in q_t.items():
            q_m_t[m][t_id] = q
        v_opt += val
    s = {}
    for m, la_t in la_m_t.items():
        s[m] = 0
        for t, la in la_t.items():
            s[m] += la @ q_m_t[m][t]
    c_m_t = {}
    d_sum = sum(s.values())
    for m, q_t in q_m_t.items():
        c_m_t[m] = {}
        for t, q in q_t.items():
            c_m_t[m][t] = q * d_sum / s[m]

    return v_opt, q_m_t, c_m_t


def optimize_app_phy(util, Q: Q_vector, wsr_phy):

    assert util == proportional_fair
    n_users = len(Q)
    assert len(Q) == n_users
    A = np.minimum(Q.q_max, Q.q_min).reshape((n_users, 1))

    best_dual_value = np.inf
    for n in range(1, 1000):
        # create and solve the approximated problem
        n_users, n = A.shape
        R = R_m_t_approx(list(range(0, n_users)), A)
        approx_value, q = time_sharing_cvx(util, R, Q)
        alpha = R.alphas
        (la, mu) = R.dual_values()
        (w_min, w_max) = Q.dual_values()
        # solve the dual problem to provide bound and update
        v_app, _ = U_Q_conj(util, la, Q)
        v_phy, c = wsr_phy(la)
        A = np.c_[A, c]

        # breaking criterium
        # U*(la) + R*(la) >= min U*(la) + R*(la) = max U(q) + R(q) >= U(q) + R_approx(q)
        # U*(la) = v_app
        # R*(la) = v_phy
        # Q(q) = approx_value
        # R_approx(q) = 0

        dual_value = v_app + v_phy
        best_dual_value = min(dual_value, best_dual_value)
        gap = abs(best_dual_value - approx_value) / abs(best_dual_value)
        LOGGER.info(
            f"Iteration {n} - Primal Approx {approx_value} - Dual Approx {dual_value} - Gap: {gap} "
        )
        if gap < 0.001:
            break
    return approx_value, q, alpha, [la, w_min, w_max, mu]
