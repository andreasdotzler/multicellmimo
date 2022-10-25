import logging

import cvxpy as cp
import numpy as np

from mcm.network_optimization import (
    I_C,
    I_C_Q,
    U_Q_conj,
    Q_vector,
    optimize_app_phy,
    proportional_fair,
)

LOGGER = logging.getLogger(__name__)


def optimize_primal_sub(A, q_min, q_max, target=None):
    # direct optimization of the problem in the primal domain via subgradient
    # unconstrained_primal_sub
    n_users = len(q_min)
    assert len(q_max) == n_users

    # add the origin to the points
    A = np.c_[np.zeros(n_users), A]
    _, n_schedules = A.shape

    rates = q_min

    max_util = sum(np.log(rates))
    for n in range(1000):

        step_size = 1
        subgradient = cp.Variable(n_users)
        d_U = cp.Variable(n_users)
        d_Q = cp.Variable(n_users)
        d_C = cp.Variable(n_users)

        scale = cp.Variable(1, nonneg=True)
        c_d_U = [d_U == scale * 1 / rates]

        c_d_Q = []
        for r_i, q_min_i, q_max_i, d_Q_i in zip(rates, q_min, q_max, d_Q):
            if abs(r_i - q_min_i) <= 10**-6:
                c_d_Q.append(d_Q_i >= 0)
            elif abs(r_i - q_max_i) <= 10**-6:
                c_d_Q.append(d_Q_i <= 0)
            else:
                c_d_Q.append(d_Q_i == 0)
        # c_d_Q = [d_Q@(q_min - rates) <=0, d_Q@(q_max - rates) <=0]
        c_d_C = []
        for a in range(n_schedules):
            c_d_C.append(d_C @ (A[:, a] - rates) <= 0)
        sub_sum = [subgradient == d_U + d_Q - d_C]

        alpha = cp.Variable(n_schedules, nonneg=True)
        feasible_C = [rates + subgradient == A @ alpha, cp.sum(alpha) <= 1]
        feasible_Q = [rates + subgradient >= q_min, rates + subgradient <= q_max]

        constraints = c_d_U + c_d_Q + c_d_C + sub_sum + feasible_Q + feasible_C
        weights = np.random.normal(size=len(rates))
        prob1 = cp.Problem(cp.Maximize(weights @ subgradient), constraints)
        # prob1 = cp.Problem(cp.Minimize(cp.sum_squares(subgradient)), constraints)
        prob1.solve()
        assert "optimal" in prob1.status
        step_size = cp.Variable(1)
        prob = cp.Problem(
            cp.Maximize(cp.sum(cp.log(rates + step_size * subgradient.value))),
            [step_size <= 1, step_size >= 0],
        )
        prob.solve()
        rates = rates + step_size.value * subgradient.value
        util = sum(np.log(rates))
        max_util = max(max_util, util)
        gap = (target - util) / abs(target)
        LOGGER.info(f"Iteration {n} - Util {util} - Gap - {gap}")

        if gap < 0.001:
            break
        # if subgradient.value @ subgradient.value < 0.00000000000001:
        # break

    return max_util, rates, None


def optimize_primal_subgradient_projected(A, q_min, q_max, target=None):

    rates = q_min
    n_users, n_schedules = A.shape
    step_size = 1
    max_util = sum(np.log(rates))
    for n in range(1000):
        LOGGER.info(
            f"Projected Subgradient: Iteration {n} - Best Primal Value {max_util}"
        )
        subgradient = 1 / rates

        r_l_1 = cp.Variable(n_users)
        alpha = cp.Variable(n_schedules, nonneg=True)

        constraints = [
            r_l_1 == A @ alpha,
            cp.sum(alpha) <= 1,
            r_l_1 >= q_min,
            r_l_1 <= q_max,
        ]
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(r_l_1 - (rates + step_size * subgradient))),
            constraints,
        )
        prob.solve()
        rates = r_l_1.value
        new_util = sum(np.log(rates))
        if target is None and abs(max_util - new_util) < 1e-6:
            break
        max_util = max(max_util, new_util)
        if target is not None:
            gap = (target - max_util) / abs(target)
            if gap < 0.001:
                break
    return max_util, rates, None


def optimize_primal_subgradient_rosen(A, q_min, q_max, target=None):

    rates = q_min
    n_users, n_schedules = A.shape
    step_size = 1
    max_util = sum(np.log(rates))
    for n in range(1000):
        LOGGER.info(
            f"Projected Subgradient: Iteration {n} - Best Primal Value {max_util}"
        )
        subgradient = 1 / rates

        p_s = cp.Variable(n_users)
        alpha = cp.Variable(n_schedules, nonneg=True)
        constraints = [
            rates + p_s == A @ alpha,
            cp.sum(alpha) <= 1,
            rates + p_s >= q_min,
            rates + p_s <= q_max,
        ]
        # prob = cp.Problem(cp.Minimize(cp.sum_squares(r_l_1 - (rates + step_size * subgradient) )), constraints)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(p_s - subgradient)), constraints)
        prob.solve()
        # rates = r_l_1.value
        rates = rates + step_size * p_s.value
        new_util = sum(np.log(rates))
        if target is None and abs(max_util - new_util) < 1e-6:
            break
        max_util = max(max_util, new_util)
        if target is not None:
            gap = (target - max_util) / abs(target)
            if gap < 0.0001:
                break
    return max_util, rates, None


def optimize_primal_column(A, q_min, q_max, target=None):
    n_users = A.shape[0]
    wsr_C_Q = I_C_Q(A, q_min, q_max)
    opt, r, alpha, _ = optimize_app_phy(
        proportional_fair,
        q_min=np.array([0.001] * n_users),
        q_max=np.array([10] * n_users),
        wsr_phy=wsr_C_Q,
    )
    return opt, r, alpha


def optimize_dual_decomp_subgradient(A, q_min, q_max, target=None):
    # Application Layer and Physical Layer Decomposition
    # Dual Sub-Gradient
    util = proportional_fair
    wsr_phy = I_C(A)
    weights = np.ones(len(q_min))
    for i in range(1000):
        v_phy, c = wsr_phy(weights)
        v_app, q = U_Q_conj(util, weights, Q_vector(q_min, q_max))
        weights -= 1 / (i + 1) * (c - q)
        if i == 0:
            r = c
        else:
            r = i / (i + 1) * r + 1 / (i + 1) * c
        primal_value = sum(np.log(r))
        dual_value = v_app + v_phy
        gap = dual_value - primal_value

        LOGGER.info(
            f"Dual Subgradient: Iteration {i} - Dual Value {dual_value}- Primal Value {primal_value} - Gap {gap}"
        )
        if gap < 1e-2:
            break
    return primal_value, r, None


def ploy_cutting_plane(U_i, q_i, c_i):
    n_user = len(q_i[0])
    # TODO check, this should be the primal version
    # alpha = cp.Variable((i+1,1), nonneg=True)
    # c_sum = cp.sum(alpha) == 1
    # c_schedule = z_i@alpha == np.zeros((len(q_min),1))
    # prob = cp.Problem(cp.Maximize(alpha.T@U_i), [c_sum, c_schedule])
    mu = cp.Variable(1)
    la = cp.Variable(n_user)
    cons = [mu >= u + la @ (c - q) for u, c, q in zip(U_i, c_i, q_i)]
    prob = cp.Problem(cp.Minimize(mu), cons)
    prob.solve()
    return prob.value, la.value
    # alpha = [c.dual_value[0] for c in cons]
    # sum([a*u for a, u in zip(alpha, U_i)]) == prob.value
    # sum([a*z for a, z in zip(alpha, z_i)]) == 0


def poly_multicut(U_i, q_i, c_i):
    mu = cp.Variable(2)
    la = cp.Variable(len(q_i[0]))
    cons_U_Q = [mu[0] >= u - la @ q for u, q in zip(U_i, q_i)]
    cons_I_C = [mu[1] >= la @ c for c in c_i]
    prob = cp.Problem(cp.Minimize(sum(mu)), cons_U_Q + cons_I_C)
    prob.solve()
    return prob.value, la.value


def poly_multicut_peruser(U_i, q_i, c_i):
    n_users = len(q_i[0])
    mu = cp.Variable(n_users + 1)
    la = cp.Variable(n_users)
    cons_U_Q = []
    for j in range(n_users):
        cons_U_Q += [mu[j] >= np.log(q[j]) - la[j] * q[j] for u, q in zip(U_i, q_i)]
    cons_I_C = [mu[j + 1] >= la @ c for c in c_i]
    prob = cp.Problem(cp.Minimize(sum(mu)), cons_U_Q + cons_I_C)
    prob.solve()
    return prob.value, la.value


def optimize_dual_cuttingplane(A, q_min, q_max, target=None):
    update = ploy_cutting_plane
    wsr_phy = I_C(A)
    util = proportional_fair
    return dual_ployhedaral_approx(util, q_min, q_max, wsr_phy, update)


def optimize_dual_multicut(A, q_min, q_max, target=None):
    update = poly_multicut
    wsr_phy = I_C(A)
    util = proportional_fair
    return dual_ployhedaral_approx(util, q_min, q_max, wsr_phy, update)


def optimize_dual_multicut_peruser(A, q_min, q_max, target=None):
    update = poly_multicut_peruser
    wsr_phy = I_C(A)
    util = proportional_fair
    return dual_ployhedaral_approx(util, q_min, q_max, wsr_phy, update)


def dual_ployhedaral_approx(util, q_min, q_max, wsr_phy, update):
    # we evaluate the dual function for lambda = [-Inf, ..., -Inf]
    # we know the result is q_min
    v_q_min = util(cp.Variable(len(q_min), value=q_min)).value
    U_i = [v_q_min]
    c_i = [q_min]
    q_i = [q_min]
    best_dual = np.Inf

    for i in range(1000):

        primal_value, weights = update(U_i, q_i, c_i)
        v_phy, c = wsr_phy(weights)
        v_app, q = U_Q_conj(util, weights, Q_vector(q_min, q_max))
        c_i.append(c)
        q_i.append(q)
        # v_app = U(q) - weights@q -> U(q) = v_app + weights@q
        U_i.append(v_app + weights @ q)

        best_dual = min(v_app + v_phy, best_dual)
        gap = (best_dual - primal_value) / abs(best_dual)

        LOGGER.info(
            f"Dual Approx: Iteration {i} - Dual Value {best_dual}- Primal Value {primal_value} - Gap {gap}"
        )
        if gap < 1e-3:
            break
    return primal_value, q, None


def optimize_app_phy_rateregionapprox(A, q_min, q_max, target=None):
    wsr_phy = I_C(A)
    util = proportional_fair
    primal_value, q, alpha, _ = optimize_app_phy(util, q_min, q_max, wsr_phy)
    return primal_value, q, alpha
