import logging

import cvxpy as cp
import numpy as np
from typing import Optional, cast, Callable, List
from mcm.network_optimization import (
    I_C,
    I_C_Q,
    U_Q_conj,
    optimize_app_phy,
    proportional_fair,
)
from mcm.regions import Q_vector_bounded, Q_vector
from mcm.my_typing import Algorithm_Result, WSR
from mcm.no_utils import solve_problem

LOGGER = logging.getLogger(__name__)


def optimize_primal_sub(A: np.ndarray, Q: Q_vector_bounded, target: float) -> Algorithm_Result:
    # direct optimization of the problem in the primal domain via subgradient
    # unconstrained_primal_sub
    # todo, only implemented for proportional fair
    # todo make targe optional
    n_users = len(Q)

    # add the origin to the points
    A = np.c_[np.zeros(n_users), A]
    _, n_schedules = A.shape

    # This alogrithm assumes we have a q_min and q_max
    rates = Q.q_min
    max_util = sum(np.log(rates))
    for n in range(1000):

        subgradient = cp.Variable(n_users)
        d_U = cp.Variable(n_users)
        d_Q = cp.Variable(n_users)
        d_C = cp.Variable(n_users)

        scale = cp.Variable(1, nonneg=True)
        c_d_U = [d_U == scale * 1 / rates]

        c_d_Q = []
        for r_i, q_min_i, q_max_i, d_Q_i in zip(rates, Q.q_min, Q.q_max, d_Q): # type: ignore
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
        feasible_Q = [rates + subgradient >= Q.q_min, rates + subgradient <= Q.q_max]

        constraints = c_d_U + c_d_Q + c_d_C + sub_sum + feasible_Q + feasible_C
        weights = np.random.normal(size=len(rates))
        # prob1 = cp.Problem(cp.Maximize(weights @ subgradient), constraints)
        # prob1 = cp.Problem(cp.Minimize(cp.sum_squares(subgradient)), constraints)
        prob1 = solve_problem(cp.Maximize(weights @ subgradient), constraints)
        assert "optimal" in prob1.status
        step_size = cp.Variable(1)
        prob = solve_problem(
            cp.Maximize(cp.sum(cp.log(rates + step_size * subgradient.value))),
            [step_size <= 1, step_size >= 0],
        )

        rates = rates + step_size.value * subgradient.value
        util = sum(np.log(rates))
        max_util = max(max_util, util)
        # use targe only if not none
        if target is not None:
            gap = (target - util) / abs(target)
        else:
            gap = (max_util - util) / abs(max_util)
        LOGGER.info(f"Iteration {n} - Util {util} - Gap - {gap}")

        if gap < 0.001:
            break


    return max_util, rates


def optimize_primal_subgradient_projected(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:

    rates = Q.q_min
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
            r_l_1 >= Q.q_min,
            r_l_1 <= Q.q_max,
        ]
        solve_problem(cp.Minimize(cast(cp.Expression, cp.sum_squares(r_l_1 - (rates + step_size * subgradient)))), constraints) #type: ignore
        rates = r_l_1.value
        new_util = sum(np.log(rates))
        if target is None and abs(max_util - new_util) < 1e-6:
            break
        max_util = max(max_util, new_util)
        if target is not None:
            gap = (target - max_util) / abs(target)
            if gap < 0.001:
                break
    return max_util, rates


def optimize_primal_subgradient_rosen(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:

    rates = Q.q_min
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
            rates + p_s >= Q.q_min,
            rates + p_s <= Q.q_max,
        ]
        # prob = cp.Problem(cp.Minimize(cp.sum_squares(r_l_1 - (rates + step_size * subgradient) )), constraints)
        prob = solve_problem(cp.Minimize(cp.sum_squares(p_s - subgradient)), constraints) # type: ignore

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
    return max_util, rates


def optimize_primal_column(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:
    n_users = A.shape[0]
    wsr_C_Q = I_C_Q(A, Q)
    # todo check why we need this bound
    # Q_up = Q_vector(q_min=np.array([0.001] * n_users), q_max=np.array([10] * n_users))
    opt, r, alpha, _ = optimize_app_phy(
        proportional_fair,
        Q,
        wsr_phy=wsr_C_Q,
    )
    return opt, r


def optimize_dual_decomp_subgradient(A: np.ndarray, Q: Q_vector, target: Optional[float] = None) -> Algorithm_Result:
    # Application Layer and Physical Layer Decomposition
    # Dual Sub-Gradient
    util = proportional_fair
    wsr_phy = I_C(A)
    weights = np.ones(len(Q))
    for i in range(1000):
        v_phy, c = wsr_phy(weights)
        v_app, q = U_Q_conj(util, weights, Q)
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
    return primal_value, r


def ploy_cutting_plane(U_i: List[np.ndarray], q_i: List[np.ndarray], c_i: List[np.ndarray]) -> Algorithm_Result:
    n_user = len(q_i[0])
    # TODO check, this should be the primal version
    # alpha = cp.Variable((i+1,1), nonneg=True)
    # c_sum = cp.sum(alpha) == 1
    # c_schedule = z_i@alpha == np.zeros((len(q_min),1))
    # prob = cp.Problem(cp.Maximize(alpha.T@U_i), [c_sum, c_schedule])
    mu = cp.Variable(1)
    la = cp.Variable(n_user)
    cons = [mu >= u + la @ (c - q) for u, c, q in zip(U_i, c_i, q_i)]
    prob = solve_problem(cp.Minimize(mu), cons)
    return prob.value, la.value
    # alpha = [c.dual_value[0] for c in cons]
    # sum([a*u for a, u in zip(alpha, U_i)]) == prob.value
    # sum([a*z for a, z in zip(alpha, z_i)]) == 0


def poly_multicut(U_i: List[np.ndarray], q_i: List[np.ndarray], c_i: List[np.ndarray]) -> Algorithm_Result:
    mu = cp.Variable(2)
    la = cp.Variable(len(q_i[0]))
    cons_U_Q = [mu[0] >= u - la @ q for u, q in zip(U_i, q_i)]
    cons_I_C = [mu[1] >= la @ c for c in c_i]
    prob = solve_problem(cp.Minimize(cp.sum(mu)), cons_U_Q + cons_I_C)
    return prob.value, la.value


def poly_multicut_peruser(U_i: List[np.ndarray], q_i: List[np.ndarray], c_i: List[np.ndarray]) -> Algorithm_Result:
    n_users = len(q_i[0])
    mu = cp.Variable(n_users + 1)
    la = cp.Variable(n_users)
    cons_U_Q = []
    for j in range(n_users):
        cons_U_Q += [mu[j] >= np.log(q[j]) - la[j] * q[j] for u, q in zip(U_i, q_i)]
    cons_I_C = [mu[j + 1] >= la @ c for c in c_i]
    prob = solve_problem(cp.Minimize(cp.sum(mu)), cons_U_Q + cons_I_C)
    return prob.value, la.value


def optimize_dual_cuttingplane(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:
    update = ploy_cutting_plane
    wsr_phy = I_C(A)
    util = proportional_fair
    return dual_ployhedaral_approx(util, Q, wsr_phy, update)


def optimize_dual_multicut(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:
    update = poly_multicut
    wsr_phy = I_C(A)
    util = proportional_fair
    return dual_ployhedaral_approx(util, Q, wsr_phy, update)


def optimize_dual_multicut_peruser(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:
    update = poly_multicut_peruser
    wsr_phy = I_C(A)
    util = proportional_fair
    return dual_ployhedaral_approx(util, Q, wsr_phy, update)


def dual_ployhedaral_approx(
        util: Callable[[cp.Variable], cp.Expression],
        Q: Q_vector_bounded,
        wsr_phy: WSR,
        update: Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], Algorithm_Result]) -> Algorithm_Result:
    # we evaluate the dual function for lambda = [-Inf, ..., -Inf]
    # we know the result is q_min
    v_q_min = util(cp.Variable(len(Q), value=Q.q_min)).value
    U_i = [v_q_min]
    c_i = [Q.q_min]
    q_i = [Q.q_min]
    best_dual = np.Inf

    for i in range(1000):

        primal_value, weights = update(U_i, q_i, c_i)
        v_phy, c = wsr_phy(weights)
        v_app, q = U_Q_conj(util, weights, Q)
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
    return primal_value, q


def optimize_app_phy_rateregionapprox(A: np.ndarray, Q: Q_vector_bounded, target: Optional[float] = None) -> Algorithm_Result:
    wsr_phy = I_C(A)
    util = proportional_fair
    primal_value, q, _, _ = optimize_app_phy(util, Q, wsr_phy)
    return primal_value, q
