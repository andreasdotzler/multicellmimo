import numpy as np
import cvxpy as cp
import logging

from typing import Tuple, List

from mcm.no_utils import InfeasibleOptimization, solve_problem
from mcm.regions import Q_vector, R_m_t_approx
from mcm.my_typing import Fractions, R_m, Util
from mcm.network import Network

LOGGER = logging.getLogger(__name__)


def check_feasible(f: Fractions, R_m: R_m, Q: Q_vector) -> None:
    if all(f[m] < 1e-3 for m in R_m):
        LOGGER.warning("no resources allocated")
        if Q.q_min and any(Q.q_min >= 0):
            raise InfeasibleOptimization("no resources allocated")
        else:
            raise NotImplementedError("no resources allocated - needs manual handling")


def time_sharing_cvx(cost_function: Util, R: R_m_t_approx, Q: Q_vector) -> Tuple[np.ndarray, np.ndarray]:

    r = cp.Variable(len(Q), pos=True)
    cons = R.cons_in_approx(r) + Q.constraints(r)
    prob = solve_problem(cp.Maximize(cost_function(r)), cons)
    return (prob.value, r.value)


def F_t_R_approx(cost_function: Util, f: Fractions, users: List[int], R_m: dict[str, R_m_t_approx], Q: Q_vector) -> Tuple[float, dict[int, float]]:
    check_feasible(f, R_m, Q)
    c_m = {}
    cons = []
    for mode, R in R_m.items():
        cons += R.cons_in_approx()
        c_m[mode] = R.c

    q_sum = cp.sum([f[mode] * c for mode, c in c_m.items() if f[mode] >= 1e-3], axis=1)
    cons += Q.constraints(q_sum)
    prob = solve_problem(cp.Maximize(cost_function(q_sum)), cons)

    return (prob.value, {user: r for user, r in zip(users, q_sum.value)})


def F_t_R_approx_conj(
    cost_function: Util, la: dict[str, np.ndarray], users: List[int], R_m: dict[str, R_m_t_approx], Q: Q_vector
) -> Tuple[np.ndarray, dict[int, np.ndarray], Fractions]:

    f = {m: cp.Variable(1, nonneg=True) for m in la}
    c_m = {}
    cons = []
    for mode, R in R_m.items():
        cons += R.cons_in_approx(sum_alphas=f[mode])
        c_m[mode] = R.c

    q_sum = cp.sum([c for mode, c in c_m.items()], axis=1)

    q = cp.Variable(len(users))
    cons.append(q == q_sum)
    cost = cost_function(q) - cp.sum([la[mode] * f[mode] for mode in f])

    cons += Q.constraints(q_sum)
    prob = solve_problem(cp.Maximize(cost), cons)

    return (
        prob.value,
        {user: r for user, r in zip(users, q_sum.value)},
        {m: f_m.value for m, f_m in f.items()},
    )


def timesharing_network(cost_function: Util, network: Network, Q: Q_vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    f = {m: cp.Variable(1, nonneg=True) for m in network.modes}
    n_users = len(network.users)
    r = cp.Variable(n_users, nonneg=True)
    transmitters = network.transmitters.values()
    c_t_m = {t: {} for t in transmitters}

    cons = []
    for t in transmitters:
        for mode, R in t.R_m_t_s.items():
            R_a = R.approx
            cons += R_a.cons_in_approx(sum_alphas=f[mode])
            c_t_m[t][mode] = R.c

    # TODO cost function per transmitter should be more elegant
    r_constraints = {}
    for t in transmitters:
        for user_index, user in enumerate(t.users):
            r_constraints[user] = r[user] == cp.sum(
                [c_m[user_index] for c_m in c_t_m[t].values()]
            )

    cons += list(r_constraints.values())
    cons.append(cp.sum([f_m for f_m in f.values()]) == 1)
    cons += Q.constraints(r)

    prob = solve_problem(cp.Maximize(cost_function(r)), cons)

    return (
        prob.value,
        r.value,
        np.array([r_constraints[user].dual_value for user in network.users]),
    )
