import numpy as np
import cvxpy as cp
import logging


from mcm.no_utils import InfeasibleOptimization, solve_problem
from mcm.regions import Q_vector, R_m_t_approx
from mcm.my_typing import Fractions, R_m

LOGGER = logging.getLogger(__name__)


def check_feasible(f: Fractions, R_m: R_m, Q: Q_vector) -> None:
    if all(f[m] < 1e-3 for m in R_m):
        LOGGER.warning("no resources allocated")
        if any(Q.q_min >= 0):
            raise InfeasibleOptimization("no resources allocated")
        else:
            raise NotImplementedError("no resources allocated - needs manual handling")


def time_sharing_cvx(cost_function, R: R_m_t_approx, Q: Q_vector):

    r = cp.Variable(len(Q), pos=True)
    cons = R.cons_in_approx(r) + Q.constraints(r)
    prob = solve_problem(cp.Maximize(cost_function(r)), cons)
    return (prob.value, r.value)


def F_t_R_approx(cost_function, f, users, R_m, Q: Q_vector):
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
    cost_function, la, users, R_m: dict[str, R_m_t_approx], Q: Q_vector
):

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


def timesharing_network(cost_function, network, Q: Q_vector):

    f = {m: cp.Variable(1, nonneg=True) for m in network.modes}
    n_users = len(network.users)
    r = cp.Variable(n_users, nonneg=True)
    transmitters = network.transmitters.values()
    c_t_m = {t: {} for t in transmitters}

    cons = []
    for t in transmitters:
        for mode, R in t.R_m_t_s.items():
            R = R.approx
            cons += R.cons_in_approx(sum_alphas=f[mode])
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


def timesharing_network_dual(cost_function, la_m_t_s, network, Q: Q_vector):

    c_m_t = {m: {} for m in network.modes}

    w_sum = 0
    for t_id, transmitter in network.transmitters.items():
        for mode, rates in transmitter.As_per_mode.items():
            users = transmitter.users_per_mode[mode]
            c_m_t[mode][t_id] = cp.Variable(len(users), nonneg=True)
            w_sum += la_m_t_s[mode][t_id] @ c_m_t[mode][t_id]

    r_constraints = {}

    for t_id, transmitter in network.transmitters.items():
        for mode in transmitter.modes:
            users = transmitter.users_per_mode[mode]
            for user_index, user in enumerate(users):
                constraint = c_m_t[mode][t_id][user_index]
                if user in r_constraints:
                    r_constraints[user] += constraint
                else:
                    r_constraints[user] = constraint

    n_users = len(r_constraints)
    r = cp.Variable(n_users, nonneg=True)
    user_rate_constraints = [r_k == r_constraints[user] for user, r_k in enumerate(r)]
    cost = cost_function(r) - w_sum


    constraints = user_rate_constraints + Q.constraints(r)

    prob = solve_problem(cp.Maximize(cost), constraints)

    return (
        prob.value,
        r.value,
        {m: {t: c.value for t, c in c_t.items()} for m, c_t in c_m_t.items()},
        [
            np.array([c.dual_value for c in user_rate_constraints]),
        ],
    )
