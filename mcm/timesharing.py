import numpy as np
import cvxpy as cp
import logging
from numpy.lib.arraysetops import unique

from mcm.no_utils import InfeasibleOptimization
from mcm.regions import Q_vector

LOGGER = logging.getLogger(__name__)


def time_sharing_cvx(cost_function, A, q_min=None, q_max=None):
    n_users, n = A.shape
    r = cp.Variable(n_users, pos=True)
    cost = cost_function(r)

    if q_min is None:
        q_min = np.zeros(r.size)
    if q_max is None:
        q_max = np.squeeze(np.asarray(np.amax(A, 1)))

    alpha = cp.Variable(n, nonneg=True)
    c1 = r == A @ alpha
    c2 = r >= q_min
    c3 = r <= q_max
    c5 = cp.sum(alpha) == 1
    constraints = [c1, c2, c3, c5]

    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()

    # check KKT
    if "infeasible" in prob.status:
        raise InfeasibleOptimization()
    assert prob.status == "optimal"
    return (
        prob.value,
        r.value,
        alpha.value,
        [c1.dual_value, c2.dual_value, c3.dual_value, c5.dual_value],
    )


def time_sharing(cost_function, A, q_min=None, q_max=None):
    value, rates, alpha, [lambda_opt, _, _, _] = time_sharing_cvx(
        cost_function, A, q_min, q_max
    )
    return value, rates, alpha, lambda_opt


def time_sharing_no_duals(costs_function, A, q_min=None, q_max=None):
    value, rates, _, _ = time_sharing_cvx(costs_function, A, q_min, q_max)
    return value, rates


def timesharing_fixed_fractions(cost_function, f, users, R_m, Q: Q_vector):
    if all(f[m] < 1e-3 for m in R_m):
        LOGGER.warning("no resources allocated")
        if any(Q.q_min >= 0):
            raise InfeasibleOptimization("no resources allocated")
        else:
            raise NotImplementedError("no resources allocated - needs manual handling")

    alphas = {}
    c_m_constraints = {}
    for mode, rates in R_m.items():
        alphas[mode] = cp.Variable(rates.shape[1], nonneg=True)

    c_m = {}
    for mode, As_m in R_m.items():
        c_m[mode] = cp.Variable(len(users), nonneg=True)
        c_m_constraints[mode] = c_m[mode] == As_m @ alphas[mode]

    # f = {m:0 if f<10**-3 else f for m,f in f.items()}
    q = cp.Variable(len(users))
    q_sum = cp.sum([f[mode] * c for mode, c in c_m.items() if f[mode] >= 1e-3], axis=1)
    # q_constraint = q == q_sum
    cost = cost_function(q_sum)

    c2 = Q.con_min(q_sum)
    c3 = Q.con_max(q_sum)
    c5 = {}
    for mode in alphas:
        c5[mode] = cp.sum(alphas[mode]) == 1

    c_m_constraints_list = []
    for mode, cons in c_m_constraints.items():
        c_m_constraints_list.append(cons)
    constraints = (
        [c2, c3] + list(c5.values()) + c_m_constraints_list  # + [q_constraint]
    )
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()
    if "optimal" not in prob.status:
        raise InfeasibleOptimization()
    q = sum([f[m] * c.value for m, c, in c_m.items()])
    return (
        prob.value,
        {user: r for user, r in zip(users, q)},
        {mode: alpha.value for mode, alpha in alphas.items()},
        {mode: c.value for mode, c in c_m.items()},
        [
            {mode: c.dual_value for mode, c in c5.items()},
            {mode: c.dual_value for mode, c in c_m_constraints.items()},
        ],
    )


def timesharing_fixed_fractions_dual(
    cost_function, la, users_per_mode, As, q_min=None, q_max=None
):

    users = []
    alphas = {}
    f = {}
    c_m_constraints = {}
    for mode, rates in As.items():
        alphas[mode] = cp.Variable(rates.shape[1], nonneg=True)
        f[mode] = cp.Variable(1, nonneg=True)
        users += users_per_mode[mode]

    users = unique(users)

    c_m = {}
    for mode, As_m in As.items():
        users = users_per_mode[mode]
        c_m[mode] = cp.Variable(len(users), nonneg=True)
        c_m_constraints[mode] = c_m[mode] == As_m @ alphas[mode]

    # f = {m:0 if f<10**-3 else f for m,f in f.items()}
    q = cp.Variable(len(users))
    q_constraint = q == cp.sum([c for mode, c in c_m.items()], axis=1)
    cost = cost_function(q) - cp.sum([la[mode] * f[mode] for mode in f])
    # cost = cp.sum(cp.log())

    # c2 = cp.sum([f[mode]*c for c in c_m.values()],axis=0 ) >= q_min
    # c3 = cp.sum([f[mode]*c for c in c_m.values()],axis=0 ) <= q_max
    c5 = {}
    for mode in alphas:
        c5[mode] = cp.sum(alphas[mode]) == f[mode]

    c_m_constraints_list = []
    for mode, cons in c_m_constraints.items():
        c_m_constraints_list.append(cons)
    constraints = (
        # [c2, c3] + list(c5.values()) + c_m_constraints_list
        list(c5.values())
        + c_m_constraints_list
        + [q_constraint]
    )
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()
    if "optimal" not in prob.status:
        raise InfeasibleOptimization()
    return (
        prob.value,
        {user: r for user, r in zip(users, q.value)},
        {mode: alpha.value for mode, alpha in alphas.items()},
        {mode: c.value for mode, c in c_m.items()},
        {mode: f_m.value for mode, f_m in f.items()},
        [
            {mode: c.dual_value for mode, c in c5.items()},
            {mode: c.dual_value for mode, c in c_m_constraints.items()},
            q_constraint.dual_value,
        ],
    )


def timesharing_network(cost_function, network, Q: Q_vector):

    alphas = {m: {} for m in network.modes}

    c_m_t = {m: {} for m in network.modes}
    c_m_t_constraints = {m: {} for m in network.modes}

    for t_id, transmitter in network.transmitters.items():
        for mode, rates in transmitter.As_per_mode.items():
            alphas[mode][t_id] = cp.Variable(rates.shape[1], nonneg=True)
            users = transmitter.users_per_mode[mode]
            c_m_t[mode][t_id] = cp.Variable(len(users), nonneg=True)
            c_m_t_constraints[mode][t_id] = (
                c_m_t[mode][t_id] == transmitter.As_per_mode[mode] @ alphas[mode][t_id]
            )

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
    cost = cost_function(r)

    c2 = Q.con_min(r)
    c3 = Q.con_max(r)

    f = cp.Variable(len(alphas), nonneg=True)
    c5 = []
    f_constraints = {}
    for (mode, alphas_per_mode), sum_m in zip(alphas.items(), f):
        f_constraints[mode] = {}
        for transmitter, alpha in alphas_per_mode.items():
            con = cp.sum(alpha) == sum_m
            f_constraints[mode][transmitter] = con
            c5.append(con)

    c6 = cp.sum(f) == 1
    c_m_t_constraints_list = []
    for mode, cons in c_m_t_constraints.items():
        for con in cons.values():
            c_m_t_constraints_list.append(con)
    constraints = user_rate_constraints + [c2] + [c3] + c5 + [c6] + c_m_t_constraints_list
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()

    if "infeasible" in prob.status:
        raise InfeasibleOptimization()
    assert "optimal" in prob.status

    return (
        prob.value,
        r.value,
        {
            m: {t: alpha.value for t, alpha in alphas_per_mode.items()}
            for m, alphas_per_mode in alphas.items()
        },
        [
            np.array([c.dual_value for c in user_rate_constraints]),
            c2.dual_value,
            c3.dual_value,
            {
                mode: {t: c.dual_value for t, c in cons.items()}
                for mode, cons in f_constraints.items()
            },
            c6.dual_value,
            {
                mode: {t: c.dual_value for t, c in cons.items()}
                for mode, cons in c_m_t_constraints.items()
            },
        ],
    )


def timesharing_network_dual(cost_function, la_m_t_s, network, q_min=None, q_max=None):

    c_m_t = {m: {} for m in network.modes}
    c_m_t_constraints = {m: {} for m in network.modes}
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

    c2 = r >= q_min
    c3 = r <= q_max

    constraints = user_rate_constraints + [c2, c3]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()

    if "infeasible" in prob.status:
        raise InfeasibleOptimization()
    assert "optimal" in prob.status

    return (
        prob.value,
        r.value,
        {m: {t: c.value for t, c in c_t.items()} for m, c_t in c_m_t.items()},
        [
            np.array([c.dual_value for c in user_rate_constraints]),
            c2.dual_value,
            c3.dual_value,
        ],
    )
