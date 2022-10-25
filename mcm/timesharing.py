import fractions
import numpy as np
import cvxpy as cp
import logging
from numpy.lib.arraysetops import unique

from mcm.no_utils import InfeasibleOptimization
from mcm.regions import Q_vector, R_m_t_approx

LOGGER = logging.getLogger(__name__)

def solve_problem(util, cons):
    prob = cp.Problem(util, cons)
    prob.solve()
    if "infeasible" in prob.status:
        raise InfeasibleOptimization()
    assert "optimal" in prob.status, f"unable to solve problem: {prob.status}"
    return prob


def check_feasible(f, R_m, Q):
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

    return (
        prob.value,
        {user: r for user, r in zip(users, q_sum.value)}
    )


def F_t_R_approx_conj(cost_function, la, users, R_m: dict[str, R_m_t_approx], Q: Q_vector):

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
        {m: f_m.value for m,f_m in f.items()}
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

    prob = solve_problem(cp.Maximize(cost), constraints)

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


