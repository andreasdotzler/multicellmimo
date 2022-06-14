"""TODO."""
import copy
import random

import cvxpy as cp
from cvxpy.constraints.nonpos import NonNeg
import numpy as np
import  logging
import copy
from numpy.lib.arraysetops import unique

from mcm.network import Network
import numpy.random

from .utils import InfeasibleOptimization
LOGGER = logging.getLogger(__name__)



# TODO we can use time_sharing network for the approximation

# solve the problem U_C_Q directly, difficult with approximation, because a lot -Inf to handle, subgradient should work
# solve U_Q + I_C check
# solve U + I_C_Q check

# need to extend to
# - one phy
# - one phy per mode
# - one phy per mode and transmitter

# need the protocol implementations

# verify the normal cone thing




# These tow functions are my very strange way to build a wsr from the time_sharing use an object
def I_C(A):
    return lambda weights: time_sharing_no_duals(weighted_sum_rate(weights), A)


def I_C_Q(A, q_min, q_max):
    return lambda weights: time_sharing_no_duals(weighted_sum_rate(weights), A, q_min, q_max)


def dual_problem_app(util, weights, q_max, q_min):
    q = cp.Variable(len(q_max))
    cost_dual = (
            util(q) - weights @ q
    )
    constraints_dual = [q >= q_min, q <= q_max]
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve(solver=cp.SCS, eps=1e-8)
    return prob_dual.value, q.value



def weighted_sum_rate(weights):
    return lambda r : weights @ r

def proportional_fair(r):
    #return cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r))
    return cp.sum(cp.log(r))

def app_layer(weights):
    return lambda r : cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r)) -  weights @ r


def optimize_network_explict(util, q_min, q_max, network: Network):
    # sum(mu_t) : mu_t <= F_t(f_t_i) - xi_t_i(f - f_t_i) \forall t \forall i, f \in F
    f_equal = 1 / len(network.modes)
    f_t = [{m: f_equal for m in network.modes}]
    d_f_n_s = []
    F_t_s = []
    max_util = None
    for i in range(0,100):
        # evaluate F(f)
        v_n, r_n, alphas_n, d_f_n, F_t = network.util_fixed_fractions(f_t[i], util, q_min, q_max)
        if max_util is None:
            max_util = v_n
        max_util = max(max_util, v_n)
        # evalute approximation
        d_f_n_s.append(d_f_n)
        F_t_s.append(F_t)
        mu = {}
        t_cons = []
        f = {mode: cp.Variable(1, nonneg=True) for mode in network.modes}
        for t in network.transmitters:
            mu[t] = cp.Variable(1)
            for j in range(i+1):
                t_cons.append(mu[t] <= F_t_s[j][t] + cp.sum([d_f_n_s[j][t][m]*(f[m] - f_t[j][m]) for m in d_f_n_s[j][t].keys()]))
        sum_con = [cp.sum(list(f.values())) == 1]
        prob = cp.Problem(cp.Maximize(cp.sum(list(mu.values()))), sum_con + t_cons)
        prob.solve()
        LOGGER.info(f"Explicit: Iteration {i} - Approximation {prob.value} - Best Primal Value {max_util}")
        assert prob.status == 'optimal'
        # solve
        f_t.append({m: ff.value[0] for m, ff in f.items()})
        if prob.value - max_util <= 1e-3:
            break
    return v_n, r_n



def optimize_network_app_phy(util, q_min, q_max, network):
    return optimize_app_phy(util, q_min, q_max, network.wsr_per_mode)

def optimize_dual_decomp_subgradient(util, q_min, q_max, wsr_phy):
    weights = np.ones(len(q_min))
    for i in range(1000):
        v_phy, c = wsr_phy(weights)
        v_app, q = dual_problem_app(util, weights, q_max, q_min)
        weights -= 1/(i+1)*(c - q)
        if i == 0:
            r = c
        else:
            r = i/(i+1)*r + 1/(i+1)*c  
        primal_value = sum(np.log(r))
        dual_value = v_app + v_phy
        gap = (dual_value - primal_value)

        LOGGER.info(f"Dual Subgradient: Iteration {i} - Dual Value {dual_value}- Primal Value {primal_value} - Gap {gap}")
        if gap < 1e-6:
            break
    return primal_value, r    


def optimize_dual_decomp_approx(util, q_min, q_max, wsr_phy):
    weights = np.ones(len(q_min))
    # strange way to evaluate util(q_min)
    v_q_min, _ = dual_problem_app(util, np.zeros_like(weights), q_min, q_min)
    # this is equal to evaluate the function for  lambda = [-Inf, ..., -Inf] 
    U_i = [v_q_min]
    z_i = [np.zeros((len(q_min),1))]
    c_i = [q_min]
    q_i = [q_min]
    best_dual = np.Inf
    for i in range(1000):
        #alpha = cp.Variable((i+1,1), nonneg=True)  
        #c_sum = cp.sum(alpha) == 1
        #c_schedule = z_i@alpha == np.zeros((len(q_min),1)) 
        #prob = cp.Problem(cp.Maximize(alpha.T@U_i), [c_sum, c_schedule])
        #prob.solve()

        mu = cp.Variable(1)
        la = cp.Variable(len(q_min))
        cons = [mu >= u + la@(c - q) for u,c,q in zip(U_i, c_i, q_i)]
        #cons = [mu >= u + la@z for u,z in zip(U_i, z_i)]
        prob = cp.Problem(cp.Minimize(mu), cons)
        prob.solve()
        # alpha = [c.dual_value[0] for c in cons]
        # sum([a*u for a, u in zip(alpha, U_i)]) == prob.value
        # sum([a*z for a, z in zip(alpha, z_i)]) == 0

        #mu = cp.Variable(2)
        #la = cp.Variable(len(q_min))
        #cons_U_Q = [mu[0] >= u - la@q for u, q in zip(U_i, q_i)]
        #cons_I_C = [mu[i+1] >= la@c  for c in c_i]
        #prob = cp.Problem(cp.Minimize(sum(mu)), cons_U_Q + cons_I_C)
        #prob.solve()

        mu = cp.Variable(len(q_min)+1)
        la = cp.Variable(len(q_min))
        cons_U_Q = []
        
        for j in range(len(q_min)):
            cons_U_Q += [mu[j] >= np.log(q[j]) - la[j]*q[j] for u, q in zip(U_i, q_i)]
        cons_I_C = [mu[j+1] >= la@c  for c in c_i]
        prob = cp.Problem(cp.Minimize(sum(mu)), cons_U_Q + cons_I_C)
        prob.solve()


        weights = la.value
        v_phy, c = wsr_phy(weights)        
        v_app, q = dual_problem_app(util, weights, q_max, q_min)
        c_i.append(c)
        q_i.append(q)
        z_i.append(c - q)
        
        # z_i = np.c_[z_i, (q - c)]
        # v_app = U(q) - weights@q -> U(q) = v_app + weights@q
        U_i.append(v_app + weights@q)
        
        primal_value = prob.value
        best_dual = min(v_app + v_phy, best_dual)
        gap = (best_dual - primal_value) / abs(best_dual)

        LOGGER.info(f"Dual Approx: Iteration {i} - Dual Value {best_dual}- Primal Value {primal_value} - Gap {gap}")
        if gap < 1e-3:
            break
    return primal_value, q    




def optimize_app_phy(util, q_min, q_max, wsr_phy):
    
    assert util == proportional_fair
    n_users = len(q_min)
    assert len(q_max) == n_users
    A = np.minimum(q_max,q_min).reshape((n_users,1))

    for n in range(1, 1000):
        # create and solve the approximated problem

        approx_value, q, alpha, [weights, w_min, w_max, mu] = time_sharing_cvx(util, A, q_min, q_max)

        import pytest
        assert 1 / q + w_min - w_max - weights == pytest.approx(np.zeros(len(q)), rel=1e-2, abs=1e-1)
        # TDODO we need to verify the weights here

        assert all(weights @ (A - q.reshape(len(q), 1)) <= 0.001)

        q_app = np.minimum(q_max, np.maximum(q_min, 1 / weights))
        q_app[weights <= 0] = q_max[weights <= 0]
        v_app = sum(np.log(q_app)) - weights @ q_app


        v_phy, c = wsr_phy(weights)
        A = np.c_[A, c]
        #print("New approximation points: ", c)
        #mu = max((weights @ A).tolist()[0])
        # TODO check the breaking criterium
        # U*(la) + R*(la) >= min U*(la) + R*(la) = max U(q) + R(q) >= U(q) + R_approx(q)
        # U*(la) = v_app
        # R*(la) = v_phy
        # Q(q) = approx_value
        # R_approx(q) = 0
        # TODO v_app + v_phy ist the current dual value, we shoul track the minmal one
        dual_value = v_app + v_phy
        LOGGER.info(f"Max_mode: Iteration {n} - Dual Approximation {approx_value} - Dual Value {dual_value}")
        if abs(dual_value - approx_value) < 0.001:
            break
    return approx_value, q




# TODO make cost function a variable
# TDOO we neet to find out what we are really doing her?
# Very likely we are doing the mode competition and keep all approximation points,
# but we should be able to to better, by using I_C_m_l and a weight for every function
# Approximation max_{c_m_t ...} V(c_m_t ...) + sum_{m} sum_{t} I_C_m_L(c_m_t)
# and inner approximation of I_C_m_l

# TODO, Does the approximation problem reformulate?

# compare to (4.97) in the thesis
def optimize_network_app_network(util, q_min, q_max, network):

    n_users = len(q_min)
    assert len(q_max) == n_users
    # we need to initialize a fake mode that is all q_min
    users_per_transmitter = {}
    users_per_node_and_transmitter_approx = {'init_mode': {}}
    for mode in network.As:
        users_per_node_and_transmitter_approx[mode] = {}
        for transmitter, users in network.users_per_mode_and_transmitter[mode].items():
            if transmitter not in users_per_transmitter:
                users_per_transmitter[transmitter] = []
            users_per_transmitter[transmitter] += users

    A_approx = {}
    A_approx['init_mode'] = {}
    for transmitter, users in users_per_transmitter.items():
        unique_users = list(set(users))
        a = q_min[unique_users]
        a = a.reshape(len(a),1)
        A_approx['init_mode'][transmitter] = a
        users_per_node_and_transmitter_approx['init_mode'][transmitter] = unique_users

    for n in range(1, 1000):
        # create and solve the approximated problem

        # time_sharing_network

        approx_value, rates, alphas, [weights, w_min, w_max, _, _, weights_m_t] = timesharing_network(util,
                                                                users_per_node_and_transmitter_approx, A_approx,
                                                                q_min, q_max)

        import pytest

        assert 1 / rates + w_min - w_max - weights == pytest.approx(np.zeros(len(rates)), rel=1e-2, abs=1e-1)
        for mode, A_approx_ts in A_approx.items():
            for transmitter, A_m_t in A_approx_ts.items():
                q_m_t = A_m_t @ alphas[mode][transmitter]
                assert all(weights_m_t[mode][transmitter] @ (A_m_t * sum(alphas[mode][transmitter]) - q_m_t.reshape(len(q_m_t), 1)) <= 0.001)
        #is lambda the same for them by definition of our duality?
        # V(c_1_1, ... , c_m_t) = max_{f, q_1, q_t} {sum_t U_t(q_t) : q_t in Q_t, q_t = sum_{f_1, ..., f_m} f_m*c_m_t, f in F}
        # V*(l_1_1, ... , l_m_t) = max_{f, q_1, q_t} {sum_t U_t(q_t) - l_m_t @ c_m_t : q_t in Q_t, q_t = sum_{f_1, ..., f_m} f_m*c_m_t, f in F}
        # ==> we pick the cheapest mode to maximize?
        # maybe we realy need to solve a different approximation problem? Or reconstruct weigths for a different but equivalen problem
        # TDODO we need to verify the weights here

        q_app = np.minimum(q_max, np.maximum(q_min, 1 / weights))
        q_app[weights <= 0] = q_max[weights <= 0]
        v_app = sum(np.log(q_app)) - weights @ q_app

        # wsr_per_mode_and_transmitter
        values, A_max = network.wsr_per_mode_and_transmitter(weights)
        # compute the maximal mode
        max_mode = None
        max_val = None
        for mode, transmitter_values in values.items():
            sum_vals = 0
            for val in transmitter_values.values():
                sum_vals += val
            if max_mode is None:
                max_mode = mode
                max_val = sum_vals
            else:
                if sum_vals > max_val:
                    max_val = sum_vals
                    max_mode = mode
        v_phy = max_val

        # update the approximation
        for mode, rates_per_transmitter in A_max.items():
            if mode not in A_approx:
                A_approx[mode] = {}
            for transmitter, rates in rates_per_transmitter.items():
                if transmitter not in A_approx[mode]:
                    users_per_node_and_transmitter_approx[mode][transmitter] = network.users_per_mode_and_transmitter[mode][transmitter]
                    A_approx[mode][transmitter] = rates.reshape((len(rates),1))
                else:
                    A_approx[mode][transmitter] = np.c_[A_approx[mode][transmitter], rates.reshape((len(rates),1))]

        dual_value = v_app + v_phy
        LOGGER.info(f"Network: Iterabtion {n} - Dual Approximation {approx_value} - Dual Value {dual_value}")
        if abs(dual_value - approx_value) < 0.001:
            break

    return approx_value, q_app, alphas


def time_sharing_cvx(cost_function, A, q_min = None, q_max = None):
    n_users, n = A.shape
    r = cp.Variable(n_users, pos=True)
    cost = cost_function(r)

    if q_min is None:
        q_min = np.zeros(r.size)
    if q_max is None:
        q_max = np.squeeze(np.asarray(np.amax(A,1)))


    alpha = cp.Variable(n, nonneg=True)
    c1 = r == A @ alpha
    c2 = r >= q_min
    c3 = r <= q_max
    c5 = cp.sum(alpha) == 1
    constraints = [c1, c2, c3, c5]

    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()

    # check KKT
    if 'infeasible' in prob.status:
        raise InfeasibleOptimization()
    assert prob.status == 'optimal'
    return prob.value, r.value, alpha.value, [c1.dual_value, c2.dual_value, c3.dual_value, c5.dual_value]

def time_sharing(cost_function, A, q_min = None, q_max = None):
    value, rates, alpha, [lambda_opt, _, _, _] = time_sharing_cvx(cost_function, A, q_min, q_max)
    return value, rates, alpha, lambda_opt

def time_sharing_no_duals(costs_function, A, q_min=None, q_max=None):
    value, rates, _, _ = time_sharing_cvx(costs_function, A, q_min, q_max)
    return value, rates

def timesharing_fixed_fractions(cost_function, f, users_per_mode, As, q_min = None, q_max = None):

    users = []
    alphas = {}
    c_m_constraints = {}
    for mode, rates in As.items():
        alphas[mode] = cp.Variable(rates.shape[1], nonneg=True)
        users += users_per_mode[mode]
    
    users = unique(users)
    r_constraints = {}
    c_m = {}
    for mode, users in users_per_mode.items():
        c_m[mode] = cp.Variable(len(users), nonneg=True)
        c_m_constraints[mode] = c_m[mode] ==  As[mode] @ alphas[mode]
        for user_index, user in enumerate(users):
            constraint = c_m[mode][user_index]
            if user in r_constraints:
                r_constraints[user] += constraint
            else:
                r_constraints[user] = constraint

    n_users = len(r_constraints)
    r = cp.Variable(n_users, nonneg=True)
    user_rate_constraints = [r_k == r_constraints[user] for user, r_k in zip(users,r)]
    cost = cost_function(r)
    if q_min is None:
        q_min = np.zeros(r.size)

    if q_max is None:
        q_max = np.ones(n_users) * 10 #np.squeeze(np.asarray(np.amax(A,1)))

 
    c2 = r >= q_min
    c3 = r <= q_max
    c5 = {}
    for mode in alphas:
        c5[mode] = cp.sum(alphas[mode]) == f[mode]

    c_m_constraints_list = []
    for mode, cons in c_m_constraints.items():
        c_m_constraints_list.append(cons)
    constraints = user_rate_constraints + [c2, c3] + list(c5.values()) + c_m_constraints_list
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()
    assert 'optimal' in prob.status 
    return (
        prob.value, 
        {user: r.value for user, r in zip(users, r)}, 
        {mode: alpha.value for mode, alpha in alphas.items()}, 
        [
            [c.dual_value for c in user_rate_constraints], 
            c2.dual_value, 
            c3.dual_value, 
            {mode: c.dual_value for mode, c in c5.items()},
            {mode: c.dual_value for mode, c in c_m_constraints.items()}
        ])

def timesharing_network(cost_function, users_per_mode_and_transmitter, As, q_min = None, q_max = None):

    alphas = {}
    for mode, rates_per_transmitter in As.items():
        alphas[mode] = {}
        for transmitter, rates in rates_per_transmitter.items():
            alphas[mode][transmitter] = cp.Variable(rates.shape[1], nonneg=True)


  
    r_constraints = {}
    c_m_t = {}
    c_m_t_constraints = {}
    for mode, transmitters_and_users in users_per_mode_and_transmitter.items():
        c_m_t[mode] = {}
        c_m_t_constraints[mode] = {}
        for transmitter, users in transmitters_and_users.items():
            c_m_t[mode][transmitter] = cp.Variable(len(users), nonneg=True)

            c_m_t_constraints[mode][transmitter] = c_m_t[mode][transmitter] ==  As[mode][transmitter] @ alphas[mode][transmitter]
            for user_index, user in enumerate(users):
                constraint = c_m_t[mode][transmitter][user_index]
                if user in r_constraints:
                    r_constraints[user] += constraint
                else:
                    r_constraints[user] = constraint



    n_users = len(r_constraints)
    r = cp.Variable(n_users, nonneg=True)
    user_rate_constraints = [r_k == r_constraints[user] for user, r_k in enumerate(r)]
    cost = cost_function(r)
    if q_min is None:
        q_min = np.zeros(r.size)
    if q_max is None:
        q_max = np.ones(n_users) * 10 #np.squeeze(np.asarray(np.amax(A,1)))

    c2 = r >= q_min
    c3 = r <= q_max
    sums_alpha_per_mode = cp.Variable(len(alphas), nonneg=True)
    c5 = []
    sums_alpha_per_mode_constraints = {}
    for (mode, alphas_per_mode), sum_m  in zip(alphas.items(), sums_alpha_per_mode):
        sums_alpha_per_mode_constraints[mode] = {}
        for transmitter, alpha in alphas_per_mode.items():
            con = cp.sum(alpha) == sum_m
            sums_alpha_per_mode_constraints[mode][transmitter] = con
            c5.append(con)

    c6 = cp.sum(sums_alpha_per_mode) == 1
    c_m_t_constraints_list = []
    for mode, cons in c_m_t_constraints.items():
        for con in cons.values():
            c_m_t_constraints_list.append(con)
    constraints = user_rate_constraints + [c2, c3] + c5 + [c6] + c_m_t_constraints_list
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()

    if 'infeasible' in prob.status:
        raise InfeasibleOptimization()
    assert 'optimal' in prob.status




    return (
        prob.value, 
        r.value, 
        {m: {t: alpha.value for t, alpha in alphas_per_mode.items()} for m, alphas_per_mode in alphas.items()},
        [np.array([
            c.dual_value for c in user_rate_constraints]), 
            c2.dual_value, 
            c3.dual_value, 
            {mode: {t: c.dual_value for t,c in cons.items()} for mode, cons in sums_alpha_per_mode_constraints.items()}, 
            c6.dual_value, 
            {mode: {t:c.dual_value for t,c in cons.items()} for mode, cons in c_m_t_constraints.items() }])

