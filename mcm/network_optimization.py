"""TODO."""
import copy
import random

import cvxpy as cp
from cvxpy.constraints.nonpos import NonNeg
import numpy as np
import  logging
import copy
from numpy.lib.arraysetops import unique


import numpy.random
from pytest import approx

from .utils import InfeasibleOptimization
LOGGER = logging.getLogger(__name__)
from typing import Callable




class Transmitter:
    def __init__(self, users_per_mode, wsr_per_mode, id=None):
        self.id = id
        self.users_per_mode : dict[str, list[int]] = users_per_mode
        self.wsr_per_mode: dict[str, Callable[[np.array],(int, np.array)]] = wsr_per_mode
        self.As_per_mode: dict[str, np.array] = {}
        self.modes = list(wsr_per_mode.keys())
        self.users = []
        for users in users_per_mode.values():
            self.users += users
        self.users = list(set(self.users))


    def wsr(self, weights, mode):
        t_weights = weights[self.users_per_mode[mode]]
        val, rates = self.wsr_per_mode[mode](t_weights)
        if mode not in self.As_per_mode:
            self.As_per_mode[mode] = rates.reshape((len(rates),1))
        else:
            self.As_per_mode[mode] = np.c_[self.As_per_mode[mode], rates.reshape((len(rates),1))]
        return val, rates


    def util_fixed_fractions(self, fractions, util, q_min, q_max):
        return timesharing_fixed_fractions(util, fractions, self.users_per_mode, self.As_per_mode, q_min, q_max)

class Network:
    def __init__(self, transmitters : Transmitter):

        self.transmitters = transmitters
        self.users = []
        self.modes = []
        for t in transmitters.values():
            self.users += t.users    
            self.modes += t.modes
        self.users = list(set(self.users))
        self.modes = list(set(self.modes))

    #def wsr(self, weights):
    #    value, rates, _, _ = timesharing_network(weighted_sum_rate(weights), self.users_per_mode_and_transmitter, self.As)
    #    return value, rates

    def initialize_approximation(self, As):
        for mode, trans_and_At in As.items():
            for trans, At in trans_and_At.items():
                self.transmitters[trans].As_per_mode[mode] = At

    def reset_approximation(self):
        for t in self.transmitters.values():
            t.As_per_mode = {}


    def wsr_per_mode(self, weights):
        max_value = -np.Inf
        mode_values, A_m = self.wsr_per_mode_and_transmitter(weights)
        w_m = {m: sum(w_m_t.values()) for m, w_m_t in mode_values.items()}
        max_mode, max_value = max(w_m.items(), key = lambda k : k[1]) 
        max_rates = np.zeros(len(weights))
        for transmitter_id, rates in A_m[max_mode].items():
            max_rates[self.transmitters[transmitter_id].users] += rates
        assert len(weights) == len(max_rates)
        return max_value, max_rates

    def wsr_per_mode_and_transmitter(self, weights):
        values = {}
        A_max = {}
        for transmitter_id, transmitter in self.transmitters.items():
            for mode in transmitter.modes:
                val, rates = transmitter.wsr(weights, mode)
                if mode not in values:
                    values[mode] = {}
                values[mode][transmitter_id] = val
                if mode not in A_max:
                    A_max[mode] = {}
                A_max[mode][transmitter_id] = rates                
        return values, A_max

    def scheduling(self, fractions, util, q_min, q_max):
        F = 0
        F_t_s = {}
        r = np.zeros(len(self.users))
        alphas = {}
        d_f = {}
        for transmitter_id, t in self.transmitters.items():
            F_t, r_t, alpha_t, [lambdas, w_min, w_max, d_f_t_m, d_c_m] = t.util_fixed_fractions(
                fractions, util, q_min[t.users], q_max[t.users])
            for mode, a in alpha_t.items():
                if mode not in alphas:
                    alphas[mode] = {}
                alphas[mode][transmitter_id] = a
            #for mode, d in d_f_t_m.items():
            #    if mode not in d_f:
            #        d_f[mode] = {}
            d_f[transmitter_id] = d_f_t_m
            F += F_t
            F_t_s[transmitter_id] = F_t
            for user, rate in r_t.items():
                r[user] += rate
        return F, r, alphas, d_f, F_t_s       
    
    def get_As(self):
        As = {m: {} for m in self.modes}
        for t_id, t in self.transmitters.items():
            for mode in t.modes:
                As[mode][t_id] = t.As_per_mode[mode]
        return As


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
def I_C_s(A):
    return lambda weights: time_sharing_no_duals(weighted_sum_rate(weights), A)

def wsr_for_A(weights, A):
    max_i = np.argmax(weights @ A)
    rates = A[:, max_i]
    return weights @ rates, rates

def I_C(A):
    return lambda weights: wsr_for_A(weights, A)



def I_C_Q(A, q_min, q_max):
    return lambda weights: time_sharing_no_duals(weighted_sum_rate(weights), A, q_min, q_max)


def dual_problem_app(util, weights, q_max = None, q_min = None):
    q = cp.Variable(len(q_max))
    cost_dual = (util(q) - weights @ q)
    constraints_dual = []
    if q_max is not None:        
        constraints_dual.append(q <= q_max)
    if q_min is not None:
        constraints_dual.append(q >= q_min)
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve()
    return prob_dual.value, q.value
    # TODO, this is a short cut for proportional fail
    #    q_app = np.minimum(q_max, np.maximum(q_min, 1 / weights))
    #    q_app[weights <= 0] = q_max[weights <= 0]
    #    v_app = sum(np.log(q_app)) - weights @ q_app
    #    assert abs(v_app_1 - v_app) <= 10**-6



def weighted_sum_rate(weights):
    return lambda r : weights @ r

def proportional_fair(r):
    #return cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r))
    return cp.sum(cp.log(r))

def app_layer(weights):
    return lambda r : cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r)) -  weights @ r

# algorithm 1
def comp_resources_dual_subgradient(util, q_min, q_max, network: Network):
    n_users = len(q_min)
    weights = np.ones(n_users)
    for n in range(1, 1000):
        # wsr_per_mode_and_transmitter
        values, A_max = network.wsr_per_mode_and_transmitter(weights)
        w_m = {m: sum(w_m_t.values()) for m, w_m_t in values.items()}
        m_opt, v_phy = max(w_m.items(), key = lambda k : k[1]) 

        
        q_app = np.minimum(q_max, np.maximum(q_min, 1 / weights))
        q_app[weights <= 0] = q_max[weights <= 0]
        v_app = sum(np.log(q_app)) - weights @ q_app
        # TODO, how can we calculate this?
        approx_value = None
   
        dual_value = v_app + v_phy
        LOGGER.info(f"Network: Iterabtion {n} - Dual Approximation {approx_value} - Dual Value {dual_value}")
        if abs(dual_value - approx_value) < 0.001:
            break

# algorithm 4
def optimize_network_explict(util, q_min, q_max, network: Network):
    # sum(mu_t) : mu_t <= F_t(f_t_i) - xi_t_i(f - f_t_i) \forall t \forall i, f \in F
    f_equal = 1 / len(network.modes)
    f_t = [{m: f_equal for m in network.modes}]
    d_f_n_s = []
    F_t_s = []
    max_util = - np.Inf
    for i in range(0,100):
        # evaluate F(f)
        v_n, r_n, alphas_n, d_f_n, F_t = network.scheduling(f_t[i], util, q_min, q_max)

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


def optimize_app_phy(util, q_min, q_max, wsr_phy):
    
    assert util == proportional_fair
    n_users = len(q_min)
    assert len(q_max) == n_users
    A = np.minimum(q_max,q_min).reshape((n_users,1))

    best_dual_value = np.inf
    for n in range(1, 1000):
        # create and solve the approximated problem
        approx_value, q, alpha, [la, w_min, w_max, mu] = time_sharing_cvx(util, A, q_min, q_max)
        
        # solve the dual problem to provide bound and update 
        v_app, _ = dual_problem_app(util, la, q_max, q_min)
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
        LOGGER.info(f"Iteration {n} - Primal Approx {approx_value} - Dual Approx {dual_value} - Gap: {gap} ")
        if gap < 0.001:
            break
    return approx_value, q, alpha, [la, w_min, w_max, mu]


# algorithm5 
def optimize_network_app_network(util, q_min, q_max, network: Network):

    n_users = len(q_min)
    assert len(q_max) == n_users
    # we need to initialize a fake mode that is all q_min
    for transmitter in network.transmitters.values():
        unique_users = list(set(transmitter.users))
        a = q_min[unique_users]
        a = a.reshape(len(a),1)
        transmitter.As_per_mode['init'] = a
        transmitter.users_per_mode['init'] = unique_users
        transmitter.modes.append('init')
        transmitter.wsr_per_mode['init'] = I_C(a)
    network.modes.append('init')

    for n in range(1, 1000):
        approx_value, _, alphas, [weights, _, _, _, _, _] = timesharing_network(util, network, q_min, q_max)                                                           

        q_app = np.minimum(q_max, np.maximum(q_min, 1 / weights))
        q_app[weights <= 0] = q_max[weights <= 0]
        v_app = sum(np.log(q_app)) - weights @ q_app

        # wsr_per_mode_and_transmitter
        values, A_max = network.wsr_per_mode_and_transmitter(weights)
        v_phy = max([sum(v.values()) for v in values.values()])
   
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

def timesharing_network(cost_function, network, q_min = None, q_max = None):

    alphas = {m: {} for m in network.modes}
    r_constraints = {}
    c_m_t = {m: {} for m in network.modes}
    c_m_t_constraints = {m: {} for m in network.modes}

    for t_id, transmitter in network.transmitters.items():
        for mode, rates in transmitter.As_per_mode.items():
            alphas[mode][t_id] = cp.Variable(rates.shape[1], nonneg=True)
            users = transmitter.users_per_mode[mode]
            c_m_t[mode][t_id] = cp.Variable(len(users), nonneg=True)
            c_m_t_constraints[mode][t_id] = c_m_t[mode][t_id] == transmitter.As_per_mode[mode] @ alphas[mode][t_id]
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