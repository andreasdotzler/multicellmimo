"""TODO."""
import copy

import cvxpy as cp
import numpy as np
import  logging
import copy
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

def weighted_sum_rate(weights):
    return lambda r : weights @ r

def proportional_fair(r):
    #return cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r))
    return cp.sum(cp.log(r))

def app_layer(weights):
    return lambda r : cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r)) -  weights @ r


def optimize_network_app_phy(q_min, q_max, wsr_phy):

    n_users = len(q_min)
    assert len(q_max) == n_users
    A = np.minimum(q_max,q_min).reshape((n_users,1))

    for n in range(1, 1000):
        # create and solve the approximated problem

        approx_value, q, alpha, [weights, w_min, w_max, mu] = time_sharing_dual(proportional_fair, A, q_min, q_max)

        import pytest
        assert 1 / q + w_min - w_max - weights == pytest.approx(np.zeros(len(q)), rel=1e-2, abs=1e-1)
        # TDODO we need to verify the weights here

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
        LOGGER.info(f"Iteraction {n} - Dual Approximation {approx_value} - Dual Value {dual_value}")
        if abs(dual_value - approx_value) < 0.001:
            break
    return approx_value, q

# TODO make cost function a variable
def optimize_network_app_network(q_min, q_max, network):
    wsr_phy = network.wsr_per_mode
    users_per_transmitter = {}


    users_per_node_and_transmitter_approx = {'init_mode': {}}
    # # initialize empty was not a good idea
    for mode in network.As:
    #     A_approx[mode] = {}
        users_per_node_and_transmitter_approx[mode] = {}
        for transmitter, users in network.users_per_mode_and_transmitter[mode].items():
    #         A_approx[mode][transmitter] = np.empty((0,len(users)))
            if transmitter not in users_per_transmitter:
                users_per_transmitter[transmitter] = []
            users_per_transmitter[transmitter] += users
            #users_per_node_and_transmitter_approx[mode][transmitter] = copy.copy(users)

    A_approx = {}
    A_approx['init_mode'] = {}
    # this iw wrong, we are iterating over modes

    for transmitter, users in users_per_transmitter.items():
        unique_users = list(set(users))
        a = q_min[unique_users]
        a = a.reshape(len(a),1)
        A_approx['init_mode'][transmitter] = a
        users_per_node_and_transmitter_approx['init_mode'][transmitter] = unique_users


    n_users = len(q_min)
    assert len(q_max) == n_users
    A = np.minimum(q_max,q_min).reshape((n_users,1))
    # we need to initialize a fake mode that is all q_min

    for n in range(1, 1000):
        # create and solve the approximated problem

        # time_sharing_network
        approx_value, q_app, alpha, [weights_1, w_min, w_max, mu] = time_sharing_dual(proportional_fair, A, q_min, q_max)
        approx_value, rates, alphas, weights = timesharing_network(proportional_fair,
                                                                users_per_node_and_transmitter_approx, A_approx,
                                                                q_min, q_max)


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
        #import ipdb; ipdb.set_trace()

        for mode, rates_per_transmitter in A_max.items():
            if mode not in A_approx:
                A_approx[mode] = {}
            for transmitter, rates in rates_per_transmitter.items():
                if transmitter not in A_approx[mode]:
                    users_per_node_and_transmitter_approx[mode][transmitter] = network.users_per_mode_and_transmitter[mode][transmitter]
                    A_approx[mode][transmitter] = rates.reshape((len(rates),1))
                else:
                    A_approx[mode][transmitter] = np.c_[A_approx[mode][transmitter], rates.reshape((len(rates),1))]

        v_phy, c = wsr_phy(weights)
        A = np.c_[A, c]
        v_phy = max_val


        dual_value = v_app + v_phy
        LOGGER.info(f"Iteraction {n} - Dual Approximation {approx_value} - Dual Value {dual_value}")
        if abs(dual_value - approx_value) < 0.001:
            import ipdb; ipdb.set_trace()
            break

    return approx_value, q_app



def time_sharing_dual(cost_function, A, q_min = None, q_max = None):
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
    #cost = sum(r)
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()

    # check KKT
    if 'infeasible' in prob.status:
        raise InfeasibleOptimization()
    assert prob.status == 'optimal'
    return prob.value, r.value, alpha.value, [c1.dual_value, c2.dual_value, c3.dual_value, c5.dual_value]

def time_sharing(cost_function, A, q_min = None, q_max = None):
    value, rates, alpha, [lambda_opt, w_min, w_max, mu] = time_sharing_dual(cost_function, A, q_min, q_max)
    return value, rates, alpha, lambda_opt


def timesharing_network(cost_function, users_per_mode_and_transmitter, As, q_min = None, q_max = None):

    alphas = {}
    for mode, rates_per_transmitter in As.items():
        alphas[mode] = {}
        for transmitter, rates in rates_per_transmitter.items():
            alphas[mode][transmitter] = cp.Variable(rates.shape[1], nonneg=True)


    r_constraints = {}
    for mode, transmitters_and_users in users_per_mode_and_transmitter.items():
        for transmitter, users in transmitters_and_users.items():
            for user_index, user in enumerate(users):
                constraint = cp.sum(alphas[mode][transmitter] @ As[mode][transmitter][user_index])
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
    for alphas_per_mode, sum_m  in zip(alphas.values(), sums_alpha_per_mode):
        for alpha in alphas_per_mode.values():
            c5.append(cp.sum(alpha) == sum_m)

    c6 = cp.sum(sums_alpha_per_mode) == 1
    constraints = user_rate_constraints + [c2, c3] + c5 + [c6]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-6)

    if 'infeasible' in prob.status:
        raise InfeasibleOptimization()
    assert 'optimal' in prob.status

    return prob.value, r.value, {m: {t: alpha.value for t, alpha in alphas_per_mode.items()} for m, alphas_per_mode in alphas.items()}, np.array([c.dual_value for c in user_rate_constraints])


class Network:
    def __init__(self, users_per_mode_and_transmitter, As):
        self.users_per_mode_and_transmitter = users_per_mode_and_transmitter
        self.As = As

    def wsr(self, weights):
        value, rates, _, _ = timesharing_network(weighted_sum_rate(weights), self.users_per_mode_and_transmitter, self.As)
        return value, rates

    def wsr_per_mode(self, weights):
        max_value = -np.Inf
        max_rates = None
        _, A_m = self.wsr_per_mode_and_transmitter(weights)
        for mode in self.As:
            mode_rates = np.zeros(len(weights))
            mode_value = 0
            for transmitter, users in self.users_per_mode_and_transmitter[mode].items():
                rates = A_m[mode][transmitter]
                value = weights[users] @ rates
                mode_rates[users] += rates
                mode_value += value
            if mode_value > max_value:
                max_value = mode_value
                max_rates = mode_rates
        assert len(weights) == len(max_rates)
        return max_value, max_rates

    def wsr_per_mode_and_transmitter(self, weights):
        values = {}
        A_max = {}
        for mode in self.As:
            A_max[mode] = {}
            values[mode] = {}
            for transmitter, users in self.users_per_mode_and_transmitter[mode].items():
                t_weights = weights[users]
                # value, rates =  _, _ = timesharing_network(weighted_sum_rate(weights), {mode: self.users_per_mode_and_transmitter[mode]},
                #                                     {mode: self.As[mode]})
                #value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(t_weights), self.As[mode][transmitter])
                A = self.As[mode][transmitter]
                max_i = np.argmax(t_weights @ A)
                rates = A[:, max_i]
                #assert t_weights @ rates == pytest.approx(value, rel=1e3, abs=1e-2)
                A_max[mode][transmitter] = rates
                values[mode][transmitter] = t_weights @ rates
        return values, A_max