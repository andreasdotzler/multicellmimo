import cvxpy as cp

import numpy as np
import logging


from mcm.timesharing import timesharing_fixed_fractions_2, time_sharing, timesharing_network, time_sharing_cvx, time_sharing_no_duals

LOGGER = logging.getLogger(__name__)
from typing import Callable



class Transmitter:
    def __init__(
        self, users_per_mode, wsr_per_mode, id=None, util=None, q_min=None, q_max=None
    ):
        self.id = id
        self.users_per_mode: dict[str, list[int]] = users_per_mode
        any_users = next(iter(self.users_per_mode.values()))
        for users in self.users_per_mode.values():
            assert (
                users == any_users
            ), "Not implemented, if we want different users per mode, some things may break"
        self.wsr_per_mode: dict[
            str, Callable[[np.array], (int, np.array)]
        ] = wsr_per_mode
        self.As_per_mode: dict[str, np.array] = {}
        self.modes = list(wsr_per_mode.keys())
        self.users = []
        for users in users_per_mode.values():
            self.users += users
        self.users = list(set(self.users))
        self.util = util
        assert q_min is None or len(q_min) == len(self.users)
        self.q_min = q_min
        assert q_max is None or len(q_max) == len(self.users)
        self.q_max = q_max
        self.weights = np.ones(len(self.users))
        self.average_transmit_rate = None
        self.iteration = 0
        self.best_dual_value = np.Inf

    def wsr(self, t_weights, mode):
        if t_weights is None:
            t_weights = self.weights
        val, rates = self.wsr_per_mode[mode](t_weights)
        if mode not in self.As_per_mode:
            self.As_per_mode[mode] = rates.reshape((len(rates), 1))
        else:
            self.As_per_mode[mode] = np.c_[
                self.As_per_mode[mode], rates.reshape((len(rates), 1))
            ]
        return val, rates

    def update_weights(self, m_opt):
        if m_opt in self.As_per_mode:
            c_t_l_1 = self.As_per_mode[m_opt][:, -1]
        else:
            c_t_l_1 = np.zeros_like(self.weights)
        ## TODO elevate this to all users!
        # = np.zeros(len(self.users))
        # c_t_l_1[self.users_per_mode[m_opt]] += c_opt
        v_phy = self.weights @ c_t_l_1
        v_app, q_t_l_1 = dual_problem_app(
            self.util, self.weights, self.q_max, self.q_min
        )
        l = self.iteration
        self.weights -= 1/(l+1)*(c_t_l_1 - q_t_l_1)
        #self.weights -= c_t_l_1 - q_t_l_1
        if l == 0:
            self.average_transmit_rate = c_t_l_1
        else:
            self.average_transmit_rate = (
                l / (l + 1) * self.average_transmit_rate + 1 / (l + 1) * c_t_l_1
            )

        primal_value = self.util(
            cp.Variable(
                len(self.average_transmit_rate), value=self.average_transmit_rate
            )
        ).value
        self.iteration += 1

        return primal_value, v_app + v_phy

    def scheduling(self, fractions, util, q_min, q_max):
        return timesharing_fixed_fractions_2(
            util, fractions, self.users_per_mode, self.As_per_mode, q_min, q_max
        )



class Network:
    def __init__(self, transmitters: Transmitter):

        self.transmitters: dict[int, Transmitter] = transmitters
        self.users = []
        self.modes = []
        for t in transmitters.values():
            self.users += t.users
            self.modes += t.modes
        self.users = list(set(self.users))
        self.modes = list(set(self.modes))

    # def wsr(self, weights):
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
        max_mode, max_value = max(w_m.items(), key=lambda k: k[1])
        max_rates = np.zeros(len(weights))
        for transmitter_id, rates in A_m[max_mode].items():
            max_rates[self.transmitters[transmitter_id].users] += rates
        assert len(weights) == len(max_rates)
        return max_value, max_rates

    def wsr_per_mode_and_transmitter(self, weights=None):
        values = {}
        A_max = {}
        for transmitter_id, transmitter in self.transmitters.items():
            for mode in transmitter.modes:
                if weights is not None:
                    t_weights = weights[transmitter.users_per_mode[mode]]
                else:
                    t_weights = None
                val, rates = transmitter.wsr(t_weights, mode)
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
            (
                F_t,
                r_t,
                alpha_t,
                c_m,
                [d_f_t_m, d_c_m, la],
            ) = t.scheduling(fractions, util, q_min[t.users], q_max[t.users])
            for mode, a in alpha_t.items():
                if mode not in alphas:
                    alphas[mode] = {}
                alphas[mode][transmitter_id] = a
            # for mode, d in d_f_t_m.items():
            #    if mode not in d_f:
            #        d_f[mode] = {}
            d_f[transmitter_id] = {m: la @ c for m,c in c_m.items()}
            F += F_t
            F_t_s[transmitter_id] = F_t
            for user, rate in r_t.items():
                r[user] += rate
        return F, r, alphas, d_f, F_t_s

    def create_init_mode(self, q_min):
        # we need to initialize a fake mode that is all q_min
        for transmitter in self.transmitters.values():
            #unique_users = list(set(transmitter.users))
            for mode in transmitter.modes:
                a = q_min[transmitter.users_per_mode[mode]]
                transmitter.As_per_mode[mode] = a.reshape(len(a), 1)


    def get_As(self):
        As = {m: {} for m in self.modes}
        for t_id, t in self.transmitters.items():
            for mode in t.modes:
                As[mode][t_id] = t.As_per_mode[mode]
        return As

    def resource_allocation(self, f_t, d_f_n_s, F_t_s):
        i = len(f_t) - 1
        mu = {}
        t_cons = []
        f = {mode: cp.Variable(1, nonneg=True) for mode in self.modes}
        for t in self.transmitters:
            mu[t] = cp.Variable(1)
            for j in range(i + 1):
                t_cons.append(
                        mu[t]
                        <= F_t_s[j][t]
                        + cp.sum(
                            [
                                d_f_n_s[j][t][m] * (f[m] - f_t[j][m])
                                for m in d_f_n_s[j][t].keys()
                            ]
                        )
                    )
        sum_con = [cp.sum(list(f.values())) == 1]
        prob = cp.Problem(cp.Maximize(cp.sum(list(mu.values()))), sum_con + t_cons)
        prob.solve()
        assert prob.status == "optimal"
        f_new = {m: ff.value[0] for m, ff in f.items()}
        return prob.value, f_new

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
    return lambda weights: time_sharing_no_duals(
        weighted_sum_rate(weights), A, q_min, q_max
    )


def dual_problem_app(util, weights, q_max=None, q_min=None):
    q = cp.Variable(len(q_max))
    cost_dual = util(q) - weights @ q
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

def dual_problem_app_f(util, weights_per_mode, f, q_max=None, q_min=None):
    q = cp.Variable(len(q_max))
    c_s = {m: cp.Variable(len(w)) for m,w in weights_per_mode.items()}

    cost_dual = util(q)
    for m, weights in weights_per_mode.items():
        cost_dual -= weights @ c_s[m]    
    const_q = 0
    for m, c in c_s.items():
        const_q += f[m]*c
    constraints_dual = [q == const_q]
    if q_max is not None:
        constraints_dual.append(q <= q_max)
    if q_min is not None:
        constraints_dual.append(q >= q_min)
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve()
    return prob_dual.value, q.value, {m: c.value for m,c in c_s.items()}

def weighted_sum_rate(weights):
    return lambda r: weights @ r


def proportional_fair(r):
    # return cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r))
    return cp.sum(cp.log(r))


def app_layer(weights):
    return (
        lambda r: cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r)) - weights @ r
    )


def optimize_app_phy(util, q_min, q_max, wsr_phy):

    assert util == proportional_fair
    n_users = len(q_min)
    assert len(q_max) == n_users
    A = np.minimum(q_max, q_min).reshape((n_users, 1))

    best_dual_value = np.inf
    for n in range(1, 1000):
        # create and solve the approximated problem
        approx_value, q, alpha, [la, w_min, w_max, mu] = time_sharing_cvx(
            util, A, q_min, q_max
        )

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
        LOGGER.info(
            f"Iteration {n} - Primal Approx {approx_value} - Dual Approx {dual_value} - Gap: {gap} "
        )
        if gap < 0.001:
            break
    return approx_value, q, alpha, [la, w_min, w_max, mu]