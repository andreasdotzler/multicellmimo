import cvxpy as cp
import numpy as np

from typing import Callable
from mcm.timesharing import timesharing_fixed_fractions
from mcm.network_optimization import U_Q_conj
from mcm.regions import Q_vector

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
        self.weights_per_mode = {}
        self.average_transmit_rate = None
        self.iteration = 0
        self.best_dual_value = np.Inf
        self.f_t = None
        self.alphas = None
        self.q = None
        self.c_m_t_s = None

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
        v_app, q_t_l_1 = U_Q_conj(
            self.util, self.weights, Q_vector(self.q_min, self.q_max)
        )
        l = self.iteration
        self.weights -= 1 / (l + 1) * (c_t_l_1 - q_t_l_1)
        # self.weights -= c_t_l_1 - q_t_l_1
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

    def scheduling(self, fractions, util, Q: Q_vector):
        self.f_t = fractions
        (F_t, r_t, alpha_t, c_m, [d_f_t_m, d_c_m]) = timesharing_fixed_fractions(
            util, fractions, self.users, self.As_per_mode, Q
        )
        # d_c_m = f_m / (sum_m f_m c_m) -> d_c_m = [0,..,0] if f_m == 0
        for m in d_c_m:
            if fractions[m] <= 1e-3:
                d_c_m[m] = np.zeros_like(d_c_m[m])
        la = 1 / sum([fractions[m] * c for m, c, in c_m.items()])
        self.weights_per_mode = d_c_m
        return F_t, r_t, alpha_t, c_m, [d_f_t_m, d_c_m, la]