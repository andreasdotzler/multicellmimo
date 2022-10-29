import cvxpy as cp
import numpy as np

from typing import Callable
from mcm.timesharing import F_t_R_approx
from mcm.network_optimization import U_Q_conj
from mcm.regions import Q_vector, R_m_t, R_m_t_approx


class Transmitter:
    def __init__(
        self, R_m_t_s: dict[str, R_m_t], id=None, util=None, Q: Q_vector = None
    ):
        self.id = id
        self.users_per_mode: dict[str, list[int]] = {
            m: R.users for m, R in R_m_t_s.items()
        }
        any_users = next(iter(self.users_per_mode.values()))
        for users in self.users_per_mode.values():
            assert (
                users == any_users
            ), "Not implemented, if we want different users per mode, some things may break"
        self.users = any_users
        self.wsr_per_mode: dict[str, Callable[[np.array], (int, np.array)]] = {
            m: R.wsr for m, R in R_m_t_s.items()
        }

        self.modes = list(R_m_t_s.keys())
        self.R_m_t_s = R_m_t_s

        self.util = util
        assert Q is None or len(Q) == len(self.users)
        self.Q = Q

        self.weights = np.ones(len(self.users))
        self.weights_per_mode = {}
        self.average_transmit_rate = None
        self.iteration = 0
        self.best_dual_value = np.Inf
        self.f_t = None
        self.alphas = None
        self.q = None
        self.c_m_t_s = None

    @property
    def As_per_mode(self):
        return {m: R.approx.A for m, R in self.R_m_t_s.items()}

    @As_per_mode.setter
    def As_per_mode(self, value):
        raise RunTimeError("Do not use this ")

    def set_approximation(self, mode, A):
        self.R_m_t_s[mode].approx = R_m_t_approx(self.users, A)

    def reset_approximations(self):
        for R in self.R_m_t_s.values():
            R.reset_approximation()

    def wsr(self, t_weights, mode):
        if t_weights is None:
            t_weights = self.weights
        self.wsr_per_mode[mode](t_weights)
        return self.R_m_t_s[mode].wsr(t_weights)

    def update_weights(self, m_opt):
        if m_opt in self.R_m_t_s:
            c_t_l_1 = self.R_m_t_s[m_opt].approx.A[:, -1]
        else:
            c_t_l_1 = np.zeros_like(self.weights)

        v_phy = self.weights @ c_t_l_1
        v_app, q_t_l_1 = U_Q_conj(self.util, self.weights, self.Q)
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
        R_m = {m: R.approx for m, R in self.R_m_t_s.items()}
        (F_t, r_t) = F_t_R_approx(util, fractions, self.users, R_m, Q)

        alpha_t = {mode: R.alphas.value for mode, R in R_m.items()}
        c_m = {mode: R.c.value for mode, R in R_m.items()}
        d_f_t_m = {mode: R.sum_alpha.dual_value for mode, R in R_m.items()}
        d_c_m = {mode: R.r_in_A_x_alpha.dual_value for mode, R in R_m.items()}

        # d_c_m = f_m / (sum_m f_m c_m) -> d_c_m = [0,..,0] if f_m == 0
        for m in d_c_m:
            if fractions[m] <= 1e-3:
                d_c_m[m] = np.zeros_like(d_c_m[m])
        la = 1 / sum([fractions[m] * c for m, c, in c_m.items()])
        self.weights_per_mode = d_c_m
        return F_t, r_t, alpha_t, c_m, [d_f_t_m, d_c_m, la]
