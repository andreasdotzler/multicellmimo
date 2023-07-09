import numpy as np
import cvxpy as cp

from typing import Any, Callable, Optional, Tuple, cast
from mcm.timesharing import F_t_R_approx
from mcm.network_optimization import U_Q_conj
from mcm.regions import Q_vector, R_m_t, R_m_t_approx
from mcm.my_typing import Fractions, Weights, Util_cvx


class Transmitter:
    def __init__(
            self, R_m_t_s: dict[str, R_m_t], id: int, util: Optional[Util_cvx] = None, Q: Optional[Q_vector] = None):
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
        self.wsr_per_mode: dict[str, Callable[[Weights], Tuple[float, np.ndarray]]] = {
            m: R.wsr for m, R in R_m_t_s.items()
        }

        self.modes = list(R_m_t_s.keys())
        self.R_m_t_s = R_m_t_s
        if util is None:
            def const_minus_inf(r: cp.Variable) -> cp.Expression:
                return cast(cp.Expression, - np.Inf)
            util = const_minus_inf
        self.util = util
        assert Q is None or len(Q) == len(self.users)
        if Q is None:
            q_min = np.ones(len(self.users)) * - np.Inf
            q_max = np.ones(len(self.users)) * np.Inf
            Q = Q_vector(q_min=q_min, q_max=q_max)
        self.Q = Q

        self.weights: Weights = np.ones(len(self.users))
        self.weights_per_mode: dict[str, Weights] = {}
        self.average_transmit_rate: np.ndarray
        self.iteration = 0
        self.best_dual_value = np.Inf
        self.f_t: Fractions
        self.alphas: np.ndarray
        self.q: np.ndarray

    @property
    def As_per_mode(self) -> dict[str, np.ndarray]:
        return {m: R.approx.A for m, R in self.R_m_t_s.items()}

    @As_per_mode.setter
    def As_per_mode(self, value: Any) -> None:
        raise RuntimeError("Do not use this")

    def set_approximation(self, mode: str, A: np.ndarray) -> None:
        self.R_m_t_s[mode].approx = R_m_t_approx(self.users, A)

    def reset_approximations(self) -> None:
        for R in self.R_m_t_s.values():
            R.reset_approximation()

    def wsr(self, t_weights: Optional[Weights], mode: str) -> Tuple[float, np.ndarray]:
        if t_weights is None:
            t_weights = self.weights
        self.wsr_per_mode[mode](t_weights)
        return self.R_m_t_s[mode].wsr(t_weights)

    def update_weights(self, m_opt: str) -> Tuple[float, float]:
        if m_opt in self.R_m_t_s:
            c_t_l_1 = self.R_m_t_s[m_opt].approx.A[:, -1]
        else:
            c_t_l_1 = np.zeros_like(self.weights)

        v_phy = float(self.weights @ c_t_l_1)
        v_app, q_t_l_1 = U_Q_conj(self.util, self.weights, self.Q)
        ell = self.iteration
        self.weights -= 1 / (ell + 1) * (c_t_l_1 - q_t_l_1)
        # self.weights -= c_t_l_1 - q_t_l_1
        if ell == 0:
            self.average_transmit_rate = c_t_l_1
        else:
            self.average_transmit_rate = (
                ell / (ell + 1) * self.average_transmit_rate + 1 / (ell + 1) * c_t_l_1
            )

        primal_value = self.util(cp.Variable(value=self.average_transmit_rate)).value
        self.iteration += 1

        return primal_value, v_app + v_phy

    def F_t_R_approx(self, fractions: Fractions, util: Util_cvx, Q: Q_vector) -> Tuple[float, dict[int, float], dict[str, np.ndarray], dict[str, np.ndarray], Tuple[dict[str, np.ndarray], dict[str, np.ndarray], float] ]:
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
        return F_t, r_t, alpha_t, c_m, (d_f_t_m, d_c_m, la)
