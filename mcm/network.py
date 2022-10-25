import cvxpy as cp
import numpy as np

from typing import Optional, Tuple, Any, Dict
from mcm.transmitter import Transmitter
from mcm.typing import A_m_t, Fractions, V_m_t, Weights
from mcm.regions import Q_vector


class Network:
    def __init__(self, transmitters: dict[int, Transmitter]):

        self.transmitters: dict[int, Transmitter] = transmitters
        self.users = []
        self.modes = []
        for t in transmitters.values():
            self.users += t.users
            self.modes += t.modes
        self.users = list(set(self.users))
        self.modes = list(set(self.modes))

    def initialize_approximation(self, As: A_m_t) -> None:
        for mode, trans_and_At in As.items():
            for trans, At in trans_and_At.items():
                self.transmitters[trans].As_per_mode[mode] = At

    def reset_approximation(self) -> None:
        for t in self.transmitters.values():
            t.As_per_mode = {}

    def wsr_per_mode(self, weights: Weights) -> Tuple[int, np.ndarray]:
        max_value = -np.Inf
        mode_values, A_m = self.wsr_per_mode_and_transmitter(weights)
        w_m = {m: sum(w_m_t.values()) for m, w_m_t in mode_values.items()}
        max_mode, max_value = max(w_m.items(), key=lambda k: k[1])
        max_rates = np.zeros(len(weights))
        for transmitter_id, rates in A_m[max_mode].items():
            max_rates[self.transmitters[transmitter_id].users] += rates
        assert len(weights) == len(max_rates)
        return max_value, max_rates

    def wsr_per_mode_and_transmitter(self, weights: Optional[Weights] = None) -> Tuple[V_m_t, A_m_t]:
        values: V_m_t = {}
        A_max: A_m_t = {}

        test: Dict[int, str] = {}
        test[1] = "dwd"
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

    def scheduling(self, fractions: Fractions, util: Any, Q: Q_vector) -> Tuple[float, np.ndarray, V_m_t, dict[int, dict[str, float]], dict[int, float]]:
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
            ) = t.scheduling(fractions, util, Q[t.users])
            for mode, a in alpha_t.items():
                if mode not in alphas:
                    alphas[mode] = {}
                alphas[mode][transmitter_id] = a
            # for mode, d in d_f_t_m.items():
            #    if mode not in d_f:
            #        d_f[mode] = {}
            d_f[transmitter_id] = {m: la @ c for m, c in c_m.items()}
            F += F_t
            F_t_s[transmitter_id] = F_t
            for user, rate in r_t.items():
                r[user] += rate
        return F, r, alphas, d_f, F_t_s

    def create_init_mode(self, q_min) -> None:
        # we need to initialize a fake mode that is all q_min
        for transmitter in self.transmitters.values():
            # unique_users = list(set(transmitter.users))
            for mode in transmitter.modes:
                a = q_min[transmitter.users_per_mode[mode]]
                transmitter.As_per_mode[mode] = a.reshape(len(a), 1)

    def get_As(self) -> A_m_t:
        As: A_m_t = {m: {} for m in self.modes}
        for t_id, t in self.transmitters.items():
            for mode in t.modes:
                As[mode][t_id] = t.As_per_mode[mode]
        return As

    def resource_allocation(self, f_t, d_f_n_s, F_t_s) -> Tuple[int, float]:
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