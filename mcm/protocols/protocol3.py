import logging
import numpy as np
import cvxpy as cp

from mcm.no_utils import InfeasibleOptimization
from mcm.network import Network
from mcm.regions import Q_vector

LOGGER = logging.getLogger(__name__)


def protocol3(util, Q: Q_vector, network: Network):
    f_equal = 1 / len(network.modes)
    f_t = {m: f_equal for m in network.modes}
    max_util = -np.Inf
    for j in range(0, 100):
        # evaluate F(f)
        v_n, r_n, alphas_n, d_f_t_m, F_t = network.scheduling(f_t, util, Q)

        # evalute approximation
        # project gradient

        d_f = {m: 0 for m in network.modes}
        for t, d_f_m in d_f_t_m.items():
            for m, d_f_i in d_f_m.items():
                d_f[m] += d_f_i
        sum_d_f = sum(d_f.values())
        d_f = {m: v / sum_d_f for m, v in d_f.items()}

        f_t_new = {m: cp.Variable(1, nonneg=True) for m in network.modes}

        for n in range(100):
            u = [
                cp.sum_squares(f_t_new[m] - (f_t[m] + 1 / (n + 1) * d_f[m]))
                for m in network.modes
            ]
            prob = cp.Problem(cp.Minimize(cp.sum(u)), [sum(f_t_new.values()) == 1])
            prob.solve()
            try:
                f_temp = {m: f.value for m, f in f_t_new.items()}
                network.scheduling(f_temp, util, Q)
                break
            except InfeasibleOptimization:
                LOGGER.info(f"Infeasible for {f_temp}")
                
        f_t = {m: f.value for m, f in f_t_new.items()}

        improvement = v_n - max_util
        max_util = max(v_n, max_util)
        LOGGER.info(
            f"Explicit: Iteration {j} - Best Primal Value {max_util} - Improvement {improvement}"
        )
        assert prob.status == "optimal"
        # solve

        if abs(improvement) <= 1e-6:
            break
    return v_n, r_n, None, None
