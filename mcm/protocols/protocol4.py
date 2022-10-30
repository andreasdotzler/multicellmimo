import logging
import numpy as np

from mcm.network import Network
from mcm.no_utils import InfeasibleOptimization
from mcm.regions import Q_vector

LOGGER = logging.getLogger(__name__)


def protocol4(util, Q: Q_vector, network: Network):
    modes = network.modes
    # sum(mu_t) : mu_t <= F_t(f_t_i) - xi_t_i(f - f_t_i) \forall t \forall i, f \in F
    f_equal = 1 / len(network.modes)
    f_t = [{m: f_equal for m in network.modes}]
    d_f_n_s = []
    F_t_s = []
    max_util = -np.Inf
    for j in range(0, 100):
        # evaluate F(f)
        for n in range(0, 100):
            try:
                v_n, r_n, alphas_n, d_f_n, F_t = network.F_t_R_appprox(f_t[j], util, Q)
                break
            except InfeasibleOptimization:
                # we hit an infeasible resource allocation,
                # unless we find a better method, we pick a random resource allocation
                # which may improve our approximation
                a = np.random.random(len(f_t[j]))
                f_t[j] = {m: a_i / a.sum() for m, a_i in zip(f_t[j].keys(), a)}
                LOGGER.info(f"infeasible {n}, try random next")

        max_util = max(max_util, v_n)
        # evalute approximation
        d_f_n_s.append(d_f_n)
        F_t_s.append(F_t)
        approx, f_new = network.resource_allocation(f_t, d_f_n_s, F_t_s)
        f_t.append(f_new)
        assert approx >= max_util
        gap = (approx - max_util) / abs(approx)
        LOGGER.info(
            f"Explicit: Iteration {j} - Approximation {approx} - Best Primal Value {max_util} - gap {gap}"
        )

        if gap <= 1e-3:
            break
    return v_n, r_n, None, None
