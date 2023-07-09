import logging
import numpy as np
import cvxpy as cp

from mcm.network import Network
from mcm.regions import Q_vector
from mcm.no_utils import InfeasibleOptimization

LOGGER = logging.getLogger(__name__)
def protocol7(util, Q: Q_vector, network: Network):
    network.create_init_mode(Q.q_min)
    # sum(mu_t) : mu_t <= F_t(f_t_i) - xi_t_i(f - f_t_i) \forall t \forall i, f \in F
    f_t = {m: 1 / len(network.modes) for m in network.modes}
    f_t_s = [f_t]

    d_f_n_s = []
    F_t_s = []
    max_util = -np.Inf
    for l in range(0, 100):
        converged = False
        for j in range(0, 100):
            # evaluate F(f)
            try:
                F = 0
                F_t_s_i = {}
                r = np.zeros(len(network.users))
                d_f = {}
                for t_id, t in network.transmitters.items():
                    F_t, r_t, alpha_t, c_m, [d_f_t_m, la_m_t, la] = t.F_t_R_approx(f_t, util, Q.q_min[t.users], Q.q_max[t.users])
                    d_f[t_id] = {m: la @ c for m,c in c_m.items()}
                    F += F_t
                    F_t_s_i[t_id] = F_t
                    for user, rate in r_t.items():
                        r[user] += rate

            except InfeasibleOptimization:
                # we hit an infeasible resource allocation,
                # unless we find a better method, we pick a random resource allocation
                # which may improve our approximation
                a = np.random.random(len(f_t))
                f_t_rand = {m: a_i / a.sum() for m,a_i in zip(f_t.keys(), a) }
                LOGGER.info(f"infeasible resource allocation {f_t}, try random next {f_t_rand}")
                f_t = f_t_rand

                continue

            max_util = max(max_util, F)
            # evalute approximation
            d_f_n_s.append(d_f)
            F_t_s.append(F_t_s_i)
            approx, f_new = network.resource_allocation(f_t_s, d_f_n_s, F_t_s)
            f_t_s.append(f_new)
            f_t = f_new
            LOGGER.info(
                f"Explicit: Iteration {j} - Approximation {approx} - Best Primal Value {max_util}"
            )
            if approx - max_util <= 1e-3:
                converged = True
                break


        w_m_t_s = {m: {} for m in network.modes}
        for transmitter_id, transmitter in network.transmitters.items():
            for mode in transmitter.modes:
                # if the resource allocation did not converge, for example because it is infeasible
                # we pick random weights for the update
                if not converged:
                    weights = np.random.random(len(transmitter.users_per_mode[mode]))
                else:
                    weights = la_m_t[mode][transmitter_id]
                val, _ = transmitter.wsr(weights, mode)
                w_m_t_s[mode][transmitter_id] = val

        v_phy = max([sum(v.values()) for v in w_m_t_s.values()])

    return F, r, None, None

