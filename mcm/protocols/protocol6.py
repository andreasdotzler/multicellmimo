import logging
import numpy as np
import pytest

from mcm.network_optimization import dual_problem_app_f
from mcm.regions import Q_vector
from mcm.network import Network
from mcm.no_utils import InfeasibleOptimization

LOGGER = logging.getLogger(__name__)


def protocol6(util, Q: Q_vector, network: Network):
    network.create_init_mode(Q.q_min)
    # sum(mu_t) : mu_t <= F_t(f_t_i) - xi_t_i(f - f_t_i) \forall t \forall i, f \in F
    f_t = {m: 0 for m in network.modes}
    f_t["reuse1"] = 1
    f_t_s = [f_t]
    # f_t[0]['reuse1'] = 0.4
    d_f_n_s = []
    F_t_s = []
    max_util = -np.Inf
    for j in range(0, 100):
        # evaluate F(f)
        try:
            F = 0
            F_t_s_i = {}
            r = np.zeros(len(network.users))
            d_f = {}
            for t_id, t in network.transmitters.items():
                # TODO until converged

                for l in range(100):
                    # Solve approximated problem
                    F_t, r_t, alpha_t, c_m, [d_f_t_m, d_c_m, la] = t.scheduling(f_t, util, Q[t.users])
                    # solve the dual problem to provide bound and update
                    v_phy = 0
                    assert 1 / np.array(list(r_t.values())) == pytest.approx(la, 1e-3)
                    for mode in t.modes:
                        v, _ = t.wsr(t_weights=la, mode=mode)
                        v_phy += f_t[mode] * v
                    # TODO we do not want this one, let us explictly calculate
                    v_app, q, c = dual_problem_app_f(util, d_c_m, f_t, Q[t.users])
                    dual_value = v_app + v_phy
                    gap = abs(dual_value - F_t) / abs(dual_value)
                    LOGGER.info(
                        f"\t transmitter {t_id} - primal {F_t} - dual {dual_value} - gap {gap}"
                    )
                    if gap < 0.0001:
                        break
                d_f[t_id] = {m: la @ c for m, c in c_m.items()}
                # d_f[t_id] = {m: max(la @ t.As_per_mode[m]) for m in t.modes}
                F += F_t
                F_t_s_i[t_id] = F_t
                for user, rate in r_t.items():
                    r[user] += rate

        except InfeasibleOptimization:
            # we hit an infeasible resource allocation,
            # unless we find a better method, we pick a random resource allocation
            # which may improve our approximation
            a = np.random.random(len(f_t))
            f_t_rand = {m: a_i / a.sum() for m, a_i in zip(f_t.keys(), a)}
            LOGGER.info(
                f"infeasible resource allocation {f_t}, try random next {f_t_rand}"
            )
            f_t_s[-1] = f_t = f_t_rand
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
        if abs(approx - max_util) / max_util <= 1e-6:
            break
    return F, r, None, None
