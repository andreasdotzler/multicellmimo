import logging
import numpy as np
import pytest

from mcm.network import Network
from mcm.timesharing import timesharing_network
from mcm.regions import Q_vector

LOGGER = logging.getLogger(__name__)


def protocol5(util, Q: Q_vector, network: Network):

    n_users = len(Q)
    assert len(Q) == n_users
    network.create_init_mode(Q.q_min)

    for n in range(1, 1000):
        approx_value, _, alphas, [weights, _, _, _, _, la_m_t] = timesharing_network(util, network, Q)
        # TODO dual_app
        q_app = np.minimum(Q.q_max, np.maximum(Q.q_min, 1 / weights))
        q_app[weights <= 0] = Q.q_max[weights <= 0]
        v_app_1 = sum(np.log(q_app)) - weights @ q_app
        # Q = Q_vector(q_min = q_min, q_max = q_max)
        # v_app, q = V_conj(network, util, la_m_t, Q)
        # assert v_app == pytest.approx(v_app_1)

        w_m_t_s = {m: {} for m in network.modes}
        for transmitter_id, transmitter in network.transmitters.items():
            for mode in transmitter.modes:
                val, _ = transmitter.wsr(la_m_t[mode][transmitter_id], mode)
                w_m_t_s[mode][transmitter_id] = val

        v_phy = max([sum(v.values()) for v in w_m_t_s.values()])

        dual_value = v_app_1 + v_phy
        LOGGER.info(
            f"Network: Iterabtion {n} - Dual Approximation {approx_value} - Dual Value {dual_value}"
        )
        if abs(dual_value - approx_value) < 0.001:
            break

    return approx_value, q_app, alphas, None
