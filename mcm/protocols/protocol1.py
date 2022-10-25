import logging
import numpy as np

from mcm.network import Network
from mcm.regions import Q_vector

LOGGER = logging.getLogger(__name__)


def protocol1(util, Q: Q_vector, network: Network):

    best_dual_value = np.Inf
    for transmitter in network.transmitters.values():
        transmitter.util = util
        transmitter.Q = Q[transmitter.users]
    for n in range(1, 1000):
        # wsr_per_mode_and_transmitter
        w_m_t_s = {m: {} for m in network.modes}
        for transmitter_id, transmitter in network.transmitters.items():
            for mode in transmitter.modes:
                val, rates = transmitter.wsr(transmitter.weights, mode)
                w_m_t_s[mode][transmitter_id] = val

        w_m = {m: sum(w_m_t.values()) for m, w_m_t in w_m_t_s.items()}
        m_opt, _ = max(w_m.items(), key=lambda k: k[1])

        dual_value = 0
        primal_value = 0
        for transmitter in network.transmitters.values():
            p, d = transmitter.update_weights(m_opt)
            dual_value += d
            primal_value += p
        assert dual_value >= primal_value
        best_dual_value = min(best_dual_value, dual_value)
        gap = (best_dual_value - primal_value) / abs(best_dual_value)
        LOGGER.info(
            f"Network: Iterabtion {n} - mopt {m_opt} - primal value {primal_value} - dual value {dual_value} - gap {gap}"
        )

        if gap <= 0.01:
            break
    rates = np.zeros(len(Q))
    for transmitter in network.transmitters.values():
        rates[transmitter.users] += transmitter.average_transmit_rate
    return sum(np.log(rates)), rates, None, None
