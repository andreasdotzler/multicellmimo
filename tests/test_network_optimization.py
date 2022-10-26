import logging


import numpy as np
import pytest

from tests.utils import gen_test_network

from mcm.protocols.protocol1 import protocol1
from mcm.protocols.protocol2 import protocol2
from mcm.protocols.protocol3 import protocol3
from mcm.protocols.protocol4 import protocol4
from mcm.protocols.protocol5 import protocol5
from mcm.protocols.protocol6 import protocol6

from mcm.timesharing import timesharing_network
from mcm.network_optimization import proportional_fair
from mcm.regions import Q_vector
from mcm.network import Network

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "algorithm",
    [
        protocol1,
        protocol2,
        protocol3,
        protocol4,
        protocol5,
        protocol6,
    ],
)
@pytest.mark.parametrize(
    "As, network", [gen_test_network(), gen_test_network(20, np.random.random)]
)
def test_global_network(As, network: Network, algorithm, seed):
    q_min = np.array([0.01] * 30)
    # q_min[0] = 0.5
    q_max = np.array([10.0] * 30)
    Q = Q_vector(q_min=q_min, q_max=q_max)
    # q_max[29] = 0.15
    network.initialize_approximation(As)
    (
        value,
        rates,
         _
    ) = timesharing_network(proportional_fair, network, Q)

    assert all(rates >= q_min * 0.97)
    assert all(rates * 0.97 <= q_max)
    verfiy_fractional_schedule(network.alphas_m_t)
    LOGGER.info(f"expected result: {value}")
    network.reset_approximation()
    # scheduling works on the approximations, let us assume here the approximation
    # is the full timesharing
    if algorithm in [protocol3, protocol4, protocol6]:
        network.initialize_approximation(As)
    opt_value_explicit, opt_q_explicit, _, _ = algorithm(
        proportional_fair, Q, network
    )
    assert opt_value_explicit == pytest.approx(value, 1e-2)
    assert opt_q_explicit == pytest.approx(rates, rel=1e-1, abs=1e-1)


def verfiy_fractional_schedule(alphas):
    total_time = 0
    for mode, alphas_per_mode in alphas.items():
        sum_per_transmitter_and_mode = []
        for alpha in alphas_per_mode.values():
            sum_per_transmitter_and_mode.append(sum(alpha))
        mean_sum = np.mean(sum_per_transmitter_and_mode)
        LOGGER.info(f"Mode {mode} - Fraction {mean_sum}")
        assert sum_per_transmitter_and_mode == pytest.approx(
            np.ones(len(sum_per_transmitter_and_mode)) * mean_sum, rel=1e-2, abs=1e-2
        )
        total_time += mean_sum
    assert total_time == pytest.approx(1, 1e-2)
