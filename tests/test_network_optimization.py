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
from mcm.no_utils import fractions_from_schedule

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
    (value, rates, _) = timesharing_network(proportional_fair, network, Q)

    assert all(rates >= q_min * 0.97)
    assert all(rates * 0.97 <= q_max)
    f = fractions_from_schedule(network.alphas_m_t)
    assert sum(f.values()) == pytest.approx(1, 1e-3)
    LOGGER.info(f"expected result: {value}")
    network.reset_approximation()
    # scheduling works on the approximations, let us assume here the approximation
    # is the full timesharing
    if algorithm in [protocol3, protocol4, protocol6]:
        network.initialize_approximation(As)
    opt_value_explicit, opt_q_explicit, _, _ = algorithm(proportional_fair, Q, network)
    assert opt_value_explicit == pytest.approx(value, 1e-2)
    assert opt_q_explicit == pytest.approx(rates, rel=1e-1, abs=1e-1)
