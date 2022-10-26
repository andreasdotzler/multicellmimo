import numpy as np
from mcm.network_optimization import I_C
from mcm.network import Network
from mcm.transmitter import Transmitter
from mcm.regions import R_m_t


def gen_test_network(n_rate_points_per_mode_and_transmitter=1, sample_function=np.ones):
    users_per_mode_and_transmitter = {
        "reuse1": {
            0: list(range(0, 10)),
            1: list(range(10, 20)),
            2: list(range(20, 30)),
        },
        "reuse3-0": {0: list(range(0, 10))},
        "reuse3-1": {1: list(range(10, 20))},
        "reuse3-2": {2: list(range(20, 30))},
    }
    users_per_transmitter = {
        0: list(range(0, 10)),
        1: list(range(10, 20)),
        2: list(range(20, 30)),
    }

    As = {}
    transmitters = {}
    wsr_transmitter_mode = {}
    users_transmitter_mode = {}
    for mode, transmitters_and_users in users_per_mode_and_transmitter.items():
        Am = As[mode] = {}
        for t, users in transmitters_and_users.items():
            Am[t] = sample_function(
                (len(users), n_rate_points_per_mode_and_transmitter)
            )
            if mode == "reuse1":
                Am[t] = Am[t] * 1 / 3
            Am[t] = Am[t] * 2

            if t not in wsr_transmitter_mode:
                wsr_transmitter_mode[t] = {}
                users_transmitter_mode[t] = {}
            wsr_transmitter_mode[t][mode] = I_C(Am[t])
            users_transmitter_mode[t][mode] = users

    for t in wsr_transmitter_mode:
        R_m_t_s = {}
        users = users_transmitter_mode[t]
        wsrs = wsr_transmitter_mode[t]
        for m in wsr_transmitter_mode[t]:
            R_m_t_s = {m: R_m_t(users[m], wsrs[m]) for m in wsrs}

        transmitters[t] = Transmitter(R_m_t_s, t)
    return As, Network(transmitters)
