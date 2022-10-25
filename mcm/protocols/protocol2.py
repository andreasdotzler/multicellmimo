from mcm.network_optimization import optimize_app_phy
from mcm.regions import Q_vector
from mcm.network import Network

def protocol2(util, Q: Q_vector, network: Network):
    return optimize_app_phy(util, Q, network.wsr_per_mode)
