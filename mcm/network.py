import logging

import cvxpy as cp
import numpy as np

from mcm.network_optimization import (
    I_C,
    I_C_Q,
    U_Q_conj,
    optimize_app_phy,
    proportional_fair,
)
