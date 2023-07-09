import numpy as np
import pytest
from typing import Callable, Optional
from mcm.algorithms import (
    optimize_primal_column,
    optimize_primal_subgradient_projected,
    optimize_dual_decomp_subgradient,
    optimize_dual_cuttingplane,
    optimize_dual_multicut,
    optimize_dual_multicut_peruser,
    optimize_app_phy_rateregionapprox,
    optimize_primal_subgradient_rosen,
)
from mcm.timesharing import time_sharing_cvx
from mcm.network_optimization import proportional_fair
from mcm.regions import Q_vector_bounded, R_m_t_approx
from mcm.my_typing import Algorithm_Result

@pytest.mark.parametrize(
    "optimize",
    [
        optimize_primal_column,
        # optimize_primal_sub,
        optimize_primal_subgradient_projected,
        optimize_primal_subgradient_rosen,
        optimize_dual_decomp_subgradient,
        optimize_dual_cuttingplane,
        optimize_dual_multicut,
        optimize_dual_multicut_peruser,
        optimize_app_phy_rateregionapprox,
    ],
)
def test_algorithms(A: np.ndarray, optimize: Callable[[np.ndarray, Q_vector_bounded, Optional[float]], Algorithm_Result]) -> None:
    n_users = A.shape[0]
    q_min = np.array([0.1] * n_users)
    q_max = np.array([10.0] * n_users)
    # q_max[0] = 0.2
    n_users, n = A.shape
    R = R_m_t_approx(list(range(0, n_users)), A)
    Q = Q_vector_bounded(q_min=q_min, q_max=q_max)
    value, rates = time_sharing_cvx(proportional_fair, R, Q)
    # TODO enforce all algorithms to return schedules
    opt_value, q = optimize(A, Q, value)
    assert opt_value == pytest.approx(value, 1e-2)
    assert rates in Q
    assert rates in R
