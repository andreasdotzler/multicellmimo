import numpy as np
import pytest
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
from mcm.regions import Q_vector, R_m_t_approx


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
def test_algorithms(A, optimize):
    n_users = A.shape[0]
    q_min = np.array([0.1] * n_users)
    q_max = np.array([10.0] * n_users)
    # q_max[0] = 0.2
    n_users, n = A.shape
    R = R_m_t_approx(list(range(0, n_users)), A)
    Q = Q_vector(q_min=q_min, q_max=q_max)
    value, rates = time_sharing_cvx(proportional_fair, R, Q)
    # TODO enforce all algorithms to return schedules
    opt_value, q, _ = optimize(A, q_min, q_max, target=value)
    assert opt_value == pytest.approx(value, 1e-2)
    # check if rates are feasible
    Q = Q_vector(q_min=q*0.95, q_max=q)    
    time_sharing_cvx(proportional_fair, R, Q)
    alpha_check = R.alphas
    assert sum(alpha_check) == pytest.approx(1, 1e-6)
    assert sum(np.log(A @ alpha_check)) == pytest.approx(value, 1e-2)
