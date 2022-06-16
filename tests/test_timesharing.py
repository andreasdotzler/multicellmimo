import numpy as np
import pytest
from mcm.algorithms import (optimize_primal_column,
                            optimize_primal_subgradient_projected,                            
                            optimize_dual_decomp_subgradient,
                            optimize_dual_cuttingplane,
                            optimize_dual_multicut,
                            optimize_dual_multicut_peruser,
                            optimize_app_phy_rateregionapprox,
                            optimize_primal_subgradient_rosen)
from mcm.network_optimization import (I_C, I_C_Q, dual_problem_app,
                                      optimize_app_phy,                                      
                                      proportional_fair, time_sharing,
                                      time_sharing_cvx, weighted_sum_rate)
from mcm.utils import InfeasibleOptimization


def test_timesharing_wsr():
    A = np.array([[4, 1], [1, 2]])
    q_min = np.array([0, 0])
    q_max = np.array([10, 10])

    weights = [1, 0]
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([4, 1], 1e-3)
    weights = [0, 1]
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([1, 2], 1e-3)
    weights = [1, 1]
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([4,1], 1e-3)
    q_min = np.array([0, 2])
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([1,2], 1e-3)
    q_min = np.array([0, 0])
    q_max = np.array([2, 2])
    value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(weights), A, q_min, q_max)
    assert rates.tolist() == pytest.approx([2,1 + 2/3])
    q_min = np.array([0, 0])
    q_max = np.array([0.5, 0])
    with pytest.raises(InfeasibleOptimization):
        time_sharing(weighted_sum_rate(weights), A, q_min, q_max)



#@pytest.mark.parametrize("A", [np.array([[0.4, 0.1], [0.1, 0.2]]).T, np.random.rand(10,200)*3])
@pytest.mark.parametrize("optimize", [optimize_primal_column, 
                                      #optimize_primal_sub, 
                                      optimize_primal_subgradient_projected, 
                                      optimize_primal_subgradient_rosen,
                                      optimize_dual_decomp_subgradient,
                                      optimize_dual_cuttingplane,
                                      optimize_dual_multicut,
                                      optimize_dual_multicut_peruser,
                                      optimize_app_phy_rateregionapprox,
                                      ])
def test_primal_sub(A, optimize):
    n_users = A.shape[0]
    q_min = np.array([0.1]*n_users)
    q_max = np.array([10.0]*n_users)
    #q_max[0] = 0.2

    value, rates, alpha, [lambda_opt, w_min, w_max, mu] = time_sharing_cvx(proportional_fair, A, q_min, q_max)
    opt_value_primal_sub, opt_q_primal_sub, _ = optimize(A, q_min, q_max, target=value)
    assert opt_value_primal_sub == pytest.approx(value, 1e-2)
    assert opt_q_primal_sub == pytest.approx(rates, rel=1e-1, abs=1e-1)




@pytest.fixture(scope="function")
def seed():
    np.random.seed(41)

@pytest.fixture(scope="function", params=[[2,2], [10,40]])
def A(request, seed):
    if request.param == [2,2]:
        return np.array([[1, 0.0], [0.0, 1]]).T
    else:
        return np.random.rand(*request.param)*3

def test_timesharing_fair(A):

    n_users = A.shape[0]
    q_min = np.array([0.1]*n_users)
    q_max = np.array([10.0]*n_users)
    # todo this currently breaks the test 
    q_max[0] = 0.2

    value, rates, alpha, [lambda_opt, w_min, w_max, mu] = time_sharing_cvx(proportional_fair, A, q_min, q_max)
    
    # verifiy KKT
    assert 1/rates + w_min - w_max - lambda_opt == pytest.approx(np.zeros(len(rates)), rel=1e-2, abs=1e-1)
    assert all(lambda_opt @ (A - rates.reshape(len(rates), 1)) <= 0.001)
    q_app = np.minimum(q_max, np.maximum(q_min, 1 / lambda_opt))
    q_app[lambda_opt <= 0] = q_max[lambda_opt <= 0]
    
    # todo assert mu = max((weights @ A).tolist()[0])
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    

    dual_value_app, q_app = dual_problem_app(proportional_fair, lambda_opt, q_max, q_min)
    assert dual_value_app == pytest.approx(sum(np.log(q_app)) - lambda_opt@q_app)
    assert rates == pytest.approx(q_app, rel=1e-3, abs=1e-1)

    rates_phy = dual_problem_phy(A, lambda_opt)
    dual_value_phy = lambda_opt @ rates
    assert lambda_opt @ rates_phy == pytest.approx(dual_value_phy)
    assert value == pytest.approx(dual_value_app + dual_value_phy)

    # now by iterative optimization
    wsr = I_C(A)
    opt_value, opt_q, _, _ = optimize_app_phy(proportional_fair, q_min, q_max, wsr)
    assert value == pytest.approx(opt_value, rel=1e-3, abs=1e-1)
    assert rates == pytest.approx(opt_q, rel=1e-3, abs=1e-1)



  


def dual_problem_phy(A, weights):
    max_i = np.argmax(weights@A)
    rates = A[:,max_i]
    return rates


