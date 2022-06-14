import cvxpy as cp

import numpy as np
import  logging

from mcm.network_optimization import I_C_Q, optimize_app_phy, proportional_fair

LOGGER = logging.getLogger(__name__)

def optimize_primal_sub(A, q_min, q_max, target=None):
    # direct optimization of the problem in the primal domain via subgradient
    # unconstrained_primal_sub
    n_users = len(q_min)
    assert len(q_max) == n_users

    # add the origin to the points
    A = np.c_[np.zeros(n_users), A]
    _, n_schedules = A.shape

    rates = q_min
  
    max_util = sum(np.log(rates))
    for n in range(1000):

        step_size = 1
        subgradient = cp.Variable(n_users)
        d_U = cp.Variable(n_users)
        d_Q = cp.Variable(n_users)
        d_C = cp.Variable(n_users)
        
        scale = cp.Variable(1, nonneg=True)        
        c_d_U = [d_U == scale*1/rates]
        
        c_d_Q = []
        for r_i, q_min_i, q_max_i, d_Q_i in zip(rates, q_min, q_max, d_Q):
            if abs(r_i - q_min_i) <= 10**-6:
                c_d_Q.append(d_Q_i >= 0)
            elif abs(r_i - q_max_i) <= 10**-6:
                c_d_Q.append(d_Q_i <= 0)
            else:
                c_d_Q.append(d_Q_i == 0)
        #c_d_Q = [d_Q@(q_min - rates) <=0, d_Q@(q_max - rates) <=0]
        c_d_C = []
        for a in range(n_schedules):
            c_d_C.append(d_C@(A[:,a] - rates) <= 0)
        sub_sum = [subgradient == d_U + d_Q - d_C]

        alpha = cp.Variable(n_schedules, nonneg=True)
        feasible_C = [rates + subgradient == A@alpha, cp.sum(alpha) <= 1]
        feasible_Q = [rates + subgradient >= q_min, rates + subgradient <= q_max]

        constraints =  c_d_U + c_d_Q + c_d_C + sub_sum + feasible_Q + feasible_C
        weights = np.random.normal(size=len(rates))
        prob1 = cp.Problem(cp.Maximize(weights @ subgradient), constraints)
        #prob1 = cp.Problem(cp.Minimize(cp.sum_squares(subgradient)), constraints)
        prob1.solve()
        assert 'optimal' in prob1.status
        step_size = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(rates + step_size * subgradient.value))),[step_size <=1, step_size>=0])
        prob.solve()
        rates = rates + step_size.value * subgradient.value 
        util = sum(np.log(rates))
        max_util = max(max_util, util)
        gap = (target - util) / abs(target)
        LOGGER.info(f"Iteration {n} - Util {util} - Gap - {gap}")
        
        if gap < 0.001:
            break
        #if subgradient.value @ subgradient.value < 0.00000000000001:
            #break

    return max_util, rates, None


def optimize_primal_subgradient_projected(A, q_min, q_max, target=None):

    rates = q_min
    n_users, n_schedules = A.shape
    step_size = 1
    max_util = sum(np.log(rates))
    for n in range(1000):
        LOGGER.info(f"Projected Subgradient: Iteration {n} - Best Primal Value {max_util}")
        subgradient = 1/rates

        r_l_1 = cp.Variable(n_users)   
        alpha = cp.Variable(n_schedules, nonneg=True)

        constraints = [r_l_1 == A@alpha, cp.sum(alpha) <= 1, r_l_1 >= q_min, r_l_1 <= q_max]  
        prob = cp.Problem(cp.Minimize(cp.sum_squares(r_l_1 - (rates + step_size * subgradient) )), constraints)
        prob.solve()
        rates = r_l_1.value
        new_util = sum(np.log(rates))
        if target is None and abs(max_util - new_util) < 1e-6:
            break
        max_util = max(max_util, new_util)
        if target is not None:
            gap = (target - max_util) / abs(target)
            if gap < 0.001:
                break        
    return max_util, rates, None


def optimize_primal_subgradient_rosen(A, q_min, q_max, target=None):

    rates = q_min
    n_users, n_schedules = A.shape
    step_size = 1
    max_util = sum(np.log(rates))
    for n in range(1000):
        LOGGER.info(f"Projected Subgradient: Iteration {n} - Best Primal Value {max_util}")
        subgradient = 1/rates

        p_s = cp.Variable(n_users)   
        alpha = cp.Variable(n_schedules, nonneg=True)
        constraints = [rates + p_s == A@alpha, cp.sum(alpha) <= 1, rates + p_s >= q_min, rates + p_s <= q_max]  
        #prob = cp.Problem(cp.Minimize(cp.sum_squares(r_l_1 - (rates + step_size * subgradient) )), constraints)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(p_s - subgradient )), constraints)
        prob.solve()
        #rates = r_l_1.value
        rates = rates + step_size * p_s.value
        new_util = sum(np.log(rates))
        if target is None and abs(max_util - new_util) < 1e-6:
            break
        max_util = max(max_util, new_util)
        if target is not None:
            gap = (target - max_util) / abs(target)
            if gap < 0.0001:
                break        
    return max_util, rates, None


def optimize_primal_column(A, q_min, q_max, target=None):
    n_users = A.shape[0]
    wsr_C_Q = I_C_Q(A, q_min, q_max)
    opt, r, alpha, _ = optimize_app_phy(proportional_fair, q_min = np.array([0.001] * n_users), q_max = np.array([10] * n_users), wsr_phy=wsr_C_Q)
  

    return opt, r, alpha
