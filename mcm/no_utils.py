import cvxpy as cp
import numpy as np

from typing import List, Union

from mcm.my_typing import x_m_t, Fractions


class InfeasibleOptimization(Exception):
    pass


def solve_problem(util: Union[cp.Maximize, cp.Minimize], cons: List[cp.constraints.constraint.Constraint]) -> cp.Problem:
    prob = cp.Problem(util, cons)
    prob.solve()  # type: ignore
    if "infeasible" in prob.status:
        raise InfeasibleOptimization()
    assert "optimal" in prob.status, f"unable to solve problem: {prob.status}"
    return prob


def d_c_m_t_X_c_m_t(d_c_m_t: x_m_t, c_m_t_s: x_m_t) -> float:
    mm = 0.0
    for m, c_t in c_m_t_s.items():
        for t, c in c_t.items():
            mm += d_c_m_t[m][t] @ c
    return mm


def fractions_from_schedule(alphas_m_t: x_m_t) -> Fractions:
    fractions: Fractions = {}
    total_time = 0.0
    for mode, alphas_t in alphas_m_t.items():
        sum_t_m = []
        for alpha in alphas_t.values():
            sum_t_m.append(sum(alpha))
        mean_sum = float(np.mean(sum_t_m))
        assert np.allclose(
            sum_t_m, np.ones_like(sum_t_m) * mean_sum, rtol=1e-02, atol=1e-02
        )
        fractions[mode] = mean_sum
        total_time += mean_sum
    return fractions
