import cvxpy as cp
class InfeasibleOptimization(Exception):
    pass

def solve_problem(util, cons):
    prob = cp.Problem(util, cons)
    prob.solve()
    if "infeasible" in prob.status:
        raise InfeasibleOptimization()
    assert "optimal" in prob.status, f"unable to solve problem: {prob.status}"
    return prob
