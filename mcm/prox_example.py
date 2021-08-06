"""TODO."""

import cvxpy as cp
import numpy as np

np.set_printoptions(precision=3)

n_users = 2
# Approximation
A = np.matrix("2.5; 1")
q_min = np.array([2.5, 1])
q_max = np.array([5, 2])
min_dual = None
for n in range(1, 1000):
    # create and solve the approximated problem
    b = np.ones(n)
    alpha = cp.Variable(n)
    r = cp.Variable(n_users)
    cost = cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(r))
    c1 = r == A @ alpha
    c2 = r >= q_min
    c3 = r <= q_max
    c4 = alpha >= 0
    c5 = b @ alpha == 1
    constraints = [c1, c2, c3, c4, c5]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)
    print(f"\n###Iteration {n} ###")
    print(f"The optimimum of approximation is {prob.value}")
    print(f"The optimal rates of the approximation are {r.value}")
    print(f"The timeshare coefficients of the approximation are {alpha.value}")
    print(f"The dual variables are {list(c.dual_value for c in constraints)}")

    # check the KKT system

    lambda_phy = c1.dual_value
    print(f"Lambda pyh is {lambda_phy}")
    # check if lambda_phy is correct
    # lambda_phy in d I_Cl
    r_opt = r.value
    for val in np.nditer(lambda_phy @ A):
        print(f"Expecting {val} - {lambda_phy@r_opt} <= 0")
        assert val - lambda_phy @ r_opt <= 0.001
    # lambda_phy in d U - d I_Q
    for k, (r_k, q_min_k, q_max_k, lambda_phy_k) in enumerate(
        zip(r_opt, q_min, q_max, lambda_phy)
    ):

        r_k_inv = 1 / r_k
        if abs(r_k - q_min_k) < 0.001:
            print(
                f"User {k} with rate {r_k:.2f} is at minmal rate {q_min_k:.2f}, expecting 1/r = {r_k_inv:.2f} < lambda_k = {lambda_phy_k:.2f}"
            )
            assert r_k_inv < lambda_phy_k + 0.001

            continue
        if abs(r_k - q_max_k) < 0.001:
            print(
                f"User {k} with rate {r_k:.2f} is at maximal rate {q_max_k:.2f}, expecting 1/r = {r_k_inv:.2f} > lambda_k = {lambda_phy_k:.2f}"
            )
            assert r_k_inv > lambda_phy_k - 0.001
            continue

        print(
            f"User {k} with rate {r_k:.2f} neither min {q_min_k:.2f} nor max {q_max_k:.2f} expecting 1/r = {r_k_inv:.2f} = lambda_k = {lambda_phy_k:.2f}"
        )
        assert abs(r_k_inv - lambda_phy_k) < 0.001

    q = cp.Variable(n_users)
    cost_dual = (
        cp.atoms.affine.sum.Sum(cp.atoms.elementwise.log.log(q)) - lambda_phy @ q
    )
    constraints_dual = [q >= q_min, q <= q_max]
    prob_dual = cp.Problem(cp.Maximize(cost_dual), constraints_dual)
    prob_dual.solve(solver=cp.SCS, eps=1e-8)
    mu = max(lambda_phy @ A)
    print(
        f"The optimimum of the dual approximation is {prob_dual.value} + {mu} = {prob_dual.value + mu}"
    )
    print(f"The optimal rates of the dual approximation are {q.value}")
    print(f"The dual variables are {c4.dual_value}")

    A_phy = np.matrix("1,1; 2,1")
    b_phy = [6, 8]
    c_phy = cp.Variable(n_users)

    prob_phy = cp.Problem(
        cp.Maximize(lambda_phy @ c_phy), [A_phy @ c_phy <= b_phy, c_phy >= 0]
    )
    prob_phy.solve(solver=cp.SCS, eps=1e-8)

    print("\nThe optimal value of the update is", prob_phy.value)
    print("The update is", c_phy.value)
    A = np.c_[A, c_phy.value]
    print("New approximation points: ", A)
    new_dual = prob_dual.value + prob_phy.value
    print("Dual value: ", new_dual)
    if not min_dual:
        min_dual = new_dual
    min_dual = min(min_dual, new_dual)
    if abs(min_dual - prob.value) < 0.001:
        break
