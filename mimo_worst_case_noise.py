import logging
import math

import cvxpy as cp
import numpy as np
import pytest

LOGGER = logging.getLogger(__name__)

inv = np.linalg.inv
det = np.linalg.det
log = np.log
eye = np.eye


def ptp_capacity_mimimax(H, R, C, P, alpha=1):
    Nrx, Ntx = H.shape
    Q = cp.Variable([Ntx, Ntx])
    Z = cp.Variable([Ntx, Ntx])
    S = alpha * R
    cost = cp.log_det(np.eye(Nrx) + np.linalg.inv(S) @ H @ Q @ H.conj().T)
    shape = Q << alpha * C + Z
    trZ = cp.trace(Z) == 0
    positivity = Q >> 0
    constraints = [shape, trZ, positivity]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value / np.log(2), Q.value, (c.dual_value for c in constraints)


def ptp_worst_case_noise(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx)
    rate_o = None
    for i in range(100):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P / sigma, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        Z = inv(W + la * P / sigma * np.eye(Ntx))
        assert np.all(np.linalg.eigvals(Z) > 0)
        LOGGER.debug(f"rate inner: {rate_i} : rate outer {rate_o}")
    return rate_i, (Z, W)


def ptp_worst_case_noise_constraint(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx) / Ntx * sigma
    rate_o = None
    for i in range(100):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        rate_o, Z = worst_case_noise_uplink(W, sigma)
        assert np.trace(Z) == pytest.approx(sigma)
        assert np.all(np.linalg.eigvals(Z) > 0)
        LOGGER.debug(f"rate inner: {rate_i} : rate outer {rate_o}")
    return rate_i, (Z, W)


def worst_case_noise_uplink(W, sigma):
    Ntx, _ = W.shape
    A = cp.Variable([Ntx, Ntx], symmetric=True)
    cost = -cp.log_det(A) + cp.trace(W @ A)
    power = cp.trace(A) <= sigma
    positivity = A >> 0
    constraints = [power, positivity]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value / np.log(2), A.value


def ptp_worst_case_noise_approx(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx) / Ntx
    f_is = []
    subgradients = []
    mu = 0
    for i in range(200):
        # todo, do not use this one, it only works for sigma=1
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)

        if rate_i - mu < 1e-4:
            break
        Z_gr = -np.linalg.inv(Z) + W
        f_is.append(rate_i - np.trace(Z @ Z_gr))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(f_is, subgradients, sigma)
        LOGGER.debug(f"Value {rate_i}, approximation {mu}")
    return rate_i, (Z, W)


def noise_outer_approximation(f_is, subgradients, sigma):
    Nrx, Ntx = subgradients[0].shape
    if Nrx > Ntx:
        raise NotImplementedError(
            "Worst-case noise approximation for Nrx > Ntx not implemented"
        )
    Z = cp.Variable([Ntx, Ntx], symmetric=True)
    mu = cp.Variable(1)
    cost = mu
    positivity = Z >> np.eye(Ntx) * 0.0001
    power = cp.trace(Z) <= sigma
    cons = [
        mu >= np.real(f_i) + cp.real(cp.trace(S @ Z))
        for S, f_i in zip(subgradients, f_is)
    ]
    constraints = cons + [positivity, power]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert "optimal" in prob.status
    return prob.value, Z.value


def ptp_worst_case_noise_gradient(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx)
    W = np.eye(Ntx)
    rate_min = None
    for i in range(10):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P / sigma, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        Z_gr = -np.linalg.inv(Z) + W + la * P / sigma * np.eye(Ntx)
        for q in range(30):
            Z_new = Z - 1 / (q + 1) * Z_gr
            Z_new = project_covariance_cvx([Z_new], 10 ** 3)[0]
            rate_new, (W_new, la) = ptp_capacity_uplink_cvx_dual(H, P / sigma, Z_new)
            LOGGER.debug(
                f"Outer iteration {i} Min_rate {rate_min}, inner iteration{q} new rate {rate_new}"
            )
            if rate_min is None or rate_new < rate_min:
                rate_min = rate_new
                Z = Z_new
                W = W_new
                break

        assert np.all(np.linalg.eigvals(Z) > 0)
    return rate_i, (Z, W)


def ptp_capacity_uplink_cvx_dual(H, P, Z):
    Nrx, Ntx = H.shape
    W = cp.Variable([Ntx, Ntx], hermitian=True)
    la = cp.Variable(1)
    cost = (
        -cp.real(cp.log_det(W))
        - np.real(log(det(Z)))
        + cp.real(cp.trace(Z @ W))
        + P * la
        - Ntx
    )
    positivity_W = W >> 0
    postitvity_la = la >= 0
    cons = cp.multiply(la, np.eye(Nrx)) >> H @ W @ H.conj().T
    constraints = [cons, positivity_W, postitvity_la]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert "optimal" in prob.status
    assert np.all(np.linalg.eigvals(W.value) > 0)
    return prob.value / np.log(2), (W.value, la.value)


def ptp_noise_cvx_dual(H, P, W, la):
    Nrx, Ntx = H.shape
    Z = cp.Variable([Ntx, Ntx])
    cost = (
        log(det(W)) - cp.log_det(Z) + cp.trace(Z @ W) + P / Ntx * la * cp.trace(Z) - Ntx
    )
    cost = -cp.log_det(Z) + cp.trace(Z @ W) + P / Ntx * la * cp.trace(Z)
    positivity_Z = Z >> 0
    constraints = [positivity_Z]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert prob.status == "optimal"
    return prob.value / np.log(2), Z.value
