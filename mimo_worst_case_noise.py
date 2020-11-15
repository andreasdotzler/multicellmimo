import logging
import math

import cvxpy as cp
import numpy as np
import pytest

from .mimo import project_covariance_cvx
from .mimo import project_covariances
from .mimo import ptp_capacity
from .utils import pinv_sqrtm


LOGGER = logging.getLogger(__name__)

inv = np.linalg.inv
pinv = np.linalg.pinv
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


def ptp_worst_case_noise_unconstraint(H, P, sigma=1):
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


def ptp_worst_case_noise_alternating(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx) / Ntx * sigma
    rate_o = None
    for i in range(1000):
        assert np.trace(Z) == pytest.approx(sigma)
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        rate_o, Z = worst_case_noise_uplink(W, sigma)
        rate_o = (rate_o - np.real(log(det(W))) + P * la - Ntx) / np.log(2)
        assert np.all(np.linalg.eigvals(Z) > 0)
        LOGGER.debug(
            f"iteration {i} - rate inner: {rate_i} - rate outer {rate_o} - diff: {rate_i-rate_o}"
        )
        if np.isclose(rate_i, rate_o, 1e-5):
            break
    return rate_i, (Z, W)


def worst_case_noise_uplink(W, sigma):
    Ntx, _ = W.shape
    Z = cp.Variable([Ntx, Ntx], hermitian=True)
    cost = -cp.real(cp.log_det(Z)) + cp.real(cp.trace(Z @ W))
    power = cp.real(cp.trace(Z)) == sigma
    positivity = Z >> 0
    constraints = [power, positivity]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value, Z.value


def ptp_worst_case_noise_approx(H, P, sigma=1, precision=1e-2):
    Nrx, Ntx = H.shape
    if Ntx == 1:
        Z = np.matrix(sigma)
        rate_i, W = ptp_capacity_dual(H.conj().T, P, Z, inf_lim=1e-3)
        return rate_i, (Z, W)
    
    Z = np.eye(Ntx) / Ntx * sigma
    f_is = []
    subgradients = []
    mu = 0
    ei_d, V_d = np.linalg.eigh(H.conj().T@H)
    inf_co = np.zeros((Ntx,Ntx))
    for i, e in enumerate(ei_d):
        if e > 1e-3:
            inf_co = inf_co + V_d[:, [i]] * 1e-3 @ V_d[:,[i]].conj().T
    inf_cons = [inf_co]
    for i in range(1000):
        # todo, do not use this one, it only works for sigma=1
        rate_i, W = ptp_capacity_dual(H.conj().T, P, Z, inf_lim=1e-3)
        LOGGER.debug(f"Iteration {i} - Value {rate_i} - Approximation {mu}")
        if rate_i == np.inf:
            inf_cons += W
            mu, Z = noise_outer_approximation(f_is, subgradients, sigma, inf_cons)
            continue
        if rate_i == pytest.approx(mu, precision):
            break
        Z_gr = -np.linalg.pinv(Z, rcond=1e-3, hermitian=True) + W
        f_is.append(rate_i - np.trace(Z @ Z_gr))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(f_is, subgradients, sigma, inf_cons)
    return rate_i, (Z, W)


def noise_outer_approximation(f_is, subgradients, sigma, inf_cons=[]):
    Ntx, _ = subgradients[0].shape
    Ss = [cp.Parameter(shape=s.shape, hermitian=True) for s in subgradients]
    for c, S in zip(Ss, subgradients):
        c.value = c.project(S)
    Z = cp.Variable([Ntx, Ntx], hermitian=True)
    mu = cp.Variable(1, pos=True)
    cost = mu
    positivity = [Z >> 0]
    power = cp.real(cp.trace(Z)) == cp.Parameter(value=sigma, complex=False)
    f_is_const = [cp.Parameter(value=f_i, complex=False) for f_i in f_is]
    cons = [
        mu >= np.real(f_i) + cp.real(cp.trace(S @ Z))
        for S, f_i in zip(Ss, f_is)
    ]
    Is = [cp.Parameter(shape=inf.shape, hermitian=True) for inf in inf_cons]
    for c, I in zip(Is, inf_cons):
        c.value = c.project(I)
    inf_constraints = [Z >> I for I in Is]
    constraints = cons + positivity + [power] + inf_constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=True, warm_start=True)
    max_retries = 10
    retry=1
    while "inaccurate" in prob.status and retry < max_retries:
        prob.solve(verbose=True, warm_start=True)
        retry+=1
     
    assert "optimal" in prob.status
    return prob.value, Z.value


def ptp_worst_case_noise_gradient(
    H, P, sigma=1, max_iter_outer=100, max_iter_inner=100
):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx)
    W = np.eye(Ntx)
    rate_min = None
    for i in range(max_iter_outer):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        Z_gr = -np.linalg.inv(Z) + W  # +la * P / sigma * np.eye(Ntx)
        Z_gr = -np.linalg.pinv(Z, rcond=1e-9, hermitian=True) + W
        for q in range(max_iter_inner):
            Z_new = Z - 1 / (10 * q + 1) * Z_gr
            Z_new = project_covariances([Z_new], sigma)[0]
            if not np.all(np.linalg.eigvals(Z_new) > 0):
                continue

            rate_new, (W_new, la) = ptp_capacity_uplink_cvx_dual(H, P, Z_new)
            LOGGER.debug(
                f"Outer iteration {i} Min_rate {rate_min}, inner iteration{q} new rate {rate_new}"
            )
            if rate_min is None or rate_new < rate_min:
                rate_min = rate_new
                Z = Z_new
                W = W_new
                break
        if q == max_iter_inner - 1:
            break
        assert np.all(np.linalg.eigvals(Z) > 0)
    return rate_i, (Z, W)


def project_gradient_cvx(X, P):
    p_gra = cp.Variable([X.shape[0], X.shape[0]], hermitian=True)
    obj = cp.Minimize(cp.sum_squares(p_gra - X))
    constraint = cp.real(cp.trace(p_gra)) == P
    prob = cp.Problem(obj, [constraint])
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert prob.status == "optimal"
    return p_gra.value


def ptp_worst_case_noise_gradient_projected(
    H, P, sigma=1, max_iter_outer=20, max_iter_inner=100000
):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx) / Ntx
    rate_min = None
    for i in range(max_iter_outer):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        # import ipdb; ipdb.set_trace()
        Z_gr = -np.linalg.inv(Z) + W  # +la * P / sigma * np.eye(Ntx)
        Z_gr = project_gradient_cvx(Z_gr, 0)
        for q in range(max_iter_inner):
            Z_new = Z - 1 / 10 * (q + 1) * Z_gr
            assert np.trace(Z_new) == pytest.approx(sigma)
            if not np.all(np.linalg.eigvals(Z_new) > 0):
                continue

            rate_new, (W_new, la) = ptp_capacity_uplink_cvx_dual(H, P, Z_new)
            LOGGER.debug(
                f"Outer iteration {i} Min_rate {rate_min}, inner iteration{q} new rate {rate_new}"
            )
            if rate_min is None or rate_new < rate_min:
                rate_min = rate_new
                Z = Z_new
                W = W_new
                break
        if q == max_iter_inner - 1:
            LOGGER.debug(f"Outer iteration {i} - Maximal inner iterations reached {q}")
            break
        assert np.all(np.linalg.eigvals(Z) > 0)
    return rate_i, (Z, W)


def ptp_capacity_dual(H, P, Z, inf_lim):
    Nrx, Ntx = H.shape
    rate, Sigma = ptp_capacity(H, P, Z, inf_lim)
    if rate == np.inf:
        return rate, Sigma
    X = eye(Nrx) + pinv_sqrtm(Z) @ H @ Sigma @ H.conj().T @ pinv_sqrtm(Z)
    W = pinv_sqrtm(Z) @ inv(X) @ pinv_sqrtm(Z)
    return rate, W


def ptp_capacity_uplink_cvx_dual(H, P, Z):
    # assert np.all(np.linalg.eigvals(Z) > 0)
    Nrx, Ntx = H.shape
    W = cp.Variable([Ntx, Ntx], hermitian=True)
    la = cp.Variable(1)
    cost = -cp.real(cp.log_det(W)) + cp.real(cp.trace(Z @ W)) + P * la - Ntx
    positivity_W = W >> 0
    postitvity_la = la >= 0
    cons = cp.multiply(la, np.eye(Nrx)) >> H @ W @ H.conj().T
    constraints = [cons, positivity_W, postitvity_la]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert "optimal" in prob.status
    assert np.all(np.linalg.eigvals(W.value) > 0)
    return np.real(prob.value - log(det(Z))) / np.log(2), (W.value, la.value)


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
