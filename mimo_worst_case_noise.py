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
    return prob.value, Q.value, (c.dual_value for c in constraints)


def ptp_worst_case_noise_approx(H, P, sigma=1, precision=1e-2):
    Nrx, Ntx = H.shape
    if Ntx == 1:
        Z = np.matrix(sigma)
        rate_i, Sigma = ptp_capacity(H.conj().T, P, Z, rcond=1e-6)
        X = eye(Ntx) + pinv_sqrtm(Z, rcond=1e-6) @ H.conj().T @ Sigma @ H @ pinv_sqrtm(
            Z, rcond=1e-6
        )
        W = pinv_sqrtm(Z, rcond=1e-6) @ inv(X) @ pinv_sqrtm(Z, rcond=1e-6)
        return rate_i, (Z, W)

    Z = np.eye(Ntx) / Ntx * sigma
    f_is = []
    subgradients = []
    mu = 0
    ei_d, V_d = np.linalg.eigh(H.conj().T @ H)
    inf_co = np.zeros((Ntx, Ntx))
    inf_cons = []
    for i, e in enumerate(ei_d):
        if e > 1e-3:
            inf_cons.append(V_d[:, [i]] * 1e-3 @ V_d[:, [i]].conj().T)
    for i in range(1000):
        rate_i, Sigma = ptp_capacity(H.conj().T, P, Z, rcond=1e-6)
        X = eye(Ntx) + pinv_sqrtm(Z, rcond=1e-6) @ H.conj().T @ Sigma @ H @ pinv_sqrtm(
            Z, rcond=1e-6
        )
        W = pinv_sqrtm(Z, rcond=1e-6) @ inv(X) @ pinv_sqrtm(Z, rcond=1e-6)
        LOGGER.debug(f"Iteration {i} - Value {rate_i} - Approximation {mu}")
        if np.allclose(rate_i, mu, rtol=precision):
            break
        Z_gr = -np.linalg.pinv(Z, rcond=1e-6, hermitian=True) + W
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
    cons = [mu >= np.real(f_i) + cp.real(cp.trace(S @ Z)) for S, f_i in zip(Ss, f_is)]
    Is = [cp.Parameter(shape=inf.shape, hermitian=True) for inf in inf_cons]
    for c, I in zip(Is, inf_cons):
        c.value = c.project(I)
    inf_constraints = [Z >> I for I in Is]
    constraints = cons + positivity + [power] + inf_constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=True, warm_start=True)
    max_retries = 10
    retry = 1
    while "inaccurate" in prob.status and retry < max_retries:
        prob.solve(verbose=True, warm_start=True)
        retry += 1

    assert "optimal" in prob.status, f"no optimal solution found, status: {prob.status}"
    return prob.value, Z.value
