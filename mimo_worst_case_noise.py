import logging
import math

import cvxpy as cp
import numpy as np
import pytest

from .mimo import project_covariance_cvx
from .mimo import project_covariances
from .mimo import ptp_capacity
from .mimo import water_filling
from .utils import pinv_sqrtm, sqrtm


LOGGER = logging.getLogger(__name__)

inv = np.linalg.inv
pinv = np.linalg.pinv
det = np.linalg.det
log = np.log
eye = np.eye


def ptp_capacity_minimax(H, R, C, Zs, eps=1e-3):
    Nrx, Ntx = H.shape
    assert (Nrx, Nrx) == R.shape
    assert (Ntx, Ntx) == C.shape
    for Z in Zs:
        assert (Ntx, Ntx) == Z.shape
    complex = any([np.iscomplex(H).any(), np.iscomplex(R).any(), np.iscomplex(C).any()])
    complex = True
    zs = cp.Variable(len(Zs), complex=False)
    Z_equal = np.zeros(Nrx)
    if complex:
        Q = cp.Variable([Ntx, Ntx], hermitian=True)
        Z = cp.Variable([Ntx, Ntx], hermitian=True)
        aC = cp.Parameter(shape=C.shape, value=C, hermitian=True)
        HTSH = cp.Parameter(shape=(Ntx, Ntx), hermitian=True)
        HTSH.value = HTSH.project(H.conj().T @ np.linalg.pinv(R, rcond=1e-6, hermitian=True) @ H)
        R_pinv = cp.Parameter(shape=(Nrx, Nrx), hermitian=True)
        R_pinv.value = R_pinv.project(np.linalg.pinv(R, rcond=1e-6, hermitian=True))
  
        for ZZ, z in zip(Zs, zs):
            Z_equal += cp.multiply(
                z, cp.Parameter(shape=(Ntx, Ntx), value=ZZ, hermitian=True)
            )

    cost = cp.log_det(np.eye(Nrx) + sqrtm(R_pinv.value) @ H @ Q @ H.conj().T@ sqrtm(R_pinv.value))
    shape = Q << aC + Z
    Zsubspace = Z == Z_equal
    positivity = Q >> 0
    constraints = [shape, Zsubspace, positivity]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=eps, max_iters=100000)
    #prob.solve()
    #prob.solve(solver=cp.SCS, warm_start=True, max_iters=1000)
    assert prob.status == "optimal"
    return log(det(eye(Ntx) + HTSH.value@Q.value)), Q.value, (c.dual_value for c in constraints)



def ptp_worst_case_noise_static(HQHT, sigma, precision=1e-2):
    Nrx = HQHT.shape[0]
    Z = np.eye(Nrx) / Nrx * sigma
    f_is = []
    subgradients = []
    mu = 0
    ei_d, V_d = np.linalg.eigh(HQHT)
    inf_cons = []
    for i, e in enumerate(ei_d):
        if e > 1e-3:
            inf_min = 1e-3
            inf_cons.append(V_d[:, [i]] * inf_min @ V_d[:, [i]].conj().T)
    for i in range(1000):
        Z_inv = np.linalg.pinv(Z, rcond=1e-6, hermitian=True)
        rate_i = np.real(log(det(eye(Nrx) + Z_inv@HQHT)))
        W = np.linalg.pinv(Z + HQHT, rcond=1e-6, hermitian=True)
        Z_gr = -Z_inv + W 
        LOGGER.debug(f"Iteration {i} - Value {rate_i:.5f} - Approximation {mu:.5f}")
        if np.allclose(rate_i, mu, rtol=precision):
            break
        f_is.append(rate_i - np.real(np.trace(Z @ Z_gr)))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(f_is, subgradients, sigma, inf_cons)
    return rate_i, Z

def ptp_worst_case_noise_approx(H, P, sigma=1, precision=1e-2, rcond=1e-6, infcond=1e-4):
    Nrx, Ntx = H.shape

    Z = np.eye(Ntx) / Ntx * sigma
    f_is = []
    subgradients = []
    mu = 0
    ei_d, V_d = np.linalg.eigh(H.conj().T @ H)
    rrr, _ = ptp_capacity(H.conj().T, P, Z)
    inf_cons = []
    for i, e in enumerate(ei_d):
        if e > infcond:
            inf_min = e*P/(math.exp(rrr) - 1)
            inf_cons.append(V_d[:, [i]] * inf_min @ V_d[:, [i]].conj().T)
    for i in range(1000):
        rate_i, Z_gr, Q = approx_inner(H, Z, P)
        LOGGER.debug(f"Iteration {i} - Value {rate_i:.5f} - Approximation {mu:.5f}")
        if np.allclose(rate_i, mu, rtol=precision):
            break
        f_is.append(rate_i - np.real(np.trace(Z @ Z_gr)))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(f_is, subgradients, sigma, inf_cons)
    return rate_i, Z, Q

def approx_inner(H, Z, P):
    ei_z, V_z = np.linalg.eigh(Z)
    above_cutoff = (ei_z > 1e-6)
    psigma_diag = 1.0 / ei_z[above_cutoff]
    V_u = V_z[:, above_cutoff]
    Z_inv = np.dot(V_u * psigma_diag, np.conjugate(V_u).T)
    ei_d, V_d = np.linalg.eigh(H@(V_u*psigma_diag)@V_u.conj().T@H.conj().T)
    ei_d = [max(e, 0) for e in ei_d]
    power = water_filling(ei_d, P)
    Sigma = V_d @ np.diag(power) @ V_d.conj().T
    rate_i = sum(math.log(1 + p * e) for p, e in zip(power, ei_d))
    HQHT = H.conj().T @ Sigma @ H
    W = np.linalg.pinv(Z + HQHT, rcond=1e-6, hermitian=True)
    Z_gr = -Z_inv + W 
    return rate_i, Z_gr, Sigma

# Idea, we could compute the worst case noise of a minimax optimization, it is the dual variable

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
    f_is_const = [cp.Parameter(value=np.real(f_i), complex=False) for f_i in f_is]
    cons = [mu >= np.real(f_i) + cp.real(cp.trace(S @ Z)) for S, f_i in zip(Ss, f_is)]
    Is = [cp.Parameter(shape=inf.shape, hermitian=True) for inf in inf_cons]
    for c, I in zip(Is, inf_cons):
        c.value = c.project(I)
    inf_constraints = [Z >> I for I in Is]
    #inf_constraints = [cp.real(cp.quad_form(I,Z)) >= 1e-5 for I in inf_cons]
    constraints = cons + positivity + [power] + inf_constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.CVXOPT, warm_start=True, max_iters=1000, kktsolver='robust', abstol=100, reltol=1e-3, refinement=1)
    #prob.solve(solver=cp.CVXOPT, warm_start=True, max_iters=1000, kktsolver='chol')
 
    max_retries = 5
    retry = 0
    while prob.status != "optimal" and retry < max_retries:
        retry += 1
        LOGGER.debug(f"Infeasible, retry {retry}")
        prob.solve(warm_start=True)
    assert "optimal" in prob.status, f"no optimal solution found, status: {prob.status}"

    return prob.value if prob.status == "optimal" else 0, Z.value
