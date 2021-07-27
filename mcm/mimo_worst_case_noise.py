import logging
import math

import cvxpy as cp
import numpy as np
import pytest

from .mimo import project_covariance_cvx
from .mimo import project_covariances
from .mimo import ptp_capacity
from .mimo import water_filling
from .mimo import MAC_cvx_with_noise_sbgr
from .utils import pinv_sqrtm, sqrtm, logdet


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
    # TODO we hardcoded complex here
    complex = True
    zs = cp.Variable(len(Zs), complex=False)
    Z_equal = np.zeros(Nrx)
    if complex:
        Q = cp.Variable([Ntx, Ntx], hermitian=True)
        Z = cp.Variable([Ntx, Ntx], hermitian=True)
        aC = cp.Parameter(shape=C.shape, value=C, hermitian=True)
        HTSH = cp.Parameter(shape=(Ntx, Ntx), hermitian=True)
        HTSH.value = HTSH.project(
            H.conj().T @ np.linalg.pinv(R, rcond=1e-6, hermitian=True) @ H
        )
        R_pinv = cp.Parameter(shape=(Nrx, Nrx), hermitian=True)
        R_pinv.value = R_pinv.project(np.linalg.pinv(R, rcond=1e-6, hermitian=True))

        for ZZ, z in zip(Zs, zs):
            Z_equal += cp.multiply(
                z, cp.Parameter(shape=(Ntx, Ntx), value=ZZ, hermitian=True)
            )

    cost = cp.log_det(
        np.eye(Nrx) + sqrtm(R_pinv.value) @ H @ Q @ H.conj().T @ sqrtm(R_pinv.value)
    )
    shape = Q << aC + Z
    Zsubspace = Z == Z_equal
    positivity = Q >> 0
    constraints = [shape, Zsubspace, positivity]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=eps, max_iters=100000)
    # prob.solve()
    # prob.solve(solver=cp.SCS, warm_start=True, max_iters=1000)
    assert prob.status == "optimal"
    return (
        logdet(eye(Ntx) + HTSH.value @ Q.value),
        Q.value,
        (c.dual_value for c in constraints),
    )


def ptp_worst_case_noise_static(HQHT, sigma, precision=1e-2):
    Nrx = HQHT.shape[0]
    Z = np.eye(Nrx) / Nrx * sigma
    f_is = []
    subgradients = []
    mu = 0
    ei_d, V_d = np.linalg.eigh(HQHT)
    inf_constraints = []
    for i, e in enumerate(ei_d):
        if e > 1e-3:
            inf_min = 1e-3
            inf_constraints.append(V_d[:, [i]] * inf_min @ V_d[:, [i]].conj().T)
    for i in range(1000):
        Z_inv = np.linalg.pinv(Z, rcond=1e-6, hermitian=True)
        rate_i = logdet(eye(Nrx) + Z_inv @ HQHT)
        W = np.linalg.pinv(Z + HQHT, rcond=1e-6, hermitian=True)
        Z_gr = -Z_inv + W
        LOGGER.debug(f"Iteration {i} - Value {rate_i:.5f} - Approximation {mu:.5f}")
        if np.allclose(rate_i, mu, rtol=precision):
            break
        f_is.append(rate_i - np.real(np.trace(Z @ Z_gr)))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(
            f_is, subgradients, sigma, inf_constraints, mini=0
        )
    return rate_i, Z


def inf_cons(H, P, rate):
    """Find constraints on the worst-case uplink noise for a user

    Keyword arguments:
    H -- channel matrix
    P -- transmit power
    Omega -- a feasible uplink noise covariance

    Given a feasible uplink covariance we compute a set of matrices Is
    such that U = V

    U = min_Omega max_Sigma logdet(I + inv(Omega) @ H.conj().T @ Sigma @ H): tr(Sigma) < P : Omega in Omegas, Omega >= Is forall I in Is 
    V = min_Omega max_Sigma logdet(I + inv(Omega) @ H.conj().T @ Sigma @ H): tr(Sigma) < P : Omega in Omegas 
    r = max_Sigma logdet(I + inv(Omega_p) @ H.conj().T @ Sigma @ H): tr(Sigma) < P

    we know u <= r

    We select the constrainst, such that the noise power in each mode of the channel is large enough that if we put all transmit power into that mode, the resulting rate is r. This guarantees a finite worst case rates when solving U instead of V. 
    """
    infcond = 1e-4
    ei_d, V_d = np.linalg.eigh(H.conj().T @ H)
    # TODO add weights and make ptp_capacity optional if no rate is supplied
    Is = []
    for i, e in enumerate(ei_d):
        if e > infcond and rate > 0:
            inf_min = e * P / (math.exp(rate) - 1)
            if inf_min > infcond:
                Is.append(V_d[:, [i]] * inf_min @ V_d[:, [i]].conj().T)
    return Is


def ptp_worst_case_noise_approx(
    H, P, sigma=1, precision=1e-2, rcond=1e-6, infcond=1e-4
):
    Nrx, Ntx = H.shape
    f_is = []
    subgradients = []
    mu = 0
    Z = np.eye(Ntx) / Ntx * sigma
    rate, _ = ptp_capacity(H.conj().T, P, Z)
    Is = inf_cons(H, P, rate)
    for i in range(1000):
        rate_i, Z_gr, Q = approx_inner_ptp(H, Z, P)
        LOGGER.debug(f"Iteration {i} - Value {rate_i:.5f} - Approximation {mu:.5f}")
        if np.allclose(rate_i, mu, rtol=precision):
            break
        f_is.append(rate_i - np.real(np.trace(Z @ Z_gr)))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(f_is, subgradients, sigma, Is)
    return rate_i, Z, Q


def MAC_worst_case_noise_approx(
    Hs, P, sigma=1, weights=None, precision=1e-2, rcond=1e-6, infcond=1e-4
):
    weights = weights or [1 for _ in Hs]
    _, Bs_antennas = Hs[0].shape
    Hs_MAC = [H.conj().T for H in Hs]
    f_is = []
    subgradients = []
    mu = 0
    Omega = np.eye(Bs_antennas) / Bs_antennas * sigma
    # TODO do a wsr once, than bounds for every user with weight and wsr_as target
    rates, _, _, _ = approx_inner_MAC(Hs_MAC, Omega, P, weights)
    wsr = sum([w * r for w, r in zip(weights, rates)])
    Is = []
    for w, H in zip(weights, Hs):
        if w > 0:
            Is += inf_cons(H, P, wsr / w)
    # Is = []
    for i in range(1000):
        rates_i, Omega_gr, Covs, order = approx_inner_MAC(Hs_MAC, Omega, P, weights)
        wsr_i = sum([w * r for w, r in zip(weights, rates_i)])
        LOGGER.debug(f"Iteration {i} - Value {wsr_i:.5f} - Approximation {mu:.5f}")
        if np.allclose(wsr_i, mu, rtol=precision):
            break
        assert wsr_i > mu
        f_is.append(wsr_i - np.real(np.trace(Omega @ Omega_gr)))
        subgradients.append(Omega_gr)
        mu, Omega = noise_outer_approximation(f_is, subgradients, sigma, Is)
    return rates_i, Omega, Covs, order


def approx_inner_ptp(H, Z, P):
    ei_z, V_z = np.linalg.eigh(Z)
    above_cutoff = ei_z > 1e-6
    psigma_diag = 1.0 / ei_z[above_cutoff]
    V_u = V_z[:, above_cutoff]
    Z_inv = np.dot(V_u * psigma_diag, np.conjugate(V_u).T)
    ei_d, V_d = np.linalg.eigh(H @ (V_u * psigma_diag) @ V_u.conj().T @ H.conj().T)
    ei_d = [max(e, 0) for e in ei_d]
    power = water_filling(ei_d, P)
    Sigma = V_d @ np.diag(power) @ V_d.conj().T
    rate_i = sum(math.log(1 + p * e) for p, e in zip(power, ei_d))
    HQHT = H.conj().T @ Sigma @ H
    W = np.linalg.pinv(Z + HQHT, rcond=1e-6, hermitian=True)
    Z_gr = -Z_inv + W
    return rate_i, Z_gr, Sigma


def approx_inner_MAC(Hs, Omega, P, weights):
    # rates, MAC_Covs, order, eye_sbgr = MAC_cvx_with_noise_sbgr([pinv_sqrtm(Omega)@H for H in Hs], P, weights, Omega=np.eye(Omega.shape[0]))
    # Omega_sbgr = pinv_sqrtm(Omega)@eye_sbgr@pinv_sqrtm(Omega)
    rates, MAC_Covs, order, Omega_sbgr = MAC_cvx_with_noise_sbgr(
        Hs, P, weights, Omega=Omega
    )
    # TODO compare the gradient to ptp

    # Omega_sbgr = pinv_sqrtm(Omega)@eye_sbgr@pinv_sqrtm(Omega)
    return rates, Omega_sbgr, MAC_Covs, order


def noise_outer_approximation(f_is, subgradients, sigma, inf_constraints=[], mini=0):
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
    Is = [cp.Parameter(shape=inf.shape, hermitian=True) for inf in inf_constraints]
    for c, I in zip(Is, inf_constraints):
        c.value = c.project(I)
    inf_constraints = [Z >> I for I in Is]
    # inf_constraints = [cp.real(cp.quad_form(I,Z)) >= 1e-5 for I in inf_cons]
    constraints = cons + positivity + [power] + inf_constraints + [mu >= mini]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(
        solver=cp.CVXOPT,
        warm_start=True,
        max_iters=1000,
        kktsolver="robust",
        abstol=100,
        reltol=1e-3,
        refinement=1,
    )
    # prob.solve(solver=cp.CVXOPT, warm_start=True, max_iters=1000, kktsolver='chol')

    max_retries = 5
    retry = 0
    while prob.status != "optimal" and retry < max_retries:
        retry += 1
        LOGGER.debug(f"Infeasible, retry {retry}")
        prob.solve(warm_start=True)
    assert "optimal" in prob.status, f"no optimal solution found, status: {prob.status}"

    return prob.value if prob.status == "optimal" else 0, Z.value
