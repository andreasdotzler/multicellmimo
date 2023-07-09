"""TODO we want typing and assert istype on uplink and downlink channels"""
import logging
import math
from typing import List, Tuple, Optional

import cvxpy as cp
import numpy as np

from .mimo import MAC_cvx, ptp_capacity, water_filling
from .my_typing import Matrix
from .mimo_utils import sqrtm, logdet

LOGGER = logging.getLogger(__name__)

inv = np.linalg.inv
pinv = np.linalg.pinv
det = np.linalg.det
log = np.log
eye = np.eye


def ptp_capacity_minimax(
    H: Matrix, R: Matrix, C: Matrix, Zs: List[Matrix], eps: float = 1e-3
) -> object:
    """Compute the minimax capacity of point-to-point link.

    Parameters
    ----------
    H
    R
    C
    Zs
    eps

    Returns
    -------
    float
    """
    Nrx, Ntx = H.shape
    assert (Nrx, Nrx) == R.shape
    assert (Ntx, Ntx) == C.shape
    for Z in Zs:
        assert (Ntx, Ntx) == Z.shape
    zs = cp.Variable(len(Zs), complex=False)
    Z_equal = np.zeros(Nrx)

    Q = cp.Variable((Ntx, Ntx), hermitian=True)
    Z = cp.Variable((Ntx, Ntx), hermitian=True)
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

    cp.log_det(
        np.eye(Nrx) + sqrtm(R_pinv.value) @ H @ Q @ H.conj().T @ sqrtm(R_pinv.value)
    )
    shape = Q << aC + Z
    Zsubspace = Z == Z_equal
    positivity = Q >> 0
    constraints = [shape, Zsubspace, positivity]

    # prob.solve()
    # prob.solve(solver=cp.SCS, warm_start=True, max_iters=1000)
    assert prob.status == "optimal"
    return (
        logdet(eye(Ntx) + HTSH.value @ Q.value),
        Q.value,
        (c.dual_value for c in constraints),
    )


def ptp_worst_case_noise_static(
    HQHT: Matrix, sigma: float, precision: float = 1e-2
) -> Tuple[float, Matrix]:
    """Compute the worst case noise for static transmission.

    Parameters
    ----------
    HQHT
    sigma
    precision

    Returns
    -------
    rate_i
    Z

    """
    Nrx = HQHT.shape[0]
    Z = np.eye(Nrx) / Nrx * sigma
    f_is = []
    subgradients = []
    mu = 0.0
    ei_d, V_d = np.linalg.eigh(HQHT)
    inf_constraints = []
    for i, e in enumerate(ei_d):
        if e > 1e-3:
            inf_min = 1e-3
            inf_constraints.append(V_d[:, [i]] * inf_min @ V_d[:, [i]].conj().T)
    rate_i = 0.0
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
            f_is, subgradients, sigma, inf_constraints, mini=0.0
        )
    return rate_i, Z


def inf_cons(H: Matrix, P: float, rate: float) -> List[Matrix]:
    r"""Find constraints on the worst-case uplink noise for a user.

    Parameters
    ----------
    H: Matrix
        Channel matrix.
    P: float
        Transmit power
    rate: float
        Maximal rate

    Returns
    -------
    Is: List[Matrix]
        TODO constrainst to be used as ...

    Given a feasible uplink covariance we compute a set of matrices Is
    such that U = V
    :math:`a^2 + b^2 = c^2`.

    .. math::

       U = \min_{\Omega} \lbrace \max_{\Sigma} \lbrace \log\det(I + \Omega^{-1} H^H \Sigma H): tr(\Sigma) < P : \Omega in \Omega, \Omega \geq Is \forall I in Is \rbrace \rbrace

    V = min_Omega max_Sigma logdet(I + inv(Omega) @ H.conj().T @ Sigma @ H): tr(Sigma) < P : Omega in Omegas
    r = max_Sigma logdet(I + inv(Omega_p) @ H.conj().T @ Sigma @ H): tr(Sigma) < P

    we know u <= r

    We select the constrainst, such that the noise power in each mode of the channel is large enough that if we put all
    transmit power into that mode, the resulting rate is r. This guarantees a finite worst case rates when solving U instead of V.
    """
    infcond = 1e-4
    ei_d, V_d = np.linalg.eigh(H.conj().T @ H)
    # TODO add weights and make ptp_capacity optional if no rate is supplied
    Is: List[Matrix] = []
    for i, e in enumerate(ei_d):
        if e > infcond and rate > 0:
            inf_min = e * P / (math.exp(rate) - 1)
            if inf_min > infcond:
                Is.append(V_d[:, [i]] * inf_min @ V_d[:, [i]].conj().T)
    return Is


def ptp_worst_case_noise_approx(
    H: Matrix, P: float, sigma: float = 1, precision: float = 1e-2
) -> Tuple[float, Matrix, Matrix]:
    """Optimize worst case noise for point-to-point channel via outer approximation.

    Parameters
    ----------
    H
    P
    sigma
    precision

    Returns
    -------
    rate_i
    Z
    Q
    """
    Nrx, Ntx = H.shape
    f_is = []
    subgradients = []
    mu = 0.0
    Z = np.eye(Ntx) / Ntx * sigma
    rate, _ = ptp_capacity(H.conj().T, P, Z)
    Is: List[Matrix] = inf_cons(H, P, rate)
    rate_i = 0.0
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
    Hs: List[Matrix],
    P: float,
    sigma: float = 1,
    weights: List[float] = None,
    precision: float = 1e-2,
) -> Tuple[List[float], Matrix, List[Matrix], List[int]]:
    """Optimize worste case noise by outer approximation.

    Parameters
    ----------
    Hs
    P
    sigma
    weights
    precision

    Returns
    -------
    rates_i
    Omega
    Covs
    order

    """
    weights = weights or [1 for _ in Hs]
    _, Bs_antennas = Hs[0].shape
    Hs_MAC = [H.conj().T for H in Hs]
    f_is = []
    subgradients = []
    mu = 0.0
    Omega = np.eye(Bs_antennas) / Bs_antennas * sigma
    # TODO do a wsr once, than bounds for every user with weight and wsr_as target
    rates, _, _, _ = MAC_cvx(Hs_MAC, P, weights, Omega)

    wsr = sum([w * r for w, r in zip(weights, rates)])
    Is = []
    for w, H in zip(weights, Hs):
        if w > 0:
            Is += inf_cons(H, P, wsr / w)
    # Is = []
    for i in range(1000):
        rates_i, Covs, order, Omega_gr = MAC_cvx(Hs_MAC, P, weights, Omega)
        wsr_i = sum([w * r for w, r in zip(weights, rates_i)])
        LOGGER.debug(f"Iteration {i} - Value {wsr_i:.5f} - Approximation {mu:.5f}")
        if np.allclose(wsr_i, mu, rtol=precision):
            break
        assert wsr_i > mu
        f_is.append(wsr_i - np.real(np.trace(Omega @ Omega_gr)))
        subgradients.append(Omega_gr)
        mu, Omega = noise_outer_approximation(f_is, subgradients, sigma, Is)
    return rates_i, Omega, Covs, order


def approx_inner_ptp(H: Matrix, Z: Matrix, P: float) -> Tuple[float, Matrix, Matrix]:
    """TODO: match with ptp MIMO. CHECK uplink and DOWNLINK Covariances

    Parameters
    ----------
    H: Matrix
        channel matrix
    Z: Matrix
        Noise covariance
    P: float
        power constraint

    Returns
    -------
    rate_i: float
        Data rate
    Z_gr: Matrix
        Noise covariance subgradient
    Sigma: Matrix
        Transmit convariance matrix


    """
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


def noise_outer_approximation(
    f_is: List[Matrix],
    subgradients: List[Matrix],
    sigma: float,
    inf_constraints: Optional[List[Matrix]] = None,
    mini: float = 0,
) -> Tuple[float, Matrix]:
    """Optimize an outer approximation of a function by subgradients.

    TODO: put the math here

    Parameters
    ----------
    f_is
    subgradients
    sigma
    inf_constraints
    mini

    Returns
    -------
    approximation: float
        approximated value
    value: Matrix
        optimizer

    """
    if inf_constraints is None:
        inf_constraints = []
    Ntx, _ = subgradients[0].shape
    Ss = [cp.Parameter(shape=s.shape, hermitian=True) for s in subgradients]
    for c, S in zip(Ss, subgradients):
        c.value = c.project(S)
    Z = cp.Variable((Ntx, Ntx), hermitian=True)
    mu = cp.Variable(1, pos=True)
    cost = mu
    positivity = [Z >> 0]
    power = cp.real(cp.trace(Z)) == cp.Parameter(value=sigma, complex=False)
    # f_is_const = [cp.Parameter(value=np.real(f_i), complex=False) for f_i in f_is]
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
