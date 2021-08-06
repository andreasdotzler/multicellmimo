"""JUHU."""
import logging
import math
import pytest

import cvxpy as cp
import numpy as np
from typing import List, Tuple, Optional
from .utils import inv_sqrtm, sqrtm, logdet, log, pinv, argsort
from .typing import Matrix

LOGGER = logging.getLogger(__name__)


def water_filling_cvx(ei: List[float], P: float) -> List[float]:
    """Water filling to optimized power allocation over multiple channels.

    Parameters
    ----------
    ei: List[float]
        Channel gains
    P: float
        Transmit power

    Returns
    -------
    power: List[float]
        Optimized power allocation
    """
    assert all([e.imag == 0 for e in ei])
    power = cp.Variable(len(ei))
    alpha = cp.Parameter(len(ei), nonneg=True)
    alpha.value = ei
    obj = cp.Maximize(cp.sum(cp.log(1 + cp.multiply(power, alpha))))
    constraints = [power >= 0, cp.sum(power) == P]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return power.value


def water_filling(ei: List[float], P: float) -> List[float]:
    """Water filling to optimized power allocation over multiple channels.

    Parameters
    ----------
    ei: List[float]
        Channel gains
    P: float
        Transmit power

    Returns
    -------
    power: List[float]
        Optimized power allocation
    """
    assert all([e.imag == 0 for e in ei])
    channel_gains = np.array(ei, dtype=np.double)
    nonzero_index = channel_gains >= 10e-16
    powers_aux = np.zeros(len(channel_gains), dtype=np.double)
    powers = np.zeros(len(channel_gains), dtype=np.double)
    while any(powers_aux <= 0):
        channel_gains_aux = channel_gains[nonzero_index]
        eta = (P + sum(1 / channel_gains_aux)) / sum(nonzero_index)
        powers_aux = eta - (1 / channel_gains_aux)
        powers = np.zeros(len(channel_gains), dtype=np.double)
        powers[nonzero_index] = powers_aux
        i = np.argmin(powers)
        nonzero_index[i] = False
    return powers[powers >= 0]


def MACtoBCtransformation(
    Hs: List[Matrix], MAC_Covs: List[Matrix], MAC_decoding_order: List[int]
) -> List[Matrix]:
    """MAC to BC conversion.

    Parameters
    ----------
    Hs
    MAC_Covs
    MAC_decoding_order

    Returns
    -------
    BC_Covs
    """
    BS_antennas = Hs[0].shape[1]
    BC_Covs = [np.zeros([BS_antennas, BS_antennas]) for _ in Hs]
    for k, ms in enumerate(MAC_decoding_order):
        H = Hs[ms]
        MS_antennas = H.shape[0]

        # Compute A
        temp_sum = np.eye(MS_antennas)
        for BC_Cov in BC_Covs:
            temp_sum = temp_sum + Hs[ms] @ BC_Cov @ Hs[ms].conj().T
        A = temp_sum

        # Compute B
        temp_sum = np.eye(BS_antennas)
        for j in MAC_decoding_order[k + 1:]:
            temp_sum = temp_sum + Hs[j].conj().T @ MAC_Covs[j] @ Hs[j]
        B = temp_sum

        # Take SVD of effective channel
        Hms_eff = inv_sqrtm(B) @ Hs[ms].conj().T @ inv_sqrtm(A)
        F, D, GH = np.linalg.svd(Hms_eff)
        Fms = F[:, : len(D)]
        Gms = GH[: len(D), :].conj().T
        np.testing.assert_almost_equal(Hms_eff, Fms @ np.diag(D) @ Gms.conj().T)
        # Compute downlink covariance (equation 35)
        BC_Cov = (
            inv_sqrtm(B)
            @ Fms
            @ Gms.conj().T
            @ sqrtm(A)
            @ MAC_Covs[ms]
            @ sqrtm(A)
            @ Gms
            @ Fms.conj().T
            @ inv_sqrtm(B)
        )
        BC_Covs[ms] = BC_Cov
    return BC_Covs


def ptp_capacity_cvx(H: Matrix, P: float, Z: Optional[Matrix] = None, rcond: float = 1e-6) -> Tuple[float, Matrix]:
    """Optimize transmit covariance for point-to-point MIMO link.

    Parameters
    ----------
    H: Matrix
        Channel matrix
    P: float
        Transmit power
    Z: Matrix
        Noise covariance

    Returns
    -------
    rate: float
        data rate
    Z: Matrix
        transmit covariance matrix
    """
    Nrx, Ntx = H.shape
    Cov = cp.Variable((Ntx, Ntx), hermitian=True)
    if Z is None:
        cost = cp.log_det(np.eye(Ntx) + Cov @ H.conj().T @ H)
    else:
        assert (Nrx, Nrx) == Z.shape
        cost = cp.log_det(np.eye(Ntx) + Cov @ H.conj().T @ pinv(Z, rcond, hermitian=True) @ H)
    power = cp.real(cp.trace(Cov)) <= P
    positivity = Cov >> 0
    constraints = [power, positivity]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value, Cov.value


def ptp_capacity(H: Matrix, P: float, Z: Optional[Matrix] = None, rcond: float = 1e-6) -> Tuple[float, Matrix]:
    """Optimize transmit covariance for point-to-point MIMO link.

    Parameters
    ----------
    H: Matrix
        Channel matrix
    P: float
        Transmit power
    Z: Matrix
        Noise covariance

    Returns
    -------
    rate: float
        data rate
    Z: Matrix
        transmit covariance matrix
    """
    if Z is None:
        HH_d = H.conj().T @ H
    else:
        HH_d = H.conj().T @ pinv(Z, rcond, hermitian=True) @ H
    ei_d, V_d = np.linalg.eigh(HH_d)
    ei_d = [max(e, 0) for e in ei_d]
    power = water_filling(ei_d, P)
    Cov = V_d @ np.diag(power) @ V_d.conj().T
    rate = sum(math.log(1 + p * e) for p, e in zip(power, ei_d))
    # TODO should return Uplink and Downlink covariance
    return rate, Cov


def sort_channels(
    Hs: List[Matrix], weights: List[float]
) -> Tuple[List[Matrix], List[float], List[int]]:
    """Sort channels in encoding order.

    Parameters
    ----------
    Hs: List[Matrix]
        channel matrices
    weights: List[float]
        user weights

    Returns
    -------
    Hs_sorted: List[Matrix]
        Sorted channel matrices
    alphas: List[float]
        Coefficients for MAC reformulation
    order
        Encoding order
    """
    order = argsort(weights)
    Hs = [Hs[k] for k in order[::-1]]
    weights_reverse = sorted(weights, reverse=True)
    alphas = [a - b for a, b in zip(weights_reverse, weights_reverse[1:] + [0])]
    return Hs, alphas, order


def MAC_cvx(Hs, P, weights, Omega=None):
    """Wrap MAC_cvc_with_hoise_sbgr."""
    rates, MAC_Covs, order, _ = MAC_cvx_with_noise_sbgr(Hs, P, weights, Omega)
    return rates, MAC_Covs, order


def MAC_cvx_with_noise_sbgr(Hs, P, weights, Omega=None):
    """Optimize uplink covriances to maximize weighted sum-rate.

    Parameters
    ----------
    Hs: List[Matrix]
        Channel Matrices
    P: float
        Transmit power constraint
    weights: List[float]
        User weights
    Omega: Optional[Matrix]
        Uplink noise covariance

    Returns
    -------
    rates: List[float]
        User rates
    MAC_Covs: List[Matrices]
        Transmit covariances
    order: List[int]
        Decodicng order

    """
    # TODO, this function does not handle a rank deficient noise very well
    MAC_Covs = []
    Xs = []
    Hs_sorted, alphas, order = sort_channels(Hs, weights)
    Hs_sorted_orig = Hs_sorted
    Nrx = Hs[0].shape[0]
    Omega = np.eye(Nrx) if Omega is None else Omega
    ei_O, V_O = np.linalg.eigh(Omega)
    above_cutoff = ei_O > 1e-3
    psigma_diag = ei_O[above_cutoff] ** -0.5
    V_u = V_O[:, above_cutoff]

    Hs_sorted_pinv = [(psigma_diag * V_u).conj().T @ H for H in Hs_sorted]
    Nrx_eff = len(psigma_diag)
    for H in Hs_sorted:
        Nrx, Ntx = H.shape
        MAC_Covs.append(cp.Variable((Ntx, Ntx), hermitian=True))
        Xs.append(cp.Variable((Nrx_eff, Nrx_eff), hermitian=True))
    cost = cp.sum([alpha * cp.log_det(X) for X, alpha in zip(Xs, alphas)])
    mat = []
    matrix_equal = np.eye(Nrx_eff)
    for X, MAC_Cov, H in zip(Xs, MAC_Covs, Hs_sorted_pinv):
        # H_e = (psigma_diag * V_u).conj().T @ H
        matrix_equal += H @ MAC_Cov @ H.conj().T
        mat.append(X << matrix_equal)
    power = cp.sum([cp.real(cp.trace(MAC_Cov)) for MAC_Cov in MAC_Covs]) <= P
    positivity = [(MAC_Cov >> 0) for MAC_Cov in MAC_Covs]
    positivity += [(X >> 0) for X in Xs]
    constraints = mat + [power] + positivity
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-12)
    assert "optimal" in prob.status
    MAC_Covs_sorted = [Cov.value for Cov in MAC_Covs]
    MAC_Covs = [None for _ in order]
    for Cov, o in zip(MAC_Covs_sorted, order[::-1]):
        MAC_Covs[o] = Cov
    rates_sorted, _ = MAC_rates_ordered(MAC_Covs_sorted, Hs_sorted_pinv)
    rates = [None for _ in order]
    for r, o in zip(rates_sorted, order[::-1]):
        rates[o] = r
    wsr = sum([w * r for w, r in zip(weights, rates)])
    if not wsr == pytest.approx(prob.value, 1e-1):
        rates = MAC_rates(MAC_Covs, Hs, order)
    Omega_sbgr = -max(weights) * pinv(Omega)
    Zs = Omega
    # alphas[1] * logdet(Xs[1].value) - alphas[0] * logdet(Xs[0].value) - min(weights) * logdet(Omega)
    for a, MAC_Cov, H in zip(alphas, MAC_Covs_sorted, Hs_sorted_orig):
        Zs = Zs + H @ MAC_Cov @ H.conj().T
        Omega_sbgr = Omega_sbgr + a * pinv(Zs)
    return rates, MAC_Covs, order, Omega_sbgr


# def MAC_rates(MAC_Covs, Hs, MAC_decoding_order, Omega=None):
#    return MAC_rates_withZs(MAC_Covs, Hs, MAC_decoding_order, Omega)[0]


def MAC_rates(MAC_Covs: List[Matrix], Hs: List[Matrix],
              MAC_decoding_order: List[int],
              Omega: Optional[Matrix] = None) -> List[float]:
    """Compute uplink data rates.

    Parameters
    ----------
    MAC_Covs: List[Matrix]
        Uplink transmit covriances
    Hs: List[Matrix]
        Channel Matrices
    MAC_decoding_order: List[int]
        Decoding order
    Omega: Optimal[Matrix]
        Uplink noise covariance

    Returns
    -------
    rates: List[float]
        Data rates

    """
    order: List[int] = list(reversed(MAC_decoding_order))
    Hs_sorted = [Hs[k] for k in order]
    MAC_Covs_sorted = [MAC_Covs[k] for k in order]
    rates_sorted, Zs = MAC_rates_ordered(MAC_Covs_sorted, Hs_sorted, Omega)
    rates: List[float] = [0] * len(order)
    for r, o in zip(rates_sorted, order):
        rates[o] = r
    return rates


def MAC_rates_ordered(MAC_Covs: List[Matrix],
                      Hs: List[Matrix],
                      Omega: Optional[Matrix] = None) -> Tuple[List[float], List[Matrix]]:
    """Compute uplink data rates.

    Parameters
    ----------
    MAC_Covs: List[Matrix]
        Transmit covariance matrices
    Hs: List[Matrix]
        Channel matrices
    Omega: Optional[Matrix]
        Uplink noise covriance

    Returns
    -------
    rates: List[float]
        Data rates
    Zs: List[Matrix]
        the ZZSS

    """
    Nrx = Hs[0].shape[0]
    Z = np.eye(Nrx) if Omega is None else Omega
    e = np.real(np.linalg.eigvalsh(Z))
    v = log(np.product(e))
    rates = []
    Zs = []
    for MAC_Cov, H in zip(MAC_Covs, Hs):
        Znew = Z + H @ MAC_Cov @ H.conj().T
        Zs.append(Znew)
        # compute rate = logdet(Znew) - logdet(Z)
        e = np.real(np.linalg.eigvalsh(Znew))
        vnew = log(np.product(e))
        rate = max(vnew - v, 0)
        rates.append(rate)
        Z = Znew
        v = vnew
    return rates, Zs


def BC_rates(BC_Covs: List[Matrix], Hs: List[Matrix], BC_encoding_order: List[int]) -> List[Optional[float]]:
    """Compute data rates for broadcast channel.

    Parameters
    ----------
    BC_Covs: List[Matrix]
        Broadcast transmit covariances
    Hs: List[Matrix]
        Channel matrices
    BC_encoding_order: List[int]
        Encoding order

    Returns
    -------
    rates: List[float]
        Data rates

    """
    rates: List[Optional[float]] = [None for _ in BC_encoding_order]
    Ntx = Hs[0].shape[1]
    Sum_INT = np.zeros([Ntx, Ntx])
    for user in reversed(BC_encoding_order):
        BC_Cov = BC_Covs[user]
        H = Hs[user]
        Nrx = H.shape[0]
        IPN = np.eye(Nrx) + H @ Sum_INT @ H.conj().T
        rate = logdet(np.eye(Nrx) + H @ BC_Cov @ H.conj().T @ np.linalg.inv(IPN))
        rates[user] = max(0, rate)
        Sum_INT = Sum_INT + BC_Cov
    assert all(rates is not None for r in rates)
    return rates


def project_eigenvalues_to_given_sum_cvx(eis, P):
    """Project eigenvalues to power constraint.

    Parameters
    ----------
    eis: List[float]
        Eigenvalues
    P: float
        Power constraint

    Returns
    -------
    projected_eis: List[float]
        Projected eigenvalues
    """
    x = cp.Variable(len(eis))
    obj = cp.Minimize(cp.sum_squares(eis - x))
    constraints = [x >= 0, cp.sum(x) == P]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-15)
    return x.value


def project_eigenvalues_to_given_sum(eis: List[float], P: float) -> List[float]:
    """Project eigenvalues to power constraint.

    Parameters
    ----------
    eis: List[float]
        Eigenvalues
    P: float
        Power constraint

    Returns
    -------
    projected_eis: List[float]
        Projected eigenvalues
    """
    assert np.all(np.imag(eis) == 0)
    # sort them
    sorted_eigenvalues = sorted(np.real(eis))

    # find current sum
    current_sum = sum(sorted_eigenvalues)
    # find number of nonzero entries after subtraction
    for values_smaller_zero, val in enumerate(sorted_eigenvalues):

        values_geq_zero = len(sorted_eigenvalues) - values_smaller_zero
        assert len(sorted_eigenvalues[values_smaller_zero:]) == values_geq_zero
        # check if this is enough
        max_sum_reduction = val * values_geq_zero + sum(
            sorted_eigenvalues[:values_smaller_zero]
        )

        if max_sum_reduction > (current_sum - P):
            break

    reduction = (
        current_sum - P - sum(sorted_eigenvalues[:values_smaller_zero])
    ) / values_geq_zero
    projected_eigenvalues = [e - reduction for e in eis]
    projected_eigenvalues[projected_eigenvalues < 0.0] = 0
    return projected_eigenvalues


def project_covariance_cvx(Xs: List[Matrix], P: float) -> List[Matrix]:
    """Project covariance matrices to power constraint.

    Implements the optimization of :func:`~mcm.mimo.project_convariances` in CVX.

    Parameters
    ----------
    Xs
    P

    Returns
    -------
    projected_Covs

    """
    Covs = [cp.Variable((X.shape[0], X.shape[0]), hermitian=True) for X in Xs]
    obj = cp.Minimize(cp.sum([cp.sum_squares(Cov - X) for Cov, X in zip(Covs, Xs)]))
    power = cp.sum([cp.real(cp.trace(Cov)) for Cov in Covs]) <= P
    positivity = [(Cov >> 0) for Cov in Covs]
    constraints = [power] + positivity
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return [Cov.value for Cov in Covs]


def project_covariances(Covs: List[Matrix], P: float) -> List[Matrix]:
    r"""Project covariance matrices to power constraint..

    .. math::

     \min_{\Omega} \lbrace \sum_{k \in K} \norm{C_k-X_k}_{F}^{2} : \sum_{k \in K}\tr(C_k) \leq P, C_k \geq 0 \forall k \in K \rbrace

    Parameters
    ----------
    Covs
        Covariances to be projected
    P
        Target power constraint

    Returns
    -------
    List[Matrix]
        Projected Covariances


    """
    eigs = []
    VVs = []
    for uMAC_Cov in Covs:
        eig, VV = np.linalg.eigh(uMAC_Cov)
        eigs.append(eig)
        VVs.append(VV)
    # project to constraint set: operate on eigenvalues
    # flatten a list of lists https://stackoverflow.com/a/952952
    eigenvalues = [item for sublist in eigs for item in sublist]
    projected = project_eigenvalues_to_given_sum(eigenvalues, P)

    if sum(eigenvalues) <= P:
        return Covs
    assert sum(projected) <= P * 1.01
    # update
    Covs = []
    sum_eigs = 0.0
    offset = 0
    for eig, VV in zip(eigs, VVs):
        new_eigs = projected[offset: offset + len(eig)]
        offset += len(eig)
        sum_eigs += sum(new_eigs)
        Covs.append(VV @ np.diag(new_eigs) @ VV.conj().T)
    assert sum_eigs <= P * 1.01
    return Covs


def MAC(
    Hs: List[Matrix],
    P: float,
    weights: List[float],
    Omega: Optional[Matrix] = None,
    rate_threshold: float = 1e-6,
    max_iterations: int = 30,
) -> Tuple[List[float], List[Optional[Matrix]], List[int]]:
    r"""
    Optimize uplink transmit covariances.

    Given channel matrices :math:`(H_1, \ldots, H_K) `

    .. math::

        [\Sigma_1, \ldots, \Sigma_K] = \
        \max_{\Sigma_1,\ldots,\Sigma} \lbrace \sum_{k \in K} w_k r_k(\Sigma_1,\ldots,\Sigma_k): \
        \sum_{k\in K} \tr(\Sigma_k) < P, \
        \Sigma_k in \geq 0 \forall k \in K \rbrace


    Parameters
    ----------
    Hs: List[Matrix]
        Channel Matrices
    P: float
        Transmit power constraint
    weights: List[float]
        User weights
    Omega: Optional[Matrix]
        Uplink noise covariance
    rate_threshold: float
        Threshold for rate change to continue optimization
    max_iterations: int
        Limit on the numnber of iterations

    Returns
    -------
    rates: List[float]
        User rates
    MAC_Covs: List[Matrices]
        Transmit covariances
    order: List[int]
        Decodicng order

    """
    # start with no rate change
    rate_change = 0.0

    Omega = np.eye(Hs[0].shape[0]) if Omega is None else Omega
    Hs_sorted, alphas, order = sort_channels(Hs, weights)
    sum_transmit_antennas = sum(H.shape[1] for H in Hs_sorted)
    MAC_Covs_sorted = [
        P / sum_transmit_antennas * np.eye(H.shape[1]) for H in Hs_sorted
    ]

    # initialize step_size parameter
    d = 1
    d0 = 1
    # compute Z matrix
    for outer_i in range(max_iterations):
        rates, Zs = MAC_rates_ordered(MAC_Covs_sorted, Hs_sorted, Omega)
        last_weighted_rate: float = sum(
            [w * r for w, r in zip(sorted(weights, reverse=True), rates)]
        )
        # precompute inverses
        Z_invs = [np.linalg.inv(Z) for Z in Zs]

        # compute the gradients
        dMAC_Covs = []
        for k, (MAC_Cov, H) in enumerate(zip(MAC_Covs_sorted, Hs_sorted)):
            dMAC_Cov = np.zeros([MAC_Cov.shape[0], MAC_Cov.shape[0]])
            for Z_inv, alpha in zip(Z_invs[k:], alphas[k:]):
                dMAC_Cov = dMAC_Cov + alpha * H.conj().T @ Z_inv @ H
            dMAC_Covs.append(dMAC_Cov)
        dMAC_Covs_trace = sum([np.trace(dMAC_Cov) for dMAC_Cov in dMAC_Covs])
        dMAC_Covs = [P / dMAC_Covs_trace * dMAC_Cov for dMAC_Cov in dMAC_Covs]
        for inner_i in range(max_iterations):
            # now update the covariances
            updated_MAC_Covs = [
                MAC_Cov + d0 / d * dMAC_Cov
                for MAC_Cov, dMAC_Cov in zip(MAC_Covs_sorted, dMAC_Covs)
            ]
            updated_MAC_Covs = project_covariances(updated_MAC_Covs, P)
            rates = MAC_rates_ordered(updated_MAC_Covs, Hs_sorted, Omega)[0]
            this_weighted_rate: float = sum(
                [w * r for w, r in zip(sorted(weights, reverse=True), rates)]
            )
            LOGGER.info(
                f"Competed inner iteration {inner_i} - current obj {this_weighted_rate} - last obj {last_weighted_rate}"
            )
            if this_weighted_rate < last_weighted_rate:
                d += 1
            else:
                rate_change = this_weighted_rate - last_weighted_rate
                MAC_Covs_sorted = updated_MAC_Covs
                break
        LOGGER.info(
            f"Competed outer iteration {outer_i} - current obj {this_weighted_rate} - rate change {rate_change}"
        )
        if rate_change / this_weighted_rate < rate_threshold:
            break

    MAC_Covs: List[Optional[Matrix]] = [None for _ in order]
    for Cov, o in zip(MAC_Covs_sorted, order[::-1]):
        MAC_Covs[o] = Cov
    rates = MAC_rates(MAC_Covs, Hs, order, Omega)

    return rates, MAC_Covs, order
