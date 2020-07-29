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


def MACtoBCtransformation(Hs, MAC_Cov, MAC_decoding_order):
    BS_antennas = Hs[0].shape[1]
    BC_Covs = []
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
        for j in MAC_decoding_order[k + 1 :]:
            temp_sum = temp_sum + Hs[j].conj().T @ MAC_Cov[j] @ Hs[j]
        B = temp_sum

        # Take SVD of effective channel
        Hms_eff = inv_sqrtm(B) @ Hs[ms].conj().T @ inv_sqrtm(A)
        F, D, GH = np.linalg.svd(Hms_eff)
        Fms = F[:, : len(D)]
        Gms = GH[: len(D), :].conj().T
        np.testing.assert_almost_equal(Hms_eff, Fms @ np.diag(D) @ Gms.conj().T)
        # Compute downlink covariance (equation 35)
        BC_Covs.append(
            (
                inv_sqrtm(B)
                @ Fms
                @ Gms.conj().T
                @ sqrtm(A)
                @ MAC_Cov[ms]
                @ sqrtm(A)
                @ Gms
                @ Fms.conj().T
                @ inv_sqrtm(B)
            )
        )
    return [BC_Covs[k] for k in MAC_decoding_order]


def inv_sqrtm(A):
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d ** -0.5) @ V_d.conj().T


def sqrtm(A):
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d ** 0.5) @ V_d.conj().T


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
        rate_i, (W, la) = ptp_capacity_cvx_dual(H, P / sigma, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        Z = inv(W + la * P / sigma * np.eye(Ntx))
        assert np.all(np.linalg.eigvals(Z) > 0)
        LOGGER.debug(f"rate inner: {rate_i} : rate outer {rate_o}")
    return rate_i, (Z, W)


def ptp_capacity_cvx_dual(H, Ps, Z):
    Nrx, Ntx = H.shape
    W = cp.Variable([Ntx, Ntx], symmetric=True)
    la = cp.Variable(1)
    cost = -cp.log_det(W) - log(det(Z)) + cp.trace(Z @ W) + Ps * la * np.trace(Z) - Ntx
    positivity_W = W >> 0
    postitvity_la = la >= 0
    cons = cp.multiply(la, np.eye(Nrx)) >> H @ W @ H.T
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


def ptp_capacity_cvx(H, P):
    Nrx, Ntx = H.shape
    Cov = cp.Variable([Ntx, Ntx], complex=True)
    I = np.eye(Ntx)
    cost = cp.log_det(I + Cov @ H.conj().T @ H)
    power = cp.real(cp.trace(Cov)) <= P
    positivity = Cov >> 0
    constraints = [power, positivity]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value / np.log(2), Cov.value


def ptp_capacity(H, P):
    HH_d = H.conj().T @ H
    ei_d, V_d = np.linalg.eigh(HH_d)
    ei_d = [max(e, 0) for e in ei_d]
    power = water_filling_iter(ei_d, P)
    Cov = V_d @ np.diag(power) @ V_d.conj().T
    rate = sum(math.log(1 + p * e, 2) for p, e in zip(power, ei_d))
    # TODO should return Uplink and Downlink covariance
    return rate, Cov


def water_filling_cvx(ei, P):
    assert all([e.imag == 0 for e in ei])
    power = cp.Variable(len(ei))
    alpha = cp.Parameter(len(ei), nonneg=True)
    alpha.value = ei
    obj = cp.Maximize(cp.sum(cp.log(1 + cp.multiply(power, alpha))))
    constraints = [power >= 0, cp.sum(power) == P]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    if prob.status == "optimal":
        return power.value
    else:
        return np.nan


def water_filling_iter(ei, P):
    assert all([e.imag == 0 for e in ei])
    channel_gains = np.array(ei)
    nonzero_index = channel_gains >= 10e-16
    powers_aux = np.zeros(len(channel_gains))
    powers = np.zeros(len(channel_gains))
    while any(powers_aux <= 0):
        LOGGER.debug(f"Waterfilling: {powers_aux} : nonzero_index {nonzero_index}")
        channel_gains_aux = channel_gains[nonzero_index]
        eta = (P + sum(1 / channel_gains_aux)) / sum(nonzero_index)
        powers_aux = eta - (1 / channel_gains_aux)
        powers = np.zeros(len(channel_gains))
        powers[nonzero_index] = powers_aux
        i = np.argmin(powers)
        nonzero_index[i] = False
    return powers[powers >= 0]


def argsort(weights, reverse=False):
    # sort channels
    # https://stackoverflow.com/a/6618543
    return [
        o
        for o, _ in sorted(
            enumerate(weights), reverse=reverse, key=lambda pair: pair[1]
        )
    ]


def sort_channels(Hs, weights):
    order = argsort(weights)
    Hs = [Hs[k] for k in order]
    alphas = np.diff([weights[o] for o in order]).tolist()
    alphas.append(weights[order[0]])
    return Hs, alphas, order


def MAC_cvx(Hs, P, weights):
    MAC_Covs = []
    Xs = []
    Hs_sorted, alphas, order = sort_channels(Hs, weights)

    for H in Hs_sorted:
        Nrx, Ntx = H.shape
        MAC_Covs.append(cp.Variable([Ntx, Ntx], complex=True))
        Xs.append(cp.Variable([Nrx, Nrx], complex=True))
    I = np.eye(Nrx)
    cost = cp.sum([alpha * cp.log_det(X) for X, alpha in zip(Xs, alphas)])
    mat = []
    matrix_equal = I
    for X, MAC_Cov, H in zip(Xs, MAC_Covs, Hs_sorted):
        matrix_equal += H @ MAC_Cov @ H.conj().T
        mat.append(X << matrix_equal)
    power = cp.sum([cp.real(cp.trace(MAC_Cov)) for MAC_Cov in MAC_Covs]) <= P
    positivity = [(MAC_Cov >> 0) for MAC_Cov in MAC_Covs]
    positivity += [(X >> 0) for X in Xs]
    constraints = mat + [power] + positivity
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-12)
    MAC_Covs_sorted = [Cov.value for Cov in MAC_Covs]
    MAC_Covs = [None for _ in order]
    for Cov, o in zip(MAC_Covs_sorted, order):
        MAC_Covs[o] = Cov
    rates = MAC_rates(MAC_Covs, Hs, order)
    return rates, MAC_Covs, order


def MAC_rates(MAC_Covs, Hs, MAC_decoding_order):
    order = list(reversed(MAC_decoding_order))
    Hs_sorted = [Hs[k] for k in order]
    MAC_Covs_sorted = [MAC_Covs[k] for k in order]
    rates_sorted = MAC_rates_ordered(MAC_Covs_sorted, Hs_sorted)[0]
    rates = [None for _ in order]
    for r, o in zip(rates_sorted, order):
        rates[o] = r
    return rates


def MAC_rates_ordered(MAC_Covs, Hs):
    Nrx = Hs[0].shape[0]
    Z = np.eye(Nrx)
    rates = []
    Zs = []
    for MAC_Cov, H in zip(MAC_Covs, Hs):
        Znew = Z + H @ MAC_Cov @ H.conj().T
        Zs.append(Znew)
        rate = (np.linalg.slogdet(Znew)[1] - np.linalg.slogdet(Z)[1]) / np.log(2)
        rates.append(rate)
        Z = Znew
    return rates, Zs


def logdet(X):
    return np.linalg.slogdet(X)[1]


def BC_rates(BC_Covs, Hs, BC_encoding_order):
    rates = [None for _ in BC_encoding_order]
    Ntx = Hs[0].shape[1]
    Sum_INT = np.zeros([Ntx, Ntx])
    for user in reversed(BC_encoding_order):
        BC_Cov = BC_Covs[user]
        H = Hs[user]
        Nrx = H.shape[0]
        IPN = np.eye(Nrx) + H @ Sum_INT @ H.conj().T
        rate = (
            logdet(np.eye(Nrx) + H @ BC_Cov @ H.conj().T @ np.linalg.inv(IPN))
        ) / np.log(2)
        rates[user] = rate
        Sum_INT = Sum_INT + BC_Cov
    assert all(rates)
    return rates


def project_eigenvalues_to_given_sum_cvx(e, P):
    # setup the objective and constraints and solve the problem
    x = cp.Variable(len(e))
    obj = cp.Minimize(cp.sum_squares(e - x))
    constraints = [x >= 0, cp.sum(x) == P]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return x.value


def project_covariance_cvx(Xs, P):
    Covs = [cp.Variable([X.shape[0], X.shape[0]], complex=True) for X in Xs]
    obj = cp.Minimize(
        cp.sum([cp.real(cp.sum_squares(Cov - X)) for Cov, X in zip(Covs, Xs)])
    )
    power = cp.sum([cp.real(cp.trace(Cov)) for Cov in Covs]) <= P
    positivity = [(Cov >> 0) for Cov in Covs]
    constraints = [power] + positivity
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return [Cov.value for Cov in Covs]


def project_covariances(Covs, P):
    eigs = []
    VVs = []
    for uMAC_Cov in Covs:
        eig, VV = np.linalg.eigh(uMAC_Cov)
        eigs.append(eig)
        VVs.append(VV)
    # project to constraint set: operate on eigenvalues
    # flatten a list of lists https://stackoverflow.com/a/952952
    eigenvalues = [item for sublist in eigs for item in sublist]
    if sum(eigenvalues) <= P:
        return Covs
    # TODO, do this without CVX
    projected = project_eigenvalues_to_given_sum_cvx(eigenvalues, P)
    assert sum(projected) <= P * 1.01
    # update
    Covs = []
    sum_eigs = 0
    offset = 0
    for eig, VV in zip(eigs, VVs):
        new_eigs = projected[offset : offset + len(eig)]
        offset += len(eig)
        sum_eigs += sum(new_eigs)
        Covs.append(VV @ np.diag(new_eigs) @ VV.conj().T)
    assert sum_eigs <= P * 1.01
    return Covs


def MAC(Hs, P, weights, rate_threshold=1e-6, max_iterations=30):
    # start with no rate change
    rate_change = 0

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
        rates, Zs = MAC_rates_ordered(MAC_Covs_sorted, Hs_sorted)
        last_weighted_rate = sum(
            [w * r for w, r in zip(sorted(weights, reverse=True), rates)]
        )
        # precompute inverses
        Z_invs = [np.linalg.inv(Z) for Z in Zs]

        # compute the gradients
        dMAC_Covs = []
        for k, (MAC_Cov, H) in enumerate(zip(MAC_Covs_sorted, Hs_sorted)):
            dMAC_Cov = np.zeros([MAC_Cov.shape[0], MAC_Cov.shape[0]])
            for Z_inv, alpha in zip(Z_invs[k:], alphas[k:]):
                dMAC_Cov = dMAC_Cov + 1 / np.log(2) * alpha * H.conj().T @ Z_inv @ H
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
            rates = MAC_rates_ordered(updated_MAC_Covs, Hs_sorted)[0]
            this_weighted_rate = sum(
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

    MAC_Covs = [None for _ in order]
    for Cov, o in zip(MAC_Covs_sorted, order):
        MAC_Covs[o] = Cov
    rates = MAC_rates(MAC_Covs, Hs, order)
    return rates, MAC_Covs, order
