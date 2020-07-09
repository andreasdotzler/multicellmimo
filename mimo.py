import logging
import math

import cvxpy as cp
import numpy as np
import pytest

LOGGER = logging.getLogger(__name__)


def MACtoBCtransformation(Hs, MAC_Cov, order):
    BS_antennas = Hs[0].shape[1]
    BC_Cov = [None for _ in MAC_Cov]
    for ms, k in enumerate(order):
        H = Hs[k]
        MS_antennas = H.shape[0]

        # Compute B
        temp_sum = np.eye(BS_antennas)
        for ll in order[k + 1 :]:
            temp_sum = temp_sum + Hs[ll].conj().T @ MAC_Cov[ll] @ Hs[ll]
        B = temp_sum

        # Compute A
        temp_sum = np.eye(MS_antennas)
        for ll in order[0:k]:
            temp_sum = temp_sum + Hs[k] @ BC_Cov[ll] @ Hs[k].conj().T
        A = temp_sum

        # Take SVD of effective channel
        Hms_eff = inv_sqrtm(B) @ Hs[k].conj().T @ inv_sqrtm(A)
        F, D, GH = np.linalg.svd(Hms_eff)
        Fms = F[:, : len(D)]
        Gms = GH[: len(D), :].conj().T
        np.testing.assert_almost_equal(Hms_eff, Fms @ np.diag(D) @ Gms.conj().T)
        # Compute downlink covariance (equation 35)
        BC_Cov[k] = (
            inv_sqrtm(B)
            @ Fms
            @ Gms.conj().T
            @ inv_sqrtm(A)
            @ MAC_Cov[k]
            @ inv_sqrtm(A)
            @ Gms
            @ Fms.conj().T
            @ inv_sqrtm(B)
        )
    return BC_Cov


def inv_sqrtm(A):
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d ** -0.5) @ V_d.conj().T


def ptp_capacity_cvx(H, P):
    Nrx, Ntx = H.shape
    Q = cp.Variable([Ntx, Ntx], complex=True)
    I = np.eye(Ntx)
    cost = cp.log_det(I + Q @ H.conj().T @ H)
    power = cp.real(cp.trace(Q)) <= P
    positivity = Q >> 0
    constraints = [power, positivity]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value / np.log(2), Q.value


def ptp_capacity(H, P):
    HH_d = H.conj().T @ H
    ei_d, V_d = np.linalg.eigh(HH_d)
    ei_d = [max(e, 0) for e in ei_d]
    power = water_filling_iter(ei_d, P)
    Q = V_d @ np.diag(power) @ V_d.conj().T
    rate = sum(math.log(1 + p * e, 2) for p, e in zip(power, ei_d))
    return rate, Q


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


def sort_channels(Hs, weights):
    # sort channels
    # https://stackoverflow.com/a/6618543
    Hs = [
        H for _, H in sorted(zip(weights, Hs), reverse=True, key=lambda pair: pair[0])
    ]
    alphas = (-np.diff(sorted(weights, reverse=True))).tolist()
    alphas.append(weights[-1])
    return Hs, alphas


def MAC(Hs, P, weights):
    Qs = []
    Xs = []
    Hs, alphas = sort_channels(Hs, weights)

    for H in Hs:
        Nrx, Ntx = H.shape
        Qs.append(cp.Variable([Ntx, Ntx], complex=True))
        Xs.append(cp.Variable([Nrx, Nrx], complex=True))
    I = np.eye(Nrx)
    cost = cp.sum([alpha * cp.log_det(X) for X, alpha in zip(Xs, alphas)])
    # cost = cp.sum([cp.log_det(I + Q @ H.conj().T @ H) for Q in Qs])
    mat = []
    matrix_equal = I
    for X, Q, H in zip(Xs, Qs, Hs):
        matrix_equal += H @ Q @ H.conj().T
        mat.append(X << matrix_equal)
    power = cp.sum([cp.real(cp.trace(Q)) for Q in Qs]) <= P
    positivity = [(Q >> 0) for Q in Qs]
    positivity += [(X >> 0) for X in Xs]
    constraints = mat + [power] + positivity
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-12)
    Qs = [Q.value for Q in Qs]
    return sum_rate(Qs, Hs)[0], Qs


def sum_rate(Qs, Hs):
    Nrx = Hs[0].shape[0]
    Z = np.eye(Nrx)
    rates = []
    Zs = []
    for Q, H in zip(Qs, Hs):
        Znew = Z + H @ Q @ H.conj().T
        Zs.append(Znew)
        rate = (np.linalg.slogdet(Znew)[1] - np.linalg.slogdet(Z)[1]) / np.log(2)
        rates.append(rate)
        Z = Znew
    return rates, Zs


def project_eigenvalues_to_given_sum_cvx(e, P):
    # setup the objective and constraints and solve the problem
    x = cp.Variable(len(e))
    obj = cp.Minimize(cp.sum_squares(e - x))
    constr = [x >= 0, cp.sum(x) == P]
    prob = cp.Problem(obj, constr)
    prob.solve()
    return np.array(x.value).squeeze()


def project_Q_cvx(X, P):
    Q = cp.Variable([X.shape[0], X.shape[0]], complex=True)
    obj = cp.Minimize(cp.sum_squares(Q - X))
    power = cp.real(cp.trace(Q)) <= P
    positivity = Q >> 0
    constraints = [power, positivity]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return Q.value


def maximize_weighted_sum_rate_in_Q(
    Hs, P, weights, rate_threshold=1e-6, max_iterations=30
):
    # start with no rate change
    rate_change = 0
    # initialize outer and inner iterations
    outer_iterations = 0
    inner_iterations = 0
    Hs, alphas = sort_channels(Hs, weights)
    sum_transmit_antennas = sum(H.shape[1] for H in Hs)
    Nrx = Hs[0].shape[0]
    Qs = [P / sum_transmit_antennas * np.eye(H.shape[1]) for H in Hs]

    # initialize step_size parameter
    d = 1
    d0 = 1
    # compute Z matrix
    for outer_i in range(max_iterations):
        rates, Zs = sum_rate(Qs, Hs)
        last_weighted_rate = sum(
            [w * r for w, r in zip(sorted(weights, reverse=True), rates)]
        )
        # precompute inverses
        Z_invs = [np.linalg.inv(Z) for Z in Zs]
        I = np.eye(Nrx)

        # compute the gradients^*

        dQ_trace = 0
        dQs = []
        for k, (Q, H) in enumerate(zip(Qs, Hs)):
            dQ = np.zeros([Q.shape[0], Q.shape[0]])
            for Z_inv, alpha in zip(Z_invs[k:], alphas[k:]):
                dQ = dQ + 1 / np.log(2) * alpha * H.conj().T @ Z_inv @ H
            dQs.append(dQ)
        dQ_trace = sum([np.trace(dQ) for dQ in dQs])
        dQs = [P / dQ_trace * dQ for dQ in dQs]
        for inner_i in range(max_iterations):
            # now update the covariances
            updated_Qs = [Q + d0 / d * dQ for Q, dQ in zip(Qs, dQs)]
            Q_cvx = project_Q_cvx(updated_Qs[0], P)
            eigs = []
            VVs = []
            for uQ in updated_Qs:
                eig, VV = np.linalg.eigh(uQ)
                eigs.append(eig)
                VVs.append(VV)
            # project to constraint set: operate on eigenvalues
            # flatten a list of lists https://stackoverflow.com/a/952952
            eigenvalues = [item for sublist in eigs for item in sublist]
            projected = project_eigenvalues_to_given_sum_cvx(eigenvalues, P)
            assert sum(projected) <= P * 1.01
            # FIXME; the hole projection needs rework
            subtract = (sum(eigenvalues) - P) / sum(projected > 1e-3)
            # update
            updated_Qs = []
            sum_eigs = 0
            offset = 0
            for eig, VV in zip(eigs, VVs):
                new_eigs = projected[offset : offset + len(eig)]
                offset += len(eig)
                sum_eigs += sum(new_eigs)
                updated_Qs.append(VV @ np.diag(new_eigs) @ VV.conj().T)
            assert sum_eigs <= P * 1.01
            rates = sum_rate(updated_Qs, Hs)[0]
            this_weighted_rate = sum(
                [w * r for w, r in zip(sorted(weights, reverse=True), rates)]
            )
            this_weighted_rate_cvs = sum(sum_rate([Q_cvx], Hs)[0])
            LOGGER.info(
                f"Competed inner iteration {inner_i} - current obj {this_weighted_rate} - last obj {last_weighted_rate}"
            )
            if this_weighted_rate < last_weighted_rate:
                d += 1
            else:
                rate_change = this_weighted_rate - last_weighted_rate
                last_weighted_rate = this_weighted_rate
                Qs = updated_Qs
                break
        LOGGER.info(
            f"Competed outer iteration {outer_i} - current obj {this_weighted_rate} - rate change {rate_change}"
        )
        if rate_change / this_weighted_rate < rate_threshold:
            break
    rates = sum_rate(Qs, Hs)[0]
    return rates, Qs
