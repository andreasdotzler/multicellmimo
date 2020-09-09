import itertools
import logging
import random

import numpy as np
import pytest

from .mimo_worst_case_noise import (
    ptp_capacity_uplink_cvx_dual,
    ptp_worst_case_noise,
    ptp_worst_case_noise_gradient,
    ptp_worst_case_noise_constraint,
    ptp_worst_case_noise_approx,
    ptp_capacity_mimimax,
)

from .mimo import (
    ptp_capacity,
    ptp_capacity_cvx,
    inv_sqrtm,
    sqrtm,
)

LOGGER = logging.getLogger(__name__)
inv = np.linalg.inv
det = np.linalg.det
log = np.log
eye = np.eye


@pytest.fixture(scope="function", autouse=True)
def seed():
    np.random.seed(42)


@pytest.mark.parametrize("Ms_antennas", [3])
@pytest.mark.parametrize("Bs_antennas", [3])
def test_minimax_ptp(Ms_antennas, Bs_antennas):
    P = 100
    H = (
        np.random.random([Ms_antennas, Bs_antennas])
        # + np.random.random([Ms_antennas, Bs_antennas]) * 1j
    )
    assert Bs_antennas == Ms_antennas, "not implemented"
    C = np.eye(Bs_antennas) * P / Bs_antennas
    R = B = np.eye(Ms_antennas) * P / Ms_antennas
    rate_d, Q, (Omega, tr, K) = ptp_capacity_mimimax(H=H, R=R, C=C, P=P)
    Sigma = -inv(R + H @ Q @ H.conj().T) + inv(R)
    rate_u = logdet(
        np.eye(Bs_antennas) + np.linalg.inv(Omega) @ H.conj().T @ Sigma @ H
    ) / np.log(2)
    assert rate_u == pytest.approx(rate_d, 1e-3)
    Phi_1 = H.conj().T @ inv(R + H @ Q @ H.conj().T) @ H + K
    Phi_2 = Omega

    # include subspace parameterization


@pytest.mark.parametrize("Ms_antennas", [1, 2, 3])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
def test_ptp_dual(Ms_antennas, Bs_antennas):
    P = 100
    sigma = 1

    H = (
        np.random.random([Ms_antennas, Bs_antennas])
        # + np.random.random([Ms_antennas, Bs_antennas]) * 1j
    )
    Z = sigma / Bs_antennas * np.eye(Bs_antennas)
    assert np.trace(Z) == pytest.approx(sigma)
    rate_p, Q_p = ptp_capacity(H, P)
    assert np.trace(Q_p) == pytest.approx(P)
    rate_d = ptp_capacity_uplink_cvx_dual(H, P / Bs_antennas, Z)[0]
    assert rate_p == pytest.approx(rate_d, 1e-3)

    # create random uplink noise
    R = np.random.random([Bs_antennas, Bs_antennas])
    Z = R @ R.T
    Z = sigma / Bs_antennas * Z
    rate_d = ptp_capacity_uplink_cvx_dual(H, P, Z)[0]
    H_eff = inv_sqrtm(Z) @ H.T
    rate_p, Sigma_eff = ptp_capacity(H_eff, P / sigma)
    rate_c = (log(det(Z + H.T @ Sigma_eff @ H)) - log(det(Z))) / np.log(2)
    assert rate_c == pytest.approx(rate_p, 1e-3)


# worst case noise, like white
@pytest.mark.parametrize("Ms_antennas", [1, 2, 3])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
def test_ptp_worstcase(Ms_antennas, Bs_antennas):
    H = (
        np.random.random([Ms_antennas, Bs_antennas])
        # np.eye( Bs_antennas)
        # + np.random.random([Ms_antennas, Bs_antennas]) * 1j
    )
    P = 1
    sigma = 1
    # we know the worst case capacity is equal to no channel knowledge and
    # selecting a white channel matrix
    Sigma_no_channel = P / Ms_antennas * eye(Ms_antennas)
    Z_no_channel = sigma / Bs_antennas * eye(Bs_antennas)

    assert np.trace(Sigma_no_channel) == pytest.approx(P)
    assert np.trace(Z_no_channel) == pytest.approx(sigma)

    rate_no_channel = log(det(eye(Bs_antennas) + P / sigma * H.T @ H)) / np.log(2)

    rate_worst_case, (Z, W) = ptp_worst_case_noise_approx(H, P, sigma)
    L = sqrtm(Z) @ W @ sqrtm(Z)
    rate_worst_case_dual = -log(det(L)) / np.log(2)
    assert rate_worst_case == pytest.approx(rate_worst_case_dual, 1e-2)
    assert rate_worst_case == pytest.approx(rate_no_channel, 1e-2)

    H_eff = inv_sqrtm(Z) @ H.T
    rate_e, Q = ptp_capacity(H_eff, P / sigma * np.trace(Z))
    assert np.trace(Q) == pytest.approx(P / sigma * np.trace(Z))
    assert rate_worst_case == pytest.approx(rate_e, 1e-2)
