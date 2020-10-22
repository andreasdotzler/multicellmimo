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
    logdet,
)

LOGGER = logging.getLogger(__name__)
inv = np.linalg.inv
det = np.linalg.det
log = np.log
eye = np.eye


@pytest.fixture(scope="function", autouse=True)
def seed():
    np.random.seed(42)


@pytest.mark.parametrize("comp", [0, 1])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
def test_uplink_noise_dual(comp, Bs_antennas):
    Ms_antennas = 2
    P = 100
    sigma = 5
    H = (
        np.random.random([Ms_antennas, Bs_antennas])
        + comp * np.random.random([Ms_antennas, Bs_antennas]) * 1j
    )
    # create random uplink noise covariance Z with tr(Z) = sigma
    R = (
        np.random.random([Bs_antennas, Bs_antennas])
        + comp * np.random.random([Bs_antennas, Bs_antennas]) * 1j
    )
    Z = R @ R.conj().T
    Z = sigma * Z / np.trace(Z)
    assert np.real(np.trace(Z)) == pytest.approx(sigma)
    H_eff = inv_sqrtm(Z) @ H.conj().T
    rate_p, Sigma_eff = ptp_capacity_cvx(H_eff, P)
    rate_i, Sigma = ptp_capacity_cvx(H.conj().T, P, Z)
    assert rate_i == pytest.approx(rate_p, 1e-3)
    rate_d, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
    L = sqrtm(Z) @ W @ sqrtm(Z)
    X = inv(L)
    assert log(det(X)) / log(2) == pytest.approx(rate_i, 1e-2)
    assert rate_i == pytest.approx(rate_d, 1e-3)


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
