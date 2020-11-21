import itertools
import logging
import random

import numpy as np
import pytest

from .mimo_worst_case_noise import (
    ptp_worst_case_noise_approx,
    ptp_capacity_mimimax,
)

from .mimo import (
    ptp_capacity,
    ptp_capacity_cvx,
)

from .utils import (
    inv_sqrtm,
    pinv_sqrtm,
    sqrtm,
    logdet,
)

LOGGER = logging.getLogger(__name__)
inv = np.linalg.inv
pinv = np.linalg.pinv
det = np.linalg.det
log = np.log
eye = np.eye


@pytest.fixture(scope="function", autouse=True)
def seed():
    np.random.seed(45)


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
def test_noise_rank_def(comp):
    P = 100
    H = np.array([[1, 0], [0, 1], [0, 0]]) + comp * 1j * np.array(
        [[1, 0], [0, 1], [0, 0]]
    )
    assert (3, 2) == H.shape
    Z = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 0]])
    rate_i, Sigma_i = ptp_capacity(H, P, Z)
    X_i = eye(3) + pinv_sqrtm(Z) @ H @ Sigma_i @ H.conj().T @ pinv_sqrtm(Z)
    assert log(det(X_i)) / log(2) == pytest.approx(rate_i, 1e-2)
    W_i = pinv_sqrtm(Z) @ inv(X_i) @ pinv_sqrtm(Z)
    assert np.allclose(pinv_sqrtm(Z) @ pinv_sqrtm(Z), pinv(Z))

    H_red = H[[0, 1], :]
    Z_red = Z[[0, 1], :][:, [0, 1]]
    rate_r, Sigma_r = ptp_capacity(H_red, P, Z_red)
    assert rate_r == pytest.approx(rate_i, 1e-2)
    H_full = np.array([[1, 0], [0, 1], [1, 1]])
    inf_lim = 1e-5
    rate_d, inf_constraint = ptp_capacity(H_full, P, Z, inf_lim=inf_lim)
    assert rate_d == np.inf
    assert np.allclose(
        inf_constraint[0], [[0, 0, 0], [0, 0, 0], [0, 0, inf_lim]]
    )  # + comp * 1j * [[1, 0, 0],[0, 1, 0], [0, 0, 0]]
    rate_f, _ = ptp_capacity(H_full, P, Z + inf_constraint[0], inf_lim=inf_lim)
    assert rate_f != np.inf
    import ipdb; ipdb.set_trace()
    rate_worst_case, (Z, W) = ptp_worst_case_noise_approx(H_full.T,P,6)


def test_noise_rank_def2():
    A = np.random.random([6, 4]) + 1j * np.random.random([6, 4])
    A = A @ A.conj().T
    import scipy.linalg

    E, K = scipy.linalg.schur(A)
    e = np.real(E.diagonal())
    pos = e > 1e-9
    T = K[:, pos]
    np.allclose(np.linalg.pinv(A), T @ np.diag(1 / e[pos]) @ T.conj().T)
    np.allclose(np.linalg.pinv(A), T @ inv(T.conj().T @ A @ T) @ T.conj().T)
    Ai = inv(A)
    print(sorted(np.real(np.linalg.eigvals(Ai))))


# worst case noise, like white
@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [1, 2, 3])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
def test_ptp_worstcase(comp, Ms_antennas, Bs_antennas):
    H = (
        np.random.random([Ms_antennas, Bs_antennas])
        + comp * 1j *np.random.random([Ms_antennas, Bs_antennas]) 
    )
    P = 1
    sigma = 1
    # we know the worst case capacity is equal to no channel knowledge and
    # selecting a white channel matrix
    Sigma_no_channel = P / Ms_antennas * eye(Ms_antennas)
    Z_no_channel = sigma / Bs_antennas * eye(Bs_antennas)

    assert np.trace(Sigma_no_channel) == pytest.approx(P)
    assert np.trace(Z_no_channel) == pytest.approx(sigma)

    rate_no_channel = np.real(log(det(eye(Bs_antennas) + P / sigma * H.conj().T @ H))) 
    rate_worst_case, (Z, W) = ptp_worst_case_noise_approx(H, P, sigma)
    assert np.allclose(rate_worst_case, rate_no_channel, rtol=1e-1)
    rate_calc = ptp_capacity(H.conj().T, P, Z)[0] 
    assert rate_worst_case == pytest.approx(rate_calc, 1e-2)



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
