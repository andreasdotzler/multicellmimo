import itertools
import logging
import random
import math

import cvxpy as cp
import numpy as np
import pytest

from .context import mcm

from mcm.mimo_worst_case_noise import (
    ptp_worst_case_noise_approx,
    MAC_worst_case_noise_approx,
    ptp_worst_case_noise_static,
    ptp_capacity_minimax,
    noise_outer_approximation,
    approx_inner_MAC,
    inf_cons,
)

from mcm.mimo import (
    ptp_capacity,
    ptp_capacity_cvx,
    MACtoBCtransformation,
    water_filling,
    MAC_cvx_with_noise_sbgr,
)

from mcm.utils import (
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


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [3])
@pytest.mark.parametrize("Bs_antennas", [3])
@pytest.mark.parametrize("sigma", [1.2])
@pytest.mark.parametrize("P", [1.3])
def test_minimax_fixed_fixed(Ms_antennas, Bs_antennas, B, C, H, P, sigma, comp):
    # Lemma 3
    p = c = np.trace(C @ C)
    n = b = np.trace(B @ B)

    # Downlink fixed noise, downlink fixed covariance, uplink optim noise, optim covariance
    r_d_f_f = logdet(eye(Ms_antennas) + inv(B) @ H @ C @ H.conj().T)
    # SIGMA = inv_sqrtm(B) @ SIGMA' @ inv_sqrtm(B)
    # OMEGA = inv_sqrtm(C) @ OMEGA' @ inv_sqrtm(C)
    # logdet(I + inv(OMEGA) @ H.conj().T @ SIGMA @ H) =
    # = logdet(I + sqrtm(C) @ inv(OMEGA') @ sqrtm(C) @ H.conj().T @ inv_sqrtm(B) @ SIGMA' @ inv_sqrtm(B) @ H)
    # = logdet(I + inv(OMEGA') @ sqrtm(C) @ H.conj().T @ inv_sqrtm(B) @ SIGMA' @ inv_sqrtm(B) @ H @ sqrtm(C)
    # = logdet(I + inv(OMEGA') @ Heff.conj().T @ SIGMA' @ Heff)
    Heff = inv_sqrtm(B) @ H @ sqrtm(C)
    # SIGMA <= p B + Y , tr(Y)  = 0
    # => trace(B@SIGMA) <= p * np.trace(B@B)
    # => trace(SIGMA') <= p * np.trace(B@B)
    # OMEGA <= n C + Zp, tr(Zp) = 0
    # => trace(C@OMEGA) <= n * np.trace(C@C)
    # => trace(OMEGA') <= n * np.trace(C@C)
    # => trace(SIGMA') / trace(OMEGA') = p * np.trace(B@B) / n * np.trace(C@C) = 1
    # min_{OMEGA') max_{SIGMA'} {{logdet(I + inv(OMEGA') @ Heff.conj().T @ SIGMA' @ Heff) : tr(SIGMA') <= 1}: tr(OMEGA') <= 1}
    r_u_o_o, O_p, S_p = ptp_worst_case_noise_approx(Heff, n * p, n * p, precision=1e-2)
    assert r_d_f_f == pytest.approx(r_u_o_o, 1e-2)
    K = np.zeros(Bs_antennas)
    Omega = H.conj().T @ pinv(B + H @ C @ H.conj().T) @ H + K
    Sigma = -inv(B + H @ C @ H.conj().T) + pinv(B)
    assert r_d_f_f == pytest.approx(
        logdet(eye(Bs_antennas) + pinv(Omega) @ H.conj().T @ Sigma @ H), 1e-3
    )

    Omega_s = Omega / np.trace(C @ Omega) * n * c
    assert np.trace(C @ Omega_s) == pytest.approx(n * c, 1e-3)
    Sigma_s = Sigma / np.trace(B @ Sigma) * p * b
    assert np.trace(B @ Sigma_s) == pytest.approx(p * b, 1e-3)

    assert r_d_f_f == pytest.approx(
        logdet(eye(Bs_antennas) + pinv(Omega_s) @ H.conj().T @ Sigma_s @ H), 1e-3
    )

    # if we use Omega_s as uplink noise, I can compute Sigma_s
    Omega_prime = sqrtm(C) @ Omega_s @ sqrtm(C)
    r_comp, Sigma_prime = ptp_capacity(Heff.conj().T, p * b, Omega_prime)
    assert np.allclose(
        inv_sqrtm(B) @ Sigma_prime @ inv_sqrtm(B), Sigma_s, rtol=1e-02, atol=1e-02
    )
    # assert np.allclose(Sigma_prime, S_p, rtol=1e-02, atol=1e-02)
    assert r_comp == pytest.approx(r_d_f_f, 1e-2)

    # check the minmax property for Omega_s, if we use Omega_s, Sigma_s as the uplink covariance we can calculate Omega_s
    X = eye(3) + pinv_sqrtm(
        Omega_prime
    ) @ Heff.conj().T @ Sigma_prime @ Heff @ pinv_sqrtm(Omega_prime)
    # TR 2.27 -inv(A) + W + mu I = 0
    Omega_comp = -eye(3) + inv(X)
    Omega_comp = Omega_comp / np.trace(Omega_comp) * np.trace(Omega_prime)
    assert np.allclose(Omega_comp, Omega_prime, rtol=1e-02, atol=1e-02)

    X = eye(3) + pinv_sqrtm(Omega_prime) @ sqrtm(C) @ H.conj().T @ inv_sqrtm(
        B
    ) @ Sigma_prime @ inv_sqrtm(B) @ H @ sqrtm(C) @ pinv_sqrtm(Omega_prime)
    # Omega_prime = sqrtm(C) @ Omega_s @ sqrtm(C)
    X = eye(3) + pinv_sqrtm(Omega_prime) @ sqrtm(C) @ H.conj().T @ Sigma_s @ H @ sqrtm(
        C
    ) @ pinv_sqrtm(Omega_prime)
    X = eye(3) + inv_sqrtm(C) @ sqrtm(C) @ inv_sqrtm((Omega_prime)) @ sqrtm(
        C
    ) @ H.conj().T @ Sigma_s @ H @ sqrtm(C) @ pinv_sqrtm(Omega_prime)
    X_l = C + sqrtm(C) @ inv_sqrtm(sqrtm(C) @ Omega_s @ sqrtm(C)) @ sqrtm(
        C
    ) @ H.conj().T @ Sigma_s @ H @ sqrtm(C) @ inv_sqrtm(
        sqrtm(C) @ Omega_s @ sqrtm(C)
    ) @ sqrtm(
        C
    )
    # TR 2.27 -inv(A) + W + mu I = 0
    Omega_comp = -eye(3) + inv(X)
    Omega_comp = Omega_comp / np.trace(Omega_comp) * np.trace(Omega_prime)
    assert np.allclose(Omega_comp, Omega_prime, rtol=1e-02, atol=1e-02)
    assert np.allclose(
        inv_sqrtm(C) @ Omega_comp @ inv_sqrtm(C), Omega_s, rtol=1e-02, atol=1e-02
    )
    # min logdet(OMEGA + H.conj().T@Sigma@H) - logdet(OMEGA) : tr(C@OMEGA) = n*c, OMEGA >= 0
    # => inv(OMEGA + H.conj().T@Sigma@H) - inv(OMEGA) = mu * C + N
    D = -inv(Omega_s + H.conj().T @ Sigma_s @ H) + inv(Omega_s)
    D = D / np.trace(D) * np.trace(C)
    assert np.allclose(D, C, rtol=1e-02, atol=1e-02)
    # min logdet(R + H@Q@H.conj().T) - logdet(R) : R <= B, R >= 0
    # => inv(R + H@Q@H.conj().T) - inv(R) = X - L : X = uB + Y | L = 0
    # => tr(B@X) == u tr(B@B) => u = tr(B@X) / tr(B@B)
    X = -inv(B + H @ C @ H.conj().T) + inv(B)
    u = np.trace(B @ X) / np.trace(B @ B)
    Y = X - u * B
    assert np.trace(B @ Y) == pytest.approx(0)

    # transformation including the uplink noise
    Q_trans = MACtoBCtransformation([Heff], [Sigma_prime], [0])[0]
    rate_downlink = logdet(eye(Ms_antennas) + Heff @ Q_trans @ Heff.conj().T)
    rate_uplink = logdet(eye(Bs_antennas) + Heff.conj().T @ Sigma_prime @ Heff)
    assert rate_downlink == pytest.approx(rate_uplink)
    rate_uplink_w_noise = logdet(
        eye(Bs_antennas) + inv(Omega_prime) @ Heff.conj().T @ Sigma_prime @ Heff
    )
    rate_downlink_w_noise = logdet(
        eye(Ms_antennas)
        + Heff
        @ inv_sqrtm(Omega_prime)
        @ Q_trans
        @ inv_sqrtm(Omega_prime)
        @ Heff.conj().T
    )

    assert r_d_f_f == pytest.approx(rate_uplink_w_noise)
    assert r_d_f_f == pytest.approx(rate_downlink_w_noise)

    rate_trans_u = logdet(
        eye(Bs_antennas)
        + inv_sqrtm(Omega_prime)
        @ sqrtm(C)
        @ H.conj().T
        @ inv_sqrtm(B)
        @ Sigma_prime
        @ inv_sqrtm(B)
        @ H
        @ sqrtm(C)
        @ inv_sqrtm(Omega_prime)
    )
    assert r_d_f_f == pytest.approx(rate_trans_u)
    rate_trans_d = logdet(
        eye(Ms_antennas)
        + inv_sqrtm(B)
        @ H
        @ sqrtm(C)
        @ inv_sqrtm(Omega_prime)
        @ Q_trans
        @ inv_sqrtm(Omega_prime)
        @ sqrtm(C)
        @ H.conj().T
        @ inv_sqrtm(B)
    )
    Q_downlink = (
        sqrtm(C) @ inv_sqrtm(Omega_prime) @ Q_trans @ inv_sqrtm(Omega_prime) @ sqrtm(C)
    )
    assert np.allclose(Q_downlink, C, rtol=1e-02, atol=1e-02)

    assert r_d_f_f == pytest.approx(rate_trans_d)

    Heff = inv_sqrtm(B) @ H @ inv_sqrtm(Omega_s)
    # transformation including the uplink noise
    Q_trans = MACtoBCtransformation([Heff], [sqrtm(B) @ Sigma_s @ sqrtm(B)], [0])[0]
    rate_downlink = logdet(eye(Ms_antennas) + Heff @ Q_trans @ Heff.conj().T)
    rate_uplink = logdet(eye(Bs_antennas) + Heff.conj().T @ Sigma_prime @ Heff)
    assert rate_downlink == pytest.approx(rate_uplink)
    assert r_d_f_f == pytest.approx(rate_uplink)
    assert r_d_f_f == pytest.approx(rate_downlink)
    rate_trans = logdet(
        eye(Bs_antennas)
        + inv_sqrtm(Omega_s)
        # @ inv_sqrtm(C)
        # @ sqrtm(C)
        @ H.conj().T @ inv_sqrtm(B) @ sqrtm(B) @ Sigma_s @ sqrtm(B) @ inv_sqrtm(B) @ H
        # @ sqrtm(C)
        # @ inv_sqrtm(C)
        @ inv_sqrtm(Omega_s)
    )
    assert r_d_f_f == pytest.approx(rate_trans)
    rate_trans = logdet(
        eye(Ms_antennas)
        + inv_sqrtm(B)
        @ H
        @ inv_sqrtm(Omega_s)
        @ Q_trans
        @ inv_sqrtm(Omega_s)
        @ H.conj().T
        @ inv_sqrtm(B)
    )
    assert r_d_f_f == pytest.approx(rate_trans)
    Q_downlink = inv_sqrtm(Omega_s) @ Q_trans @ inv_sqrtm(Omega_s)
    assert np.allclose(Q_downlink, C, rtol=1e-02, atol=1e-02)


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [3])
@pytest.mark.parametrize("Bs_antennas", [3])
@pytest.mark.parametrize("sigma", [1.2])
@pytest.mark.parametrize("P", [1.3])
def test_minimax_opt_opt(Ms_antennas, Bs_antennas, B, C, H, P, sigma, comp):
    # Lemma 3
    p = c = np.trace(C @ C)
    n = b = np.trace(B @ B)
    # Downlink optim noise, downlink optim covariance, uplink fixed noise, fixed covariance
    Heff = sqrtm(B) @ H @ inv_sqrtm(C)
    r_d_o_o, R_eff, Q_eff = ptp_worst_case_noise_approx(
        Heff.conj().T, c * n, p * b, precision=1e-4
    )
    R = inv_sqrtm(B) @ R_eff @ inv_sqrtm(B)
    Q = inv_sqrtm(C) @ Q_eff @ inv_sqrtm(C)
    assert r_d_o_o == pytest.approx(
        logdet(eye(Ms_antennas) + pinv(R) @ H @ Q @ H.conj().T), 1e-3
    )
    r_u_f_f = logdet(eye(Bs_antennas) + inv(C) @ H.conj().T @ B @ H)
    assert r_d_o_o == pytest.approx(r_u_f_f, 1e-3)

    K = np.zeros(Bs_antennas)
    Omega = H.conj().T @ pinv(R + H @ Q @ H.conj().T) @ H + K
    Sigma = -pinv(R + H @ Q @ H.conj().T) + pinv(R)
    # Omega = Omega / max(np.linalg.eigvalsh(Omega)) * max(np.linalg.eigvalsh(C))
    # Sigma = Sigma / max(np.linalg.eigvalsh(Sigma)) * max(np.linalg.eigvalsh(B))
    assert r_d_o_o == pytest.approx(
        logdet(eye(Bs_antennas) + pinv(Omega) @ H.conj().T @ Sigma @ H), 1e-3
    )

    # R = H@pinv(C + H.conj().T@B@H)@H.conj().T
    # Q = -inv(C + H.conj().T @ B @ H) + pinv(C)

    H_eff_uplink_2 = pinv_sqrtm(R) @ H @ inv_sqrtm(C)
    Q_trans = MACtoBCtransformation([H_eff_uplink_2], [sqrtm(R) @ B @ sqrtm(R)], [0])[0]
    rate_downlink_trans = logdet(
        eye(Ms_antennas) + H_eff_uplink_2 @ Q_trans @ H_eff_uplink_2.conj().T
    )
    rate_uplink_trans = logdet(
        eye(Bs_antennas)
        + H_eff_uplink_2.conj().T @ sqrtm(R) @ B @ sqrtm(R) @ H_eff_uplink_2
    )
    assert rate_downlink_trans == pytest.approx(rate_uplink_trans, 1e-2)
    assert rate_downlink_trans == pytest.approx(r_d_o_o, 1e-2)
    Q2 = inv_sqrtm(C) @ Q_trans @ inv_sqrtm(C)
    assert np.allclose(Q, Q2, rtol=1e-02, atol=1)


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [3])
@pytest.mark.parametrize("Bs_antennas", [3])
@pytest.mark.parametrize("sigma", [1.2])
@pytest.mark.parametrize("P", [1.3])
def test_minimax_fixed_opt(Ms_antennas, Bs_antennas, B, C, H, P, sigma, comp):
    p = c = np.trace(C @ C)
    n = b = np.trace(B @ B)
    # Downlink fixed noise, downlink optim covariance, uplink fixed noise, optim covariance
    r_d_f_o, Q_eff = ptp_capacity(H @ inv_sqrtm(C), np.trace(C @ C), B)
    assert np.trace(inv_sqrtm(C) @ Q_eff @ inv_sqrtm(C) @ C) == pytest.approx(c, 1e-3)
    r_u_f_o, _ = ptp_capacity(H.conj().T @ inv_sqrtm(B), p * np.trace(B @ B), n * C)
    assert r_d_f_o == pytest.approx(r_u_f_o, 1e-3)
    R = B
    Q = inv_sqrtm(C) @ Q_eff @ inv_sqrtm(C)
    K = np.zeros(Bs_antennas)
    Omega = H.conj().T @ pinv(R + H @ Q @ H.conj().T) @ H + K
    Sigma = -pinv(R + H @ Q @ H.conj().T) + pinv(R)
    assert r_d_f_o == pytest.approx(
        logdet(eye(Bs_antennas) + pinv(Omega) @ H.conj().T @ Sigma @ H), 1e-3
    )


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [3])
@pytest.mark.parametrize("Bs_antennas", [3])
@pytest.mark.parametrize("sigma", [1.2])
@pytest.mark.parametrize("P", [1.3])
def test_minimax_opt_fixed(Ms_antennas, Bs_antennas, B, C, H, P, sigma, comp):
    p = c = np.trace(C @ C)
    n = b = np.trace(B @ B)
    # Downlink optim noise, downlink fixed covariance, uplink optim noise, fixed covariance
    r_d_o_f, R = ptp_worst_case_noise_static(
        H @ C @ H.conj().T, np.trace(B @ B), precision=1e-4
    )
    r_u_o_f, _ = ptp_worst_case_noise_static(
        H.conj().T @ (p * B) @ H, n * np.trace(C @ C), precision=1e-4
    )
    K = np.zeros(Bs_antennas)
    Omega = H.conj().T @ pinv(R + H @ C @ H.conj().T) @ H + K
    Sigma = -pinv(R + H @ C @ H.conj().T) + pinv(R)
    assert r_d_o_f == pytest.approx(
        logdet(eye(Bs_antennas) + pinv(Omega) @ H.conj().T @ Sigma @ H), 1e-3
    )


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [3])
@pytest.mark.parametrize("Bs_antennas", [3])
@pytest.mark.parametrize("sigma", [2])
@pytest.mark.parametrize("P", [5])
def test_shaping(Ms_antennas, Bs_antennas, B, C, H, comp, P, sigma):
    # compute downlink covariance for noise R=B, and Q << C
    rate_shaping, Q_d, (Omega_d, nuCplusZp, K) = ptp_capacity_minimax(
        H=H, R=B, C=C, Zs=[np.zeros((3, 3))], eps=1e-8
    )
    # we expect Q = C as the optimal result
    expected_rate_shaping = logdet(eye(Ms_antennas) + inv(B) @ H @ C @ H.conj().T)
    assert rate_shaping == pytest.approx(expected_rate_shaping, 1e-3)
    # We implement a power constraint by Z = {Z : tr(Z) =0}
    Zs = []
    Zs.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
    Zs.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]))
    Zs.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))
    Zs.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]))
    Zs.append(np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]))
    if comp:
        Zs.append(np.array([[0, 1j, 0], [-1j, 0, 0], [0, 0, 0]]))
        Zs.append(np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]]))
        Zs.append(np.array([[0, 0, 0], [0, 0, 1j], [0, -1j, 0]]))
    rate_power, Sigma_u, (R_u, sigmaBplusYo, M) = ptp_capacity_minimax(
        H=H, R=B, C=C, Zs=Zs
    )
    expected_rate_power, _ = ptp_capacity(H, np.trace(C), B)
    assert rate_power == pytest.approx(expected_rate_power, 1e-3)


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [1, 2, 3])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
@pytest.mark.parametrize("sigma", [5])
@pytest.mark.parametrize("P", [73])
def test_effective_channels(comp, Ms_antennas, Bs_antennas, H, sigma, P, B, C):
    # we are testing functionality of effective channels vs. constrainsts
    # TODO test this for range of P/sigma - hint: do not use effective channels
    I_Ms = eye(Ms_antennas)
    I_Bs = eye(Bs_antennas)

    rate_ref = logdet(I_Ms + inv(B) @ H @ C @ H.conj().T)
    # compute downlink covariance for noise R=B, and Q << C

    for i, (Beff, Ceff, Bcon, Ccon) in enumerate(
        [(B, C, I_Ms, I_Bs), (I_Ms, C, B, I_Bs), (I_Ms, I_Bs, B, C), (B, I_Bs, I_Ms, C)]
    ):
        rel = 1e-2
        Heff = inv_sqrtm(Beff) @ H @ sqrtm(Ceff)
        rate_d, Q, (Omega_C, _, K) = ptp_capacity_minimax(
            H=Heff, R=Bcon, C=Ccon, Zs=[np.zeros((Bs_antennas, Bs_antennas))], eps=1e-3
        )
        Omega = Heff.conj().T @ inv(Bcon + Heff @ Q @ Heff.conj().T) @ Heff + K
        assert rate_d == pytest.approx(rate_ref, rel=rel)
        assert logdet(I_Ms + inv(Bcon) @ Heff @ Q @ Heff.conj().T) == pytest.approx(
            rate_ref, rel=rel
        )
        Sigma = -inv(Bcon + Heff @ Q @ Heff.conj().T) + inv(Bcon)
        Omega_inv = np.linalg.pinv(Omega, rcond=1e-6, hermitian=True)
        rate_u_cal = logdet(
            I_Bs
            + Omega_inv
            * np.trace(Omega @ Ccon)
            / sigma
            @ Heff.conj().T
            @ Sigma
            @ Heff
            * P
            / np.trace(Sigma @ Bcon)
        )
        rate_u_cal = logdet(
            I_Bs
            + Omega_inv
            * np.trace(Omega @ Ccon)
            @ Heff.conj().T
            @ Sigma
            @ Heff
            / np.trace(Sigma @ Bcon)
        )
        assert rate_u_cal == pytest.approx(rate_ref, rel=1e-2)
        rate_u, Sigma_eff_BC = ptp_capacity(
            Heff.conj().T @ inv_sqrtm(Bcon), P, Omega / np.trace(Omega @ Ccon) * sigma
        )
        rate_u, Sigma_eff_BC = ptp_capacity(
            Heff.conj().T @ inv_sqrtm(Bcon), 1, Omega / np.trace(Omega @ Ccon)
        )
        assert rate_u == pytest.approx(rate_ref, rel=1e-2)


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize(
    "H",
    [np.matrix([[1, 0], [0, 1], [1, 1]]), np.matrix([[1, 0, 0], [1, 0, 0], [0, 0, 1]])],
    ids=["3x2", "3x3"],
)
def test_noise_rank_def_channel(comp, H):
    # A rank deficient channel implies the noise covariance can be rank deficient
    H = H + comp * 1j * H
    P = 100
    sigma = 1
    rate_no_channel = logdet(eye(3) + P / sigma * H @ H.conj().T)
    rate_worst_case, Z, _ = ptp_worst_case_noise_approx(H.conj().T, P, sigma)
    assert rate_worst_case == pytest.approx(rate_no_channel, 1e-2)
    assert np.linalg.matrix_rank(Z, tol=1e-6, hermitian=True) == 2
    rate_noise, _ = ptp_capacity(H, P, Z)
    assert rate_noise == pytest.approx(rate_no_channel, 1e-2)


# worst case noise, like white
@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [1, 2, 3])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
@pytest.mark.parametrize("P", [3, 100])
def test_ptp_worstcase(H, Ms_antennas, Bs_antennas, P, comp):
    sigma = 3
    rate_no_channel = logdet(eye(Bs_antennas) + P / sigma * H.conj().T @ H)
    rate_worst_case, Z, Q = ptp_worst_case_noise_approx(H, P, sigma, precision=1e-2)
    assert rate_worst_case == pytest.approx(rate_no_channel, 1e-2)


def test_MAC_worst_case_simple():
    P = 10
    sigma = 3
    Bs_antennas = 2
    Ms_antennas = 1
    H_1 = np.matrix([[1, 0]])
    H_2 = np.matrix([[0, 1]])
    H = np.concatenate((H_1,H_2), axis=0)
    rate_no_channel = logdet(eye(Bs_antennas) + P / sigma * H.conj().T @ H)
    rate_worst_case, Z, Q = ptp_worst_case_noise_approx(H, P, sigma, precision=1e-2)
    rate_MAC_joint, Omega_joint, Covs_joint, order_joint = MAC_worst_case_noise_approx([H], P, sigma, precision=1e-2)
    Sigma = Covs_joint[0]
    R = H @ pinv(Omega_joint + H.conj().T @ Sigma @ H) @ H.conj().T
    Q = -pinv(Omega_joint + H.conj().T @ Sigma @ H) + pinv(Omega_joint)
    rate = logdet(eye(2*Ms_antennas) + inv(R) @ H @ Q @ H.conj().T)
    assert rate_MAC_joint[0] == pytest.approx(rate_no_channel, 1e-2)
    assert rate == pytest.approx(rate_no_channel, 1e-2)
    

    rate_MAC, Omega, Covs, order = MAC_worst_case_noise_approx([H_1, H_2], P, sigma, precision=1e-2)
    Omega_equal = eye(Bs_antennas)*sigma/Bs_antennas
    rates_check, MAC_Covs, order, Omega_sbgr = MAC_cvx_with_noise_sbgr([H_1.conj().T, H_2.conj().T], P, [1,1], Omega=Omega_equal)
    Sigma_w1 = MAC_Covs[0]
    Sigma_w2 = MAC_Covs[1]
    A1 = H_1.conj().T @ Sigma_w1 @ H_1
    A2 = H_2.conj().T @ Sigma_w2 @ H_2

    r_1 = logdet(Omega_equal + A1) - logdet(Omega_equal)
    r_2 = logdet(Omega_equal + A1 + A2) - logdet(Omega_equal + A1) 
    assert rate_MAC == pytest.approx(rates_check, rel=1e-2, abs=1e-3)
    Sigma_1 = Covs[0]
    Sigma_2 = Covs[1]
    R_1 = H_1 @ pinv(Omega + H_1.conj().T @ Sigma_1 @ H_1) @ H_1.conj().T
    R_2 = H_2 @ pinv(Omega + H_2.conj().T @ Sigma_2 @ H_2) @ H_2.conj().T
    Q_1 = -pinv(Omega + H_1.conj().T @ Sigma_1 @ H_1) + pinv(Omega)
    Q_2 = -pinv(Omega + H_2.conj().T @ Sigma_2 @ H_2) + pinv(Omega)
    r_1 = logdet(eye(Ms_antennas) + inv(R_1) @ H_1 @ Q_1 @ H_1.conj().T)
    r_2 = logdet(eye(Ms_antennas) + inv(R_2) @ H_2 @ Q_2 @ H_2.conj().T)
    assert rate_MAC == pytest.approx([r_1, r_2], rel=1e-2, abs=1e-3)

    rate_p2p1, Omega_p2p1, Cov_ptp1 = ptp_worst_case_noise_approx(H_1, P, sigma)
    rate_w1, Omega_w1, Covs_w1, order_w1 = MAC_worst_case_noise_approx([H_1, H_2], P, sigma, weights=[10,0], precision=1e-2)
    assert rate_w1 == pytest.approx([r_1, 0], rel=1e-2, abs=1e-3)
    rate_p2p2, Omega_p2p2, Cov_ptp2 = ptp_worst_case_noise_approx(H_2, P, sigma)
    rate_w2, Omega_2, Covs_2, order_w2 = MAC_worst_case_noise_approx([H_1, H_2], P, sigma, weights=[0,1], precision=1e-2)
    assert rate_w2 == pytest.approx([0, r_2], rel=1e-2, abs=1e-3)

def logpdet(A, t=1e-6):
    a = np.real(np.linalg.eigvalsh(A))
    return log(np.product(a[a>t]))


@pytest.mark.parametrize("Ms_antennas_list", [[2, 2]])
@pytest.mark.parametrize("Bs_antennas", [2])
def test_worst_case_MIMO_2user(Ms_antennas_list, Bs_antennas, H_MAC, C):
    [H_1, H_2] = H_MAC
    P = 2
    sigma = 2
    weights = [0.6, 0.4]
    rates, Omega, Covs, order = MAC_worst_case_noise_approx(H_MAC, P, sigma, weights=weights, precision=1e-4)
    [Sigma_1, Sigma_2] = Covs
    Xi_1 = Omega 
    Xi_2 = Omega + H_1.conj().T @ Sigma_1 @ H_1
    [w_1, w_2] = weights
    S_1 = w_1 * H_1 @ pinv(Xi_1 + H_1.conj().T @ Sigma_1 @ H_1) @ H_1.conj().T
    S_2 = w_2 * H_2 @ pinv(Xi_2 + H_2.conj().T @ Sigma_2 @ H_2) @ H_2.conj().T
    Q_1 = w_1 * -pinv(Xi_1 + H_1.conj().T @ Sigma_1 @ H_1) + w_1 * pinv(Xi_1)
    Q_2 = w_2 * -pinv(Xi_2 + H_2.conj().T @ Sigma_2 @ H_2) + w_2 * pinv(Xi_2)

    r_1d = logdet(eye(2) + inv(S_1) @ H_1 @ Q_1 @ H_1.conj().T)
    r_2d = logdet(eye(2) + inv(S_2) @ H_2 @ Q_2 @ H_2.conj().T)
    # if we set MS_antenns to 1 we need
    #r_1d = logdet(eye(2) + inv(2*S_1) @ H_1 @ Q_1 @ H_1.conj().T)
    #r_2d = logdet(eye(2) + inv(2*S_2) @ H_2 @ Q_2 @ H_2.conj().T)
    # TODO, we need to make the scaling automatic tr(BB)=tr(CC) 
    # TODO Q_1 and Q_2 are not really white, downlink transformation is not precice
    assert rates == pytest.approx([r_1d, r_2d], rel=1e-2, abs=1e-3)
    assert S_1 == pytest.approx(S_2 + H_1@Q_2@H_1.conj().T, rel=1e-2, abs=1e-3)


@pytest.mark.parametrize("Ms_antennas_list", [[1, 2, 3], [2, 2, 2]])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
def test_MAC_worst_case_noise(Ms_antennas_list, Bs_antennas, H_MAC, C):
    weights = [random.random() for _ in Ms_antennas_list]
    P = 2
    sigma = 2
    
    rates_MAC, Omega, Covs, order = MAC_worst_case_noise_approx(H_MAC, P, sigma, weights=weights, precision=1e-2)
    # transform to broadcast
    Xi = Omega
    rates_BC = [None for _ in Ms_antennas_list]

    for o in order[::-1]:
        Sigma = Covs[o]
        H = H_MAC[o]
        w = weights[o]
        I = eye(Ms_antennas_list[o])
        Xi_n = Xi + H.conj().T @ Sigma @ H
        S = w * H @ pinv(Xi_n) @ H.conj().T
        Q = w * -pinv(Xi_n) + w * pinv(Xi)
        Xi = Xi_n    
        rates_BC[o] = logdet(I + inv(S) @ H @ Q @ H.T)
    
    assert rates_MAC == pytest.approx(rates_BC, rel=1e-2, abs=1e-3)


@pytest.mark.parametrize("Ms_antennas_list", [[1, 2, 3], [2, 2, 2]])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
def test_infcons(Ms_antennas_list, Bs_antennas, H_MAC, C):
    # We pick a random wsr-value than bounds for every user with weight
    # such that even if all transmit power goes to this user
    # TODO matrix I is sigular, so max_Sigma { logdet(I + pinv(I) @ H.conj().T @ Sigma @ H : tr(Sigma) <= P)
    # TODO cross check with documentation of infcons
    P = 10
    wsr = random.random()*1
    weights = [random.random() for _ in Ms_antennas_list]
    Is = []
    for w, H in zip(weights, H_MAC):
        Is = inf_cons(H, P, wsr/w)
        for N_i in Is:
            r_comp, Sigma_prime = ptp_capacity(H.conj().T, P, N_i)
            assert wsr == pytest.approx(w*r_comp, 1e-2)


@pytest.mark.parametrize("Ms_antennas_list, Bs_antennas", [([1, 2, 3], 2)])
def test_MAC_rank_def_noise(Ms_antennas_list, Bs_antennas, H_MAC, C):
    P = 10
    sigma = 3
    H_1 = H_MAC[0]
    H_2 = H_MAC[1]
    Omega_w = np.matrix([[sigma, 0], [0, 0]])
    Omega_w = C
    Sigma_w1 = np.matrix([[P]]) 
    Sigma_w2 = C
    A1 = H_1.conj().T @ Sigma_w1 @ H_1
    A2 = H_2.conj().T @ Sigma_w2 @ H_2

    r_1 = logdet(Omega_w + A1) - logdet(Omega_w)
    r_2 = logdet(Omega_w + A1 + A2) - logdet(Omega_w + A1) 
    r_1 = logpdet(Omega_w + A1) - logpdet(Omega_w)
    r_2 = logpdet(Omega_w + A1 + A2) - logpdet(Omega_w + A1)
    r_w1_u = logdet(eye(Bs_antennas) + pinv(Omega_w) @ A1)
    r_w2_u = logdet(eye(Bs_antennas) + pinv(Omega_w + A1) @ A2)
    assert [r_w1_u, r_w2_u] == pytest.approx([r_1, r_2], 1e-2)
     
