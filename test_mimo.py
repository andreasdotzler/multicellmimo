import itertools
import logging
import random

import numpy as np
import pytest

from .mimo import (
    MAC,
    BC_rates,
    MAC_cvx,
    MAC_rates,
    MAC_rates_ordered,
    MACtoBCtransformation,
    inv_sqrtm,
    sqrtm,
    logdet,
    project_covariance_cvx,
    project_covariances,
    project_eigenvalues_to_given_sum_cvx,
    ptp_capacity,
    ptp_capacity_cvx,
    ptp_capacity_cvx_dual,
    ptp_capacity_mimimax,
    ptp_worst_case_noise,
    water_filling_cvx,
    water_filling_iter,
)

LOGGER = logging.getLogger(__name__)

inv = np.linalg.inv
det = np.linalg.det
log = np.log
eye = np.eye

@pytest.fixture(scope="function", autouse=True)
def seed():
    np.random.seed(42)

# worst case noise, like white
@pytest.mark.parametrize("Ms_antennas", [2, 3])
@pytest.mark.parametrize("Bs_antennas", [2, 3])
def test_ptp_worstcase(Ms_antennas, Bs_antennas):
    H = (
        np.random.random([Ms_antennas, Bs_antennas])
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

    rate_worst_case, (Z, W) = ptp_worst_case_noise(H, P, sigma)
    L = sqrtm(Z) @ W @ sqrtm(Z)
    rate_worst_case_dual = -log(det(L)) / np.log(2)
    assert rate_worst_case == pytest.approx(rate_worst_case_dual, 1e-2)
    assert rate_worst_case == pytest.approx(rate_no_channel, 1e-2)

    H_eff = inv_sqrtm(Z) @ H.T
    rate_e, Q = ptp_capacity(H_eff, P / sigma * np.trace(Z))
    assert np.trace(Q) == pytest.approx(P / sigma * np.trace(Z))
    assert rate_worst_case == pytest.approx(rate_e, 1e-2)


# TODO fix inv_sqrtm if matrix is not full rank


def test_schur_complement():
    p = 2
    q = 3
    A = np.random.random([p, p])
    A = A @ A.T
    B = np.random.random([p, q])
    C = -B.T
    D = np.random.random([q, q])
    D = D @ D.T
    M = np.block([[A, B], [C, D]])
    assert det(M) == pytest.approx(det(D) * det(A - B @ inv(D) @ C))
    assert log(det(M)) == pytest.approx(log(det(D)) + log(det(A - B @ inv(D) @ C)))


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
    rate_d = ptp_capacity_cvx_dual(H, P / Bs_antennas, Z)[0]
    assert rate_p == pytest.approx(rate_d, 1e-3)

    # create random uplink noise
    R = np.random.random([Bs_antennas, Bs_antennas])
    Z = R @ R.T
    Z = sigma / Bs_antennas * Z
    rate_d = ptp_capacity_cvx_dual(H, P, Z)[0]
    H_eff = inv_sqrtm(Z) @ H.T
    rate_p, Sigma_eff = ptp_capacity(H_eff, P / sigma)
    rate_c = (log(det(Z + H.T @ Sigma_eff @ H)) - log(det(Z))) / np.log(2)
    assert rate_c == pytest.approx(rate_p, 1e-3)


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


@pytest.mark.parametrize("Ms_antennas_list, Bs_antennas", [([1, 2, 3], 2)])
def test_MAC_rate_formulation(Ms_antennas_list, Bs_antennas):
    Hs = []
    Covs = []
    for Ms_antennas in Ms_antennas_list:
        Hs.append(
            np.random.random([Ms_antennas, Bs_antennas])
            + np.random.random([Ms_antennas, Bs_antennas]) * 1j
        )
        Co = (
            np.random.random([Ms_antennas, Ms_antennas])
            + np.random.random([Ms_antennas, Ms_antennas]) * 1j
        )
        Covs.append(Co @ Co.conj().T)
    I = np.eye(Bs_antennas)
    # r1 = logdet(I + Hs[2].conj().T @ Covs[2] @ Hs[2])
    No = (
        np.random.random([Ms_antennas, Ms_antennas])
        + np.random.random([Ms_antennas, Ms_antennas]) * 1j
    )
    Noise = 10 * I
    # decoding order: user 2, user 1, user 0 is decoded last

    HCH0 = Hs[0].conj().T @ Covs[0] @ Hs[0]
    HCH1 = Hs[1].conj().T @ Covs[1] @ Hs[1]
    HCH2 = Hs[2].conj().T @ Covs[2] @ Hs[2]

    IPN0 = Noise
    IPN1 = Noise + HCH0
    IPN1 = Noise + HCH0 + HCH1

    r0_c = logdet(I + np.linalg.inv(Noise) @ HCH0)
    r0_d = logdet(Noise + HCH0) - logdet(Noise)
    assert r0_d == pytest.approx(r0_c, 1e-3)
    r1_c = logdet(I + np.linalg.inv(Noise + HCH0) @ HCH1)
    r1_d = logdet(Noise + HCH0 + HCH1) - logdet(Noise + HCH0)
    assert r1_d == pytest.approx(r1_c, 1e-3)
    r2_c = logdet(I + np.linalg.inv(Noise + HCH0 + HCH1) @ HCH2)
    r2_d = logdet(Noise + HCH0 + HCH1 + HCH2) - logdet(Noise + HCH0 + HCH1)
    assert r2_d == pytest.approx(r2_c, 1e-3)

    w = [2, 1.5, 0.5]
    wsr_d = w[0] * r0_c + w[1] * r1_c + w[2] * r2_c
    wsr_c = w[0] * logdet(Noise + HCH0) - w[0] * logdet(Noise)
    wsr_c += w[1] * logdet(Noise + HCH0 + HCH1) - w[1] * logdet(Noise + HCH0)
    wsr_c += w[2] * logdet(Noise + HCH0 + HCH1 + HCH2) - w[2] * logdet(
        Noise + HCH0 + HCH1
    )
    wsr_e = -w[0] * logdet(Noise)
    wsr_e += (w[0] - w[1]) * logdet(Noise + HCH0)
    wsr_e += (w[1] - w[2]) * logdet(Noise + HCH0 + HCH1)
    wsr_e += w[2] * logdet(Noise + HCH0 + HCH1 + HCH2)
    assert wsr_d == pytest.approx(wsr_c, 1e-3)
    assert wsr_e == pytest.approx(wsr_c, 1e-3)


def test_BC_Rates():
    H1 = H2 = np.array([[1, 0, 0]]).T
    Cov1 = Cov2 = np.array([[1]])
    rates_01 = BC_rates([Cov1, Cov2], [H1, H2], [0, 1])
    assert rates_01[1] >= rates_01[0]
    rates_10 = BC_rates([Cov1, Cov2], [H1, H2], [1, 0])
    assert rates_10[0] >= rates_10[1]


def test_MAC_Rates():
    H1 = H2 = np.array([[1, 0, 0]])
    Cov1 = Cov2 = np.array([[1]])
    rates_01 = MAC_rates([Cov1, Cov2], [H1.T, H2.T], [0, 1])
    assert rates_01[1] >= rates_01[0]
    BC_Cov_trans_01 = MACtoBCtransformation([H1, H2], [Cov1, Cov2], [0, 1])
    BC_rates_01 = BC_rates(BC_Cov_trans_01, [H1, H2], [1, 0])
    assert rates_01 == pytest.approx(BC_rates_01, 1e-3)
    rates_10 = MAC_rates([Cov1, Cov2], [H1.T, H2.T], [1, 0])
    assert rates_10[0] >= rates_10[1]
    # assumes channels and covariances in inverse order of decoding
    # first user is decoded last -> higher rate
    rates_o1, _ = MAC_rates_ordered([Cov1, Cov2], [H1.T, H2.T])
    assert rates_o1[0] >= rates_o1[1]
    # assumes channels and covariances in inverse order of decoding
    # first user is decoded last -> higher rate
    rates_o2, _ = MAC_rates_ordered([Cov2, Cov1], [H2.T, H1.T])
    assert rates_o2[0] >= rates_o2[1]


def test_vishwanath_example1():
    """ we reproduce the numerical example in "Duality, Achievable Rates, and Sum-Rate Capacity of
    Gaussian MIMO Broadcast Channels" by Vishwanath, Jindal, Goldsmith 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.5703&rep=rep1&type=pdf
    """
    # example 1
    H1 = np.array([[1.0, 0.8], [0.5, 2.0]])
    H2 = np.array([[0.2, 1.0], [2.0, 0.5]])
    P = 1
    P1 = np.array([[0.0720, 0.1827], [0.1827, 0.4634]])
    P2 = np.array([[0.0000, 0.0026], [0.0026, 0.4646]])
    sum_rate = 2.2615 / np.log(2)
    MAC_rates, MAC_Covs, order = MAC([H1.T, H2.T], P, [1, 1])
    np.testing.assert_allclose(MAC_Covs, [P1, P2], atol=1e-3)

    MAC_rates_order_10, _ = MAC_rates_ordered([P2, P1], [H2.T, H1.T])
    MAC_rates_order_01, _ = MAC_rates_ordered([P1, P2], [H1.T, H2.T])
    assert sum(MAC_rates_order_10) == pytest.approx(sum(MAC_rates_order_01), 1e-3)
    assert sum(MAC_rates) == pytest.approx(sum(MAC_rates_order_01), 1e-3)

    S1_2 = np.array([[0.0001, -0.0069], [-0.0069, 0.4841]])
    S2_2 = np.array([[0.4849, 0.1225], [0.1225, 0.0309]])
    BC_rates_order_10 = BC_rates([S1_2, S2_2], [H1, H2], [1, 0])
    assert sum(BC_rates_order_10) == pytest.approx(sum_rate, 1e-3)
    BC_Cov_trans = MACtoBCtransformation([H1, H2], MAC_Covs, [0, 1])
    bc_rates = BC_rates(BC_Cov_trans, [H1, H2], [1, 0])
    assert bc_rates == pytest.approx(BC_rates_order_10, 1e-3)
    assert sum(bc_rates) == pytest.approx(sum_rate, 1e-3)

    S1_1 = np.array([[0.0746, 0.1932], [0.1932, 0.5004]])
    S2_1 = np.array([[0.4104, -0.0776], [-0.0776, 0.0147]])
    BC_rates_order_01 = BC_rates([S1_1, S2_1], [H1, H2], [0, 1])
    assert sum(BC_rates_order_01) == pytest.approx(sum_rate, 1e-3)
    BC_Cov_trans = MACtoBCtransformation([H1, H2], MAC_Covs, [1, 0])
    bc_rates = BC_rates(BC_Cov_trans, [H1, H2], [0, 1])
    assert bc_rates == pytest.approx(BC_rates_order_01, 1e-3)
    assert sum(bc_rates) == pytest.approx(sum_rate, 1e-3)


def test_vishwanath_example2():
    """ we redroduce the numerical example in "Duality, Achievable Rates, and Sum-Rate Capacity of
    Gaussian MIMO Broadcast Channels" by Vishwanath, Jindal, Goldsmith 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.5703&rep=rep1&type=pdf
    """
    # example 2
    H1 = np.array([[0.0, 1.0]])
    H2 = np.array([[-np.sqrt(3) / 2, -1 / 2]])
    H3 = np.array([[np.sqrt(3) / 2, -1 / 2]])
    P = 1

    I = np.eye(2)
    MAC_sum_rate = np.linalg.slogdet(
        I + 1 / 3 * (H1.conj().T @ H1 + H2.conj().T @ H2 + H3.conj().T @ H3)
    )[1] / np.log(2)
    MAC_sum_rate_paper = 0.8109 / np.log(2)
    assert MAC_sum_rate == pytest.approx(MAC_sum_rate_paper, 1e-3)
    MAC_rates, MAC_Covs, order = MAC([H1.T, H2.T, H3.T], P, [0.9999, 1, 1.0001])
    assert sum(MAC_rates) == pytest.approx(MAC_sum_rate, 1e-3)
    O = np.array([[1 / 3]])
    BC_Cov_trans = MACtoBCtransformation([H1, H2, H3], [O, O, O], [0, 1, 2])
    S1 = np.array([[0.0, 0.0], [0.0, 0.2867]])
    S2 = np.array([[0.2187, 0.1623], [0.1623, 0.1205]])
    S3 = np.array([[0.2812, -0.1624], [-0.1624, 0.0937]])
    np.testing.assert_allclose(BC_Cov_trans, [S1, S2, S3], atol=1e-3)
    bc_rates = BC_rates(BC_Cov_trans, [H1, H2, H3], [2, 1, 0])
    assert MAC_rates == pytest.approx(bc_rates, 1e-3)


@pytest.mark.parametrize("N,M", [(4, 4), (2, 4), (4, 2)])
def test_inv_sqrtm(N, M):
    H = np.random.random([N, M]) + np.random.random([N, M]) * 1j
    HH = np.eye(N) + H @ H.conj().T
    A = inv_sqrtm(HH)
    np.testing.assert_almost_equal(A @ A, np.linalg.inv(HH))


@pytest.mark.parametrize("Ms_antennas, Bs_antennas", [(2, 4), (4, 2), (4, 4)])
def test_MACtoBCtransformation_ptp(Ms_antennas, Bs_antennas):
    P = 100
    H = (
        np.random.random([Ms_antennas, Bs_antennas])
        + np.random.random([Ms_antennas, Bs_antennas]) * 1j
    )

    # H = np.array([[1j, 0], [0, 1]])
    rate_MAC, MAC_Cov = ptp_capacity(H.conj().T, P)
    assert MAC_Cov.shape == (Ms_antennas, Ms_antennas)
    rate_BC, BC_Cov = ptp_capacity(H, P)
    assert BC_Cov.shape == (Bs_antennas, Bs_antennas)

    np.testing.assert_almost_equal(H @ H.conj().T @ MAC_Cov, H @ BC_Cov @ H.conj().T)
    np.testing.assert_almost_equal(H.conj().T @ MAC_Cov @ H, H.conj().T @ H @ BC_Cov)

    BC_Cov_trans = MACtoBCtransformation([H], [MAC_Cov], [0])
    assert BC_Cov_trans[0].shape == (Bs_antennas, Bs_antennas)

    np.testing.assert_allclose(
        H @ H.conj().T @ MAC_Cov, H @ BC_Cov_trans[0] @ H.conj().T, atol=1e-3
    )
    np.testing.assert_allclose(
        H.conj().T @ MAC_Cov @ H, H.conj().T @ H @ BC_Cov_trans[0], atol=1e-3
    )

    rates_BC_calc = BC_rates(BC_Cov_trans, [H], [0])
    assert rate_BC == pytest.approx(rates_BC_calc[0], 1e-3)


@pytest.mark.parametrize("MAC_fun", [MAC, MAC_cvx])
@pytest.mark.parametrize("Ms_antennas_list, Bs_antennas", [([1, 2, 3], 2)])
def test_MACtoBCtransformation(MAC_fun, Ms_antennas_list, Bs_antennas):
    Hs = []
    for Ms_antennas in Ms_antennas_list:
        Hs.append(
            np.random.random([Ms_antennas, Bs_antennas])
            + np.random.random([Ms_antennas, Bs_antennas]) * 1j
        )
    MAC_Hs = [H.conj().T for H in Hs]
    w = list(range(1, len(Ms_antennas_list) + 1))
    w = [w * 3 for w in w]
    # broadcast, user with larges weight is encoded first https://arxiv.org/pdf/0901.2401.pdf
    # MAC decoding order is inverse that is, user with largest weight es decoded last
    for weights in itertools.permutations(w):
        MAC_decoding_order = np.argsort(weights)
        assert weights[MAC_decoding_order[-1]] >= weights[MAC_decoding_order[0]]
        mac_rates, MAC_Covs, order = MAC_fun(MAC_Hs, 100, weights)
        assert order == pytest.approx(MAC_decoding_order)
        # user with heighest weight sees no interference:
        logdet = np.linalg.slogdet
        dec_last = MAC_decoding_order[-1]
        r_h = logdet(
            np.eye(Bs_antennas)
            + MAC_Hs[dec_last] @ MAC_Covs[dec_last] @ MAC_Hs[dec_last].conj().T
        )
        # sort H and MAC_Covs
        assert r_h[1] / np.log(2) == pytest.approx(mac_rates[dec_last], 1e-3)
        mac_rates_calc = MAC_rates(MAC_Covs, MAC_Hs, MAC_decoding_order)
        assert mac_rates == pytest.approx(mac_rates_calc, 1e-3)
        # broadcast, highest weight is encoded first, lowest weight sees no interference
        enc_last = MAC_decoding_order[0]


#        BC_Cov_trans = MACtoBCtransformation(Hs, MAC_Covs, MAC_decoding_order)
#        rates_BC_calc = BC_rates(BC_Cov_trans, Hs, list(reversed(MAC_decoding_order)))
#        BC_Cov_trans_inv = MACtoBCtransformation(Hs, MAC_Covs, list(reversed(MAC_decoding_order)))
#        rates_BC_calc_inv = BC_rates(BC_Cov_trans_inv, Hs, MAC_decoding_order)
#        assert mac_rates == pytest.approx(rates_BC_calc, 1e-3)


@pytest.mark.parametrize("P", range(10))
def test_project_eigenvalues_to_given_sum_cvx(P):
    aa = np.array([1, 2, 3, 4])
    projected = project_eigenvalues_to_given_sum_cvx(aa, P)
    assert sum(projected) == pytest.approx(P, 1e-3)


@pytest.mark.parametrize(
    "H, P, r_exp, Cov_exp",
    [
        (np.array([[1, 0], [0, 0]]), 1, 1, np.array([[1, 0], [0, 0]])),
        (np.array([[1j, 0], [0, 0]]), 1, 1, np.array([[1, 0], [0, 0]])),
        (np.array([[1, 0], [0, 1]]), 2, 2, np.array([[1, 0], [0, 1]])),
        (np.array([[1j, 0], [0, 1]]), 2, 2, np.array([[1, 0], [0, 1]])),
        (np.array([[0, 1j], [1j, 0]]), 2, 2, np.array([[1, 0], [0, 1]])),
    ],
)
def test_p2p(H, P, r_exp, Cov_exp):
    rate, Cov = ptp_capacity(H, P)
    assert rate == pytest.approx(r_exp, 1e-3)
    assert np.trace(Cov) == pytest.approx(P, 1e-3)
    np.testing.assert_almost_equal(Cov, Cov_exp)


@pytest.mark.parametrize("Nrx, Ntx", [(2, 2), (5, 3), (3, 5), (1, 4), (4, 1)])
def test_p2p_uplink_downlink(Nrx, Ntx):
    H = np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j
    rate_downlink, Cov_down = ptp_capacity(H, 100)
    rate_uplink, Cov_up = ptp_capacity(H.conj().T, 100)
    assert Cov_down.shape[0] == Ntx
    assert Cov_up.shape[0] == Nrx
    assert rate_downlink == pytest.approx(rate_uplink, 1e-2)


@pytest.mark.parametrize("Nrx, Ntx", [(2, 2), (5, 3), (3, 5), (1, 4), (4, 1)])
def test_p2p_vs_cvx(Nrx, Ntx):
    H = np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j
    rate, Cov = ptp_capacity(H, 100)
    assert np.trace(Cov) == pytest.approx(100, 1e-3)
    rate_cvx, Cov_cvx = ptp_capacity_cvx(H, 100)
    assert np.real(np.trace(Cov_cvx)) == pytest.approx(100, 1e-3)
    assert rate == pytest.approx(rate_cvx, 1e-3)


@pytest.mark.parametrize("MAC_fun", [MAC_cvx, MAC])
@pytest.mark.parametrize("Ntxs, Nrx", [([1, 2, 3], 2)])
def test_MAC(MAC_fun, Ntxs, Nrx):
    Hs = []
    for Ntx in Ntxs:
        Hs.append(np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j)

    rates_cvx, MAC_Covs, order = MAC_fun(Hs, 100, [1 for _ in Ntxs])
    rates = []
    for user, (MAC_Cov, Ntx, H, r_cvx) in enumerate(zip(MAC_Covs, Ntxs, Hs, rates_cvx)):
        Ntx1, Ntx2 = MAC_Cov.shape
        assert Ntx1 == Ntx2
        assert Ntx1 == Ntx
        ind = order.index(user)
        IPN = np.eye(Nrx)
        for k in order[ind + 1 :]:
            IPN = IPN + Hs[k] @ MAC_Covs[k] @ Hs[k].conj().T
        HCH = H @ MAC_Cov @ H.conj().T
        r = (logdet(IPN + HCH) - logdet(IPN)) / np.log(2)
        assert r == pytest.approx(r_cvx, 1e-3)
        rates.append(r)
    assert sum([np.trace(MAC_Cov) for MAC_Cov in MAC_Covs]) == pytest.approx(100, 1e-3)
    assert sum(rates) == pytest.approx(sum(rates_cvx), 1e-3)


@pytest.mark.parametrize("Ntxs, Nrx", [([2], 2)])
def test_MAC_ptp(Ntxs, Nrx):
    Hs = []
    for Ntx in Ntxs:
        Hs.append(np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j)

    rate, MAC_Covs, order = MAC(Hs, 1, [1])
    rate_cvx, MAC_Covs_cvx = ptp_capacity(Hs[0], 1)

    assert sum([np.trace(MAC_Cov) for MAC_Cov in MAC_Covs]) == pytest.approx(1, 1e-2)
    assert sum(rate) == pytest.approx(rate_cvx, 1e-9)


@pytest.mark.parametrize("Ntxs, Nrx", [([1, 2, 3], 2)])
def test_MAC_vs_MACcvx(Ntxs, Nrx):
    Hs = [
        np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j for Ntx in Ntxs
    ]
    w = list(range(1, len(Ntxs) + 1))
    w = [w * 3 for w in w]
    # broadcast, user with larges weight is encoded first https://arxiv.org/pdf/0901.2401.pdf
    # MAC decoding order is inverse that is, user with largest weight es decoded last
    for weights in itertools.permutations(w):
        rates_MAC_cvx, MAC_Covs_cvx, order = MAC_cvx(Hs, 100, weights)
        rates_MAC, MAC_Covs, order = MAC(Hs, 100, weights)
        assert sum([np.trace(MAC_cov) for MAC_cov in MAC_Covs]) == pytest.approx(100, 1e-3)
        assert rates_MAC == pytest.approx(rates_MAC_cvx, abs=0.1)


@pytest.mark.parametrize("power", [3, 8, 9, 12, 100])
def test_waterfilling(power):

    gains = [0.5, 0.2, 0.1]
    p1 = water_filling_cvx(gains, power)
    assert sum(p1) == pytest.approx(power, 1e-3)
    p2 = water_filling_iter(gains, power)
    assert sum(p2) == pytest.approx(power, 1e-3)
    # the water level is 1/gain + power for all active channels
    water_level = [1 / g + p for g, p in zip(gains, p2) if p > 0]
    for w in water_level:
        assert w == pytest.approx(water_level[0], 1e-3)

    for a, b in zip(p1, p2):

        assert b == pytest.approx(a if a > 1e-3 else 0)


def test_project_covariance():
    Covs = [
        np.random.random([N, 5]) + np.random.random([N, 5]) * 1j for N in range(1, 5)
    ]
    Covs = [Cov @ Cov.conj().T for Cov in Covs]
    for P in [1, 10, 100]:
        pCovs = project_covariances(Covs, P)
        pCovs_cvx = project_covariance_cvx(Covs, P)
        # both matrix sets fulfill power constraint and have equal distance to the original set
        assert sum([np.trace(Cov).real for Cov in pCovs]) - 1e-3 <= P
        assert sum([np.trace(Cov).real for Cov in pCovs_cvx]) - 1e-3 <= P
        d = sum([np.sum((Cov - pCov) ** 2) for Cov, pCov in zip(Covs, pCovs)])
        d_cvx = sum([np.sum((Cov - pCov) ** 2) for Cov, pCov in zip(Covs, pCovs_cvx)])
        assert d == pytest.approx(d_cvx, 1e-3)
