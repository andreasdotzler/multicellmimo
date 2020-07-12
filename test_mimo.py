import logging

import numpy as np
import pytest

from .mimo import (MAC_cvx, MACtoBCtransformation, inv_sqrtm,
                   MAC,
                   project_eigenvalues_to_given_sum_cvx, ptp_capacity,
                   ptp_capacity_cvx, water_filling_cvx, water_filling_iter)

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("N,M", [(4, 4), (2, 4), (4, 2)])
def test_inv_sqrtm(N, M):
    H = np.random.random([N, M]) + np.random.random([N, M]) * 1j
    HH = np.eye(N) + H @ H.conj().T
    A = inv_sqrtm(HH)
    np.testing.assert_almost_equal(A @ A, np.linalg.inv(HH))


@pytest.mark.parametrize("Ms_antennas, Bs_antennas", [(4, 4), (2, 4), (4, 2)])
def test_MACtoBCtransformation(Ms_antennas, Bs_antennas):
    P = 100
    H = (
        np.random.random([Ms_antennas, Bs_antennas]) + np.random.random([Ms_antennas, Bs_antennas]) * 1j
    )

    # H = np.array([[1j, 0], [0, 1]])
    rate_MAC, MAC_Cov = ptp_capacity(H.conj().T, P)
    assert MAC_Cov.shape[0] == Ms_antennas
    rate_BC, BC_Cov = ptp_capacity(H, P)
    assert BC_Cov.shape[0] == Bs_antennas

    BC_Cov_trans = MACtoBCtransformation([H], [MAC_Cov], [0])
    np.testing.assert_almost_equal(H @ H.conj().T @ MAC_Cov, H @ BC_Cov @ H.conj().T)
    np.testing.assert_almost_equal(H.conj().T @ MAC_Cov @ H, H.conj().T @ H @ BC_Cov)
    np.testing.assert_allclose(H @ H.conj().T @ MAC_Cov, H @ BC_Cov_trans[0] @ H.conj().T, atol=1e-3)
    np.testing.assert_allclose(H.conj().T @ MAC_Cov @ H, H.conj().T @ H @ BC_Cov_trans[0], atol=1e-3)


@pytest.mark.parametrize("P", range(10))
def test_project_eigenvalues_to_given_sum_cvx(P):
    aa = np.array([1, 2, 3, 4])
    projected = project_eigenvalues_to_given_sum_cvx(aa, P)
    assert sum(projected) == pytest.approx(P)


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
    assert rate == pytest.approx(r_exp)
    assert np.trace(Cov) == pytest.approx(P)
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
    rate_cvx, Cov_cvx = ptp_capacity(H, 100)
    assert np.trace(Cov_cvx) == pytest.approx(100, 1e-3)
    assert rate == pytest.approx(rate_cvx, 1e-3)
    np.testing.assert_almost_equal(Cov, Cov_cvx)


@pytest.mark.parametrize("MAC_fun", [MAC_cvx, MAC])
@pytest.mark.parametrize("Ntxs, Nrx", [([1, 2, 3], 2)])
def test_MAC(MAC_fun, Ntxs, Nrx):
    Hs = []
    for Ntx in Ntxs:
        Hs.append(np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j)

    rates_cvx, MAC_Covs = MAC_fun(Hs, 100, [1 for _ in Ntxs])
    logdet = np.linalg.slogdet
    Z = np.eye(Nrx)
    rates = []
    for MAC_Cov, Ntx, H, r_cvx in zip(MAC_Covs, Ntxs, Hs, rates_cvx):
        Ntx1, Ntx2 = MAC_Cov.shape
        assert Ntx1 == Ntx2
        assert Ntx1 == Ntx
        Znew = Z + H @ MAC_Cov @ H.conj().T
        r = (logdet(Znew)[1].real - logdet(Z)[1].real) / np.log(2)
        Z = Znew
        assert r == pytest.approx(r_cvx, 1e-3)
        rates.append(r)
    assert sum([np.trace(MAC_Cov) for MAC_Cov in MAC_Covs]) == pytest.approx(100, 1e-3)
    assert sum(rates) == pytest.approx(sum(rates_cvx), 1e-3)


@pytest.mark.parametrize("Ntxs, Nrx", [([2], 2)])
def test_MAC_ptp(Ntxs, Nrx):
    Hs = []
    for Ntx in Ntxs:
        Hs.append(np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j)

    rate, MAC_Covs = MAC(Hs, 1, [1])
    rate_cvx, MAC_Covs_cvx = ptp_capacity(Hs[0], 1)

    assert sum([np.trace(MAC_Cov) for MAC_Cov in MAC_Covs]) == pytest.approx(1, 1e-2)
    assert sum(rate) == pytest.approx(rate_cvx, 1e-9)


@pytest.mark.parametrize("Ntxs, Nrx", [([1, 2, 3], 2)])
def test_MAC_vs_MACcvs(Ntxs, Nrx):
    Hs = [
        np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j for Ntx in Ntxs
    ]
    weights = [1, 1]
    rates_MAC_cvx, MAC_Covs_cvx  = MAC_cvx(Hs, 100, weights)
    rates_MAC, MAC_Covs = MAC(Hs, 100, weights)
    assert sum([np.trace(MAC_cov) for MAC_cov in MAC_Covs]) == pytest.approx(100, 1e-3)
    assert sum(rates_MAC) == pytest.approx(sum(rates_MAC_cvx), 1e-3)


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
        assert w == pytest.approx(water_level[0])

    for a, b in zip(p1, p2):

        assert b == pytest.approx(a if a > 1e-3 else 0)
