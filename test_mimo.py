import logging

import numpy as np
import pytest

from .mimo import (MAC, MACtoBCtransformation, inv_sqrtm,
                   maximize_weighted_sum_rate_in_Q,
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
    rate_MAC, Q_MAC = ptp_capacity(H.conj().T, P)
    assert Q_MAC.shape[0] == Ms_antennas
    rate_BC, Q_BC = ptp_capacity(H, P)
    assert Q_BC.shape[0] == Bs_antennas

    Q_trans = MACtoBCtransformation([H], [Q_MAC], [0])
    np.testing.assert_almost_equal(H @ H.conj().T @ Q_MAC, H @ Q_BC @ H.conj().T)
    np.testing.assert_almost_equal(H.conj().T @ Q_MAC @ H, H.conj().T @ H @ Q_BC)


@pytest.mark.parametrize("P", range(10))
def test_project_eigenvalues_to_given_sum_cvx(P):
    aa = np.array([1, 2, 3, 4])
    projected = project_eigenvalues_to_given_sum_cvx(aa, P)
    assert sum(projected) == pytest.approx(P)


@pytest.mark.parametrize(
    "H, P, r_exp, Q_exp",
    [
        (np.array([[1, 0], [0, 0]]), 1, 1, np.array([[1, 0], [0, 0]])),
        (np.array([[1j, 0], [0, 0]]), 1, 1, np.array([[1, 0], [0, 0]])),
        (np.array([[1, 0], [0, 1]]), 2, 2, np.array([[1, 0], [0, 1]])),
        (np.array([[1j, 0], [0, 1]]), 2, 2, np.array([[1, 0], [0, 1]])),
        (np.array([[0, 1j], [1j, 0]]), 2, 2, np.array([[1, 0], [0, 1]])),
    ],
)
def test_p2p(H, P, r_exp, Q_exp):
    rate, Q = ptp_capacity(H, P)
    assert rate == pytest.approx(r_exp)
    assert np.trace(Q) == pytest.approx(P)
    np.testing.assert_almost_equal(Q, Q_exp)


@pytest.mark.parametrize("Nrx, Ntx", [(2, 2), (5, 3), (3, 5), (1, 4), (4, 1)])
def test_p2p_uplink_downlink(Nrx, Ntx):
    H = np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j
    rate_downlink, Q_down = ptp_capacity(H, 100)
    rate_uplink, Q_up = ptp_capacity(H.conj().T, 100)
    assert Q_down.shape[0] == Ntx
    assert Q_up.shape[0] == Nrx
    assert rate_downlink == pytest.approx(rate_uplink, 1e-2)


@pytest.mark.parametrize("Nrx, Ntx", [(2, 2), (5, 3), (3, 5), (1, 4), (4, 1)])
def test_p2p_vs_cvx(Nrx, Ntx):
    H = np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j
    rate, Q = ptp_capacity(H, 100)
    assert np.trace(Q) == pytest.approx(100, 1e-3)
    rate_cvx, Q_cvx = ptp_capacity(H, 100)
    assert np.trace(Q_cvx) == pytest.approx(100, 1e-3)
    assert rate == pytest.approx(rate_cvx, 1e-3)
    np.testing.assert_almost_equal(Q, Q_cvx)


@pytest.mark.parametrize("MAC_fun", [MAC, maximize_weighted_sum_rate_in_Q])
@pytest.mark.parametrize("Ntxs, Nrx", [([1, 2, 3], 2)])
def test_MAC(MAC_fun, Ntxs, Nrx):
    Hs = []
    for Ntx in Ntxs:
        Hs.append(np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j)

    rates_cvx, Qs = MAC_fun(Hs, 100, [1 for _ in Ntxs])
    logdet = np.linalg.slogdet
    Z = np.eye(Nrx)
    rates = []
    for Q, Ntx, H, r_cvx in zip(Qs, Ntxs, Hs, rates_cvx):
        Ntx1, Ntx2 = Q.shape
        assert Ntx1 == Ntx2
        assert Ntx1 == Ntx
        Znew = Z + H @ Q @ H.conj().T
        r = (logdet(Znew)[1].real - logdet(Z)[1].real) / np.log(2)
        Z = Znew
        assert r == pytest.approx(r_cvx, 1e-3)
        rates.append(r)
    assert sum([np.trace(Q) for Q in Qs]) == pytest.approx(100, 1e-3)
    assert sum(rates) == pytest.approx(sum(rates_cvx), 1e-3)


@pytest.mark.parametrize("Ntxs, Nrx", [([2], 2)])
def test_MACQ_ptp(Ntxs, Nrx):
    Hs = []
    for Ntx in Ntxs:
        Hs.append(np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j)

    rate_Q, Qs = maximize_weighted_sum_rate_in_Q(Hs, 1, [1])
    rate_cvx, Q_cvx = ptp_capacity(Hs[0], 1)

    assert sum([np.trace(Q) for Q in Qs]) == pytest.approx(1, 1e-2)
    assert sum(rate_Q) == pytest.approx(rate_cvx, 1e-9)


@pytest.mark.parametrize("Ntxs, Nrx", [([1, 2, 3], 2)])
def test_MAC_vs_MACQ(Ntxs, Nrx):
    Hs = [
        np.random.random([Nrx, Ntx]) + np.random.random([Nrx, Ntx]) * 1j for Ntx in Ntxs
    ]
    weights = [1, 1]
    rates_MAC, QMACs = MAC(Hs, 100, weights)
    rates_MACQ, QMACQs = maximize_weighted_sum_rate_in_Q(Hs, 100, weights)
    assert sum([np.trace(Q) for Q in QMACQs]) == pytest.approx(100, 1e-3)
    assert sum(rates_MAC) == pytest.approx(sum(rates_MACQ))


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
