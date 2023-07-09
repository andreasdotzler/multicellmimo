import pytest
import numpy as np
import logging

from mcm.mimo_utils import eye

LOGGER = logging.getLogger(__name__)

# -*- coding: utf-8 -*-
# https://stackoverflow.com/a/62563106
import os

if os.getenv("_PYTEST_RAISE", "0") != "0":
    import pytest

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):  # type: ignore
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):  # type: ignore
        raise excinfo.value


@pytest.fixture(scope="function", autouse=True)
def seed():
    np.random.seed(52)


@pytest.fixture(scope="function", params=[[2, 2], [10, 40]])
def A(request):
    if request.param == [2, 2]:
        return np.array([[1, 0.0], [0.0, 1]]).T
    else:
        return np.random.rand(*request.param) * 3


@pytest.fixture()
def H(Ms_antennas, Bs_antennas, comp=0):
    return random_channel_matrix(Ms_antennas, Bs_antennas, comp=0)


def random_channel_matrix(Ms_antennas, Bs_antennas, comp=0):
    LOGGER.debug(
        f"creating random {'complex' if comp else 'real'} {Ms_antennas}x{Bs_antennas} channel"
    )
    if comp:
        H = np.random.default_rng().normal(
            loc=0, scale=np.sqrt(2) / 2, size=(Ms_antennas, Bs_antennas)
        ) + 1j * np.random.default_rng().normal(
            loc=0, scale=np.sqrt(2) / 2, size=(Ms_antennas, Bs_antennas)
        )
    else:
        H = np.random.default_rng().normal(
            loc=0, scale=1, size=(Ms_antennas, Bs_antennas)
        )
    return H


@pytest.fixture()
def H_MAC(Ms_antennas_list, Bs_antennas, comp=0):
    return [
        random_channel_matrix(Ms_antennas, Bs_antennas, comp)
        for Ms_antennas in Ms_antennas_list
    ]


def random_covariance(N, comp=0, scale=None):
    scale = scale or N
    Cov = np.random.default_rng().normal(
        loc=0, scale=1, size=(N, N)
    ) + 1j * comp * np.random.default_rng().normal(loc=0, scale=1, size=(N, N))
    Cov = eye(N) + Cov @ Cov.conj().T
    Cov = Cov / np.trace(Cov) * scale
    return Cov


@pytest.fixture()
def B(Ms_antennas, comp=0, sigma=None):
    return random_covariance(Ms_antennas, comp, sigma)


@pytest.fixture()
def C(Bs_antennas, comp=0, P=None):
    return random_covariance(Bs_antennas, comp, P)
