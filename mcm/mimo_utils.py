"""COPYRIGTH."""
import numpy as np

from .typing import Matrix
from typing import List


inv = np.linalg.inv
pinv = np.linalg.pinv
det = np.linalg.det
log = np.log
log2 = np.log2
eye = np.eye


def logdet(X: Matrix) -> float:
    """Compute log(det(X)).

    Parameters
    ----------
    X: Matrix
        positive semi-definite matrix

    Returns
    -------
    float
        result

    """
    return log(np.real(det(X)))


def pinv_sqrtm(A: Matrix, rcond: float = 1e-3) -> Matrix:
    """TDOD write a test.

    Parameters
    ----------
    A: Matrix
        positive semi-definite matrix
    rcond: float
        threshold for pseudo-inverse

    Returns
    -------
    B: Matrix

    """
    ei_d, V_d = np.linalg.eigh(A)
    pos = ei_d > rcond
    return V_d[:, pos] @ np.diag(ei_d[pos] ** -0.5) @ V_d[:, pos].conj().T


def inv_sqrtm(A):
    """Square root of the inverse of a positive semi-definite matrix.

    Computes :math:`B` with :math:`BB=A^{-1}`

    Parameters
    ----------
    A: Matrix
        positive semi-definite matrix

    Returns
    -------
    B: Matrix
       positive semi-definite matrix
    """
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d**-0.5) @ V_d.conj().T


def sqrtm(A: Matrix) -> Matrix:
    """Square root of a positive semi-definite matrix.

    Computes :math:`B` with :math:`BB=A`

    Parameters
    ----------
    A: Matrix
        positive semi-definite matrix

    Returns
    -------
    B: Matrix
       positive semi-definite matrix

    """
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d**0.5) @ V_d.conj().T


def argsort(weights: List[float], reverse: bool = False):
    """Sort parameters and provide list of indices.

    # https://stackoverflow.com/a/6618543

    Parameters
    ----------
    weights: List[float]
        weights
    reverse: bool
        if True, sorted TODO if false. sorted

    Returns
    -------
    order: List[int]
    """
    return [
        o
        for o, _ in sorted(
            enumerate(weights), reverse=reverse, key=lambda pair: pair[1]
        )
    ]
