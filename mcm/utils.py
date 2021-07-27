import numpy as np

inv = np.linalg.inv
pinv = np.linalg.pinv
det = np.linalg.det
log = np.log
log2 = np.log2
eye = np.eye


def logdet(X):
    return log(np.real(det(X)))


def log2det(X):
    return log2(np.real(det(X)))


def pinv_sqrtm(A, rcond=1e-3):
    ei_d, V_d = np.linalg.eigh(A)
    pos = ei_d > rcond
    return V_d[:, pos] @ np.diag(ei_d[pos] ** -0.5) @ V_d[:, pos].conj().T


def inv_sqrtm(A):
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d ** -0.5) @ V_d.conj().T


def sqrtm(A):
    ei_d, V_d = np.linalg.eigh(A)
    return V_d @ np.diag(ei_d ** 0.5) @ V_d.conj().T
