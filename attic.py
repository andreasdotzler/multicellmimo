def ptp_worst_case_noise_gradient_projected(
    H, P, sigma=1, max_iter_outer=20, max_iter_inner=100000
):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx) / Ntx
    rate_min = None
    for i in range(max_iter_outer):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        # import ipdb; ipdb.set_trace()
        Z_gr = -np.linalg.inv(Z) + W  # +la * P / sigma * np.eye(Ntx)
        Z_gr = project_gradient_cvx(Z_gr, 0)
        for q in range(max_iter_inner):
            Z_new = Z - 1 / 10 * (q + 1) * Z_gr
            assert np.trace(Z_new) == pytest.approx(sigma)
            if not np.all(np.linalg.eigvals(Z_new) > 0):
                continue

            rate_new, (W_new, la) = ptp_capacity_uplink_cvx_dual(H, P, Z_new)
            LOGGER.debug(
                f"Outer iteration {i} Min_rate {rate_min}, inner iteration{q} new rate {rate_new}"
            )
            if rate_min is None or rate_new < rate_min:
                rate_min = rate_new
                Z = Z_new
                W = W_new
                break
        if q == max_iter_inner - 1:
            LOGGER.debug(f"Outer iteration {i} - Maximal inner iterations reached {q}")
            break
        assert np.all(np.linalg.eigvals(Z) > 0)
    return rate_i, (Z, W)


def ptp_worst_case_noise_gradient(
    H, P, sigma=1, max_iter_outer=100, max_iter_inner=100
):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx)
    W = np.eye(Ntx)
    rate_min = None
    for i in range(max_iter_outer):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        Z_gr = -np.linalg.inv(Z) + W  # +la * P / sigma * np.eye(Ntx)
        Z_gr = -np.linalg.pinv(Z, rcond=1e-9, hermitian=True) + W
        for q in range(max_iter_inner):
            Z_new = Z - 1 / (10 * q + 1) * Z_gr
            Z_new = project_covariances([Z_new], sigma)[0]
            if not np.all(np.linalg.eigvals(Z_new) > 0):
                continue

            rate_new, (W_new, la) = ptp_capacity_uplink_cvx_dual(H, P, Z_new)
            LOGGER.debug(
                f"Outer iteration {i} Min_rate {rate_min}, inner iteration{q} new rate {rate_new}"
            )
            if rate_min is None or rate_new < rate_min:
                rate_min = rate_new
                Z = Z_new
                W = W_new
                break
        if q == max_iter_inner - 1:
            break
        assert np.all(np.linalg.eigvals(Z) > 0)
    return rate_i, (Z, W)


def project_gradient_cvx(X, P):
    p_gra = cp.Variable([X.shape[0], X.shape[0]], hermitian=True)
    obj = cp.Minimize(cp.sum_squares(p_gra - X))
    constraint = cp.real(cp.trace(p_gra)) == P
    prob = cp.Problem(obj, [constraint])
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert prob.status == "optimal"
    return p_gra.value


def ptp_capacity_uplink_cvx_dual(H, P, Z):
    # assert np.all(np.linalg.eigvals(Z) > 0)
    Nrx, Ntx = H.shape
    W = cp.Variable([Ntx, Ntx], hermitian=True)
    la = cp.Variable(1)
    cost = -cp.real(cp.log_det(W)) + cp.real(cp.trace(Z @ W)) + P * la - Ntx
    positivity_W = W >> 0
    postitvity_la = la >= 0
    cons = cp.multiply(la, np.eye(Nrx)) >> H @ W @ H.conj().T
    constraints = [cons, positivity_W, postitvity_la]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert "optimal" in prob.status
    assert np.all(np.linalg.eigvals(W.value) > 0)
    return np.real(prob.value - log(det(Z))), (W.value, la.value)


def ptp_noise_cvx_dual(H, P, W, la):
    Nrx, Ntx = H.shape
    Z = cp.Variable([Ntx, Ntx])
    cost = (
        log(det(W)) - cp.log_det(Z) + cp.trace(Z @ W) + P / Ntx * la * cp.trace(Z) - Ntx
    )
    cost = -cp.log_det(Z) + cp.trace(Z @ W) + P / Ntx * la * cp.trace(Z)
    positivity_Z = Z >> 0
    constraints = [positivity_Z]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert prob.status == "optimal"
    return prob.value, Z.value


def worst_case_noise_uplink(W, sigma):
    Ntx, _ = W.shape
    Z = cp.Variable([Ntx, Ntx], hermitian=True)
    cost = -cp.real(cp.log_det(Z)) + cp.real(cp.trace(Z @ W))
    power = cp.real(cp.trace(Z)) == sigma
    positivity = Z >> 0
    constraints = [power, positivity]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value, Z.value


def ptp_worst_case_noise_alternating(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx) / Ntx * sigma
    rate_o = None
    for i in range(1000):
        assert np.trace(Z) == pytest.approx(sigma)
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
        rate_o, Z = worst_case_noise_uplink(W, sigma)
        rate_o = rate_o - np.real(log(det(W))) + P * la - Ntx
        assert np.all(np.linalg.eigvals(Z) > 0)
        LOGGER.debug(
            f"iteration {i} - rate inner: {rate_i} - rate outer {rate_o} - diff: {rate_i-rate_o}"
        )
        if np.isclose(rate_i, rate_o, 1e-5):
            break
    return rate_i, (Z, W)


def ptp_worst_case_noise_unconstraint(H, P, sigma=1):
    Nrx, Ntx = H.shape
    Z = np.eye(Ntx)
    rate_o = None
    for i in range(100):
        rate_i, (W, la) = ptp_capacity_uplink_cvx_dual(H, P / sigma, Z)
        # rate_o, Z = ptp_noise_cvx_dual(H, P, W, la)
        Z = inv(W + la * P / sigma * np.eye(Ntx))
        assert np.all(np.linalg.eigvals(Z) > 0)
        LOGGER.debug(f"rate inner: {rate_i} : rate outer {rate_o}")
    return rate_i, (Z, W)


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Bs_antennas", [1, 2, 3])
@pytest.mark.parametrize("Ms_antennas", [1, 2, 3])
def test_uplink_noise_dual(comp, Bs_antennas, Ms_antennas):
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
    rate_p, Sigma_eff = ptp_capacity(H_eff, P)
    rate_i, Sigma = ptp_capacity(H.conj().T, P, Z)
    assert rate_i == pytest.approx(rate_p, 1e-2)
    rate_d, (W, la) = ptp_capacity_uplink_cvx_dual(H, P, Z)
    L = sqrtm(Z) @ W @ sqrtm(Z)
    X = inv(L)
    assert log(det(X)) == pytest.approx(rate_i, 1e-2)
    assert rate_i == pytest.approx(rate_d, 1e-2)
