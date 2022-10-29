def V_test(c_0, c_1, c_2, c_3):
    f = cp.Variable(4, nonneg=True)
    q_0 = cp.Variable(10, nonneg=True)
    q_1 = cp.Variable(10, nonneg=True)
    q_2 = cp.Variable(10, nonneg=True)
    q_3 = cp.Variable(10, nonneg=True)

    q = cp.sum([f[0] * c_0, f[1] * c_1, f[2] * c_2, f[3] * c_3])
    prob = cp.Problem(cp.Maximize(proportional_fair(q)),[cp.sum(f) == 1])
    prob.solve()
    assert "optimal" in prob.status
    util = sum(np.log(f[0].value*c_0 + f[1].value*c_1 + f[2].value*c_2 + f[3].value*c_3))
    assert util == pytest.approx(prob.value, 1e-3) 
    return q_0, q_1, q_2, q_3, q, prob, util


def V_test_2(c_0, c_1, c_2, c_3, Q):
    f = cp.Variable(2, nonneg=True)
    q_a = cp.sum([f[0] * c_0, f[1] * c_1])
    q_b = cp.sum([f[0] * c_2, f[1] * c_3])
    cost = proportional_fair(q_a) + proportional_fair(q_b)
    prob = cp.Problem(cp.Maximize(cost),[cp.sum(f) == 1] + Q.constraints(q_a) + Q.constraints(q_b))
    prob.solve()
    assert "optimal" in prob.status
    util = sum(np.log(f[0].value * c_0 + f[1].value * c_1))
    util += sum(np.log(f[0].value * c_2 + f[1].value * c_3))
    return f, q_a, q_b, prob, util

def test_V_novel_2():
    n_user = 10
    x_0 = np.random.random(n_user)
    x_1 = np.random.random(n_user)
    x_2 = np.random.random(n_user)
    x_3 = np.random.random(n_user)

    q_0 = cp.Variable(n_user, nonneg=True)
    q_1 = cp.Variable(n_user, nonneg=True)
    q_2 = cp.Variable(n_user, nonneg=True)
    q_3 = cp.Variable(n_user, nonneg=True)

    q_a = cp.sum([q_0, q_1])
    q_b = cp.sum([q_2, q_3])
    Q = Q_vector(np.zeros(10), np.ones(10)*3)

    mm_a = cp.sum([x_0 @ q_0, x_1 @ q_1])
    mm_b = cp.sum([x_2 @ q_2, x_3 @ q_3])
    prob_a = cp.Problem(cp.Maximize(proportional_fair(q_a) - mm_a), Q.constraints(q_a))

    prob_b = cp.Problem(cp.Maximize(proportional_fair(q_b) - mm_b), Q.constraints(q_b))
    prob_a.solve()
    prob_b.solve()
    #val_conf = prob.value
    c_0 = q_0.value
    c_1 = q_1.value
    c_2 = q_2.value
    c_3 = q_3.value
    la_1 = {"r1": x_0, "r2": x_1}
    la_2 = {"r1": x_2, "r2": x_3}
    la_m_t = {"r1": {"a": x_0, "b": x_2}, "r2": {"a": x_1, "b": x_3}}
    modes = {"r1", "r2"}
    T_1 = Transmitter({m: R_m_t(list(range( 0, 10)), I_C(None)) for m in modes}, "a")
    T_2 = Transmitter({m: R_m_t(list(range(10, 20)), I_C(None)) for m in modes}, "b")
    network = Network({T_1.id: T_1, T_2.id: T_2})
    val, q_m_t, c_m_t = V_conj(network, proportional_fair, la_m_t, Q_vector(np.zeros(20), np.ones(20)*3))
    prob_a_K, q_m_1 = K_conj(proportional_fair, Q, la_1)
    prob_b_K, q_m_2 = K_conj(proportional_fair, Q, la_2)
    assert prob_a.value == pytest.approx(prob_a_K, 1e-3)
    assert prob_b.value == pytest.approx(prob_b_K, 1e-3)
    assert val == pytest.approx(prob_a_K + prob_b_K, 1e-3)
    assert q_m_t["r1"]["a"] == pytest.approx(q_m_1["r1"], 1e-3, abs=1e-3)
    assert q_m_t["r1"]["b"] == pytest.approx(q_m_2["r1"], 1e-3, abs=1e-3)
    assert q_m_t["r2"]["a"] == pytest.approx(q_m_1["r2"], 1e-3, abs=1e-3)
    assert q_m_t["r2"]["b"] == pytest.approx(q_m_2["r2"], 1e-3, abs=1e-3)
    assert q_m_t["r1"]["a"] == pytest.approx(c_0, 1e-3, abs=1e-3)
    assert q_m_t["r1"]["b"] == pytest.approx(c_2, 1e-3, abs=1e-3)
    assert q_m_t["r2"]["a"] == pytest.approx(c_1, 1e-3, abs=1e-3)
    assert q_m_t["r2"]["b"] == pytest.approx(c_3, 1e-3, abs=1e-3)
    s_1 = (x_0 @ c_0 + x_2 @ c_2)
    s_2 = (x_1 @ c_1 + x_3 @ c_3) 
    s = s_1 + s_2
    f, q_a, q_b, prob, util = V_test_2(c_0 * s / s_1, c_1 * s / s_2, c_2 * s / s_1, c_3 * s / s_2, Q)

    assert prob_a.value + prob_b.value + mm_a.value + mm_b.value == pytest.approx(util, 1e-3)
    assert c_m_t["r1"]["a"] == pytest.approx(c_0 * s / s_1, 1e-3, abs=1e-3)
    assert c_m_t["r1"]["b"] == pytest.approx(c_2 * s / s_1, 1e-3, abs=1e-3)
    assert c_m_t["r2"]["a"] == pytest.approx(c_1 * s / s_2, 1e-3, abs=1e-3)
    assert c_m_t["r2"]["b"] == pytest.approx(c_3 * s / s_1, 1e-3, abs=1e-3)
    val, q_m_t, c_m_t = V_conj(network, proportional_fair, la_m_t, Q_vector(np.zeros(20), np.ones(20)*3))
    
def test_MACtoBCtransformation_with_noise():
    Ms_antennas = Bs_antennas = 3
    H = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    P = 100
    sigma = 10
    # white uplink transmit  covariance
    Sigma_white = P / Ms_antennas * eye(Ms_antennas)
    # white uplink noise covariance
    Omega_white = sigma / Bs_antennas * eye(Bs_antennas)
    rate_uplink = log(det(eye(Bs_antennas) + inv(Omega_white) @ H.conj().T @ Sigma_white @ H))
    H_eff_uplink = H @ inv_sqrtm(Omega_white)
    MAC_rates_calc = MAC_rates([Sigma_white], [H_eff_uplink.T], [0])
    assert MAC_rates_calc[0] == pytest.approx(rate_uplink, 1e-3)

    BC_Cov_trans = MACtoBCtransformation([H_eff_uplink], [Sigma_white], [0])
    BC_rates_calc = BC_rates(BC_Cov_trans, [H_eff_uplink], [0])
    assert MAC_rates_calc == pytest.approx(BC_rates_calc, 1e-3)
    

def ptp_capacity_correction_cvx(HTRH, C, Q):
    Nrx, Ntx = HTRH.shape
    Cov = cp.Variable([Ntx, Ntx], hermitian=True)
    alpha = cp.Variable(1)
    I = np.eye(Ntx)
    cost = cp.log_det(I + Cov @ HTRH)
    constraints = [Cov << C, Cov == alpha * Q, Cov >> 0]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.SCS, eps=1e-9)
    return prob.value, Cov.value


def done_here():
    rate_trans_1 = log(
        det(
            eye(3)
            + inv_sqrtm(Omega)
            @ sqrtm(C)
            @ H.conj().T
            @ inv_sqrtm(B)
            @ Sigma
            @ inv_sqrtm(B)
            @ H
            @ sqrtm(C)
            @ inv_sqrtm(Omega)
        )
    )
    assert rate_trans_1 == pytest.approx(rate_ref)

    I = eye(Ms_antennas)
    # TODO need to do this for all Sigmas, rate_trans_1 will help
    Sigma_eff_BC = -inv(I + Heff_BC@Heff_BC.conj().T) + inv(I)
    Q = -inv(Omega + H.conj().T @ Sigma @ H) + inv(Omega)
    Z = np.zeros((Ms_antennas, Ms_antennas))
    Y_o = np.zeros((Bs_antennas, Bs_antennas))
    assert np.allclose(sigmaBplusYo, R)

    rate_d = logdet(np.eye(Ms_antennas) + np.linalg.inv(R) @ H @ Q @ H.conj().T)
    Xi = Omega
    S = H @ inv(Xi + H.conj().T @ Sigma @ H) @ H.conj().T + M
    rate_h = log(det(S))
    assert rate_u == pytest.approx(rate_d, 1e-2)
    Q_trans_1 = MACtoBCtransformation([H], [Sigma], [0])

    rate_trans_1 = log(
        det(
            eye(3)
            + inv_sqrtm(Omega)
            @ H.conj().T
            @ inv_sqrtm(B)
            @ sqrtm(B)
            @ Sigma
            @ sqrtm(B)
            @ inv_sqrtm(B)
            @ H
            @ inv_sqrtm(Omega)
        )
    )

    assert rate_trans_1 == pytest.approx(rate_d, 1e-2)

    H_k = inv_sqrtm(B) @ H @ inv_sqrtm(Omega)
    Sigma_k = sqrtm(B) @ Sigma @ sqrtm(B)
    Q_trans = MACtoBCtransformation([H_k], [Sigma_k], [0])
    # the downlink noise here is B
    rate_trans = log(
        det(
            eye(3)
            + inv_sqrtm(B)
            @ H
            @ inv_sqrtm(Omega)
            @ Q_trans[0]
            @ inv_sqrtm(Omega)
            @ H.T
            @ inv_sqrtm(B)
        )
    )
    assert rate_trans == pytest.approx(rate_d, 1e-2)
    Q2 = inv_sqrtm(Omega) @ Q_trans[0] @ inv_sqrtm(Omega)
    assert np.trace(Q) / np.trace(R) == pytest.approx(np.trace(Q2) / np.trace(B))
    assert np.allclose(Q / np.trace(Q), Q2 / np.trace(Q2))

def test_transfer():
    # transformation including the uplink noise
    H_eff_uplink = pinv_sqrtm(Z) @ H @ pinv_sqrtm(Omega_white)
    Q_trans = MACtoBCtransformation([H_eff_uplink], [pinv_sqrtm(Z) @ Sigma_white @pinv_sqrtm(Z)], [0])[0]
    rate_downlink_trans = log(det(eye(Ms_antennas) + H_eff_uplink @ Q_trans @ H_eff_uplink.conj().T))
    assert rate_downlink_trans == pytest.approx(rate_uplink, 1e-2)


    Q_trans = inv_sqrtm(Omega_white) @ Q_trans @ inv_sqrtm(Omega_white)
    rate_downlink_trans_f = log(det(eye(Ms_antennas) + inv(Z) @ H@Q_trans@H.conj().T))
    assert rate_downlink_trans_f == pytest.approx(rate_uplink, 1e-2)
    Q_calc = -inv(Omega_white + H.conj().T @ Sigma_white @ H) + inv(Omega_white)
    Q_calc = Q_calc / np.trace(Q_calc) * P
    # check the min-max saddle point
    rate_downlink_worst_case_noise = log(det(eye(Ms_antennas) + inv(Z) @ HQHT))
    assert rate_downlink_worst_case_noise == pytest.approx(rate_uplink, 1e-2)

    assert rate_downlink == pytest.approx(rate_uplink, 1e-2)



@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [2])
@pytest.mark.parametrize("Bs_antennas", [2])
def test_rest(comp, Ms_antennas, Bs_antennas):
    H = np.random.random([Ms_antennas, Bs_antennas]) + comp * 1j * np.random.random(
        [Ms_antennas, Bs_antennas]
    )



    P = 100
    sigma = 10
    # white uplink transmit  covariance
    Sigma_white = P / Ms_antennas * eye(Ms_antennas)
    # white uplink noise covariance
    Q = -inv(Omega + H.conj().T @ Sigma @ H) + inv(Omega)
    Z = np.zeros((Ms_antennas, Ms_antennas))
    Y_o = np.zeros((Bs_antennas, Bs_antennas))
    assert np.allclose(sigmaBplusYo, R)

    rate_d = logdet(np.eye(Ms_antennas) + np.linalg.inv(R) @ H @ Q @ H.conj().T)
    Xi = Omega
    S = H @ inv(Xi + H.conj().T @ Sigma @ H) @ H.conj().T + M
    rate_h = log(det(S))
    assert rate_u == pytest.approx(rate_d, 1e-2)
    Q_trans_1 = MACtoBCtransformation([H], [Sigma], [0])

    rate_trans_1 = log(
        det(
            eye(3)
            + inv_sqrtm(Omega)
            @ H.conj().T
            @ inv_sqrtm(B)
            @ sqrtm(B)
            @ Sigma
            @ sqrtm(B)
            @ inv_sqrtm(B)
            @ H
            @ inv_sqrtm(Omega)
        )
    )

    assert rate_trans_1 == pytest.approx(rate_d, 1e-2)

    H_k = inv_sqrtm(B) @ H @ inv_sqrtm(Omega)
    Sigma_k = sqrtm(B) @ Sigma @ sqrtm(B)
    Q_trans = MACtoBCtransformation([H_k], [Sigma_k], [0])
    # the downlink noise here is B
    rate_trans = log(
        det(
            eye(3)
            + inv_sqrtm(B)
            @ H
            @ inv_sqrtm(Omega)
            @ Q_trans[0]
            @ inv_sqrtm(Omega)
            @ H.T
            @ inv_sqrtm(B)
        )
    )
    assert rate_trans == pytest.approx(rate_d, 1e-2)
    Q2 = inv_sqrtm(Omega) @ Q_trans[0] @ inv_sqrtm(Omega)
    assert np.trace(Q) / np.trace(R) == pytest.approx(np.trace(Q2) / np.trace(B))
    assert np.allclose(Q / np.trace(Q), Q2 / np.trace(Q2))


@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
@pytest.mark.parametrize("Ms_antennas", [2])
@pytest.mark.parametrize("Bs_antennas", [2])
def test_transfer_white(comp, Ms_antennas, Bs_antennas):
    H = np.random.random([Ms_antennas, Bs_antennas]) + comp * 1j * np.random.random(
        [Ms_antennas, Bs_antennas]
    )
    P = 100
    sigma = 10
    # white uplink transmit  covariance
    Sigma_white = P / Ms_antennas * eye(Ms_antennas)
    # white uplink noise covariance
    Omega_white = sigma / Bs_antennas * eye(Bs_antennas)
    rate_uplink = log(det(eye(Bs_antennas) + inv(Omega_white) @ H.conj().T @ Sigma_white @ H))

    # this is just for me to briefly verify the uplink downlink transformation
    Q_trans_1 = MACtoBCtransformation([H], [Sigma_white], [0])[0]
    rate_downlink_wo_noise = log(det(eye(Ms_antennas) + H @ Q_trans_1 @ H.conj().T))
    rate_uplink_wo_noise = log(det(eye(Bs_antennas) + H.conj().T @ Sigma_white @ H))
    assert rate_downlink_wo_noise == pytest.approx(rate_uplink_wo_noise)
 
    # transformation including the uplink noise
    H_eff_uplink = H @ inv_sqrtm(Omega_white)
    Q_trans_w_noise = MACtoBCtransformation([H_eff_uplink], [Sigma_white], [0])[0]
    rate_downlink_w_noise = log(det(eye(Ms_antennas) + H_eff_uplink @ Q_trans_w_noise @ H_eff_uplink.conj().T))
    rate_uplink_w_noise = log(det(eye(Bs_antennas) + H_eff_uplink.conj().T @ Sigma_white @ H_eff_uplink))
    assert rate_downlink_w_noise == pytest.approx(rate_uplink_w_noise)

    HQHT = H @ Q_trans_w_noise @ H.conj().T
    rate, Z = ptp_worst_case_noise_static(HQHT, sigma / P * np.trace(Q_trans_w_noise))
    rate_downlink = log(det(eye(Ms_antennas) + inv(Z) @ HQHT))
    assert rate_downlink == pytest.approx(rate_uplink, 1e-2)

    # transformation including the uplink noise
    H_eff_uplink_2 = pinv_sqrtm(Z) @ H @ inv_sqrtm(Omega_white)
    Q_trans = MACtoBCtransformation([H_eff_uplink_2], [sqrtm(Z) @ Sigma_white @ sqrtm(Z)], [0])[0]
    rate_downlink_trans = log(det(eye(Ms_antennas) + H_eff_uplink_2 @ Q_trans @ H_eff_uplink_2.conj().T))
    rate_uplink_trans = log(det(eye(Bs_antennas) + H_eff_uplink_2.conj().T @ sqrtm(Z) @ Sigma_white @ sqrtm(Z) @ H_eff_uplink_2))
    assert rate_downlink_trans == pytest.approx(rate_uplink_trans, 1e-2)
    assert rate_downlink_trans == pytest.approx(rate_uplink, 1e-2)

    Q_trans = inv_sqrtm(Omega_white) @ Q_trans @ inv_sqrtm(Omega_white)
    rate_downlink_trans_f = log(det(eye(Ms_antennas) + inv(Z) @ H@Q_trans@H.conj().T))
    assert rate_downlink_trans_f == pytest.approx(rate_uplink, 1e-2)
    Q_calc = -inv(Omega_white + H.conj().T @ Sigma_white @ H) + inv(Omega_white)
    Q_calc = Q_calc / np.trace(Q_calc) * P
    # check the min-max saddle point
    rate_downlink_worst_case_noise = log(det(eye(Ms_antennas) + inv(Z) @ HQHT))
    assert rate_downlink_worst_case_noise == pytest.approx(rate_uplink, 1e-2)

    assert rate_downlink == pytest.approx(rate_uplink, 1e-2)
@pytest.mark.parametrize("comp", [0, 1], ids=["real", "complex"])
def test_noise_rank_def(comp):
    P = 100
    assert (3, 2) == H.shape
    Z = np.matrix([[2, 0, 0], [0, 4, 0], [0, 0, 0]])
    rate_i, Sigma_i = ptp_capacity(H, P, Z)
    X_i = eye(3) + pinv_sqrtm(Z) @ H @ Sigma_i @ H.conj().T @ pinv_sqrtm(Z)
    assert log(det(X_i)) == pytest.approx(rate_i, 1e-2)
    W_i = pinv_sqrtm(Z) @ inv(X_i) @ pinv_sqrtm(Z)
    assert np.allclose(pinv_sqrtm(Z) @ pinv_sqrtm(Z), pinv(Z))

    H_red = H[[0, 1], :]
    Z_red = Z[[0, 1], :][:, [0, 1]]
    rate_r, Sigma_r = ptp_capacity(H_red, P, Z_red)
    rate_HTZH, Cov = ptp_capacity_HTZH(H_red.conj().T@inv(Z_red)@H_red, P)
    assert rate_r == pytest.approx(rate_HTZH, 1e-2)
    assert rate_r == pytest.approx(rate_i, 1e-2)

    rcond=1e-6
    ei_d, V_d = np.linalg.eigh(H.conj().T @ pinv(Z) @ H)
    pos = ei_d > rcond
    HTZH_red = V_d[:, pos] @ np.diag(ei_d[pos]) @ V_d[:, pos].conj().T
    rate_HTZH_red, Cov = ptp_capacity_HTZH(HTZH_red, P)
    assert rate_r == pytest.approx(rate_HTZH_red, 1e-2)

    ei_d, V_d = np.linalg.eigh(H @ H.conj().T)
    inf_cons = []
    for i, e in enumerate(ei_d):
        if e > 1e-5:
            inf_cons.append(V_d[:, [i]] * 1e-5 @ V_d[:, [i]].conj().T)



KKT
    Y = B - Sigma
    Z_o = Omega - C
    S = R

    Q2 = -inv(Omega + H.conj().T @ Sigma @ H) + inv(Omega)
    Z = np.zeros((Ms_antennas, Ms_antennas))
    Y_o = np.zeros((Bs_antennas, Bs_antennas))
    assert np.allclose(sigmaBplusYo, R)
    assert np.trace(Q) / np.trace(R) == pytest.approx(np.trace(Sigma) / np.trace(Omega), 1e-2)

    rate_d = logdet(np.eye(Ms_antennas) + np.linalg.inv(R) @ H @ Q @ H.conj().T)
    Xi = Omega
    S = H @ inv(Xi + H.conj().T @ Sigma @ H) @ H.conj().T + M
    rate_h = log(det(S))
    assert rate_u == pytest.approx(rate_d, 1e-2)

def test_noise_simple():
    P = 12
    sigma = 3
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert (3, 3) == H.shape
    rate_no_channel = log(det(eye(3) + P / sigma * H.conj().T @ H))
    LOGGER.debug(f"Rate w/o channel knowledge {rate_no_channel}")
    Z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Sigma = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    rate_known = log(det(eye(3) + inv(Z) @ H @ Sigma @ H.conj().T))
    assert rate_known == pytest.approx(rate_no_channel)
    rate_calc = ptp_capacity(H, P, Z)[0]
    assert rate_known == pytest.approx(rate_calc)
    rate_worst_case, (Z, W) = ptp_worst_case_noise_approx(H.conj().T, P, sigma)
    assert rate_worst_case == pytest.approx(rate_no_channel, 1e-2)

    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert (3, 3) == H.shape
    rate_no_channel = log(det(eye(3) + P / sigma * H.conj().T @ H))
    LOGGER.debug(f"Rate w/o channel knowledge {rate_no_channel}")
    Z = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 0]])
    Sigma = np.array([[6, 0, 0], [0, 6, 0], [0, 0, 0]])
    rate_known = log(det(eye(3) + pinv(Z) @ H @ Sigma @ H.conj().T))
    assert rate_known == pytest.approx(rate_no_channel)
    rate_calc = ptp_capacity(H, P, Z)[0]
    assert rate_known == pytest.approx(rate_calc)
    rate_worst_case, (Z, W) = ptp_worst_case_noise_approx(H.conj().T, P, sigma)
    assert rate_worst_case == pytest.approx(rate_no_channel, 1e-2)


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


def worst_case_noise_static(HQHT, sigma, precision=1e-2):
    Nrx = HQHT.shape[0]
    if Nrx == 1:
        Z = np.matrix(sigma)
        rate = log(det(eye(Nrx) + inv(Z) @ HQHT))
        return rate, Z 
    Z = np.eye(Nrx) / Nrx * sigma
    f_is = []
    subgradients = []
    mu = 0
    ei_d, V_d = np.linalg.eigh(HQHT)
    inf_co = np.zeros((Nrx, Nrx))
    inf_cons = []
    for i, e in enumerate(ei_d):
        if e > 1e-3:
            inf_cons.append(V_d[:, [i]] * 1e-3 @ V_d[:, [i]].conj().T)
    for i in range(1000):
        W = np.linalg.pinv(Z + HQHT, rcond=1e-6, hermitian=True)
        rate_i = log(det(eye(Nrx) + np.linalg.pinv(Z, rcond=1e-6, hermitian=True)@HQHT)) 
        LOGGER.debug(f"Iteration {i} - Value {rate_i} - Approximation {mu}")
        if np.allclose(rate_i, mu, rtol=precision):
            break
        Z_gr = -np.linalg.pinv(Z, rcond=1e-6, hermitian=True) + W
        f_is.append(rate_i - np.real(np.trace(Z @ Z_gr)))
        subgradients.append(Z_gr)
        mu, Z = noise_outer_approximation(f_is, subgradients, sigma, inf_cons)
    return rate_i, Z

def alternating():
    O_d = eye(3)/3*sigma
    for i in range(100):
        rate_uplink, Sigma = ptp_capacity(H.conj().T, P, O_d)
        X = eye(3) + inv_sqrtm(O_d) @ H.conj().T @ Sigma @ H @ inv_sqrtm(O_d)
        #TODO this misses the constraint for O_d >> 0
        O_d = -eye(3) + inv(X)
        O_d = O_d / np.trace(O_d) * sigma
        print(f"iteration {i} rate {rate_uplink}")
#    if Ntx == 1:
#        Z = np.matrix(sigma)
#        n_sqrd = sum(H[:]*H.conj()[:])
#        rate_i = log(1 + P/sigma*n_sqrd)
#        W = 1/(Z + P*n_sqrd)
#        return rate_i, (Z, W)



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
