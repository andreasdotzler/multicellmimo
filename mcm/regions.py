import numpy as np
import cvxpy as cp
from typing import Optional


class R_m_t:
    def __init__(self, users, wsr):
        self._wsr = wsr
        self.users = users
        self.approx = R_m_t_approx(self.users)
    
    def wsr(self, weights):
        val, rates = self._wsr(weights)
        self.approx.A = np.c_[self.approx.A, rates.reshape((len(rates), 1))]
        return val, rates

    def reset_approximation(self):
        self.approx = R_m_t_approx(self.users)


class R_m_t_approx():
    def __init__(self, users=[], A=None):
        self.users = users
        if A is not None:
            self.A = A
        else:
            self.A = np.empty([len(users), 0])

        self.c = None
        self.alphas = None
        self.r_in_A_x_alpha = None
        self.sum_alpha = None

    def cons_in_approx(self, c=None, sum_alphas=1):
        n_schedules = self.A.shape[1]
        assert n_schedules >= 0, "No approximation available"
        if c is None:
            c = cp.Variable(len(self.users), nonneg=True)
        self.c = c
        self.alphas = cp.Variable(n_schedules, nonneg=True)
        self.r_in_A_x_alpha = c == self.A @ self.alphas
        self.sum_alpha = cp.sum(self.alphas) == sum_alphas
        return [self.r_in_A_x_alpha, self.sum_alpha]

    def dual_values(self):
        return self.r_in_A_x_alpha.dual_value, self.sum_alpha.dual_value

    def in_approx(self, q):
        pass
        # todo run optimization 
    

class Q_vector:
    def __init__(self, q_min: Optional[np.ndarray] = None, q_max: Optional[np.ndarray] = None):
        self.q_max = q_max
        self.q_min = q_min
        if q_max is not None and q_min is not None:
            assert all(
                q_max >= q_min
            ), f"Error need q_max >= q_min - q_max : {q_max} q_min: {q_min} "
        self.q_geq_qmin = None
        self.q_leq_qmax = None

    def __len__(self):
        if self.q_min is not None:
            return len(self.q_min)
        elif self.q_max is not None:
            return len(self.q_max)
        else:
            return 0

    def __contains__(self, q: np.ndarray):
        return not (self.q_max is not None and any(q > self.q_max)) or (
            self.q_min is not None and any(q < self.q_min)
        )

    def __getitem__(self, users):
        return Q_vector(q_min=self.q_min[users], q_max=self.q_max[users])

    def con_min(self, q):
        if self.q_min is not None:
            self.q_geq_qmin = q >= self.q_min
            return self.q_geq_qmin
        else:
            return None
        
    def con_max(self, q):
        if self.q_max is not None:
            self.q_leq_qmax = q <= self.q_max
            return self.q_leq_qmax
        else:
            return None

    def constraints(self, q):
        cons = []
        min_con = self.con_min(q)
        if min_con is not None:
            cons.append(min_con)
        max_con = self.con_max(q)
        if min_con is not None:
            cons.append(max_con)
        return cons

    def dual_values(self):
        return self.q_geq_qmin.dual_value, self.q_leq_qmax.dual_value
        