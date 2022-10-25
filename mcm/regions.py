import numpy as np
import cvxpy as cp
from typing import Optional


class R_m_t:
    def __init__(self, wsr, users):
        self._wsr = wsr
        self.users = users
        self.A = np.empty([len(users),0])
        self.schedule = None
        self.c = None
        self.alphas = None
    
    def wsr(self, weights):
        val, rates = self._wsr(weights)
        np.c_[self.A, rates.reshape((len(rates), 1))]
        return val, rates

    def cons_in_approx(self, c=None, sum_alphas=1):
        n_schedules = self.A.shape[1]
        assert n_schedules >= 0, "No approximation available"
        if c is None:
            self.c = c = cp.Variable(len(self.users), nonneg=True)
        self.alphas = cp.Variable(n_schedules)
        return [c == self.A @ self.alphas, cp.sum(self.alphas) == sum_alphas]

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
            return q >= self.q_min
        else:
            return None
        
    def con_max(self, q):
        if self.q_max is not None:
            return q <= self.q_max
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
        