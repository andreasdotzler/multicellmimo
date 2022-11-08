from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Callable, Optional, Tuple, List, Union
from collections.abc import Sized

from mcm.no_utils import solve_problem
from mcm.my_typing import Weights


class R_m_t:
    def __init__(self, users: List[int], wsr: Callable[[Weights], Tuple[float, np.ndarray]]):
        self._wsr = wsr
        self.users = users
        self.approx = R_m_t_approx(self.users)

    def wsr(self, weights: Weights) -> Tuple[float, np.ndarray]:
        val, rates = self._wsr(weights)
        self.approx.A = np.c_[self.approx.A, rates.reshape((len(rates), 1))]
        return val, rates

    def reset_approximation(self) -> None:
        self.approx = R_m_t_approx(self.users)


class R_m_t_approx:
    def __init__(self, users: Optional[List[int]] = None, A: Optional[np.ndarray] = None, in_tol: float = 1e-3):
        if users is None:
            users = []
        self.users = users
        if A is not None:
            if len(A.shape) == 1:
                # if A is a vector we convert it to a matrix
                # the view does not copy the data
                A_v = A.view()
                A_v.shape = (A.shape[0], 1)
                self.A = A_v
            else:
                self.A = A
        else:
            self.A = np.empty([len(users), 0])

        self.c: cp.Variable
        self.alphas: cp.Variable
        self.r_in_A_x_alpha: cp.constraints.constraint.Constraint
        self.sum_alpha: cp.constraints.constraint.Constraint
        self.in_tol = in_tol

    def cons_in_approx(self, c: Optional[cp.Variable] = None, sum_alphas: Union[float, cp.Variable] = 1) -> List[cp.constraints.constraint.Constraint]:
        n_schedules = self.A.shape[1]
        assert n_schedules >= 0, "No approximation available"
        if c is None:
            c = cp.Variable(len(self.users), nonneg=True)
        self.c = c
        self.alphas = cp.Variable(n_schedules, nonneg=True)
        self.r_in_A_x_alpha = c == self.A @ self.alphas
        self.sum_alpha = cp.sum(self.alphas) == sum_alphas
        return [self.r_in_A_x_alpha, self.sum_alpha]

    def dual_values(self) -> Tuple[cp.constraints.constraint.Constraint, cp.constraints.constraint.Constraint]:
        return self.r_in_A_x_alpha.dual_value, self.sum_alpha.dual_value

    def __contains__(self, q: np.ndarray) -> bool:
        # TODO should we minimize distance?
        alphas = cp.Variable(self.A.shape[1], nonneg=True)
        solve_problem(cp.Minimize(cp.sum(alphas)), [q == self.A @ alphas])
        return sum(alphas.value) <= (1 + self.in_tol)


class Q_vector(Sized):
    def __init__(
        self, q_min: Optional[np.ndarray] = None, q_max: Optional[np.ndarray] = None
    ):
        self.q_max = q_max
        self.q_min = q_min
        if q_max is not None and q_min is not None:
            assert all(
                q_max >= q_min
            ), f"Error need q_max >= q_min - q_max : {q_max} q_min: {q_min} "
        self.q_geq_qmin: cp.constraints.constraint.Constraint
        self.q_leq_qmax: cp.constraints.constraint.Constraint

    def __len__(self) -> int:
        if self.q_min is not None:
            return len(self.q_min)
        elif self.q_max is not None:
            return len(self.q_max)
        else:
            return 0

    def __contains__(self, q: np.ndarray) -> bool:
        # TODO: compute distance and add tolerance
        return not (self.q_max is not None and any(q > self.q_max)) or (
            self.q_min is not None and any(q < self.q_min)
        )

    def __getitem__(self, users: List[int]) -> Q_vector:
        q_min: np.ndarray = self.q_min[users]
        q_max: np.ndarray = self.q_max[users]
        return Q_vector(q_min=q_min, q_max=q_max)

    def con_min(self, q: Optional[np.ndarray]) -> Optional[cp.constraints.constraint.Constraint]:
        if self.q_min is not None:
            self.q_geq_qmin = q >= self.q_min
            return self.q_geq_qmin
        else:
            return None

    def con_max(self, q: Optional[np.ndarray]) -> Optional[cp.constraints.constraint.Constraint]:
        if self.q_max is not None:
            self.q_leq_qmax = q <= self.q_max
            return self.q_leq_qmax
        else:
            return None

    def constraints(self, q: cp.Variable) -> List[cp.constraints.constraint.Constraint]:
        cons = []
        min_con = self.con_min(q)
        if min_con is not None:
            cons.append(min_con)
        max_con = self.con_max(q)
        if min_con is not None:
            cons.append(max_con)
        return cons

    def dual_values(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.q_geq_qmin.dual_value, self.q_leq_qmax.dual_value
