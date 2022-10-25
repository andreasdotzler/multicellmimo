import numpy as np
from typing import Any, NewType, Callable, Dict, Union


Matrix = NewType("Matrix", Any)
Weights = NewType("Weights", np.ndarray)
WeightsTransmitter = NewType("WeightsTransmitter", np.ndarray)
#Util = NewType("Util", Callable[[np.ndarray], (int, np.ndarray)])

A_m_t = NewType("A_m_t", Dict[str, Dict[int, np.ndarray]])
V_m_t = NewType("V_m_t", Dict[str, Dict[int, Any]])
Fractions = NewType("Fractions", Dict[str, float])
