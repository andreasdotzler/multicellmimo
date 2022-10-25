import numpy as np
from typing import Any, NewType, Callable


Matrix = NewType("Matrix", Any)
Weights = NewType("Weights", np.array)
WeightsTransmitter = NewType("WeightsTransmitter", np.array)
# Util = NewType("Util", Callable[[np.array], (int, np.array)])
