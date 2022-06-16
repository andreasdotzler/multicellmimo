import numpy as np
from typing import Any
from typing import NewType


Matrix = NewType("Matrix", Any)
Weights = NewType("Weights", np.array)
WeightsTransmitter = NewType("WeightsTransmitter", np.array)
