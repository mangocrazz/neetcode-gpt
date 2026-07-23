import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x,w) + b
        y = 1 / (1 + np.exp(-z))
        dzdw = x
        dydz = np.exp(-z) * (y ** 2)
        dldy = y - y_true
        dldw = dldy * dydz * dzdw
        dldb = dldy * dydz
        return (np.round(dldw,5),np.round(dldb,5))
