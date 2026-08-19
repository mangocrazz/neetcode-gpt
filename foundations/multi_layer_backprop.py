import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # --------------------
        # Convert to numpy
        # --------------------
        # 即使只有一个样本，也保留 batch 维
        # x:      (input_dim,) -> (1, input_dim)
        # y_true: (output_dim,) -> (1, output_dim)
        x = np.array(x, dtype=float).reshape(1, -1)
        y_true = np.array(y_true, dtype=float).reshape(1, -1)

        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)

        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)

        # ====================
        # Forward
        # ====================

        # x:      (B, input_dim)
        # W1.T:   (input_dim, H)
        # z1:     (B, H)
        z1 = x @ W1.T + b1

        # ReLU
        # a1: (B, H)
        a1 = np.maximum(0, z1)

        # a1:     (B, H)
        # W2.T:   (H, O)
        # z2:     (B, O)
        z2 = a1 @ W2.T + b2

        # MSE
        loss = np.mean((z2 - y_true) ** 2)

        # ====================
        # Backward
        # ====================

        # loss = mean((z2 - y)^2)
        #
        # y_true.size = B * O
        #
        # dz2: (B, O)
        n = y_true.size
        dz2 = 2 * (z2 - y_true) / n

        # -------- W2 --------
        #
        # dz2.T: (O, B)
        # a1:    (B, H)
        #
        # dW2:   (O, H)
        dW2 = dz2.T @ a1

        # 每个输出神经元的 bias 对所有 batch 样本求和
        # db2: (O,)
        db2 = np.sum(dz2, axis=0)

        # -------- a1 --------
        #
        # dz2: (B, O)
        # W2:  (O, H)
        #
        # da1: (B, H)
        da1 = dz2 @ W2

        # -------- ReLU --------
        #
        # ReLU'(z):
        # z > 0 -> 1
        # z <= 0 -> 0
        #
        # dz1: (B, H)
        dz1 = da1 * (z1 > 0)

        # -------- W1 --------
        #
        # dz1.T: (H, B)
        # x:     (B, input_dim)
        #
        # dW1:   (H, input_dim)
        dW1 = dz1.T @ x

        # db1: (H,)
        db1 = np.sum(dz1, axis=0)

        # ====================
        # Return
        # ====================

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }