import numpy as np


def lqr_gain(A, B, Q, R, n_iter=500):
    # 离散 Riccati 迭代：从 P = Q 开始反复更新，直到 P 不再变化
    P = Q
    for i in range(n_iter):
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return K
