import numpy as np


def state_feedback(x, K):
    # 偏多少就反着打多少：δ = -K x
    return -K @ x


def closed_loop_matrix(A, B, K):
    # 把 u = -K x 代进 x新 = A x + B u，得到 x新 = (A - B K) x
    return A - B @ K


def simulate(A, B, K, x0, n):
    # 从 x0 出发跑 n 步，返回每一步的状态和方向盘角
    x = np.array(x0, dtype=float)
    xs = [x]
    us = []
    for i in range(n):
        u = state_feedback(x, K)
        x = A @ x + B @ u
        xs.append(x)
        us.append(u[0])
    return np.array(xs), np.array(us)
