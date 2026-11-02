import numpy as np


def prediction_matrices(A, B, N):
    # 把未来 N 步一次写出来：X = Sx x0 + Su U
    # X = [x1, x2, ..., xN] 竖着排，U = [u0, u1, ..., u(N-1)] 竖着排
    nx = A.shape[0]
    nu = B.shape[1]
    Sx = np.zeros((N * nx, nx))
    Su = np.zeros((N * nx, N * nu))
    A_power = np.eye(nx)
    for k in range(N):
        A_power = A @ A_power
        Sx[k * nx:(k + 1) * nx, :] = A_power
        for j in range(k + 1):
            # u_j 经过 (k - j) 次 A 之后影响到 x_(k+1)
            Su[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = np.linalg.matrix_power(A, k - j) @ B
    return Sx, Su


def mpc_inputs(A, B, Q, R, N, x0, d=None):
    # 无约束 MPC：找让未来 N 步罚款总和最小的 U
    # d：每一步额外的“外力”（比如弯道让路线自己转向），形状 (N, nx)
    nx = A.shape[0]
    Sx, Su = prediction_matrices(A, B, N)
    _, Sd = prediction_matrices(A, np.eye(nx), N)
    Q_bar = np.kron(np.eye(N), Q)
    R_bar = np.kron(np.eye(N), R)
    free = Sx @ x0
    if d is not None:
        free = free + Sd @ np.ravel(d)
    H = Su.T @ Q_bar @ Su + R_bar
    U = -np.linalg.solve(H, Su.T @ Q_bar @ free)
    return U.reshape(N, B.shape[1])

