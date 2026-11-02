import numpy as np
from model import error_model
from lqr import lqr_gain
from mpc import prediction_matrices, mpc_inputs


def test_prediction_matches_step_by_step():
    # Sx x0 + Su U 应该和一步一步用 x新 = A x + B u 算出来的一样
    A, B = error_model(5, 0.1, 2.5)
    N = 4
    x0 = np.array([0.5, 0.1])
    U = np.array([0.1, -0.2, 0.05, 0.0])
    Sx, Su = prediction_matrices(A, B, N)
    X = Sx @ x0 + Su @ U
    x = x0
    for k in range(N):
        x = A @ x + B @ np.array([U[k]])
        assert np.allclose(X[2 * k:2 * k + 2], x)


def test_long_horizon_equals_lqr():
    # 看得足够远时，MPC 的第一步和 LQR 的 -K x 一样
    A, B = error_model(5, 0.1, 2.5)
    Q = np.eye(2)
    R = np.array([[1.0]])
    x0 = np.array([0.5, 0.1])
    U = mpc_inputs(A, B, Q, R, 100, x0)
    K = lqr_gain(A, B, Q, R)
    assert np.allclose(U[0], -K @ x0)


def test_steers_back_toward_line():
    # 车在路线左边，第一步应该往右打（δ < 0）
    A, B = error_model(5, 0.1, 2.5)
    U = mpc_inputs(A, B, np.eye(2), np.array([[1.0]]), 10, np.array([0.5, 0.0]))
    assert U[0, 0] < 0
