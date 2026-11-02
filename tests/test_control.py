import numpy as np
from model import error_model
from control import state_feedback, closed_loop_matrix, simulate


def test_feedback_sign():
    # 车在路线左边（e > 0），方向盘应该往反方向打（δ < 0）
    K = np.array([[0.5, 2.0]])
    u = state_feedback(np.array([0.5, 0.1]), K)
    assert np.allclose(u, [-0.45])


def test_closed_loop_matrix():
    A, B = error_model(5, 0.1, 2.5)
    K = np.array([[0.5, 2.0]])
    M = closed_loop_matrix(A, B, K)
    assert np.allclose(M, [[1, 0.5], [-0.1, 0.6]])


def test_zero_gain_is_drift():
    # K = 0 就是第 1 关的“方向盘不动”，20 步后 e = 1.5
    A, B = error_model(5, 0.1, 2.5)
    xs, us = simulate(A, B, np.zeros((1, 2)), [0.5, 0.1], 20)
    assert np.isclose(xs[-1, 0], 1.5)
    assert np.allclose(us, 0)


def test_good_gain_returns_to_path():
    A, B = error_model(5, 0.1, 2.5)
    K = np.array([[0.5, 2.0]])
    xs, us = simulate(A, B, K, [0.5, 0.1], 60)
    assert abs(xs[-1, 0]) < 0.01
    assert np.all(np.abs(np.linalg.eigvals(closed_loop_matrix(A, B, K))) < 1)
