import numpy as np
from model import error_model
from lqr import lqr_gain
from control import closed_loop_matrix


def test_scalar_case():
    # 一维情况 A=B=Q=R=1 有手算答案：K = 0.618（黄金分割）
    one = np.array([[1.0]])
    K = lqr_gain(one, one, one, one)
    assert np.isclose(K[0, 0], (np.sqrt(5) - 1) / 2)


def test_lqr_is_stable():
    A, B = error_model(5, 0.1, 2.5)
    K = lqr_gain(A, B, np.eye(2), np.array([[1.0]]))
    assert np.all(np.abs(np.linalg.eigvals(closed_loop_matrix(A, B, K))) < 1)


def test_bigger_r_gives_gentler_steering():
    # R 越大 = 越“心疼”打方向盘，K 就越小
    A, B = error_model(5, 0.1, 2.5)
    K_small_r = lqr_gain(A, B, np.eye(2), np.array([[0.1]]))
    K_big_r = lqr_gain(A, B, np.eye(2), np.array([[10.0]]))
    assert np.all(np.abs(K_big_r) < np.abs(K_small_r))
