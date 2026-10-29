import numpy as np
from model import error_model


def test_matrix_values():
    A, B = error_model(5, 0.1, 2.5)
    assert np.allclose(A, [[1, 0.5], [0, 1]])
    assert np.allclose(B, [[0], [0.2]])


def test_hand_calculation():
    # 就是刚才的手算题：e=0.5, eθ=0.1, δ=0.1 → [0.55, 0.12]
    A, B = error_model(5, 0.1, 2.5)
    x = np.array([0.5, 0.1])
    u = np.array([0.1])
    x_new = A @ x + B @ u
    assert np.allclose(x_new, [0.55, 0.12])
