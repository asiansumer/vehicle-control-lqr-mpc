import numpy as np
from bicycle import step


def test_straight_line():
    # 方向盘不打，沿 x 轴走 0.1 秒 × 5 m/s = 0.5 米
    x, y, theta = step(0.0, 0.0, 0.0, 5, 0.0)
    assert np.allclose([x, y, theta], [0.5, 0.0, 0.0])


def test_left_turn_increases_heading():
    # 方向盘往左打，车头角度变大
    x, y, theta = step(0.0, 0.0, 0.0, 5, 0.1)
    assert theta > 0
