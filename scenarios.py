import numpy as np
from tracking import make_path


def s_curve():
    # 和 P3 follow_path 一样的 S 弯：y = 3 sin(x/5)
    xs = np.arange(0, 60, 0.1)
    return make_path(xs, 3 * np.sin(xs / 5))


def u_turn(radius=8.0):
    # 直行 20 米 → 半径 radius 的半圆掉头 → 往回直行 20 米
    straight = np.arange(0, 20, 0.1)
    t = np.arange(-np.pi / 2, np.pi / 2, 0.1 / radius)
    xs = np.concatenate([straight, 20 + radius * np.cos(t), straight[::-1]])
    ys = np.concatenate([np.zeros_like(straight), radius + radius * np.sin(t),
                         2 * radius * np.ones_like(straight)])
    return make_path(xs, ys)
