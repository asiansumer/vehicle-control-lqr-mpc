import numpy as np
from tracking import make_path, tracking_errors, track


def straight_path():
    xs = np.arange(0, 30, 0.1)
    return make_path(xs, np.zeros_like(xs))


def test_error_signs():
    # 路线沿 x 轴：车在 y=1（路线左边）→ e=+1；车头朝左上 0.1 → eθ=+0.1
    path = straight_path()
    e, e_theta = tracking_errors(5.0, 1.0, 0.1, path, 50)
    assert np.isclose(e, 1.0)
    assert np.isclose(e_theta, 0.1)


def test_circle_curvature():
    # 半径 10 的圆，曲率应该是 1/10
    t = np.arange(0, np.pi, 0.01)
    path = make_path(10 * np.cos(t), 10 * np.sin(t))
    assert np.allclose(path["kappa"][5:-5], 0.1, atol=1e-3)


def test_lqr_pulls_back_to_line():
    # 车从路线左边 1 米出发，LQR 应该把它拉回来
    path = straight_path()
    path["y"] = path["y"] - 1.0
    log = track(path, "lqr", v=5)
    assert abs(log["e"][-1]) < 0.05


def test_start_offset():
    # start_offset=2：车一开始就在路线左边 2 米
    log = track(straight_path(), "lqr", v=5, start_offset=2.0)
    assert np.isclose(log["e"][0], 2.0)
    assert np.isclose(log["y"][0], 2.0)
