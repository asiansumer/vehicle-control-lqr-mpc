import numpy as np
from bicycle import step
from lqr import lqr_gain
from model import error_model


def wrap_angle(a):
    # 把角度换到 [-π, π)
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_path(xs, ys):
    # 由一串点算出每个点的朝向和曲率
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    headings = np.arctan2(np.gradient(ys), np.gradient(xs))
    ds = np.hypot(np.gradient(xs), np.gradient(ys))
    kappas = np.gradient(np.unwrap(headings)) / ds
    return {"x": xs, "y": ys, "heading": headings, "kappa": kappas}


def nearest_index(x, y, path, start, window=50):
    # 只在 start 往后的 window 个点里找最近点，U 弯时不会跳到对面那段路
    end = min(start + window, len(path["x"]))
    d = np.hypot(path["x"][start:end] - x, path["y"][start:end] - y)
    return start + int(np.argmin(d))


def tracking_errors(x, y, theta, path, i):
    # e：车在路线左边为正；eθ：车头比路线方向偏左为正
    dx = x - path["x"][i]
    dy = y - path["y"][i]
    h = path["heading"][i]
    e = -dx * np.sin(h) + dy * np.cos(h)
    e_theta = wrap_angle(theta - h)
    return e, e_theta


def lqr_steer(x, y, theta, path, i, K, L=2.5, feedforward=True):
    # δ = 前馈（照着弯道本身该打的角度）+ 反馈（-K x，纠正偏差）
    e, e_theta = tracking_errors(x, y, theta, path, i)
    delta = -(K @ np.array([e, e_theta]))[0]
    if feedforward:
        delta = delta + np.arctan(L * path["kappa"][i])
    return delta


def pure_pursuit_steer(x, y, theta, path, i, v, L=2.5, k_ld=0.8, ld_min=2.0):
    # 和 P3 一样：预瞄距离 Ld = max(ld_min, k_ld·v)
    Ld = max(ld_min, k_ld * v)
    d = np.hypot(path["x"][i:] - x, path["y"][i:] - y)
    far = np.nonzero(d >= Ld)[0]
    j = i + far[0] if len(far) > 0 else len(path["x"]) - 1
    tx = path["x"][j]
    ty = path["y"][j]
    alpha = wrap_angle(np.arctan2(ty - y, tx - x) - theta)
    Ld = max(np.hypot(tx - x, ty - y), 1e-6)
    return np.arctan(2 * L * np.sin(alpha) / Ld)


def path_length(path):
    return float(np.sum(np.hypot(np.diff(path["x"]), np.diff(path["y"]))))


def track(path, method, v, dt=0.1, L=2.5, max_delta=np.radians(35), Q=None, R=None,
          feedforward=True):
    # 车放在路线起点，按 method（"lqr" 或 "pp"）一直开到路线尽头
    if Q is None:
        Q = np.eye(2)
    if R is None:
        R = np.array([[1.0]])
    A, B = error_model(v, dt, L)
    K = lqr_gain(A, B, Q, R)

    x = path["x"][0]
    y = path["y"][0]
    theta = path["heading"][0]
    i = 0
    log = {"x": [x], "y": [y], "e": [0.0], "delta": []}
    last = len(path["x"]) - 1
    for k in range(int(3 * path_length(path) / (v * dt))):
        i = nearest_index(x, y, path, i)
        if i >= last:
            break
        if method == "lqr":
            delta = lqr_steer(x, y, theta, path, i, K, L, feedforward)
        else:
            delta = pure_pursuit_steer(x, y, theta, path, i, v, L)
        delta = np.clip(delta, -max_delta, max_delta)
        x, y, theta = step(x, y, theta, v, delta, dt, L)
        i = nearest_index(x, y, path, i)
        e, e_theta = tracking_errors(x, y, theta, path, i)
        log["x"].append(x)
        log["y"].append(y)
        log["e"].append(e)
        log["delta"].append(delta)
    return log
