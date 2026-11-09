import numpy as np
from bicycle import step
from lqr import lqr_gain
from mpc import mpc_inputs, mpc_inputs_limited
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


def mpc_steer(x, y, theta, path, i, A, B, Q, R, N, v, dt, u_max=None):
    # 往前看 N 步：路线在第 k 步弯多少（曲率），会让车头误差每步变化 -v·dt·κ
    e, e_theta = tracking_errors(x, y, theta, path, i)
    ds = path_length(path) / (len(path["x"]) - 1)
    ahead = max(1, int(round(v * dt / ds)))
    last = len(path["x"]) - 1
    d = np.zeros((N, 2))
    for k in range(N):
        j = min(i + k * ahead, last)
        d[k, 1] = -v * dt * path["kappa"][j]
    if u_max is None:
        U = mpc_inputs(A, B, Q, R, N, np.array([e, e_theta]), d)
    else:
        U = mpc_inputs_limited(A, B, Q, R, N, np.array([e, e_theta]), u_max, d)
    # 只执行第一步，下一步重新算
    return U[0, 0]


def track(path, method, v, dt=0.1, L=2.5, max_delta=np.radians(35), Q=None, R=None,
          feedforward=True, N=20, start_offset=0.0):
    # 车放在路线起点，按 method（"lqr"、"mpc"、"mpc_limited" 或 "pp"）一直开到路线尽头
    if Q is None:
        Q = np.eye(2)
    if R is None:
        R = np.array([[1.0]])
    A, B = error_model(v, dt, L)
    K = lqr_gain(A, B, Q, R)

    # start_offset：车一开始在路线左边多少米（负数是右边）
    theta = path["heading"][0]
    x = path["x"][0] - start_offset * np.sin(theta)
    y = path["y"][0] + start_offset * np.cos(theta)
    i = 0
    log = {"x": [x], "y": [y], "e": [start_offset], "delta": []}
    last = len(path["x"]) - 1
    for k in range(int(3 * path_length(path) / (v * dt))):
        i = nearest_index(x, y, path, i)
        if i >= last:
            break
        if method == "lqr":
            delta = lqr_steer(x, y, theta, path, i, K, L, feedforward)
        elif method == "mpc":
            delta = mpc_steer(x, y, theta, path, i, A, B, Q, R, N, v, dt)
        elif method == "mpc_limited":
            delta = mpc_steer(x, y, theta, path, i, A, B, Q, R, N, v, dt, max_delta)
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
