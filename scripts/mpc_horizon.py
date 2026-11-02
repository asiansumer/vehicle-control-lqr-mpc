import time
import numpy as np
import matplotlib.pyplot as plt
from model import error_model
from lqr import lqr_gain
from mpc import mpc_inputs
from scenarios import s_curve, u_turn
from tracking import track

# 实验 1：不考虑弯道，只看 N 越大，MPC 第一步离 LQR 有多近
A, B = error_model(5, 0.1, 2.5)
Q = np.eye(2)
R = np.array([[1.0]])
x0 = np.array([0.5, 0.1])
K = lqr_gain(A, B, Q, R)
print("LQR 第一步方向盘:", round((-K @ x0)[0], 4))
for N in [1, 5, 10, 20, 50]:
    print(f"MPC N={N:3d} 第一步方向盘: {mpc_inputs(A, B, Q, R, N, x0)[0, 0]:.4f}")

# 实验 2：真车跟线，v = 5，比较不同 N 的误差和每步计算时间
v = 5
horizons = [1, 5, 10, 20, 50]
scenes = [("S curve", s_curve()), ("U turn", u_turn())]
fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(11, 4))
print(f"\n{'场景':8s} {'方法':10s} {'平均误差':>8s} {'最大误差':>8s} {'每步毫秒':>8s}")
for scene_name, path in scenes:
    log = track(path, "lqr", v)
    lqr_err = np.mean(np.abs(log["e"]))
    print(f"{scene_name:8s} {'LQR':10s} {lqr_err:8.3f} {np.max(np.abs(log['e'])):8.3f}")
    errs = []
    times = []
    for N in horizons:
        start = time.time()
        log = track(path, "mpc", v, N=N)
        ms = (time.time() - start) / len(log["delta"]) * 1000
        e = np.abs(log["e"])
        errs.append(e.mean())
        times.append(ms)
        print(f"{scene_name:8s} {'MPC N=' + str(N):10s} {e.mean():8.3f} {e.max():8.3f} {ms:8.2f}")
    line, = ax_err.plot(horizons, errs, marker="o", label=f"MPC, {scene_name}")
    ax_err.axhline(lqr_err, color=line.get_color(), linestyle="--", label=f"LQR, {scene_name}")
    ax_time.plot(horizons, times, marker="o", label=scene_name)

ax_err.set_xscale("log")
ax_err.set_yscale("log")
ax_err.set_xlabel("horizon N")
ax_err.set_ylabel("mean |lateral error| (m)")
ax_err.legend(fontsize=8)
ax_time.set_xscale("log")
ax_time.set_xlabel("horizon N")
ax_time.set_ylabel("time per step (ms)")
ax_time.legend(fontsize=8)
fig.tight_layout()
plt.savefig("results/mpc_horizon.png", dpi=120)
plt.show()
