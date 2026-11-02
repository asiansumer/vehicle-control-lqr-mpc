import numpy as np
import matplotlib.pyplot as plt
from scenarios import s_curve, u_turn
from tracking import track

speeds = [5, 10, 15]
methods = [
    ("Pure Pursuit", "pp", True),
    ("LQR", "lqr", True),
    ("LQR no feedforward", "lqr", False),
]
scenes = [("S curve", s_curve()), ("U turn", u_turn())]

print(f"{'场景':8s} {'速度':>4s}  {'方法':20s} {'平均误差':>8s} {'最大误差':>8s} {'最大方向盘':>8s}")
for scene_name, path in scenes:
    for v in speeds:
        for name, method, ff in methods:
            log = track(path, method, v, feedforward=ff)
            e = np.abs(log["e"])
            d = np.degrees(np.max(np.abs(log["delta"])))
            print(f"{scene_name:8s} {v:4d}  {name:20s} {e.mean():8.3f} {e.max():8.3f} {d:7.1f}°")

# 画 v=15 时两个场景的轨迹和横向误差
v = 15
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for col, (scene_name, path) in enumerate(scenes):
    ax_path = axes[0, col]
    ax_err = axes[1, col]
    ax_path.plot(path["x"], path["y"], "k--", linewidth=1, label="path")
    for name, method, ff in methods:
        log = track(path, method, v, feedforward=ff)
        ax_path.plot(log["x"], log["y"], label=name)
        ax_err.plot(np.abs(log["e"]), label=name)
    ax_path.set_title(f"{scene_name}, v = {v} m/s")
    if scene_name == "U turn":
        ax_path.set_aspect("equal")
    ax_path.legend(fontsize=8)
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("|lateral error| (m)")
    ax_err.legend(fontsize=8)

fig.tight_layout()
plt.savefig("results/compare_tracking.png", dpi=120)
plt.show()
