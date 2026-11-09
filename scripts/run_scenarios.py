import numpy as np
import matplotlib.pyplot as plt
from scenarios import s_curve, u_turn, corner, straight_line
from tracking import track

# 四个场景：(名字, 路线, 车一开始偏多少米)
scenes = [
    ("S curve", s_curve(), 0.0),
    ("U turn R=8", u_turn(), 0.0),
    ("90° corner R=5", corner(), 0.0),
    ("start 4 m off", straight_line(), 4.0),
]
methods = [
    ("Pure Pursuit", "pp"),
    ("LQR", "lqr"),
    ("MPC N=10", "mpc"),
    ("MPC N=10 + limit", "mpc_limited"),
]
speeds = [5, 10, 15]

print(f"{'场景':16s} {'速度':>4s}  {'方法':18s} {'平均误差':>8s} {'最大误差':>8s} {'冲过头':>7s}")
for scene_name, path, offset in scenes:
    for v in speeds:
        for name, method in methods:
            log = track(path, method, v, N=10, start_offset=offset)
            e = np.array(log["e"][1:])
            over = max(0.0, -e.min()) if offset > 0 else 0.0
            print(f"{scene_name:16s} {v:4d}  {name:18s} {np.abs(e).mean():8.3f} {np.abs(e).max():8.3f} {over:7.3f}")

# 偏 4 米出发、调得很猛（R=0.1）时，限制才真正起作用
print("\n偏 4 米出发，R = 0.1（打方向盘很“便宜”）:")
for v in [5, 10]:
    for name, method in methods[1:]:
        log = track(straight_line(), method, v, N=10, R=np.array([[0.1]]), start_offset=4.0)
        e = np.array(log["e"])
        print(f"  v={v:2d}  {name:18s} 冲过头 {max(0.0, -e.min()):.3f} 米")

# 画 v=10 时四个场景的轨迹
v = 10
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (scene_name, path, offset) in zip(axes.ravel(), scenes):
    ax.plot(path["x"], path["y"], "k--", linewidth=1, label="path")
    for name, method in methods[:3]:
        log = track(path, method, v, N=10, start_offset=offset)
        ax.plot(log["x"], log["y"], label=name)
    ax.set_title(f"{scene_name}, v = {v} m/s")
    if scene_name != "S curve" and scene_name != "start 4 m off":
        ax.set_aspect("equal")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig("results/scenarios.png", dpi=120)
plt.show()
